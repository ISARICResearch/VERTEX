import os
import threading
import time
import uuid
from functools import lru_cache
from urllib.parse import quote, urlparse

import jwt
import requests
from dash import html
from flask import current_app, g, request
from jwt import PyJWKClient
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

from vertex.logging.logger import setup_logger
from vertex.vertex_secrets import get_database_url

logger = setup_logger(__name__)

AUTH_DATABASE_URL = get_database_url()
engine = create_engine(AUTH_DATABASE_URL)
USER_CONTEXT_TTL_SECONDS = 900
_DB_USER_CACHE: dict[str, tuple[float, dict | None]] = {}
_DB_USER_CACHE_LOCK = threading.Lock()


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def should_enable_auth(auth_settings: dict) -> bool:
    explicit_toggle = os.getenv("VERTEX_ENABLE_AUTH")
    if explicit_toggle is not None:
        return _is_truthy(explicit_toggle)

    return bool(auth_settings.get("VERTEX_BASE_URL") and auth_settings.get("COGNITO_CLIENT_ID"))


@lru_cache(maxsize=8)
def _jwks_client_for_issuer(issuer: str) -> PyJWKClient:
    discovery = requests.get(f"{issuer}/.well-known/openid-configuration", timeout=2).json()
    return PyJWKClient(discovery["jwks_uri"])


def verify_id_token(token: str) -> dict:
    unverified = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    issuer = unverified["iss"]

    signing_key = _jwks_client_for_issuer(issuer).get_signing_key_from_jwt(token)

    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=current_app.config["COGNITO_CLIENT_ID"],
        issuer=issuer,
        leeway=30,
    )


@lru_cache(maxsize=1)
def _get_auth_orm_classes():
    if engine.dialect.name != "postgresql":
        return None

    try:
        base = automap_base()
        base.prepare(autoload_with=engine, schema="public")

        users_cls = getattr(base.classes, "users", None)
        projects_cls = getattr(base.classes, "projects", None)
        mapping_cls = getattr(base.classes, "user_project_mapping", None)

        if not users_cls:
            logger.warning("Auth ORM setup missing public.users table")
            return None

        return {
            "users": users_cls,
            "projects": projects_cls,
            "user_project_mapping": mapping_cls,
        }
    except SQLAlchemyError as exc:
        logger.warning(f"Unable to reflect auth ORM models: {exc}")
        return None


def get_user_by_email_readonly(email: str) -> dict | None:
    if not email:
        return None

    orm_classes = _get_auth_orm_classes()
    if not orm_classes:
        return None

    users_cls = orm_classes["users"]

    try:
        with Session(engine) as db_session:
            user = db_session.execute(
                select(users_cls).where(func.lower(users_cls.email) == func.lower(email)).limit(1)
            ).scalar_one_or_none()

            if not user:
                return None

            return {
                "id": str(getattr(user, "id", "")),
                "email": getattr(user, "email", None),
                "is_admin": getattr(user, "is_admin", None),
                "last_login": getattr(user, "last_login", None),
                "created": getattr(user, "created", None),
                "updated": getattr(user, "updated", None),
            }
    except SQLAlchemyError as exc:
        logger.warning(f"Unable to read user from database for {email}: {exc}")
        return None


def _user_row_to_dict(user) -> dict:
    return {
        "id": str(getattr(user, "id", "")),
        "email": getattr(user, "email", None),
        "is_admin": getattr(user, "is_admin", None),
        "last_login": getattr(user, "last_login", None),
        "created": getattr(user, "created", None),
        "updated": getattr(user, "updated", None),
    }


@lru_cache(maxsize=1)
def has_project_access_tables() -> bool:
    if engine.dialect.name != "postgresql":
        return False

    try:
        inspector = inspect(engine)
        has_mapping = inspector.has_table("user_project_mapping", schema="public")
        has_projects = inspector.has_table("projects", schema="public")
        has_tables = has_mapping and has_projects
        if not has_tables:
            logger.warning(
                "Project access tables are missing (expected public.user_project_mapping and public.projects); "
                "skipping project membership lookup"
            )
        return has_tables
    except SQLAlchemyError as exc:
        logger.warning(f"Unable to probe project access tables: {exc}")
        return False


def get_user_project_ids_readonly(user_id: str) -> list[str]:
    if not user_id:
        return []

    if engine.dialect.name != "postgresql":
        return []

    if not has_project_access_tables():
        return []

    orm_classes = _get_auth_orm_classes()
    if not orm_classes:
        return []

    projects_cls = orm_classes.get("projects")
    mapping_cls = orm_classes.get("user_project_mapping")
    if not projects_cls or not mapping_cls:
        return []

    parsed_user_id: str | uuid.UUID = user_id
    try:
        parsed_user_id = uuid.UUID(user_id)
    except (ValueError, TypeError):
        parsed_user_id = user_id

    try:
        with Session(engine) as db_session:
            rows = db_session.execute(
                select(projects_cls.vertex_id)
                .join(mapping_cls, projects_cls.id == mapping_cls.project_id)
                .where(mapping_cls.user_id == parsed_user_id)
                .where(projects_cls.vertex_id.is_not(None))
                .order_by(projects_cls.vertex_id)
            ).scalars()
            return [vertex_id for vertex_id in rows if vertex_id]
    except SQLAlchemyError as exc:
        logger.warning(f"Unable to read user project IDs for user_id={user_id}: {exc}")
        return []


def _get_cached_db_user(email: str) -> dict | None:
    if not email:
        return None

    now = time.time()
    cache_key = email.strip().lower()

    with _DB_USER_CACHE_LOCK:
        cached = _DB_USER_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

    db_user = get_user_by_email_readonly(email)
    if db_user:
        db_user["allowed_project_ids"] = get_user_project_ids_readonly(db_user.get("id"))

    with _DB_USER_CACHE_LOCK:
        _DB_USER_CACHE[cache_key] = (now + USER_CONTEXT_TTL_SECONDS, db_user)

    return db_user


def get_request_user_context(auth_enabled: bool) -> dict | None:
    if not auth_enabled:
        return None

    if hasattr(g, "vertex_user_context"):
        return g.vertex_user_context

    token = request.cookies.get("vertex_id_token")
    if not token:
        g.vertex_user_context = None
        return None

    try:
        claims = verify_id_token(token)
    except jwt.PyJWTError as exc:
        logger.info(f"Invalid vertex_id_token cookie ({exc}), continuing as logged out")
        g.vertex_user_context = None
        return None

    db_user = _get_cached_db_user(claims.get("email"))
    if db_user:
        logger.info(
            "Authenticated DB user: "
            f"email={db_user.get('email')} id={db_user.get('id')} "
            f"allowed_project_ids={db_user.get('allowed_project_ids')}"
        )
    else:
        logger.warning(f"Valid Cognito token but no matching DB user for email={claims.get('email')} (read-only lookup)")

    g.vertex_user_context = {
        "claims": claims,
        "db_user": db_user,
    }
    return g.vertex_user_context


def get_request_db_user(auth_enabled: bool) -> dict | None:
    context = get_request_user_context(auth_enabled)
    if not context:
        return None
    return context.get("db_user")


def get_request_login_state(auth_enabled: bool) -> dict:
    if not auth_enabled:
        return {"is_logged_in": True, "allowed_project_ids": []}

    db_user = get_request_db_user(auth_enabled)
    if not db_user:
        return {"is_logged_in": False, "allowed_project_ids": []}

    return {
        "is_logged_in": True,
        "allowed_project_ids": db_user.get("allowed_project_ids", []),
    }


def get_request_is_logged_in(auth_enabled: bool) -> bool:
    return get_request_login_state(auth_enabled).get("is_logged_in", False)


def get_accounts_login_url(next_url: str | None = None) -> str:
    vertex_base_url = current_app.config["VERTEX_BASE_URL"].rstrip("/")
    login_url = f"{vertex_base_url}/login"
    if not next_url:
        return login_url
    return f"{login_url}?next={quote(next_url, safe='')}"


def get_accounts_logout_url(next_url: str | None = None) -> str:
    vertex_base_url = current_app.config["VERTEX_BASE_URL"].rstrip("/")
    logout_url = f"{vertex_base_url}/logout"
    if not next_url:
        return logout_url
    return f"{logout_url}?next={quote(next_url, safe='')}"


def _normalise_next_url(current_url: str | None) -> str:
    dash_internal = ("/_dash-", "/_reload-", "/auth/")

    if current_url:
        parsed = urlparse(current_url)
        path = parsed.path or "/"
        if not any(path.startswith(prefix) for prefix in dash_internal):
            return current_url

    raw_url = request.url
    if any(request.path.startswith(prefix) for prefix in dash_internal):
        return "/"
    return raw_url


def build_auth_controls(auth_enabled: bool, is_logged_in: bool, current_url: str | None = None):
    if not auth_enabled:
        return html.Div()

    next_url = _normalise_next_url(current_url)

    if is_logged_in:
        return html.A("Logout", href=get_accounts_logout_url(next_url), className="btn btn-danger btn-md")

    return html.A("Login", href=get_accounts_login_url(next_url), className="btn btn-primary btn-md")
