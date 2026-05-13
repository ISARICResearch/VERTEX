from urllib.parse import quote

import jwt
import requests
from flask import current_app, redirect, request, session, url_for

from vertex.auth.service import AUTH_DATABASE_URL, get_request_user_context
from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)


def configure_auth(app, auth_enabled: bool, auth_settings: dict) -> None:
    app.server.config.update(
        {
            "SQLALCHEMY_DATABASE_URI": AUTH_DATABASE_URL if auth_enabled else None,
            "SECRET_KEY": auth_settings.get("SECRET_KEY"),
            "COGNITO_CLIENT_ID": auth_settings.get("COGNITO_CLIENT_ID"),
            "COGNITO_CLIENT_SECRET": auth_settings.get("COGNITO_CLIENT_SECRET"),
            "COGNITO_USER_POOL_ID": auth_settings.get("COGNITO_USER_POOL_ID"),
            "COGNITO_DOMAIN": auth_settings.get("COGNITO_DOMAIN"),
            "VERTEX_BASE_URL": auth_settings.get("VERTEX_BASE_URL"),
            "AWS_REGION": auth_settings.get("AWS_REGION"),
        }
    )

    @app.server.route("/auth/login")
    def auth_login():
        next_url = request.args.get("next", request.url_root)
        cognito_domain = current_app.config.get("COGNITO_DOMAIN", "").rstrip("/")
        client_id = current_app.config.get("COGNITO_CLIENT_ID")

        if not cognito_domain or not client_id:
            logger.error("COGNITO_DOMAIN or COGNITO_CLIENT_ID not configured - cannot initiate login")
            return "Auth not configured", 500

        callback_uri = url_for("auth_callback", _external=True)
        session["auth_next"] = next_url

        authorize_url = (
            f"{cognito_domain}/oauth2/authorize"
            f"?response_type=code"
            f"&client_id={client_id}"
            f"&redirect_uri={quote(callback_uri, safe='')}"
            f"&scope=openid%20email%20profile"
        )
        return redirect(authorize_url)

    @app.server.route("/auth/callback")
    def auth_callback():
        code = request.args.get("code")
        if not code:
            logger.warning("auth_callback called without code param")
            return redirect("/")

        cognito_domain = current_app.config.get("COGNITO_DOMAIN", "").rstrip("/")
        client_id = current_app.config.get("COGNITO_CLIENT_ID")
        client_secret = current_app.config.get("COGNITO_CLIENT_SECRET")
        callback_uri = url_for("auth_callback", _external=True)

        try:
            token_resp = requests.post(
                f"{cognito_domain}/oauth2/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "redirect_uri": callback_uri,
                },
                auth=(client_id, client_secret) if client_secret else None,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.error(f"Token exchange request failed: {exc}")
            return "Login failed - token request failed", 502

        if token_resp.status_code >= 400:
            logger.error("Token exchange HTTP error: " f"status={token_resp.status_code} body={token_resp.text[:400]}")
            return "Login failed - token endpoint error", 502

        try:
            token_data = token_resp.json()
        except ValueError:
            logger.error(
                "Token exchange returned non-JSON response: " f"status={token_resp.status_code} body={token_resp.text[:400]}"
            )
            return "Login failed - invalid token response", 502
        id_token = token_data.get("id_token")

        if not id_token:
            logger.error(f"Token exchange failed: {token_data.get('error', token_data)}")
            return "Login failed - could not obtain token", 502

        claims = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
        logger.debug(
            "=== USER SESSION ===\n"
            f"  sub:            {claims.get('sub')}\n"
            f"  email:          {claims.get('email')}\n"
            f"  name:           {claims.get('name')}\n"
            f"  cognito:groups: {claims.get('cognito:groups')}\n"
            f"  exp:            {claims.get('exp')}\n"
            "==================="
        )

        next_url = session.pop("auth_next", "/")
        resp = redirect(next_url)
        resp.set_cookie("vertex_id_token", id_token, httponly=True, samesite="Lax")
        return resp

    @app.server.route("/auth/logout")
    def auth_logout():
        cognito_domain = current_app.config.get("COGNITO_DOMAIN", "").rstrip("/")
        client_id = current_app.config.get("COGNITO_CLIENT_ID")
        app_root = request.url_root.rstrip("/")
        next_url = request.args.get("next", "")
        dash_internal = ("/_dash-", "/_reload-", "/auth/")
        if not next_url or any(prefix in next_url for prefix in dash_internal):
            next_url = app_root
        if next_url.startswith("/"):
            next_url = app_root + next_url

        resp = redirect(f"{cognito_domain}/logout" f"?client_id={client_id}" f"&logout_uri={quote(next_url, safe='')}")
        resp.delete_cookie("vertex_id_token")
        return resp

    @app.server.before_request
    def validate_accounts_session():
        path = request.path

        if path.startswith(("/assets/", "/static/", "/favicon.ico", "/_reload-hash", "/auth/")):
            return None

        if not auth_enabled:
            return None

        get_request_user_context(auth_enabled)
        return None
