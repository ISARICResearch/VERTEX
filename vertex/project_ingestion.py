import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import MetaData, Table, create_engine, func, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from vertex.logging.logger import setup_logger
from vertex.vertex_secrets import get_database_url

logger = setup_logger(__name__)

REQUIRED_PROJECT_COLUMNS = ("id", "vertex_id", "owner_id", "is_public")
REQUIRED_USER_COLUMNS = ("id", "email")
REQUIRED_USER_PROJECT_MAPPING_COLUMNS = ("user_id", "project_id")


def _normalise_owner_email(value):
    if not value:
        return None
    return str(value).strip().lower()


def _normalise_is_public(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_static_projects(projects_dir: str | Path) -> list[dict]:
    root = Path(projects_dir).expanduser()
    records: list[dict] = []
    if not root.exists():
        logger.warning(f"Static projects directory does not exist: {root}")
        return records

    for project_path in sorted(root.iterdir()):
        if not project_path.is_dir() or project_path.name.startswith("."):
            continue

        config_file = project_path / "config_file.json"
        if not config_file.exists():
            logger.warning(f"Skipping folder without config_file.json: {project_path}")
            continue

        try:
            config = json.loads(config_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipping project with invalid config_file.json at {project_path}: {exc}")
            continue

        project_id = str(config.get("project_id") or "").strip() or None
        if not project_id:
            logger.warning(f"Skipping static project with missing project_id: {project_path}")
            continue

        project_name = str(config.get("project_name") or "").strip() or project_path.name
        records.append(
            {
                "project_id": project_id,
                "name": project_name,
                "project_owner": _normalise_owner_email(config.get("project_owner") or config.get("owner_email")),
                "is_public": _normalise_is_public(config.get("is_public", True)),
                "project_dir": str(project_path),
            }
        )

    return records


def _db_schema_name(engine, schema: str | None) -> str | None:
    if engine.dialect.name != "postgresql":
        return None
    return schema or "public"


def _reflect_table(engine, table_name: str, schema_name: str | None) -> Table:
    metadata = MetaData()
    return Table(table_name, metadata, autoload_with=engine, schema=schema_name)


def _require_columns(table: Table, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column_name for column_name in required_columns if column_name not in table.c]
    if missing_columns:
        raise RuntimeError(f"{table.name} table is missing required columns: {', '.join(sorted(missing_columns))}")


def _load_tables(engine, schema: str | None) -> dict:
    schema_name = _db_schema_name(engine, schema)
    inspector = inspect(engine)

    required_tables = ("projects", "users", "user_project_mapping")
    for table_name in required_tables:
        if not inspector.has_table(table_name, schema=schema_name):
            raise RuntimeError(f"Required table is missing: {table_name}")

    tables = {
        "projects": _reflect_table(engine, "projects", schema_name),
        "users": _reflect_table(engine, "users", schema_name),
        "user_project_mapping": _reflect_table(engine, "user_project_mapping", schema_name),
    }
    _require_columns(tables["projects"], REQUIRED_PROJECT_COLUMNS)
    _require_columns(tables["users"], REQUIRED_USER_COLUMNS)
    _require_columns(tables["user_project_mapping"], REQUIRED_USER_PROJECT_MAPPING_COLUMNS)
    return tables


def _project_debug_summary(project: dict) -> str:
    return json.dumps(
        {
            "project_id": project.get("project_id"),
            "name": project.get("name"),
            "project_owner": project.get("project_owner"),
            "is_public": project.get("is_public"),
            "project_dir": project.get("project_dir"),
        },
        sort_keys=True,
    )


def _project_failure_message(project: dict, exc: Exception) -> str:
    return f"Failed to ingest project: {exc}. Parsed config={_project_debug_summary(project)}"


def _build_project_insert_values(project: dict, projects_table: Table, owner_id) -> dict:
    values = {
        "vertex_id": project["project_id"],
        "owner_id": owner_id,
        "is_public": project["is_public"],
    }
    now_utc = datetime.now(timezone.utc)

    if "name" in projects_table.c:
        values["name"] = project["name"]
    if "project_dir" in projects_table.c:
        values["project_dir"] = project["project_dir"]
    if "created_at" in projects_table.c:
        values.setdefault("created_at", now_utc)
    if "updated_at" in projects_table.c:
        values.setdefault("updated_at", now_utc)
    if "created" in projects_table.c:
        values.setdefault("created", now_utc)
    if "updated" in projects_table.c:
        values.setdefault("updated", now_utc)

    id_column = projects_table.c["id"]
    id_is_uuid_like = False
    try:
        id_is_uuid_like = id_column.type.python_type is uuid.UUID
    except NotImplementedError:
        id_is_uuid_like = "uuid" in id_column.type.__class__.__name__.lower()
    if (
        id_is_uuid_like
        and not id_column.nullable
        and id_column.default is None
        and id_column.server_default is None
        and "id" not in values
    ):
        values["id"] = uuid.uuid4()

    return values


def _find_project_by_vertex_id(session: Session, projects_table: Table, project_id: str):
    row = session.execute(select(projects_table).where(projects_table.c["vertex_id"] == project_id).limit(1)).mappings().first()
    return row


def _find_user_by_email(session: Session, users_table: Table, email: str):
    return (
        session.execute(select(users_table).where(func.lower(users_table.c["email"]) == email.lower()).limit(1))
        .mappings()
        .first()
    )


def _project_has_owner_mapping(session: Session, tables: dict, project_row) -> bool:
    mapping_table = tables["user_project_mapping"]
    project_pk_value = project_row["id"]
    existing_link = session.execute(
        select(mapping_table.c["project_id"]).where(mapping_table.c["project_id"] == project_pk_value).limit(1)
    ).first()
    return existing_link is not None


def _try_link_owner(session: Session, tables: dict, project_row, owner_email: str | None, stats: dict):
    users_table = tables["users"]
    mapping_table = tables["user_project_mapping"]

    if not owner_email:
        stats["owner_missing_in_config"] += 1
        return

    project_pk_value = project_row["id"]
    user_row = _find_user_by_email(session, users_table, owner_email)

    if user_row is None:
        stats["owner_pending_users"] += 1
        logger.info(
            "Owner account not found yet; project owner link will be created later. "
            f"owner_email={owner_email} project_vertex_id={project_row.get('vertex_id')}"
        )
        return

    user_id = user_row.get("id")
    if user_id is None:
        stats["owner_linking_unavailable"] += 1
        logger.warning(f"Skipping owner linking because users row has no id for owner_email={owner_email}")
        return

    existing_link = session.execute(
        select(mapping_table)
        .where(mapping_table.c["user_id"] == user_id)
        .where(mapping_table.c["project_id"] == project_pk_value)
        .limit(1)
    ).first()

    if existing_link:
        stats["owner_links_existing"] += 1
        return

    session.execute(
        mapping_table.insert().values(
            {
                "user_id": user_id,
                "project_id": project_pk_value,
            }
        )
    )
    stats["owner_links_inserted"] += 1


def ingest_static_projects(
    database_url: str,
    projects_dir: str | Path,
    schema: str = "public",
    dry_run: bool = False,
) -> dict:
    projects = discover_static_projects(projects_dir)
    stats = {
        "projects_seen": len(projects),
        "projects_inserted": 0,
        "projects_existing": 0,
        "projects_failed": 0,
        "projects_skipped": 0,
        "owner_links_inserted": 0,
        "owner_links_existing": 0,
        "owner_pending_users": 0,
        "owner_missing_in_config": 0,
        "owner_linking_unavailable": 0,
        "owner_immutable_skipped": 0,
    }

    if not projects:
        logger.info("No static projects discovered; nothing to ingest")
        return stats

    engine = create_engine(database_url)
    tables = _load_tables(engine, schema=schema)
    projects_table = tables["projects"]
    users_table = tables["users"]

    with Session(engine) as session:
        for project in projects:
            try:
                with session.begin_nested():
                    project_already_existed = False
                    existing_row = _find_project_by_vertex_id(session, projects_table, project["project_id"])
                    if existing_row is None:
                        owner_email = project.get("project_owner")
                        if not owner_email:
                            raise RuntimeError("config_file.json is missing project_owner")

                        owner_row = _find_user_by_email(session, users_table, owner_email)
                        if owner_row is None:
                            raise RuntimeError(f"owner user does not exist for {owner_email}")

                        if not dry_run:
                            insert_values = _build_project_insert_values(project, projects_table, owner_row["id"])
                            session.execute(projects_table.insert().values(insert_values))
                            existing_row = _find_project_by_vertex_id(session, projects_table, project["project_id"])
                        stats["projects_inserted"] += 1
                    else:
                        stats["projects_existing"] += 1
                        project_already_existed = True

                    if existing_row is None:
                        stats["projects_skipped"] += 1
                        logger.warning(
                            "Could not resolve projects row after insert attempt; skipping owner mapping for "
                            f"project_id={project['project_id']}"
                        )
                        continue

                    if not dry_run:
                        if project_already_existed and _project_has_owner_mapping(session, tables, existing_row):
                            stats["owner_immutable_skipped"] += 1
                            continue
                        _try_link_owner(session, tables, existing_row, project.get("project_owner"), stats)
            except (RuntimeError, SQLAlchemyError) as exc:
                stats["projects_failed"] += 1
                logger.error(_project_failure_message(project, exc))
                continue

        if dry_run:
            session.rollback()
        else:
            session.commit()

    logger.info(
        "Static project ingestion summary: "
        f"seen={stats['projects_seen']} inserted={stats['projects_inserted']} existing={stats['projects_existing']} "
        f"failed={stats['projects_failed']} owner_links_inserted={stats['owner_links_inserted']} "
        f"owner_links_existing={stats['owner_links_existing']} owner_pending_users={stats['owner_pending_users']} "
        f"owner_immutable_skipped={stats['owner_immutable_skipped']}"
    )
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ingest static VERTEX projects into auth/access database tables")
    parser.add_argument(
        "--projects-dir",
        default=None,
        help="Directory containing static project folders (defaults to VERTEX_PROJECTS_DIR or projects/)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database URL override. Defaults to vertex configuration resolution.",
    )
    parser.add_argument("--schema", default="public", help="Database schema name for projects/users mapping tables")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to the database")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    projects_dir = Path(args.projects_dir or os.getenv("VERTEX_PROJECTS_DIR") or "projects").resolve()

    database_url = args.database_url or get_database_url()

    try:
        ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema=args.schema, dry_run=args.dry_run)
        return 0
    except (RuntimeError, SQLAlchemyError) as exc:
        logger.error(f"Static project ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
