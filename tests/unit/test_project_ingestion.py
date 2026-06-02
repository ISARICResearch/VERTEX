import json
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, String, Table, UniqueConstraint, create_engine, select
from sqlalchemy.orm import Session

from vertex.project_ingestion import ingest_static_projects


def _write_project(root: Path, folder: str, project_id: str, name: str, owner: str, is_public: bool = True):
    project_dir = root / folder
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "config_file.json").write_text(
        json.dumps(
            {
                "project_id": project_id,
                "project_name": name,
                "project_owner": owner,
                "is_public": is_public,
            },
            indent=2,
        )
        + "\n"
    )


def _prepare_schema(database_url: str):
    engine = create_engine(database_url)
    metadata = MetaData()

    users = Table(
        "users",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("email", String, unique=True, nullable=False),
        Column("is_admin", Boolean, nullable=False, default=False),
        Column("created", DateTime(timezone=True), nullable=True),
        Column("updated", DateTime(timezone=True), nullable=True),
        Column("last_login", DateTime(timezone=True), nullable=True),
    )

    projects = Table(
        "projects",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("vertex_id", String, unique=True, nullable=False),
        Column("owner_id", Integer, nullable=False),
        Column("is_public", Boolean, nullable=False),
        Column("created", DateTime(timezone=True), nullable=True),
        Column("updated", DateTime(timezone=True), nullable=True),
    )

    mapping = Table(
        "user_project_mapping",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", Integer, nullable=False),
        Column("project_id", Integer, nullable=False),
        Column("created", DateTime(timezone=True), nullable=True),
        Column("updated", DateTime(timezone=True), nullable=True),
        UniqueConstraint("user_id", "project_id", name="uq_user_project_mapping"),
    )

    metadata.create_all(engine)
    return engine, users, projects, mapping


def test_ingest_static_projects_inserts_new_rows_and_maps_existing_owners(tmp_path):
    projects_dir = tmp_path / "vertex-projects"
    projects_dir.mkdir()
    _write_project(projects_dir, "proj-a", "vertex-a", "Vertex A", "owner-a@example.com")
    _write_project(projects_dir, "proj-b", "vertex-b", "Vertex B", "owner-b@example.com")

    database_url = f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite'}"
    engine, users, projects, mapping = _prepare_schema(database_url)

    with Session(engine) as session:
        session.execute(users.insert().values(email="owner-a@example.com", is_admin=False))
        session.execute(users.insert().values(email="owner-b@example.com", is_admin=False))
        session.commit()

    stats = ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema="public")

    assert stats["projects_seen"] == 2
    assert stats["projects_inserted"] == 2
    assert stats["projects_existing"] == 0
    assert stats["owner_links_inserted"] == 2
    assert stats["owner_pending_users"] == 0

    with Session(engine) as session:
        db_projects = session.execute(select(projects.c.vertex_id)).scalars().all()
        assert sorted(db_projects) == ["vertex-a", "vertex-b"]

        mapping_rows = session.execute(select(mapping.c.user_id, mapping.c.project_id)).all()
        assert len(mapping_rows) == 2


def test_ingest_static_projects_skips_completely_invalid_json_and_exits_early(tmp_path):
    projects_dir = tmp_path / "vertex-projects"
    projects_dir.mkdir()
    broken_project_dir = projects_dir / "proj-bad"
    broken_project_dir.mkdir()
    (broken_project_dir / "config_file.json").write_text("{ this is not valid json")

    database_url = f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite'}"
    engine, users, projects, mapping = _prepare_schema(database_url)

    stats = ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema="public")

    assert stats["projects_seen"] == 0
    assert stats["projects_inserted"] == 0
    assert stats["projects_existing"] == 0

    with Session(engine) as session:
        assert session.execute(select(users.c.id)).first() is None
        assert session.execute(select(projects.c.id)).first() is None
        assert session.execute(select(mapping.c.id)).first() is None


def test_ingest_static_projects_logs_failure_when_owner_user_is_missing_for_new_project(tmp_path):
    projects_dir = tmp_path / "vertex-projects"
    projects_dir.mkdir()
    _write_project(projects_dir, "proj-a", "vertex-a", "Vertex A", "owner-a@example.com")

    database_url = f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite'}"
    engine, users, projects, mapping = _prepare_schema(database_url)

    stats = ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema="public")

    assert stats["projects_seen"] == 1
    assert stats["projects_failed"] == 1
    assert stats["projects_inserted"] == 0

    with Session(engine) as session:
        assert session.execute(select(projects.c.id)).first() is None
        assert session.execute(select(mapping.c.id)).first() is None
        assert session.execute(select(users.c.id)).first() is None


def test_ingest_static_projects_continues_after_project_failure(tmp_path):
    projects_dir = tmp_path / "vertex-projects"
    projects_dir.mkdir()
    _write_project(projects_dir, "proj-bad", "vertex-bad", "Vertex Bad", "missing-owner@example.com")
    _write_project(projects_dir, "proj-good", "vertex-good", "Vertex Good", "owner-good@example.com")

    database_url = f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite'}"
    engine, users, projects, mapping = _prepare_schema(database_url)

    with Session(engine) as session:
        session.execute(users.insert().values(email="owner-good@example.com", is_admin=False))
        session.commit()

    stats = ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema="public")

    assert stats["projects_seen"] == 2
    assert stats["projects_failed"] == 1
    assert stats["projects_inserted"] == 1
    assert stats["owner_links_inserted"] == 1

    with Session(engine) as session:
        db_projects = session.execute(select(projects.c.vertex_id)).scalars().all()
        assert db_projects == ["vertex-good"]

        mapping_rows = session.execute(select(mapping.c.user_id, mapping.c.project_id)).all()
        assert len(mapping_rows) == 1


def test_ingest_static_projects_keeps_existing_owner_mapping_immutable(tmp_path):
    projects_dir = tmp_path / "vertex-projects"
    projects_dir.mkdir()
    _write_project(projects_dir, "proj-a", "vertex-a", "Vertex A", "owner-a@example.com")

    database_url = f"sqlite+pysqlite:///{tmp_path / 'auth.sqlite'}"
    engine, users, projects, mapping = _prepare_schema(database_url)

    with Session(engine) as session:
        session.execute(users.insert().values(email="owner-a@example.com", is_admin=False))
        session.execute(users.insert().values(email="owner-b@example.com", is_admin=False))
        session.commit()

    first_stats = ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema="public")
    assert first_stats["owner_links_inserted"] == 1

    _write_project(projects_dir, "proj-a", "vertex-a", "Vertex A", "owner-b@example.com")
    second_stats = ingest_static_projects(database_url=database_url, projects_dir=projects_dir, schema="public")
    assert second_stats["projects_existing"] == 1
    assert second_stats["owner_links_inserted"] == 0
    assert second_stats["owner_immutable_skipped"] == 1

    with Session(engine) as session:
        project_pk = session.execute(select(projects.c.id).where(projects.c.vertex_id == "vertex-a")).scalar_one()
        mapping_rows = session.execute(
            select(mapping.c.user_id, mapping.c.project_id).where(mapping.c.project_id == project_pk)
        ).all()
        assert len(mapping_rows) == 1
