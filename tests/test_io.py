import json

import pytest

from vertex import descriptive_dashboard as dashboard
from vertex import io as vertex_io


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        (None, True),
    ],
)
def test_normalise_is_public(value, expected):
    assert vertex_io._normalise_is_public(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("y", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("n", False),
    ],
)
def test_as_bool_string_values(value, expected):
    assert vertex_io._as_bool(value, default=False) is expected


def test_should_save_outputs_requires_env_opt_in(monkeypatch):
    config = {"save_outputs": True}
    monkeypatch.delenv("VERTEX_ENABLE_SAVE_OUTPUTS", raising=False)
    assert vertex_io.should_save_outputs(config) is False

    monkeypatch.setenv("VERTEX_ENABLE_SAVE_OUTPUTS", "true")
    assert vertex_io.should_save_outputs(config) is True

    monkeypatch.setenv("VERTEX_ENABLE_SAVE_OUTPUTS", "0")
    assert vertex_io.should_save_outputs(config) is False


def test_get_project_record_analysis_data_source(tmp_path):
    project_dir = tmp_path / "analysis-proj"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text(
        json.dumps(
            {
                "project_name": "My Project",
                "project_id": "project-123",
                "project_owner": "Owner@Example.COM",
                "is_public": "false",
                "api_url": "https://example.test/redcap/api/",
                "api_key": "abc123",
            }
        )
    )

    record = vertex_io.get_project_record(project_dir, "analysis")

    assert record["name"] == "My Project"
    assert record["project_id"] == "project-123"
    assert record["project_owner"] == "owner@example.com"
    assert record["is_public"] is False
    assert record["project_type"] == "analysis"
    assert record["data_source"] == "api"
    assert record["path"].endswith("/analysis-proj/")


def test_get_project_record_blank_project_id_returns_none_and_logs_warning(tmp_path, caplog):
    project_dir = tmp_path / "blank-project-id"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text(
        json.dumps(
            {
                "project_name": "Blank Project ID",
                "project_id": "   ",
                "project_owner": "owner@example.com",
                "is_public": True,
            }
        )
    )

    with caplog.at_level("WARNING"):
        record = vertex_io.get_project_record(project_dir, "prebuilt")

    assert record["project_id"] is None
    assert "Invalid project_id" in caplog.text


def test_get_projects_catalog_reads_both_roots(tmp_path, monkeypatch):
    demo_root = tmp_path / "demo-projects"
    static_root = tmp_path / "projects"
    demo_root.mkdir()
    static_root.mkdir()

    demo_project = demo_root / "demo-a"
    demo_project.mkdir()
    (demo_project / "config_file.json").write_text(
        json.dumps(
            {
                "project_name": "Demo A",
                "project_id": "demo-a",
                "project_owner": "demo@example.com",
                "is_public": True,
            }
        )
    )

    static_project = static_root / "static-a"
    static_project.mkdir()
    (static_project / "config_file.json").write_text(
        json.dumps(
            {
                "project_name": "Static A",
                "project_id": "static-a",
                "project_owner": "static@example.com",
                "is_public": False,
            }
        )
    )

    monkeypatch.setattr(vertex_io, "get_demo_projects_root", lambda: demo_root)
    monkeypatch.setattr(vertex_io, "get_static_projects_root", lambda: static_root)

    catalog = vertex_io.get_projects_catalog()
    by_name = {item["name"]: item for item in catalog}

    assert len(catalog) == 2
    assert by_name["Demo A"]["project_type"] == "analysis"
    assert by_name["Static A"]["project_type"] == "prebuilt"


def test_get_config_discovers_insight_panels(analysis_project_dir):
    config = vertex_io.get_config(str(analysis_project_dir), dict(vertex_io.config_defaults))
    assert config["insight_panels"] == ["presentation"]


def test_load_vertex_from_files_parses_dates_and_forms(analysis_project_dir):
    loaded = vertex_io.load_vertex_from_files(str(analysis_project_dir), {"insight_panels_data_path": "analysis_data/"})

    assert "df_map" in loaded and "dictionary" in loaded and "df_forms_dict" in loaded
    assert str(loaded["df_map"]["pres_date"].dtype).startswith("datetime64")
    assert "presentation" in loaded["df_forms_dict"]
    assert "invalid_no_subjid" not in loaded["df_forms_dict"]


def test_get_config_static_missing_dashboard_metadata_does_not_raise(copy_fixture_project):
    project_dir = copy_fixture_project("prebuilt_public_project", "prebuilt-missing-dashboard-metadata")
    (project_dir / "dashboard_metadata.json").unlink()

    loaded_config = vertex_io.get_config(str(project_dir), dict(vertex_io.config_defaults))

    assert isinstance(loaded_config, dict)
    assert "insight_panels_path" not in loaded_config


def test_get_project_record_invalid_json_falls_back_to_directory_name(tmp_path):
    project_dir = tmp_path / "broken-config-project"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text("{ this is not valid json")

    record = vertex_io.get_project_record(project_dir, "prebuilt")

    assert record["name"] == project_dir.name
    assert record["project_type"] == "prebuilt"


def test_load_vertex_from_files_missing_dictionary_should_not_crash(copy_fixture_project):
    project_dir = copy_fixture_project("analysis_files_project", "analysis-missing-dictionary")
    (project_dir / "analysis_data" / "vertex_dictionary.csv").unlink()

    loaded = vertex_io.load_vertex_from_files(str(project_dir), {"insight_panels_data_path": "analysis_data/"})

    # Desired fallback once hardened:
    assert loaded["dictionary"].empty
    assert "df_map" in loaded


def test_load_vertex_from_files_malformed_dictionary_schema_should_not_crash(copy_fixture_project):
    project_dir = copy_fixture_project("analysis_files_project", "analysis-malformed-dictionary")
    bad_dictionary = "field_name,field_label\nsubjid,Subject ID\npres_date,Admission Date\n"
    (project_dir / "analysis_data" / "vertex_dictionary.csv").write_text(bad_dictionary)

    loaded = vertex_io.load_vertex_from_files(str(project_dir), {"insight_panels_data_path": "analysis_data/"})

    # Desired fallback once hardened:
    assert "dictionary" in loaded
    assert "df_map" in loaded


def test_get_config_malformed_analysis_config_should_apply_defaults(tmp_path):
    project_dir = tmp_path / "analysis-bad-config"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text("{ this is not valid json")
    (project_dir / "insight_panels").mkdir()
    (project_dir / "insight_panels" / "panel_a.py").write_text("def define_button():\n    return {}\n")

    config = vertex_io.get_config(str(project_dir), dict(vertex_io.config_defaults))

    # Desired behavior once hardened:
    assert "insight_panels" in config
    assert isinstance(config["insight_panels"], list)


def test_load_project_data_analysis_without_insight_panels_path_should_remain_analysis(copy_fixture_project, monkeypatch):
    project_dir = copy_fixture_project("analysis_files_project", "analysis-missing-panels-path")
    config_file = project_dir / "config_file.json"
    config = json.loads(config_file.read_text())
    config.pop("insight_panels_path", None)
    config_file.write_text(json.dumps(config, indent=2) + "\n")

    project_path = str(project_dir)
    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()
    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "analysis")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 100.0)

    loaded = dashboard.load_project_data(project_path)

    # Desired behavior once hardened:
    assert loaded["mode"] == "analysis"
