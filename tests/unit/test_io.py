import json

import pandas as pd
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


def test_get_project_name_reads_name_from_valid_config(tmp_path):
    project_dir = tmp_path / "named-project"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text(json.dumps({"project_name": "Human Friendly Name"}) + "\n")

    assert vertex_io.get_project_name(project_dir) == "Human Friendly Name"


def test_get_project_name_defaults_to_folder_name_when_missing_config(tmp_path):
    project_dir = tmp_path / "no-config-project"
    project_dir.mkdir()

    assert vertex_io.get_project_name(project_dir) == project_dir.name


def test_get_project_name_defaults_to_folder_name_when_config_invalid_json(tmp_path):
    project_dir = tmp_path / "bad-config-project"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text("{ not json")

    assert vertex_io.get_project_name(project_dir) == project_dir.name


def test_get_project_name_defaults_to_folder_name_when_project_name_missing(tmp_path):
    project_dir = tmp_path / "missing-name-project"
    project_dir.mkdir()
    (project_dir / "config_file.json").write_text(json.dumps({"project_owner": "owner@example.com"}) + "\n")

    assert vertex_io.get_project_name(project_dir) == project_dir.name


def test_save_public_outputs_writes_expected_files_and_overwrites_existing_outputs(tmp_path, monkeypatch):
    project_dir = tmp_path / "prebuilt-output-project"
    project_dir.mkdir()
    outputs_dir = project_dir / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "stale.txt").write_text("old")

    expected_buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text"]}]
    captured = {}

    def fake_get_visuals(buttons, insight_panels, **kwargs):
        captured["buttons"] = buttons
        captured["insight_panels"] = insight_panels
        captured["filepath"] = kwargs["filepath"]
        return expected_buttons

    monkeypatch.setattr(vertex_io, "get_visuals", fake_get_visuals)
    monkeypatch.setattr(vertex_io, "_VERTEX_GIT_METADATA", None)
    monkeypatch.setattr(vertex_io.subprocess, "check_output", lambda *args, **kwargs: "abcd1234\n")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(vertex_io.time, "ctime", lambda: "Mon Mar  2 12:00:00 2026")

    df_countries = pd.DataFrame([{"country_iso": "GBR", "country": "United Kingdom", "records": 1}])

    config = {
        "outputs_path": "outputs/",
        "insight_panels": ["panel_a"],
        "project_name": "Project",
        "project_id": "project-id",
        "project_owner": "owner@example.com",
        "is_public": True,
        "map_layout_center_latitude": 6,
        "map_layout_center_longitude": -75,
        "map_layout_zoom": 1.7,
    }

    vertex_io.save_public_outputs(
        buttons=[{"suffix": "panel_a"}],
        insight_panels={"panel_a": object()},
        df_map=pd.DataFrame(),
        df_countries=df_countries,
        df_forms_dict={},
        dictionary=pd.DataFrame(),
        quality_report={},
        project_path=str(project_dir),
        config_dict=config,
    )

    assert not (outputs_dir / "stale.txt").exists()
    assert (outputs_dir / "panel_a").is_dir()
    assert captured["filepath"].endswith("outputs/")

    metadata = json.loads((outputs_dir / "dashboard_metadata.json").read_text())
    assert metadata == {"insight_panels": expected_buttons}

    countries = pd.read_csv(outputs_dir / "dashboard_data.csv")
    assert list(countries.columns) == ["country_iso", "country", "records"]
    assert countries.iloc[0]["country_iso"] == "GBR"

    saved_config = json.loads((outputs_dir / "config_file.json").read_text())
    assert saved_config["project_name"] == "Project"
    assert saved_config["project_id"] == "project-id"
    assert saved_config["runtime_metadata"] == {
        "user": "tester",
        "timestamp": "Mon Mar  2 12:00:00 2026",
        "vertex_commit_sha": "abcd1234",
    }
    assert saved_config["vertex_commit_sha"] == "abcd1234"


def test_get_vertex_git_metadata_falls_back_to_env_sha(monkeypatch):
    def raise_git_error(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(vertex_io, "_VERTEX_GIT_METADATA", None)
    monkeypatch.setattr(vertex_io.subprocess, "check_output", raise_git_error)
    monkeypatch.setenv("VERTEX_GIT_SHA", "abcd1234")

    assert vertex_io._get_vertex_git_metadata() == {"commit_sha": "abcd1234"}


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, -1),
        (True, True),
        ("<b>Value</b>", "Value"),
        ("<b><i>Value</i></b>", "Value"),
    ],
)
def test_strip_html(value, expected):
    assert vertex_io.strip_html(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, -1),
        (True, True),
        ("<b>Value</b>", "<b>Value</b>"),
        ("<b><i>Value</i></b>", "<b><i>Value</i></b>"),
        ("↳ Value", " Value"),
    ],
)
def test_strip_nonstandard_unicode_chars(value, expected):
    assert vertex_io.strip_nonstandard_unicode_chars(value) == expected
