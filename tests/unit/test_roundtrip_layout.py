import json

from dash import html

from vertex import descriptive_dashboard as dashboard


def _write_roundtrip_panel(project_dir):
    panel_code = """
import pandas as pd

import isaricanalytics.IsaricDraw as idw


def define_button():
    return {"item": "Presentation", "label": "Summary"}


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    data = pd.DataFrame({"paragraphs": [f"Rows: {len(df_map)}"]})
    visual = idw.fig_text(
        data,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_id="fig_text",
        graph_label="Summary",
        graph_about="Round-trip fixture",
    )
    return (visual,)
"""
    panel_path = project_dir / "insight_panels" / "presentation.py"
    panel_path.write_text(panel_code.strip() + "\n")


def test_analysis_project_roundtrip_outputs_reload_as_prebuilt(copy_fixture_project, monkeypatch):
    project_dir = copy_fixture_project("analysis_files_project", "analysis-roundtrip")
    project_path = str(project_dir)
    outputs_path = project_dir / "outputs"

    _write_roundtrip_panel(project_dir)
    config_file = project_dir / "config_file.json"
    config = json.loads(config_file.read_text())
    config["save_outputs"] = True
    config["outputs_path"] = "outputs/"
    config_file.write_text(json.dumps(config, indent=2) + "\n")

    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()
    monkeypatch.setenv("VERTEX_ENABLE_SAVE_OUTPUTS", "true")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 100.0)
    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "analysis")

    analysis_loaded = dashboard.load_project_data(project_path)
    assert analysis_loaded["mode"] == "analysis"
    assert outputs_path.exists()

    prebuilt_path = str(outputs_path)
    dashboard.clear_project_data(project_path)
    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, prebuilt_path, "prebuilt")

    prebuilt_loaded = dashboard.load_project_data(prebuilt_path)

    assert prebuilt_loaded["mode"] == "prebuilt"
    assert len(prebuilt_loaded["buttons"]) == 1
    assert "presentation" in prebuilt_loaded["insight_panels"]

    visuals = prebuilt_loaded["insight_panels"]["presentation"].create_visuals()
    assert len(visuals) == 1
    assert visuals[0][1] == "presentation/fig_text"
    assert visuals[0][2] == "Summary"


def test_build_project_layout_analysis_smoke(copy_fixture_project, monkeypatch):
    project_dir = copy_fixture_project("analysis_files_project", "analysis-layout")
    project_path = str(project_dir)
    _write_roundtrip_panel(project_dir)

    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()
    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "analysis")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 100.0)

    project_catalog = [
        {
            "path": project_path,
            "name": "Analysis Layout",
            "project_id": "analysis-layout",
            "project_type": "analysis",
            "data_source": "files",
            "is_public": True,
        }
    ]

    layout = dashboard.build_project_layout(project_path, project_catalog, login_state=True)

    assert isinstance(layout, html.Div)


def test_build_project_layout_prebuilt_empty_countries_smoke(prebuilt_project_factory, monkeypatch):
    project_dir = prebuilt_project_factory()
    project_path = str(project_dir)
    (project_dir / "dashboard_data.csv").unlink()

    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()
    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "prebuilt")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 100.0)

    project_catalog = [
        {
            "path": project_path,
            "name": "Prebuilt Layout",
            "project_id": "prebuilt-layout",
            "project_type": "prebuilt",
            "data_source": "files",
            "is_public": True,
        }
    ]

    layout = dashboard.build_project_layout(project_path, project_catalog, login_state=False)

    assert isinstance(layout, html.Div)
