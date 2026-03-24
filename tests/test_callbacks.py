from types import SimpleNamespace

import pandas as pd
import pytest
from dash import html
from dash.exceptions import PreventUpdate
from plotly.graph_objs import Figure

from vertex import descriptive_dashboard as dashboard


def _get_wrapped_callback(name):
    for item in dashboard.app.callback_map.values():
        callback = item.get("callback")
        wrapped = getattr(callback, "__wrapped__", callback)
        if wrapped.__name__ == name:
            return wrapped
    raise KeyError(f"Callback not found: {name}")


@pytest.fixture(autouse=True)
def clear_project_cache():
    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()
    yield
    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()


def test_set_project_from_url_updates_selected_project(monkeypatch):
    callback = _get_wrapped_callback("set_project_from_url")
    visible_projects = [
        {
            "path": "/tmp/analysis-a/",
            "name": "Analysis A",
            "project_id": "analysis-a",
            "project_type": "analysis",
            "data_source": "files",
            "is_public": True,
        }
    ]
    monkeypatch.setattr(dashboard, "get_projects_catalog", lambda: visible_projects)

    selected = callback("?project=analysis-a", "/tmp/other/", True)

    assert selected == "/tmp/analysis-a/"


def test_set_project_from_url_raises_prevent_update_for_unknown_project(monkeypatch):
    callback = _get_wrapped_callback("set_project_from_url")
    visible_projects = [
        {
            "path": "/tmp/analysis-a/",
            "name": "Analysis A",
            "project_id": "analysis-a",
            "project_type": "analysis",
            "data_source": "files",
            "is_public": True,
        }
    ]
    monkeypatch.setattr(dashboard, "get_projects_catalog", lambda: visible_projects)

    with pytest.raises(PreventUpdate):
        callback("?project=missing", "/tmp/other/", True)


def test_update_url_for_project_uses_project_id(monkeypatch):
    callback = _get_wrapped_callback("update_url_for_project")
    catalog = [
        {
            "path": "/tmp/analysis-a/",
            "name": "Analysis A",
            "project_id": "analysis-a",
            "project_type": "analysis",
            "data_source": "files",
            "is_public": True,
        }
    ]
    monkeypatch.setattr(dashboard, "get_projects_catalog", lambda: catalog)

    url_query = callback("analysis-a", True)

    assert url_query == "?project=analysis-a"


def test_update_country_selection_select_all(monkeypatch):
    callback = _get_wrapped_callback("update_country_selection")
    monkeypatch.setattr(
        dashboard.dash,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "country-selectall.value"}]),
    )

    output = callback(["all"], ["GBR"], [{"label": "UK", "value": "GBR"}, {"label": "USA", "value": "USA"}])

    assert output == [
        ["all"],
        [{"label": "Unselect all", "value": "all"}],
        ["GBR", "USA"],
    ]


def test_update_country_selection_handles_initial_empty_trigger(monkeypatch):
    callback = _get_wrapped_callback("update_country_selection")
    monkeypatch.setattr(
        dashboard.dash,
        "callback_context",
        SimpleNamespace(triggered=[]),
    )

    output = callback([], ["GBR"], [{"label": "UK", "value": "GBR"}])

    assert output == [["all"], [{"label": "Unselect all", "value": "all"}], ["GBR"]]


def test_update_country_selection_handles_none_values(monkeypatch):
    callback = _get_wrapped_callback("update_country_selection")
    monkeypatch.setattr(
        dashboard.dash,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "country-selectall.value"}]),
    )

    output = callback(None, None, None)

    assert output == [[], [{"label": "Select all", "value": "all"}], []]


def test_update_map_returns_empty_figure_for_empty_filtered_cohort(monkeypatch):
    callback = _get_wrapped_callback("update_map")
    project_path = "/tmp/analysis-a/"
    dashboard.set_project_data(project_path, {"mode": "analysis", "df_map": pd.DataFrame({"subjid": [1]})})
    monkeypatch.setattr(dashboard, "filter_df_map", lambda *args, **kwargs: pd.DataFrame())

    fig = callback(
        ["Female"],
        [0, 100],
        ["GBR"],
        [0, 1],
        {"0": "2020-01-01", "1": "2020-01-02"},
        ["Death"],
        project_path,
        {"map_zoom": 1.7},
    )

    assert isinstance(fig, Figure)
    assert len(fig.data) == 1


def test_open_and_load_modal_prebuilt_uses_cached_button(monkeypatch):
    callback = _get_wrapped_callback("open_and_load_modal")
    project_path = "/tmp/prebuilt-a/"
    captured = {}

    class DummyPanel:
        def create_visuals(self, **kwargs):
            captured["visuals_kwargs"] = kwargs
            return [("fig", "id", "label", "about")]

    panel = DummyPanel()
    dashboard.set_project_data(
        project_path,
        {
            "mode": "prebuilt",
            "insight_panels": {"panel_a": panel},
            "buttons": [{"suffix": "panel_a", "item": "Item", "label": "Label"}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": '{"type":"open-modal","index":"panel_a"}.n_clicks'}]),
    )

    def fake_create_modal(visuals, button, filter_options=None):
        captured["visuals"] = visuals
        captured["button"] = button
        captured["filter_options"] = filter_options
        return "modal-content"

    monkeypatch.setattr(dashboard, "create_modal", fake_create_modal)

    is_open, modal_content, button = callback([1], project_path)

    assert is_open is True
    assert modal_content == "modal-content"
    assert button["suffix"] == "panel_a"
    assert captured["visuals_kwargs"] == {"suffix": "panel_a", "filepath": project_path}
    assert captured["filter_options"] is None


def test_open_and_load_modal_malformed_trigger_payload_prevent_update(monkeypatch):
    callback = _get_wrapped_callback("open_and_load_modal")
    monkeypatch.setattr(
        dashboard,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "not-json.n_clicks"}]),
    )

    with pytest.raises(PreventUpdate):
        callback([1], "/tmp/prebuilt-a/")


def test_update_country_selection_modal_handles_initial_empty_trigger(monkeypatch):
    callback = _get_wrapped_callback("update_country_selection_modal")
    monkeypatch.setattr(
        dashboard.dash,
        "callback_context",
        SimpleNamespace(triggered=[]),
    )

    output = callback([], ["GBR"], [{"label": "UK", "value": "GBR"}])

    assert output == [["all"], [{"label": "Unselect all", "value": "all"}], ["GBR"]]


def test_update_country_selection_modal_handles_none_values(monkeypatch):
    callback = _get_wrapped_callback("update_country_selection_modal")
    monkeypatch.setattr(
        dashboard.dash,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "country-selectall-modal.value"}]),
    )

    output = callback(None, None, None)

    assert output == [[], [{"label": "Select all", "value": "all"}], []]


def test_update_figures_returns_blank_modal_when_filtered_data_empty(monkeypatch):
    callback = _get_wrapped_callback("update_figures")
    project_path = "/tmp/analysis-a/"

    class DummyPanel:
        def create_visuals(self, **kwargs):
            return [("fig", "id", "label", "about")]

    dashboard.set_project_data(
        project_path,
        {
            "mode": "analysis",
            "df_map": pd.DataFrame({"subjid": [1], "demog_age": [25]}),
            "df_forms_dict": {"presentation": pd.DataFrame({"subjid": [1], "demog_age": [25]})},
            "dictionary": pd.DataFrame(),
            "quality_report": {},
            "insight_panels": {"panel_a": DummyPanel()},
        },
    )
    monkeypatch.setattr(dashboard, "filter_df_map", lambda *args, **kwargs: pd.DataFrame())

    result = callback(
        1,
        {"suffix": "panel_a"},
        ["Female"],
        [0, 100],
        ["GBR"],
        [0, 1],
        {"0": "2020-01-01", "1": "2020-01-02"},
        ["Death"],
        project_path,
    )

    assert result == ((), ["Female"], [0, 100], ["GBR"], [0, 1], ["Death"])


def test_update_figures_prebuilt_mode_gracefully_prevent_update():
    callback = _get_wrapped_callback("update_figures")
    project_path = "/tmp/prebuilt-a/"
    dashboard.set_project_data(project_path, {"mode": "prebuilt"})

    with pytest.raises(PreventUpdate):
        callback(
            1,
            {"suffix": "panel_a"},
            ["Female"],
            [0, 100],
            ["GBR"],
            [0, 1],
            {"0": "2020-01-01", "1": "2020-01-02"},
            ["Death"],
            project_path,
        )


def test_update_figures_rebuilds_modal_when_filtered_data_present(monkeypatch):
    callback = _get_wrapped_callback("update_figures")
    project_path = "/tmp/analysis-a/"
    captured = {}

    class DummyPanel:
        def create_visuals(self, **kwargs):
            captured["visuals_kwargs"] = kwargs
            return [("fig", "id", "label", "about")]

    dashboard.set_project_data(
        project_path,
        {
            "mode": "analysis",
            "df_map": pd.DataFrame({"subjid": [1], "demog_age": [25]}),
            "df_forms_dict": {"presentation": pd.DataFrame({"subjid": [1], "demog_age": [25]})},
            "dictionary": pd.DataFrame({"field_name": ["demog_age"]}),
            "quality_report": {"ok": True},
            "insight_panels": {"panel_a": DummyPanel()},
        },
    )
    monkeypatch.setattr(dashboard, "filter_df_map", lambda df, *args, **kwargs: df.copy())
    monkeypatch.setattr(dashboard, "get_filter_options", lambda df_map: {"sex": ["Female", "Male"]})

    def fake_create_modal(visuals, button, filter_options=None):
        captured["visuals"] = visuals
        captured["button"] = button
        captured["filter_options"] = filter_options
        return "modal-after-refresh"

    monkeypatch.setattr(dashboard, "create_modal", fake_create_modal)

    result = callback(
        1,
        {"suffix": "panel_a"},
        ["Female"],
        [0, 100],
        ["GBR"],
        [0, 1],
        {"0": "2020-01-01", "1": "2020-01-02"},
        ["Death"],
        project_path,
    )

    assert result[0] == "modal-after-refresh"
    assert captured["button"] == {"suffix": "panel_a"}
    assert captured["filter_options"] == {"sex": ["Female", "Male"]}
    assert captured["visuals_kwargs"]["suffix"] == "panel_a"
    assert captured["visuals_kwargs"]["filepath"] == project_path


def test_handle_login_logout_db_error_returns_graceful_message(monkeypatch):
    callback = _get_wrapped_callback("handle_login_logout")
    monkeypatch.setattr(
        dashboard,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "login-submit.n_clicks"}]),
    )

    class BrokenSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("db unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(dashboard, "Session", BrokenSession)

    result = callback(1, 1, 0, True, "user@example.com", "secret")

    assert result == (False, True, "Login service unavailable. Please try again.")


def test_handle_register_db_error_returns_graceful_message(monkeypatch):
    callback = _get_wrapped_callback("handle_register")
    monkeypatch.setattr(dashboard, "callback_context", SimpleNamespace(triggered_id="register-submit"))

    class BrokenSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("db unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(dashboard, "Session", BrokenSession)

    result = callback(0, 1, True, "user@example.com", "secret", "secret")

    assert result == ("Registration service unavailable. Please try again.", True)


def test_update_country_display_handles_missing_options():
    callback = _get_wrapped_callback("update_country_display")

    result = callback(["GBR"], None)

    assert isinstance(result, html.Div)
    assert result.children[-1] == "GBR"


def test_update_country_display_modal_handles_missing_options():
    callback = _get_wrapped_callback("update_country_display_modal")

    result = callback(["GBR"], None)

    assert result == "Country: GBR"
