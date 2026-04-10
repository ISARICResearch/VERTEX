import pytest

from vertex import descriptive_dashboard as dashboard


@pytest.fixture(autouse=True)
def clear_project_cache():
    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()
    yield
    dashboard.PROJECT_CACHE.clear()
    dashboard.PROJECT_CACHE_VERSION.clear()
    dashboard.PROJECT_TYPE_BY_PATH.clear()


def test_load_project_data_prebuilt_mode_returns_expected_shape(prebuilt_project_factory, monkeypatch):
    project_dir = prebuilt_project_factory()
    project_path = str(project_dir)

    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "prebuilt")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 100.0)

    loaded = dashboard.load_project_data(project_path)

    assert loaded["mode"] == "prebuilt"
    assert loaded["df_map"] is None
    assert loaded["df_forms_dict"] is None
    assert loaded["dictionary"] is None
    assert loaded["quality_report"] is None
    assert not loaded["df_countries"].empty
    assert "panel_a" in loaded["insight_panels"]
    assert isinstance(loaded["buttons"], list) and len(loaded["buttons"]) == 1


def test_load_project_data_prebuilt_uses_cache_for_unchanged_version(prebuilt_project_factory, monkeypatch):
    project_dir = prebuilt_project_factory()
    project_path = str(project_dir)

    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "prebuilt")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 42.0)

    first_load = dashboard.load_project_data(project_path)
    second_load = dashboard.load_project_data(project_path)

    assert first_load is second_load


def test_load_project_data_prebuilt_reloads_when_version_changes(prebuilt_project_factory, monkeypatch):
    project_dir = prebuilt_project_factory()
    project_path = str(project_dir)

    versions = iter([10.0, 20.0])
    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "prebuilt")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: next(versions))

    first_load = dashboard.load_project_data(project_path)
    second_load = dashboard.load_project_data(project_path)

    assert first_load is not second_load


def test_load_project_data_prebuilt_missing_dashboard_data_should_not_crash(prebuilt_project_factory, monkeypatch):
    project_dir = prebuilt_project_factory()
    project_path = str(project_dir)
    (project_dir / "dashboard_data.csv").unlink()

    monkeypatch.setitem(dashboard.PROJECT_TYPE_BY_PATH, project_path, "prebuilt")
    monkeypatch.setattr(dashboard, "get_project_version", lambda _: 100.0)

    loaded = dashboard.load_project_data(project_path)

    # Desired fallback once hardened:
    assert loaded["mode"] == "prebuilt"
    assert loaded["df_countries"].empty
