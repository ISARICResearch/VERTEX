import json
from pathlib import Path
from typing import Optional


def _load_json(path: Path):
    return json.loads(path.read_text())


def _resolve_data_path(metadata_file: Path, relative_path: str) -> Optional[Path]:
    candidates = [
        metadata_file.parent / relative_path,
        metadata_file.parent.parent / relative_path,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def test_demo_output_metadata_references_existing_data_files():
    metadata_files = [p for p in Path("demo-projects").rglob("*metadata.json") if p.name != "dashboard_metadata.json"]

    missing = []
    for metadata_file in metadata_files:
        data = json.loads(metadata_file.read_text())
        fig_data = data.get("fig_data", [])
        if not isinstance(fig_data, list):
            continue
        for rel_path in fig_data:
            if _resolve_data_path(metadata_file, rel_path) is None:
                missing.append(f"{metadata_file}: {rel_path}")

    assert not missing, "Missing fig_data files:\n" + "\n".join(missing)


def test_demo_output_dashboard_metadata_schema_is_valid():
    dashboard_files = list(Path("demo-projects").rglob("dashboard_metadata.json"))
    errors = []
    for dashboard_file in dashboard_files:
        payload = _load_json(dashboard_file)
        if not isinstance(payload, dict):
            errors.append(f"{dashboard_file}: payload must be object")
            continue
        insight_panels = payload.get("insight_panels")
        if not isinstance(insight_panels, list):
            errors.append(f"{dashboard_file}: insight_panels must be list")
            continue
        for idx, button in enumerate(insight_panels):
            if not isinstance(button, dict):
                errors.append(f"{dashboard_file}: insight_panels[{idx}] must be object")
                continue
            suffix = button.get("suffix")
            if not isinstance(suffix, str) or not suffix.strip():
                errors.append(f"{dashboard_file}: insight_panels[{idx}].suffix must be non-empty string")
            graph_ids = button.get("graph_ids")
            if graph_ids is None:
                continue
            if not isinstance(graph_ids, list):
                errors.append(f"{dashboard_file}: insight_panels[{idx}].graph_ids must be list")
                continue
            for gid in graph_ids:
                if not isinstance(gid, str) or not gid.strip():
                    errors.append(f"{dashboard_file}: graph_id must be non-empty string (button idx {idx})")
    assert not errors, "Invalid demo dashboard metadata schema:\n" + "\n".join(errors)


def test_demo_output_figure_metadata_schema_is_valid():
    metadata_files = [p for p in Path("demo-projects").rglob("*metadata.json") if p.name != "dashboard_metadata.json"]

    errors = []
    for metadata_file in metadata_files:
        payload = _load_json(metadata_file)
        fig_id = payload.get("fig_id")
        fig_name = payload.get("fig_name")
        fig_args = payload.get("fig_arguments")
        fig_data = payload.get("fig_data")

        if not isinstance(fig_id, str) or not fig_id.strip():
            errors.append(f"{metadata_file}: fig_id must be non-empty string")
        if not isinstance(fig_name, str) or not fig_name.strip():
            errors.append(f"{metadata_file}: fig_name must be non-empty string")
        if fig_args is not None and not isinstance(fig_args, dict):
            errors.append(f"{metadata_file}: fig_arguments must be object when provided")
        if not isinstance(fig_data, list):
            errors.append(f"{metadata_file}: fig_data must be list")
            continue
        for rel_path in fig_data:
            if not isinstance(rel_path, str) or not rel_path.strip():
                errors.append(f"{metadata_file}: fig_data entries must be non-empty strings")
                continue
            if _resolve_data_path(metadata_file, rel_path) is None:
                errors.append(f"{metadata_file}: missing fig_data file {rel_path}")

    assert not errors, "Invalid demo figure metadata schema:\n" + "\n".join(errors)


def test_prebuilt_fixture_has_required_root_files():
    project_root = Path("tests/fixtures/prebuilt_public_project")
    required_files = ["config_file.json", "dashboard_metadata.json", "dashboard_data.csv"]
    missing = [name for name in required_files if not (project_root / name).exists()]
    assert not missing, f"Missing required prebuilt fixture file(s): {missing}"


def test_prebuilt_fixture_dashboard_metadata_schema_is_valid():
    payload = _load_json(Path("tests/fixtures/prebuilt_public_project/dashboard_metadata.json"))
    assert isinstance(payload, dict)
    assert isinstance(payload.get("insight_panels"), list)
    for button in payload["insight_panels"]:
        assert isinstance(button, dict)
        assert isinstance(button.get("suffix"), str) and button.get("suffix")
        assert isinstance(button.get("graph_ids"), list)


def test_prebuilt_fixture_figure_metadata_schema_is_valid():
    project_root = Path("tests/fixtures/prebuilt_public_project")
    metadata_files = [p for p in project_root.rglob("*metadata.json") if p.name != "dashboard_metadata.json"]
    assert metadata_files
    for metadata_file in metadata_files:
        payload = _load_json(metadata_file)
        assert isinstance(payload.get("fig_id"), str) and payload["fig_id"]
        assert isinstance(payload.get("fig_name"), str) and payload["fig_name"]
        assert isinstance(payload.get("fig_arguments"), dict)
        assert isinstance(payload.get("fig_data"), list)


def test_prebuilt_fixture_output_metadata_references_existing_data_files():
    project_root = Path("tests/fixtures/prebuilt_public_project")
    metadata_files = [p for p in project_root.rglob("*metadata.json") if p.name != "dashboard_metadata.json"]

    missing = []
    for metadata_file in metadata_files:
        data = json.loads(metadata_file.read_text())
        fig_data = data.get("fig_data", [])
        if not isinstance(fig_data, list):
            continue
        for rel_path in fig_data:
            if _resolve_data_path(metadata_file, rel_path) is None:
                missing.append(f"{metadata_file}: {rel_path}")

    assert not missing, "Missing fixture fig_data files:\n" + "\n".join(missing)
