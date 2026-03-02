import json
from pathlib import Path
from typing import Optional


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


def test_prebuilt_fixture_has_required_root_files():
    project_root = Path("tests/fixtures/prebuilt_public_project")
    required_files = ["config_file.json", "dashboard_metadata.json", "dashboard_data.csv"]
    missing = [name for name in required_files if not (project_root / name).exists()]
    assert not missing, f"Missing required prebuilt fixture file(s): {missing}"


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
