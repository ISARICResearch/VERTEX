import json
import shutil
from pathlib import Path

import pytest

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture
def copy_fixture_project(tmp_path):
    def _copy(project_name: str, target_name: str = None) -> Path:
        source = FIXTURES_ROOT / project_name
        if not source.exists():
            raise FileNotFoundError(f"Missing fixture project: {source}")
        target = tmp_path / (target_name or project_name)
        shutil.copytree(source, target)
        return target

    return _copy


@pytest.fixture
def analysis_project_dir(copy_fixture_project):
    return copy_fixture_project("analysis_files_project", "analysis-proj")


@pytest.fixture
def prebuilt_project_factory(copy_fixture_project):
    """Copy and optionally customize a static fixture project on disk for tests."""

    def _create(
        *,
        metadata_filename="dashboard_metadata.json",
        buttons=None,
        panel_metadata_files=None,
        data_files=None,
        create_panel_dir=True,
    ):
        project_dir = copy_fixture_project("prebuilt_public_project", "static-proj")

        if buttons is None:
            metadata = json.loads((project_dir / "dashboard_metadata.json").read_text())
            buttons = metadata.get("insight_panels", [])
        _write_json(project_dir / metadata_filename, {"insight_panels": buttons})

        if create_panel_dir:
            for suffix in {button["suffix"] for button in buttons}:
                (project_dir / suffix).mkdir(parents=True, exist_ok=True)

        if panel_metadata_files is not None:
            for existing in project_dir.glob("*/*.json"):
                existing.unlink()
            target_suffix = next(iter({button["suffix"] for button in buttons}), "panel_a")
            for filename, payload in panel_metadata_files.items():
                _write_json(project_dir / target_suffix / filename, payload)

        if data_files is not None:
            for existing in project_dir.glob("*/*.csv"):
                existing.unlink()
            for rel_path, csv_content in data_files.items():
                target = project_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(csv_content)

        return project_dir

    return _create
