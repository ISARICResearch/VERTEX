import importlib.util
import json
import sys
from pathlib import Path


def _load_validator_module():
    module_path = Path(".github/actions/validate-project-outputs/validate_project_outputs.py")
    spec = importlib.util.spec_from_file_location("validate_project_outputs", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_validator_warns_for_dangling_metadata_but_keeps_exit_zero(tmp_path, capsys, monkeypatch):
    validator = _load_validator_module()
    project_root = tmp_path / "static-project"
    project_root.mkdir()
    _write_json(project_root / "config_file.json", {"project_name": "Demo"})
    (project_root / "dashboard_data.csv").write_text("country_iso,country_name,country_count\nGBR,United Kingdom,1\n")
    _write_json(
        project_root / "dashboard_metadata.json",
        {
            "insight_panels": [
                {
                    "item": "Figures",
                    "label": "Figure 1",
                    "suffix": "Figure1",
                    "graph_ids": ["Figure1/fig_bar_chart"],
                }
            ]
        },
    )
    _write_json(
        project_root / "Figure1" / "fig_bar_chart_metadata.json",
        {
            "fig_id": "Figure1/fig_bar_chart",
            "fig_name": "fig_bar_chart",
            "fig_arguments": {},
            "fig_data": ["Figure1/fig_bar_chart_data___0.csv"],
        },
    )
    (project_root / "Figure1" / "fig_bar_chart_data___0.csv").write_text("index,value\n2020,1\n")
    _write_json(
        project_root / "Figure1" / "fig_orphan_metadata.json",
        {
            "fig_id": "Figure1/fig_orphan",
            "fig_name": "fig_bar_chart",
            "fig_arguments": {},
            "fig_data": ["Figure1/missing.csv"],
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_project_outputs.py",
            "--root",
            str(project_root),
            "--require-project-files",
            "true",
        ],
    )

    exit_code = validator.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "metadata file is not referenced" in captured.out
    assert "missing fig_data file" not in captured.out


def test_validator_errors_for_missing_data_in_referenced_metadata(tmp_path, capsys, monkeypatch):
    validator = _load_validator_module()
    project_root = tmp_path / "static-project"
    project_root.mkdir()
    _write_json(project_root / "config_file.json", {"project_name": "Demo"})
    (project_root / "dashboard_data.csv").write_text("country_iso,country_name,country_count\nGBR,United Kingdom,1\n")
    _write_json(
        project_root / "dashboard_metadata.json",
        {
            "insight_panels": [
                {
                    "item": "Figures",
                    "label": "Figure 1",
                    "suffix": "Figure1",
                    "graph_ids": ["Figure1/fig_bar_chart"],
                }
            ]
        },
    )
    _write_json(
        project_root / "Figure1" / "fig_bar_chart_metadata.json",
        {
            "fig_id": "Figure1/fig_bar_chart",
            "fig_name": "fig_bar_chart",
            "fig_arguments": {},
            "fig_data": ["Figure1/missing.csv"],
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_project_outputs.py",
            "--root",
            str(project_root),
            "--require-project-files",
            "true",
        ],
    )

    exit_code = validator.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "missing fig_data file" in captured.out
