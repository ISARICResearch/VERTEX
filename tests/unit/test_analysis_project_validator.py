import importlib.util
import json
import sys
from pathlib import Path


def _load_validator_module():
    module_path = Path(".github/actions/validate-analysis-projects/validate_analysis_projects.py")
    spec = importlib.util.spec_from_file_location("validate_analysis_projects", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_analysis_validator_warns_for_unlisted_panel_but_passes(tmp_path, capsys, monkeypatch):
    validator = _load_validator_module()
    project_root = tmp_path / "demo-project"
    (project_root / "insight_panels").mkdir(parents=True)
    (project_root / "analysis_data").mkdir()
    (project_root / "analysis_data" / "df_map.csv").write_text("subjid\n1\n")
    (project_root / "analysis_data" / "vertex_dictionary.csv").write_text(
        "field_name,field_type,field_label\nsubjid,id,Subject ID\n"
    )
    (project_root / "insight_panels" / "panel_a.py").write_text(
        "def define_button():\n    return {}\n\ndef create_visuals(**kwargs):\n    return []\n"
    )
    (project_root / "insight_panels" / "extra_panel.py").write_text(
        "def define_button():\n    return {}\n\ndef create_visuals(**kwargs):\n    return []\n"
    )
    _write_json(
        project_root / "config_file.json",
        {
            "project_name": "Demo Project",
            "project_id": "demo-project",
            "project_owner": "owner@example.com",
            "is_public": True,
            "insight_panels_path": "insight_panels/",
            "insight_panels_data_path": "analysis_data/",
            "insight_panels": ["panel_a"],
        },
    )

    monkeypatch.setattr(sys, "argv", ["validate_analysis_projects.py", "--root", str(tmp_path)])

    exit_code = validator.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "not listed in config_file.json" in captured.out


def test_analysis_validator_errors_for_missing_required_panel_function(tmp_path, capsys, monkeypatch):
    validator = _load_validator_module()
    project_root = tmp_path / "demo-project"
    (project_root / "insight_panels").mkdir(parents=True)
    (project_root / "analysis_data").mkdir()
    (project_root / "analysis_data" / "df_map.csv").write_text("subjid\n1\n")
    (project_root / "analysis_data" / "vertex_dictionary.csv").write_text(
        "field_name,field_type,field_label\nsubjid,id,Subject ID\n"
    )
    (project_root / "insight_panels" / "panel_a.py").write_text("def define_button():\n    return {}\n")
    _write_json(
        project_root / "config_file.json",
        {
            "project_name": "Demo Project",
            "project_id": "demo-project",
            "project_owner": "owner@example.com",
            "is_public": True,
            "insight_panels_path": "insight_panels/",
            "insight_panels_data_path": "analysis_data/",
            "insight_panels": ["panel_a"],
        },
    )

    monkeypatch.setattr(sys, "argv", ["validate_analysis_projects.py", "--root", str(tmp_path)])

    exit_code = validator.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "missing required function 'create_visuals'" in captured.out
