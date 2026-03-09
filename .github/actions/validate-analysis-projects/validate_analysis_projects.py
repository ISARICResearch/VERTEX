#!/usr/bin/env python3

import argparse
import ast
import json
import sys
from pathlib import Path


def _load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: invalid JSON ({exc})")
        return None


def _as_text(value) -> str:
    return "" if value is None else str(value).strip()


def _panel_functions(panel_file: Path, errors: list[str]) -> set[str]:
    try:
        tree = ast.parse(panel_file.read_text(), filename=str(panel_file))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{panel_file}: could not parse panel module ({exc})")
        return set()
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _validate_project(project_root: Path, errors: list[str], warnings: list[str]):
    config_file = project_root / "config_file.json"
    payload = _load_json(config_file, errors)
    if payload is None or not isinstance(payload, dict):
        if payload is not None:
            errors.append(f"{config_file}: config must be a JSON object")
        return

    for field in ("project_name", "project_id", "project_owner", "is_public", "insight_panels_path"):
        if field not in payload:
            errors.append(f"{config_file}: missing required field '{field}'")

    insight_panels_path = _as_text(payload.get("insight_panels_path"))
    if not insight_panels_path:
        errors.append(f"{config_file}: insight_panels_path must be a non-empty string")
        return

    panels_dir = project_root / insight_panels_path
    if not panels_dir.is_dir():
        errors.append(f"{config_file}: insight_panels_path does not exist ({panels_dir})")
        return

    configured_panels = payload.get("insight_panels", [])
    if configured_panels is None:
        configured_panels = []
    if not isinstance(configured_panels, list):
        errors.append(f"{config_file}: insight_panels must be a list when provided")
        configured_panels = []

    configured_panel_names = []
    for panel_name in configured_panels:
        if not isinstance(panel_name, str) or not panel_name.strip():
            errors.append(f"{config_file}: insight_panels entries must be non-empty strings")
            continue
        configured_panel_names.append(panel_name)
        panel_file = panels_dir / f"{panel_name}.py"
        if not panel_file.exists():
            errors.append(f"{config_file}: listed insight panel '{panel_name}' is missing ({panel_file})")
            continue
        functions = _panel_functions(panel_file, errors)
        for function_name in ("define_button", "create_visuals"):
            if function_name not in functions:
                errors.append(f"{panel_file}: missing required function '{function_name}'")

    available_panels = sorted(panel_file.stem for panel_file in panels_dir.glob("*.py") if not panel_file.name.startswith("_"))
    for panel_name in available_panels:
        if panel_name not in configured_panel_names:
            warnings.append(
                f"{project_root}: panel '{panel_name}' exists in insight_panels_path but is not listed in config_file.json"
            )

    api_url = _as_text(payload.get("api_url"))
    api_key = _as_text(payload.get("api_key"))
    file_backed = not (api_url and api_key)
    if not file_backed:
        return

    analysis_data_path = _as_text(payload.get("insight_panels_data_path")) or "analysis_data/"
    analysis_data_dir = project_root / analysis_data_path
    if not analysis_data_dir.is_dir():
        errors.append(f"{config_file}: file-backed project requires analysis data directory ({analysis_data_dir})")
        return

    for required_file in ("df_map.csv", "vertex_dictionary.csv"):
        candidate = analysis_data_dir / required_file
        if not candidate.exists():
            errors.append(f"{config_file}: missing required analysis data file '{candidate}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VERTEX analysis project source structure.")
    parser.add_argument("--root", required=True, help="Root directory containing analysis project folders")
    args = parser.parse_args()

    root = Path(args.root)
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        print(f"::error ::Root path does not exist: {root}")
        return 1

    project_roots = sorted(path for path in root.iterdir() if path.is_dir() and (path / "config_file.json").exists())
    if not project_roots:
        print(f"::warning ::No analysis projects with config_file.json found under {root}")
        return 0

    for project_root in project_roots:
        _validate_project(project_root, errors, warnings)

    if warnings:
        print("::group::Analysis project warnings")
        for warning in warnings:
            print(f"::warning ::{warning}")
        print("::endgroup::")

    if errors:
        print("::group::Analysis project errors")
        for err in errors:
            print(f"::error ::{err}")
        print("::endgroup::")
        print(
            f"Validation failed: checked {len(project_roots)} analysis project(s) under {root} "
            f"with {len(errors)} error(s) and {len(warnings)} warning(s)."
        )
        return 1

    print(
        f"Validation passed: checked {len(project_roots)} analysis project(s) under {root} "
        f"with {len(errors)} error(s) and {len(warnings)} warning(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
