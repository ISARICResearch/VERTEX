#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: invalid JSON ({exc})")
        return None


def _resolve_data_path(project_root: Path, metadata_file: Path, relative_path: str) -> Path | None:
    candidates = [
        metadata_file.parent / relative_path,
        metadata_file.parent.parent / relative_path,
        project_root / relative_path,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _metadata_path_for_graph_id(project_root: Path, graph_id: str) -> Path:
    return project_root / f"{graph_id}_metadata.json"


def _validate_dashboard_schema(project_root: Path, dashboard_file: Path, errors: list[str]) -> set[Path]:
    payload = _load_json(dashboard_file, errors)
    referenced_metadata: set[Path] = set()
    if payload is None:
        return referenced_metadata
    if not isinstance(payload, dict):
        errors.append(f"{dashboard_file}: payload must be an object")
        return referenced_metadata
    insight_panels = payload.get("insight_panels")
    if not isinstance(insight_panels, list):
        errors.append(f"{dashboard_file}: insight_panels must be a list")
        return referenced_metadata

    for idx, button in enumerate(insight_panels):
        if not isinstance(button, dict):
            errors.append(f"{dashboard_file}: insight_panels[{idx}] must be an object")
            continue

        suffix = button.get("suffix")
        if not isinstance(suffix, str) or not suffix.strip():
            errors.append(f"{dashboard_file}: insight_panels[{idx}].suffix must be a non-empty string")

        graph_ids = button.get("graph_ids")
        if graph_ids is None:
            continue
        if not isinstance(graph_ids, list):
            errors.append(f"{dashboard_file}: insight_panels[{idx}].graph_ids must be a list")
            continue

        for graph_id in graph_ids:
            if not isinstance(graph_id, str) or not graph_id.strip():
                errors.append(f"{dashboard_file}: graph_ids entries must be non-empty strings")
                continue
            metadata_candidate = _metadata_path_for_graph_id(project_root, graph_id)
            referenced_metadata.add(metadata_candidate)
            if not metadata_candidate.exists():
                errors.append(f"{dashboard_file}: missing metadata file for graph_id '{graph_id}' ({metadata_candidate})")
    return referenced_metadata


def _validate_figure_metadata(project_root: Path, metadata_file: Path, errors: list[str]):
    payload = _load_json(metadata_file, errors)
    if payload is None:
        return
    if not isinstance(payload, dict):
        errors.append(f"{metadata_file}: payload must be an object")
        return

    fig_id = payload.get("fig_id")
    fig_name = payload.get("fig_name")
    fig_args = payload.get("fig_arguments")
    fig_data = payload.get("fig_data")

    if not isinstance(fig_id, str) or not fig_id.strip():
        errors.append(f"{metadata_file}: fig_id must be a non-empty string")
    if not isinstance(fig_name, str) or not fig_name.strip():
        errors.append(f"{metadata_file}: fig_name must be a non-empty string")
    if fig_args is not None and not isinstance(fig_args, dict):
        errors.append(f"{metadata_file}: fig_arguments must be an object when provided")
    if not isinstance(fig_data, list):
        errors.append(f"{metadata_file}: fig_data must be a list")
        return

    for rel_path in fig_data:
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"{metadata_file}: fig_data entries must be non-empty strings")
            continue
        if _resolve_data_path(project_root, metadata_file, rel_path) is None:
            errors.append(f"{metadata_file}: missing fig_data file '{rel_path}'")


def _warn_for_dangling_metadata(project_root: Path, referenced_metadata: set[Path], warnings: list[str]):
    all_metadata_files = {p for p in project_root.rglob("*metadata.json") if p.name != "dashboard_metadata.json"}
    for metadata_file in sorted(all_metadata_files - referenced_metadata):
        warnings.append(
            f"{metadata_file}: metadata file is not referenced by dashboard_metadata.json " "and will be ignored at runtime"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VERTEX static output schema and file references.")
    parser.add_argument("--root", required=True, help="Root directory to recursively scan")
    parser.add_argument(
        "--require-project-files",
        default="true",
        choices=["true", "false"],
        help="Require config_file.json and dashboard_data.csv next to dashboard_metadata.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    require_project_files = args.require_project_files == "true"
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        print(f"::error ::Root path does not exist: {root}")
        return 1

    dashboard_files = list(root.rglob("dashboard_metadata.json"))
    if not dashboard_files:
        print(f"::warning ::No dashboard_metadata.json files found under {root}")
        return 0

    for dashboard_file in dashboard_files:
        project_root = dashboard_file.parent
        if require_project_files:
            for required_file in ("config_file.json", "dashboard_data.csv"):
                candidate = project_root / required_file
                if not candidate.exists():
                    errors.append(f"{project_root}: missing required file '{required_file}'")

        referenced_metadata = _validate_dashboard_schema(project_root, dashboard_file, errors)
        for metadata_file in sorted(referenced_metadata):
            _validate_figure_metadata(project_root, metadata_file, errors)
        _warn_for_dangling_metadata(project_root, referenced_metadata, warnings)

    if warnings:
        print("::group::Project output schema warnings")
        for warning in warnings:
            print(f"::warning ::{warning}")
        print("::endgroup::")

    if errors:
        print("::group::Project output schema errors")
        for err in errors:
            print(f"::error ::{err}")
        print("::endgroup::")
        print(f"Found {len(errors)} validation error(s).")
        return 1

    print(f"Validation passed: checked {len(dashboard_files)} dashboard metadata file(s) under {root} with no schema errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
