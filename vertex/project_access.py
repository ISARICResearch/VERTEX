import os
import re
import unicodedata


def is_project_visible(project, auth_enabled: bool, login_state):
    if project.get("project_type") == "analysis":
        return True

    if project.get("is_public", False):
        return True

    if not auth_enabled:
        return False

    if isinstance(login_state, dict):
        if not login_state.get("is_logged_in", False):
            return False
        project_id = project.get("project_id")
        if not project_id:
            return False
        return project_id in login_state.get("allowed_project_ids", [])

    return bool(login_state)


def get_visible_projects(project_catalog, auth_enabled: bool, login_state):
    return [project for project in project_catalog if is_project_visible(project, auth_enabled, login_state)]


def get_default_project_path(visible_projects):
    for project in visible_projects:
        if project.get("project_type") == "analysis" and project.get("data_source") == "files":
            return project["path"]
    for project in visible_projects:
        if project.get("project_type") == "analysis":
            return project["path"]
    for project in visible_projects:
        if project.get("project_type") == "prebuilt":
            return project["path"]
    return visible_projects[0]["path"] if visible_projects else None


def get_project_value(project):
    return project.get("project_id") or project["path"]


def find_project_by_path(project_catalog, project_path):
    for project in project_catalog:
        if project["path"] == project_path:
            return project
    return None


def normalise_buttons(buttons):
    normalised = []
    for button in buttons or []:
        if not isinstance(button, dict):
            continue
        suffix = button.get("suffix")
        if not suffix:
            continue
        normalised.append(
            {
                **button,
                "suffix": suffix,
                "item": button.get("item") or "Insights",
                "label": button.get("label") or button.get("title") or suffix,
            }
        )
    return normalised


def resolve_project_value(selected_value, project_catalog):
    if not selected_value:
        return None
    selected_norm = selected_value.rstrip("/")
    for project in project_catalog:
        if selected_norm == get_project_value(project).rstrip("/"):
            return project["path"]
        if selected_norm == project["path"].rstrip("/"):
            return project["path"]
    return None


def resolve_project_request(query_value, project_catalog):
    if not query_value:
        return None

    requested = query_value.strip()
    if not requested:
        return None

    requested_norm = requested.rstrip("/")
    requested_lower = requested_norm.lower()
    requested_canonical = _canonical_project_key(requested_norm)
    for project in project_catalog:
        if project.get("project_id") and requested_norm == project["project_id"]:
            return project["path"]
        if requested_norm == project["path"].rstrip("/"):
            return project["path"]
        if requested_norm == os.path.basename(os.path.normpath(project["path"])):
            return project["path"]
        if requested_lower == str(project["name"]).strip().lower():
            return project["path"]

        candidates = [
            project["path"],
            os.path.basename(os.path.normpath(project["path"])),
            project.get("project_id") or "",
            str(project.get("name") or ""),
        ]
        if any(_canonical_project_key(candidate) == requested_canonical for candidate in candidates):
            return project["path"]

    return None


def _canonical_project_key(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.strip().rstrip("/").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
