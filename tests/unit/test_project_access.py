from vertex.project_access import is_project_visible


def _base_project(is_public=False):
    return {
        "project_type": "prebuilt",
        "project_id": "sivigila",
        "is_public": is_public,
    }


def test_is_project_visible_uses_db_public_first():
    project = _base_project(is_public=False)
    login_state = {
        "is_logged_in": False,
        "user_id": None,
        "allowed_project_ids": [],
        "db_project_access": {"sivigila": {"is_public": True, "owner_id": "owner-1"}},
    }

    assert is_project_visible(project, auth_enabled=True, login_state=login_state) is True


def test_is_project_visible_uses_db_owner_access():
    project = _base_project(is_public=False)
    login_state = {
        "is_logged_in": True,
        "user_id": "owner-1",
        "allowed_project_ids": [],
        "db_project_access": {"sivigila": {"is_public": False, "owner_id": "owner-1"}},
    }

    assert is_project_visible(project, auth_enabled=True, login_state=login_state) is True


def test_is_project_visible_uses_db_granted_access_mapping():
    project = _base_project(is_public=False)
    login_state = {
        "is_logged_in": True,
        "user_id": "user-2",
        "allowed_project_ids": ["sivigila"],
        "db_project_access": {"sivigila": {"is_public": False, "owner_id": "owner-1"}},
    }

    assert is_project_visible(project, auth_enabled=True, login_state=login_state) is True


def test_is_project_visible_falls_back_to_file_when_project_not_in_db():
    project = _base_project(is_public=True)
    login_state = {
        "is_logged_in": False,
        "user_id": None,
        "allowed_project_ids": [],
        "db_project_access": {},
    }

    assert is_project_visible(project, auth_enabled=True, login_state=login_state) is True


def test_is_project_visible_db_not_public_denies_logged_out_even_if_file_public():
    project = _base_project(is_public=True)
    login_state = {
        "is_logged_in": False,
        "user_id": None,
        "allowed_project_ids": [],
        "db_project_access": {"sivigila": {"is_public": False, "owner_id": "owner-1"}},
    }

    assert is_project_visible(project, auth_enabled=True, login_state=login_state) is False
