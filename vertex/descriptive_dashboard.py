import json
import os
from functools import lru_cache
from urllib.parse import parse_qs, quote

import dash
import dash_bootstrap_components as dbc
import jwt
import pandas as pd
import requests
from dash import ALL, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
from flask import current_app, redirect, request, session, url_for
from jwt import PyJWKClient
from plotly import graph_objs as go
from sqlalchemy import MetaData, create_engine

from vertex.io import (
    config_defaults,
    get_config,
    get_projects_catalog,
    load_public_dashboard,
    load_vertex_data,
    save_public_outputs,
    should_save_outputs,
)
from vertex.layout.app_layout import define_inner_layout, define_shell_layout
from vertex.layout.filters import get_filter_options
from vertex.layout.insight_panels import get_insight_panels, get_public_visuals
from vertex.layout.modals import create_modal
from vertex.logging.logger import setup_logger
from vertex.map import create_map, filter_df_map, get_countries, get_public_countries, merge_data_with_countries
from vertex.vertex_secrets import get_database_url, get_flask_auth_secrets

logger = setup_logger(__name__)

# Accounts session check
_last_check = {"time": 0, "ok": False}

AUTH_ENABLED = False


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def should_enable_auth(auth_settings: dict) -> bool:
    # Optional explicit override for local troubleshooting.
    explicit_toggle = os.getenv("VERTEX_ENABLE_AUTH")
    if explicit_toggle is not None:
        return _is_truthy(explicit_toggle)

    return bool(auth_settings.get("VERTEX_BASE_URL") and auth_settings.get("COGNITO_CLIENT_ID"))


############################################
# DATABASE SETUP
############################################

DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL)
metadata = MetaData()

############################################
# CACHE DATA
############################################

PROJECT_CACHE = {}
PROJECT_CACHE_VERSION = {}
PROJECT_TYPE_BY_PATH = {}


def get_project_data(project_path):
    return PROJECT_CACHE.get(project_path)


def set_project_data(project_path, data):
    PROJECT_CACHE[project_path] = data


def clear_project_data(project_path):
    PROJECT_CACHE.pop(project_path, None)
    PROJECT_CACHE_VERSION.pop(project_path, None)


def get_project_version(project_path):
    latest_mtime = 0.0
    for root, _, files in os.walk(project_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                latest_mtime = max(latest_mtime, os.path.getmtime(file_path))
            except OSError:
                continue
    return latest_mtime


def sync_project_type_map(project_catalog):
    global PROJECT_TYPE_BY_PATH
    PROJECT_TYPE_BY_PATH = {project["path"]: project.get("project_type") for project in project_catalog}


def is_logged_in_session(login_state):
    if not AUTH_ENABLED:
        return True
    return bool(login_state)


def is_project_visible(project, login_state):
    if project.get("project_type") == "analysis":
        return True
    if is_logged_in_session(login_state):
        return True
    return bool(project.get("is_public", False))


def get_visible_projects(project_catalog, login_state):
    return [project for project in project_catalog if is_project_visible(project, login_state)]


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
    for project in project_catalog:
        if requested_norm == project["path"].rstrip("/"):
            return project["path"]
        if requested_norm == os.path.basename(os.path.normpath(project["path"])):
            return project["path"]
        if project.get("project_id") and requested_norm == project["project_id"]:
            return project["path"]
        if requested_lower == str(project["name"]).strip().lower():
            return project["path"]

    return None


############################################
# Accounts
############################################


@lru_cache(maxsize=8)
def _jwks_client_for_issuer(issuer: str) -> PyJWKClient:
    discovery = requests.get(f"{issuer}/.well-known/openid-configuration", timeout=2).json()
    return PyJWKClient(discovery["jwks_uri"])


def verify_id_token(token: str) -> dict:
    unverified = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    issuer = unverified["iss"]

    signing_key = _jwks_client_for_issuer(issuer).get_signing_key_from_jwt(token)

    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=current_app.config["COGNITO_CLIENT_ID"],
        issuer=issuer,
        leeway=30,  # allow 30 seconds of leeway to prevent slow clock skew from breaking token validation
    )


def get_accounts_login_url(next_url: str | None = None) -> str:
    vertex_base_url = current_app.config["VERTEX_BASE_URL"].rstrip("/")
    login_url = f"{vertex_base_url}/login"
    if not next_url:
        return login_url
    return f"{login_url}?next={quote(next_url, safe='')}"


def get_accounts_logout_url(next_url: str | None = None) -> str:
    vertex_base_url = current_app.config["VERTEX_BASE_URL"].rstrip("/")
    logout_url = f"{vertex_base_url}/logout"
    if not next_url:
        return logout_url
    return f"{logout_url}?next={quote(next_url, safe='')}"


def get_request_login_state() -> bool:
    if not AUTH_ENABLED:
        return True

    token = request.cookies.get("vertex_id_token")
    if not token:
        return False

    try:
        verify_id_token(token)
    except jwt.PyJWTError as exc:
        unverified = jwt.decode(token, options={"verify_signature": False, "verify_aud": False, "verify_exp": False})
        logger.warning(
            f"Token verification failed: {exc}\n"
            f"  iss:  {unverified.get('iss')}\n"
            f"  aud:  {unverified.get('aud')}\n"
            f"  exp:  {unverified.get('exp')}\n"
            f"  configured COGNITO_CLIENT_ID: {current_app.config.get('COGNITO_CLIENT_ID')}"
        )
        return False

    return True


def build_auth_controls(is_logged_in: bool):
    if not AUTH_ENABLED:
        return html.Div()

    # Avoid capturing Dash-internal endpoints as the post-login redirect target
    _dash_internal = ("/_dash-", "/_reload-", "/auth/")
    raw_url = request.url
    next_url = "/" if any(request.path.startswith(p) for p in _dash_internal) else raw_url

    if is_logged_in:
        return html.A("Logout", href=get_accounts_logout_url(next_url), className="btn btn-danger btn-md")

    return html.A("Login", href=get_accounts_login_url(next_url), className="btn btn-primary btn-md")


############################################
# Dashboard callbacks
############################################


def register_callbacks(app):
    def current_visible_projects(login_state):
        catalog = get_projects_catalog()
        sync_project_type_map(catalog)
        visible = get_visible_projects(catalog, login_state)
        return catalog, visible

    @app.callback(
        Output("selected-project-path", "data", allow_duplicate=True),
        Input("url", "search"),
        State("selected-project-path", "data"),
        State("login-state", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def set_project_from_url(search, current_project, login_state):
        if not search:
            raise PreventUpdate

        query = parse_qs(search.lstrip("?"))
        query_project = (query.get("project") or query.get("param") or [None])[0]
        if not query_project:
            raise PreventUpdate

        _, visible_projects = current_visible_projects(login_state)
        requested_project = resolve_project_request(query_project, visible_projects)
        if not requested_project or requested_project == current_project:
            raise PreventUpdate

        return requested_project

    @app.callback(
        Output("project-body", "children"),
        Input("selected-project-path", "data"),
        State("login-state", "data"),
    )
    def load_project_layout(project_path, login_state):
        if not project_path:
            raise PreventUpdate

        project_catalog, visible_projects = current_visible_projects(login_state)
        if project_path not in [project["path"] for project in visible_projects]:
            if not visible_projects:
                return html.Div([html.H4("No projects available")])
            project_path = get_default_project_path(visible_projects)

        try:
            layout = build_project_layout(project_path, project_catalog, login_state)
        except Exception as e:
            layout = html.Div([html.H4("Error loading project"), html.Pre(str(e))])

        return layout

    ## Change the current project
    @app.callback(
        Output("selected-project-path", "data"),
        Input("project-selector", "value"),
        State("login-state", "data"),
        prevent_initial_call=True,
    )
    def set_project_path(selected_value, login_state):
        logger.info(f"Selected project is: {selected_value}")
        _, visible_projects = current_visible_projects(login_state)
        project_value = resolve_project_value(selected_value, visible_projects)
        logger.debug(f"Mapped selected project to folder: {project_value}")
        if not selected_value or not project_value:
            raise PreventUpdate
        return project_value

    @app.callback(
        Output("url", "search"),
        Input("project-selector", "value"),
        State("login-state", "data"),
        prevent_initial_call=True,
    )
    def update_url_for_project(selected_value, login_state):
        if not selected_value:
            raise PreventUpdate

        project_catalog, visible_projects = current_visible_projects(login_state)
        project_path = resolve_project_value(selected_value, visible_projects)
        if not project_path:
            raise PreventUpdate

        project = find_project_by_path(project_catalog, project_path)
        project_key = project.get("project_id") if project else None
        if not project_key:
            project_key = os.path.basename(os.path.normpath(project_path))
        return f"?project={quote(project_key, safe='')}"

    @app.callback(
        Output("project-selector", "options"),
        Output("project-selector", "value"),
        Input("login-state", "data"),
        State("selected-project-path", "data"),
        prevent_initial_call=True,
    )
    def sync_project_options(login_state, selected_project_path):
        _, visible_projects = current_visible_projects(login_state)
        options = [{"label": project["name"], "value": get_project_value(project)} for project in visible_projects]
        if not options:
            return [], None

        selected_project = find_project_by_path(visible_projects, selected_project_path)
        if selected_project:
            return options, get_project_value(selected_project)
        default_project_path = get_default_project_path(visible_projects)
        default_project = find_project_by_path(visible_projects, default_project_path)
        return options, get_project_value(default_project) if default_project else get_project_value(visible_projects[0])

    @app.callback(
        Output("selected-project-path", "data", allow_duplicate=True),
        Input("login-state", "data"),
        State("selected-project-path", "data"),
        prevent_initial_call=True,
    )
    def ensure_visible_project(login_state, selected_project_path):
        _, visible_projects = current_visible_projects(login_state)
        visible_paths = [project["path"] for project in visible_projects]
        if selected_project_path in visible_paths:
            raise PreventUpdate
        if not visible_projects:
            return None
        return get_default_project_path(visible_projects)

    @app.callback(
        Output("world-map", "figure"),
        [
            Input("sex-checkboxes", "value"),
            Input("age-slider", "value"),
            Input("country-checkboxes", "value"),
            Input("admdate-slider", "value"),
            Input("admdate-slider", "marks"),
            Input("outcome-checkboxes", "value"),
            State("selected-project-path", "data"),
        ],
        [State("map-layout", "data")],
    )
    def update_map(
        sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value, project_path, map_layout_dict
    ):
        project_data = get_project_data(project_path)

        if not project_data or project_data["mode"] != "analysis":
            raise PreventUpdate

        df_map = project_data["df_map"]
        df_filtered = filter_df_map(df_map, sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value)

        if df_filtered.empty:
            geojson = (
                "https://raw.githubusercontent.com/"
                "martynafford/natural-earth-geojson/master/"
                "50m/cultural/ne_50m_admin_0_map_units.json"
            )
            fig = go.Figure(go.Choroplethmap(geojson=geojson, featureidkey="properties.ISO_A3"), layout=map_layout_dict)
        else:
            df_countries = get_countries(df_filtered)
            fig = create_map(df_countries, map_layout_dict)

        return fig

    @app.callback(
        [Output("country-selectall", "value"), Output("country-selectall", "options"), Output("country-checkboxes", "value")],
        [Input("country-selectall", "value"), Input("country-checkboxes", "value")],
        [State("country-checkboxes", "options")],
    )
    def update_country_selection(selectall_value, country_value, country_options):
        ctx = dash.callback_context
        selectall_value = selectall_value or []
        country_value = country_value or []
        country_options = country_options or []

        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            return [["all"], [{"label": "Unselect all", "value": "all"}], country_value]

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "country-selectall":
            if "all" in selectall_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    ["all"],
                    [{"label": "Unselect all", "value": "all"}],
                    [option["value"] for option in country_options],
                ]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{"label": "Select all", "value": "all"}], []]
        elif trigger_id == "country-checkboxes":
            if len(country_value) == len(country_options):
                # All countries are selected manually
                output = [["all"], [{"label": "Unselect all", "value": "all"}], country_value]
            else:
                # Some countries are deselected
                output = [[], [{"label": "Select all", "value": "all"}], country_value]
        else:
            output = [selectall_value, [{"label": "Select all", "value": "all"}], country_value]
        return output

    @app.callback(Output("country-fade", "is_in"), [Input("country-display", "n_clicks")], [State("country-fade", "is_in")])
    def toggle_country_fade(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    @app.callback(
        Output("country-display", "children"), [Input("country-checkboxes", "value")], [State("country-checkboxes", "options")]
    )
    def update_country_display(country_value, country_options):
        if not country_value:
            output = html.Div(
                [
                    html.B("Country:"),
                    # ' (scroll down for all)',
                    html.Br(),
                    "None selected",
                ]
            )
        else:
            country_options = country_options or []
            value_label_map = {
                option.get("value"): option.get("label")
                for option in country_options
                if isinstance(option, dict) and "value" in option and "label" in option
            }

            # Build the display string
            selected_labels = [value_label_map.get(val, val) for val in country_value]
            display_text = ", ".join(selected_labels)

            if len(display_text) > 35:  # Adjust character limit as needed
                if len(selected_labels) == 1:
                    output = html.Div(
                        [
                            html.B("Country:"),
                            # ' (scroll down for all)',
                            html.Br(),
                            f"{selected_labels[0]}",
                        ]
                    )
                else:
                    output = html.Div(
                        [
                            html.B("Country:"),
                            # ' (scroll down for all)',
                            html.Br(),
                            f"{selected_labels[0]}, ",
                            f"+{len(selected_labels) - 1} more...",
                        ]
                    )
            else:
                output = html.Div([html.B("Country:"), html.Br(), f"{display_text}"])
        return output

    @app.callback(
        [
            Output("modal", "is_open"),
            Output("modal", "children"),
            Output("button", "data"),
        ],
        Input({"type": "open-modal", "index": ALL}, "n_clicks"),
        State("selected-project-path", "data"),
        prevent_initial_call=True,
    )
    def open_and_load_modal(n_clicks, project_path):
        if not n_clicks or not any(n_clicks):
            raise PreventUpdate

        try:
            triggered = callback_context.triggered or []
            prop_id = triggered[0]["prop_id"].split(".")[0]
            suffix = json.loads(prop_id)["index"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            logger.warning("open_and_load_modal received malformed callback trigger payload")
            raise PreventUpdate

        project_data = get_project_data(project_path)
        if not project_data:
            raise PreventUpdate
        logger.debug(f"open_and_load_modal: suffix requested = {suffix!r}")
        logger.debug(f"Available suffixes: {list(project_data.get('insight_panels', {}).keys())}")

        # --- get the insight panel
        insight_panel = project_data["insight_panels"].get(suffix)
        if insight_panel is None:
            logger.error(f"Insight panel {suffix} not found in project {project_path}")
            raise PreventUpdate

        # --- call create_visuals
        mode = project_data.get("mode", "analysis")
        if mode == "prebuilt":
            visuals = insight_panel.create_visuals(suffix=suffix, filepath=project_path)
            filter_options = None
        else:
            visuals = insight_panel.create_visuals(
                df_map=project_data["df_map"].copy(),
                df_forms_dict={k: v.copy() for k, v in project_data["df_forms_dict"].items()},
                dictionary=project_data["dictionary"].copy(),
                quality_report=project_data["quality_report"],
                suffix=suffix,
                filepath=project_path,
                save_inputs=False,
            )
            filter_options = get_filter_options(project_data["df_map"])

        # --- build button metadata
        if hasattr(insight_panel, "define_button"):
            button = {**insight_panel.define_button(), **{"suffix": suffix}}
        else:
            # prebuilt panels don’t define_button(); use button data already cached
            button = next((b for b in project_data["buttons"] if b["suffix"] == suffix), {"suffix": suffix})

        # --- create modal (filters only if available)
        modal_content = create_modal(visuals, button, filter_options=filter_options)

        return True, modal_content, button

    @app.callback(
        [
            Output("country-selectall-modal", "value"),
            Output("country-selectall-modal", "options"),
            Output("country-checkboxes-modal", "value"),
        ],
        [Input("country-selectall-modal", "value"), Input("country-checkboxes-modal", "value")],
        [State("country-checkboxes-modal", "options")],
    )
    def update_country_selection_modal(selectall_value, country_value, country_options):
        ctx = dash.callback_context
        selectall_value = selectall_value or []
        country_value = country_value or []
        country_options = country_options or []
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            return [["all"], [{"label": "Unselect all", "value": "all"}], country_value]

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        #
        if trigger_id == "country-selectall-modal":
            if "all" in selectall_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [["all"], [{"label": "Unselect all", "value": "all"}], [option["value"] for option in country_options]]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{"label": "Select all", "value": "all"}], []]
        elif trigger_id == "country-checkboxes-modal":
            if len(country_value) == len(country_options):
                # All countries are selected manually
                output = [["all"], [{"label": "Unselect all", "value": "all"}], country_value]
            else:
                # Some countries are deselected
                output = [[], [{"label": "Select all", "value": "all"}], country_value]
        else:
            output = [selectall_value, [{"label": "Select all", "value": "all"}], country_value]
        return output

    @app.callback(
        Output("country-fade-modal", "is_in"),
        [Input("country-display-modal", "n_clicks")],
        [State("country-fade-modal", "is_in")],
    )
    def toggle_country_fade_modal(n_clicks, is_in):
        state = is_in
        if n_clicks:
            state = not is_in
        return state

    @app.callback(
        Output("country-display-modal", "children"),
        [Input("country-checkboxes-modal", "value")],
        [State("country-checkboxes-modal", "options")],
    )
    def update_country_display_modal(country_value, country_options):
        if not country_value:
            return "Country:"

        # Create a dictionary to map values to labels
        country_options = country_options or []
        value_label_map = {
            option.get("value"): option.get("label")
            for option in country_options
            if isinstance(option, dict) and "value" in option and "label" in option
        }

        # Build the display string
        selected_labels = [value_label_map.get(val, val) for val in country_value]
        display_text = ", ".join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            output = f"{selected_labels[0]}, "
            output += f"+{len(selected_labels) - 1} more..."
        else:
            output = f"Country: {display_text}"
        return output

    @app.callback(
        [
            Output("modal", "children", allow_duplicate=True),
            Output("sex-checkboxes-modal", "value", allow_duplicate=True),
            Output("age-slider-modal", "value", allow_duplicate=True),
            Output("country-checkboxes-modal", "value", allow_duplicate=True),
            Output("admdate-slider-modal", "value", allow_duplicate=True),
            Output("outcome-checkboxes-modal", "value", allow_duplicate=True),
        ],
        [Input("submit-button-modal", "n_clicks")],
        [
            State("button", "data"),
            State("sex-checkboxes-modal", "value"),
            State("age-slider-modal", "value"),
            State("country-checkboxes-modal", "value"),
            State("admdate-slider-modal", "value"),
            State("admdate-slider-modal", "marks"),
            State("outcome-checkboxes-modal", "value"),
            State("selected-project-path", "data"),  # ✅ project path state
        ],
        prevent_initial_call=True,
    )
    def update_figures(
        n_clicks, button, sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value, project_path
    ):
        logger.debug("updating figures")
        if not button or "suffix" not in button:
            raise PreventUpdate

        suffix = button["suffix"]
        project_data = get_project_data(project_path)
        if not project_data:
            raise PreventUpdate
        if project_data.get("mode") != "analysis":
            raise PreventUpdate

        df_map = project_data.get("df_map")
        df_forms_dict = project_data.get("df_forms_dict") or {}
        dictionary = project_data.get("dictionary")
        quality_report = project_data.get("quality_report", {})
        if not isinstance(df_map, pd.DataFrame):
            raise PreventUpdate
        if not isinstance(df_forms_dict, dict):
            raise PreventUpdate
        if not isinstance(dictionary, pd.DataFrame):
            raise PreventUpdate
        insight_panel = (project_data.get("insight_panels") or {}).get(suffix)
        if insight_panel is None:
            raise PreventUpdate

        # Filter the main map
        df_map_filtered = filter_df_map(
            df_map, sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value
        )

        # Filter forms
        df_forms_filtered = {}
        for key, df_form in df_forms_dict.items():
            df_forms_filtered[key] = filter_df_map(
                df_form, sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value
            )

        # If everything is empty, return blank modal
        df_list = [df_map_filtered] + list(df_forms_filtered.values())
        if all(df.empty for df in df_list):
            return (), sex_value, age_value, country_value, admdate_value, outcome_value

        # Otherwise rebuild visuals
        visuals = insight_panel.create_visuals(
            df_map=df_map_filtered.copy(),
            df_forms_dict={k: v.copy() for k, v in df_forms_filtered.items()},
            dictionary=dictionary.copy(),
            quality_report=quality_report,
            filepath=project_path,
            suffix=suffix,
            save_inputs=False,
        )
        logger.debug(f"raw visuals type: {type(visuals)}; len? {len(visuals) if hasattr(visuals, '__len__') else 'no-len'}")

        modal = create_modal(visuals, button, get_filter_options(df_map))

        return (
            modal,
            sex_value,
            age_value,
            country_value,
            admdate_value,
            outcome_value,
        )

    # End of callbacks
    return


############################################
# Main
############################################


def build_project_layout(project_path, project_catalog, login_state):
    project_data = load_project_data(project_path)
    visible_projects = get_visible_projects(project_catalog, login_state)
    project_options = [{"label": project["name"], "value": get_project_value(project)} for project in visible_projects]
    selected_project = find_project_by_path(project_catalog, project_path)
    map_layout_dict = dict(
        map_style="carto-positron",
        map_zoom=project_data["config_dict"]["map_layout_zoom"],
        map_center={
            "lat": project_data["config_dict"]["map_layout_center_latitude"],
            "lon": project_data["config_dict"]["map_layout_center_longitude"],
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    df_countries = project_data["df_countries"]
    required_columns = {"country_iso", "country_name", "country_count"}
    if isinstance(df_countries, pd.DataFrame) and not df_countries.empty and required_columns.issubset(df_countries.columns):
        fig = create_map(df_countries, map_layout_dict)
    else:
        logger.warning(f"Project {project_path} has no valid country map data; rendering empty map.")
        geojson = (
            "https://raw.githubusercontent.com/"
            "martynafford/natural-earth-geojson/master/"
            "50m/cultural/ne_50m_admin_0_map_units.json"
        )
        fig = go.Figure(go.Choroplethmap(geojson=geojson, featureidkey="properties.ISO_A3"), layout=map_layout_dict)
    if project_data["mode"] == "analysis":
        filter_options = get_filter_options(project_data["df_map"])
    else:
        filter_options = None
    logger.debug(f"buttons: {project_data['buttons']}")
    layout = define_inner_layout(
        fig,
        project_data["buttons"],
        map_layout_dict,
        filter_options=filter_options,
        project_name=project_data["config_dict"]["project_name"],
        project_options=project_options,
        selected_project_value=get_project_value(selected_project) if selected_project else None,
    )
    return layout


def load_project_data(project_path):
    """Load project data into cache (if not already loaded) and return it."""
    logger.debug(f" Loading project data for: {project_path}")
    if not project_path:
        raise PreventUpdate

    project_type = PROJECT_TYPE_BY_PATH.get(project_path)
    if project_type is None:
        # fallback for safety if the map has not been populated yet
        project_catalog = get_projects_catalog()
        sync_project_type_map(project_catalog)
        project_type = PROJECT_TYPE_BY_PATH.get(project_path)

    current_version = get_project_version(project_path)
    cached_project = PROJECT_CACHE.get(project_path)
    cached_version = PROJECT_CACHE_VERSION.get(project_path)
    if cached_project and project_type == "analysis":
        logger.info(f" Using sticky cached analysis project data for {project_path}")
        return cached_project
    if cached_project and cached_version == current_version:
        logger.info(f" Using cached project data for {project_path}")
        return cached_project
    if cached_project and cached_version != current_version:
        logger.info(f" Project files changed on disk for {project_path}; reloading cache.")
        clear_project_data(project_path)

    logger.info(f" No cache found, loading fresh data for {project_path}")
    config_dict = get_config(project_path, config_defaults)
    # default to analysis mode
    PREBUILT = False
    if "insight_panels_path" in config_dict.keys():
        insight_panels_path = os.path.join(project_path, config_dict["insight_panels_path"])
        insight_panels, buttons = get_insight_panels(config_dict, insight_panels_path)
    else:
        PREBUILT = True
        logger.info(f" Public project detected, using dashboard_metadata.json for {project_path}")
        metadata = load_public_dashboard(project_path, config_dict)
        buttons_metadata = metadata.get("insight_panels", [])
        if not isinstance(buttons_metadata, list):
            logger.error(
                f"Project metadata insight_panels should be a list for {project_path}; "
                f"got {type(buttons_metadata).__name__}. Falling back to empty."
            )
            buttons_metadata = []
        insight_panels, buttons = get_public_visuals(project_path, buttons_metadata)
        logger.debug(f"{buttons}")
        try:
            df_countries = get_public_countries(project_path)
        except Exception as exc:
            logger.error(f"Could not read dashboard_data.csv for prebuilt project {project_path}: {exc}")
            df_countries = pd.DataFrame(columns=["country_iso", "country_name", "country_count"])

    filter_columns_dict = {
        "subjid": "subjid",
        "demog_sex": "filters_sex",
        "demog_age": "filters_age",
        "pres_date": "filters_admdate",
        "country_iso": "filters_country",
        "outco_binary_outcome": "filters_outcome",
    }

    if not PREBUILT:
        data = load_vertex_data(project_path, config_dict)
        df_map = data.get("df_map", None)
        df_forms_dict = data.get("df_forms_dict", {})
        dictionary = data.get("dictionary", None)
        quality_report = data.get("quality_report", {})
        df_map = df_map.reset_index(drop=True)
        df_map_with_countries = merge_data_with_countries(df_map)
        df_countries = get_countries(df_map_with_countries)
        df_filters = df_map_with_countries[filter_columns_dict.keys()].rename(columns=filter_columns_dict)
        df_map = pd.merge(df_map_with_countries, df_filters, on="subjid", how="left").reset_index(drop=True)
        df_forms_dict = {
            form: pd.merge(df_form, df_filters, on="subjid", how="left").reset_index(drop=True)
            for form, df_form in df_forms_dict.items()
        }
    if insight_panels:
        logger.debug(f"{list(insight_panels)[0]}")
    else:
        logger.debug("No insight panels loaded for this project.")
    project_data = {
        "mode": "prebuilt" if PREBUILT else "analysis",
        "df_map": df_map if not PREBUILT else None,
        "df_forms_dict": df_forms_dict if not PREBUILT else None,
        "dictionary": dictionary if not PREBUILT else None,
        "quality_report": quality_report if not PREBUILT else None,
        "insight_panels": insight_panels,
        "buttons": normalise_buttons(buttons),
        "config_dict": config_dict,
        "df_countries": df_countries,
    }

    PROJECT_CACHE[project_path] = project_data
    PROJECT_CACHE_VERSION[project_path] = current_version

    if should_save_outputs(config_dict):
        logger.info(f" Saving public outputs for project {project_path}")
        save_public_outputs(
            buttons, insight_panels, df_map, df_countries, df_forms_dict, dictionary, quality_report, project_path, config_dict
        )
    logger.info(f" Project data loaded and cached for {project_path}")
    return project_data


def main():
    global AUTH_ENABLED

    logger.info("Starting VERTEX")
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=os.path.join(os.path.dirname(__file__), "..", "assets"),
        title="Isaric VERTEX",
        suppress_callback_exceptions=True,
    )

    # Flask / DB config
    flask_auth_secrets = get_flask_auth_secrets()
    AUTH_ENABLED = should_enable_auth(flask_auth_secrets)
    logger.info(f"Authentication enabled: {AUTH_ENABLED}")

    app.server.config.update(
        {
            "SQLALCHEMY_DATABASE_URI": DATABASE_URL if AUTH_ENABLED else None,
            "SECRET_KEY": flask_auth_secrets.get("SECRET_KEY"),
            "COGNITO_CLIENT_ID": flask_auth_secrets.get("COGNITO_CLIENT_ID"),
            "COGNITO_CLIENT_SECRET": flask_auth_secrets.get("COGNITO_CLIENT_SECRET"),
            "COGNITO_USER_POOL_ID": flask_auth_secrets.get("COGNITO_USER_POOL_ID"),
            "COGNITO_DOMAIN": flask_auth_secrets.get("COGNITO_DOMAIN"),
            "VERTEX_BASE_URL": flask_auth_secrets.get("VERTEX_BASE_URL"),
            "AWS_REGION": flask_auth_secrets.get("AWS_REGION"),
        }
    )

    project_catalog = get_projects_catalog()
    sync_project_type_map(project_catalog)
    project_paths = [project["path"] for project in project_catalog]
    logger.debug(f" Found {len(project_paths)} projects: {project_paths}")

    def serve_layout():
        # Prefer project= for new links, but keep param= for legacy one-off deployments.
        project_catalog = get_projects_catalog()
        sync_project_type_map(project_catalog)
        login_state = get_request_login_state()
        visible_projects = get_visible_projects(project_catalog, login_state)
        default_project = get_default_project_path(visible_projects)
        requested_project = None

        try:
            query_project = request.args.get("project") or request.args.get("param")
            if query_project:
                requested_project = resolve_project_request(query_project, visible_projects)
                if requested_project is None:
                    logger.warning(f"Unknown project requested via query string: {query_project}")
        except Exception as exc:
            logger.debug(f"Unable to read request args for project selection: {exc}")

        active_project = requested_project or default_project
        return define_shell_layout(
            active_project,
            initial_body=html.Div("Loading VERTEX..."),
            auth_controls=build_auth_controls(login_state),
            initial_login_state=login_state,
        )

    app.layout = serve_layout

    # ---------------------------------------------------------------------------
    # Built-in auth routes – used when VERTEX_BASE_URL points at this server
    # (i.e. local dev).  Production deployments delegate to account.isaric.org.
    # ---------------------------------------------------------------------------

    @app.server.route("/auth/login")
    def auth_login():
        next_url = request.args.get("next", request.url_root)
        cognito_domain = current_app.config.get("COGNITO_DOMAIN", "").rstrip("/")
        client_id = current_app.config.get("COGNITO_CLIENT_ID")

        if not cognito_domain or not client_id:
            logger.error("COGNITO_DOMAIN or COGNITO_CLIENT_ID not configured – cannot initiate login")
            return "Auth not configured", 500

        callback_uri = url_for("auth_callback", _external=True)
        session["auth_next"] = next_url

        authorize_url = (
            f"{cognito_domain}/oauth2/authorize"
            f"?response_type=code"
            f"&client_id={client_id}"
            f"&redirect_uri={quote(callback_uri, safe='')}"
            f"&scope=openid%20email%20profile"
        )
        return redirect(authorize_url)

    @app.server.route("/auth/callback")
    def auth_callback():
        code = request.args.get("code")
        if not code:
            logger.warning("auth_callback called without code param")
            return redirect("/")

        cognito_domain = current_app.config.get("COGNITO_DOMAIN", "").rstrip("/")
        client_id = current_app.config.get("COGNITO_CLIENT_ID")
        client_secret = current_app.config.get("COGNITO_CLIENT_SECRET")
        callback_uri = url_for("auth_callback", _external=True)

        token_resp = requests.post(
            f"{cognito_domain}/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "redirect_uri": callback_uri,
            },
            auth=(client_id, client_secret) if client_secret else None,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        token_data = token_resp.json()
        id_token = token_data.get("id_token")

        if not id_token:
            logger.error(f"Token exchange failed: {token_data.get('error', token_data)}")
            return "Login failed – could not obtain token", 502

        # Decode without verification just to log claims (signature already verified on each request)
        claims = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
        logger.info(
            "=== USER SESSION ===\n"
            f"  sub:            {claims.get('sub')}\n"
            f"  email:          {claims.get('email')}\n"
            f"  name:           {claims.get('name')}\n"
            f"  cognito:groups: {claims.get('cognito:groups')}\n"
            f"  exp:            {claims.get('exp')}\n"
            "==================="
        )

        next_url = session.pop("auth_next", "/")
        resp = redirect(next_url)
        resp.set_cookie("vertex_id_token", id_token, httponly=True, samesite="Lax")
        return resp

    @app.server.route("/auth/logout")
    def auth_logout():
        cognito_domain = current_app.config.get("COGNITO_DOMAIN", "").rstrip("/")
        client_id = current_app.config.get("COGNITO_CLIENT_ID")
        # logout_uri must be an absolute URL registered in the Cognito app client
        app_root = request.url_root.rstrip("/")  # e.g. http://localhost:8050
        next_url = request.args.get("next", "")
        # Strip Dash-internal paths; always land on app root after logout
        _dash_internal = ("/_dash-", "/_reload-", "/auth/")
        if not next_url or any(p in next_url for p in _dash_internal):
            next_url = app_root
        # Ensure absolute URL
        if next_url.startswith("/"):
            next_url = app_root + next_url

        resp = redirect(f"{cognito_domain}/logout" f"?client_id={client_id}" f"&logout_uri={quote(next_url, safe='')}")
        resp.delete_cookie("vertex_id_token")
        return resp

    @app.server.before_request
    def validate_accounts_session():
        path = request.path

        if path.startswith(("/assets/", "/static/", "/favicon.ico", "/_reload-hash", "/auth/")):
            return None

        if not AUTH_ENABLED:
            return None

        token = request.cookies.get("vertex_id_token")
        if not token:
            return None

        try:
            verify_id_token(token)
        except jwt.PyJWTError as exc:
            logger.info(f"Invalid vertex_id_token cookie ({exc}), continuing as logged out")
            return None

        return None

    register_callbacks(app)

    return app


if __name__ == "__main__":
    app = main()
    app.run(debug=True, host="localhost", port=8051, use_reloader=False)
else:
    app = main()
    server = app.server
