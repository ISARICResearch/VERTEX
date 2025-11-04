import json
import os
import secrets
import uuid
import webbrowser

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import callback_context, html, no_update
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate
from flask_login import login_user, logout_user
from flask_security import Security, SQLAlchemyUserDatastore
from flask_security.utils import hash_password, verify_and_update_password
from flask_sqlalchemy import SQLAlchemy
from plotly import graph_objs as go
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import Session

from vertex.io import config_defaults, get_config, get_projects, load_vertex_data, save_public_outputs
from vertex.layout.app_layout import define_inner_layout, define_shell_layout
from vertex.layout.filters import get_filter_options
from vertex.layout.insight_panels import get_insight_panels
from vertex.layout.modals import create_modal
from vertex.logging.logger import setup_logger
from vertex.map import create_map, filter_df_map, get_countries, merge_data_with_countries
from vertex.models import User
from vertex.secrets import get_database_url, get_flask_auth_secrets
from vertex.translation import translate

logger = setup_logger(__name__)

# are we running locally, i.e. no db:
APP_ENV = os.getenv("APP_ENV")
AUTH_ENABLED = bool(APP_ENV)
############################################
# DATABASE SETUP
############################################

DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Flask login

db = SQLAlchemy()  # we will bind these to the app later
security = Security()

############################################
# CACHE DATA
############################################

PROJECT_CACHE = {}


def get_project_data(project_path):
    return PROJECT_CACHE.get(project_path)


def set_project_data(project_path, data):
    PROJECT_CACHE[project_path] = data


def clear_project_data(project_path):
    PROJECT_CACHE.pop(project_path, None)


############################################
# Dashboard callbacks
############################################


def register_callbacks(app):
    @app.callback(
        Output("project-body", "children"),
        Input("selected-project-path", "data"),
    )
    def load_project_layout(project_path):
        if not project_path:
            raise PreventUpdate

        try:
            layout = build_project_layout(project_path)
        except Exception as e:
            layout = html.Div([html.H4("Error loading project"), html.Pre(str(e))])

        return layout

    ## Change the current project
    @app.callback(
        Output("selected-project-path", "data"),
        Input("project-selector", "value"),
        State("project-selector", "options"),
        prevent_initial_call=True,
    )
    def set_project_path(selected_value, project_options):
        logger.info(f"Selected project is: {selected_value}")
        # This line maps the selected label back to the project folder path,
        # it is absolutely absurd that dash gives you the label and not the value of the dropdown
        project_value = next((opt["value"] for opt in project_options if opt["label"] == selected_value), None)
        logger.debug(f"Mapped selected project to folder: {project_value}")
        if not selected_value:
            raise PreventUpdate
        return selected_value

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

        if not project_data:
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
        [State("country-checkboxes", "options"), State("selected-project-path", "data")],
    )
    def update_country_selection(selectall_value, country_value, country_options, project_path):
        project_data = get_project_data(project_path)
        language = project_data["config_dict"]["language"]
        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            output = [["all"], [{"label": translate("Unselect all", language=language), "value": "all"}], country_value]

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "country-selectall":
            if "all" in selectall_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    ["all"],
                    [{"label": translate("Unselect all", language=language), "value": "all"}],
                    [option["value"] for option in country_options],
                ]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{"label": translate("Select all", language=language), "value": "all"}], []]
        elif trigger_id == "country-checkboxes":
            if len(country_value) == len(country_options):
                # All countries are selected manually
                output = [["all"], [{"label": translate("Unselect all", language=language), "value": "all"}], country_value]
            else:
                # Some countries are deselected
                output = [[], [{"label": translate("Select all", language=language), "value": "all"}], country_value]
        else:
            output = [selectall_value, [{"label": translate("Select all", language=language), "value": "all"}], country_value]
        return output

    @app.callback(Output("country-fade", "is_in"), [Input("country-display", "n_clicks")], [State("country-fade", "is_in")])
    def toggle_country_fade(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    @app.callback(
        Output("country-display", "children"),
        [Input("country-checkboxes", "value")],
        [State("country-checkboxes", "options"), State("selected-project-path", "data")],
    )
    def update_country_display(country_value, country_options, project_path):
        project_data = get_project_data(project_path)
        language = project_data["config_dict"]["language"]
        country_str = translate("Country", language=language)
        if not country_value:
            output = html.Div(
                [
                    html.B(country_str + ":"),
                    # ' (scroll down for all)',
                    html.Br(),
                    translate("None selected", language=language),
                ]
            )
        else:
            # Create a dictionary to map values to labels
            value_label_map = {option["value"]: option["label"] for option in country_options}

            # Build the display string
            selected_labels = [value_label_map[val] for val in country_value if val in value_label_map]
            display_text = ", ".join(selected_labels)

            if len(display_text) > 35:  # Adjust character limit as needed
                if len(selected_labels) == 1:
                    output = html.Div(
                        [
                            html.B(country_str + ":"),
                            # ' (scroll down for all)',
                            html.Br(),
                            f"{selected_labels[0]}",
                        ]
                    )
                else:
                    output = html.Div(
                        [
                            html.B(country_str + ":"),
                            # ' (scroll down for all)',
                            html.Br(),
                            f"{selected_labels[0]}, ",
                            f"+{len(selected_labels) - 1} more...",
                        ]
                    )
            else:
                output = html.Div([html.B(country_str + ":"), html.Br(), f"{display_text}"])
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
        if not any(n_clicks):
            raise PreventUpdate

        suffix = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])["index"]

        project_data = get_project_data(project_path)
        if not project_data:
            raise PreventUpdate

        visuals = project_data["insight_panels"][suffix].create_visuals(
            df_map=project_data["df_map"].copy(),
            df_forms_dict={k: v.copy() for k, v in project_data["df_forms_dict"].items()},
            dictionary=project_data["dictionary"].copy(),
            quality_report=project_data["quality_report"],
            suffix=suffix,
            filepath=project_path,
            save_inputs=False,
        )

        button = {**project_data["insight_panels"][suffix].define_button(), **{"suffix": suffix}}
        modal_content = create_modal(
            visuals, button, get_filter_options(project_data["df_map"]), language=project_data["config_dict"]["language"]
        )

        return True, modal_content, button

    @app.callback(
        [
            Output("country-selectall-modal", "value"),
            Output("country-selectall-modal", "options"),
            Output("country-checkboxes-modal", "value"),
        ],
        [Input("country-selectall-modal", "value"), Input("country-checkboxes-modal", "value")],
        [State("country-checkboxes-modal", "options"), State("selected-project-path", "data")],
    )
    def update_country_selection_modal(selectall_value, country_value, country_options, project_path):
        project_data = get_project_data(project_path)
        language = project_data["config_dict"]["language"]

        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            output = [["all"], [{"label": translate("Unselect all", language=language), "value": "all"}], country_value]

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        #
        if trigger_id == "country-selectall-modal":
            if "all" in selectall_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    ["all"],
                    [{"label": translate("Unselect all", language=language), "value": "all"}],
                    [option["value"] for option in country_options],
                ]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{"label": translate("Select all", language=language), "value": "all"}], []]
        elif trigger_id == "country-checkboxes-modal":
            if len(country_value) == len(country_options):
                # All countries are selected manually
                output = [["all"], [{"label": translate("Unselect all", language=language), "value": "all"}], country_value]
            else:
                # Some countries are deselected
                output = [[], [{"label": translate("Select all", language=language), "value": "all"}], country_value]
        else:
            output = [selectall_value, [{"label": translate("Select all", language=language), "value": "all"}], country_value]
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

    @app.callback(Output("auth-button-container", "children"), Input("login-state", "data"))
    def render_auth_button(is_logged_in):
        if not AUTH_ENABLED:
            return html.Div()  # No auth in local mode
        return html.Div(
            [
                dbc.Button(
                    "Login",
                    id="open-login",
                    color="primary",
                    size="md",
                    style={"display": "inline-block" if not is_logged_in else "none"},
                ),
                dbc.Button(
                    "Logout",
                    id="logout-button",
                    color="danger",
                    size="md",
                    style={"display": "inline-block" if is_logged_in else "none"},
                ),
            ]
        )

    @app.callback(
        Output("login-state", "data"),
        Output("login-modal", "is_open"),
        Output("login-output", "children"),
        Input("open-login", "n_clicks"),
        Input("login-submit", "n_clicks"),
        Input("logout-button", "n_clicks"),
        State("login-modal", "is_open"),
        State("username", "value"),
        State("password", "value"),
        prevent_initial_call=True,
    )
    def handle_login_logout(open_clicks, submit_clicks, logout_clicks, is_open, username, password):
        ctx = callback_context

        if not ctx.triggered:
            raise PreventUpdate

        if (open_clicks == 0 or open_clicks is None) and (logout_clicks == 0 or logout_clicks is None):
            return dash.no_update, False, ""

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "open-login":
            return dash.no_update, True, ""

        if trigger == "logout-button":
            logout_user()
            return False, dash.no_update, ""

        if trigger == "login-submit":
            if not username or not password:
                return False, True, "Please enter both username and password."

            with Session(engine) as session:
                user = session.query(User).filter_by(email=username.strip().lower()).first()
                logger.debug(f"User found: {user}")
                if user and verify_and_update_password(password, user):
                    login_user(user)
                    return True, False, ""
                else:
                    logger.debug(f"Invalid login attempt for user: {username}")
                    return False, True, "Invalid username or password."

        return dash.no_update, is_open, ""

    @app.callback(
        Output("register-output", "children"),
        Output("register-modal", "is_open"),
        Input("open-register", "n_clicks"),
        Input("register-submit", "n_clicks"),
        State("register-modal", "is_open"),
        State("register-email", "value"),
        State("register-password", "value"),
        State("register-confirm-password", "value"),
        prevent_initial_call=True,
    )
    def handle_register(open_clicks, submit_clicks, is_open, email, password, confirm_password):
        ctx = callback_context
        triggered = ctx.triggered_id

        if triggered == "open-register":
            return no_update, True  # Open the modal

        elif triggered == "register-submit":
            if not email or not password or not confirm_password:
                return "Please fill in all fields", True

            if password != confirm_password:
                return "Passwords do not match", True

            with Session(engine) as session:
                existing = session.query(User).filter_by(email=email.lower()).first()
                if existing:
                    return "User already exists", True

                new_user = User(
                    id=uuid.uuid4(),
                    email=email.lower(),
                    password=hash_password(password),
                    fs_uniquifier=secrets.token_urlsafe(32),
                    is_admin=False,
                )
                session.add(new_user)
                session.commit()
                return "", False  # Close modal on success

        return no_update, is_open

    @app.callback(
        Output("country-display-modal", "children"),
        [Input("country-checkboxes-modal", "value")],
        [State("country-checkboxes-modal", "options"), State("selected-project-path", "data")],
    )
    def update_country_display_modal(country_value, country_options, project_path):
        project_data = get_project_data(project_path)
        language = project_data["config_dict"]["language"]

        country_str = translate("Country", language=language)
        if not country_value:
            return country_str

        # Create a dictionary to map values to labels
        value_label_map = {option["value"]: option["label"] for option in country_options}

        # Build the display string
        selected_labels = [value_label_map[val] for val in country_value if val in value_label_map]
        display_text = ", ".join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            output = f"{selected_labels[0]}, "
            output += f"+{len(selected_labels) - 1} more..."
        else:
            output = f"{country_str}: {display_text}"
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

        df_map = project_data["df_map"]
        df_forms_dict = project_data["df_forms_dict"]
        dictionary = project_data["dictionary"]
        quality_report = project_data["quality_report"]

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
        visuals = project_data["insight_panels"][suffix].create_visuals(
            df_map=df_map_filtered.copy(),
            df_forms_dict={k: v.copy() for k, v in df_forms_filtered.items()},
            dictionary=dictionary.copy(),
            quality_report=quality_report,
            filepath=project_path,
            suffix=suffix,
            save_inputs=project_data["config_dict"]["save_filtered_public_outputs"],
        )

        modal = create_modal(visuals, button, get_filter_options(df_map), language=project_data["config_dict"]["language"])

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


def build_project_layout(project_path):
    project_data = load_project_data(project_path)
    map_layout_dict = dict(
        map_style="carto-positron",
        map_zoom=project_data["config_dict"]["map_layout_zoom"],
        map_center={
            "lat": project_data["config_dict"]["map_layout_center_latitude"],
            "lon": project_data["config_dict"]["map_layout_center_longitude"],
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    fig = create_map(project_data["df_countries"], map_layout_dict)

    filter_options = get_filter_options(project_data["df_map"])

    layout = define_inner_layout(
        fig,
        project_data["buttons"],
        filter_options,
        map_layout_dict,
        language=project_data["config_dict"]["language"],
        project_name=project_data["config_dict"]["project_name"],
    )
    return layout


def load_project_data(project_path):
    """Load project data into cache (if not already loaded) and return it."""
    logger.debug(f" Loading project data for: {project_path}")
    if not project_path:
        raise PreventUpdate

    project_data = PROJECT_CACHE.get(project_path)
    if project_data:
        logger.info(f" Using cached project data for {project_path}")
        return project_data

    logger.info(f" No cache found, loading fresh data for {project_path}")
    config_dict = get_config(project_path, config_defaults)
    insight_panels_path = os.path.join(project_path, config_dict["insight_panels_path"])
    insight_panels, buttons = get_insight_panels(config_dict, insight_panels_path)

    df_map, df_forms_dict, dictionary, quality_report = load_vertex_data(project_path, config_dict)
    df_map = df_map.reset_index(drop=True)
    df_map_with_countries = merge_data_with_countries(df_map)
    df_countries = get_countries(df_map_with_countries)

    filter_columns_dict = {
        "subjid": "subjid",
        "demog_sex": "filters_sex",
        "demog_age": "filters_age",
        "pres_date": "filters_admdate",
        "country_iso": "filters_country",
        "outco_binary_outcome": "filters_outcome",
    }

    df_filters = df_map_with_countries[filter_columns_dict.keys()].rename(columns=filter_columns_dict)
    df_map = pd.merge(df_map_with_countries, df_filters, on="subjid", how="left").reset_index(drop=True)
    df_forms_dict = {
        form: pd.merge(df_form, df_filters, on="subjid", how="left").reset_index(drop=True)
        for form, df_form in df_forms_dict.items()
    }

    project_data = {
        "df_map": df_map,
        "df_forms_dict": df_forms_dict,
        "dictionary": dictionary,
        "quality_report": quality_report,
        "insight_panels": insight_panels,
        "buttons": buttons,
        "config_dict": config_dict,
        "df_countries": df_countries,
    }

    PROJECT_CACHE[project_path] = project_data

    if config_dict["save_public_outputs"]:
        logger.info(f" Saving public outputs for project {project_path}")
        save_public_outputs(
            buttons, insight_panels, df_map, df_countries, df_forms_dict, dictionary, quality_report, project_path, config_dict
        )
    logger.info(f" Project data loaded and cached for {project_path}")
    return project_data


def main():
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
    app.server.config.update(
        {
            "SQLALCHEMY_DATABASE_URI": DATABASE_URL if AUTH_ENABLED else None,
            "SECRET_KEY": flask_auth_secrets.get("SECRET_KEY"),
            "SECURITY_PASSWORD_HASH": "bcrypt",
            "SECURITY_PASSWORD_SALT": flask_auth_secrets.get("SECURITY_PASSWORD_SALT"),
            "SECURITY_USER_IDENTITY_ATTRIBUTES": [{"email": {"mapper": "email", "case_insensitive": True}}],
        }
    )
    if AUTH_ENABLED:
        db.init_app(app.server)
        with app.server.app_context():
            global user_datastore
            user_datastore = SQLAlchemyUserDatastore(db, User, None)
            security.init_app(app.server, user_datastore)

    project_paths, names = get_projects()
    logger.debug(f" Found {len(project_paths)} projects: {project_paths}")

    for path, name in zip(project_paths, names):
        try:
            logger.debug(f" Preloading project: {name}")
            _ = load_project_data(path)  # this loads and caches it
        except Exception as e:
            logger.error(f"Failed to load project {path}: {e}")
    # load the first projects layout
    # TODO: allow this to be configurable
    active_project = project_paths[0] if project_paths else None
    if active_project is not None:
        initial_layout = build_project_layout(active_project)
        app.layout = define_shell_layout(active_project, initial_body=initial_layout)

    register_callbacks(app)

    return app


if __name__ == "__main__":
    app = main()
    webbrowser.open("http://127.0.0.1:8050", new=2, autoraise=True)
    app.run_server(debug=True, host="0.0.0.0", port=8050, use_reloader=False)
else:
    app = main()
    server = app.server
