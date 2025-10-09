import dash
import json
from dash import dcc, html, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from flask_login import LoginManager, login_user, logout_user, current_user
from flask import session as flask_session

from flask_security import SQLAlchemyUserDatastore, Security
from flask_security.utils import login_user, verify_and_update_password, hash_password
import pandas as pd
import os

import webbrowser
import secrets
from sqlalchemy import create_engine, Table, Column, String, MetaData, select, \
    TIMESTAMP, Boolean, Text, UUID
from sqlalchemy.orm import Session
import uuid
import boto3
from urllib.parse import quote_plus

from vertex.models import User, Project
from vertex.loader import get_config, load_vertex_data, config_defaults, save_public_outputs
from vertex.layout.modals import login_modal, register_modal, create_modal
from vertex.layout.app_layout import define_inner_layout, define_shell_layout
from vertex.layout.insight_panels import get_insight_panels
from vertex.layout.filters import get_filter_options
from vertex.map import create_map, get_countries, merge_data_with_countries, filter_df_map

from vertex.logging.logger import setup_logger
logger = setup_logger(__name__)

# Settings
secret_name = "rds!db-472cc9c8-1f3e-4547-b84d-9b0742de8b9a" #TODO: move to env vars
region_name = "eu-west-2"

# Create a Secrets Manager client
session = boto3.session.Session()
client = session.client(service_name='secretsmanager', region_name=region_name)

# Get secret value
response = client.get_secret_value(SecretId=secret_name)
secret = json.loads(response["SecretString"])

# Extract credentials
username = secret["username"]
password = quote_plus(secret["password"])  # URL-encode password
host = "isaric-user-db.cf4o0aos4r0d.eu-west-2.rds.amazonaws.com"
port = 5432
database = "postgres"

# Build SQLAlchemy connection string
DATABASE_URL = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}?sslmode=require"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Flask login

from flask_security import SQLAlchemyUserDatastore
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy() # we will bind these to the app later
security = Security() 


############################################
# PROJECT PATHS (CHANGE THIS)
############################################

init_project_path = 'projects/ARChetypeCRF_mpox_synthetic/'
# init_project_path = 'projects/ARChetypeCRF_dengue_synthetic/'
# init_project_path = 'projects/ARChetypeCRF_h5nx_synthetic/'
# init_project_path = 'projects/ARChetypeCRF_h5nx_synthetic_mf/'

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
            layout = html.Div([
                html.H4("Error loading project"),
                html.Pre(str(e))
            ])

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
        Output('world-map', 'figure'),
        [
            Input('sex-checkboxes', 'value'),
            Input('age-slider', 'value'),
            Input('country-checkboxes', 'value'),
            Input('admdate-slider', 'value'),
            Input('admdate-slider', 'marks'),
            Input('outcome-checkboxes', 'value'),
            State('selected-project-path', 'data'),
        ],
        [State('map-layout', 'data')],
    )
    def update_map(sex_value, age_value, country_value,
                admdate_value, admdate_marks, outcome_value,
                project_path, map_layout_dict):
        project_data = get_project_data(project_path)

        if not project_data:
            raise PreventUpdate

        df_map = project_data['df_map']
        df_filtered = filter_df_map(
            df_map, sex_value, age_value,
            country_value, admdate_value,
            admdate_marks, outcome_value
        )

        if df_filtered.empty:
            geojson = (
                "https://raw.githubusercontent.com/"
                "martynafford/natural-earth-geojson/master/"
                "50m/cultural/ne_50m_admin_0_map_units.json"
            )
            fig = go.Figure(
                go.Choroplethmap(
                    geojson=geojson,
                    featureidkey='properties.ISO_A3'
                ),
                layout=map_layout_dict
            )
        else:
            df_countries = get_countries(df_filtered)
            fig = create_map(df_countries, map_layout_dict)

        return fig


    @app.callback(
        [
            Output('country-selectall', 'value'),
            Output('country-selectall', 'options'),
            Output('country-checkboxes', 'value')
        ],
        [
            Input('country-selectall', 'value'),
            Input('country-checkboxes', 'value')
        ],
        [State('country-checkboxes', 'options')]
    )
    def update_country_selection(
            selectall_value, country_value, country_options):
        ctx = dash.callback_context

        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            output = [
                ['all'],
                [{'label': 'Unselect all', 'value': 'all'}],
                country_value
            ]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'country-selectall':
            if 'all' in selectall_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    ['all'],
                    [{'label': 'Unselect all', 'value': 'all'}],
                    [option['value'] for option in country_options],
                ]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{'label': 'Select all', 'value': 'all'}], []]
        elif trigger_id == 'country-checkboxes':
            if len(country_value) == len(country_options):
                # All countries are selected manually
                output = [
                    ['all'],
                    [{'label': 'Unselect all', 'value': 'all'}],
                    country_value
                ]
            else:
                # Some countries are deselected
                output = [
                    [],
                    [{'label': 'Select all', 'value': 'all'}],
                    country_value
                ]
        else:
            output = [
                selectall_value,
                [{'label': 'Select all', 'value': 'all'}],
                country_value
            ]
        return output

    @app.callback(
        Output('country-fade', 'is_in'),
        [Input('country-display', 'n_clicks')],
        [State('country-fade', 'is_in')]
    )
    def toggle_country_fade(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    @app.callback(
        Output('country-display', 'children'),
        [Input('country-checkboxes', 'value')],
        [State('country-checkboxes', 'options')]
    )
    def update_country_display(country_value, country_options):
        if not country_value:
            output = html.Div([
                html.B('Country:'),
                # ' (scroll down for all)',
                html.Br(),
                'None selected'
            ])
        else:
            # Create a dictionary to map values to labels
            value_label_map = {
                option['value']: option['label'] for option in country_options}

            # Build the display string
            selected_labels = [
                value_label_map[val] for val in country_value
                if val in value_label_map]
            display_text = ', '.join(selected_labels)

            if len(display_text) > 35:  # Adjust character limit as needed
                if len(selected_labels) == 1:
                    output = html.Div([
                        html.B('Country:'),
                        # ' (scroll down for all)',
                        html.Br(),
                        f'{selected_labels[0]}'])
                else:
                    output = html.Div([
                        html.B('Country:'),
                        # ' (scroll down for all)',
                        html.Br(),
                        f'{selected_labels[0]}, ',
                        f'+{len(selected_labels) - 1} more...'])
            else:
                output = html.Div([
                    html.B('Country:'),
                    html.Br(),
                    f'{display_text}'
                ])
        return output

    @app.callback(
        [
            Output("modal", "is_open"),
            Output("modal", "children"),
            Output("button", "data"), 
        ],
        Input({"type": "open-modal", "index": ALL}, "n_clicks"),
        State("selected-project-path", "data"),
        prevent_initial_call=True
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
        modal_content = create_modal(visuals, button, get_filter_options(project_data["df_map"]))

        return True, modal_content, button


    @app.callback(
        [
            Output('country-selectall-modal', 'value'),
            Output('country-selectall-modal', 'options'),
            Output('country-checkboxes-modal', 'value')
        ],
        [
            Input('country-selectall-modal', 'value'),
            Input('country-checkboxes-modal', 'value')
        ],
        [State('country-checkboxes-modal', 'options')]
    )
    def update_country_selection_modal(
            selectall_value, country_value, country_options):
        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            output = [
                ['all'],
                [{'label': 'Unselect all', 'value': 'all'}],
                country_value
            ]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        #
        if trigger_id == 'country-selectall-modal':
            if 'all' in selectall_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    ['all'],
                    [{'label': 'Unselect all', 'value': 'all'}],
                    [option['value'] for option in country_options]
                ]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{'label': 'Select all', 'value': 'all'}], []]
        elif trigger_id == 'country-checkboxes-modal':
            if len(country_value) == len(country_options):
                # All countries are selected manually
                output = [
                    ['all'],
                    [{'label': 'Unselect all', 'value': 'all'}],
                    country_value
                ]
            else:
                # Some countries are deselected
                output = [
                    [],
                    [{'label': 'Select all', 'value': 'all'}],
                    country_value
                ]
        else:
            output = [
                selectall_value,
                [{'label': 'Select all', 'value': 'all'}],
                country_value
            ]
        return output

    @app.callback(
        Output('country-fade-modal', 'is_in'),
        [Input('country-display-modal', 'n_clicks')],
        [State('country-fade-modal', 'is_in')]
    )
    def toggle_country_fade_modal(n_clicks, is_in):
        state = is_in
        if n_clicks:
            state = not is_in
        return state

    @app.callback(
        Output("auth-button-container", "children"),
        Input("login-state", "data")
    )
    def render_auth_button(is_logged_in):
        return html.Div([
            dbc.Button(
                "Login",
                id="open-login",
                color="primary",
                size="md",
                style={"display": "inline-block" if not is_logged_in else "none"}
            ),
            dbc.Button(
                "Logout",
                id="logout-button",
                color="danger",
                size="md",
                style={"display": "inline-block" if is_logged_in else "none"}
            )
        ])
            
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
        prevent_initial_call=True
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
                print(user)
                if user and verify_and_update_password(password, user):
                    login_user(user)
                    return True, False, ""
                else:
                    print(f"[DEBUG] Invalid login attempt for user: {username}")
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
        prevent_initial_call=True
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
        Output('country-display-modal', 'children'),
        [Input('country-checkboxes-modal', 'value')],
        [State('country-checkboxes-modal', 'options')]
    )
    def update_country_display_modal(country_value, country_options):
        if not country_value:
            return 'Country:'

        # Create a dictionary to map values to labels
        value_label_map = {
            option['value']: option['label'] for option in country_options}

        # Build the display string
        selected_labels = [
            value_label_map[val] for val in country_value
            if val in value_label_map]
        display_text = ', '.join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            output = f'{selected_labels[0]}, '
            output += f'+{len(selected_labels) - 1} more...'
        else:
            output = f'Country: {display_text}'
        return output

    @app.callback(
        [
            Output('modal', 'children', allow_duplicate=True),
            Output('sex-checkboxes-modal', 'value', allow_duplicate=True),
            Output('age-slider-modal', 'value', allow_duplicate=True),
            Output('country-checkboxes-modal', 'value', allow_duplicate=True),
            Output('admdate-slider-modal', 'value', allow_duplicate=True),
            Output('outcome-checkboxes-modal', 'value', allow_duplicate=True),
        ],
        [Input('submit-button-modal', 'n_clicks')],
        [
            State('button', 'data'),
            State('sex-checkboxes-modal', 'value'),
            State('age-slider-modal', 'value'),
            State('country-checkboxes-modal', 'value'),
            State('admdate-slider-modal', 'value'),
            State('admdate-slider-modal', 'marks'),
            State('outcome-checkboxes-modal', 'value'),
            State('selected-project-path', 'data'),   # ✅ project path state
        ],
        prevent_initial_call=True
    )
    def update_figures(
        n_clicks, button,
        sex_value, age_value, country_value,
        admdate_value, admdate_marks,
        outcome_value, project_path
    ):
        print("[DEBUG] updating figures")
        if not button or 'suffix' not in button:
            raise PreventUpdate

        suffix = button['suffix']
        project_data = get_project_data(project_path)
        if not project_data:
            raise PreventUpdate

        df_map = project_data['df_map']
        df_forms_dict = project_data['df_forms_dict']
        dictionary = project_data['dictionary']
        quality_report = project_data['quality_report']

        # Filter the main map
        df_map_filtered = filter_df_map(
            df_map, sex_value, age_value,
            country_value, admdate_value,
            admdate_marks, outcome_value
        )

        # Filter forms
        df_forms_filtered = {}
        for key, df_form in df_forms_dict.items():
            df_forms_filtered[key] = filter_df_map(
                df_form, sex_value, age_value,
                country_value, admdate_value,
                admdate_marks, outcome_value
            )

        # If everything is empty, return blank modal
        df_list = [df_map_filtered] + list(df_forms_filtered.values())
        if all(df.empty for df in df_list):
            return (), sex_value, age_value, country_value, admdate_value, outcome_value

        # Otherwise rebuild visuals
        visuals = project_data['insight_panels'][suffix].create_visuals(
            df_map=df_map_filtered.copy(),
            df_forms_dict={k: v.copy() for k, v in df_forms_filtered.items()},
            dictionary=dictionary.copy(),
            quality_report=quality_report,
            filepath=project_path,
            suffix=suffix,
            save_inputs=project_data['config_dict']['save_filtered_public_outputs'],
        )

        modal = create_modal(visuals, button, get_filter_options(df_map))

        return (
            modal,
            sex_value, age_value,
            country_value, admdate_value,
            outcome_value,
        )

    # End of callbacks
    return




############################################
# Main
############################################

def build_project_layout(project_path):
    print(f"[DEBUG] Building project layout for: {project_path}")
    if not project_path:
        raise PreventUpdate

    # If cached, reuse project data
    project_data = PROJECT_CACHE.get(project_path)
    if project_data:
        print(f"[DEBUG] Using cached project data for {project_path}")
    else:
        print(f"[DEBUG] No cache found, loading fresh data for {project_path}")
        config_dict = get_config(project_path, config_defaults)
        insight_panels_path = os.path.join(project_path, config_dict['insight_panels_path'])
        insight_panels, buttons = get_insight_panels(config_dict, insight_panels_path)

        df_map, df_forms_dict, dictionary, quality_report = load_vertex_data(project_path, config_dict)
        df_map = df_map.reset_index(drop=True)
        df_map_with_countries = merge_data_with_countries(df_map)
        df_countries = get_countries(df_map_with_countries)

        filter_columns_dict = {
            'subjid': 'subjid',
            'demog_sex': 'filters_sex',
            'demog_age': 'filters_age',
            'pres_date': 'filters_admdate',
            'country_iso': 'filters_country',
            'outco_binary_outcome': 'filters_outcome'
        }

        df_filters = df_map_with_countries[filter_columns_dict.keys()].rename(columns=filter_columns_dict)
        df_map = pd.merge(df_map_with_countries, df_filters, on='subjid', how='left').reset_index(drop=True)
        df_forms_dict = {
            form: pd.merge(df_form, df_filters, on='subjid', how='left').reset_index(drop=True)
            for form, df_form in df_forms_dict.items()
        }

        project_data = {
            'df_map': df_map,
            'df_forms_dict': df_forms_dict,
            'dictionary': dictionary,
            'quality_report': quality_report,
            'insight_panels': insight_panels,
            'buttons': buttons,
            'config_dict': config_dict,
            'df_countries': df_countries,  # might be useful later
        }

        PROJECT_CACHE[project_path] = project_data

        # Optional: public outputs only when building fresh
        if config_dict['save_public_outputs']:
            save_public_outputs(
                buttons, insight_panels, df_map, df_countries,
                df_forms_dict, dictionary, quality_report,
                project_path, config_dict
            )

    # Always build layout from project_data
    map_layout_dict = dict(
        map_style='carto-positron',
        map_zoom=project_data['config_dict']['map_layout_zoom'],
        map_center={
            'lat': project_data['config_dict']['map_layout_center_latitude'],
            'lon': project_data['config_dict']['map_layout_center_longitude'],
        },
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
    )
    fig = create_map(project_data['df_countries'], map_layout_dict)

    # Build filter options (from cached data)
    sex_options = [
        {'label': 'Male', 'value': 'Male'},
        {'label': 'Female', 'value': 'Female'},
        {'label': 'Other / Unknown', 'value': 'Other / Unknown'},
    ]
    max_age = max((100, project_data['df_map']['demog_age'].max()))
    age_options = {
        'min': 0, 'max': max_age, 'step': 10,
        'marks': {i: {'label': str(i)} for i in range(0, max_age + 1, 10)},
        'value': [0, max_age],
    }
    admdate_yyyymm = pd.date_range(
        start=project_data['df_map']['pres_date'].min(),
        end=project_data['df_map']['pres_date'].max(),
        freq='MS',
    )
    admdate_options = {
        'min': 0, 'max': len(admdate_yyyymm) - 1, 'step': 1,
        'marks': {i: {'label': d.strftime('%Y-%m')} for i, d in enumerate(admdate_yyyymm)},
        'value': [0, len(admdate_yyyymm) - 1],
    }
    country_options = [
        {'label': r['country_name'], 'value': r['country_iso']}
        for _, r in project_data['df_countries'].iterrows()
    ]
    outcome_options = [
        {'label': v, 'value': v}
        for v in project_data['df_map']['filters_outcome'].dropna().unique()
    ]

    filter_options = {
        'sex_options': sex_options,
        'age_options': age_options,
        'admdate_options': admdate_options,
        'country_options': country_options,
        'outcome_options': outcome_options,
    }

    layout = define_inner_layout(
        fig,
        project_data['buttons'],
        filter_options,
        map_layout_dict,
        project_data['config_dict']['project_name'],
    )
    return layout


def main():
    print('Starting VERTEX')
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=os.path.join(os.path.dirname(__file__), '..', 'assets'),
        title='Isaric VERTEX',
        suppress_callback_exceptions=True
    )

    # Flask / DB config
    app.server.config.update({
        "SQLALCHEMY_DATABASE_URI": DATABASE_URL,
        "SECRET_KEY": "mouse_trap_robot_fast_cheese_coffee_gross_back_spain",
        "SECURITY_PASSWORD_HASH": "bcrypt",
        "SECURITY_PASSWORD_SALT": "host_place_china_horse_past_arena_brand_sugar",
        "SECURITY_USER_IDENTITY_ATTRIBUTES": [
            {"email": {"mapper": "email", "case_insensitive": True}}
        ]
    })
    db.init_app(app.server)
    with app.server.app_context():
        global user_datastore
        user_datastore = SQLAlchemyUserDatastore(db, User, None)
        security.init_app(app.server, user_datastore)

    initial_layout = build_project_layout(init_project_path)
    app.layout = define_shell_layout(init_project_path, initial_body=initial_layout)

    register_callbacks(app)

    return app

if __name__ == '__main__':
    app = main()
    webbrowser.open('http://127.0.0.1:8050', new=2, autoraise=True)
    app.run_server(debug=True, host='0.0.0.0', port=8050, use_reloader=False)
else:
    app = main()
    server = app.server
