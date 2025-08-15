import dash
import json
from dash import dcc, html, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from flask_login import LoginManager, login_user, logout_user, current_user

from flask_security import SQLAlchemyUserDatastore, Security
from flask_security.utils import login_user, verify_and_update_password, hash_password
import pandas as pd
import sys
import os
import shutil
import importlib.util

import webbrowser
import requests
import secrets
from sqlalchemy import create_engine, Table, Column, String, MetaData, select, \
    TIMESTAMP, Boolean, Text, UUID
from sqlalchemy.orm import Session
import uuid
import boto3
from urllib.parse import quote_plus

from vertex.models import User, Project
import vertex.getREDCapData as getRC
from vertex.loader import get_config, load_vertex_data, config_defaults
from vertex.layout.modals import login_modal, register_modal
from vertex.layout.app_layout import define_app_layout
from vertex.map import create_map, get_countries, get_map_colorscale, interpolate_colors, merge_data_with_countries

# Settings
secret_name = "rds!db-472cc9c8-1f3e-4547-b84d-9b0742de8b9a"
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

# init_project_path = 'projects/ARChetypeCRF_mpox_synthetic/'
# init_project_path = 'projects/ARChetypeCRF_dengue_synthetic/'
# init_project_path = 'projects/ARChetypeCRF_h5nx_synthetic/'
init_project_path = 'projects/ARChetypeCRF_h5nx_synthetic_mf/'


############################################
# IMPORT
############################################


def import_from_path(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


############################################
# CACHE DATA
############################################


# def cache_memoize(cache, df_map, dd):
#     # Define a function to fetch data and cache it
#     @cache.memoize(timeout=300)  # Cache timeout of 5 minutes
#     def memoize_data():
#         return df_map.to_dict('records')
#     @cache.memoize(timeout=300)  # Cache timeout of 5 minutes
#     def memoize_dictionary():
#         return dd.to_dict('records')
#     # End of cache function
#     return


############################################
# APP LAYOUT
############################################

def generate_html_text(text):
    text_list = text.strip('\n').split('\n')
    div_list = []
    for line in text_list:
        strong_list = line.split('<strong>')
        for string in strong_list:
            if '</strong>' in string:
                strong, not_strong = string.split('</strong>')
                div_list.append(html.Div(html.Strong(strong)))
                div_list.append(html.Div(not_strong))
            else:
                div_list.append(html.Div(string))
        div_list.append(html.Br())
    div = html.Div(div_list[:-1])
    return div


############################################
# Get insight panels
############################################

def get_insight_panels(config_dict, insight_panels_path):
    # Import insight panels scripts
    insight_panels = {
        x: import_from_path(x, os.path.join(insight_panels_path, x + '.py'))
        for x in config_dict['insight_panels']}
    buttons = [
        {**ip.define_button(), **{'suffix': suffix}}
        for suffix, ip in insight_panels.items()]
    return insight_panels, buttons


def get_visuals(
        buttons, insight_panels, df_map, df_forms_dict,
        dictionary, quality_report, filepath):
    for ii in range(len(buttons)):
        suffix = buttons[ii]['suffix']
        visuals = insight_panels[suffix].create_visuals(
            df_map=df_map.copy(),
            df_forms_dict={k: v.copy() for k, v in df_forms_dict.items()},
            dictionary=dictionary.copy(), quality_report=quality_report,
            suffix=suffix, filepath=filepath, save_inputs=True)
        buttons[ii]['graph_ids'] = [id for _, id, _, _ in visuals]
    return buttons


############################################
# Modal creation
############################################


def create_modal(visuals, button, filter_options):
    if visuals is None:
        insight_children = []
        about_str = ''
    else:
        insight_children = [
            dbc.Tabs([
                dbc.Tab(dbc.Row(
                    [dbc.Col(dcc.Graph(figure=figure), id=id)]), label=label)
                for figure, id, label, _ in visuals], active_tab='tab-0')]
        # This text appears after clicking the insight panel's About button
        about_list = ['Information about each visual in the insight panel:']
        about_list += [
            '<strong>' + label + '</strong>' + about
            for _, _, label, about in visuals]
        about_str = '\n'.join(about_list)

    try:
        title = button['item'] + ': ' + button['label']
    except Exception:
        title = ''

    instructions_str = open('assets/instructions.txt', 'r').read()

    modal = [
        dbc.ModalHeader(html.H3(
            title,
            id='line-graph-modal-title',
            style={'fontSize': '2vmin', 'fontWeight': 'bold'})
        ),
        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title='Filters and Controls',
                    children=[
                        define_filters_controls_modal(**filter_options)
                    ]),
                dbc.AccordionItem(
                    title='Insights', children=insight_children)
                ], active_item='item-1')
            ], style={
                'overflowY': 'auto', 'minHeight': '75vh', 'maxHeight': '75vh'}
        ),
        define_footer_modal(
            generate_html_text(instructions_str),
            generate_html_text(about_str))
    ]
    return modal


def define_filters_controls_modal(
        sex_options, age_options, country_options,
        admdate_options,  # disease_options,
        outcome_options,
        add_row=None):
    filter_rows = [dbc.Row([
        dbc.Col([
            html.H6('Sex at birth:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Checklist(
                    id='sex-checkboxes-modal',
                    options=sex_options,
                    value=[option['value'] for option in sex_options],
                    inputStyle={'margin-right': '2px'}
                )
            ])
        ], width=2),
        dbc.Col([
            html.H6('Age:', style={'margin-right': '10px'}),
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id='age-slider-modal',
                        min=age_options['min'],
                        max=age_options['max'],
                        step=age_options['step'],
                        marks=age_options['marks'],
                        value=age_options['value']
                    )
                ], style={'width': '100%'})  # Apply style to this div
            ])
        ], width=3),
        dbc.Col([
            html.H6('Admission date:', style={'margin-right': '10px'}),
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id='admdate-slider-modal',
                        min=admdate_options['min'],
                        max=admdate_options['max'],
                        step=admdate_options['step'],
                        marks=admdate_options['marks'],
                        value=admdate_options['value']
                    )
                ], style={'width': '100%'})  # Apply style to this div
            ])
        ], width=3),
        dbc.Col([
            html.H6('Country:', style={'margin-right': '10px'}),
            html.Div([
                html.Div(
                    id='country-display-modal',
                    children='Country:', style={'cursor': 'pointer'}),
                dbc.Fade(
                    html.Div([
                        dcc.Checklist(
                            id='country-selectall-modal',
                            options=[{
                                'label': 'Select all',
                                'value': 'all'
                            }],
                            value=['all'],
                            inputStyle={'margin-right': '2px'}
                        ),
                        dcc.Checklist(
                            id='country-checkboxes-modal',
                            options=country_options,
                            value=[
                                option['value'] for option in country_options],
                            style={'overflowY': 'auto', 'maxHeight': '100px'},
                            inputStyle={'margin-right': '2px'}
                        )
                    ]),
                    id='country-fade-modal',
                    is_in=True,
                    appear=True,
                )
            ]),
        ], width=2),
        dbc.Col([
            html.H6('Outcome:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Checklist(
                    id='outcome-checkboxes-modal',
                    options=outcome_options,
                    value=[option['value'] for option in outcome_options],
                    inputStyle={'margin-right': '2px'}
                )
            ])
        ], width=2)
    ])]
    row_button = dbc.Row([
        dbc.Col([
            dbc.Button(
                'Submit',
                id='submit-button-modal',
                color='primary', className='mr-2')
            ],
            width={'size': 6, 'offset': 3},
            style={'text-align': 'center'})  # Center the button
    ])
    row_list = filter_rows + [row_button]
    if add_row is not None:
        row_list = filter_rows + [add_row, row_button]
    filters = dbc.Row([dbc.Col(row_list)])
    return filters


def define_footer_modal(instructions, about):
    footer = dbc.ModalFooter([
        html.Div([
            dbc.Button(
                'About',
                id='modal_about_popover',
                color='info', size='sm', style={'margin-right': '5px'}),
            dbc.Button(
                'Instructions',
                id='modal_instruction_popover',
                size='sm', style={'margin-right': '5px'}),
            # dbc.Button(
            #     'Download',
            #     id=f'modal_download_popover_{suffix}',
            #     size='sm', style={'margin-right': '5px'}),
            # dbc.Button('Close', id='modal_patChar_close_popover',  size='sm')
        ], className='ml-auto'),
        dbc.Popover(
            [
                dbc.PopoverHeader(
                    'Instructions',
                    style={'fontWeight': 'bold'}),
                dbc.PopoverBody(instructions)
            ],
            # id='modal-line-instructions-popover',
            # is_open=False,
            target='modal_instruction_popover',
            trigger='hover',
            placement='top',
            hide_arrow=False,
            # style={'zIndex':1}
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader('About', style={'fontWeight': 'bold'}),
                dbc.PopoverBody(about),
            ],
            # id='modal-line-guide-popover',
            # is_open=False,
            target='modal_about_popover',
            trigger='hover',
            placement='top',
            hide_arrow=False,
            # style={'zIndex':1}
        ),
    ])
    return footer


############################################
# Dashboard callbacks
############################################


def register_callbacks(
        app, insight_panels, df_map,
        df_forms_dict, dictionary, quality_report, filter_options,
        filepath, save_inputs):
    

    @app.callback(
        Output("page-content", "children"),
        Input("selected-project-path", "data"),
        prevent_initial_call=True
    )
    def load_project_layout(project_path):
        print(f"Loading project layout for: {project_path}")
        if not project_path:
            raise PreventUpdate
        
        config_dict = get_config(project_path, config_defaults)
        insight_panels_path = os.path.join(project_path, config_dict['insight_panels_path'])
        insight_panels, buttons = get_insight_panels(config_dict, insight_panels_path)


        df_map, df_forms_dict, dictionary, quality_report = load_vertex_data(init_project_path, config_dict)
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
        df_map = pd.merge(df_map_with_countries, df_filters, on='subjid', how='left')
        df_forms_dict = {
            form: pd.merge(df_form, df_filters, on='subjid', how='left')
            for form, df_form in df_forms_dict.items()
        }

        # Generate layout
        map_layout_dict = dict(
            map_style='carto-positron',
            map_zoom=config_dict['map_layout_zoom'],
            map_center={
                'lat': config_dict['map_layout_center_latitude'],
                'lon': config_dict['map_layout_center_longitude']},
            margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        )
        fig = create_map(df_countries, map_layout_dict)

        sex_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}, {'label': 'Other / Unknown', 'value': 'Other / Unknown'}]
        max_age = max((100, df_map['demog_age'].max()))
        age_options = {
            'min': 0,
            'max': max_age,
            'step': 10,
            'marks': {i: {'label': str(i)} for i in range(0, max_age + 1, 10)},
            'value': [0, max_age]
        }
        print(df_map.keys())
        admdate_yyyymm = pd.date_range(start=df_map['pres_date'].min(), end=df_map['pres_date'].max(), freq='MS')
        admdate_options = {
            'min': 0,
            'max': len(admdate_yyyymm) - 1,
            'step': 1,
            'marks': {i: {'label': d.strftime('%Y-%m')} for i, d in enumerate(admdate_yyyymm)},
            'value': [0, len(admdate_yyyymm) - 1]
        }

        country_options = [{'label': r['country_name'], 'value': r['country_iso']} for _, r in df_countries.iterrows()]
        outcome_options = [{'label': v, 'value': v} for v in df_map['filters_outcome'].dropna().unique()]

        filter_options = {
            'sex_options': sex_options,
            'age_options': age_options,
            'admdate_options': admdate_options,
            'country_options': country_options,
            'outcome_options': outcome_options,
        }

        layout = define_app_layout(fig, buttons, filter_options, map_layout_dict, config_dict['project_name'])

        return layout


    ## Change the current project
    @app.callback(
        Output("selected-project-path", "data"),
        Input("project-selector", "value"),
        prevent_initial_call=True
    )
    def set_project_path(selected_value):
        print(f"Selected project path: {selected_value}")
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
            # Input('disease-checkboxes', 'value'),
            Input('outcome-checkboxes', 'value')
        ],
        [State('map-layout', 'data')],
        prevent_initial_call=True
    )
    def update_map(
            sex_value, age_value, country_value,
            admdate_value, admdate_marks,  # disease_value,
            outcome_value,
            map_layout_dict):
        df_map['filters_age'] = df_map['filters_age'].astype(float)
        admdate_min = pd.to_datetime(
            admdate_marks[str(admdate_value[0])]['label'])
        admdate_max = pd.to_datetime(
            admdate_marks[str(admdate_value[1])]['label'])
        df_map_filtered = df_map[(
            (df_map['filters_sex'].isin(sex_value)) &
            (
                (df_map['filters_age'] >= age_value[0]) |
                (df_map['filters_age'].isna())
            ) &
            (
                (df_map['filters_age'] <= age_value[1]) |
                (df_map['filters_age'].isna())
            ) &
            (
                (df_map['filters_admdate'] >= admdate_min) |
                (df_map['filters_admdate'].isna())
            ) &
            (
                (df_map['filters_admdate'] <= admdate_max) |
                (df_map['filters_admdate'].isna())
            ) &
            # (df_map['filters_disease'].isin(disease_value)) &
            (df_map['filters_outcome'].isin(outcome_value)) &
            (df_map['filters_country'].isin(country_value))
        )]
        if df_map_filtered.empty:
            geojson = os.path.join(
                'https://raw.githubusercontent.com/',
                'martynafford/natural-earth-geojson/master/',
                '50m/cultural/ne_50m_admin_0_map_units.json')
            # geojson = os.path.join(
            #     'https://raw.githubusercontent.com/',
            #     'johan/world.geo.json/master/countries.geo.json')

            fig = go.Figure(
                go.Choroplethmap(
                    geojson=geojson,
                    featureidkey='properties.ISO_A3'
                ),
                layout=map_layout_dict)
        else:
            df_countries = get_countries(df_map_filtered)
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
                    # ' (scroll down for all)',
                    html.Br(),
                    f'{display_text}'
                ])
        return output

    @app.callback(
        [
            Output('modal', 'is_open', allow_duplicate=True),
            Output('modal', 'children', allow_duplicate=True),
            Output('modal', 'scrollable', allow_duplicate=True),
            Output('button', 'data')
        ],
        [Input({'type': 'open-modal', 'index': ALL}, 'n_clicks')],
        [State('modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_modal(n, is_open):
        ctx = callback_context
        if not ctx.triggered:
            empty_button = {'item': '', 'label': '', 'suffix': ''}
            output = is_open, [], False, empty_button
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            suffix = json.loads(button_id)['index']
            visuals = insight_panels[suffix].create_visuals(
                df_map=df_map.copy(),
                df_forms_dict={k: v.copy() for k, v in df_forms_dict.items()},
                dictionary=dictionary.copy(), quality_report=quality_report,
                suffix=suffix, filepath=filepath, save_inputs=False)
            button = {
                **insight_panels[suffix].define_button(), **{'suffix': suffix}}
            modal = create_modal(visuals, button, filter_options)
            output = not is_open, modal, True, button
        return output

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
            # Output('disease-checkboxes-modal', 'value', allow_duplicate=True),
            Output('outcome-checkboxes-modal', 'value', allow_duplicate=True)
        ],
        [Input('submit-button-modal', 'n_clicks')],
        [
            State('button', 'data'),
            State('sex-checkboxes-modal', 'value'),
            State('age-slider-modal', 'value'),
            State('country-checkboxes-modal', 'value'),
            State('admdate-slider-modal', 'value'),
            State('admdate-slider-modal', 'marks'),
            # State('disease-checkboxes-modal', 'value'),
            State('outcome-checkboxes-modal', 'value')
        ],
        prevent_initial_call=True
    )
    def update_figures(
            click, button,
            sex_value, age_value, country_value,
            admdate_value, admdate_marks,  # disease_value,
            outcome_value):
        df_map['filters_age'] = df_map['filters_age'].astype(float)
        admdate_min = pd.to_datetime(
            admdate_marks[str(admdate_value[0])]['label'])
        admdate_max = pd.to_datetime(
            admdate_marks[str(admdate_value[1])]['label'])
        df_map_filtered = df_map[(
            (df_map['filters_sex'].isin(sex_value)) &
            (
                (df_map['filters_age'] >= age_value[0]) |
                (df_map['filters_age'].isna())
            ) &
            (
                (df_map['filters_age'] <= age_value[1]) |
                (df_map['filters_age'].isna())
            ) &
            (
                (df_map['filters_admdate'] >= admdate_min) |
                (df_map['filters_admdate'].isna())
            ) &
            (
                (df_map['filters_admdate'] <= admdate_max) |
                (df_map['filters_admdate'].isna())
            ) &
            # (df_map['filters_disease'].isin(disease_value)) &
            (df_map['filters_outcome'].isin(outcome_value)) &
            (df_map['filters_country'].isin(country_value))
        )]
        df_map_filtered = df_map_filtered.reset_index(drop=True)

        df_forms_filtered = df_forms_dict.copy()
        for key in df_forms_filtered.keys():
            df_filtered = df_forms_filtered[key].copy()
            df_filtered['filters_age'] = (
                df_filtered['filters_age'].astype(float))
            df_filtered = df_filtered[(
                (df_filtered['filters_sex'].isin(sex_value)) &
                (
                    (df_filtered['filters_age'] >= age_value[0]) |
                    (df_filtered['filters_age'].isna())
                ) &
                (
                    (df_filtered['filters_age'] <= age_value[1]) |
                    (df_filtered['filters_age'].isna())
                ) &
                (
                    (df_filtered['filters_admdate'] >= admdate_min) |
                    (df_filtered['filters_admdate'].isna())
                ) &
                (
                    (df_filtered['filters_admdate'] <= admdate_max) |
                    (df_filtered['filters_admdate'].isna())
                ) &
                # (df_filtered['filters_disease'].isin(disease_value)) &
                (df_filtered['filters_outcome'].isin(outcome_value)) &
                (df_filtered['filters_country'].isin(country_value))
            )]
            df_forms_filtered[key] = df_filtered.reset_index(drop=True)

        suffix = button['suffix']
        # If all dataframes in the dict are empty, return an empty modal
        df_list = [df_map_filtered] + list(df_forms_filtered.values())
        if all([x.empty for x in df_list]):
            modal = ()
        else:
            visuals = insight_panels[suffix].create_visuals(
                df_map=df_map_filtered.copy(),
                df_forms_dict={
                    k: v.copy() for k, v in df_forms_filtered.items()},
                dictionary=dictionary.copy(),
                quality_report=quality_report,
                filepath=filepath, suffix=suffix,
                save_inputs=save_inputs)
            modal = create_modal(visuals, button, filter_options)
        output = (
            modal, sex_value, age_value, country_value,
            admdate_value,  # disease_value,
            outcome_value)
        return output

    # End of callbacks
    return




############################################
# Main
############################################


def main():
    # app.run_server(debug=True, host='0.0.0.0', port='8080')
    print('Starting VERTEX')
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=os.path.join(os.path.dirname(__file__), '..', 'assets'),
        title='Isaric VERTEX',
        suppress_callback_exceptions=True
    )
    
    app.server.config.update({
        "SQLALCHEMY_DATABASE_URI": DATABASE_URL,
        "SECRET_KEY": "mouse_trap_robot_fast_cheese_coffee_gross_back_spain",
        "SECURITY_PASSWORD_HASH": "bcrypt",
        "SECURITY_PASSWORD_SALT": "host_place_china_horse_past_arena_brand_sugar",
        "SECURITY_USER_IDENTITY_ATTRIBUTES": [
            {"email": {"mapper": "email", "case_insensitive": True}}
        ]
    })
    
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='login-state', storage_type='session', data=False),
        dcc.Store(id='selected-project-path'),
        html.Div(id='page-content'),
        html.Div(id="login-output", style={"display": "none"}),
    ])

    db.init_app(app.server)  # now bind db to app
    with app.server.app_context():
        global user_datastore
        user_datastore = SQLAlchemyUserDatastore(db, User, None)
        security.init_app(app.server, user_datastore)

    # projects_path, init_project_path = get_project_path()
    config_dict = get_config(init_project_path, config_defaults)

    insight_panels_path = os.path.join(
        init_project_path, config_dict['insight_panels_path'])
    insight_panels, buttons = get_insight_panels(
        config_dict, insight_panels_path)

    df_map, df_forms_dict, dictionary, quality_report = load_vertex_data(init_project_path, config_dict)

    df_map_with_countries = merge_data_with_countries(df_map)
    df_countries = get_countries(df_map_with_countries)
    print(df_countries)

    filter_columns_dict = {
        'subjid': 'subjid',
        'demog_sex': 'filters_sex',
        'demog_age': 'filters_age',
        'pres_date': 'filters_admdate',
        'country_iso': 'filters_country',
        'outco_binary_outcome': 'filters_outcome'
    }

    map_style = ['open-street-map', 'carto-positron']
    map_layout_dict = dict(
        map_style='carto-positron',  # alternative is 'open-street-map'
        map_zoom=config_dict['map_layout_zoom'],
        map_center={
            'lat': config_dict['map_layout_center_latitude'],
            'lon': config_dict['map_layout_center_longitude']},
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
    )

    fig = create_map(df_countries, map_layout_dict)

    sex_options = [
        {'label': 'Male', 'value': 'Male'},
        {'label': 'Female', 'value': 'Female'},
        {'label': 'Other / Unknown', 'value': 'Other / Unknown'}]

    max_age = max((100, df_map['demog_age'].max()))
    age_options = {'min': 0, 'max': max_age, 'step': 10}
    age_range = range(
        age_options['min'], age_options['max'] + 1, age_options['step'])
    age_options['marks'] = {
        ii: {
            'label': str(ii),
            'style': {
                'text-align': 'right',
                'transform-origin': 'bottom left',
                'transform': 'rotate(-45deg)',
                'margin-left': '-5px',
                'margin-top': '25px',
                'height': '70px',
                'width': '70px'}
        }
        for ii in age_range
    }
    age_options['value'] = [age_options['min'], age_options['max']]

    end_date = (df_map['pres_date'].max() + pd.DateOffset(months=1))
    end_date = end_date.strftime('%Y-%m')
    admdate_yyyymm = pd.date_range(
        start=df_map['pres_date'].min().strftime('%Y-%m'),
        end=end_date,
        freq='MS')
    admdate_yyyymm = [x.strftime('%Y-%m') for x in admdate_yyyymm]
    admdate_options = {
        'min': 0, 'max': len(admdate_yyyymm) - 1, 'step': 1}
    admdate_range = range(
        admdate_options['min'],
        admdate_options['max'] + 1,
        admdate_options['step'])
    admdate_options['marks'] = {
        ii: {
            'label': admdate_yyyymm[ii],
            'style': {
                'text-align': 'right',
                'transform-origin': 'bottom left',
                'transform': 'rotate(-45deg)',
                'margin-left': '-5px',
                'margin-top': '25px',
                'height': '70px',
                'width': '70px'}
        } for ii in admdate_range
    }
    admdate_options['value'] = [admdate_options['min'], admdate_options['max']]

    country_options = [
        {'label': x[1], 'value': x[0]}
        for x in df_countries.sort_values(by='country_iso').values]

    outcome_options = [
        {'label': 'Death', 'value': 'Death'},
        {'label': 'Censored', 'value': 'Censored'},
        {'label': 'Discharged', 'value': 'Discharged'}
    ]

    filter_options = {
        'sex_options': sex_options,
        'age_options': age_options,
        'country_options': country_options,
        'admdate_options': admdate_options,
        # 'disease_options': disease_options,
        'outcome_options': outcome_options}

    app.layout = html.Div([
        dcc.Store(id='selected-project-path', data=init_project_path),
        html.Div(id='page-content', children=define_app_layout(
            fig, buttons, filter_options,
            map_layout_dict, config_dict['project_name']))
    ])

    df_filters = df_map_with_countries[filter_columns_dict.keys()].rename(
        columns=filter_columns_dict)

    df_map = pd.merge(
        df_map_with_countries, df_filters, on='subjid', how='left')
    df_forms_dict = {
        form: pd.merge(df_form, df_filters, on='subjid', how='left')
        for form, df_form in df_forms_dict.items()}

    register_callbacks(
        app, insight_panels, df_map,
        df_forms_dict, dictionary, quality_report, filter_options,
        init_project_path, config_dict['save_filtered_public_outputs'])

    if config_dict['save_public_outputs']:
        public_path = os.path.join(
            init_project_path, config_dict['public_path'])
        if os.path.exists(public_path):
            print(f'Folder "{public_path}" already exists, removing this')
            shutil.rmtree(public_path)
        print(f'Saving files for public dashboard to "{public_path}"')
        os.makedirs(
            os.path.dirname(os.path.join(public_path, '')), exist_ok=True)
        for ip in config_dict['insight_panels']:
            os.makedirs(
                os.path.dirname(os.path.join(public_path, ip, '')),
                exist_ok=True)
        buttons = get_visuals(
            buttons, insight_panels,
            df_map=df_map, df_forms_dict=df_forms_dict,
            dictionary=dictionary, quality_report=quality_report,
            filepath=os.path.join(public_path, ''))
        os.makedirs(os.path.dirname(public_path), exist_ok=True)
        if config_dict['save_base_files_to_public_path']:
            shutil.copy('descriptive_dashboard_public.py', public_path)
            shutil.copy('IsaricDraw.py', public_path)
            shutil.copy('requirements.txt', public_path)
            assets_path = os.path.join(public_path, 'assets/')
            os.makedirs(os.path.dirname(assets_path), exist_ok=True)
            shutil.copytree('assets', assets_path, dirs_exist_ok=True)
        metadata_file = os.path.join(
            public_path, 'dashboard_metadata.json')
        with open(metadata_file, 'w') as file:
            json.dump({'insight_panels': buttons}, file, indent=4)
        data_file = os.path.join(public_path, 'dashboard_data.csv')
        df_countries.to_csv(data_file, index=False)
        config_json_file = os.path.join(
            public_path, 'config_file.json')
        with open(config_json_file, 'w') as file:
            save_config_keys = [
                'project_name', 'map_layout_center_latitude',
                'map_layout_center_longitude', 'map_layout_zoom']
            save_config_dict = {k: config_dict[k] for k in save_config_keys}
            json.dump(save_config_dict, file, indent=4)
    return app

if __name__ == '__main__':
    app = main()
    webbrowser.open('http://127.0.0.1:8050', new=2, autoraise=True)
    app.run_server(debug=True, host='0.0.0.0', port=8050, use_reloader=False)
else:
    app = main()
    server = app.server
