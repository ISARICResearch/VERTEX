from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

from vertex.layout.menu import define_menu
from vertex.layout.modals import login_modal, register_modal

def define_app_layout(
        fig, buttons, filter_options, map_layout_dict, project_name=None):
    
    title = 'VERTEX - Visual Evidence & Research Tool for EXploration'
    subtitle = 'Visual Evidence, Vital Answers'

    isaric_logo = 'ISARIC_logo.png'
    partners_logo_list = [
        'FIOCRUZ_logo.png', 'gh.png', 'puc_rio.png']
    funders_logo_list = [
        'wellcome-logo.png', 'billmelinda-logo.png',
        'uk-international-logo.png', 'FundedbytheEU.png']

    logo_style = {'height': '5vh', 'margin': '2px 10px'}

    app_layout = html.Div([
        dcc.Store(id='login-state', storage_type='session', data=False),
        dcc.Store(id='button', data={'item': '', 'label': '', 'suffix': ''}),
        dcc.Store(id='map-layout', data=map_layout_dict),
        dcc.Graph(
            id='world-map', figure=fig,
            style={'height': '92vh', 'margin': '0px'}),
        html.Div([
                html.H1(title, id='title'),
                html.P(subtitle),
                # Add a hidden button here so it always exists in the layout
                dbc.Button("Login", id="open-login", color="primary", size="sm", style={"display": "none"}),
                dbc.Button("Logout", id="logout-button", style={"display": "none"}),
                html.Div(id="auth-button-container"),
            ],
            style={
                'position': 'absolute',
                'top': 0, 'left': 10,
                'z-index': 1000}),
        define_menu(buttons, filter_options, project_name=project_name),
        html.Div(id='trigger-on-load', style={'display': 'none'}),
        html.Div(
            [
                html.Img(
                    src='/assets/logos/' + isaric_logo,
                    className='img-fluid',
                    style={'height': '7vh', 'margin': '2px 10px'}),
                html.P('In partnership with: ', style={'display': 'inline'})] +
            [html.Img(
                src='/assets/logos/' + logo,
                className='img-fluid',
                style=logo_style) for logo in partners_logo_list] +
            [html.P('    With funding from: ', style={'display': 'inline'})] +
            [html.Img(
                src='/assets/logos/' + logo,
                className='img-fluid',
                style=logo_style) for logo in funders_logo_list],
            style={
                'position': 'absolute', 'bottom': 0,
                'width': 'calc(100% - 350px)', 'margin-left': '350px',
                'background-color': '#FFFFFF',
                'z-index': 0, }),

            login_modal,
            register_modal,
    ])
    return app_layout