# modals.py
from dash import html, dcc
import dash_bootstrap_components as dbc

login_modal = dbc.Modal(
    id="login-modal",
    is_open=False,
    centered=True,
    backdrop="static",
    keyboard=False,
    children=[
        dbc.ModalHeader(dbc.ModalTitle("Login")),
        dbc.ModalBody([
            html.Form([
                dcc.Input(id="username", type="text", placeholder="Username"),
                html.Br(),
                dcc.Input(id="password", type="password", placeholder="Password"),
                html.Br(),
                html.Button("Submit", id="login-submit", n_clicks=0, type="button"),
                html.Div(id="login-output", style={"color": "red", "margin-top": "10px"}),
            ]),
            html.Div([
                html.Button("Register", id="open-register", n_clicks=0, className="btn btn-link", type="button"),
                html.Div(id="register-launcher-output")
            ])
        ]),
    ]
)

register_modal = dbc.Modal(
    id="register-modal",
    is_open=False,
    centered=True,
    backdrop="static",
    keyboard=False,
    children=[
        dbc.ModalHeader(dbc.ModalTitle("Register")),
        dbc.ModalBody([
            html.Form([
                dcc.Input(id="register-email", type="email", placeholder="Email"),
                html.Br(),
                dcc.Input(id="register-password", type="password", placeholder="Password"),
                html.Br(),
                html.Button("Register", id="register-submit", n_clicks=0, type="button"),
                html.Div(id="register-output", style={"color": "red", "margin-top": "10px"}),
            ])
        ])
    ]
)
