import dash_bootstrap_components as dbc
from dash import dcc, html

# vertex/layout/app_layout.py
from vertex.layout.footer import footer
from vertex.layout.menu import define_menu
from vertex.layout.modals import login_modal, register_modal


def define_shell_layout(init_project_path, initial_body=None):
    return html.Div(
        [
            dcc.Store(id="selected-project-path", data=init_project_path),
            dcc.Store(id="login-state", storage_type="session", data=False),
            # Header
            html.Div(
                [
                    html.H1("VERTEX - Visual Evidence & Research Tool for EXploration", id="title"),
                    html.P("Visual Evidence, Vital Answers"),
                    dbc.Button("Login", id="open-login", color="primary", size="sm", style={"display": "none"}),
                    dbc.Button("Logout", id="logout-button", style={"display": "none"}),
                    html.Div(id="auth-button-container"),
                ],
                style={"position": "absolute", "top": 0, "left": 10, "zIndex": 1000},
            ),
            # Main content area
            html.Div(id="project-body", children=initial_body),
            # Footer
            footer,
            # Modals
            login_modal,
            register_modal,
            # Insights Modal container (content is replaced dynamically)
            dbc.Modal(id="modal", children=[dbc.ModalBody("")], is_open=False, size="xl"),
        ]
    )


def define_inner_layout(fig, buttons, filter_options, map_layout_dict, project_name=None):
    return html.Div(
        [
            dcc.Store(id="button", data={"item": "", "label": "", "suffix": ""}),
            dcc.Store(id="map-layout", data=map_layout_dict),
            # Graph WITHOUT dcc.Loading → Plotly shows its built-in loading overlay instead of flashing
            dcc.Graph(id="world-map", figure=fig, style={"height": "92vh", "margin": "0px"}),
            # Side menu
            define_menu(buttons, filter_options, project_name=project_name),
            html.Div(id="trigger-on-load", style={"display": "none"}),
        ]
    )
