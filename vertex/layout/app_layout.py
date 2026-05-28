import dash_bootstrap_components as dbc
from dash import dcc, html

# vertex/layout/app_layout.py
from vertex.layout.footer import footer
from vertex.layout.menu import define_menu


def define_shell_layout(init_project_path, initial_body=None, auth_controls=None, initial_login_state=False):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="selected-project-path", data=init_project_path),
            dcc.Store(id="login-state", storage_type="session", data=initial_login_state),
            # Header
            html.Div(
                [
                    html.H1("VERTEX - Visual Evidence & Research Tool for EXploration", id="title"),
                    html.P("Visual Evidence, Vital Answers"),
                    html.Div(
                        id="auth-button-container",
                        children=auth_controls,
                        style={"position": "fixed", "top": "12px", "right": "16px", "zIndex": 2000},
                    ),
                ],
                style={"position": "absolute", "top": 0, "left": 10, "zIndex": 1000},
            ),
            # Main content area
            dcc.Loading(
                id="loading-project",
                type="default",  # or "circle", "dot"
                children=html.Div(id="project-body", children=initial_body),
            ),
            # Footer
            footer,
            # Insights Modal container (content is replaced dynamically)
            dbc.Modal(id="modal", children=[dbc.ModalBody("")], is_open=False, size="xl"),
        ]
    )


def define_inner_layout(
    fig, buttons, map_layout_dict, filter_options=None, project_name=None, project_options=None, selected_project_value=None
):
    return html.Div(
        [
            dcc.Store(id="button", data={"item": "", "label": "", "suffix": ""}),
            dcc.Store(id="map-layout", data=map_layout_dict),
            # Graph WITHOUT dcc.Loading → Plotly shows its built-in loading overlay instead of flashing
            dcc.Graph(id="world-map", figure=fig, style={"height": "92vh", "margin": "0px"}),
            # Side menu
            define_menu(
                buttons,
                filter_options=filter_options,
                project_name=project_name,
                project_options=project_options,
                selected_project_value=selected_project_value,
            ),
            html.Div(id="trigger-on-load", style={"display": "none"}),
        ]
    )
