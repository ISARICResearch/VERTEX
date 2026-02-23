import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from vertex.layout.filters import define_filters_controls
from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)


def define_menu(buttons, filter_options=None, project_name=None, project_options=None, selected_project_value=None):
    menu = pd.DataFrame(data=buttons)
    if filter_options is not None:
        menu_items = [define_filters_controls(**filter_options)]
    else:
        menu_items = []

    for item in menu["item"].unique():
        item_children = []
        for index, row in menu.loc[(menu["item"] == item)].iterrows():
            item_children.append(
                dbc.Button(
                    row["label"], id={"type": "open-modal", "index": row["suffix"]}, className="mb-2", style={"width": "100%"}
                )
            )
        menu_items.append(dbc.AccordionItem(title=item, children=item_children))

    # Header with project selector
    menu_header = dbc.ModalHeader(
        html.Div(
            [
                html.H4(
                    f"{project_name}",
                    style={
                        "fontWeight": "bold",
                        "fontSize": "1.5rem",
                        "textAlign": "center",
                        "width": "100%",
                        "margin": 0,
                    },
                ),
                html.Div(style={"margin-top": "5px"}),
                project_selector(project_options=project_options, selected_project=selected_project_value),
                html.Div(style={"margin-top": "5px"}),
            ],
            style={
                "width": "100%",
                "margin": 0,
            },
        ),
        close_button=False,
    )

    menu = html.Div(
        [
            menu_header,  # fixed top
            dbc.ModalBody(
                dbc.Accordion(menu_items, start_collapsed=True),
                style={
                    "overflowY": "auto",
                    "flexGrow": 1,
                    "maxHeight": "calc(100vh - 320px)",  # leaves room for header
                },
            ),
        ],
        style={
            "width": "350px",
            "position": "fixed",
            "bottom": 0,
            "left": 0,
            "minHeight": "45vh",
            "maxHeight": "90vh",
            "height": "auto",
            "display": "flex",
            "flexDirection": "column",
            "zIndex": 1000,
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
            "padding": "10px",
            "boxShadow": "2px 0 6px rgba(0,0,0,0.1)",
        },
    )
    return menu


def project_selector(project_options=None, selected_project=None):
    return dcc.Dropdown(
        id="project-selector",
        options=project_options or [],
        value=selected_project,
        placeholder="Change Project...",
        style={"minWidth": "300px"},
        clearable=False,
    )
