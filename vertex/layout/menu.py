from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from vertex.layout.filters import define_filters_controls

def define_menu(buttons, filter_options, project_name=None):
    menu = pd.DataFrame(data=buttons)
    menu_items = [define_filters_controls(**filter_options)]

    for item in menu['item'].unique():
        item_children = []
        for index, row in menu.loc[(menu['item'] == item)].iterrows():
            item_children.append(
                dbc.Button(
                    row['label'],
                    id={'type': 'open-modal', 'index': row['suffix']},
                    className='mb-2',
                    style={'width': '100%'}
                )
            )
        menu_items.append(
            dbc.AccordionItem(title=item, children=item_children)
        )

    # Header with project selector
    menu_header = dbc.ModalHeader(
        html.Div([
            html.Label("Project:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id="project-selector",
                options=[
                    {"label": "ARChetypeCRF_mpox_synthetic", "value": "projects/ARChetypeCRF_mpox_synthetic/"},
                    {"label": "ARChetypeCRF_dengue_synthetic", "value": "projects/ARChetypeCRF_dengue_synthetic/"},
                    {"label": "ARChetypeCRF_h5nx_synthetic", "value": "projects/ARChetypeCRF_h5nx_synthetic/"},
                    {"label": "ARChetypeCRF_h5nx_synthetic_mf", "value": "projects/ARChetypeCRF_h5nx_synthetic_mf/"},
                ],
                placeholder="Select a project...",
                value="projects/ARChetypeCRF_h5nx_synthetic_mf/",  # default
                style={"minWidth": "300px"}
            )
        ])
    )

    menu = html.Div(
        [menu_header, dbc.ModalBody([dbc.Accordion(menu_items, start_collapsed=True)])],
        style={
            'width': '350px',
            'position': 'fixed',
            'bottom': 0,
            'left': 0,
            'zIndex': 1000,
            'background-color': 'rgba(255, 255, 255, 0.8)',
            'padding': '10px'
        }
    )
    return menu
