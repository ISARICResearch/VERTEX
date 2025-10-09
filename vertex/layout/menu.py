from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from vertex.layout.filters import define_filters_controls

import json
import pathlib

from vertex.logging.logger import setup_logger
logger = setup_logger(__name__)

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
            project_selector(selected_project=project_name)
        ]),
        close_button=False,    
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

def project_selector(selected_project=None):
    projects, names = get_projects()
    options = []

    for project, name in zip(projects, names):
        options.append({"label": name, "value": project})

    return dcc.Dropdown(
        id="project-selector",
        options=options,
        placeholder="Select a project...",
        style={"minWidth": "300px"},
        clearable=False,
    )

def get_projects():
    project_path = pathlib.Path("projects/")
    logger.info(f"Looking for projects in: {project_path.resolve()}")
    projects = [p for p in project_path.iterdir() if p.is_dir()]
    names = [get_project_name(p) for p in projects]
    logger.info(f"Found projects: {[p.name for p in projects]}")
    return [str(p) + "/" for p in projects], names

def get_project_name(project_path):
    config_file = pathlib.Path(project_path) / "config_file.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                logger.debug(f"Loaded config for {project_path.name}: {config}")
                project_name = config.get("project_name", project_path.name)
        except Exception as e:
            logger.warning(f"Could not read config for {project_path.name}: {e}")
            project_name = project_path.name
    else:
        project_name = project_path.name
    return project_name