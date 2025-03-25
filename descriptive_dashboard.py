import dash
import json
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import sys
# import IsaricDraw as idw
# import redcap_config as rc_config
import getREDCapData as getRC
# from insight_panels import *
# from insight_panels.__init__ import __all__ as ip_list
import os
import shutil
import importlib.util
import webbrowser
# import dash_auth
# import flask_caching as fc


############################################
# PROJECT PATHS (CHANGE THIS)
############################################

# init_project_path = 'projects/ARChetypeCRF_mpox_synthetic/'
# init_project_path = 'projects/ARChetypeCRF_dengue_synthetic/'
init_project_path = 'projects/ARChetypeCRF_h5nx_synthetic/'

# def get_project_path():
#     with open('vertex_projects_path.txt', 'r') as f:
#         text = f.read()
#         path_dict = {
#             x.split('=')[0].strip(): eval(x.split('=')[1].strip())
#             for x in text.split('\n') if len(x) > 0
#         }
#         projects_path = os.path.join(path_dict['vertex_projects_path'], '')
#         init_project_path = os.path.join(
#             projects_path, 'projects', path_dict['init_project_name'], '')
#     return projects_path, init_project_path


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
# CONFIG
############################################


def get_config(init_project_path, config_defaults):
    config_file = os.path.join(init_project_path, 'config_file.json')
    try:
        with open(config_file, 'r') as json_data:
            config_dict = json.load(json_data)
        _ = config_dict['api_key']
        _ = config_dict['api_url']
    except Exception:
        error_message = f'''config_file.json is required in \
{init_project_path}. This file must contain both "api_key" and "api_url".'''
        print(error_message)
        raise SystemExit
    # The default for the list of insight panels is all that exist in the
    # relevant folder (which may or not be specified in config)
    if 'insight_panels_path' not in config_dict.keys():
        rel_insight_panels_path = config_defaults['insight_panels_path']
        config_dict['insight_panels_path'] = rel_insight_panels_path
    # Get a list of python files in the repository (excluding e.g. __init__.py)
    insight_panels_path = os.path.join(
        init_project_path, config_dict['insight_panels_path'])
    for (_, _, filenames) in os.walk(insight_panels_path):
        insight_panels = [
            file.split('.py')[0] for file in filenames
            if file.endswith('.py') and not file.startswith('_')]
        break
    config_defaults['insight_panels'] = insight_panels
    # Add default items where the config file doesn't include these
    config_defaults = {
        k: v
        for k, v in config_defaults.items() if k not in config_dict.keys()}
    config_dict = {**config_dict, **config_defaults}
    if any([x not in insight_panels for x in config_dict['insight_panels']]):
        print('The following insight panels in config_file.json do not exist:')
        missing_insight_panels = [
            x for x in config_dict['insight_panels']
            if x not in insight_panels]
        print('\n'.join(missing_insight_panels))
        print('These are ignored and will not appear in the dashboard.')
        config_dict['insight_panels'] = [
            x for x in config_dict['insight_panels'] if x in insight_panels]
    if any([x not in config_dict['insight_panels'] for x in insight_panels]):
        print('''The following insight panel files are not listed in \
config_file.json:''')
        missing_insight_panels = [
            x for x in insight_panels
            if x not in config_dict['insight_panels']]
        print('\n'.join(missing_insight_panels))
        print('''These will not appear in the dashboard. Please add them \
to the list "insight_panels" in config_file.json to include them in \
the dashboard.''')
    return config_dict


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
# MAP
############################################


def merge_data_with_countries(df_map):
    '''Add country variable to df_map and merge with country metadata.'''
    contries_path = 'assets/countries.csv'
    countries = pd.read_csv(contries_path, encoding='latin-1')
    countries.rename(columns={
        'Code': 'country_iso',
        'Country': 'country_name',
        'Region': 'country_region',
        'Income group': 'country_income'}, inplace=True)
    df_map = pd.merge(df_map, countries, on='country_iso', how='left')
    return df_map


def get_countries(df_map):
    df_countries = df_map[['country_iso', 'country_name', 'subjid']]
    df_countries = df_countries.groupby(
        ['country_iso', 'country_name']).count().reset_index()
    df_countries.rename(columns={'subjid': 'country_count'}, inplace=True)
    return df_countries


def interpolate_colors(colors, n):
    ''' Interpolate among multiple hex colors.'''
    # Convert all hex colors to RGB
    rgbs = [
        tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        for color in colors]

    interpolated_colors = []
    # Number of transitions is one less than the number of colors
    transitions = len(colors) - 1

    # Calculate the number of steps for each transition
    steps_per_transition = n // transitions
    steps_per_transition = (
        [steps_per_transition + 1]*(n % transitions) +
        [steps_per_transition]*(transitions - (n % transitions)))

    # Interpolate between each pair of colors
    for i in range(transitions):
        for step in range(steps_per_transition[i]):
            interpolated_rgb = [
                int(rgbs[i][j] + (float(step)/steps_per_transition[i])*(
                    rgbs[i+1][j]-rgbs[i][j]))
                for j in range(3)]
            interpolated_colors.append(
                f'rgb({interpolated_rgb[0]}, ' +
                f'{interpolated_rgb[1]},' +
                f'{interpolated_rgb[2]})')

    # Append the last color
    if len(interpolated_colors) < n:
        interpolated_colors.append(
            f'rgb({rgbs[-1][0]}, {rgbs[-1][1]}, {rgbs[-1][2]})')

    if len(rgbs) > n:
        print(len(rgbs))
        interpolated_colors = [
            f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})' for rgb in rgbs[:n]]
    return interpolated_colors


def get_map_colorscale(
        df_countries,
        map_percentile_cutoffs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]):
    cutoffs = np.percentile(
        df_countries['country_count'], map_percentile_cutoffs)
    if df_countries['country_count'].count() < len(map_percentile_cutoffs):
        cutoffs = df_countries['country_count'].sort_values()
    cutoffs = cutoffs / df_countries['country_count'].max()
    num_colors = len(cutoffs)
    cutoffs = np.insert(np.repeat(cutoffs, 2)[:-1], 0, 0)
    colors = interpolate_colors(
        ['0000FF', '00EA66', 'A7FA00', 'FFBE00', 'FF7400', 'FF3500'],
        num_colors)
    colors = np.repeat(colors, 2)
    custom_scale = [[x, y] for x, y in zip(cutoffs, colors)]
    return custom_scale


def create_map(df_countries, map_layout_dict=None):
    geojson = os.path.join(
        'https://raw.githubusercontent.com/',
        'martynafford/natural-earth-geojson/master/',
        '50m/cultural/ne_50m_admin_0_countries.json')
    # geojson = os.path.join(
    #     'https://raw.githubusercontent.com/',
    #     'datasets/geo-countries/blob/main/data/countries.geojson')
    # geojson = os.path.join(
    #     'https://raw.githubusercontent.com/',
    #     'johan/world.geo.json/master/countries.geo.json')

    map_colorscale = get_map_colorscale(df_countries)

    fig = go.Figure(go.Choroplethmap(
        geojson=geojson,
        featureidkey='properties.ADM0_A3',
        locations=df_countries['country_iso'],
        z=df_countries['country_count'],
        text=df_countries['country_name'],
        colorscale=map_colorscale,
        showscale=True,
        zmin=1,
        zmax=df_countries['country_count'].max(),
        marker_line_color='black',
        marker_opacity=0.5,
        marker_line_width=0.3,
        colorbar={
            'bgcolor': 'rgba(0,0,0,0)', 'thickness': 20, 'ticklen': 1,
            'x': 1, 'xref': 'paper', 'xanchor': 'left'},
    ))
    fig.update_layout(map_layout_dict)
    # fig.update_layout({'width': 10.5})
    return fig


############################################
# APP LAYOUT
############################################


def define_filters_and_controls(
        sex_options, age_options, country_options, outcome_options):
    filters = dbc.AccordionItem(
        title='Filters and Controls',
        children=[
            html.Label('Sex at birth:'),
            dcc.Checklist(
                id='gender-checkboxes',
                options=sex_options,
                value=[option['value'] for option in sex_options],
            ),
            html.Div(style={'margin-top': '20px'}),
            html.Label('Age:'),
            dcc.RangeSlider(
                id='age-slider',
                min=age_options['min'],
                max=age_options['max'],
                step=age_options['step'],
                marks=age_options['marks'],
                value=age_options['value']
            ),
            html.Div(style={'margin-top': '20px'}),
            html.Label('Outcome:'),
            dcc.Checklist(
                id='outcome-checkboxes',
                options=outcome_options,
                value=[option['value'] for option in outcome_options],
            ),
            html.Div(style={'margin-top': '20px'}),
            html.Div([
                html.Div(
                    id='country-display', children='Country:',
                    style={'cursor': 'pointer'}),
                dbc.Fade(
                    html.Div([
                        dcc.Checklist(
                            id='country-selectall',
                            options=[{'label': 'Select all', 'value': 'all'}],
                            value=['all']
                        ),
                        dcc.Checklist(
                            id='country-checkboxes',
                            options=country_options,
                            value=[
                                option['value'] for option in country_options],
                            style={'overflowY': 'auto', 'maxHeight': '200px'}
                        )
                    ]),
                    id='country-fade',
                    is_in=False,
                    appear=False,
                )
            ]),
        ], style={'overflowY': 'auto', 'maxHeight': '60vh'},
    )
    return filters


def define_menu(buttons, filter_options, project_name=None):
    initial_modal = dbc.Modal(
        id='modal',
        children=[dbc.ModalBody('')],  # Placeholder content
        is_open=False,
        size='xl'
    )
    menu = pd.DataFrame(data=buttons)
    menu_items = [define_filters_and_controls(**filter_options)]
    cont = 0
    for item in menu['item'].unique():
        if (cont == 0):
            item_children = [initial_modal]
        else:
            item_children = []
        cont += 1
        for index, row in menu.loc[(menu['item'] == item)].iterrows():
            item_children.append(dbc.Button(
                row['label'],
                id={'type': 'open-modal', 'index': row['suffix']},
                className='mb-2', style={'width': '100%'}))
        menu_items.append(dbc.AccordionItem(
            title=item,
            children=item_children))
    # menu = dbc.Accordion(
    #     menu_items,
    #     start_collapsed=True,
    #     style={
    #         'width': '300px', 'position': 'fixed', 'bottom': 0, 'left': 0,
    #         'z-index': 1000, 'background-color': 'rgba(255, 255, 255, 0.8)',
    #         'padding': '10px'})
    menu = [dbc.ModalBody([dbc.Accordion(menu_items, start_collapsed=True)])]
    if project_name is not None:
        menu_header = [dbc.ModalHeader(html.H1(
            project_name,
            style={
                'fontSize': '2vmin', 'fontWeight': 'bold',
                'text-align': 'center', 'padding': '5px', 'width': '300px'}
        ), close_button=False)]
        menu = menu_header + menu
    menu = html.Div(
        menu,
        style={
            'width': '300px', 'position': 'fixed', 'bottom': 0, 'left': 0,
            'z-index': 1000, 'background-color': 'rgba(255, 255, 255, 0.8)',
            'padding': '10px'})
    return menu


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
        dcc.Store(id='button', data={'item': '', 'label': '', 'suffix': ''}),
        dcc.Store(id='map-layout', data=map_layout_dict),
        # dcc.Store(id='button', data=buttons),
        dcc.Graph(
            id='world-map', figure=fig,
            style={'height': '92vh', 'margin': '0px'}),
        html.Div([
                html.H1(title, id='title'),
                html.P(subtitle)
            ],
            style={
                'position': 'absolute',
                'top': 0, 'left': 10,
                'z-index': 1000}),
        define_menu(buttons, filter_options, project_name=project_name),
        html.Div(
            [
                html.Img(
                    src='assets/logos/' + isaric_logo,
                    className='img-fluid',
                    style={'height': '7vh', 'margin': '2px 10px'}),
                html.P('In partnership with: ', style={'display': 'inline'})] +
            [html.Img(
                src='assets/logos/' + logo,
                className='img-fluid',
                style=logo_style) for logo in partners_logo_list] +
            [html.P('    With funding from: ', style={'display': 'inline'})] +
            [html.Img(
                src='assets/logos/' + logo,
                className='img-fluid',
                style=logo_style) for logo in funders_logo_list],
            style={
                'position': 'absolute', 'bottom': 0,
                'width': 'calc(100% - 300px)', 'margin-left': '300px',
                'background-color': '#FFFFFF',
                'z-index': 0, }),
    ])
    return app_layout


############################################
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
            df_map=df_map, df_forms_dict=df_forms_dict,
            dictionary=dictionary, quality_report=quality_report,
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
                ], active_item='item-0')
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
        outcome_options, add_row=None):
    row = dbc.Row([
        dbc.Col([
            html.H6('Sex at birth:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Checklist(
                    id='gender-checkboxes-modal',
                    options=sex_options,
                    value=[option['value'] for option in sex_options],
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
            html.H6('Country:', style={'margin-right': '10px'}),
            html.Div([
                html.Div(
                    id='country-display-modal',
                    children='Country:', style={'cursor': 'pointer'}),
                dbc.Fade(
                    html.Div([
                        dcc.Checklist(
                            id='country-selectall-modal',
                            options=[{'label': 'Select all', 'value': 'all'}],
                            value=['all']
                        ),
                        dcc.Checklist(
                            id='country-checkboxes-modal',
                            options=country_options,
                            value=[
                                option['value'] for option in country_options],
                            style={'overflowY': 'auto', 'maxHeight': '100px'}
                        )
                    ]),
                    id='country-fade-modal',
                    is_in=False,
                    appear=False,
                )
            ]),
        ], width=5),
        dbc.Col([
            html.H6('Outcome:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Checklist(
                    id='outcome-checkboxes-modal',
                    options=outcome_options,
                    value=[option['value'] for option in outcome_options],
                )
            ])
        ], width=2)
    ])
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
    row_list = [row, row_button]
    if add_row is not None:
        row_list = [row, add_row, row_button]
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
        # dbc.Popover(
        #     [
        #         dbc.PopoverHeader('Download', style={'fontWeight': 'bold'}),
        #         dbc.PopoverBody([
        #             html.Div('Raw data'),
        #             dbc.Button(
        #                 '.csv',
        #                 outline=True, color='secondary',
        #                 className='mr-1', id=f'csv_download_{suffix}',
        #                 style={}, size='sm'),
        #             html.Div('Chart', style={'marginTop': 5}),
        #             dbc.Button(
        #                 '.png',
        #                 outline=True, color='secondary',
        #                 className='mr-1', id=f'png_download_{suffix}',
        #                 style={}, size='sm'),
        #             html.Div(
        #                 'Advanced',
        #                 style={'marginTop': 5, 'display': 'none'}),
        #             dbc.Button(
        #                 'Downloads Area',
        #                 outline=True, color='secondary',
        #                 className='mr-1', id='btn-popover-line-download-land',
        #                 style={'display': 'none'}, size='sm'),
        #             ]),
        #     ],
        #     id=f'modal_download_popover_menu_{suffix}',
        #     target=f'modal_download_popover_{suffix}',
        #     # style={'maxHeight': '300px', 'overflowY': 'auto'},
        #     trigger='legacy',
        #     placement='top',
        #     hide_arrow=False,
        # ),
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
        Output('world-map', 'figure'),
        [
            Input('gender-checkboxes', 'value'),
            Input('age-slider', 'value'),
            Input('outcome-checkboxes', 'value'),
            Input('country-checkboxes', 'value')
        ],
        [State('map-layout', 'data')],
        prevent_initial_call=True
    )
    def update_map(genders, age_range, outcomes, countries, map_layout_dict):
        df_map['filters_age'] = df_map['filters_age'].astype(float)
        df_map_filtered = df_map[(
            (df_map['filters_sex'].isin(genders)) &
            ((
                df_map['filters_age'] >= age_range[0]) |
                df_map['filters_age'].isna()) &
            ((
                df_map['filters_age'] <= age_range[1]) |
                df_map['filters_age'].isna()) &
            (df_map['filters_outcome'].isin(outcomes)) &
            (df_map['filters_country'].isin(countries)))]
        if df_map_filtered.empty:
            geojson = os.path.join(
                'https://raw.githubusercontent.com/',
                'martynafford/natural-earth-geojson/master/',
                '50m/cultural/ne_50m_admin_0_map_units.json')
            #
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
        [Output('country-checkboxes', 'value'),
         Output('country-selectall', 'options'),
         Output('country-selectall', 'value')],
        [Input('country-selectall', 'value'),
         Input('country-checkboxes', 'value')],
        [State('country-checkboxes', 'options')]
    )
    def update_country_selection(
            select_all_value, selected_countries, all_countries_options):
        ctx = dash.callback_context

        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            output = [
                selected_countries,
                [{'label': 'Unselect all', 'value': 'all'}], ['all']]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'country-selectall':
            if 'all' in select_all_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    [option['value'] for option in all_countries_options],
                    [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{'label': 'Select all', 'value': 'all'}], []]
        elif trigger_id == 'country-checkboxes':
            if len(selected_countries) == len(all_countries_options):
                # All countries are selected manually
                output = [
                    selected_countries,
                    [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # Some countries are deselected
                output = [
                    selected_countries,
                    [{'label': 'Select all', 'value': 'all'}], []]
        else:
            output = [
                selected_countries,
                [{'label': 'Select all', 'value': 'all'}], select_all_value]
        return output

    @app.callback(
        Output('country-fade', 'is_in'),
        [Input('country-display', 'n_clicks')],
        [State('country-fade', 'is_in')]
    )
    def toggle_fade(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    @app.callback(
        Output('country-display', 'children'),
        [Input('country-checkboxes', 'value')],
        [State('country-checkboxes', 'options')]
    )
    def update_country_display(selected_values, all_options):
        if not selected_values:
            return 'Country:'

        # Create a dictionary to map values to labels
        value_label_map = {
            option['value']: option['label'] for option in all_options}

        # Build the display string
        selected_labels = [
            value_label_map[val] for val in selected_values
            if val in value_label_map]
        display_text = ', '.join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            output = f'Country: {selected_labels[0]}, '
            output += f'+{len(selected_labels) - 1} more...'
        else:
            output = f'Country: {display_text}'
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
                df_map=df_map, df_forms_dict=df_forms_dict,
                dictionary=dictionary, quality_report=quality_report,
                suffix=suffix, filepath=filepath, save_inputs=False)
            button = {
                **insight_panels[suffix].define_button(), **{'suffix': suffix}}
            modal = create_modal(visuals, button, filter_options)
            output = not is_open, modal, True, button
        return output

    @app.callback(
        [Output('country-checkboxes-modal', 'value'),
         Output('country-selectall-modal', 'options'),
         Output('country-selectall-modal', 'value')],
        [Input('country-selectall-modal', 'value'),
         Input('country-checkboxes-modal', 'value')],
        [State('country-checkboxes-modal', 'options')]
    )
    def update_country_selection_modal(
            select_all_value, selected_countries, all_countries_options):
        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            output = [
                selected_countries,
                [{'label': 'Unselect all', 'value': 'all'}], ['all']]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        #
        if trigger_id == 'country-selectall-modal':
            if 'all' in select_all_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    [option['value'] for option in all_countries_options],
                    [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{'label': 'Select all', 'value': 'all'}], []]
        elif trigger_id == 'country-checkboxes-modal':
            if len(selected_countries) == len(all_countries_options):
                # All countries are selected manually
                output = [
                    selected_countries,
                    [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # Some countries are deselected
                output = [
                    selected_countries,
                    [{'label': 'Select all', 'value': 'all'}], []]
        else:
            output = [
                selected_countries,
                [{'label': 'Select all', 'value': 'all'}], select_all_value]
        return output

    @app.callback(
        Output('country-fade-modal', 'is_in'),
        [Input('country-display-modal', 'n_clicks')],
        [State('country-fade-modal', 'is_in')]
    )
    def toggle_fade_modal(n_clicks, is_in):
        state = is_in
        if n_clicks:
            state = not is_in
        return state

    @app.callback(
        Output('country-display-modal', 'children'),
        [Input('country-checkboxes-modal', 'value')],
        [State('country-checkboxes-modal', 'options')]
    )
    def update_country_display_modal(selected_values, all_options):
        if not selected_values:
            return 'Country:'

        # Create a dictionary to map values to labels
        value_label_map = {
            option['value']: option['label'] for option in all_options}

        # Build the display string
        selected_labels = [
            value_label_map[val] for val in selected_values
            if val in value_label_map]
        display_text = ', '.join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            output = f'Country: {selected_labels[0]}, '
            output += f'+{len(selected_labels) - 1} more...'
        else:
            output = f'Country: {display_text}'
        return output

    @app.callback(
        [
            Output('modal', 'children', allow_duplicate=True),
            Output('gender-checkboxes-modal', 'value', allow_duplicate=True),
            Output('age-slider-modal', 'value', allow_duplicate=True),
            Output('outcome-checkboxes-modal', 'value', allow_duplicate=True),
            Output('country-checkboxes-modal', 'value', allow_duplicate=True)
        ],
        [Input('submit-button-modal', 'n_clicks')],
        [
            State('button', 'data'),
            State('gender-checkboxes-modal', 'value'),
            State('age-slider-modal', 'value'),
            State('outcome-checkboxes-modal', 'value'),
            State('country-checkboxes-modal', 'value')
        ],
        prevent_initial_call=True
    )
    def update_figures(
            click, button, genders, age_range, outcomes, countries):
        df_map['filters_age'] = df_map['filters_age'].astype(float)
        df_map_filtered = df_map[(
            (df_map['filters_sex'].isin(genders)) &
            ((
                df_map['filters_age'] >= age_range[0]) |
                df_map['filters_age'].isna()) &
            ((
                df_map['filters_age'] <= age_range[1]) |
                df_map['filters_age'].isna()) &
            (df_map['filters_outcome'].isin(outcomes)) &
            (df_map['filters_country'].isin(countries)))]
        df_map_filtered = df_map_filtered.reset_index(drop=True)

        df_forms_filtered = df_forms_dict.copy()
        for key in df_forms_filtered.keys():
            df_filtered = df_forms_filtered[key].copy()
            df_filtered['filters_age'] = (
                df_filtered['filters_age'].astype(float))
            df_filtered = df_filtered[(
                (df_filtered['filters_sex'].isin(genders)) &
                ((
                    df_filtered['filters_age'] >= age_range[0]) |
                    df_filtered['filters_age'].isna()) &
                ((
                    df_filtered['filters_age'] <= age_range[1]) |
                    df_filtered['filters_age'].isna()) &
                (df_filtered['filters_outcome'].isin(outcomes)) &
                (df_filtered['filters_country'].isin(countries)))]
            df_forms_filtered[key] = df_filtered.reset_index(drop=True)

        suffix = button['suffix']
        # If all dataframes in the dict are empty, return an empty modal
        df_list = [df_map_filtered] + list(df_forms_filtered.values())
        if all([x.empty for x in df_list]):
            modal = ()
        else:
            visuals = insight_panels[suffix].create_visuals(
                df_map=df_map_filtered, df_forms_dict=df_forms_filtered,
                dictionary=dictionary, quality_report=quality_report,
                filepath=filepath, suffix=suffix,
                save_inputs=save_inputs)
            modal = create_modal(visuals, button, filter_options)
        output = modal, genders, age_range, outcomes, countries
        return output

    # @dash.callback(
    #     Output('line-graph-modal-title', 'children'),
    #     [Input('main_data', 'data'),
    #      Input('dictionary', 'data')]
    # )
    # def get_data_for_modal(stored_main_data, stored_dictionary):
    #     if stored_main_data is None:
    #         raise dash.exceptions.PreventUpdate  # Prevent if no data
    #     # Convert JSON data back to a DataFrame
    #     main_data = pd.DataFrame(stored_main_data)
    #     dictionary = pd.DataFrame(stored_dictionary)
    #     return main_data, dictionary

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
        suppress_callback_exceptions=True)

    # cache = fc.Cache(
    #     app.server,
    #     config={
    #         'CACHE_TYPE': 'SimpleCache',  # Use in-memory cache
    #         'CACHE_DEFAULT_TIMEOUT': 300,  # Cache timeout in seconds (5 min)
    #     }
    # )

    # try:
    #     print('Password required')
    #     p_info = os.environ['p_info']
    #     u_info = os.environ['u_info']
    #     VALID_USERNAME_PASSWORD_PAIRS = {u_info: p_info}
    #     auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
    # except Exception:
    #     print('Password not required')

    config_defaults = {
        'project_name': None,
        'map_layout_center_latitude': 6,
        'map_layout_center_longitude': -75,
        'map_layout_zoom': 1.7,
        'save_public_outputs': False,
        'save_base_files_to_public_path': False,
        'public_path': 'PUBLIC/',
        'save_filtered_public_outputs': False,
        'insight_panels_path': 'insight_panels/',
        'insight_panels': [],
    }

    # projects_path, init_project_path = get_project_path()
    config_dict = get_config(init_project_path, config_defaults)

    insight_panels_path = os.path.join(
        init_project_path, config_dict['insight_panels_path'])
    insight_panels, buttons = get_insight_panels(
        config_dict, insight_panels_path)

    api_url = config_dict['api_url']
    api_key = config_dict['api_key']
    get_data_from_api = (api_url is not None) and (api_key is not None)

    if get_data_from_api:
        # get_data_kwargs = {}
        print('Retrieving data from the API')
        df_map, df_forms_dict, dictionary, quality_report = (
            getRC.get_redcap_data(api_url, api_key))  # **get_data_kwargs

    if get_data_from_api is False:
        try:
            vertex_dataframes_path = os.path.join(
                init_project_path, config_dict['vertex_dataframes_path'])
            vertex_dataframes = os.listdir(vertex_dataframes_path)
            df_map = pd.read_csv(
                os.path.join(vertex_dataframes_path, 'df_map.csv'))
            dictionary = pd.read_csv(
                os.path.join(vertex_dataframes_path, 'vertex_dictionary.csv'))
            # dictionary = dictionary.fillna('')
            if 'quality_report.json' in vertex_dataframes:
                quality_report_file = os.path.join(
                    vertex_dataframes_path, 'quality_report.json')
                with open(quality_report_file, 'r') as json_data:
                    quality_report = json.load(json_data)
            else:
                quality_report = {}
            exclude_files = ('df_map.csv', 'vertex_dictionary.csv')
            vertex_dataframes = [
                file for file in vertex_dataframes
                if file.endswith('.csv') and (file not in exclude_files)]
            df_forms_dict = {
                k.split('.csv')[0]: pd.read_csv(
                    os.path.join(vertex_dataframes_path, k))
                for k in vertex_dataframes}
        except Exception:
            print('Could not load the VERTEX dataframes.')
            raise

    df_map_with_countries = merge_data_with_countries(df_map)
    df_countries = get_countries(df_map_with_countries)
    print(df_countries)

    filter_columns_dict = {
        'subjid': 'subjid',
        'demog_sex': 'filters_sex',
        'demog_age': 'filters_age',
        'country_iso': 'filters_country',
        'outco_binary_outcome': 'filters_outcome'
    }

    mapbox_style = ['open-street-map', 'carto-positron']
    map_layout_dict = dict(
        mapbox_style=mapbox_style[1],
        mapbox_zoom=config_dict['map_layout_zoom'],
        mapbox_center={
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
    age_options['marks'] = {ii: str(ii) for ii in age_range}
    age_options['value'] = [age_options['min'], age_options['max']]

    country_options = [
        {'label': x[1], 'value': x[0]}
        for x in df_countries.sort_values(by='country_iso').values]

    outcome_options = [
        {'label': 'Death', 'value': 'Death'},
        {'label': 'Censored', 'value': 'Censored'},
        {'label': 'Discharged', 'value': 'Discharged'}
    ]

    filter_options = {
        'sex_options': sex_options, 'age_options': age_options,
        'country_options': country_options, 'outcome_options': outcome_options}

    app.layout = define_app_layout(
        fig, buttons, filter_options,
        map_layout_dict, config_dict['project_name'])

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
        print(f'Saving files for public dashboard to "{public_path}"')
        os.makedirs(
            os.path.dirname(os.path.join(public_path, '')), exist_ok=True)
        os.makedirs(
            os.path.dirname(os.path.join(public_path, 'data', '')),
            exist_ok=True)
        for ip in config_dict['insight_panels']:
            os.makedirs(
                os.path.dirname(os.path.join(public_path, 'data', ip, '')),
                exist_ok=True)
        buttons = get_visuals(
            buttons, insight_panels,
            df_map=df_map, df_forms_dict=df_forms_dict,
            dictionary=dictionary, quality_report=quality_report,
            filepath=os.path.join(public_path, 'data', ''))
        os.makedirs(os.path.dirname(public_path), exist_ok=True)
        if config_dict['save_base_files_to_public_path']:
            shutil.copy('descriptive_dashboard_public.py', public_path)
            shutil.copy('IsaricDraw.py', public_path)
            shutil.copy('requirements.txt', public_path)
            assets_path = os.path.join(public_path, 'assets/')
            os.makedirs(os.path.dirname(assets_path), exist_ok=True)
            shutil.copytree('assets', assets_path, dirs_exist_ok=True)
        metadata_file = os.path.join(
            public_path, 'data/dashboard_metadata.txt')
        with open(metadata_file, 'w') as metadata:
            metadata.write(repr(buttons))
        data_file = os.path.join(public_path, 'data/dashboard_data.csv')
        df_countries.to_csv(data_file, index=False)
        config_json_file = os.path.join(
            public_path, 'public_config_file.json')
        with open(config_json_file, 'w') as file:
            save_config_keys = [
                'project_name', 'map_layout_center_latitude',
                'map_layout_center_longitude', 'map_layout_zoom']
            save_config_dict = {k: config_dict[k] for k in save_config_keys}
            json.dump(save_config_dict, file)

    webbrowser.open('http://127.0.0.1:8050', new=2, autoraise=True)
    app.run_server(debug=True, use_reloader=False)
    return


if __name__ == '__main__':
    main()
