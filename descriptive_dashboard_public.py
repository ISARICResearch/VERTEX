import dash
import json
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import sys
import IsaricDraw as idw
import os


############################################
# MAP
############################################


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

    # Interpolate between each pair of colors
    for i in range(transitions):
        for step in range(steps_per_transition):
            interpolated_rgb = [
                int(rgbs[i][j] + (float(step)/steps_per_transition)*(
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
    return interpolated_colors


def get_map_colorscale(
        df_countries,
        map_percentile_cutoffs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]):
    cutoffs = np.percentile(
        df_countries['country_count'], map_percentile_cutoffs)
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
    geojson = 'https://raw.githubusercontent.com/johan/world.geo.json/master/'
    geojson = geojson + 'countries.geo.json'

    map_colorscale = get_map_colorscale(df_countries)

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=df_countries['country_iso'],
        z=df_countries['country_count'],
        text=df_countries['country_name'],
        colorscale=map_colorscale,
        showscale=True,
        zmin=1,
        zmax=df_countries['country_count'].max(),
        marker_opacity=0.5,
        marker_line_width=0,
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


def define_menu(buttons):
    initial_modal = dbc.Modal(
        id='modal',
        children=[dbc.ModalBody('')],  # Placeholder content
        is_open=False,
        size='xl'
    )
    menu = pd.DataFrame(data=buttons)
    menu.drop(columns=['visuals'], inplace=True)
    menu_items = []
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
    menu = dbc.Accordion(
        menu_items,
        start_collapsed=True,
        style={
            'width': '300px', 'position': 'fixed', 'bottom': 0, 'left': 0,
            'z-index': 1000, 'background-color': 'rgba(255, 255, 255, 0.8)',
            'padding': '10px'})
    return menu


# def define_app_layout(fig, buttons, filter_options, map_layout_dict):
def define_app_layout(fig, buttons, map_layout_dict):
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
        # define_menu(buttons, filter_options),
        define_menu(buttons),
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


def get_visuals(path, buttons):
    visuals = {}
    for file in os.listdir(path):
        if file.endswith('.txt') & file.startswith('fig'):
            new_file = eval(open(os.path.join(path, file), 'r').read())
            fig_id = new_file['fig_id']
            data = tuple(
                pd.read_csv(path + name) for name in new_file['fig_data'])
            data = data[0] if (len(data) == 1) else data
            fig_fun = eval('idw.' + new_file['fig_name'])
            visuals[fig_id] = fig_fun(data, **new_file['fig_arguments'])

    for ii in range(len(buttons)):
        buttons[ii]['visuals'] = tuple(
            [visuals[id] for id in buttons[ii]['graph_ids']])
    return buttons


############################################
# Modal creation
############################################


# def create_modal(visuals, button, filter_options):
def create_modal(visuals, button):
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
                # dbc.AccordionItem(
                #     title='Filters and Controls',
                #     children=[
                #         define_filters_controls_modal(**filter_options)
                #     ]),
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
        app, buttons):
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
            button_ind = [
                ii for ii in range(len(buttons))
                if buttons[ii]['suffix'] == suffix][0]
            visuals = buttons[button_ind]['visuals']
            button = {
                'item': buttons[button_ind]['item'],
                'label': buttons[button_ind]['label'],
                'suffix': suffix}
            modal = create_modal(visuals, button)
            output = not is_open, modal, True, button
        return output

    # End of callbacks
    return


############################################
# Main
############################################


def main():
    # app.run_server(debug=True, host='0.0.0.0', port='8080')
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True)

    mapbox_style = ['open-street-map', 'carto-positron']
    map_layout_dict = dict(
        mapbox_style=mapbox_style[1],
        mapbox_zoom=1.7,
        mapbox_center={'lat': 6, 'lon': -75},
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
    )

    path = 'test/'
    metadata_file = 'dashboard_metadata.txt'
    metadata = eval(open(os.path.join(path, metadata_file), 'r').read())
    buttons = get_visuals(path, metadata)

    df_countries = pd.read_csv(path + 'dashboard_data.csv')
    fig = create_map(df_countries, map_layout_dict)
    # geojson = 'https://raw.githubusercontent.com/johan/world.geo.json/'
    # geojson = geojson + 'master/countries.geo.json'
    # fig = go.Figure(
    #     go.Choroplethmapbox(geojson=geojson),
    #     layout=map_layout_dict)

    app.layout = define_app_layout(fig, buttons, map_layout_dict)

    _ = register_callbacks(app, buttons)

    app.run_server(debug=True)
    return


if __name__ == '__main__':
    main()
