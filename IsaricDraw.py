from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import IsaricAnalytics as ia


default_height = 430


def filters_controls(suffix, country_dropdown_options):
    row = dbc.Row([
        dbc.Col([
            html.H6('Gender:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Checklist(
                    id=f'gender-checkboxes_{suffix}',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Other / Unknown', 'value': 'Other / Unknown'}
                    ],
                    value=['Male', 'Female', 'Other / Unknown']
                )
            ])
        ], width=2),
        dbc.Col([
            html.H6('Age:', style={'margin-right': '10px'}),
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id=f'age-slider_{suffix}',
                        min=0,
                        max=100,
                        step=10,
                        marks={i: str(i) for i in range(0, 101, 10)},
                        value=[0, 100]
                    )
                ], style={'width': '100%'})  # Apply style to this div
            ])
        ], width=3),
        dbc.Col([
            html.H6('Country:', style={'margin-right': '10px'}),
            html.Div([
                html.Div(
                    id=f'country-display_{suffix}',
                    children='Country:', style={'cursor': 'pointer'}),
                dbc.Fade(
                    html.Div([
                        dcc.Checklist(
                            id=f'country-selectall_{suffix}',
                            options=[{'label': 'Select all', 'value': 'all'}],
                            value=['all']
                        ),
                        dcc.Checklist(
                            id=f'country-checkboxes_{suffix}',
                            options=country_dropdown_options,
                            value=[
                                option['value']
                                for option in country_dropdown_options],
                            style={'overflowY': 'auto', 'maxHeight': '100px'}
                        )
                    ]),
                    id=f'country-fade_{suffix}',
                    is_in=False,
                    appear=False,
                )
            ]),
        ], width=5),
        dbc.Col([
            html.H6('Outcome:', style={'margin-right': '10px'}),
            html.Div([
                dcc.Checklist(
                    id=f'outcome-checkboxes_{suffix}',
                    options=[
                        {'label': 'Death', 'value': 'Death'},
                        {'label': 'Censored', 'value': 'Censored'},
                        {'label': 'Discharged', 'value': 'Discharged'}
                    ],
                    value=['Death', 'Censored', 'Discharged']
                )
            ])
        ], width=2)
    ])
    row_button = dbc.Row([
        dbc.Col([
            row,
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Submit',
                        id=f'submit-button_{suffix}',
                        color='primary', className='mr-2')],
                    width={'size': 6, 'offset': 3},
                    style={'text-align': 'center'})  # Center the button
            ])
        ])
    ])
    return row_button


def ModalFooter(suffix, instructions, about):
    footer = dbc.ModalFooter([
        html.Div([
            dbc.Button(
                'About',
                id=f'modal_about_popover_{suffix}',
                color='info', size='sm', style={'margin-right': '5px'}),
            dbc.Button(
                'Instructions',
                id=f'modal_instruction_popover_{suffix}',
                size='sm', style={'margin-right': '5px'}),
            dbc.Button(
                'Download',
                id=f'modal_download_popover_{suffix}',
                size='sm', style={'margin-right': '5px'}),
            # dbc.Button('Close', id='modal_patChar_close_popover',  size='sm')
        ], className='ml-auto'),
        dbc.Popover(
            [
                dbc.PopoverHeader('Instructions', style={'fontWeight': 'bold'}),
                dbc.PopoverBody(instructions)
            ],
            # id='modal-line-instructions-popover',
            # is_open=False,
            target=f'modal_instruction_popover_{suffix}',
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
            target=f'modal_about_popover_{suffix}',
            trigger='hover',
            placement='top',
            hide_arrow=False,
            # style={'zIndex':1}
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader('Download', style={'fontWeight': 'bold'}),
                dbc.PopoverBody([
                    html.Div('Raw data'),
                    dbc.Button(
                        '.csv',
                        outline=True, color='secondary',
                        className='mr-1', id=f'csv_download_{suffix}',
                        style={}, size='sm'),
                    html.Div('Chart', style={'marginTop': 5}),
                    dbc.Button(
                        '.png',
                        outline=True, color='secondary',
                        className='mr-1', id=f'png_download_{suffix}',
                        style={}, size='sm'),
                    html.Div(
                        'Advanced',
                        style={'marginTop': 5, 'display': 'none'}),
                    dbc.Button(
                        'Downloads Area',
                        outline=True, color='secondary',
                        className='mr-1', id='btn-popover-line-download-land',
                        style={'display': 'none'}, size='sm'),
                    ]),
            ],
            id=f'modal_download_popover_menu_{suffix}',
            target=f'modal_download_popover_{suffix}',
            # style={'maxHeight': '300px', 'overflowY': 'auto'},
            trigger='legacy',
            placement='top',
            hide_arrow=False,
        ),
    ])
    return footer


def filters_and_controls_base_AccordionItem(country_dropdown_options):
    filters = dbc.AccordionItem(
        title='Filters and Controls',
        children=[
            html.Label('Gender:'),
            dcc.Checklist(
                id='gender-checkboxes',
                options=[
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Female', 'value': 'Female'},
                    {'label': 'Unknown', 'value': 'U'}
                    ],
                value=['Male', 'Female', 'U']
            ),
            html.Div(style={'margin-top': '20px'}),
            html.Label('Age:'),
            dcc.RangeSlider(
                id='age-slider',
                min=0,
                max=101,
                step=10,
                marks={i: str(i) for i in range(0, 91, 10)},
                value=[0, 101]
            ),
            html.Div(style={'margin-top': '20px'}),
            html.Label('Outcome:'),
            dcc.Checklist(
                id='outcome-checkboxes',
                options=[
                    {'label': 'Death', 'value': 'Death'},
                    {'label': 'Censored', 'value': 'Censored'},
                    {'label': 'Discharged', 'value': 'Discharged'}
                ],
                value=['Death', 'Censored', 'Discharged']
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
                            options=country_dropdown_options,
                            value=[
                                option['value']
                                for option in country_dropdown_options],
                            style={'overflowY': 'auto', 'maxHeight': '200px'}
                        )
                    ]),
                    id='country-fade',
                    is_in=False,
                    appear=False,
                )
            ]),
        ],
    )
    return filters


def define_menu(buttons, country_dropdown_options):
    initial_modal = dbc.Modal(
        id='modal',
        children=[dbc.ModalBody('Initial content')],  # Placeholder content
        is_open=False,
        size='xl'
    )
    menu = pd.DataFrame(data=buttons, columns=['Item', 'Label', 'Index'])
    menu_items = [
        filters_and_controls_base_AccordionItem(country_dropdown_options)]
    cont = 0
    for item in menu['Item'].unique():
        if (cont == 0):
            item_children = [initial_modal]
        else:
            item_children = []
        cont += 1
        for index, row in menu.loc[(menu['Item'] == item)].iterrows():
            item_children.append(dbc.Button(
                row['Label'],
                id={'type': 'open-modal', 'index': row['Index']},
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


############################################
############################################
# Figures
############################################
############################################


def fig_placeholder(
        df, dictionary=None, graph_id='table', graph_label='', graph_about=''):
    x = [1, 2, 3, 4, 5]
    y = [10, 14, 12, 15, 13]
    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode='markers', marker=dict(size=10, color='blue')))
    fig.update_layout(
        title='Sample Scatter Plot',
        xaxis_title='X Axis',
        yaxis_title='Y Axis')
    graph = dcc.Graph(
        id=graph_id,
        figure=fig)
    return graph, graph_label, graph_about


def fig_upset(
        df, dictionary=None,
        title='UpSet Plot',
        graph_id='upset-chart', graph_label='', graph_about=''):

    categories_reduced = ia.rename_variables(
        pd.Series(df.columns), dictionary, max_len=50).tolist()
    df = df.rename(columns=dict(zip(
        df.columns,
        ia.rename_variables(pd.Series(df.columns), dictionary).tolist())))
    categories = df.columns
    intersections = ia.compute_intersections(df)

    # Initialize subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,  # Space between the plots
        # subplot_titles=('Intersection Size', '')  # Titles for subplots
    )

    # Create bar chart traces for intersection sizes
    bar_traces = []
    for intersection, size in intersections.items():
        bar_traces.append(go.Bar(
            y=[size],
            x=[' & '.join(intersection)],
            orientation='v',
            customdata=['<br>'.join(intersection)],
            hovertemplate='%{customdata}<br><br>Intersection Size = %{y}',
            name='',
        ))

    # Add bar traces to the top subplot
    for trace in bar_traces:
        fig.add_trace(trace, row=1, col=1)

    # Create matrix scatter plot and lines
    for intersection, size in intersections.items():
        x_name = ' & '.join(intersection)
        y_coords = [
            -1 - categories.get_loc(cat)
            for cat in categories if cat in intersection]
        x_coords = [x_name] * len(y_coords)

        # Add a line connecting the points
        # Only add a line if there are at least two points
        if len(y_coords) > 1:
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hovertemplate='%{x}',
                    name=''),
                row=2, col=1)

        # Add scatter plot for each point in the intersection
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(size=10, color='black'),
                showlegend=False,
                customdata=['<br>'.join(intersection)]*len(y_coords),
                hovertemplate='%{customdata}',
                name=''),
            row=2, col=1)

    # for cat in categories:
    #     fig.add_trace(go.Scatter(
    #             x=[0, 1],
    #             y=[-1 - categories.get_loc(cat)]*2,
    #             hovertext=cat),
    #         row=2, col=1)

    # Update y-axis for the bar chart subplot
    fig.update_yaxes(title_text='Intersection Size', row=1, col=1)

    # Update y-axis for the matrix subplot to show category names
    # instead of numeric
    fig.update_yaxes(
        tickvals=[-1 - i for i in range(len(categories_reduced))],
        ticktext=categories_reduced,
        showgrid=False,
        row=2, col=1,
        labelalias=dict(zip(categories_reduced, categories))
    )

    # Hide x-axis line for the bar chart subplot
    fig.update_xaxes(showline=False, row=1, col=1)

    # Hide x-axis ticks and labels for the matrix subplot
    fig.update_xaxes(
        ticks='', showticklabels=False, showgrid=False, row=2, col=1)

    # Set the overall layout properties
    fig.update_layout(
        # title=title,
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        showlegend=False,
        height=480,  # You may need to adjust the height
        # hovermode='y',
    )

    # Return a Dash Graph component
    graph = dcc.Graph(
        id=graph_id,
        figure=fig)
    return graph, graph_label, graph_about


def fig_frequency_chart(
        df, dictionary=None,
        title='Frequency Chart', labels=['Condition', 'Proportion'],
        base_color_map=None,
        graph_id='freq-chart', graph_label='', graph_about=''):

    df['label'] = ia.rename_variables(
        df['Condition'], dictionary, max_len=40).tolist()
    df['Condition'] = ia.rename_variables(df['Condition'], dictionary)
    df = df.sort_values(by=['Proportion'], ascending=False)
    if (len(df) > 15):
        df = df.head(10)

    # Error Handling
    if not all(label in df.columns for label in labels):
        error_str = f'Dataframe must contain the following columns: {labels}'
        raise ValueError(error_str)

    # Calculate the proportion of 'Yes' for each condition and sort
    condition_proportions = df.groupby(
        labels[0])[labels[1]].mean().sort_values(ascending=True)

    sorted_conditions = condition_proportions.index.tolist()
    sorted_labels = df.set_index('Condition').loc[sorted_conditions, 'label']

    # Prepare Data Traces
    traces = []
    default_color = '#007E71'
    yes_color = (
        base_color_map.get('Yes', default_color)
        if base_color_map else default_color)
    no_color = (
        ia.hex_to_rgba(base_color_map.get('No', default_color), 0.5)
        if base_color_map else ia.hex_to_rgba(default_color, 0.5))

    for condition in sorted_conditions:
        yes_count = condition_proportions[condition]
        no_count = 1 - yes_count

        # Add 'Yes' bar
        traces.append(
            go.Bar(
                x=[yes_count],
                y=[condition],
                name='Yes',
                orientation='h',
                marker=dict(color=yes_color),
                customdata=[condition],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(condition == sorted_conditions[0]))
        )

        # Add 'No' bar
        traces.append(
            go.Bar(
                x=[no_count],
                y=[condition],
                name='No',
                orientation='h',
                marker=dict(color=no_color),
                customdata=[condition],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(condition == sorted_conditions[0]))
        )

    layout = go.Layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        barmode='stack',
        xaxis=dict(title=labels[1], range=[0, 1]),
        yaxis=dict(
            title=labels[0], automargin=True, tickmode='array',
            tickvals=sorted_conditions, ticktext=sorted_labels),
        bargap=0.1,  # Smaller gap between bars. Adjust this value as needed.
        legend=dict(x=1.05, y=1),
        margin=dict(l=100, r=100, t=100, b=50),
        height=350
    )

    fig = go.Figure(data=traces, layout=layout)

    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure=fig
    )
    return graph, graph_label, graph_about


def fig_table(
        df, dictionary,
        table_key='',
        graph_id='table-graph', graph_label='', graph_about=''):
    bf_columns = ['<b>' + x + '</b>' for x in df.columns]
    df.rename(columns=dict(zip(df.columns, bf_columns)), inplace=True)
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#bbbbbb',
            align='left'),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='#e9e9e9',
            align=['left'] + ['right']*(df.shape[1] - 1)))
    ])
    fig.update_layout(
        height=500,
        title=table_key,
        title_font=dict(size=12),
        title_y=0.08,
        title_x=0.95
    )
    graph = dcc.Graph(
        id=graph_id,
        figure=fig
    )
    return graph, graph_label, graph_about


def fig_dual_stack_pyramid(
        df, dictionary=None,
        title='Dual-Sided Stacked Pyramid Chart', base_color_map=None,
        graph_id='stacked-bar-chart', graph_label='', graph_about=''):

    df = df.loc[df['side'].isin(['Female', 'Male'])]
    # Error Handling
    required_columns = {'y_axis', 'side', 'stack_group', 'value'}
    if not required_columns.issubset(df.columns):
        error_str = 'Dataframe must contain the following columns: '
        error_str += f'{required_columns}'
        raise ValueError(error_str)
    if df.empty:
        raise ValueError('The DataFrame is empty.')
    if len(df['side'].unique()) != 2:  # TODO
        return dcc.Graph(
            id=graph_id,
            figure={}
        )
    left_side_label = df['side'].unique()[0]
    right_side_label = df['side'].unique()[1]

    # Dynamic Color Mapping
    if (base_color_map is not None) and (not isinstance(base_color_map, dict)):
        error_str = 'color_mapping must be a dictionary with stack_group'
        error_str += 'as keys and color codes as values.'
        raise ValueError(error_str)
    color_map = {}
    for stack_group, color in base_color_map.items():
        for side in df['side'].unique():
            if side == df['side'].unique()[0]:
                # Convert to RGBA with 50% opacity
                modified_color = ia.hex_to_rgba(color, 0.75)
            else:
                # Convert to RGBA with full opacity
                modified_color = ia.hex_to_rgba(color, 1)
            color_map[(side, stack_group)] = modified_color

    # Prepare Data Traces
    traces = []
    max_value = df['value'].abs().max()
    for side in df['side'].unique():
        for stack_group in df['stack_group'].unique():
            subset = df[(
                (df['side'] == side) &
                (df['stack_group'] == stack_group))]
            if subset.empty:
                continue
            # Get color from the color_map using both side and stack_group
            color = color_map.get((side, stack_group))
            x_val = (
                -subset['value'] if (side == df['side'].unique()[0])
                else subset['value'])
            traces.append(
                go.Bar(
                    y=subset['y_axis'],
                    x=x_val,
                    name=f'{side} {stack_group}',
                    orientation='h',
                    # Use the color from the color_map
                    marker=dict(color=color)
                )
            )

    # Sorting y-axis categories
    split_ranges = [
        (int(r.split('-')[0]), int(r.split('-')[1]))
        for r in df['y_axis'].unique()]
    sorted_ranges = sorted(split_ranges, key=lambda x: x[0])
    sorted_y_axis = [f'{start}-{end}' for start, end in sorted_ranges]

    max_value = max(
        df['value'].abs().max(),
        df.loc[(df['side'] != df['side'].unique()[0]), 'value'].abs().max())
    # Layout settings
    layout = go.Layout(
        title=title,
        barmode='relative',
        xaxis=dict(
            title='Count',
            range=[-max_value, max_value],
            automargin=True,
            tickvals=[-max_value, -max_value/2, 0, max_value/2, max_value],
            # Labels as positive numbers
            ticktext=[max_value, max_value/2, 0, max_value/2, max_value]
        ),
        yaxis=dict(
            title='Category',
            automargin=True,
            categoryorder='array',
            categoryarray=sorted_y_axis
        ),
        annotations=[
            dict(
                x=0.2,  # Position at 10% from the left edge of the graph
                y=1.1,  # Position just above the top of the graph
                xref='paper', yref='paper',
                text=left_side_label, showarrow=False,
                font=dict(family='Arial', size=14, color='black'),
                align='center'
            ),
            dict(
                # Position at 90% from the left edge of the graph
                # (i.e., near the right edge)
                x=0.8,
                y=1.1,  # Position just above the top of the graph
                xref='paper', yref='paper',
                text=right_side_label, showarrow=False,
                font=dict(family='Arial', size=14, color='black'),
                align='center'
            )
        ],
        shapes=[
            # Line at x=0 for reference
            dict(
                type='line',
                # Start point of the line
                # (y0=-1 to ensure it starts from the bottom)
                x0=0, y0=0,
                # End point of the line (y1=1 to ensure it goes to the top)
                x1=0, y1=1,
                # Reference to x axis and paper for y axis
                xref='x', yref='paper',
                line=dict(
                    color='Black',
                    width=2
                ),
            )
        ],
        legend=dict(x=1.05, y=1),
        margin=dict(l=100, r=100, t=100, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=default_height
    )
    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure={'data': traces, 'layout': layout}
    )
    return graph, graph_label, graph_about
