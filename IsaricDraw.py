from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import itertools
from plotly.subplots import make_subplots
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff


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
                        {'label': 'Unknown', 'value': 'U'}
                    ],
                    value=['Male', 'Female', 'U']
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
                        max=90,
                        step=10,
                        marks={i: str(i) for i in range(0, 91, 10)},
                        value=[0, 90]
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
                        {'label': 'Discharge', 'value': 'Discharge'}
                    ],
                    value=['Death', 'Censored', 'Discharge']
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
                    {'label': 'Discharge', 'value': 'Discharge'}
                ],
                value=['Death', 'Censored', 'Discharge']
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


def hex_to_rgb(hex_color):
    ''' Convert a hex color to an RGB tuple. '''
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def interpolate_colors(colors, n):
    ''' Interpolate among multiple hex colors.'''
    # Convert all hex colors to RGB
    rgbs = [hex_to_rgb(color) for color in colors]

    interpolated_colors = []
    # Number of transitions is one less than the number of colors
    transitions = len(colors) - 1

    # Calculate the number of steps for each transition
    steps_per_transition = n // transitions

    # Interpolate between each pair of colors
    for i in range(transitions):
        for step in range(steps_per_transition):
            interpolated_rgb = [
                int(rgbs[i][j] + (
                    float(step)/steps_per_transition)*(rgbs[i+1][j]-rgbs[i][j]))
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


def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    rgba_color = 'rgba(' + ', '.join(
        str(int(hex_color[i:i+hlen//3], 16))
        for i in range(0, hlen, hlen//3))
    rgba_color += f', {opacity})'
    return rgba_color


def compute_intersections(df):
    # Find all combinations of categories and their intersection sizes
    categories = df.columns
    intersections = {}
    for r in range(1, len(categories) + 1):
        for combo in itertools.combinations(categories, r):
            # Intersection is where all categories in the combo have a 1
            mask = df[list(combo)].all(axis=1)
            intersections[combo] = mask.sum()

    # Sort intersections by size in descending order
    sorted_intersections = OrderedDict(
        sorted(intersections.items(), key=lambda x: x[1], reverse=True))
    return sorted_intersections


############################################
############################################
# Figures
############################################
############################################


def fig_placeholder(df, graph_id, graph_label=None):
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
    return graph, graph_label


def fig_upset(
        df, title='UpSet Plot', graph_id='upset-chart', graph_label=None):
    categories = df.columns
    intersections = compute_intersections(df)

    # Initialize subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,  # Space between the plots
        subplot_titles=('Intersection Size', '')  # Titles for subplots
    )

    # Create bar chart traces for intersection sizes
    bar_traces = []
    for intersection, size in intersections.items():
        bar_traces.append(go.Bar(
            y=[size],
            x=[' & '.join(intersection)],
            orientation='v',
            name=' & '.join(intersection)
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

        # Add scatter plot for each point in the intersection
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(size=10, color='black'),
                showlegend=False),
            row=2, col=1)

        # Add a line connecting the points
        # Only add a line if there are at least two points
        if len(y_coords) > 1:
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False),
                row=2, col=1)

    # Update y-axis for the bar chart subplot
    fig.update_yaxes(title_text='Intersection Size', row=1, col=1)

    # Update y-axis for the matrix subplot to show category names
    # instead of numeric
    fig.update_yaxes(
        tickvals=[-1 - i for i in range(len(categories))],
        ticktext=categories,
        showgrid=False,
        row=2, col=1
    )

    # Hide x-axis line for the bar chart subplot
    fig.update_xaxes(showline=False, row=1, col=1)

    # Hide x-axis ticks and labels for the matrix subplot
    fig.update_xaxes(
        ticks='', showticklabels=False, showgrid=False, row=2, col=1)

    # Set the overall layout properties
    fig.update_layout(
        title=title,
        showlegend=False,
        height=450  # You may need to adjust the height
    )

    # Return a Dash Graph component
    graph = dcc.Graph(
        id=graph_id,
        figure=fig)
    return graph, graph_label


def fig_cumulative_bar_chart(
        df,
        title='Cumulative Bar by Timepoint', xlabel='x', ylabel='y',
        base_color_map=None,
        graph_id='cumulative-bar-chart', graph_label=None):
    # Pivot the DataFrame to get cumulative sums for each stack_group
    # at each timepoint
    # Ensure the 'timepoint' column is sorted or create a complete range if
    # necessary
    timepoints = sorted(df['timepoint'].unique())
    all_timepoints = range(int(min(timepoints)), int(max(timepoints)) + 1)

    # Create a complete DataFrame with all timepoints
    complete_df = pd.DataFrame(all_timepoints, columns=['timepoint'])

    # Merge the original dataframe with the complete timepoints dataframe to
    # fill gaps
    merged_df = pd.merge(complete_df, df, on='timepoint', how='left')

    # Pivot the merged DataFrame to get cumulative sums for each stack_group
    # at each timepoint
    pivot_df = merged_df.pivot_table(
        index='timepoint', columns='stack_group',
        values='value', aggfunc='sum')

    # Forward fill missing values and then calculate the cumulative sum
    pivot_df_ffill = pivot_df.fillna(method='ffill').cumsum()

    # Create traces for each stack_group with colors from the base_color_map
    traces = []
    for stack_group in pivot_df_ffill.columns:
        # Default to black if no color provided
        color = base_color_map.get(stack_group, '#000')
        traces.append(
            go.Bar(
                x=pivot_df_ffill.index,
                y=pivot_df_ffill[stack_group],
                name=stack_group,
                orientation='v',
                marker=dict(color=color)
            )
        )

    # Layout settings
    layout = go.Layout(
        title=title,
        barmode='stack',
        bargap=0,  # Set the gap between bars of the same category to 0
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        legend=dict(x=1.05, y=1),
        margin=dict(l=100, r=100, t=100, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=340
    )

    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure={'data': traces, 'layout': layout}
    )
    return graph, graph_label


def fig_dual_stack_pyramid(
        df,
        title='Dual-Sided Stacked Pyramid Chart', base_color_map=None,
        graph_id='stacked-bar-chart', graph_label=None):

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
                modified_color = hex_to_rgba(color, 0.75)
            else:
                # Convert to RGBA with full opacity
                modified_color = hex_to_rgba(color, 1)
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
    return graph, graph_label


def fig_sex_boxplot(
        df,
        title='Hospital Stay Length by Sex', ylabel='y', base_color_map=None,
        graph_id='sex-boxplot', graph_label=None):
    # Default color map if none provided
    if base_color_map is None:
        # Default colors for M and F
        base_color_map = {'Male': 'blue', 'Female': 'pink'}
    # Creating boxplots for each sex
    traces = [
        go.Box(
            y=df.loc[(df['sex'] == sex), 'length of hospital stay'],
            name=sex,
            marker_color=base_color_map.get(sex),
            boxpoints='outliers')  # Or 'all'
        for sex in ['Male', 'Female']]

    # Plot layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Sex'),
        yaxis=dict(title=ylabel),
        showlegend=True,
        height=340
    )

    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure={'data': traces, 'layout': layout}
    )
    return graph, graph_label


def fig_age_group_boxplot(
        df,
        title='Hospital Stay Length by Age Group', ylabel='y',
        base_color_map=None,
        graph_id='age-group-boxplot', graph_label=None):
    # Define age groups
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
    age_labels = [
        '0-10', '10-20', '20-30', '30-40', '40-50', '50-60',
        '60-70', '70-80', '80-90', '90+']
    df['age_group'] = pd.cut(
        df['age'], bins=age_bins, labels=age_labels, right=False)

    # Default color map if none provided
    if base_color_map is None:
        # Default to None for Plotly's default colors
        base_color_map = {group: None for group in age_labels}

    # Creating boxplots for each age group
    traces = [
        go.Box(
            y=df.loc[(df['age_group'] == group), 'length of hospital stay'],
            name=group,
            marker_color=base_color_map.get(group),
            boxpoints='outliers')  # Or 'all'
        for group in age_labels]

    # Plot layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Age Group'),
        yaxis=dict(title=ylabel),
        showlegend=True,
        height=340
    )

    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure={'data': traces, 'layout': layout}
    )
    return graph, graph_label


def fig_frequency_chart(
        df,
        title='Frequency Chart', labels=['Condition', 'Proportion'],
        base_color_map=None,
        graph_id='freq-chart', graph_label=None):

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

    # Prepare Data Traces
    traces = []
    default_color = '#007E71'
    yes_color = (
        base_color_map.get('Yes', default_color)
        if base_color_map else default_color)
    no_color = (
        hex_to_rgba(base_color_map.get('No', default_color), 0.5)
        if base_color_map else hex_to_rgba(default_color, 0.5))

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
                # Show legend only for the first
                showlegend=(condition == sorted_conditions[0]))
        )

    layout = go.Layout(
        title=title,
        barmode='stack',
        xaxis=dict(title=labels[1], range=[0, 1]),
        yaxis=dict(
            title=labels[0], automargin=True, tickmode='array',
            tickvals=sorted_conditions, ticktext=sorted_conditions),
        bargap=0.1,  # Smaller gap between bars. Adjust this value as needed.
        legend=dict(x=1.05, y=1),
        margin=dict(l=100, r=100, t=100, b=50),
        height=350
    )

    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure={'data': traces, 'layout': layout}
    )
    return graph, graph_label


def fig_forest_plot(
        df,
        title='Forest Plot',
        labels=['Study', 'OddsRatio', 'LowerCI', 'UpperCI'],
        graph_id='forest-plot', graph_label=None):
    # Error Handling
    if not set(labels).issubset(df.columns):
        error_str = f'Dataframe must contain the following columns: {labels}'
        raise ValueError(error_str)

    # Prepare Data Traces
    traces = []

    # Add the point estimates as scatter plot points
    traces.append(
        go.Scatter(
            x=df[labels[1]],
            y=df[labels[0]],
            mode='markers',
            name='Odds Ratio',
            marker=dict(color='blue', size=10))
    )

    # Add the confidence intervals as lines
    for index, row in df.iterrows():
        traces.append(
            go.Scatter(
                x=[row[labels[2]], row[labels[3]]],
                y=[row[labels[0]], row[labels[0]]],
                mode='lines',
                showlegend=False,
                line=dict(color='blue', width=2))
        )

    # Define layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Odds Ratio'),
        yaxis=dict(
            title='', automargin=True, tickmode='array',
            tickvals=df[labels[0]].tolist(), ticktext=df[labels[0]].tolist()),
        shapes=[
            dict(
                type='line', x0=1, y0=-0.5, x1=1, y1=len(df[labels[0]])-0.5,
                line=dict(color='red', width=2)
            )],  # Line of no effect
        margin=dict(l=100, r=100, t=100, b=50),
        height=600
    )

    # Return the dcc.Graph object with the created traces and layout
    graph = dcc.Graph(
        id=graph_id,
        figure={'data': traces, 'layout': layout}
    )
    return graph, graph_label


def fig_table(df, graph_id='table-graph', graph_label=None):
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#bbbbbb',
            align='left'),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='#e9e9e9',
            align='left'))
    ])
    fig.update_layout(height=500)
    graph = dcc.Graph(
        id=graph_id,
        figure=fig
    )
    return graph, graph_label


def fig_heatmap(df1, df2, title, graph_id='heatmap', graph_label=None):
    x_columns = df1.columns
    y_columns = df2.columns

    # Empty frequency matrix
    frequency_matrix = pd.DataFrame(index=x_columns, columns=y_columns, data=0)

    # Fill frequency matrix with count of co-occurrences
    for x_col in x_columns:
        for y_col in y_columns:
            frequency_matrix.loc[x_col, y_col] = (
                    (df1[x_col] == 1) &
                    (df2[y_col] == 1)
                ).sum()  # This is very slow!

    # Create linkage matrices for dendrograms
    linkage_rows = linkage(
        frequency_matrix.values, method='average', metric='euclidean')
    linkage_cols = linkage(
        frequency_matrix.values.T, method='average', metric='euclidean')

    # Create Dendrograms
    dendro_side = ff.create_dendrogram(
        frequency_matrix.values,
        orientation='right', labels=frequency_matrix.index.tolist(),
        linkagefun=lambda x: linkage_rows)
    dendro_top = ff.create_dendrogram(
        frequency_matrix.values.T,
        orientation='bottom', labels=frequency_matrix.columns.tolist(),
        linkagefun=lambda x: linkage_cols)

    # Remove labels from dendrograms
    for trace in dendro_top['data']:
        trace['showlegend'] = False
        trace['hoverinfo'] = 'none'
        if trace['text'] is not None:
            trace['text'] = [''] * len(trace['text'])

    for trace in dendro_side['data']:
        trace['showlegend'] = False
        trace['hoverinfo'] = 'none'
        if trace['text'] is not None:
            trace['text'] = [''] * len(trace['text'])

    # Reorder data according to the dendrogram leaves
    dendro_rows = dendro_side['layout']['yaxis']['ticktext']
    dendro_cols = dendro_top['layout']['xaxis']['ticktext']
    frequency_matrix = frequency_matrix.loc[dendro_rows, dendro_cols]

    # Create Heatmap
    heatmap = go.Heatmap(
        z=frequency_matrix.values,
        x=dendro_cols,
        y=dendro_rows,
        colorscale='jet',
        showscale=True,
        colorbar=dict(
            showticklabels=True,
            title='Frequency',
            x=0.08,
            xanchor='left',
            y=0.85,
            yanchor='middle',
            len=0.36,
            lenmode='fraction',
            thickness=10,
            thicknessmode='pixels',
        )
    )

    # Combine heatmap and dendrograms using subplots
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.4, 0.8], row_heights=[0.4, 0.8],
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'heatmap'}]],
        horizontal_spacing=0, vertical_spacing=0
    )

    # Add Dendrogram data
    for trace in dendro_top['data']:
        fig.add_trace(trace, row=1, col=2)

    for trace in dendro_side['data']:
        fig.add_trace(trace, row=2, col=1)

    # Add Heatmap data
    fig.add_trace(heatmap, row=2, col=2)

    # Update Layout
    fig.update_layout({
        'width': 800,
        'height': 800,
        'showlegend': True,
        'hovermode': 'closest',
        'title': title
    })

    # Hide axes on the dendrograms
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=False,
        showticklabels=False, ticks='', row=1, col=2)
    fig.update_yaxes(
        showgrid=False, zeroline=False, showline=False,
        showticklabels=False, ticks='', row=2, col=1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Update xaxis
    fig.update_xaxes(domain=[0.3, 1], row=2, col=2)
    fig.update_xaxes(domain=[0.3, 1], row=1, col=2)

    # Update yaxis
    fig.update_yaxes(domain=[0, 0.7], row=2, col=2)
    fig.update_yaxes(domain=[0, 0.7], row=2, col=1)

    # Show tick labels for y-axis on the right side
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(showticklabels=True, side='right', row=2, col=2)

    # Hide tick labels for top dendrogram
    fig.update_xaxes(showticklabels=False, row=1, col=2)

    # Legend update
    fig.update_layout(showlegend=True)

    graph = dcc.Graph(
        id=graph_id,
        figure=fig
    )
    return graph, graph_label


# def forest_plot(
#         df,
#         title='Forest Plot', xlabel='Odds Ratio', ylabel='Variable',
#         graph_id='forest-plot', graph_label=None):
#     df = df.sort_values(by=xlabel, ascending=False)
#     # df = df.loc[dataframe['P-Value']<=0.2]
#
#     # Create a trace for the effect sizes
#     trace_effect_sizes = go.Scatter(
#         x=df[xlabel],
#         y=df[ylabel],
#         mode='markers',
#         marker=dict(color='black', size=10),
#         name='Effect Size',
#         error_x=dict(
#             type='data',
#             symmetric=False,
#             array=df['95% CI Upper'] - df['Odds Ratio'],
#             arrayminus=df['Odds Ratio'] - df['95% CI Lower']
#         )
#     )
#
#     # Combine the trace in a list
#     data = [trace_effect_sizes]
#
#     # Define the layout
#     layout = go.Layout(
#         title=title,
#         xaxis=dict(title=xlabel, zeroline=False),
#         # Reverse axis for readability
#         yaxis=dict(title=ylabel, automargin=True, autorange='reversed'),
#         showlegend=False,
#         hovermode='closest',
#         margin=dict(l=100, r=100, t=100, b=50),
#         paper_bgcolor='white',
#         plot_bgcolor='white',
#         height=default_height+150,
#         shapes=[{
#             'type': 'line',
#             'x0': 1,
#             'y0': -0.5,
#             'x1': 1,
#             'y1': df['Variable'].nunique()-0.5,
#             'line': {
#                 'color': 'grey',
#                 'width': 2,
#                 'dash': 'dot',
#             }
#         }]
#     )
#
#     # Return the dcc.Graph object with the created trace and layout
#     graph = dcc.Graph(
#         id=graph_id,
#         figure={'data': data, 'layout': layout}
#     )
#     return graph, graph_label
