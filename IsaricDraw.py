import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


default_height = 430


############################################
############################################
# Figures
############################################
############################################


def fig_placeholder(
        df,
        title='Sample scatter plot',
        button_item='', button_label='',
        graph_id='placeholder', graph_label='', graph_about=''):
    x = [1, 2, 3, 4, 5]
    # y = [10, 14, 12, 15, 13]
    y = np.random.uniform(low=10, high=15, size=5)
    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode='markers', marker=dict(size=10, color='blue')))
    fig.update_layout(
        title=title,
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        yaxis_range=[10, 15])
    return fig, graph_id, graph_label, graph_about


def fig_upset(
        data,
        title='Upset Plot',
        height=480,
        graph_id='upset-plot', graph_label='', graph_about=''):

    counts = data[0].copy()
    intersections = data[1].copy()

    hlabel='label'
    slabel='short_label'

    hoverlabels = counts[hlabel].tolist()
    labels = counts[slabel].tolist()

    column_widths = [intersections.shape[0], counts.shape[0]]

    # Initialize subplots
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True, shared_yaxes=True,
        column_widths=column_widths,
        vertical_spacing=0.02,  # Space between the plots
        horizontal_spacing=0.01,
    )

    # Create bar chart traces for intersection sizes
    bar_traces = []
    for ii in intersections.index:
        color = rgb_to_rgba(px.colors.sequential.Purples_r[ii % 5], 1)
        hoverlabel = '<br>'.join(intersections.loc[ii, hlabel])
        n = intersections.loc[ii, 'count']
        customdata = f'Intersection of<br>{hoverlabel}<br><br>Count = {n}'
        bar_traces.append(go.Bar(
            y=[n],
            x=[ii],
            orientation='v',
            name='',
            customdata=[customdata],
            hovertemplate='%{customdata}',
            width=0.9,
            offset=-0.45,
            marker=dict(color=color),
            showlegend=False,
        ))

    # Add bar traces to the top subplot
    for trace in bar_traces:
        fig.add_trace(trace, row=1, col=1)

    bar_traces = []
    for ii in counts.index:
        hoverlabel = counts.loc[ii, hlabel]
        color = rgb_to_rgba(px.colors.sequential.Oranges_r[ii % 5], 1)
        n = counts.loc[ii, 'count']
        bar_traces.append(go.Bar(
            y=[-1 - ii],
            x=[n],
            orientation='h',
            name='',
            customdata=[f'{hoverlabel}<br><br>Count = {n}'],
            hovertemplate='%{customdata}',
            width=0.9,
            offset=-0.45,
            marker=dict(color=color),
            showlegend=False,
        ))

    # Add bar traces to the top subplot
    for trace in bar_traces:
        fig.add_trace(trace, row=2, col=2)

    # Create matrix scatter plot and lines
    for ii in intersections.index:
        intersection = intersections.loc[ii, hlabel]
        y_coords = [
            -1 - x
            for x in range(len(hoverlabels)) if hoverlabels[x] in intersection]
        x_coords = [ii]*len(y_coords)

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

    # Update y-axis for the bar chart subplot
    fig.update_yaxes(title_text='Intersection Size', row=1, col=1)

    # Update x-axis for the bar chart subplot
    fig.update_xaxes(title_text='Set Size', side='top', row=2, col=2)

    # Update y-axis for the matrix subplot to show category names
    # instead of numeric
    fig.update_yaxes(
        tickvals=[-1 - i for i in range(len(labels))],
        ticktext=labels,
        showgrid=False,
        row=2, col=1,
        labelalias=dict(zip(labels, hoverlabels))
    )

    # Hide x-axis line for the intersection size subplot
    fig.update_xaxes(showline=False, tickformat=',d', row=1, col=1)

    # Hide x-axis ticks and labels for the matrix subplot
    fig.update_xaxes(
        ticks='', showticklabels=False, showgrid=False, zeroline=False,
        row=2, col=1)

    # Hide y-axis ticks and labels for the set size subplot
    fig.update_yaxes(
        ticks='', showticklabels=False, showgrid=False, zeroline=False,
        row=2, col=2)

    # Set the overall layout properties
    fig.update_layout(
        # title=title,
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        # showlegend=False,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        height=height,  # You may need to adjust the height
        minreducedwidth=500,
    )
    return fig, graph_id, graph_label, graph_about


def fig_frequency_chart(
        df,
        title='Frequency Chart', 
        base_color_map=None,
        graph_id='freq-chart', graph_label='', graph_about=''):

    column_names=['short_label', 'proportion','label']

    # Error Handling
    if not all(col in df.columns for col in column_names):
        error_str = 'Dataframe must contain the following columns: '
        error_str += f'{column_names}'
        raise ValueError(error_str)

    # Prepare Data Traces
    traces = []
    default_color = '#007E71'
    yes_color = (
        base_color_map.get('Yes', default_color)
        if base_color_map else default_color)
    no_color = (
        hex_to_rgba(base_color_map.get('No', default_color), 0.5)
        if base_color_map else hex_to_rgba(default_color, 0.5))

    for ii in reversed(range(df.shape[0])):
        variable = df.loc[ii, column_names[0]]
        yes_count = df.loc[ii, column_names[1]]
        hlabel=df.loc[ii,column_names[2]]
        no_count = 1 - yes_count

        # Add 'Yes' bar
        traces.append(
            go.Bar(
                x=[yes_count],
                y=[variable],
                name='Yes',
                orientation='h',
                width=0.9,
                offset=-0.45,
                marker=dict(color=yes_color),
                customdata=[hlabel],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(ii == 0))
        )

        # Add 'No' bar
        traces.append(
            go.Bar(
                x=[no_count],
                y=[variable],
                name='No',
                orientation='h',
                width=0.9,
                offset=-0.45,
                marker=dict(color=no_color),
                customdata=[hlabel],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(ii == 0))
        )

    layout = go.Layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        barmode='stack',
        xaxis=dict(title=column_names[1].capitalize(), range=[0, 1]),
        yaxis=dict(
            title=column_names[0].capitalize(), automargin=True,
            tickmode='array', tickvals=df[column_names[0]],
            ticktext=df[column_names[0]]),
        bargap=0.1,  # Smaller gap between bars. Adjust this value as needed.
        # legend=dict(x=1.05, y=1),
        legend={
            'orientation': 'h',
            'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        margin=dict(l=100, r=100, t=100, b=50),
        height=350,
        minreducedwidth=500,
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig, graph_id, graph_label, graph_about


def fig_table(
        df, table_key='',
        graph_id='table-graph', graph_label='', graph_about=''):
    bf_columns = ['<b>' + x + '</b>' for x in df.columns]
    df.rename(columns=dict(zip(df.columns, bf_columns)), inplace=True)
    n = df.shape[1]
    firstwidth = 0.3
    columnwidth = [firstwidth] + [(1 - firstwidth)/(n - 1)]*(n - 1)
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#bbbbbb',
            align='left'),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='#e9e9e9',
            align=['left'] + ['right']*(n - 1)),
        columnwidth=columnwidth),
    ])
    fig.update_layout(
        height=500,
        title=table_key,
        title_font=dict(size=12),
        title_y=0.08,
        title_x=0.95,
        # minreducedwidth=500,
    )
    # metadata = {
    #     'graph_id': graph_id,
    #     'graph_label': graph_label,
    #     'graph_about': graph_about,
    #     'insight_panel': insight_panel_name}
    return fig, graph_id, graph_label, graph_about


def fig_dual_stack_pyramid(
        df, yaxis_label=None,
        title='Dual-Sided Stacked Pyramid Chart', base_color_map=None,
        graph_id='stacked-pyramid-chart', graph_label='', graph_about=''):

    # df = df.loc[df['side'].isin(['Female', 'Male'])]
    # Error Handling
    required_columns = {'y_axis', 'side', 'stack_group', 'value'}
    if not required_columns.issubset(df.columns):
        error_str = 'Dataframe must contain the following columns: '
        error_str += f'{required_columns}'
        raise ValueError(error_str)
    if df.empty:
        raise ValueError('The DataFrame is empty.')
    if len(df['side'].unique()) != 2:  # TODO
        fig = {}
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

    if yaxis_label is None:
        yaxis_label = 'Category'
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
            title=yaxis_label,
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
    fig = {'data': traces, 'layout': layout}
    return fig, graph_id, graph_label, graph_about


############################################
############################################
# Formatting: colours
############################################
############################################


def hex_to_rgb(hex_color):
    ''' Convert a hex color to an RGB tuple. '''
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb_color


def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    rgba_color = 'rgba(' + ', '.join(
        str(int(hex_color[i:i+hlen//3], 16))
        for i in range(0, hlen, hlen//3))
    rgba_color += f', {opacity})'
    return rgba_color


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add in range [0,1]
    :return: RGBA Value
    """
    rgba_color = f"rgba{rgb_value[3:-1]}, {alpha})"
    return rgba_color
