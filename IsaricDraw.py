import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

default_height = 430


def get_graph_id(graph_id_suffix, suffix, frame=1):
    fig_name = sys._getframe(frame).f_code.co_name
    if len(graph_id_suffix) != 0:
        graph_id_suffix = '_' + graph_id_suffix
    graph_id = suffix + '/' + fig_name + graph_id_suffix
    return graph_id


def save_inputs_to_file(local_args):
    fig_name = sys._getframe(1).f_code.co_name
    data = local_args.pop('df')
    # Convert to list (if not already)
    data = data if isinstance(data, tuple) else (data,)
    path = local_args['filepath']
    graph_id = get_graph_id(
        local_args['graph_id'], local_args['suffix'], frame=2)
    fig_data = [
        graph_id + '_data___' + str(ii) + '.csv'
        for ii in range(len(data))]
    _ = local_args.pop('filepath')
    _ = local_args.pop('save_inputs')
    metadata = {
        'fig_id': graph_id,
        'fig_name': fig_name,
        'fig_arguments': local_args,
        'fig_data': fig_data,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.path.join(path, graph_id + '_metadata.txt'), 'w') as file:
        file.write(repr(metadata))
    for ii in range(len(data)):
        data[ii].to_csv(
            os.path.join(path, graph_id + '_data___' + str(ii) + '.csv'),
            index=False)
    return data, metadata


############################################
############################################
# Figures
############################################
############################################


def fig_placeholder(
        df,
        title='Sample scatter plot',
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

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

    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_pie(
        df,
        title='Pie chart',
        item='',
        value='',
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    fig = px.pie(df, values=value, names=item, title=title)
    fig.update_layout(
        title=title,
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        yaxis_range=[10, 15])

    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_timelines(
        df, title='Timeline',
        label_col='', group_col='',
        start_date='start_date', end_date='end_date',
        size_col=None, min_width=2, max_width=10,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    df = df.copy()
    df[start_date] = pd.to_datetime(df[start_date])
    df[end_date] = pd.to_datetime(df[end_date])
    max_end = df[end_date].max()

    # Assign colors by group_col
    unique_groups = df[group_col].unique()
    color_map = {
        group:
            px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, group in enumerate(unique_groups)
    }

    # Assign line widths if size_col is used
    if size_col and df[size_col].notnull().any():
        values = df[size_col].fillna(0).astype(float)
        min_val, max_val = values.min(), values.max()
        if min_val == max_val:
            widths = {
                row[label_col]: (min_width + max_width) / 2
                for _, row in df.iterrows()}
        else:
            widths = {
                row[label_col]: min_width + (val - min_val) / (
                    max_val - min_val) * (max_width - min_width)
                for row, val in zip(df.to_dict(orient='records'), values)
            }
    else:
        widths = {row[label_col]: 3 for _, row in df.iterrows()}

    fig = go.Figure()

    for _, row in df.iterrows():
        y = row[label_col]
        x_start = row[start_date]
        x_end = row[end_date] if pd.notnull(row[end_date]) else max_end
        ongoing = pd.isnull(row[end_date])
        color = color_map[row[group_col]]
        width = widths[y]

        symbol = ['circle', 'arrow-right'] if ongoing else ['circle', 'circle']

        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[y, y],
            mode='lines+markers',
            line=dict(color=color, width=width),
            marker=dict(
                size=[14, 18],
                symbol=symbol,
                color=color,
                line=dict(width=1, color='black')
            ),
            name=row[group_col],
            legendgroup=row[group_col],
            showlegend=(row[group_col] not in [t.name for t in fig.data])
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis=dict(title=label_col, tickfont=dict(size=10)),
        margin=dict(l=250, r=20, t=40, b=40),
        height=300 + 40 * len(df)
    )
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_sunburst(
        df,
        title='Sunburst Chart',
        path=['level0', 'level1'], values='value',
        base_color_map=None,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    fig = px.sunburst(
        df, path=path, values=values,
    )
    fig.update_layout(title=title, title_x=0.5)
    fig.update_traces(
        sort=False,
        selector=dict(type='sunburst'),
        insidetextorientation='radial',
        customdata=path,
        hovertemplate='%{label}<br>N=%{value}<extra></extra>',
    )
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_cumulative_bar_chart(
        df,
        title='Cumulative Bar by Timepoint', xlabel='x', ylabel='y',
        base_color_map=None, barmode='stack',
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    # Generate dynamic colors if base_color_map is not provided
    if base_color_map is None:
        unique_groups = df.columns
        color_palette = px.colors.qualitative.Plotly
        base_color_map = {
            group: color_palette[i % len(color_palette)]
            for i, group in enumerate(unique_groups)}

    # Create traces for each stack_group with colors from the base_color_map
    traces = []
    for stack_group in df.columns:
        # Assign color from base_color_map
        color = base_color_map.get(stack_group, '#000')
        traces.append(
            go.Bar(
                x=df.index,
                y=df[stack_group],
                name=stack_group,
                orientation='v',
                marker=dict(color=color)
            )
        )

    # Layout settings with customized x-axis tick format
    if barmode == 'group':
        bargap = 0.1
    else:
        bargap = 0
    layout = go.Layout(
        title=title,
        barmode=barmode,
        bargap=bargap,
        xaxis=dict(
            title=xlabel,
            tickformat='%m-%Y',  # Display x-axis in MM-YYYY format
            tickvals=df.index,  # Optional: only specific dates if needed
        ),
        yaxis=dict(title=ylabel),
        legend=dict(x=1.05, y=1),
        margin=dict(l=100, r=100, t=100, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=340
    )

    fig = {'data': traces, 'layout': layout}
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_stacked_bar_chart(
        df,
        title='Bar Chart by Timepoint', xlabel='x', ylabel='y',
        base_color_map=None, barmode='stack', xaxis_tickformat='%m-%Y',
        suffix='', filepath='', save_inputs=False, height=340,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    df = df.set_index('index')
    # Generate dynamic colors if base_color_map is not provided
    if base_color_map is None:
        unique_groups = df.columns
        color_palette = px.colors.qualitative.Plotly
        base_color_map = {
            group: color_palette[i % len(color_palette)]
            for i, group in enumerate(unique_groups)}

    # Create traces for each stack_group with colors from the base_color_map
    traces = []
    for stack_group in df.columns:
        # Assign color from base_color_map
        color = base_color_map.get(stack_group, '#000')
        traces.append(
            go.Bar(
                x=df.index,
                y=df[stack_group],
                name=stack_group,
                orientation='v',
                marker=dict(color=color)
            )
        )

    # Layout settings with customized x-axis tick format
    if barmode == 'group':
        bargap = 0.1
    else:
        bargap = 0
    layout = go.Layout(
        title=title,
        barmode=barmode,
        bargap=bargap,
        xaxis=dict(
            title=xlabel,
            tickformat=xaxis_tickformat,  # Display x-axis in MM-YYYY format
            tickvals=df.index,  # Optional: only specific dates if needed
        ),
        yaxis=dict(title=ylabel),
        legend=dict(x=1.05, y=1),
        margin=dict(l=100, r=100, t=100, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=height,
    )
    fig = {'data': traces, 'layout': layout}
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


# def fig_bar_chart(
#         df,
#         title='Bar Chart by Timepoint', xlabel='x', ylabel='y',
#         base_color_map=None,
#         suffix='', filepath='', save_inputs=False,
#         graph_id='', graph_label='', graph_about=''):
#
#     if save_inputs:
#         inputs = save_inputs_to_file(locals())
#
#     df = df.set_index('index')
#     # Generate dynamic colors if base_color_map is not provided
#     if base_color_map is None:
#         unique_groups = df.columns
#         color_palette = px.colors.qualitative.Plotly
#         base_color_map = {
#             group: color_palette[i % len(color_palette)]
#             for i, group in enumerate(unique_groups)}
#
#     # Create traces for each stack_group with colors from the base_color_map
#     traces = []
#     for stack_group in df.columns:
#         # Assign color from base_color_map
#         color = base_color_map.get(stack_group, '#000')
#         traces.append(
#             go.Bar(
#                 x=df.index,
#                 y=df[stack_group],
#                 name=stack_group,
#                 orientation='v',
#                 marker=dict(color=color)
#             )
#         )
#
#     # Layout settings with customized x-axis tick format
#     layout = go.Layout(
#         title=title,
#         barmode='stack',
#         bargap=0,
#         xaxis=dict(
#             title=xlabel,
#             tickformat='%m-%Y',  # Display x-axis in MM-YYYY format
#             tickvals=df.index,  # Optional: only specific dates if needed
#         ),
#         yaxis=dict(title=ylabel),
#         legend=dict(x=1.05, y=1),
#         margin=dict(l=100, r=100, t=100, b=50),
#         paper_bgcolor='white',
#         plot_bgcolor='white',
#         height=340
#     )
#     fig = {'data': traces, 'layout': layout}
#     graph_id = get_graph_id(graph_id, suffix)
#     return fig, graph_id, graph_label, graph_about


def fig_upset(
        df,
        title='Upset Plot', height=480,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    counts = df[0].copy()
    intersections = df[1].copy()

    hlabel = 'label'
    slabel = 'short_label'

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
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_count_chart(
        df,
        title='Count Chart', base_color_map=None, height=350,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    column_names = ['label', 'count', 'short_label']

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

    for ii in reversed(range(df.shape[0])):
        hoverlabel = df.loc[ii, column_names[0]]
        yes_count = df.loc[ii, column_names[1]]
        label = df.loc[ii, column_names[2]]
        # Add 'Yes' bar
        traces.append(
            go.Bar(
                x=[yes_count],
                y=[label],
                name='Yes',
                orientation='h',
                width=0.9,
                offset=-0.45,
                marker=dict(color=yes_color),
                customdata=[hoverlabel],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(ii == 0))
        )

    xlim = [0, df[column_names[1]].max()]
    layout = go.Layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        barmode='stack',
        xaxis=dict(title=column_names[1].capitalize(), range=xlim),
        yaxis=dict(
            title='Variable', automargin=True,
            tickmode='array', tickvals=df[column_names[2]],
            ticktext=df[column_names[2]]),
        bargap=0.1,  # Smaller gap between bars. Adjust this value as needed.
        # legend=dict(x=1.05, y=1),
        legend={
            'orientation': 'h',
            'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        margin=dict(l=100, r=100, t=100, b=50),
        height=height,
        minreducedwidth=500,
    )

    fig = go.Figure(data=traces, layout=layout)
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_frequency_chart(
        df,
        title='Frequency Chart', base_color_map=None, height=350,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    column_names = ['label', 'proportion', 'short_label']

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
        hoverlabel = df.loc[ii, column_names[0]]
        yes_count = df.loc[ii, column_names[1]]
        label = df.loc[ii, column_names[2]]
        no_count = 1 - yes_count

        # Add 'Yes' bar
        traces.append(
            go.Bar(
                x=[yes_count],
                y=[label],
                name='Yes',
                orientation='h',
                width=0.9,
                offset=-0.45,
                marker=dict(color=yes_color),
                customdata=[hoverlabel],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(ii == 0))
        )

        # Add 'No' bar
        traces.append(
            go.Bar(
                x=[no_count],
                y=[label],
                name='No',
                orientation='h',
                width=0.9,
                offset=-0.45,
                marker=dict(color=no_color),
                customdata=[hoverlabel],
                hovertemplate='%{customdata}: %{x:.2f}',
                # Show legend only for the first
                showlegend=(ii == 0))
        )

    layout = go.Layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        barmode='stack',
        xaxis=dict(title=column_names[1].capitalize(), range=[0, 1]),
        yaxis=dict(
            title='Variable', automargin=True,
            tickmode='array', tickvals=df[column_names[2]],
            ticktext=df[column_names[2]]),
        bargap=0.1,  # Smaller gap between bars. Adjust this value as needed.
        # legend=dict(x=1.05, y=1),
        legend={
            'orientation': 'h',
            'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        margin=dict(l=100, r=100, t=100, b=50),
        height=height,
        minreducedwidth=500,
    )

    fig = go.Figure(data=traces, layout=layout)
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_table(
        df,
        table_key='', table_format_dict=None,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    bf_columns = ['<b>' + x + '</b>' for x in df.columns]
    df.rename(columns=dict(zip(df.columns, bf_columns)), inplace=True)
    df = df.fillna('')
    n = df.shape[1]
    firstwidth = 0.3

    default_cells_format_dict = {'align': ['left'] + ['right']*(n - 1)}
    default_header_format_dict = {'fill_color': '#bbbbbb', 'align': 'left'}

    if isinstance(table_format_dict, dict) is False:
        table_format_dict = {}

    cells_format_dict = (
        table_format_dict['cells'] if ('cells' in table_format_dict.keys())
        else {})
    if isinstance(cells_format_dict, dict) is False:
        cells_format_dict = {}
    cells_format_dict = {**default_cells_format_dict, **cells_format_dict}

    header_format_dict = (
        table_format_dict['header'] if ('header' in table_format_dict.keys())
        else {})
    if isinstance(header_format_dict, dict) is False:
        header_format_dict = {}
    header_format_dict = {**default_header_format_dict, **header_format_dict}

    if 'columnwidth' in table_format_dict.keys():
        columnwidth = table_format_dict['columnwidth']
    else:
        if n < 2:
            columnwidth = [1]
        else:
            columnwidth = [firstwidth] + [(1 - firstwidth)/(n - 1)]*(n - 1)

    fig = go.Figure(data=[go.Table(
        header={'values': list(df.columns), **header_format_dict},
        cells={'values': [df[col] for col in df.columns], **cells_format_dict},
        columnwidth=columnwidth,
        ),
    ])
    fig.update_layout(
        height=500,
        title=table_key,
        title_font=dict(size=12),
        title_y=0.08,
        title_x=0.95,
        # minreducedwidth=500,
    )
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_dual_stack_pyramid(
        df, yaxis_label=None,
        title='Dual-Sided Stacked Pyramid Chart', base_color_map=None,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    # df = df.loc[df['side'].isin(['Female', 'Male'])]
    # Error Handling
    required_columns = {'y_axis', 'side', 'stack_group', 'value'}
    if not required_columns.issubset(df.columns):
        error_str = 'Dataframe must contain the following columns: '
        error_str += f'{required_columns}'
        raise ValueError(error_str)
    if df.empty:
        raise ValueError('The DataFrame is empty.')
    # if len(df['side'].unique()) != 2:  # TODO
    #     fig = {}
    left_side_label = ''
    right_side_label = ''
    if (df['left_side'] == 1).any():
        left_side_label = df.loc[(df['left_side'] == 1), 'side'].values[0]
    if (df['left_side'] == 0).any():
        right_side_label = df.loc[(df['left_side'] == 0), 'side'].values[0]

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
            # x_val = (
            #     -subset['value'] if (side == df['side'].unique()[0])
            #     else subset['value'])
            x_val = (
                -subset['value']
                if subset['left_side'].any() else subset['value'])
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

    # # Sorting y-axis categories
    # split_ranges = [
    #     (int(r.split('-')[0]), int(r.split('-')[1]))
    #     for r in df['y_axis'].unique()]
    # sorted_ranges = sorted(split_ranges, key=lambda x: x[0])
    # sorted_y_axis = [f'{start}-{end}' for start, end in sorted_ranges]

    # max_value = max(
    #     df['value'].abs().max(),
    #     df.loc[(df['side'] != df['side'].unique()[0]), 'value'].abs().max())
    max_value = df.groupby(
        ['side', 'y_axis'], observed=True).sum()['value'].max()
    max_value += (max_value % 2)

    if yaxis_label is None:
        yaxis_label = 'Category'
    # Layout settings
    tickvals = [
        -int(max_value), -int(max_value/2), 0,
        int(max_value/2), int(max_value)]
    layout = go.Layout(
        title=title,
        barmode='relative',
        xaxis=dict(
            title='Count',
            range=[-max_value, max_value],
            automargin=True,
            tickvals=tickvals,
            # Labels as positive numbers
            ticktext=[str(abs(x)) for x in tickvals]
        ),
        yaxis=dict(
            title=yaxis_label,
            automargin=True,
            categoryorder='array',
            categoryarray=df['y_axis'],
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
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_flowchart(
        df,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    arrows = []
    arrow_to = df['arrow_to'].apply(lambda x: x.replace(' ', ''))
    arrow_to = arrow_to.loc[arrow_to != '']
    ind_start = arrow_to.index.repeat(
        arrow_to.apply(lambda x: len(x.split(',')))).tolist()
    ind_end = [int(x) for x in ','.join(arrow_to).split(',')]
    for ii in range(len(ind_start)):
        arrow_start_x = df.loc[ind_start[ii], 'x']
        arrow_start_y = df.loc[ind_start[ii], 'y']
        arrow_end_x = df.loc[ind_end[ii], 'x']
        arrow_end_y = df.loc[ind_end[ii], 'y']
        new_arrows = pd.DataFrame(columns=['x', 'y', 'ax', 'ay', 'arrowhead'])
        new_arrows['ax'] = [arrow_start_x, (arrow_end_x + arrow_start_x) / 2]
        new_arrows['ay'] = [arrow_start_y, (arrow_end_y + arrow_start_y) / 2]
        new_arrows['x'] = [(arrow_end_x + arrow_start_x) / 2, arrow_end_x]
        new_arrows['y'] = [(arrow_end_y + arrow_start_y) / 2, arrow_end_y]
        new_arrows['arrowhead'] = [1, 0]
        arrows = arrows + [new_arrows]
    arrow_data = pd.concat(arrows, axis=0).reset_index(drop=True)
    arrow_metadata = {
        'showarrow': True, 'arrowwidth': 1.5,
        'arrowcolor': 'rgba(100, 100, 100, 0.5)',
        'axref': 'x', 'ayref': 'y', 'xref': 'x', 'yref': 'y', 'text': ''}
    arrows = [
        {**arrow, **arrow_metadata} for arrow in arrow_data.to_dict('records')]

    df.drop(columns='arrow_to', inplace=True)

    annotation_metadata = {
        'showarrow': False,
        'xanchor': 'center', 'yanchor': 'middle',
        'bgcolor': 'rgba(150, 150, 150, 1)',
        'bordercolor': 'rgba(100, 100, 100, 0.5)',
        'borderwidth': 1, 'borderpad': 5}
    annotations = [
        {**annotation, **annotation_metadata}
        for annotation in df.to_dict('records')]

    layout = go.Layout(
        annotations=arrows + annotations,
        xaxis={'visible': False, 'showgrid': False, 'range': [0, 1]},
        yaxis={'visible': False, 'showgrid': False, 'range': [0, 1]},
        plot_bgcolor='rgba(0, 0, 0, 0)')
    fig = go.Figure(layout=layout)

    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_forest_plot(
        df,
        title='Forest Plot', reorder=True,
        labels=['Variable', 'OddsRatio', 'LowerCI', 'UpperCI'],
        xlabel='Odds Ratio (95% CI)',
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    # Ordering Values -> Descending Order
    if reorder:
        df = df.sort_values(by=labels[1], ascending=True)
    else:
        df = df.loc[::-1]

    # Error Handling
    if not set(labels).issubset(df.columns):
        print(df.columns)
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
                line=dict(color='blue', width=2)
            )
        )

    # Define layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Odds Ratio (95% CI)'),
        yaxis=dict(
            title='', automargin=True, tickmode='array',
            tickvals=df[labels[0]].tolist(), ticktext=df[labels[0]].tolist()),
        shapes=[
            dict(
                type='line', x0=1, y0=-0.5, x1=1, y1=len(df[labels[0]])-0.5,
                line=dict(color='red', width=2)
            )],  # Line of no effect
        margin=dict(l=100, r=100, t=100, b=50),
        height=600,
        showlegend=False
    )
    fig = {'data': traces, 'layout': layout}
    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_text(
        df,
        title='Forest Plot',
        labels=['Variable', 'OddsRatio', 'LowerCI', 'UpperCI'],
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    fig = go.Figure()

    text = '<br>'.join(df['paragraphs'].values)

    fig.add_annotation(
        x=0, y=0, text=text, showarrow=False)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, range=[-1, 1]),
        yaxis=dict(visible=False, range=[-1, 1])
    )

    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_kaplan_meier(
        df,
        p_value=None, title=None, xlabel='Time (days)', xlim=None, height=800,
        suffix='', filepath='', save_inputs=False,
        graph_id='', graph_label='', graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    df_km = df[0].copy()
    risk_table = df[1].copy()

    unique_groups = risk_table['Group'].tolist()
    colors = [
        f"hsl({i * (360 / len(unique_groups))}, 70%, 50%)"
        for i in range(len(unique_groups))]

    # Create the figure with two rows: one for the plot and one for risk table
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.1,
                        specs=[[{"type": "xy"}], [{"type": "table"}]],
                        subplot_titles=[title, "Risk Table"])

    for group, color in zip(unique_groups, colors):
        ci_lower_column = [
            col for col in df_km.columns
            if col.startswith(group + '_lower')][0]
        ci_upper_column = [
            col for col in df_km.columns
            if col.startswith(group + '_upper')][0]
        ci = df_km.set_index('timeline')
        ci = ci[[col for col in df_km.columns if col.startswith(group + '_')]]
        ci = ci.dropna()
        ci_lower = ci[ci_lower_column]
        ci_upper = ci[ci_upper_column]

        # Add confidence interval as shaded area
        fig.add_trace(go.Scatter(
            x=ci_upper.index.tolist() + ci_lower.index[::-1].tolist(),
            y=ci_upper.tolist() + ci_lower[::-1].tolist(),
            fill='toself',
            fillcolor=color.replace("hsl", "hsla").replace(")", ",0.2)"),
            line=dict(color='rgba(255,255,255,0)', shape='hv'),
            name=f"CI {group}",
            showlegend=False,
            hoverinfo='text',
            text=[f"CI {group}" for _ in range(len(ci_upper) + len(ci_lower))]
        ), row=1, col=1)

    for group, color in zip(unique_groups, colors):
        survival = df_km.set_index('timeline')[group]

        # Add survival curve
        fig.add_trace(go.Scatter(
            x=survival.index,
            y=survival.values,
            mode='lines',
            name=str(group),
            line=dict(color=color, shape='hv')
        ), row=1, col=1)

    # Add p-value annotation to the plot
    if p_value is not None:
        p_value_text = (
            'p-value: <0.0001'
            if p_value < 0.0001 else f'p-value: {p_value:.4f}')
        fig.add_annotation(
            text=p_value_text,
            x=0.95, y=95, xref='paper', yref='y1',
            showarrow=False, font=dict(size=12, color='black'),
            bgcolor='white', bordercolor='black', borderwidth=1)

    # Add risk table as second row
    fig.add_trace(go.Table(
        header=dict(
            values=[str(x) for x in risk_table.columns],
            fill_color='lightgrey',
            align='center',
            font=dict(size=14),
            height=35),
        cells=dict(
            values=[risk_table[col].tolist() for col in risk_table.columns],
            fill_color='white',
            align='center',
            font=dict(size=14),
            height=35)
    ), row=2, col=1)

    # Configure axes and layout
    fig.update_yaxes(
        title_text="Survival Probability",
        range=[0, 100],
        tickvals=np.arange(0, 110, 10),
        ticktext=[f"{i}%" for i in range(0, 110, 10)], row=1, col=1)
    fig.update_xaxes(
        title_text=xlabel,
        tickvals=risk_table.columns[1:],
        row=1, col=1)
    if xlim is not None:
        fig.update_xaxes(range=xlim)
    fig.update_layout(height=height, showlegend=True)

    graph_id = get_graph_id(graph_id, suffix)
    return fig, graph_id, graph_label, graph_about


def fig_line_chart(
        df,
        title='Line chart',
        xlabel='x',
        ylabel='y left',
        line_column='Line column',
        lower_column=None,
        upper_column=None,
        index_column='index',
        line_color=None,
        height=340,
        suffix='',
        filepath='',
        save_inputs=False,
        graph_id='',
        graph_label='',
        graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    # Ensure correct index
    df = df.set_index(index_column)

    # Create line trace
    line_trace = go.Scatter(
        x=df.index,
        y=df[line_column],
        mode='lines+markers',
        name=line_column.replace('_', ' ').title(),
        marker=dict(color=line_color),
        line=dict(color=line_color, width=2, dash='solid'),
    )

    data = [line_trace]
    yaxis_min = 0
    yaxis_max = df[line_column].max() * 1.1

    if (upper_column is not None) & (lower_column is not None):
        bounds_trace = go.Scatter(
            x=df.index.tolist() + df.index[::-1].tolist(),
            y=df[upper_column].tolist() + df[lower_column][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(150,150,150,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
        )
        data = [line_trace, bounds_trace]
        yaxis_min = min((0, df[lower_column].min() * 1.1))
        yaxis_max = df[upper_column].max() * 1.1

    # Define layout
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=xlabel,
            tickmode='array',
            tickvals=df.index,
            ticktext=df.index,  # Force display as years
            tickangle=-30,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=ylabel,
            range=[yaxis_min, yaxis_max]
        ),
        legend=dict(x=0.85, y=1, bgcolor='rgba(255,255,255,0.5)'),
        margin=dict(l=60, r=60, t=50, b=80),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=height,
    )

    fig = go.Figure(data=data, layout=layout)
    return fig, graph_id, graph_label, graph_about


def fig_bar_line_chart(
        df,
        title="Combined bar line chart",
        xlabel="x",
        ylabel_left="y left",
        ylabel_right="y right",
        bar_column="Bar column",
        line_column="Line column",
        lower_column=None,
        upper_column=None,
        index_column="index",
        bar_color=None,
        line_color=None,
        height=500,
        suffix='',
        filepath='',
        save_inputs=False,
        graph_id='',
        graph_label='',
        graph_about=''):

    if save_inputs:
        inputs = save_inputs_to_file(locals())

    # Ensure correct index
    df = df.set_index(index_column)

    # # Format x-axis labels to show only the year
    # x_labels = df.index.strftime("%Y")

    # Create bar trace
    bar_trace = go.Bar(
        x=df.index,
        y=df[bar_column],
        name=bar_column.replace("_", " ").title(),
        marker=dict(color=bar_color),
        yaxis="y"
    )

    # Create line trace
    line_trace = go.Scatter(
        x=df.index,
        y=df[line_column],
        mode="lines+markers",
        name=line_column.replace("_", " ").title(),
        marker=dict(color=line_color),
        line=dict(color=line_color, width=2, dash="solid"),
        yaxis="y2"
    )

    data = [bar_trace, line_trace]
    yaxis2_min = 0
    yaxis2_max = df[line_column].max() * 1.1

    if (upper_column is not None) & (lower_column is not None):
        bounds_trace = go.Scatter(
            x=df.index.tolist() + df.index[::-1].tolist(),
            y=df[upper_column].tolist() + df[lower_column][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(150,150,150,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            yaxis="y2"
        )
        data = [bar_trace, line_trace, bounds_trace]
        yaxis2_min = min((0, df[lower_column].min() * 1.1))
        yaxis2_max = df[upper_column].max() * 1.1

    # Define layout
    layout = go.Layout(
        title=title,
        barmode='stack',
        bargap=0.3,
        xaxis=dict(
            title=xlabel,
            tickmode="array",
            tickvals=df.index,
            ticktext=df.index,  # Force display as years
            tickangle=-30,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=ylabel_left,
            range=[0, df[bar_column].max() * 1.1]
        ),
        yaxis2=dict(
            title=ylabel_right,
            overlaying="y",
            side="right",
            showgrid=False,
            range=[yaxis2_min, yaxis2_max]
        ),
        legend=dict(x=0.85, y=1, bgcolor="rgba(255,255,255,0.5)"),
        margin=dict(l=60, r=60, t=50, b=80),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=height,
    )

    fig = go.Figure(data=data, layout=layout)
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
