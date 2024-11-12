from dash import dcc
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff
import IsaricAnalytics as ia


############################################
############################################
# Figures
############################################
############################################


def fig_cumulative_bar_chart(
        df, dictionary=None,
        title='Cumulative Bar by Timepoint', xlabel='x', ylabel='y',
        base_color_map=None,
        graph_id='cumulative-bar-chart', graph_label='', graph_about=''):
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
    return graph, graph_label, graph_about


def fig_sex_boxplot(
        df, dictionary=None,
        title='Hospital Stay Length by Sex', ylabel='y', base_color_map=None,
        graph_id='sex-boxplot', graph_label='', graph_about=''):
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
    return graph, graph_label, graph_about


def fig_age_group_boxplot(
        df, dictionary=None,
        title='Hospital Stay Length by Age Group', ylabel='y',
        base_color_map=None,
        graph_id='age-group-boxplot', graph_label='', graph_about=''):
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
    return graph, graph_label, graph_about


def fig_forest_plot(
        df, dictionary=None,
        title='Forest Plot',
        labels=['Study', 'OddsRatio', 'LowerCI', 'UpperCI'],
        graph_id='forest-plot', graph_label='', graph_about=''):
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
    return graph, graph_label, graph_about


def fig_heatmap(
        df1, df2, dictionary=None, title='Heatmap',
        graph_id='heatmap', graph_label='', graph_about=''):
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
    return graph, graph_label, graph_about


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
