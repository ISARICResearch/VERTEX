import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import pycountry
import IsaricDraw as idw
import IsaricAnalytics as ia
import getREDCapData as getRC
import redcap_config as rc_config
import plotly.graph_objs as go
from dash import dcc
import plotly.express as px

############################################
############################################
# SECTION 1: Change this section
############################################
############################################

# The suffix must be unique to each insight panel, it is used by the dashboard
suffix = 'Prog'
# Multiple insight panels can share the same research question
# Insight panels are grouped in the dashboard menu according to this
research_question = 'Outcomes'
# The combination of research question and clinical measure must be unique
clinical_measure = 'Disease progression'

# Provide a list of all ARCH data sections needed in the RAP dataframe
# Only variables from these sections will appear in the visuals
sections = [
    'dates',  # Onset & presentation (REQUIRED)
    'demog',  # Demographics (REQUIRED)
    'daily',  # Daily sections (REQUIRED)
    'asses',  # Assessment (REQUIRED)
    'outco',  # Outcome (REQUIRED)
    # 'inclu',  # Inclusion criteria
    # 'readm',  # Re-admission and previous pin
    # 'travel',  # Travel history
    'expo14',  # Exposure history in previous 14 days
    # 'preg',  # Pregnancy
    # 'infa',  # Infant
    # 'comor',  # Co-morbidities and risk factors
    # 'medic',  # Medical history
    # 'drug7',  # Medication previous 7-days
    # 'drug14',  # Medication previous 14-days
    # 'vacci',  # Vaccination
    # 'advital',  # Vital signs & assessments on admission
    # 'adsym',  # Signs and symptoms on admission
    # 'vital',  # Vital signs & assessments
    # 'sympt',  # Signs and symptoms
    'lesion',  # Skin & mucosa assessment
    # 'treat',  # Treatments & interventions
    # 'labs',  # Laboratory results
    # 'imagi',  # Imaging
    # 'medi',  # Medication
    # 'test',  # Pathogen testing
    # 'diagn',  # Diagnosis
    # 'compl',  # Complications
    # 'inter',  # Interventions
    # 'follow',  # Follow-up assessment
    # 'withd',  # Withdrawal
]

# Leftmost edge of the bins
age_groups = ['0-4', '5-17', '18-44', '45-64']


def create_visuals(df_map):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''
    dd = getRC.getDataDictionary(redcap_url, redcap_api_key)
    # variable_dict is a dictionary of lists according to variable type, which
    # are: 'binary', 'date', 'number', 'freeText', 'units', 'categorical'
    full_variable_dict = getRC.getVariableType(dd)
    # binary_list = ['binary', 'categorical', 'OneHot']
    # binary_var = sum([full_variable_dict[key] for key in binary_list], [])

    bins = [float(x.split('-')[0].strip()) for x in age_groups] + [np.inf]
    df_map['age_group'] = pd.cut(
        df_map['age'], bins=bins, labels=age_groups, right=False)

    inclu_columns = ['age_group', 'expo14_type', 'lesion_mpox_mucos']
    inclu_columns += ['lesion_mpox_skinles', 'severity', 'outcome']
    inclu_columns = [
        col for col in df_map.columns if col.split('___')[0] in inclu_columns]
    expo14_type_ranking = ['Healthcare worker exposure']
    # expo14_type_ranking = ['Sexual contact', 'Healthcare worker exposure']
    expo14_type_ranking += ['Community contact', 'Vertical transmission']
    df_sankey = df_map[inclu_columns].copy()
    df_sankey['transmission'] = 'Unknown'
    for value in expo14_type_ranking[::-1]:
        df_sankey.loc[(
            df_sankey['expo14_type___' + value] == 1), 'transmission'] = value
    lesion_columns = ['lesion_mpox_skinles___There are active skin lesions']
    lesion_columns += ['lesion_mpox_mucos___There are mucosal lesions']
    rename_dict = dict(zip(
        lesion_columns, [x.split('___')[0] for x in lesion_columns]))
    df_sankey.rename(columns=rename_dict, inplace=True)
    df_sankey['lesion_location'] = 'None'
    df_sankey.loc[(
        df_sankey['lesion_mpox_skinles'] == 1), 'lesion_location'] = 'Skin'
    df_sankey.loc[(
        df_sankey['lesion_mpox_mucos'] == 1), 'lesion_location'] = 'Mucosal'
    df_sankey.loc[(
            (df_sankey['lesion_mpox_mucos'] == 1) &
            (df_sankey['lesion_mpox_skinles'] == 1)),
        'lesion_location'] = 'Both'
    inclu_columns = ['age_group', 'transmission', 'lesion_location']
    inclu_columns += ['severity', 'outcome']
    df_sankey = df_sankey[inclu_columns].astype('object').fillna('Unknown')

    sort_values_dict = {
        'age_group': age_groups + ['Unknown'],
        'transmission': expo14_type_ranking + ['Unknown'],
        'lesion_location': ['Both', 'Skin', 'Mucosal', 'None'],
        'severity': ['Critical', 'Severe', 'Moderate', 'Mild', 'Unknown'],
        'outcome': ['Death', 'Discharged', 'Censored']}

    cmap_dict = {
        'severity': {
            'Critical': 'rgb(118, 42, 131)',
            'Severe': 'rgb(219, 196, 224)',
            'Moderate': 'rgb(200, 233, 194)',
            'Mild': 'rgb(27, 120, 55)'},
        'outcome': {
            'Death': '#DF0069',
            'Discharged': '#00C26F',
            'Censored': '#FFF500'},
        # 'age_group': ['#1b9e77', '#d95f02', '#7570b3']
    }

    sankey = fig_sankey(
        df_sankey, sort_values_dict, cmap_dict,
        graph_id='sankey1',
        graph_label='Transmission: Sankey diagram', graph_about='')

    return sankey,


def get_full_cmap_dict(
        df, cmap_dict, sort_values_dict=None,
        default_cmap_list=[
            'Oranges_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Purples_r']):
    if sort_values_dict is None:
        sort_values_dict = {}
    full_cmap_dict = cmap_dict.copy()
    plotly_colorscales = px.colors.named_colorscales()
    plotly_colorscales += [x + '_r' for x in px.colors.named_colorscales()]
    counter = 0
    for variable in df.columns:
        if variable not in sort_values_dict.keys():
            values = list(df[variable].unique())
        else:
            values = sort_values_dict[variable]
            inclu_ind = (df[variable].isin(sort_values_dict[variable]) == 0)
            values += list(df.loc[inclu_ind, variable].unique())
        unknown_ind = ('Unknown' in values)
        values = [x for x in values if x != 'Unknown']
        # Get colors
        n = len(values)
        tol = 0.1
        if variable not in full_cmap_dict.keys():
            colors = px.colors.sample_colorscale(
                default_cmap_list[counter], np.linspace(tol, 1 - tol, n))
            full_cmap_dict[variable] = dict(zip(values, colors))
            counter += 1
            if counter > len(default_cmap_list):
                counter = 0
        elif (isinstance(full_cmap_dict[variable], str) and
                (full_cmap_dict[variable].lower() not in plotly_colorscales)):
            print('Warning: colormap string not available')
            colors = px.colors.sample_colorscale(
                default_cmap_list[counter], np.linspace(tol, 1 - tol, n))
            full_cmap_dict[variable] = dict(zip(values, colors))
            counter += 1
            if counter > len(default_cmap_list):
                counter = 0
        elif isinstance(full_cmap_dict[variable], str):
            colors = px.colors.sample_colorscale(
                full_cmap_dict[variable], np.linspace(tol, 1 - tol, n))
            full_cmap_dict[variable] = dict(zip(values, colors))
        elif isinstance(full_cmap_dict[variable], list):
            repeat_n = (n // len(full_cmap_dict[variable])) + 1
            colors = full_cmap_dict[variable]*repeat_n
            colors = colors[:n]
            colors = px.colors.convert_colors_to_same_type(
                colors, 'rgb')[0][0]
            full_cmap_dict[variable] = dict(zip(values, colors))
        elif isinstance(full_cmap_dict[variable], dict):
            values = [x for x in values if x not in full_cmap_dict[variable]]
            n = len(values)
            if n > 0:
                colors = px.colors.sample_colorscale(
                    default_cmap_list[counter], np.linspace(tol, 1 - tol, n))
                full_cmap_dict[variable].update(dict(zip(values, colors)))
                counter += 1
                if counter > len(default_cmap_list):
                    counter = 0
            full_cmap_dict[variable] = {
                k: px.colors.convert_colors_to_same_type(v, 'rgb')[0][0]
                for k, v in full_cmap_dict[variable].items()}
        else:
            colors = px.colors.sample_colorscale(
                default_cmap_list[counter], np.linspace(tol, 1 - tol, n))
            full_cmap_dict[variable] = dict(zip(values, colors))
            counter += 1
            if counter > len(default_cmap_list):
                counter = 0
        if unknown_ind:
            full_cmap_dict[variable]['Unknown'] = 'rgb(190,190,190)'
    return full_cmap_dict
    #
    #
    #     inclu_in_dict = (
    #         list(full_cmap_dict[variable].keys()) + ['Unknown']
    #         if variable in full_cmap_dict.keys() else None)
    #     condition = (variable in full_cmap_dict.keys()) and (
    #         isinstance(full_cmap_dict[variable], dict)) and (
    #         df[variable].isin(inclu_in_dict).all() == 0)
    #     if condition:
    #         warning_str = 'Warning: colormap dict for ' + variable
    #         warning_str += ' does not include all values'
    #         print(warning_str)
    #         full_cmap_dict[variable] = None
    #     if variable not in full_cmap_dict.keys():
    #         full_cmap_dict[variable] = default_cmap_list[counter]
    #         counter += 1
    #         if counter > len(default_cmap_list):
    #             counter = 0
    #
    # nodes['color'] = ''
    # for variable in nodes['variable'].unique():
    #     variable_ind = (
    #         (nodes['variable'] == variable) &
    #         (nodes['node'].str.endswith('Unknown') == 0))
    #     n = variable_ind.sum()
    #     if isinstance(new_cmap_dict[variable], str):
    #         tol = 0.1
    #         color_list = px.colors.sample_colorscale(
    #             new_cmap_dict[variable], np.linspace(tol, 1 - tol, n))
    #         nodes.loc[variable_ind, 'color'] = color_list
    #     elif isinstance(new_cmap_dict[variable], list):
    #         repeat_n = (n // len(new_cmap_dict[variable])) + 1
    #         color_list = new_cmap_dict[variable]*repeat_n
    #         color_list = color_list[:n]
    #         nodes.loc[variable_ind, 'color'] = color_list
    #     elif isinstance(new_cmap_dict[variable], dict):
    #         nodes.loc[variable_ind, 'color'] = (
    #             nodes.loc[variable_ind, 'value'].map(new_cmap_dict[variable]))
    # grey = 'rgb(190,190,190)'
    # nodes.loc[(nodes['node'].str.endswith('Unknown') == 1), 'color'] = grey
    # nodes['color'] = px.colors.convert_colors_to_same_type(
    #     nodes['color'].to_list(), 'rgb')[0]
    # return full_cmap_dict


def get_source_target(df, source, target, sorter_dict):
    df_new = df.groupby([source, target], observed=True).size()
    df_new = df_new.reset_index()
    df_new[source] = df_new[source].map(lambda x: source + '___' + x)
    df_new[target] = df_new[target].map(lambda x: target + '___' + x)
    df_new = df_new.sort_values(
        by=[source, target], key=lambda z: z.map(sorter_dict))
    df_new = df_new.rename(
        columns={source: 'source', target: 'target', 0: 'value'})
    source_y = df_new.groupby('source')['value'].sum().rename('source_y')
    source_y = source_y.sort_index(key=lambda z: z.map(sorter_dict))
    source_y = (
        source_y.shift().fillna(0).cumsum() + 0.5*source_y) / source_y.sum()
    target_y = df_new.groupby('target')['value'].sum().rename('target_y')
    target_y = target_y.sort_index(key=lambda z: z.map(sorter_dict))
    target_y = (
        target_y.shift().fillna(0).cumsum() + 0.5*target_y) / target_y.sum()
    df_new = pd.merge(df_new, source_y, on='source', how='left')
    df_new = pd.merge(df_new, target_y, on='target', how='left')
    return df_new


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


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
    # merged_df = pd.merge(complete_df, df, on='timepoint', how='left')
    merged_df = df
    # Pivot the merged DataFrame to get cumulative sums for each stack_group
    # at each timepoint
    pivot_df = merged_df.pivot_table(
        index='timepoint', columns='stack_group',
        values='value', aggfunc='sum')

    pivot_df = pivot_df.fillna(0)

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


def fig_sankey(
        df_sankey, sort_values_dict, cmap_dict,
        graph_id='sankey', graph_label='', graph_about=''):
    sorter_key_list = [
        [y + '___' + x for x in sort_values_dict[y]]
        for y in sort_values_dict.keys()]

    sorter_dict = dict(zip(
        sum(sorter_key_list, []),
        sum([list(range(len(x))) for x in sort_values_dict.values()], [])))

    sankey_values_list = [
        get_source_target(df_sankey, x, y, sorter_dict)
        for x, y in zip(df_sankey.columns[:-1], df_sankey.columns[1:])]
    sankey_values = pd.concat(sankey_values_list)
    sankey_values['source_x'] = (sankey_values.index == 0).cumsum()
    sankey_values['target_x'] = (
        sankey_values['source_x'] / sankey_values['source_x'].max())
    sankey_values['source_x'] = (
        (sankey_values['source_x'] - 1) / sankey_values['source_x'].max())
    sankey_values = sankey_values.reset_index(drop=True)
    source_columns = ['source', 'source_x', 'source_y']
    sankey_source = sankey_values[source_columns].rename(
        columns=dict(zip(source_columns, ['node', 'x', 'y'])))
    target_columns = ['target', 'target_x', 'target_y']
    sankey_target = sankey_values[target_columns].rename(
        columns=dict(zip(target_columns, ['node', 'x', 'y'])))
    sankey_target = sankey_target.sort_values(
        by='node', key=lambda z: z.map(sorter_dict))
    nodes = pd.concat([sankey_source, sankey_target])
    nodes = nodes.loc[nodes.duplicated() == 0].reset_index(drop=True)
    nodes['variable'] = nodes['node'].apply(lambda x: x.split('___')[0])
    nodes['value'] = nodes['node'].apply(
        lambda x: '___'.join(x.split('___')[1:]))

    full_cmap_dict = get_full_cmap_dict(df_sankey, cmap_dict, sort_values_dict)
    nodes['color'] = nodes['node'].map({
        var + '___' + val: col
        for var in full_cmap_dict.keys()
        for val, col in full_cmap_dict[var].items()})

    nodes = nodes.reset_index().set_index('node')

    tol = 1e-4
    nodes['x'] = nodes['x']*(1 - 2*tol) + tol
    nodes['y'] = nodes['y']*(1 - 2*tol) + tol

    hovertemplate_node = '%{label}'
    hovertemplate_link = '%{source.label} to %{target.label}'

    node = {
        'x': nodes['x'].tolist(),
        'y': nodes['y'].tolist(),
        'color': nodes['color'],
        'label': [x.split('___')[1] for x in nodes.index],
        'hovertemplate': hovertemplate_node,
        'pad': 15,
        'thickness': 20,
        'line': {'color': 'black', 'width': 1.2}}
    link = {
        'source': nodes.loc[sankey_values['source'], 'index'],
        'target': nodes.loc[sankey_values['target'], 'index'],
        'color': nodes.loc[sankey_values['source'], 'color'].apply(
            lambda x: rgb_to_rgba(x, 0.2)),
        'value': sankey_values['value'],
        'hovertemplate': hovertemplate_link,
        'line': {'color': 'rgba(0,0,0,0.3)', 'width': 0.3}
    }

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        valueformat='.0f',
        node=node,
        link=link)])

    annotations = nodes.loc[(
        nodes[['variable', 'x']].duplicated() == 0), ['variable', 'x']]
    #
    # tol = 0.05
    # annotations['x'] = annotations['x']*(1 + 2*tol) - tol

    for ind in annotations.index:
        text = annotations.loc[ind, 'variable'].replace('_', ' ').capitalize()
        text = '<b>' + text + '</b>'
        fig.add_annotation(
            text=text, x=annotations.loc[ind, 'x'], y=1.1,
            showarrow=False, xanchor='center')

    graph = dcc.Graph(
        id=graph_id + suffix,
        figure=fig)
    return graph, graph_label, graph_about


############################################
############################################
# END SECTION 1: You should not need to change anything else
############################################
############################################

############################################
############################################
# SECTION 2: Config, data reading, processing, modal creation
############################################
############################################

redcap_url = rc_config.redcap_url
redcap_api_key = rc_config.redcap_api_key
site_mapping = rc_config.site_mapping

############################################
# Data reading and initial proccesing
############################################

vari_list = getRC.getVariableList(redcap_url, redcap_api_key, sections)
df_map = getRC.get_REDCAP_Single_DB(
    redcap_url, redcap_api_key, site_mapping, vari_list)

# df_severity = getRC.get_REDCAP_single_variable(redcap_url, redcap_api_key)

all_countries = pycountry.countries
countries = [
    {'label': country.name, 'value': country.alpha_3}
    for country in all_countries]
unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates(
    ).sort_values(by='slider_country')

country_dropdown_options = []
for uniq_county in range(len(unique_countries)):
    name_country = unique_countries['slider_country'].iloc[uniq_county]
    code_country = unique_countries['country_iso'].iloc[uniq_county]
    country_dropdown_options.append(
        {'label': name_country, 'value': code_country})

# This text appears after clicking the insight panel's Instructions button
instructions_str = '''
1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster).
2. Change datasets using the dropdown (country selections are remembered).
3. Hover mouse on chart for tooltip data.
4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it.
5. Toggle selected countries on/off by clicking on the legend (far right).
'''

# This text appears after clicking the insight panel's About button
about_list = [
    '<strong>' + label + '</strong>' + about
    for _, label, about in create_visuals(df_map)]
about_str = '\n'.join(
    ['Information about each visual in the insight panel:'] + about_list)

############################################
# Modal creation
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


def create_modal():
    visuals = create_visuals(df_map)

    modal = [
        dbc.ModalHeader(html.H3(
            research_question + ': ' + clinical_measure,
            id='line-graph-modal-title',
            style={'fontSize': '2vmin', 'fontWeight': 'bold'})
        ),
        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title='Filters and Controls',
                    children=[
                        idw.filters_controls(suffix, country_dropdown_options)]
                ),
                dbc.AccordionItem(
                    title='Insights',
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([
                                dbc.Col(visual, id='col-'+visual.id)
                                ]), label=label)
                            for visual, label, _ in visuals])
                    ]
                )
            ])
        ], style={
                'overflowY': 'auto', 'minHeight': '75vh', 'maxHeight': '75vh'}
        ),
        idw.ModalFooter(
            suffix,
            generate_html_text(instructions_str),
            generate_html_text(about_str))
    ]
    return modal


############################################
############################################
# SECTION 3: Callbacks
############################################
############################################

def register_callbacks(app, suffix):
    @app.callback(
        [Output(f'country-checkboxes_{suffix}', 'value'),
         Output(f'country-selectall_{suffix}', 'options'),
         Output(f'country-selectall_{suffix}', 'value')],
        [Input(f'country-selectall_{suffix}', 'value'),
         Input(f'country-checkboxes_{suffix}', 'value')],
        [State(f'country-checkboxes_{suffix}', 'options')]
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

        if trigger_id == f'country-selectall_{suffix}':
            if 'all' in select_all_value:
                # 'Select all' (now 'Unselect all') is checked
                output = [
                    [option['value'] for option in all_countries_options],
                    [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # 'Unselect all' is unchecked
                output = [[], [{'label': 'Select all', 'value': 'all'}], []]
        elif trigger_id == f'country-checkboxes_{suffix}':
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
        Output(f'country-fade_{suffix}', 'is_in'),
        [Input(f'country-display_{suffix}', 'n_clicks')],
        [State(f'country-fade_{suffix}', 'is_in')]
    )
    def toggle_fade(n_clicks, is_in):
        state = is_in
        if n_clicks:
            state = not is_in
        return state

    @app.callback(
        Output(f'country-display_{suffix}', 'children'),
        [Input(f'country-checkboxes_{suffix}', 'value')],
        [State(f'country-checkboxes_{suffix}', 'options')]
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
        [Output('col-' + visual.id, 'children')
            for visual, _, _ in create_visuals(df_map)],
        [Input(f'submit-button_{suffix}', 'n_clicks')],
        [State(f'gender-checkboxes_{suffix}', 'value'),
         State(f'age-slider_{suffix}', 'value'),
         State(f'outcome-checkboxes_{suffix}', 'value'),
         State(f'country-checkboxes_{suffix}', 'value')]
    )
    def update_figures(click, genders, age_range, outcomes, countries):
        filtered_df = df_map[(
            (df_map['slider_sex'].isin(genders)) &
            ((df_map['age'] >= age_range[0]) | df_map['age'].isna()) &
            ((df_map['age'] <= age_range[1]) | df_map['age'].isna()) &
            (df_map['outcome'].isin(outcomes)) &
            (df_map['country_iso'].isin(countries)))]

        if filtered_df.empty:
            visuals = None
        else:
            visuals = [visual for visual, _, _ in create_visuals(filtered_df)]
        return visuals

    # End of callbacks
    return
