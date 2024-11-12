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

############################################
############################################
# SECTION 1: Change this section
############################################
############################################

# The suffix must be unique to each insight panel, it is used by the dashboard
suffix = 'ComplOutco'
# Multiple insight panels can share the same research question
# Insight panels are grouped in the dashboard menu according to this
research_question = 'Outcomes'
# The combination of research question and clinical measure must be unique
clinical_measure = 'Complications and Outcomes'

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
    # 'expo14',  # Exposure history in previous 14 days
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
    # 'lesion',  # Skin & mucosa assessment
    # 'treat',  # Treatments & interventions
    # 'labs',  # Laboratory results
    # 'imagi',  # Imaging
    # 'medi',  # Medication
    # 'test',  # Pathogen testing
    # 'diagn',  # Diagnosis
    'compl',  # Complications
    # 'inter',  # Interventions
    # 'follow',  # Follow-up assessment
    # 'withd',  # Withdrawal
]


def create_visuals(df_map):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''
    dd = getRC.getDataDictionary(redcap_url, redcap_api_key)
    # variable_dict is a dictionary of lists according to variable type, which
    # are: 'binary', 'date', 'number', 'freeText', 'units', 'categorical'
    full_variable_dict = getRC.getVariableType(dd)
    binary_list = ['binary', 'categorical', 'OneHot']
    binary_var = sum([full_variable_dict[key] for key in binary_list], [])

    # Descriptive table
    column = 'outcome'
    column_reorder = ['Discharged', 'Death', 'Censored']
    # column = 'severity'
    # column_reorder = ['Mild', 'Moderate', 'Severe', 'Critical']
    inclu_columns = ia.get_variables_from_sections(
        df_map.columns, ['compl', 'outco'])
    inclu_columns += [column]
    table, totals = ia.descriptive_table(
        df_map[inclu_columns], column=column,
        full_variable_dict=full_variable_dict, return_totals=True)
    table, table_key = ia.reformat_descriptive_table(
        table, dictionary=dd, totals=totals,
        column_reorder=column_reorder, section_reorder=['compl', 'outco'])
    # table_key += '<br>Mild: <25 skin lesions, Moderate: 25-99 skin lesions, '
    # table_key += 'Severe: 100-250 skin lesions, Critical: >250 skin lesions'
    fig_table = idw.fig_table(
        table, dictionary=dd,
        table_key=table_key,
        graph_id='table_' + suffix,
        graph_label='Descriptive Table',
        graph_about='Summary of complications and interventions')

    inclu_columns = [
        col for col in df_map.columns if col.split('___')[0] in binary_var]
    inclu_columns = [col for col in inclu_columns if 'addi' not in col]

    # Complications frequency and upset charts
    proportions = ia.get_proportions(df_map[inclu_columns], ['compl'])
    intersections = ia.get_intersections(df_map, proportions=proportions)
    freq_chart = idw.fig_frequency_chart(
        proportions, dictionary=dd,
        title='Frequency of complications',
        graph_id='freq_compl_' + suffix,
        graph_label='Complications: Frequency',
        graph_about='Frequency of the ten most common complications')
    upset_plot = idw.fig_upset(
        intersections, dictionary=dd,
        title='Intersection sizes of the five most common complications',
        graph_id='upset_compl_' + suffix,
        graph_label='Complications: Intersections',
        graph_about='Intersection sizes of the five most common complications')

    return fig_table, freq_chart, upset_plot


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

print(research_question + ': ' + clinical_measure)

vari_list = getRC.getVariableList(redcap_url, redcap_api_key, sections)
df_map = getRC.get_REDCAP_Single_DB(
    redcap_url, redcap_api_key, site_mapping, vari_list)

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
# instructions_str = '''
# 1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster).
# 2. Change datasets using the dropdown (country selections are remembered).
# 3. Hover mouse on chart for tooltip data.
# 4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it.
# 5. Toggle selected countries on/off by clicking on the legend (far right).
# '''
instructions_str = '''
1. Select categories to filter by sex, age, country and outcome.
2. Click on Insights and then on each tab to view tables and figures.
3. Hover mouse on chart for tooltip data (only available for figures).
4. Zoom-in on figures using the buttons that appears when you hover the mouse near the top right of the figure.
5. To reset the chart, double-click on it.
6. Download a .png of the figure using the camera button that appears when you hover the mouse near the top right of the figure.
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
