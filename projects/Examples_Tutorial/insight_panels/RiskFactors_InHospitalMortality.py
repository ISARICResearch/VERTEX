import dash
import dash_bootstrap_components as dbc
import getREDCapData as getRC
import vertex.IsaricAnalytics as ia
import vertex.IsaricDraw as idw
import numpy as np
import pandas as pd
import pycountry
import redcap_config as rc_config
from dash import html
from dash.dependencies import Input, Output, State

############################################
############################################
# SECTION 1: Change this section
############################################
############################################

# The suffix must be unique to each insight panel, it is used by the dashboard
suffix = 'rfmortality'
# Multiple insight panels can share the same research question
# Insight panels are grouped in the dashboard menu according to this
research_question = 'Risk Factors'
# The combination of research question and clinical measure must be unique
clinical_measure = 'In-hospital mortality'

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
    'comor',  # Co-morbidities and risk factors
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
    # 'compl',  # Complications
    # 'inter',  # Interventions
    # 'follow',  # Follow-up assessment
    # 'withd',  # Withdrawal
]

# Leftmost edge of the bins
age_groups = [
    '0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
    '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
    '81-85', '86-90', '91-95', '96-100']

def create_visuals(df_map):
    """
    Function to create all visuals in the insight panel from the RAP dataframe.

    This function generates a table and a forest plot based on logistic regression 
    performed on the input dataframe. It handles multivariate and univariate logistic 
    regressions, aligns results, and formats them into visual outputs.

    Parameters:
    - cv_rap_dataframe_df (pd.DataFrame): Input dataframe containing the necessary data 
      for visual generation.

    Returns:
    - cv_table_figure_obj: A table figure with logistic regression results.
    - cv_forest_plot_figure_obj: A forest plot figure visualizing the odds ratios.
    """

    # Define outcome and predictor variables
    elr_outcome_str = 'outcome'
    elr_predictors_list = ['age_group']  # Initialize with basic predictors
    elr_predictors_list += [
        col_str for col_str in df_map.columns if 'demog_sex' in col_str
    ]

    # Prepare the input dataframe with one-hot encoding
    elr_input_dataframe_df = df_map[[elr_outcome_str] + elr_predictors_list].copy()

    # Convert categorical variables if applicable
    elr_input_dataframe_df = ia.from_dummies(
        df_map[[elr_outcome_str] + elr_predictors_list], column='demog_sex'
    )
    elr_predictors_list = ['age_group', 'demog_sex']

    # Convert the outcome column to binary (1 = Death, 0 = Other)
    elr_input_dataframe_df['outcome'] = (
        elr_input_dataframe_df['outcome'] == 'Death'
    )

    # Perform logistic regression with all predictors (multivariate)
    elr_lr_multi_df = ia.execute_logistic_regression(
        elr_input_dataframe_df, elr_outcome_str, elr_predictors_list, print_results=False
    )

    # Perform logistic regression for each predictor (univariate)
    cv_univariate_results_list = []
    for elr_predictor_str in elr_predictors_list:
        cv_univariate_results_list.append(
            ia.execute_logistic_regression(
                reg_type='uni',
                elr_dataframe_df=elr_input_dataframe_df,
                elr_outcome_str=elr_outcome_str,
                elr_predictors_list=[elr_predictor_str],
                print_results=False
            )
        )
    cv_univariate_results_df = pd.concat(cv_univariate_results_list)

    # Align 'Study' order between univariate and multivariate results
    cv_univariate_results_df['Study'] = pd.Categorical(
        cv_univariate_results_df['Study'],
        categories=elr_lr_multi_df['Study'],
        ordered=True
    )
    cv_univariate_results_df = cv_univariate_results_df.sort_values('Study').reset_index(drop=True)

    # Reset index for consistency
    elr_lr_multi_df = elr_lr_multi_df.reset_index(drop=True)
    cv_univariate_results_df = cv_univariate_results_df.reset_index(drop=True)

    # Create a copy of multivariate results to avoid altering the original dataframe
    elr_logistic_response_df = elr_lr_multi_df.copy()
    elr_logistic_response_df = elr_logistic_response_df.drop('Study', axis=1)

    # Concatenate univariate and multivariate regression results
    elr_lr_final_df = pd.concat([cv_univariate_results_df, elr_logistic_response_df], axis=1)

    # Create the final table
    elr_lr_table_df = elr_lr_final_df.copy()
    cv_table_figure_obj = idw.fig_table(
        elr_lr_table_df,
        dictionary=None,
        graph_id='table_visual_rap',
        graph_label='Logistic Regression Results Table',
        graph_about='This table contains the logistic regression results for multivariate and univariate analyses.'
    )

    # Rename columns in multivariate results for clarity
    elr_lr_multi_df.rename(
        columns={
            'OddsRatio (multi)': 'OddsRatio',
            'LowerCI (multi)': 'LowerCI',
            'UpperCI (multi)': 'UpperCI',
            'p-valor (multi)': 'p-value'
        },
        inplace=True
    )

    # Create the forest plot figure
    cv_forest_plot_figure_obj = idw.fig_forest_plot(
        elr_lr_multi_df,
        graph_id='forest_plot_visual_rap',
        graph_label='Forest Plot',
        graph_about='This plot visualizes the odds ratios from multivariate logistic regression.'
    )

    return cv_table_figure_obj, cv_forest_plot_figure_obj


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

bins = [float(x.split('-')[0].strip()) for x in age_groups] + [np.inf]
df_map['age_group'] = pd.cut(
    df_map['age'], bins=bins, labels=age_groups, right=False)

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
                                dbc.Col(visual, id='col-' + visual.id)
                                ]), label=label)
                            for visual, label, _ in visuals])]
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