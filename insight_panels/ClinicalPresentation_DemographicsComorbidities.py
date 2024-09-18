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
# Change only this section
############################################
############################################

suffix = 'DemogComor'
panel_title = 'Demographics / Comorbidities'
research_question = 'Clinical Presentation'

instructions_str = '''
1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster).
2. Change datasets using the dropdown (country selections are remembered).
3. Hover mouse on chart for tooltip data.
4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it.
5. Toggle selected countries on/off by clicking on the legend (far right).
'''

about_str = '''
Information about each visualisation in the insight panel
1. ...
2. ...
'''

# sections = ['dates', 'demog', 'comor', 'daily', 'outco']
sections = ['dates', 'demog', 'daily', 'outco']


def create_visuals(df_map):
    ############################################
    # get Variable type
    ############################################
    dd = getRC.getDataDictionary(redcap_url, redcap_api_key)
    variable_dict = getRC.getVariableType(dd)
    variables_binary = variable_dict['binary']
    # variables_date = variable_dict['date']
    variables_number = variable_dict['date']
    # variables_freeText = variable_dict['freeText']
    # variables_units = variable_dict['units']
    # variables_categorical = variable_dict['categorical']

    # Variable and label dictionary
    correct_names = dd[['field_name', 'field_label']]

    color_map = {
        'Discharge': '#00C26F',
        'Censored': '#FFF500',
        'Death': '#DF0069'}

    filter_columns = ['age_group', 'mapped_outcome', 'slider_sex']
    df_age_gender = df_map[filter_columns + ['usubjid']].groupby(
        filter_columns, observed=True).count().reset_index()
    df_age_gender.rename(
        columns={
            'slider_sex': 'side',
            'mapped_outcome': 'stack_group',
            'usubjid': 'value',
            'age_group': 'y_axis'},
        inplace=True)
    pyramid_chart = idw.fig_dual_stack_pyramid(
        df_age_gender, base_color_map=color_map,
        graph_id='age_gender_pyramid_chart_' + suffix,
        graph_label='Demographics: Population Pyramid')
    print(df_age_gender.shape)

    demog_columns = [col for col in df_map.columns if col.startswith('demog')]
    demog_columns += ['age']
    descriptive = ia.descriptive_table(
        df_map[demog_columns], correct_names,
        variables_binary, ['age'] + variables_number)
    fig_table_symp = idw.fig_table(
        descriptive,
        graph_id='demog_table_' + suffix,
        graph_label='Demographics: Descriptive Table')

    return pyramid_chart, fig_table_symp
    # proportions_comor, set_data_comor = ia.get_proportions(
    #     df_map, 'comorbidities')
    # freq_chart_comor = idw.fig_frequency_chart(
    #     proportions_comor,
    #     title='Frequency of comorbidities on presentation',
    #     graph_id='comor_freq_' + suffix,
    #     graph_label='Comorbidities on presentation: Frequency')
    # upset_plot_comor = idw.fig_upset(
    #     set_data_comor,
    #     title='Frequency of combinations of the five most common comorbidities',
    #     graph_id='comor_upset_' + suffix,
    #     graph_label='Comorbidities on presentation: Intersections')
    #
    # comor_columns = [col for col in df_map.columns if col.startswith('comor')]
    # descriptive = ia.descriptive_table(
    #     df_map[comor_columns], correct_names,
    #     variables_binary, variables_number)
    # fig_table_comor = idw.fig_table(
    #     descriptive,
    #     graph_id='comor_table_' + suffix,
    #     graph_label='Comorbidities on presentation: Descriptive Table')
    #
    # return pyramid_chart, fig_table_symp, freq_chart_comor, upset_plot_comor, fig_table_comor


############################################
############################################
# REDCap elements
############################################
############################################

redcap_url = rc_config.redcap_url
redcap_api_key = rc_config.redcap_api_key
site_mapping = rc_config.site_mapping

############################################
############################################
# Data reading and initial proccesing
############################################
############################################


def create_df_map(redcap_url, redcap_api_key, site_mapping, sections):
    vari_list = getRC.getVariableList(
        redcap_url, redcap_api_key, sections)
    df_map = getRC.get_REDCAP_Single_DB(
        redcap_url, redcap_api_key, site_mapping, vari_list)

    bins = [
        0, 6, 11, 16, 21, 26, 31, 36,
        41, 46, 51, 56, 61, 66, 71, 76,
        81, 86, 91, 96, 101]
    labels = [
        '0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
        '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
        '81-85', '86-90', '91-95', '96-100']
    df_map['age_group'] = pd.cut(
        df_map['age'], bins=bins, labels=labels, right=False)
    df_map['mapped_outcome'] = df_map['outcome']
    return df_map


df_map = create_df_map(redcap_url, redcap_api_key, site_mapping, sections)

all_countries = pycountry.countries
countries = [
    {'label': country.name, 'value': country.alpha_3}
    for country in all_countries]
sections = getRC.getDataSections(redcap_url, redcap_api_key)
vari_list = getRC.getVariableList(
    redcap_url, redcap_api_key,
    ['dates', 'demog', 'comor', 'daily', 'outco', 'labs', 'vital',
        'adsym', 'inter', 'treat'])

unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates(
    ).sort_values(by='slider_country')

country_dropdown_options = []
for uniq_county in range(len(unique_countries)):
    name_country = unique_countries['slider_country'].iloc[uniq_county]
    code_country = unique_countries['country_iso'].iloc[uniq_county]
    country_dropdown_options.append(
        {'label': name_country, 'value': code_country})

############################################
############################################
# Modal creation
############################################
############################################


def generate_html_text(text):
    text_list = text.strip('\n').split('\n')
    div_list = []
    for line in text_list:
        div_list.append(html.Div(line))
        div_list.append(html.Br())
    div = html.Div(div_list[:-1])
    return div


def create_modal():
    visuals = create_visuals(df_map)

    modal = [
        dbc.ModalHeader(html.H3(
            research_question + ': ' + panel_title,
            id="line-graph-modal-title",
            style={"fontSize": "2vmin", "fontWeight": "bold"})
        ),
        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Filters and Controls",
                    children=[
                        idw.filters_controls(suffix, country_dropdown_options)]
                ),
                dbc.AccordionItem(
                    title="Insights",
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([
                                dbc.Col(visual, id='col-'+visual.id)
                                ]), label=label)
                            for visual, label in visuals])
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
# Callbacks
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
                # "Select all" (now "Unselect all") is checked
                output = [
                    [option['value'] for option in all_countries_options],
                    [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # "Unselect all" is unchecked
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
        Output(f"country-fade_{suffix}", "is_in"),
        [Input(f"country-display_{suffix}", "n_clicks")],
        [State(f"country-fade_{suffix}", "is_in")]
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
            return "Country:"

        # Create a dictionary to map values to labels
        value_label_map = {
            option['value']: option['label'] for option in all_options}

        # Build the display string
        selected_labels = [
            value_label_map[val] for val in selected_values
            if val in value_label_map]
        display_text = ", ".join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            output = f'Country: {selected_labels[0]}, '
            output += f'+{len(selected_labels) - 1} more...'
        else:
            output = f'Country: {display_text}'
        return output

    ############################################
    ############################################
    # Specific Callbacks
    # Modify outputs
    ############################################
    ############################################

    @app.callback(
        [Output('col-' + visual.id, 'children')
            for visual, _ in create_visuals(df_map)],
        # [Output('age_gender_pyramid_chart_' + suffix, 'children'),
        #  Output('demog_table_' + suffix, 'children')],
        [Input(f'submit-button_{suffix}', 'n_clicks')],
        [State(f'gender-checkboxes_{suffix}', 'value'),
         State(f'age-slider_{suffix}', 'value'),
         State(f'outcome-checkboxes_{suffix}', 'value'),
         State(f'country-checkboxes_{suffix}', 'value')]
    )
    def update_figures(click, genders, age_range, outcomes, countries):
        filtered_df = df_map[(
            (df_map['slider_sex'].isin(genders)) &
            (df_map['age'] >= age_range[0]) &
            (df_map['age'] <= age_range[1]) &
            (df_map['outcome'].isin(outcomes)) &
            (df_map['country_iso'].isin(countries)))]

        if filtered_df.empty:
            visuals = None
        else:
            visuals = create_visuals(filtered_df)
        return visuals

    # End of callbacks
    return
