import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import IsaricDraw as idw
import pandas as pd
import plotly.graph_objs as go
import pycountry
from dash.dependencies import Input, Output, State
import dash
import numpy as np
import IsaricDraw as idw
import IsaricAnalytics as ia
import getREDCapData as getRC


############################################
############################################
## Data reading and initial proccesing 
############################################
############################################

suffix='risk_factors'

countries = [{'label': country.name, 'value': country.alpha_3} for country in pycountry.countries]

#df_map=pd.read_csv('Vertex Dashboard/assets/data/map.csv')
df_map=getRC.read_data_from_REDCAP()
df_map=df_map.dropna()
df_map_count=df_map[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()
unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates().sort_values(by='slider_country')
country_dropdown_options = [{'label': row['slider_country'], 'value': row['country_iso']}
                    for index, row in unique_countries.iterrows()]
bins = [0, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '91-95', '96-100']
df_map['age_group'] = pd.cut(df_map['age'], bins=bins, labels=labels, right=False)
df_map['mapped_outcome'] = df_map['outcome']



############################################
############################################
## Modal creation
############################################
############################################


def create_modal():
    linegraph_instructions = html.Div([
        html.Div("1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster)"),
        html.Br(),
        html.Div("2. Change datasets using the dropdown (country selections are remembered)"),
        html.Br(),
        html.Div("3. Hover mouse on chart for tooltip data "),
        html.Br(),
        html.Div("4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it."),
        html.Br(),
        html.Div("5. Toggle selected countries on/off by clicking on the legend (far right)"),
        html.Br(),
        html.Div("6. Download button will export all countries and available years for the selected dataset"),    
    ])   


    color_map = {'Discharge': '#00C26F', 'Censored': '#FFF500', 'Death': '#DF0069'}

    ## Proccessing

    
    
    risk_df=ia.risk_preprocessing(df_map)

    feature_importances,accuracy,roc,optimal_threshold,combined_df=ia.binary_model1(risk_df,risk_df.columns,'outcome')
    
    categorical_results, suitable_cat, categorical_results_t=ia.categorical_feature_outcome(risk_df,'outcome')
    fig_table_categorical=idw.table(categorical_results)

    results, suitable_num, results_t =ia.numeric_outcome_results(risk_df,'outcome')
    print(results)
    fig_table_numerical=idw.table(results)
    
    fig_table_symp=idw.fig_placeholder()
    modal = [
        dbc.ModalHeader(html.H3("Clinical Features", id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),  

        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Filters and Controls",  
                    children=[idw.filters_controls(suffix,country_dropdown_options)]
                ),                
                dbc.AccordionItem(
                    title="Insights",  
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([dbc.Col([fig_table_categorical],id='fig_table_cat_risk')]), label='Categorical variables description by outcome'),
                            dbc.Tab(dbc.Row([dbc.Col([fig_table_numerical],id='fig_table_num_risk')]), label='Continuous variables description by outcome'),
                            ##Add more tabs if needed
                        ])
                    ]
                )
            ])
        ], style={ 'overflowY': 'auto','minHeight': '75vh','maxHeight': '75vh'}),

        idw.ModalFooter(suffix,linegraph_instructions,linegraph_instructions),
    ]
    return modal    


############################################
############################################
## Callbacks
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
    def update_country_selection(select_all_value, selected_countries, all_countries_options):
        ctx = dash.callback_context

        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == f'country-selectall_{suffix}':
            if 'all' in select_all_value:
                # "Select all" (now "Unselect all") is checked
                return [[option['value'] for option in all_countries_options], [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # "Unselect all" is unchecked
                return [[], [{'label': 'Select all', 'value': 'all'}], []]

        elif trigger_id == f'country-checkboxes_{suffix}':
            if len(selected_countries) == len(all_countries_options):
                # All countries are selected manually
                return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # Some countries are deselected
                return [selected_countries, [{'label': 'Select all', 'value': 'all'}], []]

        return [selected_countries, [{'label': 'Select all', 'value': 'all'}], select_all_value]

    @app.callback(
        Output(f"country-fade_{suffix}", "is_in"),
        [Input(f"country-display_{suffix}", "n_clicks")],
        [State(f"country-fade_{suffix}", "is_in")]
    )
    def toggle_fade(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    @app.callback(
        Output(f'country-display_{suffix}', 'children'),
        [Input(f'country-checkboxes_{suffix}', 'value')],
        [State(f'country-checkboxes_{suffix}', 'options')]
    )
    def update_country_display(selected_values, all_options):
        if not selected_values:
            return "Country:"

        # Create a dictionary to map values to labels
        value_label_map = {option['value']: option['label'] for option in all_options}

        # Build the display string
        selected_labels = [value_label_map[val] for val in selected_values if val in value_label_map]
        display_text = ", ".join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            return f"Country: {selected_labels[0]}, +{len(selected_labels) - 1} more..."
        else:
            return f"Country: {display_text}"


    ############################################
    ############################################
    ## Specific Callbacks
    ## Modify outputs
    ############################################
    ############################################

    @app.callback(
        [Output('fig_table_cat_risk', 'children'),
         Output('fig_table_num_risk', 'children')],
        [Input(f'submit-button_{suffix}', 'n_clicks')],
        [State(f'gender-checkboxes_{suffix}', 'value'),
         State(f'age-slider_{suffix}', 'value'),
         State(f'outcome-checkboxes_{suffix}', 'value'),
         State(f'country-checkboxes_{suffix}', 'value')]
    )
    def update_figures(click, genders, age_range, outcomes, countries):
        filtered_df = df_map[
                        (df_map['slider_sex'].isin(genders))& 
                        (df_map['age'] >= age_range[0]) & 
                        (df_map['age'] <= age_range[1]) & 
                        (df_map['outcome'].isin(outcomes)) &
                        (df_map['country_iso'].isin(countries)) ]
        print(len(filtered_df))

        if filtered_df.empty:

            return None
        df_age_gender=filtered_df[['age_group','usubjid','mapped_outcome','slider_sex']].groupby(['age_group','mapped_outcome','slider_sex']).count().reset_index()
        df_age_gender.rename(columns={'slider_sex': 'side', 'mapped_outcome': 'stack_group', 'usubjid': 'value', 'age_group': 'y_axis'}, inplace=True)
        print(len(df_age_gender))
        color_map = {'Discharge': '#00C26F', 'Censored': '#FFF500', 'Death': '#DF0069'}


        #categorical_results, suitable_cat, categorical_results_t=ia.categorical_feature_outcome(filtered_df,'outcome')
        #fig_table_categorical=idw.table(categorical_results)

        #results, suitable_num, results_t  =ia.numeric_outcome_results(filtered_df,'outcome')
        #fig_table_numerical=idw.table(results)
        #fig_table_categorical=idw.table(pd.DataFrame(data=[[1,2,3],[4,5,6]],columns=list('abc')))
        risk_df=ia.risk_preprocessing(filtered_df)
        categorical_results, suitable_cat, categorical_results_t=ia.categorical_feature_outcome(risk_df,'outcome')
        fig_table_categorical=idw.table(categorical_results)

        results, suitable_num, results_t  =ia.numeric_outcome_results(risk_df,'outcome')
        fig_table_numerical=idw.table(results)
        return [fig_table_categorical,fig_table_numerical]

