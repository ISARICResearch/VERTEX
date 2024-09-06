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
import redcap_config as rc_config

############################################
############################################
## Data reading and initial proccesing 
############################################
############################################

suffix='risk_factors'


############################################
############################################
## Data reading and initial proccesing 
############################################
############################################

countries = [{'label': country.name, 'value': country.alpha_3} for country in pycountry.countries]

#df_map=pd.read_csv('Vertex Dashboard/assets/data/map.csv')
#df_map=getRC.read_data_from_REDCAP()

###########################
site_mapping=rc_config.site_mapping
redcap_api_key=rc_config.redcap_api_key
redcap_url=rc_config.redcap_url


sections=getRC.getDataSections(redcap_api_key)

vari_list=getRC.getVariableList(redcap_api_key,['dates','demog','comor','daily','outco','labs','vital','adsym','inter','treat'])

RF_vari_list=['outco_outcome']+getRC.getVariableList(redcap_api_key,['demog','comor','labs','vital','adsym'])

#df_map=getRC.get_REDCAP_Single_DB(redcap_url, apis_dengue,site_mapping,vari_list)
df_map=getRC.get_REDCAP_Single_DB(redcap_url, redcap_api_key,site_mapping,vari_list)

#Aqui
'''
df_RF=df_map[['age','slider_sex','income','outcome']+list(set(RF_vari_list).intersection(set(df_map.columns)))]

data=ia.remove_columns(df_RF,limit_var=90)
data=ia.num_imputation_nn(data)
coef_df,roc_auc,best_C=ia.lasso_rf(data,outcome_var='outcome')
'''

#########################################
########################################

data = {
    'Feature': ['Age', 'Diabetes', 'Hypertension', 'Cardiovascular_Disease', 
                'Chronic_Respiratory_Disease', 'Cancer', 'Renal_Disease', 'Obesity'],
    'Coefficient': [0.012, 1.235, 1.432, 1.765, 0.812, 1.125, 1.432, 0.589],
    'Odds Ratio': [1.012, 3.439, 4.187, 5.843, 2.253, 3.080, 4.187, 1.802],
    'CI Lower 95%': [1.002, 2.453, 3.276, 4.192, 1.678, 2.143, 3.276, 1.307],
    'CI Upper 95%': [1.023, 4.832, 5.360, 8.142, 3.024, 4.424, 5.360, 2.482],
    'P-value': ['0.041', '<0.005', '<0.005', '<0.005', '0.027', '<0.005', '<0.005', '0.049']
}

# Convert to DataFrame
coef_df = pd.DataFrame(data)

fig_table_coef_df=idw.table(coef_df)
########################################

##############

###########################

#df_map=getRC.get_REDCAP_Single_DB(redcap_url, redcap_api_key,site_mapping,requiered_variables)
#df_map=df_map.dropna()
df_map_count=df_map[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()
unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates().sort_values(by='slider_country')
country_dropdown_options=[]
for uniq_county in range(len(unique_countries)):
    name_country=unique_countries['slider_country'].iloc[uniq_county]
    code_country=unique_countries['country_iso'].iloc[uniq_county]
    country_dropdown_options.append({'label': name_country, 'value': code_country})
bins = [0, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '91-95', '96-100']
df_map['age_group'] = pd.cut(df_map['age'], bins=bins, labels=labels, right=False)


df_map['mapped_outcome'] = df_map['outcome']

df_age_gender=df_map[['age_group','usubjid','mapped_outcome','slider_sex']].groupby(['age_group','mapped_outcome','slider_sex']).count().reset_index()
df_age_gender.rename(columns={'slider_sex': 'side', 'mapped_outcome': 'stack_group', 'usubjid': 'value', 'age_group': 'y_axis'}, inplace=True)



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
                            #dbc.Tab(dbc.Row([dbc.Col([fig_table_categorical],id='fig_table_cat_risk')]), label='Categorical variables description by outcome'),
                            #dbc.Tab(dbc.Row([dbc.Col([fig_table_numerical],id='fig_table_num_risk')]), label='Continuous variables description by outcome'),
                            dbc.Tab(dbc.Row([dbc.Col([fig_table_coef_df],id='table_symo')]), label='Risk Factors'),
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

