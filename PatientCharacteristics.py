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

df_age_gender=df_map[['age_group','usubjid','mapped_outcome','slider_sex']].groupby(['age_group','mapped_outcome','slider_sex']).count().reset_index()
df_age_gender.rename(columns={'slider_sex': 'side', 'mapped_outcome': 'stack_group', 'usubjid': 'value', 'age_group': 'y_axis'}, inplace=True)

'''df_epiweek=df_map[['mapped_outcome','epiweek.admit','usubjid']].groupby(['mapped_outcome','epiweek.admit']).count().reset_index()
#df_epiweek['epiweek.admit']=np.round(df_epiweek['epiweek.admit']).astype('str')
df_epiweek.rename(columns={'mapped_outcome': 'stack_group', 'epiweek.admit': 'timepoint', 'usubjid': 'value'}, inplace=True)
df_epiweek=df_epiweek.dropna()

df_los=df_map[['age','slider_sex','dur_ho']].sample(5000)
df_los.rename(columns={'dur_ho': 'length of hospital stay', 'age': 'age', 'slider_sex': 'sex'}, inplace=True)'''


#proportions_comorbidities, set_data_comorbidities = ia.get_proportions(df_map,'comorbidities')

############################################
############################################
## Modal creation
############################################
############################################


def create_patient_characteristics_modal():
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
    pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='age_gender_pyramid_chart')

    proportions_symptoms, set_data_symptoms = ia.get_proportions(df_map,'symptoms')
    freq_chart_sympt = idw.frequency_chart(proportions_symptoms, title='Frequency of signs and symptoms on presentation')
    upset_plot_sympt = idw.upset(set_data_symptoms, title='Frequency of combinations of the five most common signs or symptoms')

    descriptive = ia.descriptive_table(ia.obtain_variables(df_map, 'symptoms'))
    fig_table_symp=idw.table(descriptive)


    
    #cumulative_chart = idw.cumulative_bar_chart(df_epiweek, title='Cumulative Patient Outcomes by Timepoint', base_color_map=color_map, graph_id='my-cumulative-chart')
    np.random.seed(0)

    # Generate data
    ages = np.random.randint(0, 100, size=100)  # 100 random ages between 0 and 99
    sexes = np.random.choice(['M', 'F'], size=100)  # 100 random sex assignments
    lengths_of_stay = np.random.randint(1, 30, size=100)  # 100 random lengths of stay between 1 and 29 days

    '''# Create DataFrame
    df = pd.DataFrame({
        'age': ages,
        'sex': sexes,
        'length of hospital stay': lengths_of_stay
    })
    color_map = {'Female': '#750AC8', 'Male': '#00C279'}
    boxplot_graph = idw.age_group_boxplot(df_los, base_color_map=color_map,label='Length of hospital stay')
    sex_boxplot_graph = idw.sex_boxplot(df_los, base_color_map=color_map,label='Length of hospital stay')'''
    modal = [
        dbc.ModalHeader(html.H3("Clinical Features", id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),  

        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Filters and Controls",  
                    children=[idw.filters_controls('pc',country_dropdown_options)]
                ),                
                dbc.AccordionItem(
                    title="Insights",  
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([dbc.Col([fig_table_symp],id='table_symo')]), label='Descriptive table'),
                            dbc.Tab(dbc.Row([dbc.Col(pyramid_chart,id='pyramid-chart-col')]), label='Age and Sex'),
                            dbc.Tab(dbc.Row([dbc.Col(freq_chart_sympt,id='freqSympt_chart')]), label='Signs and symptoms on presentation: Frequency'),
                            dbc.Tab(dbc.Row([dbc.Col(upset_plot_sympt,id='upsetSympt_chart')]), label='Signs and symptoms on presentation:Intersections'),
                            #dbc.Tab(dbc.Row([dbc.Col(boxplot_graph,id='boxplot_graph-col')]), label='Length of hospital stay by age group'),
                        ])
                    ]
                )
            ])
        ], style={ 'overflowY': 'auto','minHeight': '75vh','maxHeight': '75vh'}),

        idw.ModalFooter('pc',linegraph_instructions,linegraph_instructions)


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
        [Output('pyramid-chart-col', 'children'),
         Output('freqSympt_chart', 'children'),
         Output('upsetSympt_chart', 'children')],
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
        pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='age_gender_pyramid_chart')

        proportions_symptoms, set_data_symptoms = ia.get_proportions(filtered_df,'symptoms')
        freq_chart_sympt = idw.frequency_chart(proportions_symptoms, title='Frequency of signs and symptoms on presentation')
        upset_plot_sympt = idw.upset(set_data_symptoms, title='Frequency of combinations of the five most common signs or symptoms')
        return [pyramid_chart,freq_chart_sympt,upset_plot_sympt]

