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

'''outcome_mapping = {
    'Discharge': ['discharge', 'cured (confirmed by a negative covid test)', 
                'released with home care', 'released without instructions', 
                'recovery (confirmed by a negative test)', 
                'recovered (confirmed by negative covid-19 test)', 'released'],
    'Death': ['death'],
    'Censored':['ongoing care', np.nan, 'transferred', 
                'unknown outcome', 'lost to follow-up',
                'moved to facility', 'unreachable by phone', 'ran away/unknown']
}
inverted_mapping = {outcome: category for category, outcomes in outcome_mapping.items() for outcome in outcomes}
df_map['mapped_outcome'] = df_map['outcome'].apply(lambda x: inverted_mapping.get(x, 'Other'))'''
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



def filters_controls():
    row = dbc.Row([
        dbc.Col([html.H6("Gender:", style={'margin-right': '10px'}),
            html.Div([
                
                dcc.Checklist(
                    id='gender-checkboxes_pc',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Unknown', 'value': 'U'}
                    ],
                    value=['Male', 'Female', 'U']
                )
            ])
        ], width=2),
        
        dbc.Col([html.H6("Age:", style={'margin-right': '10px'}),
            html.Div([
                
                html.Div([
                    dcc.RangeSlider(
                        id='age-slider_pc',
                        min=0,
                        max=90,
                        step=10,
                        marks={i: str(i) for i in range(0, 91, 10)},
                        value=[0, 90]
                    )
                ], style={'width': '100%'})  # Apply style to this div
            ])
        ], width=3),

        dbc.Col([html.H6("Country:", style={'margin-right': '10px'}),
                html.Div([

                    html.Div(id="country-display_pc", children="Country:", style={"cursor": "pointer"}),
                    dbc.Fade(
                        html.Div([
                            dcc.Checklist(
                                id='country-selectall_pc',
                                options=[{'label': 'Select all', 'value': 'all'}],
                                value=['all']
                            ),
                            dcc.Checklist(
                                id='country-checkboxes_pc',
                                options=country_dropdown_options,
                                value=[option['value'] for option in country_dropdown_options],
                                style={'overflowY': 'auto', 'maxHeight': '100px'}
                            )
                        ]),
                        id="country-fade_pc",
                        is_in=False,
                        appear=False,
                    )
                ]),
            
        ], width=5),

        dbc.Col([html.H6("Outcome:", style={'margin-right': '10px'}),
            html.Div([
                
                dcc.Checklist(
                    id='outcome-checkboxes_pc',
                    options=[
                        {'label': 'Death', 'value': 'Death'},
                        {'label': 'Censored', 'value': 'Censored'},
                        {'label': 'Discharge', 'value': 'Discharge'}
                    ],
                    value=['Death', 'Censored', 'Discharge']
                )
            ])
        ], width=2)
    ])
    return row


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
                    children=[filters_controls()]
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

        dbc.ModalFooter([

            
            html.Div([
                dbc.Button("About", id='modal_patChar_about_popover', color='info', size='sm',style={'margin-right': '5px'}),
                dbc.Button("Instructions", id='modal_patChar_instruc_popover',  size='sm',style={'margin-right': '5px'}),
                dbc.Button("Download", id='modal_patChar_download_popover', size='sm',style={'margin-right': '5px'}),
                #dbc.Button("Close", id="modal_patChar_close_popover",  size='sm')
            ], className='ml-auto'),
                

                                
            dbc.Popover(
                [
                    dbc.PopoverHeader("Instructions", style={'fontWeight':'bold'}),
                    dbc.PopoverBody(linegraph_instructions)
                ],
                #id="modal-line-instructions-popover",
                #is_open=False,
                target="modal_patChar_instruc_popover",
                trigger="hover",
                placement="top",
                hide_arrow=False,
                #style={"zIndex":1}
            ),
            
            dbc.Popover(
                [
                    dbc.PopoverHeader("About", style={'fontWeight':'bold'}),
                    dbc.PopoverBody(linegraph_instructions),
                ],
                #id="modal-line-guide-popover",
                #is_open=False,
                target="modal_patChar_about_popover",
                trigger="hover",
                placement="top",
                hide_arrow=False,                        
                #style={"zIndex":1}
            ),
            
            dbc.Popover(
                    [
                    dbc.PopoverHeader("Download", style={'fontWeight':'bold'}),
                    dbc.PopoverBody([          
                        html.Div('Raw data'),
                        dbc.Button(".xlsx", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-xls", style={}, size='sm'),
                        dbc.Button(".csv", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-csv", style={}, size='sm'),
                        dbc.Button(".json", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-json", style={}, size='sm'),
                        html.Div('Chart', style={'marginTop':5}),
                        dbc.Button(".pdf", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-pdf", style={}, size='sm'),
                        dbc.Button(".jpg", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-jpg", style={}, size='sm'),
                        dbc.Button(".png", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-png", style={}, size='sm'),
                        dbc.Button(".svg", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-svg", style={}, size='sm'),
                        html.Div('Advanced', style={'marginTop':5,'display':'none'}),
                        dbc.Button("Downloads Area", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-land", style={'display':'none'}, size='sm'),
                        ]),
                    ],
                    id="download-popover-patChar",                                        
                    target="modal_patChar_download_popover",
                    #style={'maxHeight': '300px', 'overflowY': 'auto'},
                    trigger="legacy",
                    placement="top",
                    hide_arrow=False,
                    
            ), 
            
        ]),
    ]
    return modal    



########################################################################
########################################################################
########## Patient characteristic 
def register_callbacks(app):
    @app.callback(
        [Output('country-checkboxes_pc', 'value'),
        Output('country-selectall_pc', 'options'),
        Output('country-selectall_pc', 'value')],
        [Input('country-selectall_pc', 'value'),
        Input('country-checkboxes_pc', 'value')],
        [State('country-checkboxes_pc', 'options')]
    )
    def update_country_selection_pc(select_all_value, selected_countries, all_countries_options):
        ctx = dash.callback_context
        
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'country-selectall_pc':
            if 'all' in select_all_value:
                # "Select all" (now "Unselect all") is checked
                return [[option['value'] for option in all_countries_options], [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # "Unselect all" is unchecked
                return [[], [{'label': 'Select all', 'value': 'all'}], []]

        elif trigger_id == 'country-checkboxes':
            if len(selected_countries) == len(all_countries_options):
                # All countries are selected manually
                return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # Some countries are deselected
                return [selected_countries, [{'label': 'Select all', 'value': 'all'}], []]

        return [selected_countries, [{'label': 'Select all', 'value': 'all'}], select_all_value]

    @app.callback(
        Output("country-fade_pc", "is_in"),
        [Input("country-display_pc", "n_clicks")],
        [State("country-fade_pc", "is_in")]
    )
    def toggle_fade_pc(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    
    @app.callback(
        Output('country-display_pc', 'children'),
        [Input('country-checkboxes_pc', 'value')],
        [State('country-checkboxes_pc', 'options')]
    )
    def update_country_display_pc(selected_values, all_options):
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

    
    @app.callback(
        [Output('pyramid-chart-col', 'children'),Output('freqSympt_chart', 'children'),Output('upsetSympt_chart', 'children')],
        [
            Input('gender-checkboxes_pc', 'value'),
            Input('age-slider_pc', 'value'),
            Input('outcome-checkboxes_pc', 'value'),
            Input('country-checkboxes_pc', 'value')
        ]
    )
    def update_figures(genders, age_range, outcomes,countries):



        filtered_df = df_map[
                        #(df_map['slider_sex'].isin(genders))& 
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
    '''
    @app.callback(
        [Output('pyramid-chart-col', 'children'),
         #Output('cumulative_chart-col', 'children'), 
         #Output('sex_boxplot_graph-col', 'children'),
         #Output('boxplot_graph-col', 'children'),    
        ],
        [
            Input('gender-checkboxes_pc', 'value'),
            Input('age-slider_pc', 'value'),
            Input('outcome-checkboxes_pc', 'value'),
            Input('country-checkboxes_pc', 'value')
        ]
    )
    def update_figures(genders, age_range, outcomes,countries):
        print('entro aqui')

        outcome_mapping = {
            'Discharge': ['discharge', 'cured (confirmed by a negative covid test)', 
                        'released with home care', 'released without instructions', 
                        'recovery (confirmed by a negative test)', 
                        'recovered (confirmed by negative covid-19 test)', 'released'],
            'Death': ['death'],
            'Censored':['ongoing care', np.nan, 'transferred', 
                        'unknown outcome', 'lost to follow-up',
                        'moved to facility', 'unreachable by phone', 'ran away/unknown']
            # Add other mappings here
        }    
        df_outcomes = []
        for outcome in outcomes:
            df_outcomes.extend(outcome_mapping.get(outcome, []))

        filtered_df = df_map[(df_map['slider_sex'].isin(genders))& 
                        (df_map['age'] >= age_range[0]) & 
                        (df_map['age'] <= age_range[1]) & 
                        (df_map['outcome'].isin(df_outcomes)) &
                        (df_map['country_iso'].isin(countries)) ]
        print(len(filtered_df))

        if filtered_df.empty:
            
            return None
        df_age_gender=filtered_df[['age_group','usubjid','mapped_outcome','slider_sex']].groupby(['age_group','mapped_outcome','slider_sex']).count().reset_index()
        df_age_gender.rename(columns={'slider_sex': 'side', 'mapped_outcome': 'stack_group', 'usubjid': 'value', 'age_group': 'y_axis'}, inplace=True)
        print(len(df_age_gender))
        color_map = {'Discharge': '#00C26F', 'Censored': '#FFF500', 'Death': '#DF0069'}
        pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='age_gender_pyramid_chart')
        return [pyramid_chart]
        '''
        
        #df_epiweek=filtered_df[['mapped_outcome','epiweek.admit','usubjid']].groupby(['mapped_outcome','epiweek.admit']).count().reset_index()
#        df_epiweek.rename(columns={'mapped_outcome': 'stack_group', 'epiweek.admit': 'timepoint', 'usubjid': 'value'}, inplace=True)


#        cumulative_chart = idw.cumulative_bar_chart(df_epiweek, title='Cumulative Patient Outcomes by Timepoint', base_color_map=color_map, graph_id='my-cumulative-chart')

#        df_los=filtered_df[['age','slider_sex','dur_ho']].sample(5000)
#        df_los.rename(columns={'dur_ho': 'length of hospital stay', 'age': 'age', 'slider_sex': 'sex'}, inplace=True)
#        color_map = {'Female': '#750AC8', 'Male': '#00C279'}
#        boxplot_graph = idw.age_group_boxplot(df_los, base_color_map=color_map,label='Length of hospital stay')
#        sex_boxplot_graph = idw.sex_boxplot(df_los, base_color_map=color_map,label='Length of hospital stay')
        #return [pyramid_chart,cumulative_chart,sex_boxplot_graph,boxplot_graph]
        


########################################################################
#######################################################################
