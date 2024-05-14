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

countries = [{'label': country.name, 'value': country.alpha_3} for country in pycountry.countries]

df_comor=pd.read_csv('assets/data/comorbidities.csv')
df_comor_count=df_comor[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()
unique_countries = df_comor[['slider_country', 'country_iso']].drop_duplicates().sort_values(by='slider_country')
country_dropdown_options = [{'label': row['slider_country'], 'value': row['country_iso']}
                    for index, row in unique_countries.iterrows()]
comorbidities=[
'comorbid_asthma',
'comorbid_chronic_cardiac_disease',
'comorbid_chronic_haematological_disease',
'comorbid_chronic_kidney_disease',
'comorbid_chronic_neurological_disorder',
'comorbid_chronic_pulmonary_disease',
'comorbid_dementia',
'comorbid_diabetes',
'comorbid_hypertension',
'comorbid_immunosuppression',
'comorbid_liver_disease',
'comorbid_malignant_neoplasm',
'comorbid_malnutrition',
'comorbid_obesity',
'comorbid_other',
'comorbid_rare_diseases_and_inborn_errors_of_metabolism',
'comorbid_rheumatologic_disorder',
'comorbid_smoking',
'comorbid_transplantation',
'comorbid_tuberculosis',
'comorbid_pregnancy']
proportions = df_comor[comorbidities].apply(lambda x: x.sum() / x.count()).reset_index()
proportions.columns=['Condition', 'Proportion']
proportions=proportions.sort_values(by=['Proportion'], ascending=False)
Condition_top=proportions['Condition'].head(5)
set_data=df_comor[Condition_top]

symptoms=['symptoms_abdominal_pain',
'symptoms_altered_consciousness_confusion',
'symptoms_asymptomatic',
'symptoms_bleeding',
'symptoms_chest_pain',
'symptoms_conjunctivitis',
'symptoms_cough',
'symptoms_diarrhoea',
'symptoms_ear_pain',
'symptoms_fatigue_malaise',
'symptoms_headache',
'symptoms_history_of_fever',
'symptoms_lost_altered_sense_of_smell',
'symptoms_lost_altered_sense_of_taste',
'symptoms_lymphadenopathy',
'symptoms_muscle_aches_joint_pain',
'symptoms_runny_nose',
'symptoms_seizures',
'symptoms_severe_dehydration',
'symptoms_shortness_of_breath',
'symptoms_skin_rash',
'symptoms_sore_throat',
'symptoms_vomiting_nausea',
'symptoms_wheezing']
df_symptoms=pd.read_csv('assets/data/symptoms.csv')
proportions_symp = df_symptoms[symptoms].apply(lambda x: x.sum() / x.count()).reset_index()
proportions_symp.columns=['Condition', 'Proportion']
proportions_symp=proportions_symp.sort_values(by=['Proportion'], ascending=False)
Condition_symp_top=proportions_symp['Condition'].head(5)
set_data_symp=df_symptoms[Condition_symp_top]

def filters_controls():
    row = dbc.Row([
        dbc.Col([html.H6("Gender:", style={'margin-right': '10px'}),
            html.Div([
                
                dcc.Checklist(
                    id='gender-checkboxes_sc',
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
                        id='age-slider_sc',
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

                    html.Div(id="country-display_sc", children="Country:", style={"cursor": "pointer"}),
                    dbc.Fade(
                        html.Div([
                            dcc.Checklist(
                                id='country-selectall_sc',
                                options=[{'label': 'Select all', 'value': 'all'}],
                                value=['all']
                            ),
                            dcc.Checklist(
                                id='country-checkboxes_sc',
                                options=country_dropdown_options,
                                value=[option['value'] for option in country_dropdown_options],
                                style={'overflowY': 'auto', 'maxHeight': '100px'}
                            )
                        ]),
                        id="country-fade_sc",
                        is_in=False,
                        appear=False,
                    )
                ]),
            
        ], width=5),

        dbc.Col([html.H6("Outcome:", style={'margin-right': '10px'}),
            html.Div([
                
                dcc.Checklist(
                    id='outcome-checkboxes_sc',
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


def create_symptoms_comorbidities_modal():
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

   
    upset_plot = idw.upset(set_data, title='Frequency of combinations of the five most common comorbidities')
    upset_plot_symp = idw.upset(set_data_symp, title='Frequency of combinations of the five most common symptoms')
    #color_map = {'Yes': '#00C26F'}
    freq_chart = idw.frequency_chart(proportions, title='Frequency of comorbidities Seen at Admission Amongst COVID-19 Patients')
    freq_chart_sympt = idw.frequency_chart(proportions_symp, title='Frequency of symptoms Seen at Admission Amongst COVID-19 Patients')
    modal = [
        dbc.ModalHeader(html.H3("Symptoms and Comorbidities", id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),  

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
                            dbc.Tab(dbc.Row([dbc.Col(freq_chart,id='freqComor_chart_col')]), label='Frequency of comorbidities'),
                            dbc.Tab(dbc.Row([dbc.Col(upset_plot,id='upsetComor_chart_col')]), label='Combination of comorbidities'),
                            dbc.Tab(dbc.Row([dbc.Col(freq_chart_sympt,id='freqSympt_chart_col')]), label='Frequency of symptoms'),
                            dbc.Tab(dbc.Row([dbc.Col(upset_plot_symp,id='upsetSympt_chart_col')]), label='Symptoms of comorbidities'),
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
        [Output('country-checkboxes_sc', 'value'),
        Output('country-selectall_sc', 'options'),
        Output('country-selectall_sc', 'value')],
        [Input('country-selectall_sc', 'value'),
        Input('country-checkboxes_sc', 'value')],
        [State('country-checkboxes_sc', 'options')]
    )
    def update_country_selection_sc(select_all_value, selected_countries, all_countries_options):
        ctx = dash.callback_context
        
        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'country-selectall_sc':
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
        Output("country-fade_sc", "is_in"),
        [Input("country-display_sc", "n_clicks")],
        [State("country-fade_sc", "is_in")]
    )
    def toggle_fade_sc(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in
    @app.callback(
        Output('country-display_sc', 'children'),
        [Input('country-checkboxes_sc', 'value')],
        [State('country-checkboxes_sc', 'options')]
    )

    def update_country_display_sc(selected_values, all_options):
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
        [Output('freqComor_chart_col', 'children'),
         Output('upsetComor_chart_col', 'children'),
         Output('freqSympt_chart_col', 'children'),
         Output('upsetSympt_chart_col', 'children')    ],
        [
            Input('gender-checkboxes_sc', 'value'),
            Input('age-slider_sc', 'value'),
            Input('outcome-checkboxes_sc', 'value'),
            Input('country-checkboxes_sc', 'value')
        ]
    )
    def update_figures_sc(genders, age_range, outcomes,countries):

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

        filtered_df = df_comor[(df_comor['slider_sex'].isin(genders))& 
                        (df_comor['age'] >= age_range[0]) & 
                        (df_comor['age'] <= age_range[1]) & 
                        (df_comor['outcome'].isin(df_outcomes)) &
                        (df_comor['country_iso'].isin(countries)) ]
        filtered_df_sym = df_symptoms[(df_symptoms['slider_sex'].isin(genders))& 
                        (df_symptoms['age'] >= age_range[0]) & 
                        (df_symptoms['age'] <= age_range[1]) & 
                        (df_symptoms['outcome'].isin(df_outcomes)) &
                        (df_symptoms['country_iso'].isin(countries)) ]    

        if filtered_df.empty:
            
            return None
        proportions = filtered_df[comorbidities].apply(lambda x: x.sum() / x.count()).reset_index()
        proportions.columns=['Condition', 'Proportion']
        proportions=proportions.sort_values(by=['Proportion'], ascending=False)
        Condition_top=proportions['Condition'].head(5)
        set_data=filtered_df[Condition_top]
        upset_plot = idw.upset(set_data, title='Frequency of combinations of the five most common comorbidities')
        freq_chart = idw.frequency_chart(proportions, title='Frequency of Comorbidities Seen at Admission Amongst COVID-19 Patients')

        proportions_symp = filtered_df_sym[symptoms].apply(lambda x: x.sum() / x.count()).reset_index()
        proportions_symp.columns=['Condition', 'Proportion']
        proportions_symp=proportions_symp.sort_values(by=['Proportion'], ascending=False)
        Condition_symp_top=proportions_symp['Condition'].head(5)
        set_data_symp=filtered_df_sym[Condition_symp_top]        
        upset_plot_sympt = idw.upset(set_data_symp, title='Frequency of combinations of the five most common symptoms')
        freq_chart_sympt = idw.frequency_chart(proportions_symp, title='Frequency of symptoms Seen at Admission Amongst COVID-19 Patients')

        return [freq_chart,upset_plot,freq_chart_sympt,upset_plot_sympt]





########################################################################
#######################################################################