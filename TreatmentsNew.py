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

site_mapping=rc_config.site_mapping
redcap_api_key=rc_config.redcap_api_key
redcap_url=rc_config.redcap_url

requiered_variables=['subjid',
'dates_enrolment',
'dates_onsetdate',
'dates_adm',
'dates_admdate',
'demog_sex',
'demog_age',
'demog_age_units',
'demog_height_cm',
'demog_weight_kg',
'demog_occupation',
'demog_occupation_oth',
'demog_residence',
'demog_residence_oth',
'preg_pregnant',
'comor_chrcardiac',
'comor_hypertensi',
'comor_chrpulmona',
'comor_asthma',
'comor_chrkidney',
'comor_obesity',
'comor_liverdisea_gnrl',
'comor_hepbc',
'comor_asplenia',
'comor_chrneurolo',
'comor_malignantn',
'comor_chrhematol',
'comor_rheumatolo',
'comor_aids',
'comor_aids_cd4',
'comor_diabetes',
'comor_hba1c',
'comor_hba1c_units',
'comor_dementia',
'comor_tuberculos',
'comor_malnutriti',
'comor_smoking',
'adsym_feverplus',
'adsym_cough',
'adsym_shortbreat',
'adsym_headache',
'adsym_retroorbit',
'adsym_seizconv',
'adsym_restlessne',
'adsym_fatigue',
'adsym_myalgia',
'adsym_arthralgia',
'adsym_abdpain',
'adsym_diarrhoea',
'adsym_vomit',
'adsym_vomit2d',
'adsym_anorexia',
'adsym_skinrash',
'adsym_haemorrhag_yn',
'daily_date',
'daily_care',
'vital_highesttem',
'vital_highesttem_units',
'vital_hr',
'vital_rr',
'vital_systolicbp',
'vital_diastolicbp',
'vital_spo2',
'vital_fio2low',
'vital_fio2spo2',
'vital_fio2spo2_units',
'vital_capillaryr',
'vital_avpu',
'vital_gcs',
'vital_urineflow',
'daily_datalab',
'labs_haemo',
'labs_haemo_units',
'labs_wbccount',
'labs_lymphocyte',
'labs_lymphocyte_units',
'labs_neutrophil',
'labs_neutrophil_units',
'labs_hematocrit',
'labs_hematocrit_units',
'labs_platelets_109l',
'labs_aptt',
'labs_aptr',
'labs_prothrombin_sec',
'labs_tqinr',
'labs_altsgpt',
'labs_bilirubin',
'labs_bilirubin_units',
'labs_astsgot',
'labs_glucose_mmoll',
'labs_ggt',
'labs_ureanitro',
'labs_ureanitro_units',
'labs_lactate',
'labs_lactate_units',
'labs_creatinine',
'labs_creatinine_units',
'labs_sodium_mmoll',
'labs_potassium_mmoll',
'labs_procalcito_ngml',
'labs_crp_mgl',
'labs_ldh_ul',
'labs_creatineki_ul',
'labs_tropi',
'labs_tropi_units',
'labs_ddimer_mgl',
'labs_ferritin_ngml',
'labs_il6_pgml',
'labs_fibrinogen',
'labs_fibrinogen_units',
'labs_albumin_gl',
'labs_protein_gl',
'labs_paco2',
'labs_paco2_units',
'labs_ph',
'labs_hco3',
'labs_hco3_units',
'labs_baseexcess',
'outco_date',
'outco_outcome',
'treat_agents_acting_on_the_renin_angiotensin_system',
'treat_antibiotic_agents',
'treat_antifungal_agents',
'treat_antiinflammatory',
'treat_antimalarial_agents',
'treat_antiviral_agents',
'treat_cardiopulmonary_resuscitation',
'treat_cardiovascular_support',
'treat_colchicine',
'treat_convalescent_plasma',
'treat_corticosteroids',
'treat_experimental_agents',
'treat_high_flow_nasal_cannula',
'treat_immunoglobuli',
'treat_immunostimulants',
'treat_immunosuppressants',
'treat_inotropes_vasopressors',
'treat_interleukin_inhibitors',
'treat_invasive_ventilation',
'treat_non_invasive_ventilation',
'treat_off_label_compassionate_use_medications',
'treat_other_interventions',
'treat_pacing',
'treat_therapeutic_anticoagulant',
'inter_ivfluid_cl',
'inter_ivfluid_cr',
'inter_ivfluid_fl',
'inter_ivfluid_ge',
'inter_ivfluid_in',
'inter_ivfluid_lp',
'inter_ivfluid_os',
'inter_ivfluid_rs',
'inter_ivfluid_so',
'inter_ivfluid_ur',
'inter_ivfluid_yn',
'inter_bpt_yn',
'inter_bpt_type',
'inter_bpt_plat',
'inter_bpt_cry',
'inter_bpt_rbc',
'inter_bpt_ffp',
'inter_bpt_fibr',
'inter_ivimmunoglob',
'inter_diuretics',
'inter_nacetylcyst',
'inter_fluiddrain',
'inter_fluiddrain_plasma',
'inter_plasmaexch',
'inter_suppleox',
'inter_o2support',
'inter_o2therapy',
'inter_nivent',
'inter_ecmo',
'inter_neuromuscblock',
'inter_nitricoxide',
'inter_tracheost',
'inter_rtt',
'inter_rtt_dur',
'inter_inotrope',
'inter_inotrope_dur',
'inter_icu',
'inter_icu_date',
'inter_icudate',
'inter_icureadmit',
'inter_icuadmitdt',
'compl_shock',
'compl_reshock',
'compl_severeclot',
'compl_severeclot_r',
'compl_severeclot_s',
'compl_pleuraeff',
'compl_seizures',
'compl_focalneuro',
'compl_mening',
'compl_sepsis',
'compl_coagulopathy',
'compl_othersys',
'compl_ards',
'compl_cardiacfail',
'compl_cardiacfail_m',
'compl_myocarditis',
'compl_severebleed',
'compl_acuteclot',
'compl_acutehep',
'compl_hepatitis',
'compl_unlisted1',
'compl_unlisted2',
'compl_unlisted3',
'compl_unlisted4',
'compl_unlisted5',
'compl_unlisted6',
'compl_unlisted7',
'compl_unlisted8',
'compl_unlisted9',
'compl_unlisted10',
'compl_unlisted11',
'compl_unlisted12',
'compl_unlisted13',
'compl_unlisted14',
'compl_unlisted15',
'compl_unlisted16',
'compl_unlisted17',
'compl_unlisted18',
'compl_unlisted19',
'compl_unlisted20',
'compl_unlisted21',
'compl_unlisted22',
'compl_unlisted23',
'compl_unlisted24',
'compl_unlisted25'
]


############################################
############################################
## Data reading and initial proccesing 
############################################
############################################

countries = [{'label': country.name, 'value': country.alpha_3} for country in pycountry.countries]

#df_map=pd.read_csv('Vertex Dashboard/assets/data/map.csv')
#df_map=getRC.read_data_from_REDCAP()

###########################



sections=getRC.getDataSections(redcap_api_key)

vari_list=getRC.getVariableList(redcap_api_key,['dates','demog','comor','daily','outco','labs','vital','adsym','inter','treat', 'compl'])

df_map=getRC.get_REDCAP_Single_DB(redcap_url, redcap_api_key,site_mapping,vari_list)
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

    linegraph_about = html.Div([
        html.Div("Insights contains multiple graphs to provide a digestable summary of the data. Each graph's purpose in explained below:"),
        html.Br(),
        html.Div('1.This table provides a summary of the treatments, frequnecy and median demonstrating the raw data.'),
        html.Br(),
        html.Div('2.This bar chart shows the frequency of each treatment and proportion compared to all patients illustrating the most popular interventions.'),
        html.Br(),
        html.Div('3.This plot shows the frequency of the five most common treatments and their respective intersections demonstrating those frequently utilised together.'),
        html.Br(),
        html.Div('4.This heatmap shows the frequency of co-morbidities and treatments demonstrating the most common combinations.'),
        html.Br(),
        html.Div('5.This heatmap shows the frequency of complications and treatments demonstrating the most common combinations.'),
        html.Br(),
        html.Div('6.This violin plot shows the distribution of ages for each treatment'),
        html.Br(),
        html.Div('7.This bar chart shows the cumulative frequency of treatments over time.'),
        html.Br(),
    ])



    dd=getRC.getDataDictionary(redcap_api_key)        
    variables_binary,variables_date,variables_number,variables_freeText,variables_units,variables_categoricas=getRC.getVaribleType(dd)   

    variables_binary=[var for var in variables_binary if var.startswith('inter_')]
    variables_date=[var for var in variables_date if var.startswith('inter_')]
    variables_number=[var for var in variables_number if var.startswith('inter_')]
    variables_freeText=[var for var in variables_freeText if var.startswith('inter_')]
    variables_units=[var for var in variables_units if var.startswith('inter_')]
    variables_categoricas=[var for var in variables_categoricas if var.startswith('inter_')]
    
    #new_bin_from_cat = [element for element in list(df_map.columns) if any(element.startswith(prefix) for prefix in variables_categoricas)]

    #variables_binary=variables_binary+new_bin_from_cat

    color_map = {'Discharge': '#00C26F', 'Censored': '#FFF500', 'Death': '#DF0069'}

    correct_names=dd[['field_name','field_label']]

    #descriptive = ia.descriptive_table(ia.obtain_variables(df_map, 'symptoms'))
    descriptive = ia.descriptive_table(df_map,correct_names,variables_binary,variables_number)
    fig_table_treatx=idw.table(descriptive)

    #pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='treatx_age_gender_pyramid_chart')

    proportions_treatx, set_data_treatx = ia.get_proportions(df_map,'treatments')
    freq_chart_treatx = idw.frequency_chart(proportions_treatx, title='Frequency of Treatments Utilised')
    upset_plot_treatx = idw.upset(set_data_treatx, title='Frequency of Combinations of the Five Most Common Treatments')

    #proportions_treatx, set_data_treatx= ia.get_proportions(df_map,'treatments')
    #freq_chart_treatx = idw.frequency_chart(proportions_treatx, title='Frequency of treatments utilised')
    #upset_plot_treatx = idw.upset(set_data_treatx, title='Frequency of Combinations of the Five Most Common Treatments')

   

    symptoms_columns = [col for col in df_map.columns if col.startswith('adsym_')]
    df1=df_map[symptoms_columns]
 
    comor_columns = [col for col in df_map.columns if col.startswith('comor_')]
    df2 = df_map[comor_columns]

    compl_columns = [col for col in df_map.columns if col.startswith('compl_')]
    df3 = df_map[compl_columns]

    treatx_columns = [col for col in df_map.columns if col.startswith('inter_')]
    df4 = df_map[treatx_columns]

    mapper = {'Yes':1,'No':0}
    df1 = df1.replace(mapper)
    df2 = df2.replace(mapper) 
    df3 = df3.replace(mapper)  
    df4 = df4.replace(mapper)                  
    heatmap1=idw.heatmap(df2,df4,"Heatmap comparing Frequency of Co-morbidities and Treatments",graph_id="treatx_Heatmap2")

    heatmap2=idw.heatmap(df3,df4,"Heatmap comparing Frequency of Complications and Treatments",graph_id="treatx_Heatmap3")

    #Violin Plot
    vio_bin = 'slider_sex'
    vio_cont = 'age'

    #comor_columns = [col for col in df_map.columns if col.startswith('comor_')]
    demog_columns = [vio_bin, vio_cont]
    df11 = df_map[list(treatx_columns) + (demog_columns)]
    #violin_plot = idw.violin_plot(df11, vio_bin, vio_cont,5,'Age Distribution of Treatments by Sex', graph_id="treatx_ViolinPlot2")
    violin_plot=idw.frequency_chart(proportions_treatx, title='Frequency of Treatments Utilised')
    #Cumuative Frequency
    cml_cont = 'dates_adm'
    cml_cat = treatx_columns
    df12 = df_map[list(treatx_columns) + [cml_cont]]
    #cumulative_frequency = idw.cumulative_frequency(df12, cml_cont, cml_cat,5, 'Cumulative Frequency of Treatments', graph_id='treatx_cumulative_frequency')
    cumulative_frequency=idw.frequency_chart(proportions_treatx, title='Frequency of Treatments Utilised')
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
        dbc.ModalHeader(html.H3("Treatments", id="treatx_line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),  

        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Filters and Controls",  
                    children=[idw.filters_controls('i',country_dropdown_options)]
                ),                
                dbc.AccordionItem(
                    title="Insights",  
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([dbc.Col([fig_table_treatx],id='table_treatx')]), label='1.Descriptive table'),
                            #dbc.Tab(dbc.Row([dbc.Col(pyramid_chart,id='pyramid-chart-col')]), label='Age and Sex'),
                            dbc.Tab(dbc.Row([dbc.Col(freq_chart_treatx,id='freqTreatx_chart')]), label='2.Frequency of Treatments Utilised'),
                            dbc.Tab(dbc.Row([dbc.Col(upset_plot_treatx,id='upsetTreatx_chart')]), label='3.Intersections of Treatments'),
                            #dbc.Tab(dbc.Row([dbc.Col(boxplot_graph,id='boxplot_graph-col')]), label='Length of hospital stay by age group'),
                            #dbc.Tab(dbc.Row([dbc.Col(freq_chart_comor,id='freqcomor_chart')]), label='Comorbidities on presentation: Frequency'),
                            #dbc.Tab(dbc.Row([dbc.Col(upset_plot_comor,id='upsetcomor_chart')]), label='Comorbidities on presentation:Intersections'),
                            dbc.Tab(dbc.Row([dbc.Col(heatmap1,id='treatx_Heatmap2')]), label='4.Co-morbidities and Interventions'),
                            dbc.Tab(dbc.Row([dbc.Col(heatmap2,id='treatx_Heatmap3')]), label='5.Complications and Interventions'),
                            dbc.Tab(dbc.Row([dbc.Col(violin_plot,id='violinplot_chart')]), label='6.Age Distribution of Treatments by Sex'),
                            dbc.Tab(dbc.Row([dbc.Col(cumulative_frequency,id='cumulative_frequency')]), label='7.Cumulative Frequency of Treatments'),
                            
                        ])
                    ]
                )
            ])
        ], style={ 'overflowY': 'auto','minHeight': '75vh','maxHeight': '75vh'}),

        idw.ModalFooter('i',linegraph_instructions,linegraph_about)


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
'''
    @app.callback(
        [Output('pyramid-chart-col', 'children'),
         Output('freqTreatx_chart', 'children'),
         Output('upsetTreatx_chart', 'children'),
         Output('treatx_Heatmap1', 'children'),
         Output('treatx_Heatmap2', 'children'),
         Output('violinplot_chart', 'children'),
         Output('cumulative_frequency', 'children')],
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
        pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='treatx_age_gender_pyramid_chart')

        proportions_treatx, set_data_treatx = ia.get_proportions(filtered_df,'symptoms')
        freq_chart_treatx = idw.frequency_chart(proportions_treatx, title='Frequency of signs and symptoms on presentation')
        upset_plot_treatx = idw.upset(set_data_treatx, title='Frequency of combinations of the five most common signs or symptoms')
        treatx_Heatmap1 = idw.heatmap(filtered_df,title="Heatmap comparing Frequency of Symptoms and Treatments",graph_id="treatx_Heatmap1")
        treatx_Heatmap2 = idw.heatmap(filtered_df,title ="Heatmap comparing Frequency of Co-morbidities and Treatments",graph_id="treatx_Heatmap2")
        violin_plot = idw.violin_plot(filtered_df,5,title='Age Distribution of Treatments',graph_id="treatx_ViolinPlot2")
        cumulative_frequency = idw.cumulative_frequency(filtered_df,5, 'Cumulative Frequency of Treatments', graph_id='treatx_cumulative_frequency')
        return [pyramid_chart,freq_chart_treatx,upset_plot_treatx,treatx_Heatmap1,treatx_Heatmap2,violin_plot,cumulative_frequency]
'''
