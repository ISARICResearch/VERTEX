import numpy as np
import pandas as pd
import IsaricDraw as idw
import IsaricAnalytics as ia


def define_button():
    '''Defines the button in the main dashboard menu'''
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = 'Enrolment'
    button_label = 'Enrolment Details'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, suffix):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''
    # Provide a list of all ARC data sections needed in the insight panel
    # Only variables from these sections will appear in the visuals
    sections = [
        'dates',  # Onset & presentation (REQUIRED)
        'demog',  # Demographics (REQUIRED)
        'daily',  # Daily sections (REQUIRED)
        'asses',  # Assessment (REQUIRED)
        'outco',  # Outcome (REQUIRED)
        'inclu',  # Inclusion criteria
        'readm',  # Re-admission and previous pin
        'travel',  # Travel history
        'expo14',  # Exposure history in previous 14 days
        'preg',  # Pregnancy
        'infa',  # Infant
        'comor',  # Co-morbidities and risk factors
        'medic',  # Medical history
        'drug7',  # Medication previous 7-days
        'drug14',  # Medication previous 14-days
        'vacci',  # Vaccination
        'advital',  # Vital signs & assessments on admission
        'adsym',  # Signs and symptoms on admission
        'vital',  # Vital signs & assessments
        'sympt',  # Signs and symptoms
        'lesion',  # Skin & mucosa assessment
        'treat',  # Treatments & interventions
        'labs',  # Laboratory results
        'imagi',  # Imaging
        'medi',  # Medication
        'test',  # Pathogen testing
        'diagn',  # Diagnosis
        'compl',  # Complications
        'inter',  # Interventions
        'follow',  # Follow-up assessment
        'withd',  # Withdrawal
    ]
    variable_list = ['subjid','filters_country','site']
    variable_list += [
        col for col in ia.get_variable_list(dictionary, sections)
        if col in df_map.columns]
    df_map = df_map[variable_list].copy()

    ###############################
    ###############################
    erolment_df=df_map[['subjid', 'filters_country','dates_enrolment']]
    erolment_df['dates_enrolment'] = pd.to_datetime(erolment_df['dates_enrolment'])
    erolment_df['month_year'] = erolment_df['dates_enrolment'].dt.strftime('%m-%Y')
    erolment_df=erolment_df[['subjid', 'filters_country','month_year']].groupby(['filters_country','month_year']).nunique().reset_index()
    erolment_df.columns=['stack_group', 'timepoint', 'value']
    
    sb_df=df_map[['subjid', 'site','filters_country']].groupby(['site','filters_country']).nunique().reset_index()

    for cos in df_map:
        if cos.startswith('date'):
            print(cos)

    #fig1 = idw.fig_placeholder(
    #    df_map, dictionary=dd,
    #    graph_id='fig1_id' + suffix, graph_label='Figure 1', graph_about='')
    outcomes_tb=df_map[['outco_denguediag','outco_denguediag_class','subjid']]
    outcomes_tb=outcomes_tb.groupby(['outco_denguediag','outco_denguediag_class']).nunique().reset_index()
    outcomes_tb.columns=['Denge Diagnosis','Severity','Number of patients']
    new_row = {'Denge Diagnosis': 'No', 'outco_denguediag_class': '---', 'Number of patients':(df_map['outco_denguediag']==0).sum() }
    outcomes_tb = pd.concat([outcomes_tb, pd.DataFrame([new_row])], ignore_index=True)
    new_row = {'Denge Diagnosis': '---', 'outco_denguediag_class': '---', 'Number of patients':(df_map['outco_denguediag']!=1).sum()-(df_map['outco_denguediag']==0).sum() }
    outcomes_tb = pd.concat([outcomes_tb, pd.DataFrame([new_row])], ignore_index=True)
    outcomes_tb['Denge Diagnosis'].loc[outcomes_tb['Denge Diagnosis']==1]='Yes'

    #fig4=idw.simple_fig_table(outcomes_tb,None)
    #outcomes_tb['outco_denguediag_main'].value_counts()
    
    sb_df['site']=sb_df['site'].str.split('-', n=1).str[0]
    #########################################
    #########################################
    
    fig1 = idw.fig_placeholder(
        df_map,
        graph_id='fig1-' + suffix, graph_label='Figure 1',
        graph_about='Placeholder figure 1')
    return (fig1,)
