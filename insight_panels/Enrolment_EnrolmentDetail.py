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


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
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
    variable_list = ['subjid', 'filters_country', 'site']
    variable_list += [
        col for col in ia.get_variable_list(dictionary, sections)
        if col in df_map.columns]
    df_map = df_map[variable_list].copy()

    ###############################
    ###############################
    enrolment_df = df_map[['subjid', 'filters_country', 'dates_enrolment']]
    enrolment_df['dates_enrolment'] = pd.to_datetime(
        enrolment_df['dates_enrolment'], errors='coerce')
    enrolment_df['month_year'] = (
        enrolment_df['dates_enrolment'].dt.strftime('%m-%Y'))
    groupby_columns = ['filters_country', 'month_year']
    enrolment_df = enrolment_df[['subjid'] + groupby_columns].groupby(
        groupby_columns).nunique().reset_index()
    enrolment_df.columns = ['stack_group', 'timepoint', 'value']

    sb_df = df_map[['subjid', 'site', 'filters_country']].groupby(
        ['site', 'filters_country']).nunique().reset_index()

    for cos in df_map:
        if cos.startswith('date'):
            print(cos)

    outcomes_columns = ['outco_denguediag', 'outco_denguediag_class']
    outcomes_tb = df_map[outcomes_columns + ['subjid']]

    outcomes_tb[outcomes_columns] = outcomes_tb[outcomes_columns].fillna('---')
    # outcomes_tb['outco_denguediag_class'] = outcomes_tb['outco_denguediag_class'].fillna('---')
    # outcomes_tb['outco_denguediag'] = outcomes_tb['outco_denguediag'].fillna('---')
    outcomes_tb = outcomes_tb.groupby(
        ['outco_denguediag', 'outco_denguediag_class']).nunique()
    outcomes_tb = outcomes_tb.reset_index()
    outcomes_tb.columns = [
        'Dengue Diagnosis', 'Severity', 'Number of patients']

    outcomes_tb['Dengue Diagnosis'] = outcomes_tb['Dengue Diagnosis'].map({
        1: 'Yes', 0: 'No'})
    # outcomes_tb.loc[outcomes_tb['Dengue Diagnosis'] == 1, 'Dengue Diagnosis'] = 'Yes'
    # outcomes_tb.loc[outcomes_tb['Dengue Diagnosis'] == 0, 'Dengue Diagnosis'] = 'No'
    outcomes_tb = outcomes_tb.sort_values(by=['Dengue Diagnosis', 'Severity'])

    for qc in quality_report:
        qc_count = len(quality_report[qc])
        new_row = {
            'Dengue Diagnosis': '',
            'Severity': '<b>'+qc+'</b>',
            'Number of patients': qc_count}
        outcomes_tb = pd.concat(
            [outcomes_tb, pd.DataFrame([new_row])], ignore_index=True)

    total = outcomes_tb['Number of patients'].sum()
    new_row = {
        'Dengue Diagnosis': '',
        'Severity': '<b>Total</b>',
        'Number of patients': '<b>' + str(total) + '</b>'}
    outcomes_tb = pd.concat(
        [outcomes_tb, pd.DataFrame([new_row])], ignore_index=True)

    sb_df['site'] = sb_df['site'].str.split('-', n=1).str[0]

    # Convert 'timepoint' to datetime format and sort
    enrolment_df['timepoint'] = pd.to_datetime(
        enrolment_df['timepoint']).apply(lambda x: x.strftime('%m-%Y'))
    enrolment_df = enrolment_df.sort_values('timepoint')
    enrolment_df.rename(columns={'timepoint': 'index'}, inplace=True)

    #########################################
    #########################################

    fig_patients_bysite = idw.fig_sunburst(
        sb_df, path=['filters_country', 'site'], values='subjid',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Site Enrolment', graph_about='...')
    table_patients_bydiagnosis = idw.fig_table(
        outcomes_tb,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Number of patients by diagnosis', graph_about='...')
    fig_cumulativeenrolment = idw.fig_cumulative_bar_chart(
        enrolment_df, title='Cumulative Enrolment by date and country',
        xlabel='Month', ylabel='Number of patients',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Cumulative patient enrolment by country',
        graph_about='...')
    fig_enrolmentbtmonth = idw.fig_stacked_bar_chart(
        enrolment_df, xlabel='Month', ylabel='Number of patients',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Patient enrolment by country', graph_about='...')

    return (
        table_patients_bydiagnosis, fig_cumulativeenrolment,
        fig_enrolmentbtmonth, fig_patients_bysite)
