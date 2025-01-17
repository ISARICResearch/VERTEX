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
    button_item = 'Treatments'
    button_label = 'Interventions and Treatments'
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
    variable_list = ['subjid']
    variable_list += [
        col for col in ia.get_variable_list(dictionary, sections)
        if col in df_map.columns]
    df_map = df_map[variable_list].copy()


    # Interventions descriptive table
    split_column='outco_denguediag_class'
    #split_column_order=['']
    split_column_order=['Uncomplicated dengue','Dengue with warning signs', 'Severe dengue', 'Unkown']
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column,
        include_sections=['inter', 'treat'])
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column=split_column,
        column_reorder =split_column_order)
    fig_table = idw.fig_table(
        table, table_key=table_key,
        graph_id='table_' + suffix,
        graph_label='Descriptive Table',
        graph_about='Summary of treatments and interventions.')
    
    # Treatments frequency and upset charts
    section = 'treat'
    section_name = 'Treatments'
    df_daily = df_forms_dict['daily']


    df_upset = ia.get_descriptive_data(
        df_daily, dictionary,
        include_sections=[section], include_types=['binary', 'categorical'],include_id=True)
    
    df_upset=df_upset.groupby('subjid').max()

    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(
        df_upset, dictionary, proportions=proportions)
    about = 'Frequency of the ten most common ' + section_name.lower()
    freq_chart_treat = idw.fig_frequency_chart(
        proportions,
        title='Frequency of ' + section_name,
        graph_id=section + '_freq_' + suffix,
        graph_label=section_name + ': Frequency',
        graph_about=about)
    about = 'Intersection sizes of the five most common '
    about += section_name.lower()
    upset_plot_treat = idw.fig_upset(
        counts_intersections,
        title='Intersection sizes of ' + section_name.lower(),
        graph_id=section + '_upset_' + suffix,
        graph_label=section_name + ': Intersections',
        graph_about=about)
    
    # Interventions frequency and upset charts
    section = 'inter'
    section_name = 'Interventions'
    df_upset = ia.get_descriptive_data(
        df_map, dictionary,
        include_sections=[section], include_types=['binary', 'categorical'])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(
        df_upset, dictionary, proportions=proportions)
    about = 'Frequency of the ten most common ' + section_name.lower()
    freq_chart_inter = idw.fig_frequency_chart(
        proportions,
        title='Frequency of ' + section_name,
        graph_id=section + '_freq_' + suffix,
        graph_label=section_name + ': Frequency',
        graph_about=about)
    about = 'Intersection sizes of the five most common '
    about += section_name.lower()
    upset_plot_inter = idw.fig_upset(
        counts_intersections,
        title='Intersection sizes of ' + section_name.lower(),
        graph_id=section + '_upset_' + suffix,
        graph_label=section_name + ': Intersections',
        graph_about=about)
    


    return (fig_table,freq_chart_treat,upset_plot_treat,freq_chart_inter,upset_plot_inter)
