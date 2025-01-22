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
    button_item = 'Clinical Presentation'
    button_label = 'Signs and Symptoms'
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
        'filters',  # Filter variables (REQUIRED)
        'dates',  # Onset & presentation (REQUIRED)
        'demog',  # Demographics (REQUIRED)
        'daily',  # Daily sections (REQUIRED)
        'asses',  # Assessment (REQUIRED)
        'outco',  # Outcome (REQUIRED)
        # 'inclu',  # Inclusion criteria
        # 'readm',  # Re-admission and previous pin
        'travel',  # Travel history
        'expo14',  # Exposure history in previous 14 days
        # 'preg',  # Pregnancy
        # 'infa',  # Infant
        # 'comor',  # Co-morbidities and risk factors
        # 'medic',  # Medical history
        'drug7',  # Medication previous 7-days
        'drug14',  # Medication previous 14-days
        # 'vacci',  # Vaccination
        # 'advital',  # Vital signs & assessments on admission
        'adsym',  # Signs and symptoms on admission
        'vital',  # Vital signs & assessments
        'sympt',  # Signs and symptoms
        # 'lesion',  # Skin & mucosa assessment
        # 'treat',  # Treatments & interventions
        # 'labs',  # Laboratory results
        # 'imagi',  # Imaging
        # 'medi',  # Medication
        # 'test',  # Pathogen testing
        # 'diagn',  # Diagnosis
        # 'compl',  # Complications
        # 'inter',  # Interventions
        # 'follow',  # Follow-up assessment
        # 'withd',  # Withdrawal
        # 'country',  # Country
    ]
    variable_list = ['subjid']
    variable_list += [
        col for col in ia.get_variable_list(dictionary, sections)
        if col in df_map.columns]
    df_map = df_map[variable_list].copy()

    # Demographics and comorbidities descriptive table
    # split_column = 'outco_denguediag_class'
    # split_column_order = [
    #     'Uncomplicated dengue', 'Dengue with warning signs',
    #     'Severe dengue', 'Unknown']
    split_column = 'demog_sex'
    split_column_order = ['Female', 'Male', 'Other / Unknown']
    sections = [
        'travel', 'expo14', 'drug7', 'drug14', 'advital', 'adsym', 'sympt']
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column,
        include_sections=sections, exclude_negatives=False)
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column=split_column,
        column_reorder=split_column_order)
    fig_table = idw.fig_table(
        table, table_key=table_key,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Descriptive Table',
        graph_about='Summary of demographics and comorbidities.')

    # Symptoms on admission frequency and upset charts
    section = 'adsym'
    section_name = 'Symptoms on admission'
    df_upset = ia.get_descriptive_data(
        df_map, dictionary,
        include_sections=[section], include_types=['binary', 'categorical'])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(
        df_upset, dictionary, proportions=proportions)
    about = 'Frequency of the ten most common ' + section_name.lower()
    freq_chart_adsym = idw.fig_frequency_chart(
        proportions,
        title='Frequency of ' + section_name,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Frequency',
        graph_about=about)
    about = 'Intersection sizes of the five most common '
    about += section_name.lower()
    upset_plot_adsym = idw.fig_upset(
        counts_intersections,
        title='Intersection sizes of ' + section_name.lower(),
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Intersections',
        graph_about=about)

    # Symptoms in first 24hr frequency and upset charts
    section = 'sympt'
    section_name = 'Symptoms in first 24hr'
    df_upset = ia.get_descriptive_data(
        df_map, dictionary,
        include_sections=[section], include_types=['binary', 'categorical'])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(
        df_upset, dictionary, proportions=proportions)
    about = 'Frequency of the ten most common ' + section_name.lower()
    freq_chart_sympt = idw.fig_frequency_chart(
        proportions,
        title='Frequency of ' + section_name,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Frequency',
        graph_about=about)
    about = 'Intersection sizes of the five most common '
    about += section_name.lower()
    upset_plot_sympt = idw.fig_upset(
        counts_intersections,
        title='Intersection sizes of ' + section_name.lower(),
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Intersections',
        graph_about=about)

    return (
        fig_table, freq_chart_adsym, upset_plot_adsym,
        freq_chart_sympt, upset_plot_sympt)
