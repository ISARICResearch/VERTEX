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

    # Demographics and comorbidities descriptive table
    # split_column = 'demog_sex'
    # split_column_order = ['Female', 'Male', 'Other / Unknown']
    split_column = 'outco_binary_outcome'
    split_column_order = ['Discharged', 'Death', 'Censored']
    sections = [
        'travel', 'expo14', 'drug7', 'drug14', 'advital', 'adsym', 'sympt']
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column,
        include_sections=sections, exclude_negatives=False)
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column=split_column,
        column_reorder=split_column_order)
    fig_table = idw.fig_table(
        table, table_key=table_key + '<br><b>(SYNTHETIC DATA)</b>',
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

    about = f'Frequency of the ten most common {section_name.lower()}'
    freq_chart_adsym = idw.fig_frequency_chart(
        proportions,
        title=f'Frequency of {section_name} (SYNTHETIC DATA)',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Frequency',
        graph_about=about)

    about = f'Intersection sizes of the five most common \
    {section_name.lower()}'
    upset_plot_adsym = idw.fig_upset(
        counts_intersections,
        title=f'Intersection sizes of {section_name.lower()} (SYNTHETIC DATA)',
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

    about = f'Frequency of the ten most common {section_name.lower()}'
    freq_chart_sympt = idw.fig_frequency_chart(
        proportions,
        title=f'Frequency of {section_name} (SYNTHETIC DATA)',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Frequency',
        graph_about=about)

    about = f'Intersection sizes of the five most common \
    {section_name.lower()}'
    upset_plot_sympt = idw.fig_upset(
        counts_intersections,
        title=f'Intersection sizes of {section_name.lower()} (SYNTHETIC DATA)',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_id=section,
        graph_label=section_name + ': Intersections',
        graph_about=about)

    return (
        fig_table, freq_chart_adsym, upset_plot_adsym,
        freq_chart_sympt, upset_plot_sympt)
