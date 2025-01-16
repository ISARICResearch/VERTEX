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
    button_label = 'Demographics / Comorbidities'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, suffix):
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
        # 'travel',  # Travel history
        # 'expo14',  # Exposure history in previous 14 days
        # 'preg',  # Pregnancy
        # 'infa',  # Infant
        'comor',  # Co-morbidities and risk factors
        # 'medic',  # Medical history
        # 'drug7',  # Medication previous 7-days
        # 'drug14',  # Medication previous 14-days
        # 'vacci',  # Vaccination
        # 'advital',  # Vital signs & assessments on admission
        # 'adsym',  # Signs and symptoms on admission
        # 'vital',  # Vital signs & assessments
        # 'sympt',  # Signs and symptoms
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

    # Leftmost edge of the bins
    age_groups = [
        '0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
        '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
        '81-85', '86-90', '91-95', '96-100', '101+']
    bins = [float(x.split('-')[0].split('+')[0].strip()) for x in age_groups]
    bins = bins + [np.inf]
    df_map.loc[:, 'demog_agegroup'] = pd.cut(
        df_map['demog_age'], bins=bins, labels=age_groups, right=False)
    new_variable_dict = {
        'field_name': 'demog_agegroup',
        'form_name': 'presentation',
        'field_type': 'categorical',
        'field_label': 'Age group',
        'parent': 'demog'}
    # dictionary = ia.extend_dictionary(dictionary, new_variable_dict)  # TODO

    # Population pyramid
    color_map = {
        'Discharged': '#00C26F',
        'Censored': '#FFF500',
        'Death': '#DF0069'}
    column_dict = {
        'side': 'demog_sex',
        'y_axis': 'demog_agegroup',
        'stack_group': 'outco_outcome'}
    df_pyramid = ia.get_pyramid_data(
        df_map, column_dict, left_side='Female', right_side='Male')
    about = 'Dual-sided population pyramid, showing age, sex and outcome.'
    pyramid_chart = idw.fig_dual_stack_pyramid(
        df_pyramid,
        base_color_map=color_map, yaxis_label='Age group',
        graph_id='age_gender_pyramid_chart_' + suffix,
        graph_label='Demographics: Population Pyramid',
        graph_about=about)

    # Demographics and comorbidities descriptive table
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column='demog_sex',
        include_sections=['demog', 'comor'])
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column='demog_sex',
        column_reorder=['Female', 'Male', 'Other / Unknown'])
    fig_table = idw.fig_table(
        table, table_key=table_key,
        graph_id='table_' + suffix,
        graph_label='Descriptive Table',
        graph_about='Summary of demographics and comorbidities.')

    # Comorbodities frequency and upset charts
    df_upset = ia.get_descriptive_data(
        df_map, dictionary,
        include_sections=['comor'], include_types=['binary', 'categorical'])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(
        df_upset, dictionary, proportions=proportions)
    about = 'Frequency of the ten most common comorbodities on presentation'
    freq_chart_comor = idw.fig_frequency_chart(
        proportions,
        title='Frequency of comorbidities',
        graph_id='comor_freq_' + suffix,
        graph_label='Comorbidities: Frequency',
        graph_about=about)
    about = 'Intersection sizes of the five most common comorbidities on'
    about = about + ' presentation'
    upset_plot_comor = idw.fig_upset(
        counts_intersections,
        title='Intersection sizes of the five most common comorbidities',
        graph_id='comor_upset_' + suffix,
        graph_label='Comorbidities: Intersections',
        graph_about=about)

    return (pyramid_chart, fig_table, freq_chart_comor, upset_plot_comor)
