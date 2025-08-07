import numpy as np
import pandas as pd
import vertex.IsaricDraw as idw
import vertex.IsaricAnalytics as ia


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


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''
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
        'stack_group': 'outco_binary_outcome'}
    df_pyramid = ia.get_pyramid_data(
        df_map, column_dict, left_side='Female', right_side='Male')
    about = 'Dual-sided population pyramid, showing age, sex and outcome.'
    pyramid_chart = idw.fig_dual_stack_pyramid(
        df_pyramid,
        title='Age pyramid (SYNTHETIC DATA)',
        base_color_map=color_map, yaxis_label='Age group',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Demographics: Population Pyramid',
        graph_about=about)

    # Demographics and comorbidities descriptive table
    split_column = 'demog_sex'
    split_column_order = ['Female', 'Male', 'Other / Unknown']
    # split_column = 'outco_binary_outcome'
    # split_column_order = ['Discharged', 'Death', 'Censored']
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column,
        include_sections=['demog', 'comor'], exclude_negatives=False)
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column=split_column,
        column_reorder=split_column_order)
    fig_table = idw.fig_table(
        table, table_key=table_key + '<br><b>(SYNTHETIC DATA)</b>',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Descriptive Table',
        graph_about='Summary of demographics and comorbidities.')

    # Comorbodities frequency and upset charts
    section = 'comor'
    section_name = 'Comorbidities on presentation'
    df_upset = ia.get_descriptive_data(
        df_map, dictionary,
        include_sections=[section], include_types=['binary', 'categorical'])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(
        df_upset, dictionary, proportions=proportions)

    about = f'Frequency of the ten most common {section_name.lower()}'
    freq_chart_comor = idw.fig_frequency_chart(
        proportions,
        title=f'Frequency of {section_name} (SYNTHETIC DATA)',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label=section_name + ': Frequency',
        graph_about=about)

    about = f'Intersection sizes of the five most common \
    {section_name.lower()}'
    upset_plot_comor = idw.fig_upset(
        counts_intersections,
        title=f'Intersection sizes of {section_name.lower()} (SYNTHETIC DATA)',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label=section_name + ': Intersections',
        graph_about=about)

    return (pyramid_chart, fig_table, freq_chart_comor, upset_plot_comor)
