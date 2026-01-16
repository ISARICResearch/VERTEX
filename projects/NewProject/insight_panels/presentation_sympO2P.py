import numpy as np
import pandas as pd

import vertex.IsaricAnalytics as ia
import vertex.IsaricDraw as idw


def define_button():
    """Defines the button in the main dashboard menu"""
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = "Clinical Presentation"
    button_label = "Symptoms from onset to presentation"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """
    # Leftmost edge of the bins
    age_groups = ["0-18", "18-40", "41-80", "81+"]
    bins = [float(x.split("-")[0].split("+")[0].strip()) for x in age_groups]
    bins = bins + [np.inf]
    df_map.loc[:, "demog_agegroup"] = pd.cut(df_map["demog_age"], bins=bins, labels=age_groups, right=False)
    df_map["demog_agegroup"] = df_map["demog_agegroup"].cat.add_categories("Unknown").fillna("Unknown")
    # new_variable_dict = {
    #     'field_name': 'demog_agegroup',
    #     'form_name': 'presentation',
    #     'field_type': 'categorical',
    #     'field_label': 'Age group',
    #     'parent': 'demog'}
    # dictionary = ia.extend_dictionary(dictionary, new_variable_dict)  # TODO

    # Population pyramid
    color_map = {"Discharged": "#00C26F", "Censored": "#FFF500", "Death": "#DF0069"}
    column_dict = {"side": "demog_sex", "y_axis": "demog_agegroup", "stack_group": "outco_binary_outcome"}
    df_pyramid = ia.get_pyramid_data(df_map, column_dict, left_side="Femme", right_side="Homme")
    about = "Dual-sided population pyramid, showing age, sex and outcome."
    pyramid_chart = idw.fig_dual_stack_pyramid(
        df_pyramid,
        title="Population age pyramid",
        base_color_map=color_map,
        ylabel="Age group",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Demographics: Population Pyramid",
        graph_about=about,
    )



    split_column = "demog_sex"
    split_column_order = ["Femme", "Homme", "Other / Unknown"]
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column, include_sections=["demog", "comor"], exclude_negatives=False
    )
    table, table_key = ia.descriptive_table(df_table, dictionary, by_column=split_column, column_reorder=split_column_order)
    #table=df_map[include_columns].head()
    #table=dictionary.loc[dictionary['field_name'].isin(include_columns),['field_name','field_label']]
    #table=df_map[include_columns[8:15]].head()
    #table_key='demog_comor_table'
    '''
    fig_table = idw.fig_table(
        table,
        table_key=table_key,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table comorbidities",
        graph_about="Summary of demographics and comorbidities.",
    )
    
    # Comorbodities frequency and upset charts
    section = "comor"
    section_name = "Comorbidities on presentation"
    df_upset = ia.get_descriptive_data(df_map, dictionary, include_sections=[section], include_types=["binary", "categorical"])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(df_upset, dictionary)

    about = f"Frequency of the ten most common {section_name.lower()}"
    freq_chart_comor = idw.fig_frequency_chart(
        proportions,
        title=f"Frequency of {section_name}*",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label=section_name + ": Frequency*",
        graph_about=about,
    )

    about = f"Intersection sizes of the five most common \
    {section_name.lower()}"
    upset_plot_comor = idw.fig_upset(
        counts_intersections,
        title=f"Intersection sizes of {section_name.lower()}*",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label=section_name + ": Intersections*",
        graph_about=about,
    )
    
    table=pd.DataFrame([['a','b'],['1','2']],columns=['col1','col2'])
    #table=df_map.head()
    fig_table = idw.fig_table(
        table,
        table_key='table_key',
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table*",
        graph_about="Summary of demographics and comorbidities.",
    )'''

    split_column = "demog_sex"
    split_column_order = ["Femme", "Homme", "Other / Unknown"]
    df_table_adsym = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column, include_sections=["adsym"], exclude_negatives=False
    )
    table_adsym, table_key_adsym = ia.descriptive_table(df_table_adsym, dictionary, by_column=split_column, column_reorder=split_column_order)
    #table=df_map[include_columns].head()
    #table=dictionary.loc[dictionary['field_name'].isin(include_columns),['field_name','field_label']]
    #table=df_map[include_columns[8:15]].head()
    #table_key='demog_comor_table'
    
    fig_table_adsym = idw.fig_table(
        table_adsym,
        table_key=table_key_adsym,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table Symptoms from onset to presentation",
        graph_about="Summary of Symptoms from onset to presentation",
    )


    # sumptoms frequency and upset charts
    section = "adsym"
    section_name = "Symptoms from onset to presentation"
    df_upset = ia.get_descriptive_data(df_map, dictionary, include_sections=[section], include_types=["binary", "categorical"])
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(df_upset, dictionary)

    about = f"Frequency of the ten most common {section_name.lower()}"
    freq_chart_adsym = idw.fig_frequency_chart(
        proportions,
        title=f"Frequency of {section_name}*",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label=section_name + ": Frequency*",
        graph_about=about,
    )

    about = f"Intersection sizes of the five most common \
    {section_name.lower()}"
    upset_plot_adsym = idw.fig_upset(
        counts_intersections,
        title=f"Intersection sizes of {section_name.lower()}*",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label=section_name + ": Intersections*",
        graph_about=about,
    )
    return (fig_table_adsym,freq_chart_adsym,upset_plot_adsym)
    #return (pyramid_chart, fig_table, freq_chart_comor, upset_plot_comor)
