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
    button_item = "Daily Patient Observations"
    button_label = "Vital Signs & Assessments"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """
   
    #df_forms_dict
    daily_events=['Jour 1', 'Jour 3  (+1) ','Jour 7  (+2)',"Jour 14 (+/-2)",'Jour 28  (+/-5)','Jour 90  (+/-20)',"Jour 180 (+/-20)"]

    s=dictionary['field_name']
    vars_of_interest = s[s.str.startswith("vital_", na=False)].tolist()

    #vars_of_interest=['vital_highesttem_c','vital_hr','vital_rr']
    df_long = ia.build_all_patients_event_dataframe(
        daily_forms_data=df_forms_dict,
        daily_events=daily_events,
        variables=vars_of_interest,   # list of variable names you want summarised
        patient_col="subjid",
        day_col="day",
    )

    df_table = ia.get_descriptive_data(
        df_long,
        dictionary,
        by_column="day",
        include_sections=["vital"],   # or whichever sections include your vars
        exclude_negatives=False
    )

    table_daily, table_key_daily = ia.descriptive_table(
        df_table,
        dictionary,
        by_column="day",
        column_reorder=daily_events
    )

    fig_table_daily= idw.fig_table(
        table_daily,
        table_key=table_key_daily,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table",
        graph_about="...",
    )

    return (fig_table_daily,)
    #return (pyramid_chart, fig_table, freq_chart_comor, upset_plot_comor)
