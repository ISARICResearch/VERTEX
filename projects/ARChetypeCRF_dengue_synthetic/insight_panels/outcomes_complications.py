import pandas as pd

import vertex.IsaricAnalytics as ia
import vertex.IsaricDraw as idw


def define_button():
    """Defines the button in the main dashboard menu"""
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = "Outcomes"
    button_label = "Outcomes and Complications"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """
    # Interventions descriptive table
    # split_column = 'demog_sex'
    # split_column_order = ['Female', 'Male', 'Other / Unknown']
    split_column = "outco_binary_outcome"
    split_column_order = ["Discharged", "Death", "Censored"]
    df_table = ia.get_descriptive_data(df_map, dictionary, by_column=split_column, include_sections=["compl", "crito", "outco"])
    table, table_key = ia.descriptive_table(df_table, dictionary, by_column=split_column, column_reorder=split_column_order)
    fig_table = idw.fig_table(
        table,
        table_key=table_key,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table*",
        graph_about="Summary of outcomes and complications.",
    )

    # Treatments frequency and upset charts
    section = "compl"
    section_name = "Complications"
    df_upset = ia.get_descriptive_data(
        df_map, dictionary, include_sections=[section], include_types=["binary", "categorical"], include_subjid=False
    )
    proportions = ia.get_proportions(df_upset, dictionary)
    counts_intersections = ia.get_upset_counts_intersections(df_upset, dictionary)

    about = f"Frequency of the ten most common {section_name.lower()}"
    freq_chart_compl = idw.fig_frequency_chart(
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
    upset_plot_compl = idw.fig_upset(
        counts_intersections,
        title=f"Intersection sizes of {section_name.lower()}*",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label=section_name + ": Intersections*",
        graph_about=about,
    )

    disclaimer_text = """Disclaimer: the underlying data for these figures is \
synthetic data. Results may not be clinically relevant or accurate."""
    disclaimer_df = pd.DataFrame(disclaimer_text, columns=["paragraphs"], index=range(1))
    disclaimer = idw.fig_text(
        disclaimer_df,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="*DISCLAIMER: SYNTHETIC DATA*",
        graph_about=disclaimer_text,
    )

    return (fig_table, freq_chart_compl, upset_plot_compl, disclaimer)
