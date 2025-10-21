import pandas as pd

import vertex.IsaricAnalytics as ia
import vertex.IsaricDraw as idw


def define_button():
    """Defines the button in the main dashboard menu"""
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = "Lesions"
    button_label = "Assessment"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """

    df_photos = pd.concat(
        [
            df_forms_dict["photographs"]["subjid"],
            pd.get_dummies(df_forms_dict["photographs"]["photo_site"], prefix="photo_site", prefix_sep="___"),
        ],
        axis=1,
    )
    df_photos = df_photos.groupby("subjid").any().reset_index()

    lesion_locations = ["lesion_head", "lesion_ocular", "lesion_torso", "lesion_arms", "lesion_legs", "lesion_genit"]
    df_lesion = df_forms_dict["daily"][["subjid"] + lesion_locations]
    df_lesion = df_lesion.groupby("subjid").any().reset_index()

    df_table = pd.merge(df_lesion, df_photos, on="subjid", how="outer")
    df_table = pd.merge(df_map[["subjid", "demog_sex"]], df_table, on="subjid", how="left")

    split_column = "demog_sex"
    split_column_order = ["Female", "Male", "Other / Unknown"]
    table, table_key = ia.descriptive_table(df_table, dictionary, by_column=split_column, column_reorder=split_column_order)
    table.loc[1, "Variable"] += "<b><i> (ANY TIME DURING OBSERVATION)</b></i>"
    fig_table = idw.fig_table(
        table,
        table_key=table_key,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table*",
        graph_about="Summary of treatments and interventions.",
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

    return (fig_table, disclaimer)
