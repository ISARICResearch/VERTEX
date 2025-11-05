import pandas as pd

import vertex.IsaricDraw as idw


def define_button():
    """Defines the button in the main dashboard menu"""
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = "Enrolment"
    button_label = "Enrolment Details"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """

    df_sunburst = df_map[["subjid", "site", "filters_country"]].groupby(["site", "filters_country"]).nunique().reset_index()
    df_sunburst["site"] = df_sunburst["site"].str.split("-", n=1).str[0]

    fig_patients_bysite = idw.fig_sunburst(
        df_sunburst,
        title="Enrolment by site*",
        path=["filters_country", "site"],
        values="subjid",
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Site Enrolment*",
        graph_about="...",
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

    return (fig_patients_bysite, disclaimer)
