import pandas as pd

import vertex.IsaricAnalytics as ia
import vertex.IsaricDraw as idw


def define_button():
    """Defines the button in the main dashboard menu"""
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = "Modelling"
    button_label = "Feature Selection"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """

    variable_list = ia.get_variables_by_section_and_type(
        df_map,
        dictionary,
        required_variables=["outco_binary_outcome"],
        include_sections=["demog", "comor", "labs", "vital"],
        include_subjid=True,
    )

    df_feat = df_map.loc[(df_map["outco_binary_outcome"].isin(["Death", "Discharged"])), variable_list].copy()
    df_feat["outco_binary_outcome"] = (df_feat["outco_binary_outcome"] == "Death").astype(int)

    # y = df_feat['outco_binary_outcome']
    # subjids = df_feat['subjid']
    # df_feat.drop(columns=['subjid', 'outco_binary_outcome'], inplace=True)

    # Prep anlaysis
    df_feat = ia.impute_miss_val(
        df_feat, dictionary, outcome_column="outcome_binary_outcome", missing_threshold=0.5, verbose=True
    )
    df_feat = ia.rmv_low_var(
        df_feat, dictionary, outcome_column="outcome_binary_outcome", mad_threshold=0.05, freq_threshold=0.05, verbose=True
    )
    df_feat = ia.rmv_high_corr(
        df_feat, dictionary, outcome_column="outcome_binary_outcome", correlation_threshold=0.7, verbose=True
    )
    # df_feat['outco_binary_outcome'] = y
    # df_feat['subjid'] = subjids

    # print(df3.columns)
    # print('outcome' in df1.columns)
    # print('outcome' in df2.columns)
    # print('outcome' in df3.columns)

    # outcome_scores = (-5 + df3['comor_chrcardiac'] * 2 + df3['comor_hypertensi']*3
    # + np.random.uniform(-0.1, 0.1, df_map.shape[0]))
    # print(outcome_scores)
    # outcome_scores = 1 + np.exp(-outcome_scores.float())
    # df3['outcome'] = np.round(1 /outcome_scores)
    # df_map['outco_binary_outcome'] = df_map['outco_binary_outcome'].map({0.0: 'Discharge', 1.0: 'Death', np.nan: 'Censored'})
    # df3['outcome'] = df3['outcome'].map({0.0: 'Discharge', 1.0: 'Death'})

    # Exclude subjid
    df_feat = df_feat.drop(columns=["subjid"])
    all_results = ia.lasso_var_sel_binary(
        df_feat, outcome_column="outco_binary_outcome", random_state=42, verbose=True, threshold=0.05, metric="roc_auc"
    )

    mapping_dict = dict(zip(dictionary["field_name"], ia.format_variables(dictionary)))

    df_features = all_results[0].rename(columns={"Feature": "Variable"})
    df_features["Variable"] = df_features["Variable"].replace(mapping_dict)
    df_features["Coefficient"] = df_features["Coefficient"].apply(lambda x: f"{x:.3f}")
    df_features.rename(columns={"Coefficient": "Feature Importance"}, inplace=True)

    scores_df_display = all_results[1].copy()
    scores_df_display = scores_df_display.map(lambda x: f"{x:.3f}")
    scores_df_display.columns = ["L1 ratio = " + str(col) for col in scores_df_display.columns]
    scores_df_display.index = [
        f"<b>C = {idx:.3g}</b>" if idx < 100 else f"<b>C = {idx:.0f}</b>" for idx in scores_df_display.index
    ]
    scores_df_display = scores_df_display.reset_index()
    scores_df_display = scores_df_display.rename(columns={"index": ""})

    feature_selection_table = idw.fig_table(
        df_features,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_id="fig_table_features",
        graph_label="Feature Selection Table*",
        graph_about="""...""",
    )

    parameter_scores_table = idw.fig_table(
        scores_df_display,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_id="fig_table_scores",
        graph_label="Hyperarameter Scores Table*",
        graph_about="""...""",
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

    return (feature_selection_table, parameter_scores_table, disclaimer)
