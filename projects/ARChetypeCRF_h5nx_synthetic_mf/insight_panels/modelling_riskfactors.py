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
    button_item = 'Modelling'
    button_label = 'Risk factors'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''

    df_map['outco_lengthofstay'] = (
        df_map['outco_date'] - df_map['pres_date']).dt.days

    df_model = ia.get_modelling_data(
        df_map, dictionary,
        outcome_columns=['outco_binary_outcome', 'outco_lengthofstay'],
        include_sections=[
            'demog', 'comor', 'adsym', 'vacci', 'vital', 'sympt', 'labs'])

    # df_model = df_map.loc[(
    #     (df_model['outco_binary_outcome'].isin(['Death', 'Discharged'])) &
    #     (df_model['demog_age'] > 18))].copy()
    df_model = df_model.loc[(
        df_model['outco_binary_outcome'].isin(['Death', 'Discharged']))]
    df_model['outco_binary_outcome'] = (
        df_model['outco_binary_outcome'] == 'Death').astype(int)

    ####
    # Add custom variables to the dataframe and to the dictionary

    # Small numbers of patients in the dataframe had chronic kidney stages 4
    # and 5, so these were combined into one to help model fit (e.g. all
    # patients who had stage 5 had outcome Death, so the coefficient can't be
    # estimated properly)
    df_model['comor_chrkidney_stag___Stage 3b or 4 or 5'] = (
        df_model['comor_chrkidney_stag___Stage 3b'] |
        df_model['comor_chrkidney_stag___Stage 4'] |
        df_model['comor_chrkidney_stag___Stage 5'])
    df_model['vital_gcs___Moderate'] = (
        df_model['vital_gcs'].isin([9, 10, 11])).astype(object)
    df_model['vital_gcs___Severe'] = (df_model['vital_gcs'] < 9)
    # Some variables like platelet count can have a non-linear effect on
    # outcome (e.g. low values and high values both increase the risk)
    # The same may be true of patient age but this wasn't considered here
    df_model['labs_platelets_103ul___Low'] = (
        df_model['labs_platelets_103ul'] < 1.5)
    df_model['labs_platelets_103ul___High'] = (
        df_model['labs_platelets_103ul'] > 4.5)

    dictionary_columns = [
        'field_name', 'form_name', 'field_type', 'field_label', 'parent']
    dictionary_values = [
        [
            'comor_chrkidney_stag___Stage 3b or 4 or 5', 'presentation',
            'binary', 'Stage 3b or 4 or 5', 'comor_chrkidney_stag'],
        [
            'vital_gcs___Moderate', 'daily',
            'binary', 'Moderate (9 to 11)', 'vital_gcs'],
        [
            'vital_gcs___Severe', 'daily',
            'binary', 'Severe (less than 9)', 'vital_gcs'],
        [
            'labs_platelets_103ul___Low', 'daily',
            'binary', 'Low (under 1.5)', 'labs_platelets_103ul'],
        [
            'labs_platelets_103ul___High', 'daily',
            'binary', 'High (over 4.5)', 'labs_platelets_103ul'
        ]
    ]
    predictors_dict = pd.DataFrame(
        dictionary_values, columns=dictionary_columns).to_dict(orient='list')
    dictionary = ia.extend_dictionary(dictionary, predictors_dict, df_model)

    ####
    # Logistic regression

    predictors = [
        'demog_sex___Male',
        'demog_age',
        'demog_healthcare',
        'comor_smoking_yn',
        'comor_diabetes_yn',
        # 'comor_chrkidney',  #
        'comor_chrkidney_stag___Stage 1',
        'comor_chrkidney_stag___Stage 2',
        'comor_chrkidney_stag___Stage 3a',
        # 'comor_chrkidney_stag___Stage 3b',
        # 'comor_chrkidney_stag___Stage 4',
        # 'comor_chrkidney_stag___Stage 5',
        'comor_chrkidney_stag___Stage 3b or 4 or 5',  #
        # 'comor_liverdisease',  #
        'comor_liverdisease_type___Mild',  #
        'comor_liverdisease_type___Moderate or severe',  #
        'comor_obesity',
        'comor_hypertensi',
        'vacci_influenza_yn',
        'adsym_fever',
        'adsym_headache',
        # 'vital_highesttem_c',  #
        # 'vital_hr',  #
        # 'vital_rr',  #
        # 'vital_meanbp',  #
        'vital_gcs___Moderate',
        'vital_gcs___Severe',
        # 'sympt_fever',  #
        # 'sympt_headache',  #
        # 'labs_bilirubin_mgdl',  #
        # 'labs_creatinine_mgdl',  #
        'labs_platelets_103ul___Low',
        'labs_platelets_103ul___High',
        # 'compl_severeliver',
        # 'compl_acuterenal',
        # 'compl_ards',
        # 'inter_suppleo2'
    ]

    valid_predictors_vif, iterative_vif = (
        ia.variance_influence_factor_backwards_elimination(
            df_model, dictionary, predictors_list=predictors))

    valid_predictors_outcome = ia.remove_single_binary_outcome_predictors(
        df_model, dictionary,
        predictors_list=predictors, outcome_str='outco_binary_outcome')

    rename_columns = dict(zip(
        predictors, ['var' + str(n) for n in range(len(predictors))]))

    # Perform logistic regression with all predictors (multivariate)
    # ---- Is it possible to have reg_type='both' that does all of this in one?
    multivariate_logr_results = ia.execute_glm_regression(
        elr_dataframe_df=df_model.rename(columns=rename_columns).copy(),
        elr_outcome_str='outco_binary_outcome',
        elr_predictors_list=['var' + str(n) for n in range(len(predictors))],
        model_type='logistic',
        print_results=False)

    # Perform logistic regression for each predictor (univariate)
    univariate_logr_results = []
    for predictor in ['var' + str(n) for n in range(len(predictors))]:
        univariate_logr_results.append(
            ia.execute_glm_regression(
                elr_dataframe_df=df_model.rename(
                    columns=rename_columns).copy(),
                elr_outcome_str='outco_binary_outcome',
                elr_predictors_list=[predictor],
                model_type='logistic',
                reg_type='uni',
                print_results=False
            )
        )
    univariate_logr_results = pd.concat(univariate_logr_results)

    logr_results = pd.merge(
        multivariate_logr_results, univariate_logr_results,
        on='Study', how='outer')
    # -----

    # ----- Is it possible to move this into execute_glm_regression
    logr_results['p-value (multi)'] = (
        logr_results['p-value (multi)'].astype(float))
    logr_results['p-value (uni)'] = (
        logr_results['p-value (uni)'].astype(float))
    logr_results = logr_results.rename(columns={'Study': 'Variable'})

    logr_results['Variable'] = logr_results['Variable'].apply(
        lambda x: '___'.join(x.strip(']').split('[')).split('___True')[0])

    logr_results['Variable'] = logr_results['Variable'].apply(
        lambda x: x.split('___')[0]).map(dict(zip(
            ['var' + str(n) for n in range(len(predictors))], predictors))
    ) + logr_results['Variable'].apply(
        lambda x: '___' + x.split('___')[-1] if '___' in x else '')
    # -----

    highlight_predictors = {
        '+': [
            var for var in logr_results['Variable'].values
            if var not in valid_predictors_vif],
        '†': [
            var for var in logr_results['Variable'].values
            if var not in valid_predictors_outcome],
    }
    pvalue_significance = {'*': 0.05, '**': 0.01}
    logr_results_table = ia.regression_summary_table(
        logr_results.copy(), dictionary,
        highlight_predictors=highlight_predictors,
        pvalue_significance=pvalue_significance
    )

    table_key = {
        '+': '''(+) Multicollinearity (VIF >10 in an iterative backwards \
elimination), either remove this variable or a highly correlated variable''',
        '†': '''(†) Perfect predictor (every patient with this variable has \
only one of the outcomes), the variable should be removed from this \
analysis'''
    }
    table_key = (
        '<br>'.join([
            f'({k}) p < {str(v)}' for k, v in pvalue_significance.items()]) +
        '<br>'.join([
            v for k, v in table_key.items()
            if len(highlight_predictors[k]) > 0]))
    table_m1 = idw.fig_table(
        logr_results_table,
        table_key=table_key,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Logistic Regression for anytime in-hospital mortality*',
        graph_about='''...'''
    )

    labels = [
        'Variable', 'OddsRatio (multi)',
        'LowerCI (multi)', 'UpperCI (multi)', 'p-value (multi)']
    mapping_dict = dict(zip(
        dictionary['field_name'], ia.format_variables(dictionary)))
    # Create the forest plot figure
    forest_plot_m1 = idw.fig_forest_plot(
        logr_results[labels].replace({'Variable': mapping_dict}),
        title='Forest Plot for anytime in-hospital mortality*',
        labels=labels,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Forest Plot for anytime in-hospital mortality*',
        graph_about='''...'''
    )

    # ----- Should we have model checking of assumptions within the RAP?
    # -----

    disclaimer_text = '''Disclaimer: the underlying data for these figures is \
synthetic data. Results may not be clinically relevant or accurate.'''
    disclaimer_df = pd.DataFrame(
        disclaimer_text, columns=['paragraphs'], index=range(1))
    disclaimer = idw.fig_text(
        disclaimer_df,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='*DISCLAIMER: SYNTHETIC DATA*',
        graph_about=disclaimer_text
    )

    return (table_m1, forest_plot_m1, disclaimer)
