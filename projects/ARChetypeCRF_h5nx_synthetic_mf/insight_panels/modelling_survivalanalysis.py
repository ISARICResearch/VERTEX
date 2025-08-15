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
    button_item = 'Modelling'
    button_label = 'Survival analysis'
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

    max_lengthofstay = 15
    df_map['outco_lengthofstay'] = (
        df_map['outco_lengthofstay'].fillna(max_lengthofstay))
    ind = (df_map['outco_lengthofstay'] > max_lengthofstay)
    df_map.loc[ind, 'outco_lengthofstay'] = max_lengthofstay
    df_map.loc[ind, 'outco_binary_outcome'] = 'Censored'

    df_model = ia.get_modelling_data(
        df_map, dictionary,
        outcome_columns=['outco_binary_outcome', 'outco_lengthofstay'],
        include_sections=[
            'demog', 'comor', 'adsym', 'vacci', 'vital', 'sympt', 'labs'])

    # df_model = df_model.loc[(
    #     df_model['outco_binary_outcome'].isin(['Death', 'Discharged']))]
    df_model['outco_binary_outcome'] = (
        df_model['outco_binary_outcome'] == 'Death').astype(int)

    ####
    # Add custom variables to the dataframe and to the dictionary

    df_model['demog_agegroup'] = df_model['demog_age'].apply(
        lambda x: '>64' if (x > 64) else ('40-64' if (x > 45) else '0-39'))
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

    # Cox regression for outcome
    predictors = [
        'demog_sex___Male',
        'demog_age',
        'demog_healthcare',
        'comor_smoking_yn',
        'comor_diabetes_yn',
        'comor_chrkidney',  #
        # 'comor_chrkidney_stag___Stage 1',
        # 'comor_chrkidney_stag___Stage 2',
        # 'comor_chrkidney_stag___Stage 3a',
        # 'comor_chrkidney_stag___Stage 3b',
        # 'comor_chrkidney_stag___Stage 4',
        # 'comor_chrkidney_stag___Stage 5',
        # 'comor_chrkidney_stag___Stage 3b or 4 or 5',  #
        'comor_liverdisease',  #
        # 'comor_liverdisease_type___Mild',  #
        # 'comor_liverdisease_type___Moderate or severe',  #
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
        # 'labs_platelets_103ul___Low',
        # 'labs_platelets_103ul___High',
        # 'compl_severeliver',
        # 'compl_acuterenal',
        # 'compl_ards',
        # 'inter_suppleo2',
        # 'inter_o2support_type___High-flow nasal oxygen',
        # 'inter_o2support_type___Non-invasive ventilation',
        # 'inter_o2support_type___Invasive ventilation'
    ]

    rename_columns = dict(zip(
        predictors, ['var' + str(n) for n in range(len(predictors))]))

    # Perform logistic regression with all predictors (multivariate)
    multivariate_cox_results = ia.execute_cox_model(
        df=df_model.rename(columns=rename_columns).copy(),
        duration_col='outco_lengthofstay',
        event_col='outco_binary_outcome',
        predictors=['var' + str(n) for n in range(len(predictors))],
        )
    multivariate_cox_results.rename(columns={
        'covariate': 'Variable',
        'HR': 'HR (multi)', 'p-value': 'p-value (multi)',
        'CI_lower': 'LowerCI (multi)', 'CI_upper': 'UpperCI (multi)'
    }, inplace=True)

    # Perform logistic regression for each predictor (univariate)
    univariate_cox_results = []
    for predictor in ['var' + str(n) for n in range(len(predictors))]:
        univariate_cox_results.append(
            ia.execute_cox_model(
                df=df_model.rename(columns=rename_columns).copy(),
                duration_col='outco_lengthofstay',
                event_col='outco_binary_outcome',
                predictors=[predictor],
            )
        )
    univariate_cox_results = pd.concat(univariate_cox_results)
    univariate_cox_results.rename(columns={
        'covariate': 'Variable',
        'HR': 'HR (uni)', 'p-value': 'p-value (uni)',
        'CI_lower': 'LowerCI (uni)', 'CI_upper': 'UpperCI (uni)'
    }, inplace=True)

    cox_results = pd.merge(
        multivariate_cox_results, univariate_cox_results,
        on='Variable', how='outer')

    # ----- Is it possible to move this into execute_glm_regression
    cox_results['p-value (multi)'] = (
        cox_results['p-value (multi)'].astype(float))
    cox_results['p-value (uni)'] = (
        cox_results['p-value (uni)'].astype(float))

    cox_results['Variable'] = cox_results['Variable'].apply(
        lambda x: x.split('_True')[0])

    cox_results['Variable'] = cox_results['Variable'].apply(
        lambda x: x.split('___')[0]).map(dict(zip(
            ['var' + str(n) for n in range(len(predictors))], predictors))
    ) + cox_results['Variable'].apply(
        lambda x: '___' + x.split('___')[-1] if '___' in x else '')
    # -----

    pvalue_significance = {'*': 0.05, '**': 0.01}
    cox_results_table = ia.regression_summary_table(
        cox_results.copy(), dictionary,
        pvalue_significance=pvalue_significance, result_type='HR'
    )

    table_key = '<br>'.join([
        f'({k}) p < {str(v)}' for k, v in pvalue_significance.items()])
    table_m3 = idw.fig_table(
        cox_results_table,
        table_key=table_key,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Cox Regression for in-hospital mortality*',
        graph_about='''...'''
    )

    columns = predictors + [
        'demog_agegroup', 'outco_lengthofstay', 'outco_binary_outcome']

    df_km, df_risktable, p_value = ia.execute_kaplan_meier(
        df_model[columns],
        duration_col='outco_lengthofstay',
        event_col='outco_binary_outcome',
        group_col='demog_agegroup'
    )
    kaplanmeier_m3 = idw.fig_kaplan_meier(
        (df_km, df_risktable), p_value=p_value, index_column='Group',
        title='Kaplan-Meier Plot for in-hospital mortality*',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Kaplan-Meier Plot for in-hospital mortality*',
        graph_about='''...''')

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

    return (table_m3, kaplanmeier_m3, disclaimer)
