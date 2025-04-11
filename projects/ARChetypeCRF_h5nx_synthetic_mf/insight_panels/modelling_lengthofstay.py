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
    button_label = 'Length of stay'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''

    df_map['outco_lengthofstay'] = (
        df_map['outco_date'] - df_map['dates_admdate']).dt.days

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
    # Linear regression for length of stay

    predictors = [
        'demog_sex___Male',
        'demog_age',
        'comor_smoking_yn',
        'comor_diabetes_yn',
        'comor_chrkidney',
        'comor_liverdisease',
        'comor_hypertensi',
        'vacci_influenza_yn',
        'adsym_fever',
        'adsym_headache',
        # 'compl_ards',
        # 'inter_o2support_type___High-flow nasal oxygen',
        # 'inter_o2support_type___Non-invasive ventilation',
        # 'inter_o2support_type___Invasive ventilation'
    ]

    rename_columns = dict(zip(
        predictors, ['var' + str(n) for n in range(len(predictors))]))

    # Perform logistic regression with all predictors (multivariate)
    multivariate_linr_results = ia.execute_glm_regression(
        elr_dataframe_df=df_model.rename(columns=rename_columns).copy(),
        elr_outcome_str='outco_lengthofstay',
        elr_predictors_list=['var' + str(n) for n in range(len(predictors))],
        model_type='linear',
        print_results=False)

    # Perform logistic regression for each predictor (univariate)
    univariate_linr_results = []
    for predictor in ['var' + str(n) for n in range(len(predictors))]:
        univariate_linr_results.append(
            ia.execute_glm_regression(
                elr_dataframe_df=df_model.rename(
                    columns=rename_columns).copy(),
                elr_outcome_str='outco_lengthofstay',
                elr_predictors_list=[predictor],
                model_type='linear',
                reg_type='uni',
                print_results=False
            )
        )
    univariate_linr_results = pd.concat(univariate_linr_results)

    linr_results = pd.merge(
        multivariate_linr_results, univariate_linr_results,
        on='Study', how='outer')

    # ----- Is it possible to move this into execute_glm_regression
    linr_results['p-value (multi)'] = (
        linr_results['p-value (multi)'].astype(float))
    linr_results['p-value (uni)'] = (
        linr_results['p-value (uni)'].astype(float))
    linr_results = linr_results.rename(columns={'Study': 'Variable'})

    linr_results['Variable'] = linr_results['Variable'].apply(
        lambda x: '___'.join(x.strip(']').split('[')).split('___True')[0])

    linr_results['Variable'] = linr_results['Variable'].apply(
        lambda x: x.split('___')[0]).map(dict(zip(
            ['var' + str(n) for n in range(len(predictors))], predictors))
    ) + linr_results['Variable'].apply(
        lambda x: '___' + x.split('___')[-1] if '___' in x else '')
    # -----

    linr_results_table = ia.regression_summary_table(
        linr_results.copy(), dictionary,
        p_values={'*': 0.05, '**': 0.01}, result_type='Coefficient'
    )

    table_m2 = idw.fig_table(
        linr_results_table,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Linear Regression for hospital length of stay*',
        graph_about='''...'''
    )

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

    return (table_m2, disclaimer)
