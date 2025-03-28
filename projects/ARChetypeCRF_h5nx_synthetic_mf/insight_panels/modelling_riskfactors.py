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
    button_label = 'Risk factors'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''

    df_lr = df_map.loc[(
        df_map['outco_binary_outcome'].isin(['Death', 'Discharged']) &
        (df_map['demog_age'] > 18))].copy()
    df_lr['outco_binary_outcome'] = (df_lr['outco_binary_outcome'] == 'Death')
    df_lr['outco_binary_outcome'] = df_lr['outco_binary_outcome'].astype(int)

    # Create lots of custom variables and add to the dictionary
    df_lr['demog_age___over64'] = (df_lr['demog_age'] >= 64).astype(object)
    # df_lr.loc[df_lr['demog_age'].isna(), 'demog_age___over64'] = np.nan
    df_lr['demog_sex___Male'] = (df_lr['demog_sex'] == 'Male').astype(object)
    # df_lr.loc[df_lr['demog_sex'].isna(), 'demog_sex___Male'] = np.nan
    df_lr['comor_chrkidney_stag___3b_5'] = (
        df_lr['comor_chrkidney_stag'].isin(['Stage 3b', 'Stage 4', 'Stage 5']))
    df_lr['comor_chrkidney_stag___3b_5'] = (
        df_lr['comor_chrkidney_stag___3b_5'].astype(object))
    df_lr['comor_chrkidney_stag___1_3a'] = (
        df_lr['comor_chrkidney_stag'].isin(['Stage 1', 'Stage 2', 'Stage 3a']))
    df_lr['comor_chrkidney_stag___1_3a'] = (
        df_lr['comor_chrkidney_stag___1_3a'].astype(object))
    # columns = ['comor_chrkidney_stag___1_3a', 'comor_chrkidney_stag___3b_5']
    # df_lr.loc[df_lr['comor_chrkidney'].isna(), columns] = np.nan  # SL var
    df_lr['comor_liverdisease_type___Mild'] = (
        df_lr['comor_liverdisease_type'] == 'Mild').astype(object)
    df_lr['comor_liverdisease_type___Moderate_or_severe'] = (
            df_lr['comor_liverdisease_type'] == 'Moderate or severe'
        ).astype(object)
    # columns = [
    #     'comor_liverdisease_type___Moderate_or_severe',
    #     'comor_liverdisease_type___Mild']
    # df_lr.loc[df_lr['comor_liverdisease'].isna(), columns] = np.nan  # SL var
    df_lr['vital_gcs___9_11'] = (
        df_lr['vital_gcs'].isin([9, 10, 11])).astype(object)
    df_lr['vital_gcs___under9'] = (df_lr['vital_gcs'] < 9).astype(object)
    # columns = ['vital_gcs___9_11', 'vital_gcs___under9']
    # df_lr.loc[df_lr['vital_gcs'].isna(), columns] = np.nan
    df_lr['labs_platelets_103ul___low'] = (
        df_lr['labs_platelets_103ul'] < 1.2).astype(object)
    df_lr['labs_platelets_103ul___high'] = (
        df_lr['labs_platelets_103ul'] > 3.8).astype(object)
    # columns = [
    #     'labs_platelets_103ul___under1', 'labs_platelets_103ul___over4']
    # df_lr.loc[df_lr['labs_platelets_103ul'].isna(), columns] = np.nan
    predictors_dict = pd.DataFrame(
        columns=['form_name', 'field_type', 'field_label', 'parent'])
    predictors_dict.loc['demog_age___over64'] = [
        'presentation', 'binary', 'Over 64', 'demog_age']
    predictors_dict.loc['comor_liverdisease_type___Moderate_or_severe'] = [
        'presentation', 'binary',
        'Moderate or severe', 'comor_liverdisease_type']
    predictors_dict.loc['comor_chrkidney_stag___1_3a'] = [
        'presentation', 'binary', 'Stage 1 or 2 or 3a', 'comor_chrkidney_stag']
    predictors_dict.loc['comor_chrkidney_stag___3b_5'] = [
        'presentation', 'binary', 'Stage 3b or 4 or 5', 'comor_chrkidney_stag']
    predictors_dict.loc['vital_gcs___9_11'] = [
        'daily', 'binary', 'Moderate', 'vital_gcs']
    predictors_dict.loc['vital_gcs___under9'] = [
        'daily', 'binary', 'Severe', 'vital_gcs']
    predictors_dict.loc['labs_platelets_103ul___low'] = [
        'daily', 'binary', 'Low (under 1.2)', 'labs_platelets_103ul']
    predictors_dict.loc['labs_platelets_103ul___high'] = [
        'daily', 'binary', 'High (over 3.8)', 'labs_platelets_103ul']
    predictors_dict = predictors_dict.reset_index().rename(
        columns={'index': 'field_name'})
    predictors_dict = predictors_dict.to_dict(orient='list')
    dictionary = ia.extend_dictionary(dictionary, predictors_dict, df_lr)
    dictionary['field_label'] = dictionary['field_label'].astype(str)
    dictionary.loc[(
            dictionary['field_name'].str.startswith('adsym')),
        'field_label'] += ' (admission)'
    formatted_labels = ia.format_variables(dictionary)

    # First logistic regression

    predictors = [
        'demog_sex',
        'demog_age',
        'demog_healthcare',
        'comor_smoking_yn',
        'comor_diabetes_yn',
        'comor_chrkidney',
        # 'comor_chrkidney_stag',
        'comor_liverdisease',
        # 'comor_liverdisease_type',
        'comor_obesity',
        'comor_hypertensi',
        'vacci_influenza_yn',
        'adsym_fever',
        'adsym_headache',
        'vital_highesttem_c',
        'vital_hr',
        'vital_rr',
        'vital_meanbp',
        'vital_gcs',
        'sympt_fever',
        'sympt_headache',
        'labs_bilirubin_mgdl',
        # 'labs_creatinine_mgdl',
        'labs_platelets_103ul',
        'compl_severeliver',
        'compl_acuterenal',
        'compl_ards',
        'inter_suppleo2'
    ]

    # Perform logistic regression with all predictors (multivariate)
    multivariate_lr_results_m1 = ia.execute_glm_regression(
        elr_dataframe_df=df_lr.copy(),
        elr_outcome_str='outco_binary_outcome',
        elr_predictors_list=predictors,
        model_type='logistic',
        print_results=False)

    # Perform logistic regression for each predictor (univariate)
    univariate_lr_results_m1 = []
    for predictor in predictors:
        univariate_lr_results_m1.append(
            ia.execute_glm_regression(
                elr_dataframe_df=df_lr.copy(),
                elr_outcome_str='outco_binary_outcome',
                elr_predictors_list=[predictor],
                model_type='logistic',
                reg_type='uni',
                print_results=False
            )
        )
    univariate_lr_results_m1 = pd.concat(univariate_lr_results_m1)

    lr_results_m1 = pd.merge(
        multivariate_lr_results_m1, univariate_lr_results_m1,
        on='Study', how='outer')
    lr_results_m1 = lr_results_m1.rename(columns={'Study': 'Variable'})

    lr_results_m1['Variable'] = lr_results_m1['Variable'].apply(
        lambda x: '___'.join(x.strip(']').split('[')).split('___True')[0])
    lr_results_m1['Variable'] = lr_results_m1['Variable'].map(
        dict(zip(dictionary['field_name'], formatted_labels)))

    logr_table_m1 = idw.fig_table(
        lr_results_m1.copy(),
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Logistic Regression for anytime in-hospital mortality',
        graph_about='''...'''
    )

    labels = [
        'Variable', 'OddsRatio (multi)',
        'LowerCI (multi)', 'UpperCI (multi)', 'p-value (multi)']
    # Create the forest plot figure
    forest_plot_m1 = idw.fig_forest_plot(
        lr_results_m1[labels],
        title='Forest Plot',
        labels=labels,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Forest Plot for anytime in-hospital mortality',
        graph_about='''...'''
    )

    # Second version of logistic regression

    predictors = [
        'demog_sex___Male',
        'demog_age___over64',
        # 'demog_healthcare',
        'comor_smoking_yn',
        'comor_diabetes_yn',
        'comor_chrkidney_stag___1_3a',
        'comor_chrkidney_stag___3b_5',
        'comor_liverdisease_type___Mild',
        'comor_liverdisease_type___Moderate_or_severe',
        # 'comor_obesity',
        'comor_hypertensi',
        # 'vacci_influenza_yn',
        'adsym_fever',
        'adsym_headache',
        # 'vital_highesttem_c',
        # 'vital_hr',
        # 'vital_rr',
        # 'vital_meanbp',
        'vital_gcs___9_11',
        'vital_gcs___under9',
        # 'sympt_fever',
        # 'sympt_headache',
        # 'labs_bilirubin_mgdl',
        'labs_platelets_103ul___low',
        'labs_platelets_103ul___high',
        'compl_severeliver',
        'compl_acuterenal',
        'compl_ards',
        'inter_suppleo2'
    ]

    # Perform logistic regression with all predictors (multivariate)
    multivariate_lr_results_m2 = ia.execute_glm_regression(
        elr_dataframe_df=df_lr.copy(),
        elr_outcome_str='outco_binary_outcome',
        elr_predictors_list=predictors,
        model_type='logistic',
        print_results=False)

    # Perform logistic regression for each predictor (univariate)
    univariate_lr_results_m2 = []
    for predictor in predictors:
        univariate_lr_results_m2.append(
            ia.execute_glm_regression(
                elr_dataframe_df=df_lr.copy(),
                elr_outcome_str='outco_binary_outcome',
                elr_predictors_list=[predictor],
                model_type='logistic',
                reg_type='uni',
                print_results=False
            )
        )
    univariate_lr_results_m2 = pd.concat(univariate_lr_results_m2)

    lr_results_m2 = pd.merge(
        multivariate_lr_results_m2, univariate_lr_results_m2,
        on='Study', how='outer')
    lr_results_m2 = lr_results_m2.rename(columns={'Study': 'Variable'})

    lr_results_m2['Variable'] = lr_results_m2['Variable'].apply(
        lambda x: '___'.join(x.strip(']').split('[')).split('___True')[0])
    lr_results_m2['Variable'] = lr_results_m2['Variable'].map(
        dict(zip(dictionary['field_name'], formatted_labels)))

    logr_table_m2 = idw.fig_table(
        lr_results_m2.copy(),
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Logistic Regression for anytime in-hospital mortality',
        graph_about='''...'''
    )

    # Create the forest plot figure
    labels = [
        'Variable', 'OddsRatio (multi)',
        'LowerCI (multi)', 'UpperCI (multi)', 'p-value (multi)']
    forest_plot_m2 = idw.fig_forest_plot(
        lr_results_m2[labels],
        title='Forest Plot',
        labels=labels,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Forest Plot for anytime in-hospital mortality',
        graph_about='''...'''
    )

    # Linear regression for length of stay

    df_lr['length_of_stay'] = (
        df_lr['outco_date'] - df_lr['dates_admdate']).dt.days

    rename_dict = {
        'inter_o2support_type___High-flow nasal oxygen':
            'inter_o2support_type___3',
        'inter_o2support_type___Non-invasive ventilation':
            'inter_o2support_type___4',
        'inter_o2support_type___Invasive ventilation':
            'inter_o2support_type___5',
    }
    df_lr.rename(columns=rename_dict, inplace=True)
    dictionary['field_name'] = dictionary['field_name'].replace(rename_dict)

    predictors = [
        'demog_sex___Male',
        'demog_age___over64',
        'comor_smoking_yn',
        'comor_diabetes_yn',
        'comor_chrkidney',
        'comor_liverdisease',
        'comor_hypertensi',
        'vacci_influenza_yn',
        'adsym_fever',
        'adsym_headache',
        'compl_ards',
        'inter_o2support_type___3',
        'inter_o2support_type___4',
        'inter_o2support_type___5'
    ]

    # Perform logistic regression with all predictors (multivariate)
    multivariate_linr_results = ia.execute_glm_regression(
        elr_dataframe_df=df_lr.copy(),
        elr_outcome_str='length_of_stay',
        elr_predictors_list=predictors,
        model_type='linear',
        print_results=False)

    # Perform logistic regression for each predictor (univariate)
    univariate_linr_results = []
    for predictor in predictors:
        univariate_linr_results.append(
            ia.execute_glm_regression(
                elr_dataframe_df=df_lr.copy(),
                elr_outcome_str='length_of_stay',
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
    linr_results = linr_results.rename(columns={'Study': 'Variable'})

    linr_results['Variable'] = linr_results['Variable'].apply(
        lambda x: '___'.join(x.strip(']').split('[')).split('___True')[0])
    linr_results['Variable'] = linr_results['Variable'].map(
        dict(zip(dictionary['field_name'], formatted_labels)))

    linr_table_los = idw.fig_table(
        linr_results.copy(),
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Linear Regression for hospital length of stay',
        graph_about='''...'''
    )

    # # Create the forest plot figure
    # labels = [
    #     'Variable', 'Coefficient (multi)',
    #     'LowerCI (multi)', 'UpperCI (multi)', 'p-value (multi)']
    # forest_plot_los = idw.fig_forest_plot(
    #     linr_results[labels],
    #     title='Forest Plot',
    #     labels=labels,
    #     suffix=suffix, filepath=filepath, save_inputs=save_inputs,
    #     graph_label='Forest Plot for hospital length of stay',
    #     graph_about='''...'''
    # )

    return (
        logr_table_m1, forest_plot_m1,
        logr_table_m2, forest_plot_m2,
        linr_table_los)
