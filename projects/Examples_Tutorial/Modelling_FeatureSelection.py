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
    button_item = 'Modelling'  # CHANGE THIS
    button_label = 'Feature Selection'  # CHANGE THIS
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, suffix):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''
    # Provide a list of all ARC data sections needed in the insight panel
    # Only variables from these sections will appear in the visuals
    sections = [
    'dates',  # Onset & presentation (REQUIRED)
    'demog',  # Demographics (REQUIRED)
    'daily',  # Daily sections (REQUIRED)
    'asses',  # Assessment (REQUIRED)
    'outco',  # Outcome (REQUIRED)
    # 'inclu',  # Inclusion criteria
    # 'readm',  # Re-admission and previous pin
    # 'travel',  # Travel history
    # 'expo14',  # Exposure history in previous 14 days
    # 'preg',  # Pregnancy
    # 'infa',  # Infant
    'comor',  # Co-morbidities and risk factors
    # 'medic',  # Medical history
    # 'drug7',  # Medication previous 7-days
    # 'drug14',  # Medication previous 14-days
    # 'vacci',  # Vaccination
    'advital',  # Vital signs & assessments on admission
    'adsym',  # Signs and symptoms on admission
    'vital',  # Vital signs & assessments
    # 'sympt',  # Signs and symptoms
    'lesion',  # Skin & mucosa assessment
    # 'treat',  # Treatments & interventions
    'labs',  # Laboratory results
    # 'imagi',  # Imaging
    # 'medi',  # Medication
    # 'test',  # Pathogen testing
    # 'diagn',  # Diagnosis
    # 'compl',  # Complications
    # 'inter',  # Interventions
    # 'follow',  # Follow-up assessment
    # 'withd',  # Withdrawal
    ]

    variable_list = ['subjid']
    variable_list += [
        col for col in ia.get_variable_list(dictionary, ['comor', 'labs','vital','outco'])
        if col in df_map.columns]
    df_map = df_map[variable_list].copy()
    
    print("Initial size:", df_map.shape)
    
    df_map['outcome'] = df_map['outco_binary_outcome'].copy()
    print('outcome' in df_map.columns)
    print(df_map['outcome'].value_counts)

 #   df_map.to_csv('ISARIC_mpoxN?.csv', index=False)
 #   keep_cols = ['subjid', 'demog_age', 'demog_sex', 'comor_obesity', 'comor_aids_vrlno', 'adsym_exsal', 'outcome']
 #   df_map = df_map[keep_cols]

    df1 = df_map.loc[df_map['outco_binary_outcome'] != 'Censored']
 #   df1['outcome'] = df1['outco_binary_outcome'].copy()
    outcome_col = [col for col in df1.columns if 'binary_outcome' in col.lower()][0]  # assuming there's only one such column
    
    y = df1[outcome_col]
    if 'subjid' in df1.columns:
        subject_ids = df1['subjid'].copy()
        df1 = df1.drop(columns=['subjid'])
    else:
        subject_ids = df.index

    all_outcome_cols = [col for col in df1.columns if 'outco' in col.lower()]
    # Remove other outcome columns
    df1= df1.drop(columns=all_outcome_cols)
    
    # Prep anlaysis
    df1 = ia.impute_miss_val(df1, missing_threshold=0.5)  # df1 must contain usubjid
    df2 = ia.rmv_low_var(df1, mad_threshold=0.05, freq_threshold=0.05)  # df2 contain usubjid
    df3 = ia.rmv_high_corr(df2, correlation_threshold=0.8)  # df3 contain usubjid
    df3['outcome'] = y
    
    df3.to_csv('ISAdash_mpox2rmv.csv', index=True)
    print(df3.columns)
    print('outcome' in df1.columns)
    print('outcome' in df2.columns)
    print('outcome' in df3.columns)
    
  #  outcome_scores = (-5 + df3['comor_chrcardiac'] * 2 + df3['comor_hypertensi']*3 + np.random.uniform(-0.1, 0.1, df_map.shape[0]))
  #  print(outcome_scores)
  #  outcome_scores = 1 + np.exp(-outcome_scores.float()) 
  #  df3['outcome'] = np.round(1 /outcome_scores)
  #  df_map['outco_binary_outcome'] = df_map['outco_binary_outcome'].map({0.0: 'Discharge', 1.0: 'Death', np.nan: 'Censored'})
  #  df3['outcome'] = df3['outcome'].map({0.0: 'Discharge', 1.0: 'Death'})


  # df3.to_csv('ISARIC_prep.csv', index=False)
    all_results = ia.lasso_var_sel_binary(df3, outcome_col='outcome', random_state=42)
    df4 = all_results[0]
    
    scores_df = all_results[1]
    df_main_fields = all_results[2]
    
    scores_df_display = scores_df.copy()
    scores_df_display.columns = [str(col) for col in scores_df_display.columns]
    scores_df_display.index = [str(idx) for idx in scores_df_display.index]
    
    df_main_display = df_main_fields.copy()
    df_main_display.columns = [str(col) for col in df_main_display.columns]
    df_main_display.index = [str(idx) for idx in df_main_display.index]

    feature_selection_table = idw.fig_table(
        df4,
        graph_id='table_' + suffix,
        graph_label='Feature Selection Table',
        graph_about='...')
    
    parameter_scores_table = idw.fig_table(
        scores_df_display,
        graph_label='Parameter Scores Table',
        graph_id='table_' + suffix,
        graph_about='...')
    
    main_fields_table = idw.fig_table(
        df_main_display,
        graph_label='Main Fields Table',
        graph_id='table_' + suffix,
        graph_about='...')
    
    return feature_selection_table, parameter_scores_table , main_fields_table
