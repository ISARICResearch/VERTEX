import requests
import pandas as pd
# import os
import IsaricAnalytics as ia
import numpy as np

# def read_data_from_REDCAP():
#     contries_path = 'assets/countries.csv'
#     countries = pd.read_csv(contries_path, encoding='latin-1')
#
#     dfs = []
#     for ele_var in os.environ:
#         if ele_var.startswith('study_site'):
#             dfs.append(pd.DataFrame(
#                 [eval(os.environ[ele_var])],
#                 columns=['key', 'country_iso', 'site_id']))
#
#     # Concatenate the DataFrames
#     sites = pd.DataFrame(
#         data=[
#             ['7FA9ACD7DAB3B9BF51AE9CE797135EFD', 'COL', '1'],
#             ['25689E7C5B69F326B03B864B6FF97729', 'GBR', '2']],
#         columns=['key', 'country_iso', 'site_id'])
#
#     complete_data = pd.DataFrame()
#     for index, row in sites.iterrows():
#         try:
#             conex = {
#                 'token': row['key'],
#                 'content': 'record',
#                 'action': 'export',
#                 'format': 'json',
#                 'type': 'flat',
#                 'csvDelimiter': '',
#                 'rawOrLabel': 'label',
#                 'rawOrLabelHeaders': 'raw',
#                 'exportCheckboxLabel': 'false',
#                 'exportSurveyFields': 'false',
#                 'exportDataAccessGroups': 'false',
#                 'returnFormat': 'json'
#             }
#             # They will need to change this!
#             r = requests.post('https://ncov.medsci.ox.ac.uk/api/', data=conex)
#             print('HTTP Status: ' + str(r.status_code))
#             data = (r.json())
#             form1 = []
#             form2 = []
#             for i in data:
#                 if i['redcap_event_name'] == 'Initial Assessment / Admission':
#                     form1.append(i)
#                 elif i['redcap_event_name'] == 'Outcome / End of study':
#                     form2.append(i)
#
#             df = pd.concat([pd.DataFrame(form1), pd.DataFrame(form2)]).drop(
#                 columns='redcap_event_name').groupby('subjid').max()
#             df = df.reset_index()
#             df = ia.mapOutcomes(df)
#             df = ia.harmonizeAge(df)
#
#             country_name = countries['Country'].loc[(
#                 countries['Code'] == row['country_iso'])].iloc[0]
#             country_income = countries['Income group'].loc[(
#                 countries['Code'] == row['country_iso'])].iloc[0]
#             country_region = countries['Region'].loc[(
#                 countries['Code'] == row['country_iso'])].iloc[0]
#             df['slider_country'] = country_name
#             df['country_iso'] = row['country_iso']
#             df['income'] = country_income
#             df['region'] = country_region
#             df['epiweek.admit'] = [1]*len(df)
#             df['dur_ho'] = [0]*len(df)
#
#             df.rename(columns={
#                 'subjid': 'usubjid',
#                 'demog_age': 'age',
#                 'demog_sex': 'slider_sex',
#                 'outco_outcome': 'outcome'}, inplace=True)
#
#             complete_data = pd.concat([complete_data, df])
#
#         except Exception as e:
#             print(e)
#
#     return complete_data


def getDataDictionary(redcap_url, redcap_api_key):
    data = {
        'token': redcap_api_key,
        'content': 'metadata',
        'format': 'json',
        'returnFormat': 'json'
    }
    # Make the API request
    response = requests.post(redcap_url, data=data)
    if response.status_code == 200:
        # Convert response JSON to DataFrame
        metadata = response.json()
        df = pd.DataFrame(metadata)
    else:
        df = None
    return df


def getDataSections(redcap_url, redcap_api_key):
    data = {
        'token': redcap_api_key,
        'content': 'metadata',
        'format': 'json',
        'returnFormat': 'json'
    }
    sections = []
    # Make the API request

    response = requests.post(redcap_url, data=data)
    if response.status_code == 200:
        # Convert response JSON to DataFrame
        metadata = response.json()
        df = pd.DataFrame(metadata)
        for var_name in df['field_name']:
            sectionx = var_name.split('_')[0]
            sections.append(sectionx)
        section_list = list(set(sections))
    else:
        section_list = None
    return section_list


def getVariableList(redcap_url, redcap_api_key, sections):
    dd = getDataDictionary(redcap_url, redcap_api_key)
    sections_ids = []
    for i in dd['field_name']:
        sections_ids.append(i.split('_')[0])
    dd['Section id'] = sections_ids
    vari_list = dd['field_name'].loc[dd['Section id'].isin(sections)]
    return ['subjid'] + list(vari_list)


def getVariableType(dd):
    variables_units = [
        col for col in dd['field_name'] if col.endswith('_units')]

    variables_binary = dd.loc[(
            dd['field_type'].isin(['radio']) &
            dd['select_choices_or_calculations'].isin([
                '1, Yes | 0, No | 99, Unknown', '1, Yes | 0, No'])),
        'field_name'].tolist()

    variables_date = dd.loc[(
            dd['text_validation_type_or_show_slider_number'].isin([
                'date_dmy', 'datetime_dmy'])),
        'field_name'].tolist()

    variables_number = dd.loc[(
            dd['text_validation_type_or_show_slider_number'] == 'number'),
        'field_name'].tolist()

    variables_freeText = dd.loc[(
            (dd['field_type'] == 'text') &
            (dd['text_validation_type_or_show_slider_number'] == '') &
            (dd['field_name'].isin(variables_units) == 0)),
        'field_name'].tolist()

    variables_categorical = dd.loc[(
            (dd['field_type'] == 'radio') &
            (dd['field_name'].isin(variables_binary) == 0) &
            (dd['field_name'].isin(variables_units) == 0)),
        'field_name'].tolist()

    variables_oneHotEncoded = dd.loc[(
            dd['field_type'] == 'checkbox'),
        'field_name'].tolist()

    variable_dict = {
        'binary': variables_binary,
        'date': variables_date,
        'number': variables_number,
        'freeText': variables_freeText,
        'units': variables_units,
        'categorical': variables_categorical,
        'OneHot': variables_oneHotEncoded
    }
    return variable_dict


def get_REDCAP_Single_DB(
        redcap_url, redcap_api_key, site_mapping, required_variables):
    contries_path = 'assets/countries.csv'
    countries = pd.read_csv(contries_path, encoding='latin-1')
    conex = {
        'token': redcap_api_key,
        'content': 'record',
        'action': 'export',
        'format': 'json',
        'type': 'flat',
        'csvDelimiter': '',
        'rawOrLabel': 'label',
        'rawOrLabelHeaders': 'raw',
        'exportCheckboxLabel': 'false',
        'exportSurveyFields': 'false',
        'exportDataAccessGroups': 'false',
        'returnFormat': 'json'
    }
    r = requests.post(redcap_url, data=conex)
    print('HTTP Status: ' + str(r.status_code))
    data = (r.json())

    events = []
    for i in data:
        events.append(i['redcap_event_name'])
    events = list(set(events))

    form1 = []
    form2 = []
    form3 = []
    for i in data:
        if (i['redcap_event_name'] == 'Initial Assessment / Admission'):
            form1.append(i)
        elif (i['redcap_event_name'] == 'Daily'):
            form2.append(i)
        elif (i['redcap_event_name'] == 'Outcome / End of study'):
            form3.append(i)

    form1 = pd.DataFrame(form1)
    form2 = pd.DataFrame(form2)
    form3 = pd.DataFrame(form3)
    form1 = ia.remove_MissingDataCodes(form1)
    form2 = ia.remove_MissingDataCodes(form2)
    form3 = ia.remove_MissingDataCodes(form3)

    #form1 = form1[list(set(required_variables).intersection(set(form1.columns)))]
    #form2 = form2[list(set(required_variables).intersection(set(form2.columns)))]
    #form3 = form3[list(set(required_variables).intersection(set(form3.columns)))]

    ###
    non_nan_columns_f1 = form1.columns[form1.notna().any()].tolist()
    non_nan_columns_f2 = form2.columns[form2.notna().any()].tolist()
    non_nan_columns_f3 = form3.columns[form3.notna().any()].tolist()

    form1 = form1[non_nan_columns_f1]
    form2 = form2[non_nan_columns_f2]
    form3 = form3[non_nan_columns_f3]

    oneHotEncoded = []

    selected_columns_1 = []
    selected_columns_2 = []
    selected_columns_3 = []
    for var in required_variables:
        # Check for exact matches first
        not_in_data_dict = True
        if var in form1.columns:
            selected_columns_1.append(var)
            not_in_data_dict = False
        if var in form2.columns:
            selected_columns_2.append(var)
            not_in_data_dict = False
        if var in form3.columns:
            selected_columns_3.append(var)
            not_in_data_dict = False
        if not_in_data_dict:
            # # Check for any columns that start with the variable prefix
            # selected_columns_1.extend([col for col in form1.columns if col.startswith(var)])
            # selected_columns_2.extend([col for col in form2.columns if col.startswith(var)])
            # selected_columns_3.extend([col for col in form3.columns if col.startswith(var)])
            for col in form1:
                if col.startswith(var):
                    if form1[col].eq('Unchecked').all():
                        pass
                    else:
                        if ('___' in col):
                            oneHotEncoded.append(col)
                            selected_columns_1.append(col)
            for col in form2:
                if (col.startswith(var) and ('___' not in col)):
                    if form2[col].eq('Unchecked').all():
                        pass
                    else:
                        if ('___' in col):
                            oneHotEncoded.append(col)
                            selected_columns_2.append(col)
            for col in form3:
                if (col.startswith(var) and ('___' not in col)):
                    if form3[col].eq('Unchecked').all():
                        pass
                    else:
                        if ('___' in col):
                            oneHotEncoded.append(col)
                            selected_columns_3.append(col)

    # Now filter form1 using the selected columns
    form1 = form1[selected_columns_1]
    form2 = form2[selected_columns_2]
    form3 = form3[selected_columns_3]

    ###

    form1['country'] = form1['subjid'].str.split('-').str[0]
    form1['country'] = form1['country'].map(site_mapping)

    dates = form1[['subjid', 'dates_admdate']]

    form2 = pd.merge(form2, dates, on='subjid', how='left')

    # Ensure both columns are in datetime format
    form2['daily_date'] = pd.to_datetime(form2['daily_date'], errors='coerce')

    form2['dates_admdate'] = pd.to_datetime(
        form2['dates_admdate'], errors='coerce')

    # Calculate the difference in days and create a new column
    form2['relative_day'] = (
        form2['daily_date'] - form2['dates_admdate']).dt.days

    form3 = pd.merge(form3, dates, on='subjid', how='left')
    # Ensure both columns are in datetime format
    form3['outco_date'] = pd.to_datetime(form3['outco_date'], errors='coerce')
    form3['dates_admdate'] = pd.to_datetime(
        form3['dates_admdate'], errors='coerce')

    # Calculate the difference in days and create a new column
    form3['outcome_day'] = (
        form3['outco_date'] - form3['dates_admdate']).dt.days

    # form2_day1 = form2.loc[(form2['relative_day'] == 1)]

    form1 = ia.harmonizeAge(form1)

    form3 = ia.mapOutcomes(form3)

    dd = getDataDictionary(redcap_url, redcap_api_key)

    variable_dict = getVariableType(dd)

    complete_day1 = pd.merge(form1, form3, on='subjid', how='left')

    df_converted = ia.homogenize_variables(complete_day1)
    elements = ['subjid', 'country'] + list(variable_dict['binary'])
    elements += list(variable_dict['number'])
    elements += list(variable_dict['categorical'])
    elements += oneHotEncoded
    elements = [col for col in elements if col in df_converted.columns]

    # AQUI no sigue los onehotencoded

    df_converted = df_converted[list(set(elements))]
    df_converted = df_converted.loc[:, ~df_converted.columns.duplicated()]

    for col in variable_dict['binary']:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].apply(
                lambda x: 1 if x == 'Yes' else (0 if x == 'No' else np.nan))

    for col in oneHotEncoded:
        if col in df_converted.columns:
            try:
                df_converted[col] = df_converted[col].apply(
                    lambda x: 1 if x == 'Checked' else (0 if x == 'Unchecked' else np.nan))
            except Exception:
                print(col)
                pass

    for col in variable_dict['number']:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')

    # df_converted['sex_orig'] = df_converted['demog_sex'].copy()
    df_converted = df_converted.assign(
        slider_sex=df_converted['demog_sex'],
        age=df_converted['demog_age'])
    df_converted.rename(columns={
        'outco_outcome': 'outcome_original'
    }, inplace=True)
    df_encoded = pd.get_dummies(
        df_converted,
        columns=[
            col for col in variable_dict['categorical']
            if col in df_converted.columns],
        prefix_sep='___')
    dummy_columns = [
        col for col in df_encoded.columns
        if any(cat in col for cat in variable_dict['categorical'])]
    remove_dummy_columns = [
        element for element in dummy_columns if (
            element.endswith('___No') or element.endswith('___no') or
            element.endswith('___NO') or element.endswith('___Never smoked'))]

    df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(float)
    # for d_c in dummy_columns:
    #    df_encoded[d_c] = pd.to_numeric(df_encoded[d_c], errors='coerce')
    #    df_encoded[d_c] = df_encoded[d_c].apply(
    #            lambda x: 1 if x  else (0 if x else np.nan))

    df_encoded = pd.merge(
        df_encoded, countries,
        left_on='country', right_on='Code', how='inner')
    df_encoded.rename(columns={
        'subjid': 'usubjid',
        # 'demog_age': 'age',
        # 'sex_orig': 'slider_sex',
        'country': 'country_iso',
        'Country': 'slider_country',
        'Region': 'region',
        'Income group': 'income',
        'outcome_original': 'outcome'}, inplace=True)
    df_encoded = df_encoded.drop(columns=remove_dummy_columns)

    try:
        df_encoded = df_encoded.drop(columns=['comor_hba1c'])
    except Exception:
        pass

    return df_encoded
