import requests
import pandas as pd
# import os
import IsaricAnalytics as ia
import numpy as np
import io


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

    # FIX: rename ARC variables (to be resolved in ARC v1.1)
    daily_section = [
        var for var in df['field_name'] if var.startswith('daily_data')]
    daily_section_new = [
        var.split('daily_data')[1] + '_dailydata' for var in daily_section]
    df.replace(
        {'field_name': dict(zip(daily_section, daily_section_new))},
        inplace=True)
    df.replace(
        {'field_name': {'daily_datalab': 'labs_dailydata'}}, inplace=True)
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


def get_missing_data_codes(redcap_url, redcap_api_key):
    data = {
        'token': redcap_api_key,
        'content': 'project',
        'format': 'csv',
        'returnFormat': 'csv'
    }
    r = requests.post(redcap_url, data=data).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8')))
    if df['missing_data_codes'].isna().all():
        missing_data_codes = dict()
    else:
        missing_data_codes = df['missing_data_codes'].values[0]
        missing_data_codes = dict(zip(
            ['Metadata ' + x.split(',')[1].strip()
                for x in missing_data_codes.split('|')],
            [x.split(',')[0].strip() for x in missing_data_codes.split('|')]))
    return missing_data_codes


def remove_MissingDataCodes(df, missing_data_codes):
    # for code in missing_data_codes.keys():
    #     drop_columns = [x for x in df.columns if x.endswith('___' + code)]
    #     df.drop(columns=drop_columns, inplace=True)

    with pd.option_context('future.no_silent_downcasting', True):
        df.replace('', np.nan, inplace=True)
        # df.replace(missing_data_codes.values(), np.nan, inplace=True)
        missing_data_labels = [
            x.split('Metadata ')[-1] for x in missing_data_codes.keys()]
        df.replace(missing_data_labels, np.nan, inplace=True)
    return df


# def remove_missing_data_columns(df, missing_data_codes):
#     for code in missing_data_codes.keys():
#         drop_columns = [
#             x for x in df.columns if x.endswith('___' + code)]
#         df.drop(columns=drop_columns, inplace=True)
#     return df


def missing_to_nan(df, missing_data_codes):
    missing_options = ['Unknown', 'Not known', 'nan']
    df = df.map(lambda x: np.nan if x in missing_options else x)
    missing_values = missing_options + list(missing_data_codes.keys())
    missing_columns = [
        col for col in df.columns if col.split('___')[-1] in missing_values]
    variables = set([col.split('___')[0] for col in missing_columns])
    for variable in variables:
        variable_columns = [
            col for col in df.columns if (col.split('___')[0] == variable)]
        variable_missing_columns = [
            col for col in variable_columns
            if col.split('___')[-1] in missing_values]
        nan_ind = df[variable_missing_columns].any(axis=1)
        df.loc[nan_ind, variable_columns] = np.nan
    df.drop(columns=missing_columns, inplace=True)
    return df


def remove_AnyAdditionalOther(df, variable_suffix='addi'):
    drop_columns = [x for x in df.columns if x.endswith(variable_suffix)]
    df.drop(columns=drop_columns, inplace=True)
    return df


def rename_checkbox_variables(df, dictionary, missing_data_codes=None):
    choices_dict = ia.get_choices_label_value_dict(dictionary)
    if missing_data_codes is None:
        n_choices = choices_dict.apply(len)
        values = sum([list(x.values()) for x in choices_dict], [])
        labels = sum([list(x.keys()) for x in choices_dict], [])
    else:
        n_choices = choices_dict.apply(len) + len(missing_data_codes)
        missing_values = list(missing_data_codes.values())
        values = sum(
            [list(x.values()) + missing_values for x in choices_dict], [])
        missing_labels = list(missing_data_codes.keys())
        labels = sum(
            [list(x.keys()) + missing_labels for x in choices_dict], [])
    names = list(np.repeat(dictionary['field_name'], n_choices))
    name_values = [x + '___' + str(y).lower() for x, y in zip(names, values)]
    name_labels = [x + '___' + y for x, y in zip(names, labels)]
    df.rename(columns=dict(zip(name_values, name_labels)), inplace=True)
    return df


def get_branching_logic_variables(branching_logic):
    '''Get all variables included in the branching logic
    (including checkboxes variables)'''
    var_names = [x.split(']')[0] for x in branching_logic.split('[')[1:]]
    # Change any checkbox variable from bracket form to its one-hot column name
    var_names = [x.replace('(', '___').replace(')', '') for x in var_names]
    return var_names


def resolve_checkbox_branching_logic(df, dictionary):
    checkbox_ind = dictionary.loc[(dictionary['field_type'] == 'checkbox')]
    branching_logic_variables = (
        dictionary['branching_logic'].apply(get_branching_logic_variables))
    for ind in checkbox_ind.index:
        branching_logic_columns = [
            col for col in df.columns
            if col in branching_logic_variables.loc[ind]]
        remove_ind = df[branching_logic_columns].isna().any(axis=1)
        checkbox_columns = [
            col for col in df.columns
            if (col.split('___')[0] == dictionary.loc[ind, 'field_name'])]
        df.loc[remove_ind, checkbox_columns] = np.nan
    return df


# def get_REDCAP_Single_DB(
#         redcap_url, redcap_api_key, site_mapping, required_variables):
#     contries_path = 'assets/countries.csv'
#     countries = pd.read_csv(contries_path, encoding='latin-1')
#     conex = {
#         'token': redcap_api_key,
#         'content': 'record',
#         'action': 'export',
#         'format': 'csv',
#         'type': 'flat',
#         'csvDelimiter': '',
#         'rawOrLabel': 'label',
#         'rawOrLabelHeaders': 'raw',
#         'exportCheckboxLabel': 'false',
#         'exportSurveyFields': 'false',
#         'exportDataAccessGroups': 'false',
#         'returnFormat': 'csv'
#     }
#     r = requests.post(redcap_url, data=conex)
#     print('HTTP Status: ' + str(r.status_code))
#     data = pd.read_csv(io.StringIO(r.content.decode('utf-8')), dtype='object')
#
#     dd = getDataDictionary(redcap_url, redcap_api_key)
#     variable_dict = getVariableType(dd)
#
#     missing_data_codes = get_missing_data_codes(redcap_url, redcap_api_key)
#
#     form1 = []
#     form2 = []
#     form3 = []
#     for row in data:
#         if (row['redcap_event_name'] == 'Initial Assessment / Admission'):
#             form1.append(row)
#         elif (row['redcap_event_name'] == 'Daily'):
#             form2.append(row)
#         elif (row['redcap_event_name'] == 'Outcome / End of study'):
#             form3.append(row)
#
#     form1 = pd.DataFrame(form1)
#     form2 = pd.DataFrame(form2)
#     form3 = pd.DataFrame(form3)
#
#     form1 = rename_checkbox_variables(form1, dd, missing_data_codes)
#     form2 = rename_checkbox_variables(form2, dd, missing_data_codes)
#     form3 = rename_checkbox_variables(form3, dd, missing_data_codes)
#
#     form1 = remove_MissingDataCodes(form1, missing_data_codes)
#     # form1 = remove_AnyAdditionalOther(form1)
#     form2 = remove_MissingDataCodes(form2, missing_data_codes)
#     # form2 = remove_AnyAdditionalOther(form2)
#     form3 = remove_MissingDataCodes(form3, missing_data_codes)
#     # form3 = remove_AnyAdditionalOther(form3)
#
#     ###
#     non_nan_columns_f1 = form1.columns[form1.notna().any()].tolist()
#     non_nan_columns_f2 = form2.columns[form2.notna().any()].tolist()
#     non_nan_columns_f3 = form3.columns[form3.notna().any()].tolist()
#
#     form1 = form1[non_nan_columns_f1]
#     form2 = form2[non_nan_columns_f2]
#     form3 = form3[non_nan_columns_f3]
#
#     oneHotEncoded = []
#
#     selected_columns_1 = []
#     selected_columns_2 = []
#     selected_columns_3 = []
#     for var in required_variables:
#         # Check for exact matches first
#         not_in_data_dict = True
#         if var in form1.columns:
#             selected_columns_1.append(var)
#             not_in_data_dict = False
#         if var in form2.columns:
#             selected_columns_2.append(var)
#             not_in_data_dict = False
#         if var in form3.columns:
#             selected_columns_3.append(var)
#             not_in_data_dict = False
#         if not_in_data_dict:
#             # # Check for any columns that start with the variable prefix
#             # selected_columns_1.extend([col for col in form1.columns if col.startswith(var)])
#             # selected_columns_2.extend([col for col in form2.columns if col.startswith(var)])
#             # selected_columns_3.extend([col for col in form3.columns if col.startswith(var)])
#             for col in form1:
#                 if col.startswith(var):
#                     if form1[col].eq('Unchecked').all():
#                         pass
#                     else:
#                         if ('___' in col):
#                             oneHotEncoded.append(col)
#                             selected_columns_1.append(col)
#             for col in form2:
#                 if (col.startswith(var) and ('___' not in col)):
#                     if form2[col].eq('Unchecked').all():
#                         pass
#                     else:
#                         if ('___' in col):
#                             oneHotEncoded.append(col)
#                             selected_columns_2.append(col)
#             for col in form3:
#                 if (col.startswith(var) and ('___' not in col)):
#                     if form3[col].eq('Unchecked').all():
#                         pass
#                     else:
#                         if ('___' in col):
#                             oneHotEncoded.append(col)
#                             selected_columns_3.append(col)
#
#     # Now filter form1 using the selected columns
#     form1 = form1[selected_columns_1]
#     form2 = form2[selected_columns_2]
#     form3 = form3[selected_columns_3]
#
#     ###
#
#     form1['country'] = form1['subjid'].str.split('-').str[0]
#     form1['country'] = form1['country'].map(site_mapping)
#
#     dates = form1[['subjid', 'dates_admdate']]
#
#     form2 = pd.merge(form2, dates, on='subjid', how='left')
#
#     # Ensure both columns are in datetime format
#     form2['daily_date'] = pd.to_datetime(form2['daily_date'], errors='coerce')
#
#     form2['dates_admdate'] = pd.to_datetime(
#         form2['dates_admdate'], errors='coerce')
#
#     # Calculate the difference in days and create a new column
#     form2['relative_day'] = (
#         form2['daily_date'] - form2['dates_admdate']).dt.days
#
#     form3 = pd.merge(form3, dates, on='subjid', how='left')
#     # Ensure both columns are in datetime format
#     form3['outco_date'] = pd.to_datetime(form3['outco_date'], errors='coerce')
#     form3['dates_admdate'] = pd.to_datetime(
#         form3['dates_admdate'], errors='coerce')
#
#     # Calculate the difference in days and create a new column
#     form3['outcome_day'] = (
#         form3['outco_date'] - form3['dates_admdate']).dt.days
#
#     form1 = ia.harmonizeAge(form1)
#     mapping_dict = {
#         'Female': 'Female',
#         'Male': 'Male'
#     }
#     form1['demog_sex'] = ia.map_variable(
#         form1['demog_sex'], mapping_dict, other_value_str='Other / Unknown')
#
#     form3 = form3.groupby('subjid').first().reset_index()
#     mapping_dict = {
#         'Discharged alive': 'Discharged',
#         'Discharged against medical advice': 'Discharged',
#         'Death': 'Death',
#     }
#     form3['outco_outcome'] = ia.map_variable(
#         form3['outco_outcome'], mapping_dict, other_value_str='Censored')
#
#     complete_day1 = pd.merge(form1, form3, on='subjid', how='left')
#
#     df_converted = ia.homogenize_variables(complete_day1)
#     elements = ['subjid', 'country'] + list(variable_dict['binary'])
#     elements += list(variable_dict['number'])
#     elements += list(variable_dict['categorical'])
#     elements += oneHotEncoded
#     elements = [col for col in elements if col in df_converted.columns]
#
#     # AQUI no sigue los onehotencoded
#
#     df_converted = df_converted[list(set(elements))]
#     df_converted = df_converted.loc[:, ~df_converted.columns.duplicated()]
#
#     for col in variable_dict['binary']:
#         if col in df_converted.columns:
#             df_converted[col] = df_converted[col].apply(
#                 lambda x: 1 if x == 'Yes' else (0 if x == 'No' else np.nan))
#
#     for col in oneHotEncoded:
#         if col in df_converted.columns:
#             try:
#                 df_converted[col] = df_converted[col].apply(
#                     lambda x: 1 if x == 'Checked' else (0 if x == 'Unchecked' else np.nan))
#             except Exception:
#                 print(col)
#                 pass
#
#     for col in variable_dict['number']:
#         if col in df_converted.columns:
#             df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
#
#     # df_converted['sex_orig'] = df_converted['demog_sex'].copy()
#     df_converted = df_converted.assign(
#         slider_sex=df_converted['demog_sex'],
#         age=df_converted['demog_age'])
#     df_converted.rename(columns={
#         'outco_outcome': 'outcome_original'
#     }, inplace=True)
#     df_encoded = pd.get_dummies(
#         df_converted,
#         columns=[
#             col for col in variable_dict['categorical']
#             if col in df_converted.columns],
#         prefix_sep='___', dummy_na=True)
#     dummy_columns = [
#         col for col in df_encoded.columns
#         if any(cat in col for cat in variable_dict['categorical'])]
#
#     df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(float)
#
#     df_encoded = pd.merge(
#         df_encoded, countries,
#         left_on='country', right_on='Code', how='inner')
#     df_encoded.rename(columns={
#         'subjid': 'usubjid',
#         'country': 'country_iso',
#         'Country': 'slider_country',
#         'Region': 'region',
#         'Income group': 'income',
#         'outcome_original': 'outcome'}, inplace=True)
#
#     remove_dummy_values = ['No', 'no', 'NO', 'Never smoked']
#     for value in remove_dummy_values:
#         remove_dummy_columns = [
#             element
#             for element in dummy_columns if element.endswith('___' + value)]
#         df_encoded = df_encoded.drop(columns=remove_dummy_columns)
#
#     try:
#         df_encoded = df_encoded.drop(columns=['comor_hba1c'])
#     except Exception:
#         pass
#
#     df_encoded = ia.missing_to_nan(df_encoded, missing_data_codes)
#
#     return df_encoded


def get_REDCAP_Single_DB(
        redcap_url, redcap_api_key, site_mapping, required_variables):
    if 'lesion_mpox_sl' not in required_variables:
        required_variables += ['lesion_mpox_sl']
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

    dd = getDataDictionary(redcap_url, redcap_api_key)
    variable_dict = getVariableType(dd)

    events = []
    for row in data:
        events.append(row['redcap_event_name'])
    events = list(set(events))

    form1 = []
    form2 = []
    form3 = []
    for row in data:
        if (row['redcap_event_name'] == 'Initial Assessment / Admission'):
            form1.append(row)
        elif (row['redcap_event_name'] == 'Daily'):
            form2.append(row)
        elif (row['redcap_event_name'] == 'Outcome / End of study'):
            form3.append(row)

    form1 = pd.DataFrame(form1)
    form2 = pd.DataFrame(form2)
    form3 = pd.DataFrame(form3)

    missing_data_codes = get_missing_data_codes(redcap_url, redcap_api_key)
    form1 = rename_checkbox_variables(form1, dd, missing_data_codes)
    form2 = rename_checkbox_variables(form2, dd, missing_data_codes)
    form3 = rename_checkbox_variables(form3, dd, missing_data_codes)

    form1 = remove_MissingDataCodes(form1, missing_data_codes)
    # form1 = remove_AnyAdditionalOther(form1)
    form2 = remove_MissingDataCodes(form2, missing_data_codes)
    # form2 = remove_AnyAdditionalOther(form2)
    form3 = remove_MissingDataCodes(form3, missing_data_codes)
    # form3 = remove_AnyAdditionalOther(form3)

    form1 = resolve_checkbox_branching_logic(form1, dd)
    form2 = resolve_checkbox_branching_logic(form2, dd)
    form3 = resolve_checkbox_branching_logic(form3, dd)

    # form1 = ia.missing_to_nan(form1, missing_data_codes)
    # form2 = ia.missing_to_nan(form2, missing_data_codes)
    # form3 = ia.missing_to_nan(form3, missing_data_codes)

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
    form2['asses_date'] = pd.to_datetime(form2['asses_date'], errors='coerce')

    form2['dates_admdate'] = pd.to_datetime(
        form2['dates_admdate'], errors='coerce')

    # Calculate the difference in days and create a new column
    form2['relative_day'] = (
        form2['asses_date'] - form2['dates_admdate']).dt.days

    form3 = pd.merge(form3, dates, on='subjid', how='left')
    # Ensure both columns are in datetime format
    form3['outco_date'] = pd.to_datetime(form3['outco_date'], errors='coerce')
    form3['dates_admdate'] = pd.to_datetime(
        form3['dates_admdate'], errors='coerce')

    # Calculate the difference in days and create a new column
    form3['outcome_day'] = (
        form3['outco_date'] - form3['dates_admdate']).dt.days

    form1 = ia.harmonizeAge(form1)
    mapping_dict = {
        'Female': 'Female',
        'Male': 'Male'
    }
    form1['demog_sex'] = ia.map_variable(
        form1['demog_sex'], mapping_dict, other_value_str='Other / Unknown')

    form3 = form3.groupby('subjid').first().reset_index()
    mapping_dict = {
        'Discharged alive': 'Discharged',
        'Discharged against medical advice': 'Discharged',
        'Death': 'Death',
    }
    form3['outco_outcome'] = ia.map_variable(
        form3['outco_outcome'], mapping_dict, other_value_str='Censored')

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
        prefix_sep='___', dummy_na=True)
    dummy_columns = [
        col for col in df_encoded.columns
        if any(cat in col for cat in variable_dict['categorical'])]

    df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(float)

    df_encoded = pd.merge(
        df_encoded, countries,
        left_on='country', right_on='Code', how='inner')
    df_encoded.rename(columns={
        'subjid': 'usubjid',
        'country': 'country_iso',
        'Country': 'slider_country',
        'Region': 'region',
        'Income group': 'income',
        'outcome_original': 'outcome'}, inplace=True)

    remove_dummy_values = ['No', 'no', 'NO', 'Never smoked']
    for value in remove_dummy_values:
        remove_dummy_columns = [
            element
            for element in dummy_columns if element.endswith('___' + value)]
        df_encoded = df_encoded.drop(columns=remove_dummy_columns)

    try:
        df_encoded = df_encoded.drop(columns=['comor_hba1c'])
    except Exception:
        pass

    df_encoded = missing_to_nan(df_encoded, missing_data_codes)
    # df_encoded = remove_missing_data_columns(df_encoded, missing_data_codes)

    severity_dict = {
        'None': 'Mild',
        '1': 'Mild',
        '2-5': 'Mild',
        '6-9': 'Mild',
        '10-24': 'Mild',
        '25-49': 'Moderate',
        '50-99': 'Moderate',
        '100-250': 'Severe',
        '251-1000': 'Critical',
        '>1000': 'Critical'}
    inclu_columns = [
        col for col in df_encoded.columns
        if col.startswith('lesion_mpox_sl___')]
    df_encoded['severity'] = ia.from_dummies(
            df_encoded[inclu_columns], 'lesion_mpox_sl', missing_val='Unknown'
        ).replace(severity_dict)

    return df_encoded
