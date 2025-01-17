import requests
import pandas as pd
import numpy as np
import io


############################################
# Functions that call to the API
############################################


def get_records(redcap_url, redcap_api_key):
    '''Fetch records from the REDCap API'''
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
        'exportDataAccessGroups': 'true',
        'returnFormat': 'json'
    }
    r = requests.post(redcap_url, data=conex)
    print('HTTP Status: ' + str(r.status_code))
    data = (r.json())
    data = pd.DataFrame(data)
    return data


def get_data_dictionary(redcap_url, redcap_api_key):
    '''Fetch the data dictionary from the REDCap API'''
    conex = {
        'token': redcap_api_key,
        'content': 'metadata',
        'format': 'json',
        'returnFormat': 'json'
    }
    # Make the API request
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):
        # Convert response JSON to DataFrame
        metadata = response.json()
        df = pd.DataFrame(metadata)
    else:
        df = None
    return df


def get_form_event(redcap_url, redcap_api_key):
    '''Get events, forms and their mapppings from the REDCap API and merge
    into a single dataframe.'''
    conex = {
        'token': redcap_api_key,
        'content': 'event',
        'format': 'json',
        'returnFormat': 'json'
    }
    # Make the API request
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):
        # Convert response JSON to DataFrame
        event = response.json()
        event = pd.DataFrame(event)
    else:
        event_columns = ['event_name', 'arm_num', 'unique_event_name']
        event_columns = event_columns + ['custom_event_label', 'event_id']
        event = pd.DataFrame(columns=event_columns)

    conex = {
        'token': redcap_api_key,
        'content': 'instrument',
        'format': 'json',
        'returnFormat': 'json'
    }
    # Make the API request
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):
        # Convert response JSON to DataFrame
        form = response.json()
        form = pd.DataFrame(form).rename(columns={
            'instrument_name': 'form', 'instrument_label': 'form_label'})
    else:
        form_columns = ['form', 'form_label']
        form = pd.DataFrame(columns=form_columns)

    conex = {
        'token': redcap_api_key,
        'content': 'formEventMapping',
        'format': 'json',
        'returnFormat': 'json'
    }
    # Make the API request
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):
        # Convert response JSON to DataFrame
        form_event = response.json()
        form_event = pd.DataFrame(form_event)
    else:
        form_event_columns = ['arm_num', 'unique_event_name', 'form']
        form_event = pd.DataFrame(columns=form_event_columns)

    form_event = pd.merge(form_event, form, on='form', how='left')
    form.rename(columns={'form': 'form_name'}, inplace=True)
    form_event.rename(columns={'form': 'form_name'}, inplace=True)
    form_event = form_event.groupby(['arm_num', 'unique_event_name']).agg(
        lambda x: ','.join(x)).reset_index()
    form_event = pd.merge(
        form_event, event, on=['unique_event_name', 'arm_num'], how='left')
    return form, form_event


def get_missing_data_codes(redcap_url, redcap_api_key):
    '''Get missing data codes from REDCAP API, using the project metadata'''
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
            [x.split(',')[1].strip() for x in missing_data_codes.split('|')],
            [x.split(',')[0].strip() for x in missing_data_codes.split('|')]))
    return missing_data_codes


############################################
# Functions for processing the data dictionary
############################################


def get_value(x):
    values = [y.split(',')[0] for y in x]
    return values


def get_label(x):
    labels = [','.join(y.split(',')[1:]).strip() for y in x]
    return labels


# def get_answer_dict(x, missing_data_codes=None):
#     answer_dict =
#     if (missing_data_codes is not None) & (len(answer_dict) > 0):
#         # Add 'Missing: ' string to missing data codes to distinguish with
#         # answers that have the same label (e.g. Unknown)
#         # Also, need lower case as REDCap automatically applies this to
#         # checkbox column names with missing data codes
#         missing_data_codes = {
#             'Missing: ' + k: v.lower() for k, v in missing_data_codes.items()}
#         answer_dict = {**answer_dict, **missing_data_codes}
#     return answer_dict


def add_answer_dict(dictionary):
    '''Add a lookup dict of labels/values to the dictionary from REDCap schema.
    By default, ignore Yes/No/Unknown radio variables.'''
    new_dictionary = dictionary.copy()
    # Get categories from dictionary
    answers = new_dictionary['select_choices_or_calculations'].copy()
    # This may throw an error if there are variables of type: slider or calc
    no_answers_ind = answers.fillna('').apply(
        lambda x: (len(x) > 0) & (x.count('|') == 0) & (x.count(',') == 0))
    yes_no_unknown_ind = answers.fillna('').apply(is_yesno)

    answers.loc[(no_answers_ind | yes_no_unknown_ind)] = np.nan
    answers = answers.str.rstrip('|,').str.split(r'\|').fillna('')
    answers = answers.apply(lambda x: [y.strip() for y in x])
    # This fixes the missing answers ind
    answers = answers.apply(lambda x: [y for y in x if y != ''])
    # answers = answers.apply(
    #     get_answer_dict, missing_data_codes=missing_data_codes)
    answers = answers.apply(lambda x: dict(zip(get_label(x), get_value(x))))
    answers.name = 'answer_dict'
    new_dictionary = pd.concat([new_dictionary, answers], axis=1)
    return new_dictionary


def list_categorical_onehot_columns(dictionary_row, data, sep='___'):
    variable = dictionary_row['field_name']
    answers = dictionary_row['answer_dict'].keys()
    output = [
        variable + sep + y for y in answers if y in data[variable].values]
    return output


def list_checkbox_onehot_columns(dictionary_row, data, sep='___'):
    variable = dictionary_row['field_name']
    answers = dictionary_row['answer_dict'].keys()
    columns = [variable + sep + x for x in answers]
    output = [col for col in columns if col in data.columns]
    return output


def get_section_prefix(x):
    output = x.split('_data')[-1] if x.startswith('daily') else x.split('_')[0]
    return output


def add_onehot_variables(data, dictionary, sep='___'):
    '''Add new rows to the dictionary for onehot-encoded categorical variables,
    using only the answers that exist within the data, e.g. if checkbox columns
    exist (after removing columns with only 'Unchecked') or if radio column
    answers are present for at least one subjid.'''
    new_dictionary = dictionary.copy()
    new_dictionary['parent'] = ''
    ind = new_dictionary['field_name'].str.contains('_')
    new_dictionary.loc[ind, 'parent'] = (
        new_dictionary.loc[ind, 'field_name'].apply(lambda x: x.split('_')[0]))

    ind = (new_dictionary['answer_dict'].apply(len) > 0)
    categorical_ind = new_dictionary['field_type'].isin(['radio', 'dropdown'])
    columns = ['field_name', 'answer_dict']
    new_variables = new_dictionary.loc[(ind & categorical_ind)].copy()
    new_variables.loc[:, 'field_name'] = new_variables[columns].apply(
        list_categorical_onehot_columns, data=data, sep=sep, axis=1)

    checkbox_ind = (new_dictionary['field_type'] == 'checkbox')
    # Retain items in answer dict only if they match
    add_new_variables = new_dictionary.loc[checkbox_ind].copy()
    add_new_variables.loc[:, 'field_name'] = new_dictionary[columns].apply(
        list_checkbox_onehot_columns, data=data, sep=sep, axis=1)
    new_variables = pd.concat([new_variables, add_new_variables], axis=0)

    # Add these onehot variables directly beneath the original categorical
    # variables in the dictionary
    new_variables = new_variables.reset_index()
    n_variables = new_variables['field_name'].apply(len)
    variable_list = sum(new_variables['field_name'].tolist(), [])
    new_variables = new_variables.loc[
        np.repeat(n_variables.index, n_variables)]
    new_variables.loc[:, 'field_name'] = variable_list
    new_variables['index'] += np.hstack(
        [np.linspace(0.1, 0.9, n) for n in n_variables])
    new_variables = new_variables.set_index('index')
    new_variables.index.name = None

    # Discard information about section header and choices
    empty_columns = ['section_header', 'select_choices_or_calculations']
    new_variables.loc[:, empty_columns] = ''
    new_variables.loc[:, 'text_validation_type_or_show_slider_number'] = ''
    new_variables.loc[:, 'field_type'] = 'binary'
    new_variables.loc[:, 'field_label'] = new_variables['field_name'].apply(
        lambda x: x.split('___')[-1])
    new_variables.loc[:, 'parent'] = new_variables['field_name'].apply(
        lambda x: x.split('___')[0])

    new_dictionary = pd.concat([new_dictionary, new_variables], axis=0)
    new_dictionary = new_dictionary.sort_index().reset_index(drop=True)
    # Can drop answer_dict column now
    new_dictionary.drop(columns='answer_dict', inplace=True)

    # Add section headers as new rows in the data dictionary
    ind = (new_dictionary['section_header'] != '')
    ind = ind.loc[ind].index
    sections = pd.DataFrame('', columns=new_dictionary.columns, index=ind)
    sections['field_label'] = new_dictionary.loc[ind, 'section_header'].apply(
        lambda x: x.split(':')[0])
    sections['field_type'] = 'section'
    sections['form_name'] = new_dictionary.loc[ind, 'form_name']
    sections['field_name'] = new_dictionary.loc[ind, 'field_name'].apply(
        get_section_prefix)
    sections.index -= 0.5
    new_dictionary = pd.concat([new_dictionary, sections], axis=0)
    new_dictionary = new_dictionary.sort_index().reset_index(drop=True)
    return new_dictionary


def is_yesno(x):
    '''Check if a Yes/No/Unknown question. Remove spaces in case of different
    versions of the same string.'''
    output = x.replace(' ', '') in ('1,Yes|0,No|99,Unknown', '1,Yes|0,No')
    # output = output.isin(['1,Yes|0,No|99,Unknown', '1,Yes|0,No'])
    return output


def convert_dictionary_field_type(dictionary):
    '''Get a dictionary of variable types, based on REDCAP structure'''
    new_dictionary = dictionary.copy()
    val_column = 'text_validation_type_or_show_slider_number'

    units_ind = new_dictionary['field_name'].str.endswith('_units')
    new_dictionary.loc[units_ind, 'field_type'] = 'units'

    binary_ind = (
        new_dictionary['field_type'].isin(['radio', 'dropdown']) &
        new_dictionary['select_choices_or_calculations'].apply(is_yesno))
    # Or truefalse/yesno types
    binary_ind |= (new_dictionary['field_type'].isin(['truefalse', 'yesno']))
    new_dictionary.loc[binary_ind, 'field_type'] = 'binary'
    # Discard answer options if they exist (if a Yes/No/Unknown radio)
    new_dictionary.loc[binary_ind, 'select_choices_or_calculations'] = ''

    date_ind = new_dictionary[val_column].isin(['date_dmy', 'datetime_dmy'])
    new_dictionary.loc[date_ind, 'field_type'] = 'date'

    numeric_ind = (
        new_dictionary[val_column].isin(['number', 'integer']) |
        new_dictionary['field_type'].isin(['slider']))
    new_dictionary.loc[numeric_ind, 'field_type'] = 'numeric'

    freetext_ind = (
        new_dictionary['field_type'].isin(['text', 'notes', 'descriptive']))
    new_dictionary.loc[freetext_ind, 'field_type'] = 'freetext'

    categorical_ind = (
        new_dictionary['field_type'].isin(['radio', 'dropdown']))
    new_dictionary.loc[categorical_ind, 'field_type'] = 'categorical'

    # # Leave checkbox variable field types alone
    # onehot_ind = (new_dictionary['field_type'] == 'checkbox')
    # new_dictionary.loc[onehot_ind, 'field_type'] = ''
    return new_dictionary


def replace_with_nan_for_missing_code_checkbox(df, missing_data_codes):
    '''Convert checkbox values to NaN when a missing code checkbox is 'Checked'
    '''
    missing_data_values = [x.lower() for x in missing_data_codes.values()]
    missing_columns = [
        col for col in df.columns
        if col.split('___')[-1] in missing_data_values]

    nan_mask = (df[missing_columns] == 'Checked').T.reset_index()
    nan_mask['index'] = nan_mask['index'].apply(lambda x: x.split('___')[0])
    nan_mask = nan_mask.groupby('index').any()

    columns = [
        col for col in df.columns if col.split('___')[0] in nan_mask.index]
    nan_mask = nan_mask.loc[[col.split('___')[0] for col in columns]]
    nan_mask['column'] = columns
    nan_mask = nan_mask.set_index('column').T

    df[nan_mask] = np.nan
    return df


############################################
# Functions for processing the data
############################################


def rename_checkbox_variables(df, dictionary):
    '''Rename checkbox variable columns. By default the suffix is their answer
    option value. Convert this answer option value to the answer option name.
    '''
    checkbox_ind = (dictionary['field_type'] == 'checkbox')
    answer_dict = dictionary.loc[checkbox_ind, 'answer_dict']

    n_answers = answer_dict.apply(len)
    values = sum([list(x.values()) for x in answer_dict], [])
    labels = sum([list(x.keys()) for x in answer_dict], [])

    names = list(
        np.repeat(dictionary.loc[checkbox_ind, 'field_name'], n_answers))
    # Need lower case because REDCap automatically applies this to missing
    # value codes, if they exist
    name_values = [x + '___' + y.lower() for x, y in zip(names, values)]
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
    '''By default, a cell is marked as 'Unchecked' in the absence of the
    positive, even if the question was not asked to the subjid. If the question
    was not asked to the subjid because of the branching logic, then set this
    to be NaN instead. This does not completely check the branching logic,
    which is a data quality issue!'''
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


def harmonise_age(df, age_columns=['demog_age', 'demog_age_units']):
    '''Deprecated, age should now be included in conversion_table.csv.
    Convert age from any units into age in years.'''
    df = df.rename(
        columns=dict(zip(age_columns, ['demog_age', 'demog_age_units'])))
    df.loc[:, 'demog_age'] = pd.to_numeric(df['demog_age'], errors='coerce')
    df.loc[:, 'demog_age'] = df['demog_age'].astype(float)
    df.loc[(df['demog_age_units'] == 'Months'), 'demog_age'] *= 1/12
    df.loc[(df['demog_age_units'] == 'Days'), 'demog_age'] *= 1/365
    unit_list = ['Days', 'Months', 'Years']
    # Standardize the units to 'Years'
    df.loc[df['demog_age_units'].isin(unit_list), 'demog_age_units'] = 'Years'
    return df


def map_variable(variable, mapping_dict, other_value_str='Other / Unknown'):
    '''Map a variable according to a dict. Any non-NaN value not in the dict
    keys is converted to other_value_str.'''
    other_value_ind = (
        (variable.isin(mapping_dict.keys()) == 0) & variable.notna())
    variable = variable.map(mapping_dict)
    variable.loc[other_value_ind] = other_value_str
    return variable


def homogenise_variables(df):
    '''
    Converts variables in a DataFrame based on a conversion table.

    Parameters:
    df: DataFrame containing values and their units.
    conversion_table: DataFrame containing conversion specifications.

    Returns:
    pd.DataFrame: DataFrame with all specified values converted to the
    desired units.
    '''
    conversion_table = pd.read_csv('assets/conversion_table.csv')
    for index, row in conversion_table.iterrows():
        from_unit = row['from_unit']
        to_unit = row['to_unit']
        value_col = row['variable']
        unit_col = row['variable_unit']
        conversion_factor = row['conversion_factor']

        try:
            # Ensure that the value column is numeric
            df.loc[:, value_col] = pd.to_numeric(
                df[value_col], errors='coerce')

            # Check if the variable is labs_lymphocyte or labs_neutrophil
            check_ind = (
                value_col in ['labs_lymphocyte', 'labs_neutrophil'] and
                from_unit == '10^9/L' and
                to_unit == '%')
            if check_ind:
                # Convert absolute count to percentage using total WBC count
                total_wbc_col = 'labs_wbccount'
                if total_wbc_col in df.columns:
                    # Ensure the total WBC count column is numeric
                    df.loc[:, total_wbc_col] = pd.to_numeric(
                        df[total_wbc_col], errors='coerce')
                    # Apply conversion only to non-empty values
                    mask = (
                        (df[unit_col] == from_unit) &
                        df[value_col].notna() &
                        df[total_wbc_col].notna())
                    df.loc[mask, value_col] = 100*(
                        df.loc[mask, value_col] / df.loc[mask, total_wbc_col])
                    df.loc[mask, unit_col] = to_unit
                continue

            # Only apply the conversion if the factor is not NaN and the
            # value_col is not empty
            if not pd.isna(conversion_factor):
                mask = (df[unit_col] == from_unit) & df[value_col].notna()
                # Apply the conversion
                df.loc[mask, value_col] *= conversion_factor

            # Set all units to the target unit
            df.loc[df[unit_col] == from_unit, unit_col] = to_unit
        except Exception:
            pass
    if 'demog_age' not in conversion_table['variable']:
        try:
            df = harmonise_age(df)
        except Exception:
            pass
    return df


def convert_onehot_to_binary(df, dictionary):
    '''Convert onehot-encoded columns to True/False/NaN and discard answers
    from the data dictionary, if they exist.'''
    binary_ind = (dictionary['field_type'] == 'binary')
    binary_columns = dictionary.loc[binary_ind, 'field_name'].tolist()
    binary_columns = [col for col in binary_columns if col in df.columns]
    mapping_dict = {
        'Yes': True, 'Checked': True,
        'No': False, 'Unchecked': False, 'Unknown': np.nan}
    with pd.option_context('future.no_silent_downcasting', True):
        df.loc[:, binary_columns] = df[binary_columns].replace(mapping_dict)
    return df


############################################
# Main functions
############################################


def initial_data_processing(data, dictionary, missing_data_codes):
    '''Initial processing of complete pandas dataframe, after REDCAP API call
    '''

    # Replace empty cells or 'Unknown' with NaN
    with pd.option_context('future.no_silent_downcasting', True):
        data = data.replace(['', 'Unknown', 'unknown'], np.nan)
    # Replace missing data codes with NaN
    if missing_data_codes is not None:
        with pd.option_context('future.no_silent_downcasting', True):
            data = data.replace(list(missing_data_codes.keys()), np.nan)

    # Replace values in checkbox variables with NaN if the checkbox missing
    # data code column is 'Checked'
    data = replace_with_nan_for_missing_code_checkbox(data, missing_data_codes)

    # Remove columns where all the data is a negative answer option
    # (or missing answer)
    remove_values = ['', 'no', 'never smoked', 'unchecked', 'nan']
    remove_values += [x.lower() for x in missing_data_codes.keys()]
    remove_columns = data.columns[data.astype(str).map(
        lambda x: (x.lower().strip() in remove_values)).all(axis=0)]
    data = data[[col for col in data.columns if col not in remove_columns]]

    # Convert 'Unchecked' to NaN when a checkbox question wasn't asked
    # to a subjid because of their previous answers (i.e. the branching logic)
    # TODO: this needs updating based on branching logic values,
    # not just the variables themselves
    data = resolve_checkbox_branching_logic(data, dictionary)

    # Add a python dict of choice options to the dictionary
    new_dictionary = dictionary.copy()
    # Remove rows corresponding to the deleted columns of the data (ignore
    # checkbox columns here)
    remove_variables = [
        x for x in remove_columns.map(lambda x: x.split('___')[0])
        if x not in data.columns.map(lambda x: x.split('___')[0])]
    new_dictionary = new_dictionary.loc[(
        new_dictionary['field_name'].isin(remove_variables) == 0)]
    new_dictionary = new_dictionary.reset_index(drop=True)
    new_dictionary = add_answer_dict(new_dictionary)

    # Rename checkbox variables
    data = rename_checkbox_variables(data, new_dictionary)

    # Convert the REDCap field types and add new onehot-encoded categorical
    # variables to the dictionary (as they will be onehot-encoded in
    # descriptive analysis), without onehot-encoding these yet (because this
    # may affect imputation etc.)
    new_dictionary = add_onehot_variables(data, new_dictionary)
    new_dictionary = convert_dictionary_field_type(new_dictionary)

    columns = [
        'field_name', 'form_name', 'field_type', 'field_label', 'parent']
    new_dictionary = new_dictionary[columns]

    # Convert Yes(Checked)/No(Unchecked)/Unknown to True/False/NaN
    data = convert_onehot_to_binary(data, new_dictionary)

    # Convert numerical data to numeric type and homogenise if mixed units
    numeric_ind = (new_dictionary['field_type'] == 'numeric')
    numeric_columns = new_dictionary.loc[numeric_ind, 'field_name'].tolist()
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors='coerce')
    data = homogenise_variables(data)

    # Convert columns with dates into datetime
    date_ind = (new_dictionary['field_type'] == 'date')
    date_columns = new_dictionary.loc[date_ind, 'field_name'].tolist()
    data[date_columns] = data[date_columns].apply(
        pd.to_datetime, errors='coerce')

    # # ## TODO: Eventually move this to ISARIC Analytics instead of here?
    # # Remove columns with no data
    # data = data.dropna(axis=1, how='all')
    #
    # new_dictionary = new_dictionary.loc[(
    #     new_dictionary['field_name'].isin(remove_columns) == 0)]
    # new_dictionary = new_dictionary.reset_index(drop=True)
    return data, new_dictionary


def get_df_map(data, dictionary):
    '''Convert single-event rows into one row per patient.'''
    df_map = data.copy()
    forms = ['presentation', 'daily', 'outcome']
    columns = dictionary.loc[
        dictionary['form_name'].isin(forms), 'field_name'].tolist()
    columns = [col for col in columns if col in df_map.columns]
    ind = data['form_name'].apply(
        lambda x: any(y in x.split(',') for y in ['presentation', 'outcome']))
    df_map = df_map.loc[ind, columns]
    # # ## TODO: Should this remove all columns with no data, or just those
    # # ## from repeating events?
    # df_map = df_map.dropna(axis=1, how='all')
    '''
    # Check non-overlapping variables
    test = data.groupby('subjid').apply(
            lambda x: x.notna().sum(), include_groups=False
        ).drop(columns='redcap_event_name').isin([0, 1]).all(axis=None)
    error = 'At least one variable exists in several non-repeating forms'
    assert test, error
    '''
    # Merge into one row per subjid
    df_map = df_map.set_index('subjid').groupby(level=0).bfill()
    df_map = df_map.drop(
        columns=[col for col in df_map.columns if 'redcap' in col])
    df_map = df_map.reset_index().drop_duplicates('subjid')
    df_map = df_map.reset_index(drop=True)

    other_value_ind = (df_map['demog_sex'].isin(['Male', 'Female']) == 0)
    df_map.loc[other_value_ind, 'demog_sex'] = 'Other / Unknown'

    mapping_dict = {
        'Discharged alive': 'Discharged',
        'Discharged against medical advice': 'Discharged',
        'Death': 'Death',
    }
    df_map['outco_binary_outcome'] = map_variable(
        df_map['outco_outcome'],
        mapping_dict, other_value_str='Censored')
    outcome_dict = {}
    outcomes = ['Death', 'Discharged', 'Censored']
    outcome_dict['field_name'] = ['outco_binary_outcome']
    outcome_dict['field_name'] += [
        'outco_binary_outcome___' + x for x in outcomes]
    outcome_dict['form_name'] = 'outcome'
    outcome_dict['field_type'] = ['categorical'] + ['binary']*len(outcomes)
    outcome_dict['field_label'] = ['Outcome (binary)'] + outcomes
    outcome_dict['parent'] = ['outco'] + ['outco_binary_outcome']*len(outcomes)
    dictionary = pd.concat([
        dictionary, pd.DataFrame.from_dict(outcome_dict)], axis=0)
    dictionary = dictionary.reset_index(drop=True)
    return df_map, dictionary


def get_df_forms(data, dictionary):
    forms = dictionary['form_name'].unique()
    df_forms_dict = {}
    for form in forms:
        columns = dictionary.loc[
            dictionary['form_name'] == form, 'field_name'].tolist()
        columns = [col for col in columns if col in data.columns]
        if 'subjid' not in columns:
            columns = ['subjid'] + columns
        ind = data['form_name'].apply(lambda x: form in x.split(','))
        df_forms_dict[form] = data.loc[ind, columns].reset_index(drop=True)
    return df_forms_dict


def get_redcap_data(redcap_url, redcap_api_key, country_mapping=None):
    '''Get data from REDCap API and transform into analysis-ready dataframes'''
    data = get_records(redcap_url, redcap_api_key)
    dictionary = get_data_dictionary(redcap_url, redcap_api_key)
    missing_data_codes = get_missing_data_codes(redcap_url, redcap_api_key)

    data, new_dictionary = initial_data_processing(
        data, dictionary, missing_data_codes)

    redcap_columns = ['redcap_event_name', 'redcap_repeat_instrument']
    redcap_columns += ['redcap_repeat_instance', 'redcap_data_access_group']
    redcap_columns = [col for col in redcap_columns if col not in data.columns]
    data = pd.concat([data, pd.DataFrame(columns=redcap_columns)], axis=1)

    # Get forms and events from the API
    form, form_event = get_form_event(redcap_url, redcap_api_key)
    # Convert repeating forms from label to name
    form_dict = dict(zip(form['form_label'], form['form_name']))
    data.loc[:, 'form_name'] = data['redcap_repeat_instrument'].map(form_dict)
    # Else convert events into a string-delimited str of forms
    form_dict = dict(zip(form_event['event_name'], form_event['form_name']))
    data.loc[data['form_name'].isna(), 'form_name'] = (
        data.loc[data['form_name'].isna(), 'redcap_event_name'].map(form_dict))
    data = data.loc[data['form_name'].notna()].reset_index(drop=True)
    df_map, new_dictionary = get_df_map(data, new_dictionary)
    df_forms_dict = get_df_forms(data, new_dictionary)

    if country_mapping is None:
        dag = data[['subjid', 'redcap_data_access_group']].drop_duplicates()
        dag = dag.rename(columns={'redcap_data_access_group': 'site'})
        dag['country_iso'] = dag['site'].apply(lambda x: x.split('-')[1])
        df_map = pd.merge(df_map, dag, on='subjid', how='left')
    else:
        # TODO: Need something better here?
        try:
            df_map['country_iso'] = df_map['subjid'].str.split('-').str[0]
            df_map['country_iso'] = df_map['country'].map(country_mapping)
        except Exception:
            df_map['country_iso'] = np.nan
    # Add country_iso to dictionary
    countries = df_map['country_iso'].drop_duplicates().tolist()
    country_dict = {}
    country_dict['field_name'] = ['country', 'country_iso']
    country_dict['field_name'] += ['country_iso___' + x for x in countries]
    country_dict['form_name'] = 'presentation'
    country_dict['field_type'] = ['section', 'categorical']
    country_dict['field_type'] += ['binary']*len(countries)
    country_dict['field_label'] = ['COUNTRY', 'Country ISO Code'] + countries
    country_dict['parent'] = ['', 'country'] + ['country_iso']*len(countries)

    new_dictionary = pd.concat([
        new_dictionary, pd.DataFrame.from_dict(country_dict)], axis=0)
    new_dictionary = new_dictionary.reset_index(drop=True)

    return df_map, df_forms_dict, new_dictionary


# def convert_fixed_date_events_to_repeating_event(df, event_prefix='Day '):
#     new_columns = ['days_since_adm']
#     if 'redcap_repeat_instance' not in df.columns:
#         new_columns += ['redcap_repeat_instance']
#     add_columns = pd.DataFrame(columns=new_columns, dtype='object')
#     df = pd.concat([df, add_columns], axis=1)
#     # Get all rows that are the event_prefix followed by an integer
#     ind = (
#         df['redcap_event_name'].str.startswith(event_prefix) &
#         df['redcap_event_name'].apply(
#             lambda x: x.split(event_prefix)[-1].isdigit()))
#     df.loc[ind, 'days_since_adm'] = df.loc[ind, 'redcap_event_name'].apply(
#         lambda x: int(x.split(event_prefix)[-1]))
#     df.loc[ind, 'redcap_repeat_instance'] = sum(
#         df.loc[ind].groupby('record_id').apply(
#             lambda x: list(range(1, 1 + len(x))), include_groups=False), [])
#     return df

# def remove_empty_checkbox_variables(
#         df, dictionary,
#         missing_data_codes=None, remove_all_unchecked_columns=True):
#     '''Remove any checkbox variable columns where all entries are marked as
#     'Unchecked', including those added by REDCap for missing data codes.'''
#     # checkbox
#     # if remove_all_unchecked
#     if missing_data_codes is None:
#         remove_columns = []
#     else:
#         checkbox_columns = dictionary.loc[(
#             dictionary['field_type'] == 'checkbox'), 'field_name'].tolist()
#         missing_labels = ['Missing: ' + x for x in missing_data_codes.keys()]
#         labels = list(np.tile(missing_labels, len(checkbox_columns)))
#         names = list(np.repeat(checkbox_columns, len(missing_labels)))
#         columns = [x + '___' + y for x, y in zip(names, labels)]
#         remove_columns = [
#             col for col in df.columns
#             if (col in columns) & (df[col] == 'Unchecked').all()]
#     if remove_all_unchecked_columns:
#         remove_columns = [
#             col for col in onehot_columns if df[col].astype(str).map(
#                 lambda x: (x.lower() in ['unchecked', 'nan'])).all(axis=0)]
#         df = df[[col for col in df.columns if col not in remove_columns]]
#     df = df.drop(columns=remove_columns)
#     return df
