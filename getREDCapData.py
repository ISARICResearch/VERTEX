#!/usr/bin/env python3
"""Retrieve data from REDCap API and process into ISARIC VERTEX schema.

This module retrieves records, data dictionary and project structure
from an REDCap API (given the url and key). It processes the data,
including converting data types and harmonising across units, then
creates two analysis-ready data files (one wide and one long) and a
dictionary file. These are designed to be used with ISARIC data
analysis tools.
"""

__authors__ = 'Tom Edinburgh, Esteban Garcia-Gallo'
__credits__ = ['Esteban Garcia-Gallo', 'Tom Edinburgh']
__license__ = 'CC-BY-SA-4.0'
__version__ = '1.1.0'
__maintainer__ = 'Tom Edinburgh'
__email__ = 'data@isaric.org'

import requests
import pandas as pd
import numpy as np
import io
import os

# =============================================================================
# Functions that call to the API
# =============================================================================


def user_assigned_to_dag(redcap_url, redcap_api_key):
    """Determine if the user is assigned to a REDCap DAG.

    Request REDCap Data Access Group (DAG) list. If a user is assigned
    to a DAG, they will not be able to export this, returning a 403
    code.

    Args:
        redcap_url: str
        redcap_api_key: str

    Returns:
        dag_assigned: bool
    """

    conex = {
        'token': redcap_api_key,
        'content': 'dag',
        'format': 'csv',
        'returnFormat': 'json'
    }
    response = requests.post(redcap_url, data=conex)
    # HTTP 403 means request is understood (and user authenticated)
    # but the user does not have the necessary permissions to access
    # the requested data. This indicates that the user is assigned
    # to a DAG.
    dag_assigned = (response.status_code == 403)
    return dag_assigned


def get_records(
        redcap_url,
        redcap_api_key,
        data_access_groups=None,
        user_assigned_to_dag=False):
    """Fetch records from the REDCap API.

    If a user is assigned to a Data Access Group (DAG), it will
    also perform DAG switching via the API in order the records
    by one DAG at a time.

    Args:
        redcap_url: str
        redcap_api_key: str
        data_access_groups: list | None
        user_assigned_to_dag: bool

    Returns:
        df: pd.DataFrame
    """

    if (data_access_groups is None) or (user_assigned_to_dag is False):
        conex = {
            'token': redcap_api_key,
            'content': 'record',
            'action': 'export',
            'format': 'csv',
            'type': 'flat',
            'csvDelimiter': '',
            'rawOrLabel': 'label',
            'rawOrLabelHeaders': 'raw',
            'exportCheckboxLabel': 'false',
            'exportSurveyFields': 'false',
            'exportDataAccessGroups': 'true',
            'returnFormat': 'json'
        }
        response = requests.post(redcap_url, data=conex)
        print('HTTP Status: ' + str(response.status_code))
        df = pd.read_csv(io.StringIO(response.text), keep_default_na=False)

        # Retain only data from the specified Data Access Groups.
        if data_access_groups is not None:
            ind = df['redcap_data_access_group'].isin(data_access_groups)
            df = df.loc[ind].reset_index(drop=True)

    else:
        # Retrieve data for each DAG one at a time
        df_list = []

        for dag in data_access_groups:
            # Imitate the format of REDCap's unique group name for the DAG
            # NOTE: REDCap automatically assigns these, when a unique group is
            # duplicated it will automatically sequentially add an alphabetic
            # characters to the end. It's not practical to perform this here,
            # so we are assuming the DAGs are named sensibly.
            unique_group = dag.replace('-', '').replace(' ', '_').lower()[:18]

            conex = {
                'token': redcap_api_key,
                'content': 'dag',
                'action': 'switch',
                'dag': unique_group,
                'returnFormat': 'json'
            }
            response = requests.post(redcap_url, data=conex)

            conex = {
                'token': redcap_api_key,
                'content': 'record',
                'action': 'export',
                'format': 'csv',
                'type': 'flat',
                'csvDelimiter': '',
                'rawOrLabel': 'label',
                'rawOrLabelHeaders': 'raw',
                'exportCheckboxLabel': 'false',
                'exportSurveyFields': 'false',
                'exportDataAccessGroups': 'false',
                'returnFormat': 'json'
            }
            try:
                response = requests.post(redcap_url, data=conex)
                df_new = pd.read_csv(
                    io.StringIO(response.text), keep_default_na=False)
                df_new['redcap_data_access_group'] = dag
                df_list.append(df_new)
                print(f"""Data access group ID: {dag}, HTTP Status: \
{response.status_code}""")
            except pd.errors.EmptyDataError:
                print(f"""Data access group ID: {dag}, HTTP Status: \
{response.status_code}. Warning: Could not retrieve data from unique \
group name: {unique_group}""")
                continue

        # Combine the retrieved data for successful requests
        if len(df_list) > 0:
            df = pd.concat(df_list, axis=0)
        else:
            df = None

    return df


def get_data_dictionary(redcap_url, redcap_api_key):
    """Fetch the data dictionary from the REDCap API.

    Args:
        redcap_url: str
        redcap_api_key: str

    Returns:
        dictionary: bool
    """

    conex = {
        'token': redcap_api_key,
        'content': 'metadata',
        'format': 'csv',
        'returnFormat': 'json'
    }
    # Make the API request
    response = requests.post(redcap_url, data=conex)
    dictionary = pd.read_csv(io.StringIO(response.text), keep_default_na=False)
    return dictionary


def get_form_event(redcap_url, redcap_api_key):
    """Get events, forms and their mapppings from the REDCap API and
    merge into a single dataframe.

    Args:
        redcap_url: str
        redcap_api_key: str

    Returns:
        form: pd.DataFrame
        form_event: pd.DataFrame
    """

    # Retrieve events
    conex = {
        'token': redcap_api_key,
        'content': 'event',
        'format': 'csv',
        'returnFormat': 'json'
    }
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):  # HTTP 200 is a successful request
        event = pd.read_csv(io.StringIO(response.text), keep_default_na=False)
    else:
        # If unsuccessful, create an empty dataframe with the same columns
        event_columns = [
            'event_name',
            'arm_num',
            'unique_event_name'
            'custom_event_label',
            'event_id'
        ]
        event = pd.DataFrame(columns=event_columns)

    # Retrieve forms (instruments)
    conex = {
        'token': redcap_api_key,
        'content': 'instrument',
        'format': 'csv',
        'returnFormat': 'json'
    }
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):  # HTTP 200 is a successful request
        form = pd.read_csv(io.StringIO(response.text), keep_default_na=False)
        form = form.rename(columns={
            'instrument_name': 'form_name',
            'instrument_label': 'form_label'
        })
    else:
        # If unsuccessful, create an empty dataframe with the same columns
        form_columns = ['form_name', 'form_label']
        form = pd.DataFrame(columns=form_columns)

    # Retrieve mapping between forms and events
    conex = {
        'token': redcap_api_key,
        'content': 'formEventMapping',
        'format': 'csv',
        'returnFormat': 'json'
    }
    response = requests.post(redcap_url, data=conex)
    if (response.status_code == 200):  # HTTP 200 is a successful request
        form_event = pd.read_csv(
            io.StringIO(response.text), keep_default_na=False)
        form_event = form_event.rename(columns={'form': 'form_name'})
    else:
        # If unsuccessful, create an empty dataframe with the same columns
        form_event_columns = ['arm_num', 'unique_event_name', 'form_name']
        form_event = pd.DataFrame(columns=form_event_columns)

    # Merge the dataframes
    form_event = pd.merge(form_event, form, on='form_name', how='left')
    form_event = form_event.groupby(['arm_num', 'unique_event_name']).agg(
        lambda x: ','.join(x)).reset_index()

    form_event = pd.merge(
        form_event, event,
        on=['unique_event_name', 'arm_num'],
        how='left'
    )
    return form, form_event


def get_missing_data_codes(redcap_url, redcap_api_key):
    """Get missing data codes from REDCAP API, using the project metadata.

    Args:
        redcap_url: str
        redcap_api_key: str

    Returns:
        missing_data_codes: dict
    """

    conex = {
        'token': redcap_api_key,
        'content': 'project',
        'format': 'csv',
        'returnFormat': 'json'
    }
    response = requests.post(redcap_url, data=conex)
    df = pd.read_csv(io.StringIO(response.text), keep_default_na=False)

    # Create a dict for the missing data codes
    if df['missing_data_codes'].isna().all():
        missing_data_codes = {}
    else:
        missing_data_codes = df['missing_data_codes'].values[0]
        labels = [
            x.split(',')[1].strip() for x in missing_data_codes.split('|')]
        codes = [
            x.split(',')[0].strip() for x in missing_data_codes.split('|')]
        missing_data_codes = dict(zip(labels, codes))

    return missing_data_codes


# =============================================================================
# Functions for processing the data dictionary
# =============================================================================


def int_to_alphabet(x):
    """Convert an integer to an alphabetic string.

    Adds extra characters to the string when it runs out of
    latin alphabet letters (e.g. similar to Excel column names).

    Args:
        x: int

    Returns:
        string: str
    """

    MAX_CHAR = 26

    # Shift integer by 1, to start at 0 -> 'a'
    alpha_string = chr(ord('`') + (x - 1) % MAX_CHAR + 1)
    while (x > MAX_CHAR):
        x = x // MAX_CHAR
        alpha_string = chr(ord('`') + (x - 1) % MAX_CHAR + 1) + alpha_string

    return alpha_string


def format_field_name(field_name):
    """Format string for renaming columns.

    Converts to lower case and removes non-alphanumeric characters
    and repeated spaces.

    Args:
        field_name: str

    Returns:
        new_field_name: str
    """

    # Follow ARC convention for non-alphanumeric characters
    field_name = field_name.replace('%', 'pcnt').replace('µ', 'u')

    updated_field_name = '_'.join([
        ''.join([z for z in y if z.isalnum() or (z in ('_'))])
        for y in field_name.lower().replace('-', '_').split(' ') if y != ''])

    return updated_field_name


def add_number_to_duplicated_columns(columns):
    """Add a suffix to duplicated columns.

    This is necessary when re-formatting would otherwise result in
    duplicated column names, e.g. when two variables were named the
    identically apart from upper/lower case (then both formatted to
    lower case).

    Args:
        columns: list

    Returns:
        columns_with_suffix: list
    """

    columns = pd.Series(columns).sort_values()

    with pd.option_context('future.no_silent_downcasting', True):
        column_duplicated = columns.duplicated().astype(int)
        number = column_duplicated.diff().fillna(0).astype(int)

    number = number.apply(lambda x: '_' + str(x) if x > 0 else '')
    columns_with_suffix = (columns + number).sort_index().tolist()

    return columns_with_suffix


def is_yesno(x):
    """Check if a Yes/No/Unknown question. Remove spaces in case of
    different versions of the same string.

    Args:
        x: str

    Returns:
        yesno_bool: bool
    """

    yesno_values = ('1,yes|0,no|99,unknown', '1,yes|0,no')
    yesno_bool = (x.replace(' ', '').lower() in yesno_values)
    return yesno_bool


def get_answer_dict(redcap_dictionary, data=None):
    """Create a dict of the mapping from label to value for each field.

    Based on the REDCap schema data dictionary. Exclude Yes/No/Unknown
    variables.

    Args:
        redcap_dictionary: pd.DataFrame
        data: pd.DataFrame | None

    Returns:
        answer_dict: dict
    """

    ind = (
        (redcap_dictionary['select_choices_or_calculations'].fillna('').apply(
            is_yesno) == 0) &
        redcap_dictionary['field_type'].isin(['radio', 'checkbox', 'dropdown'])
    )

    # For each field, create a label-value dict from REDCap-formatted
    # "value1, label1 | value2, label 2 | ..." string
    answer_dict = {
        x['field_name']: {
            ','.join(z.split(',')[1:]).strip(): z.split(',')[0].strip()
            for z in x['select_choices_or_calculations'].split('|')
        }
        for _, x in redcap_dictionary.loc[ind].iterrows()}

    # Exclude answer options that don't appear in the data
    if data is not None:
        answer_dict = {
            k: {
                x: y for x, y in v.items()
                if ((k not in data.columns) or (x in data[k].values))
            }
            for k, v in answer_dict.items()
        }

    return answer_dict


def convert_field_type(redcap_dictionary):
    """Convert the REDCap variable types into ISARIC VERTEX schema field types.

    Assumes specific REDCap and ISARIC ARC/BRIDGE structure (for units).

    Args:
        redcap_dictionary: pd.DataFrame

    Returns:
        field_types: pd.Series
    """

    field_types = redcap_dictionary['field_type'].copy()

    # Units variables (will be removed after homogenising units)
    units_ind = redcap_dictionary['field_name'].str.endswith('_units')
    field_types.loc[units_ind] = 'units'

    # Binary/boolean variables
    binary_ind = (
        redcap_dictionary['field_type'].isin(['radio', 'dropdown']) &
        redcap_dictionary['select_choices_or_calculations'].apply(is_yesno)
    ) | (
        redcap_dictionary['field_type'].isin(['truefalse', 'yesno'])
    )
    field_types.loc[binary_ind] = 'bool'

    # Date/datetime variables
    val_column = 'text_validation_type_or_show_slider_number'
    date_ind = redcap_dictionary[val_column].isin(['date_dmy', 'datetime_dmy'])
    field_types.loc[date_ind] = 'datetime'

    # Numeric variables
    val_column = 'text_validation_type_or_show_slider_number'
    numeric_ind = (
        redcap_dictionary[val_column].isin(['number', 'integer']) |
        redcap_dictionary['field_type'].isin(['slider']))
    field_types.loc[numeric_ind] = 'number'

    # String variables
    # Retrieve 'string' from field_types after mapping 'number',
    # since REDCap exports numeric variables with a field_type 'text'
    freetext_ind = field_types.isin(['text', 'notes', 'descriptive'])
    field_types.loc[freetext_ind] = 'string'

    # Categorical variables (exlcuding units)
    # Retrieve 'categorical' from field_types after mapping 'units',
    # since these are radio questions (which we need to distinguish later)
    categorical_ind = field_types.isin(['radio', 'dropdown'])
    field_types.loc[categorical_ind] = 'categorical'

    # Leave checkbox variable field types alone
    checkbox_ind = (field_types == 'checkbox')
    field_types.loc[checkbox_ind, 'field_type'] = 'checkbox'
    return field_types


# =============================================================================
# Functions for processing the data
# =============================================================================


def replace_with_nan_for_missing_code_checkbox(
        df, missing_data_codes, sep='___'):
    """Impute NaN values for checkbox variables with missing data codes.

    Converts all checkbox values to NaN when the checkbox option
    corresponding to a missing data code is 'Checked'.

    Args:
        df: pd.DataFrame
        missing_data_codes: dict
        sep: str (default: '___')

    Returns:
        df: pd.DataFrame
    """

    # Get columns that contain missing data codes as an answer option
    # (following REDCap format)
    missing_data_values = [x.lower() for x in missing_data_codes.values()]
    missing_columns = [
        col for col in df.columns
        if col.split(sep)[-1] in missing_data_values]

    # Create mask (transposed) for missing data code checkbox options
    nan_mask = (df[missing_columns] == 'Checked').T.reset_index()
    nan_mask['index'] = nan_mask['index'].apply(lambda x: x.split(sep)[0])
    nan_mask = nan_mask.groupby('index').any()

    # Expand
    columns = [
        col for col in df.columns if col.split(sep)[0] in nan_mask.index]
    nan_mask = nan_mask.loc[[col.split(sep)[0] for col in columns]]
    nan_mask['column'] = columns
    nan_mask = nan_mask.set_index('column').T

    # Replace missing checkbox values with NaN instead of 'Unchecked'
    df[nan_mask] = np.nan

    return df


def combine_unlisted_variables(df, dictionary, sep='___'):
    """Combine 'unlisted' variables, as defined in ISARIC ARC/BRIDGE.

    ISARIC BRIDGE adds multiple 'unlisted' variables to allow the
    user to record additional categorical or freetext options that
    are not directly asked in a CRF (e.g. symptoms that were unknown
    when the CRF was created). This creates sets of 3 variables
    that are named {section}_unlisted_{n}item (radio/categorical),
    {section}_unlisted_{n}otherl2 (free text) and
    {section}_unlisted_{n}addi (Y/N/Unk for asking another set of 3),
    where {section} is the section of the additional variables, e.g.
    'sympt' (for symptoms), and {n} is an integer between 0 and 4
    (max allowed repetitions in BRIDGE) to differentiate between
    repeated 'unlisted' questions. The radio/categorical questions
    need to be combined into separate one-hot-encoded variables for
    categorical answer option that appears in the data. Note that
    this is very specific to REDCap databases using ISARIC
    ARC/BRIDGE structure/format.

    Args:
        df: pd.DataFrame
        dictionary: pd.DataFrame
        sep: str (default: '___')

    Returns:
        df: pd.DataFrame
        dictionary: pd.DataFrame
    """

    # Retrieve any 'unlisted' variables
    unlisted_ind = dictionary['field_name'].str.endswith('unlisted')
    unlisted_columns = dictionary.loc[unlisted_ind, 'field_name']

    # Retrieve 'unlisted_item' variables (radio)
    unlisted_item_ind = dictionary['field_name'].apply(
        lambda x: ''.join(
            [y for y in x if y.isalpha()]).endswith('unlisted_item'))
    unlisted_item_columns = dictionary.loc[unlisted_item_ind, 'field_name']
    unlisted_item_columns = [
        col for col in unlisted_item_columns if col in df.columns]

    # Create a dict of unlisted items
    unlisted_columns_dict = {
        k: [v for v in unlisted_item_columns if k in v]
        for k in unlisted_columns}

    new_dictionary_list = []
    for ind in unlisted_columns.index:
        # Get all values in the data within these columns
        column = dictionary.loc[ind, 'field_name']
        values = df[unlisted_columns_dict[column]].stack().unique()
        values = [val for val in values if val not in (np.nan, '', 'Other')]

        # Convert the multiple 'unlisted' radio variables into a
        # checkbox variable
        new_df = pd.DataFrame(
            index=df.index,
            columns=[column + '_item' + sep + x for x in values])
        for value in values:  # it's too slow...
            yes_ind = (df[unlisted_columns_dict[column]] == value).any(axis=1)
            new_df.loc[yes_ind, column + '_item' + sep + value] = True

        # Insert this in place of the 'unlisted' variable
        column_loc = df.columns.get_loc(column)
        df = pd.concat(
            [df.iloc[:, :column_loc], new_df, df.iloc[:, column_loc:]],
            axis=1)

        # Add new one-hot-encoded (checkbox) variables to dictionary
        new_dictionary_index = ind + np.linspace(0.1, 0.9, len(values))
        new_dictionary = pd.DataFrame(
            '', columns=dictionary.columns, index=new_dictionary_index)
        new_dictionary['field_name'] = [
            column + '_item' + sep + value for value in values]
        new_dictionary['field_type'] = 'bool'
        new_dictionary['field_label'] = values
        new_dictionary['section'] = dictionary.loc[ind, 'section']
        new_dictionary['field_subgroup'] = (
            dictionary.loc[ind, 'field_subgroup'])
        new_dictionary['original_field_name'] = ''
        new_dictionary['parent_field_name'] = column
        new_dictionary['form_name'] = dictionary.loc[ind, 'form_name']
        new_dictionary_list.append(new_dictionary)

    # Reorder the dictionary to add any new variables immediately after
    # the original variable
    dictionary = pd.concat([dictionary] + new_dictionary_list, axis=0)
    dictionary = dictionary.sort_index().reset_index(drop=True)

    return df, dictionary


def get_branching_logic_variables(branching_logic, sep='___'):
    """Get all variables included in the branching logic (including
    checkbox variables).
    """

    var_names = [x.split(']')[0] for x in branching_logic.split('[')[1:]]
    # Change any checkbox variable from bracket form to its one-hot column name
    var_names = [x.replace('(', sep).replace(')', '') for x in var_names]

    return var_names


def resolve_checkbox_branching_logic(df, dictionary, sep='___'):
    """Check if there is branching logic applied to checkbox
    variables. By default, a cell is marked as 'Unchecked' in the
    absence of 'Checked', even if the question was not asked to the
    participant. If the question was not asked to the participant
    because of the branching logic, then set this to be NaN instead.
    This does not completely check the branching logic, which is a
    data quality issue.
    """

    checkbox_ind = dictionary.loc[(dictionary['field_type'] == 'checkbox')]
    # Extract variables from the branching logic string
    branching_logic_variables = (
        dictionary['branching_logic'].apply(get_branching_logic_variables))
    # For each checkbox variable where the branching logic variable is missing,
    # replace the 'Unchecked' with NaN value
    for ind in checkbox_ind.index:
        branching_logic_columns = [
            col for col in df.columns
            if col in branching_logic_variables.loc[ind]]
        remove_ind = df[branching_logic_columns].isna().any(axis=1)

        # Identify checkbox columns relating to this variable
        checkbox_columns = [
            col for col in df.columns
            if (col.split(sep)[0] == dictionary.loc[ind, 'field_name'])]
        df.loc[remove_ind, checkbox_columns] = np.nan

    return df


def homogenise_variables(df, dictionary):
    """Converts variables to consistent units based on a conversion
    table.
    """

    # Load the conversion table. Try to read from GitHub as this is more likely
    # to be up to date.
    if ('conversion_table.csv' in os.listdir('assets')):
        conversion_table = pd.read_csv(
            os.path.join('assets', 'conversion_table.csv'))
    else:
        print("""There is no conversion_table.csv in assets, trying to load \
directly from ISARIC VERTEX GitHub instead.""")
        try:
            url = os.path.join(
                'https://raw.githubusercontent.com',
                'ISARICResearch/VERTEX/main/assets/conversion_table.csv')
            conversion_table = pd.read_csv(url)
        except Exception:
            raise

    # Only
    conversion_ind = (
        conversion_table['field_name'].isin(df.columns.tolist()) &
        (conversion_table['perform_conversion'] == 1))

    for _, row in conversion_table.loc[conversion_ind].iterrows():
        from_unit = row['from_unit']
        to_unit = row['to_unit']
        column = row['field_name']
        unit_column = row['unit_field_name']
        denom = row['denominator_field_name']
        error = f"""Unit conversion failed for {column} from {from_unit} to \
{to_unit}, continuing without this"""

        # Perform conversions
        if pd.isna(denom):
            conversion_function = eval(
                'lambda value: ' + row['conversion_function'])

            try:
                ind = (df[unit_column] == from_unit)
                # Execute the provided conversion
                df.loc[ind, column] = df.loc[ind, column].apply(
                    lambda value: conversion_function(value))
                df.loc[ind, unit_column] = to_unit
            except Exception:
                print(error)
                pass

        else:
            # Conversion involves a denominator
            conversion_function = eval(
                'lambda value, denominator: ' + row['conversion_function'])

            try:
                ind = (df[unit_column] == from_unit)
                # Execute the provided conversion (as a function of
                # both columns)
                df.loc[ind, column] = df.loc[ind, [column, denom]].apply(
                    lambda x: conversion_function(x[column], x[denom]),
                    axis=1)
                df.loc[ind, unit_column] = to_unit
            except Exception:
                print(error)
                pass

    # If multiple units still exist, split the variable into separate
    # variables for each unit. Otherwise update the dictionary to include
    # the unit in the field name.
    unit_ind = (
        conversion_ind &
        conversion_table['unit_field_name'].replace('', np.nan).notna() &
        (conversion_table['unit_field_name'].duplicated() == 0))
    new_variables_list = []

    for _, row in conversion_table.loc[unit_ind].iterrows():
        to_unit = row['to_unit']
        column = row['field_name']
        unit_column = row['unit_field_name']

        # If multiple units still exist
        if (df[unit_column].dropna().nunique() > 1):

            add_column_list = []
            add_series_list = []
            # Add new columns for each unit
            for unit_value in df[unit_column].dropna().unique():
                add_column = column + '_' + format_field_name(
                    unit_value.replace('%', 'pcnt').replace('µ', 'u'))
                add_series = pd.Series(np.nan, name=add_column)
                value_ind = (df[unit_column] == unit_value)
                add_series.loc[value_ind] = df.loc[value_ind, column]
                add_column_list.append(add_column)
                add_series_list.append(add_series)

            new_df = pd.concat(add_series_list, axis=1)
            column_loc = df.columns.get_loc(column)
            df = pd.concat(
                [df.iloc[:, :column_loc], new_df, df.iloc[:, column_loc:]],
                axis=1)

            # Add these to the dictionary
            dictionary_ind = (dictionary['field_name'] == column)
            dictionary_ind = dictionary_ind.loc[dictionary_ind].index.values[0]
            n_values = len(add_column_list)
            index = list(np.linspace(
                dictionary_ind + 0.1, dictionary_ind + 0.9, n_values))
            new_variables = pd.DataFrame(
                columns=dictionary.columns.tolist(), index=index)

            # Fill the dictionary
            new_variables['field_name'] = add_column_list
            new_variables['field_type'] = 'number'
            original_field_label = dictionary.loc[
                dictionary_ind, 'field_label']
            new_variables['field_label'] = [
                f'{original_field_label}  ({y})'
                for y in df[unit_column].dropna().unique().tolist()]
            new_variables['original_field_name'] = ''

            other_columns = [
                'parent_field_name', 'field_subgroup', 'section', 'form_name']
            for column in other_columns:
                new_variables[column] = [
                    dictionary.loc[dictionary_ind, column]] * n_values

            new_variables_list.append(new_variables)

            # Remove the original variable + unit columns from the dictionary
            remove_ind = (
                dictionary['field_name'].str.contains(unit_column) |
                (dictionary['field_name'] == column))
            dictionary.drop(
                index=dictionary.loc[remove_ind].index,
                inplace=True)

            # Remove the original variable and unit collumns from the data
            remove_columns = [
                col for col in df.columns
                if col in dictionary.loc[remove_ind, 'field_name'].tolist()]
            df.drop(columns=remove_columns, inplace=True)

        else:
            # Update the dictionary when all units are the same
            dictionary_ind = (dictionary['field_name'] == column)
            dictionary.loc[dictionary_ind, 'field_label'] += f' ({to_unit})'

            # Remove the original unit columns from the dictionary
            remove_ind = (
                dictionary['field_name'].str.contains(unit_column))
            dictionary.drop(
                index=dictionary.loc[remove_ind].index,
                inplace=True)

            # Remove the original unit collumns from the data
            remove_columns = [
                col for col in df.columns
                if col in dictionary.loc[remove_ind, 'field_name'].tolist()]
            df.drop(columns=remove_columns, inplace=True)

    # Reorder the dictionary to add any new variables immediately after the
    # original variable
    dictionary = pd.concat([dictionary] + new_variables_list, axis=0)
    dictionary = dictionary.sort_index().reset_index(drop=True)

    return df, dictionary


# def get_filter_columns(
#         data,
#         dictionary,
#         filter_column_mapping=None,
#         filter_values_mapping=None):
#     """Create columns for filters in one-row-per-patient wide data schema."""
#
#     # Mapping for ARC v1.1.1, which should remain fixed in future versions
#     default_filter_column_mapping = {
#         'subjid': 'subjid',
#         'filters_sex': 'demog_sex',
#         'filters_age': 'demog_age'
#     }
#
#     if filter_column_mapping is None:
#         filter_column_mapping = {
#             'subjid': 'subjid',
#             'filters_sex': 'demog_sex',
#         }
#
#     # subjid
#     try:
#         data['subjid'] = data[required_columns['subjid']].copy()
#     except KeyError:
#         print(f'`{required_columns['subjid']}` not in dataframe')
#         raise
#
#     # Filter for sex at birth
#     test = data[required_columns['demog_sex']].apply(
#         lambda x: x in required_mapping['demog_sex'].keys()).all()
#     assert test, f"""required_mapping['demog_sex'] must contain all unique
#     values in {required_columns['filters_sex']}"""
#
#     try:
#         data['filters_sex'] = data[required_columns['filters_sex']].replace(
#             required_mapping['filters_sex']).copy()
#     except KeyError:
#         print(f"""`{required_columns['filters_sex']}` not in dataframe""")
#         raise
#
#     # Age filter
#
#     numeric_columns = data.select_dtypes([float, int]).columns
#     test = (required_columns['demog_age'] in numeric_columns)
#     assert test, f"""`{required_columns['demog_age']}` not numeric"""
#
#     try:
#         data['demog_age'] = data[required_columns['demog_age']].copy()
#     except KeyError:
#         print(f"""`{required_columns['demog_age']}` not in dataframe""")
#         raise
#
#     # Outcome filter
#     test = (required_columns['outco_binary_outcome'] in data.columns)
#     assert test, f"""{required_columns['outco_binary_outcome']} not in dataframe"""
#
#     test = data[required_columns['outco_binary_outcome']].fillna('').apply(
#         lambda x: x in required_mapping['outco_binary_outcome'].keys()).all()
#     assert test, f"""required_mapping['outco_binary_outcome'] must contain all
#     unique values in {required_columns['outco_binary_outcome']}"""
#
#     data['outco_binary_outcome'] = (
#         data[required_columns['outco_binary_outcome']].fillna('').replace(
#             required_mapping['outco_binary_outcome']).copy())
#
#     # Country filter
#     url = os.path.join(
#         'https://raw.githubusercontent.com',
#         'ISARICResearch/VERTEX/main/assets/countries.csv')
#     country = pd.read_csv(url, encoding='latin-1')
#
#     test = (required_columns['country_iso'] in data.columns)
#     assert test, f"""{required_columns['country_iso']} not in dataframe"""
#
#     test = data[required_columns['country_iso']].apply(
#         lambda x: x in country['Code'].tolist()).all()
#     assert test, f"""required_mapping['country_iso'] must contain only ISO
#     codes as specified in {url}"""
#
#     data['country_iso'] = data[required_columns['country_iso']].copy()
#
#     # Date filter
#     test = (required_columns['dates_admdate'] in data.columns)
#     assert test, f"""{required_columns['dates_admdate']} not in dataframe"""
#
#     try:
#         data[required_columns['dates_admdate']].apply(
#             lambda x: pd.to_datetime(x, dayfirst=True))
#     except Exception:
#         raise
#
#     data['dates_admdate'] = data[required_columns['dates_admdate']].apply(
#         lambda x: pd.to_datetime(x, dayfirst=True)).copy()
#
#     return


############################################
# Main functions
############################################


def initial_data_processing(
        data,
        redcap_dictionary,
        missing_data_codes,
        convert_column_names=True,
        sep='___'):
    """Initial processing of REDCap exported dataframe."""

    # Replace empty cells or 'Unknown' with NaN
    with pd.option_context('future.no_silent_downcasting', True):
        data = data.replace(['', 'Unknown', 'unknown'], np.nan)

    # Replace missing data codes with NaN
    if missing_data_codes is not None:
        with pd.option_context('future.no_silent_downcasting', True):
            data = data.replace(list(missing_data_codes.keys()), np.nan)

    # Replace values in checkbox variables with NaN if the checkbox
    # missing data code column is 'Checked'
    data = replace_with_nan_for_missing_code_checkbox(data, missing_data_codes)

    # Remove columns where all the data is a negative answer option
    # (or missing answer)
    remove_values = ['', 'no', 'never smoked', 'unchecked', 'nan']
    remove_values += [x.lower() for x in missing_data_codes.keys()]
    remove_column_ind = data.astype(str).map(
        lambda x: (x.lower().strip() in remove_values)).all(axis=0)
    remove_columns = data.columns[remove_column_ind]
    data = data[[col for col in data.columns if col not in remove_columns]]

    # Convert 'Unchecked' to NaN when a checkbox question wasn't asked
    # to a subjid because of their previous answers (i.e. the
    # branching logic)
    # TODO: this needs updating based on branching logic values,
    # not just the variables themselves
    data = resolve_checkbox_branching_logic(data, redcap_dictionary)

    # Rename columns to reduce potential for errors (e.g. spaces in
    # column names is bad)
    initial_columns = data.columns.tolist()
    initial_columns = initial_columns + [
        x.split(sep)[0] for x in initial_columns if sep in x]

    if convert_column_names:
        converted_columns = [format_field_name(x) for x in initial_columns]
        converted_columns = add_number_to_duplicated_columns(converted_columns)
        rename_column_dict = dict(zip(initial_columns, converted_columns))
        data.rename(columns=rename_column_dict, inplace=True)
    else:
        rename_column_dict = {}

    # Create a new dictionary for internal VERTEX use
    dictionary_columns = [
        'field_name',
        'field_type',
        'field_label',
        'section',
        'field_subgroup',
        'parent_field_name',
        'original_field_name',
        'form_name'
    ]
    dictionary = pd.DataFrame(columns=dictionary_columns)

    dictionary['field_name'] = redcap_dictionary['field_name'].replace(
        rename_column_dict)
    dictionary['field_type'] = convert_field_type(redcap_dictionary)

    new_section_ind = (
        (redcap_dictionary['section_header'].duplicated() == 0) &
        (redcap_dictionary['section_header'] != ''))
    dictionary.loc[new_section_ind, 'section'] = redcap_dictionary.loc[
        new_section_ind, 'field_name'].apply(lambda x: x.split('_')[0])
    dictionary['section'] = dictionary['section'].ffill().fillna('')

    dictionary['field_label'] = redcap_dictionary['field_label'].copy()

    unique_branching_logic = [
        x for x in redcap_dictionary['branching_logic'].unique() if (x != '')]
    field_group_dict = dict(zip(
        unique_branching_logic,
        [int_to_alphabet(1 + x) for x in range(len(unique_branching_logic))]
    ))
    dictionary['field_subgroup'] = redcap_dictionary['branching_logic'].map(
        field_group_dict).fillna('')

    dictionary['parent_field_name'] = dictionary['section'].copy()
    dictionary['original_field_name'] = redcap_dictionary['field_name'].copy()
    dictionary['form_name'] = redcap_dictionary['form_name'].copy()

    # Add additional rows for sections
    ind = (redcap_dictionary['section_header'] != '')
    ind = ind.loc[ind].index

    sections = pd.DataFrame('', columns=dictionary.columns, index=ind)
    sections['field_label'] = (
        redcap_dictionary.loc[ind, 'section_header'].apply(
            lambda x: x.split(': ')[0]))
    sections['field_type'] = 'section'
    sections['form_name'] = redcap_dictionary.loc[ind, 'form_name']
    sections['field_name'] = redcap_dictionary.loc[ind, 'field_name'].apply(
        lambda x: x.split('_')[0])
    sections['section'] = sections['field_name']
    sections.index -= 0.5

    dictionary = pd.concat([dictionary, sections], axis=0)
    dictionary = dictionary.sort_index().reset_index(drop=True)

    # ----
    # Add one-hot-encoded variables to the dictionary (for categorical and
    # checkbox variables)

    # Get a dict of answer values and labels
    answer_dict = get_answer_dict(redcap_dictionary, data)

    # Create a dictionary for one-hot-encoded variables
    new_variables = pd.DataFrame(columns=dictionary_columns)

    # Create index in order to insert the onehot variables below their parent
    new_variables = new_variables.reset_index()
    ind = dictionary['original_field_name'].isin(answer_dict.keys())
    n_values = [len(x) for x in answer_dict.values()]
    ind = ind.loc[ind].index.tolist()
    index = [
        list(np.linspace(x + 0.1, x + 0.9, y)) for x, y in zip(ind, n_values)]
    new_variables['index'] = sum(index, [])
    new_variables = new_variables.set_index('index')
    new_variables.index.name = None

    # Create new column names for post- one-hot-encoding
    if convert_column_names:
        field_names = [
            list(format_field_name(k + sep + x) for x in v.keys())
            for k, v in answer_dict.items()]
        parents = [
            list(format_field_name(k) for x in v.keys())
            for k, v in answer_dict.items()]
    else:
        field_names = [
            list(k + sep + format_field_name(x) for x in v.keys())
            for k, v in answer_dict.items()]
        parents = [list(k for x in v.keys()) for k, v in answer_dict.items()]

    # Get REDCap column names for checkbox variables
    checkbox_ind = (redcap_dictionary['field_type'] == 'checkbox')
    original_field_names = [
        list(k + sep + x for x in v.values())
        if k in redcap_dictionary.loc[checkbox_ind, 'field_name'].tolist()
        else ['']*len(v)
        for k, v in answer_dict.items()]

    # Fill the dictionary
    new_variables['field_name'] = sum(field_names, [])
    new_variables['field_type'] = 'bool'
    new_variables['field_label'] = sum(
        [list(v.keys()) for v in answer_dict.values()], [])
    new_variables['parent_field_name'] = sum(parents, [])
    new_variables['original_field_name'] = sum(original_field_names, [])

    for column in ['section', 'field_subgroup', 'form_name']:
        new_variables[column] = (
            dictionary.loc[ind, column].repeat(n_values).tolist())

    dictionary = pd.concat([dictionary, new_variables], axis=0)
    dictionary = dictionary.sort_index().reset_index(drop=True)

    # ----
    # Rename checkbox columns (which by default contain value instead
    # of label)
    checkbox_ind = (dictionary['field_type'] == 'checkbox')
    onehot_checkbox_ind = dictionary['parent_field_name'].isin(
        dictionary.loc[checkbox_ind, 'field_name'])
    rename_dict = dict(zip(
        dictionary.loc[onehot_checkbox_ind, 'original_field_name'],
        dictionary.loc[onehot_checkbox_ind, 'field_name']
    ))
    data = data.rename(columns=rename_dict)

    # ----
    # Convert binary columns to correct type in the data
    # e.g. Yes(Checked)/No(Unchecked)/Unknown to True/False/NaN
    binary_ind = (dictionary['field_type'] == 'bool')
    binary_columns = dictionary.loc[binary_ind, 'field_name'].tolist()
    binary_columns = [col for col in binary_columns if col in data.columns]
    mapping_dict = {
        'Yes': True,
        'Checked': True,
        'No': False,
        'Unchecked': False,
        'Unknown': np.nan
    }
    with pd.option_context('future.no_silent_downcasting', True):
        data.loc[:, binary_columns] = (
            data[binary_columns].replace(mapping_dict))

    # Convert numerical data to numeric type and homogenise if mixed units
    numeric_ind = (dictionary['field_type'] == 'number')
    numeric_columns = dictionary.loc[numeric_ind, 'field_name'].tolist()
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors='coerce')
    data, dictionary = homogenise_variables(data, dictionary)

    # Convert columns with dates into datetime
    date_ind = (dictionary['field_type'] == 'datetime')
    date_columns = dictionary.loc[date_ind, 'field_name'].tolist()
    data[date_columns] = data[date_columns].apply(
        pd.to_datetime, dayfirst=True, errors='coerce')

    # Merge multiple 'unlisted' variables as defined in ISARIC ARC/BRIDGE
    # These are repeated radio questions with the same answer options
    # and branching logic that triggers the repeat question
    data, dictionary = combine_unlisted_variables(data, dictionary)

    # ----
    return data, dictionary


def get_df_map(
        data, dictionary, filter_column_mapping=None, country_mapping=None):
    """Convert single-event rows into wide one-row-per-patient dataframe."""

    # Get only data from presentation (including daily) and outcome phases
    forms = ['presentation', 'daily', 'outcome']
    columns = dictionary.loc[
        dictionary['form_name'].isin(forms), 'field_name'].tolist()
    columns = [col for col in columns if col in data.columns]
    ind = data['form_name'].apply(
        lambda x: any(y in x.split(',') for y in ['presentation', 'outcome']))

    df_map = data.loc[ind, columns].copy()

    # Merge into one row per subjid (from one row per phase per patient)
    df_map = df_map.set_index('subjid').groupby(level=0).bfill()
    df_map = df_map.drop(
        columns=[col for col in df_map.columns if 'redcap' in col])
    df_map = df_map.reset_index().drop_duplicates('subjid')
    df_map = df_map.reset_index(drop=True)

    # Create filters_sex and map other/null values
    df_map['filters_sex'] = df_map['demog_sex'].copy()
    other_value_ind = (df_map['demog_sex'].isin(['Male', 'Female']) == 0)
    df_map.loc[other_value_ind, 'filters_sex'] = 'Not specified/Unknown'

    # Create filters_date
    df_map['filters_date'] = df_map['demog_admdate'].copy()

    # Create filters_age (may be via demog_birthdate and filters_date)
    df_map['filters_age'] = df_map['demog_age'].copy()
    if ('demog_birthdate' in df_map.columns):
        ind = df_map['demog_age'].isna()
        y_diff = (
            df_map.loc[ind, 'filters_date'].dt.year -
            df_map.loc[ind, 'demog_birthdate'].dt.year)
        m_diff = (
            df_map.loc[ind, 'filters_date'].dt.month -
            df_map.loc[ind, 'demog_birthdate'].dt.month)
        df_map.loc[ind, 'filters_age'] = (y_diff - 1 * (m_diff < 0))

    # Create filters_country_iso3
    df_map['filters_country'] = df_map['dafsa'].copy()
    country_mapping = config_dict['country_mapping']
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

    # Create filters_outcome, reduce categories and map from outco_outcome
    mapping_dict = {
        'Discharged alive': 'Discharged',  # :)
        'Discharged against medical advice': 'Discharged',  # :)
        'Death': 'Death',  # :(
        'Palliative care': 'Death',  # :(
    }
    df_map['filters_outcome'] = df_map['outco_outcome'].copy().map(
        mapping_dict).fillna('Censored')

    # Add filters_outcome variable to dictionary
    outcome_dict = {}
    outcomes = ['Death', 'Discharged', 'Censored']
    outcome_dict['field_name'] = ['filters_outcome']
    outcome_dict['field_name'] += [
        'filters_outcome___' + x for x in outcomes]
    outcome_dict['form_name'] = ''
    outcome_dict['field_type'] = ['categorical'] + ['bool']*len(outcomes)
    outcome_dict['field_label'] = ['Outcome (mapped)'] + outcomes
    outcome_dict['parent_field_name'] = (
        ['outco'] + ['filters_outcome']*len(outcomes))
    outcome_dict['field_subgroup'] = ''

    # Add country_iso to dictionary
    countries = df_map['filters_country_iso3'].drop_duplicates().tolist()
    country_dict = {}
    country_dict['field_name'] = ['filters', 'filters_country_iso3']
    country_dict['field_name'] += ['filters_country_iso3___' + x for x in countries]
    country_dict['form_name'] = 'presentation'
    country_dict['field_type'] = ['section', 'categorical']
    country_dict['field_type'] += ['binary']*len(countries)
    country_dict['field_label'] = ['COUNTRY', 'Country ISO-3 Code'] + countries
    country_dict['parent_field_name'] = (
        ['', 'country'] + ['filters_country_iso3']*len(countries))
    country_dict['field_subgroup'] = ''

    dictionary = pd.concat([
        dictionary,
        pd.DataFrame.from_dict(country_dict),
        pd.DataFrame.from_dict(outcome_dict)
    ], axis=0)
    dictionary = dictionary.reset_index(drop=True)

    return df_map, dictionary


def get_df_forms(data, dictionary):
    """Convert any non-attribute data into long schema."""

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


def get_redcap_data(
        redcap_url, redcap_api_key,
        data_access_groups=None, user_assigned_to_dag=False,
        country_mapping=None):
    """Get data from REDCap API and transform into analysis-ready
    dataframes.
    """

    data = get_records(
        redcap_url, redcap_api_key,
        data_access_groups=data_access_groups,
        user_assigned_to_dag=user_assigned_to_dag)
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
    df_map, new_dictionary, quality_report = get_df_map(data, new_dictionary)
    df_forms_dict = get_df_forms(data, new_dictionary)

    return df_map, df_forms_dict, new_dictionary, quality_report
