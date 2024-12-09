import numpy as np
import pandas as pd
# import re
# import os
import scipy.stats as stats
# import researchpy as rp
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
import xgboost as xgb
import itertools
from collections import OrderedDict
from typing import List, Union
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic._utils import select_topic_representation
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler


############################################
############################################
# General preprocessing
############################################
############################################


def get_choices_value(x):
    values = [int(y.split(',')[0]) for y in x]
    return values


def get_choices_label(x):
    labels = [','.join(y.split(',')[1:]).strip() for y in x]
    return labels


def get_choices_label_value_dict(dictionary):
    # Get categories from dictionary
    choices_split = dictionary['select_choices_or_calculations'].copy()
    # This may throw an error if there are variables of type: slider or calc
    invalid_choices_ind = choices_split.fillna('').apply(
        lambda x: (len(x) > 0) & (x.count('|') == 0) & (x.count(',') == 0))
    choices_split.loc[invalid_choices_ind] = np.nan
    choices_split = choices_split.str.rstrip('|,').str.split(r'\|').fillna('')
    choices_split = choices_split.apply(lambda x: [y.strip() for y in x])
    # This fixes the missing choices ind
    choices_split = choices_split.apply(lambda x: [y for y in x if y != ''])
    choices_dict = choices_split.apply(
        lambda x: dict(zip(get_choices_label(x), get_choices_value(x))))
    return choices_dict


def rename_checkbox_variables(df, dictionary, missing_data_codes=None):
    choices_dict = get_choices_label_value_dict(dictionary)
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


def get_variables_from_sections(
        variable_list, section_list, required_variables=None):
    '''
    Get only the variables from sections, plus any required variables
    '''
    inclu_variables = []
    for section in section_list:
        inclu_variables += [
            var for var in variable_list if var.startswith(section + '_')]

    if required_variables is not None:
        required_variables = [
            var for var in required_variables if var not in inclu_variables]
        inclu_variables = required_variables + inclu_variables
    return inclu_variables


def map_variable(variable, mapping_dict, other_value_str='Other / Unknown'):
    other_ind = (variable.isin(mapping_dict.keys()) == 0)
    variable = variable.map(mapping_dict)
    variable.loc[other_ind] = other_value_str
    return variable


def harmonizeAge(df):
    # df['demog_age'] = df['demog_age'].astype(float)
    df['demog_age'] = pd.to_numeric(
        df['demog_age'], errors='coerce').astype(float)
    df.loc[(df['demog_age_units'] == 'Months'), 'demog_age'] *= 1/12
    df.loc[(df['demog_age_units'] == 'Days'), 'demog_age'] *= 1/365
    df['demog_age_units'] = 'Years'  # Standardize the units to 'Years'
    return df


def homogenize_variables(df):
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
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

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
                    df[total_wbc_col] = pd.to_numeric(
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
    return df


def getVariableType_data(data, full_variable_dict):
    variable_dict = {}
    for key in full_variable_dict.keys():
        variable_dict[key] = [
            x for x in data.columns
            if x.split('___')[0] in full_variable_dict[key]]
    return variable_dict


def from_dummies(data, column, sep='___', missing_val='No'):
    df_new = data.copy()
    columns = df_new.columns[df_new.columns.str.startswith(column + sep)]
    df_new[column + sep + missing_val] = (
        (df_new[columns].any(axis=1) == 0) |
        (df_new[columns].isna().any(axis=1)))
    df_new[columns] = df_new[columns].fillna(0)
    df_new[column] = pd.from_dummies(
        df_new[list(columns) + [column + sep + missing_val]], sep=sep)
    df_new = df_new.drop(columns=columns)
    df_new = df_new.drop(columns=column + sep + missing_val)
    return df_new


def merge_categories_except_list(
        data, column, required_values=[], merged_value='Other'):
    data.loc[(data[column].isin(required_values) == 0), column] = merged_value
    return data


def merge_cat_max_ncat(data, column, max_ncat=4, merged_value='Other'):
    required_choices_list = data[column].value_counts().head(n=max_ncat)
    required_choices_list = required_choices_list.index.tolist()
    data = merge_categories_except_list(
        data, column, required_choices_list, merged_value)
    return data


############################################
############################################
# Descriptive table
############################################
############################################


def median_iqr_str(series, dp=1, mfw=4, min_n=3):
    if series.notna().sum() < min_n:
        output_str = 'N/A'
    else:
        mfw_f = int(np.log10(max((series.quantile(0.75), 1)))) + 2 + dp
        output_str = '%*.*f' % (mfw_f, dp, series.median()) + ' ('
        output_str += '%*.*f' % (mfw_f, dp, series.quantile(0.25)) + '-'
        output_str += '%*.*f' % (mfw_f, dp, series.quantile(0.75)) + ') | '
        output_str += '%*g' % (mfw, int(series.notna().sum()))
    return output_str


def mean_std_str(series, dp=1, mfw=4, min_n=3):
    if series.notna().sum() < min_n:
        output_str = 'N/A'
    else:
        mfw_f = int(max((np.log10(series.mean(), 1)))) + 2 + dp
        output_str = '%*.*f' % (mfw_f, dp, series.mean()) + ' ('
        output_str += '%*.*f' % (mfw_f, dp, series.std()) + ') | '
        output_str += '%*g' % (mfw, int(series.notna().sum()))
    return output_str


def n_percent_str(series, dp=1, mfw=4, min_n=1):
    if series.notna().sum() < min_n:
        output_str = 'N/A'
    else:
        output_str = '%*g' % (mfw, int(series.sum())) + ' ('
        percent = 100*series.mean()
        if percent == 100:
            output_str += '100.) | '
        else:
            output_str += '%4.*f' % (dp, percent) + ') | '
        output_str += '%*g' % (mfw, int(series.notna().sum()))
    return output_str


def descriptive_table(data, column, full_variable_dict, return_totals=True):
    '''
    Descriptive table for binary (including one-hot-encoded categorical) and
    numerical variables in data. The descriptive table will have seperate
    columns for each category that exists for the variable 'column', if this
    is provided.
    '''
    df = data.copy()
    df = df.dropna(axis=1, how='all')

    df.fillna({column: 'Unknown'}, inplace=True)

    variable_dict = getVariableType_data(
        df.drop(columns=column), full_variable_dict)
    numeric_var = variable_dict['number']
    binary_var = sum([
        variable_dict[key] for key in ['binary', 'categorical', 'OneHot']], [])

    table = pd.DataFrame(
        columns=['Reported', 'All'], index=df.drop(columns=column).columns)

    table.loc[numeric_var, 'Reported'] = 'Median (IQR) | N'
    table.loc[binary_var, 'Reported'] = 'Count (%) | N'

    mfw = int(np.log10(df.shape[0])) + 1  # Min field width, for formatting
    table.loc[numeric_var, 'All'] = df[numeric_var].apply(
        lambda x: median_iqr_str(x, mfw=mfw))
    table.loc[binary_var, 'All'] = df[binary_var].apply(
        lambda x: n_percent_str(x, mfw=mfw))

    totals = pd.DataFrame(columns=['Variable', 'All'], index=[-0.5])
    totals['Variable'] = 'totals'
    totals['All'] = df.shape[0]

    if column is not None:
        choices_values = df[column].unique()
        table[list(choices_values)] = ''
        for value in choices_values:
            ind = (df[column] == value)
            mfw = int(np.log10(ind.sum())) + 1  # Min field width, for format
            table.loc[numeric_var, value] = df.loc[ind, numeric_var].apply(
                lambda x: median_iqr_str(x, mfw=mfw))
            table.loc[binary_var, value] = df.loc[ind, binary_var].apply(
                lambda x: n_percent_str(x, mfw=mfw))
            totals[value] = ind.sum()

    # Reorder rows by relevance
    table['Importance'] = df.apply(
        lambda x: x.sum() if x.name in binary_var else x.notna().sum())
    table.reset_index(inplace=True, names='Variable')

    if return_totals:
        output = table, totals
    else:
        output = table
    return output


############################################
############################################
# Descriptive table: Formatting
############################################
############################################


def rename_variables(
        variables, dictionary, missing_data_codes=None, max_len=None):
    renamed_variables = variables.copy()
    variable_dict = dict(zip(
        dictionary['field_name'], dictionary['field_label']))
    variable_split = renamed_variables.apply(lambda x: x.split('___'))
    renamed_variables = variable_split.apply(lambda x: x[0])
    renamed_variables = renamed_variables.replace(variable_dict)
    if max_len is not None:
        renamed_variables = renamed_variables.apply(
            lambda x: x if len(x) < max_len else (
                ' '.join(x[:max_len].split(' ')[:-1]) + ' ...'))
    renamed_variables = renamed_variables.apply(lambda x: '<b>' + x + '</b>')
    renamed_variables += variable_split.apply(
        lambda x: '' if ((len(x) == 1) or (x[1] == 'Yes')) else ', ' + x[1])
    return renamed_variables


def reorder_descriptive_table_columns(
        table, column_order, required_columns=['Variable', 'All']):
    df = table.copy()
    new_column_order = [
        col for col in required_columns if col not in column_order]
    new_column_order += [col for col in column_order if col in df.columns]
    new_column_order += [
        col for col in df.columns if col not in new_column_order]
    df = df[new_column_order]
    return df


def add_descriptive_table_sections(table, dictionary):
    df = table.copy()
    new_section_bool = (
        (df['Section'].duplicated() == 0) & (df['Section'] != ''))
    new_section_index = new_section_bool[new_section_bool].index
    new_section_name = df.loc[new_section_index, 'Section'].values
    insert = pd.DataFrame(
        '', columns=df.columns, index=new_section_index - 0.5)
    insert['Variable'] = new_section_name
    sections = dictionary['field_name'].apply(lambda x: x.split('_')[0])
    new_section_ind = (
        (dictionary['section_header'] != '') & (sections.duplicated() == 0))
    codes = dictionary.loc[new_section_ind, 'field_name'].apply(
        lambda x: x.split('_')[0])
    names = dictionary.loc[new_section_ind, 'section_header'].apply(
        lambda x: '<b><i>' + x.split(':')[0].strip().capitalize() + '</i></b>')
    insert['Variable'] = insert['Variable'].replace(dict(zip(codes, names)))
    df = pd.concat([df, insert]).sort_index().reset_index(drop=True)
    return df


def add_descriptive_table_repeat_variables(table):
    df = table.copy()
    new_variable_bool = (
        (df['Name'].duplicated() == 0) &
        (df['Name'].duplicated(keep='last')))
    repeat_variable_bool = df['Name'].duplicated(keep=False)
    new_variable_index = new_variable_bool[new_variable_bool].index
    new_variable_name = df.loc[new_variable_index, 'Variable'].apply(
        lambda x: x.split(',')[0]).values
    df.loc[repeat_variable_bool, 'Variable'] = df.loc[
        repeat_variable_bool, 'Variable'].apply(
            lambda x: "    ↳ " + ', '.join(x.split(', ')[1:]))
    insert = pd.DataFrame(
        '', columns=df.columns, index=new_variable_index - 0.5)
    insert['Variable'] = new_variable_name
    insert['Name'] = df.loc[new_variable_index, 'Name'].values
    insert['Reported'] = df.loc[new_variable_index, 'Reported'].values
    insert['Section'] = df.loc[new_variable_index, 'Section'].values
    if insert.shape[0] > 0:
        df = pd.concat([df, insert]).sort_index().reset_index(drop=True)
    return df


def reorder_descriptive_table(table, dictionary, section_reorder=None):
    df = table.copy()
    # Reorder rows by relevance
    df['Section'] = df['Variable'].apply(lambda x: x.split('_')[0])
    df['Name'] = df['Variable'].apply(lambda x: x.split('___')[0])
    df['Value'] = df['Variable'].apply(
        lambda x: x.split('___')[1] if (len(x.split('___')) > 1) else 'N/A')
    choices_dict = pd.concat(
        [dictionary['field_name'], get_choices_label_value_dict(dictionary)],
        axis=1).rename(columns={'field_name': 'Name'})
    df = pd.merge(df, choices_dict, on='Name', how='left')
    df = df.rename(columns={'select_choices_or_calculations': 'choices'})
    df['raw_value'] = [
        y.get(x) if x != 'N/A' else 0
        for x, y in zip(df['Value'].values, df['choices'].values)]
    grouped_df = df.groupby('Name').agg(
        name_importance=pd.NamedAgg(column='Importance', aggfunc='max'))
    df = pd.merge(df, grouped_df.reset_index(), on='Name', how='left')

    if section_reorder is not None:
        order = np.arange(len(section_reorder))
        with pd.option_context('future.no_silent_downcasting', True):
            df.replace(
                {'Section': dict(zip(section_reorder, order))}, inplace=True)
    df = df.sort_values(
        by=['Section', 'name_importance', 'Name', 'raw_value'],
        ascending=[True, False, False, True])
    if section_reorder is not None:
        with pd.option_context('future.no_silent_downcasting', True):
            df.replace(
                {'Section': dict(zip(order, section_reorder))}, inplace=True)
    df = df.reset_index(drop=True)
    remove_columns = ['name_importance', 'Importance', 'raw_value', 'choices']
    df = df.drop(columns=remove_columns)
    return df


def reformat_descriptive_table(
        table, dictionary,
        totals=None, column_reorder=None, section_reorder=None):
    df = table.copy()
    df = reorder_descriptive_table(
        df, dictionary=dictionary, section_reorder=None)
    df['Variable'] = rename_variables(df['Variable'], dictionary)
    if column_reorder is not None:
        df = reorder_descriptive_table_columns(df, column_reorder)
    df = add_descriptive_table_repeat_variables(df)
    df.loc[(df['Name'].duplicated() == 0), 'Variable'] = (
        df.loc[(df['Name'].duplicated() == 0), 'Variable'] +
        df.loc[(df['Name'].duplicated() == 0), 'Reported'].replace({
            'Count (%) | N': ' (*)',
            'Median (IQR) | N': ' (+)'
        }))
    table_key = '(*) Count (%) | N<br>(+) Median (IQR) | N'
    df = add_descriptive_table_sections(df, dictionary)
    df.drop(columns=['Reported', 'Section', 'Name', 'Value'], inplace=True)
    if totals is not None:
        totals['Variable'] = '<b>Totals</b>'
        df = pd.concat([df, totals]).sort_index().reset_index(drop=True)
    return df, table_key


############################################
############################################
# Formatting: colours
############################################
############################################


def hex_to_rgb(hex_color):
    ''' Convert a hex color to an RGB tuple. '''
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def interpolate_colors(colors, n):
    ''' Interpolate among multiple hex colors.'''
    # Convert all hex colors to RGB
    rgbs = [hex_to_rgb(color) for color in colors]

    interpolated_colors = []
    # Number of transitions is one less than the number of colors
    transitions = len(colors) - 1

    # Calculate the number of steps for each transition
    steps_per_transition = n // transitions

    # Interpolate between each pair of colors
    for i in range(transitions):
        for step in range(steps_per_transition):
            interpolated_rgb = [
                int(rgbs[i][j] + (
                    float(step)/steps_per_transition)*(rgbs[i+1][j]-rgbs[i][j]))
                for j in range(3)]
            interpolated_colors.append(
                f'rgb({interpolated_rgb[0]}, ' +
                f'{interpolated_rgb[1]},' +
                f'{interpolated_rgb[2]})')

    # Append the last color
    if len(interpolated_colors) < n:
        interpolated_colors.append(
            f'rgb({rgbs[-1][0]}, {rgbs[-1][1]}, {rgbs[-1][2]})')
    return interpolated_colors


def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    rgba_color = 'rgba(' + ', '.join(
        str(int(hex_color[i:i+hlen//3], 16))
        for i in range(0, hlen, hlen//3))
    rgba_color += f', {opacity})'
    return rgba_color


def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add in range [0,1]
    :return: RGBA Value
    """
    rgba_color = f"rgba{rgb_value[3:-1]}, {alpha})"
    return rgba_color


# def rgb_to_rgba(rgb_value, alpha):
#     """
#     Adds the alpha channel to an RGB Value and returns it as an RGBA Value
#     :param rgb_value: Input RGB Value
#     :param alpha: Alpha Value to add in range [0,1]
#     :return: RGBA Value
#     """
#     return f"rgba{rgb_value[3:-1]}, {alpha})"


############################################
############################################
# Counts
############################################
############################################


# def get_proportions(data, data_type):
#     prefix = ''
#     if data_type == 'symptoms':
#         prefix = 'adsym_'
#     elif data_type == 'comorbidities':
#         prefix = 'comor_'
#     elif data_type == 'treatments':
#         prefix = 'treat_'
#     else:
#         prefix = data_type + '_'
#
#     variables = []
#
#     for i in data:
#         if prefix in i:
#             variables.append(i)
#
#     # df = data[[
#     #     'usubjid', 'age', 'slider_sex', 'slider_country', 'outcome',
#     #     'country_iso'] + variables].copy()
#     # df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
#     df = data.copy()
#
#     proportions = df[variables].dropna(axis=1, how='all').apply(
#         lambda x: x.dropna().sum() / x.dropna().count()).reset_index()
#
#     proportions.columns = ['Condition', 'Proportion']
#     proportions = proportions.sort_values(by=['Proportion'], ascending=False)
#     Condition_top = proportions['Condition'].head(5)
#     set_data = df[Condition_top]
#     return proportions, set_data


def get_proportions(df, section_list, max_n_variables=10):
    inclu_variables = get_variables_from_sections(df.columns, section_list)

    if len(inclu_variables) == 0:
        proportions = None

    proportions = df[inclu_variables].dropna(axis=1, how='all')

    proportions = proportions.apply(
        lambda x: x.sum() / x.count()).reset_index()

    proportions.columns = ['variable', 'proportion']
    proportions = proportions.sort_values(
        by=['proportion'], ascending=False).reset_index(drop=True)
    if proportions.shape[0] > max_n_variables:
        proportions = proportions.head(max_n_variables)
    return proportions


def get_intersections(df, proportions=None, variables=None, n_variables=5):
    if proportions is not None:
        variables = proportions.sort_values(
            by='proportion', ascending=False)['variable'].head(n_variables)
    if variables is None:
        df = df.copy()
        df = df[[var for var in df.columns if df[var].sum() > 0]].fillna(0)
    else:
        variables = [var for var in variables if df[var].sum() > 0]
        df = df[variables].fillna(0).copy()

    counts = df.sum().astype(int).reset_index().rename(columns={0: 'count'})
    counts = counts.sort_values(
        by='count', ascending=False).reset_index(drop=True)
    if variables is None:
        variable_order_dict = dict(zip(counts['index'], counts.index))
        variables = counts['index'].tolist()
    else:
        variable_order_dict = dict(zip(variables, range(len(variables))))
    if n_variables is not None:
        variables = variables[:n_variables]

    intersections = df.loc[df.any(axis=1)].value_counts().reset_index()
    intersections['index'] = intersections.drop(columns='count').apply(
        lambda x: tuple(col for col in x.index if x[col] == 1), axis=1)

    # The rest is reordering to make it look prettier
    intersections = intersections.loc[(intersections['count'] > 0)]
    intersections['index_n'] = intersections['index'].apply(len)
    intersections['index_first'] = (
        intersections[variables].idxmax(axis=1).map(variable_order_dict))
    intersections['index_last'] = (
        intersections[variables].idxmin(axis=1).map(variable_order_dict))
    intersections = intersections.sort_values(
        by=['count', 'index_first', 'index_last', 'index_n'],
        ascending=[False, True, False, False])
    intersections = intersections[['index', 'count']].reset_index(drop=True)
    return counts, intersections


# def get_intersections(df, ordered_variables, n_variables=5):
#     inclu_variables = ordered_variables.head(n_variables)
#     intersections = df
#
#     # categories_reduced = rename_variables(
#     #     pd.Series(df.columns), dictionary, max_len=50).tolist()
#     # df = df.rename(columns=dict(zip(
#     #     df.columns,
#     #     ia.rename_variables(pd.Series(df.columns), dictionary).tolist())))
#     categories = inclu_variables
#     # intersections = ia.compute_intersections(df)
#
#     df = df[inclu_variables].fillna(0)
#     intersections = df.loc[df.sum(axis=1) > 0].value_counts().reset_index()
#     for r in range(1, len(categories) + 1):
#         for combo in itertools.combinations(categories, r):
#             # Intersection is where all categories in the combo have a 1
#             ind = intersections[list(combo)].all(axis=1)
#             intersections.loc[ind, 'count_all'] = intersections.loc[ind, 'count'].sum()
#
#
#     return intersections


############################################
############################################
# Clustering of free-text terms
############################################
############################################
def clean_string_list(string_list):
    """Helper function to remove nans and empty strs from a list of strings"""
    
    # Using filter() with a lambda function
    cleaned_list = list(filter(
        lambda s: (
            s is not None 
            and not (isinstance(s, float) and np.isnan(s))
            and str(s).strip() != ''
        ),
        string_list
    ))
    
    return cleaned_list


def get_clusters(terms: List[str]):
    """Function to find common topics appearing in a list of free text terms. 
    Uses the BERTopic topic modelling pipeline.
    
    Args:
        terms (List[str]): list of free text terms, for example referring to an
        'other combordities' field in a CRF
    Returns:
        clusters_df (pd.DataFrame): pandas dataframe summarizing the results of 
                                    the clustering process. Contains the 
                                    following columns:
            Topic (int): topic id
            Count (int): number of rows in terms assigned to that topic
            Name (str): name of topic, default id + keywords
            Representation (List[str]): list of keywords in topic
            Representative_Docs (List[str]): list of entries in terms which 
                                            represent the topic
            x, y (floats): coordinates of topic in an embedding space"""
    
    # remove nans and empty strings
    terms = clean_string_list(terms)

    # first define the constituent parts of the pipeline
    # how we embed the strings - default is sentence-transformers
    embedding_model = None

    # how we represent the topics - keyword extraction based on TF-IDF
    keybert_mmr = [KeyBERTInspired(), MaximalMarginalRelevance()]
    
    # we can have multiple representations if we like
    representation_model = {
        "Main": keybert_mmr,
    }

    # lots of ways to control the number of topics, including setting a fixed n
    nr_topics = "auto"

    # use bertopic topic modelling pipeline
    topic_model = BERTopic(
        embedding_model=embedding_model, # how we embed the strings, default to sentence transformers
        representation_model=representation_model,
        nr_topics=nr_topics,
    )

    # fit the model on the terms
    topics, probs = topic_model.fit_transform(documents=terms)

    # extract topic words, frequencies, and embedding coordinates
    distance_df = extract_topic_embeddings(topic_model=topic_model)

    # extract info about each topic aka cluster
    cluster_df = topic_model.get_topic_info()

    # return the combined df
    return pd.merge(cluster_df, distance_df, on='Topic', how='left')


def extract_topic_embeddings(topic_model: BERTopic,
                             topics: List[int] = None,
                             top_n_topics: int = None,
                             use_ctfidf: bool = False,
                             ):
    """Helper function to extract df with topic embedding info, from 
    bertopic.plotting._topics.visualize_topics"""

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    embeddings, c_tfidf_used = select_topic_representation(
        topic_model.c_tf_idf_,
        topic_model.topic_embeddings_,
        use_ctfidf=use_ctfidf,
        output_ndarray=True,
    )
    embeddings = embeddings[indices]

    if c_tfidf_used:
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = UMAP(n_neighbors=2, n_components=2, metric="hellinger", random_state=42).fit_transform(embeddings)
    else:
        embeddings = UMAP(n_neighbors=2, n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)

    # assemble df
    df = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "Topic": topic_list,
        }
    )

    return df

############################################
############################################
# Modelling
############################################
############################################


# def preprocessing_for_risk(data, sections=['comor', 'adsym']):
#     df_map = data
#     comor = []
#     for i in df_map:
#         if 'comor_' in i:
#             comor.append(i)
#     sympt = []
#     for i in df_map:
#         if 'adsym_' in i:
#             sympt.append(i)
#
#     sdata = df_map[sympt+comor+['age', 'slider_sex', 'outcome']].copy()
#
#     sdata = sdata.applymap(lambda x: x.lower() if isinstance(x, str) else x)
#     sdata[sympt+comor] = (sdata[sympt+comor] != 'no')
#     sdata = sdata.loc[(sdata['outcome'] != 'censored')]
#
#     outcome_binary_map = {'discharge': 0, 'death': 1}
#     sex_binary_map = {'female': 0, 'male': 1}
#     sdata['outcome'] = sdata['outcome'].map(outcome_binary_map)
#     sdata['slider_sex'] = sdata['slider_sex'].map(sex_binary_map)
#     return sdata
#
#
# def remove_columns(data, limit_var=60):
#     nan_percentage = (data.isna().sum() / len(data))*100
#     nan_percentage = nan_percentage.reset_index()
#     variables_included = nan_percentage.loc[(
#         nan_percentage[0] <= limit_var), 'index']
#     return data[variables_included]
#
#
# def num_imputation_nn(df, n_neighbor=5):
#     # Separating numerical and encoded nominal variables
#     numerical_data = df.select_dtypes(include=[np.number])
#     # Initializing the KNN Imputer
#     imputer = KNNImputer(n_neighbors=n_neighbor)
#     # Imputing missing values
#     imputed_data = imputer.fit_transform(numerical_data)
#     # Converting imputed data back to a DataFrame
#     return pd.DataFrame(imputed_data, columns=numerical_data.columns)
#
#
# def binary_model(data, variables, outcome, num_estimators=10):
#     data_path = data.dropna(subset=[outcome])
#     combined_df = data_path.dropna(subset=[outcome])
#
#     # X_Transm = combined_df[variables]
#     X = combined_df[variables]
#
#     y = combined_df[outcome]
#     le = LabelEncoder()
#     y = list(le.fit_transform(y))
#
#     # Initialize XGBoost model for classification
#     xgb_model = xgb.XGBClassifier(
#         objective='multi:softmax', num_class=len(set(y)),
#         random_state=182, use_label_encoder=False, eval_metric='mlogloss',
#         enable_categorical=True, max_depth=4, n_estimators=num_estimators)
#     for X_x in X:
#         X[X_x] = X[X_x].astype('category')
#     # Train the model
#     xgb_model.fit(X, y)
#     # Make predictions
#     predictions = xgb_model.predict(X)
#     probabilities = xgb_model.predict_proba(X)
#     combined_df['Predictions'] = predictions
#     if (len(set(y)) == 2):
#         probabilities = pd.DataFrame(data=probabilities)
#         combined_df['probabilities'] = probabilities[1]
#
#     # Evaluate the model using a classification metric
#     accuracy = accuracy_score(y, predictions)
#     roc = roc_auc_score(y, combined_df['probabilities'])
#     fpr, tpr, thresholds = roc_curve(y, combined_df['probabilities'])
#
#     # Calculate the Youden's index
#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
#
#     # Feature importances
#     importances = xgb_model.feature_importances_
#     feature_names = X.columns
#     feature_importances = pd.DataFrame(
#         {'Feature': feature_names, 'Importance': importances})
#
#     # Sort the features by importance
#     feature_importances = feature_importances.sort_values(
#         by='Importance', ascending=False)
#     return feature_importances, accuracy, roc, optimal_threshold, combined_df
#
#
# def lasso_rf(data, outcome_var='Outcome'):
#     Y = data[outcome_var]
#     X = data.drop(outcome_var, axis=1)
#
#     # Feature Scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Splitting the dataset into cross-validation set and hold-out set
#     X_cv, X_holdout, Y_cv, Y_holdout, idx_cv, idx_holdout = train_test_split(
#         X_scaled, Y, range(len(data)),
#         test_size=0.2, random_state=666, stratify=Y)
#
#     # Logistic Regression with L1 regularization
#     log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear')
#
#     # Hyperparameter tuning using GridSearchCV
#     parameters = {'C': [0.0001, 0.001, 0.01]}
#     log_reg_cv = GridSearchCV(log_reg_l1, parameters, cv=10, scoring='roc_auc')
#     log_reg_cv.fit(X_cv, Y_cv)
#
#     # Best hyperparameter value
#     best_C = log_reg_cv.best_params_['C']
#
#     # Evaluate using the best parameter on the hold-out set
#     log_reg_best = LogisticRegression(
#         penalty='l1', C=best_C, solver='liblinear')
#     log_reg_best.fit(X_cv, Y_cv)
#
#     # Predicting probabilities
#     Y_pred_proba = log_reg_best.predict_proba(X_holdout)[:, 1]
#
#     # Calculating ROC AUC
#     roc_auc = roc_auc_score(Y_holdout, Y_pred_proba)
#
#     # Print coefficients
#     feature_names = X.columns
#     coefficients = log_reg_best.coef_[0]
#     non_zero_indices = np.where(coefficients != 0)[0]
#
#     # Standard errors, CIs, and p-values
#     # intercept = log_reg_best.intercept_
#     log_reg_best.fit(X_cv[:, non_zero_indices], Y_cv)
#     standard_errors = np.sqrt(np.diag(np.linalg.inv(np.dot(
#         X_cv[:, non_zero_indices].T, X_cv[:, non_zero_indices]))))
#     z_scores = coefficients[non_zero_indices] / standard_errors
#     p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores]
#
#     # Calculate odds ratios and confidence intervals
#     odds_ratios = np.exp(coefficients[non_zero_indices])
#     conf_intervals = np.exp(
#         coefficients[non_zero_indices][:, np.newaxis] +
#         np.array([-1, 1]) * 1.96 * standard_errors[:, np.newaxis])
#
#     # Format the coefficients, OR, CI, and p-values
#     formatted_coefficients = [
#         f'{coef:.3f}' for coef in coefficients[non_zero_indices]]
#     formatted_odds_ratios = [
#         f'{or_val:.3f}' for or_val in odds_ratios]
#     formatted_conf_intervals = [
#         (f'{ci[0]:.3f}', f'{ci[1]:.3f}') for ci in conf_intervals]
#     formatted_p_values = [
#         '<0.005' if pv < 0.005 else f'{pv:.3f}' for pv in p_values]
#
#     coef_df = pd.DataFrame({
#         'Feature': feature_names[non_zero_indices],
#         'Coefficient': formatted_coefficients,
#         'Odds Ratio': formatted_odds_ratios,
#         'CI Lower 95%': [ci[0] for ci in formatted_conf_intervals],
#         'CI Upper 95%': [ci[1] for ci in formatted_conf_intervals],
#         'P-value': formatted_p_values
#     })
#
#     return coef_df, roc_auc, best_C


############################################
############################################
# Graveyard
############################################
############################################


# def mapSex(df):
#     mapping_dict = {
#         'Female': 'Female',
#         'Male': 'Male'
#     }
#     other_outcome = (df['demog_sex'].isin(mapping_dict.keys()) == 0)
#     df['demog_sex'] = df['demog_sex'].map(mapping_dict)
#     df.loc[other_outcome, 'demog_sex'] = 'Other / Unknown'
#     return df
#
#
# def mapOutcomes(df):
#     mapping_dict = {
#         'Discharged alive': 'Discharged',
#         'Discharged against medical advice': 'Discharged',
#         'Death': 'Death',
#         # 'Transfer to other facility': 'Censored',
#         # 'Still hospitalised': 'Censored',
#         # 'Palliative care': 'Censored',
#         # 'Other': 'Censored'
#     }
#     other_outcome = (df['outco_outcome'].isin(mapping_dict.keys()) == 0)
#     df['outco_outcome'] = df['outco_outcome'].map(mapping_dict)
#     df.loc[other_outcome, 'outco_outcome'] = 'Censored'
#     return df

# def rename_variables(df_variables, dictionary, missing_data_codes=None):
#     choices_dict = get_label_value_dict(dictionary, missing_data_codes)
#     variable_dict = dict(zip(
#         dictionary['field_label'], dictionary['field_name']))
#     df_variable_split = df_variables.apply(lambda x: x.split(' '))
#     df_variable_names = df_variable_split.apply(lambda x: x[0].split('___')[0])
#     df_variable_names = df_variable_names.replace(variable_dict)
#     df_choices_names = df_variable_split.apply(
#         lambda x: x[0].split('___')[1] if '___' in x[0] else '')
#     df_choices_names = 1
#     return

# def get_variables_type(data):
#     final_binary_variables = []
#     final_numeric_variables = []
#     final_categorical_variables = []
#
#     for column in data:
#         column_data = data[column].dropna()
#
#         if column_data.empty:
#             continue
#
#         # Check if the column is numeric
#         if pd.api.types.is_numeric_dtype(column_data):
#             unique_values = column_data.unique()
#             if (len(unique_values) == 2) and (set(unique_values) == {0, 1}):
#                 final_binary_variables.append(column)
#             else:
#                 try:
#                     pd.to_numeric(column_data)
#                     final_numeric_variables.append(column)
#                 except ValueError as e:
#                     print(f'An error occurred: {e}')
#                     for col_value_i in column_data:
#                         print(col_value_i)
#         else:
#             unique_values = column_data.unique()
#             # Consider column as categorical if it has a few unique values
#             if len(unique_values) <= 10:
#                 final_categorical_variables.append(column)
#
#     return final_binary_variables, final_numeric_variables, final_categorical_variables
#
#
# def categorical_feature(data, categoricals):
#     categorical_results_t = []
#     for variable in categoricals:
#         data_variable = data[[variable]].dropna()
#         category_variable = 1
#         data_aux_cat = data_variable.loc[(data_variable[variable] == 1)]
#         try:
#             n = len(data_aux_cat)
#             pe = round(100 * (n / len(data_variable)), 1)
#             categorical_results_t.append([
#                 str(variable) + ': ' + str(category_variable),
#                 str(n) + ' (' + str(pe) + ')'])
#         except Exception:
#             print(variable)
#     categorical_results_t = pd.DataFrame(
#         data=categorical_results_t, columns=['Variable', 'Count'])
#     return categorical_results_t
#
#
# def categorical_feature_outcome(data, outcome):
#     binary_variables, numeric_variables, categorical_variables = get_variables_type(data)
#     try:
#         binary_variables.remove(outcome)
#     except Exception:
#         print('Outcome not in dataframe')
#     suitable_cat = []
#
#     categorical_results = []
#     categorical_results_t = []
#     for variable in binary_variables:
#         data_variable = data[[variable, outcome]].dropna()
#         x = data_variable[variable]
#         y = data_variable[outcome]
#         data_crosstab = pd.crosstab(x, y, margins=False)
#         stat, p, dof, expected = stats.chi2_contingency(data_crosstab)
#
#         if p < 0.2:
#             suitable_cat.append(variable)
#         if p < 0.001:
#             p = '<0.001'
#         elif p <= 0.05:
#             p = str(round(p, 3))
#         else:
#             p = str(round(p, 2))
#
#         data_variable0 = data_variable.loc[(data_variable[outcome] == 0)]
#         data_variable1 = data_variable.loc[(data_variable[outcome] == 1)]
#         for category_variable in [1]:
#             data_aux_cat = data_variable.loc[(
#                 data_variable[variable] == category_variable)]
#             n = len(data_aux_cat)
#             count = data_aux_cat[outcome].value_counts().reset_index()
#             n0 = count.loc[(count[outcome] == 0), 'count']
#             n1 = count.loc[(count[outcome] == 1), 'count']
#             p0 = round(100*(n0 / len(data_variable0)), 1)
#             p1 = round(100*(n1 / len(data_variable1)), 1)
#             pe = round(100*(n / len(data_variable)), 1)
#             if len(n0) == 0:
#                 n0, p0 = 0, 0
#             else:
#                 n0 = n0.iloc[0]
#                 p0 = p0.iloc[0]
#             if len(n1) == 0:
#                 n1, p1 = 0, 0
#             else:
#                 n1 = n1.iloc[0]
#                 p1 = p1.iloc[0]
#
#             categorical_results.append([
#                 str(variable),
#                 str(n1) + ' (' + str(p1) + ')',
#                 str(n0) + ' (' + str(p0) + ')',
#                 str(n) + ' (' + str(pe) + ')',
#                 p])
#             categorical_results_t.append([
#                 str(variable) + ': ' + str(category_variable),
#                 str(n) + ' (' + str(pe) + ')'])
#
#     column1 = 'Characteristic'
#     column2 = outcome + '=1 (n=' + str(round(data[outcome].sum())) + ')'
#     column3 = outcome + '=0 (n=' + str(round(len(data) - data[outcome].sum()))
#     column3 += ')'
#     column4 = 'All cohort (n=' + str(round(len(data))) + ')'
#
#     categorical_results = pd.DataFrame(
#         data=categorical_results,
#         columns=[column1, column2, column3, column4, 'p-value'])
#
#     categorical_results_t = pd.DataFrame(
#         data=categorical_results_t, columns=['Variable', 'Count'])
#     return categorical_results, suitable_cat, categorical_results_t
#
#
# def numeric_outcome_results(data, outcome):
#     binary_variables, numeric_variables, categorical_variables = get_variables_type(data)
#     results_array = []
#     results_t = []
#     suitable_num = []
#     for variable in numeric_variables:
#         try:
#             data[variable] = pd.to_numeric(data[variable], errors='coerce')
#             data_variable = data[[variable, outcome]].dropna()
#             data0 = data_variable.loc[(data_variable[outcome] == 0), variable]
#             data1 = data_variable.loc[(data_variable[outcome] == 1), variable]
#             data_t = data_variable[variable]
#             # complete = round((100 * (len(data_variable) / len(data))), 1)
#             if len(data_variable) > 2:
#                 # On the whole variable
#                 stat, p = stats.shapiro(data_variable[variable])
#                 alpha = 0.05
#                 if p < alpha:
#                     # print('Not normal')
#                     w, p = stats.mannwhitneyu(
#                         data0, y=data1, alternative='two-sided')
#                 else:
#                     summary, results = rp.ttest(
#                         group1=data.loc[(data[outcome] == 0), variable],
#                         group1_name='0',
#                         group2=data.loc[(data[outcome] == 1), variable],
#                         group2_name='1')
#                     p = results['results'].loc[3]
#
#                 detail0 = str(round(data0.median(), 1)) + ' ('
#                 detail0 += str(round(data0.quantile(0.25), 1)) + '-'
#                 detail0 += str(round(data0.quantile(0.75), 1)) + ')'
#                 detail1 = str(round(data1.median(), 1)) + ' ('
#                 detail1 += str(round(data1.quantile(0.25), 1)) + '-'
#                 detail1 += str(round(data1.quantile(0.75), 1)) + ')'
#                 detail_t = str(round(data_t.median(), 1)) + ' ('
#                 detail_t += str(round(data_t.quantile(0.25), 1)) + '-'
#                 detail_t += str(round(data_t.quantile(0.75), 1)) + ')'
#                 if p < 0.2:
#                     suitable_num.append(variable)
#                 if p < 0.001:
#                     p = '<0.001'
#                 elif p <= 0.05:
#                     p = str(round(p, 3))
#                 else:
#                     p = str(round(p, 2))
#             else:
#                 detail0 = str(round(data0.mean(), 1)) + ' ('
#                 detail0 += str(round(data0.std(), 1)) + ')'
#                 detail1 = str(round(data1.mean(), 1)) + ' ('
#                 detail1 += str(round(data1.std(), 1)) + ')'
#                 detail_t = str(round(data_t.mean(), 1)) + ' ('
#                 detail_t += str(round(data_t.std(), 1)) + ')'
#                 p = 'N/A'
#
#             results_array.append([variable, detail1, detail0, detail_t, p])
#             results_t.append([variable, detail_t])
#         except Exception:
#             print(variable)
#
#     column1 = 'Characteristic'
#     column2 = outcome + '=1 (n=' + str(round(data[outcome].sum())) + ')'
#     column3 = outcome + '=0 (n=' + str(round(len(data) - data[outcome].sum()))
#     column3 += ')'
#     column4 = 'All cohort (n=' + str(round(len(data))) + ')'
#
#     results_df = pd.DataFrame(
#         data=results_array,
#         columns=[column1, column2, column3, column4, 'p-value'])
#     results_t = pd.DataFrame(
#         data=results_t,
#         columns=['Variable', 'median(IQR)'])
#     return results_df, suitable_num, results_t
#
#
# def numeric_results(data, numeric_variables):
#     results_t = []
#     for variable in numeric_variables:
#         try:
#             data[variable] = pd.to_numeric(
#                 data[variable], errors='coerce')
#             data_variable = data[[variable]].dropna()
#             data_t = data_variable[variable]
#             # complete = round((100 * (len(data_variable) / len(data))), 1)
#             if len(data_variable) > 2:
#                 detail_t = str(round(data_t.median(), 1)) + ' ('
#                 detail_t += str(round(data_t.quantile(0.25), 1)) + '-'
#                 detail_t += str(round(data_t.quantile(0.75), 1)) + ')'
#             else:
#                 detail_t = str(round(data_t.mean(), 1)) + ' ('
#                 detail_t += str(round(data_t.std(), 1)) + ')'
#             results_t.append([variable, detail_t])
#         except Exception:
#             print(variable)
#     results_t = pd.DataFrame(
#         data=results_t, columns=['Variable', 'median(IQR)'])
#     return results_t
#
#
# def descriptive_table(data, correct_names, categoricals, numericals):
#     categorical_results_t = categorical_feature(
#         data, list(set(categoricals).intersection(set(data.columns))))
#     numeric_results_t = numeric_results(
#         data, list(set(numericals).intersection(set(data.columns))))
#
#     table = pd.merge(
#         categorical_results_t, numeric_results_t, on='Variable', how='outer')
#
#     table['Variable'] = table['Variable'].apply(lambda x: x.split(':')[0])
#     table['Variable'] = table['Variable'].replace(
#         dict(zip(correct_names['field_name'], correct_names['field_label'])))
#     # table['Variable'] = table['Variable'].replace(correct_names)
#     table = table.fillna('')
#     return table
