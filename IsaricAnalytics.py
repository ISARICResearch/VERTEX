import warnings

import numpy as np
import pandas as pd

# from sklearn.metrics import balanced_accuracy_score, make_scorer
# from typing import List, Union
# from bertopic import BERTopic
# from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
# from bertopic._utils import select_topic_representation
# from umap import UMAP
# from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from scipy.stats import norm

# from scipy.stats import fisher_exact
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from sklearn.impute import KNNImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
# import itertools
# from collections import OrderedDict

############################################
############################################
# General preprocessing
############################################
############################################


def extend_dictionary(dictionary, new_variable_dict, data, sep='___'):
    '''Add new custom variables to the VERTEX dictionary.

    Args:
        dictionary (pd.DataFrame):
            VERTEX dictionary containing columns 'field_name', 'form_name',
            'field_type', 'field_label', 'parent'.
        new_variable_dict (dict):
            A dict with the same keys as the dictionary columns, the values for
            each item can be a string or a list.
        data (pd.DataFrame):
            pandas dataframe containing the data for the project. The columns
            of this dataframe must include the variables in
            new_variable_dict['field_type'].
        sep (str): separator for creating new one-hot-encoded variable names.

    Returns:
        dictionary (pd.DataFrame):
            VERTEX dictionary containing the original variables, plus the new
            variables and any one-hot-encoded variables derived from this.'''
    # Convert dict values to list if all are strings (otherwise a pandas error)
    if all(isinstance(v, str) for v in new_variable_dict.values()):
        new_variable_dict = {k: [v] for k, v in new_variable_dict.items()}
    new_dictionary = pd.DataFrame.from_dict(new_variable_dict)
    new_dictionary['index'] = np.nan
    dictionary = dictionary.reset_index(drop=False)
    for ind in new_dictionary.index:
        parent = new_dictionary.loc[ind, 'parent']
        if (parent not in new_dictionary['field_name'].values):
            if (parent not in dictionary['field_name'].values):
                new_ind = dictionary['index'].max() + 0.1
            else:
                new_ind = dictionary.loc[(
                    dictionary['field_name'] == parent), 'index'].max()
                parent = dictionary.loc[new_ind, 'field_name']
                while (parent in dictionary['parent'].values):
                    new_ind = dictionary.loc[(
                        dictionary['parent'] == parent), 'index'].max()
                    parent = dictionary.loc[new_ind, 'field_name']
                new_ind = new_ind + 0.1
        else:
            new_ind = new_dictionary.loc[(
                new_dictionary['field_name'] == parent), 'index'].max() + 0.1
            # new_ind = new_ind + 0.1
        new_dictionary.loc[ind, 'index'] = new_ind
    categorical_ind = new_dictionary['field_type'].isin(['categorical'])
    new_dictionary_list = [new_dictionary]
    ind_list = [
        ind for ind in categorical_ind.loc[categorical_ind].index
        if new_dictionary.loc[ind, 'field_name'] in data.columns]
    for ind in ind_list:
        variable = new_dictionary.loc[ind, 'field_name']
        options = data[variable].drop_duplicates().sort_values().dropna()
        options = [y for y in options if y not in (True, False)]
        options = [
            y for y in options
            if (variable + sep + str(y))
            not in new_dictionary['field_name'].values]
        add_options = pd.DataFrame(
            columns=dictionary.columns, index=range(len(options)))
        add_options['field_name'] = [
            variable + sep + str(y) for y in options]
        add_options['form_name'] = new_dictionary.loc[ind, 'form_name']
        add_options['field_type'] = 'binary'
        add_options['field_label'] = options
        add_options['parent'] = variable
        add_options['index'] = np.linspace(
            new_dictionary.loc[ind, 'index'] + 0.2,
            np.ceil(new_dictionary.loc[ind, 'index']) - 0.1, len(options))
        new_dictionary_list += [add_options]
    dictionary = pd.concat([dictionary] + new_dictionary_list, axis=0)
    dictionary = dictionary.sort_values(by='index').drop(columns='index')
    dictionary = dictionary.reset_index(drop=True)
    return dictionary


def get_variables_by_section_and_type(
        df, dictionary,
        required_variables=None,
        include_sections=['demog'],
        include_types=['binary', 'categorical', 'numeric'],
        exclude_suffix=[
            '_units', 'addi', 'otherl2', 'item', '_oth',
            '_unlisted', 'otherl3'],
        include_subjid=False):
    '''
    Get all variables in the dataframe from specified sections and types,
    plus any required variables.
    '''
    include_ind = dictionary['field_name'].apply(
        lambda x: x.startswith(tuple(x + '_' for x in include_sections)))
    include_ind &= dictionary['field_type'].isin(include_types)
    # include_ind &= (dictionary['field_name'].apply(
    #     lambda x: x.endswith(tuple('___' + x for x in exclude_suffix))) == 0)
    include_ind &= (dictionary['field_name'].apply(
        lambda x: x.endswith(tuple(x for x in exclude_suffix))) == 0)
    if isinstance(required_variables, list):
        include_ind |= dictionary['field_name'].isin(required_variables)
    if include_subjid:
        include_ind |= (dictionary['field_name'] == 'subjid')
    include_variables = dictionary.loc[include_ind, 'field_name'].tolist()
    include_variables = [col for col in include_variables if col in df.columns]
    return include_variables


def convert_categorical_to_onehot(
        df, dictionary, categorical_columns,
        sep='___', missing_val='nan', drop_first=False):
    '''Convert categorical variables into onehot-encoded variables.'''
    categorical_columns = [
        col for col in df.columns if col in categorical_columns]

    df.loc[:, categorical_columns] = (
        df[categorical_columns].fillna(missing_val))
    df = pd.get_dummies(
        df, columns=categorical_columns, prefix_sep=sep)

    for categorical_column in categorical_columns:
        onehot_columns = [
            var for var in df.columns
            if (var.split(sep)[0] == categorical_column)]
        # variable_type_dict['binary'] += onehot_columns
        df[onehot_columns] = df[onehot_columns].astype(object)
        if (categorical_column + sep + missing_val) in df.columns:
            mask = (df[categorical_column + sep + missing_val] == 1)
            df.loc[mask, onehot_columns] = np.nan
            df = df.drop(columns=[categorical_column + sep + missing_val])
        else:
            if drop_first:
                drop_column_ind = dictionary.apply(
                    lambda x: (
                        (x['parent'] == categorical_column) &
                        (x['field_name'].split('___')[0] == categorical_column)
                    ), axis=1)
                df = df.drop(columns=[
                    dictionary.loc[drop_column_ind, 'field_name'].values[0]])

    columns = [
        col for col in dictionary['field_name'].values if col in df.columns]
    columns += [
        col for col in df.columns
        if col not in dictionary['field_name'].values]
    df = df[columns]
    return df


def convert_onehot_to_categorical(
        df, dictionary, categorical_columns, sep='___', missing_val='nan'):
    '''Convert onehot-encoded variables into categorical variables.'''
    df = pd.concat([df, pd.DataFrame(columns=categorical_columns)], axis=1)
    for categorical_column in categorical_columns:
        onehot_columns = list(
            df.columns[df.columns.str.startswith(categorical_column + sep)])
        # Preserve missingness
        df.loc[:, categorical_column + sep + missing_val] = (
            (df[onehot_columns].any(axis=1) == 0) |
            (df[onehot_columns].isna().any(axis=1)))
        with pd.option_context('future.no_silent_downcasting', True):
            df.loc[:, onehot_columns] = df[onehot_columns].fillna(False)
        onehot_columns += [categorical_column + sep + missing_val]
        df.loc[:, categorical_column] = pd.from_dummies(
            df[onehot_columns], sep=sep)
        df = df.drop(columns=onehot_columns)

    columns = [
        col for col in dictionary['field_name'].values if col in df.columns]
    columns += [
        col for col in df.columns
        if col not in dictionary['field_name'].values]
    df = df[columns]
    return df


# def merge_categories_except_list(
#         data, column, required_values=[], merged_value='Other'):
#     data.loc[(data[column].isin(required_values) == 0), column] = merged_value
#     return data
#
#
# def merge_cat_max_ncat(data, column, max_ncat=4, merged_value='Other'):
#     required_choices_list = data[column].value_counts().head(n=max_ncat)
#     required_choices_list = required_choices_list.index.tolist()
#     data = merge_categories_except_list(
#         data, column, required_choices_list, merged_value)
#     return data
#
#
# def add_day_variables(data, date_columns):
#     '''
#     Add new variables for each date variable, for the days since admission.
#     Some values will be negative if the event occurred before admission.'''
#     try:
#         days_columns = [
#             col.split('dates_')[-1].replace('date', '').strip('_')
#             for col in date_columns]
#         days_columns = ['days_adm_to_' + x for x in days_columns]
#         data[days_columns] = np.nan
#         for days, date in zip(days_columns, date_columns):
#             data[days] = (data[date] - data['dates_admdate']).dt.days
#     except Exception:
#         pass
#     return data


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


def get_descriptive_data(
        data, dictionary, by_column=None, include_sections=['demog'],
        include_types=['binary', 'categorical', 'numeric'],
        exclude_suffix=[
            '_units', 'addi', 'otherl2', 'item', '_oth',
            '_unlisted', 'otherl3'],
        include_subjid=False, exclude_negatives=True):
    df = data.copy()

    include_columns = get_variables_by_section_and_type(
        df, dictionary,
        include_types=include_types, include_subjid=include_subjid,
        include_sections=include_sections, exclude_suffix=exclude_suffix)
    if (by_column is not None) & (by_column not in include_columns):
        include_columns = [by_column] + include_columns
    df = df[include_columns].dropna(axis=1, how='all').copy()

    # Convert categorical variables to onehot-encoded binary columns
    categorical_ind = (dictionary['field_type'] == 'categorical')
    columns = dictionary.loc[categorical_ind, 'field_name'].tolist()
    columns = [col for col in columns if col != by_column]
    df = convert_categorical_to_onehot(
        df, dictionary, categorical_columns=columns)

    if (by_column is not None) & (by_column not in df.columns):
        df = convert_onehot_to_categorical(
            df, dictionary, categorical_columns=[by_column])

    negative_values = ('no', 'never smoked')
    negative_columns = [
        col for col in df.columns
        if col.split('___')[-1].lower() in negative_values]
    if exclude_negatives:
        df.drop(columns=negative_columns, inplace=True)

    # Remove columns with only NaN values
    df = df.dropna(axis=1, how='all')
    df.fillna({by_column: 'Unknown'}, inplace=True)
    return df


def descriptive_table(
        data, dictionary, by_column=None,
        include_totals=True, column_reorder=None,
        include_raw_variable_name=False):
    '''
    Descriptive table for binary (including onehot-encoded categorical) and
    numerical variables in data. The descriptive table will have seperate
    columns for each category that exists for the variable 'by_column', if
    this is provided.
    '''
    df = data.copy()

    numeric_ind = (dictionary['field_type'] == 'numeric')
    numeric_columns = dictionary.loc[numeric_ind, 'field_name'].tolist()
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    binary_ind = (dictionary['field_type'] == 'binary')
    binary_columns = dictionary.loc[binary_ind, 'field_name'].tolist()
    binary_columns = [col for col in binary_columns if col in df.columns]

    # Add columns for section headers and categorical questions
    index = numeric_columns + binary_columns
    index += dictionary.loc[(
        dictionary['field_name'].isin(index)), 'parent'].tolist()
    table_dictionary = dictionary.loc[(dictionary['field_name'].isin(index))]
    index = table_dictionary['field_name'].tolist()

    table_columns = ['Variable', 'All']
    if by_column is not None:
        add_columns = list(df[by_column].unique())
        if column_reorder is not None:
            table_columns += [
                col for col in column_reorder if col in add_columns]
            table_columns += [
                col for col in add_columns if col not in column_reorder]
        else:
            table_columns += add_columns
    table_columns += ['Raw variable name']
    table = pd.DataFrame('', index=index, columns=table_columns)

    table['Raw variable name'] = [
        var if var in df.columns else '' for var in index]
    table['Variable'] = format_descriptive_table_variables(
        table_dictionary).tolist()

    mfw = int(np.log10(df.shape[0])) + 1  # Min field width, for formatting
    table.loc[numeric_columns, 'All'] = df[numeric_columns].apply(
        median_iqr_str, mfw=mfw)
    table.loc[binary_columns, 'All'] = df[binary_columns].apply(
        n_percent_str, mfw=mfw)

    totals = pd.DataFrame(columns=table_columns, index=['totals'])
    totals['Variable'] = '<b>Totals</b>'
    totals['All'] = df.shape[0]

    if by_column is not None:
        choices_values = df[by_column].unique()
        for value in choices_values:
            ind = (df[by_column] == value)
            mfw = int(np.log10(ind.sum())) + 1  # Min field width, for format
            table.loc[numeric_columns, value] = (
                df.loc[ind, numeric_columns].apply(median_iqr_str, mfw=mfw))
            table.loc[binary_columns, value] = (
                df.loc[ind, binary_columns].apply(n_percent_str, mfw=mfw))
            totals[value] = ind.sum()

    table = table.reset_index(drop=True)
    if include_totals:
        table = pd.concat([totals, table], axis=0).reset_index(drop=True)
    table_key = '<b>KEY</b><br>(*) Count (%) | N<br>(+) Median (IQR) | N'
    if include_raw_variable_name is False:
        table.drop(columns=['Raw variable name'], inplace=True)
    return table, table_key


############################################
############################################
# Formatting
############################################
############################################


def trim_field_label(x, max_len=40):
    if len(x) > max_len:
        x = ' '.join(x[:max_len].split(' ')[:-1]) + ' ...'
    return x


def format_descriptive_table_variables(dictionary, max_len=100, add_key=True):
    name = dictionary['field_name'].apply(
        lambda x: '   ↳ ' if '___' in x else '<b>')
    name += dictionary['field_type'].map({'section': '<i>'}).fillna('')
    name += dictionary['field_label'].apply(
        lambda x: x.split(':')[-1] if x.startswith('If') else x).apply(
        trim_field_label, max_len=max_len)
    name += dictionary['field_type'].map({'section': '</i>'}).fillna('')
    name += dictionary['field_name'].apply(
        lambda x: '' if '___' in x else '</b>')
    if add_key is True:
        field_type = dictionary['field_type'].map({
            'categorical': ' (*)',
            'binary': ' (*)',
            'numeric': ' (+)'}).fillna('')
        name += field_type*(dictionary['field_name'].str.contains('___') == 0)
    return name


def format_variables(dictionary, max_len=40):
    parent_label = dictionary['parent'].apply(
        lambda x: dictionary.loc[(
            dictionary['field_name'] == x).idxmax(), 'field_label'])
    parent_name = parent_label.apply(trim_field_label, max_len=max_len)
    name = dictionary['field_label'].apply(
        lambda x: x.split(':')[-1] if x.startswith('If') else x).apply(
        trim_field_label, max_len=max_len)
    answer_ind = dictionary['field_name'].str.contains('___')
    name = (
        ('<b>' + parent_name + '</b>, ' + name)*answer_ind +
        ('<b>' + name + '</b>')*(answer_ind == 0))
    return name


############################################
############################################
# Counts
############################################
############################################


def get_counts(df, dictionary, max_n_variables=10):
    counts = df.apply(lambda x: x.sum()).T.reset_index()

    counts.columns = ['variable', 'count']
    counts = counts.sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    if counts.shape[0] > max_n_variables:
        counts = counts.head(max_n_variables)

    short_format = format_variables(dictionary, max_len=40)
    long_format = format_variables(dictionary, max_len=1000)
    format_dict = dict(zip(dictionary['field_name'], long_format))
    short_format_dict = dict(zip(dictionary['field_name'], short_format))
    counts['label'] = counts['variable'].map(format_dict)
    counts['short_label'] = counts['variable'].map(short_format_dict)
    return counts


def get_proportions(df, dictionary, max_n_variables=10):
    proportions = df.apply(
        lambda x: (x.sum() / x.count(), x.sum())).T.reset_index()

    proportions.columns = ['variable', 'proportion', 'count']
    proportions = proportions.sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    if proportions.shape[0] > max_n_variables:
        proportions = proportions.head(max_n_variables)

    proportions = proportions.drop(columns='count')
    proportions = proportions.sort_values(
        by=['proportion'], ascending=False).reset_index(drop=True)

    short_format = format_variables(dictionary, max_len=40)
    long_format = format_variables(dictionary, max_len=1000)
    format_dict = dict(zip(dictionary['field_name'], long_format))
    short_format_dict = dict(zip(dictionary['field_name'], short_format))
    proportions['label'] = proportions['variable'].map(format_dict)
    proportions['short_label'] = proportions['variable'].map(short_format_dict)
    return proportions


def get_upset_counts_intersections(
        df, dictionary,
        proportions=None,  # Deprecated
        variables=None, n_variables=5):
    # Convert variables and column names into their formatted names
    long_format = format_variables(dictionary, max_len=1000)
    short_format = format_variables(dictionary, max_len=40)
    format_dict = dict(zip(dictionary['field_name'], long_format))
    short_format_dict = dict(zip(dictionary['field_name'], short_format))

    if variables is None:
        variables = df.columns.tolist()

    binary_columns = dictionary.loc[(
        dictionary['field_type'] == 'binary'), 'field_name'].tolist()
    variables = [col for col in variables if col in binary_columns]
    variables = [var for var in variables if df[var].sum() > 0]
    df = df[variables].astype(float).fillna(0)

    counts = df.sum().astype(int).reset_index().rename(columns={0: 'count'})
    counts = counts.sort_values(
        by='count', ascending=False).reset_index(drop=True)
    counts['short_label'] = counts['index'].map(short_format_dict)
    counts['label'] = counts['index'].map(format_dict)

    variable_order_dict = dict(zip(counts['index'], counts.index))
    variables = counts['index'].tolist()
    if n_variables is not None:
        variables = variables[:n_variables]

    df = df[variables]
    counts = counts.loc[counts['index'].isin(variables)]

    intersections = df.loc[df.any(axis=1)].value_counts().reset_index()
    intersections['index'] = intersections.drop(columns='count').apply(
        lambda x: tuple(col for col in x.index if x[col] == 1), axis=1)
    intersections['label'] = intersections['index'].apply(
        lambda x: tuple(format_dict[y] for y in x))

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
    keep_columns = ['index', 'label', 'count']
    intersections = intersections[keep_columns].reset_index(drop=True)
    return counts, intersections


def get_pyramid_data(df, column_dict, left_side='Female', right_side='Male'):
    keys = ['side', 'y_axis', 'stack_group']
    # assert all(key in tuple(column_dict.keys()) for key in keys), 'Error'
    columns = [column_dict[key] for key in keys]
    df_pyramid = df[['subjid'] + columns].copy()
    df_pyramid = df_pyramid.groupby(
        columns, observed=True).count().reset_index()
    df_pyramid.rename(columns={'subjid': 'value'}, inplace=True)
    df_pyramid.rename(
        columns={v: k for k, v in column_dict.items()}, inplace=True)
    df_pyramid = df_pyramid.loc[
        df_pyramid['side'].isin([left_side, right_side])]
    df_pyramid.loc[:, 'left_side'] = (df_pyramid['side'] == left_side)
    df_pyramid = df_pyramid.sort_values(by='y_axis').reset_index(drop=True)
    return df_pyramid


############################################
############################################
# Clustering of free-text terms
############################################
############################################

# Comment out for now until used
'''
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


def get_clusters(
        terms: List[str],
        nr_topics: Union[str, int] = 'auto'):
    """Function to find common topics appearing in a list of free text terms.
    Uses the BERTopic topic modelling pipeline.

    Args:
        terms (List[str]):
            list of free text terms, for example referring to an
            'other combordities' field in a CRF
        nr_topics (Union[str, int]): number of topics to model, 'auto' or int
            specifying desired number

    Returns:
        clusters_df (pd.DataFrame):
            pandas dataframe summarizing the results of the clustering process.
            Contains the following columns:
        Topic (int): topic id
        Count (int): number of rows in terms assigned to that topic
        Name (str): name of topic, default id + keywords
        Representation (List[str]): list of keywords in topic
        Representative_Docs (List[str]):
            list of entries in terms which represent the topic
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

    # use bertopic topic modelling pipeline
    topic_model = BERTopic(
        embedding_model=embedding_model,
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


def extract_topic_embeddings(
        topic_model: BERTopic,
        topics: List[int] = None,
        top_n_topics: int = None,
        use_ctfidf: bool = False):
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
        embeddings = UMAP(
            n_neighbors=2, n_components=2,
            metric="hellinger", random_state=42).fit_transform(embeddings)
    else:
        embeddings = UMAP(
            n_neighbors=2, n_components=2,
            metric="cosine", random_state=42).fit_transform(embeddings)

    # assemble df
    df = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "Topic": topic_list,
        }
    )
    return df
'''

############################################
############################################
# Logistic Regression from Risk Factors
############################################
############################################


def get_modelling_data(
        data, dictionary, outcome_columns,
        include_sections=[
            'demog', 'comor', 'adsym', 'vacci', 'vital', 'sympt', 'labs'],
        required_variables=None,
        include_types=['binary', 'categorical', 'numeric'],
        exclude_suffix=[
            '_units', 'addi', 'otherl2', 'item', '_oth',
            '_unlisted', 'otherl3'],
        include_subjid=False, exclude_negatives=True,
        fillna=True, drop_first=False):
    df = data.copy()

    if isinstance(outcome_columns, str):
        outcome_columns = [outcome_columns]

    include_columns = get_variables_by_section_and_type(
        df, dictionary,
        required_variables=required_variables,
        include_types=include_types, include_subjid=include_subjid,
        include_sections=include_sections, exclude_suffix=exclude_suffix)
    for outcome_column in outcome_columns:
        if (outcome_column not in include_columns):
            include_columns = [outcome_column] + include_columns
    df = df[include_columns].dropna(axis=1, how='all').copy()

    # Convert categorical variables to onehot-encoded binary columns
    categorical_ind = (dictionary['field_type'] == 'categorical')
    columns = dictionary.loc[categorical_ind, 'field_name'].tolist()
    columns = [col for col in columns if col not in tuple(outcome_columns)]
    df = convert_categorical_to_onehot(
        df, dictionary, categorical_columns=columns, drop_first=drop_first)

    binary_ind = (dictionary['field_type'] == 'binary')
    columns = dictionary.loc[binary_ind, 'field_name'].tolist()
    columns = [col for col in columns if col in df.columns]
    if fillna is True:
        with pd.option_context('future.no_silent_downcasting', True):
            df[columns] = df[columns].fillna(False)

    negative_values = ('no', 'never smoked')
    negative_columns = [
        col for col in df.columns
        if col.split('___')[-1].lower() in negative_values]
    if exclude_negatives:
        df.drop(columns=negative_columns, inplace=True)
    return df


def variance_influence_factor_backwards_elimination(
        data, dictionary, predictors_list, sep='___'):
    df = data.copy()

    numeric_ind = (dictionary['field_type'] == 'numeric')
    numeric_columns = dictionary.loc[numeric_ind, 'field_name'].tolist()
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    numeric_columns = [
        col for col in numeric_columns if col in predictors_list]

    categorical_ind = dictionary['field_type'].isin(['binary'])
    categorical_columns = dictionary.loc[
        categorical_ind, 'field_name'].tolist()
    categorical_columns = [
        col for col in categorical_columns if col in df.columns]
    categorical_columns = [
        col for col in categorical_columns if col in predictors_list]

    df[numeric_columns] = ((
            df[numeric_columns] - df[numeric_columns].mean()
        ) / df[numeric_columns].std())

    df = 1.0 * pd.concat([
            df[numeric_columns],
            df[categorical_columns]
        ], axis=1).astype(float)

    keep_columns = df.columns
    iterative_vif = pd.DataFrame(keep_columns, columns=['variable'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        iterative_vif['vif_iter_1'] = [
            variance_inflation_factor(df[keep_columns].values, ii)
            for ii in range(len(keep_columns))]
    n = 1
    while (iterative_vif['vif_iter_' + str(n)] > 10).any():
        remove_column = iterative_vif.loc[
            iterative_vif['vif_iter_' + str(n)].idxmax(), 'variable']
        remove_column = remove_column.split(sep)[0]
        keep_columns = [
            col for col in keep_columns
            if (col.split(sep)[0] != remove_column)]
        n += 1
        vif = pd.DataFrame(keep_columns, columns=['variable'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            vif['vif_iter_' + str(n)] = [
                variance_inflation_factor(df[keep_columns].values, ii)
                for ii in range(len(keep_columns))]
        iterative_vif = pd.merge(iterative_vif, vif, how='left', on='variable')
    return keep_columns, iterative_vif


def remove_single_binary_outcome_predictors(
        data, dictionary, predictors_list, outcome_str):
    """
    Removes binary predictors that are associated with only one outcome (e.g.
    if all patients with some_variable=1 have outcome=1)

    Parameters:
    - data: Pandas DataFrame containing the data.
    - outcome_str: Name of the response variable.
    - predictors_list: List of predictor variable names.

    Returns:
    - updated_predictors_list: List of predictor variable names excluding
    any that can't be used in the logistic regression model.
    """
    df = data.copy()

    numeric_ind = (dictionary['field_type'] == 'numeric')
    numeric_columns = dictionary.loc[numeric_ind, 'field_name'].tolist()
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    numeric_columns = [
        col for col in numeric_columns if col in predictors_list]

    categorical_ind = dictionary['field_type'].isin(['binary'])
    categorical_columns = dictionary.loc[
        categorical_ind, 'field_name'].tolist()
    categorical_columns = [
        col for col in categorical_columns if col in data.columns]
    categorical_columns = [
        col for col in categorical_columns if col in predictors_list]

    result = df.groupby(outcome_str)[categorical_columns].apply(
        lambda x: x.isin([True]).any() & x.isin([False]).any(),
        include_groups=False).T
    keep_columns = result.loc[result.all(axis=1)].index.tolist()
    keep_columns = keep_columns + numeric_columns
    return keep_columns


def regression_summary_table(
        table, dictionary,
        highlight_predictors=None, p_values=None, result_type='OddsRatio'):
    variables = table['Variable'].tolist()
    new_variables = variables + dictionary.loc[(
            (dictionary['field_name'].isin(variables)) &
            (dictionary['parent'].isin(['']) == 0)
        ), 'parent'].tolist()
    new_variables = list(set(new_variables))
    while (len(variables) != len(new_variables)):
        variables = new_variables
        new_variables = variables + dictionary.loc[(
                (dictionary['field_name'].isin(variables)) &
                (dictionary['parent'].isin(['']) == 0)
            ), 'parent'].tolist()
        new_variables = list(set(new_variables))
    # Reorders the variables
    dictionary = dictionary.set_index('field_name')
    nonrepeated_parent = dictionary.loc[variables].groupby('parent').apply(
        len, include_groups=False).eq(1)
    nonrepeated_parent = [
        p for p in nonrepeated_parent.loc[nonrepeated_parent].index
        if (dictionary.loc[p, 'field_type'] != 'section')]
    variables = [var for var in variables if var not in nonrepeated_parent]
    dictionary = dictionary.reset_index()

    variables = dictionary.loc[(
        dictionary['field_name'].isin(variables)), 'field_name'].tolist()

    for reg_type in ['multi', 'uni']:
        table[f'{result_type} ({reg_type})'] = table.apply(
            lambda x:
                '%.2f' % x[f'{result_type} ({reg_type})'] +
                ' (' + '%.2f' % x[f'LowerCI ({reg_type})'] +
                ', ' + '%.2f' % x[f'UpperCI ({reg_type})'] + ')', axis=1)

        if p_values is not None:
            significance = pd.Series('', index=table.index)
            for key in p_values:
                min_threshold = max(
                    v for k, v in p_values.items() if (k != key))
                if (min_threshold > p_values[key]):
                    min_threshold = 0
                ind = (
                    (table[f'p-value ({reg_type})'] < p_values[key]) &
                    (table[f'p-value ({reg_type})'] > (min_threshold - 1e-8)))
                significance.loc[ind] = f' ({key})'

            table[f'p-value ({reg_type})'] = (
                table[f'p-value ({reg_type})'].apply(
                    lambda x: '%.3f' % x) + significance)

        table = table.drop(
            columns=[f'LowerCI ({reg_type})', f'UpperCI ({reg_type})'])

    add_variables = [
        var for var in variables if var not in table['Variable'].tolist()]
    add_table = pd.DataFrame(
        '', columns=table.columns, index=range(len(add_variables)))
    add_table['Variable'] = add_variables
    table = pd.concat([table, add_table], axis=0)
    table = table.set_index('Variable').loc[variables].reset_index()

    add_key = pd.Series('', index=table.index)
    if highlight_predictors is not None:
        for key in highlight_predictors:
            add_key.loc[
                table['Variable'].isin(highlight_predictors[key])] += key
        add_key = add_key.apply(lambda x: x if x == '' else f' ({x})')

    formatted_labels_v1 = format_descriptive_table_variables(
        dictionary, add_key=False)
    formatted_labels_v2 = format_variables(dictionary)
    v1_ind = (dictionary['parent'].isin(variables))
    v2_ind = (dictionary['parent'].isin(variables) == 0)
    mapping_dict = {
        **dict(zip(
            dictionary.loc[v1_ind, 'field_name'],
            formatted_labels_v1.loc[v1_ind])),
        **dict(zip(
            dictionary.loc[v2_ind, 'field_name'],
            formatted_labels_v2.loc[v2_ind]))}
    table['Variable'] = table['Variable'].map(mapping_dict)
    table['Variable'] = table['Variable'] + add_key
    return table


def execute_glmm_regression(elr_dataframe_df, elr_outcome_str, elr_predictors_list,
                            elr_groups_str, model_type='linear',
                            print_results=True, labels=False, reg_type="multi"):
    """
    Executes a mixed effects model for linear or logistic regression.

    Parameters:
    - elr_dataframe_df: Pandas DataFrame containing the data.
    - elr_outcome_str: Name of the response variable.
    - elr_predictors_list: List of predictor variable names.
    - elr_groups_str: Name of the variable that defines the groups (random effect).
    - model_type: 'linear' for linear regression or 'logistic' for logistic regression.
    - print_results: If True, prints the summary of the results.
    - labels: (Optional) Dictionary to map variable names to readable labels.
    - reg_type: 'uni' or 'multi', to rename the output columns.

    Returns:
    - elr_summary_df: DataFrame with the model results.
    """

    # Builds the formula
    elr_formula_str = elr_outcome_str + ' ~ ' + ' + '.join(elr_predictors_list)

    # Converts predictor categorical variables
    elr_categorical_vars_list = elr_dataframe_df.select_dtypes(include=['object', 'category'])
    elr_categorical_vars_list = elr_categorical_vars_list.columns.intersection(elr_predictors_list)
    for elr_var_str in elr_categorical_vars_list:
        elr_dataframe_df[elr_var_str] = elr_dataframe_df[elr_var_str].astype('category')

    # Converts the groups column to string to ensure that the values are hashable
    elr_dataframe_df[elr_groups_str] = elr_dataframe_df[elr_groups_str].astype(str)

    if model_type.lower() == 'linear':
        # Mixed linear model using MixedLM (following your function)
        elr_model_obj = smf.mixedlm(formula=elr_formula_str,
                                    data=elr_dataframe_df,
                                    groups=elr_dataframe_df[elr_groups_str])
        elr_result_obj = elr_model_obj.fit()

        fixed_effects = elr_result_obj.fe_params
        conf_int_df = elr_result_obj.conf_int().loc[fixed_effects.index]
        pvalues = elr_result_obj.pvalues.loc[fixed_effects.index]

        elr_summary_df = pd.DataFrame({
            'Study': fixed_effects.index,
            'Coef': fixed_effects.values,
            'IC Low': conf_int_df.iloc[:, 0].values,
            'IC High': conf_int_df.iloc[:, 1].values,
            'p-value': pvalues.values
        })

    elif model_type.lower() == 'logistic':
        # Mixed logistic model using BinomialBayesMixedGLM (Bayesian approach via VB)

        # Defines vc_formula for random effect (random intercept per group)
        vc_formula = {elr_groups_str: "0 + C({})".format(elr_groups_str)}

        elr_model_obj = BinomialBayesMixedGLM.from_formula(formula=elr_formula_str,
                                                           vc_formulas=vc_formula,
                                                           data=elr_dataframe_df)
        elr_result_obj = elr_model_obj.fit_vb()

        # Extracts the fixed effect names and determines how many there are
        param_names = elr_model_obj.exog_names
        n_fixed = len(param_names)
        fixed_effects = pd.Series(elr_result_obj.params[:n_fixed], index=param_names)

        # Attempts to obtain the covariance matrix and extracts the slice corresponding to fixed effects
        try:
            cov_params = elr_result_obj.cov_params()
        except Exception:
            try:
                cov_params = elr_result_obj.vcov
            except Exception:
                cov_params = None
        if cov_params is not None:
            # If it is a DataFrame, use .iloc; otherwise, assume a NumPy array
            if hasattr(cov_params, 'iloc'):
                cov_params_fixed = cov_params.iloc[:n_fixed, :n_fixed]
            else:
                cov_params_fixed = cov_params[:n_fixed, :n_fixed]
            bse = np.sqrt(np.diag(cov_params_fixed))
            bse = pd.Series(bse, index=param_names)
            # Calculates p-values manually (Wald test, normal approximation)
            z_values = fixed_effects / bse
            pvalues = 2 * (1 - norm.cdf(np.abs(z_values)))
            pvalues = pd.Series(pvalues, index=param_names)
        else:
            bse = pd.Series(np.full(fixed_effects.shape, np.nan), index=param_names)
            pvalues = pd.Series(np.full(fixed_effects.shape, np.nan), index=param_names)

        # Calculates confidence intervals using 1.96 as the quantile of the normal distribution
        lower_ci = fixed_effects - 1.96 * bse
        upper_ci = fixed_effects + 1.96 * bse

        # Calculates Odds Ratios and corresponding intervals
        odds_ratios = np.exp(fixed_effects)
        odds_lower = np.exp(lower_ci)
        odds_upper = np.exp(upper_ci)

        elr_summary_df = pd.DataFrame({
            'Study': fixed_effects,
            'OddsRatio': odds_ratios.values,
            'IC Low': odds_lower.values,
            'IC High': odds_upper.values,
            'p-value': pvalues.values
        })
    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

    # Applies label mapping if provided
    if labels:
        def elr_parse_variable_name(var_name):
            if var_name == 'Intercept' or var_name.lower() == 'const':
                return labels.get('Intercept', 'Intercept')
            elif '[' in var_name:
                base_var = var_name.split('[')[0]
                level = var_name.split('[')[1].split(']')[0]
                base_var_name = base_var.replace('C(', '').replace(')', '').strip()
                label = labels.get(base_var_name, base_var_name)
                return f'{label} ({level})'
            else:
                var_name_clean = var_name.replace('C(', '').replace(')', '').strip()
                return labels.get(var_name_clean, var_name_clean)
        elr_summary_df['Study'] = elr_summary_df['Study'].apply(elr_parse_variable_name)

    # Removes the intercept row if present
    elr_summary_df = elr_summary_df[~elr_summary_df['Study'].isin(['Intercept', 'const'])]

    # Reorders the columns according to the model
    if model_type.lower() == 'logistic':
        elr_summary_df = elr_summary_df[['Study', 'OddsRatio', 'IC Low', 'IC High', 'p-value']]
    else:
        elr_summary_df = elr_summary_df[['Study', 'Coef', 'IC Low', 'IC High', 'p-value']]

    # Formats the numerical values
    if model_type.lower() == 'logistic':
        elr_summary_df['OddsRatio'] = elr_summary_df['OddsRatio'].round(3)
    else:
        elr_summary_df['Coef'] = elr_summary_df['Coef'].round(3)
    elr_summary_df['IC Low'] = elr_summary_df['IC Low'].round(3)
    elr_summary_df['IC High'] = elr_summary_df['IC High'].round(3)
    elr_summary_df['p-value'] = elr_summary_df['p-value'].apply(lambda x: f'{x:.4f}')

    # Renames the columns according to the reg_type parameter
    if reg_type.lower() == 'uni':
        if model_type.lower() == 'logistic':
            elr_summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (uni)',
                'IC Low': 'LowerCI (uni)',
                'IC High': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
        else:
            elr_summary_df.rename(columns={
                'Coef': 'Coef (uni)',
                'IC Low': 'LowerCI (uni)',
                'IC High': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
    else:
        if model_type.lower() == 'logistic':
            elr_summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (multi)',
                'IC Low': 'LowerCI (multi)',
                'IC High': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)
        else:
            elr_summary_df.rename(columns={
                'Coef': 'Coef (multi)',
                'IC Low': 'LowerCI (multi)',
                'IC High': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)

    if print_results:
        print(elr_summary_df)

    return elr_summary_df


def execute_glm_regression(elr_dataframe_df, elr_outcome_str, elr_predictors_list,
                           model_type='linear', print_results=True, labels=False, reg_type="Multi"):
    """
    Executes a GLM (Generalized Linear Model) for linear or logistic regression.

    Parameters:
    - elr_dataframe_df: Pandas DataFrame containing the data.
    - elr_outcome_str: Name of the response variable.
    - elr_predictors_list: List of predictor variable names.
    - model_type: 'linear' for linear regression (Gaussian) or 'logistic' for logistic regression (Binomial).
    - print_results: If True, prints the results table.
    - labels: (Optional) Dictionary to map variable names to readable labels.
    - reg_type: Type of regression ('uni' or 'multi') to rename the output columns.

    Returns:
    - summary_df: DataFrame with the model results.
    """

    # Defines the family according to model_type
    if model_type.lower() == 'logistic':
        family = sm.families.Binomial()
    elif model_type.lower() == 'linear':
        family = sm.families.Gaussian()
    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

    # Builds the formula
    formula = elr_outcome_str + ' ~ ' + ' + '.join(elr_predictors_list)

    # Converts categorical variables to 'category' type
    categorical_vars = elr_dataframe_df.select_dtypes(include=['object', 'category']).columns.intersection(elr_predictors_list)
    for var in categorical_vars:
        elr_dataframe_df[var] = elr_dataframe_df[var].astype('category')

    # Fits the GLM model
    model = smf.glm(formula=formula, data=elr_dataframe_df, family=family)
    result = model.fit()

    # Extracts the results table
    summary_table = result.summary2().tables[1].copy()

    # For logistic regression, calculates Odds Ratios; for linear, uses the coefficients directly.
    if model_type.lower() == 'logistic':
        summary_table['Odds Ratio'] = np.exp(summary_table['Coef.'])
        summary_table['IC Low'] = np.exp(summary_table['[0.025'])
        summary_table['IC High'] = np.exp(summary_table['0.975]'])

        summary_df = summary_table[['Odds Ratio', 'IC Low', 'IC High', 'P>|z|']].reset_index()
        summary_df = summary_df.rename(columns={'index': 'Study',
                                                  'Odds Ratio': 'OddsRatio',
                                                  'IC Low': 'LowerCI',
                                                  'IC High': 'UpperCI',
                                                  'P>|z|': 'p-value'})
    else:
        summary_df = summary_table[['Coef.', '[0.025', '0.975]', 'P>|z|']].reset_index()
        summary_df = summary_df.rename(columns={'index': 'Study',
                                                  'Coef.': 'Coefficient',
                                                  '[0.025': 'LowerCI',
                                                  '0.975]': 'UpperCI',
                                                  'P>|z|': 'p-value'})

    # Maps variable names to readable labels, if provided
    if labels:
        def parse_variable_name(var_name):
            if var_name == 'Intercept':
                return labels.get('Intercept', 'Intercept')
            elif '[' in var_name:
                base_var = var_name.split('[')[0]
                level = var_name.split('[')[1].split(']')[0]
                base_var_name = base_var.replace('C(', '').replace(')', '').strip()
                label = labels.get(base_var_name, base_var_name)
                return f'{label} ({level})'
            else:
                var_name_clean = var_name.replace('C(', '').replace(')', '').strip()
                return labels.get(var_name_clean, var_name_clean)
        summary_df['Study'] = summary_df['Study'].apply(parse_variable_name)

    # Reorders the columns
    if model_type.lower() == 'logistic':
        summary_df = summary_df[['Study', 'OddsRatio', 'LowerCI', 'UpperCI', 'p-value']]
    else:
        summary_df = summary_df[['Study', 'Coefficient', 'LowerCI', 'UpperCI', 'p-value']]

    # Removes the letter 'T.' from categorical variables
    summary_df['Study'] = summary_df['Study'].str.replace('T.', '')

    # Formats the numerical values
    for col in summary_df.columns[1:-1]:
        summary_df[col] = summary_df[col].round(3)
    summary_df['p-value'] = summary_df['p-value'].apply(lambda x: f'{x:.4f}')

    # Removes the intercept row if desired (optional)
    summary_df = summary_df[summary_df['Study'] != 'Intercept']

    # Renames the columns according to the type of regression
    if reg_type.lower() == 'uni':
        if model_type.lower() == 'logistic':
            summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (uni)',
                'LowerCI': 'LowerCI (uni)',
                'UpperCI': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
        else:
            summary_df.rename(columns={
                'Coefficient': 'Coefficient (uni)',
                'LowerCI': 'LowerCI (uni)',
                'UpperCI': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
    elif reg_type.lower() == 'multi':
        if model_type.lower() == 'logistic':
            summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (multi)',
                'LowerCI': 'LowerCI (multi)',
                'UpperCI': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)
        else:
            summary_df.rename(columns={
                'Coefficient': 'Coefficient (multi)',
                'LowerCI': 'LowerCI (multi)',
                'UpperCI': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)

    if print_results:
        print(summary_df)

    return summary_df


def execute_cox_model(df, duration_col, event_col, predictors, labels=None):
    """
    Performs a Cox Proportional Hazards model without weights and
    returns a summary of the results.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - duration_col: String with the name of the time variable.
    - event_col: String with the name of the outcome variable (binary event).
    - predictors: List of strings with the names of predictor variables.
    - labels (Optional):
        Dictionary mapping variable names to readable labels.
        Default is None.

    Returns:
    - summary_df: DataFrame with the results of the Cox model.
    """

    # Ensure categorical variables are treated appropriately
    categorical_vars = df.select_dtypes(
        include=['object', 'category']).columns.intersection(predictors)
    for var in categorical_vars:
        df[var] = df[var].astype('category')

    # Convert categorical variables to dummies
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Ensure numerical variables have the correct type
    df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
    df[event_col] = pd.to_numeric(df[event_col], errors='coerce')

    # Update predictors to include one-hot encoded columns
    predictors = [
        c for c in df.columns
        if c in predictors or any(
            c.startswith(p + '_') for p in categorical_vars)]

    # Remove rows with missing values in essential columns
    df = df.dropna(subset=[duration_col, event_col] + predictors)

    # Select relevant columns
    df_cox = df[[duration_col, event_col] + predictors]

    # Fit the Cox model
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=duration_col, event_col=event_col)

    # Model summary
    summary = cph.summary
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_lower'] = np.exp(
        summary['coef'] - 1.96 * summary['se(coef)'])
    summary['CI_upper'] = np.exp(
        summary['coef'] + 1.96 * summary['se(coef)'])
    # summary['p_adj'] = summary['p'].apply(
    #     lambda p: "<0.001" if p < 0.001 else round(p, 3))
    summary['p_adj'] = summary['p'].apply(lambda p: round(p, 3))

    # Select relevant columns for the final summary
    summary_df = summary[[
        'HR', 'p_adj', 'CI_lower', 'CI_upper']].reset_index()
    summary_df.rename(
        columns={'index': 'Variable', 'p_adj': 'p-value'}, inplace=True)

    # Replace variable labels if provided
    if labels:
        summary_df['Variable'] = summary_df['Variable'].map(
            labels).fillna(summary_df['Variable'])

    return summary_df



############################################
############################################
# SOME RAPS
# (basic imputation, variance check, correlation check and feature selection)
############################################
############################################


def impute_miss_val(df, missing_threshold=0.7):
    """
    Imputes missing values or drops columns based on missing value proportion and median

     Returns:
    - df: DataFrame with missing values imputed or columns dropped
    """
    # Calculate the proportion of missing values in each column
    missing_proportions = df.isnull().mean()

    # Identify columns to drop
    cols_to_drop = missing_proportions[missing_proportions > missing_threshold].index
    df = df.drop(columns=cols_to_drop)

    # Impute missing values in remaining columns
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column: impute with median
                median_value = df[col].median()
                if pd.isnull(median_value):
                    # If median cannot be computed, drop the column
                    df = df.drop(columns=[col])
                else:
                   df[col] = df[col].fillna(median_value)
            else:
                # Categorical column: impute with mode
                mode_series = df[col].mode()
                if not mode_series.empty:
                    mode_value = mode_series[0]
                    df[col] = df[col].fillna(mode_value)
                else:
                    # If mode cannot be computed, drop the column
                    df = df.drop(columns=[col])
    print("\nSummary after Imputation")
    print("Size of remaining data:", df.shape)
    return df

def rmv_low_var(df, mad_threshold=0.1, freq_threshold=0.05):
    """
    Removes numerical variables with Median Absolute Deviation (MAD) below a threshold.
    Excludes binary columns from MAD calculation.
    Removes  binary columns with very low frequencies
    Returns:
    - df: pandas DataFrame with low MAD columns removed
    """

    # Remove single-valued columns first
    single_value_cols = df.columns[df.nunique() == 1]
    df = df.drop(columns=single_value_cols)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

   # Select numeric columns and identify binary columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    binary_cols = df.columns[df.nunique() == 2]
    non_binary_cols = numeric_cols.difference(binary_cols)

    print(f"\nMAD Analysis Summary:")
    print(f"Single value columns removed: {len(single_value_cols)}")
    print(f"Total binary columns: {len(binary_cols)}")
    print(f"Total numeric columns: {len(numeric_cols)}")
    print(f"Non binary numeric columns: {len(non_binary_cols)}")
    print(f"Numeric Binary columns excluded from MAD: {len(numeric_cols) - len(non_binary_cols)}")

    # Calculate low frequency binary numeric column
    # Handle binary columns - convert to numeric first
    if len(binary_cols) > 0:
        X_bin = df[binary_cols].apply(lambda x: pd.factorize(x)[0])
        binary_counts = X_bin.apply(pd.value_counts, normalize=True)
        keep_cols = [col for col in binary_cols if
                    binary_counts[col].min() >= freq_threshold]
        X_bin = df[keep_cols]  # Keep original values for these columns
    else:
        X_bin = pd.DataFrame()  # Empty DataFrame if no binary columns

    # Normalise the numeric columns by max
    df_tmp = df
    for col in non_binary_cols:
        df[col] = df_tmp[col]/np.max(np.abs(df_tmp[col]))

    # Calculate MAD for each non-binary numeric column
    mad_values = {}
    for col in non_binary_cols:
        mad = np.median(np.abs(df[col] - np.median(df[col])))
        mad_values[col] = mad

    # Create a Series from the MAD values
    mad_series = pd.Series(mad_values)
    #print(mad_series)

    # Identify columns to keep:
    # 1. Non-numeric columns
    # 2. Binary numeric columns
    # 3. Non-binary numeric columns with MAD above threshold
    cols_to_keep = set(df.columns) - set(non_binary_cols)-set(binary_cols)  # Start with all CAT columns
    cols_to_keep.update(mad_series[mad_series >= mad_threshold].index)  # Add high MAD columns

    # Keep only the identified columns
    X_comb = pd.concat([df[list(cols_to_keep)], X_bin], axis=1)
    df = X_comb

    # Print summary for debugging

    print("High Frequency Binary columns kept:", X_bin.shape)
    print(f"Columns removed due to low MAD: {len(non_binary_cols) - len(mad_series[mad_series >= mad_threshold])}")


    return df


def rmv_high_corr(df, correlation_threshold=0.5):
    # Step 1: Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]  # DataFrame with only numeric columns

    # Step 2: Calculate the correlation matrix
    corr_matrix = df_numeric.corr().abs()

    # Step 3: Identify highly correlated columns with a double loop
    to_drop = set()  # Use a set to avoid duplicates
    num_cols = corr_matrix.shape[0]

    for i in range(num_cols):
        for j in range(i + 1, num_cols):  # Only look at the upper triangle
            if corr_matrix.iloc[i, j] > correlation_threshold:
                # Identify the columns with high correlation
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]

                # Add one of the columns to `to_drop`
                to_drop.add(col2)  # Arbitrarily drop the second column

    # Step 4: Drop highly correlated columns

    print("\nCORR Summary")
    print(f"Columns removed due to high correlation: {len(to_drop)}")
    df = df.drop(columns=list(to_drop))

    return df



def lasso_var_sel_binary(df, outcome_col='mapped_outcome', random_state=42):
    """
    Prepare data and select features using binary logistic regression with elastic net penalty.
    Specifically designed for binary outcomes only.
    """
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame")


    # Separate predictors and outcome
    y = df[outcome_col].copy()
    X_ini = df.drop(columns=[outcome_col])
    print(f"\nInitial shape of X: {X_ini.shape}")

    # Encode the binary outcome
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Verify that we have a binary outcome
    n_classes = len(np.unique(y))
    if n_classes != 2:
        raise ValueError("This function is designed for binary classification only. More than two classes found.")

    # Standardize features
    scaler = StandardScaler()
    numeric_cols = X_ini.select_dtypes(include=[np.number]).columns
    print("Column dtypes:", X_ini.dtypes)
    print("Numeric columns found:", len(numeric_cols))

    X = X_ini.copy()
    if len(numeric_cols) > 0:
        X[numeric_cols] = scaler.fit_transform(X_ini[numeric_cols])
    else:
        print("No numeric columns to standardize")
    X = pd.DataFrame(X, columns=X_ini.columns)

    print("\nOutcome classes:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
    print(f"\nInitial shape of X: {X.shape}")
    # Encode categorical predictors
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    binary_cats = [col for col in categorical_cols if X[col].nunique() == 2]
    multi_cats = [col for col in categorical_cols if X[col].nunique() > 2]

    for col in binary_cats:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Use dummies only for multi-category
    X = pd.get_dummies(X, columns=multi_cats, prefix_sep='*_*')
    print(f"\nshape of X after one-hot: {X.shape}")
    X.to_csv('ISARIC_mpox2rmv_1hot.csv', index=True)

    if X.shape[1] > 0:
        print("First actual predictor column:", X.columns[0])
    else:
        raise ValueError("No predictor columns left after dropping outcome (and ID if applicable).")



    # Fit binary logistic regression with elastic net
    # For binary classification, multi_class defaults to 'ovr', which yields a single set of coefficients.
    l1_vec = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    C_vec = np.logspace(-4, 4, 40)
    logistic = LogisticRegressionCV(
        penalty='elasticnet',
        l1_ratios= l1_vec,
        solver='saga',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        random_state=random_state,
        max_iter=5000,
        class_weight='balanced',
        Cs= C_vec,
        tol=1e-4,
        scoring='balanced_accuracy'
    )


    # Below is the original way we started
    logistic.fit(X, y)

    print("Original l1_ratios specified:", l1_vec)
    print("Model's l1_ratios attribute:", logistic.l1_ratios)
    print("Shape of logistic.scores_[1]:", logistic.scores_[1].shape)
    print("\nFirst 10 scores for each row:")
    for i in [0,1]:  # explicitly loop through both rows
        print(f"\nRow {i} (supposed to be l1_ratio = {l1_vec[i]}):")
        print(logistic.scores_[1][0, :10, i])


    scores_dt = np.mean(logistic.scores_[1],axis=0)
    print("Mean scores for varying l1_ratio and Cs")
    print(scores_dt)  # print matrix for this fold

    scores_df = pd.DataFrame(
    scores_dt,  # Transpose to get l1_ratios as rows
    index=C_vec,
    columns=l1_vec
    )

    # Label the axes
    scores_df.index.name = 'C'
    scores_df.columns.name = 'l1_ratio'
    # logistic.coef_ will have shape (1, n_features) for binary classification
    coef_df = pd.DataFrame(logistic.coef_, columns=X.columns)
    # No indexing by classes since it's binary (one row of coefficients)

    # Compute feature importance as absolute value of coefficients
    # Since there's only one class row, mean across rows is just that row
    feature_importance = np.abs(coef_df.iloc[0, :])

    # Select features with non-zero importance
    selected_features = feature_importance[feature_importance > 0.0].index.tolist()

    # Predictions
    y_pred = logistic.predict(X)

    # Performance metrics
    print("\nPerformance Metrics:")
    print("-------------------")

    # Find the best C and corresponding CV score
    best_c = logistic.C_[0]
    c_index = np.where(logistic.Cs_ == best_c)[0][0]
    all_class_scores = []
    for cl in logistic.scores_:
        all_class_scores.extend(logistic.scores_[cl][:, c_index])
    best_cv_score = np.mean(all_class_scores)

    print(f"Best C value: {best_c}")
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y, y_pred)
    print(conf_matrix)
    print("\nClassification Report:")

    target_names = [str(c) for c in label_encoder.classes_]
    print(classification_report(y, y_pred, target_names=target_names))
    print(f"Best CV score: {best_cv_score}")

    # l1_ratio_ returns the best ratio found for each class. For binary, there should be one:
    print(f"Best l1_ratio: {logistic.l1_ratio_[0]}")

    print(f"\nSelected {len(selected_features)} features")

    # Print feature importance for selected features
    print("\nFeature importance for selected features:")
    for feat in sorted(selected_features, key=lambda x: feature_importance[x], reverse=True):
        print(f"{feat}: {feature_importance[feat]:.4f}")

    X_selected = X[selected_features]
    print(f"Final shape of selected features: {X_selected.shape}")

    # Store metrics in a dictionary
    metrics = {
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(y, y_pred, target_names=label_encoder.classes_, output_dict=True),
        'accuracy': accuracy_score(y, y_pred),
        'cv_scores': all_class_scores
    }

    original_features = {}
    for feature in selected_features:
      if '*_*' in feature:
        orig_name = feature.split('*_*')[0]
        coef = abs(feature_importance[feature])
        original_features[orig_name] = max(original_features.get(orig_name, 0), coef)
      else:
        original_features[feature] = abs(feature_importance[feature])

    # Create grouped results showing all categories
    grp_results= create_grouped_results(selected_features, feature_importance)
    results_df = grp_results[0]
    sorted_fields = grp_results[1]
    categorical_fields = grp_results[2]

    # Print the grouped results
    print("\nSelected features grouped by main field:")
    print(results_df)

    # Save to CSV
    results_df.to_csv('feature_coefficients_grouped.csv', index=False)

    # Note: We may still want to keep the original X_selected DataFrame for further analysis
    X_selected = X[selected_features]
    #print(f"Final shape of selected features: {X_selected.shape}")

    # After running create_grouped_results, extract main fields
    main_fields = []
    for field in sorted_fields:  # We already have sorted_fields from earlier
        if field in categorical_fields:
            main_fields.append(field)
        else:
            main_fields.append(field)  # For regular features

    # Convert main_fields list to a single-column DataFrame
    main_fields_df = pd.DataFrame({'Main Features': main_fields})
    print(main_fields_df)

    top_params = get_parameter_ranking(logistic, n_top=20)
    print("\nTop parameter combinations:")
    print(top_params)

    return results_df, scores_df, main_fields_df, top_params, X_selected, y, selected_features, coef_df, label_encoder, feature_importance, metrics

    # Group features by main field and sort by importance
def create_grouped_results(selected_features, feature_importance):
    """
    Create a DataFrame with all categories listed under their main fields,
    with main fields sorted by their maximum coefficient magnitude.
    """
    # Step 1: Identify one-hot encoded and regular features
    categorical_fields = set()
    regular_features = []

    for feature in selected_features:
        if '*_*' in feature:
            # One-hot encoded feature
            categorical_fields.add(feature.split('*_*')[0])
        else:
            # Regular feature
            regular_features.append(feature)

    # Step 2: Organize features by main field
    field_groups = {}

    # Process categorical fields
    for field in categorical_fields:
        field_groups[field] = []
        # Find all categories for this field
        for feature in selected_features:
            if feature.startswith(field + '*_*'):
                category = feature.split('*_*')[1]
                coef = feature_importance[feature]
                field_groups[field].append({
                    'Feature': category,
                    'Coefficient': coef,
                    'AbsCoef': abs(coef)
                })

    # Process regular features
    for feature in regular_features:
        coef = feature_importance[feature]
        # For regular features, store coefficient directly
        field_groups[feature] = [{
            'Feature': '',
            'Coefficient': coef,
            'AbsCoef': abs(coef)
        }]

    # Get max coefficient for each field for sorting
    field_max_coef = {}
    for field, categories in field_groups.items():
        field_max_coef[field] = max(cat['AbsCoef'] for cat in categories)

    #  Sort fields by their max coefficient
    sorted_fields = sorted(field_groups.keys(), key=lambda f: field_max_coef[f], reverse=True)

    # Create the final DataFrame with the desired structure
    results = []

    for field in sorted_fields:
        if field in categorical_fields:
            # For categorical fields, add header row with no coefficient
            results.append({
                'Feature': field,
                'Coefficient': "..."
            })

            # Add all categories
            categories = sorted(field_groups[field], key=lambda x: x['AbsCoef'], reverse=True)
            for cat in categories:
                results.append({
                    'Feature': f"  {cat['Feature']}",  # Indent for visual grouping
                    'Coefficient': cat['Coefficient']
                })
        else:
            # For regular features, show coefficient directly
            results.append({
                'Feature': field,
                'Coefficient': field_groups[field][0]['Coefficient']
            })

    return pd.DataFrame(results), sorted_fields, categorical_fields

def get_parameter_ranking(logistic, n_top=10):
    '''
    Create a ranking of parameter combinations using stored scores
    and coefficient paths.
    '''
    # Create empty list to store parameter information
    param_scores = []

    # Get model parameters
    l1_ratios = logistic.l1_ratios_
    Cs = logistic.Cs_

    # Get first class key (for binary classification, there's only one set of scores)
    first_class = list(logistic.scores_.keys())[0]

    # Loop through all parameter combinations
    for l1_ratio_idx, l1_ratio in enumerate(l1_ratios):
        for C_idx, C in enumerate(Cs):
            # Get mean score across folds for this parameter combination
            score = np.mean(logistic.scores_[first_class][:, C_idx, l1_ratio_idx])

            # Get coefficients for this parameter combination
            coef_path = logistic.coefs_paths_[first_class][:, C_idx, l1_ratio_idx, :]

            # Average across folds
            mean_coef = np.mean(coef_path, axis=0)

            # Count non-zero coefficients
            n_features = np.sum(np.abs(mean_coef) > 1e-3)

            # Append to our list
            param_scores.append({
                'l1_ratio': l1_ratio,
                'C': C,
                'score': score,
                'n_features': n_features
            })

    # Convert to DataFrame and sort
    params_df = pd.DataFrame(param_scores)
    params_df = params_df.sort_values('score', ascending=False).head(n_top)

    return params_df


## KAPLAN MEIER ##
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def execute_kaplan_meier(df, duration_col, event_col, group_col):
    # Remove rows with missing values in relevant columns
    df = df.dropna(subset=[duration_col, event_col, group_col])
    kmf = KaplanMeierFitter()

    unique_groups = df[group_col].unique()
    survival_curves = {}
    confidence_intervals = {}
    times = np.arange(0, 61, 10)

    # Compute survival curves and confidence intervals for each group
    for group in unique_groups:
        group_data = df[df[group_col] == group]
        kmf.fit(group_data[duration_col], event_observed=group_data[event_col], label=str(group))
        survival_curves[group] = kmf.survival_function_ * 100
        ci_lower = kmf.confidence_interval_[f'{group}_lower_0.95'] * 100
        ci_upper = kmf.confidence_interval_[f'{group}_upper_0.95'] * 100
        confidence_intervals[group] = (ci_lower, ci_upper)

    # Perform log-rank test
    if len(unique_groups) == 2:
        group1_data = df[df[group_col] == unique_groups[0]]
        group2_data = df[df[group_col] == unique_groups[1]]
        result = logrank_test(group1_data[duration_col], group2_data[duration_col],
                              event_observed_A=group1_data[event_col],
                              event_observed_B=group2_data[event_col])
        p_value = result.p_value
    elif len(unique_groups) > 2:
        result = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col])
        p_value = result.p_value
    else:
        p_value = np.nan

    # Generate risk table: number of individuals at risk over time
    risk_counts = {
        group: [(df[(df[group_col] == group) & (df[duration_col] >= t)]).shape[0] for t in times]
        for group in unique_groups
    }

    risk_table = pd.DataFrame(risk_counts, index=times).T
    risk_table.insert(0, "Group", risk_table.index)

    return {
        "survival_curves": survival_curves,
        "confidence_intervals": confidence_intervals,
        "risk_table": risk_table,
        "p_value": p_value,
        "times": times
    }
