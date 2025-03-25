import numpy as np
import pandas as pd
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
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
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
                while (parent in dictionary['parent'].values):
                    parent = dictionary.loc[new_ind, 'field_name']
                    new_ind = dictionary.loc[(
                        dictionary['parent'] == parent), 'index'].max()
                new_ind = new_ind + 0.1
        else:
            new_ind = new_dictionary.loc[(
                new_dictionary['field_name'] == parent), 'index'].max() + 0.1
            # new_ind = new_ind + 0.1
        new_dictionary.loc[ind, 'index'] = new_ind
    categorical_ind = new_dictionary['field_type'].isin(['categorical'])
    new_dictionary_list = [new_dictionary]
    ind_list = [
        ind for ind in categorical_ind.index
        if new_dictionary.loc[ind, 'field_name'] in data.columns]
    for ind in ind_list:
        variable = new_dictionary.loc[ind, 'field_name']
        options = data[variable].drop_duplicates().sort_values().dropna()
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

# def get_variable_list(dictionary, sections):
#     '''Get all variables in the dictionary belonging to sections
#     (assumes ARC format)'''
#     section_ids = dictionary['field_name'].apply(lambda x: x.split('_')[0])
#     variable_list = dictionary['field_name'].loc[section_ids.isin(sections)]
#     variable_list = list(variable_list)
#     return variable_list


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
    include_ind &= (dictionary['field_name'].apply(
        lambda x: x.endswith(tuple('___' + x for x in exclude_suffix))) == 0)
    if isinstance(required_variables, list):
        include_ind |= dictionary['field_name'].isin(required_variables)
    if include_subjid:
        include_ind |= (dictionary['field_name'] == 'subjid')
    include_variables = dictionary.loc[include_ind, 'field_name'].tolist()
    include_variables = [col for col in include_variables if col in df.columns]
    return include_variables


# def get_variables_from_sections(
#         variable_list, section_list,
#         required_variables=None, exclude_suffix=None):
#     '''
#     Get only the variables from sections, plus any required variables
#     '''
#     include_variables = []
#     for section in section_list:
#         include_variables += [
#             var for var in variable_list if var.startswith(section + '_')]
#
#     if required_variables is not None:
#         required_variables = [
#             var for var in required_variables if var not in include_variables]
#         include_variables = required_variables + include_variables
#
#     if exclude_suffix is not None:
#         include_variables = [
#             var for var in include_variables
#             if (var.endswith(tuple(exclude_suffix)) == 0)]
#     return include_variables


def convert_categorical_to_onehot(
        df, dictionary, categorical_columns, sep='___', missing_val='nan'):
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

    # include_columns = dictionary.loc[(
    #     dictionary['field_type'].isin(include_types)), 'field_name'].tolist()
    # include_columns = [col for col in include_columns if col in df.columns]
    include_columns = get_variables_by_section_and_type(
        df, dictionary,
        include_types=include_types, include_subjid=include_subjid,
        include_sections=include_sections, exclude_suffix=exclude_suffix)
    if (by_column is not None) & (by_column not in include_columns):
        include_columns = [by_column] + include_columns
    # if include_subjid:
    #     include_columns = ['subjid'] + include_columns
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


def format_descriptive_table_variables(dictionary, max_len=100):
    name = dictionary['field_name'].apply(
        lambda x: '   ↳ ' if '___' in x else '<b>')
    name += dictionary['field_type'].map({'section': '<i>'}).fillna('')
    name += dictionary['field_label'].apply(
        lambda x: x.split(':')[-1] if x.startswith('If') else x).apply(
        trim_field_label, max_len=max_len)
    name += dictionary['field_type'].map({'section': '</i>'}).fillna('')
    name += dictionary['field_name'].apply(
        lambda x: '' if '___' in x else '</b>')
    field_type = dictionary['field_type'].map({
        'categorical': ' (*)', 'binary': ' (*)', 'numeric': ' (+)'}).fillna('')
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


def get_proportions(df, dictionary, max_n_variables=10):
    proportions = df.apply(lambda x: x.sum() / x.count()).reset_index()

    proportions.columns = ['variable', 'proportion']
    proportions = proportions.sort_values(
        by=['proportion'], ascending=False).reset_index(drop=True)
    if proportions.shape[0] > max_n_variables:
        proportions = proportions.head(max_n_variables)

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
    # df = df.rename(columns=format_dict).copy()
    # if variables is not None:
    #     variables = [format_dict[var] for var in variables]
    # if proportions is not None:
    #     variables = proportions.sort_values(
    #         by='proportion', ascending=False)['variable'].head(n_variables)
    #     variables = variables.tolist()

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
    # if variables is None:
    #     variable_order_dict = dict(zip(counts['index'], counts.index))
    #     variables = counts['index'].tolist()
    # else:
    #     variable_order_dict = dict(zip(variables, range(len(variables))))
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
        embedding_model=embedding_model,  # how we embed the strings, default to sentence transformers
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
    summary['p_adj'] = summary['p'].apply(
        lambda p: "<0.001" if p < 0.001 else round(p, 3))

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
