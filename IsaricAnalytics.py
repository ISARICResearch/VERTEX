import numpy as np
import pandas as pd
from typing import List, Union
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic._utils import select_topic_representation
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
# from sklearn.impute import KNNImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.impute import KNNImputer
# import xgboost as xgb
# import itertools
# from collections import OrderedDict


############################################
############################################
# General preprocessing
############################################
############################################


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
        include_totals=True, column_reorder=None):
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
    table = pd.DataFrame('', index=index, columns=table_columns)

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
        df, dictionary, proportions=None, variables=None, n_variables=5):
    # Convert variables and column names into their formatted names
    long_format = format_variables(dictionary, max_len=1000)
    short_format = format_variables(dictionary, max_len=40)
    format_dict = dict(zip(dictionary['field_name'], long_format))
    short_format_dict = dict(zip(dictionary['field_name'], short_format))
    # df = df.rename(columns=format_dict).copy()
    # if variables is not None:
    #     variables = [format_dict[var] for var in variables]
    if proportions is not None:
        variables = proportions.sort_values(
            by='proportion', ascending=False)['variable'].head(n_variables)
        variables = variables.tolist()

    variables = [var for var in variables if df[var].sum() > 0]
    df = df[variables].astype(float).fillna(0)

    counts = df.sum().astype(int).reset_index().rename(columns={0: 'count'})
    counts = counts.sort_values(
        by='count', ascending=False).reset_index(drop=True)
    counts['short_label'] = counts['index'].map(short_format_dict)
    counts['label'] = counts['index'].map(format_dict)
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


############################################
############################################
# Logistic Regression from Risk Factors
############################################
############################################


def execute_logistic_regression(
        elr_dataframe_df, elr_outcome_str, elr_predictors_list,
        print_results=True, labels=False, reg_type="multi"):
    """
    Performs a logistic regression and returns a table with the coefficients
    and effects of the predictor variables.

    Parameters:
    - elr_dataframe_df (pd.DataFrame): DataFrame containing the data.
    - elr_outcome_str (str): Name of the outcome variable.
    - elr_predictors_list (list):
        List of strings with the names of the predictor variables.
    - print_results (bool, optional):
        Flag to print the regression results. Default is True.
    - labels (dict, optional):
        Dictionary mapping variable names to readable labels.
        Can accept an empty dictionary.
    - reg_type (str, optional):
        Type of regression ('multi' for multivariate, 'uni' for univariate).
        Default is "multi".
    Returns:
    - elr_summary_df (pd.DataFrame):
        DataFrame with the results of the logistic regression.
    """

    # Prepare the formula for the model
    elr_formula_str = elr_outcome_str + ' ~ ' + ' + '.join(elr_predictors_list)

    # Identify categorical variables that are also predictors
    elr_categorical_vars_list = elr_dataframe_df.select_dtypes(
        include=['object', 'category'])
    elr_categorical_vars_list = elr_categorical_vars_list.columns.intersection(
        elr_predictors_list)

    # Convert categorical variables to the 'category' data type
    for elr_var_str in elr_categorical_vars_list:
        elr_dataframe_df[elr_var_str] = (
            elr_dataframe_df[elr_var_str].astype('category'))

    # Fit the logistic regression model
    elr_model_obj = smf.glm(
        formula=elr_formula_str,
        data=elr_dataframe_df, family=sm.families.Binomial())
    elr_result_obj = elr_model_obj.fit()

    # Extract the summary table from the regression results
    elr_summary_table_df = elr_result_obj.summary2().tables[1]

    # Calculate Odds Ratios and confidence intervals
    elr_summary_table_df['Odds Ratio'] = np.exp(elr_summary_table_df['Coef.'])
    elr_summary_table_df['IC Low'] = np.exp(elr_summary_table_df['[0.025'])
    elr_summary_table_df['IC High'] = np.exp(elr_summary_table_df['0.975]'])

    # Select relevant columns and rename them as needed
    elr_summary_df = elr_summary_table_df[[
        'Odds Ratio', 'IC Low', 'IC High', 'P>|z|']]
    elr_summary_df = elr_summary_df.rename(columns={'P>|z|': 'p-value'})
    elr_summary_df = elr_summary_df.reset_index()
    elr_summary_df.rename(
        columns={
            'index': 'Study',
            'Odds Ratio': 'OddsRatio',
            'IC Low': 'LowerCI',
            'IC High': 'UpperCI'
        }, inplace=True
    )

    # Map variable names to readable labels
    if labels:
        def elr_parse_variable_name(var_name):
            if var_name == 'Intercept':
                return labels.get('Intercept', 'Intercept')
            elif '[' in var_name:
                base_var = var_name.split('[')[0]
                level = var_name.split('[')[1].split(']')[0]
                base_var_name = base_var.replace(
                    'C(', '').replace(')', '').strip()
                label = labels.get(base_var_name, base_var_name)
                return f'{label} ({level})'
            else:
                var_name_clean = var_name.replace(
                    'C(', '').replace(')', '').strip()
                return labels.get(var_name_clean, var_name_clean)

        elr_summary_df['Study'] = elr_summary_df['Study'].apply(
            elr_parse_variable_name)

    # Reorder the columns
    elr_summary_df = elr_summary_df[[
        'Study', 'OddsRatio', 'LowerCI', 'UpperCI', 'p-value']]

    # Format numerical values
    elr_summary_df['OddsRatio'] = elr_summary_df['OddsRatio'].round(2)
    elr_summary_df['LowerCI'] = elr_summary_df['LowerCI'].round(2)
    elr_summary_df['UpperCI'] = elr_summary_df['UpperCI'].round(2)
    elr_summary_df['p-value'] = elr_summary_df['p-value'].apply(
        lambda x: f'{x:.4f}')

    # Remove the letter 'T.' from categorical variables
    elr_summary_df['Study'] = elr_summary_df['Study'].str.replace('T.', '')

    # Remove intercept from the results
    elr_summary_df = elr_summary_df[elr_summary_df['Study'] != 'Intercept']

    # Rename columns based on regression type
    if reg_type == 'uni':
        elr_summary_df.rename(columns={
            'OddsRatio': 'OddsRatio (uni)',
            'LowerCI': 'LowerCI (uni)',
            'UpperCI': 'UpperCI (uni)',
            'p-value': 'p-value (uni)'
        }, inplace=True)
    else:
        elr_summary_df.rename(columns={
            'OddsRatio': 'OddsRatio (multi)',
            'LowerCI': 'LowerCI (multi)',
            'UpperCI': 'UpperCI (multi)',
            'p-value': 'p-value (multi)'
        }, inplace=True)

    # Print results if the flag is set
    if print_results:
        print(elr_summary_df)

    return elr_summary_df


############################################
############################################
# Modelling
############################################
############################################

############################################
############################################
# Graveyard
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
#
#
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
