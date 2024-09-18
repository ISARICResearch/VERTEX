import numpy as np
import pandas as pd
# import re
# import os
import scipy.stats as stats
import researchpy as rp
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
import xgboost as xgb


def risk_preprocessing(data):
    df_map = data
    comor = []
    for i in df_map:
        if 'comor_' in i:
            comor.append(i)
    sympt = []
    for i in df_map:
        if 'adsym_' in i:
            sympt.append(i)

    sdata = df_map[sympt+comor+['age', 'slider_sex', 'outcome']].copy()

    sdata = sdata.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    sdata[sympt+comor] = (sdata[sympt+comor] != 'no')
    sdata = sdata.loc[(sdata['outcome'] != 'censored')]

    outcome_binary_map = {'discharge': 0, 'death': 1}
    sex_binary_map = {'female': 0, 'male': 1}
    sdata['outcome'] = sdata['outcome'].map(outcome_binary_map)
    sdata['slider_sex'] = sdata['slider_sex'].map(sex_binary_map)
    return sdata


def obtain_variables(data, data_type):
    prefix = ''
    if data_type == 'symptoms':
        prefix = 'adsym_'
    elif data_type == 'comorbidities':
        prefix = 'comor_'
    variables = []

    for i in data:
        if prefix in i:
            variables.append(i)

    df = data[[
        'usubjid', 'age', 'slider_sex', 'slider_country',
        'outcome', 'country_iso'] + variables].copy()
    if (data_type == 'symptoms') or (data_type == 'comorbidities'):
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    return df


def get_proportions(data, data_type):
    prefix = ''
    if data_type == 'symptoms':
        prefix = 'adsym_'
    elif data_type == 'comorbidities':
        prefix = 'comor_'
    elif data_type == 'treatments':
        prefix = 'treat_'

    variables = []

    for i in data:
        if prefix in i:
            variables.append(i)

    df = data[[
        'usubjid', 'age', 'slider_sex', 'slider_country', 'outcome',
        'country_iso'] + variables].copy()
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    proportions = df[variables].apply(
        lambda x: x.dropna().sum() / x.dropna().count()).reset_index()

    proportions.columns = ['Condition', 'Proportion']
    proportions = proportions.sort_values(by=['Proportion'], ascending=False)
    Condition_top = proportions['Condition'].head(5)
    set_data = df[Condition_top]
    return proportions, set_data


def mapOutcomes(df):
    mapping_dict = {
        'Discharged alive': 'Discharge',
        'Transfer to other facility': 'Censored',
        'Discharged against medical advice': 'Discharge',
        'Death': 'Death',
        'Still hospitalised': 'Censored',
        'Palliative discharge': 'Censored'
    }
    df['outco_outcome'] = df['outco_outcome'].map(mapping_dict)
    return df


def remove_MissingDataCodes(df):
    MissingDataCodes = [
        'Unknown', 'No information', 'Not asked', 'Not applicable']
    with pd.option_context('future.no_silent_downcasting', True):
        df.replace('', np.nan, inplace=True)
        df.replace(MissingDataCodes, np.nan, inplace=True)
    return df


def harmonizeAge(df):
    # df['demog_age'] = df['demog_age'].astype(float)
    df['demog_age'] = pd.to_numeric(df['demog_age'], errors='coerce')
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


def get_variables_type(data):
    final_binary_variables = []
    final_numeric_variables = []
    final_categorical_variables = []

    for column in data:
        column_data = data[column].dropna()

        if column_data.empty:
            continue

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(column_data):
            unique_values = column_data.unique()
            if (len(unique_values) == 2) and (set(unique_values) == {0, 1}):
                final_binary_variables.append(column)
            else:
                try:
                    pd.to_numeric(column_data)
                    final_numeric_variables.append(column)
                except ValueError as e:
                    print(f'An error occurred: {e}')
                    for col_value_i in column_data:
                        print(col_value_i)
        else:
            unique_values = column_data.unique()
            # Consider column as categorical if it has a few unique values
            if len(unique_values) <= 10:
                final_categorical_variables.append(column)

    return final_binary_variables, final_numeric_variables, final_categorical_variables


def categorical_feature(data, categoricals):
    categorical_results_t = []
    for variable in categoricals:
        data_variable = data[[variable]].dropna()
        category_variable = 1
        data_aux_cat = data_variable.loc[(data_variable[variable] == 1)]
        try:
            n = len(data_aux_cat)
            pe = round(100 * (n / len(data_variable)), 1)
            categorical_results_t.append([
                str(variable) + ': ' + str(category_variable),
                str(n) + ' (' + str(pe) + ')'])
        except Exception:
            print(variable)
    categorical_results_t = pd.DataFrame(
        data=categorical_results_t, columns=['Variable', 'Count'])
    return categorical_results_t


def categorical_feature_outcome(data, outcome):
    binary_variables, numeric_variables, categorical_variables = get_variables_type(data)
    try:
        binary_variables.remove(outcome)
    except Exception:
        print('Outcome not in dataframe')
    suitable_cat = []

    categorical_results = []
    categorical_results_t = []
    for variable in binary_variables:
        data_variable = data[[variable, outcome]].dropna()
        x = data_variable[variable]
        y = data_variable[outcome]
        data_crosstab = pd.crosstab(x, y, margins=False)
        stat, p, dof, expected = stats.chi2_contingency(data_crosstab)

        if p < 0.2:
            suitable_cat.append(variable)
        if p < 0.001:
            p = '<0.001'
        elif p <= 0.05:
            p = str(round(p, 3))
        else:
            p = str(round(p, 2))

        data_variable0 = data_variable.loc[(data_variable[outcome] == 0)]
        data_variable1 = data_variable.loc[(data_variable[outcome] == 1)]
        for category_variable in [1]:
            data_aux_cat = data_variable.loc[(
                data_variable[variable] == category_variable)]
            n = len(data_aux_cat)
            count = data_aux_cat[outcome].value_counts().reset_index()
            n0 = count.loc[(count[outcome] == 0), 'count']
            n1 = count.loc[(count[outcome] == 1), 'count']
            p0 = round(100*(n0 / len(data_variable0)), 1)
            p1 = round(100*(n1 / len(data_variable1)), 1)
            pe = round(100*(n / len(data_variable)), 1)
            if len(n0) == 0:
                n0, p0 = 0, 0
            else:
                n0 = n0.iloc[0]
                p0 = p0.iloc[0]
            if len(n1) == 0:
                n1, p1 = 0, 0
            else:
                n1 = n1.iloc[0]
                p1 = p1.iloc[0]

            categorical_results.append([
                str(variable),
                str(n1) + ' (' + str(p1) + ')',
                str(n0) + ' (' + str(p0) + ')',
                str(n) + ' (' + str(pe) + ')',
                p])
            categorical_results_t.append([
                str(variable) + ': ' + str(category_variable),
                str(n) + ' (' + str(pe) + ')'])

    column1 = 'Characteristic'
    column2 = outcome + '=1 (n=' + str(round(data[outcome].sum())) + ')'
    column3 = outcome + '=0 (n=' + str(round(len(data) - data[outcome].sum()))
    column3 += ')'
    column4 = 'All cohort (n=' + str(round(len(data))) + ')'

    categorical_results = pd.DataFrame(
        data=categorical_results,
        columns=[column1, column2, column3, column4, 'p-value'])

    categorical_results_t = pd.DataFrame(
        data=categorical_results_t, columns=['Variable', 'Count'])
    return categorical_results, suitable_cat, categorical_results_t


def numeric_outcome_results(data, outcome):
    binary_variables, numeric_variables, categorical_variables = get_variables_type(data)
    results_array = []
    results_t = []
    suitable_num = []
    for variable in numeric_variables:
        try:
            data[variable] = pd.to_numeric(data[variable], errors='coerce')
            data_variable = data[[variable, outcome]].dropna()
            data0 = data_variable.loc[(data_variable[outcome] == 0), variable]
            data1 = data_variable.loc[(data_variable[outcome] == 1), variable]
            data_t = data_variable[variable]
            # complete = round((100 * (len(data_variable) / len(data))), 1)
            if len(data_variable) > 2:
                # On the whole variable
                stat, p = stats.shapiro(data_variable[variable])
                alpha = 0.05
                if p < alpha:
                    # print('Not normal')
                    w, p = stats.mannwhitneyu(
                        data0, y=data1, alternative='two-sided')
                else:
                    summary, results = rp.ttest(
                        group1=data.loc[(data[outcome] == 0), variable],
                        group1_name='0',
                        group2=data.loc[(data[outcome] == 1), variable],
                        group2_name='1')
                    p = results['results'].loc[3]

                detail0 = str(round(data0.median(), 1)) + ' ('
                detail0 += str(round(data0.quantile(0.25), 1)) + '-'
                detail0 += str(round(data0.quantile(0.75), 1)) + ')'
                detail1 = str(round(data1.median(), 1)) + ' ('
                detail1 += str(round(data1.quantile(0.25), 1)) + '-'
                detail1 += str(round(data1.quantile(0.75), 1)) + ')'
                detail_t = str(round(data_t.median(), 1)) + ' ('
                detail_t += str(round(data_t.quantile(0.25), 1)) + '-'
                detail_t += str(round(data_t.quantile(0.75), 1)) + ')'
                if p < 0.2:
                    suitable_num.append(variable)
                if p < 0.001:
                    p = '<0.001'
                elif p <= 0.05:
                    p = str(round(p, 3))
                else:
                    p = str(round(p, 2))
            else:
                detail0 = str(round(data0.mean(), 1)) + ' ('
                detail0 += str(round(data0.std(), 1)) + ')'
                detail1 = str(round(data1.mean(), 1)) + ' ('
                detail1 += str(round(data1.std(), 1)) + ')'
                detail_t = str(round(data_t.mean(), 1)) + ' ('
                detail_t += str(round(data_t.std(), 1)) + ')'
                p = 'N/A'

            results_array.append([variable, detail1, detail0, detail_t, p])
            results_t.append([variable, detail_t])
        except Exception:
            print(variable)

    column1 = 'Characteristic'
    column2 = outcome + '=1 (n=' + str(round(data[outcome].sum())) + ')'
    column3 = outcome + '=0 (n=' + str(round(len(data) - data[outcome].sum()))
    column3 += ')'
    column4 = 'All cohort (n=' + str(round(len(data))) + ')'

    results_df = pd.DataFrame(
        data=results_array,
        columns=[column1, column2, column3, column4, 'p-value'])
    results_t = pd.DataFrame(
        data=results_t,
        columns=['Variable', 'median(IQR)'])
    return results_df, suitable_num, results_t


def numeric_results(data, numeric_variables):
    results_t = []
    for variable in numeric_variables:
        try:
            data[variable] = pd.to_numeric(
                data[variable], errors='coerce')
            data_variable = data[[variable]].dropna()
            data_t = data_variable[variable]
            # complete = round((100 * (len(data_variable) / len(data))), 1)
            if len(data_variable) > 2:
                detail_t = str(round(data_t.median(), 1)) + ' ('
                detail_t += str(round(data_t.quantile(0.25), 1)) + '-'
                detail_t += str(round(data_t.quantile(0.75), 1)) + ')'
            else:
                detail_t = str(round(data_t.mean(), 1)) + ' ('
                detail_t += str(round(data_t.std(), 1)) + ')'
            results_t.append([variable, detail_t])
        except Exception:
            print(variable)
    results_t = pd.DataFrame(
        data=results_t, columns=['Variable', 'median(IQR)'])
    return results_t


def descriptive_table(data, correct_names, categoricals, numericals):
    categorical_results_t = categorical_feature(
        data, list(set(categoricals).intersection(set(data.columns))))
    numeric_results_t = numeric_results(
        data, list(set(numericals).intersection(set(data.columns))))

    table = pd.merge(
        categorical_results_t, numeric_results_t, on='Variable', how='outer')

    table = table.fillna('')
    return table


def binary_model1(data, variables, outcome, num_estimators=10):
    data_path = data.dropna(subset=[outcome])
    combined_df = data_path.dropna(subset=[outcome])

    # X_Transm = combined_df[variables]
    X = combined_df[variables]

    y = combined_df[outcome]
    le = LabelEncoder()
    y = list(le.fit_transform(y))

    # Initialize XGBoost model for classification
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax', num_class=len(set(y)),
        random_state=182, use_label_encoder=False, eval_metric='mlogloss',
        enable_categorical=True, max_depth=4, n_estimators=num_estimators)
    for X_x in X:
        X[X_x] = X[X_x].astype('category')
    # Train the model
    xgb_model.fit(X, y)
    # Make predictions
    predictions = xgb_model.predict(X)
    probabilities = xgb_model.predict_proba(X)
    combined_df['Predictions'] = predictions
    if (len(set(y)) == 2):
        probabilities = pd.DataFrame(data=probabilities)
        combined_df['probabilities'] = probabilities[1]

    # Evaluate the model using a classification metric
    accuracy = accuracy_score(y, predictions)
    roc = roc_auc_score(y, combined_df['probabilities'])
    fpr, tpr, thresholds = roc_curve(y, combined_df['probabilities'])

    # Calculate the Youden's index
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Feature importances
    importances = xgb_model.feature_importances_
    feature_names = X.columns
    feature_importances = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances})

    # Sort the features by importance
    feature_importances = feature_importances.sort_values(
        by='Importance', ascending=False)
    return feature_importances, accuracy, roc, optimal_threshold, combined_df


def remove_columns(data, limit_var=60):
    nan_percentage = (data.isna().sum() / len(data))*100
    nan_percentage = nan_percentage.reset_index()
    variables_included = nan_percentage.loc[(
        nan_percentage[0] <= limit_var), 'index']
    return data[variables_included]


def num_imputation_nn(df, n_neighbor=5):
    # Separating numerical and encoded nominal variables
    numerical_data = df.select_dtypes(include=[np.number])
    # Initializing the KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbor)
    # Imputing missing values
    imputed_data = imputer.fit_transform(numerical_data)
    # Converting imputed data back to a DataFrame
    return pd.DataFrame(imputed_data, columns=numerical_data.columns)


def lasso_rf(data, outcome_var='Outcome'):
    Y = data[outcome_var]
    X = data.drop(outcome_var, axis=1)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset into cross-validation set and hold-out set
    X_cv, X_holdout, Y_cv, Y_holdout, idx_cv, idx_holdout = train_test_split(
        X_scaled, Y, range(len(data)),
        test_size=0.2, random_state=666, stratify=Y)

    # Logistic Regression with L1 regularization
    log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear')

    # Hyperparameter tuning using GridSearchCV
    parameters = {'C': [0.0001, 0.001, 0.01]}
    log_reg_cv = GridSearchCV(log_reg_l1, parameters, cv=10, scoring='roc_auc')
    log_reg_cv.fit(X_cv, Y_cv)

    # Best hyperparameter value
    best_C = log_reg_cv.best_params_['C']

    # Evaluate using the best parameter on the hold-out set
    log_reg_best = LogisticRegression(
        penalty='l1', C=best_C, solver='liblinear')
    log_reg_best.fit(X_cv, Y_cv)

    # Predicting probabilities
    Y_pred_proba = log_reg_best.predict_proba(X_holdout)[:, 1]

    # Calculating ROC AUC
    roc_auc = roc_auc_score(Y_holdout, Y_pred_proba)

    # Print coefficients
    feature_names = X.columns
    coefficients = log_reg_best.coef_[0]
    non_zero_indices = np.where(coefficients != 0)[0]

    # Standard errors, CIs, and p-values
    # intercept = log_reg_best.intercept_
    log_reg_best.fit(X_cv[:, non_zero_indices], Y_cv)
    standard_errors = np.sqrt(np.diag(np.linalg.inv(np.dot(
        X_cv[:, non_zero_indices].T, X_cv[:, non_zero_indices]))))
    z_scores = coefficients[non_zero_indices] / standard_errors
    p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores]

    # Calculate odds ratios and confidence intervals
    odds_ratios = np.exp(coefficients[non_zero_indices])
    conf_intervals = np.exp(
        coefficients[non_zero_indices][:, np.newaxis] +
        np.array([-1, 1]) * 1.96 * standard_errors[:, np.newaxis])

    # Format the coefficients, OR, CI, and p-values
    formatted_coefficients = [
        f'{coef:.3f}' for coef in coefficients[non_zero_indices]]
    formatted_odds_ratios = [
        f'{or_val:.3f}' for or_val in odds_ratios]
    formatted_conf_intervals = [
        (f'{ci[0]:.3f}', f'{ci[1]:.3f}') for ci in conf_intervals]
    formatted_p_values = [
        '<0.005' if pv < 0.005 else f'{pv:.3f}' for pv in p_values]

    coef_df = pd.DataFrame({
        'Feature': feature_names[non_zero_indices],
        'Coefficient': formatted_coefficients,
        'Odds Ratio': formatted_odds_ratios,
        'CI Lower 95%': [ci[0] for ci in formatted_conf_intervals],
        'CI Upper 95%': [ci[1] for ci in formatted_conf_intervals],
        'P-value': formatted_p_values
    })

    return coef_df, roc_auc, best_C
