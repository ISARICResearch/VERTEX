import pandas as pd
import scipy.stats as stats
import os
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import researchpy as rp


def risk_preprocessing(data):
    df_map=data
    comor=[]
    for i in df_map:
        if 'comor_' in i :
            comor.append(i)
    sympt=[]
    for i in df_map:
        if 'adsym_' in i :
            sympt.append(i)
    
    sdata=df_map[sympt+comor+['age','slider_sex','outcome']]  

    sdata = sdata.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    sdata2=sdata
    sdata2[sympt+comor]=sdata[sympt+comor]!='no'
    sdata2=sdata2.loc[sdata2['outcome']!='censored']

    outcome_binary_map={'discharge':0,  'death':1}
    sex_binary_map={'female':0,  'male':1}
    sdata2['outcome'] = sdata2['outcome'].map(outcome_binary_map)
    sdata2['slider_sex'] = sdata2['slider_sex'].map(sex_binary_map)
    return sdata2

def obtain_variables(data,data_type):
    if data_type == 'symptoms':
        prefix='adsym_'
    elif data_type == 'comorbidities':
        prefix='comor_'
    variables=[]

    for i in data:
        if prefix in i :
            variables.append(i)

    df=data[['usubjid','age','slider_sex','slider_country','outcome','country_iso']+variables].copy()
    if data_type == 'symptoms' or data_type == 'comorbidities':
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        df[variables]=df[variables]!='no'
    return df
    

def get_proportions(data,data_type):
    if data_type == 'symptoms':
        prefix='adsym_'
    elif data_type == 'comorbidities':
        prefix='comor_'
    variables=[]

    for i in data:
        if prefix in i :
            variables.append(i)
            
    df=data[['usubjid','age','slider_sex','slider_country','outcome','country_iso']+variables].copy()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df=df!='no'
    proportions = df[variables].apply(lambda x: x.sum() / x.count()).reset_index()
    proportions.columns=['Condition', 'Proportion']
    proportions=proportions.sort_values(by=['Proportion'], ascending=False)
    Condition_top=proportions['Condition'].head(5)
    set_data=df[Condition_top]
    return proportions,set_data

def mapOutcomes(df):
    mapping_dict = {
        "Discharged alive": "Discharge",
        "Transfer to other facility": "Censored",
        "Discharged against medical advice": "Discharge",
        "Death": "Death",
        "Still hospitalised": "Censored",
        "Palliative discharge": "Censored"
    }
    df['outco_outcome'] = df['outco_outcome'].map(mapping_dict)

    return df

def harmonizeAge(df):
    df['demog_age']=df['demog_age'].astype(float)
    df['demog_age'].loc[df['demog_age_units'] == 'Months'] = df['demog_age'] / 12
    df['demog_age'].loc[df['demog_age_units'] == 'Days'] = df['demog_age']/ 365
    df['demog_age_units'] = 'Years'  # Standardize the units to 'Years'
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
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                final_binary_variables.append(column)
            else:
                try:
                    pd.to_numeric(column_data)
                    final_numeric_variables.append(column)
                except ValueError as e:
                    print(f"An error occurred: {e}")
                    for col_value_i in column_data:
                        print(col_value_i)
        else:
            unique_values = column_data.unique()
            # Consider column as categorical if it has a few unique values
            if len(unique_values) <= 10:
                final_categorical_variables.append(column)

    return final_binary_variables, final_numeric_variables, final_categorical_variables

def categorical_feature(data):
    categoricals, numeric_variables, categorical_variables = get_variables_type(data)

    suitable_cat = []

    categorical_results = []
    categorical_results_t = []
    for variable in categoricals: 
        data_variable=data[[variable]].dropna()
        category_variable =1
        data_aux_cat = data_variable.loc[data_variable[variable] == 1]
        n = len(data_aux_cat)
        pe = round(100 * (n / len(data_variable)), 1)

        categorical_results_t.append([str(variable) + ': ' + str(category_variable),
                                      str(n) + ' (' + str(pe) + ')'])  
    categorical_results_t = pd.DataFrame(data=categorical_results_t, columns=['Variable', 'Count'])
    return categorical_results_t

def categorical_feature_outcome(data, outcome):
    categoricals, numeric_variables, categorical_variables = get_variables_type(data)
    try:
        categoricals.remove(outcome)
    except:
        print('Outcome not in dataframe')
    suitable_cat = []
    
    categorical_results = []
    categorical_results_t = []
    for variable in categoricals: 

        # variable = 'MaritalStatus'

        data_variable = data[[variable, outcome]].dropna()
        x = data_variable[variable]
        y = data_variable[outcome]
        data_crosstab = pd.crosstab(x, y, margins=False)
        stat, p, dof, expected = chi2_contingency(data_crosstab) 
        if p < 0.2:
            suitable_cat.append(variable)   

        if p < 0.001:
            p = '<0.001'
        elif p <= 0.05:
            p = str(round(p, 3))   
        else:
            p = str(round(p, 2))         

        data_variable0 = data_variable.loc[data_variable[outcome] == 0]
        data_variable1 = data_variable.loc[data_variable[outcome] == 1]
        for category_variable in [1]:
            data_aux_cat = data_variable.loc[data_variable[variable] == category_variable]
            n = len(data_aux_cat)
            count = data_aux_cat[outcome].value_counts().reset_index()
            n0 = count['count'].loc[count[outcome] == 0]
            n1 = count['count'].loc[count[outcome] == 1]
            p0 = round(100 * (n0 / len(data_variable0)), 1)
            p1 = round(100 * (n1 / len(data_variable1)), 1)
            pe = round(100 * (n / len(data_variable)), 1)
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

            categorical_results.append([str(variable),
                                        str(n1) + ' (' + str(p1) + ')',
                                        str(n0) + ' (' + str(p0) + ')',
                                        str(n) + ' (' + str(pe) + ')',
                                        p])
            categorical_results_t.append([str(variable) + ': ' + str(category_variable),
                                          str(n) + ' (' + str(pe) + ')'])  

    column1 = 'Characteristic'
    column2 = outcome + '=1 (n=' + str(round(data[outcome].sum())) + ')'
    column3 = outcome + '=0 (n=' + str(round(len(data) - data[outcome].sum())) + ')'
    column4 = 'All cohort (n=' + str(round(len(data))) + ')'

    categorical_results = pd.DataFrame(data=categorical_results, columns=[column1, column2, column3, column4, 'p-value'])

    categorical_results_t = pd.DataFrame(data=categorical_results_t, columns=['Variable', 'Count'])
    return categorical_results, suitable_cat, categorical_results_t



def numeric_outcome_results(data, outcome):    
        categoricals, numeric_variables, categorical_variables = get_variables_type(data)
        results = []
        results_t = []
        suitable_num = []
        for variable in numeric_variables: 
            try:
                data[variable] = pd.to_numeric(data[variable], errors='coerce')
                data_variable = data[[variable, outcome]].dropna()
                data0 = data_variable[variable][data_variable[outcome] == 0]
                data1 = data_variable[variable][data_variable[outcome] == 1]
                data_t = data_variable[variable]
                complete = round((100 * (len(data_variable) / len(data))), 1)
                if len(data_variable) > 2:
                    stat, p = stats.shapiro(data_variable[variable]) # On the whole variable

                    alpha = 0.05
                    if p < alpha:
                        # print('Not normal')
                        w, p = mannwhitneyu(data0, y=data1, alternative="two-sided")    
                    else:
                        summary, results = rp.ttest(group1=data[variable].loc[data[outcome] == 0], group1_name="0",
                                                    group2=data[variable].loc[data[outcome] == 1], group2_name="1")    
                        p = results['results'].loc[3]

                    detail0 = str(round(data0.median(), 1)) + ' (' + str(round(data0.quantile(0.25), 1)) + '-' + str(round(data0.quantile(0.75), 1)) + ')'
                    detail1 = str(round(data1.median(), 1)) + ' (' + str(round(data1.quantile(0.25), 1)) + '-' + str(round(data1.quantile(0.75), 1)) + ')'
                    detail_t = str(round(data_t.median(), 1)) + ' (' + str(round(data_t.quantile(0.25), 1)) + '-' + str(round(data_t.quantile(0.75), 1)) + ')'
                    if p < 0.2:
                        suitable_num.append(variable)
                    if p < 0.001:
                        p = '<0.001'
                    elif p <= 0.05:
                        p = str(round(p, 3))   
                    else:
                        p = str(round(p, 2))   

                else:
                    # results.append([variable, 'Cannot be calculated', 'Conforming Data: ' + str(complete) + '%']) 
                    detail0 = str(round(data0.mean(), 1)) + ' (' + str(round(data0.std(), 1)) + ')'
                    detail1 = str(round(data1.mean(), 1)) + ' (' + str(round(data1.std(), 1)) + ')'
                    detail_t = str(round(data_t.mean(), 1)) + ' (' + str(round(data_t.std(), 1)) + ')'
                    p = 'N/A'

                results.append([variable, detail1, detail0, detail_t, p])
                results_t.append([variable, detail_t])
            except:
                print(variable)

        column1 = 'Characteristic'
        column2 = outcome + '=1 (n=' + str(round(data[outcome].sum())) + ')'
        column3 = outcome + '=0 (n=' + str(round(len(data) - data[outcome].sum())) + ')'
        column4 = 'All cohort (n=' + str(round(len(data))) + ')'

        results = pd.DataFrame(data=results, columns=[column1, column2, column3, column4, 'p-value'])
        results_t = pd.DataFrame(data=results_t, columns=['Variable', 'median(IQR)'])
        return results, suitable_num, results_t  


def numeric_results(data):    
    categoricals, numeric_variables, categorical_variables = get_variables_type(data)
    results = []
    results_t = []
    suitable_num = []
    for variable in numeric_variables: 
        try:
            data[variable] = pd.to_numeric(data[variable], errors='coerce')
            data_variable = data[[variable]].dropna()
            data_t = data_variable[variable]
            complete = round((100 * (len(data_variable) / len(data))), 1)
            if len(data_variable) > 2:    
                detail_t = str(round(data_t.median(), 1)) + ' (' + str(round(data_t.quantile(0.25), 1)) + '-' + str(round(data_t.quantile(0.75), 1)) + ')'
            else:
                detail_t = str(round(data_t.mean(), 1)) + ' (' + str(round(data_t.std(), 1)) + ')'
                p = 'N/A'
            results_t.append([variable, detail_t])
        except:
            print(variable)
    results_t = pd.DataFrame(data=results_t, columns=['Variable', 'median(IQR)'])
    return results_t  



def descriptive_table(data):
    categorical_results_t=categorical_feature(data)
    numeric_results_t=numeric_results(data)
    table=pd.merge(categorical_results_t, numeric_results_t,on='Variable',how='outer')

    table=table.fillna(' ')
    return table