import requests
import json
import pandas as pd
import os
import IsaricAnalytics as ia

def read_data_from_REDCAP():
    # Assuming your JSON content is in a file named 'data.json'
  
    file_path = "assets/sites.json"
    
  
    contries_path="assets/countries.csv"
    countries=pd.read_csv(contries_path,encoding='latin-1')
    dfs=[]
    for ele_var in os.environ:
        if ele_var.startswith("study_site"):
          dfs.append(pd.DataFrame([eval(os.environ[ele_var])], columns=['key', 'country_iso', 'site_id']))
    # Concatenate the DataFrames
    #sites = pd.concat(dfs, ignore_index=True)
    sites = pd.DataFrame(data=[['7FA9ACD7DAB3B9BF51AE9CE797135EFD','COL','1'],['25689E7C5B69F326B03B864B6FF97729','GBR','2']] ,columns=['key', 'country_iso', 'site_id'])
    complete_data=pd.DataFrame()
    for index, row in sites.iterrows():
        try:
            conex = {
                'token': row['key'],
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
            r = requests.post('https://ncov.medsci.ox.ac.uk/api/',data=conex)
            print('HTTP Status: ' + str(r.status_code))
            data=(r.json())
            form1=[]
            form2=[]
            for i in data:
                if i['redcap_event_name']=='Initial Assessment / Admission':
                    form1.append(i)
                elif i['redcap_event_name']=='Outcome / End of study':
                    form2.append(i)
            df = pd.concat([pd.DataFrame(form1),pd.DataFrame(form2)]).drop(columns='redcap_event_name').groupby('subjid').max().reset_index()
            df=ia.mapOutcomes(df)
            df=ia.harmonizeAge(df)
            
            country_name=countries['Country'].loc[countries['Code']==row['country_iso']].iloc[0]
            country_income=countries['Income group'].loc[countries['Code']==row['country_iso']].iloc[0]
            country_region=countries['Region'].loc[countries['Code']==row['country_iso']].iloc[0]
            df['slider_country']=country_name
            df['country_iso']=row['country_iso']
            df['income']=country_income
            df['region']=country_region
            df['epiweek.admit']=[1]*len(df)
            df['dur_ho']=[0]*len(df)
            
            df.rename(columns={'subjid':'usubjid',
                                'demog_age':'age',
                                'demog_sex':'slider_sex',
                                'outco_outcome':'outcome'}, inplace=True)
            
            complete_data=pd.concat([complete_data,df])
        
        except Exception as e:
            print(e)
    
    return complete_data

