"""loader.py"""
import os
import json
import pandas as pd
import shutil

import vertex.getREDCapData as getRC
from vertex.layout.insight_panels import get_insight_panels, get_visuals

from vertex.logging.logger import setup_logger
logger = setup_logger(__name__)

config_defaults = {
    'project_name': None,
    "data_access_groups": None,
    'map_layout_center_latitude': 6,
    'map_layout_center_longitude': -75,
    'map_layout_zoom': 1.7,
    'save_public_outputs': False,
    'save_base_files_to_public_path': False,
    'public_path': 'PUBLIC/',
    'save_filtered_public_outputs': False,
    'insight_panels_path': 'insight_panels/',
    'insight_panels': [],
}

def get_config(project_path, config_defaults):
    config_file = os.path.join(project_path, 'config_file.json')
    try:
        with open(config_file, 'r') as json_data:
            config_dict = json.load(json_data)
        _ = config_dict['api_key']
        _ = config_dict['api_url']
    except Exception:
        error_message = f'''config_file.json is required in \
{project_path}. This file must contain both "api_key" and "api_url".'''
        print(error_message)
        raise SystemExit
    # The default for the list of insight panels is all that exist in the
    # relevant folder (which may or not be specified in config)
    if 'insight_panels_path' not in config_dict.keys():
        rel_insight_panels_path = config_defaults['insight_panels_path']
        config_dict['insight_panels_path'] = rel_insight_panels_path
    # Get a list of python files in the repository (excluding e.g. __init__.py)
    insight_panels_path = os.path.join(
        project_path, config_dict['insight_panels_path'])
    for (_, _, filenames) in os.walk(insight_panels_path):
        insight_panels = [
            file.split('.py')[0] for file in filenames
            if file.endswith('.py') and not file.startswith('_')]
        break
    config_defaults['insight_panels'] = insight_panels
    # Add default items where the config file doesn't include these
    config_defaults = {
        k: v
        for k, v in config_defaults.items() if k not in config_dict.keys()}
    config_dict = {**config_dict, **config_defaults}
    if any([x not in insight_panels for x in config_dict['insight_panels']]):
        print('The following insight panels in config_file.json do not exist:')
        missing_insight_panels = [
            x for x in config_dict['insight_panels']
            if x not in insight_panels]
        print('\n'.join(missing_insight_panels))
        print('These are ignored and will not appear in the dashboard.')
        config_dict['insight_panels'] = [
            x for x in config_dict['insight_panels'] if x in insight_panels]
    if any([x not in config_dict['insight_panels'] for x in insight_panels]):
        print('''The following insight panel files are not listed in \
                config_file.json:''')
        missing_insight_panels = [
            x for x in insight_panels
            if x not in config_dict['insight_panels']]
        print('\n'.join(missing_insight_panels))
        print('''These will not appear in the dashboard. Please add them \
                to the list "insight_panels" in config_file.json to include them in \
                the dashboard.''')
    return config_dict

def load_vertex_data(project_path, config_dict):
    api_url = config_dict['api_url']
    api_key = config_dict['api_key']
    get_data_from_api = (api_url is not None) and (api_key is not None)
    
    if get_data_from_api:
        df_map, df_forms_dict, dictionary, quality_report = load_vertex_from_api(api_url, api_key, config_dict)
    else:
        print(f'Loading data from {project_path}')
        df_map, df_forms_dict, dictionary, quality_report = load_vertex_from_files(project_path, config_dict)
    print("dfm keys", df_map.keys())
    return df_map, df_forms_dict, dictionary, quality_report

def load_vertex_from_api(api_url, api_key, config_dict):
    """Load data from the REDCap API."""
    print('Retrieving data from the API')
    user_assigned_to_dag = getRC.user_assigned_to_dag(api_url, api_key)
    get_data_kwargs = {
        'data_access_groups': config_dict['data_access_groups'],
        'user_assigned_to_dag': user_assigned_to_dag}
    df_map, df_forms_dict, dictionary, quality_report = (
        getRC.get_redcap_data(api_url, api_key, **get_data_kwargs))
    return df_map, df_forms_dict, dictionary, quality_report

def load_vertex_from_files(project_path, config_dict):

    try:
        vertex_dataframes_path = os.path.join(
            project_path, config_dict['vertex_dataframes_path'])
        vertex_dataframes = os.listdir(vertex_dataframes_path)
        dictionary = pd.read_csv(
            os.path.join(vertex_dataframes_path, 'vertex_dictionary.csv'),
            dtype={'field_label': 'str'},
            keep_default_na=False)
        str_ind = dictionary['field_type'].isin(
            ['freetext', 'categorical'])
        str_columns = dictionary.loc[str_ind, 'field_name'].tolist()
        non_str_columns = dictionary.loc[(
            str_ind == 0), 'field_name'].tolist()
        # num_ind = dictionary['field_type'].isin(['numeric'])
        # num_columns = dictionary.loc[num_ind, 'field_name'].tolist()
        dtype_dict = {
            **{x: 'str' for x in str_columns},
            # **{x: 'float' for x in num_columns}
        }
        # pandas tries to infer NaN values, sometimes this causes issues
        # solution is to ignore str columns, otherwise there are errors if
        # e.g. 'None' is an answer option
        pandas_default_na_values = [
            '',
            ' ',
            '#N/A',
            '#N/A N/A',
            '#NA',
            '-1.#IND',
            '-1.#QNAN',
            '-NaN',
            '-nan',
            '1.#IND',
            '1.#QNAN',
            '<NA>',
            'N/A',
            'NA',
            'NULL',
            'NaN',
            'None',
            'n/a',
            'nan',
            'null'
        ]
        na_values = {
            **{x: pandas_default_na_values for x in non_str_columns},
            **{x: '' for x in str_columns}
        }
        df_map = pd.read_csv(
            os.path.join(vertex_dataframes_path, 'df_map.csv'),
            dtype=dtype_dict,
            keep_default_na=False,
            na_values=na_values
        )
        # Fix dates
        date_variables = dictionary.loc[(
            dictionary['field_type'] == 'date'), 'field_name'].tolist()
        df_map[date_variables] = df_map[date_variables].apply(
            lambda x: x.apply(lambda y: pd.to_datetime(y)))
        quality_report = {}
        exclude_files = ('df_map.csv', 'vertex_dictionary.csv')
        vertex_dataframes = [
            file for file in vertex_dataframes
            if file.endswith('.csv') and (file not in exclude_files)]
        df_forms_dict = {}
        for file in vertex_dataframes:
            df_form = pd.read_csv(
                os.path.join(vertex_dataframes_path, file),
                dtype=dtype_dict,
                keep_default_na=False,
                na_values=na_values
            )
            if 'subjid' in df_form.columns:
                key = file.split('.csv')[0]
                df_forms_dict[key] = df_form
            else:
                print(f'{file} does not include subjid, ignoring this.')
        return {
            'df_map': df_map,
            'dictionary': dictionary,
            'df_forms_dict': df_forms_dict,
            'quality_report': quality_report,
        }
    except Exception:
        print('Could not load the VERTEX dataframes.')
        raise

def save_public_outputs(buttons, insight_panels, df_map, df_countries, df_forms_dict,
        dictionary, quality_report, project_path, config_dict):
    """Save public outputs to the PUBLIC folder."""
    public_path = os.path.join(
        project_path, config_dict['public_path'])
    if os.path.exists(public_path):
        print(f'Folder "{public_path}" already exists, removing this')
        shutil.rmtree(public_path)
    print(f'Saving files for public dashboard to "{public_path}"')
    os.makedirs(
        os.path.dirname(os.path.join(public_path, '')), exist_ok=True)
    for ip in config_dict['insight_panels']:
        os.makedirs(
            os.path.dirname(os.path.join(public_path, ip, '')),
            exist_ok=True)
    buttons = get_visuals(
        buttons, insight_panels,
        df_map=df_map, df_forms_dict=df_forms_dict,
        dictionary=dictionary, quality_report=quality_report,
        filepath=os.path.join(public_path, ''))
    os.makedirs(os.path.dirname(public_path), exist_ok=True)
    if config_dict['save_base_files_to_public_path']:
        shutil.copy('descriptive_dashboard_public.py', public_path)
        shutil.copy('IsaricDraw.py', public_path)
        shutil.copy('requirements.txt', public_path)
        assets_path = os.path.join(public_path, 'assets/')
        os.makedirs(os.path.dirname(assets_path), exist_ok=True)
        shutil.copytree('assets', assets_path, dirs_exist_ok=True)
    metadata_file = os.path.join(
        public_path, 'dashboard_metadata.json')
    with open(metadata_file, 'w') as file:
        json.dump({'insight_panels': buttons}, file, indent=4)
    data_file = os.path.join(public_path, 'dashboard_data.csv')
    df_countries.to_csv(data_file, index=False)
    config_json_file = os.path.join(
        public_path, 'config_file.json')
    with open(config_json_file, 'w') as file:
        save_config_keys = [
            'project_name', 'map_layout_center_latitude',
            'map_layout_center_longitude', 'map_layout_zoom']
        save_config_dict = {k: config_dict[k] for k in save_config_keys}
        json.dump(save_config_dict, file, indent=4)
    logger.info(f'Public dashboard files saved to {public_path}')