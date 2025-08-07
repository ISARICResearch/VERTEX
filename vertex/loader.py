"""loader.py"""
import os
import json

import vertex.getREDCapData as getRC

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
    api_url = config_dict.get('api_url')
    api_key = config_dict.get('api_key')
    get_data_from_api = api_url is not None and api_key is not None

    if get_data_from_api:
        user_assigned_to_dag = getRC.user_assigned_to_dag(api_url, api_key)
        get_data_kwargs = {
            'data_access_groups': config_dict['data_access_groups'],
            'user_assigned_to_dag': user_assigned_to_dag}
        return getRC.get_redcap_data(api_url, api_key, **get_data_kwargs)
    else:
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
        except Exception:
            print('Could not load the VERTEX dataframes.')
            raise
