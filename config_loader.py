

############################################
# CONFIG
############################################
import os
import json

def get_config(init_project_path, config_defaults):
    config_file = os.path.join(init_project_path, 'config_file.json')
    try:
        with open(config_file, 'r') as json_data:
            config_dict = json.load(json_data)
        _ = config_dict['api_key']
        _ = config_dict['api_url']
    except Exception:
        error_message = f'''config_file.json is required in \
{init_project_path}. This file must contain both "api_key" and "api_url".'''
        print(error_message)
        raise SystemExit
    # The default for the list of insight panels is all that exist in the
    # relevant folder (which may or not be specified in config)
    if 'insight_panels_path' not in config_dict.keys():
        rel_insight_panels_path = config_defaults['insight_panels_path']
        config_dict['insight_panels_path'] = rel_insight_panels_path
    # Get a list of python files in the repository (excluding e.g. __init__.py)
    insight_panels_path = os.path.join(
        init_project_path, config_dict['insight_panels_path'])
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
