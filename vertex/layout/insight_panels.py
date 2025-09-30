import importlib
import sys
import os

def import_from_path(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_insight_panels(config_dict, insight_panels_path):
    # Import insight panels scripts
    insight_panels = {
        x: import_from_path(x, os.path.join(insight_panels_path, x + '.py'))
        for x in config_dict['insight_panels']}
    buttons = [
        {**ip.define_button(), **{'suffix': suffix}}
        for suffix, ip in insight_panels.items()]
    return insight_panels, buttons

def get_visuals(
        buttons, insight_panels, df_map, df_forms_dict,
        dictionary, quality_report, filepath):
    for ii in range(len(buttons)):
        suffix = buttons[ii]['suffix']
        visuals = insight_panels[suffix].create_visuals(
            df_map=df_map.copy(),
            df_forms_dict={k: v.copy() for k, v in df_forms_dict.items()},
            dictionary=dictionary.copy(), quality_report=quality_report,
            suffix=suffix, filepath=filepath, save_inputs=True)
        buttons[ii]['graph_ids'] = [id for _, id, _, _ in visuals]
    return buttons