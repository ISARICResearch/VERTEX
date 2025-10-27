import importlib
import json
import os
import sys

import pandas as pd
import plotly.graph_objs as go

from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)


def import_from_path(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_insight_panels(config_dict, insight_panels_path):
    # Import insight panels scripts
    insight_panels = {
        x: import_from_path(x, os.path.join(insight_panels_path, x + ".py")) for x in config_dict["insight_panels"]
    }
    buttons = [{**ip.define_button(), **{"suffix": suffix}} for suffix, ip in insight_panels.items()]
    return insight_panels, buttons


def get_visuals(buttons, insight_panels, df_map, df_forms_dict, dictionary, quality_report, filepath):
    for ii in range(len(buttons)):
        suffix = buttons[ii]["suffix"]
        visuals = insight_panels[suffix].create_visuals(
            df_map=df_map.copy(),
            df_forms_dict={k: v.copy() for k, v in df_forms_dict.items()},
            dictionary=dictionary.copy(),
            quality_report=quality_report,
            suffix=suffix,
            filepath=filepath,
            save_inputs=True,
        )
        buttons[ii]["graph_ids"] = [id for _, id, _, _ in visuals]
    return buttons


def get_public_visuals(path, buttons):
    visuals_by_suffix = {}
    for ii in range(len(buttons)):
        suffix = buttons[ii]["suffix"]
        graph_ids = tuple(x.split("/")[-1] for x in buttons[ii]["graph_ids"])
        metadata_dir = os.path.join(path, suffix)
        try:
            metadata_files = os.listdir(metadata_dir)
        except FileNotFoundError:
            logger.warning(f"No directory for suffix {suffix} at {metadata_dir}")
            visuals_by_suffix[suffix] = StaticInsightPanel([])
            continue

        # prefer explicit and/startswith for clarity
        metadata_files = [
            f
            for f in os.listdir(os.path.join(path, suffix))
            if f.endswith(".json") and any(f.startswith(gid) for gid in graph_ids)
        ]
        logger.debug(f"[get_public_visuals] graph_ids={graph_ids}, matched_files={metadata_files}")

        logger.info(f" Loading public visuals from {metadata_files} ")
        suffix_visuals = []
        for filename in metadata_files:
            new_file_path = os.path.join(metadata_dir, filename)
            with open(new_file_path, "r") as fh:
                new_file = json.load(fh)
            fig_id = new_file["fig_id"]
            logger.debug(f" fig_id: {fig_id} ")
            data = tuple(pd.read_csv(os.path.join(path, name)) for name in new_file["fig_data"])
            data = data[0] if (len(data) == 1) else data

            try:
                fig_fun = eval("idw." + new_file["fig_name"])
                new_file["fig_arguments"]["save_inputs"] = False

                # call the drawing function
                fig_ret = fig_fun(data, **new_file["fig_arguments"])

                # NORMALIZE RETURN VALUE:
                # Cases we want to accept:
                # 1) fig_fun returns (figure_obj, fig_id, label, about) -> use as-is
                # 2) fig_fun returns Figure/dict -> construct tuple using metadata
                if isinstance(fig_ret, (list, tuple)) and len(fig_ret) == 4:
                    # defensive check: first element should be a Figure-like or a dict
                    first = fig_ret[0]
                    if isinstance(first, (go.Figure, dict)):
                        figure_obj, returned_id, returned_label, returned_about = fig_ret
                        suffix_visuals.append((figure_obj, returned_id, returned_label, returned_about))
                    else:
                        # unexpected shape: fallback to constructing from metadata
                        logger.warning(f"Unexpected fig tuple first element type for {fig_id}: {type(first)}; using fallback.")
                        suffix_visuals.append((fig_ret, fig_id, new_file["fig_name"], new_file.get("about", "")))
                else:
                    # fig_fun returned a single figure object (or something else)
                    # Accept go.Figure or dict (plotly JSON); otherwise log and pack anyway
                    if isinstance(fig_ret, (go.Figure, dict)):
                        suffix_visuals.append((fig_ret, fig_id, new_file["fig_name"], new_file.get("about", "")))
                    else:
                        logger.warning(f"fig_fun for {fig_id} returned unexpected type {type(fig_ret)}; wrapping anyway.")
                        suffix_visuals.append((fig_ret, fig_id, new_file["fig_name"], new_file.get("about", "")))

            except AttributeError as e:
                logger.error(f" Could not load figure {fig_id} : {e} ")
                # remove this graph id from the button list to avoid broken references
                try:
                    buttons[ii]["graph_ids"].remove(fig_id)
                except ValueError:
                    pass
            except Exception as e:
                logger.exception(f"Unexpected error while loading {fig_id}: {e}")
                try:
                    buttons[ii]["graph_ids"].remove(fig_id)
                except ValueError:
                    pass

        # wrap visuals in our StaticInsightPanel stub so we can call create_visuals
        visuals_by_suffix[suffix] = StaticInsightPanel(suffix_visuals)
    logger.debug(visuals_by_suffix[suffix]._visuals)
    return visuals_by_suffix, buttons


class StaticInsightPanel:
    def __init__(self, visuals):
        self._visuals = visuals

    def create_visuals(self, **_):
        return self._visuals
