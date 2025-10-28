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
        logger.debug(f"\n[get_public_visuals] ===== Processing suffix: {suffix} =====")
        metadata_dir = os.path.join(path, suffix)

        if not os.path.isdir(metadata_dir):
            logger.warning(f"[get_public_visuals] Directory missing: {metadata_dir}")
            visuals_by_suffix[suffix] = StaticInsightPanel([])
            continue

        all_files = os.listdir(metadata_dir)
        logger.debug(f"[get_public_visuals] All files in {metadata_dir}: {all_files}")

        # get graph_ids (may be empty)
        graph_ids = tuple(x.split("/")[-1] for x in buttons[ii].get("graph_ids", []))
        logger.debug(f"[get_public_visuals] graph_ids from buttons: {graph_ids}")

        # select candidate metadata files
        metadata_files = [f for f in all_files if f.endswith(".json")]
        if graph_ids:
            metadata_files = [f for f in metadata_files if any(f.startswith(gid) for gid in graph_ids)]
        logger.debug(f"[get_public_visuals] JSON metadata_files selected: {metadata_files}")

        if not metadata_files:
            logger.warning(f"[get_public_visuals] No JSON metadata files found for suffix={suffix}")
            visuals_by_suffix[suffix] = StaticInsightPanel([])
            continue

        suffix_visuals = []

        for filename in metadata_files:
            new_file_path = os.path.join(metadata_dir, filename)
            logger.debug(f"[get_public_visuals] --- Reading metadata: {new_file_path}")
            try:
                with open(new_file_path, "r") as fh:
                    new_file = json.load(fh)
            except Exception as e:
                logger.error(f"[get_public_visuals] ERROR reading JSON {filename}: {e}")
                continue

            fig_id = new_file.get("fig_id")
            fig_name = new_file.get("fig_name")
            fig_args = new_file.get("fig_arguments", {})
            fig_data_files = new_file.get("fig_data", [])
            logger.debug(f"[get_public_visuals] fig_id={fig_id}, fig_name={fig_name}, data_files={fig_data_files}")
            logger.debug(f"[get_public_visuals] fig_args keys={list(fig_args.keys())}")

            # Check data file existence and load
            data_paths = [os.path.join(path, name) for name in fig_data_files]
            missing = [p for p in data_paths if not os.path.exists(p)]
            if missing:
                logger.warning(f"[get_public_visuals] Missing CSV(s) for {fig_id}: {missing}")
            try:
                data = tuple(pd.read_csv(p) for p in data_paths if os.path.exists(p))
                data = data[0] if len(data) == 1 else data
                logger.debug(
                    f"[get_public_visuals] Loaded data type={type(data)}, "
                    f"shape(s)={[getattr(d,'shape',None) for d in (data if isinstance(data,tuple) else [data])]}"
                )
            except Exception as e:
                logger.error(f"[get_public_visuals] ERROR reading CSVs for {fig_id}: {e}")
                continue

            # Now call figure builder
            try:
                import vertex.IsaricDraw as idw

                if not hasattr(idw, fig_name):
                    logger.error(f"[get_public_visuals] Draw function not found: idw.{fig_name}")
                    continue
                fig_fun = getattr(idw, fig_name)
                fig_args["save_inputs"] = False
                fig_ret = fig_fun(data, **fig_args)
                logger.debug(
                    f"[get_public_visuals] fig_fun returned type={type(fig_ret)} "
                    f"len={(len(fig_ret) if hasattr(fig_ret,'__len__') else 'no-len')}"
                )

                # Normalize result
                if isinstance(fig_ret, (list, tuple)) and len(fig_ret) == 4:
                    first = fig_ret[0]
                    logger.debug(f"[get_public_visuals] inner tuple types={[type(x) for x in fig_ret]}")
                    if isinstance(first, (go.Figure, dict)):
                        suffix_visuals.append(fig_ret)
                    else:
                        logger.warning(
                            f"[get_public_visuals] Unexpected first element type for {fig_id}:"
                            f"{type(first)}; skipping normalization"
                        )
                        suffix_visuals.append((fig_ret, fig_id, fig_name, ""))
                else:
                    suffix_visuals.append((fig_ret, fig_id, fig_name, ""))
                logger.debug(f"[get_public_visuals] Appended visual for {fig_id}, total now={len(suffix_visuals)}")
            except Exception as e:
                logger.exception(f"[get_public_visuals] ERROR building figure for {fig_id}: {e}")
                continue

        logger.debug(f"[get_public_visuals] Final visuals count for suffix={suffix}: {len(suffix_visuals)}")
        visuals_by_suffix[suffix] = StaticInsightPanel(suffix_visuals)

    logger.debug(f"[get_public_visuals] ===== Done: suffixes={list(visuals_by_suffix.keys())} =====")
    logger.debug(list(visuals_by_suffix)[0])
    return visuals_by_suffix, buttons


class StaticInsightPanel:
    def __init__(self, visuals):
        self._visuals = visuals

    def create_visuals(self, **_):
        return self._visuals
