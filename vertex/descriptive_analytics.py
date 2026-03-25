__all__ = [
    "get_project_data",
]

# -- IMPORTS --

# -- Standard libraries --
import pathlib
import typing

# -- 3rd party libraries --
import click
import pandas as pd

# -- Internal libraries --
from vertex.io import (
    config_defaults,
    get_config,
    load_vertex_data,
    save_insight_panel_visuals,
    save_public_outputs,
)
from vertex.layout.insight_panels import (
    get_insight_panels,
    get_public_visuals,
)
from vertex.logging.logger import setup_logger
from vertex.map import (
    get_countries,
    merge_data_with_countries,
)

logger = setup_logger(__name__)


def get_project_data(project_path: str | pathlib.Path) -> dict[str, typing.Any]:
    """:py:class:`dict` : Retrieves project data from the project path folder.

    Parameters
    ----------
    project_path : str, pathlib.Path
        The incoming project path as a plain string (`py:class:str`) or
        :py:class:`pathlib.Path` object.

    Returns
    -------
    dict
        The project data dict

    Examples
    --------
    >>> project_data = get_project_data('demo-projects/ARChetypeCRF_dengue_synthetic/')
    2026-03-20 11:36:23 [INFO] vertex.io: Retrieving data from redcap API
    2026-03-20 11:36:23 [INFO] vertex.getREDCapData: REDCap data pipeline start
    2026-03-20 11:36:23 [INFO] vertex.getREDCapData: REDCap records export: requesting all records
    2026-03-20 11:36:27 [INFO] vertex.getREDCapData: REDCap records export complete in 3.7s (rows=6495)
    2026-03-20 11:36:27 [INFO] vertex.getREDCapData: REDCap step get_records finished in 3.7s
    2026-03-20 11:36:27 [INFO] vertex.getREDCapData: REDCap step get_data_dictionary finished in 0.2s
    2026-03-20 11:36:28 [INFO] vertex.getREDCapData: REDCap step get_missing_data_codes finished in 0.2s
    2026-03-20 11:36:30 [INFO] vertex.getREDCapData: REDCap step initial_data_processing finished in 2.3s (rows=6495, cols=685)
    2026-03-20 11:36:30 [INFO] vertex.getREDCapData: REDCap step get_form_event finished in 0.5s
    2026-03-20 11:36:31 [INFO] vertex.getREDCapData: REDCap step get_df_map finished in 0.1s (rows=1000)
    2026-03-20 11:36:31 [INFO] vertex.getREDCapData: REDCap step get_df_forms finished in 0.0s (forms=4)
    2026-03-20 11:36:31 [INFO] vertex.getREDCapData: REDCap data pipeline complete in 7.2s
    >>> project_data
    {'buttons': [{'item': 'Enrolment',
       'label': 'Enrolment Details',
       'suffix': 'enrolment_details'},
       ...
     ...
     ...
    'insight_panels_data_path': None,
    'write_api_cache': False}}
    """
    _project_path = project_path
    if not isinstance(_project_path, pathlib.Path):
        _project_path = pathlib.Path(_project_path).resolve()

    config_dict = get_config(_project_path, config_defaults)

    insight_panels_path = _project_path.joinpath(config_dict["insight_panels_path"])
    insight_panels, buttons = get_insight_panels(config_dict, insight_panels_path)

    filter_columns_dict = {
        "subjid": "subjid",
        "demog_sex": "filters_sex",
        "demog_age": "filters_age",
        "pres_date": "filters_admdate",
        "country_iso": "filters_country",
        "outco_binary_outcome": "filters_outcome",
    }

    data = load_vertex_data(_project_path, config_dict)
    df_map = data.get("df_map", None)
    df_forms_dict = data.get("df_forms_dict", {})
    dictionary = data.get("dictionary", None)
    quality_report = data.get("quality_report", {})
    df_map = df_map.reset_index(drop=True)
    df_map_with_countries = merge_data_with_countries(df_map)
    df_countries = get_countries(df_map_with_countries)
    df_filters = df_map_with_countries[filter_columns_dict.keys()].rename(columns=filter_columns_dict)
    df_map = pd.merge(df_map_with_countries, df_filters, on="subjid", how="left").reset_index(drop=True)
    df_forms_dict = {
        form: pd.merge(df_form, df_filters, on="subjid", how="left").reset_index(drop=True)
        for form, df_form in df_forms_dict.items()
    }

    logger.debug(f"{list(insight_panels)[0]}")

    return {
        "buttons": buttons,
        "insight_panels": insight_panels,
        "df_map": df_map,
        "df_countries": df_countries,
        "df_forms_dict": df_forms_dict,
        "dictionary": dictionary,
        "quality_report": quality_report,
        "project_path": _project_path,
        "config_dict": config_dict,
    }


@click.command()
@click.argument("project-path")
def main(project_path: str | pathlib.Path) -> None:
    """:py:class:`NoneType` : The entrypoint function.

    Parameters
    ----------
    project_path : str, pathlib.Path
        The project path as a plain string or :py:class:`pathlib.Path` object.

    """
    if not isinstance(project_path, pathlib.Path):
        project_path = pathlib.Path(project_path).resolve()

    # 1. Get project data from the project path, which includes buttons,
    #    insight_panels, df_map, df_countries, df_forms_dict, dictionary,
    #    quality_report, project_path, config_dict
    logger.info(f'Loading project data from project path: "{project_path}"')
    project_data = get_project_data(project_path)

    # 2. Save the analytics outputs to file (to an outputs subfolder inside
    #    the project path)
    project_outputs_path = project_path.joinpath(project_data["config_dict"]["outputs_path"])
    logger.info(f'Saving outputs to "{project_outputs_path}"')
    save_public_outputs(**project_data)
    logger.info(f'Saving insight panel figures to "{project_outputs_path}"')
    insight_panels, _ = get_public_visuals(project_outputs_path, project_data["buttons"])
    save_insight_panel_visuals(insight_panels, project_outputs_path, project_outputs_path.joinpath("visuals"))
