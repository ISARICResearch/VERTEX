__all__ = [
    'get_project_data',
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
    get_projects_catalog,
    load_public_dashboard,
    load_vertex_data,
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
    """Retrieves project data from the project path folder.

    Parameters
    ----------
    project_path : str, pathlib.Path
        The incoming project path as a plain string (`py:class:str`) or
        :py:class:`pathlib.Path` object.

    Returns
    -------
    dict
        The project data dict
    """
    _project_path = pathlib.Path(project_path).resolve()

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
        "project_path": project_path,
        "config_dict": config_dict,
    }


@click.command()
@click.argument('project-path')
def main(project_path: str) -> Any:
    project_path = pathlib.Path(project_path).resolve()

    # 1. Get project data from the project path, which includes buttons,
    #    insight_panels, df_map, df_countries, df_forms_dict, dictionary,
    #    quality_report, project_path, config_dict
    #
    logger.info(f'Loading project data from project path: "{project_path}"')
    project_data = get_project_data(project_path)

    # 2. Save the analytics outputs to file (to an outputs folder inside
    #.   the project path)
    outputs_path = project_path.joinpath(project_data["config_dict"]["outputs_path"])
    logger.info(f'Saving outputs to "{outputs_path}"')
    save_public_outputs(**project_data)
