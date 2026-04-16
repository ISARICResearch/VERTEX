__all__ = [
    "get_project_data",
]

# -- IMPORTS --

# -- Standard libraries --
import pathlib

# -- 3rd party libraries --
import click

# -- Internal libraries --
from vertex.io import (
    get_project_data,
    save_insight_panel_visuals,
    save_public_outputs,
)
from vertex.layout.insight_panels import (
    get_public_visuals,
)
from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)


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

    # 2. Save the public outputs to file (to an outputs subfolder inside
    #    the project path)
    project_outputs_path = project_path.joinpath(project_data["config_dict"]["outputs_path"])
    logger.info(f'Saving public outputs to "{project_outputs_path}"')
    save_public_outputs(**project_data)

    logger.info(f'Saving all figures to "{project_outputs_path}"')
    insight_panels, _ = get_public_visuals(project_outputs_path, project_data["buttons"])
    save_insight_panel_visuals(insight_panels, project_outputs_path, project_outputs_path.joinpath("visuals"))
