"""io.py"""

import glob
import json
import os
import re
import shutil
import subprocess
import time
import typing
from pathlib import Path

import pandas as pd

import vertex.getREDCapData as getRC
from vertex.layout.insight_panels import get_visuals
from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)
_VERTEX_GIT_METADATA = None

config_defaults = {
    "project_name": None,
    "project_id": None,
    "project_owner": None,
    "is_public": True,
    "data_access_groups": None,
    "map_layout_center_latitude": 6,
    "map_layout_center_longitude": -75,
    "map_layout_zoom": 1.7,
    "save_outputs": False,
    "outputs_path": "outputs/",
    "insight_panels_path": "insight_panels/",
    "insight_panels": [],
    "insight_panels_data_path": None,
    "write_api_cache": False,
}


def get_demo_projects_root():
    return Path("demo-projects/").expanduser()


def get_static_projects_root():
    return Path(os.getenv("VERTEX_PROJECTS_DIR") or "projects/").expanduser()


def _normalise_project_id(value, project_path):
    if value in (None, ""):
        return None
    project_id = str(value).strip()
    if not project_id:
        logger.warning(f"Invalid project_id in {project_path}/config_file.json: {value!r}")
        return None
    return project_id


def _normalise_owner_email(value):
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def _normalise_is_public(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _get_vertex_git_metadata():
    global _VERTEX_GIT_METADATA
    if _VERTEX_GIT_METADATA is not None:
        return _VERTEX_GIT_METADATA

    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        metadata = {"commit_sha": commit_sha or None}
    except Exception:
        env_sha = (os.getenv("VERTEX_GIT_SHA") or "").strip()
        metadata = {"commit_sha": env_sha or None}

    _VERTEX_GIT_METADATA = metadata
    return metadata


def _get_vertex_runtime_metadata(config_dict):
    git_metadata = _get_vertex_git_metadata()
    latest_sha = git_metadata.get("commit_sha")
    return {
        "user": os.environ.get("USER", None),
        "timestamp": time.ctime(),
        "vertex_commit_sha": latest_sha,
    }


def _validate_project_config(config, project_path, project_type):
    required_fields = ["project_name", "project_id", "project_owner", "is_public"]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        logger.warning(f"Project config missing required fields in {project_path}: {missing_fields}")

    project_name = config.get("project_name")
    if project_name is None or str(project_name).strip() == "":
        logger.warning(f"Project config has invalid project_name in {project_path}")

    owner = config.get("project_owner")
    if owner is not None:
        owner_str = str(owner).strip()
        if owner_str and "@" not in owner_str:
            logger.warning(f"Project config has non-email project_owner in {project_path}: {owner!r}")

    if "is_public" in config and not isinstance(config.get("is_public"), bool):
        logger.warning(
            f"Project config field is_public should be boolean in {project_path}; "
            f"coercing from {type(config.get('is_public')).__name__}"
        )

    if project_type == "analysis" and "insight_panels_path" not in config:
        logger.warning(f"Analysis project missing insight_panels_path in {project_path}")


def should_save_outputs(config_dict):
    if not config_dict.get("save_outputs", False):
        return False

    # Prevent accidental rewrites of tracked output files; opt-in explicitly.
    env_flag = os.getenv("VERTEX_ENABLE_SAVE_OUTPUTS")
    if env_flag is None:
        return False
    return env_flag.strip().lower() in {"1", "true", "yes", "y"}


def get_config(project_path, config_defaults):
    local_defaults = dict(config_defaults)
    config_file = os.path.join(project_path, "config_file.json")
    config_dict = {}
    try:
        with open(config_file, "r") as json_data:
            config_dict = json.load(json_data)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Could not read config_file.json: {e}, using defaults")

    if "project_owner" not in config_dict and "owner_email" in config_dict:
        # temporary compatibility
        config_dict["project_owner"] = config_dict["owner_email"]

    # If no insight_panels_path is specified, then this is a static project which requires dashboard_metadata.json.
    if "insight_panels_path" not in config_dict.keys():
        default_insight_panels_path = local_defaults.get("insight_panels_path", "insight_panels/")
        guessed_insight_panels_path = os.path.join(project_path, default_insight_panels_path)
        if os.path.isdir(guessed_insight_panels_path):
            logger.warning(f"Config for {project_path} has no insight_panels_path; defaulting to {default_insight_panels_path}")
            config_dict["insight_panels_path"] = default_insight_panels_path
        else:
            if not os.path.exists(os.path.join(project_path, "dashboard_metadata.json")):
                logger.error("Could not read dashboard_metadata.json in static project, cannot proceed.")
                logger.error(
                    "please define insight_panels_path for data processing or "
                    "add a dashboard_metadata.json file for static projects."
                )
            static_defaults = {
                k: v
                for k, v in local_defaults.items()
                if k not in {"insight_panels_path", "insight_panels", "insight_panels_data_path", "write_api_cache"}
            }
            return {**static_defaults, **config_dict}

    insight_panels_path = os.path.join(project_path, config_dict["insight_panels_path"])
    insight_panels = []
    if not os.path.isdir(insight_panels_path):
        logger.error(f"Insight panels path does not exist: {insight_panels_path}")
    else:
        for _, _, filenames in os.walk(insight_panels_path):
            insight_panels = [file.split(".py")[0] for file in filenames if file.endswith(".py") and not file.startswith("_")]
            break
    local_defaults["insight_panels"] = insight_panels
    missing_defaults = {k: v for k, v in local_defaults.items() if k not in config_dict.keys()}
    config_dict = {**config_dict, **missing_defaults}
    if not isinstance(config_dict.get("insight_panels"), list):
        logger.warning(
            f"Config field insight_panels should be a list in {project_path}; "
            f"got {type(config_dict.get('insight_panels')).__name__}, defaulting to []"
        )
        config_dict["insight_panels"] = []

    if any([x not in insight_panels for x in config_dict["insight_panels"]]):
        missing_insight_panels = [x for x in config_dict["insight_panels"] if x not in insight_panels]
        logger.warning(
            f"The following insight panels are ignored and will not appear in the dashboard: {missing_insight_panels}"
        )
        config_dict["insight_panels"] = [x for x in config_dict["insight_panels"] if x in insight_panels]

    if any([x not in config_dict["insight_panels"] for x in insight_panels]):
        missing_insight_panels = [x for x in insight_panels if x not in config_dict["insight_panels"]]
        logger.warning(
            f"The following insight panels are available but not included in the dashboard "
            f"(add these to config_file.json to include them): {missing_insight_panels}"
        )

    return config_dict


def get_project_record(project_path, project_type):
    config_file = Path(project_path) / "config_file.json"
    config = {}
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read config for {project_path.name}: {e}")

    if "project_owner" not in config and "owner_email" in config:
        config["project_owner"] = config["owner_email"]

    _validate_project_config(config, project_path, project_type)

    data_source = None
    if project_type == "analysis":
        has_api_url = bool(str(config.get("api_url", "")).strip())
        has_api_key = bool(str(config.get("api_key", "")).strip())
        data_source = "api" if (has_api_url and has_api_key) else "files"

    return {
        "path": str(project_path) + "/",
        "name": config.get("project_name", project_path.name),
        "project_id": _normalise_project_id(config.get("project_id"), project_path),
        "project_owner": _normalise_owner_email(config.get("project_owner")),
        "is_public": _normalise_is_public(config.get("is_public", True)),
        "project_type": project_type,
        "data_source": data_source,
    }


def get_projects_catalog():
    project_catalog = []
    roots = [("analysis", get_demo_projects_root()), ("prebuilt", get_static_projects_root())]
    for project_type, root in roots:
        logger.info(f"Looking for {project_type} projects in: {root.resolve()}")
        if not root.exists():
            logger.warning(f"Project directory does not exist: {root}")
            continue
        for project_path in root.iterdir():
            if not project_path.is_dir() or project_path.name.startswith("."):
                continue
            if not (project_path / "config_file.json").exists():
                logger.warning(f"Skipping folder without config_file.json: {project_path}")
                continue
            project_catalog.append(get_project_record(project_path, project_type))
    logger.info(f"Found {len(project_catalog)} total projects.")
    return project_catalog


def get_projects():
    projects = get_projects_catalog()
    return [item["path"] for item in projects], [item["name"] for item in projects]


def _get_vertex_dataframes_path(project_path, config_dict):
    data_path = config_dict.get("insight_panels_data_path") or "analysis_data/"
    return os.path.join(project_path, data_path)


def _has_vertex_data_cache(project_path, config_dict):
    vertex_dataframes_path = _get_vertex_dataframes_path(project_path, config_dict)
    required_files = [
        os.path.join(vertex_dataframes_path, "df_map.csv"),
        os.path.join(vertex_dataframes_path, "vertex_dictionary.csv"),
    ]
    return all(os.path.exists(file_path) for file_path in required_files)


def _save_vertex_data_cache(project_path, config_dict, data):
    vertex_dataframes_path = _get_vertex_dataframes_path(project_path, config_dict)
    os.makedirs(vertex_dataframes_path, exist_ok=True)

    data["df_map"].to_csv(os.path.join(vertex_dataframes_path, "df_map.csv"), index=False)
    data["dictionary"].to_csv(os.path.join(vertex_dataframes_path, "vertex_dictionary.csv"), index=False)
    for form_name, df_form in data["df_forms_dict"].items():
        df_form.to_csv(os.path.join(vertex_dataframes_path, f"{form_name}.csv"), index=False)

    quality_report_path = os.path.join(vertex_dataframes_path, "quality_report.json")
    with open(quality_report_path, "w") as file:
        json.dump(data.get("quality_report", {}), file, indent=2)
        file.write("\n")

    metadata_path = os.path.join(vertex_dataframes_path, "vertex_runtime_metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(_get_vertex_runtime_metadata(config_dict), file, indent=2)
        file.write("\n")

    logger.info(f"Saved API snapshot cache to {vertex_dataframes_path}")


def load_vertex_data(project_path, config_dict):
    api_url = config_dict.get("api_url")
    api_key = config_dict.get("api_key")
    api_url_str = "" if api_url is None else str(api_url).strip()
    api_key_str = "" if api_key is None else str(api_key).strip()
    get_data_from_api = bool(api_url_str and api_key_str)
    logger.debug(f"api_url: {api_url}, api_key: {'***' if api_key else None}")

    write_api_cache = _as_bool(os.getenv("VERTEX_WRITE_API_CACHE"), _as_bool(config_dict.get("write_api_cache"), False))

    if get_data_from_api:
        data = load_vertex_from_api(api_url_str, api_key_str, config_dict)
        if write_api_cache:
            try:
                _save_vertex_data_cache(project_path, config_dict, data)
            except Exception as e:
                logger.warning(f"Failed to persist API snapshot cache for {project_path}: {e}")
    else:
        logger.info(f"Loading data from {project_path}")
        data = load_vertex_from_files(project_path, config_dict)
    return data


def load_public_dashboard(project_path, config_dict):
    metadata_file = os.path.join(project_path, config_dict.get("dashboard_metadata") or "dashboard_metadata.json")
    try:
        with open(metadata_file, "r") as file:
            metadata = json.load(file)
    except Exception as exc:
        logger.error(f"Could not read dashboard metadata from {metadata_file}: {exc}")
        metadata = {"insight_panels": []}
    if not isinstance(metadata, dict):
        logger.error(f"Dashboard metadata in {metadata_file} must be a JSON object; got {type(metadata).__name__}")
        return {"insight_panels": []}
    return metadata


def load_vertex_from_api(api_url, api_key, config_dict):
    """Load data from the REDCap API."""
    logger.info("Retrieving data from redcap API")
    user_assigned_to_dag = getRC.user_assigned_to_dag(api_url, api_key)
    get_data_kwargs = {"data_access_groups": config_dict["data_access_groups"], "user_assigned_to_dag": user_assigned_to_dag}
    df_map, df_forms_dict, dictionary, quality_report = getRC.get_redcap_data(api_url, api_key, **get_data_kwargs)
    return {
        "df_map": df_map,
        "dictionary": dictionary,
        "df_forms_dict": df_forms_dict,
        "quality_report": quality_report,
    }


def load_vertex_from_files(project_path, config_dict):
    vertex_dataframes_path = _get_vertex_dataframes_path(project_path, config_dict)
    dictionary_columns = ["field_name", "field_type", "field_label", "form_name", "parent", "branching_logic"]
    if not os.path.isdir(vertex_dataframes_path):
        logger.error(f"Could not load VERTEX dataframes path: {vertex_dataframes_path}")
        return {
            "df_map": pd.DataFrame(),
            "dictionary": pd.DataFrame(columns=dictionary_columns),
            "df_forms_dict": {},
            "quality_report": {},
        }

    vertex_dataframes = os.listdir(vertex_dataframes_path)
    dictionary_path = os.path.join(vertex_dataframes_path, "vertex_dictionary.csv")
    try:
        dictionary = pd.read_csv(dictionary_path, dtype={"field_label": "str"}, keep_default_na=False)
    except Exception as exc:
        logger.error(f"Could not read vertex dictionary at {dictionary_path}: {exc}")
        dictionary = pd.DataFrame(columns=dictionary_columns)

    for required_col in ("field_name", "field_type", "field_label"):
        if required_col not in dictionary.columns:
            logger.error(f"Dictionary missing required column '{required_col}' at {dictionary_path}; applying fallback.")
            dictionary[required_col] = ""

    str_ind = dictionary["field_type"].isin(["freetext", "categorical"])
    str_columns = dictionary.loc[str_ind, "field_name"].dropna().astype(str).tolist()
    non_str_columns = dictionary.loc[(str_ind == 0), "field_name"].dropna().astype(str).tolist()
    dtype_dict = {**{x: "str" for x in str_columns}}

    pandas_default_na_values = [
        "",
        " ",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    ]
    na_values = {**{x: pandas_default_na_values for x in non_str_columns}, **{x: "" for x in str_columns}}

    df_map_path = os.path.join(vertex_dataframes_path, "df_map.csv")
    try:
        df_map = pd.read_csv(df_map_path, dtype=dtype_dict, keep_default_na=False, na_values=na_values)
    except Exception as exc:
        logger.error(f"Could not read df_map at {df_map_path}: {exc}")
        df_map = pd.DataFrame()

    if not df_map.empty:
        date_variables = dictionary.loc[(dictionary["field_type"] == "date"), "field_name"].tolist()
        date_variables = [col for col in date_variables if col in df_map.columns]
        if date_variables:
            df_map[date_variables] = df_map[date_variables].apply(lambda x: x.apply(lambda y: pd.to_datetime(y)))

    quality_report = {}
    exclude_files = ("df_map.csv", "vertex_dictionary.csv")
    vertex_dataframes = [file for file in vertex_dataframes if file.endswith(".csv") and (file not in exclude_files)]
    df_forms_dict = {}
    for file in vertex_dataframes:
        file_path = os.path.join(vertex_dataframes_path, file)
        try:
            df_form = pd.read_csv(file_path, dtype=dtype_dict, keep_default_na=False, na_values=na_values)
        except Exception as exc:
            logger.warning(f"Could not read form data from {file_path}: {exc}")
            continue
        if "subjid" in df_form.columns:
            key = file.split(".csv")[0]
            df_forms_dict[key] = df_form
        else:
            logger.warning(f"{file} does not include subjid, ignoring.")
    return {
        "df_map": df_map,
        "dictionary": dictionary,
        "df_forms_dict": df_forms_dict,
        "quality_report": quality_report,
    }


def get_project_name(project_path):
    config_file = Path(project_path) / "config_file.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                logger.debug(f"Loaded config for {project_path.name}: {config}")
                project_name = config.get("project_name", project_path.name)
        except Exception as e:
            logger.warning(f"Could not read config for {project_path.name}: {e}")
            project_name = project_path.name
    else:
        project_name = project_path.name
    return project_name


def save_public_outputs(
    buttons, insight_panels, df_map, df_countries, df_forms_dict, dictionary, quality_report, project_path, config_dict
):
    """Save outputs to the `outputs` folder."""
    outputs_path = os.path.join(project_path, config_dict["outputs_path"])
    if os.path.exists(outputs_path):
        logger.warning(f'Folder "{outputs_path}" already exists, removing this')
        shutil.rmtree(outputs_path)
    logger.info(f'Saving files for static dashboard to "{outputs_path}"')
    os.makedirs(os.path.dirname(os.path.join(outputs_path, "")), exist_ok=True)
    for ip in config_dict["insight_panels"]:
        os.makedirs(os.path.dirname(os.path.join(outputs_path, ip, "")), exist_ok=True)
    buttons = get_visuals(
        buttons,
        insight_panels,
        df_map=df_map,
        df_forms_dict=df_forms_dict,
        dictionary=dictionary,
        quality_report=quality_report,
        filepath=os.path.join(outputs_path, ""),
    )
    os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
    metadata_file = os.path.join(outputs_path, "dashboard_metadata.json")
    with open(metadata_file, "w") as file:
        json.dump({"insight_panels": buttons}, file, indent=4)
        file.write("\n")
    data_file = os.path.join(outputs_path, "dashboard_data.csv")
    df_countries.to_csv(data_file, index=False)
    config_json_file = os.path.join(outputs_path, "config_file.json")
    with open(config_json_file, "w") as file:
        save_config_keys = [
            "project_name",
            "project_id",
            "project_owner",
            "is_public",
            "map_layout_center_latitude",
            "map_layout_center_longitude",
            "map_layout_zoom",
        ]
        save_config_dict = {k: config_dict.get(k) for k in save_config_keys}
        runtime_metadata = _get_vertex_runtime_metadata(config_dict)
        save_config_dict["runtime_metadata"] = runtime_metadata
        save_config_dict["vertex_commit_sha"] = runtime_metadata["vertex_commit_sha"]
        json.dump(save_config_dict, file, indent=4)
        file.write("\n")
    logger.info(f"Public dashboard files saved to {outputs_path}")


def save_insight_panel_visuals(
    insight_panels: dict[str, typing.Any], public_outputs_path: str | Path, visual_outputs_path: str | Path
) -> None:
    """:py:class:`NoneType` : Saves/writes all figures/plots in the insight panels to an output folder.

    Parameters
    ----------
    insight_panels : dict
        The insight panels dict.

    public_outputs_path : str, pathlib.Path
        The project public outputs path where the CSVs are stored.

    visual_outputs_path : str, pathlib.Path
        The output folder path to which all the visual artifacts,
        currently, just figures, will be saved/exported. This will
        usually be in a `visuals` subfolder within the project
        public outputs path.
    """
    _public_outputs_path = public_outputs_path
    if not isinstance(_public_outputs_path, Path):
        _public_outputs_path = Path(public_outputs_path).resolve()

    _visual_outputs_path = visual_outputs_path
    if not isinstance(_visual_outputs_path, Path):
        _visual_outputs_path = Path(_visual_outputs_path).resolve()
    if _visual_outputs_path.exists():
        logger.warning(f'Clearing pre-existing visual outputs folder "{_visual_outputs_path}"')
        shutil.rmtree(_visual_outputs_path)
    _visual_outputs_path.mkdir()

    for suffix in insight_panels:
        suffix_visuals_path = _visual_outputs_path.joinpath(suffix)
        if not suffix_visuals_path.exists():
            suffix_visuals_path.mkdir()

        logger.info(f'Saving "{suffix}" insight panel non-table figures to "{suffix_visuals_path}"')
        suffix_visuals = insight_panels[suffix].create_visuals()
        for idx in range(len(suffix_visuals)):
            fig, fig_text = suffix_visuals[idx][:2]
            if "table" in fig_text:
                continue
            fig_text = f"{fig_text.split('/')[-1]}.png"
            filepath = suffix_visuals_path.joinpath(fig_text)
            filepath.touch()
            fig.write_image(
                filepath, format="png", width=(fig.layout.minreducedwidth * 2.25), height=fig.layout.height, scale=3
            )
            logger.info(f'Saved "{fig_text}" to "{filepath}"')

        suffix_csv_source_path = _public_outputs_path.joinpath(suffix)
        logger.info(f'Copying "{suffix}" table CSVs from CSV output subfolder "{suffix_csv_source_path}"')
        table_csvs = copy_figure_table_csvs(suffix_csv_source_path, suffix_visuals_path)
        logger.info("Cleaning figure table CSVs")
        for csv in table_csvs:
            logger.info(f"Cleaning figure table CSV {csv}")
            clean_figure_table(pd.read_csv(csv)).to_csv(suffix_visuals_path.joinpath(csv.name), index=False)


def copy_figure_table_csvs(source_path: str | Path, target_path: str | Path) -> tuple[Path]:
    """:py:class:`tuple` : Copies figure table CSV filepaths found in the source folder to the target folder.

    The source and target folders must exist, as it is not within the function
    scope to create new folders, but simply to copy files between preexisting
    folders.

    The copied file retains the same name as the original, the copy method
    used is :py:func:`shutil.copy2`, which attempts to preserve file metadata,
    and nothing is copied if the source folder does not contain any figure
    table CSVs.

    Returns a tuple of copied CSV filepaths for reference.

    Parameters
    ----------
    source_path : str, pathlib.Path
        The souce folder path from which to copy figure table CSVs, if they
        exist.
    target_path : str, pathlib.Path
        The copy target folder path.

    Raises
    ------
    FileNotFoundError
        If either the source or target folder does not exist.

    Returns
    -------
    tuple
        A tuple of copied CSV filepaths.
    """
    _source_path = Path(source_path).resolve()
    if not _source_path.exists():
        raise FileNotFoundError(f'The source folder "{_source_path}" does not exist!')

    _target_path = Path(target_path).resolve()
    if not _target_path.exists():
        raise FileNotFoundError(f'The target folder "{_target_path}" does not exist!')

    copies = []

    logger.info(f'Copying figure table CSVs from "{_source_path}" to "{_target_path}"')
    for i, source_csv_path in enumerate(map(Path, glob.glob(f"{_source_path.joinpath('fig_table*.csv')}"))):
        logger.info(f'Copying "{source_csv_path}" to "{_target_path}"')
        shutil.copy2(source_csv_path, _target_path)
        copies.append(_target_path.joinpath(source_csv_path.name))

    if len(copies) == 0:
        logger.warning(f'No figure table CSVs found in "{_source_path}"')
    else:
        logger.info(f'{len(copies)} figure table CSVs copied to "{_target_path}"')

    return tuple(copies)


def strip_html(value: typing.Any) -> str | typing.Any:
    """:py:class:`typing.Any` : Strip HTML elements from a value.

    Parameters
    ----------
    value : typing.Any
        A value.

    Returns
    -------
    str, typing.Any
        Either a string stripped of all HTML elements, or the original non-
        string value.
    """
    if isinstance(value, str):
        return re.sub(r"<.*?>", "", value)

    return value


def strip_nonstandard_unicode_chars(value: typing.Any) -> str | typing.Any:
    """:py:class:`typing.Any` : Strip any non-standard (usually, non-alphabetic) Unicode characters from a value.

    The non-standard Unicode characters of interest are defined within the
    function itself, and are currently limited to the "↳" (U+21B3) character,
    but may be extended to include other characters.

    Parameters
    ----------
    value : typing.Any
        A value.

    Returns
    -------
    str, typing.Any
        Either a string stripped of all non-standard Unicode characters, or the
        original non- string value.
    """
    nonstandard_unicode_chars = "↳"

    if isinstance(value, str):
        return re.sub(rf"[{nonstandard_unicode_chars}]", "", value)

    return value


def clean_figure_table(figure_table: pd.DataFrame) -> pd.DataFrame:
    """:py:class:pandas.DataFrame : A cleaned figure table dataframe.

    The cleaning steps are unique to the Plotly graph object table format from
    which the table CSVs were originally, which contain HTML styling elements
    and non-standard (non-alphabetic) Unicode characters. The cleaning is the
    removal of such characters.

    Parameters
    ----------
    figure_table : pandas.DataFrame
        The original figure table as a Pandas dataframe.

    Returns
    -------
    pandas.DataFrame
        The cleaned figure table.
    """
    # The use of `pandas.DataFrame.map` here is not absolutely optimal, as
    # `map` applies changes across the dataframe element-wise, but is the
    # safer choice given that the dataframe may contain a number of non-string
    # columns which cannot be known in advance, while the cleaning steps
    # currently only apply to string values.
    return figure_table.map(strip_html).map(strip_nonstandard_unicode_chars)
