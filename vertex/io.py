"""io.py"""

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

import vertex.getREDCapData as getRC
from vertex.layout.insight_panels import get_visuals
from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)

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
    config_file = os.path.join(project_path, "config_file.json")
    config_dict = {}
    try:
        with open(config_file, "r") as json_data:
            config_dict = json.load(json_data)
    except IOError as e:
        logger.error(f"Could not read config_file.json: {e}, using defaults")

    if "project_owner" not in config_dict and "owner_email" in config_dict:
        # temporary compatibility
        config_dict["project_owner"] = config_dict["owner_email"]

    # If no insight_panels_path is specified, then this is a static project which requires dashboard_metadata.json.
    if "insight_panels_path" not in config_dict.keys():
        if not os.path.exists(os.path.join(project_path, "dashboard_metadata.json")):
            logger.error("Could not read dashboard_metadata.json in static project, cannot proceed.")
            logger.error(
                "please define insight_panels_path for data processing or "
                "add a dashboard_metadata.json file for static projects."
            )
        return config_dict

    insight_panels_path = os.path.join(project_path, config_dict["insight_panels_path"])
    for _, _, filenames in os.walk(insight_panels_path):
        insight_panels = [file.split(".py")[0] for file in filenames if file.endswith(".py") and not file.startswith("_")]
        break
    config_defaults["insight_panels"] = insight_panels
    config_defaults = {k: v for k, v in config_defaults.items() if k not in config_dict.keys()}
    config_dict = {**config_dict, **config_defaults}

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
    with open(metadata_file, "r") as file:
        metadata = json.load(file)
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
    try:
        vertex_dataframes_path = _get_vertex_dataframes_path(project_path, config_dict)
        vertex_dataframes = os.listdir(vertex_dataframes_path)
        dictionary = pd.read_csv(
            os.path.join(vertex_dataframes_path, "vertex_dictionary.csv"), dtype={"field_label": "str"}, keep_default_na=False
        )
        str_ind = dictionary["field_type"].isin(["freetext", "categorical"])
        str_columns = dictionary.loc[str_ind, "field_name"].tolist()
        non_str_columns = dictionary.loc[(str_ind == 0), "field_name"].tolist()
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
        df_map = pd.read_csv(
            os.path.join(vertex_dataframes_path, "df_map.csv"), dtype=dtype_dict, keep_default_na=False, na_values=na_values
        )
        date_variables = dictionary.loc[(dictionary["field_type"] == "date"), "field_name"].tolist()
        date_variables = [col for col in date_variables if col in df_map.columns]
        if date_variables:
            df_map[date_variables] = df_map[date_variables].apply(lambda x: x.apply(lambda y: pd.to_datetime(y)))
        quality_report = {}
        exclude_files = ("df_map.csv", "vertex_dictionary.csv")
        vertex_dataframes = [file for file in vertex_dataframes if file.endswith(".csv") and (file not in exclude_files)]
        df_forms_dict = {}
        for file in vertex_dataframes:
            df_form = pd.read_csv(
                os.path.join(vertex_dataframes_path, file), dtype=dtype_dict, keep_default_na=False, na_values=na_values
            )
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
    except Exception:
        logger.error("Could not load the VERTEX dataframes.")
        raise


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
        runtime_metadata = {"user": os.environ.get("USER", None), "timestamp": time.ctime()}
        save_config_dict["runtime_metadata"] = runtime_metadata
        json.dump(save_config_dict, file, indent=4)
        file.write("\n")
    logger.info(f"Public dashboard files saved to {outputs_path}")
