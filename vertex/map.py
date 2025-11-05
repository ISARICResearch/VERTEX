import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests


def merge_data_with_countries(df_map, add_capital_location=False):
    """Add country variable to df_map and merge with country metadata."""
    contries_path = "assets/countries.csv"
    countries = pd.read_csv(contries_path, encoding="latin-1")

    geojson = os.path.join(
        "https://raw.githubusercontent.com/",
        "martynafford/natural-earth-geojson/master/",
        "50m/cultural/ne_50m_populated_places_simple.json",
    )
    capitals = json.loads(requests.get(geojson).text)
    features = ["adm0_a3", "latitude", "longitude", "featurecla"]
    capitals = [{k: x["properties"][k] for k in features} for x in capitals["features"]]
    capitals = pd.DataFrame.from_dict(capitals)
    capitals = capitals.sort_values(by=["adm0_a3", "featurecla"])
    capitals = capitals.drop_duplicates(["adm0_a3"]).reset_index(drop=True)
    capitals.drop(columns=["featurecla"], inplace=True)
    capitals.rename(columns={"adm0_a3": "Code"}, inplace=True)

    countries = pd.merge(countries, capitals, how="left", on="Code")

    countries.rename(
        columns={
            "Code": "country_iso",
            "Country": "country_name",
            "Region": "country_region",
            "Income group": "country_income",
            "latitude": "country_capital_lat",
            "longitude": "country_capital_lon",
        },
        inplace=True,
    )
    df_map = pd.merge(df_map, countries, on="country_iso", how="left")
    return df_map


def get_countries(df_map):
    df_countries = df_map[["country_iso", "country_name", "subjid"]]
    df_countries = df_countries.groupby(["country_iso", "country_name"]).count().reset_index()
    df_countries.rename(columns={"subjid": "country_count"}, inplace=True)
    return df_countries


def get_public_countries(path):
    data_file = os.path.join(path, "dashboard_data.csv")
    df_countries = pd.read_csv(data_file)
    return df_countries


def interpolate_colors(colors, n):
    """Interpolate among multiple hex colors."""
    # Convert all hex colors to RGB
    rgbs = [tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) for color in colors]

    interpolated_colors = []
    # Number of transitions is one less than the number of colors
    transitions = len(colors) - 1

    # Calculate the number of steps for each transition
    steps_per_transition = n // transitions
    steps_per_transition = [steps_per_transition + 1] * (n % transitions) + [steps_per_transition] * (
        transitions - (n % transitions)
    )

    # Interpolate between each pair of colors
    for i in range(transitions):
        for step in range(steps_per_transition[i]):
            interpolated_rgb = [
                int(rgbs[i][j] + (float(step) / steps_per_transition[i]) * (rgbs[i + 1][j] - rgbs[i][j])) for j in range(3)
            ]
            interpolated_colors.append(f"rgb({interpolated_rgb[0]}, " + f"{interpolated_rgb[1]}," + f"{interpolated_rgb[2]})")

    # Append the last color
    if len(interpolated_colors) < n:
        interpolated_colors.append(f"rgb({rgbs[-1][0]}, {rgbs[-1][1]}, {rgbs[-1][2]})")

    if len(rgbs) > n:
        interpolated_colors = [f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})" for rgb in rgbs[:n]]
    return interpolated_colors


def get_map_colorscale(df_countries, map_percentile_cutoffs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]):
    cutoffs = np.percentile(df_countries["country_count"], map_percentile_cutoffs)
    if df_countries["country_count"].count() < len(map_percentile_cutoffs):
        cutoffs = df_countries["country_count"].sort_values()
    cutoffs = cutoffs / df_countries["country_count"].max()
    num_colors = len(cutoffs)
    cutoffs = np.insert(np.repeat(cutoffs, 2)[:-1], 0, 0)
    colors = interpolate_colors(["0000FF", "00EA66", "A7FA00", "FFBE00", "FF7400", "FF3500"], num_colors)
    colors = np.repeat(colors, 2)
    custom_scale = [[x, y] for x, y in zip(cutoffs, colors)]
    return custom_scale


def create_map(df_countries, map_layout_dict=None):
    geojson = os.path.join(
        "https://raw.githubusercontent.com/",
        "martynafford/natural-earth-geojson/master/",
        "50m/cultural/ne_50m_admin_0_countries.json",
    )

    map_colorscale = get_map_colorscale(df_countries)

    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            featureidkey="properties.ADM0_A3",
            locations=df_countries["country_iso"],
            z=df_countries["country_count"],
            text=df_countries["country_name"],
            colorscale=map_colorscale,
            showscale=True,
            zmin=1,
            zmax=df_countries["country_count"].max(),
            marker_line_color="black",
            marker_opacity=0.5,
            marker_line_width=0.3,
            colorbar={
                "bgcolor": "rgba(255,255,255,1)",
                "thickness": 20,
                "ticklen": 5,
                "x": 1,
                "xref": "paper",
                "xanchor": "right",
                "xpad": 5,
            },
        )
    )
    fig.update_layout(map_layout_dict)
    # fig.update_layout({'width': 10.5})
    return fig


def filter_df_map(df_map, sex_value, age_value, country_value, admdate_value, admdate_marks, outcome_value):
    df_map["filters_age"] = df_map["filters_age"].astype(float)
    admdate_min = pd.to_datetime(admdate_marks[str(admdate_value[0])]["label"])
    admdate_max = pd.to_datetime(admdate_marks[str(admdate_value[1])]["label"]) + pd.DateOffset(months=1)
    df_map_filtered = df_map[
        (df_map["filters_sex"].isin(sex_value))
        & ((df_map["filters_age"] >= age_value[0]) | df_map["filters_age"].isna())
        & ((df_map["filters_age"] <= age_value[1]) | df_map["filters_age"].isna())
        & ((df_map["filters_admdate"] >= admdate_min) | df_map["filters_admdate"].isna())
        & ((df_map["filters_admdate"] <= admdate_max) | df_map["filters_admdate"].isna())
        & (df_map["filters_outcome"].isin(outcome_value))
        & (df_map["filters_country"].isin(country_value))
    ]
    return df_map_filtered.reset_index(drop=True)
