import pandas as pd

from vertex.layout.filters import get_filter_options
from vertex.map import filter_df_map


def test_filter_df_map_applies_expected_filters():
    df_map = pd.DataFrame(
        {
            "subjid": ["A", "B", "C"],
            "filters_sex": ["Female", "Male", "Female"],
            "filters_age": [25, 65, None],
            "filters_admdate": pd.to_datetime(["2024-01-10", "2024-02-10", "2024-03-10"]),
            "filters_outcome": ["Discharged", "Death", "Discharged"],
            "filters_country": ["GBR", "USA", None],
        }
    )
    admdate_marks = {
        "0": {"label": "2024-01"},
        "1": {"label": "2024-02"},
        "2": {"label": "2024-03"},
    }

    filtered = filter_df_map(
        df_map=df_map,
        sex_value=["Female"],
        age_value=[0, 40],
        country_value=["GBR"],
        admdate_value=[0, 2],
        admdate_marks=admdate_marks,
        outcome_value=["Discharged"],
    )

    # Row C survives because country is null and null is explicitly allowed.
    assert filtered["subjid"].tolist() == ["A", "C"]


def test_get_filter_options_returns_expected_shape():
    df_map = pd.DataFrame(
        {
            "demog_age": [18, 42, 80],
            "pres_date": pd.to_datetime(["2024-01-01", "2024-03-15", "2024-04-01"]),
            "filters_outcome": ["Discharged", "Death", "Discharged"],
            "filters_country": ["GBR", "USA", "BRA"],
            "filters_sex": ["Female", "Male", "Female"],
        }
    )

    options = get_filter_options(df_map)

    assert set(options.keys()) == {
        "sex_options",
        "age_options",
        "admdate_options",
        "country_options",
        "outcome_options",
    }
    assert options["age_options"]["min"] == 0
    assert options["age_options"]["max"] >= 100
    assert options["admdate_options"]["max"] >= options["admdate_options"]["min"]
    assert len(options["country_options"]) == 3
