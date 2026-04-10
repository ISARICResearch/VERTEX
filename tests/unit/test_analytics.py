import pandas as pd

from vertex.IsaricAnalytics import (
    convert_categorical_to_onehot,
    convert_onehot_to_categorical,
    from_timeA_to_timeB,
)


def test_categorical_onehot_roundtrip():
    df = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
    dictionary = pd.DataFrame(
        {
            "field_name": ["cat", "num"],
            "field_type": ["categorical", "numeric"],
            "field_label": ["Category", "Number"],
            "parent": ["root", "root"],
        }
    )

    encoded = convert_categorical_to_onehot(df.copy(), dictionary, categorical_columns=["cat"])
    assert "cat___a" in encoded.columns
    assert "cat___b" in encoded.columns

    decoded = convert_onehot_to_categorical(encoded.copy(), dictionary, categorical_columns=["cat"])
    assert decoded["cat"].tolist() == ["a", "b"]


def test_from_time_a_to_time_b_days():
    df = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "end_date": pd.to_datetime(["2024-01-03", "2024-01-10"]),
        }
    )
    dictionary = pd.DataFrame(
        {
            "field_name": ["end_date"],
            "form_name": ["outcome"],
            "field_type": ["date"],
            "field_label": ["End date"],
            "parent": ["dates"],
        }
    )

    out_df, out_dictionary = from_timeA_to_timeB(
        data=df.copy(),
        dictionary=dictionary.copy(),
        timeA_column="start_date",
        timeB_column="end_date",
        timediff_column="days_between",
        timediff_label="Days between dates",
        time_unit="days",
    )

    assert out_df["days_between"].tolist() == [2, 5]
    assert "days_between" in out_dictionary["field_name"].values
