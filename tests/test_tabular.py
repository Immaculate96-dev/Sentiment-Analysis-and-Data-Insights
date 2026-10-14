import numpy as np
import pandas as pd

import tabular_analysis as ta
from demo_data import make_demo_dataframe


def test_types_on_demo_data():
    df, types = ta.prepare_dataframe(make_demo_dataframe())
    assert types["review_id"] == "id"
    assert types["date"] == "datetime"
    assert types["price"] == "numeric"
    assert types["product"] == "categorical"
    assert types["review_text"] == "text"


def test_text_numbers_and_money_are_converted():
    raw = pd.DataFrame({"Amount": [f"${i * 13:,}" for i in range(60)]})
    df, types = ta.prepare_dataframe(raw)
    assert types["Amount"] == "numeric"
    assert df["Amount"].iloc[2] == 26


def test_day_first_dates_detected_from_data():
    raw = pd.DataFrame({"d": pd.date_range("2024-01-07", periods=40, freq="W").strftime("%d/%m/%Y")})
    df, types = ta.prepare_dataframe(raw)
    assert types["d"] == "datetime"
    assert df["d"].min() == pd.Timestamp("2024-01-07")


def test_ambiguous_dates_follow_the_option():
    raw = pd.DataFrame({"d": ["03/04/2024", "05/06/2024", "01/02/2024"] * 10})
    month_first, _ = ta.prepare_dataframe(raw, dayfirst=False)
    day_first, _ = ta.prepare_dataframe(raw, dayfirst=True)
    assert month_first["d"].min() == pd.Timestamp("2024-01-02")
    assert day_first["d"].min() == pd.Timestamp("2024-02-01")
    assert month_first.attrs["date_order"]["d"] == "ambiguous_month"


def test_empty_constant_and_binary_columns():
    raw = pd.DataFrame({"a": [None] * 30, "b": ["x"] * 30, "c": [0, 1] * 15})
    _, types = ta.prepare_dataframe(raw)
    assert types == {"a": "empty", "b": "constant", "c": "categorical"}


def test_duplicate_column_names_are_made_unique():
    raw = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"])
    df, _ = ta.prepare_dataframe(raw)
    assert list(df.columns) == ["x", "x_2"]


def test_iqr_outliers():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
    count, _, upper = ta.iqr_outliers(s)
    assert count == 1 and upper < 1000
    assert ta.iqr_outliers(pd.Series([5] * 10))[0] == 0


def test_strong_correlation_is_reported():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    raw = pd.DataFrame({"x": x, "y": x * 2 + rng.normal(scale=0.1, size=200), "z": rng.normal(size=200)})
    df, types = ta.prepare_dataframe(raw)
    texts = [t for _, t in ta.generate_insights(df, types)]
    assert any("`x` and `y` are strongly correlated" in t for t in texts)


def test_every_column_pair_produces_a_result():
    df, types = ta.prepare_dataframe(make_demo_dataframe())
    usable = [c for c in df.columns if types[c] in ta.ANALYSABLE]
    for x in usable:
        for y in usable:
            fig, text = ta.relationship(df, x, y, types)
            assert text
