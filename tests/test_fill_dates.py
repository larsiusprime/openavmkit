"""Filling blanks never puts text into a date column: a missing date stays missing, and the output can be written."""

import pandas as pd

from openavmkit.cleaning import _fill_with


def test_a_missing_date_stays_missing(tmp_path):
    df = pd.DataFrame({"demo_date": pd.to_datetime(["2023-01-01", None]), "zoning": ["R1", None]})
    df = _fill_with(df, "demo_date", "UNKNOWN")
    df = _fill_with(df, "zoning", "UNKNOWN")
    assert pd.api.types.is_datetime64_any_dtype(df["demo_date"]) and df["demo_date"].isna().sum() == 1
    assert df["zoning"].tolist() == ["R1", "UNKNOWN"]
    df.to_parquet(tmp_path / "x.parquet")  # writes: no dates mixed with text
    filled = _fill_with(df, "demo_date", pd.Timestamp("2020-01-01"))
    assert filled["demo_date"].isna().sum() == 0


def test_the_clean_stage_keeps_dates_as_dates(tmp_path):
    from openavmkit.cleaning import _fill_unknown_values_per_model_group

    settings = {
        "modeling": {"metadata": {"valuation_date": "2026-01-01"}, "model_groups": {"g": {}}},
        "field_classification": {"important": {}, "land": {"categorical": ["demo_date", "reno_date", "zoning"]}},
    }
    u = pd.DataFrame(
        {
            "key": ["a", "b", "c"],
            "model_group": ["g"] * 3,
            "demo_date": pd.to_datetime(["2023-01-01", None, None]),
            "reno_date": pd.to_datetime([None, "2022-02-02", None]),
            "zoning": ["R1", None, "R2"],
        }
    )
    out = _fill_unknown_values_per_model_group(u, settings)
    for c in ("demo_date", "reno_date"):
        assert pd.api.types.is_datetime64_any_dtype(out[c]), c
    assert out["zoning"].tolist() == ["R1", "UNKNOWN", "R2"]
    out.to_parquet(tmp_path / "u.parquet")
