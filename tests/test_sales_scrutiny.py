"""Sales scrutiny: drops as settings, the keep file, the flags file, and later passes."""

import os

import pandas as pd

from openavmkit.cleaning import write_filled
from openavmkit.data import SalesUniversePair
from openavmkit.sales_scrutiny_study import _calc_sales_scrutiny, read_keep_keys, run_heuristics
from openavmkit.utilities.settings import get_sales_scrutiny_drops


def _cluster(per_area: list[float]) -> pd.DataFrame:
    n = len(per_area)
    area = [1000, 1010, 990, 1005, 995, 1000, 1002, 998][:n]
    return pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(n)],
            "key_sale": [f"s{i}" for i in range(n)],
            "ss_id": ["c"] * n,
            "bldg_area_finished_sqft": area,
            "sale_price": [p * a for p, a in zip(per_area, area)],
            "sale_price_impr_sqft": per_area,
        }
    )


def test_a_second_pass_finds_the_outlier_another_one_hid():
    # One sale at 10x the median stretches the spread so far that one at 0.02x isn't a deviation below it.
    df = _cluster([100, 105, 95, 102, 98, 1000, 2, 101])
    once = _calc_sales_scrutiny(df, "sale_price_impr_sqft")
    assert once.loc[once["flagged"], "key_sale"].tolist() == ["s5"]

    twice = _calc_sales_scrutiny(df, "sale_price_impr_sqft", passes=2)
    flagged = twice[twice["flagged"]].set_index("key_sale")
    assert sorted(flagged.index) == ["s5", "s6"]
    assert flagged.loc["s5", "scrutiny_pass"] == 1 and "4x_median" in flagged.loc["s5", "rules"]
    # Compared with the other six, without the one that hid it.
    assert flagged.loc["s6", "scrutiny_pass"] == 2
    assert flagged.loc["s6", "count"] == 7
    assert flagged.loc["s6", "rules"] == "0.35x_median"
    assert flagged.loc["s6", "anomaly_2"] == True  # noqa: E712 - low price, low price per area
    # Later passes use the ratio thresholds only: nothing ordinary is set aside.
    assert len(flagged) == 2


def test_nothing_flagged_keeps_every_sale():
    df = _cluster([100, 105, 95, 102, 98, 101])
    out = _calc_sales_scrutiny(df, "sale_price_impr_sqft", passes=2)
    assert len(out) == 6 and not out["flagged"].any()


def _sales():
    universe = pd.DataFrame({"key": [f"p{i}" for i in range(4)], "bldg_year_built": [1990] * 4})
    sales = pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(4)],
            "key_sale": [f"s{i}" for i in range(4)],
            "sale_date": pd.to_datetime(["2020-01-02"] * 4),
            "sale_year": [2020] * 4,
            "sale_price": [100_000, 100_000, 300_000, 400_000],
            "valid_sale": [True] * 4,
            "vacant_sale": [False] * 4,
        }
    )
    return SalesUniversePair(sales=sales, universe=universe)


def test_kept_sales_are_flagged_but_not_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("in")
    pd.DataFrame({"key_sale": ["s1"]}).to_csv("in/keep.csv", index=False)
    settings = {"analysis": {"sales_scrutiny": {"keep_file": "in/keep.csv"}}, "locality": {"units": "imperial"}}
    assert read_keep_keys(settings) == {"s1"}

    out = run_heuristics(_sales(), settings, drop=True)
    # s0 and s1 sold on one day for one price: both flagged, s1 kept.
    assert sorted(out.sales["key_sale"]) == ["s1", "s2", "s3"]
    flags = pd.read_parquet("out/sales_scrutiny/heuristics/flags.parquet")
    assert sorted(flags["key_sale"]) == ["s0", "s1"]
    assert set(flags["check"]) == {"dupe_date_price"}


def test_drops_are_settings():
    assert get_sales_scrutiny_drops({}) == (False, True)
    assert get_sales_scrutiny_drops({}, clusters=True) == (True, True)
    s = {"analysis": {"sales_scrutiny": {"drop": {"clusters": True, "heuristics": False}}}}
    assert get_sales_scrutiny_drops(s) == (True, False)


def test_filled_values_are_recorded(tmp_path):
    before = pd.DataFrame({"key": ["a", "b", "c"], "x": [1.0, None, 3.0], "y": [None, None, None], "z": [1, 2, 3]})
    blank = {c: before[c].isna() for c in before if before[c].hasnans}
    after = before.assign(x=[1.0, 2.0, 3.0])  # y left blank: no fill for it
    out = write_filled(after, blank, path=str(tmp_path / "filled.parquet"))
    assert out.columns.tolist() == ["key", "x"]
    assert out["x"].tolist() == [False, True, False]
    assert pd.read_parquet(tmp_path / "filled.parquet").equals(out)


def test_each_heuristic_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = {
        "analysis": {"sales_scrutiny": {"heuristics": {"dupe_date_price": False}}},
        "locality": {"units": "imperial"},
    }
    out = run_heuristics(_sales(), settings, drop=True)
    assert sorted(out.sales["key_sale"]) == ["s0", "s1", "s2", "s3"]
