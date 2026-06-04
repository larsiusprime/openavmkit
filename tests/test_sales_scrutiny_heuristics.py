"""Tests for the duplicate-detection heuristics in ``run_heuristics``.

Regression coverage for the ``flag_dupe_date_price`` heuristic: distinct parcels
conveyed in a single multi-parcel deed share a sale date and price but are NOT
duplicates and must be kept, while a genuine same-parcel repeat must still be
dropped.
"""
import pandas as pd

from openavmkit.data import SalesUniversePair
from openavmkit.sales_scrutiny_study import run_heuristics


SETTINGS = {"analysis": {"sales_scrutiny": {}}}


def _sale(key, key_sale, date, price):
    return {
        "key": key,
        "key_sale": key_sale,
        "sale_date": date,
        "sale_price": price,
        "sale_year": int(date[:4]),
        "vacant_sale": False,
        "bldg_year_built": 0,
    }


def _make_sup(sales_rows):
    sales = pd.DataFrame(sales_rows)
    keys = sorted(set(sales["key"]))
    universe = pd.DataFrame({"key": keys, "is_vacant": [False] * len(keys)})
    return SalesUniversePair(sales=sales, universe=universe)


def test_dupe_date_price_keeps_distinct_parcels_in_one_deed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Three DISTINCT parcels conveyed in a single multi-parcel deed: same date,
    # same (deed-total) price. These are not duplicate reports and must survive.
    sup = _make_sup(
        [
            _sale("p1", "s1", "2020-01-01", 75000),
            _sale("p2", "s2", "2020-01-01", 75000),
            _sale("p3", "s3", "2020-01-01", 75000),
        ]
    )
    out = run_heuristics(sup, SETTINGS, drop=True)
    assert set(out.sales["key_sale"]) == {"s1", "s2", "s3"}


def test_dupe_date_price_still_flags_same_parcel_repeat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The SAME parcel reported twice at the same date and price is a genuine
    # duplicate report and must still be dropped; the distinct-parcel deed lots
    # alongside it must be kept.
    sup = _make_sup(
        [
            _sale("p1", "s1", "2020-01-01", 75000),
            _sale("p2", "s2", "2020-01-01", 75000),
            _sale("p5", "s4", "2021-01-01", 50000),
            _sale("p5", "s5", "2021-01-01", 50000),
        ]
    )
    out = run_heuristics(sup, SETTINGS, drop=True)
    survivors = set(out.sales["key_sale"])
    assert {"s1", "s2"}.issubset(survivors)
    assert survivors.isdisjoint({"s4", "s5"})


def test_dupe_date_price_keeps_one_parcel_sold_twice_on_different_dates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # One parcel with two legitimate sales at DIFFERENT dates is not a duplicate
    # and both rows must be kept.
    sup = _make_sup(
        [
            _sale("p1", "s1", "2018-05-01", 40000),
            _sale("p1", "s2", "2022-09-01", 60000),
        ]
    )
    out = run_heuristics(sup, SETTINGS, drop=True)
    assert set(out.sales["key_sale"]) == {"s1", "s2"}


def test_dupe_date_price_with_jurisdiction_keeps_distinct_parcels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The jurisdiction-scoped branch must still keep distinct parcels that share
    # a date and price, and still drop a true same-parcel repeat.
    sales = [
        _sale("p1", "s1", "2020-01-01", 75000),
        _sale("p2", "s2", "2020-01-01", 75000),
        _sale("p3", "s3", "2021-01-01", 50000),
        _sale("p3", "s4", "2021-01-01", 50000),
    ]
    for row in sales:
        row["county"] = "Acme"
    sup = _make_sup(sales)
    settings = {"analysis": {"sales_scrutiny": {"jurisdiction": "county"}}}
    out = run_heuristics(sup, settings, drop=True)
    survivors = set(out.sales["key_sale"])
    assert {"s1", "s2"}.issubset(survivors)
    assert survivors.isdisjoint({"s3", "s4"})
