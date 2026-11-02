"""Multi-parcel ("bulk") deed heuristics in run_heuristics.

A bulk deed stamps one bundle price onto every parcel it covers, so those sales must be
dropped. Unrelated sales on separate, known deeds that merely share a date and price must not
be. The end-to-end cases run through ``run_heuristics`` the way ``process_data`` sets things up:
``sale_parcel_count`` is counted before invalid sales are filtered out.
"""
import pandas as pd
import pytest

from openavmkit.data import SalesUniversePair
from openavmkit.sales_scrutiny_study import run_heuristics
from openavmkit.utilities.sales_scrutiny import (
    SALE_PARCEL_COUNT,
    count_parcels_per_deed,
    deed_group_ids,
    flag_bulk_deeds,
    flag_dupe_date_price,
    flag_dupe_deed_date,
)

BUNDLE = 300_000  # one deed-total price stamped on every parcel of a bulk deed


def _sale(key, date, price, deed, valid=True, key_sale=None):
    return {
        "key": key,
        # The documented convention: key_sale = parcel id + sale date (one row per parcel).
        "key_sale": key_sale if key_sale is not None else f"{key}---{date}",
        "sale_date": pd.Timestamp(date),
        "sale_price": price,
        "deed_num": deed,
        "valid_sale": valid,
        "vacant_sale": 1,
        "sale_year": pd.Timestamp(date).year,
        "bldg_year_built": 0,
    }


def _run(rows, use_deed_id):
    """Mimic process_data (count parcels per deed, then drop invalid sales), then run_heuristics."""
    deed_id = "deed_num" if use_deed_id else None
    sales = pd.DataFrame(rows)
    counts = count_parcels_per_deed(sales, deed_id)
    if counts is not None:
        sales[SALE_PARCEL_COUNT] = counts
    sales = sales[sales["valid_sale"]].reset_index(drop=True)
    universe = pd.DataFrame({"key": sorted(set(r["key"] for r in rows)), "neighborhood": "N1"})
    ss = {"location": "neighborhood"}
    if deed_id:
        ss["deed_id"] = deed_id
    sup = run_heuristics(
        SalesUniversePair(universe=universe, sales=sales),
        {"analysis": {"sales_scrutiny": ss}},
        drop=True,
    )
    return set(sales["key"]) - set(sup.sales["key"])


BULK_3 = [
    _sale("p1", "2020-01-01", BUNDLE, "D1"),
    _sale("p2", "2020-01-01", BUNDLE, "D1"),
    _sale("p3", "2020-01-01", BUNDLE, "D1"),
]
ORPHAN = [
    _sale("p1", "2020-01-01", BUNDLE, "D1"),
    _sale("p2", "2020-01-01", BUNDLE, "D1", valid=False),
    _sale("p3", "2020-01-01", BUNDLE, "D1", valid=False),
]


@pytest.mark.parametrize(
    "rows, use_deed_id, expected_dropped",
    [
        pytest.param(BULK_3, True, {"p1", "p2", "p3"}, id="bulk deed, deed_id set"),
        # Without deed IDs, repeated date+price is the only bulk-deed signal: it must still fire.
        pytest.param(BULK_3, False, {"p1", "p2", "p3"}, id="bulk deed, no deed_id"),
        pytest.param(
            [_sale(k, "2020-01-01", BUNDLE, "D1", key_sale="D1") for k in ("p1", "p2", "p3")],
            False, {"p1", "p2", "p3"}, id="bulk deed, key_sale is the deed number",
        ),
        pytest.param(ORPHAN, True, {"p1"}, id="orphan bulk deed, deed_id set"),
        pytest.param(
            [dict(r, key_sale="D1") for r in ORPHAN], False, {"p1"},
            id="orphan bulk deed, key_sale is the deed number",
        ),
        pytest.param(
            [_sale("q1", "2020-03-02", 250_000, "D7"), _sale("q2", "2020-03-02", 250_000, "D8")],
            True, set(), id="coincidence on separate deeds is kept",
        ),
        pytest.param(
            [_sale(f"l{i}", "2020-05-05", 80_000, f"D2{i}") for i in range(3)],
            True, set(), id="builder lots on separate deeds are kept",
        ),
        pytest.param(
            [
                _sale("g1", "2020-07-07", BUNDLE, "D1"),
                _sale("g2", "2020-07-07", BUNDLE, "D1"),
                _sale("g3", "2020-07-07", BUNDLE, "D2"),
            ],
            True, {"g1", "g2"}, id="bulk deed plus unrelated coincidence",
        ),
        pytest.param(
            [_sale("h1", "2020-08-08", 120_000, "D31"), _sale("h2", "2020-08-08", 120_000, None)],
            True, {"h1", "h2"}, id="missing deed in a date+price group stays suspicious",
        ),
        pytest.param(
            [_sale("i1", "2020-09-09", 100_000, None), _sale("i2", "2020-09-09", 200_000, None)],
            True, set(), id="missing deeds are not one shared deed",
        ),
        pytest.param(
            [
                _sale("z1", "2021-01-01", 50_000, "D9", key_sale="z1---a"),
                _sale("z1", "2021-01-01", 50_000, "D9", key_sale="z1---b"),
            ],
            False, {"z1"}, id="same parcel recorded twice",
        ),
        pytest.param(
            [_sale("s1", "2019-01-01", 90_000, "D40"), _sale("s1", "2022-06-01", 140_000, "D41")],
            True, set(), id="ordinary resale is kept",
        ),
    ],
)
def test_run_heuristics_bulk_deed_scenarios(tmp_path, monkeypatch, rows, use_deed_id, expected_dropped):
    monkeypatch.chdir(tmp_path)  # run_heuristics writes its reports to ./out
    assert _run(rows, use_deed_id) == expected_dropped


def test_bulk_deed_report_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(ORPHAN, True)
    assert (tmp_path / "out" / "sales_scrutiny" / "multi_parcel_bulk_deeds.xlsx").exists()


def _df(rows):
    return pd.DataFrame(rows)


def test_deed_group_ids_uses_deed_and_date_and_masks_missing():
    df = _df([
        _sale("a", "2020-01-01", 1, "D1"),
        _sale("b", "2020-01-01", 1, "D1"),
        _sale("c", "2020-02-01", 1, "D1"),  # same deed number, different date
        _sale("d", "2020-01-01", 1, None),
        _sale("e", "2020-01-01", 1, "  "),
    ])
    ids = deed_group_ids(df, "deed_num")
    assert ids[0] == ids[1] != ids[2]
    assert ids[3:].isna().all()


def test_deed_group_ids_uses_jurisdiction_only_when_present():
    df = _df([_sale("a", "2020-01-01", 1, "D1"), _sale("b", "2020-01-01", 1, "D1")])
    df["city"] = ["X", "Y"]
    assert deed_group_ids(df, "deed_num", "city").nunique() == 2
    assert deed_group_ids(df, "deed_num", "not_a_column").nunique() == 1


def test_deed_group_ids_falls_back_to_key_sale():
    df = _df([_sale("a", "2020-01-01", 1, "D1")])
    assert deed_group_ids(df, None).tolist() == ["a---2020-01-01"]
    assert deed_group_ids(df.drop(columns="key_sale"), None) is None


def test_count_parcels_per_deed_counts_distinct_parcels():
    df = _df([
        _sale("a", "2020-01-01", 1, "D1"),
        _sale("b", "2020-01-01", 1, "D1"),
        _sale("b", "2020-01-01", 1, "D1", key_sale="b---dupe"),  # same parcel again
        _sale("c", "2020-01-01", 1, None),
    ])
    counts = count_parcels_per_deed(df, "deed_num")
    assert str(counts.dtype) == "Int64"
    assert counts[:3].tolist() == [2, 2, 2]
    assert counts.isna()[3]
    assert count_parcels_per_deed(df.drop(columns="key"), "deed_num") is None


def test_flag_dupe_deed_date_ignores_missing_deeds():
    df = _df([
        _sale("a", "2020-01-01", 1, "D1"),
        _sale("b", "2020-01-01", 2, "D1"),
        _sale("c", "2020-01-01", 3, None),
        _sale("d", "2020-01-01", 4, None),
    ])
    assert flag_dupe_deed_date(df, "deed_num").tolist() == [True, True, False, False]


def test_flag_dupe_date_price_without_deed_ids_flags_every_repeat():
    df = _df([
        _sale("a", "2020-01-01", 250_000, "D7"),
        _sale("b", "2020-01-01", 250_000, "D8"),
        _sale("c", "2020-01-01", 260_000, "D9"),
    ])
    assert flag_dupe_date_price(df).tolist() == [True, True, False]
    # With deed IDs the separate-deed coincidence is spared.
    assert flag_dupe_date_price(df, deed_id="deed_num").tolist() == [False, False, False]


def test_flag_dupe_date_price_respects_jurisdiction():
    df = _df([_sale("a", "2020-01-01", 250_000, "D7"), _sale("b", "2020-01-01", 250_000, "D8")])
    df["city"] = ["X", "Y"]
    assert flag_dupe_date_price(df, jurisdiction="city").tolist() == [False, False]


def test_flag_bulk_deeds():
    df = pd.DataFrame({SALE_PARCEL_COUNT: pd.array([1, 3, pd.NA], dtype="Int64")})
    assert flag_bulk_deeds(df).tolist() == [False, True, False]
    assert flag_bulk_deeds(pd.DataFrame({"x": [1, 2]})).tolist() == [False, False]
