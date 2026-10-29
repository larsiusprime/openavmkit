"""Sales scrutiny as a diagnosis: the five patterns read strictly (price, price per area and size each low, in range or
high), sales that fit none flagged only, and the checks for misclassified vacancy and sales between family."""

import pandas as pd
import pytest

from openavmkit.data import SalesUniversePair
from openavmkit.sales_scrutiny_study import (
    _calc_sales_scrutiny,
    _flag_rows,
    _strict_patterns,
    run_heuristics,
    surnames,
)


def _cluster(extra: list[tuple[float, float]]) -> pd.DataFrame:
    """Twelve ordinary sales, 1,500 to 2,600 sq ft at about $200 a sq ft, and the ones to diagnose (area, price)."""
    rows = [(1500 + 100 * i, (1500 + 100 * i) * (198 if i % 2 else 202)) for i in range(12)] + extra
    n = len(rows)
    return pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(n)],
            "key_sale": [f"s{i}" for i in range(n)],
            "ss_id": ["c"] * n,
            "bldg_area_finished_sqft": [a for a, _ in rows],
            "sale_price": [p for _, p in rows],
            "sale_price_impr_sqft": [p / a for a, p in rows],
        }
    )


def _diagnosis(extra, strict=True) -> pd.Series:
    out = _calc_sales_scrutiny(_cluster(extra), "sale_price_impr_sqft", strict=strict)
    flagged = out[out["flagged"]]
    assert flagged["key_sale"].tolist()[:1] == ["s12"]
    return flagged.iloc[0]


@pytest.mark.parametrize(
    "sale, pattern, readings",
    [
        ((600, 410_000), 1, ("in", "high", "low")),  # the size is wrong, most likely
        ((2000, 150_000), 2, ("low", "low", "in")),  # a low price: not a market sale?
        ((2000, 900_000), 3, ("high", "high", "in")),  # a high price: more than this parcel?
        ((1700, 400_000), 4, ("in", "high", "in")),  # an ordinary price, high for its size
        ((2400, 330_000), 5, ("in", "low", "in")),  # an ordinary price, low for its size
    ],
)
def test_each_flagged_sale_fits_one_pattern(sale, pattern, readings):
    row = _diagnosis([sale])
    assert (row["price_rating"], row["per_area_rating"], row["size_rating"]) == readings
    assert [i for i in range(1, 6) if row[f"anomaly_{i}"] == True]  # noqa: E712 == [pattern]
    assert row["anomaly_other"] == False  # noqa: E712
    assert row["size"] == sale[0] and row["price_median"] > 0 and row["size_median"] > 0


def test_the_classic_reading_overlaps():
    # Classic: "low price" is any price under the median, and size is always "in range": two patterns at once.
    row = _diagnosis([(600, 410_000)], strict=False)
    assert sum(row[f"anomaly_{i}"] == True for  # noqa: E712
 i in range(1, 6)) > 1


def test_a_small_clusters_outlier_reads_low_by_its_ratio():
    # Seven sales: a $10 deed stretches the spread so far that it isn't 2 deviations below anything. It's still
    # low: 0.35x the median or less, as it was flagged.
    # (The sample's own: 320 S Mendenhall St, sold for $12 among houses at $135 to $358 a sq ft.)
    rows = list(zip([3285, 2610, 2925, 3462, 4115, 3354], [135, 202, 230, 276, 306, 358], strict=True))
    rows = [(a, a * p) for a, p in rows]
    df = pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(7)],
            "key_sale": [f"s{i}" for i in range(7)],
            "ss_id": ["c"] * 7,
            "bldg_area_finished_sqft": [a for a, _ in rows] + [2968],
            "sale_price": [p for _, p in rows] + [12],
            "sale_price_impr_sqft": [p / a for a, p in rows] + [12 / 2968],
        }
    )
    out = _calc_sales_scrutiny(df, "sale_price_impr_sqft", strict=True)
    row = out[out["key_sale"] == "s6"].iloc[0]
    assert row["flagged"] and abs(row["price_dist"]) < 2
    assert (row["price_rating"], row["per_area_rating"], row["size_rating"]) == ("low", "low", "in")
    assert row["anomaly_2"] == True and row["anomaly_other"] == False  # noqa: E712


def test_a_sale_that_fits_no_pattern_is_flagged_only():
    # A low price but a high price per area, its size in range (a wide spread of sizes): none of the five.
    r = lambda *x: pd.Series(list(x))  # noqa: E731
    patterns = _strict_patterns(r("low", "high", "in"), r("high", "low", "high"), r("in", "in", "high"))
    assert [k for k, v in patterns.items() if v[0]] == []
    assert [k for k, v in patterns.items() if v[2]] == ["anomaly_1"]
    # Flag rows say "other" and flag_only for such a sale; it's never dropped.
    out = _calc_sales_scrutiny(_cluster([(1700, 400_000)]), "sale_price_impr_sqft", strict=True)
    out.loc[out["flagged"], ["anomaly_4", "anomaly_other"]] = [False, True]
    out["vacant"], out["area"], out["sale_price_time_adj"] = False, "impr_sqft", out["sale_price"]
    flags = _flag_rows([out], "g")
    assert flags["pattern"].tolist() == ["other"] and flags["flag_only"].all()


@pytest.mark.parametrize(
    "name, pattern, expected",
    [
        ("SMITH JOHN A", None, ["SMITH"]),
        ("SMITH JOHN & MARY", None, ["SMITH"]),  # Mary is a Smith too
        ("SMITH JOHN & JONES MARY", None, ["SMITH", "JONES"]),
        ("SMITH JOHN AND MARY JONES", None, ["SMITH", "MARY"]),  # surname first: read as written
        ("JOHN & MARY SMITH", r"(?P<surname>[A-Za-z'\-]+)\s*$", ["SMITH"]),  # Mary is a Smith too
        ("SMITH", None, ["SMITH"]),
        ("SMITH JOHN & ACME LLC", None, []),
        ("O'BRIEN-KELLY PAT", None, ["O'BRIEN-KELLY"]),
        ("SMITH FAMILY TRUST", None, []),  # not a person
        ("ACME HOLDINGS LLC", None, []),
        ("", None, []),
        (None, None, []),
        ("Smith, John A.", r"^\s*(?P<surname>[^,]+),", ["SMITH"]),
        ("John A. Smith Jr", r"(?P<surname>[A-Za-z'\-]+)\s*(?:,?\s*(?:JR|SR|II|III|IV)\.?)?\s*$", ["SMITH"]),
    ],
)
def test_surnames(name, pattern, expected):
    assert (surnames(name, pattern) if pattern else surnames(name)) == expected


def _sup():
    universe = pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(8)],
            "bldg_year_built": [1990, 2023, 1990, 1990, 1990, 0, 0, 0],
            "neighborhood": ["a"] * 7 + ["b"],
        }
    )
    sales = pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(8)],
            "key_sale": [f"s{i}" for i in range(8)],
            "sale_date": pd.to_datetime([f"2021-0{i + 1}-02" for i in range(8)]),
            "sale_year": [2021] * 8,
            "sale_price": [300_000, 40_000, 310_000, 290_000, 305_000, 60_000, 200_000, 150_000],
            "valid_sale": [True] * 8,
            "vacant_sale": [False, False, False, False, False, True, True, True],
            "deed_grantor": ["SMITH JOHN", "A", "B", "LEE ANN & BOB", "SMITH TRUST", "X", "Y", "Z"],
            "deed_grantee": ["SMITH MARY", "C", "D", "PARK JO & LEE KIM", "SMITH JANE", "W", "V", "U"],
        }
    )
    return SalesUniversePair(sales=sales, universe=universe)


def _settings(**heuristics):
    return {
        "analysis": {
            "sales_scrutiny": {
                "location": "neighborhood",
                "heuristics": {"dupe_date_price": False, **heuristics},
                "drop": {"heuristics": True},
            }
        },
        "locality": {"units": "imperial"},
    }


def test_the_newer_checks_are_off_unless_turned_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = run_heuristics(_sup(), _settings(), drop=True)
    assert len(out.sales) == 8


def test_built_after_the_sale_vacant_priced_improved_and_same_surname(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    on = {"built_after_sale": True, "vacant_priced_improved": True, "same_surname": True}
    out = run_heuristics(_sup(), _settings(**on), drop=True)
    flags = pd.read_parquet("out/sales_scrutiny/heuristics/flags.parquet").set_index("key_sale")
    # s1: sold as improved in 2021, its building built in 2023: land when sold.
    assert flags.loc["s1", "check"] == "built_after_sale" and flags.loc["s1", "detail"] == "built 2023"
    # s6: vacant, at 200,000 two thirds of neighborhood a's median improved sale (300,000). s5 at 60,000 isn't;
    # s7's neighborhood has no improved sales to compare with.
    assert flags.loc["s6", "check"] == "vacant_priced_improved" and "67%" in flags.loc["s6", "detail"]
    # s0 Smith to Smith; s3 Lee to (Park and) Lee. s4's seller is a trust, not a person.
    surname = flags[flags["check"] == "same_surname"]
    assert dict(surname["detail"]) == {"s0": "SMITH", "s3": "LEE"}
    assert sorted(set(flags.index)) == ["s0", "s1", "s3", "s6"]
    assert not flags["flag_only"].any()
    assert sorted(out.sales["key_sale"]) == ["s2", "s4", "s5", "s7"]


def test_surname_fields_win_over_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sup = _sup()
    sup.sales["deed_grantor_surname"] = [None, "OKAFOR", None, None, None, None, None, None]
    sup.sales["deed_grantee_surname"] = [None, "okafor", None, None, None, None, None, None]
    run_heuristics(sup, _settings(same_surname=True), drop=False)
    flags = pd.read_parquet("out/sales_scrutiny/heuristics/flags.parquet")
    assert sorted(flags["key_sale"]) == ["s0", "s1", "s3"]


def test_nominal_prices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sup = _sup()
    sup.sales.loc[2, "sale_price"] = 10
    sup.sales.loc[3, "sale_price"] = 999
    run_heuristics(sup, _settings(nominal_price=True), drop=False)
    flags = pd.read_parquet("out/sales_scrutiny/heuristics/flags.parquet")
    nominal = flags[flags["check"] == "nominal_price"].set_index("key_sale")
    assert sorted(nominal.index) == ["s2", "s3"] and nominal.loc["s2", "detail"] == "10"
    s = _settings(nominal_price=True)
    s["analysis"]["sales_scrutiny"]["nominal_price"] = {"below": 100}
    run_heuristics(sup, s, drop=False)
    flags = pd.read_parquet("out/sales_scrutiny/heuristics/flags.parquet")
    assert sorted(flags.loc[flags["check"] == "nominal_price", "key_sale"]) == ["s2"]
