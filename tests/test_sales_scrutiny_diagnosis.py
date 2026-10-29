"""Sales scrutiny as a diagnosis: the five patterns read strictly (price, price per area and size each low, in range or
high), sales that fit none flagged only, and the checks for misclassified vacancy and sales between family."""

import pandas as pd
import pytest

from openavmkit.data import SalesUniversePair
from openavmkit.sales_scrutiny_study import _calc_sales_scrutiny, _flag_rows, run_heuristics, surnames


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


def test_a_sale_that_fits_no_pattern_is_flagged_only():
    # Eight sales at 4.5x the median: each crosses the 4x rule, but so many stretch the spread that none is 2
    # deviations out. Flagged ("other"), never dropped: the user decides.
    out = _calc_sales_scrutiny(_cluster([(2000, 2000 * 900)] * 8), "sale_price_impr_sqft", strict=True)
    row = out[out["flagged"]].iloc[0]
    assert row["anomaly_other"] == True and row["per_area_rating"] == "in"  # noqa: E712
    out["vacant"], out["area"], out["sale_price_time_adj"] = False, "impr_sqft", out["sale_price"]
    flags = _flag_rows([out.assign(sale_price_impr_sqft=out["sale_price_impr_sqft"])], "g")
    assert set(flags["pattern"]) == {"other"} and flags["flag_only"].all()
    # Classic reads them as pattern 4 (its size test always passes) and drops them with the rest.
    classic = _calc_sales_scrutiny(_cluster([(2000, 2000 * 900)] * 8), "sale_price_impr_sqft")
    assert classic["flagged"].sum() == 8 and not classic["anomaly_other"].any()


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
