"""Tests for the duplicate-detection heuristic ``flag_dupe_date_price``.

Regression coverage for ``flag_dupe_date_price``: distinct parcels conveyed in a
single multi-parcel deed share a sale date and price but are NOT duplicates and
must be kept, while a genuine same-parcel repeat must still be flagged.
"""
import pandas as pd

from openavmkit.utilities.sales_scrutiny import flag_dupe_date_price


def _sale(key, key_sale, date, price):
    return {
        "key": key,
        "key_sale": key_sale,
        "sale_date": date,
        "sale_price": price,
    }


def _df(sales_rows):
    return pd.DataFrame(sales_rows)


def _flagged_sales(df):
    if "flag_dupe_date_price" not in df.columns:
        return set()
    return set(df.loc[df["flag_dupe_date_price"].eq(True), "key_sale"])


def test_dupe_date_price_keeps_distinct_parcels_in_one_deed():
    # Three DISTINCT parcels conveyed in a single multi-parcel deed: same date,
    # same (deed-total) price. These are not duplicate reports and must not be
    # flagged.
    df = _df(
        [
            _sale("p1", "s1", "2020-01-01", 75000),
            _sale("p2", "s2", "2020-01-01", 75000),
            _sale("p3", "s3", "2020-01-01", 75000),
        ]
    )
    out = flag_dupe_date_price(df)
    assert _flagged_sales(out) == set()


def test_dupe_date_price_still_flags_same_parcel_repeat():
    # The SAME parcel reported twice at the same date and price is a genuine
    # duplicate report and must still be flagged; the distinct-parcel deed lots
    # alongside it must not be.
    df = _df(
        [
            _sale("p1", "s1", "2020-01-01", 75000),
            _sale("p2", "s2", "2020-01-01", 75000),
            _sale("p5", "s4", "2021-01-01", 50000),
            _sale("p5", "s5", "2021-01-01", 50000),
        ]
    )
    out = flag_dupe_date_price(df)
    assert _flagged_sales(out) == {"s4", "s5"}


def test_dupe_date_price_keeps_one_parcel_sold_twice_on_different_dates():
    # One parcel with two legitimate sales at DIFFERENT dates is not a duplicate
    # and neither row must be flagged.
    df = _df(
        [
            _sale("p1", "s1", "2018-05-01", 40000),
            _sale("p1", "s2", "2022-09-01", 60000),
        ]
    )
    out = flag_dupe_date_price(df)
    assert _flagged_sales(out) == set()


def test_dupe_date_price_with_jurisdiction_keeps_distinct_parcels():
    # The jurisdiction-scoped branch must still keep distinct parcels that share
    # a date and price, and still flag a true same-parcel repeat.
    sales = [
        _sale("p1", "s1", "2020-01-01", 75000),
        _sale("p2", "s2", "2020-01-01", 75000),
        _sale("p3", "s3", "2021-01-01", 50000),
        _sale("p3", "s4", "2021-01-01", 50000),
    ]
    for row in sales:
        row["county"] = "Acme"
    df = _df(sales)
    out = flag_dupe_date_price(df, jurisdiction="county")
    assert _flagged_sales(out) == {"s3", "s4"}


def test_dupe_date_price_without_parcel_key_falls_back_to_sale_key():
    df = pd.DataFrame(
        {
            "key_sale": ["s1", "s2", "s3", "s3"],
            "sale_date": ["2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01"],
            "sale_price": [75000, 75000, 50000, 50000],
        }
    )

    out = flag_dupe_date_price(df)

    assert _flagged_sales(out) == {"s3"}


def test_dupe_date_price_null_parcel_key_falls_back_to_sale_key():
    df = _df(
        [
            _sale(None, "s1", "2020-01-01", 75000),
            _sale(None, "s2", "2020-01-01", 75000),
            _sale(None, "s3", "2021-01-01", 50000),
            _sale(None, "s3", "2021-01-01", 50000),
        ]
    )

    out = flag_dupe_date_price(df)

    assert _flagged_sales(out) == {"s3"}


def test_dupe_date_price_blank_parcel_key_falls_back_to_sale_key():
    df = _df(
        [
            _sale("", "s1", "2020-01-01", 75000),
            _sale(" ", "s2", "2020-01-01", 75000),
            _sale("", "s3", "2021-01-01", 50000),
            _sale("", "s3", "2021-01-01", 50000),
        ]
    )

    out = flag_dupe_date_price(df)

    assert _flagged_sales(out) == {"s3"}
