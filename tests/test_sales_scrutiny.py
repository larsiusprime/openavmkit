"""Tests for sales-scrutiny heuristics."""

import pandas as pd

from openavmkit.sales_scrutiny_study import flag_bulk_deeds


def test_flag_bulk_deeds_precomputed_count_catches_orphan():
    """When ``sale_parcel_count`` is present (computed at ingestion before the sales
    table is thinned to one row per parcel), a lone surviving "orphan" row of a
    multi-parcel deed is still flagged -- the case the duplicate-based heuristics
    miss once the sibling rows are dropped."""
    df = pd.DataFrame(
        {
            "key": ["A", "B", "C"],
            "key_sale": ["deed1", "deed2", "deed3"],
            # deed1 covered 15 parcels but only this orphan row survived dedup;
            # deed2 was a normal single-parcel sale; deed3 a 2-parcel bundle.
            "sale_parcel_count": [15, 1, 2],
        }
    )
    flagged = flag_bulk_deeds(df)
    assert flagged.tolist() == [True, False, True]


def test_flag_bulk_deeds_fallback_counts_parcels_per_deed():
    """Without the precomputed column, fall back to counting distinct parcels per
    deed in the current table (catches bulk deeds whose duplicate rows survive)."""
    df = pd.DataFrame(
        {
            "key": ["A", "B", "C", "D"],
            # deed1 recorded against 3 parcels; deed2 a single-parcel sale.
            "key_sale": ["deed1", "deed1", "deed1", "deed2"],
        }
    )
    flagged = flag_bulk_deeds(df)
    assert flagged.tolist() == [True, True, True, False]


def test_flag_bulk_deeds_precomputed_takes_precedence_over_fallback():
    """The precomputed count wins even when per-table parcel counts disagree
    (e.g. all sibling rows were dropped, so the table shows one row per deed)."""
    df = pd.DataFrame(
        {
            "key": ["A", "B"],
            "key_sale": ["deed1", "deed2"],  # each appears once in the table
            "sale_parcel_count": [8, 1],      # ...but deed1 truly spanned 8 parcels
        }
    )
    flagged = flag_bulk_deeds(df)
    assert flagged.tolist() == [True, False]


def test_flag_bulk_deeds_no_usable_columns_returns_all_false():
    """No ``sale_parcel_count`` and no ``key`` -> nothing can be inferred."""
    df = pd.DataFrame({"key_sale": ["deed1", "deed2"]})
    flagged = flag_bulk_deeds(df)
    assert flagged.tolist() == [False, False]
    assert list(flagged.index) == list(df.index)


def test_flag_bulk_deeds_handles_na_count():
    """A NaN ``sale_parcel_count`` (deed never seen at ingestion) is treated as 1."""
    df = pd.DataFrame(
        {
            "key": ["A", "B"],
            "key_sale": ["deed1", "deed2"],
            "sale_parcel_count": pd.array([pd.NA, 3], dtype="Int64"),
        }
    )
    flagged = flag_bulk_deeds(df)
    assert flagged.tolist() == [False, True]
