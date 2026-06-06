"""Tests for per-location area-statistic enrichment (openavmkit.area_stats)."""
import numpy as np
import pandas as pd

import openavmkit.area_stats as area_stats
from openavmkit.area_stats import enrich_sup_area_stats
from openavmkit.data import SalesUniversePair
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_fields_numeric,
)


def _base_settings(area_cfg: dict) -> dict:
    """Settings with a minimal field classification plus an area_stats config block."""
    return {
        "locality": {"name": "Testville", "slug": "test"},
        "data": {"process": {"enrich": {"area_stats": area_cfg}}},
        "field_classification": {
            "impr": {
                "numeric": ["bldg_area_finished_sqft"],
                "categorical": [],
                "boolean": [],
            },
            "land": {"numeric": [], "categorical": ["zoning"], "boolean": []},
            "other": {"numeric": [], "categorical": [], "boolean": []},
        },
    }


def _characteristic_sup() -> SalesUniversePair:
    """A small universe (6 parcels, 3 neighborhoods) plus two sales."""
    universe = pd.DataFrame(
        {
            "key": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "neighborhood": ["A", "A", "A", "B", "B", "C"],
            "bldg_area_finished_sqft": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "zoning": ["R1", "R1", "R2", "R1", "R1", "R2"],
        }
    )
    sales = pd.DataFrame(
        {
            "key": ["u1", "u4"],
            "key_sale": ["s1", "s4"],
        }
    )
    return SalesUniversePair(sales, universe)


def test_numeric_mean_groupby_and_min_count():
    """mean matches a hand-computed group mean; groups below min_count become NaN."""
    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["bldg_area_finished_sqft"],
            "stats": ["mean"],
            "min_count": 2,
        }
    )
    out = enrich_sup_area_stats(_characteristic_sup(), settings)
    col = "area_stat_neighborhood_bldg_area_finished_sqft_mean"
    by_key = out.universe.set_index("key")[col]

    # A = mean(100,200,300)=200 ; B = mean(400,500)=450 ; C has 1 parcel -> NaN
    assert by_key["u1"] == 200.0
    assert by_key["u4"] == 450.0
    assert pd.isna(by_key["u6"])

    # count column is always emitted and is NOT masked by min_count
    counts = out.universe.set_index("key")["area_stat_neighborhood_count"]
    assert counts["u1"] == 3
    assert counts["u4"] == 2
    assert counts["u6"] == 1


def test_features_propagate_to_sales():
    """Universe-level features are stamped onto sales by parcel key."""
    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["bldg_area_finished_sqft"],
            "stats": ["mean"],
            "min_count": 1,
        }
    )
    out = enrich_sup_area_stats(_characteristic_sup(), settings)
    col = "area_stat_neighborhood_bldg_area_finished_sqft_mean"
    sales_by_key = out.sales.set_index("key")[col]
    assert sales_by_key["u1"] == 200.0  # parcel u1 is in neighborhood A
    assert sales_by_key["u4"] == 450.0  # parcel u4 is in neighborhood B


def test_categorical_mode_and_mode_frac():
    """mode returns the dominant category; mode_frac returns its share (numeric)."""
    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["zoning"],
            "categorical_stats": ["mode", "mode_frac"],
            "min_count": 1,
        }
    )
    out = enrich_sup_area_stats(_characteristic_sup(), settings)
    mode = out.universe.set_index("key")["area_stat_neighborhood_zoning_mode"]
    frac = out.universe.set_index("key")["area_stat_neighborhood_zoning_mode_frac"]

    # Neighborhood A: zoning R1,R1,R2 -> mode R1, frac 2/3
    assert mode["u1"] == "R1"
    assert frac["u1"] == 2.0 / 3.0
    # Neighborhood B: zoning R1,R1 -> mode R1, frac 1.0
    assert mode["u4"] == "R1"
    assert frac["u4"] == 1.0


def test_sale_derived_uses_train_valid_only(monkeypatch):
    """Sale-derived fields aggregate over training valid sales only."""
    universe = pd.DataFrame(
        {"key": ["u1", "u2", "u3", "u4"], "neighborhood": ["A", "A", "A", "A"]}
    )
    sales = pd.DataFrame(
        {
            "key": ["u1", "u2", "u3", "u4"],
            "key_sale": ["s1", "s2", "s3", "s4"],
            "neighborhood": ["A", "A", "A", "A"],
            "sale_price": [100.0, 200.0, 999.0, 5000.0],
            "valid_sale": [True, True, True, False],
        }
    )
    sup = SalesUniversePair(sales, universe)

    # s1,s2,s4 are training; s3 is test. s4 is invalid. -> only s1,s2 contribute.
    monkeypatch.setattr(
        area_stats,
        "get_train_test_keys",
        lambda df, s: (np.array(["s1", "s2", "s4"]), np.array(["s3"])),
    )

    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["sale_price"],
            "stats": ["mean"],
            "min_count": 1,
        }
    )
    out = enrich_sup_area_stats(sup, settings)
    mean = out.universe.set_index("key")["area_stat_neighborhood_sale_price_mean"]
    # mean of training valid sales in A = mean(100, 200) = 150
    assert mean["u1"] == 150.0


def test_exclude_test_keys_drops_test_parcels(monkeypatch):
    """exclude_test_keys removes test-key parcels from characteristic aggregation too."""
    universe = pd.DataFrame(
        {
            "key": ["u1", "u2", "u3"],
            "neighborhood": ["A", "A", "A"],
            "bldg_area_finished_sqft": [100.0, 200.0, 900.0],
        }
    )
    sales = pd.DataFrame(
        {
            "key": ["u1", "u2", "u3"],
            "key_sale": ["s1", "s2", "s3"],
            "neighborhood": ["A", "A", "A"],
        }
    )
    sup = SalesUniversePair(sales, universe)
    # s3 (parcel u3) is the test sale; exclude it from the universe aggregation.
    monkeypatch.setattr(
        area_stats,
        "get_train_test_keys",
        lambda df, s: (np.array(["s1", "s2"]), np.array(["s3"])),
    )

    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["bldg_area_finished_sqft"],
            "stats": ["mean"],
            "min_count": 1,
            "exclude_test_keys": True,
        }
    )
    out = enrich_sup_area_stats(sup, settings)
    mean = out.universe.set_index("key")[
        "area_stat_neighborhood_bldg_area_finished_sqft_mean"
    ]
    # u3 excluded -> mean(100, 200) = 150 (not 400 with u3's 900 included)
    assert mean["u1"] == 150.0


def test_generated_fields_are_auto_classified():
    """Derived columns are discoverable via get_fields_numeric/categorical with right kind."""
    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["bldg_area_finished_sqft", "zoning", "sale_price"],
            "stats": ["mean"],
            "categorical_stats": ["mode", "mode_frac"],
            "min_count": 1,
        }
    )
    numeric = set(get_fields_numeric(settings))
    categorical = set(get_fields_categorical(settings))

    assert "area_stat_neighborhood_bldg_area_finished_sqft_mean" in numeric
    assert "area_stat_neighborhood_sale_price_mean" in numeric
    assert "area_stat_neighborhood_zoning_mode_frac" in numeric
    assert "area_stat_neighborhood_count" in numeric
    # mode output is categorical, not numeric
    assert "area_stat_neighborhood_zoning_mode" in categorical
    assert "area_stat_neighborhood_zoning_mode" not in numeric


def test_report_ranks_features_by_correlation(monkeypatch):
    """report_area_stats returns a ranked table; a strong feature ranks at/near the top."""
    from openavmkit.area_stats import report_area_stats

    rng = np.random.default_rng(0)
    n = 120
    nbhd = rng.choice(["A", "B", "C", "D"], n)
    base = {"A": 100.0, "B": 200.0, "C": 300.0, "D": 400.0}
    sqft = np.array([base[x] for x in nbhd]) + rng.normal(0, 10, n)
    price = sqft * 50 + rng.normal(0, 200, n)

    universe = pd.DataFrame(
        {
            "key": [f"u{i}" for i in range(n)],
            "neighborhood": nbhd,
            "bldg_area_finished_sqft": sqft,
        }
    )
    sales = pd.DataFrame(
        {
            "key": [f"u{i}" for i in range(n)],
            "key_sale": [f"s{i}" for i in range(n)],
            "neighborhood": nbhd,
            "bldg_area_finished_sqft": sqft,
            "sale_price": price,
            "valid_sale": [True] * n,
        }
    )
    sup = SalesUniversePair(sales, universe)

    settings = _base_settings(
        {
            "locations": ["neighborhood"],
            "fields": ["bldg_area_finished_sqft"],
            "stats": ["mean", "std"],
            "min_count": 3,
        }
    )
    out = enrich_sup_area_stats(sup, settings)
    ranked = report_area_stats(out, settings, outpath=None)

    assert list(ranked.columns) == [
        "variable",
        "corr_strength",
        "corr_clarity",
        "corr_score",
    ]
    assert not ranked.empty
    # the neighborhood mean sqft should be strongly correlated with price
    top = ranked.iloc[0]
    assert "area_stat_neighborhood_bldg_area_finished_sqft_mean" in set(ranked["variable"])
    assert top["corr_strength"] > 0.5


def test_no_config_is_noop():
    """With no area_stats config, the sup is returned unchanged."""
    settings = {"data": {"process": {"enrich": {}}}, "field_classification": {}}
    sup = _characteristic_sup()
    out = enrich_sup_area_stats(sup, settings)
    assert list(out.universe.columns) == list(sup.universe.columns)
