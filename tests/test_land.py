import numpy as np
import pandas as pd
import pytest

from openavmkit.land import calc_lycd_land_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(units="imperial"):
    return {"locality": {"units": units}}


def _make_df():
    """Return a tiny universe with two model groups and a mix of
    vacant / improved properties."""
    return pd.DataFrame({
        "key":                ["A1", "A2", "A3", "A4", "B1", "B2", "B3"],
        "model_group":        ["G1", "G1", "G1", "G1", "G2", "G2", "G2"],
        "model_market_value": [200_000, 250_000, 300_000, 50_000,
                               400_000, 500_000, 80_000],
        "land_area_sqft":     [5_000, 6_000, 7_000, 4_000,
                               10_000, 12_000, 8_000],
        # A4 and B3 are vacant lots; the rest are improved
        "is_vacant":          [False, False, False, True,
                               False, False, True],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lycd_scalar_alloc():
    """With a fixed 20 % allocation every group gets the same land rate formula."""
    df = _make_df()
    settings = _make_settings()
    result = calc_lycd_land_values(df, settings, land_alloc=0.20)

    assert "lycd_land_value" in result.columns
    assert "lycd_local_land_rate" in result.columns
    assert "lycd_land_alloc" in result.columns

    # --- Group G1 ---
    # improved: A1(200k/5k), A2(250k/6k), A3(300k/7k)
    # median_mv_improved  = 250_000
    # median_lot_improved = 6_000
    # local_land_rate     = (250_000 * 0.20) / 6_000  ≈ 8.3333
    g1_rate = (250_000 * 0.20) / 6_000
    for _, row in result[result["model_group"] == "G1"].iterrows():
        assert abs(row["lycd_local_land_rate"] - g1_rate) < 1e-6
        expected_lv = min(g1_rate * row["land_area_sqft"], row["model_market_value"])
        assert abs(row["lycd_land_value"] - expected_lv) < 1e-2
        assert abs(row["lycd_land_alloc"] - 0.20) < 1e-10

    # --- Group G2 ---
    # improved: B1(400k/10k), B2(500k/12k)
    # median_mv_improved  = 450_000
    # median_lot_improved = 11_000
    # local_land_rate     = (450_000 * 0.20) / 11_000
    g2_rate = (450_000 * 0.20) / 11_000
    for _, row in result[result["model_group"] == "G2"].iterrows():
        assert abs(row["lycd_local_land_rate"] - g2_rate) < 1e-6


def test_lycd_dict_alloc():
    """Per-group allocations are applied correctly."""
    df = _make_df()
    settings = _make_settings()
    alloc = {"G1": 0.15, "G2": 0.25}
    result = calc_lycd_land_values(df, settings, land_alloc=alloc)

    g1_rows = result[result["model_group"] == "G1"]
    g2_rows = result[result["model_group"] == "G2"]

    assert (g1_rows["lycd_land_alloc"] - 0.15).abs().max() < 1e-10
    assert (g2_rows["lycd_land_alloc"] - 0.25).abs().max() < 1e-10


def test_lycd_auto_alloc_from_vacant():
    """When land_alloc=None the allocation is derived from vacant vs improved
    per-unit values within each group."""
    df = _make_df()
    settings = _make_settings()
    result = calc_lycd_land_values(df, settings, land_alloc=None)

    # --- Group G1 ---
    # vacant:  A4 → 50_000 / 4_000 = 12.5  $/sqft  (only one, median = 12.5)
    # improved per-unit medians:
    #   A1: 200_000/5_000 = 40.0
    #   A2: 250_000/6_000 ≈ 41.667
    #   A3: 300_000/7_000 ≈ 42.857
    #   median ≈ 41.667
    # implied alloc = 12.5 / 41.667 ≈ 0.3000
    g1_vacant_rate = 50_000 / 4_000
    g1_improved_rates = sorted([200_000 / 5_000, 250_000 / 6_000, 300_000 / 7_000])
    g1_improved_median = np.median(g1_improved_rates)
    expected_g1_alloc = g1_vacant_rate / g1_improved_median

    g1_rows = result[result["model_group"] == "G1"]
    assert (g1_rows["lycd_land_alloc"] - expected_g1_alloc).abs().max() < 1e-6

    # land values must be non-negative
    assert (result["lycd_land_value"] >= 0).all()


def test_lycd_clamping():
    """Land value is clamped to [0, market_value]."""
    # Construct a case where the raw rate would overshoot
    df = pd.DataFrame({
        "key":                ["X1", "X2"],
        "model_group":        ["G1", "G1"],
        "model_market_value": [100_000, 100_000],
        "land_area_sqft":     [1_000, 100_000],   # huge lot → would overshoot
        "is_vacant":          [False, False],
    })
    settings = _make_settings()
    result = calc_lycd_land_values(df, settings, land_alloc=0.20)

    # No land value may exceed the market value
    assert (result["lycd_land_value"] <= result["model_market_value"] + 1e-6).all()
    assert (result["lycd_land_value"] >= 0).all()


def test_lycd_metric_units():
    """Works with metric (sqm) settings."""
    df = pd.DataFrame({
        "key":                ["M1", "M2", "M3"],
        "model_group":        ["G1", "G1", "G1"],
        "model_market_value": [200_000, 300_000, 50_000],
        "land_area_sqm":      [500, 700, 400],
        "is_vacant":          [False, False, True],
    })
    settings = _make_settings(units="metric")
    result = calc_lycd_land_values(df, settings, land_alloc=0.20)

    # median improved: mv=250k, lot=600 sqm
    # rate = (250_000 * 0.20) / 600
    expected_rate = (250_000 * 0.20) / 600
    assert (result["lycd_local_land_rate"] - expected_rate).abs().max() < 1e-6


def test_lycd_no_vacant_falls_back_to_global():
    """A group with no vacant properties falls back to the global allocation."""
    df = pd.DataFrame({
        "key":                ["A1", "A2", "B1", "B2", "B3"],
        "model_group":        ["G1", "G1", "G2", "G2", "G2"],
        "model_market_value": [200_000, 300_000, 400_000, 500_000, 80_000],
        "land_area_sqft":     [5_000, 7_000, 10_000, 12_000, 8_000],
        # G1 has no vacant; G2 has B3 as vacant
        "is_vacant":          [False, False, False, False, True],
    })
    settings = _make_settings()
    result = calc_lycd_land_values(df, settings, land_alloc=None)

    # Both groups should receive a valid (non-NaN) allocation
    assert result["lycd_land_alloc"].notna().all()
    # The G1 allocation should equal the global allocation
    g1_alloc = result[result["model_group"] == "G1"]["lycd_land_alloc"].iloc[0]
    g2_alloc = result[result["model_group"] == "G2"]["lycd_land_alloc"].iloc[0]

    # G2 has its own vacant data so it will differ; G1 must be the global fallback
    # Global: all vacant = B3 (80k/8k=10), all improved = A1,A2,B1,B2
    all_improved_rates = sorted([200_000/5_000, 300_000/7_000, 400_000/10_000, 500_000/12_000])
    global_improved_median = np.median(all_improved_rates)
    global_vacant_rate = 80_000 / 8_000
    expected_global_alloc = global_vacant_rate / global_improved_median

    assert abs(g1_alloc - expected_global_alloc) < 1e-6
