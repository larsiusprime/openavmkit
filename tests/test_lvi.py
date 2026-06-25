"""Tests for the openavmkit.lvi land-value-integrity battery.

Two layers:
  * Synthetic "recover-known-land" tests — build a universe + sales with a KNOWN land truth and
    confirm the battery reads it back (A3 level, A1 independence, A7 sanity, A5 gradient), plus
    unit tests for config parsing, filter-driven evidence streams, prime_comp, and reconstruct_rcn.
  * A Wake regression test that pins today's headline numbers — skipped automatically when the
    cleaned sup isn't present (so it runs locally, skips in CI).
"""
import os

import numpy as np
import pandas as pd
import pytest

from openavmkit.lvi import run_land_value_integrity
from openavmkit.lvi.config import load_lvi_configs, GroupConfig
from openavmkit.lvi import evidence, battery
from openavmkit.lvi.report import scorecard


# --------------------------------------------------------------------------- synthetic fixture

def make_synthetic(land_factor=1.0, n_nbhd=6, per=40, seed=1):
    """A small synthetic jurisdiction with a KNOWN land truth.

    Per neighborhood i, land has a base rate (8 + 4*i $/sqft) plus parcel-level noise that is
    INDEPENDENT of the building (so A1's partial R^2 ~ 0). The series under test assesses land at
    ``land_factor`` x the true land (1.0 = perfect; 0.8 = 20% under). A few oversized / odd-shaped
    vacant lots are planted to exercise prime_comp. Returns (universe_df, sales_df)."""
    rng = np.random.default_rng(seed)
    cost_psf = 100.0
    urows, srows = [], []
    k = 0
    for i in range(n_nbhd):
        base_psf = 8.0 + 4.0 * i
        typical_lot = 8000.0 + 1000.0 * i
        for j in range(per):
            k += 1
            key = f"{k:06d}"
            nb = f"NB{i:02d}"
            improved = (j % 2 == 0)
            # most lots typical-sized; a couple oversized; one odd-shaped
            if j == per - 1:
                la = typical_lot * 6.0          # oversized -> fails prime size band
            else:
                la = typical_lot * rng.uniform(0.85, 1.15)
            rect = 0.30 if j == per - 2 else 0.85   # one irregular -> fails prime shape
            psf = base_psf * rng.uniform(0.9, 1.1)  # within-nbhd noise, independent of building
            true_land = psf * la
            lat = 35.7 + 0.05 * i + rng.uniform(-0.005, 0.005)
            lon = -78.6 - 0.05 * i + rng.uniform(-0.005, 0.005)
            if improved:
                ba = rng.uniform(1500, 3000)
                quality = int(rng.integers(2, 6))
                new = (j % 4 == 0)
                pctgood = 100.0 if new else float(rng.choice([85.0, 70.0]))
                yr = 2025 if new else int(rng.integers(1960, 2010))
                building = cost_psf * ba
                impr_v = building
                la_band = "S" if la < typical_lot else "L"
                urows.append(dict(
                    key=key, neighborhood_filled=nb, model_group="single_family",
                    land_area_sqft=la, bldg_area_finished_sqft=ba, is_vacant=False,
                    geom_rectangularity_num=rect, latitude=lat, longitude=lon,
                    land_he_id=f"{nb}_{la_band}", impr_he_id=f"Q{quality}_{int(ba // 500)}",
                    bldg_condition_pct=pctgood, bldg_year_built=yr,
                    bldg_age_years=2026 - yr, bldg_quality_num=quality, bldg_condition_num=pctgood / 100.0,
                    assr_land_value=land_factor * true_land, assr_impr_value=impr_v,
                    assr_market_value=land_factor * true_land + impr_v,
                    land_value=land_factor * true_land, impr_value=impr_v,
                    total_value=land_factor * true_land + impr_v))
                srows.append(dict(
                    key=key, sale_price=true_land + building, sale_price_time_adj=true_land + building,
                    sale_date="2025-06-01", bldg_year_built=yr, vacant_sale=False, disq_flag="A",
                    valid_sale=True, valid_for_ratio_study=True, valid_for_land_ratio_study=False,
                    bldg_condition_pct=pctgood, sale_age_days=200, model_group="single_family"))
            else:
                urows.append(dict(
                    key=key, neighborhood_filled=nb, model_group="single_family",
                    land_area_sqft=la, bldg_area_finished_sqft=0.0, is_vacant=True,
                    geom_rectangularity_num=rect, latitude=lat, longitude=lon,
                    land_he_id=f"{nb}_{'S' if la < typical_lot else 'L'}", impr_he_id="VAC",
                    bldg_condition_pct=0.0, bldg_year_built=np.nan,
                    bldg_age_years=np.nan, bldg_quality_num=np.nan, bldg_condition_num=np.nan,
                    assr_land_value=land_factor * true_land, assr_impr_value=0.0,
                    assr_market_value=land_factor * true_land,
                    land_value=land_factor * true_land, impr_value=0.0,
                    total_value=land_factor * true_land))
                srows.append(dict(
                    key=key, sale_price=true_land, sale_price_time_adj=true_land,
                    sale_date="2025-06-01", bldg_year_built=np.nan, vacant_sale=True, disq_flag="A",
                    valid_sale=True, valid_for_ratio_study=False, valid_for_land_ratio_study=True,
                    bldg_condition_pct=0.0, sale_age_days=200, model_group="single_family"))
    return pd.DataFrame(urows), pd.DataFrame(srows)


SF_FILTERS = {"land_value_integrity": {"single_family": {
    "land_evidence_filter": ["and", ["==", "vacant_sale", True], ["==", "valid_for_land_ratio_study", True]],
    "cost_residual_filter": ["and", ["==", "vacant_sale", False], [">=", "bldg_condition_pct", 95]],
}}}


def _cfg():
    return load_lvi_configs(SF_FILTERS)["single_family"]


# --------------------------------------------------------------------------- config

def test_config_defaults_and_skip_unnamed():
    settings = {"land_value_integrity": {
        "__defaults": {"evidence": {"frozen_sov": False}},
        "single_family": {"land_evidence_filter": ["==", "vacant_sale", True]},
        "commercial": {"land_evidence_filter": ["==", "vacant_sale", True], "evidence": {"frozen_sov": True}},
    }}
    cfgs = load_lvi_configs(settings)
    assert set(cfgs) == {"single_family", "commercial"}        # __defaults skipped
    assert cfgs["single_family"].evidence.frozen_sov is False  # default merged in
    assert cfgs["commercial"].evidence.frozen_sov is True       # group override wins
    assert cfgs["single_family"].verdicts.a3_cod == (15, 25)    # IAAO default
    assert cfgs["single_family"].fields.land_value == "land_value"


def test_config_requires_land_evidence_filter():
    with pytest.raises(ValueError):
        load_lvi_configs({"land_value_integrity": {"single_family": {}}})


# --------------------------------------------------------------------------- evidence

def test_reconstruct_rcn():
    df = pd.DataFrame({"assr_impr_value": [80.0, 0.0, 100.0], "bldg_condition_pct": [80.0, 90.0, 50.0]})
    rcn = evidence.reconstruct_rcn(df)
    assert rcn.iloc[0] == pytest.approx(100.0)   # 80 / 0.80
    assert np.isnan(rcn.iloc[1])                 # impr<=0 -> NaN
    assert rcn.iloc[2] == pytest.approx(200.0)   # 100 / 0.50


def test_streams_are_filter_driven():
    u, s = make_synthetic()
    obs = evidence.build_land_observations(s, u, _cfg())
    assert set(obs["kind"].unique()) <= {"direct", "cost_residual"}
    assert (obs["kind"] == "direct").sum() > 0
    assert (obs["kind"] == "cost_residual").sum() > 0


def test_prime_comp_drops_oversized_and_odd():
    u, s = make_synthetic()
    cfg = _cfg()
    obs_on = evidence.build_land_observations(s, u, cfg)
    cfg_off = load_lvi_configs({"land_value_integrity": {"single_family": {
        "land_evidence_filter": ["and", ["==", "vacant_sale", True], ["==", "valid_for_land_ratio_study", True]],
        "prime": {"enabled": False}}}})["single_family"]
    obs_off = evidence.build_land_observations(s, u, cfg_off)
    n_on = (obs_on["kind"] == "direct").sum()
    n_off = (obs_off["kind"] == "direct").sum()
    assert n_on < n_off              # the oversized/odd lots are screened out when prime_comp is on


# --------------------------------------------------------------------------- battery (recovery)

def test_a3_recovers_known_level_perfect():
    u, s = make_synthetic(land_factor=1.0)
    cfg = _cfg()
    obs = evidence.build_land_observations(s, u, cfg)
    res = battery.test_A3("assessor", u, obs, cfg)
    assert res["median_ratio"] == pytest.approx(1.0, abs=0.03)
    assert res["cod"] < 12


def test_a3_recovers_known_underassessment():
    u, s = make_synthetic(land_factor=0.8)
    cfg = _cfg()
    obs = evidence.build_land_observations(s, u, cfg)
    res = battery.test_A3("assessor", u, obs, cfg)
    assert res["median_ratio"] == pytest.approx(0.8, abs=0.03)


def test_a7_clean_then_violation():
    u, s = make_synthetic()
    cfg = _cfg()
    u2 = u.copy()
    u2["_land_psf"] = u2["land_value"] / u2["land_area_sqft"]
    u2["_impr_psf"] = u2["impr_value"] / u2["bldg_area_finished_sqft"]
    a7, _ = battery.test_A7("assessor", u2, cfg)
    assert a7["pct_violations"] == pytest.approx(0.0, abs=1e-6)  # land+impr=total by construction
    u2.loc[u2.index[0], "total_value"] *= 2.0                    # break the identity on one parcel
    a7b, _ = battery.test_A7("assessor", u2, cfg)
    assert a7b["pct_violations"] > 0


def test_a1_land_independent_of_improvements():
    u, s = make_synthetic()
    cfg = _cfg()
    u["_land_psf"] = u["land_value"] / u["land_area_sqft"]
    res = battery.test_A1("assessor", u, cfg)
    # land was built independent of building features -> low partial R^2
    assert np.isfinite(res["partial_r2"])
    assert res["partial_r2"] < 0.15


def test_a5_tracks_market_gradient():
    u, s = make_synthetic(land_factor=1.0)
    cfg = _cfg()
    obs = evidence.build_land_observations(s, u, cfg)
    res = battery.test_A5("assessor", u, obs, cfg)
    assert np.isfinite(res["rho_market"])
    assert res["rho_market"] > 0.7        # assessed land tracks the per-neighborhood market gradient


def test_end_to_end_run_and_scorecard():
    u, s = make_synthetic(land_factor=0.85)
    res = run_land_value_integrity({"assessor": u}, s, _cfg())
    assert set(res.results["assessor"]) >= {"total", "A1", "A2", "A3", "A5", "A6", "A7", "A8", "B3", "VE"}
    assert "A0" in res.diagnostics and "depreciation" in res.diagnostics
    assert "propagation" in res.support
    pkt = res.packets["assessor"]
    assert "land_integrity_confidence" in pkt.columns
    assert isinstance(res.scorecard(), str) and "A3" in res.scorecard()


# --------------------------------------------------------------------------- Wake regression (skipped if no data)

WAKE = os.path.join("notebooks", "pipeline", "data", "us-nc-wake", "out", "2-clean-sup.pickle")


@pytest.mark.skipif(not os.path.exists(WAKE), reason="Wake cleaned sup not present (local-only data)")
def test_wake_regression():
    from openavmkit.pipeline import read_pickle
    from openavmkit.data import get_hydrated_sales_from_sup
    cwd = os.getcwd(); os.chdir(os.path.dirname(os.path.dirname(WAKE)).rsplit("out", 1)[0] or ".")
    try:
        sup = read_pickle("out/2-clean-sup")
    finally:
        os.chdir(cwd)
    u = sup.universe[sup.universe["model_group"] == "single_family"].copy(); u["key"] = u["key"].astype(str)
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["model_group"] == "single_family") & (s["valid_sale"] == True) & (s["sale_price"] > 0)].copy()
    s["key"] = s["key"].astype(str)
    cfg = load_lvi_configs({"land_value_integrity": {"single_family": {
        "land_evidence_filter": ["and", ["==", "vacant_sale", True],
                                 ["==", "valid_for_land_ratio_study", True], ["isin", "disq_flag", ["A", "C"]]],
        "cost_residual_filter": ["and", ["==", "vacant_sale", False], [">=", "bldg_condition_pct", 95]],
    }}})["single_family"]
    u["land_value"], u["impr_value"], u["total_value"] = u["assr_land_value"], u["assr_impr_value"], u["assr_market_value"]
    res = run_land_value_integrity({"assessor": u}, s, cfg)
    r = res.results["assessor"]
    assert r["A1"]["partial_r2"] == pytest.approx(0.093, abs=0.02)
    assert r["A3"]["median_ratio"] == pytest.approx(0.85, abs=0.02)
    assert r["A3"]["cod"] == pytest.approx(25, abs=3)
    assert r["A8"]["impr_chd"] == pytest.approx(6.3, abs=1.0)
