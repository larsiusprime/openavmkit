"""Tests for the OSM nearest-distance / proximity enrichment.

Validates the pre-filter + reproject-once implementation of
``_do_perform_distance_calculations_osm`` against an independent brute-force nearest
distance, covering the no-max path, the max_distance buffer pre-filter (gating + far-parcel
handling), and equivalence of the pre-projected-parcels fast path.
"""
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import box, Point

from openavmkit.data import _do_perform_distance_calculations_osm
from openavmkit.utilities.geometry import get_crs


def _parcels():
    # Six small parcels marching east of a feature; near Pittsburgh so UTM 17N is chosen.
    polys = [box(-79.90 + i * 0.004, 40.40, -79.90 + i * 0.004 + 0.0005, 40.4005) for i in range(6)]
    return gpd.GeoDataFrame({"key": [f"p{i}" for i in range(6)]}, geometry=polys, crs="EPSG:4326")


def _feature():
    return gpd.GeoDataFrame({"id": [0]}, geometry=[Point(-79.905, 40.4002)], crs="EPSG:4326")


def _brute_force_meters(parcels, feat):
    """Independent nearest distance (meters) parcel-geometry → feature-geometry."""
    crs = get_crs(parcels, "equal_distance")
    p = parcels.to_crs(crs)
    fg = feat.to_crs(crs).geometry.iloc[0]
    return p.geometry.distance(fg).values


def test_osm_distance_no_max_matches_bruteforce(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels, feat = _parcels(), _feature()
    out = _do_perform_distance_calculations_osm(parcels.copy(), feat, "nomax", max_distance=None, unit="m")
    bf = _brute_force_meters(parcels, feat)
    assert np.allclose(out["dist_to_nomax"].values, bf, atol=1e-2)
    assert np.allclose(out["proximity_to_nomax"].values, bf.max() - bf, atol=1e-2)
    assert out["within_nomax"].all()


def test_osm_distance_prefilter_gates_and_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels, feat = _parcels(), _feature()
    bf = _brute_force_meters(parcels, feat)  # increasing west→east
    md = float((bf[2] + bf[3]) / 2.0)  # cut between parcel 2 and 3
    out = _do_perform_distance_calculations_osm(parcels.copy(), feat, "gated", max_distance=md, unit="m")

    in_range = bf <= md
    assert (out["within_gated"].values == in_range).all()
    # in-range distances match brute force; out-of-range are NaN (proximity 0)
    assert np.allclose(out["dist_to_gated"].values[in_range], bf[in_range], atol=1e-2)
    assert np.isnan(out["dist_to_gated"].values[~in_range]).all()
    mx = bf[in_range].max()
    assert np.allclose(out["proximity_to_gated"].values[in_range], mx - bf[in_range], atol=1e-2)
    assert np.allclose(out["proximity_to_gated"].values[~in_range], 0.0)


def test_osm_distance_parcels_proj_equivalent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels, feat = _parcels(), _feature()
    crs = get_crs(parcels, "equal_distance")
    pproj = parcels[["key", "geometry"]].to_crs(crs)

    bf = _brute_force_meters(parcels, feat)
    md = float((bf[2] + bf[3]) / 2.0)
    # The pre-projected fast path must produce identical results to re-projecting internally.
    slow = _do_perform_distance_calculations_osm(parcels.copy(), feat, "slow", max_distance=md, unit="m")
    fast = _do_perform_distance_calculations_osm(parcels.copy(), feat, "fast", max_distance=md, unit="m", parcels_proj=pproj)
    for c in ("dist_to_", "within_", "proximity_to_"):
        np.testing.assert_allclose(
            slow[f"{c}slow"].values.astype(float),
            fast[f"{c}fast"].values.astype(float),
            atol=1e-6, equal_nan=True,
        )
