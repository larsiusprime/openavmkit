"""Overture intersection caches must store ONLY the computed per-parcel stats (keyed by
bbox) and graft them onto the LIVE input frame -- never a snapshot of the input frame.
Otherwise the cache goes stale when callers add input columns (e.g. 'address') and the
downstream cache integrity check fires a false positive."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from openavmkit.utilities.overture import OvertureService

STAT = "bldg_area_footprint_sqft"


def _svc():
    # The stats-cache helpers don't use instance state, so bypass __init__.
    return OvertureService.__new__(OvertureService)


def _gdf(keys, extra=None):
    data = {"key": list(keys)}
    if extra:
        data.update(extra)
    return gpd.GeoDataFrame(data, geometry=[Point(i, i) for i in range(len(keys))], crs=4326)


def test_cache_preserves_new_input_columns(tmp_path):
    svc = _svc()
    cp = str(tmp_path / "intersections_area.parquet")
    # cache built from a frame WITHOUT 'address'
    old = _gdf(["a", "b"], {STAT: [100.0, 200.0]})
    svc._stats_cache_save(cp, old, [STAT])
    # later the live frame has the SAME parcels but a new 'address' column
    live = _gdf(["a", "b"], {"address": ["123 MAIN ST", "9 OAK DR"]})
    out = svc._stats_cache_load(cp, live, [STAT])
    assert out is not None
    assert "address" in out.columns                  # live column preserved
    assert list(out[STAT]) == [100.0, 200.0]          # cached stat grafted on
    assert "geometry" in out.columns


def test_cache_recomputes_when_parcel_set_changes(tmp_path):
    svc = _svc()
    cp = str(tmp_path / "intersections_area.parquet")
    svc._stats_cache_save(cp, _gdf(["a"], {STAT: [100.0]}), [STAT])
    # a new parcel 'c' is not covered by the cache -> must force recompute (return None)
    live = _gdf(["a", "c"], {"address": ["x", "z"]})
    assert svc._stats_cache_load(cp, live, [STAT]) is None


def test_cache_absent_returns_none(tmp_path):
    svc = _svc()
    assert svc._stats_cache_load(str(tmp_path / "nope.parquet"), _gdf(["a"]), [STAT]) is None
