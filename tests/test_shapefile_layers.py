"""User shapefiles: GeoJSON loads like the other GIS formats; distances from a user's layer need no "name"
column and fail loudly."""

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from openavmkit.data import _enrich_df_distances, _snoop_column_names, load_dataframe


def _parcels() -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {"key": [str(n) for n in range(5)]},
        geometry=[box(-79.80 + n * 0.01, 36.10, -79.795 + n * 0.01, 36.105) for n in range(5)],
        crs="EPSG:4326",
    )
    c = gdf.geometry.centroid
    gdf["latitude"], gdf["longitude"] = c.y, c.x
    return gdf


def test_geojson_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()
    zones = gpd.GeoDataFrame({"ZONE": ["R1", "C2"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    zones.to_file(tmp_path / "in" / "zones.geojson", driver="GeoJSON")
    assert _snoop_column_names("in/zones.geojson") == ["ZONE", "geometry"]
    df = load_dataframe({"filename": "zones.geojson", "load": {"zone": "ZONE"}}, {"locality": {"units": "imperial"}})
    assert list(df["zone"]) == ["R1", "C2"] and df.crs.to_epsg() == 4326


def test_distances_from_a_layer_without_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schools = gpd.GeoDataFrame(geometry=[Point(-79.80, 36.1025)], crs="EPSG:4326")  # no "name" column
    settings = {"enabled": True, "schools": {"source": "schools", "max_distance": 5, "unit": "km"}}
    out = _enrich_df_distances(_parcels(), settings, {"schools": schools})
    assert "dist_to_schools" in out and out["dist_to_schools"].notna().all()
    assert out["dist_to_schools"].is_monotonic_increasing  # the lots run east, away from the school


def test_a_broken_source_layer_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = {"enabled": True, "schools": {"source": "not_loaded", "max_distance": 5}}
    with pytest.raises(ValueError, match="schools"):
        _enrich_df_distances(_parcels(), settings, {})
