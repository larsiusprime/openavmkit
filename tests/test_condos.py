"""Tests for openavmkit.condos.resolve_condos (condo geometry borrowing + grouping +
land-share allocation)."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from openavmkit.condos import (
    resolve_condos,
    LAND_ALLOC_FIELD,
    BORROWED_FLAG,
    _polygon_areas_sqft,
)


def _building_poly():
    # A small footprint near Eagle County, CO (lon/lat, EPSG:4326).
    return Polygon([
        (-106.7000, 39.6000),
        (-106.6996, 39.6000),
        (-106.6996, 39.6003),
        (-106.7000, 39.6003),
    ])


def _settings(condos):
    return {
        "data": {"process": {"merge": {"universe": ["char"]}, "condos": condos}},
        "field_classification": {},
    }


def _frames(land_vals):
    poly = _building_poly()
    geo = gpd.GeoDataFrame(
        {"key": ["B001"], "parcel_num": ["123456789001"]},
        geometry=[poly], crs=4326,
    )
    char = pd.DataFrame({
        "key": ["B001", "U2", "U3"],                       # one already-mapped + two orphans
        "parcel_num": ["123456789001", "123456789002", "123456789003"],
        "property_use": ["CONDOS-IMP"] * 3,
        "bldg_area_finished_sqft": [1000.0, 2000.0, 1000.0],
        "land_area_sqft": land_vals,
    })
    return {"geo_parcels": geo, "char": char}, poly


def _condos_cfg(land_share):
    return {
        "enabled": True,
        "select": ["contains", "property_use", ["CONDOS-IMP"]],
        "link": {"method": "id_prefix", "id_field": "parcel_num", "prefix_len": 9, "from": "geo_parcels"},
        "group_field": "condo_group",
        "borrow_geometry": True,
        "land_share": land_share,
    }


def test_id_prefix_borrow_group_and_autoregister():
    dfs, poly = _frames([0.05 * 43560] * 3)
    s = _settings(_condos_cfg({"method": "field", "field": "land_area_sqft"}))
    out = resolve_condos(dfs, s)
    g, c = out["geo_parcels"], out["char"]

    # B001 already had geometry; U2 and U3 get borrowed rows (distinct keys preserved).
    assert len(g) == 3
    assert set(g["key"]) == {"B001", "U2", "U3"}
    assert int(g[BORROWED_FLAG].sum()) == 2
    # All three units grouped under the 9-digit building id.
    assert (c["condo_group"] == "123456789").all()
    # Borrowed geometry equals the building footprint.
    assert g.loc[g["key"] == "U2"].geometry.iloc[0].equals(poly)
    # Fields auto-registered as a categorical location + numeric land field.
    fc = s["field_classification"]
    assert "condo_group" in fc["land"]["categorical"]
    assert "condo_group" in fc["important"]["locations"]
    assert LAND_ALLOC_FIELD in fc["land"]["numeric"]


def test_disabled_is_noop():
    dfs, _ = _frames([100.0] * 3)
    s = _settings(_condos_cfg({"method": "field", "field": "land_area_sqft"}))
    s["data"]["process"]["condos"]["enabled"] = False
    out = resolve_condos(dfs, s)
    assert len(out["geo_parcels"]) == 1
    assert "condo_group" not in out["char"].columns


def test_floor_area_land_share_sums_to_parcel():
    # Floors 1000/2000/1000 -> shares 0.25/0.5/0.25 of the building land area.
    dfs, _ = _frames([None, None, None])
    s = _settings(_condos_cfg({"method": "floor_area", "floor_field": "bldg_area_finished_sqft"}))
    out = resolve_condos(dfs, s)
    c = out["char"].set_index("key")
    parcel_area = float(_polygon_areas_sqft(dfs["geo_parcels"]).iloc[0])

    alloc = c[LAND_ALLOC_FIELD].astype(float)
    # Allocations sum to the parcel area and are proportional to floor area.
    assert abs(alloc.sum() - parcel_area) < 1e-3 * parcel_area
    assert abs(alloc["U2"] - 2 * alloc["B001"]) < 1e-6
    # condos' land_area_sqft is overwritten with the per-unit share.
    assert abs(c.loc["U2", "land_area_sqft"] - alloc["U2"]) < 1e-6
