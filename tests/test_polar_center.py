"""Polar coordinates are measured from locality.center, which is in degrees, whatever the parcels' coordinates."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openavmkit.data import _enrich_polar_coordinates


def _parcels(crs: str) -> gpd.GeoDataFrame:
    # A row of lots near Greensboro, NC, given in NC State Plane (US feet) or in degrees.
    lots = gpd.GeoDataFrame(
        {"key": [str(n) for n in range(10)]},
        geometry=[box(1_760_000 + n * 100, 850_000, 1_760_080 + n * 100, 850_120) for n in range(10)],
        crs="EPSG:2264",
    )
    return lots.to_crs(crs)


@pytest.mark.parametrize("crs", ["EPSG:2264", "EPSG:4326"])
def test_a_center_in_degrees_is_read_as_degrees(crs):
    gdf = _parcels(crs)
    middle = gdf.to_crs(4326).geometry.union_all().centroid
    from_middle = _enrich_polar_coordinates(gdf, {"locality": {}})
    given = {"locality": {"center": {"longitude": middle.x, "latitude": middle.y}}}
    from_center = _enrich_polar_coordinates(gdf, given)
    # The same point, given or worked out: the same distances (a few metres at most, from the
    # middle being worked out in the file's own coordinates).
    assert from_center["polar_radius"].to_numpy() == pytest.approx(from_middle["polar_radius"].to_numpy(), abs=5)
    assert from_center["polar_radius"].max() < 1_000  # metres: the lots span about 300 m


@pytest.mark.parametrize("crs", ["EPSG:2264", "EPSG:4326"])
def test_the_middle_is_the_dissolved_middle_where_parcels_dont_overlap(crs):
    """Without locality.center, the middle is the area-weighted mean of the parcels' centroids: where they don't
    overlap, the same point as the centroid of all of them dissolved into one, without the dissolve's seconds."""
    from openavmkit.utilities.settings import get_center

    gdf = _parcels(crs)
    union = gdf.geometry.union_all().centroid
    assert get_center({"locality": {}}, gdf) == pytest.approx((union.x, union.y), rel=1e-12, abs=1e-9)


def test_stacked_duplicates_count_once_as_in_the_dissolve():
    from openavmkit.utilities.geometry import area_weighted_center

    gdf = _parcels("EPSG:2264")
    # condo units stacked five deep on the first lot's footprint
    stacked = gpd.GeoDataFrame(
        {"key": [f"u{n}" for n in range(5)]}, geometry=[gdf.geometry.iloc[0]] * 5, crs=gdf.crs
    )
    both = gpd.GeoDataFrame(pd.concat([gdf, stacked], ignore_index=True), crs=gdf.crs)
    union = both.geometry.union_all().centroid
    weighted = area_weighted_center(both.geometry)
    assert (weighted.x, weighted.y) == pytest.approx((union.x, union.y), rel=1e-12, abs=1e-9)


def test_points_give_their_plain_mean_and_nothing_gives_none():
    from shapely.geometry import Point

    from openavmkit.utilities.geometry import area_weighted_center

    pts = gpd.GeoSeries([Point(0, 0), Point(2, 0), Point(4, 6), None], crs="EPSG:2264")
    m = area_weighted_center(pts)
    assert (m.x, m.y) == pytest.approx((2.0, 2.0))
    assert area_weighted_center(gpd.GeoSeries([], crs="EPSG:2264")) is None


def test_no_shapes_and_no_center_says_so():
    from openavmkit.utilities.settings import get_center

    empty = gpd.GeoDataFrame({"key": []}, geometry=[], crs="EPSG:2264")
    with pytest.raises(ValueError, match="locality.center"):
        get_center({"locality": {}}, empty)
