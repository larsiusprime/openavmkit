"""Polar coordinates are measured from locality.center, which is in degrees, whatever the parcels' coordinates."""

import geopandas as gpd
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
