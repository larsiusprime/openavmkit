import os

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from openavmkit.utilities.dem import (
    DEMService,
    bbox_in_usgs_coverage,
    _utm_crs_for_bbox,
)


def test_bbox_in_usgs_coverage_conus():
    # central Virginia
    assert bbox_in_usgs_coverage((-78.0, 37.0, -77.5, 37.5))


def test_bbox_in_usgs_coverage_alaska():
    # Anchorage area
    assert bbox_in_usgs_coverage((-150.0, 61.0, -149.5, 61.5))


def test_bbox_in_usgs_coverage_hawaii():
    assert bbox_in_usgs_coverage((-156.0, 20.5, -155.5, 21.0))


def test_bbox_in_usgs_coverage_outside():
    # mid-Atlantic ocean
    assert not bbox_in_usgs_coverage((-30.0, 30.0, -29.5, 30.5))
    # central Europe
    assert not bbox_in_usgs_coverage((10.0, 48.0, 11.0, 49.0))


def test_utm_crs_for_bbox_north_zone():
    crs = _utm_crs_for_bbox((-77.5, 37.0, -77.0, 37.5))
    assert "+proj=utm" in crs
    assert "+zone=18" in crs
    assert "+north" in crs


def test_utm_crs_for_bbox_south_zone():
    crs = _utm_crs_for_bbox((-58.5, -34.7, -58.3, -34.5))
    assert "+south" in crs


def test_compute_parcel_stats_on_synthetic_raster(tmp_path):
    """Build a synthetic UTM DEM with a known linear ramp and verify per-parcel
    mean/stdev/slope match analytical expectations."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # 100x100 grid, 10m pixels in a fictional UTM zone, elevations 0..990 east-to-west
    # gradient (so slope per pixel is 1m rise over 10m run -> arctan(0.1) ~ 5.71 deg).
    width, height = 100, 100
    pixel_size = 10.0
    origin_x, origin_y = 500000.0, 4000000.0  # arbitrary UTM coords
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    # elevation = 10 * column index (0..990 meters east-to-west)
    elevation = np.tile(np.arange(width) * 10.0, (height, 1)).astype("float32")

    utm_crs = "+proj=utm +zone=10 +north +datum=WGS84 +units=m +no_defs"
    dem_path = tmp_path / "dem.tif"
    with rasterio.open(
        dem_path, "w",
        driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=utm_crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)

    service = DEMService({})
    slope_path = service.compute_slope_raster(dem_path)

    # Build a parcel that covers pixels (col 10..19, row 10..19) inclusive ->
    # elevations 100..190, mean ~145, stdev ~ stdev(arange(100,200,10)) ~ 28.72.
    minx = origin_x + 10 * pixel_size
    maxx = origin_x + 20 * pixel_size
    maxy = origin_y - 10 * pixel_size
    miny = origin_y - 20 * pixel_size
    parcel = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    # Also a flat parcel: just one column wide -> stdev ~ 0
    flat = Polygon([
        (origin_x + 50 * pixel_size, origin_y - 50 * pixel_size),
        (origin_x + 51 * pixel_size, origin_y - 50 * pixel_size),
        (origin_x + 51 * pixel_size, origin_y - 40 * pixel_size),
        (origin_x + 50 * pixel_size, origin_y - 40 * pixel_size),
    ])

    gdf = gpd.GeoDataFrame({"id": ["ramp", "flat"]}, geometry=[parcel, flat], crs=utm_crs)
    stats = service.compute_parcel_stats(gdf, dem_path, slope_path)

    # Ramp parcel: each of 10 columns has elevations [100, 110, ..., 190], mean=145.
    assert stats.loc[gdf.index[0], "elevation_mean_m"] == pytest.approx(145.0, abs=5.0)
    expected_std = np.std(np.arange(100, 200, 10))
    assert stats.loc[gdf.index[0], "elevation_stdev_m"] == pytest.approx(expected_std, abs=2.0)
    # Slope: arctan(10/10) = arctan(1) = 45 deg? No — 10m rise over 10m horizontal between
    # adjacent pixels => gradient is 1.0 => slope = arctan(1) = 45 deg.
    assert stats.loc[gdf.index[0], "slope_mean_deg"] == pytest.approx(45.0, abs=2.0)

    # Flat parcel (single column width): stdev should be ~0 since one column,
    # all rows have same elevation.
    assert stats.loc[gdf.index[1], "elevation_stdev_m"] == pytest.approx(0.0, abs=1.0)


def test_dem_service_resolution_validation(tmp_path):
    """Invalid resolutions are rejected before any network call."""
    service = DEMService({})
    with pytest.raises(ValueError, match="10m, 30m, or 60m"):
        service.get_dem_for_bbox((-78.0, 37.0, -77.5, 37.5), resolution_m=5)


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="Set RUN_NETWORK_TESTS=1 to enable USGS network integration test",
)
def test_dem_service_integration_petersburg(tmp_path, monkeypatch):
    """End-to-end fetch + reproject + slope + stats for a tiny Petersburg, VA bbox."""
    pytest.importorskip("rasterio")
    pytest.importorskip("seamless_3dep")

    monkeypatch.chdir(tmp_path)
    bbox = (-77.41, 37.21, -77.39, 37.23)  # tiny AOI in Petersburg, VA

    service = DEMService({})
    dem_path = service.get_dem_for_bbox(bbox, resolution_m=30, verbose=True)
    utm_path = service.reproject_to_utm(dem_path, bbox, verbose=True)
    slope_path = service.compute_slope_raster(utm_path, verbose=True)

    # one big parcel covering the whole AOI
    poly = Polygon([
        (bbox[0], bbox[1]), (bbox[2], bbox[1]),
        (bbox[2], bbox[3]), (bbox[0], bbox[3]),
    ])
    gdf = gpd.GeoDataFrame({"id": ["aoi"]}, geometry=[poly], crs="EPSG:4326")
    stats = service.compute_parcel_stats(gdf, utm_path, slope_path)

    # Petersburg is roughly 30-60m elevation; verify result is in a plausible range.
    elev = stats.iloc[0]["elevation_mean_m"]
    assert 0 < elev < 200, f"unexpected elevation: {elev}"
    assert stats.iloc[0]["elevation_stdev_m"] >= 0
    assert 0 <= stats.iloc[0]["slope_mean_deg"] < 45
