"""
USGS 3DEP Digital Elevation Model (DEM) service.

Fetches DEM tiles from the USGS 3D Elevation Program for a given bounding box,
mosaics them, reprojects to a local UTM CRS, derives a slope raster, and
computes per-parcel zonal statistics (mean elevation, stdev elevation, mean
slope). Used by the DEM enrichment step (``data.process.enrich.dem``) in
``openavmkit.data``.

USGS 3DEP covers CONUS, AK, HI, and PR at 10m / 30m / 60m resolution.
For bounding boxes outside that coverage the caller should warn-and-skip;
this module's coverage check is the authoritative gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

# Rough lat/lon bounds for USGS 3DEP coverage regions.
_USGS_COVERAGE_REGIONS = [
    (-125.0, 24.0, -66.5, 49.5),    # CONUS
    (-180.0, 51.0, -129.0, 72.0),   # Alaska (incl. Aleutians)
    (-161.0, 18.5, -154.5, 22.5),   # Hawaii
    (-68.0, 17.5, -65.0, 18.6),     # Puerto Rico
]


def bbox_in_usgs_coverage(bbox: Tuple[float, float, float, float]) -> bool:
    """Return True if any part of the WGS84 bbox overlaps USGS 3DEP coverage."""
    west, south, east, north = bbox
    for r_w, r_s, r_e, r_n in _USGS_COVERAGE_REGIONS:
        if west <= r_e and east >= r_w and south <= r_n and north >= r_s:
            return True
    return False


class DEMService:
    """Service for fetching USGS 3DEP DEMs and computing per-parcel stats.

    Lazy-imports ``rasterio`` and ``seamless_3dep`` so the rest of the package
    can be loaded in environments that don't have them installed.
    """

    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        self.cache_dir = Path("cache") / "dem"

    @staticmethod
    def _bbox_key(bbox: Tuple[float, float, float, float], res: int) -> str:
        payload = {"bbox": [round(c, 6) for c in bbox], "res": int(res), "src": "usgs_3dep"}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def get_dem_for_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        resolution_m: int = 10,
        verbose: bool = False,
    ) -> Path:
        """Download (or read cached) USGS 3DEP DEM for a WGS84 bbox.

        Returns the path to a single mosaiced GeoTIFF in EPSG:4326.
        Cache layout: ``cache/dem/<key>/`` for raw tiles, ``cache/dem/<key>_mosaic.tif``
        for the mosaic. The key is a short hash of (bbox, resolution).
        """
        if resolution_m not in (10, 30, 60):
            raise ValueError(
                f"USGS 3DEP only supports 10m, 30m, or 60m resolutions; got {resolution_m}"
            )

        key = self._bbox_key(bbox, resolution_m)
        tile_dir = self.cache_dir / key
        mosaic_path = self.cache_dir / f"{key}_mosaic.tif"

        if mosaic_path.exists():
            if verbose:
                print(f"--> using cached DEM mosaic ({mosaic_path.name})")
            return mosaic_path

        tile_dir.mkdir(parents=True, exist_ok=True)

        import seamless_3dep as s3dep

        if verbose:
            print(f"--> downloading USGS 3DEP DEM @ {resolution_m}m for bbox {bbox}")
        tiff_files = s3dep.get_dem(bbox, str(tile_dir), res=resolution_m)
        if not tiff_files:
            raise RuntimeError("seamless_3dep returned no DEM tiles for the given bbox")

        self._mosaic_tiles(tiff_files, mosaic_path)
        return mosaic_path

    @staticmethod
    def _mosaic_tiles(tiff_files, out_path: Path) -> None:
        import rasterio
        from rasterio.merge import merge

        srcs = [rasterio.open(t) for t in tiff_files]
        try:
            mosaic, transform = merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                    "count": mosaic.shape[0],
                }
            )
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic)
        finally:
            for s in srcs:
                s.close()

    def reproject_to_utm(
        self,
        src_path: Path,
        bbox_wgs84: Tuple[float, float, float, float],
        verbose: bool = False,
    ) -> Path:
        """Reproject a DEM GeoTIFF to a local UTM CRS so pixel sizes are in meters."""
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        utm_crs = _utm_crs_for_bbox(bbox_wgs84)
        out_path = src_path.with_name(src_path.stem + "_utm.tif")
        if out_path.exists():
            return out_path

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, utm_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update(
                {
                    "crs": utm_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )
            with rasterio.open(out_path, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=utm_crs,
                        resampling=Resampling.bilinear,
                    )
        if verbose:
            print(f"--> reprojected DEM to {utm_crs}")
        return out_path

    def compute_slope_raster(self, utm_dem_path: Path, verbose: bool = False) -> Path:
        """Compute a slope raster (degrees) from a UTM-projected DEM."""
        import rasterio

        out_path = utm_dem_path.with_name(utm_dem_path.stem + "_slope.tif")
        if out_path.exists():
            return out_path

        with rasterio.open(utm_dem_path) as src:
            dem = src.read(1).astype("float64")
            nodata = src.nodata
            transform = src.transform
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)

            mask = _nodata_mask(dem, nodata)
            if mask.any():
                dem = np.where(mask, np.nan, dem)

            # np.gradient with float spacings returns (d/dy, d/dx) for a 2D array.
            grad_y, grad_x = np.gradient(dem, pixel_size_y, pixel_size_x)
            slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
            slope_deg = np.degrees(slope_rad).astype("float32")
            slope_deg = np.where(mask, np.float32(-9999.0), slope_deg)

            meta = src.meta.copy()
            meta.update({"dtype": "float32", "count": 1, "nodata": -9999.0})
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(slope_deg, 1)

        if verbose:
            print(f"--> computed slope raster ({out_path.name})")
        return out_path

    def compute_parcel_stats(
        self,
        gdf: gpd.GeoDataFrame,
        dem_path: Path,
        slope_path: Path,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Compute per-parcel mean/stdev elevation (meters) and mean slope (degrees).

        Returns a DataFrame indexed by ``gdf.index`` with three columns:
        ``elevation_mean_m``, ``elevation_stdev_m``, ``slope_mean_deg``.
        Parcels whose geometry has no valid raster cells receive NaN.
        """
        import rasterio
        from rasterio.mask import mask as rio_mask

        with rasterio.open(dem_path) as dem_src, rasterio.open(slope_path) as slope_src:
            gdf_proj = gdf.to_crs(dem_src.crs) if gdf.crs != dem_src.crs else gdf
            dem_nodata = dem_src.nodata
            slope_nodata = slope_src.nodata

            elev_mean = np.full(len(gdf_proj), np.nan)
            elev_std = np.full(len(gdf_proj), np.nan)
            slope_mean = np.full(len(gdf_proj), np.nan)

            for i, geom in enumerate(gdf_proj.geometry.values):
                if geom is None or geom.is_empty:
                    continue
                try:
                    dem_arr, _ = rio_mask(dem_src, [geom], crop=True, all_touched=True)
                    slope_arr, _ = rio_mask(slope_src, [geom], crop=True, all_touched=True)
                except ValueError:
                    # geometry does not overlap raster
                    continue

                dem_vals = dem_arr[0]
                slope_vals = slope_arr[0]
                dem_vals = dem_vals[~_nodata_mask(dem_vals, dem_nodata)]
                slope_vals = slope_vals[~_nodata_mask(slope_vals, slope_nodata)]

                if dem_vals.size > 0:
                    elev_mean[i] = float(np.mean(dem_vals))
                    elev_std[i] = float(np.std(dem_vals))
                if slope_vals.size > 0:
                    slope_mean[i] = float(np.mean(slope_vals))

        if verbose:
            valid = np.isfinite(elev_mean).sum()
            print(f"--> computed DEM stats for {valid}/{len(gdf)} parcels")

        return pd.DataFrame(
            {
                "elevation_mean_m": elev_mean,
                "elevation_stdev_m": elev_std,
                "slope_mean_deg": slope_mean,
            },
            index=gdf.index,
        )


def _nodata_mask(arr: np.ndarray, nodata) -> np.ndarray:
    """Return a boolean mask of cells that should be treated as nodata.

    Handles three cases that ``arr != nodata`` mishandles:

    - ``nodata is None``  – mask any non-finite cells (NaN/inf).
    - ``nodata is NaN``   – ``nan != nan`` is True in IEEE 754, so the naive
      comparison would *keep* NaN cells; we mask them via ``np.isnan``.
    - ``nodata`` finite   – plain equality, plus ``np.isnan`` for any stray
      NaNs introduced by reprojection.
    """
    if nodata is None:
        return ~np.isfinite(arr)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isnan(arr)
    return (arr == nodata) | np.isnan(arr)


def _utm_crs_for_bbox(bbox: Tuple[float, float, float, float]) -> str:
    """Return a UTM proj string for the centroid of the given WGS84 bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    centroid_lon = (min_lon + max_lon) / 2
    centroid_lat = (min_lat + max_lat) / 2
    utm_zone = int((centroid_lon + 180) / 6) + 1
    hemisphere = "north" if centroid_lat >= 0 else "south"
    return f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"


def init_service_dem(settings: dict = None) -> DEMService:
    """Factory mirroring ``init_service_openstreetmap``."""
    return DEMService(settings)
