import binascii
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import json

from typing import Optional, Tuple
from shapely import wkt, wkb
from shapely.geometry.base import BaseGeometry
from pyproj import CRS, Geod
from shapely import Polygon, MultiPolygon, LineString
import pyarrow.parquet as pq

from openavmkit.utilities.timing import TimingData

geod = Geod(ellps="WGS84")

CRS84_ALIASES = {
    "OGC:CRS84",
    "CRS84",
    "URN:OGC:DEF:CRS:OGC:1.3:CRS84",
    "HTTP://WWW.OPENGIS.NET/DEF/CRS/OGC/1.3/CRS84",
    "HTTPS://WWW.OPENGIS.NET/DEF/CRS/OGC/1.3/CRS84",
}

def get_crs(gdf: gpd.GeoDataFrame, projection_type: str) -> pyproj.CRS:
    """Returns the appropriate CRS for a GeoDataFrame based on the specified projection
    type.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.
    projection_type : str
        Type of projection ('latlon', 'equal_area', 'equal_distance').

    Returns
    -------
    pyproj.CRS
        Appropriate CRS for the specified projection type.
    """
    # Ensure the GeoDataFrame is in EPSG:4326
    gdf = gdf.to_crs("EPSG:4326")

    # Calculate the centroid of the entire GeoDataFrame

    # supress user warning:
    warnings.filterwarnings("ignore", category=UserWarning)

    # get centroid:
    lon, lat = gdf.centroid.x.mean(), gdf.centroid.y.mean()

    return get_crs_from_lat_lon(lat, lon, projection_type)


def get_crs_from_lat_lon(
    lat: float, lon: float, projection_type: str, units: str = "m"
) -> pyproj.CRS:
    """Return a Coordinate Reference System (CRS) suitable for the requested projection type at the given
    latitude/longitude location.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    projection_type : str
        The desired projection type. Legal values are "latlon", "equal_area", "equal_distance", and "conformal"
    units : str
        The desired units. Legal values are "ft" and "m"

    Returns
    -------
    CRS
        The appropriately Coordinate Reference System

    Notes
    -----
    The following kind of CRS will be returned for each of the ``projection_type`` values:

      - 'latlon'       : geographic (EPSG:4326)
      - 'equal_area'   : local azimuthal equal-area (LAEA)
      - 'equal_distance': azimuthal equidistant (AEQD)
      - 'conformal'    : local UTM zone (minimal distortion in shape/angle)
    """
    if projection_type == "latlon":
        return CRS.from_epsg(4326)

    elif projection_type == "equal_area":
        # Lambert Azimuthal Equal‐Area about your centroid
        return CRS.from_proj4(
            f"+proj=laea +lat_0={lat} +lon_0={lon} +datum=WGS84 +units={units}"
        )

    elif projection_type == "equal_distance":
        # Azimuthal Equidistant about your centroid
        return CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units={units}"
        )

    elif projection_type == "conformal":
        # Use the UTM zone for minimal scale error over a small area
        # pyproj can detect the right UTM zone automatically:
        return CRS.from_user_input(
            f"+proj=utm +zone={int((lon+180)/6)+1} +datum=WGS84 +units={units}"
        )

    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")


def is_likely_epsg4326(gdf: gpd.GeoDataFrame) -> bool:
    """Checks if the GeoDataFrame is likely using EPSG:4326.

    This is a heuristic function that inspects the geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to check

    Returns
    -------
    bool
        True if it's likely EPSG:4326, False if it's not.

    """
    # Check if geometries have lat/lon coordinates within typical ranges
    for geom in gdf.geometry.head(10):  # Check first 10 geometries for efficiency
        if geom.is_empty:
            continue
        if not geom.bounds:
            continue
        min_x, min_y, max_x, max_y = geom.bounds
        # Longitude range: -180 to 180, Latitude range: -90 to 90
        if not (-180 <= min_x <= 180 and -180 <= max_x <= 180):
            return False
        if not (-90 <= min_y <= 90 and -90 <= max_y <= 90):
            return False
    return True


def safe_normalize_to_4326(crs):
    """
    Returns True if .to_crs(4326) is fidelity-safe (datum already WGS84 geographic or projected on WGS84),
    False otherwise.
    """
    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        # Geographic WGS84?
        name = (crs.name or "").upper()
        auth = crs.to_authority()
        if auth == ("EPSG", "4326"):
            return True
        if name.find("WGS 84") >= 0 or str(crs).upper().find("OGC:CRS84") >= 0:
            return True
        return False
    else:
        # Projected: check the base/geodetic CRS
        base = crs.get_geodetic_crs()
        if base and "WGS 84" in (base.name or ""):
            return True   # e.g., Web Mercator on WGS84
        return False


def offset_coordinate_feet(lat: float, lon: float, lat_feet: float, lon_feet: float) -> tuple[float, float]:
    """Offset a lat/long coordinate by the designated number of feet

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    lat_feet : float
        Number of feet to offset latitude by
    lon_feet : float
        Number of feet to offset longitude by

    Returns
    -------
    tuple[float, float]
        The offset latitude, longitude pair (in degrees)
    """
    lat_km = lat_feet * 0.0003048
    lon_km = lon_feet * 0.0003048
    return offset_coordinate_km(lat, lon, lat_km, lon_km)


def offset_coordinate_miles(lat, lon, lat_miles, lon_miles) -> tuple[float, float]:
    """Offset a lat/long coordinate by the designated number of miles

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    lat_miles : float
        Number of miles to offset latitude by
    lon_miles : float
        Number of miles to offset longitude by

    Returns
    -------
    tuple[float, float]
        The offset latitude, longitude pair (in degrees)
    """
    lat_km = lat_miles * 1.60934
    lon_km = lon_miles * 1.60934
    return offset_coordinate_km(lat, lon, lat_km, lon_km)


def offset_coordinate_m(lat: float, lon: float, lat_m: float, lon_m: float) -> tuple[float, float]:
    """Offset a lat/long coordinate by the designated number of meters

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    lat_m : float
        Number of meters to offset latitude by
    lon_m : float
        Number of meters to offset longitude by

    Returns
    -------
    tuple[float, float]
        The offset latitude, longitude pair (in degrees)
    """
    lat_km = lat_m / 1000
    lon_km = lon_m / 1000
    return offset_coordinate_km(lat, lon, lat_km, lon_km)



def offset_coordinate_km(lat: float, lon: float, lat_km: float, lon_km: float) -> tuple[float, float]:
  # shift north/south
  lon_ns, lat_ns, _ = geod.fwd(
    lon, lat,
    0 if lat_km >= 0 else 180,
    abs(lat_km) * 1000
  )

  # shift east/west
  lon_ew, lat_ew, _ = geod.fwd(
    lon, lat,
    90 if lon_km >= 0 else 270,
    abs(lon_km) * 1000
  )

  new_lat = lat_ns
  new_lon = lon_ew

  return new_lat, new_lon



def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """
  Calculates the distance in kilometers between two latitude/longitude points.
  :param lat1: Latitude of the first point.
  :param lon1: Longitude of the first point.
  :param lat2: Latitude of the second point.
  :param lon2: Longitude of the second point.
  :return: Distance in kilometers.
  """
  _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
  return dist_m / 1000.0



def create_geo_circle(lat: float, lon: float, crs: pyproj.CRS, radius_km: float, num_points: int=100) -> gpd.GeoDataFrame:
    """Creates a GeoDataFrame containing a circle centered at the specified latitude and
    longitude.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    crs : CRS
        Coordinate Reference System
    radius_km : float
        Radius of the circle, in kilometers
    num_points : int
        Number of points used to approximate the circle

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the circle
    """
    # Create a list of points around the circle
    points = []
    for i in range(num_points):
        angle = 2 * 3.14159 * i / num_points
        x = radius_km * math.cos(angle)
        y = radius_km * math.sin(angle)
        pt_lat, pt_lon = offset_coordinate_km(lat, lon, x, y)
        points.append(shapely.Point(pt_lon, pt_lat))

    points.append(points[0])
    polygon = shapely.Polygon(points)

    # Create a GeoDataFrame from the points
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
    return gdf


def create_geo_rect_shape_deg(lat: float, lon: float, width_deg: float, height_deg: float, anchor_point: str="center") -> Polygon:
    """Creates a GeoDataFrame containing a rectangle centered at the specified latitude
    and longitude.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    height_deg : float
        The height of the rectangle in degrees
    width_deg : float
        The height of the rectangle in degrees
    anchor_point : str
        The anchor point of the rectangle ("center" or "nw")

    Returns
    -------
    Polygon
        A shapely Polygon object representing the rectangle.
    """

    if anchor_point == "center":
        off_nw_x = -width_deg / 2
        off_sw_x = -width_deg / 2

        off_ne_x = width_deg / 2
        off_se_x = width_deg / 2

        off_nw_y = height_deg / 2
        off_ne_y = height_deg / 2

        off_se_y = -height_deg / 2
        off_sw_y = -height_deg / 2
    elif anchor_point == "nw":
        off_nw_x = 0
        off_sw_x = 0

        off_ne_x = width_deg
        off_se_x = width_deg

        off_nw_y = 0
        off_ne_y = 0

        off_se_y = -height_deg
        off_sw_y = -height_deg
    else:
        raise ValueError("Invalid anchor point. Choose 'center' or 'nw'.")

    # Calculate the four corners of the rectangle
    nw_lat, nw_lon = lat + off_nw_y, lon + off_nw_x  # NW
    ne_lat, ne_lon = lat + off_ne_y, lon + off_ne_x  # NE
    se_lat, se_lon = lat + off_se_y, lon + off_se_x  # SE
    sw_lat, sw_lon = lat + off_sw_y, lon + off_sw_x  # SW

    # Order: NW → NE → SE → SW → NW (to close polygon)
    polygon_coords = [
        (nw_lon, nw_lat),
        (ne_lon, ne_lat),
        (se_lon, se_lat),
        (sw_lon, sw_lat),
        (nw_lon, nw_lat),
    ]

    # Create a Polygon
    polygon = Polygon(polygon_coords)
    return polygon


def create_geo_rect_shape_km(lat: float, lon: float, width_km: float, height_km: float, anchor_point: str="center") -> Polygon:
    """Creates a GeoDataFrame containing a rectangle centered at the specified latitude
    and longitude.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    width_km : float
        Width of the rectangle in kilometers
    height_km : float
        Height of the rectangle in kilometers
    anchor_point : str
        Anchor point of the rectangle ("center", "now")

    Returns
    -------
    Polygon
        A polygon representing a rectangle.
    """

    if anchor_point == "center":
        off_nw_x = -width_km / 2
        off_sw_x = -width_km / 2

        off_ne_x = width_km / 2
        off_se_x = width_km / 2

        off_nw_y = height_km / 2
        off_ne_y = height_km / 2

        off_se_y = -height_km / 2
        off_sw_y = -height_km / 2
    elif anchor_point == "nw":
        off_nw_x = 0
        off_sw_x = 0

        off_ne_x = width_km
        off_se_x = width_km

        off_nw_y = 0
        off_ne_y = 0

        off_se_y = -height_km
        off_sw_y = -height_km
    else:
        raise ValueError("Invalid anchor point. Choose 'center' or 'nw'.")

    # Calculate the four corners of the rectangle
    nw_lat, nw_lon = offset_coordinate_km(lat, lon, off_nw_y, off_nw_x)  # NW
    ne_lat, ne_lon = offset_coordinate_km(lat, lon, off_ne_y, off_ne_x)  # NE
    se_lat, se_lon = offset_coordinate_km(lat, lon, off_se_y, off_se_x)  # SE
    sw_lat, sw_lon = offset_coordinate_km(lat, lon, off_sw_y, off_sw_x)  # SW

    # Order: NW → NE → SE → SW → NW (to close polygon)
    polygon_coords = [
        (nw_lon, nw_lat),
        (ne_lon, ne_lat),
        (se_lon, se_lat),
        (sw_lon, sw_lat),
        (nw_lon, nw_lat),
    ]

    # Create a Polygon
    polygon = Polygon(polygon_coords)
    return polygon


def create_geo_rect(lat: float, lon: float, crs: pyproj.CRS, width_km: float, height_km: float, anchor_point: str="center") -> gpd.GeoDataFrame:
    """Creates a GeoDataFrame containing a rectangle centered at the specified latitude
    and longitude.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    crs : CRS
        Coordinate Reference System
    width_km : float
        Width in kilometers
    height_km : float
        Height in kilometers
    anchor_point : str, optional
        Anchor point of the rectangle ("center", "nw")

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the rectangle.
    """

    polygon = create_geo_rect_shape_km(
        lat, lon, width_km, height_km, anchor_point=anchor_point
    )

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

    return gdf


def ensure_geometries(df: pd.DataFrame, geom_col: str="geometry", crs: pyproj.CRS=None) -> gpd.GeoDataFrame:
    """Parse a DataFrame whose `geom_col` may be:

      - Shapely geometries
      - WKT strings
      - WKB bytes/bytearray
      - Hex-encoded WKB strings (with or without "0x" prefix)
      - numpy.bytes_ scalars, memoryviews, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    geom_col : str
        name of the geometry column
    crs : CRS, optional
        Coordinate Reference System

    Returns
    -------
    gpd.GeoDataFrame
        a brand-new GeoDataFrame with a _clean_ geometry column.
    """
    # Copy into a plain DataFrame (drops any old GeoDataFrame metadata)
    # Handle both pandas and bodo.pandas DataFrames
    if hasattr(df, 'to_pandas'):
        data = df.to_pandas().copy()
    else:
        data = df.copy()
    
    if crs is None:
        if "geometry" in df:
            try:
                crs = df.crs
            except AttributeError:
                crs = None

    def _parse(val):
        # 1) Nulls
        if val is None or (hasattr(val, '__float__') and pd.isna(val)):
            return None
        # 2) Already Shapely?
        if isinstance(val, BaseGeometry):
            return val
        # 3) Geo-interface dicts
        if hasattr(val, "__geo_interface__"):
            from shapely.geometry import shape

            return shape(val)
        # 4) Strings: try WKT first, then hex WKB
        if isinstance(val, str):
            s = val.strip()
            if s.lower().startswith("0x"):
                s = s[2:]
            try:
                return wkt.loads(val)
            except Exception:
                raw = binascii.unhexlify(s)
                return wkb.loads(raw)
        # 5) Bytes-like
        if isinstance(val, (bytes, bytearray, memoryview)):
            raw = val.tobytes() if isinstance(val, memoryview) else val
            return wkb.loads(raw)
        # 6) numpy bytes or other numpy scalar
        if isinstance(val, np.generic):
            try:
                b = bytes(val)
                return wkb.loads(b)
            except Exception:
                pass
        # 7) Anything with tobytes()
        if hasattr(val, "tobytes"):
            try:
                raw = val.tobytes()
                return wkb.loads(raw)
            except Exception:
                pass

        raise TypeError(f"Cannot parse geometry of type {type(val)}")

    # Parse into a plain pandas Series of Shapely objects
    parsed = data[geom_col].apply(_parse)

    # Build a GeoSeries (so GeoPandas trusts it)
    geom_series = gpd.GeoSeries(parsed.values.tolist(), index=data.index, crs=crs)

    # Drop the old column and assemble
    df_clean = data.drop(columns=[geom_col])
    return gpd.GeoDataFrame(df_clean, geometry=geom_series, crs=crs)


def clean_geometry(gdf: gpd.GeoDataFrame, ensure_polygon: bool=True, target_crs: str|int=None):
    """Preprocess a GeoDataFrame by diagnosing and fixing common geometry issues.

    Parameters
    ----------
    gdf : GeoDataFrame
        The input GeoDataFrame with geometries.
    ensure_polygon : bool
        If True, removes non-polygon geometries.
    target_crs : str | int, optional
        If specified, ensures the GeoDataFrame is in this CRS.

    Returns
    -------
    gpd.GeoDataFrame
        A cleaned and fixed GeoDataFrame.
    """
    
    gdf = ensure_geometries(gdf, crs=gdf.crs)

    # Drop null geometries
    warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)
    gdf = gdf[gdf.geometry.notna()]
    warnings.filterwarnings("default", "GeoSeries.notna", UserWarning)

    # Fix invalid geometries using buffer(0)
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )

    # Remove empty geometries
    gdf = gdf[~gdf.is_empty]

    # Ensure all polygons are properly closed (for Polygons and MultiPolygons)
    def close_polygon(geom):
        if isinstance(geom, Polygon):
            if not geom.exterior.is_closed:
                return Polygon(list(geom.exterior.coords) + [geom.exterior.coords[0]])
        return geom

    gdf["geometry"] = gdf["geometry"].apply(close_polygon)

    # Remove geometries with fewer than 4 points (invalid for polygons)
    def valid_polygon(geom):
        if isinstance(geom, Polygon):
            return len(geom.exterior.coords) >= 4
        elif isinstance(geom, MultiPolygon):
            for poly in list(geom.geoms):
                if len(poly.exterior.coords) < 4:
                    return False
            return True
        return False

    gdf = gdf[gdf.geometry.apply(valid_polygon)]
    
    # Remove non-polygon geometries if ensure_polygon is True
    if ensure_polygon:
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    
    # Ensure the CRS is consistent
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def detect_triangular_lots(
    geom: shapely.geometry.base.BaseGeometry, compactness_threshold: float=0.85, angle_tolerance: float=10, min_aspect: float=0.5, max_aspect: float=2.0
) -> bool:
    """
    Determine if a geometry is approximately triangular based on compactness,
    hull vertex angles, and bounding box aspect ratio.

    This function evaluates a shape by:

    1. Computing the ratio of its area to its convex hull area (compactness).
    2. Checking that the convex hull has at most three non-nearly-180° angles within
       the specified tolerance, indicating a triangle-like hull.
    3. Verifying the bounding box aspect ratio is within [min_aspect, max_aspect].

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input polygonal geometry to test (must have an 'area' and 'convex_hull').
    compactness_threshold : float, default 0.85
        Minimum allowed ratio of geom.area to its convex_hull.area.
        Shapes with more concave boundaries (lower ratios) are rejected.
    angle_tolerance : float, default 10
        Degrees of tolerance from 180° for hull vertex angles.
    min_aspect : float, default 0.5
        Minimum width/height ratio of the geometry's bounding box.
    max_aspect : float, default 2.0
        Maximum width/height ratio of the geometry's bounding box.

    Returns
    -------
    bool
        True if geom passes all triangularity checks; False otherwise.

    """
    hull = geom.convex_hull
    area_ratio = geom.area / hull.area
    if area_ratio < compactness_threshold:
        return False

    # Check approximate triangular shape
    coords = list(hull.exterior.coords[:-1])
    edges = [
        LineString([coords[i], coords[(i + 1) % len(coords)]])
        for i in range(len(coords))
    ]

    # Calculate angles
    def edge_angle(edge1, edge2):
        vec1 = np.array(edge1.coords[1]) - np.array(edge1.coords[0])
        vec2 = np.array(edge2.coords[1]) - np.array(edge2.coords[0])
        angle = np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
        return np.degrees(abs(angle))

    angles = [
        edge_angle(edges[i], edges[(i + 1) % len(edges)]) for i in range(len(edges))
    ]
    near_180 = sum(abs(180 - angle) < angle_tolerance for angle in angles)
    if len(edges) - near_180 > 3:
        return False

    # Check bounding box aspect ratio
    bounds = geom.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    aspect_ratio = width / height
    if not (min_aspect <= aspect_ratio <= max_aspect):
        return False

    return True


def get_exterior_coords(geom: Polygon) -> list | None:
    """Gets a list of all the exterior coordinates, regardless of whether the Geometry is a Polygon or MultiPolygon

    Parameters
    ----------
    geom : shapely.Polygon
        The shapely polygon whose exterior coordinates you want

    Returns
    -------
    list
        The list of exterior coordinates
    """
    if geom.geom_type == "Polygon":
        return list(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        # Return a list of exterior coordinates for each polygon in the MultiPolygon
        return [list(poly.exterior.coords) for poly in geom.geoms]
    else:
        return None  # or handle other geometry types if necessary


def identify_irregular_parcels(
    gdf: gpd.GeoDataFrame,
    verbose: bool=False,
    tolerance: int=10,
    complex_threshold: int=12,
    rectangularity_threshold: float=0.75,
    elongation_threshold: float=5,
) -> gpd.GeoDataFrame:
    """
    Detect and flag irregular parcel geometries based on shape metrics.

    Applies a sequence of geometric tests to identify triangular, overly
    complex, or elongated parcel shapes.  The input GeoDataFrame is temporarily
    projected to EPSG:3857 for distance-based operations, then restored to its
    original CRS.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain a ``geometry`` column of Polygon geometries, plus
        precomputed fields:

        - ``geom_rectangularity_num`` : numeric rectangularity measure (0–1).
        - ``geom_aspect_ratio``     : width/height ratio of each parcel.
    verbose : bool, default False
        If True, print timing and progress for each processing phase.
    tolerance : int, default 10
        Simplification tolerance (projection units) for reducing geometry
        complexity before analysis.
    complex_threshold : int, default 12
        Minimum vertex count (post-simplification) for a parcel to be
        considered 'complex' when rectangularity is low.
    rectangularity_threshold : float, default 0.75
        Maximum rectangularity below which a high-vertex-count geometry is
        flagged as irregular.
    elongation_threshold : float, default 5
        Minimum bounding-box aspect ratio that qualifies a parcel as
        'elongated'.

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of the input with these added columns:

        - ``is_geom_triangular`` : bool flag for approximate triangular shapes.
        - ``geom_vertices``      : int count of vertices after simplification.
        - ``is_geom_complex``    : bool flag for complex non-rectangular shapes.
        - ``is_geom_elongated``  : bool flag for elongated shapes.
        - ``is_geom_irregular``  : bool flag if any irregular condition is met.

    Notes
    -----
    - The original CRS is preserved in the final output.
    - Triangle detection delegates to :func:`detect_triangular_lots`.
    - Complex shapes satisfy both: vertex count ≥ ``complex_threshold`` AND
      rectangularity ≤ ``rectangularity_threshold``.
    - Elongation is evaluated on the axis-aligned bounding box of each parcel.
    """
    if verbose:
        print(f"--> identifying irregular parcels...")

    t = TimingData()
    t.start("all")
    t.start("setup")
    old_crs = gdf.crs
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:3857")
    gdf["simplified_geometry"] = gdf.geometry.simplify(
        tolerance, preserve_topology=True
    )
    t.stop("setup")

    t.start("tri")
    gdf["is_geom_triangular"] = gdf["simplified_geometry"].apply(detect_triangular_lots)
    t.stop("tri")
    if verbose:
        _t = t.get("setup")
        print(f"----> simplified geometry...({_t:.2f}s)")
        _t = t.get("tri")
        print(f"----> identified triangular parcels...({_t:.2f}s)")

    # Detect complex geometry based on rectangularity and vertex count
    t.start("complex")
    gdf["geom_vertices"] = gdf["simplified_geometry"].apply(
        lambda geom: len(geom.exterior.coords) if geom.geom_type == "Polygon" else 0
    )
    gdf["is_geom_complex"] = (gdf["geom_vertices"].ge(complex_threshold)) & (
        gdf["geom_rectangularity_num"].le(rectangularity_threshold)
    )
    t.stop("complex")
    if verbose:
        _t = t.get("complex")
        print(f"----> identified complex geometry...({_t:.2f}s)")
    
    t.start("long")
    gdf["is_geom_elongated"] = gdf["geom_aspect_ratio"].ge(elongation_threshold)
    t.stop("long")
    if verbose:
        _t = t.get("long")
        print(f"----> identified elongated parcels...({_t:.2f}s)")
    
    t.start("finish")
    # Combine criteria for irregular lots
    gdf["is_geom_irregular"] = (
        gdf["is_geom_complex"] | gdf["is_geom_elongated"] | gdf["is_geom_triangular"]
    )
    
    # Fill any NA values with false:
    for field in [
        "is_geom_complex",
        "is_geom_elongated",
        "is_geom_triangular",
        "is_geom_irregular",
    ]:
        gdf[field] = gdf[field].fillna(False)
        gdf[field] = gdf[field].astype(bool)

    gdf = gdf.drop(columns="simplified_geometry")
    gdf = gdf.to_crs(old_crs)
    t.stop("finish")
    if verbose:
        _t = t.get("finish")
        print(f"----> finished up...({_t:.2f}s)")
    t.stop("all")
    if verbose:
        _t = t.get("all")
        print(f"--> identified irregular parcels (total)...({_t:.2f}s)")

    return gdf


def geolocate_point_to_polygon(
    gdf: gpd.GeoDataFrame,
    df_in: pd.DataFrame,
    lat_field: str,
    lon_field: str,
    parcel_id_field: str,
) -> pd.DataFrame:
    """
    Assign each point (latitude/longitude) in a DataFrame to a containing polygon ID.

    Converts input latitude/longitude columns into Point geometries, performs a
    spatial join against a GeoDataFrame of parcel polygons, and returns the
    original DataFrame augmented with the matching parcel identifier.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame of polygon geometries.  Must include:

        - A geometry column of type Polygon or MultiPolygon.
        - The column named by `parcel_id_field` containing unique parcel IDs.
    df_in : pandas.DataFrame
        Input DataFrame with at least the latitude and longitude columns.
        Any existing `geometry` column will be dropped.
    lat_field : str
        Name of the column in `df_in` containing latitude values.
    lon_field : str
        Name of the column in `df_in` containing longitude values.
    parcel_id_field : str
        Name of the parcel ID column in `gdf` to attach to each point.

    Returns
    -------
    pandas.DataFrame
        Copy of `df_in` with:

        - Any preexisting `geometry` column removed.
        - A new column `parcel_id_field` containing the ID of the polygon in `gdf`
          that contains each point.  Rows with no containing polygon will have NaN.

    Notes
    -----
    - Input DataFrame is temporarily converted to a GeoDataFrame with Point
      geometries; its CRS is set to match `gdf`.
    - Spatial join uses the 'within' predicate to ensure points fall inside
      parcels.
    - A temporary index column is used to preserve original row order.
    """
    # Make a local copy
    df = df_in.copy()

    if "geometry" in df.columns:
        warnings.warn(
            "You're doing a `lat_lon` merge of a DataFrame that also has a `geometry` column. The geometry column will be discarded."
        )

    # Drop fields that don't have lat/lon values
    df = df[df[lon_field].notna() & df[lat_field].notna()]

    # Strip old geometry if it exists and replace it with points corresponding to lat/lon fields
    df = df.drop(columns="geometry", errors="ignore")
    df["geometry"] = df.apply(
        lambda row: shapely.geometry.Point(row[lon_field], row[lat_field]), axis=1
    )
    df = gpd.GeoDataFrame(df, geometry="geometry")

    # Match the CRS
    df.set_crs(gdf.crs, inplace=True)

    # reset index and set a "temp_index" of numerical index values
    df.reset_index(drop=True)
    df["temp_index"] = df.index

    gdf_keys = gdf[[parcel_id_field, "geometry"]]

    # deduplicate:
    gdf_keys = gdf_keys.drop_duplicates(subset=[parcel_id_field])

    # perform the spatial join, each lat/lon matched to the parcel it's inside of
    gdf_joined = gpd.sjoin(df, gdf_keys, how="left", predicate="within")

    # now we have a DataFrame that matches each row in df_in to a `parcel_id_field` in gdf

    # merge parcel_id_field back onto df:
    df = df.merge(
        gdf_joined[[parcel_id_field, "temp_index"]], on="temp_index", how="left"
    )

    # drop the temporary index and the point geometry:
    df = df.drop(columns=["temp_index", "geometry"])

    # now df has the parcel_id_field for each lat/lon pair in gdf

    return df


def _normalize_crs_value(crs_value) -> CRS:
    # Strings: catch OGC/URN/URL variants of CRS84 and map to EPSG:4326
    if isinstance(crs_value, str):
        s = crs_value.strip()
        if s.upper() in CRS84_ALIASES:
            return CRS.from_epsg(4326)
        # Otherwise let pyproj try
        return CRS.from_user_input(s)

    # Dicts: typically PROJJSON
    if isinstance(crs_value, dict):
        try:
            return CRS.from_user_input(crs_value)
        except Exception:
            pass

    # Already a CRS?
    if isinstance(crs_value, CRS):
        return crs_value

    # Give pyproj one more chance
    return CRS.from_user_input(crs_value)


def detect_crs_from_parquet(path: str, geom_col: str = "geometry") -> Tuple[Optional[CRS], str]:
    """
    Return (crs, geometry_column_used). crs is a pyproj.CRS or None.
    Reads only GeoParquet metadata; does not infer from coordinates.
    """
    pf = pq.ParquetFile(path)

    def _extract_crs(md: dict) -> Tuple[Optional[CRS], str]:
        if not md:
            return None, geom_col

        geo_raw = (md.get(b"geo") if isinstance(md, dict) else None)
        if not geo_raw:
            return None, geom_col

        info = json.loads(geo_raw.decode("utf-8"))
        cols = info.get("columns", {}) or {}

        # Find metadata for the requested geometry column, or the first geometry-ish column
        colmeta = cols.get(geom_col)
        geom_name = geom_col
        if colmeta is None and isinstance(cols, dict):
            for name, cm in cols.items():
                if isinstance(cm, dict) and str(cm.get("encoding", "")).upper() in {"WKB", "WKT", "GEOARROW"}:
                    geom_name = name
                    colmeta = cm
                    break
            else:
                return None, geom_name

        # GeoParquet: "crs" (string/projjson) or legacy "crs_wkt"
        crs_value = colmeta.get("crs") or colmeta.get("crs_wkt")

        # OPTIONAL: treat missing CRS as CRS84 per common GeoParquet convention
        # (Uncomment if you want this behavior.)
        # if not crs_value:
        #     try:
        #         return CRS.from_epsg(4326), geom_name
        #     except Exception:
        #         return None, geom_name

        if not crs_value:
            return None, geom_name

        try:
            return _normalize_crs_value(crs_value), geom_name
        except Exception:
            return None, geom_name

    # Schema-level metadata (usual place)
    schema_md = getattr(pf.schema_arrow, "metadata", None) or {}
    crs, used_geom = _extract_crs(schema_md)
    if crs:
        return crs, used_geom

    # Fallback: some writers stash geo metadata in file metadata
    file_md = pf.metadata
    if file_md and file_md.metadata:
        return _extract_crs(file_md.metadata)

    return None, geom_col