import binascii
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt, wkb
from geopy import Point
from geopy.distance import distance
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
from shapely import Polygon, MultiPolygon, LineString

from openavmkit.utilities.timing import TimingData


def get_crs(gdf, projection_type):
  """
  Returns the appropriate CRS for a GeoDataFrame based on the specified projection type.

  Parameters:
      gdf (gpd.GeoDataFrame): Input GeoDataFrame.
      projection_type (str): Type of projection ('latlon', 'equal_area', 'equal_distance').

  Returns:
      pyproj.CRS: Appropriate CRS for the specified projection type.
  """
  # Ensure the GeoDataFrame is in EPSG:4326
  gdf = gdf.to_crs("EPSG:4326")

  # Calculate the centroid of the entire GeoDataFrame

  # supress user warning:
  warnings.filterwarnings("ignore", category=UserWarning)

  # get centroid:
  lon, lat = gdf.centroid.x.mean(), gdf.centroid.y.mean()

  return get_crs_from_lat_lon(lat, lon, projection_type)


def get_crs_from_lat_lon(lat, lon, projection_type, units="m"):
  """
  Return a CRS suitable for:
    - 'latlon'       : geographic (EPSG:4326)
    - 'equal_area'   : local azimuthal equal-area (LAEA)
    - 'equal_distance': azimuthal equidistant (AEQD)
    - 'conformal'    : local UTM zone (minimal distortion in shape/angle)
  """
  if projection_type == 'latlon':
    return CRS.from_epsg(4326)

  elif projection_type == 'equal_area':
    # Lambert Azimuthal Equal‐Area about your centroid
    return CRS.from_proj4(
      f"+proj=laea +lat_0={lat} +lon_0={lon} +datum=WGS84 +units={units}"
    )

  elif projection_type == 'equal_distance':
    # Azimuthal Equidistant about your centroid
    return CRS.from_proj4(
      f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units={units}"
    )

  elif projection_type == 'conformal':
    # Use the UTM zone for minimal scale error over a small area
    # pyproj can detect the right UTM zone automatically:
    return CRS.from_user_input(f"+proj=utm +zone={int((lon+180)/6)+1} +datum=WGS84 +units={units}")

  else:
    raise ValueError(f"Unknown projection_type: {projection_type}")


def is_likely_epsg4326(gdf: gpd.GeoDataFrame) -> bool:
  """
  Checks if the GeoDataFrame is likely using EPSG:4326.
  This is a heuristic function that inspects the geometries.
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


def get_bounds_in_meters(gdf: gpd.GeoDataFrame) -> (float, float):
  """
  Returns the bounds of the GeoDataFrame in meters as (width, height)
  :param gdf: The GeoDataFrame
  :return: The bounds in meters
  """

  # Cast to a local projection appropriate for calculating distance:
  crs = get_crs(gdf, 'equal_distance')
  gdf = gdf.to_crs(crs)

  # Calculate the bounds in meters
  bounds = gdf.total_bounds

  # Calculate the width and height in meters
  width = bounds[2] - bounds[0]
  height = bounds[3] - bounds[1]

  return width, height


def get_bounds_in_feet(gdf: gpd.GeoDataFrame) -> (float, float):
  """
  Returns the bounds of the GeoDataFrame in feet as (width, height)
  :param gdf: The GeoDataFrame
  :return: The bounds in feet
  """

  meters = get_bounds_in_meters(gdf)
  feet = (meters[0] * 3.28084, meters[1] * 3.28084)
  return feet


def auto_select_grid_size(gdf: gpd.GeoDataFrame, zoom: float = 1.0) -> (int, int):
  """
  Automatically selects a grid size based on the bounds of the GeoDataFrame and the median size of a parcel
  :param gdf:
  :param zoom: how much to zoom in/out on the grid (> 1.0 means zoom out, coarser grid, < 1.0 means zoom in, finer grid)
  :return:
  """

  # Get the bounds in feet
  bounds = get_bounds_in_feet(gdf)

  # Get the median parcel size in feet
  median_size = gdf["land_area_sqft"].median()

  # Get the sqrt of the median parcel size to get its median dimension
  median_length = median_size ** 0.5

  median_length *= zoom

  # Calculate the number of rows and columns
  rows = int((bounds[1] / median_length) + 0.5)
  cols = int((bounds[0] / median_length) + 0.5)

  return rows, cols


def select_grid_size_from_size_str(gdf: gpd.GeoDataFrame, size_str: str) -> (int, int):
  try:
    if "ft" in size_str:
      num = float(size_str.replace("ft", ""))
      bounds = get_bounds_in_feet(gdf)
    elif "m" in size_str:
      num = float(size_str.replace("m", ""))
      bounds = get_bounds_in_meters(gdf)
    elif "parcel" in size_str:
      num = float(size_str.replace("parcel", ""))
      return auto_select_grid_size(gdf, num)
    else:
      raise ValueError(f"Invalid size string: {size_str}: must have 'ft', 'm', or 'parcel' suffix")
  except Exception as e:
    raise ValueError(f"Error parsing size string ({size_str}): only 'ft', 'm', 'parcel' are allowed. Error = {e}")

  if num <= 0:
    raise ValueError(f"Size must be greater than zero: {size_str}")

  rows = int((bounds[1] / num) + 0.5)
  cols = int((bounds[0] / num) + 0.5)

  return rows, cols


def scale_coords(x_in, y_in, x_max: float, y_max: float, x_min: float, y_min: float):
  x_range = abs(x_max - x_min)
  y_range = abs(y_max - y_min)
  x_out = (x_in - x_min) / x_range
  y_out = (y_in - y_min) / y_range
  return x_out, y_out


def stamp_geo_field_onto_df(df_in: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, field: str):
  df = df_in.copy()

  # Compute centroids
  df["centroid"] = df.geometry.centroid

  # Perform spatial join using 'intersects'
  df = df.set_geometry("centroid")
  joined = df.sjoin(gdf[[field, "geometry"]], predicate="intersects", how="left")

  # If multiple matches, keep only the one with the smallest polygon area
  if not joined.empty:
    # Merge to get polygon areas
    joined = joined.merge(gdf[[field, "geometry"]], left_on=field, right_on=field)
    joined["area"] = joined.geometry_y.area  # Area of intersected polygons

    # Keep only smallest area polygon per original row
    joined = joined.loc[joined.groupby(df.index)["area"].idxmin()]

    # Drop temporary columns
    joined = joined.drop(columns=["centroid", "geometry_y", "area", "index_right"])

    # Restore original geometry
    joined = joined.drop(columns=["centroid"])
    joined = joined.set_geometry(df_in.geometry.name)
  else:
    # No matches, so just drop the temporary centroid column
    df = df.drop(columns=["centroid"])
    df = df.set_geometry(df_in.geometry.name)
    joined = df

  return joined


def offset_coordinate_feet(lat, lon, lat_feet, lon_feet) -> (float, float):
  lat_km = lat_feet * 0.0003048
  lon_km = lon_feet * 0.0003048
  return offset_coordinate_km(lat, lon, lat_km, lon_km)


def offset_coordinate_miles(lat, lon, lat_miles, lon_miles) -> (float, float):
  lat_km = lat_miles * 1.60934
  lon_km = lon_miles * 1.60934
  return offset_coordinate_km(lat, lon, lat_km, lon_km)


def offset_coordinate_m(lat, lon, lat_m, lon_m) -> (float, float):
  lat_km = lat_m / 1000
  lon_km = lon_m / 1000
  return offset_coordinate_km(lat, lon, lat_km, lon_km)


def offset_coordinate_km(lat, lon, lat_km, lon_km):
  start = Point(lat, lon)

  # Latitude shift (North/South)
  new_lat = distance(kilometers=abs(lat_km)).destination(
    start, bearing=0 if lat_km >= 0 else 180
  ).latitude

  # Longitude shift (East/West)
  new_lon = distance(kilometers=abs(lon_km)).destination(
    start, bearing=90 if lon_km >= 0 else 270
  ).longitude

  return new_lat, new_lon


def distance_km(lat1, lon1, lat2, lon2):
  """
  Calculates the distance in kilometers between two latitude/longitude points.
  :param lat1: Latitude of the first point.
  :param lon1: Longitude of the first point.
  :param lat2: Latitude of the second point.
  :param lon2: Longitude of the second point.
  :return: Distance in kilometers.
  """
  return distance((lat1, lon1), (lat2, lon2)).km


def create_geo_circle(lat, lon, crs, radius_km, num_points=100):
  """
  Creates a GeoDataFrame containing a circle centered at the specified latitude and longitude.
  :param lat: The latitude of the center of the circle.
  :param lon: The longitude of the center of the circle.
  :param crs: The CRS of the circle.
  :param radius_km: The radius of the circle in kilometers.
  :param num_points: The number of points to use to approximate the circle.
  :return: A GeoDataFrame containing the circle.
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


def create_geo_rect_shape_deg(lat, lon, width_deg, height_deg, anchor_point="center"):
  """
  Creates a GeoDataFrame containing a rectangle centered at the specified latitude and longitude.
  :param lat: The latitude of the center of the rectangle.
  :param lon: The longitude of the center of the rectangle.
  :param width_deg: The width of the rectangle in degrees.
  :param height_deg: The height of the rectangle in degrees.
  :param anchor_point: The anchor point of the rectangle (center, nw).
  :return: A GeoDataFrame containing the rectangle.
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
  polygon_coords = [(nw_lon, nw_lat), (ne_lon, ne_lat), (se_lon, se_lat), (sw_lon, sw_lat), (nw_lon, nw_lat)]

  # Create a Polygon
  polygon = Polygon(polygon_coords)
  return polygon


def create_geo_rect_shape_km(lat, lon, width_km, height_km, anchor_point="center"):
  """
  Creates a GeoDataFrame containing a rectangle centered at the specified latitude and longitude.
  :param lat: The latitude of the center of the rectangle.
  :param lon: The longitude of the center of the rectangle.
  :param width_km: The width of the rectangle in kilometers.
  :param height_km: The height of the rectangle in kilometers.
  :param anchor_point: The anchor point of the rectangle (center, nw).
  :return: A GeoDataFrame containing the rectangle.
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
  polygon_coords = [(nw_lon, nw_lat), (ne_lon, ne_lat), (se_lon, se_lat), (sw_lon, sw_lat), (nw_lon, nw_lat)]

  # Create a Polygon
  polygon = Polygon(polygon_coords)
  return polygon


def create_geo_rect(lat, lon, crs, width_km, height_km, anchor_point="center"):
  """
  Creates a GeoDataFrame containing a rectangle centered at the specified latitude and longitude.
  :param lat: The latitude of the center of the rectangle.
  :param lon: The longitude of the center of the rectangle.
  :param crs: The CRS of the rectangle.
  :param width_km: The width of the rectangle in kilometers.
  :param height_km: The height of the rectangle in kilometers.
  :param anchor_point: The anchor point of the rectangle (center, nw).
  :return: A GeoDataFrame containing the rectangle.
  """

  polygon = create_geo_rect_shape_km(lat, lon, width_km, height_km, anchor_point="center")

  # Create a GeoDataFrame
  gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

  return gdf


def ensure_geometries(df, geom_col="geometry", crs=None):
  """
  Parse a DataFrame whose `geom_col` may be:
    - Shapely geometries
    - WKT strings
    - WKB bytes/bytearray
    - Hex‐encoded WKB strings (with or without "0x" prefix)
    - numpy.bytes_ scalars, memoryviews, etc.

  Returns a brand‑new GeoDataFrame with a _clean_ geometry column.
  """
  # Copy into a plain DataFrame (drops any old GeoDataFrame metadata)
  data = pd.DataFrame(df).copy()

  def _parse(val):
    # 1) Nulls
    if val is None or (isinstance(val, float) and pd.isna(val)):
      return None
    # 2) Already Shapely?
    if isinstance(val, BaseGeometry):
      return val
    # 3) Geo‑interface dicts
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
    # 5) Bytes‐like
    if isinstance(val, (bytes, bytearray, memoryview)):
      raw = val.tobytes() if isinstance(val, memoryview) else val
      return wkb.loads(raw)
    # 6) numpy bytes_ or other numpy scalar
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
  geom_series = gpd.GeoSeries(parsed.values.tolist(),
    index=data.index,
    crs=crs)

  # Drop the old column and assemble
  df_clean = data.drop(columns=[geom_col])
  return gpd.GeoDataFrame(df_clean,
    geometry=geom_series,
    crs=crs)


def clean_geometry(gdf, ensure_polygon=True, target_crs=None):
  """
    Preprocess a GeoDataFrame by diagnosing and fixing common geometry issues.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame with geometries.
        ensure_polygon (bool): If True, removes non-polygon geometries.
        target_crs (str or int, optional): If specified, ensures the GeoDataFrame is in this CRS.

    Returns:
        GeoDataFrame: A cleaned and fixed GeoDataFrame.
    """

  # Drop null geometries
  warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
  gdf = gdf[gdf.geometry.notna()]
  warnings.filterwarnings('default', 'GeoSeries.notna', UserWarning)

  # Fix invalid geometries using buffer(0)
  gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

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


def detect_triangular_lots(geom, compactness_threshold=0.85, angle_tolerance=10, min_aspect=0.5, max_aspect=2.0):
  hull = geom.convex_hull
  area_ratio = geom.area / hull.area
  if area_ratio < compactness_threshold:
    return False

  # Check approximate triangular shape
  coords = list(hull.exterior.coords[:-1])
  edges = [LineString([coords[i], coords[(i + 1) % len(coords)]]) for i in range(len(coords))]

  # Calculate angles
  def edge_angle(edge1, edge2):
    vec1 = np.array(edge1.coords[1]) - np.array(edge1.coords[0])
    vec2 = np.array(edge2.coords[1]) - np.array(edge2.coords[0])
    angle = np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
    return np.degrees(abs(angle))

  angles = [edge_angle(edges[i], edges[(i + 1) % len(edges)]) for i in range(len(edges))]
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


def get_exterior_coords(geom):
  if geom.geom_type == "Polygon":
    return list(geom.exterior.coords)
  elif geom.geom_type == "MultiPolygon":
    # Return a list of exterior coordinates for each polygon in the MultiPolygon
    return [list(poly.exterior.coords) for poly in geom.geoms]
  else:
    return None  # or handle other geometry types if necessary


def identify_irregular_parcels(gdf, verbose=False, tolerance=10, complex_threshold=12, rectangularity_threshold=0.75, elongation_threshold=5):

  if verbose:
    print(f"--> identifying irregular parcels...")

  t = TimingData()
  t.start("all")
  t.start("setup")
  old_crs = gdf.crs
  if gdf.crs.is_geographic:
    gdf = gdf.to_crs("EPSG:3857")
  gdf["simplified_geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
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
  gdf["is_geom_complex"] = (gdf["geom_vertices"].ge(complex_threshold)) & (gdf["geom_rectangularity_num"].le(rectangularity_threshold))
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
      gdf["is_geom_complex"] |
      gdf["is_geom_elongated"] |
      gdf["is_geom_triangular"]
  )

  # Fill any NA values with false:
  for field in ["is_geom_complex", "is_geom_elongated", "is_geom_triangular", "is_geom_irregular"]:
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


def geolocate_point_to_polygon(gdf: gpd.GeoDataFrame, df_in: pd.DataFrame, lat_field: str, lon_field: str, parcel_id_field: str):
  # Make a local copy
  df = df_in.copy()

  if "geometry" in df.columns:
    warnings.warn("You're doing a `lat_lon` merge of a DataFrame that also has a `geometry` column. The geometry column will be discarded.")

  # Drop fields that don't have lat/lon values
  df = df[df[lon_field].notna() & df[lat_field].notna()]

  # Strip old geometry if it exists and replace it with points corresponding to lat/lon fields
  df = df.drop(columns="geometry", errors="ignore")
  df["geometry"] = df.apply(lambda row: shapely.geometry.Point(row[lon_field], row[lat_field]), axis=1)
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
  df = df.merge(gdf_joined[[parcel_id_field, "temp_index"]], on="temp_index", how="left")

  # drop the temporary index and the point geometry:
  df = df.drop(columns=["temp_index", "geometry"])

  # now df has the parcel_id_field for each lat/lon pair in gdf

  return df