import math
import warnings

import geopandas as gpd
import shapely
from geopy import Point
from geopy.distance import distance
from pyproj import CRS
from shapely import Polygon, MultiPolygon


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

  if projection_type == 'latlon':
    # Return WGS 84 (EPSG:4326)
    return CRS.from_epsg(4326)

  elif projection_type == 'equal_area':
    # Return an equal-area projection centered on the centroid, ensuring meters
    return CRS.from_proj4(
      f"+proj=aea +lat_1={lat-5} +lat_2={lat+5} +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    )

  elif projection_type == 'equal_distance':
    # Return an Azimuthal Equidistant projection centered on the centroid (meters)
    return CRS.from_proj4(
      f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    )

  else:
    raise ValueError("Invalid projection_type. Choose 'latlon', 'equal_area', or 'equal_distance'.")


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
    joined = joined.set_geometry(df_in.geometry.name)
  else:
    # No matches, so just drop the temporary centroid column
    df = df.drop(columns=["centroid"])
    return df  # Return original dataframe with field unmodified (all NaN)

  return joined


def offset_coordinate_miles(lat, lon, lat_miles, lon_miles) -> (float, float):
  lat_km = lat_miles * 1.60934
  lon_km = lon_miles * 1.60934
  return offset_coordinate_km(lat, lon, lat_km, lon_km)


def offset_coordinate_km(lat, lon, lat_km, lon_km):
  """Offsets a coordinate by lat_km km north and lon_km km east."""
  start = Point(lat, lon)  # ✅ Correct (lat, lon) ordering

  # Move latitude (North/South) - No need for scaling
  new_lat = distance(kilometers=abs(lat_km)).destination(start, bearing=0 if lat_km >= 0 else 180).latitude

  # Scale longitude movement by cos(latitude)
  lon_adjusted_km = lon_km / abs(math.cos(math.radians(lat)))

  # Move longitude (East/West) with proper scaling
  new_lon = distance(kilometers=abs(lon_adjusted_km)).destination(start, bearing=90 if lon_km >= 0 else 270).longitude

  return new_lat, new_lon


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



def create_geo_rect(lat, lon, crs, width_km, height_km):
  """
  Creates a GeoDataFrame containing a rectangle centered at the specified latitude and longitude.
  :param lat: The latitude of the center of the rectangle.
  :param lon: The longitude of the center of the rectangle.
  :param crs: The CRS of the rectangle.
  :param width_km: The width of the rectangle in kilometers.
  :param height_km: The height of the rectangle in kilometers.
  :return: A GeoDataFrame containing the rectangle.
  """
  # Calculate the four corners of the rectangle
  nw_lat, nw_lon = offset_coordinate_km(lat, lon, height_km / 2, -width_km / 2)  # NW
  ne_lat, ne_lon = offset_coordinate_km(lat, lon, height_km / 2, width_km / 2)   # NE
  se_lat, se_lon = offset_coordinate_km(lat, lon, -height_km / 2, width_km / 2)  # SE
  sw_lat, sw_lon = offset_coordinate_km(lat, lon, -height_km / 2, -width_km / 2) # SW

  # Order: NW → NE → SE → SW → NW (to close polygon)
  polygon_coords = [(nw_lon, nw_lat), (ne_lon, ne_lat), (se_lon, se_lat), (sw_lon, sw_lat), (nw_lon, nw_lat)]

  # Create a Polygon
  polygon = Polygon(polygon_coords)

  # Create a GeoDataFrame
  gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

  return gdf


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

# def detect_triangular_lots(geom, compactness_threshold=0.85, angle_tolerance=10, min_aspect=0.5, max_aspect=2.0):
#   hull = geom.convex_hull
#   area_ratio = geom.area / hull.area
#   if area_ratio < compactness_threshold:
#     return False
#
#   # Check approximate triangular shape
#   coords = list(hull.exterior.coords[:-1])
#   edges = [LineString([coords[i], coords[(i + 1) % len(coords)]]) for i in range(len(coords))]
#
#   # Calculate angles
#   def edge_angle(edge1, edge2):
#     vec1 = np.array(edge1.coords[1]) - np.array(edge1.coords[0])
#     vec2 = np.array(edge2.coords[1]) - np.array(edge2.coords[0])
#     angle = np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
#     return np.degrees(abs(angle))
#
#   angles = [edge_angle(edges[i], edges[(i + 1) % len(edges)]) for i in range(len(edges))]
#   near_180 = sum(abs(180 - angle) < angle_tolerance for angle in angles)
#   if len(edges) - near_180 > 3:
#     return False
#
#   # Check bounding box aspect ratio
#   bounds = geom.bounds
#   width = bounds[2] - bounds[0]
#   height = bounds[3] - bounds[1]
#   aspect_ratio = width / height
#   if not (min_aspect <= aspect_ratio <= max_aspect):
#     return False
#
#   return True
#
#
# def identify_irregular_parcels(gdf, tolerance=10, complex_threshold=12, rectangularity_threshold=0.75, elongation_threshold=5):
#
#   old_crs = gdf.crs
#   if gdf.crs.is_geographic:
#     gdf = gdf.to_crs("EPSG:3857")
#   tolerance = 10
#   gdf["simplified_geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
#   gdf["is_triangular"] = gdf["simplified_geometry"].apply(detect_triangular_lots)
#
#   # Detect complex geometry based on rectangularity and vertex count
#   gdf["num_vertices"] = gdf["simplified_geometry"].apply(
#     lambda geom: len(geom.exterior.coords) if geom.type == "Polygon" else 0
#   )
#   gdf["is_complex_geometry"] = (gdf["num_vertices"].ge(complex_threshold)) & (gdf["rectangularity"].le(rectangularity_threshold))
#
#   # Aspect ratio to detect elongated lots
#   bounds = np.array(gdf["simplified_geometry"].apply(lambda geom: geom.bounds).to_list())
#   widths = bounds[:, 2] - bounds[:, 0]
#   heights = bounds[:, 3] - bounds[:, 1]
#   gdf["aspect_ratio"] = np.maximum(widths, heights) / np.minimum(widths, heights)
#   gdf["is_elongated"] = gdf["aspect_ratio"].ge(elongation_threshold)
#
#   # Combine criteria for irregular lots
#   gdf["is_irregular"] = (
#       gdf["is_complex_geometry"] |
#       gdf["is_elongated"] |
#       gdf["is_triangular"]
#   )
#
#   gdf = gdf.drop(columns="simplified_geometry")
#   gdf = gdf.to_crs(old_crs)
#
#   return gdf
