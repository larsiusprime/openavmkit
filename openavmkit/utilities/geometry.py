import geopandas as gpd
from pyproj import CRS


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

  # Calculate the centroid of the GeoDataFrame
  centroid = gdf.union_all().centroid
  lat, lon = centroid.y, centroid.x

  if projection_type == 'latlon':
    # Return WGS 84 (EPSG:4326)
    return CRS.from_epsg(4326)

  elif projection_type == 'equal_area':
    # Return an equal-area projection centered on the centroid
    return CRS.from_proj4(
      f"+proj=aea +lat_1={lat-5} +lat_2={lat+5} +lat_0={lat} +lon_0={lon}"
    )

  elif projection_type == 'equal_distance':
    # Return an Azimuthal Equidistant projection centered on the centroid
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

  return (width, height)


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

