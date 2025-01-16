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