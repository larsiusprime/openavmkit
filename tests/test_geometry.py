import geopandas as gpd
import pandas as pd
from IPython.core.display_functions import display
from pyproj import Transformer
from shapely import Point

from openavmkit.utilities.geometry import get_crs


def _get_test_cases():
  return [
    gpd.GeoDataFrame(geometry=[Point( -77.0369,  38.9072)], crs="EPSG:4326"), # Washington DC
    gpd.GeoDataFrame(geometry=[Point(   2.3522,  48.8566)], crs="EPSG:4326"), # Paris
    gpd.GeoDataFrame(geometry=[Point( 139.6917,  35.6895)], crs="EPSG:4326"), # Tokyo
    gpd.GeoDataFrame(geometry=[Point(  -0.1278,  51.5074)], crs="EPSG:4326"), # London
    gpd.GeoDataFrame(geometry=[Point(-122.4194,  37.7749)], crs="EPSG:4326"), # San Francisco
    gpd.GeoDataFrame(geometry=[Point( 151.2093, -33.8688)], crs="EPSG:4326"), # Sydney
    gpd.GeoDataFrame(geometry=[Point( -74.0060,  40.7128)], crs="EPSG:4326"), # New York City
    gpd.GeoDataFrame(geometry=[Point(  55.2708,  25.2048)], crs="EPSG:4326"), # Dubai
    gpd.GeoDataFrame(geometry=[Point(  13.4050,  52.5200)], crs="EPSG:4326"), # Berlin
    gpd.GeoDataFrame(geometry=[Point(  37.6173,  55.7558)], crs="EPSG:4326")  # Moscow
  ]

def test_crs_convert():
  test_cases = _get_test_cases()

  def generate_expected_crs(lat, lon, projection_type):
    if projection_type == 'latlon':
      return "EPSG:4326"
    elif projection_type == 'equal_area':
      return f"+proj=aea +lat_1={lat-5} +lat_2={lat+5} +lat_0={lat} +lon_0={lon} +type=crs"
    elif projection_type == 'equal_distance':
      return f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +type=crs"
    else:
      raise ValueError("Invalid projection_type.")

  # Run the test cases for each projection type
  results = []
  projection_types = ['latlon', 'equal_area', 'equal_distance']

  for idx, gdf in enumerate(test_cases):
    for proj_type in projection_types:
      crs = get_crs(gdf, proj_type)
      expected = generate_expected_crs(gdf.geometry.y[0], gdf.geometry.x[0], proj_type)
      actual = crs.to_string()
      results.append((f"Test Case {idx + 1}", proj_type, expected, actual, expected==actual))

  results_df = pd.DataFrame(results, columns=["Test Case", "Projection Type", "Expected CRS", "Actual CRS", "Match"])

  # assert that Expected CRS = Actual CRS in all cases
  assert results_df["Expected CRS"].equals(results_df["Actual CRS"])


def test_crs_function():
  test_cases = _get_test_cases()

  def ensure_functionality(lat, lon, crs_string):
    try:
      transformer_to = Transformer.from_crs("EPSG:4326", crs_string, always_xy=True)
      transformer_back = Transformer.from_crs(crs_string, "EPSG:4326", always_xy=True)

      # Forward transformation: WGS84 to specified CRS
      x, y = transformer_to.transform(lon, lat)

      # Backward transformation: Specified CRS to WGS84
      lon_back, lat_back = transformer_back.transform(x, y)

      passing = abs(lon_back-lon) < 1e-6 and abs(lat_back-lat) < 1e-6

      if not passing:
        print(f"Failed for {lat}, {lon}, {crs_string}")
        print(f"Forward: {x}, {y}")
        print(f"Backward: {lon_back}, {lat_back}")
        print(f"Accuracy: {abs(lon_back-lon)}, {abs(lat_back-lat)}")

      return {
        "CRS Valid": True,
        "Forward Transform": (x, y),
        "Backward Transform (EPSG:4326)": (lon_back, lat_back),
        "Accuracy": (abs(lon_back-lon), abs(lat_back-lat)),
        "Pass": passing
      }

    except Exception as e:
      return {"CRS Valid": False, "Error": str(e)}

  # Run the test cases for each projection type
  results = []
  projection_types = ['latlon', 'equal_area', 'equal_distance']

  for idx, gdf in enumerate(test_cases):
    for proj_type in projection_types:
      crs = get_crs(gdf, proj_type)
      result = ensure_functionality(gdf.geometry.y[0], gdf.geometry.x[0], crs.to_string())
      results.append((
        f"Test Case {idx + 1}",
        proj_type,
        result["CRS Valid"],
        result["Forward Transform"],
        result["Backward Transform (EPSG:4326)"],
        result["Accuracy"],
        result["Pass"]
      ))

  results_df = pd.DataFrame(results, columns=["Test Case", "Projection Type", "CRS Valid", "Forward Transform", "Backward Transform (EPSG:4326)", "Accuracy", "Pass"])
  assert results_df["CRS Valid"].all() and results_df["Pass"].all()