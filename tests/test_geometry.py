import geopandas as gpd
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from pyproj import Transformer
from shapely import Point, Polygon

from openavmkit.data import perform_spatial_joins
from openavmkit.modeling import plot_value_surface
from openavmkit.utilities.assertions import dfs_are_equal
from openavmkit.utilities.geometry import get_crs, offset_coordinate_miles, create_geo_rect, create_geo_circle, \
  offset_coordinate_km


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
    gpd.GeoDataFrame(geometry=[Point(  37.6173,  55.7558)], crs="EPSG:4326"), # Moscow
    gpd.GeoDataFrame(geometry=[Point( -95.3698,  29.7604)], crs="EPSG:4326")  # Houston
  ]

def test_crs_convert():
  test_cases = _get_test_cases()

  def generate_expected_crs(lat, lon, projection_type):
    if projection_type == 'latlon':
      return "EPSG:4326"
    elif projection_type == 'equal_area':
      return f"+proj=aea +lat_1={lat-5} +lat_2={lat+5} +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +type=crs"
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

  print("")
  print("===")
  display(results_df["Expected CRS"].values)
  print("---")
  display(results_df["Actual CRS"].values)

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


def test_spatial_join_contains_centroid():

  ice_cream_map = {
    "0": "vanilla",
    "1": "chocolate",
    "2": "strawberry",
    "3": "mint",
    "4": "rocky road"
  }

  data = {
    "key": [],
    "x": [],
    "y": [],
    "ice_cream": []
  }
  geo_data = {
    "key": [],
    "lat": [],
    "lon": [],
    "geometry": []
  }

  origin_lat = 29.7604
  origin_lon = -95.3698

  # set the south-east point 10 km south-east of the north-west point:
  origin_se_lat, origin_se_lon = offset_coordinate_km(origin_lat, origin_lon, -10, 10)

  i = 0
  for x in range(0, 11):
    for y in range(0, 11):
      data["key"].append(f"{x}-{y}")
      data["x"].append(x)
      data["y"].append(y)
      data["ice_cream"].append(ice_cream_map[str(i % 5)])

      lat, lon = offset_coordinate_km(origin_lat, origin_lon, -y, x)

      geo_data["key"].append(f"{x}-{y}")
      geo_data["lat"].append(lat)
      geo_data["lon"].append(lon)

      # construct a polygonal (square) parcel centered on lat/lon, with a width/height of 1 km:
      nw_lat, nw_lon = offset_coordinate_km(lat, lon, lat_km= 0.5, lon_km=-0.5)  # Top-left
      ne_lat, ne_lon = offset_coordinate_km(lat, lon, lat_km= 0.5, lon_km= 0.5)  # Top-right
      se_lat, se_lon = offset_coordinate_km(lat, lon, lat_km=-0.5, lon_km= 0.5)  # Bottom-right
      sw_lat, sw_lon = offset_coordinate_km(lat, lon, lat_km=-0.5, lon_km=-0.5)  # Bottom-left

      nw = (nw_lon, nw_lat)
      ne = (ne_lon, ne_lat)
      se = (se_lon, se_lat)
      sw = (sw_lon, sw_lat)

      geo_data["geometry"].append(Polygon([nw, ne, se, sw, nw]))

      i += 1

  ice_cream_to_hex = {
    "vanilla": "#F5DEB3",
    "chocolate": "#8B4513",
    "strawberry": "#FF69B4",
    "mint": "#98FB98",
    "rocky road": "#8B4513"
  }

  df = pd.DataFrame(data=data)
  df["rgb"] = df["ice_cream"].map(ice_cream_to_hex)

  gdf_parcels = gpd.GeoDataFrame(data=geo_data, crs="EPSG:4326")
  gdf_parcels = gdf_parcels.merge(df, on="key")

  nw_quadrant_lat, nw_quadrant_lon = offset_coordinate_km(origin_lat, origin_lon, -2.5, 2.5)
  se_quadrant_lat, se_quadrant_lon = offset_coordinate_km(origin_se_lat, origin_se_lon, 2.5, -2.5)

  gdf_square = create_geo_rect(nw_quadrant_lat, nw_quadrant_lon, "EPSG:4326", 5.01, 5.01)
  gdf_circle = create_geo_circle(se_quadrant_lat, se_quadrant_lon, "EPSG:4326", 2.51)

  # Map colors to the GeoDataFrame

  gdf_parcels = gdf_parcels.to_crs("EPSG:4326")
  gdf_square = gdf_square.to_crs("EPSG:4326")
  gdf_circle = gdf_circle.to_crs("EPSG:4326")

  gdf_square["color_square"] = "#FF0000"
  gdf_circle["color_circle"] = "#0000FF"

  dataframes = {
    "geo_parcels": gdf_parcels,
    "square": gdf_square,
    "circle": gdf_circle
  }

  s_geom = [
    "square",
    "circle"
  ]

  gdf = perform_spatial_joins(s_geom, dataframes)

  gdf["rgb"] = gdf["color_square"].combine_first(gdf["color_circle"]).combine_first(gdf["rgb"])

  df_results = gdf[["key", "rgb"]]
  data_expected = {
    'key': ['0-0', '0-1', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8', '0-9', '0-10',
            '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10',
            '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10',
            '3-0', '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10',
            '4-0', '4-1', '4-2', '4-3', '4-4', '4-5', '4-6', '4-7', '4-8', '4-9', '4-10',
            '5-0', '5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7', '5-8', '5-9', '5-10',
            '6-0', '6-1', '6-2', '6-3', '6-4', '6-5', '6-6', '6-7', '6-8', '6-9', '6-10',
            '7-0', '7-1', '7-2', '7-3', '7-4', '7-5', '7-6', '7-7', '7-8', '7-9', '7-10',
            '8-0', '8-1', '8-2', '8-3', '8-4', '8-5', '8-6', '8-7', '8-8', '8-9', '8-10',
            '9-0', '9-1', '9-2', '9-3', '9-4', '9-5', '9-6', '9-7', '9-8', '9-9', '9-10',
            '10-0', '10-1', '10-2', '10-3', '10-4', '10-5', '10-6', '10-7', '10-8', '10-9', '10-10'],
     "rgb": ['#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#8B4513', '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3',
             '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3', '#8B4513',
             '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#98FB98', '#8B4513', '#F5DEB3', '#8B4513', '#FF69B4',
             '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#8B4513', '#F5DEB3', '#8B4513', '#FF69B4', '#98FB98',
             '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#F5DEB3', '#8B4513', '#FF69B4', '#98FB98', '#8B4513',
             '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#8B4513', '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3',
             '#8B4513', '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3', '#8B4513', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#8B4513',
             '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3', '#8B4513', '#FF69B4', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#FF69B4',
             '#98FB98', '#8B4513', '#F5DEB3', '#8B4513', '#FF69B4', '#98FB98', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#98FB98',
             '#8B4513', '#F5DEB3', '#8B4513', '#FF69B4', '#98FB98', '#8B4513', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#8B4513',
             '#F5DEB3', '#8B4513', '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3', '#8B4513', '#FF69B4', '#98FB98', '#8B4513', '#F5DEB3']
  }
  df_expected = pd.DataFrame(data=data_expected)

  assert dfs_are_equal(df_expected, df_results)

  fig, ax = plt.subplots(figsize=(10, 6))
  gdf_parcels.plot(color=gdf["rgb"], edgecolor="black", ax=ax)

  gdf_square.plot(ax=ax, color="none", edgecolor="black")

  # plot the circle, connect the lines:
  gdf_circle.plot(ax=ax, color="none", edgecolor="black")

  # Create a legend manually
  plt.show()