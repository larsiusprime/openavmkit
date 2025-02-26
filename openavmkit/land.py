import os
import pickle
import warnings

import osmnx as ox

import networkx as nx
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from shapely import MultiLineString
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from numpy import ma
from rasterio import features

import pandas as pd
import geopandas as gpd
from IPython.core.display_functions import display
from geopandas import GeoDataFrame
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from skimage.measure import label, regionprops, find_contours

from openavmkit.benchmark import MultiModelResults, _calc_benchmark, run_ensemble, \
  optimize_ensemble_allocation
from openavmkit.data import get_sales
from openavmkit.modeling import SingleModelResults, plot_value_surface
from openavmkit.quality_control import check_land_values
from openavmkit.utilities.data import div_field_z_safe, add_sqft_fields
from openavmkit.utilities.geometry import select_grid_size_from_size_str, get_crs
from openavmkit.utilities.plotting import plot_histogram_df
from openavmkit.utilities.settings import get_model_group_ids

from skimage import measure
from skimage.morphology import skeletonize, remove_small_objects

from openavmkit.utilities.stats import calc_correlations


def run_land_analysis(
    df_in: pd.DataFrame,
    settings: dict,
    verbose: bool = False
):
  model_group_ids = get_model_group_ids(settings)
  for model_group in model_group_ids:
    _run_land_analysis(df_in, settings, model_group, verbose)


def finalize_land_values(
    df_in: pd.DataFrame,
    settings: dict,
    generate_boundaries: bool,
    verbose: bool = False
) -> pd.DataFrame:
  model_group_ids = get_model_group_ids(settings)
  df_all_values : pd.DataFrame | None = None
  for model_group in model_group_ids:
    df_values = df_in[df_in["model_group"].eq(model_group)].copy()
    outpath = f"out/models/{model_group}/_cache/land_analysis.pickle"
    if os.path.exists(outpath):
      df_finalize = pd.read_pickle(outpath)
      df_finalize = _finalize_land_values(
        df_in,
        df_finalize,
        model_group,
        settings,
        generate_boundaries,
        verbose
      )
      df_values = df_values.merge(df_finalize[[
        "key",
        "model_market_value",
        "model_impr_value",
        "model_land_value"
      ]], on="key", how="left")
      df_values = add_sqft_fields(df_values)
      if df_all_values is None:
        df_all_values = df_values
      else:
        df_all_values = pd.concat([df_all_values, df_values], ignore_index=True)
  df_all_values.reset_index(inplace=True, drop=True)
  new_fields = [field for field in df_all_values.columns.values if field != "key"]
  df_return = df_in.copy()
  df_return = df_return[[col for col in df_return if col not in new_fields]]
  df_return = df_return.merge(df_all_values, on="key", how="left")

  os.makedirs(f"out/models/", exist_ok=True)
  gdf = gpd.GeoDataFrame(df_return, geometry="geometry")
  gdf.to_parquet(f"out/models/predictions.parquet")

  return gdf


def _finalize_land_values(
    df_orig: pd.DataFrame,
    df_in: pd.DataFrame,
    model_group: str,
    settings: dict,
    generate_boundaries: bool = False,
    verbose: bool = False
):
  df = df_in.copy()

  # Derive the final land values
  df["model_land_value"] = df["model_market_value"] * df["model_land_alloc"]
  df["model_impr_value"] = df["model_market_value"] - df["model_land_value"]

  # Apply basic sanity check / error correction to land values
  df = check_land_values(df, model_group)

  df["model_land_value_land_sqft"] = div_field_z_safe(df["model_land_value"], df["land_area_sqft"])
  df["model_market_value_land_sqft"] = div_field_z_safe(df["model_market_value"], df["land_area_sqft"])
  df["model_market_value_impr_sqft"] = div_field_z_safe(df["model_market_value"], df["bldg_area_finished_sqft"])

  # Save the results
  outpath = f"out/models/{model_group}/_cache/land_analysis_final.pickle"

  # STEP 5: Find variables correlated with land value

  df_subset = df_orig[df_orig["model_group"].eq(model_group)]
  df_sales = get_sales(df_subset, settings)
  df_sales = df_sales.merge(df[["key", "model_market_value", "model_land_value", "model_land_value_land_sqft"]], on="key", how="left")
  df_sales["model_market_value_impr_sqft"] = div_field_z_safe(df_sales["model_market_value"], df_sales["bldg_area_finished_sqft"])
  df_sales["model_market_value_land_sqft"] = div_field_z_safe(df_sales["model_market_value"], df_sales["land_area_sqft"])

  dep_vars = settings.get("modeling", {}).get("models", {}).get("main", {}).get("default", {}).get("dep_vars", [])
  dep_vars = dep_vars + ["assr_market_value", "assr_land_value", "model_market_value", "model_market_value_land_sqft", "model_market_value_impr_sqft"]

  print("LAND VALUE")
  X_corr = df_sales[["model_land_value"] + dep_vars]
  corrs = calc_correlations(X_corr)
  print("INITIAL")
  display(corrs["initial"])
  print("")
  print("FINAL)")
  display(corrs["final"])
  print("")
  print("LAND VALUE PER SQFT")
  X_corr = df_sales[["model_land_value_land_sqft"] + dep_vars]
  corrs = calc_correlations(X_corr)
  print("INITIAL")
  display(corrs["initial"])
  print("")
  print("FINAL)")
  display(corrs["final"])


  # Super tiny slivers of land will have insane $/sqft values
  df["model_market_value_land_sqft"] = div_field_z_safe(df["model_market_value"], df["land_area_sqft"])
  df["model_market_value_impr_sqft"] = div_field_z_safe(df["model_market_value"], df["bldg_area_finished_sqft"])
  df_not_tiny = df[df["land_area_sqft"].gt(5000)]

  plot_value_surface(
    "Land value per sqft",
    df_not_tiny["model_land_value_land_sqft"],
    gdf=df,
    cmap="viridis",
    norm="log"
  )

  plot_value_surface(
    "Market value per land sqft",
    df_not_tiny["model_market_value_land_sqft"],
    gdf=df,
    cmap="viridis",
    norm="log"
  )

  outpath = f"out/models/{model_group}/_images/"
  os.makedirs(outpath, exist_ok=True)
  value_field = "model_market_value_impr_sqft"

  if generate_boundaries:
    generate_raster(
      df_not_tiny,
      outpath,
      value_field,
      plot=True
    )

  return df



def save_raster(
    filepath: str,
    n_rows: int,
    n_cols: int,
    raster: np.ndarray,
    nodata_value: float,
    crs: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float
):
  pixel_width = (maxx - minx) / n_cols
  pixel_height = (maxy - miny) / n_rows
  transform = from_origin(minx, maxy, pixel_width, pixel_height)

  with rasterio.open(
      filepath,
      'w',
      driver='GTiff',
      height=n_rows,
      width=n_cols,
      count=1,
      dtype=raster.dtype,
      crs=crs,  # Use the same CRS as the input GeoDataFrame
      transform=transform,
      nodata=nodata_value
  ) as dst:
    dst.write(raster, 1)


# Generate a raster from the final land values on a map
def generate_raster(
    gdf_in: gpd.GeoDataFrame,
    path: str,
    field: str,
    plot: bool = False
):
  """
  Rasterize the value surface represented by gdf_in[field] onto a raster grid.

  Parameters:
    gdf_in (GeoDataFrame): Input GeoDataFrame.
                           IMPORTANT: It should be in a projected CRS
                           (e.g., an equal area or equal distance projection)
                           so that linear measurements (e.g., pixel sizes) are correct.
    path (str): Output file path for the raster (e.g., a GeoTIFF file).
    field (str): The name of the field in gdf_in whose values will be rasterized.
    plot (bool): If True, display the raster using matplotlib.
  """

  crs = get_crs(gdf_in, "equal_area")
  gdf = gdf_in.to_crs(crs)

  # Check that the field exists
  if field not in gdf.columns:
    raise ValueError(f"Field '{field}' not found in the provided GeoDataFrame.")

  # Get the bounds of all geometries: (minx, miny, maxx, maxy)
  minx, miny, maxx, maxy = gdf.total_bounds

  size_str = "1parcel"
  n_rows, n_cols = select_grid_size_from_size_str(gdf, size_str)
  n_rows *= 2
  n_cols *= 2
  print(f"{size_str} --> n_rows = {n_rows}, n_cols = {n_cols}")

  pixel_width = (maxx - minx) / n_cols
  pixel_height = (maxy - miny) / n_rows

  # Define the affine transform.
  # from_origin expects: (west, north, xsize, ysize). Note that ysize here is positive,
  # and the raster grid will be created top-down.
  transform = from_origin(minx, maxy, pixel_width, pixel_height)

  # Define a nodata value for the raster
  nodata_value = -9999

  if gdf[field].isna().any():
    gdf[field] = gdf[field].fillna(nodata_value)

  # Prepare the shapes (geometry, value) pairs.
  # We iterate over the rows of the GeoDataFrame and pair each geometry with its associated value.
  shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[field]))

  # Rasterize the geometries into a numpy array.
  raster = features.rasterize(
    shapes=shapes,
    out_shape=(n_rows, n_cols),
    transform=transform,
    fill=nodata_value,
    dtype='float32'
  )

  filepath = f"{path}/{field}.tiff"
  # Write the raster to disk as a GeoTIFF
  save_raster(filepath, n_rows, n_cols, raster, nodata_value, gdf.crs, minx, miny, maxx, maxy)

  # Optionally, plot the result
  if plot:
    vmax = max(0, gdf[field].quantile(0.95))
    norm = Normalize(vmin=0, vmax=vmax)

    dpi = 100
    figsize = (raster.shape[1] / dpi, raster.shape[0] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # The extent is defined as (minx, maxx, miny, maxy)
    ax.imshow(raster, cmap='viridis', extent=(minx, maxx, miny, maxy), norm=norm, origin='upper')
    fig.colorbar(label=field, mappable=ax.images[0])

    # Everywhere the raster is nodata, overlay with bright red pixels at only those locations:
    nodata_mask = ma.masked_where(raster != nodata_value, raster)

    ax.imshow(nodata_mask, cmap='Reds', extent=(minx, maxx, miny, maxy), origin='upper')

    ax.set_title("Rasterized Value Surface")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.show()

    convex_hull = gdf.union_all().convex_hull
    print(f"Convex Hull: {type(convex_hull)}")

    find_areas_from_negative_space(
      raster,
      path,
      convex_hull,
      gdf.crs,
      n_rows,
      n_cols,
      minx,
      miny,
      maxx,
      maxy,
      pixel_width,
      pixel_height,
      nodata_value
    )

    find_areas_from_negative_space(
      raster,
      path,
      convex_hull,
      gdf.crs,
      n_rows,
      n_cols,
      minx,
      miny,
      maxx,
      maxy,
      pixel_width,
      pixel_height,
      nodata_value,
      sigma = 1.0,
      threshold = 0.25,
      prefix="small_"
    )

    # find_areas_from_energy_gradient(
    #   raster,
    #   path,
    #   convex_hull,
    #   gdf.crs,
    #   n_rows,
    #   n_cols,
    #   minx,
    #   miny,
    #   maxx,
    #   maxy,
    #   pixel_width,
    #   pixel_height,
    #   nodata_value
    # )


def remove_isolated_islands(mask, min_area=200, min_major_axis=40):
  """
  Remove connected regions from a binary mask that are smaller than a given area
  or that do not have a sufficiently long major axis.

  Parameters:
    mask (np.ndarray): Boolean array representing the binary mask.
    min_area (int): Minimum area (in pixels) required to keep a region.
    min_major_axis (float): Minimum length of the major axis required.

  Returns:
    np.ndarray: A new boolean mask with small or isolated regions removed.
  """
  labeled = label(mask)
  filtered_mask = np.zeros_like(mask, dtype=bool)

  for region in regionprops(labeled):
    # Check if the region meets at least one of the criteria.
    # (You might use an "and" instead of "or" if you want to be stricter.)
    if region.area >= min_area or region.major_axis_length >= min_major_axis:
      filtered_mask[labeled == region.label] = True

  return filtered_mask


def pixel_to_coords(coords, minx, maxy, pixel_width, pixel_height):
  xs = minx + coords[:, 1] * pixel_width
  ys = maxy - coords[:, 0] * pixel_height
  return xs, ys


def extract_line_segments_from_skeleton(skeleton, minx, maxy, pixel_width, pixel_height):
  """
  Convert a skeletonized binary image (where True indicates a skeleton pixel)
  into a list of ordered line segments. This method decomposes each connected
  component of the skeleton graph into segments by splitting at junctions.

  For components that are cycles (every node has degree 2), the cycle is returned
  as a single segment.

  Parameters:
    skeleton: 2D boolean numpy array (skeletonized image)
    minx: spatial x-coordinate for pixel (row=0, col=0)
    maxy: spatial y-coordinate for pixel (row=0, col=0)
    pixel_width: spatial resolution in x
    pixel_height: spatial resolution in y

  Returns:
    segments_spatial: List of segments, where each segment is a list of (x, y) pairs.
  """
  segments_spatial = []
  # Label connected components (using 8-connectivity)
  labeled = measure.label(skeleton, connectivity=2)
  for label_val in np.unique(labeled):
    if label_val == 0:
      continue  # skip background
    comp_mask = (labeled == label_val)
    # Build a graph for this component: nodes are pixel positions (row, col)
    G = nx.Graph()
    pixels = list(map(tuple, np.argwhere(comp_mask)))
    for pix in pixels:
      G.add_node(pix)
    for (r, c) in pixels:
      # For 8-connected neighbors, check offsets in [-1,0,1] for both directions.
      for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
          if dr == 0 and dc == 0:
            continue
          neighbor = (r + dr, c + dc)
          if neighbor in G:
            G.add_edge((r, c), neighbor)

    # Now decompose the graph into segments.
    segs = []       # Will hold lists of pixel tuples
    visited_edges = set()  # Use a set to record edges we have processed (as frozensets)

    # Identify junction nodes: those with degree not equal to 2.
    junctions = [n for n in G.nodes() if G.degree(n) != 2]

    if junctions:
      # For each junction, trace out each edge that emanates from it.
      for node in junctions:
        for neighbor in G.neighbors(node):
          edge = frozenset({node, neighbor})
          if edge in visited_edges:
            continue
          seg = [node, neighbor]
          visited_edges.add(edge)
          prev = node
          current = neighbor
          # Follow the chain until we hit a junction or a dead end.
          while G.degree(current) == 2 and current not in junctions:
            # Get the neighbor of current that is not prev.
            nbs = list(G.neighbors(current))
            next_node = nbs[0] if nbs[0] != prev else nbs[1]
            edge = frozenset({current, next_node})
            if edge in visited_edges:
              break
            seg.append(next_node)
            visited_edges.add(edge)
            prev, current = current, next_node
          segs.append(seg)
    else:
      # If there are no junctions, then the component is a cycle.
      # In that case, pick an arbitrary node and trace until you return.
      arbitrary_node = next(iter(G.nodes()))
      seg = [arbitrary_node]
      current = arbitrary_node
      prev = None
      while True:
        neighbors = list(G.neighbors(current))
        # Choose the neighbor that is not the previous node (if available).
        if prev is None:
          next_node = neighbors[0]
        else:
          next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
        seg.append(next_node)
        if next_node == arbitrary_node:
          break
        prev, current = current, next_node
      segs.append(seg)

    # Convert each segment from pixel coordinates to spatial coordinates.
    for seg in segs:
      seg_arr = np.array(seg)
      xs, ys = pixel_to_coords(seg_arr, minx, maxy, pixel_width, pixel_height)
      seg_spatial = list(zip(xs, ys))
      segments_spatial.append(seg_spatial)

  return segments_spatial


def shatter_polygon_with_lines(line_segments, convex_hull):
  # 1. Convert all line segments to Shapely LineString objects, filtering invalid ones
  valid_lines = []
  for seg in line_segments:
    try:
      ls = LineString(seg)
      if ls.is_valid and not ls.is_empty:
        valid_lines.append(ls)
    except Exception as e:
      print(f"Skipping invalid segment {seg}: {e}")

  if not valid_lines:
    print("No valid line segments to process.")
    return []

  # 2. Node the line segments: Ensure all intersections are split
  try:
    noded_lines = unary_union(valid_lines)
  except Exception as e:
    print(f"Error during unary_union: {e}")
    return []

  # 3. Ensure noded_lines is iterable
  if isinstance(noded_lines, LineString):
    noded_lines = [noded_lines]
  elif isinstance(noded_lines, MultiLineString):
    noded_lines = list(noded_lines.geoms)
  else:
    print("Unexpected geometry type after unary_union.")
    return []

  # 4. Use polygonize to find all closed loops
  polygons = list(polygonize(noded_lines))

  # remove any polygons whose centroid is outside the convex hull:
  print(f"Before: {len(polygons)} polygons")

  polygons = [p for p in polygons if convex_hull.contains(p.centroid)]

  print(f"After: {len(polygons)} polygons")

  # define "small_area" in a way not dependent on the local coordinate scale & system:
  small_area = 0.0001 * convex_hull.area

  # for any tiny polygons, merge them with the nearest polygon:
  for i in range(len(polygons)):
    p = polygons[i]
    if p.area < small_area:
      # find the nearest polygon:
      nearest = None
      nearest_dist = float('inf')
      p2_j = -1
      for j in range(len(polygons)):
        p2 = polygons[j]
        if i == j:
          continue
        dist = p.distance(p2)
        if dist < nearest_dist:
          nearest = p2
          nearest_dist = dist
          p2_j = j

      # merge the two polygons:
      p = unary_union([p, nearest])

      # update the records:
      polygons[i] = None
      polygons[p2_j] = p

  polygons = [p for p in polygons if p is not None]

  # unpack any multipolygons:
  flattened_polygons = []
  for poly in polygons:
    if poly.geom_type == "Polygon":
      flattened_polygons.append(poly)
    elif poly.geom_type == "MultiPolygon":
      flattened_polygons.extend(list(poly.geoms))  # Unpack MultiPolygon into individual Polygons



  return flattened_polygons


def find_areas_from_negative_space(
    raster: np.ndarray,
    path: str,
    convex_hull,
    crs: str,
    n_rows: int,
    n_cols: int,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    pixel_width: float,
    pixel_height: float,
    nodata_value: float,
    sigma: float = 2.0,    # Standard deviation for Gaussian kernel
    threshold: float = 0.5, # Threshold on blurred mask (0-1),
    min_size: int = 50,
    island_min_area: int = 200, # Minimum area for a region to be kept (tunable)
    island_min_major: float = 40,  # Minimum major axis length for a region to be kept (tunable)
    prefix: str = ""
):

  # --- Step 1: Create and Blur the Road Mask ---
  # Create a binary mask where roads (nodata) are 1.
  road_mask : np.ndarray = (raster == nodata_value).astype(float)

  # Apply a Gaussian blur: thick roads yield higher values, thin roads lower values.
  blurred_mask : np.ndarray = gaussian_filter(road_mask, sigma=sigma)

  # Threshold the blurred mask so that only regions above a certain "thickness" are kept.
  big_road_mask : np.ndarray = (blurred_mask > threshold)

  # --- Step 2: Remove Tiny Islands ---
  # Remove small connected components that are likely noise.
  big_road_mask_clean : np.ndarray = remove_small_objects(big_road_mask, min_size=min_size)

  big_road_mask_filtered = remove_isolated_islands(big_road_mask_clean,
    min_area=island_min_area,
    min_major_axis=island_min_major)

  # --- Step 3: Compute the Skeleton (Medial Axis) ---
  # This gives you a one-pixel wide line through the middle of each road region.
  skeleton : np.ndarray = skeletonize(big_road_mask_filtered)

  # Write out each layer:
  save_raster(f"{path}{prefix}road_mask.tiff", n_rows, n_cols, road_mask, nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}{prefix}blurred_mask.tiff", n_rows, n_cols, blurred_mask, nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}{prefix}big_road_mask.tiff", n_rows, n_cols, big_road_mask.astype(float), nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}{prefix}big_road_mask_clean.tiff", n_rows, n_cols, big_road_mask_clean.astype(float), nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}{prefix}big_road_mask_filtered.tiff", n_rows, n_cols, big_road_mask_filtered.astype(float), nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}{prefix}skeleton.tiff", n_rows, n_cols, skeleton.astype(float), nodata_value, crs, minx, miny, maxx, maxy)

  segments = extract_line_segments_from_skeleton(skeleton, minx, maxy, pixel_width, pixel_height)

  # add line segments from the convex_hull to segments:
  for i in range(len(convex_hull.exterior.coords) - 1):
    segments.append([convex_hull.exterior.coords[i], convex_hull.exterior.coords[i + 1]])

  # write out the segments as a geodataframe parquet:
  gdf = gpd.GeoDataFrame(geometry=[LineString(seg) for seg in segments])
  gdf.crs = crs
  gdf.to_parquet(f"{path}{prefix}road_segments.parquet")

  polygons = shatter_polygon_with_lines(segments, convex_hull)

  # write out the polygons as a geodataframe parquet:
  gdf_poly = gpd.GeoDataFrame(geometry=polygons)
  gdf_poly.crs = crs
  gdf_poly.to_parquet(f"{path}{prefix}road_polygons.parquet")

  print("DONE")

  # # --- Step 4: Plot the Results ---
  # fig, ax = plt.subplots(figsize=(10, 10))
  #
  # # Plot the original raster (for context)
  # im = ax.imshow(raster, cmap="viridis", extent=(minx, maxx, miny, maxy), origin="upper")
  # ax.set_title("Big Road Centerlines (Skeletons)")
  # ax.set_xlabel("X coordinate")
  # ax.set_ylabel("Y coordinate")
  # fig.colorbar(im, ax=ax, label="Value")
  #
  # # Overlay the skeleton: plot skeleton pixels as red dots.
  # skel_rows, skel_cols = np.where(skeleton)
  # xs = minx + skel_cols * pixel_width
  # ys = maxy - skel_rows * pixel_height
  # ax.scatter(xs, ys, color="red", s=3, label="Road Skeleton")
  #
  # # Overlay the line segments: plot each segment as a line.
  # for segment in segments:
  #   xs, ys = zip(*segment)
  #   ax.plot(xs, ys, linewidth=2, color="blue")
  #
  # # Overlay the polygons: plot each polygon as an outlined shape with no fill.
  # for polygon in polygons:
  #   xs, ys = zip(*polygon.exterior.coords)
  #   ax.plot(xs, ys, linewidth=1, color="white")
  #
  # ax.legend()
  # plt.show()


def find_areas_from_energy_gradient(
  raster: np.ndarray,
  path: str,
  convex_hull,
  crs: str,
  n_rows: int,
  n_cols: int,
  minx: float,
  miny: float,
  maxx: float,
  maxy: float,
  pixel_width: float,
  pixel_height: float,
  nodata_value: float,
  sigma: float = 2.0,    # Standard deviation for Gaussian kernel
  low_threshold=0.1,
  high_threshold=0.3
):

  # 1. Smooth the raster to create a "heightmap" with reduced noise.
  smoothed = gaussian_filter(raster, sigma=sigma)

  # (Optional: You could apply a morphological erosion here if needed, e.g. using skimage.morphology.erosion.)

  # 2. Use the Canny edge detector to detect edges in the smoothed image.
  #    This will highlight where the value changes sharply.
  edges = canny(smoothed, low_threshold=low_threshold, high_threshold=high_threshold)

  # 3. Skeletonize the binary edge image to thin the detected boundaries to one pixel wide.
  skeleton = skeletonize(edges)

  # 4. Extract contours (continuous curves) from the skeleton.
  #    find_contours returns a list of (row, col) coordinates for each detected curve.
  boundaries = find_contours(skeleton, level=0.5)

  # (Optional: If you want to convert pixel coordinates to spatial coordinates,
  #  you can apply your affine transform here.)
  save_raster(f"{path}_smoothed.tiff", n_rows, n_cols, smoothed, nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}_edges.tiff", n_rows, n_cols, edges.astype(float), nodata_value, crs, minx, miny, maxx, maxy)
  save_raster(f"{path}_skeleton.tiff", n_rows, n_cols, skeleton.astype(float), nodata_value, crs, minx, miny, maxx, maxy)

  fig, axes = plt.subplots(1, 4, figsize=(16, 4))
  ax = axes.ravel()
  ax[0].imshow(smoothed, cmap='viridis')
  ax[0].set_title("Filled")
  ax[1].imshow(smoothed, cmap='viridis')
  ax[1].set_title("Smoothed")
  ax[2].imshow(edges, cmap='gray')
  ax[2].set_title("Canny Edges")
  ax[3].imshow(skeleton, cmap='gray')
  for boundary in boundaries:
    ax[3].plot(boundary[:, 1], boundary[:, 0], linewidth=2, color='red')
  ax[3].set_title("Skeleton & Boundaries")
  for a in ax:
    a.axis('off')
  plt.tight_layout()
  plt.show()

  return boundaries


def _run_land_analysis(
    df_in: pd.DataFrame,
    settings: dict,
    model_group: str,
    verbose: bool = False
):
  instructions = settings.get("modeling", {}).get("instructions", {})
  allocation = instructions.get("allocation", {})

  results_map = {
    "main": {},
    "hedonic": {},
    "vacant": {}
  }

  land_fields = []
  land_results: dict[str: SingleModelResults] = {}

  # STEP 1: Gather results from the main, hedonic, and vacant models
  for key in ["main", "hedonic", "vacant"]:
    short_key = key[0]
    if key == "main":
      models = instructions.get("main", {}).get("run", [])
      if "ensemble" not in models:
        models.append("ensemble")
    else:
      models = allocation.get(key, [])
    outpath = f"out/models/{model_group}/{key}"

    if verbose:
      print(f"key = {key}")
    if len(models) > 0:
      for model in models:
        if verbose:
          print(f"----> model = {model}")
        filepath = f"{outpath}/model_{model}.pickle"
        if os.path.exists(filepath):
          with open(filepath, "rb") as file:
            results = pickle.load(file)
            if results.df_multiverse is not None:
              print(f"Yes multiverse found for model: {model}")
              df = results.df_multiverse[["key"]].copy()
              df.loc[:, "prediction"] = results.pred_multi
            else:
              print(f"No multiverse found for model: {model}, using universe instead")
              df = results.df_universe[["key"]].copy()
              df.loc[:, "prediction"] = results.pred_univ
            results_map[key][model] = df
            if key != "main":
              land_results[f"{short_key}_{model}"] = results
              land_fields.append(f"{short_key}_{model}")

  df_all_alloc = results_map["main"]["ensemble"].copy()
  df_all_land_values = df_all_alloc.copy()
  df_all_land_values = df_all_land_values[["key"]].merge(df_in, on="key", how="left")
  all_alloc_names = []

  bins = 400

  # STEP 2: Calculate land allocations for each model

  for key in ["hedonic", "vacant"]:
    short_key = key[0]
    df_alloc = results_map["main"]["ensemble"].copy()
    alloc_names = []
    entries = results_map[key]
    for model in entries:

      pred_main = None  #results_map["main"].get(model)

      if pred_main is None:
        warnings.warn(f"No main model found for model: {model}, using ensemble instead")
        pred_main = results_map["main"].get("ensemble")

      pred_land = results_map[key].get(model).rename(columns={"prediction": "prediction_land"})
      df = pred_main.merge(pred_land, on="key", how="left")
      alloc_name = f"{short_key}_{model}"
      df.loc[:, alloc_name] = df["prediction_land"] / df["prediction"]

      df_alloc = df_alloc.merge(df[["key", alloc_name]], on="key", how="left")
      df_all_alloc = df_all_alloc.merge(df[["key", alloc_name]], on="key", how="left")

      df2 = df.copy().rename(columns={"prediction_land": alloc_name})
      df_all_land_values = df_all_land_values.merge(df2[["key", alloc_name]], on="key", how="left")

      alloc_names.append(alloc_name)
      all_alloc_names.append(alloc_name)

    df_alloc["allocation_ensemble"] = df_alloc[alloc_names].median(axis=1)

    plot_histogram_df(
      df=df_alloc,
      fields=alloc_names,
      xlabel="% of value attributable to land",
      ylabel="Number of parcels",
      title=f"Land allocation -- {key}",
      bins=bins,
      x_lim=(0.0, 1.0)
    )
    plot_histogram_df(
      df=df_alloc,
      fields=["allocation_ensemble"],
      xlabel="% of value attributable to land",
      ylabel="Number of parcels",
      title=f"Land allocation -- {key}, ensemble",
      bins=bins,
      x_lim=(0.0, 1.0)
    )

  plot_histogram_df(
    df=df_all_alloc,
    fields=all_alloc_names,
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- all",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  df_all_alloc["allocation_ensemble"] = df_all_alloc[all_alloc_names].median(axis=1)
  plot_histogram_df(
    df=df_all_alloc,
    fields=["allocation_ensemble"],
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- all, ensemble",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  # STEP 3: Optimize the ensemble allocation

  print(f"Putting it all together...")

  all_land_results = MultiModelResults(
    land_results,
    _calc_benchmark(land_results)
  )

  best_ensemble = optimize_ensemble_allocation(
    df=None,
    model_group=model_group,
    vacant_only=True,
    ind_var="sale_price_time_adj",
    ind_var_test="sale_price_time_adj",
    all_results=all_land_results,
    settings=settings,
    verbose=verbose
  )

  print(f"Best land ensemble --> {best_ensemble}")

  # Run the ensemble model
  ensemble_results = run_ensemble(
    df=df_in,
    model_group=model_group,
    vacant_only=True,
    hedonic=False,
    ind_var="sale_price_time_adj",
    ind_var_test="sale_price_time_adj",
    outpath=outpath,
    ensemble_list=best_ensemble,
    all_results=all_land_results,
    settings=settings,
    verbose=verbose,
    df_multiverse=df_in
  )

  all_land_results.add_model("ensemble", ensemble_results)

  print("LAND ENSEMBLE BENCHMARK:")
  print(all_land_results.benchmark.print())

  drop_alloc_names = [name for name in all_alloc_names if name not in best_ensemble]
  df_all_alloc = df_all_alloc.drop(columns=drop_alloc_names)

  plot_histogram_df(
    df=df_all_alloc,
    fields=best_ensemble,
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- best ensemble (components)",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  df_all_alloc["allocation_ensemble"] = df_all_alloc[best_ensemble].median(axis=1)
  plot_histogram_df(
    df=df_all_alloc,
    fields=["allocation_ensemble"],
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- best ensemble",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  # STEP 4: Finalize the results
  df_finalize = df_all_alloc.drop(columns=best_ensemble)
  df_finalize = df_finalize.rename(
    columns={"allocation_ensemble": "model_land_alloc", "prediction": "model_market_value"})
  df_finalize = df_finalize.merge(
    df_in[["key", "geometry", "latitude", "longitude", "land_area_sqft", "bldg_area_finished_sqft"]], on="key",
    how="left")
  df_finalize = GeoDataFrame(df_finalize, geometry="geometry", crs=df_in.crs)

  outpath = f"out/models/{model_group}/_cache/land_analysis.pickle"
  os.makedirs(os.path.dirname(outpath), exist_ok=True)
  df_finalize.to_pickle(outpath)

  df_finalize = add_sqft_fields(df_finalize)
  gdf = GeoDataFrame(df_finalize, geometry="geometry", crs=df_in.crs)
  gdf.to_parquet(f"out/models/{model_group}/_cache/land_analysis.parquet")


def get_median_value_with_kdtree(row, field, tree, valid_gdf, k=3, counter=[0]):

  counter[0] += 1
  if counter[0] % 1000 == 0:
    print(f"Processed {counter[0]} features")

  # Use the centroid of the current feature.
  centroid = row.geometry.centroid
  # Query the KDTree for the k nearest neighbors.
  distances, indices = tree.query([centroid.x, centroid.y], k=k)

  # In case only one neighbor is found (if k==1), ensure indices is iterable.
  if np.isscalar(indices):
    indices = [indices]

  # Retrieve the corresponding values.
  neighbor_values = valid_gdf.iloc[indices][field]

  # Return the median value.
  return neighbor_values.median()


def paint_and_blur_median_value(
    gdf: gpd.GeoDataFrame,
    field: str
):
  # Assume gdf is your GeoDataFrame with a 'price' column.
  # Separate features with valid prices.
  gdf_valid = gdf[~gdf[field].isna()].copy()

  # Compute the centroids of the valid features.
  # Create an array of (x, y) coordinates for the centroids.
  valid_coords = np.array([
    (geom.x, geom.y) for geom in gdf_valid.geometry.centroid
  ])

  print(f"got {len(valid_coords)} valid coords")

  # Build the KDTree from the valid feature coordinates.
  tree = cKDTree(valid_coords)

  # Create a mask for rows with missing 'price'.
  mask_missing = gdf[field].isna()
  mask_all = mask = pd.Series(True, index=gdf.index)

  print("FINAL THING")

  # Apply the KDTree-based function to fill missing values.
  gdf.loc[mask_missing, field] = gdf[mask_missing].apply(
    lambda row: get_median_value_with_kdtree(row, field, tree, gdf_valid, k=3),
    axis=1
  )

  print("SECOND VERSE, SAME AS THE FIRST")

  gdf.loc[mask_all, field] = gdf[mask_all].apply(
    lambda row: get_median_value_with_kdtree(row, field, tree, gdf_valid, k=3),
    axis=1
  )

  return gdf


def extract_lines(geom):
  """
  Given a geometry (LineString, MultiLineString or GeometryCollection),
  return a list of all LineString components.
  """
  lines = []
  if geom.geom_type == 'LineString':
    lines.append(geom)
  elif geom.geom_type == 'MultiLineString':
    lines.extend(list(geom.geoms))
  elif geom.geom_type == 'GeometryCollection':
    for part in geom:
      lines.extend(extract_lines(part))
  return lines


def get_edge_lengths(polygon):
  """
  Compute the lengths of the edges of the polygon’s minimum rotated rectangle.
  """
  mrr = polygon.minimum_rotated_rectangle
  coords = list(mrr.exterior.coords)
  edge_lengths = []
  for i in range(len(coords) - 1):
    p1, p2 = coords[i], coords[i+1]
    edge_lengths.append(LineString([p1, p2]).length)
  return edge_lengths


def is_polygon_small(poly, min_area, min_width):
  """
  Returns True if the polygon is considered too small:
    - Its area is below min_area, or
    - Its minimum rotated rectangle has an edge below min_width.
  """
  if poly.area < min_area:
    return True
  edges = get_edge_lengths(poly)
  if edges and min(edges) < min_width:
    return True
  return False


def merge_small_polygons(gdf, min_area, min_width):
  """
  Iteratively find polygons that are “too small” and merge them with
  their largest nearest neighbor.
  """
  gdf = gdf.copy().reset_index(drop=True)
  iteration = 0
  max_iterations = 1000
  while iteration < max_iterations:
    iteration += 1
    gdf['small'] = gdf.geometry.apply(lambda poly: is_polygon_small(poly, min_area, min_width))

    count_small = gdf['small'].sum()
    print(f"iteration {iteration}/1000, small = {count_small}")

    if not gdf['small'].any():
      break  # No small polygons remain

    # Process one small polygon at a time (take the first one)
    small_idx = gdf[gdf['small']].index[0]
    small_poly = gdf.loc[small_idx, 'geometry']

    # Look for candidate polygons that touch or intersect the small polygon
    candidates = gdf.drop(index=small_idx)
    touching = candidates[candidates.geometry.apply(lambda g: g.touches(small_poly) or g.intersects(small_poly))]
    if not touching.empty:
      candidate_idx = touching.geometry.area.idxmax()  # largest area among touching polygons
    else:
      # Otherwise, choose the nearest polygon by centroid distance
      small_centroid = small_poly.centroid
      candidates = candidates.copy()
      candidates['dist'] = candidates.geometry.centroid.distance(small_centroid)
      candidate_idx = candidates['dist'].idxmin()

    # Merge the small polygon with the chosen candidate
    candidate_poly = gdf.loc[candidate_idx, 'geometry']
    new_poly = candidate_poly.union(small_poly)
    gdf.at[candidate_idx, 'geometry'] = new_poly

    # Remove the small polygon from the GeoDataFrame
    gdf = gdf.drop(index=small_idx).reset_index(drop=True)
  return gdf.drop(columns='small', errors='ignore')

# --- Main Function ---

def process_county(county_name, min_area_threshold=500, min_width_threshold=10):
  """
  Given a county name (e.g., "Guilford County, North Carolina"), this function:
    1. Retrieves the county boundary polygon.
    2. Queries highways (with tag highway = motorway/primary/secondary/tertiary/trunk)
       within the county polygon.
    3. Creates a network of lines (highways + county boundary), polygonizes it,
       and clips the result to the county boundary.
    4. Merges polygons that are “too small” (by area or narrowness) into their largest
       nearest neighbor.

  Returns a GeoDataFrame of the resulting polygons (in EPSG:3857).
  """
  # --- Step 1: Retrieve County Boundary ---
  print("Retrieving county boundary…")
  county_gdf = ox.geocode_to_gdf(county_name)
  if county_gdf.empty:
    raise ValueError(f"Could not geocode county: {county_name}")
  county_polygon = county_gdf.geometry.iloc[0]

  # --- Reproject to a metric CRS (EPSG:3857) ---
  county_gdf = county_gdf.to_crs(epsg=3857)
  county_polygon = county_gdf.geometry.iloc[0]

  # --- Step 2: Query Highways Using the County Polygon ---
  tags = {'highway': ['motorway', 'primary', 'secondary', 'tertiary', 'trunk']}
  print("Querying highways from OSM…")
  # Use features_from_place since that's available in your version
  highways = ox.features_from_place(county_name, tags)
  # Filter to keep only LineString and MultiLineString geometries
  highways = highways[highways.geometry.type.isin(['LineString', 'MultiLineString'])]
  highways = highways.to_crs(epsg=3857)

  # --- Step 3: Combine Lines and Polygonize ---
  print("Extracting lines from highways and county boundary…")
  all_lines = []
  # Extract lines from highway geometries
  for geom in highways.geometry:
    all_lines.extend(extract_lines(geom))
  # Also add the county boundary (its exterior) as lines
  county_boundary_line = county_polygon.boundary
  all_lines.extend(extract_lines(county_boundary_line))

  print("Polygonizing the line network…")
  merged_lines = unary_union(all_lines)
  raw_polygons = list(polygonize(merged_lines))
  # Clip polygons to the county boundary
  polygons = [poly.intersection(county_polygon) for poly in raw_polygons if poly.intersects(county_polygon)]
  polygons = [poly for poly in polygons if poly.is_valid and not poly.is_empty and poly.area > 0]
  poly_gdf = gpd.GeoDataFrame(geometry=polygons, crs=highways.crs)

  # --- Step 4: Merge Small Polygons ---
  print("Merging small polygons…")
  poly_gdf_cleaned = merge_small_polygons(poly_gdf, min_area_threshold, min_width_threshold)

  return poly_gdf_cleaned