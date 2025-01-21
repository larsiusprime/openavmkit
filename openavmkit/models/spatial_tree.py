import colorsys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.neighbors import KDTree
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw

from openavmkit.utilities.geometry import get_crs
from openavmkit.utilities.stats import calc_mse


############################
# Power Law + COD
############################


def power_law(size: float, a: float, b: float) -> float:
  return a * (size ** -b)


def power_law_np(sizes: np.ndarray, a: float, b: float) -> np.ndarray:
  return a * (sizes ** -b)


def linear_fit(x: float, slope: float, b: float):
  return x * slope + b


def linear_fit_np(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
  return x * slope + intercept


def compute_cod(predicted: np.ndarray, actual: np.ndarray) -> float:
  """
  Compute Coefficient of Dispersion (COD), in percent.
  COD = 100 * (1/n) * sum( |ratio_i - median(ratio)| / median(ratio) )

  ratio_i = predicted / actual

  Returns COD as a float, e.g. 15.2 for 15.2%
  """
  ratio = predicted / actual
  med_ratio = np.median(ratio)
  if med_ratio == 0:
    return 999999.0  # fallback for degenerate case
  abs_diff = np.abs(ratio - med_ratio)
  cod = 100.0 * np.mean(abs_diff / med_ratio)
  return cod


############################
# KDTree Node
############################

class KDTreeNode:
  def __init__(
      self,
      min_x: float, max_x: float,
      min_y: float, max_y: float,
      data_indices: np.ndarray,
      depth: int = 0
  ):
    indent_arrow = "--" * depth
    indent_arrow += ">"
    print(f"{indent_arrow}New KDTreeNode ({min_x}, {max_x}, {min_y}, {max_y}) @ depth {depth} with len {len(data_indices)}")

    self.min_x = min_x
    self.max_x = max_x
    self.min_y = min_y
    self.max_y = max_y

    self.data_indices = data_indices  # indexes of all points in this node
    self.depth = depth

    # Fitted parameters
    self.equation: Optional[str] = None
    self.a: Optional[float] = None
    self.b: Optional[float] = None
    self.slope: Optional[float] = None
    self.intercept: Optional[float] = None
    self.mse: Optional[float] = float('nan')

    # We'll store a centroid for blending
    self.center_x: Optional[float] = None
    self.center_y: Optional[float] = None

    # Children
    self.children: List['KDTreeNode'] = []

  def is_leaf(self) -> bool:
    return len(self.children) == 0

  def contains(self, x: float, y: float) -> bool:
    return (
        self.min_x <= x < self.max_x and
        self.min_y <= y < self.max_y
    )

  def get_density(self) -> float:
    return len(self.data_indices) / ((self.max_x - self.min_x) * (self.max_y - self.min_y))

  def get_max_value(self, value: str) -> float:
    return self._get_value(value, True)

  def get_min_value(self, value: str) -> float:
    return self._get_value(value, False)

  def _get_value(self, value: str, is_max: bool) -> float:
    amount = 0.0
    if value == "a":
      amount = self.a
    if value == "b":
      amount = self.b
    if value == "len":
      if len(self.children) == 0:
        amount = len(self.data_indices)
      else:
        amount = 0
    if value == "density":
      if len(self.children) == 0:
        density = self.get_density()
        print(f"density = {density}")
        if not np.isinf(density) and not np.isnan(density):
          amount = density
      else:
        amount = 0
    if value == "mse":
      amount = self.mse
    for child in self.children:
      if child is not None:
        child_value = child._get_value(value, is_max)
        if is_max:
          if child_value > amount:
            amount = child_value
        else:
          if child_value < amount:
            amount = child_value
    return amount

############################
# Fitting & COD for a Node
############################

def fit_power_law_node(df: pd.DataFrame, node: KDTreeNode, ind_var: str, size_var: str):
  """
  Fit power law Price ~ a * (size^(-b)) using only the data in node.data_indices.
  Update node.a, node.b in-place.
  """
  idx = node.data_indices

  sizes = df[size_var].values[idx]
  prices = df[ind_var].values[idx]

  # We want a>0, b in (0,1)
  lower_bounds = (1e-9, 0.0)
  upper_bounds = (1e9, 0.999999)

  # initial guesses: a ~ median of prices, b ~ 0.5
  p0 = [np.median(prices), 0.5]

  try:
    # Fit power law first
    popt, pcov = curve_fit(
      power_law,
      sizes,
      prices,
      p0=p0,
      bounds=(lower_bounds, upper_bounds),
      maxfev=2000
    )
    a, b = popt
    power_law_mse = calc_mse(power_law_np(sizes, a, b), prices)
    node.a = a
    node.b = b

    # Fit linear as well and compare
    p0 = [0.5, np.median(prices)]

    popt, pcov = curve_fit(
      linear_fit,
      sizes,
      prices,
      p0=p0,
      maxfev=2000
    )
    a, b = popt
    linear_fit_mse = calc_mse(linear_fit_np(sizes, a, b), prices)
    node.slope = a
    node.intercept = b

    if power_law_mse < linear_fit_mse:
      node.equation = "power_law"
      node.mse = power_law_mse
    else:
      node.equation = "linear"
      node.mse = linear_fit_mse

    indent_arrow = "--" * node.depth + ">"
    print(f"{indent_arrow}Fitted a={node.a:.2f}, b={node.b:.2f}, MSE={node.mse:.2f}")
  except RuntimeError:
    # fallback if curve_fit fails
    node.a = np.median(prices)
    node.b = 0.5
    node.mse = float('inf')


############################
# Splitting a Node
############################

def split_node(node: KDTreeNode, df: gpd.GeoDataFrame, split_vertical:int, min_children: int) -> List[KDTreeNode]:
  """
  Split node's bounding box into 2 quadrants, either vertically or horizontal
  Return list of up 2 child KDTreeNode's with data assigned.
  """
  mid_x = 0.5 * (node.min_x + node.max_x)
  mid_y = 0.5 * (node.min_y + node.max_y)

  if split_vertical:
    boxes = [
      (node.min_x, node.max_x, node.min_y, mid_y),  # Bottom
      (node.min_x, node.max_x, mid_y, node.max_y)   # Top
    ]
  else:
    boxes = [
      (node.min_x, mid_x, node.min_y, node.max_y),  # Left
      (mid_x, node.max_x, node.min_y, node.max_y)   # Right
    ]

  child_nodes = []

  centroids = df.geometry.centroid
  x_vals = centroids.x.values[node.data_indices]
  y_vals = centroids.y.values[node.data_indices]

  for (min_x, max_x, min_y, max_y) in boxes:
    mask = (
        (x_vals >= min_x) & (x_vals < max_x) &
        (y_vals >= min_y) & (y_vals < max_y)
    )
    child_idx = node.data_indices[mask]
    if len(child_idx) == 0:
      continue  # skip empty child
    child = KDTreeNode(min_x, max_x, min_y, max_y, child_idx, node.depth + 1)
    if len(child.data_indices) < min_children:
      return []  # skip split if any child is too small
    child_nodes.append(child)

  return child_nodes


############################
# Building the KDTree
############################

def build_kdtree(
    df: gpd.GeoDataFrame,
    root_node: KDTreeNode,
    ind_var: str,
    size_var: str,
    min_samples: int,
    max_depth: int
):
  """
  Recursively fit local power law
  """
  # 1. Fit local power-law
  fit_power_law_node(df, root_node, ind_var, size_var)

  # 2. Check split conditions
  if (
      len(root_node.data_indices) > min_samples and
      root_node.depth < max_depth
  ):
    # 3. Attempt to split
    vertical_children = None
    horizontal_children = None

    depth = root_node.depth
    indent_arrow = "--" * depth + ">"
    print(f"{indent_arrow}Splitting node with {len(root_node.data_indices)} data points")

    for split_vertical in [True, False]:
      print(f"{indent_arrow} --> Splitting {'vertically' if split_vertical else 'horizontally'}")
      children = split_node(root_node, df, split_vertical, min_samples)
      print(f"{indent_arrow} --> Got {len(children)} children")
      if len(children) == 1:
        # If only 1 child actually had data, no real split
        continue
      elif len(children) == 0:
        continue

      for child in children:
        fit_power_law_node(df, child, ind_var, size_var)

      if split_vertical:
        vertical_children = children
      else:
        horizontal_children = children

    if vertical_children is not None and horizontal_children is not None:
      vertical_mse = sum([child.mse for child in vertical_children])
      horizontal_mse = sum([child.mse for child in horizontal_children])
      print(f"{indent_arrow} --> Vertical MSE: {vertical_mse:.2f}, Horizontal MSE: {horizontal_mse:.2f}")
      if vertical_mse <= horizontal_mse:
        best_children = vertical_children
      else:
        best_children = horizontal_children
    elif vertical_children is not None:
      best_children = vertical_children
    elif horizontal_children is not None:
      best_children = horizontal_children
    else:
      return

    root_node.children = best_children
    for child in best_children:
      build_kdtree(df, child, ind_var, size_var, min_samples, max_depth)


############################
# Leaf Gathering & Centroids (for blending)
############################

def gather_leaves(node: KDTreeNode) -> List[KDTreeNode]:
  """
  Recursively collect all leaf nodes in the kdtree.
  """
  if node.is_leaf():
    return [node]
  else:
    leaves = []
    for child in node.children:
      leaves.extend(gather_leaves(child))
    return leaves


def compute_leaf_centroid(df: pd.DataFrame, node: KDTreeNode) -> Tuple[float, float]:
  """
  Compute (center_x, center_y) as the mean of x/y among the node's data points.
  """
  idx = node.data_indices
  x_vals = df.geometry.x.values[idx]
  y_vals = df.geometry.y.values[idx]
  c_x = x_vals.mean()
  c_y = y_vals.mean()
  return c_x, c_y


def assign_leaf_centroids(df: pd.DataFrame, root_node: KDTreeNode) -> List[KDTreeNode]:
  """
  Traverse all leaves, compute centroids, store in node.center_x / node.center_y.
  Return the list of leaves for convenience.
  """
  leaves = gather_leaves(root_node)
  for leaf in leaves:
    c_x, c_y = compute_leaf_centroid(df, leaf)
    leaf.center_x = c_x
    leaf.center_y = c_y
  return leaves


############################
# Building a KD-Tree of Leaves for Fast Blending
############################

def build_leaf_kdtree(leaves: List[KDTreeNode]) -> KDTree:
  """
  Build a sklearn KDTree from the leaf centroids for fast nearest-neighbor lookups.
  Return the KDTree and optionally the centroids array for reference.
  """
  coords = []
  for leaf in leaves:
    if leaf.center_x is None or leaf.center_y is None:
      raise ValueError("Leaf centroid not assigned.")
    coords.append([leaf.center_x, leaf.center_y])
  coords = np.array(coords)
  tree = KDTree(coords)
  return tree, coords


def visualize_kdtree(root: KDTreeNode, max_pixels_wide: int, max_pixels_tall: int):
  # Normalize dimensions to maintain aspect ratio
  width = root.max_x - root.min_x
  height = root.max_y - root.min_y
  aspect_ratio = width / height

  # max_a = np.log(root.get_max_value("a"))
  # min_a = np.log(root.get_min_value("a"))
  # max_b = np.log(root.get_max_value("b"))
  # min_b = np.log(root.get_min_value("b"))
  max_e = np.log(root.get_max_value("mse"))
  min_e = np.log(root.get_min_value("mse"))

  eq = root.equation

  # range_a = max_a - min_a
  # range_b = max_b - min_b
  range_e = max_e - min_e

  #print(f"range_a: {range_a}, range_b: {range_b}, range_e: {range_e}")

  if max_pixels_wide / max_pixels_tall > aspect_ratio:
    canvas_width = int(max_pixels_tall * aspect_ratio)
    canvas_height = max_pixels_tall
  else:
    canvas_width = max_pixels_wide
    canvas_height = int(max_pixels_wide / aspect_ratio)

  # Scale factors for normalization
  x_scale = canvas_width / width
  y_scale = canvas_height / height

  def normalize_x(x):
    return int((x - root.min_x) * x_scale)

  def normalize_y(y):
    return int((y - root.min_y) * y_scale)

  # Initialize the canvas
  image = Image.new("RGB", (canvas_width, canvas_height), "white")
  draw = ImageDraw.Draw(image)

  def calculate_color(_child: KDTreeNode):

    if len(_child.data_indices) == 0:
      return 255, 255, 255

    # a = (np.log(_child.a) - min_a) / range_a
    # b = (np.log(_child.b) - min_b) / range_b
    c = (np.log(_child.mse) - min_e) / range_e

    if np.isnan(c):
      c = 0.0

    # red = int(a * 255)
    # blue = int(b * 255)
    red = int(c * 255)
    green = 0
    blue = 0

    if _child.equation == "power_law":
      blue = 255
    else:
      blue = 0

    return red, 0, blue


  # Recursive function to draw the kdtree
  def draw_node(node, base_color):
    # Normalize and draw the current node's bounding box
    x0 = normalize_x(node.min_x)
    y0 = normalize_y(node.min_y)
    x1 = normalize_x(node.max_x)
    y1 = normalize_y(node.max_y)

    draw.rectangle([x0, y0, x1, y1], outline=base_color, width=1, fill=base_color)

    # If the node is a leaf, draw the data_indices in the center
    if node.is_leaf():
      center_x = (x0 + x1) // 2
      center_y = (y0 + y1) // 2

      #txt_a = (np.log(node.a) - min_a) / range_a
      #txt_b = (np.log(node.b) - min_b) / range_b
      txt_eq = "P" if node.equation == "power_law" else "L"
      txt_e = np.sqrt(node.mse)/100
      #(np.log(node.mse) - min_e) / range_e

      #text = f"a={txt_a:.2f}\nb={txt_b:.2f}\ne={txt_e:.2f}"
      text = f"{txt_eq}\ne={txt_e:,.0f}"
      draw.text((center_x, center_y), text, fill="white", anchor="mm", stroke_width=1, stroke_fill=(0,0,0))

    # Recurse into children with adjusted colors
    for i, child in enumerate(node.children):
      child_color = calculate_color(child)
      draw_node(child, child_color)

  # Starting colors for the quadrants
  start_colors = {
    "NW": (0x80, 0x00, 0x00),
    "NE": (0x00, 0x80, 0x00),
    "SE": (0x00, 0x00, 0x80),
    "SW": (0x80, 0x80, 0x80),
  }

  # Draw the root node and its descendants
  for i, child in enumerate(root.children):
    direction = ["NW", "NE", "SW", "SE"][i]
    draw_node(child, start_colors[direction])

  # Save or display the image
  image.show()
  return image


############################
# Prediction with or without blending
############################

def kdtree_predict(
    df: pd.DataFrame,
    root_node: KDTreeNode,
    lat: float,
    lon: float,
    size: float,
    blending: bool = False,
    leaves: Optional[List[KDTreeNode]] = None,
    leaf_tree: Optional[KDTree] = None,
    k: int = 3
) -> float:
  """
  Predict price at (lat, lon) for lot 'size'.

  If blending=False:
    - We descend the kdtree to find the single leaf that contains (lat, lon) and use (a, b).

  If blending=True:
    - We assume we have a list of all leaves (`leaves`) plus a KDTree (`leaf_tree`).
    - We do a quick KNN query on the KD-tree to find the k nearest leaf centroids.
    - Then do IDW on their (a, b) to produce a smoothed parameter estimate.
  """

  if not blending:
    # "No-blend" approach: descend the kdtree to find the leaf
    node = root_node
    while not node.is_leaf():
      found_child = None
      for child in node.children:
        if child.contains(lat, lon):
          found_child = child
          break
      if found_child is None:
        # (lat, lon) not found in any child box
        # (maybe on boundary). We'll just stay in 'node'
        break
      node = found_child

    # Use node's fit
    a, b = node.a, node.b
    if a is None or b is None:
      return 0.0
    return power_law(size, a, b)

  else:
    if leaves is None or leaf_tree is None:
      raise ValueError("Must provide 'leaves' and 'leaf_tree' for blending.")

    # 1. Query the KDTree for nearest leaf centroids
    # KDTrees require 2D input, so we pass [[lat, lon]]
    dist, idx = leaf_tree.query([[lat, lon]], k=k)
    # dist, idx shapes: (1, k)
    dist = dist[0]  # shape (k,)
    idx = idx[0]    # shape (k,)

    # If the nearest leaf is exactly distance=0, just use it
    if dist[0] < 1e-12:
      leaf = leaves[idx[0]]
      if leaf.a is None or leaf.b is None:
        return 0.0
      return power_law(size, leaf.a, leaf.b)

    # 2. IDW weights with small epsilon
    epsilon = 1e-12
    w = 1.0 / (dist + epsilon)
    wsum = np.sum(w)
    wnorm = w / wsum

    # 3. Weighted average of 'a' and 'b'
    a_vals = np.array([leaves[i].a for i in idx])
    b_vals = np.array([leaves[i].b for i in idx])

    a_blend = np.sum(wnorm * a_vals)
    b_blend = np.sum(wnorm * b_vals)

    return power_law(size, a_blend, b_blend)


def run_spatial_tree(
    df_in: gpd.GeoDataFrame,
    ind_var: str,
    size_var: str,
    cod_threshold: float = 20.0,
    min_samples: int = 10,
    max_depth: int = 6
):
  # 1) Generate Synthetic Data
  # np.random.seed(42)
  # n_data = 2000
  # lat_vals = np.random.uniform(30.0, 31.0, n_data)
  # lon_vals = np.random.uniform(-100.0, -99.0, n_data)
  # sizes = np.random.uniform(1000, 100000, n_data)  # lot sizes

  # # Suppose the true underlying model is a=300, b=0.4, plus some noise
  # true_a, true_b = 300.0, 0.4
  # def ground_truth_price(sz):
  #   return power_law(sz, true_a, true_b)
  #
  # base_price = ground_truth_price(sizes)
  # noise = np.random.normal(0, 0.2, n_data) * base_price
  # sale_prices = base_price + noise
  #
  # df = pd.DataFrame({
  #   'lat': lat_vals,
  #   'lon': lon_vals,
  #   'lot_size': sizes,
  #   'sale_price': sale_prices
  # })

  crs_equal_distance = get_crs(df_in, "equal_distance")
  df = df_in.to_crs(crs_equal_distance)

  # 2) Build the Root Node (full bounding box)

  min_y, max_y = df.bounds['miny'].min(), df.bounds['maxy'].max()
  min_x, max_x = df.bounds['minx'].min(), df.bounds['maxx'].max()

  all_indices = np.arange(len(df))
  root_node = KDTreeNode(min_x, max_x, min_y, max_y, all_indices, depth=0)

  build_kdtree(df, root_node, ind_var, size_var, cod_threshold, min_samples, max_depth)

  # 3) Make predictions
  # Try some test location near middle
  test_lat, test_lon = 30.5, -99.5
  test_size = 5000.0

  # Without blending
  price_no_blend = kdtree_predict(
    df, root_node, test_lat, test_lon, test_size,
    blending=False
  )

  # # Assign centroids to leaves (for blending)
  # leaves = assign_leaf_centroids(df, root_node)
  #
  # # Build a KD-Tree for fast leaf lookups
  # leaf_tree, leaf_centroids = build_leaf_kdtree(leaves)
  #
  # # With blending
  # price_blend = kdtree_predict(
  #   df, root_node, test_lat, test_lon, test_size,
  #   blending=True,
  #   leaves=leaves,       # list of leaf nodes
  #   leaf_tree=leaf_tree, # KDTree object
  #   k=3
  # )
