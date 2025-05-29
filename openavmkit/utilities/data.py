import os
import pickle

import numpy as np
import pandas as pd
import osmnx as ox
from collections import defaultdict

from scipy.spatial._ckdtree import cKDTree

from openavmkit.utilities.settings import get_model_group_ids


def clean_column_names(df: pd.DataFrame):
  """
  Clean the column names in a DataFrame by replacing forbidden characters with legal representations.
  For one-hot encoded columns (containing '='), ensures clean formatting.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :returns: DataFrame with cleaned column names.
  :rtype: pandas.DataFrame
  """
  # Find column names that contain forbidden characters and replace them with legal representations.
  replace_map = {
    "[": "_",
    "]": "_",
    "<NA>": "_NA_",
    "/": "_",
    "\\": "_",
    ":": "_",
    "*": "_",
    "?": "_",
    "\"": "_",
    "<": "_",
    ">": "_",
    "|": "_",
    " ": "_",  # Replace spaces with underscores
    "-": "_",  # Replace hyphens with underscores
    ",": "_",  # Replace commas with underscores
    ";": "_",  # Replace semicolons with underscores
    ".": "_",  # Replace periods with underscores
    "(": "_",  # Replace parentheses with underscores
    ")": "_"
  }

  # First pass - replace special characters
  for key in replace_map:
    df.columns = df.columns.str.replace(key, replace_map[key], regex=False)
  
  # Second pass - clean up one-hot encoded column names
  new_columns = []
  for col in df.columns:
    if "=" in col:
      # Handle one-hot encoded columns
      base, value = col.split("=", 1)
      # Clean up the base and value
      base = base.strip()
      value = value.strip()
      # Replace multiple underscores with single underscore
      base = "_".join(filter(None, base.split("_")))
      value = "_".join(filter(None, value.split("_")))
      new_col = f"{base}__{value}"  # Use double underscore as separator
    else:
      # For non-one-hot columns, just clean up multiple underscores
      new_col = "_".join(filter(None, col.split("_")))
    
    new_columns.append(new_col)
  
  df.columns = new_columns
  return df


def clean_series(series: pd.Series):
  replace_map = {
    "[": "_LBRKT_",
    "]": "_RBRKT_",
    "<NA>": "_NA_",
    "/": "_SLASH_",
    "\\": "_BSLASH_",
    ":": "_COLON_",
    "*": "_STAR_",
    "?": "_QMARK_",
    "\"": "_DQUOT_",
    "<": "_LT_",
    ">": "_GT_",
    "|": "_PIPE_"
  }

  for key in replace_map:
    series = series.str.replace(key, replace_map[key], regex=False)


def div_field_z_safe(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray):
  """
  Perform a divide-by-zero-safe division of two series or arrays, replacing division by zero with None.

  :param numerator: Numerator series or array.
  :type numerator: pandas.Series or numpy.ndarray
  :param denominator: Denominator series or array.
  :type denominator: pandas.Series or numpy.ndarray
  :returns: The result of the division with divide-by-zero cases replaced by None.
  :rtype: pandas.Series or numpy.ndarray
  """
  # Get the index of all rows where the denominator is zero.
  idx_denominator_zero = (denominator == 0)

  # Get the numerator and denominator for rows where the denominator is not zero.
  series_numerator = numerator[~idx_denominator_zero]
  series_denominator = denominator[~idx_denominator_zero]

  # Make a copy of the denominator and convert to a float type.
  result = denominator.copy().astype("Float64")

  # Replace all values where denominator is zero with None.
  result[idx_denominator_zero] = None

  # Replace other values with the result of the division.
  result[~idx_denominator_zero] = series_numerator / series_denominator
  return result


def div_z_safe(df: pd.DataFrame, numerator: str, denominator: str):
  """
  Perform a divide-by-zero-safe division of two columns in a DataFrame, replacing division by zero with None.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param numerator: Name of the column to use as the numerator.
  :type numerator: str
  :param denominator: Name of the column to use as the denominator.
  :type denominator: str
  :returns: A pandas Series with the result of the safe division.
  :rtype: pandas.Series
  """
  # Get the index of all rows where the denominator is zero.
  idx_denominator_zero = df[denominator].eq(0)

  # Get the numerator and denominator for rows where the denominator is not zero.
  series_numerator = df.loc[~idx_denominator_zero, numerator]
  series_denominator = df.loc[~idx_denominator_zero, denominator]

  # Make a copy of the denominator.
  result = df[denominator].copy()

  # Replace values where denominator is zero with None.
  result[idx_denominator_zero] = None

  # Replace other values with the result of the division.

  result = result.astype("Float64") # ensure it can accept the result

  result[~idx_denominator_zero] = series_numerator / series_denominator
  return result


def dataframe_to_markdown(df: pd.DataFrame):
  """
  Convert a DataFrame to a markdown-formatted string.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :returns: Markdown string representation of the DataFrame.
  :rtype: str
  """
  header = "| " + " | ".join(df.columns) + " |"
  separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
  rows = "\n".join(
    "| " + " | ".join(row) + " |" for row in df.astype(str).values
  )
  return f"{header}\n{separator}\n{rows}"


def rename_dict(dict, renames):
  """
  Rename the keys of a dictionary according to a given rename map.

  :param dict_obj: Original dictionary.
  :type dict_obj: dict
  :param renames: Dictionary mapping old keys to new keys.
  :type renames: dict
  :returns: New dictionary with keys renamed.
  :rtype: dict
  """
  new_dict = {}
  for key in dict:
    new_key = renames.get(key, key)
    new_dict[new_key] = dict[key]
  return new_dict


def do_per_model_group(df_in: pd.DataFrame, settings: dict, func: callable, params: dict, key: str ="key", verbose: bool = False, instructions = None) -> pd.DataFrame:
  """
  Apply a function to each subset of the DataFrame grouped by 'model_group', updating rows based on matching indices.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param func: Function to apply to each subset.
  :type func: callable
  :param params: Additional parameters for the function.
  :type params: dict
  :param key: Column name to use as the index for alignment (default is "key").
  :type key: str, optional
  :param verbose: If True, prints progress information (default is False).
  :type verbose: bool, optional
  :param instructions: Optional instructions for the function.
  :type instructions: any, optional
  :returns: Modified DataFrame with updates from the function.
  :rtype: pandas.DataFrame
  """
  df = df_in.copy()

  if instructions is None:
    instructions = {}

  model_groups = get_model_group_ids(settings, df_in)
  verbose = params.get("verbose", verbose)

  for model_group in model_groups:
    if pd.isna(model_group):
      continue

    if verbose:
      print(f"Processing model group: {model_group}")

    # Copy params locally to avoid side effects.
    params_local = params.copy()
    params_local["model_group"] = model_group

    # Filter the subset using .loc to avoid SettingWithCopyWarning
    mask = df["model_group"].eq(model_group)
    df_sub = df.loc[mask].copy()

    # Apply the function.
    df_sub_updated = func(df_sub, **params_local)

    if df_sub_updated is not None:
      # Ensure consistent data types between df and the updated subset.
      just_stomp_columns = instructions.get("just_stomp_columns", [])
      if len(just_stomp_columns) > 0:
        for col in just_stomp_columns:
          if col in df_sub_updated.columns:
            df.loc[mask, col] = df_sub_updated[col]
      else:
        for col in df_sub_updated.columns:
          if col == key:
            continue
          df = combine_dfs(df, df_sub_updated[[key, col]], df2_stomps=True, index=key)

  return df


def merge_and_stomp_dfs(df1: pd.DataFrame, df2: pd.DataFrame, df2_stomps=False, on: str|list = "key", how: str = "left") -> pd.DataFrame:

  common_columns = [col for col in df1.columns if col in df2.columns]
  df_merge = pd.merge(df1, df2, on=on, how=how, suffixes=("_1", "_2"))
  suffixed_columns = [col + "_1" for col in common_columns] + [col + "_2" for col in common_columns]
  suffixed_columns = [col for col in suffixed_columns if col in df_merge.columns]

  for col in common_columns:
    if col == on or (isinstance(on, list) and col in on):
      continue
    if df2_stomps:
      # prefer df2's column value everywhere df2 has a non-null value
      # Filter out empty entries before combining
      df2_col = df_merge[col + "_2"].dropna()
      df1_col = df_merge[col + "_1"].dropna()
      if df2_col.size > 0 and df1_col.size > 0:
        df_merge[col] = df2_col.combine_first(df1_col)
      elif df2_col.size > 0:
        df_merge[col] = df2_col
      else:
        df_merge[col] = df1_col
    else:
      # prefer df1's column value everywhere df1 has a non-null value
      # Filter out empty entries before combining
      df1_col = df_merge[col + "_1"].dropna()
      df2_col = df_merge[col + "_2"].dropna()
      if df1_col.size > 0 and df2_col.size > 0:
        df_merge[col] = df1_col.combine_first(df2_col)
      elif df1_col.size > 0:
        df_merge[col] = df1_col
      else:
        df_merge[col] = df2_col

  df_merge.drop(columns=suffixed_columns, inplace=True)
  return df_merge


def combine_dfs(df1: pd.DataFrame, df2: pd.DataFrame, df2_stomps=False, index: str = "key") -> pd.DataFrame:
  """
  Combine two DataFrames on a given index column.

  If df2_stomps is False, NA values in df1 are filled with values from df2.
  If df2_stomps is True, values in df1 are overwritten by those in df2 for matching keys.

  :param df1: First DataFrame.
  :type df1: pandas.DataFrame
  :param df2: Second DataFrame.
  :type df2: pandas.DataFrame
  :param df2_stomps: Flag indicating if df2 values should overwrite df1 values (default is False).
  :type df2_stomps: bool, optional
  :param index: Column name to use as the index for alignment (default is "key").
  :type index: str, optional
  :returns: Combined DataFrame.
  :rtype: pandas.DataFrame
  """
  df = df1.copy()
  # Save the original index for restoration
  original_index = df.index.copy()

  # Work on a copy so we don't modify df2 outside this function.
  df2 = df2.copy()

  # Set the index to the key column for alignment.
  df.index = df[index]
  df2.index = df2[index]

  # Iterate over columns in df2 (skip the key column).
  for col in df2.columns:
    if col == index:
      continue
    if col in df.columns:
      # Find the common keys to avoid KeyErrors if df2 has extra keys.
      common_idx = df.index.intersection(df2.index)
      if df2_stomps:
        # Overwrite all values in df for common keys.
        df.loc[common_idx, col] = df2.loc[common_idx, col]
      else:
        # For common keys, fill only NA values.
        na_mask = pd.isna(df.loc[common_idx, col])
        # Only assign where df2 has a value and df is NA.
        df.loc[common_idx[na_mask], col] = df2.loc[common_idx[na_mask], col]
    else:
      # Add the new column, aligning by index.
      # (Rows in df without a corresponding key in df2 will get NaN.)
      df[col] = df2[col]

  # Restore the original index.
  df.index = original_index
  return df


def add_sqft_fields(df_in: pd.DataFrame):
  """
  Add per-square-foot fields to the DataFrame for land and improvement values.

  This function creates new columns based on existing value fields and area fields.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :returns: DataFrame with additional per-square-foot fields.
  :rtype: pandas.DataFrame
  """
  df = df_in.copy()
  land_sqft = ["model_market_value", "model_land_value", "assr_market_value", "assr_land_value"]
  impr_sqft = ["model_market_value", "model_impr_value", "assr_market_value", "assr_impr_value"]
  for field in land_sqft:
    if field in df:
      df[field + "_land_sqft"] = div_field_z_safe(df[field], df["land_area_sqft"])
  for field in impr_sqft:
    if field in df:
      df[field + "_impr_sqft"] = div_field_z_safe(df[field], df["bldg_area_finished_sqft"])
  return df


def cache(path: str, logic: callable):
  """
  Cache a computed result to disk.

  If the file at the given path exists, load and return the cached result. Otherwise,
  compute the result using the provided callable, save it, and return it.

  :param path: File path for the cache.
  :type path: str
  :param logic: A callable that computes the result.
  :type logic: callable
  :returns: The cached or computed result.
  :rtype: Any
  """
  outpath = path
  if os.path.exists(outpath):
    with open(outpath, "rb") as f:
      return pickle.load(f)
  result = logic()
  os.makedirs(os.path.dirname(outpath), exist_ok=True)
  with open(outpath, "wb") as f:
    pickle.dump(result, f)
  return result


def count_values_in_common(a: pd.DataFrame, b: pd.DataFrame, a_field: str, b_field: str = None):
    if b_field is None:
      b_field = a_field
    a_values = set(a[a_field].dropna().unique())
    b_values = set(b[b_field].dropna().unique())
    a_in_b = a_values.intersection(b_values)
    b_in_a = b_values.intersection(a_values)
    return len(a_in_b), len(b_in_a)


def ensure_categories(
    df: pd.DataFrame,
    df_other: pd.DataFrame,
    field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
  if (isinstance(df[field].dtype, pd.CategoricalDtype) and
      isinstance(df_other[field].dtype, pd.CategoricalDtype)):

    # union keeps order of appearance in the first operands
    cats = df[field].cat.categories.union(df_other[field].cat.categories)

    # give *both* Series the identical category list
    df[field]        = df[field].cat.set_categories(cats)
    df_other[field]  = df_other[field].cat.set_categories(cats)

  return df, df_other


def align_categories(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Ensure that *every* column that is categorical on either side ends up
  categorical on both sides and that they share the **union** of categories.
  """

  for col in df_left.columns.union(df_right.columns):

    left_is_cat  = isinstance(df_left.get(col, pd.Series(dtype="object")).dtype,
      pd.CategoricalDtype)
    right_is_cat = isinstance(df_right.get(col, pd.Series(dtype="object")).dtype,
      pd.CategoricalDtype)

    # If exactly one side is categorical, convert the other side first
    if left_is_cat and not right_is_cat:
      df_right[col] = pd.Categorical(df_right[col],
        categories=df_left[col].cat.categories)
      right_is_cat = True
    elif right_is_cat and not left_is_cat:
      df_left[col] = pd.Categorical(df_left[col],
        categories=df_right[col].cat.categories)
      left_is_cat = True

    # Now, if both are categorical, give them the same (union) category list
    if left_is_cat and right_is_cat:
      cats = df_left[col].cat.categories.union(df_right[col].cat.categories)
      df_left[col]  = df_left[col].cat.set_categories(cats)
      df_right[col] = df_right[col].cat.set_categories(cats)

  return df_left, df_right


def encode_city_blocks():

  # ─── 1. Download & simplify the OSM network ────────────────────────────────────
  place = "Cambridge, MA, USA"
  highway_types = [
    "motorway","trunk","primary","secondary",
    "tertiary","unclassified","residential","service"
  ]
  custom_filter = f'["highway"~"{"|".join(highway_types)}"]'
  G = ox.graph_from_place(
    place,
    network_type="drive",
    simplify=True,
    custom_filter=custom_filter
  )

  # ─── 2. Extract edges GeoDataFrame with u/v node IDs ───────────────────────────
  edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()[[
    "u", "v", "geometry", "name", "highway", "osmid"
  ]]

  # ─── 3. Explode multilines, drop empties, reproject to metric CRS ─────────────
  #    (choose a suitable projected CRS for distance-based ops)
  crs_eq = "EPSG:3857"
  edges = (
    edges
    .explode(index_parts=False)
    .dropna(subset=["geometry"])
    .to_crs(crs_eq)
    .reset_index(drop=True)
  )

  # ─── 4. Unwrap any list‐values & fill missing names ───────────────────────────
  edges["road_name"] = edges["name"].apply(
    lambda v: v[0] if isinstance(v, (list, tuple)) else v
  )
  edges["road_type"] = edges["highway"].apply(
    lambda v: v[0] if isinstance(v, (list, tuple)) else v
  )
  # fallback: use osmid as a string if name was null
  edges["road_name"] = edges["road_name"].fillna(edges["osmid"].astype(str))

  # ─── 5. Build node→roads mapping, skipping service‐type roads ──────────────────
  node_to_names = defaultdict(set)
  for _, row in edges[["u","v","road_name","road_type"]].iterrows():
    if row["road_type"] == "service":
      # never include service roads as cross‐streets
      continue
    node_to_names[row.u].add(row.road_name)
    node_to_names[row.v].add(row.road_name)

  # ─── 6. Helper: pick the first “other” road at a junction ─────────────────────
  def first_other(names_set, self_name):
    for nm in names_set:
      if nm != self_name:
        return nm
    return "?"

  # ─── 7. Compute cross‐street names at each end ────────────────────────────────
  edges["cross_w"] = [
    first_other(node_to_names[u], rn)
    for u, rn in zip(edges["u"], edges["road_name"])
  ]
  edges["cross_e"] = [
    first_other(node_to_names[v], rn)
    for v, rn in zip(edges["v"], edges["road_name"])
  ]

  # ─── 8. Build the final name_loc field ────────────────────────────────────────
  edges["name_loc"] = (
      edges["road_name"]
      + " between "
      + edges["cross_w"]
      + " and "
      + edges["cross_e"]
  )

  # ─── 9. (Optional) drop service‐road segments entirely ────────────────────────
  # edges = edges[edges.road_type != "service"]

  # ─── Done! ─────────────────────────────────────────────────────────────────────
  print(edges[["u","v","road_name","cross_w","cross_e","name_loc"]].head())


def calc_spatial_lag(df_sample: pd.DataFrame, df_univ: pd.DataFrame, value_fields:list[str], neighbors:int = 5, exclude_self_in_sample: bool = False) -> pd.DataFrame:

  df = df_univ.copy()

  # Build a cKDTree from df_sales coordinates

  # we TRAIN on these coordinates -- coordinates that are NOT in the test set
  coords_train = df_sample[['latitude', 'longitude']].values
  tree = cKDTree(coords_train)

  # we PREDICT on these coordinates -- all the coordinates in the universe
  coords_all = df[['latitude', 'longitude']].values

  for value_field in value_fields:
    if value_field not in df_sample:
      print("Value field not in df_sample, skipping")
      continue

    # Choose the number of nearest neighbors to use
    k = neighbors  # You can adjust this number as needed

    # Query the tree: for each parcel in df_universe, find the k nearest parcels
    # distances: shape (n_universe, k); indices: corresponding indices in df_sales
    distances, indices = tree.query(coords_all, k=k)

    if exclude_self_in_sample:
      distances = distances[:, 1:]  # Exclude self-distance
      indices = indices[:, 1:]  # Exclude self-index

    # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
    if k < 2:
      raise ValueError("k must be at least 2 to compute spatial lag.")

    # For each universe parcel, compute sigma as the mean distance to its k neighbors.
    sigma = distances.mean(axis=1, keepdims=True)

    # Handle zeros in sigma
    sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

    # Compute Gaussian kernel weights for all neighbors
    weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))

    # Normalize the weights so that they sum to 1 for each parcel
    weights_norm = weights / weights.sum(axis=1, keepdims=True)

    # Get the values corresponding to the neighbor indices
    parcel_values = df_sample[value_field].values
    neighbor_values = parcel_values[indices]  # shape (n_universe, k)

    # Compute the weighted average (spatial lag) for each parcel in the universe
    spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_values)).sum(axis=1)

    # Add the spatial lag as a new column
    df[f"spatial_lag_{value_field}"] = spatial_lag

    median_value = df_sample[value_field].median()
    df[f"spatial_lag_{value_field}"] = df[f"spatial_lag_{value_field}"].fillna(median_value)

  return df


def load_model_results(model_group: str, model_name: str, subset: str = "universe", model_type: str = "main"):
  outpath = f"out/models/{model_group}/{model_type}"

  filepath = f"{outpath}/{model_name}"
  if os.path.exists(filepath):
    fpred = f"{filepath}/pred_{subset}.parquet"
    if not os.path.exists(fpred):
      fpred = f"{filepath}/pred_{model_name}_{subset}.parquet"

    if os.path.exists(fpred):
      df = pd.read_parquet(fpred)
      if "key_x" in df:
        # If the DataFrame has a 'key_x' column, rename it to 'key'
        df.rename(columns={"key_x": "key"}, inplace=True)
      df = df[["key", "prediction"]].copy()
      return df

  fpred_results = f"{filepath}/pred_{subset}.pkl"
  if os.path.exists(fpred_results):
    if model_type != "main":
      with open (fpred_results, "rb") as file:
        results = pickle.load(file)
        if subset == "universe":
          df = results.df_universe[["key", "prediction"]].copy()
        elif subset == "sales":
          df = results.df_sales[["key", "prediction"]].copy()
        elif subset == "test":
          df = results.df_test[["key", "prediction"]].copy()
        return df

  return None