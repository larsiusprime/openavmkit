import os
import pickle

import numpy as np
import pandas as pd

from openavmkit.utilities.settings import get_model_group_ids


def clean_column_names(df: pd.DataFrame):
  """
  Clean the column names in a DataFrame by replacing forbidden characters with legal representations.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :returns: DataFrame with cleaned column names.
  :rtype: pandas.DataFrame
  """
  # Find column names that contain forbidden characters and replace them with legal representations.
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
    df.columns = df.columns.str.replace(key, replace_map[key])
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


def do_per_model_group(df_in: pd.DataFrame, settings: dict, func: callable, params: dict, key: str ="key", verbose: bool = False) -> pd.DataFrame:
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
  :returns: Modified DataFrame with updates from the function.
  :rtype: pandas.DataFrame
  """
  df = df_in.copy()
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
    if col == on:
      continue
    if df2_stomps:
      # prefer df2's column value everywhere df2 has a non-null value
      df_merge[col] = df_merge[col + "_2"].combine_first(df_merge[col + "_1"])
    else:
      # prefer df1's column value everywhere df1 has a non-null value
      df_merge[col] = df_merge[col + "_1"].combine_first(df_merge[col + "_2"])

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
