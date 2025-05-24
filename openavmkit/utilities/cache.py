import os
import json
import pickle

import pandas as pd
import geopandas as gpd

from openavmkit.utilities.assertions import objects_are_equal, dicts_are_equal, dfs_are_equal
from openavmkit.utilities.geometry import ensure_geometries


def write_cache(
    filename: str,
    payload: dict | str | pd.DataFrame | gpd.GeoDataFrame | bytes,
    signature: dict | str,
    filetype: str
):
  extension = _get_extension(filetype)
  path = f"cache/{filename}.{extension}"
  base_path = os.path.dirname(path)
  os.makedirs(base_path, exist_ok=True)
  if filetype == "dict":
    with open(path, "w") as file:
      json.dump(payload, file)
  elif filetype == "str":
    with open(path, "w") as file:
      file.write(payload)
  elif filetype == "pickle":
    with open(path, "wb") as file:
      pickle.dump(payload, file)
  elif filetype == "df":
    if isinstance(payload, pd.DataFrame):
      if isinstance(payload, gpd.GeoDataFrame):
        payload.to_parquet(path, engine="pyarrow")
      else:
        payload.to_parquet(path)
    else:
      raise TypeError("Payload must be a DataFrame for df type.")

  if type(signature) is dict:
    sig_ext = "json"
  elif type(signature) is str:
    sig_ext = "txt"
  else:
    raise TypeError(f"Unsupported type for signature value: {type(signature)} sig = {signature}")

  signature_path = f"cache/{filename}.signature.{sig_ext}"
  with open(signature_path, "w") as file:
    if sig_ext == "json":
      json.dump(signature, file)
    else:
      file.write(signature)


def read_cache(
    filename: str,
    filetype: str
):
  extension = _get_extension(filetype)
  path = f"cache/{filename}.{extension}"
  if os.path.exists(path):
    if filetype == "dict":
      with open(path, "r") as file:
        return json.load(file)
    elif filetype == "str":
      with open(path, "r") as file:
        return file.read()
    elif filetype == "pickle":
      with open(path, "rb") as file:
        return pickle.load(file)
    elif filetype == "df":
      try:
        df = gpd.read_parquet(path)
        if "geometry" in df:
          df = gpd.GeoDataFrame(df, geometry="geometry")
          ensure_geometries(df, "geometry", df.crs)
      except ValueError:
        df = pd.read_parquet(path)
      return df
  return None


def check_cache(
    filename: str,
    signature: dict | str,
    filetype: str
):
  ext = _get_extension(filetype)
  path = f"cache/{filename}"
  match = _match_signature(path, signature)
  if match:
    path_exists = os.path.exists(f"{path}.{ext}")
    return path_exists
  return False


def clear_cache(
    filename: str,
    filetype: str
):
  ext = _get_extension(filetype)
  path = f"cache/{filename}"
  if os.path.exists(f"{path}.{ext}"):
    os.remove(f"{path}.{ext}")
  if os.path.exists(f"{path}.cols{ext}"):
    os.remove(f"{path}..cols.{ext}")
  if os.path.exists(f"{path}.rows{ext}"):
    os.remove(f"{path}.rows{ext}")
  if os.path.exists(f"{path}.signature.json"):
    os.remove(f"{path}.signature.json")
  if os.path.exists(f"{path}.cols.signature.json"):
    os.remove(f"{path}.cols.signature.json")
  if os.path.exists(f"{path}.rows.signature.json"):
    os.remove(f"{path}.rows.signature.json")


def write_cached_df(
    df_orig: pd.DataFrame,
    df_new: pd.DataFrame,
    filename: str,
    key: str = "key",
    extra_signature: dict | str = None
)-> pd.DataFrame | None:

  df_new = df_new.copy()

  orig_cols = set(df_orig.columns)
  new_cols  = [c for c in df_new.columns if c not in orig_cols]
  common    = [c for c in df_new.columns if c in orig_cols]

  orig_rows_by_key = df_orig[key].values
  new_rows_by_key = df_new[key].values
  if len(orig_rows_by_key) > len(new_rows_by_key):
    added_rows = [key for key in new_rows_by_key if key not in orig_rows_by_key]
    orig_set = set(orig_rows_by_key)
    new_set = set(new_rows_by_key)
    added_rows = []
    deleted_rows = list(orig_set - new_set)
  elif len(orig_rows_by_key) < len(new_rows_by_key):
    orig_set = set(orig_rows_by_key)
    new_set = set(new_rows_by_key)
    added_rows = list(new_set - orig_set)
    deleted_rows = []
  else:
    added_rows = []
    deleted_rows = []


  modified = []
  for c in common:
    col_new = df_new[c].reset_index(drop=True)
    col_orig = df_orig[c].reset_index(drop=True)

    is_different = False
    if len(col_new) == len(col_orig):
      values_equal = col_new.values == col_orig.values
      na_equal = col_new.isna() & col_orig.isna()

      count_na_equal = na_equal.sum()
      count_values_equal = values_equal.sum()

      count_to_match = len(col_new)

      all_equal = (count_na_equal == count_to_match and count_values_equal == count_to_match)
      if not all_equal:
        is_different = True
    else:
      is_different = True

    if is_different:
      modified.append(c)
      continue

  changed_cols = new_cols + modified
  if not changed_cols:
    # nothing new or modified → no cache update needed
    return df_orig

  the_cols = changed_cols
  if key not in the_cols:
    the_cols = [key]+changed_cols

  df_diff_cols = df_new[the_cols].copy()
  df_diff_cols = df_diff_cols[~df_diff_cols[key].isin(added_rows)]
  signature = _get_df_signature(df_orig, extra_signature)
  df_type = "df"
  write_cache(f"{filename}.cols", df_diff_cols, signature, df_type)
  if len(deleted_rows) > 0:
    df_new = df_new[~df_new[key].isin(deleted_rows)].copy()

  if len(added_rows) > 0:
    df_diff_rows = df_new[df_new[key].isin(added_rows)].copy()
    if not df_diff_rows.empty:
      write_cache(f"{filename}.rows", df_diff_rows, signature, df_type)

  df_cached = get_cached_df(df_orig, filename, key, extra_signature)

  are_equal = dfs_are_equal(df_new, df_cached, allow_weak=True, primary_key=key)
  if not are_equal:
    raise ValueError(f"Cached DataFrame does not match the original DataFrame.")

  return df_cached


def get_cached_df(
    df: pd.DataFrame,
    filename: str,
    key: str = "key",
    extra_signature: dict | str = None,
    only_signature: dict | str = None
)->pd.DataFrame | gpd.GeoDataFrame | None:

  if only_signature is not None:
    signature = only_signature
  else:
    signature = _get_df_signature(df, extra_signature)

  filename_rows = f"{filename}.rows"
  filename_cols = f"{filename}.cols"

  df_merged = None

  if check_cache(filename_cols, signature, "df"):
    # Merge new columns
    df_diff = read_cache(filename_cols, "df")
    if not df_diff is None and not df_diff.empty:
      df_diff[key] = df_diff[key].astype(df[key].dtype)

      cols_to_replace = [c for c in df_diff.columns if c != key]

      # Drop the columns that are going to be replaced
      df_base = df.drop(columns=cols_to_replace, errors="ignore")

      # Drop the keys that are not in the diff
      df_base = df_base[df_base["key"].isin(df_diff[key])].copy()

      df_merged = df_base.merge(df_diff, how="left", on=key)

      if isinstance(df_diff, gpd.GeoDataFrame):
        df_merged = gpd.GeoDataFrame(df_merged, geometry="geometry")
        df_merged = ensure_geometries(df_merged, "geometry", df_diff.crs)

  if check_cache(filename_rows, signature, "df"):
    # Add new rows
    df_diff = read_cache(filename_rows, "df")
    if not df_diff is None and not df_diff.empty:
      df_diff[key] = df_diff[key].astype(df[key].dtype)

      if df_merged is None:
        df_merged = df.copy()

      # add the new rows onto the end of the DataFrame
      df_merged = pd.concat([df_merged, df_diff], ignore_index=True)


  return df_merged


def _get_df_signature(df: pd.DataFrame, extra: dict | str = None):
  sorted_columns = sorted(df.columns)
  signature = {
    "num_rows": len(df),
    "num_columns": len(df.columns),
    "columns": sorted_columns,
    "checksum": _cheap_checksum(df)
  }
  if extra is not None:
    signature["extra"] = extra
  return signature


def _cheap_checksum(df: pd.DataFrame):
  checksum = {}
  return checksum
  # for col in df.columns:
  #   # if it's geometry:
  #   # if it's numeric:
  #   if pd.api.types.is_numeric_dtype(df[col]):
  #     checksum[col] = float(df[col].sum())
  #   elif col == "geometry":
  #     # just note how many geometry rows are not null:
  #     checksum[col] = float((~df[col].isna()).sum())
  #   else:
  #     try:
  #       checksum[col] = str(df[col].value_counts())
  #     except TypeError:
  #       checksum[col] = float(df[col].apply(lambda x: str(x).encode("utf-8")).sum())
  # return checksum

def _match_signature(
    filename: str,
    signature: dict | str
)->bool:
  if type(signature) is dict:
    sig_ext = "json"
  elif type(signature) is str:
    sig_ext = "txt"
  else:
    raise TypeError(f"Unsupported type for signature value: {type(signature)}")
  sig_file = f"{filename}.signature.{sig_ext}"
  match = False
  if os.path.exists(sig_file):
    if sig_ext == "json":
      with open(sig_file, "r") as file:
        cache_signature = json.load(file)
      match = dicts_are_equal(signature, cache_signature)
    else:
      with open(sig_file, "r") as file:
        cache_signature = file.read()
      match = signature == cache_signature
  return match


def _get_extension(filetype:str):
  if filetype == "dict":
    return "json"
  elif filetype == "str":
    return "txt"
  elif filetype == "df":
    return "parquet"
  elif filetype == "pickle":
    return "pickle"
  elif filetype == "json":
    raise ValueError(f"Filetype 'json' is unsupported, did you mean 'dict'?")
  elif filetype == "txt" or filetype == "text":
    raise ValueError(f"Filetype '{filetype}' is unsupported, did you mean 'str'?")
  elif filetype == "parquet":
    raise ValueError(f"Filetype 'parquet' is ambiguous: please use 'df' instead")
  raise ValueError(f"Unsupported filetype: '{filetype}'")
