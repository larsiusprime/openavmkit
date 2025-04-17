from datetime import datetime

import numpy as np
import pandas as pd

from openavmkit.filters import resolve_filter
from openavmkit.utilities.data import div_field_z_safe
from openavmkit.utilities.geometry import get_crs


def perform_tweaks(df_in: pd.DataFrame, tweak: list):
  df = df_in.copy()
  for entry in tweak:
    field = entry.get("field")
    key_field = entry.get("key")
    values = entry.get("values", {})
    if field not in df:
      raise ValueError(f"Field not found: \"{field}\"")
    if key_field not in df:
      raise ValueError(f"Key not found: \"{key_field}\"")
    for key in values:
      value = values[key]
      df.loc[df[key_field].eq(key), field] = value
  return df

def perform_calculations(df_in: pd.DataFrame, calc: dict):
  df = df_in.copy()
  
  for new_field in calc:
    entry = calc[new_field]
    new_value = _do_calc(df, entry)
    df[new_field] = new_value
    
    # Keep only essential debug output for valid_sale
    if new_field == 'valid_sale':
      valid_count = df[new_field].sum()
      print(f"Valid sales: {valid_count} out of {len(df)} total")
  
  # remove temporary columns
  for col in df.columns:
    if col.startswith("__temp_"):
      del df[col]
  
  return df


def _crawl_calc_dict_for_fields(calc_dict: dict):
  fields = []
  for field in calc_dict:
    calc_list = calc_dict[field]
    fields += _crawl_calc_list_for_fields(calc_list)
  return list(set(fields))


def _crawl_calc_list_for_fields(calc_list: list):
  fields = []
  if len(calc_list) > 1:
    entries = calc_list[1:]
    for entry in entries:
      if isinstance(entry, list):
        fields += _crawl_calc_list_for_fields(entry)
      elif isinstance(entry, str):
        if not entry.startswith("str:"):
          fields.append(entry)
  return list(set(fields))


def _calc_resolve(df: pd.DataFrame, value, i:int=0):
  if isinstance(value, str):
    # If it's a string, two possibilities:
    # 1. It's prepended with "str:" --> interpret as a string literal
    # 2. It's a column name --> return the column as a series
    if value.startswith("str:"):
      text = value[4:]
      # Return a constant value as a series
      return text, i
    else:
      if value in df:
        # Return a matching column
        return df[value], i
      else:
        raise ValueError(f"Field not found: \"{value}\". If this was meant as a string constant, prefix it with \"str:\"")
  elif isinstance(value, list):
    # If it's a list of literals, return it as is
    if all(isinstance(x, (int, float)) for x in value):
      return value, i
    # Otherwise, return the result of a recursive calculation
    i += 1
    return _do_calc(df, value, i), i
  # If it's a numeric literal, return it as is
  elif isinstance(value, (int, float)):
    return value, i
  return value, i


def _do_calc(df_in: pd.DataFrame, entry: list, i:int=0):
  df = df_in
  if entry is None or len(entry) == 0:
    raise ValueError("Empty calculation entry")
  op = entry[0]

  # N-ary operations
  if op == "values":
    elements = entry[1:]
    fields = []
    for element in elements:
      if isinstance(element, str) and element in df:
        fields.append(element)
      else:
        element, i = _calc_resolve(df, value=element, i=i+1)
        field_name = f"__temp_{i}"
        df[field_name] = element
        fields.append(field_name)
    return df[fields]

  # Filter operations
  if op == "?":
    return resolve_filter(df, entry[1])

  # Unary operations (LHS only)
  lhs = None
  if len(entry) > 1:
    lhs = entry[1]
    lhs, i = _calc_resolve(df, value=lhs, i=i)

  if op == "asint":
    return (lhs.astype("Float64")).astype("Int64")
  elif op == "asfloat":
    return lhs.astype(float)
  elif op == "asstr":
    return lhs.astype(str)
  elif op == "floor":
    return np.floor(lhs)
  elif op == "ceil":
    return np.ceil(lhs)
  elif op == "round":
    return np.round(lhs)
  elif op == "abs":
    return np.abs(lhs)
  elif op == "strip":
    return lhs.astype(str).str.strip()
  elif op == "striplzero":
    return lhs.astype(str).str.lstrip("0")
  elif op == "stripkey":
    return lhs.astype(str).str.replace(r'\s+', '', regex=True).str.lstrip("0")
  elif op == "set":
    return lhs
  elif op == "not":
    return ~lhs

  # Binary operations (LHS & RHS)
  rhs = None
  if len(entry) > 2:
    rhs = entry[2]
    rhs, i = _calc_resolve(df, value=rhs, i=i)

  if op == "==":
    if isinstance(rhs, str):
      result = lhs.astype(str).str.strip().eq(str(rhs).strip())
      return result
    return lhs.eq(rhs)
  elif op == "!=":
    if isinstance(rhs, str):
      result = lhs.astype(str).str.strip().ne(str(rhs).strip())
      return result
    return lhs.ne(rhs)
  elif op == "+":
    return lhs + rhs
  elif op == "-":
    return lhs - rhs
  elif op == "*":
    return lhs * rhs
  elif op == "/":
    return lhs / rhs
  elif op == "//":
    return (lhs // rhs).astype(int)
  elif op == "/0":
    return div_field_z_safe(lhs, rhs)
  elif op == "round_nearest":
    value = lhs / rhs
    value = np.round(value)
    return value * rhs
  elif op == "map":
    lhs = lhs.astype(str)
    return lhs.map(rhs).fillna(lhs)
  elif op == "fillna":
    lhs.loc[pd.isna(lhs)]
  elif op == "replace":
    for key in rhs:
      old = key
      new = rhs[key]
      is_regex = rhs.get("regex", False)
      lhs = lhs.astype(str).str.replace(old, new, regex=is_regex)
    return lhs
  elif op == "contains":
    result = lhs.astype(str).str.contains(str(rhs), na=False)
    return result.fillna(False).astype(bool)
  elif op == "contains_case_insensitive":
    if "grantor_name" in str(lhs.name):
      lhs_upper = lhs.astype(str).str.upper()
      patterns = ["CITY", "MAYOR AND CITY COUNCIL", "MAYOR & CITY COUNCIL"]
      result = pd.Series(False, index=lhs.index)
      for pattern in patterns:
        result = result | lhs_upper.str.contains(pattern, na=False)
      return result.fillna(False).astype(bool)
    else:
      result = lhs.astype(str).str.upper().str.contains(str(rhs).upper(), na=False)
      return result.fillna(False).astype(bool)
  elif op == "isin":
    if all(isinstance(x, str) for x in rhs):
      result = lhs.astype(str).str.strip().isin([str(x).strip() for x in rhs])
      return result
    return lhs.isin(rhs)
  elif op == "and":
    return lhs & rhs
  elif op == "or":
    return lhs | rhs
  elif op == "split_before":
    return lhs.astype(str).str.split(rhs, expand=False).str[0]
  elif op == "split_after":
    parts = lhs.astype(str).str.partition(rhs)
    return parts[2].mask(parts[1] == "", parts[0])
  elif op == "join":
    result = lhs.astype(str).apply(lambda x: f"{rhs}".join(x), axis=1)
    return result
  elif op == "datetime":
    result = pd.to_datetime(lhs, format=rhs)
    return result
  elif op == "datetimestr":
    result = pd.to_datetime(lhs, format=rhs)
    str_value = result.dt.strftime("%Y-%m-%d")
    return str_value
  elif op == "substr":
    if type(rhs) is dict:
      a = rhs.get("left", None)
      b = rhs.get("right", None)
      if a is not None:
        if b is not None:
          return lhs.astype(str).str[a:b]
        else:
          return lhs.astype(str).str[a:]
      else:
        return lhs.astype(str).str[:b]
    raise ValueError(f"Right-hand side value for operator 'substr' must be a dict containing 'left' and/or 'right' keys, found '{type(rhs)}' = {rhs}")
  elif op == "geo_area":
    if "geometry" in df_in:
      ea_crs = get_crs(df_in, "equal_area")
      df_ea = df_in.to_crs(ea_crs)
      series_area = df_ea.geometry.area
      if lhs == "sqft":
        return series_area * 10.7639
      elif lhs == "sqm":
        return series_area
      elif lhs == "acres":
        return (series_area * 10.7639) / 43560
      elif lhs == "sqkm":
        return series_area / 1e6
      elif lhs == "hectares":
        return series_area / 10000
      else:
        raise ValueError(f"Unknown area unit: {lhs}. Only 'sqft', 'sqm', 'acres', 'sqkm', 'hectares' are supported.")
    else:
      raise ValueError("'area' calculation can only be performed on a geodataframe containing a 'geometry' column!")

  raise ValueError(f"Unknown operation: {op}")

