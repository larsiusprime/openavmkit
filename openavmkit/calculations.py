import numpy as np
import pandas as pd
from openavmkit.utilities.data import div_field_z_safe


def crawl_calc_dict_for_fields(calc_dict: dict):
  fields = []
  for field in calc_dict:
    calc_list = calc_dict[field]
    fields += _crawl_calc_list_for_fields(calc_list)
  return list(set(fields))


def _crawl_calc_list_for_fields(calc_list: list):
  fields = []
  if len(calc_list) > 1:
    sub_list = calc_list[1:]
    for entry in sub_list:
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
    # Return the result of a recursive calculation
    i += 1
    return _do_calc(df, value, i), i
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
        # It's an existing field name
        fields.append(element)
      else:
        # It's a fresh calculation

        # resolve the field
        element, i = _calc_resolve(df, value=element, i=i+1)

        # store it under a temporary name as a new field
        field_name = f"__temp_{i}"
        df[field_name] = element
        fields.append(field_name)

    # return the selection of field names & calculations
    return df[fields]

  # Unary operations (LHS only)

  lhs = None
  if len(entry) > 1:
    lhs = entry[1]
    lhs, i = _calc_resolve(df, value=lhs, i=i)

  if op == "asint":
    return lhs.astype(int)
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

  # Binary operations (LHS & RHS)

  rhs = None
  if len(entry) > 2:
    rhs = entry[2]
    rhs, i = _calc_resolve(df, value=rhs, i=i)

  if op == "+":
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
  elif op == "join":
    result = lhs.astype(str).apply(lambda x: f"{rhs}".join(x), axis=1)
    return result


  raise ValueError(f"Unknown operation: {op}")


def perform_calculations(df_in: pd.DataFrame, calc: dict):
  df = df_in.copy()
  for new_field in calc:
    entry = calc[new_field]
    new_value = _do_calc(df, entry)
    df[new_field] = new_value
  # remove temporary columns
  for col in df.columns:
    if col.startswith("__temp_"):
      del df[col]
  return df