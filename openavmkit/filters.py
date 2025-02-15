import pandas as pd


def select_filter(df: pd.DataFrame, f: list) -> pd.DataFrame:
  resolved_index = resolve_filter(df, f)
  return df.loc[resolved_index]


def resolve_bool_filter(df: pd.DataFrame, f: list) -> pd.Series:
  operator = f[0]
  values = f[1:]

  final_index = None

  for v in values:
    selected_index = resolve_filter(df, v)

    if final_index is None:
      final_index = selected_index
      continue

    if operator == "and":
      final_index = final_index & selected_index
    elif operator == "nand":
      final_index = ~(final_index & selected_index)
    elif operator == "or":
      final_index = final_index | selected_index
    elif operator == "nor":
      final_index = ~(final_index | selected_index)
    elif operator == "xor":
      final_index = final_index ^ selected_index
    elif operator == "xnor":
      final_index = ~(final_index ^ selected_index)

  return final_index


def resolve_filter(df: pd.DataFrame, f: list) -> pd.Series:
  operator = f[0]

  # check if operator is FilterOperatorBool:
  if is_bool_operator(operator):
    return resolve_bool_filter(df, f)
  else:
    field = f[1]
    if len(f) == 3:
      value = f[2]
    else:
      value = None

    if isinstance(value, str):
      if value.startswith("str:"):
        value = value[4:]

    if operator == ">": return df[field].fillna(0).gt(value)
    if operator == "<": return df[field].fillna(0).lt(value)
    if operator == ">=": return df[field].fillna(0).ge(value)
    if operator == "<=": return df[field].fillna(0).le(value)
    if operator == "==": return df[field].fillna(0).eq(value)
    if operator == "!=": return df[field].fillna(0).ne(value)
    if operator == "isin": return df[field].isin(value)
    if operator == "notin": return ~df[field].isin(value)
    if operator == "isempty": return pd.isna(df[field]) | df[field].astype(str).str.strip().eq("")
    if operator == "iszero": return df[field].eq(0)
    if operator == "contains":
      selection = df[field].str.contains(value[0])
      for v in value[1:]: selection |= df[field].str.contains(v)
      return selection

  raise ValueError(f"Unknown operator {operator}")


def validate_filter_list(filters: list[list]):
  for f in filters:
    validate_filter(f)
  return True


def validate_filter(f: list):
  operator = f[0]
  if operator in ["and", "or"]:
    pass
  else:
    value = f[2]

    if operator in [">", "<", ">=", "<="]:
      if not isinstance(value, (int, float, bool)):
        raise ValueError(f"Value must be a number for operator {operator}")
    if operator in ["isin", "notin"]:
      if not isinstance(value, list):
        raise ValueError(f"Value must be a list for operator {operator}")
    if operator == "contains":
      if not isinstance(value, str):
        raise ValueError(f"Value must be a string for operator {operator}")
  return True


def is_basic_operator(s : str) -> bool:
  return s in ["<", ">", "<=", ">=", "==", "!=", "isin", "notin", "contains"]


def is_bool_operator(s : str) -> bool:
  return s in ["and", "or", "nand", "nor", "xor", "xnor"]