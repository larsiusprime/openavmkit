import pandas as pd


def select_filter(df: pd.DataFrame, f: list) -> pd.DataFrame:
  """
  Select a subset of the DataFrame based on a list of filters.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param f: Filter expressed as a list.
  :type f: list
  :returns: Filtered DataFrame.
  :rtype: pandas.DataFrame
  """
  resolved_index = resolve_filter(df, f)
  return df.loc[resolved_index]


def resolve_not_filter(df: pd.DataFrame, f: list) -> pd.Series:
  """
  Resolve a NOT filter.

  The first element of the filter list must be "not", followed by a filter list.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param f: Filter list.
  :type f: list
  :returns: Boolean Series resulting from applying the NOT operator.
  :rtype: pandas.Series
  """
  if len(f) < 2:
    raise ValueError("NOT operator requires at least one argument")

  values = f[1:]
  if len(values) > 1:
    raise ValueError(f"NOT operator only accepts one argument")

  selected_index = resolve_filter(df, values[0])
  return ~selected_index


def resolve_bool_filter(df: pd.DataFrame, f: list) -> pd.Series:
  """
  Resolve a list of filters using a boolean operator.

  Iterates through each filter in the list (after the operator) and combines their boolean indices
  using the specified boolean operator ("and", "or", "nand", "nor", "xor", "xnor").

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param f: List where the first element is the boolean operator and the remaining elements are filter objects.
  :type f: list
  :returns: Boolean Series resulting from applying the boolean operator.
  :rtype: pandas.Series
  """
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


def resolve_filter(df: pd.DataFrame, f: list | str | bool | None, rename_map: dict = None) -> pd.Series:
  """
  Resolve a filter expression into a boolean Series.
  Handles various filter formats including:
  - None or empty list -> all True
  - Boolean -> constant series
  - String -> treated as field name
  - List with format [operator, field, value] -> comparison operation
  - List with format [operator, field] -> unary operation
  - List with format [operator, ...filters] -> boolean operation
  - List with format ["?", filter] -> pass through filter

  :param df: DataFrame to filter.
  :type df: pandas.DataFrame
  :param f: Filter expression.
  :type f: list | str | bool | None
  :param rename_map: Optional mapping of original to renamed columns.
  :type rename_map: dict, optional
  :returns: Boolean Series.
  :rtype: pandas.Series
  """
  # Handle base cases
  if f is None:
    return pd.Series(True, index=df.index)
  if isinstance(f, bool):
    return pd.Series(f, index=df.index)
  if isinstance(f, str):
    # If it's just a field name, check if it exists and is boolean
    field_to_use = _resolve_field_name(df, f, rename_map)
    if field_to_use is not None:
      if df[field_to_use].dtype == bool:
        return df[field_to_use]
    return pd.Series(True, index=df.index)
  if len(f) == 0:
    return pd.Series(True, index=df.index)
  if len(f) == 1:
    # Single element list - try to interpret as field name
    if isinstance(f[0], str):
      field_to_use = _resolve_field_name(df, f[0], rename_map)
      if field_to_use is not None and df[field_to_use].dtype == bool:
        return df[field_to_use]
    return pd.Series(True, index=df.index)

  # Get operator
  operator = f[0]

  # Handle boolean operations
  if operator == "and":
    result = pd.Series(True, index=df.index)
    for entry in f[1:]:
      result = result & resolve_filter(df, entry, rename_map)
    return result
  if operator == "or":
    result = pd.Series(False, index=df.index)
    for entry in f[1:]:
      result = result | resolve_filter(df, entry, rename_map)
    return result
  if operator == "not":
    if len(f) < 2:
      raise ValueError(f"'not' operator requires a filter argument, got: {f}")
    return ~resolve_filter(df, f[1], rename_map)
  if operator == "?":
    if len(f) < 2:
      raise ValueError(f"'?' operator requires a filter argument, got: {f}")
    return resolve_filter(df, f[1], rename_map)

  try:
    # Get field and resolve name
    if len(f) < 2:
      raise ValueError(f"Operation requires at least [operator, field], got: {f}")
    
    field = f[1]
    field_to_use = _resolve_field_name(df, field, rename_map)
    if field_to_use is None:
      raise ValueError(f"Field not found: \"{field}\" (also tried looking up original/renamed versions)")

    # Handle unary operations (no value needed)
    if operator in ["iszeroempty", "isna", "notna"]:
      if operator == "iszeroempty": return df[field_to_use].isna() | df[field_to_use].eq(0)
      if operator == "isna": return df[field_to_use].isna()
      if operator == "notna": return df[field_to_use].notna()

    # Handle comparison operations (require value)
    if len(f) < 3:
      raise ValueError(f"Comparison operation requires [operator, field, value], got: {f}")

    # Get value and handle string constants
    value = f[2]
    if isinstance(value, list):
      if len(value) > 0 and value[0] == "str:":
        value = value[1]
    elif isinstance(value, str):
      if value.startswith("str:"):
        value = value[4:]

    # Perform the comparison
    if operator == ">": return df[field_to_use].fillna(0).gt(value)
    if operator == "<": return df[field_to_use].fillna(0).lt(value)
    if operator == ">=": return df[field_to_use].fillna(0).ge(value)
    if operator == "<=": return df[field_to_use].fillna(0).le(value)
    if operator == "==": return df[field_to_use].eq(value)
    if operator == "!=": return df[field_to_use].ne(value)
    if operator == "isin": return df[field_to_use].isin(value)
    if operator == "contains": 
      if isinstance(value, list):
        # Create a regex pattern that matches any of the values
        pattern = "|".join(map(str, value))
        return df[field_to_use].str.contains(pattern, na=False)
      return df[field_to_use].str.contains(value, na=False)
    if operator == "contains_case_insensitive": 
      if isinstance(value, list):
        # Create a regex pattern that matches any of the values
        pattern = "|".join(map(str, value))
        return df[field_to_use].str.contains(pattern, case=False, na=False)
      return df[field_to_use].str.contains(value, case=False, na=False)
    raise ValueError(f"Unknown operator: {operator}")
  except Exception as e:
    raise ValueError(f"Error processing filter {f}: {str(e)}")


def _resolve_field_name(df: pd.DataFrame, field: str, rename_map: dict = None) -> str | None:
  """
  Helper function to resolve a field name using the rename map.
  Returns the resolved field name if found, None otherwise.

  :param df: DataFrame containing fields.
  :type df: pandas.DataFrame
  :param field: Field name to resolve.
  :type field: str
  :param rename_map: Optional mapping of original to renamed columns.
  :type rename_map: dict, optional
  :returns: Resolved field name or None if not found.
  :rtype: str | None
  """
  if field in df:
    return field
  if rename_map:
    # Create reverse map for looking up original names
    reverse_map = {v: k for k, v in rename_map.items()}
    if field in reverse_map and reverse_map[field] in df:
      return reverse_map[field]
    elif field in rename_map and rename_map[field] in df:
      return rename_map[field]
  return None


def validate_filter_list(filters: list[list]):
  """
  Validate a list of filter lists.

  :param filters: List of filters (each filter is a list).
  :type filters: list[list]
  :returns: True if all filters are valid.
  :rtype: bool
  """
  for f in filters:
    validate_filter(f)
  return True


def validate_filter(f: list):
  """
  Validate a single filter list.

  Checks that the filter's operator is appropriate for the value type.

  :param f: Filter expressed as a list.
  :type f: list
  :returns: True if the filter is valid.
  :rtype: bool
  :raises ValueError: If the value type does not match the operator requirements.
  """
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


def _is_basic_operator(s: str) -> bool:
  """
  Check if the operator is a basic comparison operator.

  :param s: Operator as a string.
  :type s: str
  :returns: True if it is a basic operator.
  :rtype: bool
  """
  return s in ["<", ">", "<=", ">=", "==", "!=", "isin", "notin", "contains"]


def _is_bool_operator(s: str) -> bool:
  """
  Check if the operator is a boolean operator.

  :param s: Operator as a string.
  :type s: str
  :returns: True if it is a boolean operator.
  :rtype: bool
  """
  return s in ["and", "or", "nand", "nor", "xor", "xnor"]
