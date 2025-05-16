import importlib
import json
import os
import warnings
import geopandas as gpd

import pandas as pd
from datetime import datetime

def load_settings(settings_file: str = "in/settings.json", settings_object: dict = None, error=True):
  if settings_object is None:
    try:
      with open(settings_file, "r") as f:
        settings = json.load(f)
    except FileNotFoundError:
      cwd = os.getcwd()
      full_path = os.path.join(cwd, settings_file)
      exists = os.path.exists(full_path)
      msg = f"Could not find settings file: {settings_file}. Go to '{cwd}' and create a settings.json file there! {full_path} exists? {exists}"
      if error:
        raise FileNotFoundError(msg)
      else:
        warnings.warn(msg)

  else:
    settings = settings_object

  template = _load_settings_template()
  # merge settings with template; settings will overwrite template values
  settings = _merge_settings(template, settings)
  base_dd = {
    "data_dictionary": _load_data_dictionary_template()
  }
  settings = _merge_settings(base_dd, settings)
  settings = _remove_comments_from_settings(settings)
  settings = _replace_variables(settings)
  return settings


def get_model_group(s: dict, key: str):
  return s.get("modeling", {}).get("model_groups", {}).get(key, {})


def get_valuation_date(s: dict):
  val_date_str: str | None = s.get("modeling", {}).get("metadata", {}).get("valuation_date", None)

  if val_date_str is None:
    # return January 1 of this year:
    return datetime(datetime.now().year, 1, 1)

  # process the date from string to datetime using format YYYY-MM-DD:
  val_date = datetime.strptime(val_date_str, "%Y-%m-%d")
  return val_date


def get_center(s: dict, gdf: gpd.GeoDataFrame=None)-> tuple[float, float]:
  center : dict | None = s.get("locality", {}).get("center", None)
  if center is not None:
    if "longitude" not in center or "latitude" not in center:
      raise ValueError("Could not find both 'longitude' and 'latitude' in 'settings.locality.center'!")
    latitude = center["latitude"]
    longitude = center["longitude"]
    return longitude, latitude
  elif gdf is not None:
    # calculate the center of the gdf
    centroid = gdf.geometry.unary_union.centroid
    return centroid.x, centroid.y
  else:
    raise ValueError("Could not find locality.center in settings!")


def get_fields_land(s: dict, df: pd.DataFrame=None):
  return _get_fields(s, "land", df)


def get_fields_land_as_list(s: dict, df: pd.DataFrame=None):
  fields = get_fields_land(s, df)
  return fields.get("categorical", []) + fields.get("numeric", []) + fields.get("boolean", [])


def get_fields_impr(s: dict, df: pd.DataFrame=None):
  return _get_fields(s, "impr", df)


def get_fields_impr_as_list(s: dict, df: pd.DataFrame=None):
  fields = get_fields_impr(s, df)
  return fields.get("categorical", []) + fields.get("numeric", []) + fields.get("boolean", [])


def get_fields_other(s: dict, df: pd.DataFrame=None):
  return _get_fields(s, "other", df)


def get_fields_other_as_list(s: dict, df: pd.DataFrame=None):
  fields = get_fields_other(s, df)
  return fields.get("categorical", []) + fields.get("numeric", []) + fields.get("boolean", [])


def get_fields_date(s: dict, df: pd.DataFrame):
  # TODO: add to this as necessary
  all_date_fields = [
    "sale_date",
    "date"
  ]
  date_fields = [field for field in all_date_fields if field in df]
  for field in df:
    if "_date" in field and field not in date_fields:
      date_fields.append(field)

  return date_fields


def get_fields_boolean(s: dict, df: pd.DataFrame = None, types: list[str] = None):
  """
  Get all boolean fields from settings.

  :param s: Settings dictionary.
  :type s: dict
  :param df: Optional DataFrame to filter fields.
  :type df: pandas.DataFrame, optional
  :param types: List of field types to include (e.g., ["land", "impr", "other"]).
  :type types: list[str], optional
  :returns: List of boolean field names.
  :rtype: list[str]
  """
  if types is None:
    types = ["land", "impr", "other"]
  bools = []
  
  # Determine which boolean field to get based on na_handling
  field_type = "boolean"

  if "land" in types:
    bools += s.get("field_classification", {}).get("land", {}).get(field_type, [])
  if "impr" in types:
    bools += s.get("field_classification", {}).get("impr", {}).get(field_type, [])
  if "other" in types:
    bools += s.get("field_classification", {}).get("other", {}).get(field_type, [])
  
  if df is not None:
    bools = [bool for bool in bools if bool in df]
  return bools


def get_fields_categorical(s: dict, df: pd.DataFrame = None, include_boolean: bool = False, types: list[str] = None):
  if types is None:
    types = ["land", "impr", "other"]
  cats = []
  if "land" in types:
    cats += s.get("field_classification", {}).get("land", {}).get("categorical", [])
  if "impr" in types:
    cats += s.get("field_classification", {}).get("impr", {}).get("categorical", [])
  if "other" in types:
    cats += s.get("field_classification", {}).get("other", {}).get("categorical", [])
  if include_boolean:
    if "land" in types:
      cats += s.get("field_classification", {}).get("land", {}).get("boolean", [])
    if "impr" in types:
      cats += s.get("field_classification", {}).get("impr", {}).get("boolean", [])
    if "other" in types:
      cats += s.get("field_classification", {}).get("other", {}).get("boolean", [])
  if df is not None:
    cats = [cat for cat in cats if cat in df]
  return cats


def get_fields_numeric(s: dict, df: pd.DataFrame = None, include_boolean: bool = False, types: list[str] = None):
  if types is None:
    types = ["land", "impr", "other"]
  nums = []
  if "land" in types:
    nums += s.get("field_classification", {}).get("land", {}).get("numeric", [])
  if "impr" in types:
    nums += s.get("field_classification", {}).get("impr", {}).get("numeric", [])
  if "other" in types:
    nums += s.get("field_classification", {}).get("other", {}).get("numeric", [])
  if include_boolean:
    if "land" in types:
      nums += s.get("field_classification", {}).get("land", {}).get("boolean", [])
    if "impr" in types:
      nums += s.get("field_classification", {}).get("impr", {}).get("boolean", [])
    if "other" in types:
      nums += s.get("field_classification", {}).get("other", {}).get("boolean", [])
  if df is not None:
    nums = [num for num in nums if num in df]
  return nums


def get_variable_interactions(entry: dict, settings: dict, df: pd.DataFrame = None):
  interactions: dict | None = entry.get("interactions", None)
  if interactions is None:
    return {}
  is_default = interactions.get("default", False)
  if is_default:
    result = {}
    fields_land = get_fields_categorical(settings, df, include_boolean=True, types=["land"])
    fields_impr = get_fields_categorical(settings, df, include_boolean=True, types=["impr"])
    for field in fields_land:
      result[field] = "land_area_sqft"
    for field in fields_impr:
      result[field] = "bldg_area_finished_sqft"
    return result
  else:
    return interactions.get("fields", {})


def get_data_dictionary(settings: dict):
  return settings.get("data_dictionary", {})


def get_grouped_fields_from_data_dictionary(dd: dict, group: str, types: list[str] = None) -> list[str]:
  result = []
  for key in dd:
    entry = dd[key]
    if group in entry.get("groups", []):
      if types is None or entry.get("type") in types:
        result.append(key)
  return result


def get_model_group_ids(settings: dict, df: pd.DataFrame = None):
  modeling = settings.get("modeling", {})

  # Get the model groups defined in the settings
  model_groups = modeling.get("model_groups", {})

  # Get the preferred order, if any
  order = modeling.get("instructions", {}).get("model_group_order", [])

  if df is not None:
    # If a dataframe is provided, filter out model groups that are not present in the DataFrame
    model_groups_in_df = df["model_group"].unique()
    model_group_ids = [key for key in model_groups if key in model_groups_in_df]
  else:
    model_group_ids = [key for key in model_groups]

  # Order the model groups according to the preferred order
  ordered_ids = [key for key in order if key in model_group_ids]
  unordered_ids = [key for key in model_group_ids if key not in ordered_ids]
  ordered_ids += unordered_ids

  return ordered_ids


def get_small_area_unit(settings: dict):
  base_units = settings.get("locality", {}).get("units", "imperial")
  if base_units == "imperial":
    return "sqft"
  elif base_units == "metric":
    return "sqm"


def get_large_area_unit(settings: dict):
  base_units = settings.get("locality", {}).get("units", "imperial")
  if base_units == "imperial":
    return "acre"
  elif base_units == "metric":
    return "ha" #hectare


def get_short_distance_unit(settings: dict):
  base_units = settings.get("locality", {}).get("units", "imperial")
  if base_units == "imperial":
    return "ft"
  elif base_units == "metric":
    return "m"


def get_long_distance_unit(settings: dict):
  base_units = settings.get("locality", {}).get("units", "imperial")
  if base_units == "imperial":
    return "mile"
  elif base_units == "metric":
    return "km"


# Private


def _apply_dd_to_df_cols(
    df: pd.DataFrame,
    settings: dict,
    one_hot_descendants: dict = None,
    dd_field: str = "name"
) -> pd.DataFrame:
  dd = settings.get("data_dictionary", {})

  rename_map = {}
  for column in df.columns:
    rename_map[column] = dd.get(column, {}).get(dd_field, column)

  if one_hot_descendants is not None:
    for ancestor in one_hot_descendants:
      descendants = one_hot_descendants[ancestor]
      for descendant in descendants:
        rename_map[descendant] = dd.get(ancestor, {}).get(dd_field, ancestor) + " = " + descendant[len(ancestor)+1:]

  df = df.rename(columns=rename_map)
  return df


def apply_dd_to_df_rows(
    df: pd.DataFrame,
    column: str,
    settings: dict,
    one_hot_descendants: dict = None,
    dd_field: str = "name"
) -> pd.DataFrame:
  dd = settings.get("data_dictionary", {})

  df[column] = df[column].map(lambda x: dd.get(x, {}).get(dd_field, x))
  if one_hot_descendants is not None:
    one_hot_rename_map = {}
    for ancestor in one_hot_descendants:
      descendants = one_hot_descendants[ancestor]
      for descendant in descendants:
        one_hot_rename_map[descendant] = dd.get(ancestor, {}).get(dd_field, ancestor) + " = " + descendant[len(ancestor)+1:]
    df[column] = df[column].map(lambda x: one_hot_rename_map.get(x, x))
  return df


def _get_fields(s: dict, type: str, df: pd.DataFrame = None):
  cats = s.get("field_classification", {}).get(type, {}).get("categorical", [])
  nums = s.get("field_classification", {}).get(type, {}).get("numeric", [])
  bools = s.get("field_classification", {}).get(type, {}).get("boolean", [])

  if df is not None:
    cats = [c for c in cats if c in df]
    nums = [n for n in nums if n in df]
    bools = [b for b in bools if b in df]
  return {
    "categorical": cats,
    "numeric": nums,
    "boolean": bools
  }


def _get_base_dir(s: dict):
  slug = s.get("locality", {}).get("slug", None)
  if slug is None:
    raise ValueError("Could not find settings.locality.slug!")
  return slug


def _process_settings(settings: dict):
  s = settings.copy()

  # Step 1: remove any and all keys that are prefixed with the string "__":
  s = _remove_comments_from_settings(s)

  # Step 2: do variable replacement:
  s = _replace_variables(s)

  return s


def _remove_comments_from_settings(s: dict):
  comment_token = "__"
  keys_to_remove = []
  for key in s:
    entry = s[key]
    if key.startswith(comment_token):
      keys_to_remove.append(key)
    elif isinstance(entry, dict):
      s[key] = _remove_comments_from_settings(entry)
  for k in keys_to_remove:
    del s[k]
  return s


def _replace_variables(settings: dict):

  result = settings.copy()
  failsafe = 999
  changes = 1

  while changes > 0 and failsafe > 0:
    result, changes = _do_replace_variables(result, settings)
    failsafe -= 1

  return result



def _do_replace_variables(node: dict | list | str,  settings: dict, var_token: str = "$$"):
  # For each key-value pair, search for values that are strings prefixed with $$, and replace them accordingly

  changes = 0
  replacement = node

  if isinstance(node, str):
    # Case 1 -- node is string
    str_value = str(node)
    if str_value.startswith(var_token):
      var_name = str_value[len(var_token):]
      var_value = _lookup_variable_in_settings(settings, var_name)
      replacement = var_value
      if replacement is None:
        raise ValueError(f"Variable {var_name} not found in settings!")
      changes += 1

  elif isinstance(node, dict):
    # Case 2 -- node is a dict
    _replacements = {}
    for key in node:
      entry = node[key]
      replacement, _changes = _do_replace_variables(entry, settings, var_token)
      if _changes > 0:
        _replacements[key] = replacement
        changes += _changes
    if changes > 0:
      for key in _replacements:
        node[key] = _replacements[key]
    replacement = node

  elif isinstance(node, list):
    # Case 3 -- node is a list. Go through each entry in the list.
    _replacements = {}
    for i, entry in enumerate(node):
      replacement, _changes = _do_replace_variables(entry, settings, var_token)
      if _changes > 0:
        _replacements[i] = replacement
        changes += _changes
    if changes > 0:
      for i in _replacements:
        node[i] = _replacements[i]
    replacement = node

  return replacement, changes


def _lookup_variable_in_settings(s: dict, var_name: str, path: list[str] = None):
  if path is None:
    # no path is provided, but the variable name exists:
    # split it by periods, if it has any
    path = var_name.split(".")

  if path is not None and len(path) > 0:
    first_bit = path[0]
    if first_bit in s:
      if len(path) == 1:
        # this is the last bit of the path
        return s[first_bit]
      else:
        return _lookup_variable_in_settings(s[first_bit], "", path[1:])

  return None


def _load_data_dictionary_template():
  with importlib.resources.open_text("openavmkit.resources.settings", f"data_dictionary.json", encoding="utf-8") as file:
    data_dictionary = json.load(file)
  return data_dictionary


def _load_settings_template():
  with importlib.resources.open_text("openavmkit.resources.settings", f"settings.template.json", encoding="utf-8") as file:
    settings = json.load(file)
  return settings


def is_key_in(object: dict, key: str):
  flags = ["+", "!"]
  for flag in ["+", "!", ""]:
    if f"{flag}{key}" in object:
      return True, flag
  return False, ""


def _strip_flags(settings: dict|list):
  flags = ["+", "!"]

  if isinstance(settings, list):
    for i, item in enumerate(settings):
      if isinstance(item, list) or isinstance(item, dict):
        settings[i] = _strip_flags(item)
    return settings

  if isinstance(settings, dict):
    keys_in_settings = [key for key in settings]

    for key_ in keys_in_settings:
      if key_ not in settings:
        continue
      entry = settings[key_]
      key = key_
      for flag in flags:
        if key_.startswith(flag):
          key = key_[1:]
          settings[key] = settings[key_]
          del settings[key_]
      if isinstance(entry, dict):
        entry = _strip_flags(entry)
        settings[key] = entry
      elif isinstance(entry, list):
        for i, item in enumerate(entry):
          if isinstance(item, list) or isinstance(item, dict):
            entry[i] = _strip_flags(item)
      settings[key] = entry
  return settings


def _merge_settings(template: dict, local: dict, indent:str= ""):
  # Start by copying the template
  merged = template.copy()

  # Iterate over keys of local:
  for key_ in local:

    key = key_
    local_stomps = False
    if key_.startswith("!"):
      local_stomps = True
      key = key_[1:]

    entry_l = local[key_]

    key_exists, flag = is_key_in(template, key)

    # If the key is in both template and local, reconcile them:
    if key_exists:
      local_key = f"{flag}{key}"
      add_template = False
      if not local_stomps and flag == "+":
        add_template = True

      if local_stomps:
        merged[key] = entry_l
      else:
        entry_t = template[local_key]
        if isinstance(entry_t, dict) and isinstance(entry_l, dict):
          # If both are dictionaries, merge them recursively:
          merged[key] = _merge_settings(entry_t, entry_l, indent + "  ")
        elif isinstance(entry_t, list) and isinstance(entry_l, list):
          if add_template:
            # If both are lists, add any new local items that aren't already in template:
            for item in entry_l:
              if item not in entry_t:
                entry_t.append(item)
            merged[key] = entry_t
          else:
            merged[key] = entry_l
        else:
          merged[key] = entry_l

      if flag != "" and local_key in merged:
        del merged[local_key]

    else:
      merged[key] = entry_l

  merged = _strip_flags(merged)

  return merged
