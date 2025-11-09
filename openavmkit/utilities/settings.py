import importlib
import json
import os
import warnings
import geopandas as gpd

import pandas as pd
from datetime import datetime


def load_settings(
    settings_file: str = "in/settings.json", settings_object: dict = None, error:bool=True, warning:bool=True
) -> dict | None:
    """
    Load settings file from disk

    Parameters
    ----------
    settings_file : str
        Path to the settings file
    settings_object : dict, optional
        Already loaded settings object
    error : bool, optional
        Whether to raise errors or simply emit warnings if something is wrong
    warning : bool, optional
        Whether to emit warnings if something is wrong

    Returns
    -------
    dict
        The settings object
    """
    settings : dict | None = None

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
                if warning:
                    warnings.warn(msg)

    else:
        settings = settings_object

    if settings is None:
        return None

    template = _load_settings_template()
    # merge settings with template; settings will overwrite template values
    settings = _merge_settings(template, settings)
    base_dd = {"data_dictionary": _load_data_dictionary_template()}
    settings = _merge_settings(base_dd, settings)
    settings = _remove_comments_from_settings(settings)
    settings = _replace_variables(settings)
    return settings


def get_model_group(s: dict, key: str) -> dict:
    """
    Get a model group definition object from the settings dictionary

    Parameters
    ----------
    s : dict
        Settings object
    key : str
        The name of the model group

    Returns
    -------
    dict
        Model group definition
    """
    return s.get("modeling", {}).get("model_groups", {}).get(key, {})


def get_valuation_date(s: dict) -> datetime:
    """
    Get the valuation date from the settings dictionary

    Parameters
    ----------
    s : dict
        Settings dictionary

    Returns
    -------
    datetime
        The valuation date
    """
    val_date_str: str | None = (
        s.get("modeling", {}).get("metadata", {}).get("valuation_date", None)
    )

    if val_date_str is None:
        # return January 1 of this year:
        return datetime(datetime.now().year, 1, 1)

    # process the date from string to datetime using format YYYY-MM-DD:
    val_date = datetime.strptime(val_date_str, "%Y-%m-%d")
    return val_date


def get_look_back_dates(s: dict):
    rs = s.get("analysis", {}).get("ratio_study", {})
    look_back_years = rs.get("look_back_years", 1)
    val_date = get_valuation_date(s)
        
    # Look back N years BEFORE the valuation date
    look_back_date = val_date - pd.DateOffset(years=look_back_years)
    
    return look_back_date, val_date


def get_center(s: dict, gdf: gpd.GeoDataFrame = None) -> tuple[float, float]:
    """
    Get the centroid of all the provided parcel geometry

    Parameters
    ----------
    s : dict
        Settings dictionary
    gdf : gpd.GeoDataFrame
        Parcel geometry

    Return
    ------
    tuple[float, float]
        Centroid of all the parcel geometry
    """
    center: dict | None = s.get("locality", {}).get("center", None)
    if center is not None:
        if "longitude" not in center or "latitude" not in center:
            raise ValueError(
                "Could not find both 'longitude' and 'latitude' in 'settings.locality.center'!"
            )
        latitude = center["latitude"]
        longitude = center["longitude"]
        return longitude, latitude
    elif gdf is not None:
        # calculate the center of the gdf
        centroid = gdf.geometry.unary_union.centroid
        return centroid.x, centroid.y
    else:
        raise ValueError("Could not find locality.center in settings!")


def get_fields_land(s: dict, df: pd.DataFrame = None) -> dict:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to land.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        All fields pertaining to land, organized as a dictionary containing three keys:

          - "categorical": list of categorical fields
          - "numeric": list of numerical fields
          - "boolean": list of boolean fields
    """
    fields_land = _get_fields(s, "land", df)
    fields_unclassified = _get_unclassified_fields(s, df)

    for field in fields_unclassified:
        if field.startswith("dist_to_") or field.startswith("within_") or field.startswith("proximity_to_") or field.startswith("spatial_lag_"):
            fields_land["numeric"].append(field)

    for key in fields_land:
        # remove duplicates:
        fields_land[key] = list(set(fields_land[key]))

    return fields_land


def get_fields_land_as_list(s: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to land.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list
        A list of all field names pertaining to land
    """
    fields = get_fields_land(s, df)
    return (
        fields.get("categorical", [])
        + fields.get("numeric", [])
        + fields.get("boolean", [])
    )


def get_fields_impr(s: dict, df: pd.DataFrame = None) -> dict:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        All fields pertaining to buildings/improvements, organized as a dictionary containing three keys:

          - "categorical": list of categorical fields
          - "numeric": list of numerical fields
          - "boolean": list of boolean fields
    """
    return _get_fields(s, "impr", df)


def get_fields_impr_as_list(s: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list
        A list of all field names pertaining to buildings/improvements
    """
    fields = get_fields_impr(s, df)
    return (
        fields.get("categorical", [])
        + fields.get("numeric", [])
        + fields.get("boolean", [])
    )


def get_fields_other(s: dict, df: pd.DataFrame = None) -> dict:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to neither land nor
    buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        All fields pertaining neither to land nor to buildings/improvements,
        organized as a dictionary containing three keys:

          - "categorical": list of categorical fields
          - "numeric": list of numerical fields
          - "boolean": list of boolean fields
    """
    return _get_fields(s, "other", df)


def get_fields_other_as_list(s: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to neither land nor to
    buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list
        A list of all field names pertaining neither to land nor to buildings/improvements
    """
    fields = get_fields_other(s, df)
    return (
        fields.get("categorical", [])
        + fields.get("numeric", [])
        + fields.get("boolean", [])
    )


def get_fields_date(s: dict, df: pd.DataFrame):
    """
    Get all fields pertaining to dates

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list[str]
        List of field names pertaining to dates
    """

    # TODO: add to this as necessary
    all_date_fields = ["sale_date", "date"]
    date_fields = [field for field in all_date_fields if field in df]
    for field in df:
        if "_date" in field and field not in date_fields:
            date_fields.append(field)

    return date_fields


def get_fields_boolean(
    s: dict,
    df: pd.DataFrame = None,
    types: list[str] = None
) -> list[str]:
    """
    Retrieve boolean field names based on settings and optional filters.

    Parameters
    ----------
    s : dict
        Settings dictionary containing field configurations.
    df : pandas.DataFrame, optional
        DataFrame to filter fields by presence. Defaults to None.
    types : list[str], optional
        List of field classification types to include (e.g., ["land", "impr", "other"]).
        Defaults to None, which includes all types.

    Returns
    -------
    list[str]
        List of boolean field names matching the specified criteria.
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


def get_fields_categorical(
    s: dict,
    df: pd.DataFrame = None,
    include_boolean: bool = False,
    types: list[str] = None,
) -> list[str]:
    """
    Retrieve categorical field names based on settings and optional filters.

    Parameters
    ----------
    s : dict
        Settings dictionary containing field configurations.
    df : pandas.DataFrame, optional
        DataFrame to filter fields by presence. Defaults to None.
    include_boolean : bool, optional
        Whether to include boolean fields in the results or not. Defaults to False.
    types : list[str], optional
        List of field classification types to include (e.g., ["land", "impr", "other"]).
        Defaults to None, which includes all types.

    Returns
    -------
    list[str]
        List of categorical field names matching the specified criteria.
    """
    if types is None:
        types = ["land", "impr", "other"]
    cats = []
    if "land" in types:
        cats += s.get("field_classification", {}).get("land", {}).get("categorical", [])
    if "impr" in types:
        cats += s.get("field_classification", {}).get("impr", {}).get("categorical", [])
    if "other" in types:
        cats += (
            s.get("field_classification", {}).get("other", {}).get("categorical", [])
        )
    if include_boolean:
        if "land" in types:
            cats += s.get("field_classification", {}).get("land", {}).get("boolean", [])
        if "impr" in types:
            cats += s.get("field_classification", {}).get("impr", {}).get("boolean", [])
        if "other" in types:
            cats += (
                s.get("field_classification", {}).get("other", {}).get("boolean", [])
            )
    if df is not None:
        cats = [cat for cat in cats if cat in df]
    return cats


def get_fields_numeric(
    s: dict,
    df: pd.DataFrame = None,
    include_boolean: bool = False,
    types: list[str] = None,
) -> list[str]:
    """
     Retrieve numeric field names based on settings and optional filters.

     Parameters
     ----------
     s : dict
         Settings dictionary containing field configurations.
     df : pandas.DataFrame, optional
         DataFrame to filter fields by presence. Defaults to None.
     include_boolean : bool, optional
         Whether to include boolean fields in the results or not. Defaults to False.
     types : list[str], optional
         List of field classification types to include (e.g., ["land", "impr", "other"]).
         Defaults to None, which includes all types.

     Returns
     -------
     list[str]
         List of numeric field names matching the specified criteria.
     """
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
            nums += (
                s.get("field_classification", {}).get("other", {}).get("boolean", [])
            )
    if df is not None:
        nums = [num for num in nums if num in df]
    return nums


def get_variable_interactions(entry: dict, settings: dict, df: pd.DataFrame = None) -> dict:
    """
    Get variable interaction information from a dictionary object

    Parameters
    ----------
    entry : dict
        The dictionary object that may contain variable interactions
    settings : dict
        Global settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        Interactions dictionary which maps field names to other field names, indicating variable interactions.

        Example:
        Interacting a categorical field like "neighborhood" with a numeric field like "land_area_sqft" means that
        every one-hot-encoded descendant like "neighborhood=River Heights" will be multiplied against the numeric
        value of "land_area_sqft", so this is a way to interact neighborhood dummies with land size.
    """
    interactions: dict | None = entry.get("interactions", None)
    if interactions is None:
        return {}
    is_default = interactions.get("default", False)
    if is_default:
        result = {}
        fields_land = get_fields_categorical(
            settings, df, include_boolean=True, types=["land"]
        )
        fields_impr = get_fields_categorical(
            settings, df, include_boolean=True, types=["impr"]
        )
        for field in fields_land:
            result[field] = "land_area_sqft"
        for field in fields_impr:
            result[field] = "bldg_area_finished_sqft"
        return result
    else:
        return interactions.get("fields", {})


def get_data_dictionary(settings: dict) -> dict:
    """
    Get the data dictionary object

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    dict
        The data dictionary for this locality
    """
    return settings.get("data_dictionary", {})


def get_grouped_fields_from_data_dictionary(
    dd: dict, group: str, types: list[str] = None
) -> list[str]:
    """
    Get all field names from the data dictionary of the named group and, optionally, of the designated types.

    Parameters
    ----------
    dd : dict
        The data dictionary
    group : str
        Name of a particular group in the data dictionary
    types : list, optional
        If None, returns all field names in the group. If not, targets only those fields that match the
        listed types. Legal values are: "boolean", "str", "number", "percent", "date"

    Returns
    -------
    list[str]
        A list of field names belonging to the specified group
    """
    result = []
    for key in dd:
        entry = dd[key]
        if group in entry.get("groups", []):
            if types is None or entry.get("type") in types:
                result.append(key)
    return result


def get_model_group_ids(settings: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all model group ids specified in settings, in the preferred order specified by the user

    Parameters
    ----------
    settings : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list[str]
        Ordered list of model group ids
    """
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


def get_small_area_unit(settings: dict) -> str|None:
    """
    Get the designated "small" area unit (square feet or square meters)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "sqft" if units are imperial and "sqm" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "sqft"
    elif base_units == "metric":
        return "sqm"


def get_large_area_unit(settings: dict)-> str|None:
    """
    Get the designated "large" area unit (acre or hectare)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "acre" if units are imperial and "ha" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "acre"
    elif base_units == "metric":
        return "ha"  # hectare


def get_short_distance_unit(settings: dict) -> str|None:
    """
    Get the designated "short" distance unit (foot or meter)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "ft" if units are imperial and "m" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "ft"
    elif base_units == "metric":
        return "m"


def get_long_distance_unit(settings: dict) -> str|None:
    """
    Get the designated "long" distance unit (mile or kilometer)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "mile" if units are imperial and "km" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "mile"
    elif base_units == "metric":
        return "km"


#######################################
# PRIVATE
#######################################



def _apply_dd_to_df_cols(
    df: pd.DataFrame,
    settings: dict,
    one_hot_descendants: dict = None,
    dd_field: str = "name",
) -> pd.DataFrame:
    dd = settings.get("data_dictionary", {})

    rename_map = {}
    for column in df.columns:
        rename_map[column] = dd.get(column, {}).get(dd_field, column)

    if one_hot_descendants is not None:
        for ancestor in one_hot_descendants:
            descendants = one_hot_descendants[ancestor]
            for descendant in descendants:
                rename_map[descendant] = (
                    dd.get(ancestor, {}).get(dd_field, ancestor)
                    + " = "
                    + descendant[len(ancestor) + 1 :]
                )

    df = df.rename(columns=rename_map)
    return df


def _apply_dd_to_df_rows(
    df: pd.DataFrame,
    column: str,
    settings: dict,
    one_hot_descendants: dict = None,
    dd_field: str = "name",
) -> pd.DataFrame:
    dd = settings.get("data_dictionary", {})

    df[column] = df[column].map(lambda x: dd.get(x, {}).get(dd_field, x))
    if one_hot_descendants is not None:
        one_hot_rename_map = {}
        for ancestor in one_hot_descendants:
            descendants = one_hot_descendants[ancestor]
            for descendant in descendants:
                one_hot_rename_map[descendant] = (
                    dd.get(ancestor, {}).get(dd_field, ancestor)
                    + " = "
                    + descendant[len(ancestor) + 1 :]
                )
        df[column] = df[column].map(lambda x: one_hot_rename_map.get(x, x))
    return df


def _get_unclassified_fields(s: dict, df: pd.DataFrame = None):
    # Get all fields that are not classified as categorical, numeric, or boolean
    all = []
    for t in ["land", "impr", "other"]:
        cats = s.get("field_classification", {}).get(t, {}).get("categorical", [])
        nums = s.get("field_classification", {}).get(t, {}).get("numeric", [])
        bools = s.get("field_classification", {}).get(t, {}).get("boolean", [])
        all += cats + nums + bools

    if df is not None:
        all = [f for f in all if f in df]
        for col in df:
            if col not in all:
                all.append(col)

    return all


def _get_fields(s: dict, type: str, df: pd.DataFrame = None) -> dict:
    cats = s.get("field_classification", {}).get(type, {}).get("categorical", [])
    nums = s.get("field_classification", {}).get(type, {}).get("numeric", [])
    bools = s.get("field_classification", {}).get(type, {}).get("boolean", [])

    if df is not None:
        cats = [c for c in cats if c in df]
        nums = [n for n in nums if n in df]
        bools = [b for b in bools if b in df]

    return {"categorical": cats, "numeric": nums, "boolean": bools}


def _get_base_dir(s: dict) -> str:
    slug: str|None = s.get("locality", {}).get("slug", None)
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


def _remove_comments_from_settings(s: dict) -> dict:
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


def _replace_variables(settings: dict) -> dict:

    result = settings.copy()
    failsafe = 999
    changes = 1

    while changes > 0 and failsafe > 0:
        result, changes = _do_replace_variables(result, settings)
        failsafe -= 1

    return result


def _do_replace_variables(
    node: dict | list | str, settings: dict, var_token: str = "$$"
) -> tuple[dict|list|str, int]:
    # For each key-value pair, search for values that are strings prefixed with $$, and replace them accordingly

    changes = 0
    replacement = node

    if isinstance(node, str):
        # Case 1 -- node is string
        str_value = str(node)
        if str_value.startswith(var_token):
            var_name = str_value[len(var_token) :]
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
    with importlib.resources.open_text(
        "openavmkit.resources.settings", f"data_dictionary.json", encoding="utf-8"
    ) as file:
        data_dictionary = json.load(file)
    return data_dictionary


def _load_settings_template():
    with importlib.resources.open_text(
        "openavmkit.resources.settings", f"settings.template.json", encoding="utf-8"
    ) as file:
        settings = json.load(file)
    return settings


def _is_key_in(object: dict, key: str) -> tuple[bool, str]:
    flags = ["+", "!"]
    for flag in ["+", "!", ""]:
        if f"{flag}{key}" in object:
            return True, flag
    return False, ""


def _strip_flags(settings: dict | list) -> dict | list:
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


def _merge_settings(template: dict, local: dict, indent: str = ""):
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

        key_exists, flag = _is_key_in(template, key)

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


def _get_sales(
    df_in: pd.DataFrame,
    settings: dict,
    vacant_only: bool = False,
    df_univ: pd.DataFrame = None,
) -> pd.DataFrame:
    """Retrieve valid sales from the input DataFrame. Also simulates removed buildings if
    applicable.

    Filters for sales with a positive sale price, valid_sale marked True. If vacant_only
    is True, only includes rows where vacant_sale is True.
    """
    df = df_in.copy().reset_index(drop=True)

    if "vacant_sale" in df:
        # check for vacant sales:
        idx_vacant_sale = df["vacant_sale"].eq(True)

        # simulate removed buildings for vacant sales
        # (if we KNOW it was a vacant sale, then the building characteristics have to go)
        df = _simulate_removed_buildings(df, settings, idx_vacant_sale)

        # TODO: smell
        if "is_vacant" not in df and df_univ is not None:
            df = df.merge(df_univ[["key", "is_vacant"]], on="key", how="left")

        if "model_group" not in df and df_univ is not None:
            df = df.merge(df_univ[["key", "model_group"]], on="key", how="left")

        # if a property was NOT vacant at time of sale, but is vacant now, then the sale is invalid:
        idx_is_vacant = df["is_vacant"].eq(True)
        df.loc[~idx_vacant_sale & idx_is_vacant, "valid_sale"] = False
        idx_valid_sale = df["valid_sale"].eq(True)
        

    # Use sale_price_time_adj if it exists, otherwise use sale_price
    sale_field = "sale_price_time_adj" if "sale_price_time_adj" in df else "sale_price"
    idx_sale_price = df[sale_field].gt(0)
    
    
    idx_valid_sale = df["valid_sale"].eq(True)
    
    idx_all = idx_sale_price & idx_valid_sale & (idx_is_vacant if vacant_only else True)
    
    idx_vacant_sale = df["vacant_sale"].eq(True)
    
    
    if vacant_only:
        idx_all = idx_sale_price & idx_valid_sale & idx_vacant_sale
    else:
        idx_all = idx_sale_price & idx_valid_sale
    

    df_sales: pd.DataFrame = df[idx_all].copy()

    return df_sales


def _is_series_all_bools(series: pd.Series) -> bool:
    dtype = series.dtype
    if dtype == bool:
        return True
    # check all unique values:
    uniques = series.unique()
    for unique in uniques:
        if type(unique) != bool:
            return False
    return True


def _get_max_ratio_study_trim(settings: dict, model_group: str)->float:
    trim = settings.get("analysis",{}).get("ratio_study",{}).get("trim",{})
    entry = trim.get(model_group, trim.get("default", {}))
    return entry.get("max_percent", 0.1)


def _simulate_removed_buildings(
    df: pd.DataFrame, settings: dict, idx_vacant: pd.Series = None
) -> pd.DataFrame:
    """Simulate removed buildings by changing improvement fields to values that reflect
    the absence of a building.

    For all improvement fields, fills categorical fields with "UNKNOWN", numeric fields
    with 0, and boolean fields with False for the rows specified by idx_vacant (or all
    rows if idx_vacant is None).
    """
    if idx_vacant is None:
        # do the whole thing:
        idx_vacant = df.index

    fields_impr = get_fields_impr(settings, df)

    # fill unknown values for categorical improvements:
    fields_impr_cat = fields_impr["categorical"]
    fields_impr_num = fields_impr["numeric"]
    fields_impr_bool = fields_impr["boolean"]

    for field in fields_impr_cat:
        if not hasattr(df[field].dtype, 'categories'):
            df[field] = df[field].astype("category")
        # add UNKNOWN if needed
        if "UNKNOWN" not in df[field].cat.categories:
            df[field] = df[field].cat.add_categories(["UNKNOWN"])

    for field in fields_impr_cat:
        df.loc[idx_vacant, field] = "UNKNOWN"

    for field in fields_impr_num:
        df.loc[idx_vacant, field] = 0

    for field in fields_impr_bool:
        # Convert to boolean type first if needed
        if df[field].dtype != bool:
            df[field] = df[field].astype(bool)
        df.loc[idx_vacant, field] = False

    # just to be safe, ensure that the "bldg_area_finished_sqft" field is set to 0 for vacant sales
    # and update "is_vacant" to perfectly match
    # TODO: if we add support for a custom vacancy filter, we will need to adjust this
    if "bldg_area_finished_sqft" in df:
        df.loc[idx_vacant, "bldg_area_finished_sqft"] = 0
        # Convert is_vacant to boolean first
        if "is_vacant" not in df or df["is_vacant"].dtype != bool:
            df["is_vacant"] = False
        df.loc[idx_vacant, "is_vacant"] = True

    return df
