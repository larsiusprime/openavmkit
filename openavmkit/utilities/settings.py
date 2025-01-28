import json
from datetime import datetime

import pandas as pd


def get_base_dir(s: dict):
	slug = s.get("locality", {}).get("slug", None)
	if slug is None:
		raise ValueError("Could not find settings.locality.slug!")
	return slug


def get_modeling_group(s: dict, key: str):
	return s.get("modeling", {}).get("modeling_groups", {}).get(key, {})


def get_valuation_date(s: dict):
	val_date_str: str | None = s.get("modeling", {}).get("metadata", {}).get("valuation_date", None)

	if val_date_str is None:
		# return January 1 of this year:
		return datetime(datetime.now().year, 1, 1)

	# process the date from string to datetime using format YYYY-MM-DD:
	val_date = datetime.strptime(val_date_str, "%Y-%m-%d")
	return val_date


def load_settings(settings_file: str = "settings.json"):
	# this assumes you've set your root directory already
	with open(settings_file, "r") as f:
		settings = json.load(f)
	template = load_settings_template()
	# merge settings with template; settings will overwrite template values
	settings = merge_settings(template, settings)
	base_dd = {
		"data_dictionary": load_data_dictionary_template()
	}
	settings = merge_settings(base_dd, settings)
	return settings


def load_data_dictionary_template():
	with open("../data_dictionary.json", "r") as f:
		data_dictionary = json.load(f)
	return data_dictionary


def load_settings_template():
	# this assumes you've set your root directory already
	with open("../settings.template.json", "r") as f:
		settings = json.load(f)
	return settings


def merge_settings(template: dict, local: dict, indent:str=""):
	# Start by copying the template
	merged = template.copy()

	# Iterate over keys of local:
	for key in local:
		entry_l = local[key]
		# If the key is in both template and local, reconcile them:
		if key in template:
			entry_t = template[key]
			if isinstance(entry_t, dict) and isinstance(entry_l, dict):
				# If both are dictionaries, merge them recursively:
				merged[key] = merge_settings(entry_t, entry_l, indent+"  ")
			elif isinstance(entry_t, list) and isinstance(entry_l, list):
				# If both are lists, add any new local items that aren't already in template:
				for item in entry_l:
					if item not in entry_t:
						entry_t.append(item)
				merged[key] = entry_t
			else:
				merged[key] = entry_l
		else:
			merged[key] = entry_l

	return merged


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


def get_fields_boolean(s: dict, df: pd.DataFrame = None, types: list[str] = None):
	if types is None:
		types = ["land", "impr", "other"]
	bools = []
	if "land" in types:
		bools += s.get("field_classification", {}).get("land", {}).get("boolean", [])
	if "impr" in types:
		bools += s.get("field_classification", {}).get("impr", {}).get("boolean", [])
	if "other" in types:
		bools += s.get("field_classification", {}).get("other", {}).get("boolean", [])
	if df is not None:
		bools = [bool for bool in bools if bool in df]
	return bools


def get_fields_categorical(s: dict, df: pd.DataFrame = None, include_boolean: bool = True, types: list[str] = None):
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
		fields_land = get_fields_categorical(settings, df, types=["land"])
		fields_impr = get_fields_categorical(settings, df, types=["impr"])
		for field in fields_land:
			result[field] = "land_area_sqft"
		for field in fields_impr:
			result[field] = "bldg_area_finished_sqft"
		return result
	else:
		return interactions.get("fields", {})


def get_data_dictionary(settings: dict):
	return settings.get("data_dictionary", {})


def apply_dd_to_df_cols(
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


def get_model_group_ids(settings: dict, df: pd.DataFrame = None):
	modeling = settings.get("modeling", {})
	model_groups = modeling.get("modeling_groups", {})
	if df is not None:
		model_groups_in_df = df["model_group"].unique()
		model_group_ids = [key for key in model_groups if key in model_groups_in_df]
	else:
		model_group_ids = [key for key in model_groups]
	return model_group_ids