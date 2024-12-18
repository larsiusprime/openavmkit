import json
from datetime import datetime

import pandas as pd


def get_base_dir(s: dict):
	slug = s.get("locality", {}).get("slug", None)
	if slug is None:
		raise ValueError("Could not find settings.locality.slug!")
	return slug


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
	return merge_settings(template, settings)


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


def get_fields_impr(s: dict, df: pd.DataFrame=None):
	return _get_fields(s, "impr", df)


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
	cats = []
	if "land" in types:
		cats += s.get("field_classification", {}).get("land", {}).get("numeric", [])
	if "impr" in types:
		cats += s.get("field_classification", {}).get("impr", {}).get("numeric", [])
	if "other" in types:
		cats += s.get("field_classification", {}).get("other", {}).get("numeric", [])
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
