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


def get_fields_categorical(s: dict, df: pd.DataFrame = None, include_boolean: bool = True):
	land_cats = s.get("field_classification", {}).get("land", {}).get("categorical", [])
	impr_cats = s.get("field_classification", {}).get("impr", {}).get("categorical", [])
	other_cats = s.get("field_classification", {}).get("other", {}).get("categorical", [])
	cats = land_cats + impr_cats + other_cats
	if include_boolean:
		land_bools = s.get("field_classification", {}).get("land", {}).get("boolean", [])
		impr_bools = s.get("field_classification", {}).get("impr", {}).get("boolean", [])
		other_bools = s.get("field_classification", {}).get("other", {}).get("boolean", [])
		bools = land_bools + impr_bools + other_bools
		cats += bools
	if df is not None:
		cats = [cat for cat in cats if cat in df]
	return cats


def get_fields_numeric(s: dict, df: pd.DataFrame = None, include_boolean: bool = False):
	land_nums = s.get("field_classification", {}).get("land", {}).get("numeric", [])
	impr_nums = s.get("field_classification", {}).get("impr", {}).get("numeric", [])
	other_nums = s.get("field_classification", {}).get("other", {}).get("numeric", [])
	nums = land_nums + impr_nums + other_nums
	if include_boolean:
		land_bools = s.get("field_classification", {}).get("land", {}).get("boolean", [])
		impr_bools = s.get("field_classification", {}).get("impr", {}).get("boolean", [])
		other_bools = s.get("field_classification", {}).get("other", {}).get("boolean", [])
		bools = land_bools + impr_bools + other_bools
		nums += bools
	if df is not None:
		nums = [num for num in nums if num in df]
	return nums