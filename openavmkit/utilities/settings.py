import json
from datetime import datetime


def get_valuation_date(settings: dict):
	val_date_str: str | None = settings.get("meta", {}).get("valuation_date", None)

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
	settings_template = load_settings_template()
	# merge settings with template; settings will overwrite template values
	merge_settings(settings_template, settings)
	return settings


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
			print(f"{indent}Merging key: {key}\n{indent}-> Template: {entry_t}\n{indent}-> Local: {entry_l}")
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