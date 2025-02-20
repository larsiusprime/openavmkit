import pandas as pd



from openavmkit.data import SalesUniversePair, get_field_classifications
from openavmkit.utilities.data import combine_dfs
from openavmkit.utilities.settings import get_valuation_date, get_fields_categorical, get_fields_boolean, \
	get_grouped_fields_from_data_dictionary, get_data_dictionary, get_model_group_ids, get_fields_numeric


def fill_median_impr_field(df_in, field, model_group=None):

	df = df_in.copy()

	if model_group is not None:
		df = df[df["model_group"].eq(model_group)]

	df = df[df["bldg_area_finished_sqft"].ge(1)]
	values = df[field]
	median = values.median()

	if pd.isna(median):
		median = 0
	elif df[field].dtype == "int64" or df[field].dtype == "Int64" or df[field].dtype == "int" or df[field].dtype == "Int":
		median = int(median)

	df.loc[
		df[field].isna() &
		df["bldg_area_finished_sqft"].ge(1),
		field
	] = median
	df.loc[
		df[field].isna() &
		df["bldg_area_finished_sqft"].eq(0) | df["bldg_area_finished_sqft"].isna(),
		field
	] = 0

	df_return = df_in.copy()
	df_return.loc[df.index, field] = df[field]

	return df_return


def identify_unknown_values(df: pd.DataFrame, settings: dict):
	# remove common columns:
	to_remove = ["key", "key2", "key3", "model_group"]

	data = {
		"field": [],
		"missing": [],
		"percent": [],
		"type": [],
		"class": []
	}

	field_map = get_field_classifications(settings)

	results = {}

	for col in df.columns.values:
		if col in to_remove:
			continue
		# count missing:
		missing = df[col].isna().sum()
		if missing == 0:
			continue
		perc_missing = missing / len(df)
		ftype = field_map.get(col, {}).get("type", "other")
		fclass = field_map.get(col, {}).get("class", "unclassified")
		data["field"].append(col)
		data["missing"].append(missing)
		data["percent"].append(perc_missing)
		data["type"].append(ftype)
		data["class"].append(fclass)
		results[col] = {
			"missing": missing,
			"percent": perc_missing,
			"type": ftype,
			"class": fclass
		}

	df_missing = pd.DataFrame(data)
	df_missing = df_missing.sort_values(by=["type","class","percent"], ascending=[True,True,False])

	return results


def infer_fill_missing_logic(df: pd.DataFrame, settings: dict):
	data = identify_unknown_values(df, settings)
	dd = get_data_dictionary(settings)

	get_grouped_fields_from_data_dictionary(dd, "building")

	for key in data:
		entry = data[key]
		dd_entry = dd.get(key, {})
		missing = entry["missing"]
		if missing > 0:
			dd_type = dd_entry.get("type", "unknown")

			# basic defaults:
			if dd_type in ["string"]:
				entry["fill"] = "unknown"
			elif dd_type in ["number", "percent"]:
				entry["fill"] = 0

			# special defaults:
			if key.startswith("dist_to_"):
				# TODO: since we calculate this, we should guarantee this comes out clean upstream
				entry["type"] = "land"
				entry["class"] = "numeric"
				entry["fill"] = "max"
			elif key in [
				"sale_year",
				"sale_month",
				"sale_age_days",
				"sale_quarter",
				"sale_price",
				"sale_price_time_adj",
				"assr_land_value",
				"assr_impr_value",
				"assr_market_value"
			]:
				entry["class"] = "numeric"
				entry["fill"] = 0



	# for key in data:
	# 	entry = data[key]
	# 	missing = entry["missing"]
	# 	if missing > 0:
	# 		ftype = entry["type"]
	# 		fclass = entry["class"]
	# 		if fclass == "unclassified":
	# 			if key.startswith("dist_to_"):
	# 				entry["type"] = "land"
	# 				entry["class"] = "numeric"
	# 				entry["fill"] = "max"
	# 			elif key in [
	# 				"sale_year",
	# 				"sale_month",
	# 				"sale_age_days",
	# 				"sale_quarter",
	# 				"sale_price",
	# 				"sale_price_time_adj",
	# 				"assr_land_value",
	# 				"assr_impr_value",
	# 				"assr_market_value"
	# 			]:
	# 				entry["class"] = "numeric"
	# 				entry["fill"] = 0
	# 			elif ftype == "categorical":
	# 				entry["fill"] = "UNKNOWN"
	# 		elif fclass == "numeric":
	# 			if key.startswith("noise_"):
	# 				entry["fill"] = 0




def sup_fill_unknown_values(sup: SalesUniversePair, settings: dict):
	df_sales = sup["sales"].copy()
	df_univ = sup["universe"].copy()

	# Fill ALL unknown values for the universe
	df_univ = fill_unknown_values_per_model_group(df_univ, settings)

	# For sales, fill ONLY the unknown values that pertain to sales metadata
	# df_sales can contain characteristics, but we want to preserve the blanks in those fields because they function
	# as overlays on top of the universe data
	dd = get_data_dictionary(settings)
	sale_fields = get_grouped_fields_from_data_dictionary(dd, "sale")
	sale_fields = [field for field in sale_fields if field in df_sales]

	df_sales_subset = df_sales[sale_fields].copy()
	df_sales_subset = fill_unknown_values(df_sales_subset, settings)
	for col in df_sales_subset:
		df_sales[col] = df_sales_subset[col]

	# max_rows_old = pd.get_option("display.max_rows")
	# pd.set_option("display.max_rows", None)
	# print("Identifying unknown values in SALES...")
	# identify_unknown_values(df_sales, settings)
	#
	# print("Identifying unknown values in UNIVERSE...")
	# identify_unknown_values(df_sales, settings)
	# pd.set_option("display.max_rows", max_rows_old)

	sup.set("sales", df_sales)
	sup.set("universe", df_univ)

	return sup


def fill_with(df: pd.DataFrame, field: str, value):
	df.loc[df[field].isna(), field] = value
	return df


def fill_thing(df: pd.DataFrame, field: str, fill_method: str):
	if fill_method == "zero":
		df = fill_with(df, field, 0)
	elif fill_method == "unknown":
		df = fill_with(df, field, "UNKNOWN")
	elif fill_method == "none":
		df = fill_with(df, field, "NONE")
	elif fill_method == "false":
		df = fill_with(df, field, False)
	elif fill_method == "mode":
		df = fill_with(df, field, df[~df[field].isna()][field].mode().iloc[0])
	elif fill_method == "median":
		df = fill_with(df, field, df[~df[field].isna()][field].median())
	elif fill_method == "mean":
		df = fill_with(df, field, df[~df[field].isna()][field].mean())
	elif fill_method == "max":
		df = fill_with(df, field, df[~df[field].isna()][field].max())
	elif fill_method == "min":
		df = fill_with(df, field, df[~df[field].isna()][field].min())
	return df


def fill_unknown_values_per_model_group(df_in: pd.DataFrame, settings: dict):
	df = df_in.copy()
	model_groups = get_model_group_ids(settings, df)

	for model_group in model_groups:
		df_mg = df[df["model_group"].eq(model_group)].copy()
		df_mg = fill_unknown_values(df_mg, settings)
		df.loc[df_mg.index, :] = df_mg

	return df


def fill_unknown_values(df, settings: dict):
	fills = settings.get("data", {}).get("process", {}).get("fill", {})

	for key in fills:
		fill_list = fills[key]
		for field in fill_list:
			if field not in df:
				continue
			fill_method = key
			if key.endswith("_impr"):
				fill_method = key[:-5]
				df_impr = df[df["is_vacant"].eq(False)].copy()
				df_impr = fill_thing(df_impr, field, fill_method)
				df.loc[df_impr.index, field] = df_impr[field]
			elif key.endswith("_vacant"):
				fill_method = key[:-7]
				df_vacant = df[df["is_vacant"].eq(True)].copy()
				df_vacant = fill_thing(df_vacant, field, fill_method)
				df.loc[df_vacant.index, field] = df_vacant[field]
			else:
				df = fill_thing(df, field, fill_method)

	valuation_date = get_valuation_date(settings)
	valuation_year = valuation_date.year

	if "bldg_age_years" in df:
		df["bldg_age_years"] = valuation_year - df["bldg_year_built"]

	if "bldg_effective_age_years" in df:
		df["bldg_effective_age_years"] = valuation_year - df["bldg_effective_year_built"]

	#remaining fields get auto-filled

	cat_fields = get_fields_categorical(settings, df, include_boolean=False)
	bool_fields = get_fields_boolean(settings, df)

	if cat_fields is not None:
		for field in cat_fields:
			if field in df:
				df[field] = df[field].astype("str")
				df[field] = df[field].fillna("UNKNOWN")

	if bool_fields is not None:
		for field in bool_fields:
			if field in df:
				df[field] = df[field].fillna(False).infer_objects(copy=False).astype(bool)

	return df


def clean_valid_sales(sup: SalesUniversePair, settings : dict):
	# load metadata
	val_date = get_valuation_date(settings)
	val_year = val_date.year
	metadata = settings.get("modeling", {}).get("metadata", {})
	use_sales_from = metadata.get("use_sales_from", val_year - 5)

	df_sales = sup["sales"].copy()
	df_univ = sup["universe"]

	# temporarily merge in universe's vacancy status (how the parcel is now)
	df_univ_vacant = df_univ[["key", "is_vacant"]].copy().rename(columns={"is_vacant": "univ_is_vacant"})
	df_sales = df_sales.merge(df_univ_vacant, on="key", how="left")

	# mark which sales are to be used (only those that are valid and within the specified time frame)
	df_sales.loc[df_sales["sale_year"].lt(use_sales_from), "valid_sale"] = False

	# initialize these -- we want to further determine which valid sales are valid for ratio studies
	df_sales["valid_for_ratio_study"] = False
	df_sales["valid_for_land_ratio_study"] = False

	# NORMAL RATIO STUDIES:
	# If it's a valid sale, and its vacancy status matches its status at time of sale, it's valid for a ratio study
	# This is because how it looked at time of sale matches how it looks now, so the prediction is comparable to the sale
	# If the vacancy status has changed since it sold, we can't meaningfully compare sale price to current valuation
	df_sales.loc[
		df_sales["valid_sale"] &
		df_sales["vacant_sale"].eq(df_sales["univ_is_vacant"]),
		"valid_for_ratio_study"
	] = True

	# LAND RATIO STUDIES:
	# If it's a valid sale, and it was vacant at time of sale, it's valid for a LAND ratio study regardless of whether it
	# is valid for a normal ratio study. That's because we will come up with a land value prediction no matter what, and
	# we can always compare that to what it sold for, as long as it was vacant at time of sale
	# we can always compare that to what it sold for, as long as it was vacant at time of sale
	df_sales.loc[
		df_sales["valid_sale"] &
		df_sales["vacant_sale"].eq(True),
		"valid_for_land_ratio_study"
	] = True

	# scrub sales info from invalid sales
	idx_invalid = df_sales["valid_sale"].eq(False)
	fields_to_scrub = [
		"sale_date",
		"sale_price",
		"sale_year",
		"sale_month",
		"sale_day",
		"sale_quarter",
		"sale_year_quarter",
		"sale_year_month",
		"sale_age_days"
	]

	for field in fields_to_scrub:
		if field in df_sales:
			df_sales.loc[idx_invalid, field] = None

	print(f"Using {len(df_sales[df_sales['valid_sale'].eq(True)])} sales...")
	print(f"--> {len(df_sales[df_sales['vacant_sale'].eq(True)])} vacant sales")
	print(f"--> {len(df_sales[df_sales['vacant_sale'].eq(False)])} improved sales")
	print(f"--> {len(df_sales[df_sales['valid_for_ratio_study'].eq(True)])} valid for ratio study")
	print(f"--> {len(df_sales[df_sales['valid_for_land_ratio_study'].eq(True)])} valid for land ratio study")

	df_sales = df_sales.drop(columns=["univ_is_vacant"])

	sup.update_sales(df_sales)

	return sup
