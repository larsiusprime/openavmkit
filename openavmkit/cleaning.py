import pandas as pd

from openavmkit.data import SalesUniversePair
from openavmkit.utilities.settings import get_valuation_date, get_fields_categorical, get_fields_boolean, \
	get_grouped_fields_from_data_dictionary, get_data_dictionary


def fill_median_impr_field(df, field):
	values = df[df["bldg_area_finished_sqft"].ge(1)][field]
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
	return df


def sup_fill_unknown_values(sup: SalesUniversePair, settings: dict):
	df_sales = sup["sales"].copy()
	df_univ = sup["universe"].copy()

	# Fill ALL unknown values for the universe
	df_univ = fill_unknown_values(df_univ, settings)

	# For sales, fill ONLY the unknown values that pertain to sales metadata
	# df_sales can contain characteristics, but we want to preserve the blanks in those fields
	dd = get_data_dictionary(settings)
	sale_fields = get_grouped_fields_from_data_dictionary(dd, "sale")
	sale_fields = [field for field in sale_fields if field in df_sales]

	df_sales_subset = df_sales[sale_fields].copy()
	df_sales_subset = fill_unknown_values(df_sales_subset, settings)
	for col in df_sales_subset:
		df_sales[col] = df_sales_subset[col]

	sup.set("sales", df_sales)
	sup.set("universe", df_univ)

	return sup


def fill_unknown_values(df, settings: dict):
	fills = [
		"bldg_area_finished_sqft",
		"bldg_quality_num",
		"bldg_condition_num"
	]

	impr_fills = [
		"bldg_area_finished_sqft",
		"bldg_quality_num",
		"bldg_condition_num"
	]

	cat_fields = get_fields_categorical(settings, df, include_boolean=False)
	bool_fields = get_fields_boolean(settings, df)

	for fill in fills:
		if fill in impr_fills:
			if fill in df:
				df = fill_median_impr_field(df, fill)

	# Special handling of age fields:
	for fill in ["bldg_year_built", "bldg_effective_year_built"]:
		if fill in df:
			df = fill_median_impr_field(df, fill)

	valuation_date = get_valuation_date(settings)
	valuation_year = valuation_date.year

	if "bldg_age_years" in df:
		df["bldg_age_years"] = valuation_year - df["bldg_year_built"]

	if "bldg_effective_age_years" in df:
		df["bldg_effective_age_years"] = valuation_year - df["bldg_effective_year_built"]

	if cat_fields is not None:
		for field in cat_fields:
			if field in df:
				df[field] = df[field].astype("str")
				df[field] = df[field].fillna("UNKNOWN")

	if bool_fields is not None:
		for field in bool_fields:
			if field in df:
				df[field] = df[field].fillna(False).astype(bool)

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
