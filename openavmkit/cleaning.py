import pandas as pd
from openavmkit.utilities.settings import get_valuation_date, get_fields_categorical, get_fields_boolean


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
			df = fill_median_impr_field(df, fill)

	# Special handling of age fields:
	for fill in ["bldg_year_built", "bldg_effective_year_built"]:
		df = fill_median_impr_field(df, fill)

	valuation_date = get_valuation_date(settings)
	valuation_year = valuation_date.year

	df["bldg_age_years"] = valuation_year - df["bldg_year_built"]
	df["bldg_effective_age_years"] = valuation_year - df["bldg_effective_year_built"]

	if cat_fields is not None:
		for field in cat_fields:
			df[field] = df[field].astype("str")
			df[field] = df[field].fillna("UNKNOWN")

	if bool_fields is not None:
		for field in bool_fields:
			df[field] = df[field].fillna(False).astype(bool)

	return df


def clean_valid_sales(df, settings):
	# load metadata
	val_date = get_valuation_date(settings)
	val_year = val_date.year
	metadata = settings.get("modeling", {}).get("metadata", {})
	use_sales_from = metadata.get("use_sales_from", val_year - 5)

	# mark which sales are to be used
	df.loc[df["sale_year"].lt(use_sales_from), "valid_sale"] = False

	# scrub sales info from invalid sales
	idx_invalid = df["valid_sale"].eq(False)
	df.loc[idx_invalid, "sale_date"] = None
	df.loc[idx_invalid, "sale_price"] = None

	print(f"Using {len(df[df['valid_sale'].eq(1)])} sales...")
	return df
