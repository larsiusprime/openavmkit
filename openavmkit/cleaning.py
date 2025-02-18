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
