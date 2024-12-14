import pandas as pd

from openavmkit.utilities.settings import get_valuation_date


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


def fill_unknown_values_per_model_group(df, settings: dict, categorical_fields: list[str]=None):
	model_groups = df["model_group"].unique()

	df_return: pd.DataFrame | None = None

	for model_group in model_groups:
		df_group = df[df["model_group"].eq(model_group)]
		df_group = fill_unknown_values(df_group, settings, categorical_fields)

		if df_return is None:
			df_return = df_group
		else:
			df_return = pd.concat([df_return, df_group], ignore_index=True)

	return df_return


def fill_unknown_values(df, settings: dict, categorical_fields: list[str]=None):
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

	if categorical_fields is not None:
		for field in categorical_fields:
			df[field] = df[field].astype("str")
			df[field] = df[field].fillna("UNKNOWN")

	return df

