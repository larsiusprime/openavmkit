import pandas as pd
import geopandas as gpd
from pandas import Series

from openavmkit.utilities.settings import get_fields_categorical, get_fields_impr, get_fields_boolean, \
	get_fields_numeric


def enrich_time(df: pd.DataFrame) -> pd.DataFrame:
	if "sale_date" not in df:
		raise ValueError("The dataframe does not contain a 'sale_date' column.")
	# ensure "sale_date" is a datetime object:
	df["sale_date"] = pd.to_datetime(df["sale_date"])
	# create a "sale_year" column if it does not exist:
	if "sale_year" not in df:
		df["sale_year"] = df["sale_date"].dt.year
	if "sale_month" not in df:
		df["sale_month"] = df["sale_date"].dt.month
	if "sale_quarter" not in df:
		df["sale_quarter"] = df["sale_date"].dt.quarter
	if "sale_year_month" not in df:
		# format sale date in the form of "YYYY-MM"
		df["sale_year_month"] = df["sale_date"].dt.to_period("M").astype("str")
	if "sale_year_quarter" not in df:
		# format sale date in the form of "YYYY-QX"
		df["sale_year_quarter"] = df["sale_date"].dt.to_period("Q").astype("str")
	return df


def simulate_removed_buildings(df: pd.DataFrame, idx_vacant: Series, settings: dict):
	fields_impr = get_fields_impr(settings, df)

	# Step 3: fill unknown values for categorical improvements:
	fields_impr_cat = fields_impr["categorical"]
	fields_impr_num = fields_impr["numeric"]
	fields_impr_bool = fields_impr["boolean"]

	for field in fields_impr_cat:
		df.loc[idx_vacant, field] = "UNKNOWN"

	for field in fields_impr_num:
		df.loc[idx_vacant, field] = 0

	for field in fields_impr_bool:
		df.loc[idx_vacant, field] = False

	return df


def get_sale_field(settings: dict) -> str:
	ta = settings.get("modeling", {}).get("instructions", {}).get("time_adjustment", {})
	use = ta.get("use", True)
	if use:
		return "sale_price_time_adj"
	return "sale_price"


def get_vacant_sales(df_in: pd.DataFrame, settings: dict, invert:bool = False) -> pd.DataFrame:
	df = df_in.copy()
	df = boolify_column_in_df(df, "vacant_sale")
	idx_vacant_sale = df["vacant_sale"].eq(True)
	if invert:
		idx_vacant_sale = ~idx_vacant_sale
	df_vacant_sales = df[idx_vacant_sale].copy()
	return df_vacant_sales


def get_vacant(df_in: pd.DataFrame, settings: dict, invert:bool = False) -> pd.DataFrame:
	# TODO : custom vacant filter
	df = df_in.copy()

	if df["is_vacant"].dtype in ["object", "string", "str"]:
		df["is_vacant"] = df["is_vacant"].str.lower().str.strip()
		df["is_vacant"] = df["is_vacant"].replace(["true", "t", "1"], True)
		df["is_vacant"] = df["is_vacant"].replace(["false", "f", "0"], False)

	df = boolify_column_in_df(df, "is_vacant")

	idx_vacant = df["is_vacant"].eq(True)
	if invert:
		idx_vacant = ~idx_vacant
	df_vacant = df[idx_vacant].copy()
	return df_vacant


def get_sales(df_in: pd.DataFrame, settings: dict) -> pd.DataFrame:

	df = df_in.copy()
	df = boolify_column_in_df(df, "valid_sale")

	if "vacant_sale" in df:
		# check for vacant sales:
		df = boolify_column_in_df(df, "vacant_sale")
		idx_vacant_sale = df["vacant_sale"].eq(True)
		df = simulate_removed_buildings(df, idx_vacant_sale, settings)

		# if a property was NOT vacant at time of sale, but is vacant now, then the sale is invalid, because we don't
		# know e.g. what the square footage was at the time of sale
		# TODO: deal with this better when we support frozen characteristics
		df.loc[
			~idx_vacant_sale &
			df["is_vacant"].eq(True),
			"valid_sale"
		] = False

	# get the sales
	df_sales : pd.DataFrame = df[df["sale_price"].gt(0) & df["valid_sale"].eq(True)].copy()

	return df_sales


def boolify_series(series: pd.Series):
	if series.dtype in ["object", "string", "str"]:
		series = series.str.lower().str.strip()
		series = series.replace(["true", "t", "1"], 1)
		series = series.replace(["false", "f", "0"], 0)
	series = series.fillna(0)
	series = series.astype(bool)
	return series


def boolify_column_in_df(df: pd.DataFrame, field: str):
	series = df[field]
	series = boolify_series(series)
	df[field] = series
	return df


def get_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
	locations = settings.get("field_classification", {}).get("important", {}).get("locations", [])
	if df is not None:
		locations = [loc for loc in locations if loc in df]
	return locations


def get_important_fields(settings: dict, df: pd.DataFrame = None) -> list[str]:
	imp = settings.get("field_classification", {}).get("important", {})
	fields = imp.get("fields", {})
	print(f"get important fields imp = {imp}")
	print(f"fields = {fields}")
	list_fields = []
	if df is not None:
		for field in fields:
			other_name = fields[field]
			if other_name in df:
				list_fields.append(other_name)
	return list_fields


def get_important_field(settings: dict, field_name: str, df: pd.DataFrame = None) -> str | None:
	imp = settings.get("field_classification", {}).get("important", {})
	other_name = imp.get("fields", {}).get(field_name, None)
	if df is not None:
		if other_name is not None and other_name in df:
			return other_name
		else:
			return None
	return other_name


def get_dtypes_from_settings(settings: dict):
	cats = get_fields_categorical(settings, include_boolean=False)
	bools = get_fields_boolean(settings)
	nums = get_fields_numeric(settings, include_boolean=False)
	dtypes = {}
	for c in cats:
		dtypes[c] = "string"
	for b in bools:
		dtypes[b] = "bool"
	for n in nums:
		dtypes[n] = "Float64"
	return dtypes



def load_data(settings: dict) -> pd.DataFrame:
		"""
		Load the data from the settings.
		"""
		s_data = settings.get("data", {})
		s_load = s_data.get("load", {})
		dataframes = []

		dtype_map = get_dtypes_from_settings(settings)

		fields_cat = get_fields_categorical(settings, include_boolean=False)
		fields_bool = get_fields_boolean(settings)
		fields_num = get_fields_numeric(settings, include_boolean=False)

		for key in s_load:
			entry = s_load[key]
			filename = entry.get("filename", None)
			if filename is None:
				continue
			ext = filename.split(".")[-1]

			df: pd.DataFrame = None

			if ext == "parquet":
				try:
					df = gpd.read_parquet(filename)
				except ValueError as e:
					df = pd.read_parquet(filename)
			elif ext == "csv":
				df = pd.read_csv(filename, dtype=dtype_map)
			else:
				raise ValueError(f"Unsupported file extension: {ext}")

			# Fix up the types appropriately
			for col in df.columns:
				if col in fields_cat:
					df[col] = df[col].astype("string")
				elif col in fields_bool:
					df[col] = boolify_series(df[col])
				elif col in fields_num:
					df[col] = df[col].astype("Float64")

			dataframes.append(df)

		df = merge_list_of_dfs(dataframes, settings)

		return df


def merge_list_of_dfs(dfs: list[pd.DataFrame], settings: dict) -> pd.DataFrame:
		"""
		Merge the dataframes.
		"""
		s_data = settings.get("data", {})
		s_merge = s_data.get("merge", {})
		merged = None
		for key in s_merge:
			entry = s_merge[key]
			how = entry.get("how", "left")
			on = entry.get("on", "key")
			for df in dfs:
				if merged is None:
					merged = df
				else:
					merged = pd.merge(merged, df, how=how, on=on)
		return merged


