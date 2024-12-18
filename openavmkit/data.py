import pandas as pd
import geopandas as gpd
from pandas import Series

from openavmkit.utilities.settings import get_fields_categorical, get_fields_land, get_fields_impr


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
		df.loc[idx_vacant, field] = 0

	return df



def get_sales(df_in: pd.DataFrame, settings: dict) -> pd.DataFrame:
	# Step 1: get the sales
	df_sales : pd.DataFrame = df_in[df_in["sale_price"].gt(0) & df_in["valid_sale"].ge(1)].copy()

	if "vacant_sale" in df_in:
		# Step 2: check for vacant sales:
		idx_vacant_sale = df_sales["vacant_sale"].eq(1)
		df_sales = simulate_removed_buildings(df_sales, idx_vacant_sale, settings)

	return df_sales



def load_data(settings: dict) -> pd.DataFrame:
		"""
		Load the data from the settings.
		"""
		s_data = settings.get("data", {})
		s_load = s_data.get("load", {})
		dataframes = []
		for key in s_load:
			entry = s_load[key]
			filename = entry.get("filename", None)
			if filename is None:
				continue
			ext = filename.split(".")[-1]
			if ext == "parquet":
				try:
					df = gpd.read_parquet(filename)
				except ValueError as e:
					df = pd.read_parquet(filename)
				dataframes.append(df)
			elif ext == "csv":
				# TODO: process dtypes
				df = pd.read_csv(filename)
				dataframes.append(df)
		df = merge_dataframes(dataframes, settings)
		return df


def merge_dataframes(dfs: list[pd.DataFrame], settings: dict) -> pd.DataFrame:
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