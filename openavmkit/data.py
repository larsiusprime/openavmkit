import pandas as pd
import geopandas as gpd


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