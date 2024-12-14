import pandas as pd
import geopandas as gpd


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