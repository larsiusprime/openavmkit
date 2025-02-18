import os
import pickle

import numpy as np
import pandas as pd

def clean_column_names(df: pd.DataFrame):
	# find column names that contain forbidden characters and replace them with legal representations:
	replace_map = {
		"[": "_LBRKT_",
		"]": "_RBRKT_",
		"<NA>": "_NA_",
		"<": "_LT_",
	}
	for key in replace_map:
		df.columns = df.columns.str.replace(key, replace_map[key])
	return df


def div_field_z_safe(numerator: pd.Series|np.ndarray, denominator: pd.Series|np.ndarray):
	# perform a divide-by-zero-safe division of the two series, replacing divide by zero values with NaN:

	# get the index of all rows where the denominator is zero:
	idx_denominator_zero = (denominator == 0)

	# get the series of the numerator and denominator for all rows where the denominator is not zero:
	series_numerator = numerator[~idx_denominator_zero]
	series_denominator = denominator[~idx_denominator_zero]

	# make a copy of the denominator
	result = denominator.copy()

	# replace all values where it is zero with None
	result[idx_denominator_zero] = None

	# replace all other values with the result of the division
	result[~idx_denominator_zero] = series_numerator / series_denominator
	return result

def div_z_safe(df: pd.DataFrame, numerator: str, denominator: str):
	# perform a divide-by-zero-safe division of the two columns, replacing divide by zero values with NaN:

	# get the index of all rows where the denominator is zero:
	idx_denominator_zero = df[denominator].eq(0)

	# get the series of the numerator and denominator for all rows where the denominator is not zero:
	series_numerator = df.loc[~idx_denominator_zero, numerator]
	series_denominator = df.loc[~idx_denominator_zero, denominator]

	# make a copy of the denominator
	result = df[denominator].copy()

	# replace all values where it is zero with None
	result[idx_denominator_zero] = None

	# replace all other values with the result of the division
	result[~idx_denominator_zero] = series_numerator / series_denominator
	return result


# Function to manually build Markdown
def dataframe_to_markdown(df: pd.DataFrame):
	# Create the header
	header = "| " + " | ".join(df.columns) + " |"
	separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
	rows = "\n".join(
		"| " + " | ".join(row) + " |" for row in df.astype(str).values
	)
	return f"{header}\n{separator}\n{rows}"


def rename_dict(dict, renames):
	# rename the keys of a dictionary according to a rename map:
	new_dict = {}
	for key in dict:
		new_key = renames.get(key, key)
		new_dict[new_key] = dict[key]
	return new_dict


def do_per_model_group(df_in: pd.DataFrame, func: callable, params: dict) -> pd.DataFrame:
	"""
  Apply a function to each subset of the DataFrame grouped by 'model_group',
  updating rows for which the indices match.

  Parameters:
      df_in (pd.DataFrame): Input DataFrame.
      func (callable): A function to apply to each subset.
      params (dict): Additional parameters for the function.

  Returns:
      pd.DataFrame: Modified DataFrame with updates from the function.
  """
	df = df_in.copy()
	model_groups = df["model_group"].unique()

	for model_group in model_groups:

		if pd.isna(model_group):
			continue

		# Copy params locally to avoid side effects
		params_local = params.copy()
		params_local["model_group"] = model_group

		# Filter the subset
		df_sub = df[df["model_group"].eq(model_group)]

		# Apply the function
		df_sub_updated = func(df_sub, **params_local)

		if df_sub_updated is not None:
			# Ensure consistent data types between df and df_sub_updated
			for col in df_sub_updated.columns:
				df = combine_dfs(df, df_sub_updated[["key", col]], df2_stomps=True)

	return df


def combine_dfs(df1: pd.DataFrame, df2: pd.DataFrame, df2_stomps=False, index="key") -> pd.DataFrame:
	"""
  Combine the dataframes on a given index column.

  If df2_stomps is False, NA values in df1 are filled with values from df2.
  If df2_stomps is True, values in df1 are overwritten by those in df2 for matching keys.
  """
	df = df1.copy()
	# Save the original index for restoration
	original_index = df.index.copy()

	# Work on a copy so we don’t modify df2 outside this function.
	df2 = df2.copy()

	# Set the index to the key column for alignment.
	df.index = df[index]
	df2.index = df2[index]

	# Iterate over columns in df2 (skip the key column).
	for col in df2.columns:
		if col == index:
			continue
		if col in df.columns:
			# Find the common keys to avoid KeyErrors if df2 has extra keys.
			common_idx = df.index.intersection(df2.index)
			if df2_stomps:
				# Overwrite all values in df for common keys.
				df.loc[common_idx, col] = df2.loc[common_idx, col]
			else:
				# For common keys, fill only NA values.
				na_mask = pd.isna(df.loc[common_idx, col])
				# Only assign where df2 has a value and df is NA.
				df.loc[common_idx[na_mask], col] = df2.loc[common_idx[na_mask], col]
		else:
			# Add the new column, aligning by index.
			# (Rows in df without a corresponding key in df2 will get NaN.)
			df[col] = df2[col]

	# Restore the original index.
	df.index = original_index
	return df


def add_sqft_fields(df_in: pd.DataFrame):
	df = df_in.copy()
	land_sqft = ["model_market_value", "model_land_value", "assr_market_value", "assr_land_value"]
	impr_sqft = ["model_market_value", "model_impr_value", "assr_market_value", "assr_impr_value"]
	for field in land_sqft:
		if field in df:
			df[field + "_land_sqft"] = div_field_z_safe(df[field], df["land_area_sqft"])
	for field in impr_sqft:
		if field in df:
			df[field + "_impr_sqft"] = div_field_z_safe(df[field], df["bldg_area_finished_sqft"])
	return df


def cache(path : str, logic : callable):
	outpath = path
	if os.path.exists(outpath):
		with open(outpath, "rb") as f:
			return pickle.load(f)
	result = logic()
	os.makedirs(os.path.dirname(outpath), exist_ok=True)
	with open(outpath, "wb") as f:
		pickle.dump(result, f)
	return result
