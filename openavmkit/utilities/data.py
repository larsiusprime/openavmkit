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


def div_field_z_safe(numerator: pd.Series, denominator: pd.Series):
	# perform a divide-by-zero-safe division of the two series, replacing divide by zero values with NaN:

	# get the index of all rows where the denominator is zero:
	idx_denominator_zero = denominator.eq(0)

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


def combine_dfs(df1:pd.DataFrame, df2:pd.DataFrame, df2_stomps=False, index="key") -> pd.DataFrame:
		"""
		Combine the dataframes on a given index column.
		"""
		df = df1.copy()

		index_orig = df.index

		df.index = df[index]
		df2.index = df2[index]

		for column in df2:
			if column == index:
				continue
			if column in df:
				# if the column already exists
				if df2_stomps:
					# fill all rows in df that match df2 with values from df2
					df.loc[df2.index, column] = df2[column]
				else:
					# fill NA values in df with values from df2
					df.loc[pd.isna(df[column]), column] = df2[column]
			else:
				# if the column does not exist, then add it
				df[column] = df2[column]

		# reset the index to the original one
		df.index = index_orig

		return df
