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


