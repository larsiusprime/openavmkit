import pandas as pd

def clean_column_names(df: pd.DataFrame):
	# find column names that contain forbidden characters and replace them with legal representations:
	replace_map = {
		"[": "_LBRKT_",
		"]": "_RBRKT_",
		"<": "_LT_",
	}
	for key in replace_map:
		df.columns = df.columns.str.replace(key, replace_map[key])
	return df