import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import Series

from openavmkit.calculations import crawl_calc_dict_for_fields
from openavmkit.utilities.settings import get_fields_categorical, get_fields_impr, get_fields_boolean, \
	get_fields_numeric, get_model_group_ids, get_fields_date


def enrich_time(df: pd.DataFrame, time_formats: dict) -> pd.DataFrame:
	for key in time_formats:
		time_format = time_formats[key]
		if key in df:
			df[key] = pd.to_datetime(df[key], format=time_format, errors="coerce")

	for prefix in ["sale_"]:
		df = _enrich_time_field(df, prefix, add_year_month=True, add_year_quarter=True)

	return df


def _enrich_time_field(
		df: pd.DataFrame,
		prefix: str,
		add_year_month: bool = True,
		add_year_quarter: bool = True
) -> pd.DataFrame:

	if f"{prefix}_date" not in df:
		# Check if we have _year, _month, and _day:
		if f"{prefix}_year" in df and f"{prefix}_month" in df and f"{prefix}_day" in df:
			date_str_series = (
					df[f"{prefix}_year"].astype(str).str.pad(4,fillchar="0") + "-" +
					df[f"{prefix}_month"].astype(str).str.pad(2,fillchar="0") + "-" +
					df[f"{prefix}_day"].astype(str).str.pad(2,fillchar="0")
			)
			df[f"{prefix}_date"] = pd.to_datetime(date_str_series, format="%Y-%m-%d", errors="coerce")
		else:
			raise ValueError(f"The dataframe does not contain a '{prefix}_date' column.")

	# ensure f"{prefix}_date" is a datetime object:
	df[f"{prefix}_date"] = pd.to_datetime(df[f"{prefix}_date"], format="%Y-%m-%d", errors="coerce")

	# create a f"{prefix}_year" column if it does not exist:
	if f"{prefix}_year" not in df:
		df[f"{prefix}_year"] = df[f"{prefix}_date"].dt.year
	if f"{prefix}_month" not in df:
		df[f"{prefix}_month"] = df[f"{prefix}_date"].dt.month
	if f"{prefix}_quarter" not in df:
		df[f"{prefix}_quarter"] = df[f"{prefix}_date"].dt.quarter

	if add_year_month:
		if f"{prefix}_year_month" not in df:
			# format sale date in the form of "YYYY-MM"
			df[f"{prefix}_year_month"] = df[f"{prefix}_date"].dt.to_period("M").astype("str")

	if add_year_quarter:
		if f"{prefix}_year_quarter" not in df:
			# format sale date in the form of "YYYY-QX"
			df[f"{prefix}_year_quarter"] = df[f"{prefix}_date"].dt.to_period("Q").astype("str")

	checks = ["_year", "_month", "_day", "_year_month", "_year_quarter"]
	for check in checks:
		# Verify that the derived field a) exists and b) matches the value in the date field:
		if f"{prefix}{check}" in df:
			if f"{prefix}_date" in df:
				if check in ["_year", "_month", "_day"]:
					date_value = None
					if check == "_year":
						date_value = df[f"{prefix}_date"].dt.year.astype("Int64")
					elif check == "_month":
						date_value = df[f"{prefix}_date"].dt.month.astype("Int64")
					elif check == "_day":
						date_value = df[f"{prefix}_date"].dt.day.astype("Int64")
					if not df[f"{prefix}{check}"].astype("Int64").equals(date_value):
						# Count how many fields differ:
						n_diff = df[f"{prefix}{check}"].astype("Int64").ne(date_value).sum()
						raise ValueError(f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows.")
				elif check in ["_year_month", "_year_quarter"]:
					date_value = None
					if check == "_year_month":
						date_value = df[f"{prefix}_date"].dt.to_period("M").astype("str")
					elif check == "_year_quarter":
						date_value = df[f"{prefix}_date"].dt.to_period("Q").astype("str")
					if not df[f"{prefix}{check}"].equals(date_value):
						# Count how many fields differ:
						n_diff = df[f"{prefix}{check}"].ne(date_value).sum()
						raise ValueError(f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows.")

	return df


def old_enrich_time(df: pd.DataFrame) -> pd.DataFrame:
	if "sale_date" not in df:
		raise ValueError("The dataframe does not contain a 'sale_date' column.")
	# ensure "sale_date" is a datetime object:
	df["sale_date"] = pd.to_datetime(df["sale_date"], format="%Y-%m-%d", errors="coerce")
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


def simulate_removed_buildings(df: pd.DataFrame, settings: dict, idx_vacant: Series = None):

	if idx_vacant is None:
		# do the whole thing:
		idx_vacant = df.index

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


def  get_vacant(df_in: pd.DataFrame, settings: dict, invert:bool = False) -> pd.DataFrame:
	# TODO : support custom vacant filter from user settings
	df = df_in.copy()

	is_vacant_dtype = df["is_vacant"].dtype
	if is_vacant_dtype != bool:
		raise ValueError(f"The 'is_vacant' column must be a boolean type (found: {is_vacant_dtype})")

	idx_vacant = df["is_vacant"].eq(True)
	if invert:
		idx_vacant = ~idx_vacant
	df_vacant = df[idx_vacant].copy()
	return df_vacant


def get_sales(df_in: pd.DataFrame, settings: dict, vacant_only: bool = False) -> pd.DataFrame:
	df = df_in.copy()
	valid_sale_dtype = df["valid_sale"].dtype
	if valid_sale_dtype != bool:
		raise ValueError(f"The 'valid_sale' column must be a boolean type (found: {valid_sale_dtype})")

	if "vacant_sale" in df:
		vacant_sale_dtype = df["vacant_sale"].dtype
		if vacant_sale_dtype != bool:
			raise ValueError(f"The 'vacant_sale' column must be a boolean type (found: {vacant_sale_dtype})")
		# check for vacant sales:
		idx_vacant_sale = df["vacant_sale"].eq(True)
		df = simulate_removed_buildings(df, settings, idx_vacant_sale)

		# if a property was NOT vacant at time of sale, but is vacant now, then the sale is invalid, because we don't
		# know e.g. what the square footage was at the time of sale
		# TODO: deal with this better when we support frozen characteristics
		df.loc[
			~idx_vacant_sale &
			df["is_vacant"].eq(True),
			"valid_sale"
		] = False

	# get the sales
	df_sales : pd.DataFrame = df[
		df["sale_price"].gt(0) &
		df["valid_sale"].eq(True) &
		(df["vacant_sale"].eq(True) if vacant_only else True)
	].copy()

	return df_sales


def boolify_series(series: pd.Series):
	if series.dtype in ["object", "string", "str"]:
		series = series.astype(str).str.lower().str.strip()
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


def load_dataframes(settings: dict) -> dict[str : pd.DataFrame]:
	"""
  Load the data from the settings.
  """
	s_data = settings.get("data", {})
	s_load = s_data.get("load", {})
	dataframes = {}

	# TODO: should we even make it optional to include_booleans? Or at least make it false by default?
	fields_cat = get_fields_categorical(settings, include_boolean=False)
	fields_bool = get_fields_boolean(settings)
	fields_num = get_fields_numeric(settings, include_boolean=False)

	for key in s_load:
		entry = s_load[key]
		df = load_dataframe(entry, settings, fields_cat, fields_bool, fields_num)
		if df is not None:
			dataframes[key] = dataframes

	return dataframes



def load_dataframe(entry: dict, settings: dict, fields_cat: list = None, fields_bool: list = None, fields_num: list = None) -> pd.DataFrame | None:
	filename = entry.get("filename", None)
	if filename is None:
		return None
	ext = str(filename).split(".")[-1]

	e_load = entry.get("load", {})

	rename_map = {}
	dtype_map = {}
	extra_map = {}
	cols_to_load = []
	for rename_key in e_load:
		original = e_load[rename_key]
		original_key = None
		if isinstance(original, list):
			if len(original) > 0:
				original_key = original[0]
				cols_to_load = [original_key]
				rename_map[original_key] = rename_key
			if len(original) > 1:
				dtype_map[original_key] = original[1]
			if len(original) > 2:
				extra_map[rename_key] = original[2]

	# Get a list of every field that is either renamed or used in a calculation:
	fields_in_calc = crawl_calc_dict_for_fields(entry.get("calc", {}))
	cols_to_load += fields_in_calc
	# These are the columns we will actually load:
	cols_to_load = list(set(cols_to_load))

	# Read the actual file:
	if ext == "parquet":
		if dtype_map:
			warnings.warn("dtypes are ignored when loading parquet files.")
		try:
			df = gpd.read_parquet(filename, columns=cols_to_load)
		except ValueError as e:
			df = pd.read_parquet(filename, columns=cols_to_load)
	elif ext == "csv":
		df = pd.read_csv(filename, usecols=cols_to_load, dtype=dtype_map)
	else:
		raise ValueError(f"Unsupported file extension: {ext}")

	# Perform renames:
	df = df.rename(columns=rename_map)

	# Fix up the types appropriately
	if fields_cat is None:
		fields_cat = get_fields_categorical(settings, include_boolean=False)
	if fields_bool is None:
		fields_bool = get_fields_boolean(settings)
	if fields_num is None:
		fields_num = get_fields_numeric(settings, include_boolean=False)

	for col in df.columns:
		if col in fields_cat:
			df[col] = df[col].astype("string")
		elif col in fields_bool:
			df[col] = boolify_series(df[col])
		elif col in fields_num:
			df[col] = df[col].astype("Float64")

	# Fix up all the dates
	date_fields = get_fields_date(settings, df)
	time_format_map = {}
	for xkey in extra_map:
		if xkey in date_fields:
			# The third parameter specifies a date, if it's a date field
			time_format_map[xkey] = extra_map[xkey]

	# Ensure that all date fields have a time format specified:
	for dkey in date_fields:
		if dkey not in time_format_map:
			raise ValueError(f"Date field '{dkey}' does not have a time format specified.")

	# Enrich the time fields (e.g. add year, month, quarter, etc., and ensure all sub-fields match the date field)
	df = enrich_time(df, time_format_map)

	# Handle duplicated rows
	dupes = entry.get("dupes", {})
	df = handle_duplicated_rows(df, dupes)

	return df


def handle_duplicated_rows(df_in: pd.DataFrame, dupes: dict) -> pd.DataFrame:

	subset = dupes.get("subset", "key")

	# Count duplicates:
	num_dupes = df_in.duplicated(subset=subset).sum()

	if num_dupes > 0:
		sort_by = dupes.get("sort_by", ["key", "asc"])
		if not isinstance(sort_by, list):
			raise ValueError("sort_by must be a list of string pairs of the form [<field_name>, <asc|desc>]")
		if len(sort_by) == 2:
			if isinstance(sort_by[0], str) and isinstance(sort_by[1], str):
				sort_by = [sort_by]
		else:
			for entry in sort_by:
				if not isinstance(entry, list):
					raise ValueError(f"sort_by must be a list of string pairs of the form [<field_name>, <asc|desc>], but found a non-list entry: {entry}")
				elif len(entry) != 2:
					raise ValueError(f"sort_by must be a list of string pairs of the form [<field_name>, <asc|desc], but found an entry with {len(entry)} members: {entry}")
				elif not isinstance(entry[0], str) or not isinstance(entry[1], str):
					raise ValueError(f"sort_by must be a list of string pairs of the form [<field_name>, <asc|desc], but found an entry with non-string members: {entry}")

		df = df_in.copy()
		bys = [x[0] for x in sort_by]
		ascendings = [x[1] == "asc" for x in sort_by]
		df = df.sort_values(by=bys, ascending=ascendings)
		df = df.drop_duplicates(subset=subset, keep="first")

		return df.reset_index(drop=True)

	return df_in


def old_load_data(settings: dict) -> pd.DataFrame:
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


def write_canonical_splits(
		df_sales_in: pd.DataFrame,
		settings: dict
):
	df_sales = get_sales(df_sales_in, settings)
	model_groups = get_model_group_ids(settings, df_sales)
	instructions = settings.get("modeling", {}).get("instructions", {})
	test_train_frac = instructions.get("test_train_frac", 0.8)
	random_seed = instructions.get("random_seed", 1337)
	for model_group in model_groups:
		_write_canonical_split(model_group, df_sales, settings, test_train_frac, random_seed)


def _perform_canonical_split(
		model_group: str,
		df_sales_in: pd.DataFrame,
		settings: dict,
		test_train_fraction: float = 0.8,
		random_seed: int = 1337
):
	# Select only the relevant model group
	df = df_sales_in[df_sales_in["model_group"].eq(model_group)].copy()

	# Divide universe & sales into vacant & improved
	df_v = get_vacant_sales(df, settings)
	df_i = df.drop(df_v.index)

	# Do the split for vacant & improved property separately
	np.random.seed(random_seed)
	df_v_train = df_v.sample(frac=test_train_fraction)
	df_v_test = df_v.drop(df_v_train.index)

	df_i_train = df_i.sample(frac=test_train_fraction)
	df_i_test = df_i.drop(df_i_train.index)

	# Then piece them back together
	df_test = pd.concat([df_v_test, df_i_test]).reset_index(drop=True)
	df_train = pd.concat([df_v_train, df_i_train]).reset_index(drop=True)

	# Now the vacant test set is always guaranteed to be a perfect subset of the vacant + improved test set
	return df_test, df_train


def _write_canonical_split(
		model_group: str,
		df_sales_in: pd.DataFrame,
		settings: dict,
		test_train_fraction: float = 0.8,
		random_seed: int = 1337
):

	df_test, df_train = _perform_canonical_split(
		model_group,
		df_sales_in,
		settings,
		test_train_fraction,
		random_seed
	)

	outpath = f"out/models/{model_group}/_data"
	os.makedirs(outpath, exist_ok=True)

	df_train[["key"]].to_csv(f"{outpath}/train_keys.csv", index=False)
	df_test[["key"]].to_csv(f"{outpath}/test_keys.csv", index=False)


def read_split_keys(
		model_group: str
):
	path = f"out/models/{model_group}/_data"
	train_path = f"{path}/train_keys.csv"
	test_path = f"{path}/test_keys.csv"
	if not os.path.exists(train_path) or not os.path.exists(test_path):
		raise ValueError("No split keys found.")
	train_keys = pd.read_csv(train_path)["key"].astype(str).values
	test_keys = pd.read_csv(test_path)["key"].astype(str).values
	return test_keys, train_keys