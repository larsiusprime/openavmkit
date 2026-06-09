import numpy as np
import pandas as pd
from IPython.display import display

from openavmkit.data import _perform_canonical_split, _handle_duplicated_rows, _perform_ref_tables, _merge_dict_of_dfs, \
	_do_enrich_year_built, enrich_time, SalesUniversePair, get_hydrated_sales_from_sup, _enrich_permits, \
	compute_lookback_test_size, _resolve_strat_fields_improved, _build_strat_label, _stratified_test_sample, \
	_three_tier_split
from openavmkit.modeling import DataSplit
from openavmkit.utilities.assertions import dfs_are_equal, series_are_equal
from openavmkit.utilities.data import div_df_z_safe, merge_and_stomp_dfs, combine_dfs
from openavmkit.utilities.settings import get_valuation_date
from openavmkit.data import _boolify_series

def test_div_z_safe():
	print("")
	df = pd.DataFrame({
		"numerator": [1, 2, 3, 4, 5],
		"denominator": [0, 1, 2, 0, 4]
	})
	result = div_df_z_safe(df, "numerator", "denominator")
	assert result.isna().sum() == 2
	assert result.astype(str).eq(["<NA>","2.0","1.5","<NA>","1.25"]).all()


def test_split_keys():
	keys = [f"{i}" for i in range(10000)]

	df = pd.DataFrame(data={"key": keys})
	df["model_group"] = "residential_sf"
	df["valid_sale"] = False

	# Quick synthetic data:
	# - 10% of the data are sales
	# - 10% of the data are vacant

	df["valid_sale"] = False
	df["is_vacant"] = False
	df["vacant_sale"] = False
	df["bldg_area_finished_sqft"] = 0.0
	df["land_area_sqft"] = 0.0
	df["sale_price"] = 0.0
	df["sale_date"] = None
	df["sale_year"] = None
	df["sale_month"] = None
	df["sale_day"] = None

	#### START ANNOYING BLOCK ####

	# Set 10% of the rows to be valid sales
	df.loc[df["key"].astype(int) % 10 == 0, "valid_sale"] = True

	# Number numerically from 0 starting from the first
	df["sale_id"] = -1
	df.loc[df["valid_sale"].eq(True), "sale_id"] = df["valid_sale"].cumsum()

	df["non_sale_id"] = -1
	df["not_sale"] = df["valid_sale"].eq(False)
	df.loc[df["valid_sale"].eq(False), "non_sale_id"] = df["not_sale"].cumsum()

	# Set 10% of the sales to vacant:
	df.loc[df["sale_id"].astype(int) % 10 == 0, "is_vacant"] = True

	# Set 10% of the non-sales to vacant:
	df.loc[df["non_sale_id"].astype(int) % 10 == 0, "is_vacant"] = True

	#### END ANNOYING BLOCK ####

	df.loc[df["is_vacant"].eq(True) & df["valid_sale"].eq(True), "vacant_sale"] = True

	df["land_area_sqft"] = 10000.0
	df.loc[df["is_vacant"].eq(True), "bldg_area_finished_sqft"] = 0.0
	df.loc[df["is_vacant"].eq(False), "bldg_area_finished_sqft"] = 2000.0
	df["sale_price"] = df["valid_sale"] * ((df["bldg_area_finished_sqft"] * 80.0) + (df["land_area_sqft"] * 20.0))
	df["sale_date"] = None
	df.loc[df["valid_sale"].eq(True), "sale_date"] = "2023-01-01"
	df["sale_date"] = pd.to_datetime(df["sale_date"])
	df["key_sale"] = df["key"].astype(str) + "-" + df["sale_date"].astype(str)
	df["sale_year"] = None
	df["sale_month"] = None
	df["sale_day"] = None
	df["sale_age_days"] = None
	df.loc[df["valid_sale"].eq(True), "sale_year"] = df["sale_date"].dt.year
	df.loc[df["valid_sale"].eq(True), "sale_month"] = df["sale_date"].dt.month
	df.loc[df["valid_sale"].eq(True), "sale_day"] = df["sale_date"].dt.day
	df.loc[df["valid_sale"].eq(True), "sale_age_days"] = 0

	df_sales = df[df["valid_sale"].eq(True)].copy()

	df_test, df_train = _perform_canonical_split("residential_sf", df_sales,{}, test_train_fraction=0.8)

	test_keys = df_test["key_sale"].tolist()
	train_keys = df_train["key_sale"].tolist()

	count_vacant = len(df_sales[df_sales["is_vacant"].eq(True)])
	count_improved = len(df_sales[df_sales["is_vacant"].eq(False)])

	expected_train = len(df_sales) * 0.8
	expected_test = len(df_sales) * 0.2

	expected_train_vacant = count_vacant * 0.8
	expected_test_vacant = count_vacant * 0.2

	expected_train_improved = count_improved * 0.8
	expected_test_improved = count_improved * 0.2

	# Assert that the key splits are the expected lengths
	assert(len(test_keys) == expected_test)
	assert(len(train_keys) == expected_train)

	# Assert that test & train are the expected length
	assert(df_test.shape[0] + df_train.shape[0] == df_sales.shape[0])
	assert(df_test.shape[0] == expected_test)
	assert(df_train.shape[0] == expected_train)

	# Assert that the expected number of vacant & improved sales exist
	assert(df_test[df_test["is_vacant"].eq(True)].shape[0] == expected_test_vacant)
	assert(df_train[df_train["is_vacant"].eq(True)].shape[0] == expected_train_vacant)

	assert(df_test[df_test["is_vacant"].eq(False)].shape[0] == expected_test_improved)
	assert(df_train[df_train["is_vacant"].eq(False)].shape[0] == expected_train_improved)

	ds = DataSplit(
		name="",
		df_sales=df_sales,
		df_universe=df,
		model_group="residential_sf",
		settings={},
		dep_var="sale_price",
		dep_var_test="sale_price",
		ind_vars=["bldg_area_finished_sqft", "land_area_sqft"],
		categorical_vars=[],
		interactions={},
		test_keys=test_keys,
		train_keys=train_keys,
		vacant_only=False,
	)
	ds.split()

	ds_v = DataSplit(
		name="",
		df_sales=df_sales,
		df_universe=df,
		model_group="residential_sf",
		settings={},
		dep_var="sale_price",
		dep_var_test="sale_price",
		ind_vars=["bldg_area_finished_sqft", "land_area_sqft"],
		categorical_vars=[],
		interactions={},
		test_keys=test_keys,
		train_keys=train_keys,
		vacant_only=True,
	)
	ds_v.split()

	# Assert that both flavors of splits generated the expected lengths
	assert(ds.df_train.shape[0] == expected_train)
	assert(ds.df_test.shape[0] == expected_test)
	assert(ds_v.df_train.shape[0] == expected_train_vacant)
	assert(ds_v.df_test.shape[0] == expected_test_vacant)

	def a_equals_b(a: pd.DataFrame, b: pd.DataFrame):
		a_keys = a["key"].tolist()
		b_keys = b["key"].tolist()
		return set(a_keys) == set(b_keys)

	def a_is_subset_of_b(a: pd.DataFrame, b: pd.DataFrame):
		a_keys = a["key"].tolist()
		b_keys = b["key"].tolist()
		return set(a_keys).issubset(set(b_keys))
		result = set(a_keys).issubset(set(b_keys))
		return result

	def a_is_superset_of_b(a: pd.DataFrame, b: pd.DataFrame):
		a_keys = a["key"].tolist()
		b_keys = b["key"].tolist()
		return set(a_keys).issuperset(set(b_keys))

	# Assert that the test sets obey certain relationships:

	# ds_v.test is a strict subset of ds.test (vacant test sales only has sales also found in the vacant+improved test sales)
	assert a_is_subset_of_b(ds_v.df_test, ds.df_test)

	# ds.test is a strict superset of ds_v.test (vacant+improved test sales includes all sales found in vacant test sales)
	assert a_is_superset_of_b(ds.df_test, ds_v.df_test)

	# now intentionally screw up the data and assert the tests are FALSE (guard against broken tests yielding false positives)

	# find a key that is in ds_v and df_test:
	keys_in_ds_v = ds_v.df_test["key_sale"].tolist()
	keys_in_ds = ds.df_test["key_sale"].tolist()

	keys_in_both_ds_v_and_ds = list(set(keys_in_ds_v) & set(keys_in_ds))
	first_key_in_both_ds_v_and_ds = keys_in_both_ds_v_and_ds[0]
	second_key_in_both_ds_v_and_ds = keys_in_both_ds_v_and_ds[1]

	# remove a key from ds_v we know is in df_test:
	ds_v.df_test = ds_v.df_test[ds_v.df_test["key_sale"] != first_key_in_both_ds_v_and_ds]

	# remove a key from df_test we know is in ds_v:
	ds.df_test = ds.df_test[ds.df_test["key_sale"] != second_key_in_both_ds_v_and_ds]

	# All of these should return false now:
	assert a_is_subset_of_b(ds_v.df_test, ds.df_test) == False
	assert a_is_superset_of_b(ds.df_test, ds_v.df_test) == False


def test_duplicates():
	data = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0", "0", "1", "2"],
		"sale_price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 100, 100, 200, 300],
	}
	df = pd.DataFrame(data=data)

	dupes = {
		"subset": "key",
		"sort_by": ["key", "asc"],
		"drop": True
	}

	data_expected = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
		"sale_price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
	}
	df_expected = pd.DataFrame(data=data_expected)
	df_results = _handle_duplicated_rows(df, dupes)
	df_results = df_results.sort_values(by="key").reset_index(drop=True)
	df_expected = df_expected.sort_values(by="key").reset_index(drop=True)

	assert dfs_are_equal(df_results, df_expected, primary_key="key")

	data = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0", "0", "1", "2"],
		"sale_price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 100, 100, 200, 300],
		"sale_year": [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 1992, 1996, 1993, 1999],
	}
	df = pd.DataFrame(data=data)

	dupes = {
		"subset": "key",
		"sort_by": [["key", "asc"], ["sale_year", "desc"]],
		"drop": True
	}

	data_expected = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
		"sale_price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
		"sale_year": [1996, 1993, 1999, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000],
	}

	df_expected = pd.DataFrame(data=data_expected)
	df_results = _handle_duplicated_rows(df, dupes)

	df_results = df_results.sort_values(by="key").reset_index(drop=True)
	df_expected = df_expected.sort_values(by="key").reset_index(drop=True)

	assert dfs_are_equal(df_results, df_expected, primary_key="key")


def test_dupes_allow_keeps_all_rows():
	# Regression: a keyed source declared dupes:"allow" must keep ALL rows, not silently
	# de-dupe on key. get_dupes() resolves "allow" -> {"allow": True}; _handle_duplicated_rows
	# must honor that. (Previously "allow" resolved to {} and was de-duped on the "key" default.)
	from openavmkit.utilities.settings import get_dupes

	# "allow" resolves to an explicit, distinct signal (not the {} no-dupes default)
	assert get_dupes({"dupes": "allow"}) == {"allow": True}
	# the no-dupes-specified default still means "de-dupe on key"
	assert get_dupes({}) == {}

	data = {
		"key": ["0", "0", "0", "1", "1", "2"],
		"date": ["a", "b", "c", "d", "e", "f"],
	}
	df = pd.DataFrame(data=data)

	# resolved-allow keeps every row even though 'key' has duplicates
	assert len(_handle_duplicated_rows(df, get_dupes({"dupes": "allow"}))) == 6
	# the raw string is also honored (defensive guard)
	assert len(_handle_duplicated_rows(df, "allow")) == 6
	# sanity: the default {} still de-dupes on key
	assert len(_handle_duplicated_rows(df, {})) == 3


def test_ref_table():
	print("")

	data = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"],
		"zoning": ["R1", "R1", "R2", "R2", "R3", "C1", "C1", "C2", "C2", "R1", "M1", "M2", "M3", "M1"]
	}
	df = pd.DataFrame(data=data)

	data_ref_table = {
		"zoning_id": ["R1", "R2", "R3", "C1", "C2", "M1", "M2", "M3"],
		"zoning_density": [1, 2, 3, 1, 2, 1, 2, 3],
		"zoning_code": ["residential", "residential", "residential", "commercial", "commercial", "mixed-use", "mixed-use", "mixed-use"],
		"zoning_class": ["R", "R", "R", "C", "C", "M", "M", "M"],
		"zoning_resi_allowed": [True, True, True, False, False, True, True, True],
		"zoning_comm_allowed": [False, False, False, True, True, True, True, True],
		"zoning_mixed_use": [False, False, False, False, False, True, True, True]
	}
	df_ref_table = pd.DataFrame(data=data_ref_table)

	ref_table = {
		"id": "ref_zoning",
		"key_ref_table": "zoning_id",
		"key_target": "zoning",
		"add_fields": ["zoning_density", "zoning_code", "zoning_class", "zoning_resi_allowed", "zoning_comm_allowed", "zoning_mixed_use"]
	}

	dataframes = {
		"ref_zoning": df_ref_table
	}

	data_expected = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"],
		"zoning": ["R1", "R1", "R2", "R2", "R3", "C1", "C1", "C2", "C2", "R1", "M1", "M2", "M3", "M1"],
		"zoning_density": [1, 1, 2, 2, 3, 1, 1, 2, 2, 1, 1, 2, 3, 1],
		"zoning_code": ["residential", "residential", "residential", "residential", "residential", "commercial", "commercial", "commercial", "commercial", "residential", "mixed-use", "mixed-use", "mixed-use", "mixed-use"],
		"zoning_class": ["R", "R", "R", "R", "R", "C", "C", "C", "C", "R", "M", "M", "M", "M"],
		"zoning_resi_allowed": [True, True, True, True, True, False, False, False, False, True, True, True, True, True],
		"zoning_comm_allowed": [False, False, False, False, False, True, True, True, True, False, True, True, True, True],
		"zoning_mixed_use": [False, False, False, False, False, False, False, False, False, False, True, True, True, True]
	}
	df_expected = pd.DataFrame(data=data_expected)
	df_results = _perform_ref_tables(df, ref_table, dataframes)

	# Test the case where the keys are different
	assert dfs_are_equal(df_expected, df_results, primary_key="key")

	# Test the case where we do it in two lookups
	ref_tables = [
		{
			"id": "ref_zoning",
			"key_ref_table": "zoning_id",
			"key_target": "zoning",
			"add_fields": ["zoning_density", "zoning_code", "zoning_class"]
		},
		{
			"id": "ref_zoning",
			"key_ref_table": "zoning_id",
			"key_target": "zoning",
			"add_fields": ["zoning_resi_allowed", "zoning_comm_allowed", "zoning_mixed_use"]
		},
	]

	df_results = _perform_ref_tables(df, ref_tables, dataframes)

	assert dfs_are_equal(df_expected, df_results, primary_key="key")

	# Test the case where the keys are identical
	dataframes["ref_zoning"] = dataframes["ref_zoning"].rename(columns={"zoning_id": "zoning"})
	ref_table["key_ref_table"] = "zoning"

	df_results = _perform_ref_tables(df, ref_table, dataframes)

	assert dfs_are_equal(df_expected, df_results, primary_key="key")


def test_merge_conflicts():

	datas = {
		"a": {
			"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
			"fruit": ["apple", None, None, None, "elderberry", "fig", "grape", None, None, None],
		},
		"b": {
			"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
			"fruit": [None, "banana", "cherry", "date", None, None, None, None, None, None],
		},
		"c": {
			"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
			"fruit": [None, None, None, None, None, None, None, "honeydew", "kiwi", "lemon"],
		}
	}

	dfs = {}

	for data in datas:
		df = pd.DataFrame(data=datas[data])
		dfs[data] = df

	_merge_dict_of_dfs(
		dataframes=dfs,
		merge_list=["a", "b", "c"],
		settings={}
	)


def test_enrich_year_built():
	data = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
		"sale_date": [None, None, None, "2021-01-01", None, None, None, None, None, "2022-11-15", None],
		"valid_sale": [False, False, False, True, False, False, False, False, False, True, False],
		"sale_price": [None, None, None, 100000, None, None, None, None, None, 200000, None],
		"bldg_year_built": [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000]
	}

	df = pd.DataFrame(data=data)

	df_sales = df[df["valid_sale"].eq(True)].copy().reset_index(drop=True)
	df_univ = df.copy()

	val_date = pd.to_datetime("2025-01-01")

	expected_univ = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
		"sale_date": [None, None, None, "2021-01-01", None, None, None, None, None, "2022-11-15", None],
		"valid_sale": [False, False, False, True, False, False, False, False, False, True, False],
		"sale_price": [None, None, None, 100000, None, None, None, None, None, 200000, None],
		"bldg_year_built": [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000],
		"sale_year": [None, None, None, "2021", None, None, None, None, None, "2022", None],
		"sale_month": [None, None, None, "1", None, None, None, None, None, "11", None],
		"sale_day": [None, None, None, "1", None, None, None, None, None, "15", None],
		"sale_quarter": [None, None, None, "1", None, None, None, None, None, "4", None],
		"sale_year_month": ["NaT", "NaT", "NaT", "2021-01", "NaT", "NaT", "NaT", "NaT", "NaT", "2022-11", "NaT"],
		"sale_year_quarter": ["NaT", "NaT", "NaT", "2021Q1", "NaT", "NaT", "NaT", "NaT", "NaT", "2022Q4", "NaT"],
		"sale_age_days": [None, None, None, 1461, None, None, None, None, None, 778, None],
		"bldg_age_years": [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
	}

	expected_sales = {
		"key": ["3", "9"],
		"sale_date": ["2021-01-01", "2022-11-15"],
		"valid_sale": [True, True],
		"sale_price": [100000.0, 200000.0],
		"bldg_year_built": [1993, 1999],
		"sale_year": ["2021", "2022"],
		"sale_month": ["1", "11"],
		"sale_day": ["1", "15"],
		"sale_quarter": ["1", "4"],
		"sale_year_month": ["2021-01", "2022-11"],
		"sale_year_quarter": ["2021Q1", "2022Q4"],
		"sale_age_days": [1461, 778],
		"bldg_age_years": [28.0, 23.0]
	}

	time_formats = {"sale_date":"%Y-%m-%d"}

	test_settings = {
		"modeling":{
			"metadata":{
				"valuation_date": "2025-01-01"
			}
		}
	}

	df_univ = enrich_time(df_univ, time_formats, test_settings)
	df_sales = enrich_time(df_sales, time_formats, test_settings)

	df_univ = _do_enrich_year_built(df_univ, "bldg_year_built", "bldg_age_years", val_date, False)
	df_sales = _do_enrich_year_built(df_sales, "bldg_year_built", "bldg_age_years", val_date, True)

	df_univ_expected = pd.DataFrame(data=expected_univ)
	df_sales_expected = pd.DataFrame(data=expected_sales)

	for thing in ["sale_year", "sale_month", "sale_quarter", "sale_age_days"]:
		df_univ[thing] = df_univ[thing].astype("Int64").astype("string")
		df_sales[thing] = df_sales[thing].astype("Int64").astype("string")
		df_univ_expected[thing] = df_univ_expected[thing].astype("Int64").astype("string")
		df_sales_expected[thing] = df_sales_expected[thing].astype("Int64").astype("string")

	for thing in ["sale_date", "sale_year_month", "sale_year_quarter", "sale_age_days"]:
		df_univ[thing] = df_univ[thing].astype("string")
		df_sales[thing] = df_sales[thing].astype("string")
		df_univ_expected[thing] = df_univ_expected[thing].astype("string")
		df_sales_expected[thing] = df_sales_expected[thing].astype("string")

	assert dfs_are_equal(df_univ, df_univ_expected, primary_key="key")
	assert dfs_are_equal(df_sales, df_sales_expected, primary_key="key")


def test_get_sales_from_sup():
	data = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
		"sale_date": [None, None, None, "2021-01-01", None, None, None, None, None, "2022-11-15", None],
		"valid_sale": [False, False, False, True, False, False, False, False, False, True, False],
		"sale_price": [None, None, None, 100000, None, None, None, None, None, 200000, None],
		"bldg_year_built": [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000],
		"bldg_quality_txt": ["average", "average", "average", "average", "average", "average", "average", "average", "average", "average", "average"],
		"land_class": ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"],
	}

	df = pd.DataFrame(data=data)

	df_sales = df[df["valid_sale"].eq(True)].copy().reset_index(drop=True)
	df_sales = df_sales.drop(columns=["land_class"])
	df_sales["bldg_quality_txt"] = "good"

	df_univ = df.copy()

	sup = SalesUniversePair(sales=df_sales, universe=df_univ)

	df_sales_hydrated = get_hydrated_sales_from_sup(sup).reset_index(drop=True)

	data_expected = {
		"key": ["3", "9"],
		"sale_date": ["2021-01-01", "2022-11-15"],
		"valid_sale": [True, True],
		"sale_price": [100000.0, 200000.0],
		"bldg_year_built": [1993, 1999],
		"bldg_quality_txt": ["good", "good"],
		"land_class": ["R", "R"],
	}
	df_expected = pd.DataFrame(data=data_expected)

	assert dfs_are_equal(df_sales_hydrated, df_expected, primary_key="key")


def test_combine_dfs():
	data_1 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["apple", "banana", "cherry", "date", "elderberry", None, None, None, None, None],
		"color": [None, "yellow", "red", "brown", "purple", "green", "purple", "green", "brown", "yellow"]
	}

	data_2 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["APPLE", "BANANA", "CHERRY", "DATE", "ELDERBERRY", "FIG", "GRAPE", "HONEYDEW", "KIWI", "LEMON"],
		"color": ["RED", "YELLOW", "RED", "BROWN", "PURPLE", "GREEN", "PURPLE", "GREEN", "BROWN", "YELLOW"]
	}

	data_3 = {
		"key": ["0", "1", "2", "3"],
		"fruit": ["grape", "graper", "grapest", "graperlative"],
		"color": ["purple", "purpler", "purplest", "purplerlative"]
	}

	df1 = pd.DataFrame(data=data_1)
	df2 = pd.DataFrame(data=data_2)
	df3 = pd.DataFrame(data=data_3)

	expected_1 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["apple", "banana", "cherry", "date", "elderberry", "FIG", "GRAPE", "HONEYDEW", "KIWI", "LEMON"],
		"color": ["RED", "yellow", "red", "brown", "purple", "green", "purple", "green", "brown", "yellow"]
	}
	expected_2 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["APPLE", "BANANA", "CHERRY", "DATE", "ELDERBERRY", "FIG", "GRAPE", "HONEYDEW", "KIWI", "LEMON"],
		"color": ["RED", "YELLOW", "RED", "BROWN", "PURPLE", "GREEN", "PURPLE", "GREEN", "BROWN", "YELLOW"]
	}
	expected_3 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["apple", "banana", "cherry", "date", "elderberry", None, None, None, None, None],
		"color": ["purple", "yellow", "red", "brown", "purple", "green", "purple", "green", "brown", "yellow"]
	}
	expected_4 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["grape", "graper", "grapest", "graperlative", "elderberry", None, None, None, None, None],
		"color": ["purple", "purpler", "purplest", "purplerlative", "purple", "green", "purple", "green", "brown", "yellow"]
	}
	expected1 = pd.DataFrame(data=expected_1)
	expected2 = pd.DataFrame(data=expected_2)
	expected3 = pd.DataFrame(data=expected_3)
	expected4 = pd.DataFrame(data=expected_4)

	merged1 = combine_dfs(df1, df2, df2_stomps=False)
	merged2 = combine_dfs(df1, df2, df2_stomps=True)
	merged3 = combine_dfs(df1, df3, df2_stomps=False)
	merged4 = combine_dfs(df1, df3, df2_stomps=True)

	display(merged1)
	display(expected1)

	assert dfs_are_equal(merged1, expected1, primary_key="key")
	assert dfs_are_equal(merged2, expected2, primary_key="key")
	assert dfs_are_equal(merged3, expected3, primary_key="key")
	assert dfs_are_equal(merged4, expected4, primary_key="key")


def test_merge_and_stomp_dfs():
	data_1 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["apple", "banana", "cherry", "date", "elderberry", None, None, None, None, None],
		"color": [None, "yellow", "red", "brown", "purple", "green", "purple", "green", "brown", "yellow"]
	}

	data_2 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["APPLE", "BANANA", "CHERRY", "DATE", "ELDERBERRY", "FIG", "GRAPE", "HONEYDEW", "KIWI", "LEMON"],
		"color": ["RED", "YELLOW", "RED", "BROWN", "PURPLE", "GREEN", "PURPLE", "GREEN", "BROWN", "YELLOW"]
	}

	data_3 = {
		"key": ["0", "1", "2", "3"],
		"fruit": ["grape", "graper", "grapest", "graperlative"],
		"color": ["purple", "purpler", "purplest", "purplerlative"]
	}

	df1 = pd.DataFrame(data=data_1)
	df2 = pd.DataFrame(data=data_2)
	df3 = pd.DataFrame(data=data_3)

	expected_1 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["apple", "banana", "cherry", "date", "elderberry", "FIG", "GRAPE", "HONEYDEW", "KIWI", "LEMON"],
		"color": ["RED", "yellow", "red", "brown", "purple", "green", "purple", "green", "brown", "yellow"]
	}
	expected_2 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["APPLE", "BANANA", "CHERRY", "DATE", "ELDERBERRY", "FIG", "GRAPE", "HONEYDEW", "KIWI", "LEMON"],
		"color": ["RED", "YELLOW", "RED", "BROWN", "PURPLE", "GREEN", "PURPLE", "GREEN", "BROWN", "YELLOW"]
	}
	expected_3 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["apple", "banana", "cherry", "date", "elderberry", None, None, None, None, None],
		"color": ["purple", "yellow", "red", "brown", "purple", "green", "purple", "green", "brown", "yellow"]
	}
	expected_4 = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"fruit": ["grape", "graper", "grapest", "graperlative", "elderberry", None, None, None, None, None],
		"color": ["purple", "purpler", "purplest", "purplerlative", "purple", "green", "purple", "green", "brown", "yellow"]
	}
	expected1 = pd.DataFrame(data=expected_1)
	expected2 = pd.DataFrame(data=expected_2)
	expected3 = pd.DataFrame(data=expected_3)
	expected4 = pd.DataFrame(data=expected_4)

	merged1 = merge_and_stomp_dfs(df1, df2, df2_stomps=False)
	merged2 = merge_and_stomp_dfs(df1, df2, df2_stomps=True)
	merged3 = merge_and_stomp_dfs(df1, df3, df2_stomps=False)
	merged4 = merge_and_stomp_dfs(df1, df3, df2_stomps=True)

	assert dfs_are_equal(merged1, expected1, primary_key="key")
	assert dfs_are_equal(merged2, expected2, primary_key="key")
	assert dfs_are_equal(merged3, expected3, primary_key="key")
	assert dfs_are_equal(merged4, expected4, primary_key="key")


def test_update_sales():
	sales = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"sale_price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
		"sale_price_time_adj": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
		"sale_date": ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01"],
		"key_sale": ["0---2025-01-01", "1---2025-01-01", "2---2025-01-01", "3---2025-01-01", "4---2025-01-01", "5---2025-01-01", "6---2025-01-01", "7---2025-01-01", "8---2025-01-01", "9---2025-01-01"],
		"suspicious": [True, True, True, False, False, False, False, False, False, False],
		"valid_sale": [True, True, True, True, True, True, True, True, True, True]
	}
	univ = {
		"key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
		"bldg_area_finished_sqft": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
		"land_area_sqft": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
	}
	df_sales = pd.DataFrame(sales)
	df_univ = pd.DataFrame(univ)

	sup = SalesUniversePair(
		sales=df_sales,
		universe=df_univ
	)

	df_sales = sup.sales.copy()

	df_sales.loc[df_sales["suspicious"].eq(True), "valid_sale"] = False
	df_sales = df_sales[df_sales["valid_sale"].eq(True)]

	num_valid_before = len(sup.sales[sup.sales["valid_sale"].eq(True)])
	len_before = len(sup.sales)

	sup.update_sales(df_sales, allow_remove_rows=True)

	num_valid_after = len(sup.sales[sup.sales["valid_sale"].eq(True)])
	len_after = len(sup.sales)

	assert num_valid_before == 10
	assert len_before == 10
	assert num_valid_after == 7
	assert len_after == 7


def test_permits_teardown_sales():
	print("")

	sales = {
		"key": ["0", "1", "2", "3"],
		"valid_sale": [True, True, True, True],
		"vacant_sale": [False, False, False, False],
		"sale_price": [1, 1, 1, 1],
		"sale_date": [
			"2020-06-01",
			"2020-06-01",
			"2020-06-01",
			"2020-06-01"
		]
	}

	nan = float('nan')
	# A teardown sale = buyer purchases, then demolishes within max_days_to_demo (default 365).
	# Demos BEFORE the sale are sales of already-cleared lots and are not flagged.
	permits = {
		"key": ["0", "1", "2", "3", "3", "3"],
		"is_teardown": [True, True, True, True, True, True],
		"date": [
			              # One demo each for keys 0, 1, 2
			"2020-07-01", # IN-WINDOW:  1 month AFTER sale  -> teardown
			"2020-01-01", # BEFORE:     5 months BEFORE sale -> not a teardown (cleared lot)
			"2021-07-01", # OUT-WINDOW: 13 months AFTER sale -> not a teardown (too late)

			              # 3 demo dates for key 3 -- de-duplicate to the closest valid demo
			              # AFTER the sale, which is 2020-07-01 (30 days out).
			"2020-07-01", # in-window
			"2020-01-01", # before sale (ignored)
			"2021-07-01"  # too late
		]
	}
	expected = {
		"key": ["0", "1", "2", "3"],
		"valid_sale": [True, True, True, True],
		# is_teardown_sale=True flips vacant_sale to True (sale of effectively land-only)
		"vacant_sale": [True, False, False, True],
		"sale_price": [1, 1, 1, 1],
		"sale_date": ["2020-06-01", "2020-06-01", "2020-06-01", "2020-06-01"],
		"key_sale": ["0---2020-06-01", "1---2020-06-01", "2---2020-06-01", "3---2020-06-01"],
		"is_teardown_sale": [True, False, False, True],
		# demo_date is preserved for every key that had any matching permit, even if
		# the demo was before the sale (days_to_demo gets NaN'd, demo_date stays).
		"demo_date": ["2020-07-01", "2020-01-01", "2021-07-01", "2020-07-01"],
		"days_to_demo":[30.0, nan, 395.0, 30.0]
	}

	df_expected = pd.DataFrame(data=expected)

	df_sales = pd.DataFrame(data=sales)
	df_sales["key_sale"] = df_sales["key"] + "---" + df_sales["sale_date"]
	df_sales["sale_date"] = pd.to_datetime(df_sales["sale_date"], format="%Y-%m-%d")
	df_permits = pd.DataFrame(data=permits)
	df_permits["date"] = pd.to_datetime(df_permits["date"], format="%Y-%m-%d")

	settings = {
		"data":{
			"process":{
				"enrich":{
					"sales":{
						"permits":{
							"sources": ["permits"]
						}
					}
				}
			}
		}
	}
	s_enrich_sales = settings.get("data", {}).get("process", {}).get("enrich", {}).get("sales")
	dataframes = {"permits": df_permits}

	df_results = _enrich_permits(
		df_sales,
		s_enrich_sales,
		dataframes,
		settings,
		is_sales=True,
		verbose=True
	)

	assert dfs_are_equal(df_expected, df_results, allow_weak=True)


def test_permits_reno_sales():
	print("")

	sales = {
		"key": ["0", "1", "2", "3", "4"],
		"valid_sale": [True, True, True, True, True],
		"vacant_sale": [False, False, False, False, False],
		"sale_price": [1, 1, 1, 1, 1],
		"sale_date": [
			"2020-06-01",
			"2020-06-01",
			"2020-06-01",
			"2020-06-01",
			"2020-06-01",
		]
	}

	nan = float('nan')
	permits = {
		"key": ["0", "1", "2", "3", "3", "3", "4", "4", "4"],
		"is_renovation": [True, True, True, True, True, True, True, True, True],
		"renovation_num": [2, 3, 3, 1, 2, 3, 3, 2, 1],
		"renovation_txt": ["medium", "major", "major", "minor", "medium", "major", "major", "medium", "minor"],
		"date": [
			# reno dates for keys 1, 2, and 3
			"2020-05-01", # before the sale, picked
			"2020-07-01", # after the sale, dismissed
			"2010-06-01", # before the sale, picked

			# 3 reno dates, all for key 3 -- should de-duplicate and pick the last one (best one)
			"2020-05-01",
			"2020-05-10",
			"2020-05-20",

			# 4 reno dates, all for key 4 -- should de-duplicate and pick the first one (best one)
			"2020-05-01",
			"2020-06-01",
			"2020-07-01"
		]
	}
	expected = {
		"key": ["0", "1", "2", "3", "4"],
		"valid_sale": [True, True, True, True, True],
		"vacant_sale": [False, False, False, False, False],
		"sale_price": [1, 1, 1, 1, 1],
		"sale_date": ["2020-06-01", "2020-06-01", "2020-06-01", "2020-06-01", "2020-06-01"],
		"key_sale": ["0---2020-06-01", "1---2020-06-01", "2---2020-06-01", "3---2020-06-01", "4---2020-06-01"],
		"is_renovated": [True, False, True, True, True],
		"reno_date": ["2020-05-01", None, "2010-06-01", "2020-05-20", "2020-05-01"],
		"renovation_num": [2, None, 3, 3, 3],
		"renovation_txt": ["medium", None, "major", "major", "major"],
		"days_to_reno": [-31.0, None, -3653.0, -12.0, -31.0]
	}

	df_expected = pd.DataFrame(data=expected)
	df_expected["reno_date"] = pd.to_datetime(df_expected["reno_date"], format="%Y-%m-%d")

	df_sales = pd.DataFrame(data=sales)
	df_sales["key_sale"] = df_sales["key"] + "---" + df_sales["sale_date"]
	df_sales["sale_date"] = pd.to_datetime(df_sales["sale_date"], format="%Y-%m-%d")
	df_permits = pd.DataFrame(data=permits)
	df_permits["date"] = pd.to_datetime(df_permits["date"], format="%Y-%m-%d")

	settings = {
		"data":{
			"process":{
				"enrich":{
					"sales":{
						"permits":{
							"sources": ["permits"]
						}
					}
				}
			}
		}
	}
	s_enrich_sales = settings.get("data", {}).get("process", {}).get("enrich", {}).get("sales")
	dataframes = {"permits": df_permits}

	df_results = _enrich_permits(
		df_sales,
		s_enrich_sales,
		dataframes,
		settings,
		is_sales=True,
		verbose=True
	)

	assert dfs_are_equal(df_expected, df_results, allow_weak=True)


def test_boolify_series():

	bool_series = pd.Series([True, False, True, False, None])
	boolean_series = pd.Series([True, False, True, False, None]).astype("boolean")
	int_series = pd.Series([1, 0, 1, 0, None])
	mixed_series = pd.Series([1, 0, True, False, None])
	str_series_1 = pd.Series(["true", "false", "t", "f", ""])
	str_series_2 = pd.Series(["1", "0", "TRUE", "FALSE", "none"])
	str_series_3 = pd.Series(["T", "F", "y", "n", "unknown"])

	bool_series = _boolify_series(bool_series)
	boolean_series = _boolify_series(boolean_series)
	int_series = _boolify_series(int_series)
	mixed_series = _boolify_series(mixed_series)
	str_series_1 = _boolify_series(str_series_1)
	str_series_2 = _boolify_series(str_series_2)
	str_series_3 = _boolify_series(str_series_3)

	expected_series = pd.Series([True, False, True, False, False])

	series_are_equal(expected_series, bool_series)
	series_are_equal(expected_series, boolean_series)
	series_are_equal(expected_series, int_series)
	series_are_equal(expected_series, mixed_series)
	series_are_equal(expected_series, str_series_1)
	series_are_equal(expected_series, str_series_2)
	series_are_equal(expected_series, str_series_3)


# ---------------------------------------------------------------------------
# Tests for the new canonical-split helpers: 30/15/2x rule, three-tier split,
# stratification. Each test names the high-level behavior it pins down.
# ---------------------------------------------------------------------------


def test_lookback_size_defaults_preserve_legacy_fill():
	# With no constraints set, the function takes as many lookback sales as the test
	# set needs (capped by availability). This preserves the legacy behavior of
	# "fill the test set from lookback first."
	assert compute_lookback_test_size(test_count=53, lb_size=75, nlb_size=188) == 53
	assert compute_lookback_test_size(test_count=200, lb_size=75, nlb_size=188) == 75


def test_lookback_size_cap_limits_overrepresentation():
	# 2x cap: lookback share should not exceed 2x the non-lookback share.
	# With test=53, LB=75, NLB=188: cap = 2*53*75 / (188 + 2*75) = 23.5 -> 23
	n = compute_lookback_test_size(53, 75, 188, cap_ratio=2.0)
	assert n == 23

	# Verify the cap actually keeps overrepresentation at ≤ 2x:
	non_lb_test = 53 - n
	lb_share = n / 75
	nlb_share = non_lb_test / 188
	assert lb_share <= 2.0 * nlb_share + 1e-9


def test_lookback_size_petersburg_15_2x():
	# Petersburg shape (53, 75, 188) with floor=15, cap=2.0 → cap binds at 23.
	# Floor (15) is well below cap (23), so cap is what determines the result.
	n = compute_lookback_test_size(53, 75, 188, floor=15, cap_ratio=2.0)
	assert n == 23


def test_lookback_size_floor_wins_when_cap_would_go_too_low():
	# 20 lookback, 200 non-lb, test=44, floor=15, cap=2.0
	# cap_l = 2*44*20 / (200+40) = 7.3 -> 7  (would violate the 15 floor)
	# floor wins → returns 15 even though it violates 2x slightly. The floor is a
	# hard requirement for a usable ratio-study sample size.
	n = compute_lookback_test_size(44, 20, 200, floor=15, cap_ratio=2.0)
	assert n == 15


def test_lookback_size_thin_lookback_returns_all_available():
	# 10 lookback total. Even with floor=15, we can only return 10.
	n = compute_lookback_test_size(42, 10, 200, floor=15, cap_ratio=2.0)
	assert n == 10


def test_lookback_size_abundant_lookback_takes_more_than_typical_floor():
	# Big lookback (500), small non-lb (100), floor=15, cap=2.0.
	# cap_l = 2 * 120 * 500 / (100 + 1000) = 109.
	# We take as many as cap allows — 109 — well above any reasonable floor.
	# Floor is a *minimum*, not a ceiling; abundance is fine.
	n = compute_lookback_test_size(120, 500, 100, floor=15, cap_ratio=2.0)
	assert n == 109


def test_lookback_size_no_non_lookback_disables_cap():
	# When there are no non-lookback sales, the cap can't be meaningfully computed
	# (no other group to overrepresent against). The cap is disabled and the function
	# fills the test set from lookback up to the available count. This is the case
	# the legacy synthetic test_split_keys exercises.
	assert compute_lookback_test_size(180, 900, 0, floor=15, cap_ratio=2.0) == 180
	assert compute_lookback_test_size(20, 100, 0, floor=15, cap_ratio=2.0) == 20


def test_lookback_size_edge_cases():
	assert compute_lookback_test_size(0, 75, 188) == 0
	assert compute_lookback_test_size(53, 0, 188) == 0


def test_resolve_strat_fields_improved_defaults_prefer_effective_age():
	df = pd.DataFrame({
		"bldg_age_years": [10, 20, 30],
		"bldg_effective_age_years": [5, 15, 25],
		"bldg_area_finished_sqft": [1000, 2000, 3000],
		"sale_year": [2023, 2024, 2025],
	})
	settings = {"locality": {"units": "imperial"}}
	fields = _resolve_strat_fields_improved(df, settings, user_override=None)
	# Effective age preferred when present; sqft chosen for imperial units;
	# sale_year always appended.
	assert "bldg_effective_age_years" in fields
	assert "bldg_age_years" not in fields
	assert "bldg_area_finished_sqft" in fields
	assert "sale_year" in fields


def test_resolve_strat_fields_improved_falls_back_to_actual_age():
	df = pd.DataFrame({
		"bldg_age_years": [10, 20, 30],
		"bldg_area_finished_sqft": [1000, 2000, 3000],
		"sale_year": [2023, 2024, 2025],
	})
	settings = {"locality": {"units": "imperial"}}
	fields = _resolve_strat_fields_improved(df, settings, user_override=None)
	# No effective_age → fall back to actual age
	assert "bldg_age_years" in fields
	assert "bldg_effective_age_years" not in fields


def test_resolve_strat_fields_improved_user_override_respected():
	df = pd.DataFrame({
		"neighborhood": ["A", "B", "C"],
		"bldg_quality_num": [3, 4, 5],
		"sale_year": [2023, 2024, 2025],
	})
	settings = {}
	fields = _resolve_strat_fields_improved(df, settings,
		user_override=["neighborhood", "bldg_quality_num"])
	assert fields == ["neighborhood", "bldg_quality_num", "sale_year"]
	# Defaults (age, area) are NOT included when user overrides.
	assert "bldg_age_years" not in fields


def test_resolve_strat_fields_improved_drops_missing_columns():
	df = pd.DataFrame({"sale_year": [2023, 2024]})
	settings = {}
	fields = _resolve_strat_fields_improved(df, settings, user_override=None)
	# Defaults reference age and area which don't exist in this df → dropped.
	# sale_year survives.
	assert fields == ["sale_year"]


def test_build_strat_label_numeric_is_quantile_binned():
	df = pd.DataFrame({"x": list(range(100))})
	label = _build_strat_label(df, ["x"], n_bins=4)
	# 100 distinct values quantile-binned to 4 strata → 4 unique label values.
	assert label.nunique() == 4


def test_build_strat_label_categorical_preserved_as_is():
	df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
	label = _build_strat_label(df, ["cat"], n_bins=4)
	assert set(label.unique()) == {"A", "B", "C"}


def test_build_strat_label_combined_uses_cross_product():
	df = pd.DataFrame({
		"year": [2023, 2024, 2023, 2024],
		"cat": ["A", "A", "B", "B"],
	})
	label = _build_strat_label(df, ["year", "cat"], n_bins=4)
	# 2 years × 2 categories, all distinct combinations → 4 unique labels.
	assert label.nunique() == 4


def test_build_strat_label_empty_fields_returns_none():
	df = pd.DataFrame({"x": [1, 2, 3]})
	assert _build_strat_label(df, [], n_bins=4) is None
	assert _build_strat_label(df, ["nonexistent_field"], n_bins=4) is None


def test_stratified_test_sample_preserves_class_proportion():
	rng = np.random.RandomState(0)
	df = pd.DataFrame({
		"key_sale": [f"k{i}" for i in range(200)],
		"sale_year": [2023] * 100 + [2024] * 100,
	})
	train, test = _stratified_test_sample(df, n_test=40, random_seed=1337,
		strat_fields=["sale_year"])
	# Each year's representation in test should be proportional (20 each, ±1 from
	# sklearn's rounding).
	test_counts = test["sale_year"].value_counts()
	assert abs(test_counts.get(2023, 0) - 20) <= 1
	assert abs(test_counts.get(2024, 0) - 20) <= 1
	# And train+test = total
	assert len(train) + len(test) == len(df)
	assert len(test) == 40


def test_stratified_test_sample_falls_back_when_strata_too_thin():
	# Each combination of (year, cat) has only 1 sample — sklearn would raise.
	# Helper should drop fields and fall back gracefully.
	df = pd.DataFrame({
		"key_sale": [f"k{i}" for i in range(4)],
		"sale_year": [2023, 2024, 2025, 2026],
		"cat": ["A", "B", "C", "D"],
	})
	train, test = _stratified_test_sample(df, n_test=2, random_seed=1337,
		strat_fields=["sale_year", "cat"])
	assert len(test) == 2
	assert len(train) == 2


def test_three_tier_split_post_val_goes_to_test():
	# A jurisdiction where all sales are post-valuation: they should ALL go to test,
	# and the training set is empty (no leakage).
	df = pd.DataFrame({
		"key_sale": [f"k{i}" for i in range(10)],
		"sale_age_days": [-30] * 10,  # all post-val
		"sale_year": [2026] * 10,
		"vacant_sale": [False] * 10,
		"bldg_area_finished_sqft": [1500.0] * 10,
	})
	test, train = _three_tier_split(df, test_count=5, look_back_days=365,
		floor=15, cap_ratio=2.0,
		strat_fields=["sale_year"], random_seed=1337)
	# We never train on post-val sales; all 10 are reserved.
	assert len(train) == 0
	# Test got all 10 (even though we only asked for 5) — post-val takes priority
	# and is not down-sampled.
	assert len(test) == 10


def test_three_tier_split_cap_balances_train_share():
	# Petersburg-like: 117 pre-lookback (2023-24), 75 lookback (2025), test_count=53.
	# With cap=2.0, lookback contribution to test is limited; some 2025 sales go
	# to training where the model needs them.
	rng = np.random.RandomState(0)
	rows = []
	val_date = pd.Timestamp("2026-01-01")
	for yr, n in [(2023, 117), (2024, 71), (2025, 75)]:
		for i in range(n):
			sd = pd.Timestamp(f"{yr}-06-15")
			rows.append({
				"key_sale": f"{yr}-{i:04d}",
				"sale_age_days": (val_date - sd).days,
				"sale_year": yr,
				"vacant_sale": False,
				"bldg_area_finished_sqft": 1500.0,
			})
	df = pd.DataFrame(rows)
	test, train = _three_tier_split(df, test_count=53, look_back_days=365,
		floor=15, cap_ratio=2.0,
		strat_fields=["sale_year"], random_seed=1337)

	test_2025 = (test["sale_year"] == 2025).sum()
	train_2025 = (train["sale_year"] == 2025).sum()
	# At the 2x cap: 23 lookback → test, 52 lookback → train. The exact split is
	# determined by the cap math; we assert the cap is binding (23) and the floor
	# (15) is not binding for this shape.
	assert test_2025 == 23
	assert train_2025 == 52
	# 2025 is no longer 100% of test — older years contribute the remaining 30.
	assert len(test) == 53
	assert (test["sale_year"] < 2025).sum() == 30


def test_three_tier_split_legacy_default_fills_test_from_lookback():
	# With both constraints disabled (floor=None, cap_ratio=None), three-tier split
	# reproduces the old "lookback fills test first" behavior. For sf_suburban this
	# puts all 53 test slots into 2025.
	val_date = pd.Timestamp("2026-01-01")
	rows = []
	for yr, n in [(2023, 117), (2024, 71), (2025, 75)]:
		for i in range(n):
			sd = pd.Timestamp(f"{yr}-06-15")
			rows.append({
				"key_sale": f"{yr}-{i:04d}",
				"sale_age_days": (val_date - sd).days,
				"sale_year": yr,
				"vacant_sale": False,
				"bldg_area_finished_sqft": 1500.0,
			})
	df = pd.DataFrame(rows)
	test, train = _three_tier_split(df, test_count=53, look_back_days=365,
		floor=None, cap_ratio=None,
		strat_fields=["sale_year"], random_seed=1337)
	test_2025 = (test["sale_year"] == 2025).sum()
	# Legacy fill: all 53 test slots come from lookback.
	assert test_2025 == 53
	assert len(test) == 53


def test_canonical_split_petersburg_30_15_2x_end_to_end():
	# Builds a Petersburg-like sf_suburban dataset and verifies that the new
	# settings produce a temporally balanced split.
	val_date = pd.Timestamp("2026-01-01")
	rows = []
	for yr, n in [(2023, 117), (2024, 71), (2025, 75)]:
		for i in range(n):
			sd = pd.Timestamp(f"{yr}-06-15")
			rows.append({
				"key": f"{yr}-{i:04d}",
				"key_sale": f"{yr}-{i:04d}---{sd.date()}",
				"model_group": "sf",
				"valid_sale": True,
				"vacant_sale": False,
				"is_vacant": False,
				"sale_price": 200000.0,
				"sale_date": sd,
				"sale_year": yr,
				"sale_month": 6,
				"sale_day": 15,
				"sale_age_days": (val_date - sd).days,
				"bldg_area_finished_sqft": 1500.0,
				"land_area_sqft": 5000.0,
			})
	df = pd.DataFrame(rows)
	settings = {
		"modeling": {
			"metadata": {"valuation_date": "2026-01-01"},
			"instructions": {
				"test_lookback_floor": 15,
				"test_lookback_cap_ratio": 2.0,
			},
		},
		"analysis": {"ratio_study": {"look_back_years": 1}},
	}
	df_test, df_train = _perform_canonical_split("sf", df, settings,
		test_train_fraction=0.8)
	assert len(df_test) == 53
	assert len(df_train) == 210

	test_2025 = (df_test["sale_year"] == 2025).sum()
	train_2025 = (df_train["sale_year"] == 2025).sum()
	# The cap is the binding constraint; 23 lookback in test, 52 lookback in train.
	assert test_2025 == 23
	assert train_2025 == 52
	# 2025 share of training is now meaningful (~25%), not the ~10% from the
	# legacy fill.
	assert train_2025 / len(df_train) > 0.20


def test_canonical_split_default_applies_15_2x_rule():
	# The default constants (floor=15, cap_ratio=2.0) are baked into
	# _perform_canonical_split so jurisdictions get a sensible holdout shape without
	# having to opt in. With Petersburg-like proportions the cap is binding at 23.
	val_date = pd.Timestamp("2026-01-01")
	rows = []
	for yr, n in [(2023, 117), (2024, 71), (2025, 75)]:
		for i in range(n):
			sd = pd.Timestamp(f"{yr}-06-15")
			rows.append({
				"key": f"{yr}-{i:04d}",
				"key_sale": f"{yr}-{i:04d}---{sd.date()}",
				"model_group": "sf",
				"valid_sale": True,
				"vacant_sale": False,
				"is_vacant": False,
				"sale_price": 200000.0,
				"sale_date": sd,
				"sale_year": yr,
				"sale_month": 6,
				"sale_day": 15,
				"sale_age_days": (val_date - sd).days,
				"bldg_area_finished_sqft": 1500.0,
				"land_area_sqft": 5000.0,
			})
	df = pd.DataFrame(rows)
	# NO modeling.instructions block — defaults should apply.
	settings = {
		"modeling": {"metadata": {"valuation_date": "2026-01-01"}},
		"analysis": {"ratio_study": {"look_back_years": 1}},
	}
	df_test, df_train = _perform_canonical_split("sf", df, settings,
		test_train_fraction=0.8)
	# Default rule kicks in: 23 lookback (2025) test, 52 lookback train.
	test_2025 = (df_test["sale_year"] == 2025).sum()
	train_2025 = (df_train["sale_year"] == 2025).sum()
	assert test_2025 == 23
	assert train_2025 == 52


def test_canonical_split_explicit_none_disables_default_rule():
	# Setting all three knobs to None in settings explicitly opts out of the default
	# rule and restores the legacy "fill test from lookback first" behavior.
	val_date = pd.Timestamp("2026-01-01")
	rows = []
	for yr, n in [(2023, 117), (2024, 71), (2025, 75)]:
		for i in range(n):
			sd = pd.Timestamp(f"{yr}-06-15")
			rows.append({
				"key": f"{yr}-{i:04d}",
				"key_sale": f"{yr}-{i:04d}---{sd.date()}",
				"model_group": "sf",
				"valid_sale": True,
				"vacant_sale": False,
				"is_vacant": False,
				"sale_price": 200000.0,
				"sale_date": sd,
				"sale_year": yr,
				"sale_month": 6,
				"sale_day": 15,
				"sale_age_days": (val_date - sd).days,
				"bldg_area_finished_sqft": 1500.0,
				"land_area_sqft": 5000.0,
			})
	df = pd.DataFrame(rows)
	settings = {
		"modeling": {
			"metadata": {"valuation_date": "2026-01-01"},
			"instructions": {
				"test_lookback_floor": None,
				"test_lookback_cap_ratio": None,
			},
		},
		"analysis": {"ratio_study": {"look_back_years": 1}},
	}
	df_test, df_train = _perform_canonical_split("sf", df, settings,
		test_train_fraction=0.8)
	test_2025 = (df_test["sale_year"] == 2025).sum()
	train_2025 = (df_train["sale_year"] == 2025).sum()
	# Legacy fill: all 53 test slots come from lookback; only 22 of 75 lookback
	# sales remain for training.
	assert test_2025 == 53
	assert train_2025 == 22