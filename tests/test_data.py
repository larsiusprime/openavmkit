import numpy as np
import pandas as pd
from IPython.display import display

from openavmkit.data import _perform_canonical_split, _handle_duplicated_rows, _perform_ref_tables, _merge_dict_of_dfs, \
	_do_enrich_year_built, enrich_time, SalesUniversePair, get_hydrated_sales_from_sup, _enrich_permits
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
		hedonic=False
	)
	ds.split()

	ds_v = DataSplit(
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
		hedonic=False
	)
	ds_v.split()

	ds_h = DataSplit(
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
		hedonic=True
	)
	ds_h.split()

	# Assert that all three flavors of splits generated the expected lengths
	assert(ds.df_train.shape[0] == expected_train)
	assert(ds.df_test.shape[0] == expected_test)
	assert(ds_v.df_train.shape[0] == expected_train_vacant)
	assert(ds_v.df_test.shape[0] == expected_test_vacant)
	assert(ds_h.df_train.shape[0] == expected_train_vacant)
	assert(ds_h.df_test.shape[0] == expected_test_vacant)

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

	# ds_v.test is equivalent to ds_h.test (they both test against vacant sales)
	assert a_equals_b(ds_v.df_test, ds_h.df_test)

	# ds_v.test is a strict subset of ds.test (vacant test sales only has sales also found in the vacant+improved test sales)
	assert a_is_subset_of_b(ds_v.df_test, ds.df_test)

	# ds.test is a strict superset of ds_v.test (vacant+improved test sales includes all sales found in vacant test sales)
	assert a_is_superset_of_b(ds.df_test, ds_v.df_test)

	# ds_h.test is a strict subset of ds.test (hedonic test sales only has sales also found in the vacant+improved test sales)
	assert a_is_subset_of_b(ds_h.df_test, ds.df_test)

	# ds.test is a strict superset of ds_h.test (vacant+improved test sales includes all sales found in hedonic test sales)
	assert a_is_superset_of_b(ds.df_test, ds_h.df_test)

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
	assert a_equals_b(ds_v.df_test, ds_h.df_test) == False
	assert a_is_subset_of_b(ds_v.df_test, ds.df_test) == False
	assert a_is_superset_of_b(ds.df_test, ds_v.df_test) == False
	assert a_is_subset_of_b(ds_h.df_test, ds.df_test) == False
	assert a_is_superset_of_b(ds.df_test, ds_h.df_test) == False


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

	df_univ = enrich_time(df_univ, time_formats, {})
	df_sales = enrich_time(df_sales, time_formats, {})

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
	permits = {
		"key": ["0", "1", "2", "3", "3", "3"],
		"is_teardown": [True, True, True, True, True, True],
		"date": [
			              # Demo dates for keys 1, 2, and 3
			"2020-07-01", # too early: one month AFTER the sale
			"2020-01-01", # just right: five months BEFORE the sale
			"2021-07-01", # too late: thirteen months AFTER the sale

			              # 3 demo dates, all for key 3 -- should de-duplicate and pick the middle one
			"2020-07-01", # too early
			"2020-01-01", # just right
			"2021-07-01"  # too late
		]
	}
	expected = {
		"key": ["0", "1", "2", "3"],
		"valid_sale": [True, True, True, True],
		"vacant_sale": [False, True, False, True],
		"sale_price": [1, 1, 1, 1],
		"sale_date": ["2020-06-01", "2020-06-01", "2020-06-01", "2020-06-01"],
		"key_sale": ["0---2020-06-01", "1---2020-06-01", "2---2020-06-01", "3---2020-06-01"],
		"is_teardown_sale": [False, True, False, True],
		"demo_date": ["2020-07-01", "2020-01-01", "2021-07-01", "2020-01-01"],
		"days_to_demo":[nan, 152.0, nan, 152.0]
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