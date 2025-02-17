import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from openavmkit.data import _perform_canonical_split, handle_duplicated_rows, perform_ref_tables, merge_dict_of_dfs, \
	_enrich_year_built, enrich_time
from openavmkit.modeling import DataSplit
from openavmkit.utilities.assertions import dfs_are_equal
from openavmkit.utilities.data import div_z_safe


def test_div_z_safe():
	print("")
	df = pd.DataFrame({
		"numerator": [1, 2, 3, 4, 5],
		"denominator": [0, 1, 2, 0, 4]
	})
	result = div_z_safe(df, "numerator", "denominator")
	assert result.isna().sum() == 2
	assert result.astype(str).eq(["nan","2.0","1.5","nan","1.25"]).all()


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
	df.loc[df["valid_sale"].eq(True), "sale_date"] = "2025-01-01"
	df["sale_date"] = pd.to_datetime(df["sale_date"])
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

	test_keys = df_test["key"].tolist()
	train_keys = df_train["key"].tolist()

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
		df=df,
		model_group="residential_sf",
		settings={},
		ind_var="sale_price",
		ind_var_test="sale_price",
		dep_vars=["bldg_area_finished_sqft", "land_area_sqft"],
		categorical_vars=[],
		interactions={},
		test_keys=test_keys,
		train_keys=train_keys,
		vacant_only=False,
		hedonic=False
	)
	ds.split()

	ds_v = DataSplit(
		df=df,
		model_group="residential_sf",
		settings={},
		ind_var="sale_price",
		ind_var_test="sale_price",
		dep_vars=["bldg_area_finished_sqft", "land_area_sqft"],
		categorical_vars=[],
		interactions={},
		test_keys=test_keys,
		train_keys=train_keys,
		vacant_only=True,
		hedonic=False
	)
	ds_v.split()

	ds_h = DataSplit(
		df=df,
		model_group="residential_sf",
		settings={},
		ind_var="sale_price",
		ind_var_test="sale_price",
		dep_vars=["bldg_area_finished_sqft", "land_area_sqft"],
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

	# remove the first row from ds_v.df_test:
	ds_v.df_test = ds_v.df_test.iloc[1:]

	# remove the last row from ds.df_test:
	ds.df_test = ds.df_test.iloc[:-1]

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
	df_results = handle_duplicated_rows(df, dupes)
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
	df_results = handle_duplicated_rows(df, dupes)

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
	df_results = perform_ref_tables(df, ref_table, dataframes)

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

	df_results = perform_ref_tables(df, ref_tables, dataframes)

	assert dfs_are_equal(df_expected, df_results, primary_key="key")

	# Test the case where the keys are identical
	dataframes["ref_zoning"] = dataframes["ref_zoning"].rename(columns={"zoning_id": "zoning"})
	ref_table["key_ref_table"] = "zoning"

	df_results = perform_ref_tables(df, ref_table, dataframes)

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

	merge_dict_of_dfs(
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
		"sale_quarter": [None, None, None, "1", None, None, None, None, None, "4", None],
		"sale_year_month": ["NaT", "NaT", "NaT", "2021-01", "NaT", "NaT", "NaT", "NaT", "NaT", "2022-11", "NaT"],
		"sale_year_quarter": ["NaT", "NaT", "NaT", "2021Q1", "NaT", "NaT", "NaT", "NaT", "NaT", "2022Q4", "NaT"],
		"bldg_age_years": [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25],
	}

	expected_sales = {
		"key": ["3", "9"],
		"sale_date": ["2021-01-01", "2022-11-15"],
		"valid_sale": [True, True],
		"sale_price": [100000.0, 200000.0],
		"bldg_year_built": [1993, 1999],
		"sale_year": ["2021", "2022"],
		"sale_month": ["1", "11"],
		"sale_quarter": ["1", "4"],
		"sale_year_month": ["2021-01", "2022-11"],
		"sale_year_quarter": ["2021Q1", "2022Q4"],
		"bldg_age_years": [28.0, 23.0]
	}

	time_formats = {"sale_date":"%Y-%m-%d"}

	df_univ = enrich_time(df_univ, time_formats)
	df_sales = enrich_time(df_sales, time_formats)

	df_univ = _enrich_year_built(df_univ, "bldg_year_built", "bldg_age_years", val_date, False)
	df_sales = _enrich_year_built(df_sales, "bldg_year_built", "bldg_age_years", val_date, True)

	df_univ_expected = pd.DataFrame(data=expected_univ)
	df_sales_expected = pd.DataFrame(data=expected_sales)

	for thing in ["sale_year", "sale_month", "sale_quarter"]:
		df_univ[thing] = df_univ[thing].astype("Int64").astype("string")
		df_sales[thing] = df_sales[thing].astype("Int64").astype("string")
		df_univ_expected[thing] = df_univ_expected[thing].astype("Int64").astype("string")
		df_sales_expected[thing] = df_sales_expected[thing].astype("Int64").astype("string")

	for thing in ["sale_date", "sale_year_month", "sale_year_quarter"]:
		df_univ[thing] = df_univ[thing].astype("string")
		df_sales[thing] = df_sales[thing].astype("string")
		df_univ_expected[thing] = df_univ_expected[thing].astype("string")
		df_sales_expected[thing] = df_sales_expected[thing].astype("string")

	assert dfs_are_equal(df_univ, df_univ_expected, primary_key="key")
	assert dfs_are_equal(df_sales, df_sales_expected, primary_key="key")