import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from openavmkit.data import _perform_canonical_split
from openavmkit.modeling import DataSplit
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

