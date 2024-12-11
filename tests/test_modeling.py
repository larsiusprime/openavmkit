import logging

import pandas as pd
from IPython.core.display_functions import display

from openavmkit.benchmark import run_benchmark, format_benchmark_df
from openavmkit.cleaning import fill_unknown_values_per_model_group
from openavmkit.horizontal_equity_study import cluster_by_location_and_big_five
from openavmkit.synthetic_data import generate_basic


def test_models_guilford():
	df = pd.read_parquet("data/nc-guilford/universe_clean.parquet")
	pd.set_option('display.max_rows',None)
	pd.set_option('display.max_rows',15)

	df = fill_unknown_values_per_model_group(df, {})
	df = df[df["model_group"].eq("residential_sf")].copy().reset_index(drop=True)

	ind_var = "sale_price"
	dep_vars = [
		"bldg_area_finished_sqft",
		"land_area_sqft",
		"bldg_quality_num",
		"bldg_condition_num",
		"bldg_age_years",
		"dist_to_cbd"
	]

	models = [
		"garbage",
		# "garbage_normal*",
		# "mean*",
		# "median*",
		# "naive_sqft*",
		"mra"
		# "gwr",
		# "lightgbm",
		# "catboost",
		# "xgboost"
	]
	df_test, df_full = run_benchmark(df, ind_var, dep_vars, models, verbose=True, save_params=True, use_saved_params=True)

	print("Test set:")
	print(format_benchmark_df(df_test))
	print("")
	print("Full set:")
	print(format_benchmark_df(df_full))

def test_models_synthetic():
	print("")
	df = generate_basic(100)
	ind_var = "sale_price"
	dep_vars = [
		"bldg_area_finished_sqft",
		"land_area_sqft",
		"bldg_quality_num",
		"bldg_condition_num",
		"bldg_age_years",
		"distance_from_cbd"
	]

	# Assign equity cluster ID's
	df["he_id"] = cluster_by_location_and_big_five(df, "neighborhood", [])
	models = [
		# #"garbage",
		# "garbage*",
		# #"garbage_normal",
		# "garbage_normal*",
		# #"mean",
		# "mean*",
		# #"median",
		# "median*",
		# #"naive_sqft",
		# "naive_sqft*",
		"mra",
		# "gwr",
		# "lightgbm",
		# "catboost",
		# "xgboost"
	]
	df_test, df_full = run_benchmark(df, ind_var, dep_vars, models, verbose=True, save_params=True, use_saved_params=True)

	print("Test set:")
	print(format_benchmark_df(df_test))
	print("")
	print("Full set:")
	print(format_benchmark_df(df_full))