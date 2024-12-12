import logging

import pandas as pd
from IPython.core.display_functions import display

from openavmkit.benchmark import run_benchmark, format_benchmark_df
from openavmkit.cleaning import fill_unknown_values_per_model_group
from openavmkit.horizontal_equity_study import cluster_by_location_and_big_five
from openavmkit.synthetic_data import generate_basic
from openavmkit.utilities.settings import get_valuation_date


def test_models_guilford():
	df = pd.read_parquet("data/nc-guilford/universe_clean.parquet")
	pd.set_option('display.max_rows',None)
	pd.set_option('display.max_rows',15)

	df = fill_unknown_values_per_model_group(df, {})
	df = df[df["model_group"].eq("residential_sf")].copy().reset_index(drop=True)

	ind_var = "sale_price"
	dep_vars = {
		"default": [
			"bldg_area_finished_sqft",
			"land_area_sqft",
			"bldg_quality_num",
			"bldg_condition_num",
			"bldg_age_years",
			"dist_to_cbd",
			"latitude",
			"longitude",
			"rectangularity_num",
			"slope_mean",
			"elevation_mean",
			"dist_to_universities",
			"dist_to_colleges",
			"dist_to_greenspace",
			"dist_to_airport"
		],
		"gwr": [
			"bldg_area_finished_sqft",
			"bldg_age_years",
			"land_area_sqft",
			"rectangularity_num",
			"dist_to_cbd",
			"elevation_mean",
		]
	}

	#df["he_id"] = cluster_by_location_and_big_five(df, "neighborhood", [], verbose=True)
	models = [
		"mra",
		"gwr",
		"lightgbm",
		"catboost",
		"xgboost",
		"garbage",
		"garbage_normal",
		"mean",
		"median",
		"naive_sqft"
	]

	# select only recent sales
	val_date = get_valuation_date({})
	val_year = val_date.year
	sales_back_to_year = val_year - 1

	df.loc[df["sale_year"].lt(sales_back_to_year), "valid_sale"] = 0

	print(f"Using {len(df[df['valid_sale'].eq(1)])} sales...")

	df_test, df_full = run_benchmark(df, ind_var, dep_vars, models, outdir="nc-guilford", verbose=True, save_params=True, use_saved_params=True)

	print("Test set:")
	print(format_benchmark_df(df_test))
	print("")
	print("Full set:")
	print(format_benchmark_df(df_full))

def test_models_synthetic():
	print("")
	df = generate_basic(100)
	ind_var = "sale_price"
	dep_vars = {
		"default":[
			"bldg_area_finished_sqft",
			"land_area_sqft",
			"bldg_quality_num",
			"bldg_condition_num",
			"bldg_age_years",
			"distance_from_cbd"
		]
	}

	# Assign equity cluster ID's
	df["he_id"] = cluster_by_location_and_big_five(df, "neighborhood", [], verbose=True)
	models = [
		#"garbage",
		"garbage*",
		#"garbage_normal",
		"garbage_normal*",
		#"mean",
		"mean*",
		#"median",
		"median*",
		#"naive_sqft",
		"naive_sqft*",
		"mra",
		"gwr",
		"lightgbm",
		"catboost",
		"xgboost"
	]
	df_test, df_full = run_benchmark(df, ind_var, dep_vars, models, outdir="synthetic-basic", verbose=True, save_params=True, use_saved_params=True)

	print("Test set:")
	print(format_benchmark_df(df_test))
	print("")
	print("Full set:")
	print(format_benchmark_df(df_full))