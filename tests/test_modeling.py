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

	categorical_fields = [
		"census_tract",
		"census_block_group",
		"city",
		"zoning",
		"zoning_class",
		"zoning_desc",
		"zoning_class_desc",
		"school_district",
		"rectangularity_txt",
		"is_triangular",
		"is_complex_geometry",
		"is_elongated",
		"is_irregular",
		"land_class",
		"bldg_class",
		"bldg_plumbing",
		"bldg_count",
		"neighborhood",
		"bldg_desc",
		"bldg_type",
		"bldg_style",
		"bldg_exterior",
		"bldg_heating",
		"bldg_ac",
		"bldg_fixtures",
		"bldg_foundation",
		"bldg_quality_txt",
		"bldg_condition_txt",
		"vacant_sale",
		"new_construction",
		"newish_construction",
		"osm_street_name",
		"osm_street_type",
		"osm_corner_lot_type",
		"osm_is_corner_lot",
		"osm_corner_lot_street",
		"osm_is_cul_de_sac",
		"osm_waterfront_name",
		"osm_is_waterfront",
		"osm_golf_course_name",
		"osm_on_golf_course",
		"osm_park_name",
		"osm_on_park",
		"osm_playground_name",
		"osm_on_playground",
		"osm_swimming_pool_name",
		"osm_on_swimming_pool",
		"zoning_category",
		"is_vacant",
		"potential_vacant_sale",
		"valid_for_ratio_study",
		"valid_for_land_ratio_study",
		"valid_for_modeling",
		"warning_vacant_discrepancy",
		"warning_vacant_positive_year_built",
		"warning_vacant_positive_impr_numeric",
		"warning_vacant_has_impr_categorical",
		"model_group",
		"is_below_buildable_size",
		"he_id"
	]

	df = fill_unknown_values_per_model_group(df, settings={}, categorical_fields=categorical_fields)
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
			"dist_to_airport",
			"sale_age_days",
			#"neighborhood",
			#"census_tract",
			"bldg_style"
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
		#"gwr",
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

	df_test, df_full = run_benchmark(
		df,
		ind_var,
		dep_vars,
		models,
		categorical_fields,
		outdir="nc-guilford",
		verbose=True,
		save_params=True,
		use_saved_params=True
	)

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
			"bldg_type",
			"distance_from_cbd"
		]
	}

	cat_vars = [
		"bldg_type"
	]

	# Assign equity cluster ID's
	df["he_id"] = cluster_by_location_and_big_five(df, "neighborhood", [], verbose=True)
	models = [
		# "garbage*",
		# "garbage_normal*",
		# "mean*",
		# "median*",
		# "naive_sqft*",
		"mra",
		"gwr",
		"lightgbm",
		"catboost",
		"xgboost"
	]
	df_test, df_full = run_benchmark(df, ind_var, dep_vars, models, cat_vars, outdir="synthetic-basic", verbose=True, save_params=True, use_saved_params=True)

	print("Test set:")
	print(format_benchmark_df(df_test))
	print("")
	print("Full set:")
	print(format_benchmark_df(df_full))