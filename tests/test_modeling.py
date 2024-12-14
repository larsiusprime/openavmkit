import os

import pandas as pd
from IPython.core.display_functions import display

from openavmkit.benchmark import run_benchmark, format_benchmark_df
from openavmkit.cleaning import fill_unknown_values_per_model_group
from openavmkit.data import load_data
from openavmkit.horizontal_equity_study import cluster_by_location_and_big_five
from openavmkit.synthetic_data import generate_basic
from openavmkit.utilities.settings import get_valuation_date, load_settings, get_fields_categorical


def test_models_guilford():
	print("")
	# set working directory to the library's root/tests/data:
	os.chdir("data/nc-guilford")

	# load the settings:
	settings = load_settings()

	# load the data
	df = load_data(settings)

	# clean the data
	df = fill_unknown_values_per_model_group(df, settings)

	# select a subset of the data
	df = df[df["model_group"].eq("residential_sf")].copy().reset_index(drop=True)

	if "he_id" not in df:
		df["he_id"] = cluster_by_location_and_big_five(df, "neighborhood", [], verbose=True)

	# load metadata
	val_date = get_valuation_date(settings)
	val_year = val_date.year
	metadata = settings.get("modeling", {}).get("metadata", {})
	use_sales_from = metadata.get("use_sales_from", val_year - 5)

	# mark which sales are to be used
	df.loc[df["sale_year"].lt(use_sales_from), "valid_sale"] = 0

	print(f"Using {len(df[df['valid_sale'].eq(1)])} sales...")

	# run the predictive models
	df_test, df_full = run_benchmark(
		df,
		settings,
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
	# set working directory to the library's root/tests/data:
	os.chdir("data/zz-synthetic")

	# load the settings:
	settings = load_settings()

	# load the data
	df = generate_basic(100)

	# calculate horizontal equity cluster ID's
	df["he_id"] = cluster_by_location_and_big_five(df, "neighborhood", [], verbose=True)

	# run the predictive models
	df_test, df_full = run_benchmark(
		df,
		settings,
		verbose=True,
		save_params=True,
		use_saved_params=True
	)

	print("Test set:")
	print(format_benchmark_df(df_test))
	print("")
	print("Full set:")
	print(format_benchmark_df(df_full))