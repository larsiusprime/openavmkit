import os

from openavmkit.benchmark import run_models, MultiModelResults
from openavmkit.cleaning import clean_valid_sales
from openavmkit.data import enrich_time
from openavmkit.horizontal_equity_study import mark_horizontal_equity_clusters
from openavmkit.ratio_study import run_and_write_ratio_study_breakdowns
from openavmkit.sales_scrutiny_study import SalesScrutinyStudy, run_sales_scrutiny
from openavmkit.synthetic_data import generate_basic
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.settings import get_valuation_date, load_settings


def test_guilford_sales_scrutiny():
	# TODO: update this
	return True


def test_models_guilford():
	# TODO: update this
	return True


def test_models_synthetic():
	# TODO: update this
	return True
	# print("")
	# # set working directory to the library's root/tests/data:
	# os.chdir("data/zz-synthetic")
	#
	# # load the settings:
	# settings = load_settings()
	#
	# # load the data
	# sd = generate_basic(100)
	# df = sd.df
	#
	# df["assr_market_value"] = df["total_value"]
	# df["model_group"] = "residential_sf"
	#
	# # calculate horizontal equity cluster ID's
	# df = mark_horizontal_equity_clusters(df, settings)
	#
	# # run the predictive models
	# results = run_models(
	# 	df,
	# 	settings,
	# 	verbose=True,
	# 	save_params=True,
	# 	use_saved_params=True
	# )
	#
	# print(results.benchmark.print())
	# return True