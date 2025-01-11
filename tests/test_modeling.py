import os

from openavmkit.benchmark import run_models, get_variable_recommendations, MultiModelResults
from openavmkit.cleaning import fill_unknown_values_per_model_group
from openavmkit.data import load_data, enrich_time
from openavmkit.horizontal_equity_study import make_clusters, mark_horizontal_equity_clusters
from openavmkit.modeling import SingleModelResults
from openavmkit.ratio_study import run_and_write_ratio_study_breakdowns
from openavmkit.sales_scrutiny_study import SalesScrutinyStudy, clean_sales
from openavmkit.synthetic_data import generate_basic
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.settings import get_valuation_date, load_settings, get_fields_categorical


def test_guilford_sales_scrutiny():
	print("")
	# set working directory to the library's root/tests/data:
	os.chdir("data/nc-guilford")

	# load the settings:
	settings = load_settings()

	# load the data
	df = load_data(settings)

	# clean the data
	df = fill_unknown_values_per_model_group(df, settings)

	model_group = "residential_sf"

	# select a subset of the data
	df = df[df["model_group"].eq(model_group)].copy().reset_index(drop=True)

	if "he_id" not in df:
		df = mark_horizontal_equity_clusters(df, settings)

	# load metadata
	val_date = get_valuation_date(settings)
	val_year = val_date.year
	metadata = settings.get("modeling", {}).get("metadata", {})
	use_sales_from = metadata.get("use_sales_from", val_year - 5)

	# mark which sales are to be used
	df.loc[df["sale_year"].lt(use_sales_from), "valid_sale"] = 0

	# scrub sales info from invalid sales
	idx_invalid = df["valid_sale"].eq(0)
	df.loc[idx_invalid, "sale_date"] = None
	df.loc[idx_invalid, "sale_price"] = None

	# enrich time:
	df = enrich_time(df)
	df = enrich_time_adjustment(df, settings, verbose=True)

	# run sales validity:
	ss = SalesScrutinyStudy(df, settings, model_group=model_group)
	ss.write(f"out")



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

	model_group = "residential_sf"

	# select a subset of the data
	df = df[df["model_group"].eq(model_group)].copy().reset_index(drop=True)

	if "he_id" not in df:
		df = mark_horizontal_equity_clusters(df, settings)

	# load metadata
	val_date = get_valuation_date(settings)
	val_year = val_date.year
	metadata = settings.get("modeling", {}).get("metadata", {})
	use_sales_from = metadata.get("use_sales_from", val_year - 5)

	# mark which sales are to be used
	df.loc[df["sale_year"].lt(use_sales_from), "valid_sale"] = 0

	# scrub sales info from invalid sales
	idx_invalid = df["valid_sale"].eq(0)
	df.loc[idx_invalid, "sale_date"] = None
	df.loc[idx_invalid, "sale_price"] = None

	print(f"Using {len(df[df['valid_sale'].eq(1)])} sales...")

	# enrich time:
	df = enrich_time(df)
	df = enrich_time_adjustment(df, settings, verbose=True)

	df = clean_sales(df, settings, model_group, verbose=True)

	# run the predictive models
	results : MultiModelResults = run_models(
		df,
		settings,
		verbose=True,
		save_params=True,
		use_saved_params=True
	)

	# run ratio study reports
	run_and_write_ratio_study_breakdowns(
		settings,
		results.model_results["ensemble"].df_sales,
		model_group,
		"out",
		iterations=100
	)

	df_univ = results.model_results["ensemble"].df_universe
	df_sales = results.model_results["ensemble"].df_sales
	df_univ[["key", "model_group", "sale_date", "sale_price", "sale_price_time_adj", "assr_market_value", "prediction"]].to_csv("out/univ.csv")
	df_sales[["key", "model_group", "sale_date", "sale_price", "sale_price_time_adj", "assr_market_value", "prediction"]].to_csv("out/sales.csv")

	print(results.benchmark.print())


def test_models_synthetic():
	print("")
	# set working directory to the library's root/tests/data:
	os.chdir("data/zz-synthetic")

	# load the settings:
	settings = load_settings()

	# load the data
	sd = generate_basic(100)
	df = sd.df

	df["assr_market_value"] = df["total_value"]

	# calculate horizontal equity cluster ID's
	df["he_id"] = mark_horizontal_equity_clusters(df, settings)

	# run the predictive models
	results = run_models(
		df,
		settings,
		verbose=True,
		save_params=True,
		use_saved_params=True
	)

	print(results.benchmark.print())