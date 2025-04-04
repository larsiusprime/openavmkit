import os
import pickle
import warnings

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster
from statsmodels.nonparametric.kernel_regression import KernelReg
from xgboost import XGBRegressor

from openavmkit.data import get_important_field, get_locations, _read_split_keys, SalesUniversePair, \
	get_hydrated_sales_from_sup, get_report_locations, get_sales
from openavmkit.modeling import run_mra, run_gwr, run_xgboost, run_lightgbm, run_catboost, SingleModelResults, \
	run_garbage, run_average, run_naive_sqft, predict_garbage, \
	run_kernel, run_local_sqft, run_pass_through, predict_average, predict_naive_sqft, predict_local_sqft, \
	predict_pass_through, predict_kernel, predict_gwr, predict_xgboost, predict_catboost, predict_lightgbm, \
	GarbageModel, AverageModel, DataSplit, predict_lars, run_ground_truth, predict_ground_truth
from openavmkit.reports import MarkdownReport, _markdown_to_pdf
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import div_z_safe, dataframe_to_markdown
from openavmkit.utilities.format import fancy_format
from openavmkit.utilities.modeling import NaiveSqftModel, LocalSqftModel, PassThroughModel, GWRModel, MRAModel, \
	LarsModel, GroundTruthModel
from openavmkit.utilities.settings import get_fields_categorical, get_variable_interactions, get_valuation_date, \
	get_model_group, apply_dd_to_df_rows, get_model_group_ids
from openavmkit.utilities.stats import calc_vif_recursive_drop, calc_t_values_recursive_drop, \
	calc_p_values_recursive_drop, calc_elastic_net_regularization, calc_correlations, calc_r2, \
	calc_cross_validation_score, calc_cod
from openavmkit.utilities.timing import TimingData


# Public:

class BenchmarkResults:
	"""
  Container for benchmark results.

  Attributes:
      df_time (pd.DataFrame): DataFrame containing timing information.
      df_stats_test (pd.DataFrame): DataFrame with statistics for the test set.
      df_stats_full (pd.DataFrame): DataFrame with statistics for the full universe.
  """
	df_time: pd.DataFrame
	df_stats_test: pd.DataFrame
	df_stats_full: pd.DataFrame

	def __init__(self, df_time: pd.DataFrame, df_stats_test: pd.DataFrame, df_stats_full: pd.DataFrame):
		"""
    Initialize a BenchmarkResults instance.

    :param df_time: DataFrame containing timing data.
    :type df_time: pandas.DataFrame
    :param df_stats_test: DataFrame with test set statistics.
    :type df_stats_test: pandas.DataFrame
    :param df_stats_full: DataFrame with full universe statistics.
    :type df_stats_full: pandas.DataFrame
    """
		self.df_time = df_time
		self.df_stats_test = df_stats_test
		self.df_stats_full = df_stats_full

	def print(self) -> str:
		"""
    Return a formatted string summarizing the benchmark results.

    :returns: A string that includes timings, test set stats, and universe set stats.
    :rtype: str
    """
		result = "Timings:\n"
		result += _format_benchmark_df(self.df_time)
		result += "\n\n"
		result += "Test set:\n"
		result += _format_benchmark_df(self.df_stats_test)
		result += "\n\n"
		result += "Universe set:\n"
		result += _format_benchmark_df(self.df_stats_full)
		result += "\n\n"
		return result


class MultiModelResults:
	"""
  Container for results from multiple models along with a benchmark.

  Attributes:
      model_results (dict[str, SingleModelResults]): Dictionary mapping model names to their results.
      benchmark (BenchmarkResults): Benchmark results computed from the model results.
  """
	model_results: dict[str, SingleModelResults]
	benchmark: BenchmarkResults

	def __init__(
			self,
			model_results: dict[str, SingleModelResults],
			benchmark: BenchmarkResults
	):
		"""
    Initialize a MultiModelResults instance.

    :param model_results: Dictionary of individual model results.
    :type model_results: dict[str, SingleModelResults]
    :param benchmark: Benchmark results.
    :type benchmark: BenchmarkResults
    """
		self.model_results = model_results
		self.benchmark = benchmark

	def add_model(
			self,
			model: str,
			results: SingleModelResults
	):
		"""
    Add a new model's results and update the benchmark.

    :param model: The model name.
    :type model: str
    :param results: The results for the given model.
    :type results: SingleModelResults
    :returns: None
    """
		self.model_results[model] = results
		# Recalculate the benchmark based on updated model results.
		self.benchmark = _calc_benchmark(self.model_results)


def get_variable_recommendations(
		df_sales: pd.DataFrame,
		df_universe: pd.DataFrame,
		vacant_only: bool,
		settings: dict,
		model_group: str,
		verbose: bool = False
):
	"""
  Determine which variables are most likely to be meaningful in a model.

  This function examines sales and universe data, applies feature selection via
  correlations, elastic net regularization, R², p-values, t-values, and VIF, and produces
  a set of recommended variables along with a written report.

  :param df_sales: The sales data.
  :type df_sales: pandas.DataFrame
  :param df_universe: The parcel universe data.
  :type df_universe: pandas.DataFrame
  :param vacant_only: Whether to consider only vacant sales.
  :type vacant_only: bool
  :param settings: The settings dictionary.
  :type settings: dict
  :param model_group: The model group to consider.
  :type model_group: str
  :param verbose: If True, prints additional debugging information.
  :type verbose: bool, optional	
  :returns: A dictionary with keys "variables" (the best variables list) and "report" (the generated report).
  :rtype: dict
  """
	if verbose:
		print("")

	report = MarkdownReport("variables")

	if "sale_price_time_adj" not in df_sales:
		warnings.warn("Time adjustment was not found in sales data. Calculating now...")
		df_sales = enrich_time_adjustment(df_sales, settings, verbose=verbose)

	ds = _prepare_ds(df_sales, df_universe, model_group, vacant_only, settings)
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()

	feature_selection = settings.get("modeling", {}).get("instructions", {}).get("feature_selection", {})
	thresh = feature_selection.get("thresholds", {})

	X_sales = ds.X_sales[ds.ind_vars]
	y_sales = ds.y_sales

	# Correlation
	X_corr = ds.df_sales[[ds.dep_var] + ds.ind_vars]
	corr_results = calc_correlations(X_corr, thresh.get("correlation", 0.1))

	# Elastic net regularization
	try:
		enr_coefs = calc_elastic_net_regularization(X_sales, y_sales, thresh.get("enr", 0.01))
	except ValueError as e:
		nulls_in_X = X_sales[X_sales.isna().any(axis=1)]
		print(f"Found {len(nulls_in_X)} rows with nulls in X:")
		raise e

	# R² values
	r2_values = calc_r2(ds.df_sales, ds.ind_vars, y_sales)

	# P Values
	p_values = calc_p_values_recursive_drop(X_sales, y_sales, thresh.get("p_value", 0.05))

	# T Values
	t_values = calc_t_values_recursive_drop(X_sales, y_sales, thresh.get("t_value", 2))

	# VIF
	vif = calc_vif_recursive_drop(X_sales, thresh.get("vif", 10))

	# Generate final results & recommendations
	df_results = _calc_variable_recommendations(
		ds=ds,
		settings=settings,
		correlation_results=corr_results,
		enr_results=enr_coefs,
		r2_values_results=r2_values,
		p_values_results=p_values,
		t_values_results=t_values,
		vif_results=vif,
		report=report
	)

	curr_variables = df_results["variable"].tolist()
	best_variables = curr_variables.copy()
	best_score = float('inf')

	y = ds.y_sales
	while len(curr_variables) > 0:
		X = ds.df_sales[curr_variables]
		cv_score = calc_cross_validation_score(X, y)
		if cv_score < best_score:
			best_score = cv_score
			best_variables = curr_variables.copy()
		worst_idx = df_results["weighted_score"].idxmin()
		worst_variable = df_results.loc[worst_idx, "variable"]
		curr_variables.remove(worst_variable)
		# Remove the variable from the results dataframe.
		df_results = df_results[df_results["variable"].ne(worst_variable)]
		if verbose:
			print(f"--> score: {cv_score:,.0f}  {len(curr_variables)} variables: {curr_variables}")

	# Create a table from the list of best variables.
	df_best = pd.DataFrame(best_variables, columns=["Variable"])
	df_best["Rank"] = range(1, len(df_best) + 1)
	df_best["Description"] = df_best["Variable"]
	df_best = apply_dd_to_df_rows(df_best, "Variable", settings, ds.one_hot_descendants, "name")
	df_best = apply_dd_to_df_rows(df_best, "Description", settings, ds.one_hot_descendants, "description")
	df_best = df_best[["Rank", "Variable", "Description"]]
	df_best.loc[
		df_best["Variable"].eq(df_best["Description"]),
		"Description"
	] = ""
	df_best.set_index("Rank", inplace=True)
	report.set_var("summary_table", df_best.to_markdown())

	report = generate_variable_report(report, settings, model_group, best_variables)

	return {
		"variables": best_variables,
		"report": report
	}


def generate_variable_report(
		report: MarkdownReport,
		settings: dict,
		model_group: str,
		best_variables: list[str]
):
	"""
  Generate a variable selection report.

  This function updates the markdown report with various threshold values, weights,
  and summary tables based on the best variables.

  :param report: The markdown report object.
  :type report: MarkdownReport
  :param settings: The settings dictionary.
  :type settings: dict
  :param model_group: The model group identifier.
  :type model_group: str
  :param best_variables: List of selected best variables.
  :type best_variables: list[str]
  :returns: The updated markdown report.
  :rtype: MarkdownReport
  """
	locality = settings.get("locality", {})
	report.set_var("locality", locality.get("name", "...LOCALITY..."))

	mg = get_model_group(settings, model_group)
	report.set_var("val_date", get_valuation_date(settings).strftime("%Y-%m-%d"))
	report.set_var("model_group", mg.get("name", mg))

	instructions = settings.get("modeling", {}).get("instructions", {})
	feature_selection = instructions.get("feature_selection", {})
	thresh = feature_selection.get("thresholds", {})

	report.set_var("thresh_correlation", thresh.get("correlation"), fmt=".2f")
	report.set_var("thresh_enr_coef", thresh.get("enr_coef"), fmt=".2f")
	report.set_var("thresh_vif", thresh.get("vif"), fmt=".2f")
	report.set_var("thresh_p_value", thresh.get("p_value"), fmt=".2f")
	report.set_var("thresh_t_value", thresh.get("t_value"), fmt=".2f")
	report.set_var("thresh_adj_r2", thresh.get("adj_r2"), fmt=".2f")

	weights = feature_selection.get("weights", {})
	df_weights = pd.DataFrame(weights.items(), columns=["Statistic", "Weight"])
	df_weights["Statistic"] = df_weights["Statistic"].map({
		"vif": "VIF",
		"p_value": "P-value",
		"t_value": "T-value",
		"corr_score": "Correlation",
		"enr_coef": "ENR",
		"coef_sign": "Coef. sign",
		"adj_r2": "R-squared"
	})
	df_weights.set_index("Statistic", inplace=True)
	report.set_var("pre_model_weights", df_weights.to_markdown())

	# TODO: Construct summary and post-model tables as needed.
	post_model_table = "...POST MODEL TABLE..."
	report.set_var("post_model_table", post_model_table)

	return report


def run_models(
		sup: SalesUniversePair,
		settings: dict,
		save_params: bool = True,
		use_saved_params: bool = True,
		save_results: bool = True,
		use_saved_results: bool = True,
		verbose: bool = False,
		run_main: bool = True,
		run_vacant: bool = True,
		run_hedonic: bool = True
):
	"""
  Runs predictive models on the given SalesUniversePair. This function takes detailed instructions from the provided
	settings dictionary and handles all the internal details like splitting the data, training the models, and saving the
	results. It performs basic statistic analysis on each model, and optionally combines results into an ensemble model.

	If "run_main" is true, it will run normal models as well as hedonic models (if the user so specifies), "hedonic" in
	this context meaning models that attempt to generate a land value and an improvement value separately. If "run_vacant"
	is true, it will run vacant models as well -- models that only use vacant models as evidence to generate land values.

  This function iterates over model groups and runs models for both main and vacant cases.

  :param sup: Sales and universe data.
  :type sup: SalesUniversePair
  :param settings: The settings dictionary.
  :type settings: dict
  :param save_params: Whether to save model parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to use saved model parameters.
  :type use_saved_params: bool, optional
  :param save_results: Whether to save model results.
  :type save_results: bool, optional
  :param use_saved_results: Whether to use saved model results.
  :type use_saved_results: bool, optional
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :param run_main: Whether to run main (non-vacant) models.
  :type run_main: bool, optional
  :param run_vacant: Whether to run vacant models.
  :type run_vacant: bool, optional
  :returns: The MultiModelResults containing all model results and benchmarks.
  :rtype: MultiModelResults
  """
	s = settings
	s_model = s.get("modeling", {})
	s_inst = s_model.get("instructions", {})
	model_groups = s_inst.get("model_groups", [])

	df_univ = sup["universe"]

	if len(model_groups) == 0:
		model_groups = get_model_group_ids(settings, df_univ)

	dict_all_results = {}

	for model_group in model_groups:
		if verbose:
			print("")
			print(f"*** Running models for model_group: {model_group} ***")
			print("")
		for vacant_only in [False, True]:
			if vacant_only and not run_vacant:
				continue
			if not vacant_only and not run_main:
				continue
			mg_results = _run_models(sup, model_group, settings, vacant_only, save_params, use_saved_params, save_results, use_saved_results, verbose, run_hedonic)
			dict_all_results[model_group] = mg_results

	write_out_all_results(sup, dict_all_results)

	return dict_all_results


def write_out_all_results(sup:SalesUniversePair, all_results:dict):
	df_all = None

	for model_group in all_results:
		mm_results:MultiModelResults = all_results[model_group]
		ensemble:SingleModelResults = mm_results.model_results["ensemble"]

		df_univ_local = ensemble.df_universe[["key", ensemble.field_prediction]].rename(columns={ensemble.field_prediction: "market_value"})
		df_univ_local["model_group"] = model_group

		if df_all is None:
			df_all = df_univ_local
		else:
			df_all = pd.concat([df_all, df_univ_local])

	df_univ = sup.universe.copy()
	df_univ = df_univ.merge(df_all, on="key", how="left")

	outpath = "out/models/all_model_groups"
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	df_univ.to_csv(f"{outpath}/universe.csv", index=False)
	df_univ.to_parquet(f"{outpath}/universe.parquet", index=False)


# Private functions:


def _calc_benchmark(model_results: dict[str, SingleModelResults]):
	"""
  Calculate benchmark statistics from individual model results.

  :param model_results: Dictionary mapping model names to SingleModelResults.
  :type model_results: dict[str, SingleModelResults]
  :returns: BenchmarkResults computed from the model results.
  :rtype: BenchmarkResults
  """
	data_time = {
		"model": [],
		"total": [],
		"param": [],
		"train": [],
		"test": [],
		"univ": [],
		"multi": [],
		"chd": []
	}

	data = {
		"model": [],
		"subset": [],
		"utility_score": [],
		"count_sales": [],
		"count_univ": [],
		"mse": [],
		"rmse": [],
		"r2": [],
		"adj_r2": [],
		"median_ratio": [],
		"cod": [],
		"cod_trim": [],
		"prd": [],
		"prb": [],
		"chd": []
	}
	for key in model_results:
		for kind in ["test", "univ"]:
			results = model_results[key]
			if kind == "test":
				pred_results = results.pred_test
				subset = "Test set"
			else:
				pred_results = results.pred_sales
				subset = "Universe set"

			data["model"].append(key)
			data["subset"].append(subset)
			data["utility_score"].append(results.utility)
			data["count_sales"].append(pred_results.ratio_study.count)
			data["count_univ"].append(results.df_universe.shape[0])
			data["mse"].append(pred_results.mse)
			data["rmse"].append(pred_results.rmse)
			data["r2"].append(pred_results.r2)
			data["adj_r2"].append(pred_results.adj_r2)
			data["median_ratio"].append(pred_results.ratio_study.median_ratio)
			data["cod"].append(pred_results.ratio_study.cod)
			data["cod_trim"].append(pred_results.ratio_study.cod_trim)
			data["prd"].append(pred_results.ratio_study.prd)
			data["prb"].append(pred_results.ratio_study.prb)

			chd_results = None
			if kind == "univ":
				chd_results = results.chd
				tim = results.timing.results
				data_time["model"].append(key)
				data_time["total"].append(tim.get("total"))
				data_time["param"].append(tim.get("parameter_search"))
				data_time["train"].append(tim.get("train"))
				data_time["test"].append(tim.get("predict_test"))
				data_time["univ"].append(tim.get("predict_univ"))
				data_time["multi"].append(tim.get("predict_multi"))
				data_time["chd"].append(tim.get("chd"))
			data["chd"].append(chd_results)

	df = pd.DataFrame(data)
	df_test = df[df["subset"].eq("Test set")].drop(columns=["subset"])
	df_full = df[df["subset"].eq("Universe set")].drop(columns=["subset"])
	df_time = pd.DataFrame(data_time)

	df_test.set_index("model", inplace=True)
	df_full.set_index("model", inplace=True)
	df_time.set_index("model", inplace=True)

	results = BenchmarkResults(
		df_time,
		df_test,
		df_full
	)
	return results


def _format_benchmark_df(df: pd.DataFrame):
	"""
  Format a benchmark DataFrame for display.

  :param df: The DataFrame to format.
  :type df: pandas.DataFrame
  :returns: A markdown-formatted string representation of the DataFrame.
  :rtype: str
  """
	formats = {
		"utility_score": fancy_format,
		"count_sales": "{:,.0f}",
		"count_univ": "{:,.0f}",
		"mse": fancy_format,
		"rmse": fancy_format,
		"r2": "{:.2f}",
		"adj_r2": "{:.2f}",
		"median_ratio": "{:.2f}",
		"cod": "{:.2f}",
		"cod_trim": "{:.2f}",
		"prd": "{:.2f}",
		"prb": "{:.2f}",
		"total": fancy_format,
		"param": fancy_format,
		"train": fancy_format,
		"test": fancy_format,
		"univ": fancy_format,
		"multi": fancy_format,
		"chd": fancy_format
	}

	for col in df.columns:
		if col in formats:
			if callable(formats[col]):
				df[col] = df[col].apply(formats[col])
			else:
				df[col] = df[col].apply(lambda x: formats[col].format(x))
	return df.transpose().to_markdown()


def _predict_one_model(
		smr: SingleModelResults,
		model: str,
		outpath: str,
		settings: dict,
		use_saved_results: bool = False,
		verbose: bool = False
) -> SingleModelResults:
	"""
  Predict results for one model, using saved results if available.

  :param smr: The single model results container.
  :type smr: SingleModelResults
  :param model: The model name.
  :type model: str
  :param outpath: Output directory path.
  :type outpath: str
  :param settings: The settings dictionary.
  :type settings: dict
  :param use_saved_results: If True, loads saved results if they exist.
  :type use_saved_results: bool, optional
  :param verbose: If True, prints additional output.
  :type verbose: bool, optional
  :returns: Updated SingleModelResults.
  :rtype: SingleModelResults
  """
	model_name = model
	out_pickle = f"{outpath}/model_{model_name}.pickle"
	ds = smr.ds
	if use_saved_results and os.path.exists(out_pickle):
		with open(out_pickle, "rb") as file:
			results = pickle.load(file)
		return results

	timing = TimingData()
	timing.start("total")

	results: SingleModelResults | None = None

	if model_name == "garbage":
		garbage_model: GarbageModel = smr.model
		results = predict_garbage(ds, garbage_model, timing, verbose)
	elif model_name == "garbage_normal":
		garbage_model: GarbageModel = smr.model
		results = predict_garbage(ds, garbage_model, timing, verbose)
	elif model_name == "mean":
		mean_model: AverageModel = smr.model
		results = predict_average(ds, mean_model, timing, verbose)
	elif model_name == "median":
		median_model: AverageModel = smr.model
		results = predict_average(ds, median_model, timing, verbose)
	elif model_name == "naive_sqft":
		sqft_model: NaiveSqftModel = smr.model
		results = predict_naive_sqft(ds, sqft_model, timing, verbose)
	elif model_name == "local_sqft":
		sqft_model: LocalSqftModel = smr.model
		results = predict_local_sqft(ds, sqft_model, timing, verbose)
	elif model_name == "lars":
		lars_model: LarsModel = smr.model
		results = predict_lars(ds, lars_model, timing, verbose)
	elif model_name == "assessor":
		assr_model: PassThroughModel = smr.model
		results = predict_pass_through(ds, assr_model, timing, verbose)
	elif model_name == "ground_truth":
		assr_model: GroundTruthModel = smr.model
		results = predict_ground_truth(ds, assr_model, timing, verbose)
	elif model_name == "mra":
		# MRA is a special case where we have to call run_ instead of predict_, because there's delicate state mangling.
		# We pass the pretrained `model` object to run_mra() to get it to skip training and move straight to prediction
		model: MRAModel = smr.model
		results = run_mra(ds, model.intercept, verbose, model)
	elif model_name == "kernel":
		kernel_reg: KernelReg = smr.model
		results = predict_kernel(ds, kernel_reg, timing, verbose)
	elif model_name == "gwr":
		gwr_model: GWRModel = smr.model
		results = predict_gwr(ds, gwr_model, timing, verbose)
	elif model_name == "xgboost":
		xgb_regressor: XGBRegressor = smr.model
		results = predict_xgboost(ds, xgb_regressor, timing, verbose)
	elif model_name == "lightgbm":
		lightgbm_regressor: Booster = smr.model
		results = predict_lightgbm(ds, lightgbm_regressor, timing, verbose)
	elif model_name == "catboost":
		catboost_regressor: CatBoostRegressor = smr.model
		results = predict_catboost(ds, catboost_regressor, timing, verbose)

	_write_model_results(results, outpath, settings)

	with open(out_pickle, "wb") as file:
		pickle.dump(results, file)

	return results


def get_data_split_for(
		name: str,
		model_group: str,
		location_fields: list[str] | None,
		ind_vars: list[str],
		df_sales: pd.DataFrame,
		df_universe: pd.DataFrame,
		settings: dict,
		dep_var: str,
		dep_var_test: str,
		fields_cat: list[str],
		interactions: dict,
		test_keys: list[str],
		train_keys: list[str],
		vacant_only: bool,
		hedonic: bool,
		hedonic_test_against_vacant_sales: bool = True,
		df_multiverse: pd.DataFrame | None = None
):
	"""
  Prepare a DataSplit object for a given model.

  :param name: Model name.
  :type name: str
  :param model_group: The model group identifier.
  :type model_group: str
  :param location_fields: List of location fields.
  :type location_fields: list[str] or None
  :param ind_vars: List of independent variables.
  :type ind_vars: list[str]
  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame
  :param settings: The settings dictionary.
  :type settings: dict
  :param dep_var: Dependent variable for training.
  :type dep_var: str
  :param dep_var_test: Dependent variable for testing.
  :type dep_var_test: str
  :param fields_cat: List of categorical fields.
  :type fields_cat: list[str]
  :param interactions: Dictionary of variable interactions.
  :type interactions: dict
  :param test_keys: Keys for test split.
  :type test_keys: list[str]
  :param train_keys: Keys for training split.
  :type train_keys: list[str]
  :param vacant_only: Whether to consider only vacant sales.
  :type vacant_only: bool
  :param hedonic: Whether to use hedonic pricing.
  :type hedonic: bool
  :param df_multiverse: Optional multiverse DataFrame.
  :type df_multiverse: pandas.DataFrame or None
  :returns: A DataSplit object.
  :rtype: DataSplit
  """
	if name == "local_sqft":
		_ind_vars = location_fields + ["bldg_area_finished_sqft", "land_area_sqft"]
	elif name == "assessor":
		_ind_vars = ["assr_land_value"] if hedonic else ["assr_market_value"]
	elif name == "ground_truth":
		_ind_vars = ["true_land_value"] if hedonic else ["true_market_value"]
	else:
		_ind_vars = ind_vars
		if name == "gwr" or name == "kernel":
			exclude_vars = ["latitude", "longitude", "latitude_norm", "longitude_norm"]
			_ind_vars = [var for var in _ind_vars if var not in exclude_vars]

	return DataSplit(
		df_sales,
		df_universe,
		model_group,
		settings,
		dep_var,
		dep_var_test,
		_ind_vars,
		fields_cat,
		interactions,
		test_keys,
		train_keys,
		vacant_only=vacant_only,
		hedonic=hedonic,
		df_multiverse=df_multiverse
	)


def run_one_model(
		df_multiverse: pd.DataFrame | None,
		df_sales: pd.DataFrame,
		df_universe: pd.DataFrame,
		vacant_only: bool,
		model_group: str,
		model: str,
		model_entries: dict,
		settings: dict,
		dep_var: str,
		dep_var_test: str,
		best_variables: list[str],
		fields_cat: list[str],
		outpath: str,
		save_params: bool,
		use_saved_params: bool,
		save_results: bool,
		use_saved_results: bool,
		verbose: bool = False,
		hedonic: bool = False,
		test_keys: list[str] | None = None,
		train_keys: list[str] | None = None
) -> SingleModelResults | None:
	"""
  Run a single model based on provided parameters and return its results.

  :param df_multiverse: Multiverse DataFrame.
  :type df_multiverse: pandas.DataFrame
  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool
  :param model_group: Model group identifier.
  :type model_group: str
  :param model: Model name.
  :type model: str
  :param model_entries: Dictionary of model configuration entries.
  :type model_entries: dict
  :param settings: Settings dictionary.
  :type settings: dict
  :param dep_var: Dependent variable for training.
  :type dep_var: str
  :param dep_var_test: Dependent variable for testing.
  :type dep_var_test: str
  :param best_variables: List of best variables selected.
  :type best_variables: list[str]
  :param fields_cat: List of categorical fields.
  :type fields_cat: list[str]
  :param outpath: Output path for saving results.
  :type outpath: str
  :param save_params: Whether to save parameters.
  :type save_params: bool
  :param use_saved_params: Whether to use saved parameters.
  :type use_saved_params: bool
  :param save_results: Whether to save results.
  :type save_results: bool
  :param use_saved_results: Whether to load saved results if available.
  :type use_saved_results: bool
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :param hedonic: Whether to use hedonic pricing.
  :type hedonic: bool, optional
  :param test_keys: Optional list of test keys (will be read from disk if not provided)
  :type test_keys: list[str] or None
  :param train_keys: Optional list of training keys (will be read from disk if not provided)
  :type train_keys: list[str] or None
  :returns: SingleModelResults if successful, else None.
  :rtype: SingleModelResults or None
  """
	model_name = model
	out_pickle = f"{outpath}/model_{model_name}.pickle"
	if use_saved_results and os.path.exists(out_pickle):
		with open(out_pickle, "rb") as file:
			results = pickle.load(file)
		return results

	entry: dict | None = model_entries.get(model, None)
	default_entry: dict | None = model_entries.get("default", {})
	if entry is None:
		entry = default_entry
		if entry is None:
			raise ValueError(f"Model entry for {model} not found, and there is no default entry!")

	if "*" in model:
		sales_chase = 0.01
		model_name = model.replace("*", "")
	else:
		sales_chase = False

	if verbose:
		print(f" running model {model} on {len(df_sales)} rows...")

	are_dep_vars_default = entry.get("ind_vars", None) is None
	ind_vars: list | None = entry.get("ind_vars", default_entry.get("ind_vars", None))
	if ind_vars is None:
		raise ValueError(f"ind_vars not found for model {model}")

	if are_dep_vars_default and verbose:
		if set(ind_vars) != set(best_variables):
			print(f"--> using default variables, auto-optimized variable list: {best_variables}")
		ind_vars = best_variables

	interactions = get_variable_interactions(entry, settings, df_sales)
	location_fields = get_locations(settings, df_sales)

	if test_keys is None or train_keys is None:
		test_keys, train_keys = _read_split_keys(model_group)
	else:
		if verbose:
			print(f"--> using provided test/train keys")
			print(f"--> test keys: {len(test_keys)}")
			print(f"--> train keys: {len(train_keys)}")

	ds = get_data_split_for(
		name=model_name,
		model_group=model_group,
		location_fields=location_fields,
		ind_vars=ind_vars,
		df_sales=df_sales,
		df_universe=df_universe,
		settings=settings,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		fields_cat=fields_cat,
		interactions=interactions,
		test_keys=test_keys,
		train_keys=train_keys,
		vacant_only=vacant_only,
		hedonic=hedonic,
		hedonic_test_against_vacant_sales=True,
		df_multiverse=df_multiverse
	)

	if len(ds.y_sales) < 15:
		if verbose:
			print(f"--> model {model} has less than 15 sales. Skipping...")
		return None

	intercept = entry.get("intercept", True)

	if model_name == "garbage":
		results = run_garbage(ds, normal=False, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "garbage_normal":
		results = run_garbage(ds, normal=True, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "mean":
		results = run_average(ds, average_type="mean", sales_chase=sales_chase, verbose=verbose)
	elif model_name == "median":
		results = run_average(ds, average_type="median", sales_chase=sales_chase, verbose=verbose)
	elif model_name == "naive_sqft":
		results = run_naive_sqft(ds, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "local_sqft":
		results = run_local_sqft(ds, location_fields=location_fields, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "assessor":
		results = run_pass_through(ds, verbose=verbose)
	elif model_name == "ground_truth":
		results = run_ground_truth(ds, verbose=verbose)
	elif model_name == "mra":
		results = run_mra(ds, intercept=intercept, verbose=verbose)
	elif model_name == "kernel":
		results = run_kernel(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "gwr":
		results = run_gwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "xgboost":
		results = run_xgboost(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "lightgbm":
		results = run_lightgbm(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "catboost":
		results = run_catboost(ds, outpath, save_params, use_saved_params, verbose=verbose)
	else:
		raise ValueError(f"Model {model_name} not found!")

	_write_model_results(results, outpath, settings)

	with open(out_pickle, "wb") as file:
		pickle.dump(results, file)

	return results


def run_one_hedonic_model(
		df_multiverse: pd.DataFrame,
		df_sales: pd.DataFrame,
		df_univ: pd.DataFrame,
		settings: dict,
		model: str,
		smr: SingleModelResults,
		model_group: str,
		dep_var: str,
		dep_var_test: str,
		fields_cat: list[str],
		outpath: str,
		hedonic_test_against_vacant_sales: bool = True,
		use_saved_results: bool = False,
		verbose: bool = False
):
	location_field_neighborhood = get_important_field(settings, "loc_neighborhood", df_sales)
	location_field_market_area = get_important_field(settings, "loc_market_area", df_sales)
	location_fields = [location_field_neighborhood, location_field_market_area]

	ds = get_data_split_for(
		name=model,
		model_group=model_group,
		location_fields=location_fields,
		ind_vars=smr.ind_vars,
		df_sales=df_sales,
		df_universe=df_univ,
		settings=settings,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		fields_cat=fields_cat,
		interactions=smr.ds.interactions.copy(),
		test_keys=smr.ds.test_keys,
		train_keys=smr.ds.train_keys,
		vacant_only=False,
		hedonic=True,
		hedonic_test_against_vacant_sales=hedonic_test_against_vacant_sales,
		df_multiverse=df_multiverse
	)
	# We call this here because we are re-running prediction without first calling run(), which would call this
	ds.split()
	if hedonic_test_against_vacant_sales and len(ds.y_sales) < 15:
		print(f"Skipping hedonic model because there are not enough sale records...")
		return None
	smr.ds = ds
	results = _predict_one_model(
		smr=smr,
		model=model,
		outpath=outpath,
		settings=settings,
		use_saved_results=use_saved_results,
		verbose=verbose
	)
	return results




def _assemble_model_results(results: SingleModelResults, settings: dict):
	"""
  Assemble model results into DataFrames for sales, universe, and test sets.

  :param results: Single model results.
  :type results: SingleModelResults
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: A dictionary mapping keys ("sales", "universe", "test", and optionally "multiverse") to DataFrames.
  :rtype: dict
  """
	locations = get_report_locations(settings)
	fields = ["key", "geometry", "prediction",
						"assr_market_value", "assr_land_value",
						"true_market_value", "true_land_value",
						"sale_price", "sale_price_time_adj", "sale_date"] + locations
	fields = [field for field in fields if field in results.df_sales.columns]

	dfs = {
		"sales": results.df_sales[["key_sale"]+fields].copy(),
		"universe": results.df_universe[fields].copy(),
		"test": results.df_test[["key_sale"]+fields].copy()
	}

	if results.df_multiverse is not None:
		dfs["multiverse"] = results.df_multiverse[fields].copy()

	for key in dfs:
		df = dfs[key]
		df["prediction_ratio"] = div_z_safe(df, "prediction", "sale_price_time_adj")
		if "assr_market_value" in df:
			df["assr_ratio"] = div_z_safe(df, "assr_market_value", "sale_price_time_adj")
		else:
			df["assr_ratio"] = None
		if "true_market_value" in df:
			df["true_vs_sale_ratio"] = div_z_safe(df, "true_market_value", "sale_price_time_adj")
			df["pred_vs_true_ratio"] = div_z_safe(df, "prediction", "true_market_value")
		for location in locations:
			if location in df:
				df[f"prediction_cod_{location}"] = None
				df[f"assr_cod_{location}"] = None
				location_values = df[location].unique()
				for value in location_values:
					predictions = df.loc[df[location].eq(value), "prediction_ratio"].values
					predictions = predictions[~pd.isna(predictions)]
					df.loc[df[location].eq(value), f"prediction_cod_{location}"] = calc_cod(predictions)

					if "assr_market_value" in df:
						assr_ratios = df.loc[df[location].eq(value), "assr_ratio"].values
						assr_ratios = assr_ratios[~pd.isna(assr_ratios)]
						df.loc[df[location].eq(value), f"assr_cod_{location}"] = calc_cod(assr_ratios)
					if "true_market_value" in df:
						true_vs_sales_ratios = df.loc[df[location].eq(value), "true_vs_sale_ratio"].values
						true_vs_sales_ratios = true_vs_sales_ratios[~pd.isna(true_vs_sales_ratios)]
						df.loc[df[location].eq(value), f"true_vs_sale_cod_{location}"] = calc_cod(true_vs_sales_ratios)

						pred_vs_true_ratios = df.loc[df[location].eq(value), "pred_vs_true_ratio"].values
						pred_vs_true_ratios = pred_vs_true_ratios[~pd.isna(pred_vs_true_ratios)]
						df.loc[df[location].eq(value), f"pred_vs_true_cod_{location}"] = calc_cod(pred_vs_true_ratios)

	return dfs


def _write_model_results(results: SingleModelResults, outpath: str, settings: dict):
	"""
  Write model results to disk in parquet and CSV formats.

  :param results: Single model results.
  :type results: SingleModelResults
  :param outpath: Output directory path.
  :type outpath: str
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: None
  """
	dfs = _assemble_model_results(results, settings)
	path = f"{outpath}/{results.type}"
	if "*" in path:
		path = path.replace("*", "_star")
	os.makedirs(path, exist_ok=True)
	for key in dfs:
		df = dfs[key]
		df.to_parquet(f"{path}/pred_{key}.parquet")
		if "geometry" in df:
			df = df.drop(columns=["geometry"])
		df.to_csv(f"{path}/pred_{key}.csv", index=False)

	results.df_sales.to_csv(f"{path}/sales.csv", index=False)
	results.df_universe.to_csv(f"{path}/universe.csv", index=False)


def _write_ensemble_model_results(
		results: SingleModelResults,
		outpath: str,
		settings: dict,
		dfs: dict[str, pd.DataFrame],
		ensemble_list: list[str]
):
	"""
  Write ensemble model results to disk.

  :param results: Single model results for the ensemble.
  :type results: SingleModelResults
  :param outpath: Output directory path.
  :type outpath: str
  :param settings: Settings dictionary.
  :type settings: dict
  :param dfs: Dictionary of DataFrames with ensemble predictions.
  :type dfs: dict[str, pandas.DataFrame]
  :param ensemble_list: List of models used in the ensemble.
  :type ensemble_list: list[str]
  :returns: None
  """
	dfs_basic = _assemble_model_results(results, settings)
	path = f"{outpath}/{results.type}"
	os.makedirs(path, exist_ok=True)
	for key in dfs_basic:
		prim_keys = ["key"]
		merge_key = "key"
		if key in ["sales", "test"]:
			prim_keys.append("key_sale")
			merge_key = "key_sale"
		df_basic = dfs_basic[key]
		df_ensemble = dfs[key]
		df_ensemble = df_ensemble[prim_keys + ensemble_list]
		df = df_basic.merge(df_ensemble, on=merge_key, how="left")
		df.to_parquet(f"{path}/pred_ensemble_{key}.parquet")
		df.to_csv(f"{path}/pred_ensemble_{key}.csv", index=False)


def _optimize_ensemble_allocation(
		df_sales: pd.DataFrame | None,
		df_universe: pd.DataFrame | None,
		model_group: str,
		vacant_only: bool,
		dep_var: str,
		dep_var_test: str,
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False,
		hedonic: bool = False,
		ensemble_list: list[str] = None
):
	"""
  Select the models that produce the best land allocation results for an ensemble model.

  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame or None
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame or None
  :param model_group: Model group identifier.
  :type model_group: str
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool
  :param dep_var: Dependent variable for training.
  :type dep_var: str
  :param dep_var_test: Dependent variable for testing.
  :type dep_var_test: str
  :param all_results: MultiModelResults containing individual model results.
  :type all_results: MultiModelResults
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :param hedonic: Whether to use hedonic pricing.
  :type hedonic: bool, optional
  :param ensemble_list: Optional list of models to consider for the ensemble.
  :type ensemble_list: list[str] or None
  :returns: The best ensemble list.
  :rtype: list[str]
  """
	timing = TimingData()
	timing.start("total")
	timing.start("setup")

	if df_sales is None:
		first_key = list(all_results.model_results.keys())[0]
		df_universe = all_results.model_results[first_key].ds.df_universe_orig
		df_sales = all_results.model_results[first_key].ds.df_sales_orig

	test_keys, train_keys = _read_split_keys(model_group)

	ds = DataSplit(
		df_sales,
		df_universe,
		model_group,
		settings,
		dep_var,
		dep_var_test,
		[],
		[],
		{},
		test_keys,
		train_keys,
		vacant_only=vacant_only,
		hedonic=hedonic
	)

	vacant_status = "vacant" if vacant_only else "main"
	df_test = ds.df_test
	df_univ = ds.df_universe
	instructions = settings.get("modeling", {}).get("instructions", {})

	if ensemble_list is None:
		ensemble_list = instructions.get(vacant_status, {}).get("ensemble", [])

	if len(ensemble_list) == 0:
		ensemble_list = [key for key in all_results.model_results.keys()]

	if "assessor" in ensemble_list:
		ensemble_list.remove("assessor")

	if "ground_truth" in ensemble_list:
		ensemble_list.remove("ground_truth")

	best_list = []
	best_score = float('inf')

	while len(ensemble_list) > 1:
		best_score, best_list = _optimize_ensemble_allocation_iteration(
			df_test,
			df_univ,
			timing,
			all_results,
			ds,
			best_score,
			best_list,
			ensemble_list,
			verbose
		)

	if verbose:
		print(f"Best score = {best_score:8.0f}, ensemble = {best_list}")
	return best_list


def _optimize_ensemble_allocation_iteration(
		df_test: pd.DataFrame,
		df_univ: pd.DataFrame,
		timing: TimingData,
		all_results: MultiModelResults,
		ds: DataSplit,
		best_score: float,
		best_list: list[str],
		ensemble_list: list[str],
		verbose: bool = False
):
	"""
  Perform one iteration of ensemble allocation optimization.

  :param df_test: Test DataFrame.
  :type df_test: pandas.DataFrame
  :param df_univ: Universe DataFrame.
  :type df_univ: pandas.DataFrame
  :param timing: TimingData object.
  :type timing: TimingData
  :param all_results: MultiModelResults containing model results.
  :type all_results: MultiModelResults
  :param ds: DataSplit object.
  :type ds: DataSplit
  :param best_score: Current best score.
  :type best_score: float
  :param best_list: Current best ensemble list.
  :type best_list: list[str]
  :param ensemble_list: List of models to consider in this iteration.
  :type ensemble_list: list[str]
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :returns: Tuple containing best score and best ensemble list.
  :rtype: tuple(float, list[str])
  """
	df_test_ensemble = df_test[["key_sale", "key"]].copy()
	df_univ_ensemble = df_univ[["key"]].copy()
	if len(ensemble_list) == 0:
		ensemble_list = [key for key in all_results.model_results.keys()]
	timing.stop("setup")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("train")
	for m_key in ensemble_list:
		m_results = all_results.model_results[m_key]
		df_test_ensemble[m_key] = m_results.pred_test.y_pred
		df_univ_ensemble[m_key] = m_results.pred_univ
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_test")

	timing.start("predict_sales")
	timing.stop("predict_sales")

	timing.start("predict_univ")
	y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_univ")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"ensemble",
		model="ensemble",
		y_pred_test=y_pred_test_ensemble.to_numpy(),
		y_pred_sales=None,
		y_pred_univ=y_pred_univ_ensemble.to_numpy(),
		timing=timing,
		verbose=verbose
	)
	score = results.utility

	timing.stop("total")

	if verbose:
		print(f"score = {score:5.0f}, best = {best_score:5.0f}, ensemble = {ensemble_list}...")

	if score < best_score and len(ensemble_list) >= 3:
		best_score = score
		best_list = ensemble_list.copy()

	# identify the WORST individual model:
	worst_model = None
	worst_score = float('-inf')
	for key in ensemble_list:
		if key in all_results.model_results:
			model_results = all_results.model_results[key]
			model_score = model_results.utility

			if model_score > worst_score:
				worst_score = model_score
				worst_model = key

	if worst_model is not None and len(ensemble_list) > 1:
		ensemble_list.remove(worst_model)

	return best_score, best_list


def _optimize_ensemble(
		df_sales: pd.DataFrame | None,
		df_universe: pd.DataFrame | None,
		model_group: str,
		vacant_only: bool,
		dep_var: str,
		dep_var_test: str,
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False,
		hedonic: bool = False,
		ensemble_list: list[str] = None
):
	"""
  Optimize the ensemble allocation over all iterations.

  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame or None
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame or None
  :param model_group: Model group identifier.
  :type model_group: str
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool
  :param dep_var: Dependent variable for training.
  :type dep_var: str
  :param dep_var_test: Dependent variable for testing.
  :type dep_var_test: str
  :param all_results: MultiModelResults containing model results.
  :type all_results: MultiModelResults
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :param hedonic: Whether to use hedonic pricing.
  :type hedonic: bool, optional
  :param ensemble_list: Optional list of models to consider.
  :type ensemble_list: list[str] or None
  :returns: The best ensemble list.
  :rtype: list[str]
  """
	timing = TimingData()
	timing.start("total")
	timing.start("setup")

	if df_sales is None:
		first_key = list(all_results.model_results.keys())[0]
		df_universe = all_results.model_results[first_key].ds.df_universe_orig
		df_sales = all_results.model_results[first_key].ds.df_sales_orig

	test_keys, train_keys = _read_split_keys(model_group)
	ds = DataSplit(
		df_sales,
		df_universe,
		model_group,
		settings,
		dep_var,
		dep_var_test,
		[],
		[],
		{},
		test_keys,
		train_keys,
		vacant_only=vacant_only,
		hedonic=hedonic
	)

	vacant_status = "vacant" if vacant_only else "main"
	df_test = ds.df_test
	df_univ = ds.df_universe
	instructions = settings.get("modeling", {}).get("instructions", {})

	if ensemble_list is None:
		ensemble_list = instructions.get(vacant_status, {}).get("ensemble", [])

	if len(ensemble_list) == 0:
		ensemble_list = [key for key in all_results.model_results.keys()]

	if "assessor" in ensemble_list:
		ensemble_list.remove("assessor")

	if "ground_truth" in ensemble_list:
		ensemble_list.remove("ground_truth")

	best_list = []
	best_score = float('inf')

	while len(ensemble_list) > 1:
		best_score, best_list = _optimize_ensemble_iteration(
			df_test,
			df_univ,
			timing,
			all_results,
			ds,
			best_score,
			best_list,
			ensemble_list,
			verbose
		)

	if verbose:
		print(f"Best score = {best_score:8.0f}, ensemble = {best_list}")
	return best_list


def _optimize_ensemble_iteration(
		df_test: pd.DataFrame,
		df_univ: pd.DataFrame,
		timing: TimingData,
		all_results: MultiModelResults,
		ds: DataSplit,
		best_score: float,
		best_list: list[str],
		ensemble_list: list[str],
		verbose: bool = False
):
	df_test_ensemble = df_test[["key_sale", "key"]].copy()
	df_univ_ensemble = df_univ[["key"]].copy()
	if len(ensemble_list) == 0:
		ensemble_list = [key for key in all_results.model_results.keys()]
	timing.stop("setup")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("train")
	for m_key in ensemble_list:
		m_results = all_results.model_results[m_key]
		df_test_ensemble[m_key] = m_results.pred_test.y_pred
		df_univ_ensemble[m_key] = m_results.pred_univ
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_test")

	timing.start("predict_sales")
	timing.stop("predict_sales")

	timing.start("predict_univ")
	y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_univ")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"ensemble",
		model="ensemble",
		y_pred_test=y_pred_test_ensemble.to_numpy(),
		y_pred_sales=None,
		y_pred_univ=y_pred_univ_ensemble.to_numpy(),
		timing=timing,
		verbose=verbose
	)
	timing.stop("total")

	score = results.utility

	if verbose:
		print(f"score = {score:5.0f}, best = {best_score:5.0f}, ensemble = {ensemble_list}...")

	if score < best_score: # and len(ensemble_list) >= 3:
		best_score = score
		best_list = ensemble_list.copy()

	# identify the WORST individual model:
	worst_model = None
	worst_score = float('-inf')
	for key in ensemble_list:
		if key in all_results.model_results:
			model_results = all_results.model_results[key]

			if model_results.utility > worst_score:
				worst_score = model_results.utility
				worst_model = key

	if worst_model is not None and len(ensemble_list) > 1:
		ensemble_list.remove(worst_model)

	return best_score, best_list


def _run_ensemble(
		df_sales: pd.DataFrame,
		df_universe: pd.DataFrame,
		model_group: str,
		vacant_only: bool,
		hedonic: bool,
		dep_var: str,
		dep_var_test: str,
		outpath: str,
		ensemble_list: list[str],
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False,
		df_multiverse: pd.DataFrame = None
):
	"""
  Run the ensemble model based on the given ensemble list and write results.

  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame
  :param model_group: Model group identifier.
  :type model_group: str
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool
  :param hedonic: Whether to use hedonic pricing.
  :type hedonic: bool
  :param dep_var: Dependent variable for training.
  :type dep_var: str
  :param dep_var_test: Dependent variable for testing.
  :type dep_var_test: str
  :param outpath: Output path for results.
  :type outpath: str
  :param ensemble_list: List of models to include in the ensemble.
  :type ensemble_list: list[str]
  :param all_results: MultiModelResults containing model results.
  :type all_results: MultiModelResults
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :param df_multiverse: Optional multiverse DataFrame.
  :type df_multiverse: pandas.DataFrame or None
  :returns: SingleModelResults for the ensemble.
  :rtype: SingleModelResults
  """
	timing = TimingData()
	timing.start("total")
	timing.start("setup")

	test_keys, train_keys = _read_split_keys(model_group)
	ds = DataSplit(
		df_sales,
		df_universe,
		model_group,
		settings,
		dep_var,
		dep_var_test,
		[],
		[],
		{},
		test_keys,
		train_keys,
		vacant_only=vacant_only,
		hedonic=hedonic,
		df_multiverse=df_multiverse
	)
	ds.split()

	df_test = ds.df_test
	df_sales = ds.df_sales
	df_univ = ds.df_universe
	df_multi = ds.df_multiverse

	df_test_ensemble = df_test[["key_sale", "key"]].copy()
	df_sales_ensemble = df_sales[["key_sale", "key"]].copy()
	df_univ_ensemble = df_univ[["key"]].copy()

	if df_multi is not None:
		df_multi_ensemble = ds.df_multiverse[["key"]].copy()
	else:
		df_multi_ensemble = None

	if len(ensemble_list) == 0:
		ensemble_list = [key for key in all_results.model_results.keys()]
	timing.stop("setup")

	timing.start("parameter_search")
	timing.stop("parameter_search")
	timing.start("train")
	for m_key in ensemble_list:
		m_results = all_results.model_results[m_key]
		_df_test = m_results.df_test[["key_sale"]].copy()
		_df_test.loc[:, m_key] = m_results.pred_test.y_pred
		_df_sales = m_results.df_sales[["key_sale"]].copy()
		_df_sales.loc[:, m_key] = m_results.pred_sales.y_pred
		_df_univ = m_results.df_universe[["key"]].copy()
		_df_univ.loc[:, m_key] = m_results.pred_univ
		df_test_ensemble = df_test_ensemble.merge(_df_test, on="key_sale", how="left")
		df_sales_ensemble = df_sales_ensemble.merge(_df_sales, on="key_sale", how="left")
		df_univ_ensemble = df_univ_ensemble.merge(_df_univ, on="key", how="left")
		if df_multi is not None:
			_df_multi = m_results.df_multiverse[["key"]].copy()
			_df_multi.loc[:, m_key] = m_results.pred_multi
			df_multi_ensemble = df_multi_ensemble.merge(_df_multi, on="key", how="left")
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales_ensemble = df_sales_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_sales")

	timing.start("predict_univ")
	y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_univ")

	timing.start("predict_multi")
	if df_multi is not None:
		y_pred_multi_ensemble = df_multi_ensemble[ensemble_list].median(axis=1)
	else:
		y_pred_multi_ensemble = None
	timing.stop("predict_multi")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"ensemble",
		model="ensemble",
		y_pred_test=y_pred_test_ensemble.to_numpy(),
		y_pred_sales=y_pred_sales_ensemble.to_numpy(),
		y_pred_univ=y_pred_univ_ensemble.to_numpy(),
		timing=timing,
		verbose=verbose,
		y_pred_multi=y_pred_multi_ensemble.to_numpy() if y_pred_multi_ensemble is not None else None
	)
	timing.stop("total")

	dfs = {
		"sales": df_sales_ensemble,
		"universe": df_univ_ensemble,
		"test": df_test_ensemble,
	}
	if df_multi_ensemble is not None:
		dfs["multiverse"] = df_multi_ensemble

	_write_ensemble_model_results(results, outpath, settings, dfs, ensemble_list)

	return results


def _prepare_ds(
		df_sales: pd.DataFrame,
		df_universe: pd.DataFrame,
		model_group: str,
		vacant_only: bool,
		settings: dict
):
	"""
  Prepare a DataSplit object for modeling.

  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame
  :param model_group: Model group identifier.
  :type model_group: str
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: A DataSplit object.
  :rtype: DataSplit
  """
	s = settings
	s_model = s.get("modeling", {})
	vacant_status = "vacant" if vacant_only else "main"
	model_entries = s_model.get("models", {}).get(vacant_status, {})
	entry: dict | None = model_entries.get("model", model_entries.get("default", {}))

	ind_vars: list | None = entry.get("ind_vars", None)
	if ind_vars is None:
		raise ValueError(f"ind_vars not found for model 'default'")

	fields_cat = get_fields_categorical(s, df_sales)
	interactions = get_variable_interactions(entry, s, df_sales)

	instructions = s.get("modeling", {}).get("instructions", {})
	dep_var = instructions.get("dep_var", "sale_price")
	dep_var_test = instructions.get("dep_var_test", "sale_price_time_adj")

	test_keys, train_keys = _read_split_keys(model_group)

	ds = DataSplit(
		df_sales=df_sales,
		df_universe=df_universe,
		model_group=model_group,
		settings=settings,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		ind_vars=ind_vars,
		categorical_vars=fields_cat,
		interactions=interactions,
		test_keys=test_keys,
		train_keys=train_keys,
		vacant_only=vacant_only
	)
	return ds


def _calc_variable_recommendations(
		ds: DataSplit,
		settings: dict,
		correlation_results: dict,
		enr_results: dict,
		r2_values_results: pd.DataFrame,
		p_values_results: dict,
		t_values_results: dict,
		vif_results: dict,
		report: MarkdownReport = None
):
	"""
  Calculate variable recommendations based on various statistical metrics.

  :param ds: DataSplit object containing the data.
  :type ds: DataSplit
  :param settings: Settings dictionary.
  :type settings: dict
  :param correlation_results: Correlation analysis results.
  :type correlation_results: dict
  :param enr_results: Elastic net regularization results.
  :type enr_results: dict
  :param r2_values_results: R² values DataFrame.
  :type r2_values_results: pandas.DataFrame
  :param p_values_results: P-value analysis results.
  :type p_values_results: dict
  :param t_values_results: T-value analysis results.
  :type t_values_results: dict
  :param vif_results: VIF analysis results.
  :type vif_results: dict
  :param report: Optional MarkdownReport object.
  :type report: MarkdownReport or None
  :returns: DataFrame with variable recommendations.
  :rtype: pandas.DataFrame
  """
	feature_selection = settings.get("modeling", {}).get("instructions", {}).get("feature_selection", {})
	thresh = feature_selection.get("thresholds", {})
	weights = feature_selection.get("weights", {})

	df = pd.merge(correlation_results["final"], enr_results["final"], on="variable", how="outer")
	df = pd.merge(df, r2_values_results, on="variable", how="outer")
	df = pd.merge(df, p_values_results["final"], on="variable", how="outer")
	df = pd.merge(df, t_values_results["final"], on="variable", how="outer")
	df = pd.merge(df, vif_results["final"], on="variable", how="outer")

	df["weighted_score"] = 0

	# remove "const" from df:
	df = df[df["variable"].ne("const")]

	adj_r2_thresh = thresh.get("adj_r2", 0.1)
	df.loc[df["adj_r2"].gt(adj_r2_thresh), "weighted_score"] += 1

	weight_corr_score = weights.get("corr_score", 1)
	weight_enr_coef = weights.get("enr_coef", 1)
	weight_p_value = weights.get("p_value", 1)
	weight_t_value = weights.get("t_value", 1)
	weight_vif = weights.get("vif", 1)
	weight_coef_sign = weights.get("coef_sign", 1)

	df.loc[df["corr_score"].notna(), "weighted_score"] += weight_corr_score
	df.loc[df["enr_coef"].notna(), "weighted_score"] += weight_enr_coef
	df.loc[df["p_value"].notna(), "weighted_score"] += weight_p_value
	df.loc[df["t_value"].notna(), "weighted_score"] += weight_t_value
	df.loc[df["vif"].notna(), "weighted_score"] += weight_vif

	# check if "enr_coefficient", "t_value", and "coef_sign" are pointing in the same direction:
	df.loc[
		df["enr_coef_sign"].eq(df["t_value_sign"]) &
		df["enr_coef_sign"].eq(df["coef_sign"]),
		"signs_match"
	] = 1

	df.loc[df["signs_match"].eq(1), "weighted_score"] += weight_coef_sign
	df = df.sort_values(by="weighted_score", ascending=False)

	if report is not None:
		dfr = df.copy()
		dfr = dfr.rename(columns={
			"variable": "Variable",
			"corr_score": "Correlation",
			"enr_coef": "ENR",
			"adj_r2": "R-squared",
			"p_value": "P Value",
			"t_value": "T Value",
			"vif": "VIF",
			"signs_match": "Coef. sign",
			"weighted_score": "Weighted Score"
		})

		# Correlation:
		thresh_corr = thresh.get("correlation", 0.1)
		report.set_var("thresh_corr", thresh_corr, ".2f")
		corr_fields = ["variable", "corr_strength", "corr_clarity", "corr_score"]
		corr_renames = {
			"variable": "Variable",
			"corr_strength": "Strength",
			"corr_clarity": "Clarity",
			"corr_score": "Score"
		}

		# VIF:
		thresh_vif = thresh.get("vif", 10)
		vif_renames = {
			"variable": "Variable",
			"vif": "VIF"
		}

		# P-value:
		thresh_p_value = thresh.get("p_value", 0.05)
		p_value_renames = {
			"variable": "Variable",
			"p_value": "P-value"
		}

		# T-value:
		thresh_t_value = thresh.get("t_value", 2)
		t_value_renames = {
			"variable": "Variable",
			"t_value": "T-value"
		}

		# ENR:
		thresh_enr = thresh.get("enr", 0.1)
		enr_renames = {
			"variable": "Variable",
			"enr_coef": "Coefficient"
		}

		# R-Squared:
		thresh_r2 = thresh.get("adj_r2", 0.1)
		r2_renames = {
			"variable": "Variable",
			"adj_r2": "R-squared"
		}

		# Coef signs:
		coef_sign_renames = {
			"variable": "Variable",
			"enr_coef_sign": "ENR sign",
			"t_value_sign": "T-value sign",
			"coef_sign": "Coef. sign"
		}

		for state in ["initial", "final"]:
			# Correlation:
			dfr_corr = correlation_results[state][corr_fields].copy()
			dfr_corr["Pass/Fail"] = dfr_corr["corr_score"].apply(lambda x: "✅" if x > thresh_corr else "❌")
			for field in corr_fields:
				if field == "variable":
					continue
				if field not in dfr_corr:
					print("missing field", field)
				dfr_corr[field] = dfr_corr[field].apply(lambda x: f"{x:.2f}").astype("string")

			dfr_corr = dfr_corr.rename(columns=corr_renames)
			dfr_corr["Rank"] = range(1, len(dfr_corr) + 1)
			dfr_corr = dfr_corr[["Rank", "Variable", "Strength", "Clarity", "Score", "Pass/Fail"]]
			dfr_corr.set_index("Rank", inplace=True)
			dfr_corr = apply_dd_to_df_rows(dfr_corr, "Variable", settings, ds.one_hot_descendants)
			report.set_var(f"table_corr_{state}", dataframe_to_markdown(dfr_corr))

			# TODO: refactor this down to DRY it out a bit

			# VIF:
			dfr_vif = vif_results[state][["variable", "vif"]].copy()
			dfr_vif = dfr_vif.sort_values(by="vif", ascending=True)
			dfr_vif["Pass/Fail"] = dfr_vif["vif"].apply(lambda x: "✅" if x < thresh_vif else "❌")
			dfr_vif["vif"] = dfr_vif["vif"].apply(lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:,.0f}").astype("string")
			dfr_vif = dfr_vif.rename(columns=vif_renames)
			dfr_vif["Rank"] = range(1, len(dfr_vif) + 1)
			dfr_vif = dfr_vif[["Rank", "Variable", "VIF", "Pass/Fail"]]
			dfr_vif.set_index("Rank", inplace=True)
			dfr_vif = apply_dd_to_df_rows(dfr_vif, "Variable", settings, ds.one_hot_descendants)
			report.set_var(f"table_vif_{state}", dataframe_to_markdown(dfr_vif))

			# P-value:
			dfr_p_value = p_values_results[state][["variable", "p_value"]].copy()
			dfr_p_value = dfr_p_value[dfr_p_value["variable"].ne("const")]
			dfr_p_value = dfr_p_value.sort_values(by="p_value", ascending=True)
			dfr_p_value["Pass/Fail"] = dfr_p_value["p_value"].apply(lambda x: "✅" if x < thresh_p_value else "❌")
			dfr_p_value["p_value"] = dfr_p_value["p_value"].apply(lambda x: f"{x:.3f}").astype("string")
			dfr_p_value = dfr_p_value.rename(columns=p_value_renames)
			dfr_p_value["Rank"] = range(1, len(dfr_p_value) + 1)
			dfr_p_value = dfr_p_value[["Rank", "Variable", "P-value", "Pass/Fail"]]
			dfr_p_value.set_index("Rank", inplace=True)
			dfr_p_value = apply_dd_to_df_rows(dfr_p_value, "Variable", settings, ds.one_hot_descendants)
			report.set_var(f"table_p_value_{state}", dataframe_to_markdown(dfr_p_value))

			# T-value:
			dfr_t_value = t_values_results[state][["variable", "t_value"]].copy()
			dfr_t_value = dfr_t_value[dfr_t_value["variable"].ne("const")]
			dfr_t_value = dfr_t_value.sort_values(by="t_value", ascending=False, key=abs)
			dfr_t_value["Pass/Fail"] = dfr_t_value["t_value"].apply(lambda x: "✅" if abs(x) > thresh_t_value else "❌")
			dfr_t_value["t_value"] = dfr_t_value["t_value"].apply(lambda x: f"{x:.2f}").astype("string")
			dfr_t_value = dfr_t_value.rename(columns=t_value_renames)
			dfr_t_value["Rank"] = range(1, len(dfr_t_value) + 1)
			dfr_t_value = dfr_t_value[["Rank", "Variable", "T-value", "Pass/Fail"]]
			dfr_t_value.set_index("Rank", inplace=True)
			dfr_t_value = apply_dd_to_df_rows(dfr_t_value, "Variable", settings, ds.one_hot_descendants)
			report.set_var(f"table_t_value_{state}", dataframe_to_markdown(dfr_t_value))

			# ENR:
			dfr_enr = enr_results[state][["variable", "enr_coef"]].copy()
			dfr_enr = dfr_enr.sort_values(by="enr_coef", ascending=False, key=abs)
			dfr_enr["Pass/Fail"] = dfr_enr["enr_coef"].apply(lambda x: "✅" if abs(x) > thresh_enr else "❌")
			dfr_enr["enr_coef"] = dfr_enr["enr_coef"].apply(lambda x: f"{x:.2f}" if abs(x) < 100 else f"{x:,.0f}").astype("string")
			dfr_enr = dfr_enr.rename(columns=enr_renames)
			dfr_enr["Rank"] = range(1, len(dfr_enr) + 1)
			dfr_enr = dfr_enr[["Rank", "Variable", "Coefficient", "Pass/Fail"]]
			dfr_enr.set_index("Rank", inplace=True)
			dfr_enr = apply_dd_to_df_rows(dfr_enr, "Variable", settings, ds.one_hot_descendants)
			report.set_var(f"table_enr_{state}", dataframe_to_markdown(dfr_enr))

			# R-squared
			dfr_r2 = r2_values_results.copy()
			dfr_r2 = dfr_r2.sort_values(by="adj_r2", ascending=False)
			dfr_r2["Pass/Fail"] = dfr_r2["adj_r2"].apply(lambda x: "✅" if x > thresh_r2 else "❌")
			dfr_r2["adj_r2"] = dfr_r2["adj_r2"].apply(lambda x: f"{x:.2f}").astype("string")
			dfr_r2 = dfr_r2.rename(columns=r2_renames)
			dfr_r2["Rank"] = range(1, len(dfr_r2) + 1)
			dfr_r2 = dfr_r2[["Rank", "Variable", "R-squared", "Pass/Fail"]]
			dfr_r2.set_index("Rank", inplace=True)
			dfr_r2 = apply_dd_to_df_rows(dfr_r2, "Variable", settings, ds.one_hot_descendants)
			if state == "final":
				dfr_r2 = dfr_r2[dfr_r2["Pass/Fail"].eq("✅")]
			report.set_var(f"table_adj_r2_{state}", dataframe_to_markdown(dfr_r2))

			# Coef sign:
			dfr_coef_sign = enr_results[state][["variable", "enr_coef_sign"]].copy()
			dfr_coef_sign = dfr_coef_sign.merge(t_values_results[state][["variable", "t_value_sign"]], on="variable", how="outer")
			dfr_coef_sign = dfr_coef_sign.merge(r2_values_results[["variable", "coef_sign"]], on="variable", how="outer")
			dfr_coef_sign["signs_match"] = False
			dfr_coef_sign.loc[
				dfr_coef_sign["enr_coef_sign"].eq(dfr_coef_sign["t_value_sign"]) &
				dfr_coef_sign["enr_coef_sign"].eq(dfr_coef_sign["coef_sign"]),
				"signs_match"
			] = True
			dfr_coef_sign["Pass/Fail"] = dfr_coef_sign["signs_match"].apply(lambda x: "✅" if x else "❌")
			dfr_coef_sign = dfr_coef_sign.sort_values(by="signs_match", ascending=False)
			dfr_coef_sign = dfr_coef_sign[dfr_coef_sign["variable"].ne("const")]
			dfr_coef_sign = dfr_coef_sign.rename(columns=coef_sign_renames)
			dfr_coef_sign = dfr_coef_sign[["Variable", "ENR sign", "T-value sign", "Coef. sign", "Pass/Fail"]]
			for field in ["ENR sign", "T-value sign", "Coef. sign"]:
				dfr_coef_sign[field] = dfr_coef_sign[field].apply(lambda x: f"{x:.0f}").astype("string")
			dfr_coef_sign = apply_dd_to_df_rows(dfr_coef_sign, "Variable", settings, ds.one_hot_descendants)
			if state == "final":
				dfr_coef_sign = dfr_coef_sign[dfr_coef_sign["Pass/Fail"].eq("✅")]
			report.set_var(f"table_coef_sign_{state}", dataframe_to_markdown(dfr_coef_sign))


		dfr["Rank"] = range(1, len(dfr) + 1)
		dfr = apply_dd_to_df_rows(dfr, "Variable", settings, ds.one_hot_descendants)

		dfr = dfr[["Rank", "Weighted Score", "Variable", "VIF", "P Value", "T Value", "ENR", "Correlation", "Coef. sign", "R-squared"]]
		dfr.set_index("Rank", inplace=True)
		for col in dfr.columns:
			if col == "R-squared":
				dfr[col] = dfr[col].apply(lambda x: "✅" if x > adj_r2_thresh else "❌")
			elif col == "Coef. sign":
				dfr[col] = dfr[col].apply(lambda x: "✅" if x == 1 else "❌")
			elif col not in ["Rank", "Weighted Score", "Variable"]:
				dfr[col] = dfr[col].apply(lambda x: "✅" if not pd.isna(x) else "❌")
		report.set_var("pre_model_table", dfr.to_markdown())

	return df


def _run_hedonic_models(
		settings: dict,
		model_group: str,
		vacant_only: bool,
		models_to_run: list[str],
		all_results: MultiModelResults,
		df_sales: pd.DataFrame,
		df_universe: pd.DataFrame,
		dep_var: str,
		dep_var_test: str,
		fields_cat: list[str],
		use_saved_results: bool = True,
		verbose: bool = False,
		df_multiverse: pd.DataFrame = None
):
	"""
  Run hedonic models and ensemble them, then update the benchmark.

  :param settings: Settings dictionary.
  :type settings: dict
  :param model_group: Model group identifier.
  :type model_group: str
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool
  :param models_to_run: List of models to run.
  :type models_to_run: list[str]
  :param all_results: MultiModelResults containing current model results.
  :type all_results: MultiModelResults
  :param df_sales: Sales DataFrame.
  :type df_sales: pandas.DataFrame
  :param df_universe: Universe DataFrame.
  :type df_universe: pandas.DataFrame
  :param dep_var: Dependent variable for training.
  :type dep_var: str
  :param dep_var_test: Dependent variable for testing.
  :type dep_var_test: str
  :param fields_cat: List of categorical fields.
  :type fields_cat: list[str]
  :param use_saved_results: Whether to use saved results if available.
  :type use_saved_results: bool, optional
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :param df_multiverse: Optional multiverse DataFrame.
  :type df_multiverse: pandas.DataFrame or None
  :returns: None
  """
	hedonic_results = {}
	# Run hedonic models
	outpath = f"out/models/{model_group}/hedonic"
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	location_field_neighborhood = get_important_field(settings, "loc_neighborhood", df_sales)
	location_field_market_area = get_important_field(settings, "loc_market_area", df_sales)
	location_fields = [location_field_neighborhood, location_field_market_area]

	# Re-run the models one by one and stash the results
	for model in models_to_run:
		if model not in all_results.model_results:
			continue
		smr = all_results.model_results[model]
		ds = get_data_split_for(
			name=model,
			model_group=model_group,
			location_fields=location_fields,
			ind_vars=smr.ind_vars,
			df_sales=df_sales,
			df_universe=df_universe,
			settings=settings,
			dep_var=dep_var,
			dep_var_test=dep_var_test,
			fields_cat=fields_cat,
			interactions=smr.ds.interactions.copy(),
			test_keys=smr.ds.test_keys,
			train_keys=smr.ds.train_keys,
			vacant_only=False,
			hedonic=True,
			hedonic_test_against_vacant_sales=True,
			df_multiverse=df_multiverse
		)
		# We call this here because we are re-running prediction without first calling run(), which would call this
		ds.split()
		if len(ds.y_sales) < 15:
			print(f"Skipping hedonic model because there are not enough sale records....")
			return
		smr.ds = ds
		results = _predict_one_model(
			smr=smr,
			model=model,
			outpath=outpath,
			settings=settings,
			use_saved_results=use_saved_results,
			verbose=verbose
		)
		if results is not None:
			hedonic_results[model] = results

	all_hedonic_results = MultiModelResults(
		model_results=hedonic_results,
		benchmark=_calc_benchmark(hedonic_results)
	)
	best_ensemble = _optimize_ensemble(
		df_sales=df_sales,
		df_universe=df_universe,
		model_group=model_group,
		vacant_only=vacant_only,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		all_results=all_hedonic_results,
		settings=settings,
		verbose=verbose,
		hedonic=True
	)
	# Run the ensemble model
	ensemble_results = _run_ensemble(
		df_sales=df_sales,
		df_universe=df_universe,
		model_group=model_group,
		vacant_only=vacant_only,
		hedonic=True,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		outpath=outpath,
		ensemble_list=best_ensemble,
		all_results=all_results,
		settings=settings,
		verbose=verbose,
		df_multiverse=df_multiverse
	)

	out_pickle = f"{outpath}/model_ensemble.pickle"
	with open(out_pickle, "wb") as file:
		pickle.dump(ensemble_results, file)

	# Calculate final results, including ensemble
	print("HEDONIC BENCHMARK")
	all_hedonic_results.add_model("ensemble", ensemble_results)
	print(all_hedonic_results.benchmark.print())


def _run_models(
		sup: SalesUniversePair,
		model_group: str,
		settings: dict,
		vacant_only: bool = False,
		save_params: bool = True,
		use_saved_params: bool = True,
		save_results: bool = False,
		use_saved_results: bool = True,
		verbose: bool = False,
		run_hedonic: bool = True
):
	"""
  Run models for a given model group and process ensemble results.

  :param sup: Sales and universe data.
  :type sup: SalesUniversePair
  :param model_group: Model group identifier.
  :type model_group: str
  :param settings: Settings dictionary.
  :type settings: dict
  :param vacant_only: Whether to use only vacant sales.
  :type vacant_only: bool, optional
  :param save_params: Whether to save model parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to use saved parameters.
  :type use_saved_params: bool, optional
  :param save_results: Whether to save results.
  :type save_results: bool
  :param use_saved_results: Whether to use saved results if available.
  :type use_saved_results: bool, optional
  :param verbose: If True, prints additional information.
  :type verbose: bool, optional
  :returns: MultiModelResults containing all models and the final ensemble.
  :rtype: MultiModelResults
  """
	df_univ = sup["universe"]
	df_sales = get_hydrated_sales_from_sup(sup)
	df_multi = df_univ.copy()

	df_sales = df_sales[df_sales["model_group"].eq(model_group)].copy()
	df_univ = df_univ[df_univ["model_group"].eq(model_group)].copy()

	s = settings
	s_model = s.get("modeling", {})
	s_inst = s_model.get("instructions", {})
	vacant_status = "vacant" if vacant_only else "main"

	dep_var = s_inst.get("dep_var", "sale_price")
	dep_var_test = s_inst.get("dep_var_test", "sale_price_time_adj")
	fields_cat = get_fields_categorical(s, df_univ)
	models_to_run = s_inst.get(vacant_status, {}).get("run", None)
	model_entries = s_model.get("models").get(vacant_status, {})
	if models_to_run is None:
		models_to_run = list(model_entries.keys())

	# Enforce that horizontal equity cluster ID's have already been calculated
	if "he_id" not in df_univ:
		raise ValueError("Could not find equity cluster ID's in the dataframe (he_id)")

	model_results = {}
	outpath = f"out/models/{model_group}/{vacant_status}"
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	df_sales_count = get_sales(df_sales, settings, vacant_only, df_univ)

	if len(df_sales_count) == 0:
		print(f"No sales records found for model_group: {model_group}, vacant_only: {vacant_only}. Skipping...")
		return

	if len(df_sales_count) < 15:
		warnings.warn(f"For model_group: {model_group}, vacant_only: {vacant_only}, there are fewer than 15 sales records. Model might not be any good!")

	var_recs = get_variable_recommendations(
		df_sales,
		df_univ,
		vacant_only,
		settings,
		model_group,
		verbose=True,
	)
	best_variables = var_recs["variables"]
	var_report = var_recs["report"]
	var_report_md = var_report.render()

	os.makedirs(f"{outpath}/reports", exist_ok=True)
	with open(f"{outpath}/reports/variable_report.md", "w", encoding="utf-8") as f:
		f.write(var_report_md)

	pdf_path = f"{outpath}/reports/variable_report.pdf"
	formats = settings.get("analysis", {}).get("report", {}).get("formats", None)
	_markdown_to_pdf(var_report_md, pdf_path, css_file="variable", formats=formats)

	any_results = False

	# Run the models one by one and stash the results
	for model in models_to_run:
		results = run_one_model(
			df_multiverse=df_multi,
			df_sales=df_sales,
			df_universe=df_univ,
			vacant_only=vacant_only,
			model_group=model_group,
			model=model,
			model_entries=model_entries,
			settings=settings,
			dep_var=dep_var,
			dep_var_test=dep_var_test,
			best_variables=best_variables,
			fields_cat=fields_cat,
			outpath=outpath,
			save_params=save_params,
			use_saved_params=use_saved_params,
			save_results=save_results,
			use_saved_results=use_saved_results,
			verbose=verbose
		)
		if results is not None:
			model_results[model] = results
			any_results = True
		else:
			print(f"Could not generate results for model: {model}")

	if not any_results:
		print(f"No results generated for model_group: {model_group}, vacant_only: {vacant_only}. Skipping...")
		return

	# Calculate initial results (ensemble will use them)
	all_results = MultiModelResults(
		model_results=model_results,
		benchmark=_calc_benchmark(model_results)
	)

	best_ensemble = _optimize_ensemble(
		df_sales=df_sales,
		df_universe=df_univ,
		model_group=model_group,
		vacant_only=vacant_only,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		all_results=all_results,
		settings=settings,
		verbose=verbose
	)

	# Run the ensemble model
	ensemble_results = _run_ensemble(
		df_sales=df_sales,
		df_universe=df_univ,
		model_group=model_group,
		vacant_only=vacant_only,
		hedonic=False,
		dep_var=dep_var,
		dep_var_test=dep_var_test,
		outpath=outpath,
		ensemble_list=best_ensemble,
		all_results=all_results,
		settings=settings,
		verbose=verbose,
		df_multiverse=df_multi
	)

	out_pickle = f"{outpath}/model_ensemble.pickle"
	with open(out_pickle, "wb") as file:
		pickle.dump(ensemble_results, file)

	# Calculate final results, including ensemble
	all_results.add_model("ensemble", ensemble_results)

	print("")
	if vacant_only:
		print(f"VACANT BENCHMARK")
	else:
		print(f"MAIN BENCHMARK")
	print(all_results.benchmark.print())

	if not vacant_only and run_hedonic:
		_run_hedonic_models(
			settings=settings,
			model_group=model_group,
			vacant_only=vacant_only,
			models_to_run=models_to_run,
			all_results=all_results,
			df_sales=df_sales,
			df_universe=df_univ,
			dep_var=dep_var,
			dep_var_test=dep_var_test,
			fields_cat=fields_cat,
			use_saved_results=use_saved_results,
			verbose=verbose,
			df_multiverse=df_multi
		)

	return all_results
