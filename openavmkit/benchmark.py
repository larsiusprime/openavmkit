import os
import pickle

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import Booster
from statsmodels.nonparametric.kernel_regression import KernelReg
from xgboost import XGBRegressor

from openavmkit.data import get_important_field, get_locations, read_split_keys
from openavmkit.modeling import run_mra, run_gwr, run_xgboost, run_lightgbm, run_catboost, SingleModelResults, \
	run_garbage, \
	run_average, run_naive_sqft, DataSplit, run_kernel, run_local_sqft, run_assessor, predict_garbage, \
	GarbageModel, predict_average, AverageModel, predict_naive_sqft, predict_local_sqft, predict_assessor, predict_kernel, \
	predict_gwr, predict_xgboost, predict_catboost, predict_lightgbm
from openavmkit.reports import MarkdownReport, markdown_to_pdf
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import div_z_safe, dataframe_to_markdown
from openavmkit.utilities.format import fancy_format
from openavmkit.utilities.modeling import NaiveSqftModel, LocalSqftModel, AssessorModel, GWRModel, MRAModel
from openavmkit.utilities.settings import get_fields_categorical, get_variable_interactions, get_valuation_date, \
	get_modeling_group, apply_dd_to_df_rows
from openavmkit.utilities.stats import calc_vif_recursive_drop, calc_t_values_recursive_drop, \
	calc_p_values_recursive_drop, calc_elastic_net_regularization, calc_correlations, calc_r2, \
	calc_cross_validation_score, calc_cod
from openavmkit.utilities.timing import TimingData


class BenchmarkResults:
	df_time: pd.DataFrame
	df_stats_test: pd.DataFrame
	df_stats_full: pd.DataFrame

	def __init__(self, df_time: pd.DataFrame, df_stats_test: pd.DataFrame, df_stats_full: pd.DataFrame):
		self.df_time = df_time
		self.df_stats_test = df_stats_test
		self.df_stats_full = df_stats_full


	def print(self) -> str:
		result = "Timings:\n"
		result += format_benchmark_df(self.df_time)
		result += "\n\n"
		result += "Test set:\n"
		result += format_benchmark_df(self.df_stats_test)
		result += "\n\n"
		result += "Universe set:\n"
		result += format_benchmark_df(self.df_stats_full)
		result += "\n\n"
		return result


class MultiModelResults:
	model_results: dict[str, SingleModelResults]
	benchmark: BenchmarkResults

	def __init__(
			self,
			model_results: dict[str, SingleModelResults],
			benchmark: BenchmarkResults
	):
		self.model_results = model_results
		self.benchmark = benchmark


	def add_model(
			self,
			model: str,
			results: SingleModelResults
	):
		self.model_results[model] = results
		# recalculate the benchmark
		self.benchmark = _calc_benchmark(self.model_results)


def _calc_benchmark(model_results: dict[str, SingleModelResults]):
	data_time = {
		"model": [],
		"total": [],
		"param": [],
		"train": [],
		"test": [],
		"univ": [],
		"chd": [],
	}

	data = {
		"model":[],
		"subset":[],
		"utility_score": [],
		"count_sales":[],
		"count_univ":[],
		"mse":[],
		"rmse":[],
		"r2":[],
		"adj_r2":[],
		"median_ratio":[],
		"cod":[],
		"cod_trim":[],
		"prd":[],
		"prb":[],
		"chd":[]
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
				data_time["chd"].append(tim.get("chd"))

			data["chd"].append(chd_results)

	df = pd.DataFrame(data)

	df_test = df[df["subset"].eq("Test set")].drop(columns=["subset"])
	df_full = df[df["subset"].eq("Universe set")].drop(columns=["subset"])
	df_time = pd.DataFrame(data_time)

	# set index to the model column:
	df_test.set_index("model", inplace=True)
	df_full.set_index("model", inplace=True)
	df_time.set_index("model", inplace=True)

	results = BenchmarkResults(
		df_time,
		df_test,
		df_full
	)

	return results

def format_benchmark_df(df: pd.DataFrame):

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
			# check if formats[col] is a function or a string
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

	model_name = model

	out_pickle = f"{outpath}/model_{model_name}.pickle"
	if use_saved_results and os.path.exists(out_pickle):
		with open(out_pickle, "rb") as file:
			results = pickle.load(file)
		return results

	ds = smr.ds

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
	elif model_name == "local_naive_sqft":
		sqft_model: LocalSqftModel = smr.model
		results = predict_local_sqft(ds, sqft_model, timing, verbose)
	elif model_name == "local_smart_sqft":
		sqft_model: LocalSqftModel = smr.model
		results = predict_local_sqft(ds, sqft_model, timing, verbose)
	elif model_name == "assessor":
		assr_model: AssessorModel = smr.model
		results = predict_assessor(ds, assr_model, timing, verbose)
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

	# write out the results:
	write_model_results(results, outpath, settings)

	with open(out_pickle, "wb") as file:
		pickle.dump(results, file)

	return results


def _get_data_split_for(
		name: str,
		model_group: str,
		location_fields: list[str] | None,
		dep_vars: list[str],
		df: pd.DataFrame,
		settings: dict,
		ind_var: str,
		ind_var_test: str,
		fields_cat: list[str],
		interactions: dict,
		test_keys: list[str],
		train_keys: list[str],
		vacant_only: bool,
		hedonic: bool,
		df_multiverse: pd.DataFrame | None = None
):
	if name == "local_naive_sqft":
		_dep_vars = location_fields + ["bldg_area_finished_sqft", "land_area_sqft"]
	elif name == "local_smart_sqft":
		_dep_vars = ["ss_id"] + location_fields + ["bldg_area_finished_sqft", "land_area_sqft"]
	elif name == "assessor":
		if hedonic:
			_dep_vars = ["assr_land_value"]
		else:
			_dep_vars = ["assr_market_value"]
	else:
		_dep_vars = dep_vars

	return DataSplit(
		df,
		model_group,
		settings,
		ind_var,
		ind_var_test,
		_dep_vars,
		fields_cat,
		interactions,
		test_keys,
		train_keys,
		vacant_only=vacant_only,
		hedonic=hedonic,
		df_multiverse=df_multiverse
	)


def _run_one_model(
		df_multiverse: pd.DataFrame,
		df: pd.DataFrame,
		vacant_only: bool,
		model_group: str,
		model: str,
		model_entries: dict,
		settings: dict,
		ind_var: str,
		ind_var_test: str,
		best_variables: list[str],
		fields_cat: list[str],
		outpath: str,
		save_params: bool,
		use_saved_params: bool,
		use_saved_results: bool,
		verbose: bool = False,
		hedonic: bool = False
) -> SingleModelResults:

	model_name = model

	out_pickle = f"{outpath}/model_{model_name}.pickle"
	if use_saved_results and os.path.exists(out_pickle):
		with open(out_pickle, "rb") as file:
			results = pickle.load(file)
		return results

	entry: dict | None = model_entries.get(model, None)
	default_entry: dict | None = model_entries.get("default", None)
	if entry is None:
		entry = default_entry
		if entry is None:
			raise ValueError(f"Model entry for {model} not found, and there is no default entry!")

	# TODO: make this more elegant
	if "*" in model:
		sales_chase = 0.01
		model_name = model.replace("*", "")
	else:
		sales_chase = False

	if verbose:
		print(f" running model {model}...")

	are_dep_vars_default = entry.get("dep_vars", None) is None

	dep_vars : list | None = entry.get("dep_vars", default_entry.get("dep_vars", None))
	if dep_vars is None:
		raise ValueError(f"dep_vars not found for model {model}")

	if are_dep_vars_default:
		if verbose:
			if set(dep_vars) != set(best_variables):
				print(f"--> using default variables, auto-optimized variable list: {best_variables}")
		dep_vars = best_variables

	interactions = get_variable_interactions(entry, settings, df)

	location_fields = get_locations(settings, df)

	test_keys, train_keys = read_split_keys(model_group)

	ds = _get_data_split_for(
		model_name,
		model_group,
		location_fields,
		dep_vars,
		df,
		settings,
		ind_var,
		ind_var_test,
		fields_cat,
		interactions,
		test_keys,
		train_keys,
		vacant_only,
		hedonic,
		df_multiverse
	)

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
	elif model_name == "local_naive_sqft":
		results = run_local_sqft(ds, location_fields=location_fields, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "local_smart_sqft":
		results = run_local_sqft(ds, location_fields=["ss_id"] + location_fields, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "assessor":
		results = run_assessor(ds, verbose=verbose)
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

	# write out the results:
	write_model_results(results, outpath, settings)

	with open(out_pickle, "wb") as file:
		pickle.dump(results, file)

	return results


def _assemble_model_results(results: SingleModelResults, settings: dict):
	locations = get_locations(settings)

	fields = ["key", "geometry", "prediction", "assr_market_value", "assr_market_land_value", "sale_price", "sale_price_time_adj", "sale_date"] + locations
	fields = [field for field in fields if field in results.df_sales.columns]

	dfs = {
		"sales": results.df_sales[fields].copy(),
		"universe": results.df_universe[fields].copy(),
		"test": results.df_test[fields].copy()
	}

	if results.df_multiverse is not None:
		dfs["multiverse"] = results.df_multiverse[fields].copy()

	for key in dfs:
		df = dfs[key]
		df["prediction_ratio"] = div_z_safe(df, "prediction", "sale_price_time_adj")
		df["assr_ratio"] = div_z_safe(df, "assr_market_value", "sale_price_time_adj")
		for location in locations:
			if location in df:
				df[f"prediction_cod_{location}"] = None
				df[f"assr_cod_{location}"] = None
				location_values = df[location].unique()
				for value in location_values:
					predictions = df.loc[df[location].eq(value), "prediction_ratio"].values
					predictions = predictions[~pd.isna(predictions)]
					df.loc[df[location].eq(value), f"prediction_cod_{location}"] = calc_cod(predictions)

					assr_predictions = df.loc[df[location].eq(value), "assr_ratio"].values
					assr_predictions = assr_predictions[~pd.isna(assr_predictions)]
					df.loc[df[location].eq(value), f"assr_cod_{location}"] = calc_cod(assr_predictions)
	return dfs


def write_model_results(results: SingleModelResults, outpath: str, settings: dict):
	dfs = _assemble_model_results(results, settings)
	path = f"{outpath}/{results.type}"
	if "*" in path:
		path = path.replace("*", "_star")
	os.makedirs(path, exist_ok=True)
	for key in dfs:
		df = dfs[key]
		df.to_parquet(f"{path}/pred_{key}.parquet")
		df.to_csv(f"{path}/pred_{key}.csv", index=False)


def write_ensemble_model_results(
		results: SingleModelResults,
		outpath: str,
		settings: dict,
		dfs: dict[str, pd.DataFrame],
		ensemble_list: list[str]
):
	dfs_basic = _assemble_model_results(results, settings)
	path = f"{outpath}/{results.type}"
	os.makedirs(path, exist_ok=True)
	for key in dfs_basic:
		df_basic = dfs_basic[key]
		df_ensemble = dfs[key]
		df_ensemble = df_ensemble[["key"] + ensemble_list]
		df = df_basic.merge(df_ensemble, on="key", how="left")
		df.to_parquet(f"{path}/pred_ensemble_{key}.parquet")
		df.to_csv(f"{path}/pred_ensemble_{key}.csv", index=False)


def optimize_ensemble_allocation(
		df: pd.DataFrame | None,
		model_group: str,
		vacant_only: bool,
		ind_var: str,
		ind_var_test: str,
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False,
		hedonic: bool = False,
		ensemble_list: list[str] = None
):
	timing = TimingData()
	timing.start("total")
	timing.start("setup")

	if df is None:
		# get first key from all_results.model_results:
		first_key = list(all_results.model_results.keys())[0]
		# get the universe dataframe from the first model:
		df = all_results.model_results[first_key].ds.df_universe_orig

	test_keys, train_keys = read_split_keys(model_group)

	ds = DataSplit(
		df,
		model_group,
		settings,
		ind_var,
		ind_var_test,
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

	# Never use an assessor's model in an ensemble!
	if "assessor" in ensemble_list:
		ensemble_list.remove("assessor")

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
	df_test_ensemble = df_test[["key"]].copy()
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


def optimize_ensemble(
		df: pd.DataFrame | None,
		model_group: str,
		vacant_only: bool,
		ind_var: str,
		ind_var_test: str,
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False,
		hedonic: bool = False,
		ensemble_list: list[str] = None
):
	timing = TimingData()
	timing.start("total")
	timing.start("setup")

	if df is None:
		# get first key from all_results.model_results:
		first_key = list(all_results.model_results.keys())[0]
		# get the universe dataframe from the first model:
		df = all_results.model_results[first_key].ds.df_universe_orig

	test_keys, train_keys = read_split_keys(model_group)

	ds = DataSplit(
		df,
		model_group,
		settings,
		ind_var,
		ind_var_test,
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

	# Never use an assessor's model in an ensemble!
	if "assessor" in ensemble_list:
		ensemble_list.remove("assessor")

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
	df_test_ensemble = df_test[["key"]].copy()
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

	if score < best_score and len(ensemble_list) >= 3:
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


def run_ensemble(
		df: pd.DataFrame,
		model_group: str,
		vacant_only: bool,
		hedonic: bool,
		ind_var: str,
		ind_var_test: str,
		outpath: str,
		ensemble_list: list[str],
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False,
		df_multiverse: pd.DataFrame = None
):
	timing = TimingData()

	timing.start("total")

	timing.start("setup")

	test_keys, train_keys = read_split_keys(model_group)

	ds = DataSplit(
		df,
		model_group,
		settings,
		ind_var,
		ind_var_test,
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

	df_test_ensemble = df_test[["key"]].copy()
	df_sales_ensemble = df_sales[["key"]].copy()
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

		_df_test = m_results.df_test[["key"]].copy()
		_df_test.loc[:, m_key] = m_results.pred_test.y_pred

		_df_sales = m_results.df_sales[["key"]].copy()
		_df_sales.loc[:, m_key] = m_results.pred_sales.y_pred

		_df_univ = m_results.df_universe[["key"]].copy()
		_df_univ.loc[:, m_key] = m_results.pred_univ

		df_test_ensemble = df_test_ensemble.merge(_df_test, on="key", how="left")
		df_sales_ensemble = df_sales_ensemble.merge(_df_sales, on="key", how="left")
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

	write_ensemble_model_results(results, outpath, settings, dfs, ensemble_list)
	return results


def _prepare_ds(
	df: pd.DataFrame,
	model_group: str,
	vacant_only: bool,
	settings: dict
):
	s = settings
	s_model = s.get("modeling", {})
	vacant_status = "vacant" if vacant_only else "main"
	model_entries = s_model.get("models", {}).get(vacant_status, {})
	entry: dict | None = model_entries.get("model", model_entries.get("default", {}))

	dep_vars : list | None = entry.get("dep_vars", None)
	if dep_vars is None:
		raise ValueError(f"dep_vars not found for model 'default'")

	fields_cat = get_fields_categorical(s, df)
	interactions = get_variable_interactions(entry, s, df)

	instructions = s.get("modeling", {}).get("instructions", {})
	ind_var = instructions.get("ind_var", "sale_price")
	ind_var_test = instructions.get("ind_var_test", "sale_price_time_adj")

	test_keys, train_keys = read_split_keys(model_group)

	ds = DataSplit(
		df,
		model_group,
		settings,
		ind_var,
		ind_var_test,
		dep_vars,
		fields_cat,
		interactions,
		test_keys,
		train_keys,
		vacant_only
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


def get_variable_recommendations(
		df: pd.DataFrame,
		vacant_only: bool,
		settings: dict,
		model: str,
		model_group: str,
		verbose: bool = False
):
	if verbose:
		print("")

	report = MarkdownReport("variables")

	df = enrich_time_adjustment(df, settings, verbose=verbose)
	ds = _prepare_ds(df, model_group, vacant_only, settings)
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()

	feature_selection = settings.get("modeling", {}).get("instructions", {}).get("feature_selection", {})
	thresh = feature_selection.get("thresholds", {})

	X_sales = ds.X_sales[ds.dep_vars]
	y_sales = ds.y_sales

	# Correlation
	X_corr = ds.df_sales[[ds.ind_var] + ds.dep_vars]
	corr_results = calc_correlations(X_corr, thresh.get("correlation", 0.1))

	# Elastic net regularization
	enr_coefs = calc_elastic_net_regularization(X_sales, y_sales, thresh.get("enr", 0.01))

	# R^2 values
	r2_values = calc_r2(ds.df_sales, ds.dep_vars, y_sales)

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
		# remove the variable from the dataframe:
		df_results = df_results[df_results["variable"].ne(worst_variable)]
		if verbose:
			print(f"--> score: {cv_score:,.0f}  {len(curr_variables)} variables: {curr_variables}")

	# make a table from the list of best variables:
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

	report = generate_variable_report(
		report,
		settings,
		model_group,
		best_variables
	)

	return {
		"variables": best_variables,
		"report": report
	}


def generate_variable_report(
		report: MarkdownReport,
		settings: dict,
		modeling_group: str,
		best_variables: list[str]
):

	locality = settings.get("locality", {})
	report.set_var("locality", locality.get("name", "...LOCALITY..."))

	mg = get_modeling_group(settings, modeling_group)
	report.set_var("val_date", get_valuation_date(settings).strftime("%Y-%m-%d"))
	report.set_var("modeling_group", mg.get("name", mg))


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

	# TODO: construct these
	#summary_table = "...SUMMARY TABLE..."
	#report.set_var("summary_table", summary_table)

	post_model_table = "...POST MODEL TABLE..."
	report.set_var("post_model_table", post_model_table)

	return report


def run_models(
		df: pd.DataFrame,
		settings: dict,
		save_params: bool = True,
		use_saved_params: bool = True,
		use_saved_results: bool = True,
		verbose: bool = False,
		run_main: bool = True,
		run_vacant: bool = True
):
	s = settings
	s_model = s.get("modeling", {})
	s_inst = s_model.get("instructions", {})
	model_groups = s_inst.get("model_groups", [])
	if len(model_groups) == 0:
		model_groups = df["model_group"].unique()
		model_groups = [mg for mg in model_groups if not pd.isna(mg) and str(mg) != "<NA>"]

	for model_group in model_groups:
		if verbose:
			print(f"*** Running models for model_group: {model_group} ***")
		for vacant_only in [False, True]:
			if vacant_only:
				if not run_vacant:
					continue
			else:
				if not run_main:
					continue
			_run_models(df, model_group, settings, vacant_only, save_params, use_saved_params, use_saved_results, verbose)

def _run_hedonic_models(
		settings: dict,
		model_group: str,
		vacant_only: bool,
		models_to_run: list[str],
		all_results: MultiModelResults,
		df: pd.DataFrame,
		ind_var: str,
		ind_var_test: str,
		fields_cat: list[str],
		use_saved_results: bool = True,
		verbose: bool = False,
		df_multiverse: pd.DataFrame = None
):
	hedonic_results = {}
	# Run hedonic models
	outpath = f"out/models/{model_group}/hedonic"
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	location_field_neighborhood = get_important_field(settings, "loc_neighborhood", df)
	location_field_market_area = get_important_field(settings, "loc_market_area", df)
	location_fields = [location_field_neighborhood, location_field_market_area]

	# Re-run the models one by one and stash the results
	for model in models_to_run:

		smr = all_results.model_results[model]
		ds = _get_data_split_for(
			model,
			model_group,
			location_fields,
			smr.dep_vars,
			df,
			settings,
			ind_var,
			ind_var_test,
			fields_cat,
			smr.ds.interactions.copy(),
			smr.ds.test_keys,
			smr.ds.train_keys,
			vacant_only=False,
			hedonic=True,
			df_multiverse=df_multiverse
		)

		# TODO: there is a bug here because the number of rows in df_test winds up different across models (224 vs 227)

		# We call this here because we are re-running prediction without first calling run(), which would call this
		ds.split()

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

	best_ensemble = optimize_ensemble(
		df=df,
		model_group=model_group,
		vacant_only=vacant_only,
		ind_var=ind_var,
		ind_var_test=ind_var_test,
		all_results=all_hedonic_results,
		settings=settings,
		verbose=verbose,
		hedonic=True
	)

	# Run the ensemble model
	ensemble_results = run_ensemble(
		df=df,
		model_group=model_group,
		vacant_only=vacant_only,
		hedonic=True,
		ind_var=ind_var,
		ind_var_test=ind_var_test,
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
		df_in: pd.DataFrame,
		model_group: str,
		settings: dict,
		vacant_only: bool = False,
		save_params: bool = True,
		use_saved_params: bool = True,
		use_saved_results: bool = True,
		verbose: bool = False
):

	df = df_in[df_in["model_group"] == model_group].copy()

	s = settings
	s_model = s.get("modeling", {})
	s_inst = s_model.get("instructions", {})
	vacant_status = "vacant" if vacant_only else "main"

	ind_var = s_inst.get("ind_var", "sale_price")
	ind_var_test = s_inst.get("ind_var_test", "sale_price")
	fields_cat = get_fields_categorical(s, df)
	models_to_run = s_inst.get(vacant_status, {}).get("run", None)
	model_entries = s_model.get("models").get(vacant_status, {})
	if models_to_run is None:
		models_to_run = list(model_entries.keys())

	# Enforce that horizontal equity cluster ID's have already been calculated
	if "he_id" not in df:
		raise ValueError("Could not find equity cluster ID's in the dataframe (he_id)")

	model_results = {}
	outpath = f"out/models/{model_group}/{vacant_status}"
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	var_recs = get_variable_recommendations(
		df,
		vacant_only,
		settings,
		"default",
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
	markdown_to_pdf(var_report_md, pdf_path, css_file="variable")

	# Run the models one by one and stash the results
	for model in models_to_run:
		results = _run_one_model(
			df_multiverse=df_in,
			df=df,
			vacant_only=vacant_only,
			model_group=model_group,
			model=model,
			model_entries=model_entries,
			settings=settings,
			ind_var=ind_var,
			ind_var_test=ind_var_test,
			best_variables=best_variables,
			fields_cat=fields_cat,
			outpath=outpath,
			save_params=save_params,
			use_saved_params=use_saved_params,
			use_saved_results=use_saved_results,
			verbose=verbose
		)
		if results is not None:
			model_results[model] = results

	# Calculate initial results (ensemble will use them)
	all_results = MultiModelResults(
		model_results=model_results,
		benchmark=_calc_benchmark(model_results)
	)

	best_ensemble = optimize_ensemble(
		df=df,
		model_group=model_group,
		vacant_only=vacant_only,
		ind_var=ind_var,
		ind_var_test=ind_var_test,
		all_results=all_results,
		settings=settings,
		verbose=verbose
	)

	# Run the ensemble model
	ensemble_results = run_ensemble(
		df=df,
		model_group=model_group,
		vacant_only=vacant_only,
		hedonic=False,
		ind_var=ind_var,
		ind_var_test=ind_var_test,
		outpath=outpath,
		ensemble_list=best_ensemble,
		all_results=all_results,
		settings=settings,
		verbose=verbose,
		df_multiverse=df_in
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

	if not vacant_only:
		_run_hedonic_models(
			settings,
			model_group,
			vacant_only,
			models_to_run,
			all_results,
			df,
			ind_var,
			ind_var_test,
			fields_cat,
			use_saved_results,
			verbose,
			df_multiverse=df_in
		)

	return all_results