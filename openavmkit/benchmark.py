import os

import numpy as np
import pandas as pd

from openavmkit.modeling import run_mra, run_gwr, run_xgboost, run_lightgbm, run_catboost, SingleModelResults, run_garbage, \
	run_average, run_naive_sqft, DataSplit, run_kernel, run_mgwr
from openavmkit.time_adjustment import apply_time_adjustment
from openavmkit.utilities.data import div_z_safe
from openavmkit.utilities.settings import get_fields_categorical, get_variable_interactions
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
		result += "Full set:\n"
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
		"full": [],
		"chd": [],
	}

	data = {
		"model":[],
		"subset":[],
		"utility_score": [],
		"mse":[],
		"rmse":[],
		"r2":[],
		"adj_r2":[],
		"median_ratio":[],
		"cod":[],
		"prd":[],
		"prb":[],
		"chd":[]
	}
	for key in model_results:
		for kind in ["test", "full"]:
			results = model_results[key]
			if kind == "test":
				pred_results = results.pred_test
				subset = "Test set"
			else:
				pred_results = results.pred_sales
				subset = "Full set"
			data["model"].append(key)
			data["subset"].append(subset)
			data["utility_score"].append(results.utility)
			data["mse"].append(pred_results.mse)
			data["rmse"].append(pred_results.rmse)
			data["r2"].append(pred_results.r2)
			data["adj_r2"].append(pred_results.adj_r2)
			data["median_ratio"].append(pred_results.ratio_study.median_ratio)
			data["cod"].append(pred_results.ratio_study.cod)
			data["prd"].append(pred_results.ratio_study.prd)
			data["prb"].append(pred_results.ratio_study.prb)

			chd_results = None

			if kind == "full":
				chd_results = results.chd
				tim = results.timing.results
				data_time["model"].append(key)
				data_time["total"].append(tim["total"])
				data_time["param"].append(tim["parameter_search"])
				data_time["train"].append(tim["train"])
				data_time["test"].append(tim["predict_test"])
				data_time["full"].append(tim["predict_full"])
				data_time["chd"].append(tim["chd"])

			data["chd"].append(chd_results)

	df = pd.DataFrame(data)

	df_test = df[df["subset"].eq("Test set")].drop(columns=["subset"])
	df_full = df[df["subset"].eq("Full set")].drop(columns=["subset"])
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

	def fancy_format(num):
		if np.isinf(num):
			if num > 0:
				return " ∞"
			else:
				return "-∞"
		if pd.isna(num):
			return "N/A"
		if num == 0:
			return '0.00'
		if num < 1:
			return '{:.2f}'.format(num)
		num = float('{:.3g}'.format(num))
		magnitude = 0
		while abs(num) >= 1000 and abs(num) > 1e-6:
			magnitude += 1
			num /= 1000.0
		return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

	formats = {
		"utility_score": fancy_format,
		"mse": fancy_format,
		"rmse": fancy_format,
		"r2": "{:.2f}",
		"adj_r2": "{:.2f}",
		"median_ratio": "{:.2f}",
		"cod": "{:.2f}",
		"prd": "{:.2f}",
		"prb": "{:.2f}",
		"total": fancy_format,
		"param": fancy_format,
		"train": fancy_format,
		"test": fancy_format,
		"full": fancy_format,
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


def _run_one_model(
		df: pd.DataFrame,
		model: str,
		model_entries: dict,
		settings: dict,
		ind_var: str,
		ind_var_test: str,
		fields_cat: list[str],
		outpath: str,
		save_params: bool,
		use_saved_params: bool,
		verbose: bool = False
) -> SingleModelResults:
	model_name = model
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

	dep_vars : list | None = entry.get("dep_vars", default_entry.get("dep_vars", None))
	if dep_vars is None:
		raise ValueError(f"dep_vars not found for model {model}")

	interactions = get_variable_interactions(entry, settings, df)

	instructions = settings.get("modeling", {}).get("instructions", {})
	test_train_frac = instructions.get("test_train_frac", 0.8)
	random_seed = instructions.get("random_seed", 1337)

	ds = DataSplit(
		df,
		settings,
		ind_var,
		ind_var_test,
		dep_vars,
		fields_cat,
		interactions,
		test_train_frac,
		random_seed
	)

	intercept = entry.get("intercept", True)

	results = None

	if model_name == "garbage":
		results = run_garbage(ds, normal=False, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "garbage_normal":
		results = run_garbage(ds, normal=True, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "mean":
		results = run_average(ds, type="mean", sales_chase=sales_chase, verbose=verbose)
	elif model_name == "median":
		results = run_average(ds, type="median", sales_chase=sales_chase, verbose=verbose)
	elif model_name == "naive_sqft":
		results = run_naive_sqft(ds, sales_chase=sales_chase, verbose=verbose)
	elif model_name == "mra":
		results = run_mra(ds, intercept=intercept, verbose=verbose)
	elif model_name == "kernel":
		results = run_kernel(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "gwr":
		results = run_gwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "mgwr":
		results = run_mgwr(ds, outpath, intercept, save_params, use_saved_params, verbose=verbose)
	elif model_name == "xgboost":
		results = run_xgboost(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "lightgbm":
		results = run_lightgbm(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "catboost":
		results = run_catboost(ds, outpath, save_params, use_saved_params, verbose=verbose)

	# write out the results:
	write_model_results(results, outpath)

	return results


def _assemble_model_results(results: SingleModelResults):
	fields = ["key", "prediction", "assr_market_value", "sale_price", "sale_price_time_adj", "sale_date"]

	dfs = {
		"sales": results.df_sales[fields].copy(),
		"universe": results.df_universe[fields].copy(),
		"test": results.df_test[fields].copy()
	}

	for key in dfs:
		df = dfs[key]
		df["prediction_ratio"] = div_z_safe(df, "prediction", "sale_price")
		df["assr_ratio"] = div_z_safe(df, "assr_market_value", "sale_price")
	return dfs


def write_model_results(results: SingleModelResults, outpath: str):
	dfs = _assemble_model_results(results)
	path = f"{outpath}/{results.type}"
	os.makedirs(path, exist_ok=True)
	for key in dfs:
		df = dfs[key]
		df.to_parquet(f"{path}/pred_{key}.parquet")
		df.to_csv(f"{path}/pred_{key}.csv", index=False)


def write_ensemble_model_results(
		results: SingleModelResults,
		outpath: str,
		dfs: dict[str, pd.DataFrame],
		ensemble_list: list[str]
):
	dfs_basic = _assemble_model_results(results)
	path = f"{outpath}/{results.type}"
	os.makedirs(path, exist_ok=True)
	for key in dfs_basic:
		df_basic = dfs_basic[key]
		df_ensemble = dfs[key]
		df_ensemble = df_ensemble[["key"] + ensemble_list]
		df = df_basic.merge(df_ensemble, on="key", how="left")
		df.to_parquet(f"{path}/pred_ensemble_{key}.parquet")
		df.to_csv(f"{path}/pred_ensemble_{key}.csv", index=False)


def run_ensemble(
		df: pd.DataFrame,
		ind_var: str,
		ind_var_test: str,
		outpath: str,
		all_results: MultiModelResults,
		settings: dict,
		verbose: bool = False
):
	timing = TimingData()

	timing.start("total")

	timing.start("setup")

	instructions = settings.get("modeling", {}).get("instructions", {})
	test_train_frac = instructions.get("test_train_frac", 0.8)
	random_seed = instructions.get("random_seed", 1337)

	ds = DataSplit(
		df,
		settings,
		ind_var,
		ind_var_test,
		[],
		[],
		{},
		test_train_frac,
		random_seed
	)

	df_test = ds.df_test
	df_sales = ds.df_sales
	df_univ = ds.df_universe
	instructions = settings.get("modeling", {}).get("instructions", {})
	ensemble_list = instructions.get("ensemble", [])
	df_test_ensemble = df_test[["key"]].copy()
	df_sales_ensemble = df_sales[["key"]].copy()
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
		df_sales_ensemble[m_key] = m_results.pred_sales.y_pred
		df_univ_ensemble[m_key] = m_results.pred_univ
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales_ensemble = df_sales_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_sales")

	timing.start("predict_full")
	y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
	timing.stop("predict_full")

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
		verbose=verbose
	)
	timing.stop("total")

	dfs = {
		"sales": df_sales_ensemble,
		"universe": df_univ_ensemble,
		"test": df_test_ensemble
	}
	write_ensemble_model_results(results, outpath, dfs, ensemble_list)
	return results


def run_models(
		df: pd.DataFrame,
		settings: dict,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	# Gather settings
	s = settings
	s_model = s.get("modeling", {})
	s_inst = s_model.get("instructions", {})
	model_dict = s_model.get("models", None)
	if model_dict is None:
		raise ValueError("settings.modeling.models not found!")

	# Apply time adjustment if necessary
	if "sale_price_time_adj" not in df:
		if verbose:
			print("Applying time adjustment...")
		period = s_inst.get("time_adjustment", {}).get("period", "Q")
		df = apply_time_adjustment(df.copy(), settings, period=period, verbose=verbose)

	ind_var = s_inst.get("ind_var", "sale_price_time_adj")
	ind_var_test = s_inst.get("ind_var_test", "sale_price")
	fields_cat = get_fields_categorical(s, df)
	models_to_run = s_inst.get("run", None)
	model_entries = s_model.get("models", {})
	if models_to_run is None:
		models_to_run = list(model_entries.keys())

	# Enforce that horizontal equity cluster ID's have already been calculated
	if "he_id" not in df:
		raise ValueError("Could not find equity cluster ID's in the dataframe (he_id)")

	model_results = {}
	outpath = f"out"

	df.to_parquet(f"{outpath}/df.parquet")

	# Run the models one by one and stash the results
	for model in models_to_run:
		results = _run_one_model(
			df=df,
			model=model,
			model_entries=model_entries,
			settings=settings,
			ind_var=ind_var,
			ind_var_test=ind_var_test,
			fields_cat=fields_cat,
			outpath=outpath,
			save_params=save_params,
			use_saved_params=use_saved_params,
			verbose=verbose
		)
		if results is not None:
			model_results[model] = results

	# Calculate initial results (ensemble will use them)
	all_results = MultiModelResults(
		model_results=model_results,
		benchmark=_calc_benchmark(model_results)
	)

	# Run the ensemble model
	ensemble_results = run_ensemble(
		df=df,
		ind_var=ind_var,
		ind_var_test=ind_var_test,
		outpath=outpath,
		all_results=all_results,
		settings=settings,
		verbose=verbose
	)

	# Calculate final results, including ensemble
	all_results.add_model("ensemble", ensemble_results)
	return all_results