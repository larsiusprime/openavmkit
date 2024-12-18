import numpy as np
import pandas as pd

from openavmkit.modeling import run_mra, run_gwr, run_xgboost, run_lightgbm, run_catboost, SingleModelResults, run_garbage, \
	run_average, run_naive_sqft, DataSplit, run_kernel, run_mgwr
from openavmkit.time_adjustment import apply_time_adjustment
from openavmkit.utilities.settings import get_fields_categorical, get_variable_interactions


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
):
	model_name = model
	entry = model_entries.get(model, None)
	if entry is None:
		entry = model_entries.get("default", None)
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

	dep_vars = entry.get("dep_vars", None)
	if dep_vars is None:
		raise ValueError(f"dep_vars not found for model {model}")

	interactions = get_variable_interactions(entry, settings, df)
	ds = DataSplit(df, ind_var, ind_var_test, dep_vars, fields_cat, interactions)

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
		results = run_mra(ds, verbose=verbose)
	elif model_name == "kernel":
		results = run_kernel(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "gwr":
		results = run_gwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "mgwr":
		results = run_mgwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "xgboost":
		results = run_xgboost(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "lightgbm":
		results = run_lightgbm(ds, outpath, save_params, use_saved_params, verbose=verbose)
	elif model_name == "catboost":
		results = run_catboost(ds, outpath, save_params, use_saved_params, verbose=verbose)

	return results



def run_models(
		df: pd.DataFrame,
		settings: dict,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	# Apply time adjustment if necessary
	if "sale_price_time_adj" not in df:
		df = apply_time_adjustment(df.copy())

	# Gather settings
	s = settings
	s_model = s.get("modeling", {})
	model_dict = s_model.get("models", None)
	if model_dict is None:
		raise ValueError("settings.modeling.models not found!")

	ind_var = s_model.get("ind_var", "sale_price_time_adj")
	ind_var_test = s_model.get("ind_var_test", "sale_price")
	fields_cat = get_fields_categorical(s, df)
	s_inst = s_model.get("instructions", {})
	models_to_run = s_inst.get("run", None)
	model_entries = s_model.get("models", {})
	if models_to_run is None:
		models_to_run = list(model_entries.keys())

	# Enforce that horizontal equity cluster ID's have already been calculated
	if "he_id" not in df:
		raise ValueError("Could not find equity cluster ID's in the dataframe (he_id)")

	model_results = {}
	outpath = f"out"

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

	return MultiModelResults(
		model_results=model_results,
		benchmark=_calc_benchmark(model_results)
	)