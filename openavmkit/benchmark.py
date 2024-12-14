import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from openavmkit.modeling import run_mra, run_gwr, run_xgboost, run_lightgbm, run_catboost, ModelResults, run_garbage, \
	run_average, run_naive_sqft, DataSplit
from openavmkit.utilities.settings import get_fields_categorical, get_fields_numeric, get_variable_interactions


def _calc_benchmark(model_results: dict[str, ModelResults]):
	data = {
		"model":[],
		"subset":[],
		"t_total": [],
		"t_param": [],
		"t_train": [],
		"t_predict": [],
		"t_test": [],
		"t_full": [],
		"t_chd": [],
		"utility": [],
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
			subset = ""
			if kind == "test":
				pred_results = results.pred_test
				subset = "Test set"
			else:
				pred_results = results.pred_sales
				subset = "Full set"
			data["model"].append(key)
			data["subset"].append(subset)
			data["utility"].append(results.utility)
			data["mse"].append(pred_results.mse)
			data["rmse"].append(pred_results.rmse)
			data["r2"].append(pred_results.r2)
			data["adj_r2"].append(pred_results.adj_r2)
			data["median_ratio"].append(pred_results.ratio_study.median_ratio)
			data["cod"].append(pred_results.ratio_study.cod)
			data["prd"].append(pred_results.ratio_study.prd)
			data["prb"].append(pred_results.ratio_study.prb)

			tim = results.timing.results

			data["t_total"].append(tim["total"])
			data["t_param"].append(tim["parameter_search"])
			data["t_train"].append(tim["train"])
			data["t_predict"].append(0)
			data["t_test"].append(tim["predict_test"])
			data["t_full"].append(tim["predict_full"])
			data["t_chd"].append(tim["chd"])

			if kind == "full":
				data["chd"].append(results.chd)
			else:
				data["chd"].append(None)
	df = pd.DataFrame(data)

	df_test = df[df["subset"].eq("Test set")].drop(columns=["subset"])
	df_test["t_predict"] = df["t_test"]
	df_test = df_test.drop(columns=["t_test", "t_full"])

	df_full = df[df["subset"].eq("Full set")].drop(columns=["subset"])
	df_full["t_predict"] = df["t_full"]
	df_full = df_full.drop(columns=["t_test", "t_full"])

	# set index to the model column:
	df_test.set_index("model", inplace=True)
	df_full.set_index("model", inplace=True)
	return df_test, df_full

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
		"utility": fancy_format,
		"mse": fancy_format,
		"rmse": fancy_format,
		"r2": "{:.2f}",
		"adj_r2": "{:.2f}",
		"median_ratio": "{:.2f}",
		"cod": "{:.2f}",
		"prd": "{:.2f}",
		"prb": "{:.2f}",
		"chd": fancy_format,
		"t_total": fancy_format,
		"t_param": fancy_format,
		"t_train": fancy_format,
		"t_predict": fancy_format,
		"t_chd": fancy_format
	}

	for col in df.columns:
		if col in formats:
			# check if formats[col] is a function or a string
			if callable(formats[col]):
				df[col] = df[col].apply(formats[col])
			else:
				df[col] = df[col].apply(lambda x: formats[col].format(x))

	return df.transpose().to_markdown()


def run_benchmark(
		df: pd.DataFrame,
		settings: dict,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	s = settings
	s_model = s.get("modeling", {})
	ind_var = s_model.get("ind_var", "sale_price")

	model_dict = s_model.get("models", None)
	if model_dict is None:
		raise ValueError("settings.modeling.models not found!")

	fields_cat = get_fields_categorical(s, df)

	s_inst = s_model.get("instructions", {})

	models_to_run = s_inst.get("run", None)
	model_entries = s_model.get("models", {})
	if models_to_run is None:
		models_to_run = list(model_entries.keys())

	if "he_id" not in df:
		raise ValueError("Could not find equity cluster ID's in the dataframe (he_id)")

	results = None
	model_results = {}

	outpath = f"out"

	for model in models_to_run:
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

		print(f" running model {model}...")

		dep_vars = entry.get("dep_vars", None)
		if dep_vars is None:
			raise ValueError(f"dep_vars not found for model {model}")

		interactions = get_variable_interactions(entry, settings, df)

		ds = DataSplit(df, ind_var, dep_vars, fields_cat, interactions)

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
		elif model_name == "gwr":
			results = run_gwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
		elif model_name == "xgboost":
			results = run_xgboost(ds, outpath, save_params, use_saved_params, verbose=verbose)
		elif model_name == "lightgbm":
			results = run_lightgbm(ds, outpath, save_params, use_saved_params, verbose=verbose)
		elif model_name == "catboost":
			results = run_catboost(ds, outpath, save_params, use_saved_params, verbose=verbose)
		if results is not None:
			model_results[model] = results

	return _calc_benchmark(model_results)