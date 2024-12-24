import os

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from tabulate import tabulate

from openavmkit.modeling import run_mra, run_gwr, run_xgboost, run_lightgbm, run_catboost, SingleModelResults, run_garbage, \
	run_average, run_naive_sqft, DataSplit, run_kernel, run_mgwr
from openavmkit.reports import MarkdownReport
from openavmkit.time_adjustment import apply_time_adjustment, enrich_time_adjustment
from openavmkit.utilities.data import div_z_safe, dataframe_to_markdown
from openavmkit.utilities.settings import get_fields_categorical, get_variable_interactions, get_valuation_date, \
	get_modeling_group, apply_dd_to_df
from openavmkit.utilities.stats import calc_vif, calc_vif_recursive_drop, calc_t_values, calc_t_values_recursive_drop, \
	calc_p_values_recursive_drop, calc_elastic_net_regularization, calc_correlations, calc_r2, calc_cross_validation_score
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
		"count":[],
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
			data["count"].append(pred_results.ratio_study.count)
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
		"count": "{:.0f}",
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
		best_variables: list[str],
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

	are_dep_vars_default = entry.get("dep_vars", None) is None

	dep_vars : list | None = entry.get("dep_vars", default_entry.get("dep_vars", None))
	if dep_vars is None:
		raise ValueError(f"dep_vars not found for model {model}")

	# if not are_dep_vars_default:
	# 	get_variable_recommendations(
	# 		df,
	# 		settings
	#
	# 	)

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


def optimize_ensemble(
		df: pd.DataFrame,
		ind_var: str,
		ind_var_test: str,
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
	df_univ = ds.df_universe
	instructions = settings.get("modeling", {}).get("instructions", {})
	ensemble_list = instructions.get("ensemble", [])
	if len(ensemble_list) == 0:
		ensemble_list = [key for key in all_results.model_results.keys()]

	best_list = []
	best_score = float('inf')

	while len(ensemble_list) > 0:
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
			y_pred_sales=None,
			y_pred_univ=y_pred_univ_ensemble.to_numpy(),
			timing=timing,
			verbose=verbose
		)
		timing.stop("total")

		score = results.utility

		if verbose:
			print(f"score = {score:5.0f}, best = {best_score:5.0f}, ensemble = {ensemble_list}...")

		if score < best_score:
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
					if verbose:
						print(f"--> kicking score {worst_score:5.0f}, model = {worst_model}")

		ensemble_list.remove(worst_model)

	if verbose:
		print(f"Best score = {best_score:5.0}, ensemble = {best_list}")
	return best_list

def run_ensemble(
		df: pd.DataFrame,
		ind_var: str,
		ind_var_test: str,
		outpath: str,
		ensemble_list: list[str],
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


def _prepare_ds(
	df: pd.DataFrame,
	settings: dict,
	model: str = None
):
	s = settings
	s_model = s.get("modeling", {})
	model_entries = s_model.get("models")
	entry: dict | None = model_entries.get("model", model_entries.get("default"))

	dep_vars : list | None = entry.get("dep_vars", None)
	if dep_vars is None:
		raise ValueError(f"dep_vars not found for model 'default'")

	fields_cat = get_fields_categorical(s, df)
	interactions = get_variable_interactions(entry, s, df)

	instructions = s.get("modeling", {}).get("instructions", {})
	ind_var = instructions.get("ind_var", "sale_price")
	ind_var_test = instructions.get("ind_var_test", "sale_price_time_adj")
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

		thresh_corr = thresh.get("correlation", 0.1)
		report.set_var("thresh_corr", thresh_corr, ".2f")
		corr_fields = ["variable", "corr_strength", "corr_clarity", "corr_score"]
		corr_renames = {
			"variable": "Variable",
			"corr_strength": "Strength",
			"corr_clarity": "Clarity",
			"corr_score": "Score"
		}
		for state in ["initial", "final"]:
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
			dfr_corr = apply_dd_to_df(dfr_corr, "Variable", settings, ds.one_hot_descendants)
			markdown_output = dataframe_to_markdown(dfr_corr)
			display(markdown_output)
			report.set_var(f"table_corr_{state}", markdown_output)

		dfr["Rank"] = range(1, len(dfr) + 1)
		dfr = apply_dd_to_df(dfr, "Variable", settings, ds.one_hot_descendants)

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
		settings: dict,
		model: str,
		modeling_group: str,
		verbose: bool = False
):
	if verbose:
		print("")

	report = MarkdownReport("variables")

	df = enrich_time_adjustment(df, settings, verbose=verbose)
	ds = _prepare_ds(df, settings, model)
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

	pd.set_option('display.max_columns', None)
	display(df_results)

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
			if verbose:
				print(f"--> BEST SO FAR!")
		worst_idx = df_results["weighted_score"].idxmin()
		worst_score = df_results.loc[worst_idx, "weighted_score"]
		worst_variable = df_results.loc[worst_idx, "variable"]
		curr_variables.remove(worst_variable)
		# remove the variable from the dataframe:
		df_results = df_results[df_results["variable"].ne(worst_variable)]
		if verbose:
			print(f"{len(curr_variables)} variables: {curr_variables}")
			print(f"--> score: {cv_score:,.0f}")
			print(f"-->  drop: {worst_variable}, score {worst_score}...")

	# make a table from the list of best variables:
	df_best = pd.DataFrame(best_variables, columns=["Variable"])
	df_best["Rank"] = range(1, len(df_best) + 1)
	df_best["Description"] = df_best["Variable"]
	df_best = apply_dd_to_df(df_best, "Variable", settings, ds.one_hot_descendants, "name")
	df_best = apply_dd_to_df(df_best, "Description", settings, ds.one_hot_descendants, "description")
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
		modeling_group,
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

	table_vif_initial = "...VIF INITIAL..."
	report.set_var("table_vif_initial", table_vif_initial)

	table_vif_final = "...VIF FINAL..."
	report.set_var("table_vif_final", table_vif_final)

	table_p_value_initial = "...P VALUE INITIAL..."
	report.set_var("table_p_value_initial", table_p_value_initial)

	table_p_value_final = "...P VALUE FINAL..."
	report.set_var("table_p_value_final", table_p_value_final)

	table_p_value_initial = "...T VALUE INITIAL..."
	report.set_var("table_t_value_initial", table_p_value_initial)

	table_p_value_final = "...T VALUE FINAL..."
	report.set_var("table_t_value_final", table_p_value_final)

	table_enr_initial = "...ENR INITIAL..."
	report.set_var("table_enr_initial", table_enr_initial)

	table_enr_final = "...ENR FINAL..."
	report.set_var("table_enr_final", table_enr_final)

	table_r_squared_initial = "...R SQUARED INITIAL..."
	report.set_var("table_r_squared_initial", table_r_squared_initial)

	table_r_squared_final = "...R SQUARED FINAL..."
	report.set_var("table_r_squared_final", table_r_squared_final)

	table_coef_initial = "...COEF INITIAL..."
	report.set_var("table_coef_initial", table_coef_initial)

	table_coef_final = "...COEF FINAL..."
	report.set_var("table_coef_final", table_coef_final)

	return report


def run_models(
		df: pd.DataFrame,
		settings: dict,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	s = settings
	s_model = s.get("modeling", {})
	s_inst = s_model.get("instructions", {})

	df = enrich_time_adjustment(df, s, verbose=verbose)

	ind_var = s_inst.get("ind_var", "sale_price")
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

	var_recs = get_variable_recommendations(
		df,
		settings,
		"default",
		"residential_sf",
		verbose=True,
	)
	best_variables = var_recs["variables"]
	var_report = var_recs["report"]
	var_report_md = var_report.render()
	os.makedirs(f"{outpath}/reports", exist_ok=True)
	with open(f"{outpath}/reports/variable_report.md", "w", encoding="utf-8") as f:
		f.write(var_report_md)

	# Run the models one by one and stash the results
	for model in models_to_run:
		results = _run_one_model(
			df=df,
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
		ind_var=ind_var,
		ind_var_test=ind_var_test,
		all_results=all_results,
		settings=settings,
		verbose=verbose
	)

	# Run the ensemble model
	ensemble_results = run_ensemble(
		df=df,
		ind_var=ind_var,
		ind_var_test=ind_var_test,
		outpath=outpath,
		ensemble_list=best_ensemble,
		all_results=all_results,
		settings=settings,
		verbose=verbose
	)

	# Calculate final results, including ensemble
	all_results.add_model("ensemble", ensemble_results)
	return all_results