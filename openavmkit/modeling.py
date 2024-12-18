import json
import math
import os
import polars as pl
from IPython.core.display_functions import display
from joblib import Parallel, delayed
from typing import Union

import numpy as np
import statsmodels.api as sm
import pandas as pd
import xgboost
import lightgbm as lgb
import catboost
from catboost import CatBoostRegressor, Pool
from lightgbm import Booster
from mgwr.gwr import MGWR, GWR
from mgwr.gwr import _compute_betas_gwr, Kernel
from mgwr.kernels import local_cdist
from mgwr.sel_bw import Sel_BW
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import RegressionResults
from xgboost import XGBRegressor

from openavmkit.ratio_study import RatioStudy
from openavmkit.utilities.data import clean_column_names
from openavmkit.utilities.stats import quick_median_chd
from openavmkit.utilities.tuning import tune_lightgbm, tune_xgboost, tune_catboost
from openavmkit.utilities.timing import TimingData

PredictionModel = Union[RegressionResults, XGBRegressor, Booster, CatBoostRegressor, GWR, MGWR, None]

class PredictionResults:
	ind_var: str
	dep_vars: list[str]
	y: pd.Series
	y_pred: np.ndarray
	mse: float
	rmse: float
	r2: float
	adj_r2: float
	ratio_study: RatioStudy

	def __init__(self,
			ind_var: str,
			dep_vars: list[str],
			y: pd.Series,
			y_pred: np.ndarray
	):
		self.ind_var = ind_var
		self.dep_vars = dep_vars
		self.y = y
		self.y_pred = y_pred

		self.mse = mean_squared_error(y, y_pred)
		self.rmse = np.sqrt(self.mse)
		self.r2 = 1 - self.mse / np.var(y)

		# Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]
		#
		# where:
		#
		# R2: The R2 of the model
		# n: The number of observations
		# k: The number of predictor variables

		n = len(y_pred)
		k = len(dep_vars)

		self.adj_r2 = 1 - ((1 - self.r2)*(n-1)/(n-k-1))
		self.ratio_study = RatioStudy(y_pred, y.to_numpy())

class DataSplit:
	df_sales: pd.DataFrame
	df_universe: pd.DataFrame
	df_train: pd.DataFrame
	df_test: pd.DataFrame
	X_univ: pd.DataFrame
	X_sales: pd.DataFrame
	y_sales: pd.Series
	X_train: pd.DataFrame
	y_train: pd.Series
	X_test: pd.DataFrame
	y_test: pd.Series
	test_train_frac: float
	random_seed: int
	ind_var: str
	ind_var_test: str
	dep_vars: list[str]
	categorical_vars: list[str]
	interactions: dict
	one_hot_descendants: dict
	days_field: str

	def __init__(self,
			df: pd.DataFrame,
			ind_var: str,
			ind_var_test: str,
			dep_vars: list[str],
			categorical_vars: list[str],
			interactions: dict,
			test_train_frac: float = 0.8,
			random_seed: int = 1337,
			days_field: str = "sale_age_days"
	):

		self.df_universe = df.copy()
		self.df_sales = df[df["valid_sale"].eq(1)].copy().reset_index(drop=True)

		# Pre-sort dataframes so that rolling origin cross-validation can assume oldest observations first:
		if days_field in self.df_universe:
			self.df_universe.sort_values(by=days_field, ascending=False, inplace=True)
		else:
			raise ValueError(f"Field '{days_field}' not found in dataframe.")

		if days_field in self.df_sales:
			self.df_sales.sort_values(by=days_field, ascending=False, inplace=True)
		else:
			raise ValueError(f"Field '{days_field}' not found in dataframe.")

		self.ind_var = ind_var
		self.ind_var_test = ind_var_test
		self.dep_vars = dep_vars.copy()
		self.categorical_vars = categorical_vars.copy()
		self.interactions = interactions.copy()
		self.one_hot_descendants = {}
		self.random_seed = random_seed
		self.days_field = days_field
		self.test_train_frac = test_train_frac
		self.split()


	def copy(self):
		# Return a deep copy
		return DataSplit(
			self.df_universe.copy(),
			self.ind_var,
			self.ind_var_test,
			self.dep_vars,
			self.categorical_vars,
			self.interactions,
			self.test_train_frac,
			self.random_seed,
			self.days_field
		)


	def encode_categoricals_as_categories(self):

		if len(self.categorical_vars) == 0:
			return self

		ds = self.copy()

		# Ensure all categorical variables are encoded with the "categorical" dtype:
		for col in ds.categorical_vars:
			ds.df_universe[col] = ds.df_universe[col].astype("category")
			ds.df_sales[col] = ds.df_sales[col].astype("category")

		return ds


	def encode_categoricals_with_one_hot(self):

		if len(self.categorical_vars) == 0:
			return self

		ds = self.copy()

		dep_vars = self.dep_vars

		cat_vars = [col for col in dep_vars if col in self.categorical_vars]

		old_cols = ds.df_universe.columns.values

		# One-hot encode the categorical variables, perform this on ds rather than self, do it for everything:
		ds.df_universe = pd.get_dummies(ds.df_universe, columns=[col for col in cat_vars if col in ds.df_universe], drop_first=True)
		ds.df_sales = pd.get_dummies(ds.df_sales, columns=[col for col in cat_vars if col in ds.df_sales], drop_first=True)
		ds.df_train = pd.get_dummies(ds.df_train, columns=[col for col in cat_vars if col in ds.df_train], drop_first=True)
		ds.df_test = pd.get_dummies(ds.df_test, columns=[col for col in cat_vars if col in ds.df_test], drop_first=True)

		ds.df_universe = clean_column_names(ds.df_universe)
		ds.df_sales = clean_column_names(ds.df_sales)
		ds.df_train = clean_column_names(ds.df_train)
		ds.df_test = clean_column_names(ds.df_test)

		# Remove the original categorical variables:
		ds.df_universe = ds.df_universe.drop(columns=[col for col in cat_vars if col in ds.df_universe])
		ds.df_sales = ds.df_sales.drop(columns=[col for col in cat_vars if col in ds.df_sales])
		ds.df_train = ds.df_train.drop(columns=[col for col in cat_vars if col in ds.df_train])
		ds.df_test = ds.df_test.drop(columns=[col for col in cat_vars if col in ds.df_test])

		new_cols = [col for col in ds.df_train.columns.values if col not in old_cols]
		dep_vars += new_cols
		dep_vars = [col for col in dep_vars if col in ds.df_train.columns]
		ds.dep_vars = dep_vars

		# sort cat vars so the longest strings come first:
		cat_vars = sorted(cat_vars, key=len, reverse=True)

		ds.one_hot_descendants = {}
		matched = []
		for col in new_cols:
			for orig_col in cat_vars:
				if col in matched:
					continue
				if orig_col in col:
					if orig_col not in ds.one_hot_descendants:
						ds.one_hot_descendants[orig_col] = []
					ds.one_hot_descendants[orig_col].append(col)
					matched.append(col)


		# Ensure that only columns found in df_train are in the other dataframes:
		ds.df_universe = ds.df_universe[ds.df_train.columns]
		ds.df_sales = ds.df_sales[ds.df_train.columns]

		test_cols = [col for col in ds.df_train.columns if col in ds.df_test.columns]
		extra_cols = [col for col in ds.df_train.columns if col not in test_cols]

		ds.df_test = ds.df_test[test_cols]
		# add extra_cols, set them to zero:
		for col in extra_cols:
			ds.df_test[col] = 0.0

		return ds


	def split(self):
		# separate df into train & test:
		np.random.seed(self.random_seed)
		self.df_train = self.df_sales.sample(frac=self.test_train_frac)
		self.df_test = self.df_sales.drop(self.df_train.index)

		self.df_train = self.df_train.reset_index(drop=True)
		self.df_test = self.df_test.reset_index(drop=True)

		# sort again because sampling shuffles order:
		self.df_train.sort_values(by=self.days_field, ascending=False, inplace=True)
		self.df_test.sort_values(by=self.days_field, ascending=False, inplace=True)

		_df_univ = self.df_universe.copy()
		_df_sales = self.df_sales.copy()
		_df_train = self.df_train.copy()
		_df_test = self.df_test.copy()

		# if interactions is not empty, multiply the fields together:
		if self.interactions is not None and len(self.interactions) > 0:
			for parent_field, fill_field in self.interactions.items():
				target_fields = []
				if parent_field in self.one_hot_descendants:
					target_fields = self.one_hot_descendants[parent_field].copy()
				if parent_field not in self.categorical_vars:
					target_fields += parent_field
				for target_field in target_fields:
					if target_field in _df_univ:
						_df_univ[target_field] = _df_univ[target_field] * _df_univ[fill_field]
					if target_field in _df_sales:
						_df_sales[target_field] = _df_sales[target_field] * _df_sales[fill_field]
					if target_field in _df_train:
						_df_train[target_field] = _df_train[target_field] * _df_train[fill_field]
					if target_field in _df_test:
						_df_test[target_field] = _df_test[target_field] * _df_test[fill_field]

		self.X_univ = _df_univ[self.dep_vars]

		self.X_sales = _df_sales[self.dep_vars]
		self.y_sales = _df_sales[self.ind_var]

		self.X_train = _df_train[self.dep_vars]
		self.y_train = _df_train[self.ind_var]

		# convert all Float64 to float64 in X_train:
		for col in self.X_train.columns:
			# if it's a Float64 or a boolean, convert it to float64
			if (self.X_train[col].dtype == "Float64" or
					self.X_train[col].dtype == "Int64" or
					self.X_train[col].dtype == "boolean" or
					self.X_train[col].dtype == "bool"
			):
				self.X_train[col] = self.X_train[col].astype("float64")

		self.X_test = _df_test[self.dep_vars]
		self.y_test = _df_test[self.ind_var_test]

class ModelResults:
	type: str
	ind_var: str
	ind_var_test: str
	dep_vars: list[str]
	model: PredictionModel
	pred_test: PredictionResults
	pred_sales: PredictionResults
	pred_univ: np.ndarray
	chd: float
	utility: float
	timing: TimingData

	def __init__(self,
			df: pd.DataFrame,
			field_prediction: str,
			field_horizontal_equity_id: str,
			type: str,
			ind_var: str,
			ind_var_test: str,
			dep_vars: list[str],
			model: PredictionModel,
			y_test: pd.Series,
			y_pred_test: np.ndarray,
			y_sales: pd.Series,
			y_pred_sales: np.ndarray,
			y_pred_univ: np.ndarray,
			timing: TimingData,
			verbose: bool = False
	):
		self.type = type
		self.ind_var = ind_var
		self.ind_var_test = ind_var_test
		self.dep_vars = dep_vars
		self.model = model

		timing.start("stats_test")
		self.pred_test = PredictionResults(ind_var_test, dep_vars, y_test, y_pred_test)
		timing.stop("stats_test")

		timing.start("stats_sales")
		self.pred_sales = PredictionResults(ind_var_test, dep_vars, y_sales, y_pred_sales)
		timing.stop("stats_sales")

		if verbose:
			print("--> calculating CHD...")
		self.pred_univ = y_pred_univ
		timing.start("chd")
		# TODO: finish converting other stuff to polars
		pl_df = pl.DataFrame(df)
		self.chd = quick_median_chd(pl_df, field_prediction, field_horizontal_equity_id)
		timing.stop("chd")
		if verbose:
			print("----> done")

		timing.start("utility")
		self.utility = model_utility_score(self)
		timing.stop("utility")
		self.timing = timing

	def summary(self):
		str = ""

		str += (f"Model type: {self.type}\n")
		# Print the # of rows in test & full set
		# Print the MSE, RMSE, R2, and Adj R2 for test & full set
		str += (f"-->Test set, rows: {len(self.pred_test.y)}\n")
		str += (f"---->RMSE   : {self.pred_test.rmse:8.0f}\n")
		str += (f"---->R2     : {self.pred_test.r2:8.4f}\n")
		str += (f"---->Adj R2 : {self.pred_test.adj_r2:8.4f}\n")
		str += (f"---->M.Ratio: {self.pred_test.ratio_study.median_ratio:8.4f}\n")
		str += (f"---->COD    : {self.pred_test.ratio_study.cod:8.4f}\n")
		str += (f"---->PRD    : {self.pred_test.ratio_study.prd:8.4f}\n")
		str += (f"---->PRB    : {self.pred_test.ratio_study.prb:8.4f}\n")
		str += (f"\n")
		str += (f"-->Full set, rows: {len(self.pred_sales.y)}\n")
		str += (f"---->RMSE   : {self.pred_sales.rmse:8.0f}\n")
		str += (f"---->R2     : {self.pred_sales.r2:8.4f}\n")
		str += (f"---->Adj R2 : {self.pred_sales.adj_r2:8.4f}\n")
		str += (f"---->M.Ratio: {self.pred_sales.ratio_study.median_ratio:8.4f}\n")
		str += (f"---->COD    : {self.pred_sales.ratio_study.cod:8.4f}\n")
		str += (f"---->PRD    : {self.pred_sales.ratio_study.prd:8.4f}\n")
		str += (f"---->PRB    : {self.pred_sales.ratio_study.prb:8.4f}\n")
		str += (f"---->CHD    : {self.chd:8.4f}\n")

		str += (f"\n")
		if self.type == "mra":
			# print the coefficients?
			pass
		elif self.type == "gwr":
			# print the coefficients?
			pass
		elif self.type == "xgboost":
			# print the feature importance?
			pass
		elif self.type == "lightgbm":
			# print the feature importance?
			pass
		return str


def model_utility_score(
		model_results: ModelResults
):
	# We want to minimize:
	# 1. error
	# 2. the difference between the median ratio and 1
	# 3. the COD
	# 4. the CHD

	weight_dist_ratio = 1000.00
	weight_cod = 1.00
	weight_chd = 1.00
	weight_sales_chase = 7.5

	cod = model_results.pred_test.ratio_study.cod
	chd = model_results.chd

	# Is the median ratio over 1.05? Penalize over-estimates; err on the side of under-estimates
	ratio_over_penalty = 2 if model_results.pred_test.ratio_study.median_ratio < 1.05 else 1

	# calculate base score
	dist_ratio_score = abs(1.0 - model_results.pred_test.ratio_study.median_ratio) * weight_dist_ratio * ratio_over_penalty
	cod_score = cod * weight_cod
	chd_score = chd * weight_chd

	# penalize very low COD's with bad horizontal equity
	sales_chase_score = ((1.0/cod) * chd) * weight_sales_chase
	final_score = dist_ratio_score + cod_score + chd_score + sales_chase_score
	return final_score


def run_mra(
		ds: DataSplit,
		intercept: bool = True,
		verbose: bool = False
):
		timing = TimingData()

		timing.start("total")

		timing.start("setup")
		ds = ds.encode_categoricals_with_one_hot()
		ds.split()
		if intercept:
			ds.X_train = sm.add_constant(ds.X_train)
			ds.X_test = sm.add_constant(ds.X_test)
			ds.X_sales = sm.add_constant(ds.X_sales)
			ds.X_univ = sm.add_constant(ds.X_univ)
		timing.stop("setup")

		timing.start("parameter_search")
		timing.stop("parameter_search")

		timing.start("train")
		linear_model = sm.OLS(ds.y_train, ds.X_train)
		fitted_model = linear_model.fit()
		# convert to Float64 dtype
		timing.stop("train")

		# predict on test set:
		timing.start("predict_test")
		y_pred_test = fitted_model.predict(ds.X_test).to_numpy()
		timing.stop("predict_test")

		# predict on the sales set:
		timing.start("predict_sales")
		y_pred_sales = fitted_model.predict(ds.X_sales).to_numpy()
		timing.stop("predict_sales")

		# predict on the full set:
		timing.start("predict_full")
		y_pred_univ = fitted_model.predict(ds.X_univ).to_numpy()
		timing.stop("predict_full")

		timing.stop("total")

		df = ds.df_universe
		ind_var = ds.ind_var
		ind_var_test = ds.ind_var_test
		dep_vars = ds.dep_vars

		# gather the predictions
		df["prediction"] = y_pred_univ

		results = ModelResults(
			df,
			"prediction",
			"he_id",
			"mra",
			ind_var,
			ind_var_test,
			dep_vars,
			fitted_model,
			ds.y_test,
			y_pred_test,
			ds.y_sales,
			y_pred_sales,
			y_pred_univ,
			timing,
			verbose=verbose
		)

		return results


def run_gwr(
		ds: DataSplit,
		outpath: str,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	"""
	Runs a GWR model
	:param ds: The data split object containing processed input data
	:param outpath: The output path
	:param save_params: Whether to save the parameters
	:param use_saved_params: Whether to use saved parameters
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("setup")
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()
	u_train = ds.df_train['longitude']
	v_train = ds.df_train['latitude']
	coords_train = list(zip(u_train, v_train))

	u_test = ds.df_test['longitude']
	v_test = ds.df_test['latitude']
	coords_test = list(zip(u_test, v_test))

	u_sales = ds.df_sales['longitude']
	v_sales = ds.df_sales['latitude']
	coords_sales = list(zip(u_sales, v_sales))

	u = ds.df_universe['longitude']
	v = ds.df_universe['latitude']
	coords_univ = list(zip(u,v))

	y_train = ds.y_train.to_numpy().reshape((-1, 1))

	X_train = ds.X_train.values
	X_test = ds.X_test.values
	X_sales = ds.X_sales.values
	X_univ = ds.X_univ.values

	# add a very small amount of random noise to every row in every column of X_train:
	# this is to prevent singular matrix errors in the GWR
	X_train += np.random.normal(0, 1e-6, X_train.shape)
	X_test += np.random.normal(0, 1e-6, X_test.shape)

	timing.stop("setup")

	timing.start("parameter_search")
	gwr_bw = -1.0

	if verbose:
		print("Tuning GWR: searching for optimal bandwidth...")

	if use_saved_params:
		if os.path.exists(f"{outpath}/gwr_bw.json"):
			gwr_bw = json.load(open(f"{outpath}/gwr_bw.json", "r"))
			if verbose:
				print(f"--> using saved bandwidth: {gwr_bw:0.2f}")

	if gwr_bw < 0:
		gwr_selector = Sel_BW(coords_train, y_train, X_train)
		gwr_bw = gwr_selector.search()

		if save_params:
			os.makedirs(outpath, exist_ok=True)
			json.dump(gwr_bw, open(f"{outpath}/gwr_bw.json", "w"))
		if verbose:
			print(f"--> optimal bandwidth = {gwr_bw:0.2f}")

	timing.stop("parameter_search")

	timing.start("train")
	gwr = GWR(coords_train, y_train, X_train, gwr_bw)
	gwr_fit = gwr.fit()
	gwr = gwr_fit.model
	timing.stop("train")

	np_coords_test = np.array(coords_test)
	timing.start("predict_test")
	gwr_result_test = gwr.predict(
		np_coords_test,
		X_test
	)
	y_pred_test = gwr_result_test.predictions.flatten()
	timing.stop("predict_test")

	timing.start("predict_sales")
	if verbose:
		print("GWR: predicting sales set...")
	y_pred_sales = _run_gwr_prediction(
		coords_sales,
		coords_train,
		X_sales,
		X_train,
		gwr_bw,
		y_train,
		verbose=verbose
	).flatten()
	timing.stop("predict_sales")

	timing.start("predict_full")
	if verbose:
		print("GWR: predicting full set...")
	y_pred_univ = _run_gwr_prediction(
		coords_univ,
		coords_train,
		X_univ,
		X_train,
		gwr_bw,
		y_train,
		verbose=verbose
	).flatten()
	timing.stop("predict_full")

	df = ds.df_universe
	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"gwr",
		ind_var,
		ind_var_test,
		dep_vars,
		gwr,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)
	timing.stop("total")

	return results


def run_xgboost(
		ds: DataSplit,
		outpath: str,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	"""
	Runs an XGBoost model
	:param ds: The data split object containing processed input data
	:param outpath: The output path
	:param save_params: Whether to save the parameters
	:param use_saved_params: Whether to use saved parameters
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("setup")
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()
	timing.stop("setup")

	timing.start("parameter_search")
	params = _get_params("XGBoost", "xgboost", ds, tune_xgboost, outpath, save_params, use_saved_params, verbose)
	timing.stop("parameter_search")

	timing.start("train")

	xgboost_model = xgboost.XGBRegressor(**params)
	xgboost_model.fit(ds.X_train, ds.y_train)
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test = xgboost_model.predict(ds.X_test)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales = xgboost_model.predict(ds.X_sales)
	timing.stop("predict_sales")

	timing.start("predict_full")
	y_pred_univ = xgboost_model.predict(ds.X_univ)
	timing.stop("predict_full")

	timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"xgboost",
		ind_var,
		ind_var_test,
		dep_vars,
		xgboost_model,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)
	return results


def run_lightgbm(
		ds: DataSplit,
		outpath: str,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	"""
	Runs a LightGBM model
	:param ds: The data split object containing processed input data
	:param outpath: The output path
	:param save_params: Whether to save the parameters
	:param use_saved_params: Whether to use saved parameters
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("setup")
	ds = ds.encode_categoricals_as_categories()
	ds.split()
	timing.stop("setup")

	timing.start("parameter_search")
	params = _get_params("LightGBM", "lightgbm", ds, tune_lightgbm, outpath, save_params, use_saved_params, verbose)
	timing.stop("parameter_search")

	timing.start("train")
	cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]
	lgb_train = lgb.Dataset(ds.X_train, ds.y_train, categorical_feature=cat_vars)
	lgb_test  = lgb.Dataset(ds.X_test,  ds.y_test,  categorical_feature=cat_vars, reference=lgb_train)

	params["verbosity"] = -1

	gbm = lgb.train(
		params,
		lgb_train,
		num_boost_round=1000,
		valid_sets=lgb_test,
		callbacks=[
			lgb.early_stopping(stopping_rounds=5, verbose=False),
			lgb.log_evaluation(period=0)
		]
	)
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test = gbm.predict(ds.X_test, num_iteration=gbm.best_iteration)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales = gbm.predict(ds.X_sales, num_iteration=gbm.best_iteration)
	timing.stop("predict_sales")

	timing.start("predict_full")
	y_pred_univ = gbm.predict(ds.X_univ, num_iterations=gbm.best_iteration)
	timing.stop("predict_full")

	timing.stop("total")

	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars
	df = ds.df_universe

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"lightgbm",
		ind_var,
		ind_var_test,
		dep_vars,
		gbm,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)
	return results


def run_catboost(
		ds: DataSplit,
		outpath: str,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	"""
	Runs a CatBoost model
	:param ds: The data split object containing processed input data
	:param outpath: The output path
	:param save_params: Whether to save the parameters
	:param use_saved_params: Whether to use saved parameters
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("parameter_search")
	params = _get_params("CatBoost", "catboost", ds, tune_catboost, outpath, save_params, use_saved_params, verbose)
	timing.stop("parameter_search")

	timing.start("setup")

	params["verbose"] = False
	cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]
	catboost_model = catboost.CatBoostRegressor(**params)
	train_pool = Pool(data=ds.X_train, label=ds.y_train, cat_features=cat_vars)
	test_pool = Pool(data=ds.X_test, label=ds.y_test, cat_features=cat_vars)
	sales_pool = Pool(data=ds.X_sales, label=ds.y_sales, cat_features=cat_vars)
	univ_pool = Pool(data=ds.X_univ, cat_features=cat_vars)

	timing.stop("setup")

	timing.start("train")
	catboost_model.fit(train_pool)
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test = catboost_model.predict(test_pool)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales = catboost_model.predict(sales_pool)
	timing.stop("predict_sales")

	timing.start("predict_full")
	y_pred_univ = catboost_model.predict(univ_pool)
	timing.stop("predict_full")

	timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"catboost",
		ind_var,
		ind_var_test,
		dep_vars,
		catboost_model,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)

	return results


def run_garbage(
		ds: DataSplit,
		normal: bool = False,
		sales_chase: float = 0.0,
		verbose: bool = False
):
	"""
	Runs a garbage model that simply predicts random values between the min and max of the training set.
	:param ds: The data split object containing processed input data
	:param normal: Whether to use a normal or uniform distribution when randomly picking
	:param sales_chase: If not 0, simulate sales chasing by predicting the sales set as the test set, with this much noise
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("setup")
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()
	timing.stop("setup")

	timing.start("train")
	min_value = ds.y_train.min()
	max_value = ds.y_train.max()
	timing.stop("train")

	timing.start("predict_test")
	if normal:
		y_pred_test = np.random.normal(loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_test))
	else:
		y_pred_test = np.random.uniform(min_value, max_value, len(ds.X_test))
	timing.stop("predict_test")

	timing.start("predict_sales")
	if normal:
		y_pred_sales = np.random.normal(loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_sales))
	else:
		y_pred_sales = np.random.uniform(min_value, max_value, len(ds.X_sales))
	timing.stop("predict_sales")

	timing.start("predict_full")
	if normal:
		y_pred_univ = np.random.normal(loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_univ))
	else:
		y_pred_univ = np.random.uniform(min_value, max_value, len(ds.X_univ))
	timing.stop("predict_full")

	timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars

	if sales_chase:
		y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
		y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
		y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

	name = "garbage"
	if normal:
		name = "garbage_normal"
	if sales_chase:
		name += "*"

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		name,
		ind_var,
		ind_var_test,
		dep_vars,
		None,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)

	return results


def run_average(
		ds: DataSplit,
		type: str = "mean",
		sales_chase: float = 0.0,
		verbose: bool = False
):
	"""
	Runs a garbage model that simply predicts the average of the training set for everything
	:param ds: The data split object containing processed input data
	:param type: The type of average to use ("mean" or "median")
	:param sales_chase: If not 0, simulate sales chasing by predicting the sales set as the test set, with this much noise
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("setup")
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()
	timing.stop("setup")

	timing.start("train")
	min_value = ds.y_train.min()
	max_value = ds.y_train.max()
	timing.stop("train")

	timing.start("predict_test")
	if type == "median":
		# get a series of equal length to ds.X_test filled with the mean of the training set
		y_pred_test = np.full(len(ds.X_test), ds.y_train.median())
	else:
		y_pred_test = np.full(len(ds.X_test), ds.y_train.mean())
	timing.stop("predict_test")

	timing.start("predict_sales")
	if type == "median":
		y_pred_sales = np.full(len(ds.X_sales), ds.y_train.median())
	else:
		y_pred_sales = np.full(len(ds.X_sales), ds.y_train.mean())
	timing.stop("predict_sales")

	timing.start("predict_full")
	if type == "median":
		y_pred_univ = np.full(len(ds.X_univ), ds.y_train.median())
	else:
		y_pred_univ = np.full(len(ds.X_univ), ds.y_train.mean())
	timing.stop("predict_full")

	timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars

	if sales_chase:
		y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
		y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
		y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

	name = "mean"
	if type == "median":
		name = "median"
	if sales_chase:
		name += "*"

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		name,
		ind_var,
		ind_var_test,
		dep_vars,
		None,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)

	return results


def run_naive_sqft(
		ds: DataSplit,
		sales_chase: float = 0.0,
		verbose: bool = False
):
	"""
	Runs a garbage model that simply predicts the median $/sqft of the training set for everything
	:param ds: The data split object containing processed input data
	:param type: The type of average to use ("mean" or "median")
	:param sales_chase: If not 0, simulate sales chasing by predicting the sales set as the test set, with this much noise
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("setup")
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()
	timing.stop("setup")

	timing.start("train")

	X_train = ds.X_train
	# filter out vacant land where bldg_area_finished_sqft is zero:
	X_train_improved = X_train[X_train["bldg_area_finished_sqft"].gt(0)]
	X_train_vacant = X_train[X_train["bldg_area_finished_sqft"].eq(0)]

	ind_per_built_sqft = (ds.y_train / X_train_improved["bldg_area_finished_sqft"]).median()
	ind_per_land_sqft = (ds.y_train / X_train_vacant["land_area_sqft"]).median()
	if pd.isna(ind_per_built_sqft):
		ind_per_built_sqft = 0
	if pd.isna(ind_per_land_sqft):
		ind_per_land_sqft = 0

	if verbose:
		print("Tuning Naive Sqft: searching for optimal parameters...")
		print(f"--> optimal improved $/finished sqft = {ind_per_built_sqft:0.2f}")
		print(f"--> optimal vacant   $/land     sqft = {ind_per_land_sqft:0.2f}")

	timing.stop("train")

	timing.start("predict_test")
	X_test = ds.X_test
	X_test_improved = X_test[X_test["bldg_area_finished_sqft"].gt(0)]
	X_test_vacant = X_test[X_test["bldg_area_finished_sqft"].eq(0)]
	X_test["prediction_impr"] = X_test_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
	X_test["prediction_vacant"] = X_test_vacant["land_area_sqft"] * ind_per_land_sqft
	X_test["prediction"] = np.where(X_test["bldg_area_finished_sqft"].gt(0), X_test["prediction_impr"], X_test["prediction_vacant"])
	y_pred_test = X_test["prediction"].to_numpy()
	X_test.drop(columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True)
	timing.stop("predict_test")

	timing.start("predict_sales")
	X_sales = ds.X_sales
	X_sales_improved = X_sales[X_sales["bldg_area_finished_sqft"].gt(0)]
	X_sales_vacant = X_sales[X_sales["bldg_area_finished_sqft"].eq(0)]
	X_sales["prediction_impr"] = X_sales_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
	X_sales["prediction_vacant"] = X_sales_vacant["land_area_sqft"] * ind_per_land_sqft
	X_sales["prediction"] = np.where(X_sales["bldg_area_finished_sqft"].gt(0), X_sales["prediction_impr"], X_sales["prediction_vacant"])
	y_pred_sales = X_sales["prediction"].to_numpy()
	X_sales.drop(columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True)
	timing.stop("predict_sales")

	timing.start("predict_full")
	X_univ = ds.X_univ
	X_univ_improved = X_univ[X_univ["bldg_area_finished_sqft"].gt(0)]
	X_univ_vacant = X_univ[X_univ["bldg_area_finished_sqft"].eq(0)]
	X_univ["prediction_impr"] = X_univ_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
	X_univ["prediction_vacant"] = X_univ_vacant["land_area_sqft"] * ind_per_land_sqft
	X_univ["prediction"] = np.where(X_univ["bldg_area_finished_sqft"].gt(0), X_univ["prediction_impr"], X_univ["prediction_vacant"])
	y_pred_univ = X_univ["prediction"].to_numpy()
	X_univ.drop(columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True)
	timing.stop("predict_full")

	timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var
	ind_var_test = ds.ind_var_test
	dep_vars = ds.dep_vars

	if sales_chase:
		y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
		y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
		y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

	name = "naive_sqft"
	if sales_chase:
		name += "*"

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		name,
		ind_var,
		ind_var_test,
		dep_vars,
		None,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)

	return results


##### PRIVATE:



def _sales_chase_univ(df_in, ind_var, y_pred_univ):
	"""
	Simulates sales chasing (obviously never use this in a real model, only intended for studying bad behavior)
	:param df_in:
	:param y_pred_univ:
	:return:
	"""
	df_univ = df_in[[ind_var]].copy()
	df_univ["prediction"] = y_pred_univ.copy()
	df_univ.loc[df_univ[ind_var].gt(0), "prediction"] = df_univ[ind_var]
	return df_univ["prediction"].to_numpy()


def _gwr_predict(model, points, P, exog_scale=None, exog_resid=None, fit_params={}):
	"""
	Standalone function for GWR predictions for a larger set of samples in one go.

	Parameters
	----------
	model        : GWR instance
								 The trained GWR model.
	points       : array-like
								 n*2, collection of n sets of (x,y) coordinates for prediction.
	P            : array-like
								 n*k, independent variables for prediction.
	exog_scale   : scalar, optional
								 Estimated scale from the training set. If None, computed from the model.
	exog_resid   : array-like, optional
								 Residuals from the training set. If None, computed from the model.
	fit_params   : dict, optional
								 Parameters for fitting the model.

	Returns
	-------
	results      : dict
								 A dictionary with keys "params" and "predy" containing the predictions.
	"""
	# Use model's fit method to get training scale and residuals if not provided
	if (exog_scale is None) and (exog_resid is None):
		train_gwr = model.fit(**fit_params)
		exog_scale = train_gwr.scale
		exog_resid = train_gwr.resid_response
	elif (exog_scale is not None) and (exog_resid is not None):
		pass  # Use provided scale and residuals
	else:
		raise ValueError("exog_scale and exog_resid must both either be None or specified.")

	# Add intercept column to P if the model includes a constant
	if model.constant:
		P = np.hstack([np.ones((len(P), 1)), P])

	# Perform predictions for all points
	results = Parallel(n_jobs=model.n_jobs)(
		delayed(_local_gwr_predict_external)(
			model, point, predictors
		) for point, predictors in zip(points, P)
	)

	# Extract results
	params = np.array([res[0] for res in results])
	y_pred = np.array([res[1] for res in results])

	return {"params": params, "y_pred": y_pred}


def _local_gwr_predict_external(model, point, predictors):
	# Ensure point and predictors are NumPy arrays
	point = np.asarray(point).reshape(1, -1)  # shape: (1, 2)
	predictors = np.asarray(predictors)

	# Use Kernel with points, giving i=0
	# This tells Kernel: "Compute distances from points[0] to model.coords"
	weights = Kernel(
		0,
		model.coords,
		model.bw,
		fixed=model.fixed,
		function=model.kernel,
		spherical=model.spherical,
		points=point  # Here we pass our prediction point
	).kernel.reshape(-1, 1)

	# Compute local regression betas
	betas, _ = _compute_betas_gwr(model.y, model.X, weights)

	# Predict response
	y_pred = np.dot(predictors, betas)[0]
	return betas.reshape(-1), y_pred


def _run_gwr_prediction(
		coords,
		coords_train,
		X,
		X_train,
		gwr_bw,
		y_train,
		verbose:bool = False
):
	gwr = GWR(coords_train, y_train, X_train, gwr_bw)
	gwr_results = _gwr_predict(gwr, coords, X)
	y_pred = gwr_results["y_pred"]

	return y_pred


def _get_params(name:str, slug:str, ds:DataSplit, tune_func, outpath:str, save_params:bool, use_saved_params:bool, verbose:bool):
	if verbose:
		print(f"Tuning {name}: searching for optimal parameters...")

	params = None
	if use_saved_params:
		if os.path.exists(f"{outpath}/{slug}_params.json"):
			params = json.load(open(f"{outpath}/{slug}_params.json", "r"))
			if verbose:
				print(f"--> using saved parameters: {params}")
	if params is None:
		params = tune_func(ds.X_train, ds.y_train, verbose=verbose)
		if verbose:
			print(f"--> optimal parameters = {params}")
		if save_params:
			os.makedirs(outpath, exist_ok=True)
			json.dump(params, open(f"{outpath}/{slug}_params.json", "w"))
	return params