import logging
import math
import warnings
from typing import Union

import numpy as np
import statsmodels.api as sm
import pandas as pd
import xgboost
import lightgbm as lgb
import catboost
from IPython.core.display_functions import display
from catboost import CatBoostRegressor
from lightgbm import Booster
from mgwr.gwr import MGWR, GWR
from mgwr.sel_bw import Sel_BW
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import RegressionResults
from xgboost import XGBRegressor

from openavmkit.ratio_study import RatioStudy
from openavmkit.stats import quick_median_chd
from openavmkit.utilities.tuning import tune_lightgbm, tune_xgboost, tune_catboost
from openavmkit.utilities.timing import TimingData

PredictionModel = Union[RegressionResults, XGBRegressor, Booster, CatBoostRegressor, GWR, MGWR]

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
		self.ratio_study = RatioStudy(y_pred, y)

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

	def __init__(self,
			df: pd.DataFrame,
			ind_var: str,
			dep_vars: list[str],
			test_train_frac: float = 0.8,
			random_seed: int = 1337,
			days_field: str = "sale_age_days"
	):

		self.df_universe = df
		self.df_sales = df[df["valid_sale"].eq(1)].reset_index(drop=True)

		# Pre-sort dataframes so that rolling origin cross-validation can assume oldest observations first:
		if days_field in self.df_universe:
			self.df_universe.sort_values(by=days_field, ascending=False, inplace=True)
		else:
			raise ValueError(f"Field '{days_field}' not found in dataframe.")

		if days_field in self.df_sales:
			self.df_sales.sort_values(by=days_field, ascending=False, inplace=True)
		else:
			raise ValueError(f"Field '{days_field}' not found in dataframe.")

		# separate df into train & test:
		np.random.seed(random_seed)
		self.df_train = self.df_sales.sample(frac=test_train_frac)
		self.df_test = self.df_sales.drop(self.df_train.index)

		self.df_train = self.df_train.reset_index(drop=True)
		self.df_test = self.df_test.reset_index(drop=True)

		# sort again because sampling shuffles order:
		self.df_train.sort_values(by=days_field, ascending=False, inplace=True)
		self.df_test.sort_values(by=days_field, ascending=False, inplace=True)

		self.X_univ = self.df_universe[dep_vars]

		self.X_sales = self.df_sales[dep_vars]
		self.y_sales = self.df_sales[ind_var]

		self.X_train = self.df_train[dep_vars]
		self.y_train = self.df_train[ind_var]

		self.X_test = self.df_test[dep_vars]
		self.y_test = self.df_test[ind_var]

class ModelResults:
	type: str
	ind_var: str
	dep_vars: list[str]
	model: PredictionModel
	pred_test: PredictionResults
	pred_sales: PredictionResults
	pred_univ: np.ndarray
	chd: float
	timing: TimingData

	def __init__(self,
			df: pd.DataFrame,
			field_prediction: str,
			field_horizontal_equity_id: str,
			type: str,
			ind_var: str,
			dep_vars: list[str],
			model: PredictionModel,
			y_test: pd.Series,
			y_pred_test: np.ndarray,
			y_sales: pd.Series,
			y_pred_sales: np.ndarray,
			y_pred_univ: np.ndarray,
			timing: TimingData
	):
		self.type = type
		self.ind_var = ind_var
		self.dep_vars = dep_vars
		self.model = model
		self.pred_test = PredictionResults(ind_var, dep_vars, y_test, y_pred_test)
		self.pred_sales = PredictionResults(ind_var, dep_vars, y_sales, y_pred_sales)
		self.pred_univ = y_pred_univ
		self.chd = quick_median_chd(df, field_prediction, field_horizontal_equity_id)
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

def run_mra(
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str],
		intercept: bool = True,
		verbose: bool = False
):
		timing = TimingData()
		ds = DataSplit(df, ind_var, dep_vars)

		timing.start("total")

		timing.start("setup")
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
		timing.stop("train")

		# predict on test set:
		timing.start("predict_test")
		y_pred_test = fitted_model.predict(ds.X_test)
		timing.stop("predict_test")

		# predict on the sales set:
		timing.start("predict_sales")
		y_pred_sales = fitted_model.predict(ds.X_sales)
		timing.stop("predict_sales")

		# predict on the full set:
		timing.start("predict_full")
		y_pred_univ = fitted_model.predict(ds.X_univ)
		timing.stop("predict_full")

		timing.stop("total")

		# gather the predictions
		df["prediction"] = y_pred_univ
		results = ModelResults(
			df,
			"prediction",
			"he_id",
			"mra",
			ind_var,
			dep_vars,
			fitted_model,
			ds.y_test,
			y_pred_test,
			ds.y_sales,
			y_pred_sales,
			y_pred_univ,
			timing
		)
		return results


def run_gwr(
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str],
		verbose: bool = False
):
	timing = TimingData()
	ds = DataSplit(df, ind_var, dep_vars)

	timing.start("total")

	timing.start("setup")
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

	y_train = ds.y_train.values.reshape((-1, 1))

	X_train = ds.X_train.values
	X_test = ds.X_test.values
	X_sales = ds.X_sales.values
	X_univ = ds.X_univ.values
	timing.stop("setup")

	timing.start("parameter_search")
	if verbose:
		print("Tuning GWR: searching for optimal bandwidth...")
	gwr_selector = Sel_BW(coords_train, y_train, X_train)
	gwr_bw = gwr_selector.search()
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
	y_pred_sales = _run_gwr_prediction_iterations(
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
	y_pred_univ = _run_gwr_prediction_iterations(
		coords_univ,
		coords_train,
		X_univ,
		X_train,
		gwr_bw,
		y_train,
		verbose=verbose
	).flatten()
	timing.stop("predict_full")

	timing.stop("total")

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"gwr",
		ind_var,
		dep_vars,
		gwr,
		ds.y_test,
		y_pred_test,
		ds.y_sales,
		y_pred_sales,
		y_pred_univ,
		timing
	)
	return results


def run_xgboost(
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str],
		verbose: bool = False
):
	timing = TimingData()
	ds = DataSplit(df, ind_var, dep_vars)

	timing.start("total")

	timing.start("setup")
	timing.stop("setup")

	timing.start("parameter_search")
	params = tune_xgboost(ds.X_train, ds.y_train, verbose=verbose)
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

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"xgboost",
		ind_var,
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
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str],
		verbose: bool = False
):
	timing = TimingData()
	ds = DataSplit(df, ind_var, dep_vars)

	timing.start("total")

	timing.start("setup")
	timing.stop("setup")

	timing.start("parameter_search")
	params = tune_lightgbm(ds.X_train, ds.y_train, verbose=verbose)
	timing.stop("parameter_search")

	timing.start("train")
	lgb_train = lgb.Dataset(ds.X_train, ds.y_train)
	lgb_test = lgb.Dataset(ds.X_test, ds.y_test, reference=lgb_train)

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

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"lightgbm",
		ind_var,
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
		df_in: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str],
		verbose: bool = False
):
	timing = TimingData()
	df = df_in.copy()
	ds = DataSplit(df, ind_var, dep_vars)

	timing.start("total")

	timing.start("parameter_search")
	params = tune_catboost(ds.X_train, ds.y_train, verbose=verbose)
	timing.stop("parameter_search")

	timing.start("setup")

	print(params)

	catboost_model = catboost.CatBoostRegressor(**params)
	timing.stop("setup")

	timing.start("train")
	catboost_model.fit(ds.X_train, ds.y_train)
	timing.stop("train")

	timing.start("predict_test")
	y_pred_test = catboost_model.predict(ds.X_test)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales = catboost_model.predict(ds.X_sales)
	timing.stop("predict_sales")

	timing.start("predict_full")
	y_pred_univ = catboost_model.predict(ds.X_univ)
	timing.stop("predict_full")

	timing.stop("total")

	df["prediction"] = y_pred_univ
	results = ModelResults(
		df,
		"prediction",
		"he_id",
		"catboost",
		ind_var,
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


##### PRIVATE:


def _run_gwr_prediction_iterations(
		coords,
		coords_train,
		X,
		X_train,
		gwr_bw,
		y_train,
		verbose:bool = False
):
	y_pred = np.array([])
	iterations = int(math.ceil(len(coords) / len(coords_train)))
	segment_size = len(coords_train)
	for i in range(0, iterations):
		coords_segment = coords[i*segment_size: (i+1)*segment_size]
		X_segment = X[i*segment_size: (i+1)*segment_size]

		np_coords_segment = np.array(coords_segment)

		gwr = GWR(coords_train, y_train, X_train, gwr_bw)

		# TODO: you have to reconstruct the model each time you predict
		# this is for two reasons:
		# 1. the model can only predict a batch size as large as the training run
		# 2. you can't call predict() more than once
		# this works around those limitations for now, at the cost of inflated prediction time

		if verbose:
			print(f"--> GWR prediction iteration {i+1}/{iterations}")

		gwr_fit = gwr.fit()
		gwr = gwr_fit.model
		gwr_result_segment = gwr.predict(
			np_coords_segment,
			X_segment
		)
		if i == 0:
			y_pred = gwr_result_segment.predictions
		else:
			y_pred = np.concatenate((y_pred, gwr_result_segment.predictions))
	return y_pred
