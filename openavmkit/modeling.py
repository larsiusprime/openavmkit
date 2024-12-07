import math
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
			random_seed: int = 1337
	):
		self.df_universe = df
		self.df_sales = df[df["valid_sale"].eq(1)].reset_index(drop=True)

		# separate df in train & test:
		np.random.seed(random_seed)
		self.df_train = self.df_sales.sample(frac=test_train_frac)
		self.df_test = self.df_sales.drop(self.df_train.index)

		self.df_train = self.df_train.reset_index(drop=True)
		self.df_test = self.df_test.reset_index(drop=True)

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
			y_pred_univ: np.ndarray
	):
		self.type = type
		self.ind_var = ind_var
		self.dep_vars = dep_vars
		self.model = model
		self.pred_test = PredictionResults(ind_var, dep_vars, y_test, y_pred_test)
		self.pred_sales = PredictionResults(ind_var, dep_vars, y_sales, y_pred_sales)
		self.pred_univ = y_pred_univ
		self.chd = quick_median_chd(df, field_prediction, field_horizontal_equity_id)

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
		intercept: bool = True
):
		ds = DataSplit(df, ind_var, dep_vars)

		if intercept:
			ds.X_train = sm.add_constant(ds.X_train)
			ds.X_test = sm.add_constant(ds.X_test)
			ds.X_sales = sm.add_constant(ds.X_sales)
			ds.X_univ = sm.add_constant(ds.X_univ)

		linear_model = sm.OLS(ds.y_train, ds.X_train)
		fitted_model = linear_model.fit()

		# predict on test set:
		y_pred_test = fitted_model.predict(ds.X_test)

		# predict on the sales set:
		y_pred_sales = fitted_model.predict(ds.X_sales)

		# predict on the full set:
		y_pred_univ = fitted_model.predict(ds.X_univ)

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
			y_pred_univ
		)
		return results


def run_gwr(
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str]
):
	ds = DataSplit(df, ind_var, dep_vars)

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

	gwr_selector = Sel_BW(coords_train, y_train, X_train)
	gwr_bw = gwr_selector.search()
	gwr = GWR(coords_train, y_train, X_train, gwr_bw)
	gwr_fit = gwr.fit()
	gwr = gwr_fit.model

	np_coords_test = np.array(coords_test)
	gwr_result_test = gwr.predict(
		np_coords_test,
		X_test
	)
	y_pred_test = gwr_result_test.predictions

	y_pred_sales = _run_gwr_prediction_iterations(
		coords_sales,
		coords_train,
		X_sales,
		X_train,
		gwr_bw,
		y_train
	)

	y_pred_univ = _run_gwr_prediction_iterations(
		coords_univ,
		coords_train,
		X_univ,
		X_train,
		gwr_bw,
		y_train
	)

	y_pred_test = y_pred_test.flatten()
	y_pred_sales = y_pred_sales.flatten()
	y_pred_univ = y_pred_univ.flatten()

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
		y_pred_univ
	)
	return results


def run_xgboost(
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str]
):
	ds = DataSplit(df, ind_var, dep_vars)

	xgboost_model = xgboost.XGBRegressor(
		n_estimators=100,
		max_depth=4,
		learning_rate=0.1,
		random_state=42
	)
	xgboost_model.fit(ds.X_train, ds.y_train)

	y_pred_test = xgboost_model.predict(ds.X_test)
	y_pred_sales = xgboost_model.predict(ds.X_sales)
	y_pred_univ = xgboost_model.predict(ds.X_univ)

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
		y_pred_univ
	)
	return results


def run_lightgbm(
		df: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str]
):
	ds = DataSplit(df, ind_var, dep_vars)

	params = {
		"boosting_type": "gbdt",
		"objective": "regression",
		"metric": {"l2", "l1"},
		"num_leaves": 31,
		"learning_rate": 0.05,
		"feature_fraction": 0.9,
		"bagging_fraction": 0.8,
		"bagging_freq": 5,
		"verbose": 0,
	}

	lgb_train = lgb.Dataset(ds.X_train, ds.y_train)
	lgb_test = lgb.Dataset(ds.X_test, ds.y_test, reference=lgb_train)

	gbm = lgb.train(
		params,
		lgb_train,
		num_boost_round=20,
		valid_sets=lgb_test,
		callbacks=[lgb.early_stopping(stopping_rounds=5)]
	)

	y_pred_test = gbm.predict(ds.X_test, num_iteration=gbm.best_iteration)
	y_pred_sales = gbm.predict(ds.X_sales, num_iteration=gbm.best_iteration)
	y_pred_univ = gbm.predict(ds.X_univ, num_iterations=gbm.best_iteration)

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
		y_pred_univ
	)
	return results


def run_catboost(
		df_in: pd.DataFrame,
		ind_var: str,
		dep_vars: list[str]
):
	df = df_in.copy()
	ds = DataSplit(df, ind_var, dep_vars)

	catboost_model = catboost.CatBoostRegressor(
		iterations=100,
		depth=4,
		learning_rate=0.1,
		loss_function="RMSE"
	)
	catboost_model.fit(ds.X_train, ds.y_train)

	y_pred_test = catboost_model.predict(ds.X_test)
	y_pred_sales = catboost_model.predict(ds.X_sales)
	y_pred_univ = catboost_model.predict(ds.X_univ)

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
		y_pred_univ
	)

	return results


##### PRIVATE:


def _run_gwr_prediction_iterations(
		coords,
		coords_train,
		X,
		X_train,
		gwr_bw,
		y_train
):
	y_pred = np.array([])
	iterations = int(math.ceil(len(coords) / len(coords_train)))
	segment_size = len(coords_train)
	for i in range(0, iterations):
		coords_segment = coords[i*segment_size: (i+1)*segment_size]
		X_segment = X[i*segment_size: (i+1)*segment_size]

		np_coords_segment = np.array(coords_segment)

		gwr = GWR(coords_train, y_train, X_train, gwr_bw)
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
