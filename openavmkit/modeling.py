import json
import os
import pickle

import polars as pl
from IPython.core.display_functions import display
from joblib import Parallel, delayed
from typing import Union


import numpy as np
import statsmodels.api as sm
import pandas as pd
import geopandas as gpd
import xgboost
import lightgbm as lgb
import catboost
from catboost import CatBoostRegressor, Pool
from lightgbm import Booster
from matplotlib import pyplot as plt, ticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from mgwr.gwr import GWR
from mgwr.gwr import _compute_betas_gwr, Kernel

from mgwr.sel_bw import Sel_BW
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.regression.linear_model import RegressionResults
from xgboost import XGBRegressor

from openavmkit.data import get_sales, simulate_removed_buildings
from openavmkit.ratio_study import RatioStudy
from openavmkit.utilities.format import fancy_format
from openavmkit.utilities.geometry import select_grid_size_from_size_str
from openavmkit.utilities.modeling import GarbageModel, AverageModel, NaiveSqftModel, LocalSqftModel, AssessorModel, \
	GWRModel, MRAModel
from openavmkit.utilities.data import clean_column_names, div_field_z_safe
from openavmkit.utilities.stats import quick_median_chd
from openavmkit.tuning import tune_lightgbm, tune_xgboost, tune_catboost
from openavmkit.utilities.timing import TimingData

PredictionModel = Union[
	MRAModel,
	XGBRegressor,
	Booster,
	CatBoostRegressor,
	GWR,
	KernelReg,
	GarbageModel,
	AverageModel,
	NaiveSqftModel,
	LocalSqftModel,
	AssessorModel,
	GWRModel,
	str,
	None
]

class PredictionResults:

	def __init__(self,
			ind_var: str,
			dep_vars: list[str],
			prediction_field: str,
			df: pd.DataFrame
	):
		self.ind_var = ind_var
		self.dep_vars = dep_vars

		y = df[ind_var].to_numpy()
		y_pred = df[prediction_field].to_numpy()

		self.y = y
		self.y_pred = y_pred

		df_valid = df[df["valid_for_ratio_study"].eq(True)]

		y = df_valid[ind_var].to_numpy()
		y_pred = df_valid[prediction_field].to_numpy()

		y_clean = y[~pd.isna(y_pred)]
		y_pred_clean = y_pred[~pd.isna(y_pred)]

		if len(y_clean) > 0:
			self.mse = mean_squared_error(y_clean, y_pred_clean)
			self.rmse = np.sqrt(self.mse)
			self.r2 = 1 - self.mse / np.var(y_clean)
		else:
			self.mse = float('nan')
			self.rmse = float('nan')
			self.r2 = float('nan')

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
		self.ratio_study = RatioStudy(y_pred_clean, y_clean)

class DataSplit:

	counter: int = 0

	def __init__(self,
			df: pd.DataFrame | None,
      model_group: str,
			settings: dict,
			ind_var: str,
			ind_var_test: str,
			dep_vars: list[str],
			categorical_vars: list[str],
			interactions: dict,
      test_keys: list[str],
      train_keys: list[str],
			vacant_only: bool = False,
			hedonic: bool = False,
			days_field: str = "sale_age_days",
      df_multiverse: pd.DataFrame | None = None,
			init: bool = True
  ):
    if not init:
      return

    self.settings = settings.copy()

    # An *unmodified* copy of the original model group universe, that will remain unmodified
    self.df_universe_orig = df.copy()
    self.df_universe = df.copy()

    # The parcel "multiverse" is a parcel universe that contains *all* model groups, not just the current model group
    self.df_multiverse_orig = None
    self.df_multiverse = None
    if df_multiverse is not None:
      self.df_multiverse_orig = df_multiverse.copy()
      self.df_multiverse = df_multiverse.copy()

    self.df_sales = get_sales(df, settings, vacant_only).reset_index(drop=True)

    self._df_sales = self.df_sales.copy()

    self.test_keys = test_keys
    self.train_keys = train_keys

			if hedonic:
				# transform df_universe & df_sales such that all improved characteristics are removed
      self.df_universe = simulate_removed_buildings(self.df_universe, settings)
      self.df_sales = simulate_removed_buildings(self.df_sales, settings)
      if self.df_multiverse is not None:
        self.df_multiverse = simulate_removed_buildings(self.df_multiverse, settings)

    # we also need to limit the sales set, but we can't do that AFTER we've split

			# Pre-sort dataframes so that rolling origin cross-validation can assume oldest observations first:
			if days_field in self.df_universe:
				self.df_universe.sort_values(by=days_field, ascending=False, inplace=True)
			else:
				raise ValueError(f"Field '{days_field}' not found in dataframe.")

			if days_field in self.df_sales:
				self.df_sales.sort_values(by=days_field, ascending=False, inplace=True)
			else:
				raise ValueError(f"Field '{days_field}' not found in dataframe.")

    self.model_group = model_group
			self.ind_var = ind_var
			self.ind_var_test = ind_var_test
			self.dep_vars = dep_vars.copy()
			self.categorical_vars = categorical_vars.copy()
			self.interactions = interactions.copy()
    self.one_hot_descendants = {}
    self.vacant_only = vacant_only
    self.hedonic = hedonic
    self.days_field = days_field
    self.split()


	def copy(self):
		# Return a deep copy
		ds = DataSplit(
			None,
      "",
			{},
			"",
			"",
			[],
			[],
      {},
      [],
      [],
      False,
			False,
			"",
			init=False
		)
    # manually copy every field:
    ds.settings = self.settings.copy()
    ds.model_group = self.model_group
    ds.df_sales = self.df_sales.copy()
    ds.df_universe = self.df_universe.copy()
    ds.df_universe_orig = self.df_universe_orig.copy()
    ds._df_sales = self._df_sales.copy()
    ds.df_train = self.df_train.copy()
		ds.df_test = self.df_test.copy()
		ds.X_univ = self.X_univ.copy()
		ds.X_sales = self.X_sales.copy()
		ds.y_sales = self.y_sales.copy()
		ds.X_train = self.X_train.copy()
		ds.y_train = self.y_train.copy()
		ds.X_test = self.X_test.copy()
    ds.y_test = self.y_test.copy()
    ds.test_keys = self.test_keys.copy()
    ds.train_keys = self.train_keys.copy()
    ds.vacant_only = self.vacant_only
    ds.hedonic = self.hedonic
		ds.ind_var = self.ind_var
		ds.ind_var_test = self.ind_var_test
		ds.dep_vars = self.dep_vars.copy()
    ds.categorical_vars = self.categorical_vars.copy()
    ds.interactions = self.interactions.copy()
    ds.one_hot_descendants = self.one_hot_descendants.copy()
    ds.days_field = self.days_field

    if self.df_multiverse is not None:
      ds.df_multiverse = self.df_multiverse.copy()
      ds.df_multiverse_orig = self.df_multiverse_orig.copy()
    else:
      ds.df_multiverse = None
      ds.df_multiverse_orig = None
    return ds


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

    if ds.df_multiverse is not None:
      ds.df_multiverse = pd.get_dummies(ds.df_multiverse, columns=[col for col in cat_vars if col in ds.df_multiverse], drop_first=True)
      ds.df_multiverse = clean_column_names(ds.df_multiverse)
      ds.df_multiverse = ds.df_multiverse.drop(columns=[col for col in cat_vars if col in ds.df_multiverse])

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
    if ds.df_multiverse is not None:
      ds.df_multiverse = ds.df_multiverse[ds.df_train.columns]

    test_cols = [col for col in ds.df_train.columns if col in ds.df_test.columns]
    extra_cols = [col for col in ds.df_train.columns if col not in test_cols]

		ds.df_test = ds.df_test[test_cols]
		# add extra_cols, set them to zero:
		for col in extra_cols:
			ds.df_test[col] = 0.0

		return ds


  def split(self):

    test_keys = self.test_keys
    train_keys = self.train_keys

    # separate df into train & test:

    # select the rows that are in the test_keys:
    self.df_test = self._df_sales[self._df_sales["key"].astype(str).isin(test_keys)].reset_index(drop=True)
    self.df_train = self._df_sales.drop(self.df_test.index)

    self.df_test = self.df_test.reset_index(drop=True)
    self.df_train = self.df_train.reset_index(drop=True)

    # sort again because sampling shuffles order:
    self.df_test.sort_values(by=self.days_field, ascending=False, inplace=True)
    self.df_train.sort_values(by=self.days_field, ascending=False, inplace=True)

		if self.hedonic:
			# if it's a hedonic model, we're predicting land value, and are thus testing against vacant land only:
			# we have to do this here, AFTER the split, to ensure that the selected rows are from the same subsets

			# get the sales that are actually vacant, from the original set of sales
			_df_sales = get_sales(self._df_sales, self.settings, True).reset_index(drop=True)

			# now, select only those records from the modified base sales set that are also in the above set,
			# but use the rows from the modified base sales set
			_df_sales = self.df_sales[self.df_sales["key"].isin(_df_sales["key"])].reset_index(drop=True)

			# use these as our sales
			self.df_sales = _df_sales

			# set df_test/train to only those rows that are also in sales:
			# we don't need to use get_sales() because they've already been transformed to vacant
			self.df_test = self.df_test[self.df_test["key"].isin(self.df_sales["key"])].reset_index(drop=True)
			self.df_train = self.df_train[self.df_train["key"].isin(self.df_sales["key"])].reset_index(drop=True)

    _df_univ = self.df_universe.copy()
    _df_sales = self.df_sales.copy()
    _df_train = self.df_train.copy()
    _df_test = self.df_test.copy()
    _df_multi = self.df_multiverse.copy() if self.df_multiverse is not None else None

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
          if _df_multi is not None and target_field in _df_multi:
            _df_multi[target_field] = _df_multi[target_field] * _df_multi[fill_field]

    dep_vars = [col for col in self.dep_vars if col in _df_univ.columns]
    self.X_univ = _df_univ[dep_vars]

    dep_vars = [col for col in self.dep_vars if col in _df_sales.columns]
		self.X_sales = _df_sales[dep_vars]
		self.y_sales = _df_sales[self.ind_var]

		dep_vars = [col for col in self.dep_vars if col in _df_train.columns]
    self.X_train = _df_train[dep_vars]
    self.y_train = _df_train[self.ind_var]

    if _df_multi is not None:
      dep_vars = [col for col in self.dep_vars if col in _df_multi.columns]
      self.X_multiverse = _df_multi[dep_vars]
      self.y_multiverse = _df_multi[self.ind_var]
    else:
      self.X_multiverse = None
      self.y_multiverse = None

    # convert all Float64 to float64 in X_train:
    for col in self.X_train.columns:
      # if it's a Float64 or a boolean, convert it to float64
      if (self.X_train[col].dtype == "Float64" or
          self.X_train[col].dtype == "Int64" or
					self.X_train[col].dtype == "boolean" or
					self.X_train[col].dtype == "bool"
			):
				self.X_train.loc[:, col] = self.X_train[col].astype("float64")

		dep_vars = [col for col in self.dep_vars if col in _df_test.columns]
		self.X_test = _df_test[dep_vars]
		self.y_test = _df_test[self.ind_var_test]


class SingleModelResults:

	def __init__(self,
			ds: DataSplit,
			field_prediction: str,
			field_horizontal_equity_id: str,
			type: str,
			model: PredictionModel,
			y_pred_test: np.ndarray,
			y_pred_sales: np.ndarray | None,
			y_pred_univ: np.ndarray,
      timing: TimingData,
      verbose: bool = False,
      y_pred_multi: np.ndarray | None = None
	):
		self.ds = ds

		df_univ = ds.df_universe.copy()
		df_sales = ds.df_sales.copy()
    df_test = ds.df_test.copy()

    df_univ[field_prediction] = y_pred_univ
    df_test[field_prediction] = y_pred_test

    self.df_universe = df_univ
    self.df_test = df_test

    if y_pred_sales is not None:
      df_sales[field_prediction] = y_pred_sales
      self.df_sales = df_sales

    if y_pred_multi is not None and ds.df_multiverse is not None:
      df_multi = ds.df_multiverse.copy()
      df_multi[field_prediction] = y_pred_multi
      self.df_multiverse = df_multi
    else:
      df_multi = None
      self.df_multiverse = None

    self.type = type
    self.ind_var = ds.ind_var
		self.ind_var_test = ds.ind_var_test
		self.dep_vars = ds.dep_vars.copy()
		self.model = model

		timing.start("stats_test")
		self.pred_test = PredictionResults(self.ind_var_test, self.dep_vars, field_prediction, df_test)
		timing.stop("stats_test")

		timing.start("stats_sales")
		if y_pred_sales is not None:
			self.pred_sales = PredictionResults(self.ind_var_test, self.dep_vars, field_prediction, df_sales)
		timing.stop("stats_sales")

		self.pred_univ = y_pred_univ
		timing.start("chd")
		df_univ_valid = df_univ

		df_univ_valid = pd.DataFrame(df_univ_valid)  # Ensure it's a Pandas DataFrame

		# drop problematic columns:
		df_univ_valid.drop(columns=["geometry"], errors="ignore", inplace=True)

		# convert all category and string[python] types to string:
		for col in df_univ_valid.columns:
			if df_univ_valid[col].dtype in ["category", "string"]:
				df_univ_valid[col] = df_univ_valid[col].astype("str")

		pl_df = pl.DataFrame(df_univ_valid)
		self.chd = quick_median_chd(pl_df, field_prediction, field_horizontal_equity_id)
		timing.stop("chd")

		timing.start("utility")
		self.utility = model_utility_score(self)
		timing.stop("utility")
		self.timing = timing

	def summary(self):
		str = ""

		str += f"Model type: {self.type}\n"
		# Print the # of rows in test & full set
		# Print the MSE, RMSE, R2, and Adj R2 for test & full set
		str += f"-->Test set, rows: {len(self.pred_test.y)}\n"
		str += f"---->RMSE   : {self.pred_test.rmse:8.0f}\n"
		str += f"---->R2     : {self.pred_test.r2:8.4f}\n"
		str += f"---->Adj R2 : {self.pred_test.adj_r2:8.4f}\n"
		str += f"---->M.Ratio: {self.pred_test.ratio_study.median_ratio:8.4f}\n"
		str += f"---->COD    : {self.pred_test.ratio_study.cod:8.4f}\n"
		str += f"---->PRD    : {self.pred_test.ratio_study.prd:8.4f}\n"
		str += f"---->PRB    : {self.pred_test.ratio_study.prb:8.4f}\n"
		str += f"\n"
		str += f"-->Full set, rows: {len(self.pred_sales.y)}\n"
		str += f"---->RMSE   : {self.pred_sales.rmse:8.0f}\n"
		str += f"---->R2     : {self.pred_sales.r2:8.4f}\n"
		str += f"---->Adj R2 : {self.pred_sales.adj_r2:8.4f}\n"
		str += f"---->M.Ratio: {self.pred_sales.ratio_study.median_ratio:8.4f}\n"
		str += f"---->COD    : {self.pred_sales.ratio_study.cod:8.4f}\n"
		str += f"---->PRD    : {self.pred_sales.ratio_study.prd:8.4f}\n"
    str += f"---->PRB    : {self.pred_sales.ratio_study.prb:8.4f}\n"
    str += f"---->CHD    : {self.chd:8.4f}\n"
    str += f"\n"
    return str


def model_utility_score(
		model_results: SingleModelResults
):
	# We want to minimize:
	# 1. error
	# 2. the difference between the median ratio and 1
  # 3. the COD
  # 4. the CHD

  # LOWER IS BETTER

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


def predict_mra(
		ds: DataSplit,
		model: MRAModel,
		timing: TimingData,
		verbose: bool = False
):
	fitted_model: RegressionResults = model.fitted_model

	# predict on test set:
	timing.start("predict_test")
	y_pred_test = fitted_model.predict(ds.X_test).to_numpy()
	timing.stop("predict_test")

	# predict on the sales set:
	timing.start("predict_sales")
	y_pred_sales = fitted_model.predict(ds.X_sales).to_numpy()
  timing.stop("predict_sales")

  # predict on the full set:
  timing.start("predict_univ")
  y_pred_univ = fitted_model.predict(ds.X_univ).to_numpy()
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    y_pred_multi = fitted_model.predict(ds.X_multiverse).to_numpy()
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    "mra",
    model,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
  )

	return results


def run_mra(
		ds: DataSplit,
		intercept: bool = True,
		verbose: bool = False,
		model: MRAModel | None = None
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
      if ds.X_multiverse is not None:
        ds.X_multiverse = sm.add_constant(ds.X_multiverse)
		timing.stop("setup")

		timing.start("parameter_search")
		timing.stop("parameter_search")

		ds.X_train = ds.X_train.astype(float)
		ds.y_train = ds.y_train.astype(float)

		timing.start("train")
		if model is None:
			linear_model = sm.OLS(ds.y_train, ds.X_train)
			fitted_model = linear_model.fit()
			model = MRAModel(fitted_model, intercept)
		timing.stop("train")

		return predict_mra(ds, model, timing, verbose)


def predict_assessor(
		ds: DataSplit,
		assr_model: AssessorModel,
		timing: TimingData,
		verbose: bool = False
):
	field = assr_model.field
	if ds.hedonic:
		field = ds.dep_vars[0]

	# predict on test set:
	timing.start("predict_test")
	y_pred_test = ds.X_test[field].to_numpy()
	timing.stop("predict_test")

	# predict on the sales set:
	timing.start("predict_sales")
	y_pred_sales = ds.X_sales[field].to_numpy()
	timing.stop("predict_sales")

  # predict on the full set:
  timing.start("predict_univ")
  y_pred_univ = ds.X_univ[field].to_numpy()
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    y_pred_multi = ds.X_multiverse[field].to_numpy()
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    "assessor",
    assr_model,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
  )

  return results


def run_assessor(
		ds: DataSplit,
		verbose: bool = False
):
	timing = TimingData()

	timing.start("total")

	timing.start("setup")
	ds.split()
	timing.stop("setup")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("train")
	timing.stop("train")

	assr_model = AssessorModel(ds.dep_vars[0])
	return predict_assessor(ds, assr_model, timing, verbose)


def predict_kernel(
		ds: DataSplit,
		kr: KernelReg,
		timing: TimingData,
		verbose: bool = False
):

	u_test = ds.df_test['longitude']
	v_test = ds.df_test['latitude']

	u_sales = ds.df_sales['longitude']
	v_sales = ds.df_sales['latitude']

	u = ds.df_universe['longitude']
	v = ds.df_universe['latitude']

	vars_test = (u_test, v_test)
	for col in ds.X_test.columns:
		vars_test += (ds.X_test[col].to_numpy(),)

	vars_sales = (u_sales, v_sales)
	for col in ds.X_sales.columns:
		vars_sales += (ds.X_sales[col].to_numpy(),)

	vars_univ = (u, v)
	for col in ds.X_univ.columns:
		vars_univ += (ds.X_univ[col].to_numpy(),)

	X_test = np.column_stack(vars_test)
	X_sales = np.column_stack(vars_sales)
  X_univ = np.column_stack(vars_univ)


  if ds.df_multiverse is not None:
    u_multi = ds.df_multiverse['longitude']
    v_multi = ds.df_multiverse['latitude']
    vars_multi = (u_multi, v_multi)
    for col in ds.X_multiverse.columns:
      vars_multi += (ds.X_multiverse[col].to_numpy(),)
    X_multi = np.column_stack(vars_multi)
  else:
    u_multi = None
    v_multi = None
    X_multi = None

  if verbose:
    print(f"--> predicting on test set...")
  # Predict at original locations:
	timing.start("predict_test")
	y_pred_test, _ = kr.fit(X_test, verbose=verbose)
	timing.stop("predict_test")

	if verbose:
		print(f"--> predicting on sales set...")
	timing.start("predict_sales")
	y_pred_sales, _ = kr.fit(X_sales, verbose=verbose)
	timing.stop("predict_sales")

  if verbose:
    print(f"--> predicting on full set...")
  timing.start("predict_univ")
  y_pred_univ, _ = kr.fit(X_univ, verbose=verbose)
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    y_pred_multi, _ = kr.fit(X_multi, verbose=verbose)
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"mra",
		kr,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
  )

	return results


def run_kernel(
		ds: DataSplit,
		outpath: str,
		save_params: bool = False,
		use_saved_params: bool = False,
		verbose: bool = False
):
	timing = TimingData()

	timing.start("total")

	timing.start("setup")
	ds = ds.encode_categoricals_with_one_hot()
	ds.split()
	u_train = ds.df_train['longitude']
	v_train = ds.df_train['latitude']

	vars_train = (u_train, v_train)
	for col in ds.X_train.columns:
		vars_train += (ds.X_train[col].to_numpy(),)

	X_train = np.column_stack(vars_train)

	y_train = ds.y_train.to_numpy()
	timing.stop("setup")

	timing.start("parameter_search")

	kernel_bw = None
	if use_saved_params:
		if os.path.exists(f"{outpath}/kernel_bw.pkl"):
			with open(f"{outpath}/kernel_bw.pkl", "rb") as f:
				kernel_bw = pickle.load(f)
				# if kernel_bw is not the same length as the number of variables:
				if len(kernel_bw) != X_train.shape[1]:
					print(f"-->saved bandwidth ({len(kernel_bw)} does not match the number of variables ({X_train.shape[1]}), regenerating...")
					kernel_bw = None
			if verbose:
				print(f"--> using saved bandwidth: {kernel_bw}")
	if kernel_bw is None:
		kernel_bw = "cv_ls"
		if verbose:
			print(f"--> searching for optimal bandwidth...")
	timing.stop("parameter_search")

	timing.start("train")
	# TODO: can adjust this to handle categorical data better
	var_type = "c" * X_train.shape[1]
	defaults = EstimatorSettings(efficient=True)
	kr = KernelReg(endog=y_train, exog=X_train, var_type=var_type, bw=kernel_bw, defaults=defaults)
	kernel_bw = kr.bw
	if save_params:
		os.makedirs(outpath, exist_ok=True)
		with open(f"{outpath}/kernel_bw.pkl", "wb") as f:
			pickle.dump(kernel_bw, f)
	if verbose:
		print(f"--> optimal bandwidth = {kernel_bw}")
	timing.stop("train")

	return predict_kernel(ds, kr, timing, verbose)


def predict_gwr(
		ds: DataSplit,
    gwr_model: GWRModel,
    timing: TimingData,
    verbose: bool,
    diagnostic: bool = False,
    intercept: bool = True
):
	timing.start("train")
	# You have to re-train GWR before each prediction, so we move training to the predict function
	gwr = GWR(gwr_model.coords_train, gwr_model.y_train, gwr_model.X_train, gwr_model.gwr_bw)
	gwr.fit()
	timing.stop("train")

	gwr_bw = gwr_model.gwr_bw
	coords_train = gwr_model.coords_train
	X_train = gwr_model.X_train
	y_train = gwr_model.y_train

	X_test = ds.X_test.values
	X_test = X_test.astype(np.float64)

	X_sales = ds.X_sales.values
	X_univ = ds.X_univ.values
	X_sales = X_sales.astype(np.float64)
	X_univ = X_univ.astype(np.float64)

	u_test = ds.df_test['longitude']
	v_test = ds.df_test['latitude']
	coords_test = list(zip(u_test, v_test))

	u_sales = ds.df_sales['longitude']
	v_sales = ds.df_sales['latitude']
	coords_sales = list(zip(u_sales, v_sales))

	u = ds.df_universe['longitude']
	v = ds.df_universe['latitude']
  coords_univ = list(zip(u,v))

  if ds.df_multiverse is not None:
    X_multi = ds.X_multiverse.values
    X_multi = X_multi.astype(np.float64)
    u_multi = ds.df_multiverse['longitude']
    v_multi = ds.df_multiverse['latitude']
    coords_multi = list(zip(u_multi, v_multi))
  else:
    X_multi = None
    coords_multi = None

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
    plot=False,
    intercept=intercept
	).flatten()
	timing.stop("predict_sales")

  timing.start("predict_univ")
	if verbose:
		print("GWR: predicting full set...")
	y_pred_univ = _run_gwr_prediction(
		coords_univ,
    coords_train,
    X_univ,
    X_train,
    gwr_bw,
    y_train,
    plot=True,
    intercept=intercept,
    gdf=ds.df_universe,
    dep_vars=ds.dep_vars
  ).flatten()
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    if verbose:
      print("GWR: predicting multiverse...")
    y_pred_multi = _run_gwr_prediction(
      coords_multi,
      coords_train,
      X_multi,
      X_train,
      gwr_bw,
      y_train,
      plot=False,
      intercept=intercept
    ).flatten()
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  model_name = "gwr"
  if diagnostic:
    model_name = "diagnostic_gwr"

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    model_name,
    gwr_model,
		y_pred_test,
		y_pred_sales,
    y_pred_univ,
    timing,
    y_pred_multi=y_pred_multi
	)
	timing.stop("total")

	return results


def run_gwr(
		ds: DataSplit,
		outpath: str,
		save_params: bool = False,
		use_saved_params: bool = False,
    verbose: bool = False,
    diagnostic: bool = False
):
  """
  Runs a GWR model
  :param ds: The data split object containing processed input data
  :param outpath: The output path
  :param save_params: Whether to save the parameters
  :param use_saved_params: Whether to use saved parameters
  :param verbose: Whether to print verbose output
  :param diagnostic:
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

	y_train = ds.y_train.to_numpy().reshape((-1, 1))

	X_train = ds.X_train.values

	# add a very small amount of random noise to every row in every column of X_train:
	# this is to prevent singular matrix errors in the GWR
	X_train += np.random.normal(0, 1e-6, X_train.shape)

	# ensure that every dtype of every column in X_* is a float and not an object:
	X_train = X_train.astype(np.float64)

  # ensure that every dtype of y_train is a float and not an object:
  y_train = y_train.astype(np.float64)

  timing.stop("setup")

  model_name = "gwr"
  if diagnostic:
    model_name = "diagnostic_gwr"

  timing.start("parameter_search")
	gwr_bw = -1.0

	if verbose:
		print("Tuning GWR: searching for optimal bandwidth...")

	if use_saved_params:
    if os.path.exists(f"{outpath}/gwr_bw.json"):
      gwr_bw = json.load(open(f"{outpath}/{model_name}_bw.json", "r"))
      if verbose:
        print(f"--> using saved bandwidth: {gwr_bw:0.2f}")

  if gwr_bw < 0:
    gwr_selector = Sel_BW(coords_train, y_train, X_train)
    gwr_bw = gwr_selector.search()

    if save_params:
      os.makedirs(outpath, exist_ok=True)
      json.dump(gwr_bw, open(f"{outpath}/{model_name}_bw.json", "w"))
    if verbose:
			print(f"--> optimal bandwidth = {gwr_bw:0.2f}")

	timing.stop("parameter_search")

	X_train = np.asarray(X_train, dtype=np.float64)

	gwr_model = GWRModel(coords_train, X_train, y_train, gwr_bw)

  return predict_gwr(ds, gwr_model, timing, verbose, diagnostic)


def predict_xgboost(
		ds: DataSplit,
		xgboost_model: xgboost.XGBRegressor,
		timing: TimingData,
		verbose: bool = False
):
	timing.start("predict_test")
	y_pred_test = xgboost_model.predict(ds.X_test)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales = xgboost_model.predict(ds.X_sales)
	timing.stop("predict_sales")

  timing.start("predict_univ")
  y_pred_univ = xgboost_model.predict(ds.X_univ)
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    y_pred_multi = xgboost_model.predict(ds.X_multiverse)
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"xgboost",
		xgboost_model,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
		timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
	)
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

	return predict_xgboost(
		ds,
		xgboost_model,
		timing,
		verbose
	)


def predict_lightgbm(
		ds: DataSplit,
		gbm: lgb.Booster,
		timing: TimingData,
		verbose: bool = False
):
	timing.start("predict_test")
	y_pred_test = gbm.predict(ds.X_test, num_iteration=gbm.best_iteration)
	timing.stop("predict_test")

	timing.start("predict_sales")
  y_pred_sales = gbm.predict(ds.X_sales, num_iteration=gbm.best_iteration)
  timing.stop("predict_sales")

  timing.start("predict_univ")
  y_pred_univ = gbm.predict(ds.X_univ, num_iterations=gbm.best_iteration)
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    y_pred_multi = gbm.predict(ds.X_multiverse, num_iterations=gbm.best_iteration)
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"lightgbm",
		gbm,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
		timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
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
	lgb_test = lgb.Dataset(ds.X_test,  ds.y_test,  categorical_feature=cat_vars, reference=lgb_train)

	params["verbosity"] = -1

	num_boost_round = 1000
	if "num_iterations" in params:
		num_boost_round = params.pop("num_iterations")

	gbm = lgb.train(
		params,
		lgb_train,
		num_boost_round=num_boost_round,
		valid_sets=[lgb_test],
		callbacks=[
			lgb.early_stopping(stopping_rounds=5, verbose=False),
			lgb.log_evaluation(period=0)
		]
	)
	timing.stop("train")

	return predict_lightgbm(ds, gbm, timing, verbose)


def predict_catboost(
		ds: DataSplit,
		catboost_model: catboost.CatBoostRegressor,
		timing: TimingData,
		verbose: bool = False
):
	cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]

	test_pool = Pool(data=ds.X_test, label=ds.y_test, cat_features=cat_vars)
	sales_pool = Pool(data=ds.X_sales, label=ds.y_sales, cat_features=cat_vars)
	univ_pool = Pool(data=ds.X_univ, cat_features=cat_vars)

	timing.start("predict_test")
	y_pred_test = catboost_model.predict(test_pool)
	timing.stop("predict_test")

	timing.start("predict_sales")
	y_pred_sales = catboost_model.predict(sales_pool)
	timing.stop("predict_sales")

  timing.start("predict_univ")
  y_pred_univ = catboost_model.predict(univ_pool)
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    multi_pool = Pool(data=ds.X_multiverse, cat_features=cat_vars)
    y_pred_multi = catboost_model.predict(multi_pool)
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		"catboost",
		catboost_model,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
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

	timing.stop("setup")

	timing.start("train")
	catboost_model.fit(train_pool)
	timing.stop("train")

	return predict_catboost(ds, catboost_model, timing, verbose)


def predict_garbage(
		ds: DataSplit,
		garbage_model: GarbageModel,
		timing: TimingData,
		verbose: bool = False
):
	timing.start("predict_test")
	normal = garbage_model.normal
	min_value = garbage_model.min_value
	max_value = garbage_model.max_value
	sales_chase = garbage_model.sales_chase

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

  timing.start("predict_univ")
  if normal:
    y_pred_univ = np.random.normal(loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_univ))
  else:
    y_pred_univ = np.random.uniform(min_value, max_value, len(ds.X_univ))
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    if normal:
      y_pred_multi = np.random.normal(loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_multiverse))
    else:
      y_pred_multi = np.random.uniform(min_value, max_value, len(ds.X_multiverse))
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

	timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var

	if sales_chase:
		y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
		y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))
    if y_pred_multi is not None:
      y_pred_multi = _sales_chase_univ(df, ind_var, y_pred_multi) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_multi))

	name = "garbage"
	if normal:
		name = "garbage_normal"
	if sales_chase:
		name += "*"

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		name,
		garbage_model,
		y_pred_test,
		y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
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

	garbage_model = GarbageModel(min_value, max_value, sales_chase, normal)

	return predict_garbage(ds, garbage_model, timing, verbose)


def predict_average(
		ds: DataSplit,
		average_model: AverageModel,
		timing: TimingData,
		verbose: bool = False
):
	timing.start("predict_test")
	type = average_model.type
	sales_chase = average_model.sales_chase

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

  timing.start("predict_univ")
  if type == "median":
    y_pred_univ = np.full(len(ds.X_univ), ds.y_train.median())
  else:
    y_pred_univ = np.full(len(ds.X_univ), ds.y_train.mean())
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    if type == "median":
      y_pred_multi = np.full(len(ds.X_multiverse), ds.y_train.median())
    else:
      y_pred_multi = np.full(len(ds.X_multiverse), ds.y_train.mean())
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

  df = ds.df_universe
	ind_var = ds.ind_var

	if sales_chase:
		y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
		y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))
    if y_pred_multi is not None:
      y_pred_multi = _sales_chase_univ(df, ind_var, y_pred_multi) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_multi))

	name = "mean"
	if type == "median":
		name = "median"
	if sales_chase:
		name += "*"

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		name,
		average_model,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
  )

	return results


def run_average(
		ds: DataSplit,
		average_type: str = "mean",
		sales_chase: float = 0.0,
		verbose: bool = False
):
	"""
	Runs a garbage model that simply predicts the average of the training set for everything
	:param ds: The data split object containing processed input data
	:param average_type: The type of average to use ("mean" or "median")
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
	timing.stop("train")

	average_model = AverageModel(average_type, sales_chase)
	return predict_average(ds, average_model, timing, verbose)


def predict_naive_sqft(
		ds: DataSplit,
		sqft_model: NaiveSqftModel,
		timing: TimingData,
		verbose: bool = False
):
	timing.start("predict_test")

	ind_per_built_sqft = sqft_model.ind_per_built_sqft
	ind_per_land_sqft = sqft_model.ind_per_land_sqft
	sales_chase = sqft_model.sales_chase

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

  timing.start("predict_univ")
  X_univ = ds.X_univ
  X_univ_improved = X_univ[X_univ["bldg_area_finished_sqft"].gt(0)]
  X_univ_vacant = X_univ[X_univ["bldg_area_finished_sqft"].eq(0)]
  X_univ["prediction_impr"] = X_univ_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
  X_univ["prediction_vacant"] = X_univ_vacant["land_area_sqft"] * ind_per_land_sqft
  X_univ["prediction"] = np.where(X_univ["bldg_area_finished_sqft"].gt(0), X_univ["prediction_impr"], X_univ["prediction_vacant"])
  y_pred_univ = X_univ["prediction"].to_numpy()
  X_univ.drop(columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True)
  timing.stop("predict_univ")

  timing.start("predict_multi")
  if ds.df_multiverse is not None:
    X_multi = ds.X_multiverse
    X_multi_improved = X_multi[X_multi["bldg_area_finished_sqft"].gt(0)]
    X_multi_vacant = X_multi[X_multi["bldg_area_finished_sqft"].eq(0)]
    X_multi["prediction_impr"] = X_multi_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
    X_multi["prediction_vacant"] = X_multi_vacant["land_area_sqft"] * ind_per_land_sqft
    X_multi["prediction"] = np.where(X_multi["bldg_area_finished_sqft"].gt(0), X_multi["prediction_impr"], X_multi["prediction_vacant"])
    y_pred_multi = X_multi["prediction"].to_numpy()
    X_multi.drop(columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True)
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var

	if sales_chase:
		y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
		y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))
    if y_pred_multi is not None:
      y_pred_multi = _sales_chase_univ(df, ind_var, y_pred_multi) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_multi))

	name = "naive_sqft"
	if sales_chase:
		name += "*"

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		name,
		sqft_model,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
    timing,
    verbose=verbose,
    y_pred_multi=y_pred_multi
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

	sqft_model = NaiveSqftModel(ind_per_built_sqft, ind_per_land_sqft, sales_chase)

	return predict_naive_sqft(ds, sqft_model, timing, verbose)


def predict_local_sqft(
		ds: DataSplit,
		sqft_model: LocalSqftModel,
		timing: TimingData,
		verbose: bool = False
):
	timing.start("predict_test")

	loc_map = sqft_model.loc_map
	location_fields = sqft_model.location_fields
	overall_per_impr_sqft = sqft_model.overall_per_impr_sqft
	overall_per_land_sqft = sqft_model.overall_per_land_sqft
	sales_chase = sqft_model.sales_chase

	# intent is to create a primary-keyed dataframe that we can fill with the appropriate local $/sqft value
	# we will merge this in to the main dataframes, then mult. local size by local $/sqft value to predict
	df_land = ds.df_universe[["key"] + location_fields].copy()
	df_impr = ds.df_universe[["key"] + location_fields].copy()

	# start with zero
	df_land["per_land_sqft"] = 0
	df_impr["per_impr_sqft"] = 0

	# go from most specific to the least specific location (first to last)
	for location_field in location_fields:
		df_sqft_impr, df_sqft_land = loc_map[location_field]

		df_impr = df_impr.merge(df_sqft_impr[[location_field, f"{location_field}_per_impr_sqft"]], on=location_field, how="left")
		df_land = df_land.merge(df_sqft_land[[location_field, f"{location_field}_per_land_sqft"]], on=location_field, how="left")

		df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = df_impr[f"{location_field}_per_impr_sqft"]
		df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = df_land[f"{location_field}_per_land_sqft"]

		df_sqft_land.to_csv(f"debug_local_sqft_{len(location_fields)}_{location_field}_sqft_land.csv", index=False)
		df_land.to_csv(f"debug_local_sqft_{len(location_fields)}_{location_field}_land.csv", index=False)

	# any remaining zeroes get filled with the locality-wide median value
	df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
	df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft

	X_test = ds.X_test

	df_impr = df_impr[["key", "per_impr_sqft"]]
	df_land = df_land[["key", "per_land_sqft"]]

	# merge the df_sqft_land/impr values into the X_test dataframe:
	X_test["key"] = ds.df_test["key"]
	X_test = X_test.merge(df_land, on="key", how="left")
	X_test = X_test.merge(df_impr, on="key", how="left")
	X_test.loc[X_test["per_impr_sqft"].isna() | X_test["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
	X_test.loc[X_test["per_land_sqft"].isna() | X_test["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
	X_test = X_test.drop(columns=["key"])

	X_test_improved = X_test[X_test["bldg_area_finished_sqft"].gt(0)]
	X_test_vacant = X_test[X_test["bldg_area_finished_sqft"].eq(0)]
	X_test["prediction_impr"] = X_test_improved["bldg_area_finished_sqft"] * X_test_improved["per_impr_sqft"]
	X_test["prediction_land"] = X_test_vacant["land_area_sqft"] * X_test_vacant["per_land_sqft"]
	X_test["prediction"] = np.where(X_test["bldg_area_finished_sqft"].gt(0), X_test["prediction_impr"], X_test["prediction_land"])

	y_pred_test = X_test["prediction"].to_numpy()
	# TODO: later, don't drop these columns, use them to predict land value everywhere
	X_test.drop(columns=["prediction_impr", "prediction_land", "prediction", "per_impr_sqft", "per_land_sqft"], inplace=True)
	timing.stop("predict_test")

	timing.start("predict_sales")
	X_sales = ds.X_sales

	# merge the df_sqft_land/impr values into the X_sales dataframe:
	X_sales["key"] = ds.df_sales["key"]
	X_sales = X_sales.merge(df_land, on="key", how="left")
	X_sales = X_sales.merge(df_impr, on="key", how="left")
	X_sales.loc[X_sales["per_impr_sqft"].isna() | X_sales["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
	X_sales.loc[X_sales["per_land_sqft"].isna() | X_sales["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
	X_sales = X_sales.drop(columns=["key"])

	X_sales_improved = X_sales[X_sales["bldg_area_finished_sqft"].gt(0)]
	X_sales_vacant = X_sales[X_sales["bldg_area_finished_sqft"].eq(0)]
	X_sales["prediction_impr"] = X_sales_improved["bldg_area_finished_sqft"] * X_sales_improved["per_impr_sqft"]
	X_sales["prediction_land"] = X_sales_vacant["land_area_sqft"] * X_sales_vacant["per_land_sqft"]
	X_sales["prediction"] = np.where(X_sales["bldg_area_finished_sqft"].gt(0), X_sales["prediction_impr"], X_sales["prediction_land"])
	y_pred_sales = X_sales["prediction"].to_numpy()
	X_sales.drop(columns=["prediction_impr", "prediction_land", "prediction", "per_impr_sqft", "per_land_sqft"], inplace=True)
  timing.stop("predict_sales")

  timing.start("predict_univ")
  X_univ = ds.X_univ

  # merge the df_sqft_land/impr values into the X_univ dataframe:
  X_univ["key"] = ds.df_universe["key"]
  X_univ = X_univ.merge(df_land, on="key", how="left")
  X_univ = X_univ.merge(df_impr, on="key", how="left")
  X_univ.loc[X_univ["per_impr_sqft"].isna() | X_univ["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
  X_univ.loc[X_univ["per_land_sqft"].isna() | X_univ["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
  X_univ = X_univ.drop(columns=["key"])

  X_univ_improved = X_univ[X_univ["bldg_area_finished_sqft"].gt(0)]
  X_univ_vacant = X_univ[X_univ["bldg_area_finished_sqft"].eq(0)]
  X_univ["prediction_impr"] = X_univ_improved["bldg_area_finished_sqft"] * X_univ_improved["per_impr_sqft"]
  X_univ["prediction_land"] = X_univ_vacant["land_area_sqft"] * X_univ_vacant["per_land_sqft"]
  X_univ.loc[X_univ["prediction_impr"].isna() | X_univ["prediction_impr"].eq(0)] = overall_per_impr_sqft
  X_univ.loc[X_univ["prediction_land"].isna() | X_univ["prediction_land"].eq(0)] = overall_per_land_sqft
  X_univ["prediction"] = np.where(X_univ["bldg_area_finished_sqft"].gt(0), X_univ["prediction_impr"], X_univ["prediction_land"])
  y_pred_univ = X_univ["prediction"].to_numpy()
  X_univ.drop(columns=["prediction_impr", "prediction_land", "prediction", "per_impr_sqft", "per_land_sqft"], inplace=True)
  timing.stop("predict_univ")

  timing.start("predict_multi")
  X_multi = ds.X_multiverse
  if X_multi is not None:
    X_multi["key"] = ds.df_multiverse["key"]
    X_multi = X_multi.merge(df_land, on="key", how="left")
    X_multi = X_multi.merge(df_impr, on="key", how="left")
    X_multi.loc[X_multi["per_impr_sqft"].isna() | X_multi["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
    X_multi.loc[X_multi["per_land_sqft"].isna() | X_multi["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
    X_multi = X_multi.drop(columns=["key"])

    X_multi_improved = X_multi[X_multi["bldg_area_finished_sqft"].gt(0)]
    X_multi_vacant = X_multi[X_multi["bldg_area_finished_sqft"].eq(0)]
    X_multi["prediction_impr"] = X_multi_improved["bldg_area_finished_sqft"] * X_multi_improved["per_impr_sqft"]
    X_multi["prediction_land"] = X_multi_vacant["land_area_sqft"] * X_multi_vacant["per_land_sqft"]
    X_multi.loc[X_multi["prediction_impr"].isna() | X_multi["prediction_impr"].eq(0)] = overall_per_impr_sqft
    X_multi.loc[X_multi["prediction_land"].isna() | X_multi["prediction_land"].eq(0)] = overall_per_land_sqft
    X_multi["prediction"] = np.where(X_multi["bldg_area_finished_sqft"].gt(0), X_multi["prediction_impr"], X_multi["prediction_land"])
    y_pred_multi = X_multi["prediction"].to_numpy()
    X_multi.drop(columns=["prediction_impr", "prediction_land", "prediction", "per_impr_sqft", "per_land_sqft"], inplace=True)
  else:
    y_pred_multi = None
  timing.stop("predict_multi")

  timing.stop("total")

	df = ds.df_universe
	ind_var = ds.ind_var

	if sales_chase:
    y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
    y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, ind_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))
    if y_pred_multi is not None:
      y_pred_multi = _sales_chase_univ(df, ind_var, y_pred_multi) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_multi))

	if "ss_id" in location_fields:
		name = "local_smart_sqft"
	else:
		name = "local_naive_sqft"

	if sales_chase:
		name += "*"

	results = SingleModelResults(
		ds,
		"prediction",
		"he_id",
		name,
		sqft_model,
		y_pred_test,
		y_pred_sales,
		y_pred_univ,
    timing,
    y_pred_multi=y_pred_multi
	)

	return results


def run_local_sqft(
		ds: DataSplit,
		location_fields: list[str],
		sales_chase: float = 0.0,
		verbose: bool = False
):
	"""
	Runs a model that simply predicts the median $/sqft of the training set on a per-location basis
	:param ds: The data split object containing processed input data
	:param location_fields: The fields to use for location-based prediction
	:param sales_chase: If not 0, simulate sales chasing by predicting the sales set as the test set, with this much noise
	:param verbose: Whether to print verbose output
	:return: The model results
	"""
	timing = TimingData()

	timing.start("total")

	timing.start("parameter_search")
	timing.stop("parameter_search")

	timing.start("setup")
	ds.split()
	timing.stop("setup")

	timing.start("train")

	X_train = ds.X_train

	# filter out vacant land where bldg_area_finished_sqft is zero:
	X_train_improved = X_train[X_train["bldg_area_finished_sqft"].gt(0)]

	# filter out improved land where bldg_area_finished_sqft is > zero:
	X_train_vacant = X_train[X_train["bldg_area_finished_sqft"].eq(0)]

	# our aim is to construct a dataframe which will contain the local $/sqft values for each individual location value,
	# for multiple location fields. We will then use this to calculate final values for every permutation, and merge
	# that onto our main dataframe to assign $/sqft values from which to generate our final predictions

	loc_map = {}

	for location_field in location_fields:

		data_sqft_land = {}
		data_sqft_impr = {}

		if location_field not in ds.df_train:
			print(f"Location field {location_field} not found in dataset")
			continue

		data_sqft_land[location_field] = []
		data_sqft_land[f"{location_field}_per_land_sqft"] = []

		data_sqft_impr[location_field] = []
		data_sqft_impr[f"{location_field}_per_impr_sqft"] = []

		# for every specific location, calculate the local median $/sqft for improved & vacant property
		for loc in ds.df_train[location_field].unique():
			y_train_loc = ds.y_train[ds.df_train[location_field].eq(loc)]
			X_train_loc = ds.X_train[ds.df_train[location_field].eq(loc)]

			X_train_loc_improved = X_train_loc[X_train_loc["bldg_area_finished_sqft"].gt(0)]
			X_train_loc_vacant = X_train_loc[X_train_loc["bldg_area_finished_sqft"].eq(0)]

			if len(X_train_loc_improved) > 0:
				local_per_impr_sqft = (y_train_loc / X_train_loc_improved["bldg_area_finished_sqft"]).median()
			else:
				local_per_impr_sqft = 0

			if len(X_train_loc_vacant) > 0:
				local_per_land_sqft = (y_train_loc / X_train_loc_vacant["land_area_sqft"]).median()
			else:
				local_per_land_sqft = 0

			# some values will be null so replace them with zeros
			if pd.isna(local_per_impr_sqft):
				local_per_impr_sqft = 0
			if pd.isna(local_per_land_sqft):
				local_per_land_sqft = 0

			data_sqft_impr[location_field].append(loc)
			data_sqft_land[location_field].append(loc)

			data_sqft_impr[f"{location_field}_per_impr_sqft"].append(local_per_impr_sqft)
			data_sqft_land[f"{location_field}_per_land_sqft"].append(local_per_land_sqft)

		for key in data_sqft_impr:
			print(f"--> {key}: {len(data_sqft_impr[key])}")

		# create dataframes from the calculated values
		df_sqft_impr = pd.DataFrame(data=data_sqft_impr)
		df_sqft_land = pd.DataFrame(data=data_sqft_land)

		loc_map[location_field] = (df_sqft_impr, df_sqft_land)

	# calculate the median overall values
	overall_per_impr_sqft = (ds.y_train / X_train_improved["bldg_area_finished_sqft"]).median()
	overall_per_land_sqft = (ds.y_train / X_train_vacant["land_area_sqft"]).median()

	timing.stop("train")
	if verbose:
		print("Tuning Naive Sqft: searching for optimal parameters...")
		print(f"--> optimal improved $/finished sqft (overall) = {overall_per_impr_sqft:0.2f}")
		print(f"--> optimal vacant   $/land     sqft (overall) = {overall_per_land_sqft:0.2f}")

	sqft_model = LocalSqftModel(loc_map, location_fields, overall_per_impr_sqft, overall_per_land_sqft, sales_chase)

	return predict_local_sqft(ds, sqft_model, timing, verbose)


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


def _gwr_predict(model, points, P, exog_scale=None, exog_resid=None, fit_params=None):
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
	if fit_params is None:
		fit_params = {}

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
    plot: bool = False,
    gdf: gpd.GeoDataFrame = None,
    dep_vars: list[str] = None,
    intercept: bool = True
):
  gwr = GWR(coords_train, y_train, X_train, gwr_bw, constant=intercept)
	gwr_results = _gwr_predict(gwr, coords, X)
  params = gwr_results["params"]
  y_pred = gwr_results["y_pred"]

  print(f"params shape = {params.shape}")

  # the shape of params is (n, k), where n is the number of points to predict and k is the number of predictors
  # we want to visualize each "layer" of the prediction surface individually, so we grab one set of predictions for each predictor

  x_coords, y_coords = np.array(coords).T

  print(f"X shape = {X.shape}")
  print(f"X type = {type(X)}")

  print(f"dep_vars = {dep_vars}")
  print(f"params shape = {params.shape}")

  var = ""
  if plot:
    print(f"gdf exists ? {gdf is not None}")
    print(f"gdf cols = {gdf.columns.values}")
    for i in range(params.shape[1]):
      contributions = params[:, i]

      if i == 0:
        var = "Intercept"
      else:
        var = dep_vars[i-1] if dep_vars is not None else f"Variable {i-1}"

      plot_value_surface(f"Prediction contribution for {var}", contributions, x_coords, y_coords, gdf)

    plot_value_surface("Prediction", y_pred, x_coords, y_coords, gdf)

    if dep_vars is not None and "land_area_sqft" in dep_vars:
      # get the index of the land area sqft variable
      land_size_index = dep_vars.index("land_area_sqft")

      print(f"Divide {var} by {dep_vars[land_size_index]}")

      # we normalize this by dividing each contribution by the value of its corresponding variable value in X:
      #contributions = div_field_z_safe(contributions, X[:, land_size_index])
      _y_pred_land_sqft = div_field_z_safe(y_pred, X[:, land_size_index])
      plot_value_surface("Prediction / land sqft", _y_pred_land_sqft, x_coords, y_coords, gdf)

  return y_pred


def _get_params(name:str, slug:str, ds:DataSplit, tune_func, outpath:str, save_params:bool, use_saved_params:bool, verbose:bool, **kwargs):
  if verbose:
		print(f"Tuning {name}: searching for optimal parameters...")

	params = None
	if use_saved_params:
		if os.path.exists(f"{outpath}/{slug}_params.json"):
			params = json.load(open(f"{outpath}/{slug}_params.json", "r"))
			if verbose:
				print(f"--> using saved parameters: {params}")
	if params is None:
    params = tune_func(ds.X_train, ds.y_train, verbose=verbose, **kwargs)
		if verbose:
			print(f"--> optimal parameters = {params}")
		if save_params:
			os.makedirs(outpath, exist_ok=True)
			json.dump(params, open(f"{outpath}/{slug}_params.json", "w"))
  return params


def plot_value_surface(title: str, values: np.array, x_coords: np.array, y_coords: np.array, gdf: gpd.GeoDataFrame, center_on_zero: bool = True):
  plt.clf()
  plt.figure(figsize=(12, 8))

  plt.title(title)
  vmin = np.quantile(values, 0.05)
  vmax = np.quantile(values, 0.95)

  norm = None
  if center_on_zero:
    vmin = min(0, vmin)
    vcenter = max(0, vmin)
    vmax = max(0, vmax)

    if vmax > abs(vmin):
      vmin = -vmax
    if abs(vmin) > vmax:
      vmax = abs(vmin)
    # Define normalization to center zero on white
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
  else:
    # Define normalization to start at zero, center on the median value and cap at 95th percentile
    vmin = min(0, vmin)
    vcenter = max(0, np.quantile(values, 0.50))
    vmax = max(0, vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


  #plt.scatter(x_coords, y_coords, c=values, cmap="coolwarm", s=2, norm=norm)

  gdf_slice = gdf[["geometry"]].copy()
  gdf_slice["values"] = values

  # plot the contributions as polygons using the same color map and vmin/vmax:
  ax = gdf_slice.plot(column="values", cmap="coolwarm", norm=norm, ax=plt.gca())
  mappable = ax.collections[0]

  cbar = plt.colorbar(mappable, ax=ax)
  cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fancy_format(x)))
  cbar.set_label("Value ($)", fontsize=12)
  plt.show()