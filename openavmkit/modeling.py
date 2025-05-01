import json
import os
import pickle
import warnings

import numpy.linalg
from IPython.core.display import display
import polars as pl
from joblib import Parallel, delayed
from typing import Union, Any, Dict
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import statsmodels.api as sm
import pandas as pd
import geopandas as gpd
import xgboost
import lightgbm as lgb
import catboost
from catboost import CatBoostRegressor, Pool
from lightgbm import Booster
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm
from matplotlib.ticker import FuncFormatter
from mgwr.gwr import GWR
from mgwr.gwr import _compute_betas_gwr, Kernel

from mgwr.sel_bw import Sel_BW
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.regression.linear_model import RegressionResults
from xgboost import XGBRegressor

from openavmkit.data import get_sales, simulate_removed_buildings, _enrich_time_field, _enrich_sale_age_days, \
  SalesUniversePair, get_hydrated_sales_from_sup
from openavmkit.ratio_study import RatioStudy
from openavmkit.utilities.format import fancy_format
from openavmkit.utilities.modeling import GarbageModel, AverageModel, NaiveSqftModel, LocalSqftModel, PassThroughModel, \
  GWRModel, MRAModel, LarsModel, GroundTruthModel, SpatialLagModel
from openavmkit.utilities.data import clean_column_names, div_field_z_safe
from openavmkit.utilities.settings import get_valuation_date
from openavmkit.utilities.stats import quick_median_chd_pl, calc_mse_r2_adj_r2, calc_prb
from openavmkit.tuning import tune_lightgbm, tune_xgboost, tune_catboost
from openavmkit.utilities.timing import TimingData


from scipy.optimize import minimize

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
  LarsModel,
  PassThroughModel,
  SpatialLagModel,
  GroundTruthModel,
  GWRModel,
  str,
  None
]


class LandPredictionResults:

  def __init__(self,
      land_prediction_field: str,
      impr_prediction_field: str,
      total_prediction_field: str,
      dep_var: str,
      ind_vars: list[str],
      sup: SalesUniversePair
  ):

    necessary_fields = [
      land_prediction_field,
      impr_prediction_field,
      total_prediction_field,
      dep_var,
      "land_he_id",
      "impr_he_id",
      "he_id",
      "is_vacant",
      "land_area_sqft",
      "bldg_area_finished_sqft"
    ]

    use_sales_not_univ = False
    for field in necessary_fields:
      if field not in sup.universe:
        if "sale" not in field:
          raise ValueError(f"Necessary field '{field}' not found in universe DataFrame.")

    df = get_hydrated_sales_from_sup(sup)

    for field in necessary_fields + ["valid_sale", "vacant_sale", "valid_for_land_ratio_study", "valid_for_ratio_study"]:
      if field not in df:
        raise ValueError(f"Necessary field '{field}' not found in sales DataFrame.")

    self.land_prediction_field = land_prediction_field
    self.impr_prediction_field = impr_prediction_field
    self.total_prediction_field = total_prediction_field

    df_univ = sup.universe.copy()

    df_univ["land_allocation"] = div_field_z_safe(df_univ[land_prediction_field], df_univ[total_prediction_field])
    df_univ["impr_allocation"] = div_field_z_safe(df_univ[impr_prediction_field], df_univ[total_prediction_field])

    # Phase 1: Accuracy
    if "sale" in dep_var:
      df = df[df["valid_for_land_ratio_study"].eq(True)].copy()
      land_predictions = df[land_prediction_field]
      sale_prices = df[dep_var]
    elif dep_var == "true_land_value":
      df = df_univ.copy()
      land_predictions = df[land_prediction_field]
      sale_prices = df[dep_var]
    else:
      raise ValueError(f"Unsupported dep_var '{dep_var}' for land prediction results.")

    self.land_ratio_study = RatioStudy(land_predictions, sale_prices)
    mse, r2, adj_r2 = calc_mse_r2_adj_r2(land_predictions, sale_prices, len(ind_vars))
    self.mse = mse
    self.rmse = np.sqrt(mse)
    self.r2 = r2
    self.adj_r2 = adj_r2
    self.prb, _, _ = calc_prb(land_predictions, sale_prices)

    df_univ_valid = df_univ.drop(columns="geometry", errors="ignore").copy()

    # convert all category and string[python] types to string:
    for col in df_univ_valid.columns:
      if df_univ_valid[col].dtype in ["category", "string"]:
        df_univ_valid[col] = df_univ_valid[col].astype("str")
    pl_df = pl.DataFrame(df_univ_valid)

    # Phase 2: Consistency
    self.total_chd = quick_median_chd_pl(pl_df, total_prediction_field, "he_id")
    self.land_chd = quick_median_chd_pl(pl_df, land_prediction_field, "land_he_id")
    self.impr_chd = quick_median_chd_pl(pl_df, impr_prediction_field, "impr_he_id")

    # Phase 3: Sanity

    # Hard rules
    count = len(df_univ)
    count_land_null = len(df_univ[df_univ[land_prediction_field].isna()])
    count_land_negative = len(df_univ[df_univ[land_prediction_field].lt(0)])
    count_land_invalid = len(df_univ[
      df_univ[land_prediction_field].lt(0) |
      df_univ[land_prediction_field].isna()
    ])
    self.perc_land_null = count_land_null / count
    self.perc_land_negative = count_land_negative / count
    self.perc_land_invalid = count_land_invalid / count

    count_impr_null = len(df_univ[df_univ[impr_prediction_field].isna()])
    count_impr_negative = len(df_univ[df_univ[impr_prediction_field].lt(0)])
    count_impr_invalid = len(df_univ[
      df_univ[impr_prediction_field].lt(0) |
      df_univ[impr_prediction_field].isna()
    ])
    self.perc_impr_null = count_impr_null / count
    self.perc_impr_negative = count_impr_negative / count
    self.perc_impr_invalid = count_impr_invalid / count

    count_dont_add_up = len(df_univ[(
      df_univ[total_prediction_field] - np.abs(
        df_univ[land_prediction_field] +
        df_univ[impr_prediction_field]
      )).gt(1e-6)]
    )
    count_land_overshoot = len(df_univ[
      df_univ[land_prediction_field].gt(df_univ[total_prediction_field])
    ])
    count_vacant_land_not_100 = len(df_univ[
      df_univ["is_vacant"].eq(True) &
      df_univ["land_allocation"].lt(1.0)
    ])
    self.perc_dont_add_up = count_dont_add_up / count
    self.perc_land_overshoot = count_land_overshoot / count
    self.perc_vacant_land_not_100 = count_vacant_land_not_100 / count

    # Soft rules
    count_improved_land_over_100 = len(df_univ[
      df_univ["is_vacant"].eq(False) &
      df_univ["land_allocation"].gt(1.0)
    ])
    self.perc_improved_land_over_100 = count_improved_land_over_100 / count

    self.utility_score = 0
    self.utility_score = land_utility_score(self)

    # Paired sales analysis tests:
    # Control for location:
    # - Land allocation inversely correlated with floor area ratio
    # - Land value / sqft decreases as total land size increases
    # - Land value increases as total land size increases
    # - Within location, control for one at a time: size/quality/condition:
    #   - Condition positively correlated with impr value
    #   - Quality positively correlated with impr value
    #   - Age *mostly* negatively correlated with impr value





class PredictionResults:
  """
  Container for prediction results and associated performance metrics.

  Attributes:
      dep_var (str): The independent variable used for prediction
      ind_vars (list[str]): List of dependent variables
      y (numpy.ndarray): Ground truth values
      y_pred (numpy.ndarray): Predicted values
      mse (float): Mean squared error
      rmse (float): Root mean squared error
      r2 (float): R-squared
      adj_r2 (float): Adjusted R-squared
      ratio_study (RatioStudy): RatioStudy object
  """

  def __init__(self,
      dep_var: str,
      ind_vars: list[str],
      prediction_field: str,
      df: pd.DataFrame):
    """
    Initialize a PredictionResults instance.

    Converts the specified prediction column in the DataFrame to a NumPy array,
    computes performance metrics on the subset of data that is valid for ratio study,
    and stores the computed values.

    :param dep_var: The independent variable (e.g., sale price).
    :type dep_var: str
    :param ind_vars: List of dependent variable names.
    :type ind_vars: list[str]
    :param prediction_field: Name of the field containing model predictions.
    :type prediction_field: str
    :param df: DataFrame on which predictions were computed.
    :type df: pandas.DataFrame
    """
    self.dep_var = dep_var
    self.ind_vars = ind_vars

    y = df[dep_var].to_numpy()
    y_pred = df[prediction_field].to_numpy()

    self.y = y
    self.y_pred = y_pred

    df_valid = df[df["valid_for_ratio_study"].eq(True)]

    y = df_valid[dep_var].to_numpy()
    y_pred = df_valid[prediction_field].to_numpy()

    # get a mask of all NaN values in y_pred:
    y_pred_mask = ~pd.isna(y_pred)
    y_mask = ~pd.isna(y)

    # select only values that are not NaN in either:
    y_clean = y[y_mask & y_pred_mask]
    y_pred_clean = y_pred[y_mask & y_pred_mask]

    if len(y_clean) > 0 and len (y_pred_clean) > 0:
      self.mse = mean_squared_error(y_clean, y_pred_clean)
      self.rmse = np.sqrt(self.mse)
      var_y = np.var(y_clean)
      if var_y == 0:
        self.r2 = float('nan')  # R² undefined when variance is 0
      else:
        self.r2 = 1 - self.mse / var_y
    else:
      self.mse = float('nan')
      self.rmse = float('nan')
      self.r2 = float('nan')

    n = len(y_pred)
    k = len(ind_vars)
    divisor = n - k - 1
    if divisor <= 0 or pd.isna(self.r2):
      self.adj_r2 = float('nan')  # Adjusted R² undefined with insufficient df or undefined R²
    else:
      self.adj_r2 = 1 - ((1 - self.r2) * (n - 1) / divisor)
    self.ratio_study = RatioStudy(y_pred_clean, y_clean)


class DataSplit:
  """
  Encapsulates the splitting of data into training, test, and other subsets. Handles all the internals and keeps things
  organized so you don't have to worry about it.

  Attributes:
      df_sales (pd.DataFrame): Sales data after processing.
      df_universe (pd.DataFrame): Universe (parcel) data after processing.
      df_train (pd.DataFrame): Training subset of sales data.
      df_test (pd.DataFrame): Test subset of sales data.
      X_train, X_test, X_univ, etc.: Feature matrices for different subsets.
      y_train, y_test: Target arrays for training and testing.
      Other attributes store configuration and settings.
  """

  counter: int = 0

  def __init__(self,
      df_sales: pd.DataFrame | None,
      df_universe: pd.DataFrame | None,
      model_group: str,
      settings: dict,
      dep_var: str,
      dep_var_test: str,
      ind_vars: list[str],
      categorical_vars: list[str],
      interactions: dict,
      test_keys: list[str],
      train_keys: list[str],
      vacant_only: bool = False,
      hedonic: bool = False,
      days_field: str = "sale_age_days",
      hedonic_test_against_vacant_sales=True,
      init: bool = True):
    """
    Initialize a DataSplit instance by processing and splitting sales and universe data.

    Performs several operations:
     - Saves unmodified copies of original data.
     - Adds missing columns to universe data.
     - Enriches time fields and calculates sale age.
     - Splits sales data into training and test sets.
     - Pre-sorts data for rolling origin cross-validation.
     - Applies interactions if specified.

    :param df_sales: Sales DataFrame.
    :type df_sales: pandas.DataFrame or None
    :param df_universe: Universe (parcel) DataFrame.
    :type df_universe: pandas.DataFrame or None
    :param model_group: Model group identifier.
    :type model_group: str
    :param settings: Settings dictionary.
    :type settings: dict
    :param dep_var: Dependent variable name.
    :type dep_var: str
    :param dep_var_test: Dependent variable name for testing.
    :type dep_var_test: str
    :param ind_vars: List of independent variable names.
    :type ind_vars: list[str]
    :param categorical_vars: List of categorical variable names.
    :type categorical_vars: list[str]
    :param interactions: Dictionary defining interactions between variables.
    :type interactions: dict
    :param test_keys: List of keys for test set.
    :type test_keys: list[str]
    :param train_keys: List of keys for training set.
    :type train_keys: list[str]
    :param vacant_only: Whether to consider only vacant sales.
    :type vacant_only: bool, optional
    :param hedonic: Whether to use hedonic adjustments.
    :type hedonic: bool, optional
    :param days_field: Field name for sale age in days.
    :type days_field: str, optional
    :param init: Whether to perform initialization (default True).
    :type init: bool, optional
    :raises ValueError: If required fields are missing.
    """
    if not init:
      return

    self.settings = settings.copy()

    # An *unmodified* copy of the original model group universe/sales, that will remain unmodified
    self.df_universe_orig = df_universe.copy()
    self.df_sales_orig = df_sales.copy()

    # The working copy of the model group universe, that *will* be modified
    self.df_universe = df_universe.copy()

    # Set "sales" fields in the universe so that columns match
    set_to_zero = ["sale_age_days"]
    set_to_false = ["valid_sale", "vacant_sale", "valid_for_ratio_study", "valid_for_land_ratio_study"]
    set_to_none = ["ss_id", "sale_price", "sale_price_time_adj"]

    for col in set_to_zero:
      self.df_universe[col] = 0
    for col in set_to_false:
      self.df_universe[col] = False
    for col in set_to_none:
      self.df_universe[col] = None

    # Set sale dates in the universe to match the valuation date
    val_date = get_valuation_date(settings)
    self.df_universe["sale_date"] = val_date
    self.df_universe = _enrich_time_field(self.df_universe, "sale")
    self.df_universe = _enrich_sale_age_days(self.df_universe, settings)

    self.df_sales = get_sales(df_sales, settings, vacant_only).reset_index(drop=True)

    self._df_sales = self.df_sales.copy()

    self.test_keys = test_keys
    self.train_keys = train_keys

    self.train_sizes = np.zeros_like(train_keys)

    self.train_he_ids = np.zeros_like(train_keys)
    self.train_land_he_ids = np.zeros_like(train_keys)
    self.train_impr_he_ids = np.zeros_like(train_keys)

    self.df_test:pd.DataFrame|None = None
    self.df_train:pd.DataFrame|None = None

    if hedonic:
      # transform df_universe & df_sales such that all improved characteristics are removed
      self.df_universe = simulate_removed_buildings(self.df_universe, settings)
      self.df_sales = simulate_removed_buildings(self.df_sales, settings)

    # we also need to limit the sales set, but we can't do that AFTER we've split

    # Pre-sort dataframes so that rolling origin cross-validation can assume oldest observations first:
    self.df_universe.sort_values(by="key", ascending=False, inplace=True)

    if days_field in self.df_sales:
      self.df_sales.sort_values(by=days_field, ascending=False, inplace=True)
    else:
      raise ValueError(f"Field '{days_field}' not found in dataframe.")

    self.model_group = model_group
    self.dep_var = dep_var
    self.dep_var_test = dep_var_test
    self.ind_vars = ind_vars.copy()
    self.categorical_vars = categorical_vars.copy()
    self.interactions = interactions.copy()
    self.one_hot_descendants = {}
    self.vacant_only = vacant_only
    self.hedonic = hedonic
    self.hedonic_test_against_vacant_sales = hedonic_test_against_vacant_sales
    self.days_field = days_field
    self.split()

  def copy(self):
    """
    Return a deep copy of the DataSplit instance.

    :returns: A deep copy of the current DataSplit.
    :rtype: DataSplit
    """
    ds = DataSplit(
      None,
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
    ds.df_sales_orig = self.df_sales_orig.copy()
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
    ds.train_sizes = self.train_sizes.copy()
    ds.train_he_ids = self.train_he_ids.copy()
    ds.train_land_he_ids = self.train_land_he_ids.copy()
    ds.train_impr_he_ids = self.train_impr_he_ids.copy()
    ds.vacant_only = self.vacant_only
    ds.hedonic = self.hedonic
    ds.hedonic_test_against_vacant_sales = self.hedonic_test_against_vacant_sales
    ds.dep_var = self.dep_var
    ds.dep_var_test = self.dep_var_test
    ds.ind_vars = self.ind_vars.copy()
    ds.categorical_vars = self.categorical_vars.copy()
    ds.interactions = self.interactions.copy()
    ds.one_hot_descendants = self.one_hot_descendants.copy()
    ds.days_field = self.days_field

    return ds


  def encode_categoricals_as_categories(self):
    """
    Convert all categorical variables in sales and universe DataFrames to the 'category' dtype.

    :returns: The updated DataSplit instance.
    :rtype: DataSplit
    """
    if len(self.categorical_vars) == 0:
      return self

    ds = self.copy()

    for col in ds.categorical_vars:
      ds.df_universe[col] = ds.df_universe[col].astype("category")
      ds.df_sales[col] = ds.df_sales[col].astype("category")

    return ds

  def reconcile_fields_with_foreign(self, foreign_ds):
    """
    Reconcile this DataSplit's fields with those of a provided reference DataSplit (foreign_ds).

    The function performs the following:
      1. One-hot encodes its own categorical columns using its existing encoding method.
      2. Reindexes each DataFrame (train, test, universe, sales)
         so that their columns exactly match the reference DataSplit's train columns.

    Parameters:
      foreign_ds (DataSplit): The DataSplit instance whose fields should be matched (e.g., the model's ds).

    Returns:
      DataSplit: The updated self with reconciled columns.
    """

    # check if foreign is one hot descended by checking if descendents is an empty object
    if foreign_ds.one_hot_descendants is None or len(foreign_ds.one_hot_descendants) == 0:
      # if so nothing is to be done here
      return self

    # First, ensure that self is one-hot encoded.
    ds_encoded = self.encode_categoricals_with_one_hot()

    # Use the train split of the foreign DataSplit as the reference.
    reference_columns = foreign_ds.df_train.columns

    # Define a helper function to reindex a DataFrame split.
    def reindex_df(df):
      return df.reindex(columns=reference_columns, fill_value=0.0)

    # Reindex all splits in the local DataSplit so that their columns match the reference.
    ds_encoded.df_train    = reindex_df(ds_encoded.df_train)
    ds_encoded.df_test     = reindex_df(ds_encoded.df_test)
    ds_encoded.df_universe = reindex_df(ds_encoded.df_universe)
    ds_encoded.df_sales    = reindex_df(ds_encoded.df_sales)

    # Update the independent variables metadata (if applicable)
    ds_encoded.ind_vars = [col for col in reference_columns if col in ds_encoded.ind_vars]

    # Optionally, you might also update any other metadata such as one-hot descendants mapping.
    # For example, if you previously built a mapping from original categorical variables to one-hot encoded columns,
    # you can rebuild or adjust it here.

    # Build a mapping of original categorical variables to their one-hot encoded descendant columns.
    ds_encoded.one_hot_descendants = {
      col: [descendant for descendant in reference_columns if descendant.startswith(f"{col}_")]
      for col in ds_encoded.categorical_vars
    }

    return ds_encoded


  def encode_categoricals_with_one_hot(self):
    """
    One-hot encode the categorical variables in all data splits using a consistent encoder.
    This implementation:

    - Collects the union of categorical values from universe, sales, train, test.
    - Uses scikit-learn's OneHotEncoder (with handle_unknown="ignore" and drop='first') so that output
      dummy columns are consistent across splits.
    - Transforms each DataFrame so that each split (universe, sales, train, test,)
      ends up with the same set of dummy columns (reindexing missing ones to 0).

    Returns:
        DataSplit: The updated DataSplit instance with one-hot encoded features.
    """
    # If no categorical variables to encode, return self
    if len(self.categorical_vars) == 0:
      return self

    ds = self.copy()

    # Identify the categorical variables that need encoding.
    # We restrict to those that appear in the independent variables.
    cat_vars = [col for col in ds.ind_vars if col in self.categorical_vars]

    # Collect data from all splits where a categorical column is present.
    dataframes_for_union = []
    for df in [ds.df_universe, ds.df_sales, ds.df_train, ds.df_test]:
      present_cols = [col for col in cat_vars if col in df.columns]
      if present_cols:
        dataframes_for_union.append(df[present_cols])

    # Concatenate all categorical data for a full view of unique values.
    if dataframes_for_union:
      union_df = pd.concat(dataframes_for_union, axis=0)
    else:
      return ds  # Nothing to encode

    # Build a dictionary of union categories for each categorical variable.
    union_categories = {}
    for col in cat_vars:
      if col in union_df.columns:
        # Drop missing values and sort the unique values (order matters if using drop-first)
        union_categories[col] = sorted(union_df[col].dropna().unique())

    # Create the OneHotEncoder:
    # - The 'categories' parameter is provided as a list following the order in cat_vars.
    # - handle_unknown="ignore" ensures that any new category seen later is handled gracefully.
    # - drop='first' mimics drop_first=True in pd.get_dummies (avoid dummy-variable trap)
    encoder = OneHotEncoder(
      categories=[union_categories[col] for col in cat_vars],
      handle_unknown='ignore',
      drop='first',
      sparse_output=False
    )

    # Prepare a DataFrame for fitting the encoder.
    # Ensure all categorical columns appear, even if some are missing from union_df.
    df_for_encoding = pd.DataFrame()
    for col in cat_vars:
      if col in union_df.columns:
        df_for_encoding[col] = union_df[col]
      else:
        # If somehow missing, create column filled with NaN.
        df_for_encoding[col] = np.nan

    # Fit the encoder on the union of the categorical data.
    encoder.fit(df_for_encoding)

    # Retrieve feature names generated by the encoder.
    try:
      onehot_feature_names = encoder.get_feature_names_out(cat_vars)
    except AttributeError:
      onehot_feature_names = encoder.get_feature_names(cat_vars)

    # Define a helper function to transform a DataFrame.
    def transform_df(df):
      df_tmp = df.copy()
      # Make sure all categorical columns are present for transformation.
      for col in cat_vars:
        if col not in df_tmp.columns:
          df_tmp[col] = np.nan
      # Subset to our categorical columns in the expected order.
      df_cats = df_tmp[cat_vars]
      # Transform using the fitted OneHotEncoder; result is a NumPy array.
      onehot_arr = encoder.transform(df_cats)
      # Create a DataFrame from the dummy array with proper column names.
      onehot_df = pd.DataFrame(onehot_arr, columns=onehot_feature_names, index=df.index)
      # Drop the original categorical columns from the DataFrame.
      df_tmp = df_tmp.drop(columns=cat_vars, errors='ignore')
      # Concatenate the dummy DataFrame onto the non-categorical features.
      df_transformed = pd.concat([df_tmp, onehot_df], axis=1)
      return df_transformed

    # Transform every split.
    ds.df_universe = transform_df(ds.df_universe)
    ds.df_sales    = transform_df(ds.df_sales)
    ds.df_train    = transform_df(ds.df_train)
    ds.df_test     = transform_df(ds.df_test)

    # Clean column names.
    ds.df_universe = clean_column_names(ds.df_universe)
    ds.df_sales    = clean_column_names(ds.df_sales)
    ds.df_train    = clean_column_names(ds.df_train)
    ds.df_test     = clean_column_names(ds.df_test)

    # Ensure that all data splits have the same columns and in the same order.
    # We use the training data columns as the reference.
    base_columns = ds.df_train.columns
    ds.df_universe = ds.df_universe.reindex(columns=base_columns, fill_value=0.0)
    ds.df_sales    = ds.df_sales.reindex(columns=base_columns, fill_value=0.0)
    ds.df_test     = ds.df_test.reindex(columns=base_columns, fill_value=0.0)

    # Here, we update ds.ind_vars to include only the columns present in df_train.
    ds.ind_vars = [col for col in base_columns if col in ds.ind_vars or col in onehot_feature_names]

    # Build a mapping of original categorical variables to their one-hot encoded descendant columns.
    ds.one_hot_descendants = {
      orig: [col for col in onehot_feature_names if col.startswith(f"{orig}_")]
      for orig in cat_vars
    }

    return ds

  def split(self):
    """
    Split the sales DataFrame into training and test sets based on provided keys.

    Uses the test_keys and train_keys to partition the sales data. Also sorts the splits by the specified days_field.
    If the model is hedonic, further filters the sales set to vacant records.
    """
    test_keys = self.test_keys

    # separate df into train & test:

    # select the rows that are in the test_keys:
    self.df_test = self.df_sales[self.df_sales["key_sale"].astype(str).isin(test_keys)].reset_index(drop=True)
    self.df_train = self.df_sales.drop(self.df_test.index)

    self.df_test = self.df_test.reset_index(drop=True)
    self.df_train = self.df_train.reset_index(drop=True)

    # sort again because sampling shuffles order:
    self.df_test.sort_values(by=self.days_field, ascending=False, inplace=True)
    self.df_train.sort_values(by=self.days_field, ascending=False, inplace=True)

    if self.hedonic and self.hedonic_test_against_vacant_sales:
      # if it's a hedonic model, we're predicting land value, and are thus testing against vacant land only:
      # we have to do this here, AFTER the split, to ensure that the selected rows are from the same subsets

      # get the sales that are actually vacant, from the original set of sales
      _df_sales = get_sales(self._df_sales, self.settings, True).reset_index(drop=True)

      # now, select only those records from the modified base sales set that are also in the above set,
      # but use the rows from the modified base sales set
      _df_sales = self.df_sales[self.df_sales["key_sale"].isin(_df_sales["key_sale"])].reset_index(drop=True)

      # use these as our sales
      self.df_sales = _df_sales

      # set df_test/train to only those rows that are also in sales:
      # we don't need to use get_sales() because they've already been transformed to vacant
      self.df_test = self.df_test[self.df_test["key_sale"].isin(self.df_sales["key_sale"])].reset_index(drop=True)
      self.df_train = self.df_train[self.df_train["key_sale"].isin(self.df_sales["key_sale"])].reset_index(drop=True)

    _df_univ = self.df_universe.copy()
    _df_sales = self.df_sales.copy()
    _df_train = self.df_train.copy()
    _df_test = self.df_test.copy()

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

    ind_vars = [col for col in self.ind_vars if col in _df_univ.columns]
    self.X_univ = _df_univ[ind_vars]

    ind_vars = [col for col in self.ind_vars if col in _df_sales.columns]
    self.X_sales = _df_sales[ind_vars]
    self.y_sales = _df_sales[self.dep_var]

    ind_vars = [col for col in self.ind_vars if col in _df_train.columns]

    self.X_train = _df_train[ind_vars]
    self.y_train = _df_train[self.dep_var]

    idx_vacant = _df_train["bldg_area_finished_sqft"] <= 0

    # set the train sizes to the building area for improved properties, and the land area for vacant properties
    _df_train["size"] = _df_train["bldg_area_finished_sqft"]
    _df_train.loc[idx_vacant, "size"] = _df_train["land_area_sqft"]
    self.train_sizes = _df_train["size"]

    # make sure it's a float64
    self.train_sizes = self.train_sizes.astype("float64")

    # set the cluster to the "he_id":
    if "he_id" in _df_train:
      self.train_he_ids = _df_train["he_id"]

    if "land_he_id" in _df_train:
      self.train_land_he_ids = _df_train["land_he_id"]

    if "impr_he_id" in _df_train:
      self.train_impr_he_ids = _df_train["impr_he_id"]

    # convert all Float64 to float64 in X_train:
    for col in self.X_train.columns:
      # if it's a Float64 or a boolean, convert it to float64
      if (self.X_train[col].dtype == "Float64" or
          self.X_train[col].dtype == "Int64" or
          self.X_train[col].dtype == "boolean" or
          self.X_train[col].dtype == "bool"
      ):
        self.X_train = self.X_train.astype({col: "float64"})

    ind_vars = [col for col in self.ind_vars if col in _df_test.columns]
    self.X_test = _df_test[ind_vars]
    self.y_test = _df_test[self.dep_var_test]


class SingleModelResults:
  """
  Container for results from a single model prediction.

  Attributes:
      ds (DataSplit): The data split object used
      df_universe (pd.DataFrame): Universe DataFrame
      df_test (pd.DataFrame): Test DataFrame
      df_sales (pd.DataFrame, optional): Sales DataFrame
      type (str): Model type identifier
      dep_var (str): Independent variable name
      ind_vars (list[str]): Dependent variable names
      model (PredictionModel): The model used for prediction
      pred_test (PredictionResults): Results for the test set
      pred_sales (PredictionResults, optional): Results for the sales set
      pred_univ: Predictions for the universe (all parcels in the current scope, such as a model group)
      chd (float): Calculated CHD value
      utility (float): Composite utility score, used for comparing models
      timing (TimingData): Timing data for different phases of the model run
  """

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
      verbose: bool = False
  ):
    """
    Initialize SingleModelResults by attaching predictions and computing performance metrics.

    :param ds: DataSplit object containing all necessary splits.
    :type ds: DataSplit
    :param field_prediction: The field name for predictions
    :type field_prediction: str
    :param field_horizontal_equity_id: The field name for the horizontal equity ID
    :type field_horizontal_equity_id: str
    :param type: Model type identifier
    :type type: str
    :param model: The model used
    :type model: PredictionModel
    :param y_pred_test: Predictions on the test set
    :type y_pred_test: numpy.ndarray
    :param y_pred_sales: Predictions on the sales set
    :type y_pred_sales: numpy.ndarray or None
    :param y_pred_univ: Predictions on the universe set
    :type y_pred_univ: numpy.ndarray
    :param timing: TimingData object.
    :type timing: TimingData
    :param verbose: Whether to print verbose output.
    :type verbose: bool, optional
    """
    self.ds = ds

    df_univ = ds.df_universe.copy()
    df_sales = ds.df_sales.copy()
    df_test = ds.df_test.copy()

    self.field_prediction = field_prediction

    df_univ[field_prediction] = y_pred_univ

    df_test[field_prediction] = y_pred_test

    self.df_universe = df_univ
    self.df_test = df_test

    if y_pred_sales is not None:
      df_sales[field_prediction] = y_pred_sales
      self.df_sales = df_sales

    self.type = type
    self.dep_var = ds.dep_var
    self.dep_var_test = ds.dep_var_test
    self.ind_vars = ds.ind_vars.copy()
    self.model = model

    timing.start("stats_test")
    self.pred_test = PredictionResults(self.dep_var_test, self.ind_vars, field_prediction, df_test)
    timing.stop("stats_test")

    timing.start("stats_sales")
    if y_pred_sales is not None:
      self.pred_sales = PredictionResults(self.dep_var_test, self.ind_vars, field_prediction, df_sales)
    timing.stop("stats_sales")

    self.pred_univ = y_pred_univ

    self._deal_with_log_and_sqft()

    timing.start("chd")
    df_univ_valid = df_univ.copy()
    df_univ_valid = pd.DataFrame(df_univ_valid)  # Ensure it's a Pandas DataFrame
    # drop problematic columns:
    df_univ_valid.drop(columns=["geometry"], errors="ignore", inplace=True)

    # convert all category and string[python] types to string:
    for col in df_univ_valid.columns:
      if df_univ_valid[col].dtype in ["category", "string"]:
        df_univ_valid[col] = df_univ_valid[col].astype("str")
    pl_df = pl.DataFrame(df_univ_valid)

    # TODO: This might need to be changed to be the $/sqft value rather than the total value
    self.chd = quick_median_chd_pl(pl_df, field_prediction, field_horizontal_equity_id)
    timing.stop("chd")

    timing.start("utility")
    self.utility = model_utility_score(self)
    timing.stop("utility")
    self.timing = timing


  def _deal_with_log_and_sqft(self):
    # if it's a log model, we need to exponentiate the predictions
    if self.dep_var.startswith("log_"):
      self.pred_sales.y_pred = np.exp(self.pred_sales.y_pred)
      self.pred_univ = np.exp(self.pred_univ)
    if self.dep_var_test.startswith("log_"):
      self.pred_test.y_pred = np.exp(self.pred_test.y_pred)

    # if it's a sqft model, we need to further multiply the predictions by the size
    for suffix in ["_size", "_land_sqft", "_impr_sqft"]:
      if self.dep_var.endswith(suffix):
        self.pred_sales.y_pred = self.pred_sales.y_pred * self.ds.df_sales[suffix]
        self.pred_univ = self.pred_univ * self.ds.df_universe[suffix]
      if self.dep_var_test.startswith("log_"):
        self.pred_test.y_pred = self.pred_test.y_pred * self.ds.df_test[suffix]



  def summary(self):
    """
    Generate a summary string of model performance.

    The summary includes model type, number of rows in test & universe sets, RMSE, R², adjusted R²,
    median ratio, COD, PRD, PRB, and CHD.

    :returns: Summary string.
    :rtype: str
    """
    str = ""
    str += f"Model type: {self.type}\n"
    # Print the # of rows in test & universe set
    # Print the MSE, RMSE, R2, and Adj R2 for test & universe set
    str += f"-->Test set, rows: {len(self.pred_test.y)}\n"
    str += f"---->RMSE   : {self.pred_test.rmse:8.0f}\n"
    str += f"---->R2     : {self.pred_test.r2:8.4f}\n"
    str += f"---->Adj R2 : {self.pred_test.adj_r2:8.4f}\n"
    str += f"---->M.Ratio: {self.pred_test.ratio_study.median_ratio:8.4f}\n"
    str += f"---->COD    : {self.pred_test.ratio_study.cod:8.4f}\n"
    str += f"---->PRD    : {self.pred_test.ratio_study.prd:8.4f}\n"
    str += f"---->PRB    : {self.pred_test.ratio_study.prb:8.4f}\n"
    str += f"\n"
    str += f"-->Universe set, rows: {len(self.pred_sales.y)}\n"
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


def land_utility_score(land_results: LandPredictionResults):
  # Utility score is a composite score based on the following:
  # 1. Accuracy:
  #   - Land ratio study median ratio
  #   - Land ratio study untrimmed COD
  # 2. Consistency:
  #   - Land CHD
  #   - Impr CHD
  # 3. Sanity:
  #   - All the various sanity checks

  # Normalization values
  cod_base = 15
  chd_land_base = 15
  chd_impr_base = 30 # we're more tolerant of higher CHD values for improvement than for land
  dist_ratio_base = 0.01

  # Weights
  weight_dist_ratio = 10.0
  weight_chd_land = 10.0
  weight_chd_impr = 10.0
  weight_sanity = 100.0

  weight_cod = 1.0
  weight_invalid = 2.0
  weight_overshoot = 10.0
  weight_undershoot = 1.0

  # penalize over-estimates; err on the side of under-estimates
  ratio_over_penalty = 2 if land_results.land_ratio_study.median_ratio < 1.05 else 1

  cod = land_results.land_ratio_study.cod
  dist_ratio = abs(1.0 - cod)

  # Normalize the scores around the base values
  cod_score = cod / cod_base
  dist_ratio_score = dist_ratio / dist_ratio_base
  chd_land_score = land_results.land_chd / chd_land_base
  chd_impr_score = land_results.impr_chd / chd_impr_base

  # Calculate weighted components
  weighted_cod_score = cod_score * weight_cod
  weighted_dist_ratio_score = dist_ratio_score * weight_dist_ratio * ratio_over_penalty

  weighted_chd_land_score = chd_land_score * weight_chd_land
  weighted_chd_impr_score = chd_impr_score * weight_chd_impr
  weighted_chd_score = weighted_chd_land_score + weighted_chd_impr_score

  # sanity
  perc_invalid = ((100 * land_results.perc_land_invalid) +
                  (100 * land_results.perc_impr_invalid) +
                  (100 * land_results.perc_dont_add_up))
  perc_overshoot = (100 * land_results.perc_land_overshoot)
  perc_undershoot = (100 * land_results.perc_vacant_land_not_100)

  perc_invalid *= weight_invalid
  perc_overshoot *= weight_overshoot
  perc_undershoot *= weight_undershoot

  sanity_score = (perc_invalid + perc_overshoot + perc_undershoot)
  weighted_sanity_score = sanity_score * weight_sanity

  final_score = weighted_dist_ratio_score + weighted_cod_score + weighted_chd_score + weighted_sanity_score
  return final_score



def model_utility_score(model_results: SingleModelResults):
  """
  Compute a utility score for a model based on error, median ratio, COD, and CHD. Lower scores are better.

  This function is the weighted average of the following: median ratio distance from 1.0, COD, CHD. It also adds a
  penalty for suspiciously low COD values, to punish sales chasing.

  :param model_results: SingleModelResults object.
  :type model_results: SingleModelResults
  :returns: Computed utility score.
  :rtype: float
  """
  weight_dist_ratio = 1000.00
  weight_cod = 1.50
  weight_chd = 1.00
  weight_sales_chase = 7.5

  cod = model_results.pred_test.ratio_study.cod
  chd = model_results.chd

  # Penalize over-estimates; err on the side of under-estimates
  ratio_over_penalty = 2 if model_results.pred_test.ratio_study.median_ratio < 1.05 else 1

  # calculate base score
  dist_ratio_score = abs(1.0 - model_results.pred_test.ratio_study.median_ratio) * weight_dist_ratio * ratio_over_penalty
  cod_score = cod * weight_cod
  chd_score = chd * weight_chd

  # penalize very low COD's with bad horizontal equity
  if cod == 0.0:
    cod = 1e-6
  sales_chase_score = ((1.0/cod) * chd) * weight_sales_chase
  final_score = dist_ratio_score + cod_score + chd_score + sales_chase_score
  return final_score


def safe_predict(callable, X: Any, params: Dict[str, Any] = None):
  """
  Safely obtain predictions from a callable model function (Returns an empty array if the input is empty).

  :param callable: Prediction function.
  :type callable: callable
  :param X: Input features.
  :type X: Any
  :param params: Additional parameters for the callable.
  :type params: dict, optional
  :returns: Predicted values as a NumPy array.
  :rtype: numpy.ndarray
  """
  if len(X) == 0:
    return np.array([])
  if params is None:
    params = {}
  return callable(X, **params)


def predict_mra(ds: DataSplit, model: MRAModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a Multiple Regression Analysis (MRA) model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param model: MRAModel instance.
  :type model: MRAModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: SingleModelResults with predictions.
  :rtype: SingleModelResults
  """
  fitted_model: RegressionResults = model.fitted_model

  # predict on test set:
  timing.start("predict_test")
  y_pred_test = safe_predict(fitted_model.predict, ds.X_test)
  timing.stop("predict_test")

  # predict on the sales set:
  timing.start("predict_sales")
  y_pred_sales = safe_predict(fitted_model.predict, ds.X_sales)
  timing.stop("predict_sales")

  # predict on the universe set:
  timing.start("predict_univ")
  y_pred_univ = safe_predict(fitted_model.predict, ds.X_univ)
  timing.stop("predict_univ")

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
    verbose=verbose
  )

  return results


def run_mra(ds: DataSplit, intercept: bool = True, verbose: bool = False, model: MRAModel | None = None):
  """
  Train an MRA model and return its prediction results.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param intercept: Whether to include an intercept in the model.
  :type intercept: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :param model: Optional pre-trained MRAModel.
  :type model: MRAModel or None
  :returns: Prediction results from the MRA model.
  :rtype: SingleModelResults
  """
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

  ds.X_train = ds.X_train.astype(float)
  ds.y_train = ds.y_train.astype(float)

  timing.start("train")
  if model is None:
    linear_model = sm.OLS(ds.y_train, ds.X_train)
    fitted_model = linear_model.fit()
    model = MRAModel(fitted_model, intercept)
  timing.stop("train")

  return predict_mra(ds, model, timing, verbose)


def predict_ground_truth(ds: DataSplit, ground_truth_model: GroundTruthModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a ground truth model.

  Uses the observed field (e.g. sale price) as the "prediction" and compares it against the ground truth field (e.g. true market value in a synthetic model)

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param ground_truth_model: GroundTruthModel instance.
  :type ground_truth_model: GroundTruthModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: SingleModelResults with assessor predictions.
  :rtype: SingleModelResults
  """
  observed_field = ground_truth_model.observed_field
  ground_truth_field = ground_truth_model.ground_truth_field

  model_name = "ground_truth"

  # predict on test set:
  timing.start("predict_test")
  y_pred_test = ds.df_test[observed_field].to_numpy()
  timing.stop("predict_test")

  # predict on the sales set:
  timing.start("predict_sales")
  y_pred_sales = ds.df_sales[observed_field].to_numpy()
  timing.stop("predict_sales")

  # predict on the universe set:
  timing.start("predict_univ")
  y_pred_univ = ds.df_universe[observed_field].to_numpy()# ds.X_univ[observed_field].to_numpy()
  timing.stop("predict_univ")

  timing.stop("total")

  ds = ds.copy()
  ds.dep_var = ground_truth_field
  ds.dep_var_test = ground_truth_field

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    model_name,
    ground_truth_model,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose
  )

  return results


def predict_spatial_lag(ds: DataSplit, model: SpatialLagModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a spatial lag model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param model: SpatialLagModel instance.
  :type model: SpatialLagModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: SingleModelResults with spatial lag predictions.
  :rtype: SingleModelResults
  """

  if model.per_sqft == False:
    field = ds.ind_vars[0]

    # predict on test set:
    timing.start("predict_test")
    y_pred_test = ds.X_test[field].to_numpy()
    timing.stop("predict_test")

    # predict on the sales set:
    timing.start("predict_sales")
    y_pred_sales = ds.X_sales[field].to_numpy()
    timing.stop("predict_sales")

    # predict on the universe set:
    timing.start("predict_univ")
    y_pred_univ = ds.X_univ[field].to_numpy()
    timing.stop("predict_univ")

  else:
    field_impr_sqft = ""
    field_land_sqft = ""
    for field in ds.ind_vars:
      if "spatial_lag" in field:
        if "impr_sqft" in field:
          field_impr_sqft = field
        if "land_sqft" in field:
          field_land_sqft = field
    if field_impr_sqft == "":
      raise ValueError("No field found for spatial lag with 'impr_sqft'")
    if field_land_sqft == "":
      raise ValueError("No field found for spatial lag with 'land_sqft'")

    # predict on test set:
    timing.start("predict_test")
    idx_vacant_test = ds.X_test["bldg_area_finished_sqft"].le(0)
    y_pred_test = ds.X_test[field_impr_sqft].to_numpy() * ds.X_test["bldg_area_finished_sqft"].to_numpy()
    y_pred_test[idx_vacant_test] = (
        ds.X_test[field_land_sqft].to_numpy()[idx_vacant_test] *
        ds.X_test["land_area_sqft"].to_numpy()[idx_vacant_test]
    )
    timing.stop("predict_test")

    # predict on the sales set:
    timing.start("predict_sales")
    idx_vacant_sales = ds.X_sales["bldg_area_finished_sqft"].le(0)
    y_pred_sales = ds.X_sales[field_impr_sqft].to_numpy() * ds.X_sales["bldg_area_finished_sqft"].to_numpy()
    y_pred_sales[idx_vacant_sales] = (
        ds.X_sales[field_land_sqft].to_numpy()[idx_vacant_sales] *
        ds.X_sales["land_area_sqft"].to_numpy()[idx_vacant_sales]
    )
    timing.stop("predict_sales")

    # predict on the universe set:
    timing.start("predict_univ")
    idx_vacant_univ = ds.X_univ["bldg_area_finished_sqft"].le(0)
    y_pred_univ = ds.X_univ[field_impr_sqft].to_numpy() * ds.X_univ["bldg_area_finished_sqft"].to_numpy()
    y_pred_univ[idx_vacant_univ] = (
        ds.X_univ[field_land_sqft].to_numpy()[idx_vacant_univ] *
        ds.X_univ["land_area_sqft"].to_numpy()[idx_vacant_univ]
    )
    timing.stop("predict_univ")

  timing.stop("total")

  name = "spatial_lag"
  if model.per_sqft:
    name = "spatial_lag_per_sqft"

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    name,
    model,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose
  )

  return results


def predict_pass_through(ds: DataSplit, model: PassThroughModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using an assessor model.

  Uses the specified field from the assessor model (or the first dependent variable if hedonic)
  to extract predictions directly from the input DataFrames.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param model: PassThroughModel instance.
  :type model: PassThroughModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: SingleModelResults with assessor predictions.
  :rtype: SingleModelResults
  """
  field = model.field

  # TODO: genericize this to take any field name and label
  model_name = "assessor"

  if ds.hedonic:
    field = ds.ind_vars[0]

  # predict on test set:
  timing.start("predict_test")
  y_pred_test = ds.X_test[field].to_numpy()
  timing.stop("predict_test")

  # predict on the sales set:
  timing.start("predict_sales")
  y_pred_sales = ds.X_sales[field].to_numpy()
  timing.stop("predict_sales")

  # predict on the universe set:
  timing.start("predict_univ")
  y_pred_univ = ds.X_univ[field].to_numpy()
  timing.stop("predict_univ")

  timing.stop("total")

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    model_name,
    model,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose
  )

  return results


def run_ground_truth(ds: DataSplit, verbose: bool = False):
  """
  Run a ground truth model by performing data splitting and returning predictions.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the ground truth model.
  :rtype: SingleModelResults
  """
  timing = TimingData()

  timing.start("total")

  timing.start("setup")
  ds.split()
  timing.stop("setup")

  timing.start("parameter_search")
  timing.stop("parameter_search")

  timing.start("train")
  timing.stop("train")

  ground_truth_model = GroundTruthModel(
    observed_field=ds.dep_var,
    ground_truth_field=ds.ind_vars[0]
  )
  return predict_ground_truth(ds, ground_truth_model, timing, verbose)


def run_spatial_lag(ds: DataSplit, per_sqft: bool = False, verbose: bool = False):
  """
  Run a spatial lag model by performing data splitting and returning predictions.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param per_sqft: Whether to normalize the model by sqft.
  :type per_sqft: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the spatial lag model.
  :rtype: SingleModelResults
  """
  timing = TimingData()

  timing.start("total")

  timing.start("setup")
  ds.split()
  timing.stop("setup")

  timing.start("parameter_search")
  timing.stop("parameter_search")

  timing.start("train")
  timing.stop("train")

  model = SpatialLagModel(per_sqft=per_sqft)
  return predict_spatial_lag(ds, model, timing, verbose)


def run_pass_through(ds: DataSplit, verbose: bool = False):
  """
  Run an assessor model by performing data splitting and returning predictions.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the assessor model.
  :rtype: SingleModelResults
  """
  timing = TimingData()

  timing.start("total")

  timing.start("setup")
  ds.split()
  timing.stop("setup")

  timing.start("parameter_search")
  timing.stop("parameter_search")

  timing.start("train")
  timing.stop("train")

  model = PassThroughModel(ds.ind_vars[0])
  return predict_pass_through(ds, model, timing, verbose)


def predict_kernel(ds: DataSplit, kr: KernelReg, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a kernel regression model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param kr: KernelReg model instance.
  :type kr: KernelReg
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the kernel regression model.
  :rtype: SingleModelResults
  """
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

  if verbose:
    print(f"--> predicting on test set...")
  # Predict at original locations:
  timing.start("predict_test")
  y_pred_test = np.zeros(X_test.shape[0])
  if kr is not None:
    try:
      y_pred_test, _ = kr.fit(X_test)
    except numpy.linalg.LinAlgError as e:
      print(f"--> Error in kernel regression: {e}")
      y_pred_test = np.zeros(X_test.shape[0])
  timing.stop("predict_test")

  if verbose:
    print(f"--> predicting on sales set...")
  timing.start("predict_sales")
  y_pred_sales = np.zeros(X_sales.shape[0])
  if kr is not None:
    try:
      y_pred_sales, _ = kr.fit(X_sales)
    except numpy.linalg.LinAlgError as e:
      print(f"--> Error in kernel regression: {e}")
      y_pred_sales = np.zeros(X_sales.shape[0])
  timing.stop("predict_sales")

  if verbose:
    print(f"--> predicting on universe set...")
  timing.start("predict_univ")
  y_pred_univ = np.zeros(X_univ.shape[0])
  if kr is not None:
    try:
      y_pred_univ, _ = kr.fit(X_univ)
    except numpy.linalg.LinAlgError as e:
      print(f"--> Error in kernel regression: {e}")
      y_pred_univ = np.zeros(X_univ.shape[0])
  timing.stop("predict_univ")

  timing.stop("total")

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    "kernel",
    kr,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing,
    verbose=verbose
  )

  return results


def run_kernel(ds: DataSplit, outpath: str, save_params: bool = False, use_saved_params: bool = False, verbose: bool = False):
  """
  Run a kernel regression model by tuning its bandwidth and returning predictions.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param outpath: Path to store output parameters.
  :type outpath: str
  :param save_params: Whether to save the tuned parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to load saved parameters.
  :type use_saved_params: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the kernel regression model.
  :rtype: SingleModelResults
  """
  timing = TimingData()

  timing.start("total")

  timing.start("setup")
  ds = ds.encode_categoricals_with_one_hot()
  ds.split()
  u_train = ds.df_train['longitude']
  v_train = ds.df_train['latitude']
  vars_train = (u_train, v_train)

  for col in ds.X_train.columns:

    # check if every value is the same:
    if ds.X_train[col].nunique() == 1:
      # add a very small amount of random noise
      # this is to prevent singular matrix errors in the Kernel regression
      ds.X_train[col] += np.random.normal(0, 1e-6, ds.X_train[col].shape)

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
  try:
    kr = KernelReg(endog=y_train, exog=X_train, var_type=var_type, bw=kernel_bw, defaults=defaults)
    kernel_bw = kr.bw
    if save_params:
      os.makedirs(outpath, exist_ok=True)
      with open(f"{outpath}/kernel_bw.pkl", "wb") as f:
        pickle.dump(kernel_bw, f)
  except numpy.linalg.LinAlgError as e:
    print(f"--> Error in kernel regression: {e}")
    print("Kernel regression failed. Please check your data.")
    kr = None

  if verbose:
    print(f"--> optimal bandwidth = {kernel_bw}")
  timing.stop("train")

  return predict_kernel(ds, kr, timing, verbose)


def predict_gwr(ds: DataSplit, gwr_model: GWRModel, timing: TimingData, verbose: bool, diagnostic: bool = False, intercept: bool = True):
  """
  Generate predictions using a Geographically Weighted Regression (GWR) model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param gwr_model: GWRModel instance containing training data and parameters.
  :type gwr_model: GWRModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool
  :param diagnostic: If True, use diagnostic mode.
  :type diagnostic: bool, optional
  :param intercept: Whether the model includes an intercept.
  :type intercept: bool, optional
  :returns: Prediction results from the GWR model.
  :rtype: SingleModelResults
  """
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
  coords_univ = list(zip(u, v))

  np_coords_test = np.array(coords_test)
  timing.start("predict_test")

  if len(np_coords_test) == 0 or len(X_test) == 0:
    y_pred_test = np.array([])
  else:
    gwr_result_test = gwr.predict(
      np_coords_test,
      X_test
    )
    y_pred_test = gwr_result_test.predictions.flatten()
  timing.stop("predict_test")

  timing.start("predict_sales")
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
    ind_vars=ds.ind_vars
  ).flatten()
  timing.stop("predict_univ")

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
    timing
  )
  timing.stop("total")

  return results


def run_gwr(ds: DataSplit, outpath: str, save_params: bool = False, use_saved_params: bool = False, verbose: bool = False, diagnostic: bool = False):
  """
  Run a GWR model by tuning its bandwidth and generating predictions.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param outpath: Output path for saving parameters.
  :type outpath: str
  :param save_params: Whether to save tuned parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to load saved parameters.
  :type use_saved_params: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :param diagnostic: If True, run in diagnostic mode.
  :type diagnostic: bool, optional
  :returns: Prediction results from the GWR model.
  :rtype: SingleModelResults
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
    bw_max = len(y_train)

    try:
      gwr_selector = Sel_BW(coords_train, y_train, X_train)
      gwr_bw = gwr_selector.search(bw_max=bw_max)
    except ValueError:
      if len(y_train) < 100:
        # Set n_jobs to 1 in case the # of cores exceeds the number of rows
        gwr_selector = Sel_BW(coords_train, y_train, X_train, fixed=True, n_jobs = 1)
        gwr_bw = gwr_selector.search()
      else:
        # Use default n_jobs
        gwr_selector = Sel_BW(coords_train, y_train, X_train, fixed=True)
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


def predict_xgboost(ds: DataSplit, xgboost_model: xgboost.XGBRegressor, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using an XGBoost model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param xgboost_model: Trained XGBRegressor instance.
  :type xgboost_model: xgboost.XGBRegressor
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the XGBoost model.
  :rtype: SingleModelResults
  """
  timing.start("predict_test")
  y_pred_test = safe_predict(xgboost_model.predict, ds.X_test)
  timing.stop("predict_test")

  timing.start("predict_sales")
  y_pred_sales = safe_predict(xgboost_model.predict, ds.X_sales)
  timing.stop("predict_sales")

  timing.start("predict_univ")
  y_pred_univ = safe_predict(xgboost_model.predict, ds.X_univ)
  timing.stop("predict_univ")

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
    verbose=verbose
  )
  return results


def run_xgboost(ds: DataSplit, outpath: str, save_params: bool = False, use_saved_params: bool = False, verbose: bool = False):
  """
  Run an XGBoost model by tuning parameters, training, and predicting.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param outpath: Output path for saving parameters.
  :type outpath: str
  :param save_params: Whether to save tuned parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to load saved parameters.
  :type use_saved_params: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the XGBoost model.
  :rtype: SingleModelResults
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

  # Print timing information for XGBoost model
  # if verbose:
  #   print("\n***** XGBoost Model Timing *****")
  #   print(timing.print())
  #   print("*********************************\n")

  return predict_xgboost(ds, xgboost_model, timing, verbose)


def predict_lightgbm(ds: DataSplit, gbm: lgb.Booster, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a LightGBM model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param gbm: Trained LightGBM Booster.
  :type gbm: lgb.Booster
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the LightGBM model.
  :rtype: SingleModelResults
  """
  timing.start("predict_test")
  y_pred_test = safe_predict(gbm.predict, ds.X_test, {"num_iteration": gbm.best_iteration})
  timing.stop("predict_test")

  timing.start("predict_sales")
  y_pred_sales = safe_predict(gbm.predict, ds.X_sales, {"num_iteration": gbm.best_iteration})
  timing.stop("predict_sales")

  timing.start("predict_univ")
  y_pred_univ = safe_predict(gbm.predict, ds.X_univ, {"num_iteration": gbm.best_iteration})
  timing.stop("predict_univ")

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
    verbose=verbose
  )
  return results


def run_lightgbm(ds: DataSplit, outpath: str, save_params: bool = False, use_saved_params: bool = False, verbose: bool = False):
  """
  Run a LightGBM model by tuning parameters, training, and predicting.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param outpath: Output path for saving parameters.
  :type outpath: str
  :param save_params: Whether to save tuned parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to load saved parameters.
  :type use_saved_params: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the LightGBM model.
  :rtype: SingleModelResults
  """
  timing = TimingData()

  timing.start("total")

  timing.start("setup")
  ds = ds.encode_categoricals_with_one_hot()
  ds.split()
  timing.stop("setup")

  timing.start("parameter_search")
  params = _get_params("LightGBM", "lightgbm", ds, tune_lightgbm, outpath, save_params, use_saved_params, verbose)
  timing.stop("parameter_search")

  timing.start("train")
  cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]
  lgb_train = lgb.Dataset(ds.X_train, ds.y_train, categorical_feature=cat_vars)
  lgb_test = lgb.Dataset(ds.X_test, ds.y_test, categorical_feature=cat_vars, reference=lgb_train)

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

  # Print timing information for LightGBM model
  # if verbose:
  #   print("\n***** LightGBM Model Timing *****")
  #   print(timing.print())
  #   print("*********************************\n")
  
  return predict_lightgbm(ds, gbm, timing, verbose)


def predict_catboost(ds: DataSplit, catboost_model: catboost.CatBoostRegressor, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a CatBoost model.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param catboost_model: Trained CatBoostRegressor.
  :type catboost_model: catboost.CatBoostRegressor
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the CatBoost model.
  :rtype: SingleModelResults
  """
  cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]

  timing.start("predict_test")
  if len(ds.y_test) == 0:
    y_pred_test = np.array([])
  else:
    test_pool = Pool(data=ds.X_test, label=ds.y_test, cat_features=cat_vars)
    y_pred_test = catboost_model.predict(test_pool)
  timing.stop("predict_test")

  timing.start("predict_sales")
  if len(ds.y_sales) == 0:
    y_pred_sales = np.array([])
  else:
    sales_pool = Pool(data=ds.X_sales, label=ds.y_sales, cat_features=cat_vars)
    y_pred_sales = catboost_model.predict(sales_pool)
  timing.stop("predict_sales")

  timing.start("predict_univ")
  if len(ds.X_univ) == 0:
    y_pred_univ = np.array([])
  else:
    univ_pool = Pool(data=ds.X_univ, cat_features=cat_vars)
    y_pred_univ = catboost_model.predict(univ_pool)
  timing.stop("predict_univ")

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
    verbose=verbose
  )

  return results


def run_catboost(ds: DataSplit, outpath: str, save_params: bool = False, use_saved_params: bool = False, verbose: bool = False):
  """
  Run a CatBoost model by tuning parameters, training, and predicting.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param outpath: Output path for saving parameters.
  :type outpath: str
  :param save_params: Whether to save tuned parameters.
  :type save_params: bool, optional
  :param use_saved_params: Whether to load saved parameters.
  :type use_saved_params: bool, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the CatBoost model.
  :rtype: SingleModelResults
  """
  timing = TimingData()

  timing.start("total")

  timing.start("parameter_search")
  params = _get_params("CatBoost", "catboost", ds, tune_catboost, outpath, save_params, use_saved_params, verbose)
  timing.stop("parameter_search")

  timing.start("setup")
  params["verbose"] = False
  params["train_dir"] = f"{outpath}/catboost/catboost_info"
  os.makedirs(params["train_dir"], exist_ok=True)
  cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]
  catboost_model = catboost.CatBoostRegressor(**params)
  train_pool = Pool(data=ds.X_train, label=ds.y_train, cat_features=cat_vars)
  timing.stop("setup")

  timing.start("train")
  catboost_model.fit(train_pool)
  timing.stop("train")

  return predict_catboost(ds, catboost_model, timing, verbose)


def predict_garbage(ds: DataSplit, garbage_model: GarbageModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a "garbage" model that produces random values.

  If sales_chase is specified, adjusts predictions to simulate sales chasing behavior.

  Needless to say, you should not use this model in production.  

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param garbage_model: GarbageModel instance containing configuration.
  :type garbage_model: GarbageModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the garbage model.
  :rtype: SingleModelResults
  """
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

  timing.stop("total")

  df = ds.df_universe
  dep_var = ds.dep_var

  if sales_chase:
    y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
    y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

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
    verbose=verbose
  )

  return results


def run_garbage(ds: DataSplit, normal: bool = False, sales_chase: float = 0.0, verbose: bool = False):
  """
  Run a garbage model that predicts random values within a range derived from the training set.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param normal: If True, use a normal distribution; otherwise, use a uniform distribution.
  :type normal: bool, optional
  :param sales_chase: Factor for simulating sales chasing (default 0.0 means no adjustment).
  :type sales_chase: float, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the garbage model.
  :rtype: SingleModelResults
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


def predict_average(ds: DataSplit, average_model: AverageModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions by simply using the average (mean or median) of the training set.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param average_model: AverageModel instance with configuration.
  :type average_model: AverageModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the average model.
  :rtype: SingleModelResults
  """
  timing.start("predict_test")
  type = average_model.type
  sales_chase = average_model.sales_chase

  if type == "median":
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

  timing.stop("total")

  df = ds.df_universe
  dep_var = ds.dep_var

  if sales_chase:
    y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
    y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

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
    verbose=verbose
  )

  return results


def run_average(ds: DataSplit, average_type: str = "mean", sales_chase: float = 0.0, verbose: bool = False):
  """
  Run an average model that predicts either the mean or median of the training set for all predictions.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param average_type: "mean" or "median" indicating which statistic to use.
  :type average_type: str, optional
  :param sales_chase: Factor for simulating sales chasing (default 0.0 means no adjustment).
  :type sales_chase: float, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the average model.
  :rtype: SingleModelResults
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


def predict_naive_sqft(ds: DataSplit, sqft_model: NaiveSqftModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a naive per-square-foot model.

  Separately computes predictions for improved and vacant properties based on bldg_area_finished_sqft
  and land_area_sqft, then combines them.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param sqft_model: NaiveSqftModel instance containing per-square-foot multipliers.
  :type sqft_model: NaiveSqftModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the naive sqft model.
  :rtype: SingleModelResults
  """
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

  timing.stop("total")

  df = ds.df_universe
  dep_var = ds.dep_var

  if sales_chase:
    y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
    y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

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
    verbose=verbose
  )

  return results


def run_naive_sqft(ds: DataSplit, sales_chase: float = 0.0, verbose: bool = False):
  """
  Run a naive per-square-foot model that predicts based on median $/sqft from the training set.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param sales_chase: Factor for simulating sales chasing (default 0.0 means no adjustment).
  :type sales_chase: float, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the naive sqft model.
  :rtype: SingleModelResults
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


def predict_local_sqft(ds: DataSplit, sqft_model: LocalSqftModel, timing: TimingData, verbose: bool = False):
  """
  Generate predictions using a local per-square-foot model that uses location-specific values.

  Merges local per-square-foot values computed for different location fields with the test set,
  and then computes predictions based on improved and vacant properties.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param sqft_model: LocalSqftModel instance containing location-specific multipliers.
  :type sqft_model: LocalSqftModel
  :param timing: TimingData object.
  :type timing: TimingData
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: Prediction results from the local sqft model.
  :rtype: SingleModelResults
  """
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
  df_land["per_land_sqft"] = 0.0  # Initialize as float
  df_impr["per_impr_sqft"] = 0.0  # Initialize as float

  # go from most specific to the least specific location (first to last)
  for location_field in location_fields:
    df_sqft_impr, df_sqft_land = loc_map[location_field]

    df_impr = df_impr.merge(df_sqft_impr[[location_field, f"{location_field}_per_impr_sqft"]], on=location_field, how="left")
    df_land = df_land.merge(df_sqft_land[[location_field, f"{location_field}_per_land_sqft"]], on=location_field, how="left")

    df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = df_impr[f"{location_field}_per_impr_sqft"]
    df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = df_land[f"{location_field}_per_land_sqft"]

    # df_sqft_land.to_csv(f"debug_local_sqft_{len(location_fields)}_{location_field}_sqft_land.csv", index=False)
    # df_land.to_csv(f"debug_local_sqft_{len(location_fields)}_{location_field}_land.csv", index=False)

  # any remaining zeroes get filled with the locality-wide median value
  df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
  df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft

  X_test = ds.X_test

  df_impr = df_impr[["key", "per_impr_sqft"]]
  df_land = df_land[["key", "per_land_sqft"]]

  # merge the df_sqft_land/impr values into the X_test dataframe:
  X_test["key_sale"] = ds.df_test["key_sale"]
  X_test["key"] = ds.df_test["key"]
  X_test = X_test.merge(df_land, on="key", how="left")
  X_test = X_test.merge(df_impr, on="key", how="left")
  X_test.loc[X_test["per_impr_sqft"].isna() | X_test["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
  X_test.loc[X_test["per_land_sqft"].isna() | X_test["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
  X_test = X_test.drop(columns=["key_sale", "key"])

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
  X_sales["key_sale"] = ds.df_sales["key_sale"]
  X_sales["key"] = ds.df_sales["key"]
  X_sales = X_sales.merge(df_land, on="key", how="left")
  X_sales = X_sales.merge(df_impr, on="key", how="left")
  X_sales.loc[X_sales["per_impr_sqft"].isna() | X_sales["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
  X_sales.loc[X_sales["per_land_sqft"].isna() | X_sales["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
  X_sales = X_sales.drop(columns=["key_sale", "key"])

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
  X_univ.loc[X_univ["prediction_impr"].isna() | X_univ["prediction_impr"].eq(0), "prediction_impr"] = overall_per_impr_sqft
  X_univ.loc[X_univ["prediction_land"].isna() | X_univ["prediction_land"].eq(0), "prediction_land"] = overall_per_land_sqft
  X_univ["prediction"] = np.where(X_univ["bldg_area_finished_sqft"].gt(0), X_univ["prediction_impr"], X_univ["prediction_land"])
  y_pred_univ = X_univ["prediction"].to_numpy()
  X_univ.drop(columns=["prediction_impr", "prediction_land", "prediction", "per_impr_sqft", "per_land_sqft"], inplace=True)
  timing.stop("predict_univ")

  timing.stop("total")

  df = ds.df_universe
  dep_var = ds.dep_var

  if sales_chase:
    y_pred_test = ds.y_test * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_test))
    y_pred_sales = ds.y_sales * np.random.choice([1-sales_chase, 1+sales_chase], len(ds.y_sales))
    y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice([1-sales_chase, 1+sales_chase], len(y_pred_univ))

  name = "local_sqft"

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
    timing
  )

  return results


def _prepredict_lars_sqft(ds: DataSplit, sqft_model: LocalSqftModel, timing: TimingData, verbose: bool = False):

  if verbose:
    print(f"Prepredicting lars sqft...")
  timing.start("train")
  loc_map = sqft_model.loc_map
  location_fields = sqft_model.location_fields
  overall_per_impr_sqft = sqft_model.overall_per_impr_sqft
  overall_per_land_sqft = sqft_model.overall_per_land_sqft

  # intent is to create a primary-keyed dataframe that we can fill with the appropriate local $/sqft value
  # we will merge this in to the main dataframes, then mult. local size by local $/sqft value to predict
  df_land = ds.df_universe[["key"] + location_fields].copy()
  df_impr = ds.df_universe[["key"] + location_fields].copy()

  # start with zero
  df_land["per_land_sqft"] = 0.0  # Initialize as float
  df_impr["per_impr_sqft"] = 0.0  # Initialize as float
  df_land["prediction_land"] = 0.0
  df_impr["prediction"] = 0.0

  # go from most specific to the least specific location (first to last)
  for location_field in location_fields:
    df_sqft_impr, df_sqft_land = loc_map[location_field]

    df_impr = df_impr.merge(df_sqft_impr[[location_field, f"{location_field}_per_impr_sqft"]], on=location_field, how="left")
    df_land = df_land.merge(df_sqft_land[[location_field, f"{location_field}_per_land_sqft"]], on=location_field, how="left")

    df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = df_impr[f"{location_field}_per_impr_sqft"]
    df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = df_land[f"{location_field}_per_land_sqft"]

  # any remaining zeroes get filled with the locality-wide median value
  df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
  df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft

  df_impr = df_impr[["key", "per_impr_sqft"]]
  df_land = df_land[["key", "per_land_sqft"]]

  X_train = ds.X_train

  # merge the df_sqft_land/impr values into the X_sales dataframe:
  X_train["key_sale"] = ds.df_train["key_sale"]
  X_train = X_train.merge(df_land, on="key_sale", how="left")
  X_train = X_train.merge(df_impr, on="key_sale", how="left")
  X_train.loc[X_train["per_impr_sqft"].isna() | X_train["per_impr_sqft"].eq(0), "per_impr_sqft"] = overall_per_impr_sqft
  X_train.loc[X_train["per_land_sqft"].isna() | X_train["per_land_sqft"].eq(0), "per_land_sqft"] = overall_per_land_sqft
  X_train = X_train.drop(columns=["key_sale", "key"])

  X_train_improved = X_train[X_train["bldg_area_finished_sqft"].gt(0)]
  X_train["prediction_land"] = X_train_improved["land_area_sqft"] * X_train_improved["per_land_sqft"]
  X_train["prediction_impr"] = X_train_improved["bldg_area_finished_sqft"] * X_train_improved["per_impr_sqft"]
  X_train["prediction"] = X_train["prediction_land"] + X_train["prediction_impr"]

  timing.stop("train")
  if verbose:
    print(f"Done...")

  return X_train, timing


def run_local_sqft(ds: DataSplit, location_fields: list[str], sales_chase: float = 0.0, verbose: bool = False):
  sqft_model, timing = _run_local_sqft(ds, location_fields, sales_chase, verbose)
  return predict_local_sqft(ds, sqft_model, timing, verbose)


def run_lars(ds: DataSplit, location_fields: list[str], sales_chase: float = 0.0, verbose: bool = False):
  sqft_model, timing = _run_lars_sqft(ds, location_fields, sales_chase, verbose)

  # Pre-predict just the training set to get baseline values
  X_train, timing = _prepredict_lars_sqft(ds, sqft_model, timing, verbose)

  y_train = ds.y_train

  # X_train now has these new fields:
  # - "prediction_impr",
  # - "prediction_land",
  # - "prediction",
  # - "per_impr_sqft",
  # - "per_land_sqft"

  # We do some renaming:
  X_train.rename(columns={
    "per_impr_sqft": "baseline_impr_rate",
    "per_land_sqft": "baseline_land_rate"
  }, inplace=True)

  # Now we organize our land cluster ids and our improvement cluster ids
  land_he_ids = ds.train_land_he_ids
  impr_he_ids = ds.train_impr_he_ids
  unique_land_he_ids = np.unique(land_he_ids)
  unique_impr_he_ids = np.unique(impr_he_ids)

  # We pack them together into one list of parameters
  # Essentially, we're going to derive unique local adjustment values per cluster id
  n_land = len(unique_land_he_ids)
  n_impr = len(unique_impr_he_ids)

  if n_land == 0:
    raise ValueError("No land clusters found.")
  if n_impr == 0:
    raise ValueError("No improvement clusters found.")

  initial_params = np.zeros(n_land + n_impr)

  # Check we have the necessary fields:
  necessary = ["baseline_land_rate", "baseline_impr_rate", "bldg_area_finished_sqft", "land_area_sqft"]
  for field in necessary:
    if field not in X_train:
      raise ValueError(f"Required field \"{field}\" not found in training data.")

  if verbose:
    print("Optimizing L.A.R.S. model")

  def objective_wrapper(params, *args):
    # Increment the counter each time this function is called
    objective_wrapper.calls += 1
    verbose = False
    if objective_wrapper.calls % 100 == 0:
      print(f"Objective function called {objective_wrapper.calls} times")
      verbose = True
    # Call the original objective function
    return _objective_lars(params, *args, verbose=verbose)

  objective_wrapper.calls = 0

  # Optimize the objective function
  result = minimize(
    objective_wrapper,
    initial_params,
    args=(land_he_ids, impr_he_ids, X_train, y_train),
    method='L-BFGS-B'
  )

  if verbose:
    print("Minimum loss achieved:", result.fun)

  # unpack the optimal parameters into land_x and impr_x
  land_x = result.x[:n_land]
  impr_x = result.x[n_land:]

  # create a map of cluster id to optimal local rate for land and impr respectively:
  land_map = {}
  impr_map = {}

  for i, land_id in enumerate(unique_land_he_ids):
    land_map[land_id] = land_x[i]

  for i, impr_id in enumerate(unique_impr_he_ids):
    impr_map[impr_id] = impr_x[i]

  # Now we have a map of land adjustment values per id, and improvement adjustment values per id

  if verbose:
    print("Optimal parameters:")
    print("Land:")
    for land_id in land_map:
      print(f"--> {land_id}: {land_map[land_id]:0.2f}")
    print("")
    print("Improvement:")
    for impr_id in impr_map:
      print(f"--> {impr_id}: {impr_map[impr_id]:0.2f}")

  # Pack it all up and predict
  lars_model = LarsModel(sqft_model, land_map, impr_map)
  return predict_lars(ds, lars_model, timing, verbose)


def predict_lars(ds: DataSplit, lars_model: LarsModel, timing: TimingData, verbose: bool = False):

  # First, predict local sqft the normal way using the LocalSqftModel
  results = predict_local_sqft(ds, lars_model.sqft_model, timing, verbose)

  # Timer was just stopped in the above call, so start it again
  timing.start("total")

  # Now, calculate the L.A.R.S. adjustments for each prediction set

  timing.start("predict_test")
  X_test = ds.df_test[ds.ind_vars + ["impr_he_id", "land_he_id"]].copy()

  #ds.X_test.copy()

  # For each cluster, there is a unique improvement and land adjustment rate, look it up and apply it
  # (If a particular cluster was out of the training sample, we will have no adjustment for it, i.e. it will be 0.0)
  # Once we have the right rate, we multiply it by the appropriate size unit
  X_test["impr_adjustment"] = X_test["impr_he_id"].map(lars_model.impr_adjustments).fillna(0.0) * X_test["bldg_area_finished_sqft"].to_numpy()
  X_test["land_adjustment"] = X_test["land_he_id"].map(lars_model.land_adjustments).fillna(0.0) * X_test["land_area_sqft"].to_numpy()

  # The total local adjustment is the sum of the local land and improvement adjustment
  X_test["total_adjustment"] = X_test["impr_adjustment"] + X_test["land_adjustment"]

  # We take the baseline prediction and add the local adjustment
  y_pred_test = results.pred_test.y_pred + X_test["total_adjustment"]
  timing.stop("predict_test")

  # Do the same thing for everything else:

  timing.start("predict_sales")

  X_sales = ds.df_sales[ds.ind_vars + ["impr_he_id", "land_he_id"]].copy()
  #X_sales = ds.X_sales.copy()

  X_sales["impr_adjustment"] = X_sales["impr_he_id"].map(lars_model.impr_adjustments).fillna(0.0) * X_sales["bldg_area_finished_sqft"].to_numpy()
  X_sales["land_adjustment"] = X_sales["land_he_id"].map(lars_model.land_adjustments).fillna(0.0) * X_sales["land_area_sqft"].to_numpy()
  X_sales["total_adjustment"] = X_sales["impr_adjustment"] + X_sales["land_adjustment"]
  y_pred_sales = results.pred_sales.y_pred + X_sales["total_adjustment"]
  timing.stop("predict_sales")

  timing.start("predict_univ")
  X_univ = ds.df_universe[ds.ind_vars + ["impr_he_id", "land_he_id"]].copy()
  X_univ["impr_adjustment"] = X_univ["impr_he_id"].map(lars_model.impr_adjustments).fillna(0.0) * X_univ["bldg_area_finished_sqft"].to_numpy()
  X_univ["land_adjustment"] = X_univ["land_he_id"].map(lars_model.land_adjustments).fillna(0.0) * X_univ["land_area_sqft"].to_numpy()
  X_univ["total_adjustment"] = X_univ["impr_adjustment"] + X_univ["land_adjustment"]
  y_pred_univ = results.pred_univ + X_univ["total_adjustment"]
  timing.stop("predict_univ")

  timing.stop("total")

  name = "lars"

  results = SingleModelResults(
    ds,
    "prediction",
    "he_id",
    name,
    lars_model,
    y_pred_test,
    y_pred_sales,
    y_pred_univ,
    timing
  )

  return results

# Private functions:


def _run_lars_sqft(ds: DataSplit, location_fields: list[str], sales_chase: float = 0.0, verbose: bool = False)->(LocalSqftModel, TimingData):
  """
  Run a local per-square-foot model that predicts values based on location-specific median $/sqft.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param location_fields: List of location field names to use.
  :type location_fields: list[str]
  :param sales_chase: Factor for simulating sales chasing (default 0.0 means no adjustment).
  :type sales_chase: float, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: LocalSqftModel instance and TimingData object.
  :rtype: (LocalSqftModel, TimingData)
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
  y_train_improved = ds.y_train[X_train["bldg_area_finished_sqft"].gt(0)]

  # filter out improved land where bldg_area_finished_sqft is > zero:
  X_train_vacant = X_train[X_train["bldg_area_finished_sqft"].eq(0)]
  y_train_vacant = ds.y_train[X_train["bldg_area_finished_sqft"].eq(0)]

  # our aim is to construct a dataframe which will contain the local $/sqft values for each individual location value,
  # for multiple location fields. We will then use this to calculate final values for every permutation, and merge
  # that onto our main dataframe to assign $/sqft values from which to generate our final predictions

  loc_map = {}

  # Calculate a very generic prior for the median land allocation
  # median land size x median price/land sqft --> median land value
  # median impr size x median price/impr sqft --> median total value
  median_impr_sqft = X_train_improved["bldg_area_finished_sqft"].median()
  median_land_sqft = X_train_improved["land_area_sqft"].median()
  median_per_impr_sqft = div_field_z_safe(y_train_improved, X_train_improved["bldg_area_finished_sqft"]).median()
  median_per_land_sqft = div_field_z_safe(y_train_vacant, X_train_vacant["land_area_sqft"]).median()
  median_total_value = median_per_impr_sqft * median_impr_sqft
  median_land_value = median_per_land_sqft * median_land_sqft
  median_land_alloc = median_land_value / median_total_value

  if verbose:
    print("Prior median values:")
    print(f"-->impr_sqft: {median_impr_sqft}")
    print(f"-->land_sqft: {median_land_sqft}")
    print(f"-->per_impr_sqft: {median_per_impr_sqft}")
    print(f"-->per_land_sqft: {median_per_land_sqft}")
    print(f"-->total_value: {median_total_value}")
    print(f"-->land_value: {median_land_value}")
    print(f"-->land_alloc: {median_land_alloc}")

  # If it winds up whacky then just pick a conservative value for the baseline
  if median_land_alloc >= 1.0 or median_land_alloc <= 0:
    median_land_alloc = 0.15

  for location_field in location_fields:
    data_sqft_land = {}

    if location_field not in ds.df_train:
      print(f"Location field {location_field} not found in dataset")
      continue

    data_sqft_land[location_field] = []
    data_sqft_land[f"{location_field}_per_land_sqft"] = []

    # for every specific location, calculate the local median $/sqft for vacant property
    for loc in ds.df_train[location_field].unique():
      y_train_loc = ds.y_train[ds.df_train[location_field].eq(loc)]
      X_train_loc = ds.X_train[ds.df_train[location_field].eq(loc)]
      X_train_loc_vacant = X_train_loc[X_train_loc["bldg_area_finished_sqft"].eq(0)]

      if len(X_train_loc_vacant) > 0:
        local_per_land_sqft = (y_train_loc / X_train_loc_vacant["land_area_sqft"]).median()
      else:
        local_per_land_sqft = 0

      # some values will be null so replace them with zeros
      if pd.isna(local_per_land_sqft):
        local_per_land_sqft = 0

      data_sqft_land[location_field].append(loc)
      data_sqft_land[f"{location_field}_per_land_sqft"].append(local_per_land_sqft)

    # create dataframes from the calculated values
    df_sqft_land = pd.DataFrame(data=data_sqft_land)
    loc_map[location_field] = [None, df_sqft_land]

  for location_field in location_fields:
    data_sqft_impr = {}
    if location_field not in ds.df_train:
      print(f"Location field {location_field} not found in dataset")
      continue

    data_sqft_impr[location_field] = []
    data_sqft_impr[f"{location_field}_per_impr_sqft"] = []

    df_sqft_land = loc_map[location_field][1]

    # for every specific location, calculate the local median $/sqft for improved property
    for loc in ds.df_train[location_field].unique():
      y_train_loc = ds.y_train[ds.df_train[location_field].eq(loc)]
      X_train_loc = ds.X_train[ds.df_train[location_field].eq(loc)]

      X_train_loc_improved = X_train_loc[X_train_loc["bldg_area_finished_sqft"].gt(0)]

      if len(X_train_loc_improved) > 0:
        local_per_impr_sqft = (y_train_loc / X_train_loc_improved["bldg_area_finished_sqft"]).median()
      else:
        local_per_impr_sqft = 0

      # some values will be null so replace them with zeros
      if pd.isna(local_per_impr_sqft):
        local_per_impr_sqft = 0

      # Now we try to make the land fit
      local_per_land_sqft = df_sqft_land[df_sqft_land[location_field].eq(loc)][f"{location_field}_per_land_sqft"].values[0]
      X_train_loc["prediction"] = X_train_loc["bldg_area_finished_sqft"] * local_per_impr_sqft

      # We need to make sure the land value is not greater than the total value
      failsafe = 99
      done = False
      land_changed = False
      while (not done) and (failsafe > 0):
        X_train_loc["prediction_land"] = X_train_loc["land_area_sqft"] * local_per_land_sqft
        X_train_loc["prediction_impr"] = X_train_loc["prediction"] - X_train_loc["prediction_land"]

        # if no "prediction_impr" values are zero, we're good:
        if X_train_loc["prediction_impr"].lt(0).any():
          done = True
        else:
          local_per_land_sqft *= 0.9 # reduce by 10% and try again
          land_changed = True

        # update local $/impr sqft rate
        local_per_impr_sqft = div_field_z_safe(X_train_loc["prediction_impr"], X_train_loc["bldg_area_finished_sqft"]).median()

        failsafe -= 1

      if failsafe <= 0:
        warnings.warn(f"Land value failed to converge for '{location_field}' = '{loc}")
        local_per_land_sqft = 0.0
        land_changed = True

      # Store the improved sqft value
      data_sqft_impr[location_field].append(loc)
      data_sqft_impr[f"{location_field}_per_impr_sqft"].append(local_per_impr_sqft)

      if land_changed:
        # Update the land sqft value:
        df_sqft_land.loc[df_sqft_land[location_field].eq(loc), f"{location_field}_per_land_sqft"] = local_per_land_sqft
        if verbose:
          print(f"--> Land changed for '{location_field}' = '{loc}' to {local_per_land_sqft:0.2f}")

    # create dataframes from the calculated values
    df_sqft_impr = pd.DataFrame(data=data_sqft_impr)

    if verbose:
      print("")
      print("df sqft land")
      display(df_sqft_land)
      print("")
      print("df sqft impr")
      display(df_sqft_impr)

    loc_map[location_field][0] = df_sqft_impr
    loc_map[location_field][1] = df_sqft_land

  # guaranteed to "fit" properly
  overall_per_land_sqft = median_per_impr_sqft * median_land_alloc
  overall_per_impr_sqft = median_per_impr_sqft - overall_per_land_sqft

  timing.stop("train")
  if verbose:
    print("Tuning L.A.R.S. Sqft: searching for optimal parameters...")
    print(f"--> optimal improved $/finished sqft (overall) = {overall_per_impr_sqft:0.2f}")
    print(f"--> optimal vacant   $/land     sqft (overall) = {overall_per_land_sqft:0.2f}")

  return LocalSqftModel(loc_map, location_fields, overall_per_impr_sqft, overall_per_land_sqft, sales_chase), timing



def _run_local_sqft(ds: DataSplit, location_fields: list[str], sales_chase: float = 0.0, verbose: bool = False)->(LocalSqftModel, TimingData):
  """
  Run a local per-square-foot model that predicts values based on location-specific median $/sqft.

  :param ds: DataSplit object.
  :type ds: DataSplit
  :param location_fields: List of location field names to use.
  :type location_fields: list[str]
  :param sales_chase: Factor for simulating sales chasing (default 0.0 means no adjustment).
  :type sales_chase: float, optional
  :param verbose: Whether to print verbose output.
  :type verbose: bool, optional
  :returns: LocalSqftModel instance and TimingData object.
  :rtype: (LocalSqftModel, TimingData)
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

    # create dataframes from the calculated values
    df_sqft_impr = pd.DataFrame(data=data_sqft_impr)
    df_sqft_land = pd.DataFrame(data=data_sqft_land)

    loc_map[location_field] = (df_sqft_impr, df_sqft_land)

  # calculate the median overall values
  overall_per_impr_sqft = (ds.y_train / X_train_improved["bldg_area_finished_sqft"]).median()
  overall_per_land_sqft = (ds.y_train / X_train_vacant["land_area_sqft"]).median()

  timing.stop("train")
  if verbose:
    print("Tuning Local Sqft: searching for optimal parameters...")
    print(f"--> optimal improved $/finished sqft (overall) = {overall_per_impr_sqft:0.2f}")
    print(f"--> optimal vacant   $/land     sqft (overall) = {overall_per_land_sqft:0.2f}")

  return LocalSqftModel(loc_map, location_fields, overall_per_impr_sqft, overall_per_land_sqft, sales_chase), timing


def _objective_lars(params, land_he_ids: np.ndarray, impr_he_ids: np.ndarray, X_train: pd.DataFrame, y_train: np.ndarray, verbose=False):

  # Get unique clusters and inverse indices for vectorized mapping
  unique_land, inv_idx_land = np.unique(land_he_ids, return_inverse=True)
  unique_impr, inv_idx_impr = np.unique(impr_he_ids, return_inverse=True)

  n_land = len(unique_land)
  n_impr = len(unique_impr)

  # First n_loc parameters correspond to adjustments to the land rate,
  # and the next n_impr parameters to adjustments to the improvement rate.
  # we unpack these into separate variables
  land_rate_adj = params[:n_land]
  impr_rate_adj = params[n_land:n_land+n_impr]

  # Vectorized adjustment of the unit rates
  adjusted_land_rate = X_train['baseline_land_rate'].to_numpy() + land_rate_adj[inv_idx_land]
  adjusted_impr_rate = X_train['baseline_impr_rate'].to_numpy() + impr_rate_adj[inv_idx_impr]

  # Multiply by the land/impr sizes to get land, improvement, and total value predictions
  adjusted_land_value = adjusted_land_rate * X_train['land_area_sqft'].to_numpy()
  adjusted_impr_value = adjusted_impr_rate * X_train['bldg_area_finished_sqft'].to_numpy()

  total_prediction = adjusted_land_value + adjusted_impr_value

  neg_land_values = adjusted_land_value < 0
  neg_impr_values = adjusted_impr_value < 0

  # Penalize negative predictions
  abs_neg_values = np.sum(np.abs(adjusted_land_value[neg_land_values]) / X_train[neg_land_values]["land_area_sqft"]) + \
  np.sum(np.abs(adjusted_impr_value[neg_impr_values]) / X_train[neg_impr_values]["bldg_area_finished_sqft"])

  # Mean Absolute Error between the total prediction and observed ground truth
  mae = np.mean(np.abs(total_prediction - y_train)) / np.mean(X_train["bldg_area_finished_sqft"])

  # Compute uniformity penalties: variance within each cluster
  land_variances = []
  for land_id in unique_land:
    mask = (land_he_ids == land_id)
    if np.sum(mask) > 1:
      land_variances.append(np.var(adjusted_land_rate[mask]))

  impr_variances = []
  for impr_id in unique_impr:
    mask = (impr_he_ids == impr_id)
    if np.sum(mask) > 1:
      impr_variances.append(np.var(adjusted_impr_rate[mask]))

  # Average variance within clusters (if any clusters have >1 member)
  land_var = np.mean(land_variances) if len(land_variances) > 0 else 0.0
  impr_var = np.mean(impr_variances) if len(impr_variances) > 0 else 0.0

  # Weights to balance the prediction error and uniformity penalties
  alpha = 1.00  # weight for MAE
  beta  = 0.00  # weight for land uniformity penalty
  gamma = 1.00  # weight for improvement uniformity penalty
  delta = 10.00  # weight for negative prediction penalty

  total_loss = alpha * mae + beta * land_var + gamma * impr_var + delta * abs_neg_values

  if verbose:
    print(f"params.sum() = {params.sum()} mae = {mae:,.2f} abs_neg_values = {abs_neg_values:,.2f} land_var = {land_var:,.4f} impr_var = {impr_var:,.2f} total_loss = {total_loss:,.2f}")
  return total_loss


def _sales_chase_univ(df_in, dep_var, y_pred_univ):
  """
  Simulate sales chasing behavior for universe predictions.

  Intended for studying bad behavior; adjusts predictions such that if the observed value is greater than zero,
  it uses the observed value. Should not be used in actual production.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param dep_var: Independent variable column name.
  :type dep_var: str
  :param y_pred_univ: Array of predictions for the universe.
  :type y_pred_univ: numpy.ndarray
  :returns: Adjusted predictions as a NumPy array.
  :rtype: numpy.ndarray
  """
  df_univ = df_in[[dep_var]].copy()
  df_univ["prediction"] = y_pred_univ.copy()
  df_univ.loc[df_univ[dep_var].gt(0), "prediction"] = df_univ[dep_var]
  return df_univ["prediction"].to_numpy()


def _gwr_predict(model, points, P, exog_scale=None, exog_resid=None, fit_params=None):
  """
  Standalone function for GWR predictions for multiple samples.

  :param model: Trained GWR model.
  :type model: GWR
  :param points: Array-like (n*2) of (x, y) coordinates for prediction.
  :type points: array-like
  :param P: Array-like (n*k) of independent variables for prediction.
  :type P: array-like
  :param exog_scale: Optional scale from training; if None, computed from the model.
  :type exog_scale: scalar, optional
  :param exog_resid: Optional residuals from training; if None, computed from the model.
  :type exog_resid: array-like, optional
  :param fit_params: Additional parameters for fitting.
  :type fit_params: dict, optional
  :returns: Dictionary with keys "params" and "y_pred".
  :rtype: dict
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
  """
  Helper function for GWR prediction on a single point.

  :param model: Trained GWR model.
  :type model: GWR
  :param point: Single (x, y) coordinate.
  :type point: array-like
  :param predictors: Predictor vector for the point.
  :type predictors: array-like
  :returns: Tuple of (local betas, predicted value).
  :rtype: tuple(numpy.ndarray, float)
  """
  point = np.asarray(point).reshape(1, -1)
  predictors = np.asarray(predictors)
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


def _run_gwr_prediction(coords, coords_train, X, X_train, gwr_bw, y_train, plot: bool = False, gdf: gpd.GeoDataFrame = None, ind_vars: list[str] = None, intercept: bool = True):
  """
  Run GWR predictions for a set of points.

  Uses the _gwr_predict function to obtain predictions, with optional plotting for diagnostic purposes.

  :param coords: List of coordinate tuples for prediction.
  :type coords: list
  :param coords_train: Training set coordinates.
  :type coords_train: list
  :param X: Predictor matrix for prediction.
  :type X: numpy.ndarray
  :param X_train: Predictor matrix for training.
  :type X_train: numpy.ndarray
  :param gwr_bw: Bandwidth for GWR.
  :type gwr_bw: float
  :param y_train: Training target values.
  :type y_train: numpy.ndarray
  :param plot: If True, produce diagnostic plots.
  :type plot: bool, optional
  :param gdf: Optional GeoDataFrame for plotting.
  :type gdf: geopandas.GeoDataFrame, optional
  :param ind_vars: List of dependent variable names.
  :type ind_vars: list[str], optional
  :param intercept: Whether an intercept is used.
  :type intercept: bool, optional
  :returns: Predicted values as a NumPy array.
  :rtype: numpy.ndarray
  """
  gwr = GWR(coords_train, y_train, X_train, gwr_bw, constant=intercept)
  gwr_results = _gwr_predict(gwr, coords, X)
  y_pred = gwr_results["y_pred"]

  # params = gwr_results["params"]
  # print(f"params shape = {params.shape}")
  #
  # # the shape of params is (n, k), where n is the number of points to predict and k is the number of predictors
  # # we want to visualize each "layer" of the prediction surface individually, so we grab one set of predictions for each predictor
  #
  # x_coords, y_coords = np.array(coords).T
  #
  # print(f"X shape = {X.shape}")
  # print(f"X type = {type(X)}")
  #
  # print(f"ind_vars = {ind_vars}")
  # print(f"params shape = {params.shape}")
  #
  # var = ""
  # if plot:
  #   print(f"gdf exists ? {gdf is not None}")
  #   print(f"gdf cols = {gdf.columns.values}")
  #   for i in range(params.shape[1]):
  #     contributions = params[:, i]
  #
  #     if i == 0:
  #       var = "Intercept"
  #     else:
  #       var = ind_vars[i-1] if ind_vars is not None else f"Variable {i-1}"
  #
  #     plot_value_surface(f"Prediction contribution for {var}", contributions, x_coords, y_coords, gdf)
  #
  #   plot_value_surface("Prediction", y_pred, x_coords, y_coords, gdf)
  #
  #   if ind_vars is not None and "land_area_sqft" in ind_vars:
  #     # get the index of the land area sqft variable
  #     land_size_index = ind_vars.index("land_area_sqft")
  #
  #     print(f"Divide {var} by {ind_vars[land_size_index]}")
  #
  #     # we normalize this by dividing each contribution by the value of its corresponding variable value in X:
  #     #contributions = div_field_z_safe(contributions, X[:, land_size_index])
  #     _y_pred_land_sqft = div_field_z_safe(y_pred, X[:, land_size_index])
  #     plot_value_surface("Prediction / land sqft", _y_pred_land_sqft, x_coords, y_coords, gdf)

  return y_pred


def _get_params(name: str, slug: str, ds: DataSplit, tune_func, outpath: str, save_params: bool, use_saved_params: bool, verbose: bool, **kwargs):
  """
  Obtain model parameters by tuning, with option to save or load saved parameters.

  :param name: Name of the model.
  :type name: str
  :param slug: Slug identifier for file naming.
  :type slug: str
  :param ds: DataSplit object.
  :type ds: DataSplit
  :param tune_func: Function to tune parameters.
  :type tune_func: callable
  :param outpath: Output path for saving parameters.
  :type outpath: str
  :param save_params: Whether to save tuned parameters.
  :type save_params: bool
  :param use_saved_params: Whether to load saved parameters.
  :type use_saved_params: bool
  :param verbose: Whether to print verbose output.
  :type verbose: bool
  :returns: Tuned model parameters as a dictionary.
  :rtype: dict
  """
  if verbose:
    print(f"Tuning {name}: searching for optimal parameters...")

  params = None
  if use_saved_params:
    if os.path.exists(f"{outpath}/{slug}_params.json"):
      params = json.load(open(f"{outpath}/{slug}_params.json", "r"))
      if verbose:
        print(f"--> using saved parameters")
  if params is None:
    params = tune_func(
      ds.X_train,
      ds.y_train,
      sizes=ds.train_sizes,
      he_ids=ds.train_he_ids,
      verbose=verbose,
      cat_vars=ds.categorical_vars,
      **kwargs
    )
    if save_params:
      os.makedirs(outpath, exist_ok=True)
      json.dump(params, open(f"{outpath}/{slug}_params.json", "w"))
  return params


def plot_value_surface(title: str, values: np.array, gdf: gpd.GeoDataFrame, cmap: str = None, norm: str = None):
  """
  Plot a value surface over spatial data.

  Creates a plot of the given values on the geometries in the provided GeoDataFrame using a color map and normalization.

  :param title: Plot title.
  :type title: str
  :param values: Array of values to plot.
  :type values: numpy.ndarray
  :param gdf: GeoDataFrame containing geometries.
  :type gdf: geopandas.GeoDataFrame
  :param cmap: Colormap to use (default "coolwarm" if None).
  :type cmap: str, optional
  :param norm: Normalization method: "two_slope", "log", or None.
  :type norm: str, optional
  :returns: None
  """
  plt.clf()
  plt.figure(figsize=(12, 8))

  plt.title(title)
  vmin = np.quantile(values, 0.05)
  vmax = np.quantile(values, 0.95)

  norm = None

  if norm == "two_slope":
    vmin = min(0, vmin)
    vcenter = max(0, vmin)
    vmax = max(0, vmax)

    if vmax > abs(vmin):
      vmin = -vmax
    if abs(vmin) > vmax:
      vmax = abs(vmin)
    # Define normalization to center zero on white
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
  elif norm == "log":
    # Define normalization to start at zero, center on the median value and cap at 95th percentile
    norm = LogNorm(vmin=vmin, vmax=vmax)
  else:
    # Define normalization to start at zero, center on the median value and cap at 95th percentile
    vmin = min(0, vmin)
    vmax = max(0, vmax)
    # one slope
    norm = Normalize(vmin=vmin, vmax=vmax)

  if cmap is None:
    cmap = "coolwarm"

  gdf_slice = gdf[["geometry"]].copy()
  gdf_slice["values"] = values

  # plot the contributions as polygons using the same color map and vmin/vmax:
  ax = gdf_slice.plot(column="values", cmap=cmap, norm=norm, ax=plt.gca())
  mappable = ax.collections[0]

  cbar = plt.colorbar(mappable, ax=ax)
  cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fancy_format(x)))
  cbar.set_label("Value ($)", fontsize=12)
  plt.show()


def simple_ols(
    df: pd.DataFrame,
    ind_var: str,
    dep_var: str
):
  y = df[dep_var].copy()
  X = df[ind_var].copy()
  X = sm.add_constant(X)
  X = X.astype(np.float64)
  model = sm.OLS(y, X).fit()

  return {
    "slope": model.params[ind_var],
    "intercept": model.params["const"],
    "r2": model.rsquared,
    "adj_r2": model.rsquared_adj
  }