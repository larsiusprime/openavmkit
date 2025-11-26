import json
import os
import pickle
from datetime import date
from numpy.linalg import LinAlgError
import polars as pl
from joblib import Parallel, delayed
from typing import Union, Any, Dict
from pygam import LinearGAM, s, te
from scipy.optimize import curve_fit
from pygam.callbacks import CallBack

from scipy.spatial._ckdtree import cKDTree
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import statsmodels.api as sm
import pandas as pd
import geopandas as gpd
import xgboost as xgb
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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.regression.linear_model import RegressionResults
from xgboost import XGBRegressor

from openavmkit.data import (
    _get_sales,
    _simulate_removed_buildings,
    _enrich_time_field,
    _enrich_sale_age_days,
    SalesUniversePair,
    get_hydrated_sales_from_sup,
    get_sale_field,
    get_train_test_keys,
    filter_df_by_date_range
)
from openavmkit.filters import select_filter
from openavmkit.ratio_study import RatioStudy
from openavmkit.utilities.plotting import plot_scatterplot
from openavmkit.utilities.somers import (
    get_unit_ft,
    get_lot_value_ft,
    get_size_in_somers_units_ft,
    get_size_in_somers_units_m,
)
from openavmkit.utilities.format import fancy_format
from openavmkit.utilities.modeling import (
    GarbageModel,
    AverageModel,
    NaiveSqftModel,
    LocalSqftModel,
    PassThroughModel,
    GWRModel,
    MRAModel,
    GroundTruthModel,
    SpatialLagModel,
    LandSLICEModel
)
from openavmkit.utilities.data import (
    clean_column_names,
    div_series_z_safe,
    calc_spatial_lag,
)
from openavmkit.utilities.settings import get_valuation_date, _get_max_ratio_study_trim, get_look_back_dates
from openavmkit.utilities.stats import (
    quick_median_chd_pl,
    calc_mse_r2_adj_r2,
    calc_prb,
    trim_outliers_mask,
)
from openavmkit.tuning import _tune_lightgbm, _tune_xgboost, _tune_catboost
from openavmkit.utilities.timing import TimingData

pd.set_option("future.no_silent_downcasting", True)

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
    PassThroughModel,
    SpatialLagModel,
    GroundTruthModel,
    GWRModel,
    LandSLICEModel,
    str,
    None,
]


class LandPredictionResults:

    def __init__(
        self,
        land_prediction_field: str,
        impr_prediction_field: str,
        total_prediction_field: str,
        dep_var: str,
        ind_vars: list[str],
        sup: SalesUniversePair,
        max_trim: float
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
            "bldg_area_finished_sqft",
        ]

        use_sales_not_univ = False
        for field in necessary_fields:
            if field not in sup.universe:
                if "sale" not in field:
                    raise ValueError(
                        f"Necessary field '{field}' not found in universe DataFrame."
                    )

        df = get_hydrated_sales_from_sup(sup)

        for field in necessary_fields + [
            "valid_sale",
            "vacant_sale",
            "valid_for_land_ratio_study",
            "valid_for_ratio_study",
        ]:
            if field not in df:
                raise ValueError(
                    f"Necessary field '{field}' not found in sales DataFrame."
                )

        self.land_prediction_field = land_prediction_field
        self.impr_prediction_field = impr_prediction_field
        self.total_prediction_field = total_prediction_field

        df_univ = sup.universe.copy()

        df_univ["land_allocation"] = div_series_z_safe(
            df_univ[land_prediction_field], df_univ[total_prediction_field]
        )
        df_univ["impr_allocation"] = div_series_z_safe(
            df_univ[impr_prediction_field], df_univ[total_prediction_field]
        )

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
            raise ValueError(
                f"Unsupported dep_var '{dep_var}' for land prediction results."
            )

        self.land_ratio_study = RatioStudy(land_predictions, sale_prices, max_trim)
        mse, r2, adj_r2 = calc_mse_r2_adj_r2(
            land_predictions, sale_prices, len(ind_vars)
        )
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
        count_land_invalid = len(
            df_univ[
                df_univ[land_prediction_field].lt(0)
                | df_univ[land_prediction_field].isna()
            ]
        )
        self.perc_land_null = count_land_null / count
        self.perc_land_negative = count_land_negative / count
        self.perc_land_invalid = count_land_invalid / count

        count_impr_null = len(df_univ[df_univ[impr_prediction_field].isna()])
        count_impr_negative = len(df_univ[df_univ[impr_prediction_field].lt(0)])
        count_impr_invalid = len(
            df_univ[
                df_univ[impr_prediction_field].lt(0)
                | df_univ[impr_prediction_field].isna()
            ]
        )
        self.perc_impr_null = count_impr_null / count
        self.perc_impr_negative = count_impr_negative / count
        self.perc_impr_invalid = count_impr_invalid / count

        count_dont_add_up = len(
            df_univ[
                (
                    df_univ[total_prediction_field]
                    - np.abs(
                        df_univ[land_prediction_field] + df_univ[impr_prediction_field]
                    )
                ).gt(1e-6)
            ]
        )
        count_land_overshoot = len(
            df_univ[df_univ[land_prediction_field].gt(df_univ[total_prediction_field])]
        )
        count_vacant_land_not_100 = len(
            df_univ[df_univ["is_vacant"].eq(True) & df_univ["land_allocation"].lt(1.0)]
        )
        self.perc_dont_add_up = count_dont_add_up / count
        self.perc_land_overshoot = count_land_overshoot / count
        self.perc_vacant_land_not_100 = count_vacant_land_not_100 / count

        # Soft rules
        count_improved_land_over_100 = len(
            df_univ[df_univ["is_vacant"].eq(False) & df_univ["land_allocation"].gt(1.0)]
        )
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

    Attributes
    ----------
    dep_var : str
        The independent variable used for prediction.
    ind_vars : list[str]
        List of dependent variables.
    y : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.
    mse : float
        Mean squared error.
    rmse : float
        Root mean squared error.
    mape : float
        Mean absolute percent error.
    r2 : float
        R-squared.
    adj_r2 : float
        Adjusted R-squared.
    ratio_study : RatioStudy
        RatioStudy object.
    df : pd.DataFrame
        DataFrame corresponding to y and y_pred
    """

    def __init__(
        self, 
        dep_var: str, 
        ind_vars: list[str], 
        prediction_field: str, 
        df: pd.DataFrame,
        max_trim: float,
        is_land_predictions: bool = False
    ):
        """
        Initialize a PredictionResults instance.

        Converts the specified prediction column in the DataFrame to a NumPy array,
        computes performance metrics on the subset of data that is valid for ratio study,
        and stores the computed values.

        Parameters
        ----------
        dep_var : str
            The independent variable (e.g., sale price).
        ind_vars : list[str]
            List of dependent variable names.
        prediction_field : str
            Name of the field containing model predictions.
        df : pandas.DataFrame
            DataFrame on which predictions were computed.
        max_trim : float
            The maximum amount of records allowed to be trimmed in a ratio study
        is_land_predictions : bool
            Whether these predictions are for land or not
        """

        self.dep_var = dep_var
        self.ind_vars = ind_vars
        
        y = df[dep_var].to_numpy()
        
        df[dep_var] = pd.to_numeric(df[dep_var], errors="coerce")
        df[prediction_field] = pd.to_numeric(df[prediction_field], errors="coerce")
        
        valid_field = "valid_for_ratio_study"
        if is_land_predictions:
            valid_field = "valid_for_land_ratio_study"
            
        # select only values that are not NaN in either and are valid for ratio study:
        df_clean = df[
            df[valid_field] & 
            ~pd.isna(df[dep_var]) & 
            ~pd.isna(df[prediction_field])
        ]
        self.df = df_clean
        
        y = df_clean[dep_var].to_numpy()
        y_pred = df_clean[prediction_field].to_numpy()
        
        
        self.y = y
        self.y_pred = y_pred
        
        if len(y) > 0 and len(y_pred) > 0:
            self.mse = mean_squared_error(y, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mape = mean_absolute_percentage_error(y, y_pred)
            var_y = np.var(y)

            if var_y == 0:
                self.r2 = float("nan")  # R² undefined when variance is 0
                self.slope = float("nan")
            else:
                df = pd.DataFrame(data={"y": y, "y_pred": y_pred})
                ols_results = simple_ols(df, "y", "y_pred")
                self.r2 = ols_results["r2"]
                self.slope = ols_results["slope"]

            y_ratio = y_pred / y
            mask = trim_outliers_mask(y_ratio, max_trim)

            y_pred_trim = y_pred[mask]
            y_trim = y[mask]

            if len(y_trim) > 0 and len(y_pred_trim) > 0:
                self.mse_trim = mean_squared_error(y_trim, y_pred_trim)
                self.rmse_trim = np.sqrt(self.mse_trim)
                self.mape_trim = mean_absolute_percentage_error(y_trim, y_pred_trim)
                var_y_trim = np.var(y_trim)
                if var_y_trim == 0:
                    self.r2_trim = float("nan")
                    self.slope_trim = float("nan")
                else:
                    df = pd.DataFrame(data={"y": y_trim, "y_pred": y_pred_trim})
                    ols_results = simple_ols(df, "y", "y_pred")
                    self.r2_trim = ols_results["r2"]
                    self.slope_trim = ols_results["slope"]
            else:
                self.mse_trim = float("nan")
                self.rmse_trim = float("nan")
                self.mape_trim = float("nan")
                self.r2_trim = float("nan")
                self.slope_trim = float("nan")

            n_trim = len(y_pred_trim)
            k = len(ind_vars)

            divisor = n_trim - k - 1
            if divisor <= 0 or pd.isna(self.r2_trim):
                self.adj_r2_trim = float("nan")
            else:
                self.adj_r2_trim = 1 - ((1 - self.r2_trim) * (n_trim - 1) / divisor)

        else:
            self.mse = float("nan")
            self.rmse = float("nan")
            self.mape = float("nan")
            self.r2 = float("nan")
            self.adj_r2 = float("nan")
            self.slope = float("nan")
            self.mse_trim = float("nan")
            self.rmse_trim = float("nan")
            self.r2_trim = float("nan")
            self.slope_trim = float("nan")
            self.adj_r2_trim = float("nan")

        n = len(y_pred)
        k = len(ind_vars)
        divisor = n - k - 1
        if divisor <= 0 or pd.isna(self.r2):
            self.adj_r2 = float(
                "nan"
            )  # Adjusted R² undefined with insufficient df or undefined R²
        else:
            self.adj_r2 = 1 - ((1 - self.r2) * (n - 1) / divisor)

        self.ratio_study = RatioStudy(y_pred, y, max_trim)


class DataSplit:
    """
    Encapsulates the splitting of data into training, test, and other subsets.

    Handles all the internals and keeps things organized so you don't have to worry about it.

    Attributes
    ----------
    df_sales : pd.DataFrame
        Sales data after processing.
    df_universe : pd.DataFrame
        Universe (parcel) data after processing.
    df_train : pd.DataFrame
        Training subset of sales data.
    df_test : pd.DataFrame
        Test subset of sales data.
    X_train : pd.DataFrame
        Feature matrix for the training data.
    X_test : pd.DataFrame
        Feature matrix for the test data.
    X_univ : pd.DataFrame
        Feature matrix for the universe data.
    y_train : np.ndarray
        Target array for training.
    y_test : np.ndarray
        Target array for testing.
    ...
        Other attributes storing configuration, validation splits, and settings.
    """

    counter: int = 0

    def __init__(
        self,
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
        hedonic_test_against_vacant_sales: bool = True,
        init: bool = True,
    ):
        """
        Initialize a DataSplit instance by processing and splitting sales and universe data.

        Performs several operations:

        - Saves unmodified copies of original data.
        - Adds missing columns to universe data.
        - Enriches time fields and calculates sale age.
        - Splits sales data into training and test sets.
        - Pre-sorts data for rolling origin cross-validation.
        - Applies interactions if specified.

        Parameters
        ----------
        df_sales : pandas.DataFrame or None
            Sales DataFrame.
        df_universe : pandas.DataFrame or None
            Universe (parcel) DataFrame.
        model_group : str
            Model group identifier.
        settings : dict
            Settings dictionary.
        dep_var : str
            Dependent variable name.
        dep_var_test : str
            Dependent variable name for testing.
        ind_vars : list[str]
            List of independent variable names.
        categorical_vars : list[str]
            List of categorical variable names.
        interactions : dict
            Dictionary defining interactions between variables.
        test_keys : list[str]
            List of keys for the test set.
        train_keys : list[str]
            List of keys for the training set.
        vacant_only : bool, optional
            Whether to consider only vacant sales. Defaults to False.
        hedonic : bool, optional
            Whether to use hedonic adjustments. Defaults to False.
        days_field : str, optional
            Field name for sale age in days. Defaults to "sale_age_days".
        hedonic_test_against_vacant_sales : bool, optional
            Whether to test hedonic models against vacant sales. Defaults to True.
        init : bool, optional
            Whether to perform initialization. Defaults to True.

        Raises
        ------
        ValueError
            If required fields are missing.
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
        set_to_false = [
            "valid_sale",
            "vacant_sale",
            "valid_for_ratio_study",
            "valid_for_land_ratio_study",
        ]
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

        self.df_sales = _get_sales(df_sales, settings, vacant_only).reset_index(
            drop=True
        )

        self._df_sales = self.df_sales.copy()

        self.test_keys = test_keys
        self.train_keys = train_keys

        self.train_sizes = np.zeros_like(train_keys)

        self.train_he_ids = np.zeros_like(train_keys)
        self.train_land_he_ids = np.zeros_like(train_keys)
        self.train_impr_he_ids = np.zeros_like(train_keys)

        self.df_test: pd.DataFrame | None = None
        self.df_train: pd.DataFrame | None = None

        if hedonic:
            # transform df_universe & df_sales such that all improved characteristics are removed
            self.df_universe = _simulate_removed_buildings(self.df_universe, settings)
            self.df_sales = _simulate_removed_buildings(self.df_sales, settings)

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
    
    def is_land_predictions(self)->bool:
        return self.vacant_only or self.hedonic

    def copy(self):
        """
        Return a deep copy of the DataSplit instance.

        Returns
        -------
        DataSplit
            A deep copy of the current DataSplit.
        """
        ds = DataSplit(
            None, None, "", {}, "", "", [], [], {}, [], [], False, False, "", init=False
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

        Returns
        -------
        DataSplit
            The updated DataSplit instance.
        """

        if len(self.categorical_vars) == 0:
            return self

        ds = self.copy()

        for col in ds.categorical_vars:
            ds.df_universe[col] = ds.df_universe[col].astype("category")
            if "UNKNOWN" not in ds.df_universe[col].cat.categories:
                ds.df_universe[col].cat.add_categories(["UNKNOWN"])

            ds.df_sales[col] = ds.df_sales[col].astype("category")
            if "UNKNOWN" not in ds.df_sales[col].cat.categories:
                ds.df_sales[col].cat.add_categories(["UNKNOWN"])

        return ds

    def reconcile_fields_with_foreign(self, foreign_ds):
        """Reconcile this DataSplit's fields with those of a provided reference DataSplit
        (foreign_ds).

        The function performs the following:

          1. One-hot encodes its own categorical columns using its existing encoding method.
          2. Reindexes each DataFrame (train, test, universe, sales)
             so that their columns exactly match the reference DataSplit's train columns.

        Parameters
        ----------
        foreign_ds : DataSplit
            The DataSplit instance whose fields should be matched (e.g., the model's ds).

        Returns
        -------
        DataSplit
            The updated self with reconciled columns.
        """

        # check if foreign is one hot descended by checking if descendents is an empty object
        if (
            foreign_ds.one_hot_descendants is None
            or len(foreign_ds.one_hot_descendants) == 0
        ):
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
        ds_encoded.df_train = reindex_df(ds_encoded.df_train)
        ds_encoded.df_test = reindex_df(ds_encoded.df_test)
        ds_encoded.df_universe = reindex_df(ds_encoded.df_universe)
        ds_encoded.df_sales = reindex_df(ds_encoded.df_sales)

        # Update the independent variables metadata (if applicable)
        ds_encoded.ind_vars = [
            col for col in reference_columns if col in ds_encoded.ind_vars
        ]

        # Optionally, you might also update any other metadata such as one-hot descendants mapping.
        # For example, if you previously built a mapping from original categorical variables to one-hot encoded columns,
        # you can rebuild or adjust it here.

        # Build a mapping of original categorical variables to their one-hot encoded descendant columns.
        ds_encoded.one_hot_descendants = {
            col: [
                descendant
                for descendant in reference_columns
                if descendant.startswith(f"{col}_")
            ]
            for col in ds_encoded.categorical_vars
        }

        return ds_encoded

    def encode_categoricals_with_one_hot(self, exceptions: list[str] = None):
        """
        One-hot encode categorical variables in the DataSplit instance.

        Parameters
        ----------
        exceptions : list[str], optional
            List of categorical variables to exclude from encoding.

        Returns
        -------
        DataSplit
            A new DataSplit instance with one-hot encoded categorical variables.
        """

        # If no categorical variables to encode, return self
        if len(self.categorical_vars) == 0:
            return self

        ds = self.copy()

        # Identify the categorical variables that need encoding.
        # We restrict to those that appear in the independent variables.
        cat_vars = [col for col in ds.ind_vars if col in self.categorical_vars]
        cat_vars = [col for col in cat_vars if col not in (exceptions or [])]

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
                # If the column is of categorical type, ensure "missing" is a known category
                if hasattr(union_df[col].dtype, 'categories'):
                    if "missing" not in union_df[col].cat.categories:
                        current_col_series = union_df[col]
                        try:
                            current_col_series = current_col_series.cat.add_categories(
                                "missing"
                            )
                        except (
                            ValueError
                        ):  # "missing" might already exist due to concurrent modification or previous runs
                            if "missing" not in current_col_series.cat.categories:
                                raise  # Reraise if it genuinely failed to add and was not there
                        union_df[col] = current_col_series

                # Fill NaN with a string placeholder before getting unique categories
                filled_series = union_df[col].fillna("missing")
                filled_series = filled_series.infer_objects(copy=False)
                filled_series = filled_series.astype(str)
                union_categories[col] = sorted(filled_series.unique())
            else:
                # If col is not in union_df, it means it's all NaN or wasn't present.
                # We'll represent its only category as "missing".
                union_categories[col] = ["missing"]

        # Create the OneHotEncoder:
        # - The 'categories' parameter is provided as a list following the order in cat_vars.
        # - handle_unknown="ignore" ensures that any new category seen later is handled gracefully.
        # - drop='first' mimics drop_first=True in pd.get_dummies (avoid dummy-variable trap)
        encoder = OneHotEncoder(
            categories=[union_categories[col] for col in cat_vars],
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )

        # Prepare a DataFrame for fitting the encoder.
        # Ensure all categorical columns appear, even if some are missing from union_df.
        df_for_encoding = pd.DataFrame()
        for col in cat_vars:
            if col in union_df.columns:
                filled_series = union_df[col].fillna("missing")
                filled_series = filled_series.infer_objects(copy=False)
                df_for_encoding[col] = filled_series
            else:
                # If somehow missing, create column filled with our placeholder.
                df_for_encoding[col] = "missing"

        # Ensure all columns in df_for_encoding are of string type if they are categorical,
        # to prevent issues if a column was all NaN and became float before fillna.
        for col in cat_vars:
            if col in df_for_encoding.columns:
                df_for_encoding[col] = df_for_encoding[col].astype(str)

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
                    df_tmp[col] = "missing"  # Use the same placeholder
                else:
                    # If the column is of categorical type, ensure "missing" is a known category
                    if pd.api.types.is_categorical_dtype(df_tmp[col].dtype):
                        if "missing" not in df_tmp[col].cat.categories:
                            # Assign back as add_categories may return a new Series
                            df_tmp[col] = df_tmp[col].cat.add_categories("missing")

                    filled_series = (
                        df_tmp[col].fillna("missing").infer_objects(copy=False)
                    )
                    df_tmp[col] = filled_series
                # Ensure the column is string type before transform
                df_tmp[col] = df_tmp[col].astype(str)

            # Subset to our categorical columns in the expected order.
            df_cats = df_tmp[cat_vars]
            if len(df_cats) > 0:
                # Transform using the fitted OneHotEncoder; result is a NumPy array.
                onehot_arr = encoder.transform(df_cats)
                # Create a DataFrame from the dummy array with proper column names.
                onehot_df = pd.DataFrame(
                    onehot_arr, columns=onehot_feature_names, index=df.index
                )
                # Drop the original categorical columns from the DataFrame.
                df_tmp = df_tmp.drop(columns=cat_vars, errors="ignore")
                # Concatenate the dummy DataFrame onto the non-categorical features.
                df_transformed = pd.concat([df_tmp, onehot_df], axis=1)
            else:
                df_transformed = df_tmp
            return df_transformed

        # Transform every split.
        ds.df_universe = transform_df(ds.df_universe)
        ds.df_sales = transform_df(ds.df_sales)
        ds.df_train = transform_df(ds.df_train)
        ds.df_test = transform_df(ds.df_test)

        # Clean column names.
        ds.df_universe = clean_column_names(ds.df_universe)
        ds.df_sales = clean_column_names(ds.df_sales)
        ds.df_train = clean_column_names(ds.df_train)
        ds.df_test = clean_column_names(ds.df_test)

        # Ensure that all data splits have the same columns and in the same order.
        # We use the training data columns as the reference.
        base_columns = ds.df_train.columns
        ds.df_universe = ds.df_universe.reindex(columns=base_columns, fill_value=0.0)
        ds.df_sales = ds.df_sales.reindex(columns=base_columns, fill_value=0.0)
        ds.df_test = ds.df_test.reindex(columns=base_columns, fill_value=0.0)

        # Here, we update ds.ind_vars to include only the columns present in df_train.
        ds.ind_vars = [
            col
            for col in base_columns
            if col in ds.ind_vars or col in onehot_feature_names
        ]

        # Build a mapping of original categorical variables to their one-hot encoded descendant columns.
        ds.one_hot_descendants = {
            orig: [col for col in onehot_feature_names if col.startswith(f"{orig}_")]
            for orig in cat_vars
        }

        return ds

    def split(self):
        """
        Split the sales DataFrame into training and test sets based on provided keys.

        Uses the `test_keys` and `train_keys` to partition the sales data. Also sorts the splits
        by the specified `days_field`. If the model is hedonic, further filters the sales set
        to vacant records.
        """

        test_keys = self.test_keys

        # separate df into train & test:

        # select the rows that are in the test_keys:
        self.df_test = self.df_sales[
            self.df_sales["key_sale"].astype(str).isin(test_keys)
        ].reset_index(drop=True)
        self.df_train = self.df_sales[
            ~self.df_sales["key_sale"].astype(str).isin(test_keys)
        ].reset_index(drop=True)

        # self.df_train = self.df_sales.drop(self.df_test.index)

        keys_in_df_test = self.df_test["key_sale"].astype(str).unique()
        keys_in_df_train = self.df_train["key_sale"].astype(str).unique()
        keys_in_df_sales = self.df_sales["key_sale"].astype(str).unique()
        # assert that the keys in keys_in_df_test are found in keys_in_df_sales:
        # assert that the union of keys_in_df_test and keys_in_df_train is equal to keys_in_df_sales:
        assert len(set(keys_in_df_test).union(set(keys_in_df_train))) == len(
            set(keys_in_df_sales)
        ), f"Union of keys in df_test and df_train is not equal to keys in df_sales: {set(keys_in_df_test).union(set(keys_in_df_train))} != {set(keys_in_df_sales)}"

        # assert that the keys in keys_in_df_test are not found in keys_in_df_train:
        assert (
            len(set(keys_in_df_test).intersection(set(keys_in_df_train))) == 0
        ), f"Keys in df_test are also found in df_train: {set(keys_in_df_test).intersection(set(keys_in_df_train))}"

        # assert that the keys in keys_in_df_train ARE found in keys_in_df_sales:
        assert (
            len(set(keys_in_df_train).difference(set(keys_in_df_sales))) == 0
        ), f"Keys in df_train are not found in df_sales: {set(keys_in_df_train).difference(set(keys_in_df_sales))}"

        # assert that the keys in keys_in_df_test ARE found in keys_in_df_sales:
        assert (
            len(set(keys_in_df_test).difference(set(keys_in_df_sales))) == 0
        ), f"Keys in df_sales are not found in df_test: {set(keys_in_df_test).difference(set(keys_in_df_sales))}"

        # sort again because sampling shuffles order:
        self.df_test.sort_values(by=self.days_field, ascending=False, inplace=True)
        self.df_train.sort_values(by=self.days_field, ascending=False, inplace=True)

        if self.hedonic and self.hedonic_test_against_vacant_sales:
            # if it's a hedonic model, we're predicting land value, and are thus testing against vacant land only:
            # we have to do this here, AFTER the split, to ensure that the selected rows are from the same subsets

            # get the sales that are actually vacant, from the original set of sales
            _df_sales = _get_sales(self._df_sales, self.settings, True).reset_index(
                drop=True
            )

            # now, select only those records from the modified base sales set that are also in the above set,
            # but use the rows from the modified base sales set
            _df_sales = self.df_sales[
                self.df_sales["key_sale"].isin(_df_sales["key_sale"])
            ].reset_index(drop=True)

            # use these as our sales
            self.df_sales = _df_sales

            # set df_test/train to only those rows that are also in sales:
            # we don't need to use get_sales() because they've already been transformed to vacant
            self.df_test = self.df_test[
                self.df_test["key_sale"].isin(self.df_sales["key_sale"])
            ].reset_index(drop=True)
            self.df_train = self.df_train[
                self.df_train["key_sale"].isin(self.df_sales["key_sale"])
            ].reset_index(drop=True)

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
                        _df_univ[target_field] = (
                            _df_univ[target_field] * _df_univ[fill_field]
                        )
                    if target_field in _df_sales:
                        _df_sales[target_field] = (
                            _df_sales[target_field] * _df_sales[fill_field]
                        )
                    if target_field in _df_train:
                        _df_train[target_field] = (
                            _df_train[target_field] * _df_train[fill_field]
                        )
                    if target_field in _df_test:
                        _df_test[target_field] = (
                            _df_test[target_field] * _df_test[fill_field]
                        )

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
            try:
                if (
                    self.X_train[col].dtype == "Float64"
                    or self.X_train[col].dtype == "Int64"
                    or self.X_train[col].dtype == "boolean"
                    or self.X_train[col].dtype == "bool"
                ):
                    self.X_train = self.X_train.astype({col: "float64"})
            except AttributeError as e:
                raise AttributeError(f"Error converting column '{col}': {e}")

        ind_vars = [col for col in self.ind_vars if col in _df_test.columns]
        self.X_test = _df_test[ind_vars]
        self.y_test = _df_test[self.dep_var_test]


class SingleModelResults:
    """
    Container for results from a single prediction model.

    Attributes
    ----------
    ds : DataSplit
        The data split object used.
    df_universe : pd.DataFrame
        Universe DataFrame.
    df_test : pd.DataFrame
        Test DataFrame.
    df_sales : pd.DataFrame, optional
        Sales DataFrame.
    type : str
        Model type identifier.
    dep_var : str
        Independent variable name.
    ind_vars : list[str]
        Dependent variable names.
    model : PredictionModel
        The model used for prediction.
    pred_test : PredictionResults
        Results for the test set.
    pred_train : PredictionResults
        Results for the training set
    pred_sales : PredictionResults, optional
        Results for the sales set.
    pred_univ : Any
        Predictions for the universe (all parcels in the current scope, such as a model group).
    chd : float
        Calculated CHD value.
    utility_test : float
        Composite utility score for the test set, used for comparing models.
    utility_train : float
        Composite utility score for the training set, used for comparing models.
    is_land_predictions : bool
        Whether these results are land predictions or not.
    timing : TimingData
        Timing data for different phases of the model run.
    """

    def __init__(
        self,
        ds: DataSplit,
        field_prediction: str,
        field_horizontal_equity_id: str,
        type: str,
        model: PredictionModel,
        y_pred_test: np.ndarray,
        y_pred_sales: np.ndarray | None,
        y_pred_univ: np.ndarray,
        timing: TimingData | None = None,
        verbose: bool = False,
        sale_filter: list = None
    ):
        """
        Initialize SingleModelResults by attaching predictions and computing performance metrics.

        Parameters
        ----------
        ds : DataSplit
            DataSplit object containing all necessary splits.
        field_prediction : str
            The field name for predictions.
        field_horizontal_equity_id : str
            The field name for the horizontal equity ID.
        type : str
            Model type identifier.
        model : PredictionModel
            The model used.
        y_pred_test : numpy.ndarray
            Predictions on the test set.
        y_pred_sales : numpy.ndarray or None
            Predictions on the sales set.
        y_pred_univ : numpy.ndarray
            Predictions on the universe set.
        timing : TimingData, optional
            TimingData object.
        verbose : bool, optional
            Whether to print verbose output.
        sale_filter : list, optional
            Filter to apply to sales.
        """

        self.ds = ds
        
        self.is_land_predictions = ds.is_land_predictions()

        max_trim = _get_max_ratio_study_trim(ds.settings, ds.model_group)

        df_univ = ds.df_universe.copy()
        df_sales = ds.df_sales.copy()
        df_test = ds.df_test.copy()

        self.field_prediction = field_prediction
        self.field_horizontal_equity_id = field_horizontal_equity_id

        df_univ[field_prediction] = y_pred_univ
        df_test[field_prediction] = y_pred_test

        if sale_filter is not None:
            sales_before = len(df_sales)
            test_before = len(df_test)
            df_sales = select_filter(df_sales, sale_filter)
            df_test = select_filter(df_test, sale_filter)
            sales_after = len(df_sales)
            test_after = len(df_test)
            if verbose:
                print(f"{sales_after}/{sales_before} sales records passed filter")
                print(f"{test_after}/{test_before} test records passed filter")

        self.verbose = verbose
        self.sale_filter = sale_filter

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

        if timing is None:
            timing = TimingData()
        timing.start("stats_test")
        self.pred_test = PredictionResults(
            self.dep_var_test, self.ind_vars, field_prediction, df_test, max_trim, self.is_land_predictions
        )
        self.df_test = self.pred_test.df.copy()
        timing.stop("stats_test")

        timing.start("stats_sales")

        self.pred_train = None
        self.pred_sales = None
        self.pred_sales_lookback = None

        if y_pred_sales is not None:
            self.pred_sales = PredictionResults(
                self.dep_var_test, self.ind_vars, field_prediction, df_sales, max_trim, self.is_land_predictions
            )
            self.df_sales = self.pred_sales.df.copy()

            # If we have predictions for sales, we also have predictions for the training subset
            df_train = df_sales.copy()
            if sale_filter is not None:
                train_before = len(df_train)
                df_train = select_filter(df_train, sale_filter)
                train_after = len(df_train)
                if verbose:
                    print(
                        f"{train_after}/{train_before} training records passed filter"
                    )

            df_train = df_train[df_train["key_sale"].isin(ds.train_keys)]
            self.pred_train = PredictionResults(
                self.dep_var_test, self.ind_vars, field_prediction, df_train, max_trim, self.is_land_predictions
            )
            self.df_train = self.pred_train.df
            
            # Get prediction results ONLY for the lookback period
            start_date, end_date = get_look_back_dates(ds.settings)
            self.df_sales_lookback = filter_df_by_date_range(df_sales, start_date, end_date)
            
            self.pred_sales_lookback = PredictionResults(
                self.dep_var_test, self.ind_vars, field_prediction, self.df_sales_lookback, max_trim, self.is_land_predictions
            )
            self.df_sales_lookback = self.pred_sales_lookback.df
            

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
        if field_horizontal_equity_id in df_univ_valid:
            self.chd = quick_median_chd_pl(
                pl_df, field_prediction, field_horizontal_equity_id
            )
        else:
            self.chd = float("nan")
        timing.stop("chd")

        timing.start("utility")
        self.utility_test = self.pred_test.mape * 100
        if y_pred_sales is not None:
            self.utility_train = self.pred_train.mape * 100
            self.utility_sales_lookback = self.pred_sales_lookback.mape * 100
        else:
            self.utility_train = float("nan")
            self.utility_sales_lookback = float("nan")
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
                self.pred_sales.y_pred = (
                    self.pred_sales.y_pred * self.ds.df_sales[suffix]
                )
                self.pred_univ = self.pred_univ * self.ds.df_universe[suffix]
            if self.dep_var_test.startswith("log_"):
                self.pred_test.y_pred = self.pred_test.y_pred * self.ds.df_test[suffix]

    def summary(self) -> str:
        """
        Generate a summary string of model performance.

        The summary includes model type, number of rows in test & universe sets, RMSE, R²,
        adjusted R², median ratio, COD, PRD, PRB, and CHD.

        Returns
        -------
        str
            Summary string.
        """

        str = ""
        str += f"Model type: {self.type}\n"
        # Print the # of rows in test & all sales set
        # Print the MSE, RMSE, R2, and Adj R2 for test & all sales set
        str += f"-->Test set, rows: {len(self.pred_test.y)}\n"
        str += f"---->RMSE   : {self.pred_test.rmse:8.0f}\n"
        str += f"---->R2     : {self.pred_test.r2:8.4f}\n"
        str += f"---->Adj R2 : {self.pred_test.adj_r2:8.4f}\n"
        str += f"---->Slope  : {self.pred_test.slope:8.4f}\n"
        str += f"---->M.Ratio: {self.pred_test.ratio_study.median_ratio:8.4f}\n"
        str += f"---->COD    : {self.pred_test.ratio_study.cod:8.4f}\n"
        str += f"---->PRD    : {self.pred_test.ratio_study.prd:8.4f}\n"
        str += f"---->PRB    : {self.pred_test.ratio_study.prb:8.4f}\n"
        str += f"\n"
        str += f"-->All sales set, rows: {len(self.pred_sales.y)}\n"
        str += f"---->RMSE   : {self.pred_sales_lookback.rmse:8.0f}\n"
        str += f"---->R2     : {self.pred_sales_lookback.r2:8.4f}\n"
        str += f"---->Adj R2 : {self.pred_sales_lookback.adj_r2:8.4f}\n"
        str += f"---->Slope  : {self.pred_sales_lookback.slope:8.4f}\n"
        str += f"---->M.Ratio: {self.pred_sales_lookback.ratio_study.median_ratio:8.4f}\n"
        str += f"---->COD    : {self.pred_sales_lookback.ratio_study.cod:8.4f}\n"
        str += f"---->PRD    : {self.pred_sales_lookback.ratio_study.prd:8.4f}\n"
        str += f"---->PRB    : {self.pred_sales_lookback.ratio_study.prb:8.4f}\n"
        str += f"---->CHD    : {self.chd:8.4f}\n"
        str += f"\n"
        return str


def land_utility_score(land_results: LandPredictionResults) -> float:
    """Calculates a "land utility score", based on the following:

    1. Accuracy:

      - Land ratio study median ratio
      - Land ratio study untrimmed COD

    2. Consistency:

      - Land CHD
      - Impr CHD

    3. Sanity:

      - Null and negative predictions
      - Overshoot allocations (> 1.0)
      - Undershoot allocations (vacant land < 1.0)

    Parameters
    ----------
    land_results : LandPredictionResults

    Returns
    -------
    float
        The calculated land utility score

    """
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
    chd_impr_base = (
        30  # we're more tolerant of higher CHD values for improvement than for land
    )
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
    weighted_dist_ratio_score = (
        dist_ratio_score * weight_dist_ratio * ratio_over_penalty
    )

    weighted_chd_land_score = chd_land_score * weight_chd_land
    weighted_chd_impr_score = chd_impr_score * weight_chd_impr
    weighted_chd_score = weighted_chd_land_score + weighted_chd_impr_score

    # sanity
    perc_invalid = (
        (100 * land_results.perc_land_invalid)
        + (100 * land_results.perc_impr_invalid)
        + (100 * land_results.perc_dont_add_up)
    )
    perc_overshoot = 100 * land_results.perc_land_overshoot
    perc_undershoot = 100 * land_results.perc_vacant_land_not_100

    perc_invalid *= weight_invalid
    perc_overshoot *= weight_overshoot
    perc_undershoot *= weight_undershoot

    sanity_score = perc_invalid + perc_overshoot + perc_undershoot
    weighted_sanity_score = sanity_score * weight_sanity

    final_score = (
        weighted_dist_ratio_score
        + weighted_cod_score
        + weighted_chd_score
        + weighted_sanity_score
    )
    return final_score


def model_utility_score(
    model_results: SingleModelResults, test_set: bool = False
) -> float:
    """
    Compute a utility score for a model based on error, median ratio, COD, and CHD.

    Lower scores are better. This function is the weighted average of the following:
    median ratio distance from 1.0, COD, CHD. It also adds a penalty for suspiciously low
    COD values, to punish sales chasing.

    Parameters
    ----------
    model_results : SingleModelResults
        SingleModelResults object.
    test_set : bool, optional
        If True, compute the score using the test set results. Defaults to False.

    Returns
    -------
    float
        Computed utility score.
    """

    weight_dist_ratio = 1000.00
    weight_cod = 1.50
    weight_chd = 1.00
    weight_sales_chase = 7.5

    if test_set:
        pred = model_results.pred_test
    else:
        pred = model_results.pred_train

    cod = pred.ratio_study.cod
    chd = model_results.chd

    # Penalize over-estimates; err on the side of under-estimates
    ratio_over_penalty = 2 if pred.ratio_study.median_ratio < 1.05 else 1

    # calculate base score
    dist_ratio_score = (
        abs(1.0 - pred.ratio_study.median_ratio)
        * weight_dist_ratio
        * ratio_over_penalty
    )
    cod_score = cod * weight_cod
    chd_score = chd * weight_chd

    # penalize very low COD's with bad horizontal equity
    if cod == 0.0:
        cod = 1e-6
    sales_chase_score = ((1.0 / cod) * chd) * weight_sales_chase
    final_score = dist_ratio_score + cod_score + chd_score + sales_chase_score
    return final_score


def safe_predict(callable, X: Any, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Safely obtain predictions from a callable model function.

    Returns an empty array if the input is empty.

    Parameters
    ----------
    callable : callable
        Prediction function.
    X : Any
        Input features.
    params : Dict[str, Any], optional
        Additional parameters for the callable.

    Returns
    -------
    numpy.ndarray
        Predicted values as a NumPy array.
    """

    if len(X) == 0:
        return np.array([])
    if params is None:
        params = {}
    return callable(X, **params)


def predict_mra(
    ds: DataSplit, model: MRAModel, timing: TimingData, verbose: bool = False
) -> SingleModelResults:
    """
    Generate predictions using a Multiple Regression Analysis (MRA) model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    model : MRAModel
        Fitted MRA model instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Container with predictions and associated performance metrics.
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
        verbose=verbose,
    )

    return results


def run_mra(
    ds: DataSplit,
    intercept: bool = True,
    verbose: bool = False,
    model: MRAModel | None = None,
) -> SingleModelResults:
    """
    Train an MRA model and return its prediction results.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    intercept : bool, optional
        Whether to include an intercept in the model. Defaults to True.
    verbose : bool, optional
        Whether to print verbose output. Defaults to False.
    model : MRAModel or None, optional
        Optional pre-trained MRAModel. Defaults to None.

    Returns
    -------
    SingleModelResults
        Prediction results from the MRA model.
    """
    timing = TimingData()

    timing.start("total")

    timing.start("setup")
    ds = ds.encode_categoricals_with_one_hot()
    ds.split()
    
    if intercept:
        ds.X_train = sm.add_constant(ds.X_train, has_constant='add')
        ds.X_test = sm.add_constant(ds.X_test, has_constant='add')
        ds.X_sales = sm.add_constant(ds.X_sales, has_constant='add')
        ds.X_univ = sm.add_constant(ds.X_univ, has_constant='add')

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


def predict_ground_truth(
    ds: DataSplit,
    ground_truth_model: GroundTruthModel,
    timing: TimingData,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions using a ground truth model.

    Uses the observed field (e.g., sale price) as the "prediction" and compares it against
    the ground truth field (e.g., true market value in a synthetic model).

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    ground_truth_model : GroundTruthModel
        GroundTruthModel instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Container with predictions and associated performance metrics.
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
    y_pred_univ = ds.df_universe[
        observed_field
    ].to_numpy()  # ds.X_univ[observed_field].to_numpy()
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
        verbose=verbose,
    )

    return results


def predict_spatial_lag(
    ds: DataSplit, model: SpatialLagModel, timing: TimingData, verbose: bool = False
) -> SingleModelResults:
    """
    Generate predictions using a spatial lag model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    model : SpatialLagModel
        SpatialLagModel instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, prints verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Container with spatial lag predictions and associated performance metrics.
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

        if verbose:
            print(
                f"Spatial lag SQFT model, impr={field_impr_sqft}, land={field_land_sqft}"
            )

        # predict on test set:
        timing.start("predict_test")
        idx_vacant_test = ds.X_test["bldg_area_finished_sqft"].le(0)
        if ds.vacant_only or ds.hedonic:
            y_pred_test = (
                ds.X_test[field_land_sqft].to_numpy()
                * ds.X_test["land_area_sqft"].to_numpy()
            )
        else:
            y_pred_test = (
                ds.X_test[field_impr_sqft].to_numpy()
                * ds.X_test["bldg_area_finished_sqft"].to_numpy()
            )
            y_pred_test[idx_vacant_test] = (
                ds.X_test[field_land_sqft].to_numpy()[idx_vacant_test]
                * ds.X_test["land_area_sqft"].to_numpy()[idx_vacant_test]
            )
        timing.stop("predict_test")

        # predict on the sales set:
        timing.start("predict_sales")
        idx_vacant_sales = ds.X_sales["bldg_area_finished_sqft"].le(0)
        if ds.vacant_only or ds.hedonic:
            y_pred_sales = (
                ds.X_sales[field_land_sqft].to_numpy()
                * ds.X_sales["land_area_sqft"].to_numpy()
            )
        else:
            y_pred_sales = (
                ds.X_sales[field_impr_sqft].to_numpy()
                * ds.X_sales["bldg_area_finished_sqft"].to_numpy()
            )
            y_pred_sales[idx_vacant_sales] = (
                ds.X_sales[field_land_sqft].to_numpy()[idx_vacant_sales]
                * ds.X_sales["land_area_sqft"].to_numpy()[idx_vacant_sales]
            )
        timing.stop("predict_sales")

        # predict on the universe set:
        timing.start("predict_univ")
        idx_vacant_univ = ds.X_univ["bldg_area_finished_sqft"].le(0)

        if ds.vacant_only or ds.hedonic:
            y_pred_univ = (
                ds.X_univ[field_land_sqft].to_numpy()
                * ds.X_univ["land_area_sqft"].to_numpy()
            )
        else:
            y_pred_univ = (
                ds.X_univ[field_impr_sqft].to_numpy()
                * ds.X_univ["bldg_area_finished_sqft"].to_numpy()
            )
            y_pred_univ[idx_vacant_univ] = (
                ds.X_univ[field_land_sqft].to_numpy()[idx_vacant_univ]
                * ds.X_univ["land_area_sqft"].to_numpy()[idx_vacant_univ]
            )
        timing.stop("predict_univ")

    timing.stop("total")

    name = "spatial_lag"
    if model.per_sqft:
        name = "spatial_lag_sqft"

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
        verbose=verbose,
    )

    return results


def predict_pass_through(
    ds: DataSplit, model: PassThroughModel, timing: TimingData, verbose: bool = False
) -> SingleModelResults:
    """
    Generate predictions using an assessor model.

    Uses the specified field from the assessor model (or the first dependent variable if
    hedonic) to extract predictions directly from the input DataFrames.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    model : PassThroughModel
        PassThroughModel instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Container with assessor model predictions and associated performance metrics.
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
        verbose=verbose,
    )

    return results


def run_ground_truth(ds: DataSplit, verbose: bool = False) -> SingleModelResults:
    """
    Run a ground truth model by performing data splitting and returning predictions.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    verbose : bool, optional
        Whether to print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the ground truth model.
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
        observed_field=ds.dep_var, ground_truth_field=ds.ind_vars[0]
    )
    return predict_ground_truth(ds, ground_truth_model, timing, verbose)


def run_spatial_lag(
    ds: DataSplit, per_sqft: bool = False, verbose: bool = False
) -> SingleModelResults:
    """
    Run a spatial lag model by performing data splitting and returning predictions.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    per_sqft : bool, optional
        Whether to normalize the model by square footage. Defaults to False.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the spatial lag model.
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


def run_pass_through(ds: DataSplit, verbose: bool = False) -> SingleModelResults:
    """
    Run an assessor model by performing data splitting and returning predictions.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the assessor model.
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


def predict_kernel(
    ds: DataSplit, kr: KernelReg, timing: TimingData, verbose: bool = False
) -> SingleModelResults:
    """
    Generate predictions using a kernel regression model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    kr : KernelReg
        KernelReg model instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the kernel regression model.
    """

    u_test = ds.df_test["longitude"]
    v_test = ds.df_test["latitude"]

    u_sales = ds.df_sales["longitude"]
    v_sales = ds.df_sales["latitude"]

    u = ds.df_universe["longitude"]
    v = ds.df_universe["latitude"]

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
        except LinAlgError as e:
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
        except LinAlgError as e:
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
        except LinAlgError as e:
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
        verbose=verbose,
    )

    return results


def run_kernel(
    ds: DataSplit,
    outpath: str,
    save_params: bool = False,
    use_saved_params: bool = False,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Run a kernel regression model by tuning its bandwidth and returning predictions.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    outpath : str
        Path to store output parameters.
    save_params : bool, optional
        Whether to save the tuned parameters. Defaults to False.
    use_saved_params : bool, optional
        Whether to load saved parameters. Defaults to False.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the kernel regression model.
    """

    timing = TimingData()

    timing.start("total")

    timing.start("setup")
    ds = ds.encode_categoricals_with_one_hot()
    ds.split()
    u_train = ds.df_train["longitude"]
    v_train = ds.df_train["latitude"]
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
                    print(
                        f"-->saved bandwidth ({len(kernel_bw)} does not match the number of variables ({X_train.shape[1]}), regenerating..."
                    )
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
        kr = KernelReg(
            endog=y_train,
            exog=X_train,
            var_type=var_type,
            bw=kernel_bw,
            defaults=defaults,
        )
        kernel_bw = kr.bw
        if save_params:
            os.makedirs(outpath, exist_ok=True)
            with open(f"{outpath}/kernel_bw.pkl", "wb") as f:
                pickle.dump(kernel_bw, f)
    except LinAlgError as e:
        print(f"--> Error in kernel regression: {e}")
        print("Kernel regression failed. Please check your data.")
        kr = None

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
    intercept: bool = True,
) -> SingleModelResults:
    """
    Generate predictions using a Geographically Weighted Regression (GWR) model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    gwr_model : GWRModel
        GWRModel instance containing training data and parameters.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool
        If True, print verbose output.
    diagnostic : bool, optional
        If True, run in diagnostic mode. Defaults to False.
    intercept : bool, optional
        Whether the model includes an intercept. Defaults to True.

    Returns
    -------
    SingleModelResults
        Prediction results from the GWR model.
    """

    timing.start("train")
    # You have to re-train GWR before each prediction, so we move training to the predict function
    gwr = GWR(
        gwr_model.coords_train, gwr_model.y_train, gwr_model.X_train, gwr_model.gwr_bw
    )
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

    u_test = ds.df_test["longitude"]
    v_test = ds.df_test["latitude"]
    coords_test = list(zip(u_test, v_test))

    u_sales = ds.df_sales["longitude"]
    v_sales = ds.df_sales["latitude"]
    coords_sales = list(zip(u_sales, v_sales))

    u = ds.df_universe["longitude"]
    v = ds.df_universe["latitude"]
    coords_univ = list(zip(u, v))

    np_coords_test = np.array(coords_test)
    timing.start("predict_test")

    if len(np_coords_test) == 0 or len(X_test) == 0:
        y_pred_test = np.array([])
    else:
        gwr_result_test = gwr.predict(np_coords_test, X_test)
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
        intercept=intercept,
    ).flatten()
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ = _run_gwr_prediction(
        coords_univ, coords_train, X_univ, X_train, gwr_bw, y_train, intercept=intercept
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
        timing,
    )
    timing.stop("total")

    return results


def run_gwr(
    ds: DataSplit,
    outpath: str,
    save_params: bool = False,
    use_saved_params: bool = False,
    verbose: bool = False,
    diagnostic: bool = False,
) -> SingleModelResults:
    """
    Run a GWR model by tuning its bandwidth and generating predictions.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    outpath : str
        Output path for saving parameters.
    save_params : bool, optional
        Whether to save tuned parameters. Defaults to False.
    use_saved_params : bool, optional
        Whether to load saved parameters. Defaults to False.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.
    diagnostic : bool, optional
        If True, run in diagnostic mode. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the GWR model.
    """

    timing = TimingData()

    timing.start("total")

    timing.start("setup")
    ds = ds.encode_categoricals_with_one_hot()
    ds.split()
    u_train = ds.df_train["longitude"]
    v_train = ds.df_train["latitude"]
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
                gwr_selector = Sel_BW(
                    coords_train, y_train, X_train, fixed=True, n_jobs=1
                )
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


def predict_xgboost(
    ds: DataSplit,
    xgboost_model: xgb.XGBRegressor,
    timing: TimingData,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions using an XGBoost model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    xgboost_model : xgb.XGBRegressor
        Trained XGBRegressor instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the XGBoost model.
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
        verbose=verbose,
    )
    return results


def run_xgboost(
    ds: DataSplit,
    outpath: str,
    save_params: bool = False,
    use_saved_params: bool = False,
    verbose: bool = False,
    n_trials: int = 50
) -> SingleModelResults:
    """
    Run an XGBoost model by tuning parameters, training, and predicting.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    outpath : str
        Output path for saving parameters.
    save_params : bool, optional
        Whether to save tuned parameters. Defaults to False.
    use_saved_params : bool, optional
        Whether to load saved parameters. Defaults to False.
    n_trials : int, optional
        How many trials do run during parameter search. Defaults to 50.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the XGBoost model.
    """

    timing = TimingData()

    timing.start("total")

    ds = ds.encode_categoricals_with_one_hot()
    ds.split()

    # Fix for object-typed boolean columns (especially 'within_*' fields)
    for col in ds.X_train.columns:
        if col.startswith("within_") or (
            ds.X_train[col].dtype == "object"
            and ds.X_train[col].isin([True, False]).all()
        ):
            if verbose:
                print(f"Converting column {col} from {ds.X_train[col].dtype} to bool")
            ds.X_train[col] = ds.X_train[col].astype(bool)
            if col in ds.X_test.columns:
                ds.X_test[col] = ds.X_test[col].astype(bool)
            if col in ds.X_univ.columns:
                ds.X_univ[col] = ds.X_univ[col].astype(bool)
            if col in ds.X_sales.columns:
                ds.X_sales[col] = ds.X_sales[col].astype(bool)

    parameters = _get_params(
        "XGBoost",
        "xgboost",
        ds,
        _tune_xgboost,
        outpath,
        save_params,
        use_saved_params,
        verbose,
        n_trials=n_trials
    )

    parameters["verbosity"] = 0
    parameters["tree_method"] = "auto"
    parameters["device"] = "cpu"
    parameters["objective"] = "reg:squarederror"
    # parameters["eval_metric"] = "rmse"
    xgboost_model = xgb.XGBRegressor(**parameters)

    timing.start("train")
    xgboost_model.fit(ds.X_train, ds.y_train)
    timing.stop("train")

    return predict_xgboost(ds, xgboost_model, timing, verbose)


def predict_lightgbm(
    ds: DataSplit, gbm: lgb.Booster, timing: TimingData, verbose: bool = False
) -> SingleModelResults:
    """
    Generate predictions using a LightGBM model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    gbm : lgb.Booster
        Trained LightGBM Booster.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the LightGBM model.
    """

    timing.start("predict_test")
    y_pred_test = safe_predict(
        gbm.predict, ds.X_test, {"num_iteration": gbm.best_iteration}
    )
    timing.stop("predict_test")

    timing.start("predict_sales")
    y_pred_sales = safe_predict(
        gbm.predict, ds.X_sales, {"num_iteration": gbm.best_iteration}
    )
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ = safe_predict(
        gbm.predict, ds.X_univ, {"num_iteration": gbm.best_iteration}
    )
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
        verbose=verbose,
    )
    return results


def run_lightgbm(
    ds: DataSplit,
    outpath: str,
    save_params: bool = False,
    use_saved_params: bool = False,
    n_trials: int = 50,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Run a LightGBM model by tuning parameters, training, and predicting.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    outpath : str
        Output path for saving parameters.
    save_params : bool, optional
        Whether to save tuned parameters. Defaults to False.
    use_saved_params : bool, optional
        Whether to load saved parameters. Defaults to False.
    n_trials : int, optional
        How many trials do run during parameter search. Defaults to 50.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the LightGBM model.
    """

    timing = TimingData()

    timing.start("total")

    timing.start("setup")
    ds = ds.encode_categoricals_with_one_hot()
    ds.split()

    # Fix for object-typed boolean columns (especially 'within_*' fields)
    for col in ds.X_train.columns:
        if col.startswith("within_") or (
            ds.X_train[col].dtype == "object"
            and ds.X_train[col].isin([True, False]).all()
        ):
            if verbose:
                print(f"Converting column {col} from {ds.X_train[col].dtype} to bool")
            ds.X_train[col] = ds.X_train[col].astype(bool)
            if col in ds.X_test.columns:
                ds.X_test[col] = ds.X_test[col].astype(bool)
            if col in ds.X_univ.columns:
                ds.X_univ[col] = ds.X_univ[col].astype(bool)
            if col in ds.X_sales.columns:
                ds.X_sales[col] = ds.X_sales[col].astype(bool)

    timing.stop("setup")

    timing.start("parameter_search")
    params = _get_params(
        "LightGBM",
        "lightgbm",
        ds,
        _tune_lightgbm,
        outpath,
        save_params,
        use_saved_params,
        verbose,
        n_trials=n_trials
    )

    # Remove any problematic parameters that might cause errors with forced splits
    problematic_params = [
        "forcedsplits_filename",
        "forced_splits_filename",
        "forced_splits_file",
        "forced_splits",
    ]
    for param in problematic_params:
        if param in params:
            if verbose:
                print(
                    f"Removing problematic parameter '{param}' from LightGBM parameters"
                )
            params.pop(param, None)

    timing.stop("parameter_search")

    timing.start("train")
    cat_vars = [var for var in ds.categorical_vars if var in ds.X_train.columns.values]
    lgb_train = lgb.Dataset(ds.X_train, ds.y_train, categorical_feature=cat_vars)
    lgb_test = lgb.Dataset(
        ds.X_test, ds.y_test, categorical_feature=cat_vars, reference=lgb_train
    )

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
            lgb.log_evaluation(period=0),
        ],
    )
    timing.stop("train")

    # Print timing information for LightGBM model
    # if verbose:
    #   print("\n***** LightGBM Model Timing *****")
    #   print(timing.print())
    #   print("*********************************\n")

    return predict_lightgbm(ds, gbm, timing, verbose)


def predict_catboost(
    ds: DataSplit,
    catboost_model: catboost.CatBoostRegressor,
    timing: TimingData,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions using a CatBoost model.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    catboost_model : catboost.CatBoostRegressor
        Trained CatBoostRegressor instance.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the CatBoost model.
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
        verbose=verbose,
    )

    return results


def run_catboost(
    ds: DataSplit,
    outpath: str,
    save_params: bool = False,
    use_saved_params: bool = False,
    n_trials: int = 50,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Run a CatBoost model by tuning parameters, training, and predicting.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    outpath : str
        Output path for saving parameters.
    save_params : bool, optional
        Whether to save tuned parameters. Defaults to False.
    use_saved_params : bool, optional
        Whether to load saved parameters. Defaults to False.
    n_trials : int, optional
        How many trials do run during parameter search. Defaults to 50.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the CatBoost model.
    """

    timing = TimingData()

    timing.start("total")

    timing.start("setup")
    ds = ds.encode_categoricals_as_categories()
    ds.split()
    timing.stop("setup")

    timing.start("parameter_search")
    params = _get_params(
        "CatBoost",
        "catboost",
        ds,
        _tune_catboost,
        outpath,
        save_params,
        use_saved_params,
        verbose,
        n_trials=n_trials
    )
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


def predict_slice(
    ds: DataSplit,
    slice_model: LandSLICEModel, 
    timing: TimingData,
    verbose: bool = False
) -> SingleModelResults:
    
    timing.start("predict_test")
    if len(ds.y_test) == 0:
        y_pred_test = np.array([])
    else:
        y_pred_test = slice_model.predict(ds.df_test)
    timing.stop("predict_test")

    timing.start("predict_sales")
    if len(ds.y_sales) == 0:
        y_pred_sales = np.array([])
    else:
        y_pred_sales = slice_model.predict(ds.df_sales)
    timing.stop("predict_sales")

    timing.start("predict_univ")
    if len(ds.X_univ) == 0:
        y_pred_univ = np.array([])
    else:
        y_pred_univ = slice_model.predict(ds.df_universe)
    timing.stop("predict_univ")

    timing.stop("total")

    results = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        "catboost",
        slice_model,
        y_pred_test,
        y_pred_sales,
        y_pred_univ,
        timing,
        verbose=verbose,
    )

    return results


def run_slice(
    ds: DataSplit,
    verbose: bool = False
) -> SingleModelResults:
    
    timing = TimingData()
    
    timing.start("total")
    
    timing.start("setup")
    ds = ds.encode_categoricals_with_one_hot()
    ds.split()
    timing.stop("setup")

    timing.start("parameter_search")
    timing.stop("parameter_search")

    timing.start("train")
    
    df_in = ds.df_train[["land_area_sqft",ds.dep_var,"latitude","longitude"]].copy()
    slice_model = fit_land_SLICE_model(
        df_in,
        "land_area_sqft",
        ds.dep_var,
        verbose
    )
    timing.stop("train")

    return predict_slice(ds, slice_model, timing, verbose)


def predict_garbage(
    ds: DataSplit,
    garbage_model: GarbageModel,
    timing: TimingData,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions using a "garbage" model that produces random values.

    If sales_chase is specified, adjusts predictions to simulate sales chasing behavior.

    Needless to say, you should not use this model in production.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    garbage_model : GarbageModel
        Instance containing configuration.
    timing : TimingData
        TimingData object.
    verbose : bool, optional
        Whether to print verbose output.

    Returns
    -------
    SingleModelResults
        Prediction results from the garbage model.
    """

    timing.start("predict_test")
    normal = garbage_model.normal
    min_value = garbage_model.min_value
    max_value = garbage_model.max_value
    sales_chase = garbage_model.sales_chase

    if normal:
        y_pred_test = np.random.normal(
            loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_test)
        )
    else:
        y_pred_test = np.random.uniform(min_value, max_value, len(ds.X_test))
    timing.stop("predict_test")

    timing.start("predict_sales")
    if normal:
        y_pred_sales = np.random.normal(
            loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_sales)
        )
    else:
        y_pred_sales = np.random.uniform(min_value, max_value, len(ds.X_sales))
    timing.stop("predict_sales")

    timing.start("predict_univ")
    if normal:
        y_pred_univ = np.random.normal(
            loc=ds.y_train.mean(), scale=ds.y_train.std(), size=len(ds.X_univ)
        )
    else:
        y_pred_univ = np.random.uniform(min_value, max_value, len(ds.X_univ))
    timing.stop("predict_univ")

    timing.stop("total")

    df = ds.df_universe
    dep_var = ds.dep_var

    if sales_chase:
        y_pred_test = ds.y_test * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_test)
        )
        y_pred_sales = ds.y_sales * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_sales)
        )
        y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(y_pred_univ)
        )

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
    )

    return results


def run_garbage(
    ds: DataSplit,
    normal: bool = False,
    sales_chase: float = 0.0,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Run a garbage model that predicts random values within a range derived from the training set.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    normal : bool, optional
        If True, use a normal distribution; otherwise, use a uniform distribution. Defaults to False.
    sales_chase : float, optional
        Factor for simulating sales chasing (default 0.0 means no adjustment). Defaults to 0.0.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the garbage model.
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
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions by simply using the average (mean or median) of the training set.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    average_model : AverageModel
        AverageModel instance with configuration.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the average model.
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
        y_pred_test = ds.y_test * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_test)
        )
        y_pred_sales = ds.y_sales * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_sales)
        )
        y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(y_pred_univ)
        )

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
    )

    return results


def run_average(
    ds: DataSplit,
    average_type: str = "mean",
    sales_chase: float = 0.0,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Run an average model that predicts either the mean or median of the training set for all predictions.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    average_type : str, optional
        "mean" or "median" indicating which statistic to use. Defaults to "mean".
    sales_chase : float, optional
        Factor for simulating sales chasing (default 0.0 means no adjustment). Defaults to 0.0.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the average model.
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
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions using a naive per-square-foot model.

    Separately computes predictions for improved and vacant properties based on
    `bldg_area_finished_sqft` and `land_area_sqft`, then combines them.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    sqft_model : NaiveSqftModel
        NaiveSqftModel instance containing per-square-foot multipliers.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the naive square-foot model.
    """

    timing.start("predict_test")

    ind_per_built_sqft = sqft_model.dep_per_built_sqft
    ind_per_land_sqft = sqft_model.dep_per_land_sqft
    sales_chase = sqft_model.sales_chase

    X_test = ds.X_test
    X_test_improved = X_test[X_test["bldg_area_finished_sqft"].gt(0)]
    X_test_vacant = X_test[X_test["bldg_area_finished_sqft"].eq(0)]
    X_test["prediction_impr"] = (
        X_test_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
    )
    X_test["prediction_vacant"] = X_test_vacant["land_area_sqft"] * ind_per_land_sqft
    X_test["prediction"] = np.where(
        X_test["bldg_area_finished_sqft"].gt(0),
        X_test["prediction_impr"],
        X_test["prediction_vacant"],
    )
    y_pred_test = X_test["prediction"].to_numpy()
    X_test.drop(
        columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True
    )
    timing.stop("predict_test")

    timing.start("predict_sales")
    X_sales = ds.X_sales
    X_sales_improved = X_sales[X_sales["bldg_area_finished_sqft"].gt(0)]
    X_sales_vacant = X_sales[X_sales["bldg_area_finished_sqft"].eq(0)]
    X_sales["prediction_impr"] = (
        X_sales_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
    )
    X_sales["prediction_vacant"] = X_sales_vacant["land_area_sqft"] * ind_per_land_sqft
    X_sales["prediction"] = np.where(
        X_sales["bldg_area_finished_sqft"].gt(0),
        X_sales["prediction_impr"],
        X_sales["prediction_vacant"],
    )
    y_pred_sales = X_sales["prediction"].to_numpy()
    X_sales.drop(
        columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True
    )
    timing.stop("predict_sales")

    timing.start("predict_univ")
    X_univ = ds.X_univ
    X_univ_improved = X_univ[X_univ["bldg_area_finished_sqft"].gt(0)]
    X_univ_vacant = X_univ[X_univ["bldg_area_finished_sqft"].eq(0)]
    X_univ["prediction_impr"] = (
        X_univ_improved["bldg_area_finished_sqft"] * ind_per_built_sqft
    )
    X_univ["prediction_vacant"] = X_univ_vacant["land_area_sqft"] * ind_per_land_sqft
    X_univ["prediction"] = np.where(
        X_univ["bldg_area_finished_sqft"].gt(0),
        X_univ["prediction_impr"],
        X_univ["prediction_vacant"],
    )
    y_pred_univ = X_univ["prediction"].to_numpy()
    X_univ.drop(
        columns=["prediction_impr", "prediction_vacant", "prediction"], inplace=True
    )
    timing.stop("predict_univ")

    timing.stop("total")

    df = ds.df_universe
    dep_var = ds.dep_var

    if sales_chase:
        y_pred_test = ds.y_test * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_test)
        )
        y_pred_sales = ds.y_sales * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_sales)
        )
        y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(y_pred_univ)
        )

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
    )

    return results


def run_naive_sqft(
    ds: DataSplit,
    sales_chase: float = 0.0,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Run a naive per-square-foot model that predicts based on median $/sqft from the training set.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object.
    sales_chase : float, optional
        Factor for simulating sales chasing (default 0.0 means no adjustment). Defaults to 0.0.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the naive square-foot model.
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

    ind_per_built_sqft = (
        ds.y_train / X_train_improved["bldg_area_finished_sqft"]
    ).median()
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
    verbose: bool = False,
) -> SingleModelResults:
    """
    Generate predictions using a local per-square-foot model that uses location-specific values.

    This function merges location-specific per-square-foot values computed for different
    location fields with the test set, then computes predictions separately for improved
    and vacant properties and combines them.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    sqft_model : LocalSqftModel
        LocalSqftModel instance containing location-specific multipliers.
    timing : TimingData
        TimingData object for recording performance metrics.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the local per-square-foot model.
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
        count_zero_impr = df_impr["per_impr_sqft"].eq(0).sum()
        count_zero_land = df_land["per_land_sqft"].eq(0).sum()

        df_impr = df_impr.merge(
            df_sqft_impr[[location_field, f"{location_field}_per_impr_sqft"]],
            on=location_field,
            how="left",
        )
        df_land = df_land.merge(
            df_sqft_land[[location_field, f"{location_field}_per_land_sqft"]],
            on=location_field,
            how="left",
        )

        df_impr.loc[df_impr["per_impr_sqft"].eq(0), "per_impr_sqft"] = df_impr[
            f"{location_field}_per_impr_sqft"
        ]
        df_land.loc[df_land["per_land_sqft"].eq(0), "per_land_sqft"] = df_land[
            f"{location_field}_per_land_sqft"
        ]

        after_count_zero_impr = df_impr["per_impr_sqft"].eq(0).sum()
        after_count_zero_land = df_land["per_land_sqft"].eq(0).sum()

        if verbose:
            print(
                f"Painting local sqft values for {location_field}, {len(df_sqft_impr[location_field].unique())} location values..."
            )
            delta_impr = count_zero_impr - after_count_zero_impr
            delta_land = count_zero_land - after_count_zero_land
            print(
                f"--> painted {delta_impr} impr values, {after_count_zero_impr} remaining zeroes"
            )
            print(
                f"--> painted {delta_land} land values, {after_count_zero_land} remaining zeroes"
            )

        # do_debug = True
        #
        # if do_debug:
        #   path = "main"
        #   if ds.vacant_only:
        #     path = "vacant"
        #   elif ds.hedonic:
        #     path = "hedonic"
        #
        #   out_path = f"out/models/{ds.model_group}/{path}/local_sqft"
        #   df_sqft_land.to_csv(f"{out_path}/debug_local_sqft_{len(location_fields)}_{location_field}_sqft_land.csv", index=False)
        #   df_land.to_csv(f"{out_path}debug_local_sqft_{len(location_fields)}_{location_field}_land.csv", index=False)
        #   df_sqft_impr.to_csv(f"{out_path}/debug_local_sqft_{len(location_fields)}_{location_field}_sqft_impr.csv", index=False)
        #   df_impr.to_csv(f"{out_path}/debug_local_sqft_{len(location_fields)}_{location_field}_impr.csv", index=False)

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
    X_test.loc[
        X_test["per_impr_sqft"].isna() | X_test["per_impr_sqft"].eq(0), "per_impr_sqft"
    ] = overall_per_impr_sqft
    X_test.loc[
        X_test["per_land_sqft"].isna() | X_test["per_land_sqft"].eq(0), "per_land_sqft"
    ] = overall_per_land_sqft
    X_test = X_test.drop(columns=["key_sale", "key"])

    X_test["prediction_impr"] = (
        X_test["bldg_area_finished_sqft"] * X_test["per_impr_sqft"]
    )
    X_test["prediction_land"] = X_test["land_area_sqft"] * X_test["per_land_sqft"]

    if ds.vacant_only or ds.hedonic:
        X_test["prediction"] = X_test["prediction_land"]
    else:
        X_test["prediction"] = np.where(
            X_test["bldg_area_finished_sqft"].gt(0),
            X_test["prediction_impr"],
            X_test["prediction_land"],
        )

    y_pred_test = X_test["prediction"].to_numpy()
    # TODO: later, don't drop these columns, use them to predict land value everywhere
    X_test.drop(
        columns=[
            "prediction_impr",
            "prediction_land",
            "prediction",
            "per_impr_sqft",
            "per_land_sqft",
        ],
        inplace=True,
    )
    timing.stop("predict_test")

    timing.start("predict_sales")
    X_sales = ds.X_sales

    # merge the df_sqft_land/impr values into the X_sales dataframe:
    X_sales["key_sale"] = ds.df_sales["key_sale"]
    X_sales["key"] = ds.df_sales["key"]
    X_sales = X_sales.merge(df_land, on="key", how="left")
    X_sales = X_sales.merge(df_impr, on="key", how="left")
    X_sales.loc[
        X_sales["per_impr_sqft"].isna() | X_sales["per_impr_sqft"].eq(0),
        "per_impr_sqft",
    ] = overall_per_impr_sqft
    X_sales.loc[
        X_sales["per_land_sqft"].isna() | X_sales["per_land_sqft"].eq(0),
        "per_land_sqft",
    ] = overall_per_land_sqft
    X_sales = X_sales.drop(columns=["key_sale", "key"])

    X_sales["prediction_impr"] = (
        X_sales["bldg_area_finished_sqft"] * X_sales["per_impr_sqft"]
    )
    X_sales["prediction_land"] = X_sales["land_area_sqft"] * X_sales["per_land_sqft"]

    if ds.vacant_only or ds.hedonic:
        X_sales["prediction"] = X_sales["prediction_land"]
    else:
        X_sales["prediction"] = np.where(
            X_sales["bldg_area_finished_sqft"].gt(0),
            X_sales["prediction_impr"],
            X_sales["prediction_land"],
        )

    y_pred_sales = X_sales["prediction"].to_numpy()
    X_sales.drop(
        columns=[
            "prediction_impr",
            "prediction_land",
            "prediction",
            "per_impr_sqft",
            "per_land_sqft",
        ],
        inplace=True,
    )
    timing.stop("predict_sales")

    timing.start("predict_univ")
    X_univ = ds.X_univ

    # merge the df_sqft_land/impr values into the X_univ dataframe:
    X_univ["key"] = ds.df_universe["key"]
    X_univ = X_univ.merge(df_land, on="key", how="left")
    X_univ = X_univ.merge(df_impr, on="key", how="left")
    X_univ.loc[
        X_univ["per_impr_sqft"].isna() | X_univ["per_impr_sqft"].eq(0), "per_impr_sqft"
    ] = overall_per_impr_sqft
    X_univ.loc[
        X_univ["per_land_sqft"].isna() | X_univ["per_land_sqft"].eq(0), "per_land_sqft"
    ] = overall_per_land_sqft
    X_univ["prediction_impr"] = (
        X_univ["bldg_area_finished_sqft"] * X_univ["per_impr_sqft"]
    )
    X_univ["prediction_land"] = X_univ["land_area_sqft"] * X_univ["per_land_sqft"]
    X_univ = X_univ.drop(columns=["key"])

    X_univ.loc[
        X_univ["prediction_impr"].isna() | X_univ["prediction_impr"].eq(0),
        "per_impr_sqft",
    ] = overall_per_impr_sqft
    X_univ.loc[
        X_univ["prediction_land"].isna() | X_univ["prediction_land"].eq(0),
        "per_land_sqft",
    ] = overall_per_land_sqft
    X_univ["prediction_impr"] = (
        X_univ["bldg_area_finished_sqft"] * X_univ["per_impr_sqft"]
    )
    X_univ["prediction_land"] = X_univ["land_area_sqft"] * X_univ["per_land_sqft"]

    if ds.vacant_only or ds.hedonic:
        X_univ["prediction"] = X_univ["prediction_land"]
    else:
        X_univ["prediction"] = np.where(
            X_univ["bldg_area_finished_sqft"].gt(0),
            X_univ["prediction_impr"],
            X_univ["prediction_land"],
        )
    y_pred_univ = X_univ["prediction"].to_numpy()
    X_univ.drop(
        columns=[
            "prediction_impr",
            "prediction_land",
            "prediction",
            "per_impr_sqft",
            "per_land_sqft",
        ],
        inplace=True,
    )
    timing.stop("predict_univ")

    timing.stop("total")

    df = ds.df_universe
    dep_var = ds.dep_var

    if sales_chase:
        y_pred_test = ds.y_test * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_test)
        )
        y_pred_sales = ds.y_sales * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(ds.y_sales)
        )
        y_pred_univ = _sales_chase_univ(df, dep_var, y_pred_univ) * np.random.choice(
            [1 - sales_chase, 1 + sales_chase], len(y_pred_univ)
        )

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
        timing,
    )

    return results


def run_local_sqft(
    ds: DataSplit,
    location_fields: list[str],
    sales_chase: float = 0.0,
    verbose: bool = False,
):
    """
    Run a local per-square-foot model that predicts values based on location-specific median $/sqft.

    Parameters
    ----------
    ds : DataSplit
        DataSplit object containing train/test/universe splits.
    location_fields : list[str]
        List of location field names to use.
    sales_chase : float, optional
        Factor for simulating sales chasing (default 0.0 means no adjustment). Defaults to 0.0.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    SingleModelResults
        Prediction results from the local per-square-foot model.
    """
    sqft_model, timing = _run_local_sqft(ds, location_fields, sales_chase, verbose)
    return predict_local_sqft(ds, sqft_model, timing, verbose)


def _run_local_sqft(
    ds: DataSplit,
    location_fields: list[str],
    sales_chase: float = 0.0,
    verbose: bool = False,
) -> (LocalSqftModel, TimingData):
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

            X_train_loc_improved = X_train_loc[
                X_train_loc["bldg_area_finished_sqft"].gt(0)
            ]
            X_train_loc_vacant = X_train_loc[
                X_train_loc["bldg_area_finished_sqft"].eq(0)
            ]

            if len(X_train_loc_improved) > 0:
                y_train_loc_improved = y_train_loc[
                    X_train_loc["bldg_area_finished_sqft"].gt(0)
                ]
                local_per_impr_sqft = (
                    y_train_loc_improved
                    / X_train_loc_improved["bldg_area_finished_sqft"]
                ).median()
            else:
                local_per_impr_sqft = 0.0

            if len(X_train_loc_vacant) > 0:
                y_train_loc_vacant = y_train_loc[
                    X_train_loc["bldg_area_finished_sqft"].eq(0)
                ]
                local_per_land_sqft = (
                    y_train_loc_vacant / X_train_loc_vacant["land_area_sqft"]
                ).median()
            else:
                local_per_land_sqft = 0.0

            # some values will be null so replace them with zeros
            if pd.isna(local_per_impr_sqft):
                local_per_impr_sqft = 0.0
            if pd.isna(local_per_land_sqft):
                local_per_land_sqft = 0.0

            data_sqft_impr[location_field].append(loc)
            data_sqft_land[location_field].append(loc)

            data_sqft_impr[f"{location_field}_per_impr_sqft"].append(
                local_per_impr_sqft
            )
            data_sqft_land[f"{location_field}_per_land_sqft"].append(
                local_per_land_sqft
            )

        # create dataframes from the calculated values
        df_sqft_impr = pd.DataFrame(data=data_sqft_impr)
        df_sqft_land = pd.DataFrame(data=data_sqft_land)

        loc_map[location_field] = (df_sqft_impr, df_sqft_land)

    # calculate the median overall values
    overall_per_impr_sqft = (
        ds.y_train / X_train_improved["bldg_area_finished_sqft"]
    ).median()
    overall_per_land_sqft = (ds.y_train / X_train_vacant["land_area_sqft"]).median()

    timing.stop("train")
    if verbose:
        print("Tuning Local Sqft: searching for optimal parameters...")
        print(
            f"--> optimal improved $/finished sqft (overall) = {overall_per_impr_sqft:0.2f}"
        )
        print(
            f"--> optimal vacant   $/land     sqft (overall) = {overall_per_land_sqft:0.2f}"
        )

    return (
        LocalSqftModel(
            loc_map,
            location_fields,
            overall_per_impr_sqft,
            overall_per_land_sqft,
            sales_chase,
        ),
        timing,
    )


def _sales_chase_univ(
    df_in: pd.DataFrame, dep_var: str, y_pred_univ: np.ndarray
) -> np.ndarray:
    """
    Simulate sales chasing behavior for universe predictions.

    This function adjusts predictions so that, for each record, if the observed
    value (in `df_in[dep_var]`) is greater than zero, the prediction is replaced
    by the observed value. Intended for studying undesirable “sales chasing” behavior.

    **SHOULD NOT BE USED IN ACTUAL PRODUCTION FOR PREDICTIONS FOR OBVIOUS REASONS.**

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame containing the observed values.
    dep_var : str
        Name of the dependent variable column in `df_in`.
    y_pred_univ : numpy.ndarray
        Array of predictions for the universe.

    Returns
    -------
    numpy.ndarray
        Adjusted predictions as a NumPy array.
    """

    df_univ = df_in[[dep_var]].copy()
    df_univ["prediction"] = y_pred_univ.copy()
    df_univ.loc[df_univ[dep_var].gt(0), "prediction"] = df_univ[dep_var]
    return df_univ["prediction"].to_numpy()


def _gwr_predict(model, points, P, exog_scale=None, exog_resid=None, fit_params=None):
    """Standalone function for GWR predictions for multiple samples."""
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
        raise ValueError(
            "exog_scale and exog_resid must both either be None or specified."
        )

    # Add intercept column to P if the model includes a constant
    if model.constant:
        P = np.hstack([np.ones((len(P), 1)), P])

    # Perform predictions for all points
    results = Parallel(n_jobs=model.n_jobs)(
        delayed(_local_gwr_predict_external)(model, point, predictors)
        for point, predictors in zip(points, P)
    )

    # Extract results
    params = np.array([res[0] for res in results])
    y_pred = np.array([res[1] for res in results])

    return {"params": params, "y_pred": y_pred}


def _local_gwr_predict_external(model, point, predictors):
    """Helper function for GWR prediction on a single point."""
    point = np.asarray(point).reshape(1, -1)
    predictors = np.asarray(predictors)
    weights = Kernel(
        0,
        model.coords,
        model.bw,
        fixed=model.fixed,
        function=model.kernel,
        spherical=model.spherical,
        points=point,  # Here we pass our prediction point
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
    intercept: bool = True,
):
    """Run GWR predictions for a set of points."""
    gwr = GWR(coords_train, y_train, X_train, gwr_bw, constant=intercept)
    gwr_results = _gwr_predict(gwr, coords, X)
    y_pred = gwr_results["y_pred"]
    return y_pred


def _get_params(
    name: str,
    slug: str,
    ds: DataSplit,
    tune_func,
    outpath: str,
    save_params: bool,
    use_saved_params: bool,
    verbose: bool,
    **kwargs,
):
    """Obtain model parameters by tuning, with option to save or load saved parameters."""
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
            **kwargs,
        )
        if save_params:
            os.makedirs(outpath, exist_ok=True)
            json.dump(params, open(f"{outpath}/{slug}_params.json", "w"))
    return params


def plot_value_surface(
    title: str,
    values: np.ndarray,
    gdf: gpd.GeoDataFrame,
    cmap: str = None,
    norm: str = None,
) -> None:
    """
    Plot a value surface over spatial data.

    Creates a plot of the given values on the geometries in the provided GeoDataFrame
    using a color map and normalization.

    Parameters
    ----------
    title : str
        Plot title.
    values : numpy.ndarray
        Array of values to plot.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries.
    cmap : str, optional
        Colormap to use (default is "coolwarm" if None).
    norm : str, optional
        Normalization method: "two_slope", "log", or None.
    """

    # TODO: Why is this in modeling and not somewhere related to plotting?

    plt.clf()
    plt.figure(figsize=(12, 8))

    plt.title(title)
    vmin = np.quantile(values, 0.05)
    vmax = np.quantile(values, 0.95)

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


def simple_mra(df: pd.DataFrame, ind_vars: list[str], dep_var: str):
    """Run a simple multiple regression on the provided data, using multiple predictors

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to run the regression on

    ind_vars : list[str]
        List of independent variables (predictors)

    dep_var : str
        Dependent variable (what you are trying to predict)

    Returns
    -------
    dict
        Dictionary containing the following values:

          - "coefs" (dictionary of coefficients keyed by the variable name)
          - "intercept"
          - "r2"
          - "adj_r2"
          - "pval"
          - "mse"
          - "rmse"
          - "std_err"
    """
    y = df[dep_var].copy()
    X = df[ind_vars].copy()
    X = sm.add_constant(X, has_constant='add')
    X = X.astype(np.float64)
    model = sm.OLS(y, X).fit()

    return {
        "coefs": {ind_var: model.params[ind_var] for ind_var in ind_vars},
        "intercept": model.params["const"],
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "pval": model.pvalues[ind_vars],
        "mse": model.mse_resid,
        "rmse": np.sqrt(model.mse_resid),
        "std_err": model.bse[ind_vars],
    }


def simple_ols(df: pd.DataFrame, ind_var: str, dep_var: str, intercept: bool = True):
    """Run a simple ordinary-least-squares regression on the provided data, using a single predictor

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to run the regression on

    ind_var : str
        Independent variable (predictor)

    dep_var : str
        Dependent variable (what you are trying to predict)

    Returns
    -------
    dict
        Dictionary containing the following values:

          - "slope"
          - "intercept"
          - "r2"
          - "adj_r2"
          - "pval"
          - "mse"
          - "rmse"
          - "std_err"

    """

    y = df[dep_var].copy()
    X = df[ind_var].copy()
    if intercept:
        X = sm.add_constant(X, has_constant='add')
    X = X.astype(np.float64)
    model = sm.OLS(y, X).fit()

    return {
        "slope": model.params[ind_var],
        "intercept": model.params.get("const", 0.0),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "pval": model.pvalues[ind_var],
        "mse": model.mse_resid,
        "rmse": np.sqrt(model.mse_resid),
        "std_err": model.bse[ind_var],
    }


def _greedy_nn_limited(
    lat: float, lon: float, start_idx: int = 0, k: int = 16
) -> np.ndarray:
    """Greedy nearest-neighbor on flat coords with limited-k search.

    - Projects lat/lon to (x,y) in meters via equirectangular.
    - At each step, queries up to k nearest; if all are visited, tries more,
      but never requests more than n-1, and if still stuck picks the first
      unvisited.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    start_idx : int
        Starting index (defaults to 0)
    k : int
        How many neighbors (defaults to 16)

    Returns
    -------
    np.ndarray
        Ordered list of indices corresponding to neighbors

    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    n = len(lat)

    # 1) project to x,y in meters
    mean_lat = np.deg2rad(lat.mean())
    m_per_deg = 111_320.0
    xs = (lon - lon.mean()) * m_per_deg * np.cos(mean_lat)
    ys = (lat - lat.mean()) * m_per_deg
    pts = np.vstack((xs, ys)).T

    # 2) build tree once
    tree = cKDTree(pts)

    visited = np.zeros(n, bool)
    order = np.empty(n, int)
    current = start_idx

    for i in range(n):
        order[i] = current
        visited[current] = True

        # if this was the last point, break
        if i == n - 1:
            break

        kk = k
        next_pt = None
        while True:
            # never ask for more than n-1 neighbors (excluding self)
            kk = min(kk, n - 1)
            dists, idxs = tree.query(pts[current], kk + 1)  # +1 to skip self
            # scan for the first unvisited
            for cand in idxs[1:]:  # skip idxs[0] == current
                if cand < n and not visited[cand]:
                    next_pt = cand
                    break
            if next_pt is not None:
                break
            if kk >= n - 1:
                # all other points must be visited? or we're at the very end.
                # fallback: pick the first unvisited by simple search.
                unvis = np.nonzero(~visited)[0]
                next_pt = unvis[0]
                break
            # otherwise, try a bigger neighborhood
            kk *= 2

        current = next_pt

    return order


def _choose_m(n_obs: int) -> int:
    if n_obs < 3_000:
        return 1
    elif n_obs < 30_000:
        return 2
    elif n_obs < 300_000:
        return 3
    else:
        return 4


def _yatchew_estimate(m, y, Z, Xs, robust=True):
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Xs.shape[1] != 2:
        raise ValueError("smooth_vars must have exactly two columns (lat, lon)")

    lat, lon = Xs.T
    order = _greedy_nn_limited(lat, lon, k=16)
    y, Z = y[order], Z[order]

    for _ in range(m):
        y = y[m:] - y[:-m]
        Z = Z[m:] - Z[:-m]

    res = sm.OLS(y, Z).fit()
    return res.get_robustcov_results("HC1") if robust else res


def _kolbe_et_al_transform(
    df_in: pd.DataFrame,
    sale_field: str,
    bldg_fields: list[str],
    units: str = "ft",
    log: bool = True,
    drop_zeros: bool = True,  # ← new flag
):
    df = df_in.copy()  # keep original untouched

    # -- 1. Normalize by size units ----------------------------------------------------
    if units == "ft":
        df["SIZE"] = df["land_area_sqft"]
    else:
        df["SIZE"] = df["land_area_m2"]

    # -- 2. Raw ratios ------------------------------------------------------
    if sale_field in df:
        df["_price_per_SIZE"] = df[sale_field] / df["SIZE"]

    for col in bldg_fields:
        df[f"_{col}_per_SIZE"] = df[col] / df["SIZE"]

    # -- 3. Optionally drop rows that would break the log -------------------
    if log and drop_zeros:
        if sale_field in df:
            keep = df["_price_per_SIZE"] > 0
        else:
            keep = pd.Series(True, index=df.index)
        for col in bldg_fields:
            keep &= df[f"_{col}_per_SIZE"] > 0
        df = df.loc[keep].reset_index(drop=True)

    # -- 4. Build y and Z ---------------------------------------------------
    if log:
        if sale_field in df:
            y = np.log(df["_price_per_SIZE"].to_numpy(float))
        else:
            y = np.zeros(df.shape[0])
        Z = np.log(df[[f"_{c}_per_SIZE" for c in bldg_fields]].to_numpy(float))
    else:
        if sale_field in df:
            y = df["_price_per_SIZE"].to_numpy(float)
        else:
            y = np.zeros(df.shape[0])
        Z = df[[f"_{c}_per_SIZE" for c in bldg_fields]].to_numpy(float)

    # drop the helper columns
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

    return y, Z, df


def _kolbe_yatchew(
    df_train_in: pd.DataFrame,
    df_test_in: pd.DataFrame,
    df_univ_in: pd.DataFrame,
    bldg_fields: list[str],
    settings: dict,
    units: str = "ft",
    log: bool = False,
    robust: bool = True,
    verbose: bool = False,
):
    sale_field = get_sale_field(settings)

    df = df_train_in.copy()
    df_univ = df_univ_in.copy()

    # 1. Transform variables according to Kolbe, et al.
    y, Z, df = _kolbe_et_al_transform(
        df, sale_field, bldg_fields, units=units, log=log
    )
    y_test, Z_test, df_test = _kolbe_et_al_transform(
        df_test_in, sale_field, bldg_fields, units=units, log=log
    )
    y_univ, Z_univ, df_univ = _kolbe_et_al_transform(
        df_univ, sale_field, bldg_fields, units=units, log=log
    )

    # 2. Run Yatchew
    m = _choose_m(len(df))
    Xs = df[["latitude", "longitude"]].to_numpy(float)

    res = _yatchew_estimate(m, y, Z, Xs, robust=robust)

    def kys_predict_impr(_df, _Z, _res, _log: bool):
        # ------- 1. improvement prediction (β̂′Z) ------------------------------
        _df["impr_pred"] = _Z @ _res.params
        if _log:
            _df["impr_value_per_SIZE"] = np.exp(_df["impr_pred"])
        else:
            _df["impr_value_per_SIZE"] = _df["impr_pred"]
        _df["impr_value"] = _df["impr_value_per_SIZE"] * _df["SIZE"]
        return _df

    def kys_predict_land(_df, _y, _Z, _res, _log: bool):
        # ------- 2. land residual ----------------------------------------
        _df["land_resid"] = _y - (_Z @ _res.params)

        # ------- 3. Improvement + land in $/SU -------------------------------
        if _log:
            _df["land_value_per_SIZE"] = np.exp(_df["land_resid"])
        else:
            _df["land_value_per_SIZE"] = _df["land_resid"]

        # ------- 4. Dollar values per parcel ---------------------------------
        _df["land_value"] = _df["land_value_per_SIZE"] * _df["SIZE"]
        return _df

    def kys_predict_market_value(_df, _log: bool):
        # ------- 5. Total market value ---------------------------------------
        if _log:
            _df["market_value_per_SIZE"] = np.exp(_df["impr_pred"] + _df["land_resid"])
        else:
            _df["market_value_per_SIZE"] = _df["impr_pred"] + _df["land_resid"]

        _df["market_value"] = _df["market_value_per_SIZE"] * _df["SIZE"]

        # Optional: sanity check (should be ~0 except for FP round-off)
        _df["market_value_check"] = _df["market_value"] - (
            _df["impr_value"] + _df["land_value"]
        )

        return _df

    # 3.  Add structure and location components ------------------------------
    df = kys_predict_impr(df, Z, res, log)
    df = kys_predict_land(df, y, Z, res, log)
    df = kys_predict_market_value(df, log)

    mse, r2, adj_r2 = calc_mse_r2_adj_r2(
        df["market_value"].to_numpy(), df[sale_field].to_numpy(), len(bldg_fields)
    )
    slope, _ = np.polyfit(df["market_value"], df[sale_field], 1)

    # plot "market_value" vs. sale_field:
    plot_scatterplot(
        df,
        "market_value",
        sale_field,
        title="Kolbe Yatchew",
        xlabel="Predicted Value ($)",
        ylabel="Observed Value ($)",
        best_fit_line=True,
        perfect_fit_line=True,
    )

    if verbose:
        print(f"Kolbe Yatchew: {units} units")
        print(f"  MSE    = {mse:.2f}")
        print(f"  R2     = {r2:.4f}")
        print(f"  Adj R2 = {adj_r2:.4f}")
        print(f"  Slope  = {slope:.4f}")

    # Predict on the test set

    # Get spatial lag of land value per SU
    df_univ = calc_spatial_lag(df, df_univ, ["land_value_per_SIZE"])
    df_univ = df_univ.rename(
        columns={"spatial_lag_land_value_per_SIZE": "land_value_per_SIZE"}
    )
    df_univ["land_value"] = df_univ["land_value_per_SIZE"] * df_univ["SIZE"]

    suffix = "_area"

    df_univ = df_univ.merge(df_univ_in[["key", "model_group"]], on="key", how="left")

    df_univ.to_parquet(f"out/kolbe_yatchew{suffix}.parquet")

    # Merge this onto the test set
    df_test = df_test.merge(
        df_univ[["key", "land_value_per_SIZE"]], on="key", how="left"
    )

    df_test = kys_predict_impr(df_test, Z_test, res, log)
    df_test["land_value"] = df_test["land_value_per_SIZE"] * df_test["SIZE"]
    df_test["market_value"] = df_test["impr_value"] + df_test["land_value"]

    mse, r2, adj_r2 = calc_mse_r2_adj_r2(
        df_test["market_value"].to_numpy(),
        df_test[sale_field].to_numpy(),
        len(bldg_fields),
    )
    try:
        slope, _ = np.polyfit(df_test["market_value"], df_test[sale_field], 1)
    except LinAlgError as e:
        print(f"LinAlgError in np.polyfit: {e}")
        slope = np.nan

    # plot "market_value" vs. sale_field:
    plot_scatterplot(
        df_test,
        "market_value",
        sale_field,
        title="Kolbe Yatchew (Test Set)",
        xlabel="Predicted Value ($)",
        ylabel="Observed Value ($)",
        best_fit_line=True,
        perfect_fit_line=True,
    )

    if verbose:
        print(f"Kolbe Yatchew (Test Set): {units} units")
        print(f"  MSE    = {mse:.2f}")
        print(f"  R2     = {r2:.4f}")
        print(f"  Adj R2 = {adj_r2:.4f}")
        print(f"  Slope  = {slope:.4f}")

    return res, df


def _calc_spatial_lag(
    df_sample: pd.DataFrame,
    df_univ: pd.DataFrame,
    value_fields: list[str],
    neighbors: int = 5,
    exclude_self_in_sample: bool = False,
) -> pd.DataFrame:

    df = df_univ.copy()

    # Build a cKDTree from df_sales coordinates

    # we TRAIN on these coordinates -- coordinates that are NOT in the test set
    coords_train = df_sample[["latitude", "longitude"]].values
    tree = cKDTree(coords_train)

    # we PREDICT on these coordinates -- all the coordinates in the universe
    coords_all = df[["latitude", "longitude"]].values

    for value_field in value_fields:
        print(f"Value field = {value_field}")
        if value_field not in df_sample:
            print("Value field not in df_sample, skipping")
            continue

        # Choose the number of nearest neighbors to use
        k = neighbors  # You can adjust this number as needed

        # Query the tree: for each parcel in df_universe, find the k nearest parcels
        # distances: shape (n_universe, k); indices: corresponding indices in df_sales
        distances, indices = tree.query(coords_all, k=k)

        if exclude_self_in_sample:
            distances = distances[:, 1:]  # Exclude self-distance
            indices = indices[:, 1:]  # Exclude self-index

        # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
        if k < 2:
            raise ValueError("k must be at least 2 to compute spatial lag.")

        # For each universe parcel, compute sigma as the mean distance to its k neighbors.
        sigma = distances.mean(axis=1, keepdims=True)

        # Handle zeros in sigma
        sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

        # Compute Gaussian kernel weights for all neighbors
        weights = np.exp(-(distances**2) / (2 * sigma**2))

        # Normalize the weights so that they sum to 1 for each parcel
        weights_norm = weights / weights.sum(axis=1, keepdims=True)

        # Get the values corresponding to the neighbor indices
        parcel_values = df_sample[value_field].values
        neighbor_values = parcel_values[indices]  # shape (n_universe, k)

        # Compute the weighted average (spatial lag) for each parcel in the universe
        spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_values)).sum(
            axis=1
        )

        # Add the spatial lag as a new column
        df[f"spatial_lag_{value_field}"] = spatial_lag

        median_value = df_sample[value_field].median()
        df[f"spatial_lag_{value_field}"] = df[f"spatial_lag_{value_field}"].fillna(
            median_value
        )

    return df


def _derive_land_values(
    sup: SalesUniversePair,
    bldg_fields: list[str],
    model_group: str,
    settings: dict,
    log: bool = True,
    verbose: bool = False,
):
    # TODO: Experimental

    df_sales = get_hydrated_sales_from_sup(sup)

    # Filter only to our current model group
    df_sales = df_sales[df_sales["model_group"].eq(model_group)].copy()

    # Filter out outliers in terms of price:
    sale_field = get_sale_field(settings)
    df_sales = df_sales[
        df_sales[sale_field].gt(df_sales[sale_field].quantile(0.05))
        & df_sales[sale_field].lt(df_sales[sale_field].quantile(0.95))
    ]

    # Filter out outliers in terms of size:
    df_sales = df_sales[
        df_sales["land_area_sqft"].ge(df_sales["land_area_sqft"].quantile(0.05))
        & df_sales["land_area_sqft"].le(df_sales["land_area_sqft"].quantile(0.95))
    ]

    train_keys, test_keys = get_train_test_keys(df_sales, settings)
    df_train = df_sales[df_sales["key_sale"].isin(test_keys)]
    df_test = df_sales[df_sales["key_sale"].isin(train_keys)]

    results, df = _kolbe_yatchew(
        df_train,
        df_test,
        sup.universe,
        bldg_fields,
        settings,
        log=log,
        verbose=verbose,
    )

    if verbose:
        print(results.summary())

    return results, df


def fit_land_SLICE_model(
    df_in : pd.DataFrame,
    size_field: str = "land_area_sqft",
    value_field: str = "land_value",
    verbose: bool = False
)->LandSLICEModel:
    """
    Fits land values using SLICE: "Smooth Location with Increasing-Concavity Equation"
    
    This model takes already-existing raw per-parcel land values and separates the contribution of land size and locational premium.
    It also enforces three constraints: 
    1. Locational premium must change smoothly over space
    2. Land value in any fixed location must increase monotonically with land size
    3. The marginal value of each additional unit of land size must decrease monotonically
    
    The output is an object that encodes the final fitted land values, the locational premiums, and the local land factors. Fitted land
    values are derived by simply multiplying locational premium times local land factor.
    
    Parameters
    ----------
    df_in : pd.DataFrame
        Input data
    size_field : str
        The name of your land size field
    value_field : str
        The name of your land value field
    verbose : bool
        Whether to print verbose output
    """
    
    
    class Progress(CallBack):
        def on_loop_end(self, diff):
            # self.iter is automatically tracked inside Callback
            print(f"iter {self.iter:>3d}   dev.change={diff:9.3e}")
    
    if verbose:
        print("Fitting land SLICE model...")


    df = df_in[[value_field, size_field, "latitude", "longitude"]].copy()
    med_land_size = float(np.median(df[size_field]))

    # Y = Size-detrended location factor
    df["Y"] = div_series_z_safe(
        df[value_field],
        np.sqrt(
            df[size_field] / med_land_size
        )
    )

    if verbose:
        print("-->fitting thin-plate spline for location factor...")
        
    # Fit a thin-plate spline for location factor L(lat, lon)
    basis = te(0, 1, n_splines=40, spline_order=3)
    gam_L : LinearGAM = LinearGAM(
        basis,
        max_iter=40,
        callbacks=[Progress()],
        verbose=verbose
    )
    gam_L.fit(
        df[['latitude', 'longitude']].values,
        np.log(df['Y']).values
    )

    if verbose:
        print("-->estimating initial location factor...")
    # L_hat = Initial estimated location factor (mostly depends on latitude/longitude)
    df['L_hat'] = np.exp(gam_L.predict(df[['latitude', 'longitude']].values))

    # Z = Location-detrended land values (mostly depends on size)
    df["Z"] = df[value_field] / df["L_hat"]

    # Define a power law curve function
    def power_curve(s, alpha, beta):
        return alpha * (s / med_land_size)**beta

    # Solve for location-detrended-land-value and observed size to fit the power law curve
    # - with bounds: alpha>0 (always positive), 0<beta<1 (monotonic-up & concave)
    # - this enforces that land increases in value with size, but with diminishing returns to marginal size
    if verbose:
        print("-->fitting power law curve for size factor...")
    popt, _ = curve_fit(
        f=power_curve,
        p0=[np.median(df["Z"]),0.5],
        xdata=df[size_field].values,
        ydata=df["Z"].values,
        bounds=([0, 1e-6], [np.inf, 0.999])
    )

    # Coefficients for the power law curve:
    alpha_hat, beta_hat = popt

    # Function to call the power law curve with memorized coefficients and a given size
    def F_hat(s):
        return power_curve(np.asarray(s), alpha_hat, beta_hat)

    if verbose:
        print("-->tightening up values with one more iteration...")

    # Tighten up our values with an extra iteration
    df["Y2"] = df[value_field] / F_hat(df[size_field])
    gam_L2 : LinearGAM = gam_L.fit(df[["latitude", "longitude"]], np.log(df["Y2"]))   # refit L

    if verbose:
        print("-->estimating final location factor...")

    # L_hat = Final estimated location factor
    df["L_hat"] = np.exp( gam_L2.predict(df[["latitude", "longitude"]]))

    # could refit L_hat once more here if desired
    return LandSLICEModel(
        alpha_hat,
        beta_hat,
        gam_L2,
        med_land_size,
        size_field
    )

##############################
