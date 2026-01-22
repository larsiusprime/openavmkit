from __future__ import annotations
import numpy as np
from statsmodels.regression.linear_model import RegressionResults
from pygam import LinearGAM, s, te
import pandas as pd
from typing import Any

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

class GarbageModel:
    """An intentionally bad predictive model, to use as a sort of control. Produces random predictions.

    Attributes
    ----------
    min_value : float
        The minimum value of to "predict"
    max_value : float
        The maximum value of to "predict"
    sales_chase : float
        Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
        parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
        ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
        **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
    normal : bool
        If True, the randomly generated predictions follow a normal distribution based on the observed sale price's
        standard deviation. If False, randomly generated predictions follow a uniform distribution between min and max.
    """
    def __init__(
        self, min_value: float, max_value: float, sales_chase: float, normal: bool
    ):
        """Initialize a GarbageModel

        Parameters
        ----------
        min_value : float
            The minimum value of to "predict"
        max_value : float
            The maximum value of to "predict"
        sales_chase : float
            Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
            parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
            ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
            **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
        normal : bool
            If True, the randomly generated predictions follow a normal distribution based on the observed sale price's
            standard deviation. If False, randomly generated predictions follow a uniform distribution between min and max.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.sales_chase = sales_chase
        self.normal = normal


class AverageModel:
    """An intentionally bad predictive model, to use as a sort of control. Produces predictions equal to the average of
    observed sale prices.

    Attributes
    ----------
    type : str
        The type of average to use
    sales_chase : float
        Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
        parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
        ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
        **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
    """
    def __init__(self, type: str, sales_chase: float):
        """Initialize an AverageModel

        Parameters
        ----------
        type : str
            The type of average to use
        sales_chase : float
            Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
            parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
            ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
            **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
        """
        self.type = type
        self.sales_chase = sales_chase


class NaiveAreaModel:
    """An intentionally bad predictive model, to use as a sort of control. Produces predictions equal to the prevailing
    average price/area of land or building, multiplied by the observed size of the parcel's land or building, depending
    on whether it's vacant or improved.

    Attributes
    ----------
    dep_per_built_area: float
        Dependent variable value divided by improved square footage
    dep_per_land_area: float
        Dependent variable value divided by land square footage
    sales_chase : float
        Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
        parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
        ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
        **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
    """
    def __init__(
        self, dep_per_built_area: float, dep_per_land_area: float, sales_chase: float
    ):
        """Initialize a NaiveAreaModel

        Parameters
        ----------
        dep_per_built_area: float
            Dependent variable value divided by improved square footage
        dep_per_land_area: float
            Dependent variable value divided by land square footage
        sales_chase : float
            Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
            parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
            ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
            **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
        """
        self.dep_per_built_area = dep_per_built_area
        self.dep_per_land_area = dep_per_land_area
        self.sales_chase = sales_chase


class LocalAreaModel:
    """Produces predictions equal to the localized average price/area of land or building, multiplied by the observed
    size of the parcel's land or building, depending on whether it's vacant or improved.

    Unlike ``NaiveAreaModel``, this model is sensitive to location, based on user-specified locations, and might
    actually result in decent predictions.

    Attributes
    ----------
    loc_map : dict[str : tuple[DataFrame, DataFrame]
        A dictionary that maps location field names to localized per-area values. The dictionary itself is keyed by the
        names of the location fields themselves (e.g. "neighborhood", "market_region", "census_tract", etc.) or whatever
        the user specifies.

        Each entry is a tuple containing two DataFrames:

          - Values per improved square foot
          - Values per land square foot

        Each DataFrame is keyed by the unique *values* for the given location. (e.g. "River heights", "Meadowbrook",
        etc., if the location field in question is "neighborhood") The other field in each DataFrame will be
        ``{location_field}_per_impr_{unit}`` or ``{location_field}_per_land_{unit}``
    location_fields : list
        List of location fields used (e.g. "neighborhood", "market_region", "census_tract", etc.)
    overall_per_impr_area : float
        Fallback value per improved square foot, to use for parcels of unspecified location. Based on the
        overall average value for the dataset.
    overall_per_land_area : float
        Fallback value per land square foot, to use for parcels of unspecified location. Based on the overall average
        value for the dataset.
    sales_chase : float
        Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
        parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
        ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
        **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
    """

    def __init__(
        self,
        loc_map: dict,
        location_fields: list,
        overall_per_impr_area: float,
        overall_per_land_area: float,
        sales_chase: float,
    ):
        """Initialize a LocalAreaModel

        Parameters
        ----------
        loc_map : dict[str : tuple[DataFrame, DataFrame]
            A dictionary that maps location field names to localized per-area values. The dictionary itself is keyed by the
            names of the location fields themselves (e.g. "neighborhood", "market_region", "census_tract", etc.) or whatever
            the user specifies.

            Each entry is a tuple containing two DataFrames:

              - Values per improved square foot
              - Values per land square foot

            Each DataFrame is keyed by the unique *values* for the given location. (e.g. "River heights", "Meadowbrook",
            etc., if the location field in question is "neighborhood") The other field in each DataFrame will be
            ``{location_field}_per_impr_{unit}`` or ``{location_field}_per_land_{unit}``
        location_fields : list
            List of location fields used (e.g. "neighborhood", "market_region", "census_tract", etc.)
        overall_per_impr_area : float
            Fallback value per improved square foot, to use for parcels of unspecified location. Based on the
            overall average value for the dataset.
        overall_per_land_area : float
            Fallback value per land square foot, to use for parcels of unspecified location. Based on the overall average
            value for the dataset.
        sales_chase : float
            Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
            parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
            ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
            **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
        """
        self.loc_map = loc_map
        self.location_fields = location_fields
        self.overall_per_impr_area = overall_per_impr_area
        self.overall_per_land_area = overall_per_land_area
        self.sales_chase = sales_chase


class GroundTruthModel:
    """Mostly only used in Synthetic models, where you want to compare against simulation ``ground_truth`` instead of
    observed sale price, which you can never do in real life.

    Attributes
    ----------
    observed_field : str
        The field that represents observed sale prices
    ground_truth_field : str
        The field that represents platonic ground truth
    """
    def __init__(self, observed_field: str, ground_truth_field: str):
        """Initialize a GroundTruthModel object

        Parameters
        ----------
        observed_field : str
            The field that represents observed sale prices
        ground_truth_field : str
            The field that represents platonic ground truth
        """
        self.observed_field = observed_field
        self.ground_truth_field = ground_truth_field


class SpatialLagModel:
    """Use a spatial lag field as your prediction

    Attributes
    ----------
    per_area : bool
        If True, normalize by area unit. If False, use the direct value of the spatial lag field.

    """
    def __init__(self, per_area: bool):
        """Initialize a SpatialLagModel

        Parameters
        ----------
        per_area : bool
            If True, normalize by square foot. If False, use the direct value of the spatial lag field.
        """
        self.per_area = per_area


class PassThroughModel:
    """Mostly used for representing existing valuations to compare against, such as the Assessor's values

    Attributes
    ----------
    field : str
        The field that holds the values you want to pass through as predictions

    """
    def __init__(
        self,
        field: str,
        engine: str
    ):
        """Initialize a PassThroughModel

        Parameters
        ----------
        field : str
            The field that holds the values you want to pass through as predictions
        engine : str
            The model engine ("assessor" or "pass_through")
        """
        self.field = field
        self.engine = engine


class GWRModel:
    """Geographic Weighted Regression Model

    Attributes
    ----------
    coords_train : list[tuple[float, float]]
        list of geospatial coordinates corresponding to each observation in the training set
    X_train : np.ndarray
        2D array of independent variables' values from the training set
    y_train : np.ndarray
        1D array of dependent variable's values from the training set
    gwr_bw : float
        Bandwidth for GWR calculation
    df_params_test : pd.DataFrame
        Coefficients for the test set
    df_params_sales : pd.DataFrame
        Coefficients for the sales set
    df_params_universe : pd.DataFrame
        Coefficients for the universe set

    """
    def __init__(
        self,
        coords_train: list[tuple[float, float]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        gwr_bw: float
    ):
        """
        Parameters
        ----------
        coords_train : list[tuple[float, float]]
            list of geospatial coordinates corresponding to each observation in the training set
        X_train : np.ndarray
            2D array of independent variables' values from the training set
        y_train : np.ndarray
            1D array of dependent variable's values from the training set
        gwr_bw : float
            Bandwidth for GWR calculation
        """
        self.coords_train = coords_train
        self.X_train = X_train
        self.y_train = y_train
        self.gwr_bw = gwr_bw
        self.df_params_sales = None
        self.df_params_univ = None
        self.df_params_test = None


class LandSLICEModel:

    """
    SLICE stands for "Smooth Location w/ Increasing-Concavity Equation."
    
    Attributes
    ----------
    alpha : float
    beta : float
    gam_L : LinearGAM
    med_size : float
    size_field : str
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gam_L: LinearGAM,
        med_size: float,
        size_field: str
    ):
        """
        ...
        
        Parameters
        ----------
        alpha: float
        beta : float
        gam_L : LinearGAM
        med_size : float
        size_field : str
        """
        self.alpha = alpha
        self.beta = beta
        self.gam_L = gam_L
        self.med_size = med_size
        self.size_field = size_field


    def predict_size_factor(size_value: float):
        return self.alpha * (size_value / self.med_size)**self.beta

    
    def predict(
        self,
        df_in: pd.DataFrame,
        location_factor: str = "location_factor",
        size_factor: str = "size_factor",
        prediction: str = "land_value"
    ):
        df = df_in.copy()
        for field in ["latitude", "longitude", self.size_field]:
            if field not in df:
                raise ValueError(f"Required field {field} is missing from dataframe!")

        # Get location factor from Lat & Lon
        df[location_factor] = np.exp(
            self.gam_L.predict(df[["latitude", "longitude"]])
        )

        # Get size factor from power curve
        df[size_factor] = self.alpha * (np.asarray(df[self.size_field]) / self.med_size)**self.beta

        # Prediction is simply location premium times size factor
        return df[location_factor] * df[size_factor]
        

    def predict_df(
        self,
        df: pd.DataFrame,
        location_factor: str = "location_factor",
        size_factor: str = "size_factor",
        prediction: str = "land_value"
    ) -> pd.DataFrame:
        for field in ["latitude", "longitude", self.size_field]:
            if field not in df:
                raise ValueError(f"Required field {field} is missing from dataframe!")

        # Get location factor from Lat & Lon
        df[location_factor] = np.exp(
            self.gam_L.predict(df[["latitude", "longitude"]])
        )

        # Get size factor from power curve
        df[size_factor] = self.alpha * (np.asarray(df[self.size_field]) / self.med_size)**self.beta

        # Prediction is simply location premium times size factor
        df[prediction] = df[location_factor] * df[size_factor]
        return df


@dataclass
class TreeBasedCategoricalData:
    """
    Stores categorical metadata needed to reproduce LightGBM-compatible
    categorical encodings and generate numeric matrices for SHAP.
    """

    feature_names: List[str]
    categorical_cols: List[str]
    category_levels: Dict[str, List]
    bool_cols: List[str]

    # ---------- construction ----------

    @classmethod
    def from_training_data(
        cls,
        X_train: pd.DataFrame,
        categorical_cols: List[str],
    ) -> "TreeBasedCategoricalData":
        """
        Build metadata from training data AFTER categoricals have been
        converted to pandas 'category' dtype.
        """
        feature_names = list(X_train.columns)

        cat_cols = [
            c for c in categorical_cols
            if c in X_train.columns
            and pd.api.types.is_categorical_dtype(X_train[c])
        ]

        category_levels = {
            c: list(X_train[c].cat.categories)
            for c in cat_cols
        }

        bool_cols = [
            c for c in X_train.columns
            if pd.api.types.is_bool_dtype(X_train[c])
            or str(X_train[c].dtype) == "boolean"
        ]

        return cls(
            feature_names=feature_names,
            categorical_cols=cat_cols,
            category_levels=category_levels,
            bool_cols=bool_cols,
        )

    # ---------- enforcement ----------

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reapply categorical + boolean structure to a dataframe.
        Unknown categories are preserved but will map to NaN later.
        """
        X = X.reindex(columns=self.feature_names)

        # enforce booleans
        for c in self.bool_cols:
            if c in X.columns:
                X[c] = X[c].astype("boolean")

        # enforce categorical levels
        for c, levels in self.category_levels.items():
            if c in X.columns:
                X[c] = pd.Categorical(X[c], categories=levels)

        return X

    # ---------- SHAP / numeric view ----------

    def to_numeric_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert dataframe to a numeric matrix compatible with SHAP.
        Categoricals -> integer codes, unknowns/missing -> np.nan.
        """
        X = self.apply(X)
        out = X.copy()

        for c in self.categorical_cols:
            codes = out[c].cat.codes.astype(np.float64)
            codes[codes == -1] = np.nan
            out[c] = codes

        for c in self.bool_cols:
            out[c] = out[c].astype("Float64").astype(np.float64)

        return out.to_numpy(dtype=np.float64)


class LightGBMModel:
    """LightGBM Model
    
    Attributes
    ----------
    booster: Booster
        The trained LightGBM Booster model
    cat_data: TreeBasedCategoricalData
    """
    def __init__(self, booster, cat_data):
        self.booster = booster
        self.cat_data = cat_data


class GPBoostModel:
    """GPBoost Model

    Attributes
    ----------
    booster: Booster
        The trained GPBoost Booster model
    cat_data: TreeBasedCategoricalData
    """
    def __init__(self, booster, cat_data):
        self.booster = booster
        self.cat_data = cat_data


class XGBoostModel:
    """XGBoost Model
    
    Attributes
    ----------
    regressor: XGBRegressor
        The trained XGBoost XGBRegressor model
    cat_data: TreeBasedCategoricalData
    """
    def __init__(self, regressor, cat_data):
        self.regressor = regressor
        self.cat_data = cat_data


class CatBoostModel:
    """CatBoost Model
    
    Attributes
    ----------
    regressor: CatBRegressor
        The trained CatBoost CatBRegressor model
    cat_data: TreeBasedCategoricalData
    """
    def __init__(self, regressor, cat_data):
        self.regressor = regressor
        self.cat_data = cat_data


class MRAModel:
    """Multiple Regression Analysis Model

    Plain 'ol (multiple) linear regression

    Attributes
    ----------
    fitted_model: RegressionResults
        Fitted model from running the regression
    intercept : bool
        Whether the model was fit with an intercept or not.
    """
    def __init__(self, fitted_model: RegressionResults, intercept: bool):
        self.fitted_model = fitted_model
        self.intercept = intercept


class MultiMRAModel:
    """
    Multi-MRA (hierarchical local OLS) model.

    For each location field (e.g. "block", "neighborhood", ...), and for each
    distinct value of that field, we fit a separate OLS regression using the
    same set of independent variables.

    We store:
      - A global OLS coefficient vector (fallback when no local model applies)
      - A mapping from (location_field, location_value) -> coefficient vector
      - The feature_names (column order) used for all regressions
      - Whether an intercept was used
      - The location_fields (ordered most specific -> least specific)
      
    Attributes
    ----------
    coef_map : dict[str, dict[Any, np.ndarray]]
        Mapping from location field name to a dict mapping location value -> coefficient vector (aligned with feature_names).
    global_coef : np.ndarray
        Coefficient vector for the global OLS regression.
    feature_names : list[str]
        Ordered list of feature names used for all regressions.
    intercept : bool
        Whether an intercept column was used.
    location_fields : list[str]
        Location fields in order from most specific to least specific.
    """

    def __init__(
        self,
        coef_map: dict[str, dict[Any, np.ndarray]],
        global_coef: np.ndarray,
        feature_names: list[str],
        intercept: bool,
        location_fields: list[str],
    ):
        """
        Parameters
        ----------
        coef_map : dict[str, dict[Any, np.ndarray]]
            Mapping from location field name to a dict mapping
            location value -> coefficient vector (aligned with feature_names).
        global_coef : np.ndarray
            Coefficient vector for the global OLS regression.
        feature_names : list[str]
            Ordered list of feature names used for all regressions.
        intercept : bool
            Whether an intercept column was used.
        location_fields : list[str]
            Location fields in order from most specific to least specific.
        """
        self.coef_map = coef_map
        self.global_coef = global_coef
        self.feature_names = feature_names
        self.intercept = intercept
        self.location_fields = location_fields
       
# Multi-MRA optimization:

@dataclass(frozen=True)
class GreedyResult:
    variables: List[str]
    cv_r2: float
    train_r2: float


def _ols_train_and_loocv_r2(X_design: np.ndarray, y: np.ndarray, sst: float) -> Tuple[float, float]:
    """
    Returns (train_r2, loocv_r2) using the leverage shortcut for LOOCV.
    X_design includes intercept.
    """
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    yhat = X_design @ beta
    resid = y - yhat
    sse_train = float(resid.T @ resid)

    XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    h = np.sum((X_design @ XtX_inv) * X_design, axis=1)

    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    loocv_resid = resid / denom
    sse_cv = float(loocv_resid.T @ loocv_resid)

    train_r2 = 1.0 - (sse_train / sst)
    cv_r2 = 1.0 - (sse_cv / sst)
    return train_r2, cv_r2


def greedy_forward_loocv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k_max: Optional[int] = None,
    min_gain: float = 0.002,          # stop if best add improves CV-R² < 0.2%
    standardize: bool = True,
    prescreen_k: Optional[int] = 15,  # optionally screen to top K by |corr| for speed
) -> GreedyResult:
    """
    Greedy forward selection maximizing LOOCV R² (fast).
    Auto-detects intercept handling:
      - If X contains a 'const' column, treats it as intercept (always included, not selectable, not standardized).
      - Otherwise, adds an intercept internally (as before).

    Assumes X, y are numeric, aligned, and NaN-free.
    """

    yv = y.to_numpy(dtype=float)
    cols_all = list(X.columns)

    has_const = "const" in cols_all
    if has_const:
        # Treat provided const as the intercept (force-include)
        const_vec = X["const"].to_numpy(dtype=float).reshape(-1, 1)

        # Optional sanity check (not required, but helpful during debugging):
        # if not (np.nanstd(const_vec) < 1e-12):
        #     raise ValueError("'const' column exists but is not (near-)constant; cannot treat as intercept safely.")

        X_no_const = X.drop(columns=["const"])
        cols = list(X_no_const.columns)
        Xmat = X_no_const.to_numpy(dtype=float)
    else:
        const_vec = None
        cols = cols_all
        Xmat = X.to_numpy(dtype=float)

    n, p = Xmat.shape
    if p == 0 and not has_const:
        return GreedyResult([], cv_r2=float("-inf"), train_r2=float("-inf"))

    # SST
    y_centered = yv - yv.mean()
    sst = float(y_centered.T @ y_centered)
    if sst == 0.0:
        return GreedyResult([], cv_r2=0.0, train_r2=0.0)

    # Standardize X for stability (recommended)
    # IMPORTANT: never standardize the intercept; we already removed it above if present.
    if standardize and p > 0:
        mu = Xmat.mean(axis=0)
        sd = Xmat.std(axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xmat = (Xmat - mu) / sd

    # Adaptive size cap if not provided
    if k_max is None:
        # if p==0 but has_const, we still just fit const-only
        k_max = 0 if p == 0 else min(p, max(2, n // 5))
    k_max = max(0, min(k_max, p))

    # Optional prescreen to reduce p cheaply
    if prescreen_k is not None and prescreen_k < p and p > 0:
        y_std = y_centered.std(ddof=0)
        if y_std == 0:
            return GreedyResult([], cv_r2=0.0, train_r2=0.0)
        x_std = Xmat.std(axis=0, ddof=0)
        x_std = np.where(x_std == 0, 1.0, x_std)
        corr = (Xmat.T @ y_centered) / (n * x_std * y_std)
        keep_idx = np.argsort(np.abs(corr))[-prescreen_k:]
        keep_idx = np.sort(keep_idx)
        Xmat = Xmat[:, keep_idx]
        cols = [cols[i] for i in keep_idx]
        p = Xmat.shape[1]
        k_max = min(k_max, p)

    selected: List[int] = []
    remaining = set(range(p))

    # baseline
    if has_const:
        # intercept is provided as a column; baseline is const-only
        X0 = const_vec
        best_train_r2, best_cv_r2 = _ols_train_and_loocv_r2(X0, yv, sst)
    else:
        # original behavior: intercept-only baseline
        X0 = np.ones((n, 1))
        best_train_r2, best_cv_r2 = _ols_train_and_loocv_r2(X0, yv, sst)

    while remaining and len(selected) < k_max:
        step_best_cv = best_cv_r2
        step_best_train = best_train_r2
        step_best_j = None

        for j in list(remaining):
            idxs = selected + [j]
            Xsub = Xmat[:, idxs]

            if has_const:
                # Use provided intercept column
                X_design = np.column_stack([const_vec, Xsub])
            else:
                # Add intercept internally (original behavior)
                X_design = np.column_stack([np.ones(n), Xsub])

            train_r2, cv_r2 = _ols_train_and_loocv_r2(X_design, yv, sst)
            if cv_r2 > step_best_cv + 1e-12:
                step_best_cv = cv_r2
                step_best_train = train_r2
                step_best_j = j

        if step_best_j is None or (step_best_cv - best_cv_r2) < min_gain:
            break

        selected.append(step_best_j)
        remaining.remove(step_best_j)
        best_cv_r2 = step_best_cv
        best_train_r2 = step_best_train

    # Return only non-const variables (const is forced-in when present)
    return GreedyResult([cols[i] for i in selected], best_cv_r2, best_train_r2)
