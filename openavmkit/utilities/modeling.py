import numpy as np
from statsmodels.regression.linear_model import RegressionResults
from pygam import LinearGAM, s, te
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


class NaiveSqftModel:
    """An intentionally bad predictive model, to use as a sort of control. Produces predictions equal to the prevailing
    average price/sqft of land or building, multiplied by the observed size of the parcel's land or building, depending
    on whether it's vacant or improved.

    Attributes
    ----------
    dep_per_built_sqft: float
        Dependent variable value divided by improved square footage
    dep_per_land_sqft: float
        Dependent variable value divided by land square footage
    sales_chase : float
        Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
        parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
        ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
        **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
    """
    def __init__(
        self, dep_per_built_sqft: float, dep_per_land_sqft: float, sales_chase: float
    ):
        """Initialize a NaiveSqftModel

        Parameters
        ----------
        dep_per_built_sqft: float
            Dependent variable value divided by improved square footage
        dep_per_land_sqft: float
            Dependent variable value divided by land square footage
        sales_chase : float
            Simulates sales chasing. If 0.0, no sales chasing will occur. For any other value, predictions against sold
            parcels will chase (copy) the observed sale price, with a bit of random noise equal to the value of
            ``sales_chase``. So ``sales_chase=0.05`` will copy each sale price with 5% random noise.
            **NOTE**: This is for analytical purposes only, one should not intentionally chase sales when working in actual production.
        """
        self.dep_per_built_sqft = dep_per_built_sqft
        self.dep_per_land_sqft = dep_per_land_sqft
        self.sales_chase = sales_chase


class LocalSqftModel:
    """Produces predictions equal to the localized average price/sqft of land or building, multiplied by the observed
    size of the parcel's land or building, depending on whether it's vacant or improved.

    Unlike ``NaiveSqftModel``, this model is sensitive to location, based on user-specified locations, and might
    actually result in decent predictions.

    Attributes
    ----------
    loc_map : dict[str : tuple[DataFrame, DataFrame]
        A dictionary that maps location field names to localized per-sqft values. The dictionary itself is keyed by the
        names of the location fields themselves (e.g. "neighborhood", "market_region", "census_tract", etc.) or whatever
        the user specifies.

        Each entry is a tuple containing two DataFrames:

          - Values per improved square foot
          - Values per land square foot

        Each DataFrame is keyed by the unique *values* for the given location. (e.g. "River heights", "Meadowbrook",
        etc., if the location field in question is "neighborhood") The other field in each DataFrame will be
        ``{location_field}_per_impr_sqft`` or ``{location_field}_per_land_sqft``
    location_fields : list
        List of location fields used (e.g. "neighborhood", "market_region", "census_tract", etc.)
    overall_per_impr_sqft : float
        Fallback value per improved square foot, to use for parcels of unspecified location. Based on the
        overall average value for the dataset.
    overall_per_land_sqft : float
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
        overall_per_impr_sqft: float,
        overall_per_land_sqft: float,
        sales_chase: float,
    ):
        """Initialize a LocalSqftModel

        Parameters
        ----------
        loc_map : dict[str : tuple[DataFrame, DataFrame]
            A dictionary that maps location field names to localized per-sqft values. The dictionary itself is keyed by the
            names of the location fields themselves (e.g. "neighborhood", "market_region", "census_tract", etc.) or whatever
            the user specifies.

            Each entry is a tuple containing two DataFrames:

              - Values per improved square foot
              - Values per land square foot

            Each DataFrame is keyed by the unique *values* for the given location. (e.g. "River heights", "Meadowbrook",
            etc., if the location field in question is "neighborhood") The other field in each DataFrame will be
            ``{location_field}_per_impr_sqft`` or ``{location_field}_per_land_sqft``
        location_fields : list
            List of location fields used (e.g. "neighborhood", "market_region", "census_tract", etc.)
        overall_per_impr_sqft : float
            Fallback value per improved square foot, to use for parcels of unspecified location. Based on the
            overall average value for the dataset.
        overall_per_land_sqft : float
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
        self.overall_per_impr_sqft = overall_per_impr_sqft
        self.overall_per_land_sqft = overall_per_land_sqft
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
    per_sqft : bool
        If True, normalize by square foot. If False, use the direct value of the spatial lag field.

    """
    def __init__(self, per_sqft: bool):
        """Initialize a SpatialLagModel

        Parameters
        ----------
        per_sqft : bool
            If True, normalize by square foot. If False, use the direct value of the spatial lag field.
        """
        self.per_sqft = per_sqft


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
    ):
        """Initialize a PassThroughModel

        Parameters
        ----------
        field : str
            The field that holds the values you want to pass through as predictions
        """
        self.field = field


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

    """
    def __init__(
        self,
        coords_train: list[tuple[float, float]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        gwr_bw: float,
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