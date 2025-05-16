import numpy as np
from statsmodels.regression.linear_model import RegressionResults


class GarbageModel:
  def __init__(self, min_value: float, max_value: float, sales_chase: float, normal: bool):
    self.min_value = min_value
    self.max_value = max_value
    self.sales_chase = sales_chase
    self.normal = normal


class AverageModel:
  def __init__(self, type: str, sales_chase: float):
    self.type = type
    self.sales_chase = sales_chase


class NaiveSqftModel:
  def __init__(
      self,
      ind_per_built_sqft: float,
      ind_per_land_sqft: float,
      sales_chase: float
  ):
    self.ind_per_built_sqft = ind_per_built_sqft
    self.ind_per_land_sqft = ind_per_land_sqft
    self.sales_chase = sales_chase


class LocalSqftModel:
  def __init__(
      self,
      loc_map: dict,
      location_fields: list,
      overall_per_impr_sqft: float,
      overall_per_land_sqft: float,
      sales_chase: float
  ):
    self.loc_map = loc_map
    self.location_fields = location_fields
    self.overall_per_impr_sqft = overall_per_impr_sqft
    self.overall_per_land_sqft = overall_per_land_sqft
    self.sales_chase = sales_chase


class LocalSomersModel:
  def __init__(
      self,
      loc_map: dict,
      location_fields: list,
      overall_per_impr_sqft: float,
      overall_land_unit_ft: float,
      sales_chase: float
  ):
    self.loc_map = loc_map
    self.location_fields = location_fields
    self.overall_per_impr_sqft = overall_per_impr_sqft
    self.overall_land_unit_ft = overall_land_unit_ft
    self.sales_chase = sales_chase


class GroundTruthModel:
  def __init__(
      self,
      observed_field: str,
      ground_truth_field: str
  ):
    self.observed_field = observed_field
    self.ground_truth_field = ground_truth_field


class SpatialLagModel:
  def __init__(
      self,
      per_sqft: bool
  ):
    self.per_sqft = per_sqft


class PassThroughModel:
  def __init__(
    self,
    field: str,
  ):
    self.field = field


class GWRModel:
  def __init__(self,
      coords_train: list[tuple[float, float]],
      X_train: np.ndarray,
      y_train: np.ndarray,
      gwr_bw: float
  ):
    self.coords_train = coords_train
    self.X_train = X_train
    self.y_train = y_train
    self.gwr_bw = gwr_bw


class MRAModel:
  def __init__(self, fitted_model: RegressionResults, intercept: bool):
    self.fitted_model = fitted_model
    self.intercept = intercept