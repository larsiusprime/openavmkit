import numpy as np


def get_unit_ft(lot_value: np.ndarray | float, frontage_ft: np.ndarray | float, depth_ft: np.ndarray | float):
  """
  Given a lot's total value, frontage, and depth, calculate the somers unit-foot value
  :param lot_value: The total value of the lot
  :param frontage_ft: The frontage of the lot, in feet
  :param depth_ft: The depth of the lot, in feet
  :return: The value of the somers unit-foot for the lot (1 ft of frontage x 100 ft of depth)
  """
  # Calculate the Somers unit-foot value
  return lot_value / (get_depth_percent_ft(depth_ft) * frontage_ft)


def get_unit_m(lot_value: np.ndarray | float, frontage_m: np.ndarray | float, depth_m: np.ndarray | float):
  """
  Given a lot's total value, frontage, and depth, calculate the somers unit-foot value (but in meters)
  :param lot_value: The total value of the lot
  :param frontage_m: The frontage of the lot, in meters
  :param depth_m: The depth of the lot, in meters
  :return: The value of the somers unit-foot for the lot (30.48 cm of frontage x 30.48 m of depth)
  """
  # Convert to feet
  depth_ft = depth_m / 0.3048
  frontage_ft = frontage_m / 0.3048
  return get_unit_ft(lot_value, frontage_ft, depth_ft)


def get_lot_value_ft(unit_value: np.ndarray | float, frontage_ft: np.ndarray | float, depth_ft: np.ndarray | float):
  """
  Given a unit value, frontage, and depth, calculate the value of the lot
  :param unit_value: The value of the unit lot (1 ft of frontage x 100 ft of depth)
  :param frontage_ft: The frontage of the lot
  :param depth_ft: The depth of the lot
  :return: The Somers system value of the lot
  """
  # Calculate the value of the lot using the Somers formula
  return get_depth_percent_ft(depth_ft) * unit_value * frontage_ft


def get_lot_value_m(unit_value: np.ndarray | float, frontage_m: np.ndarray | float, depth_m: np.ndarray | float):
  """
  Given a unit value, frontage, and depth, calculate the value of the lot
  :param unit_value: The value of the unit lot (30.48 cm of frontage x 30.48 m of depth)
  :param frontage_m: The frontage of the lot
  :param depth_m: The
  :return:
  """
  depth_ft = depth_m / 0.3048
  frontage_ft = frontage_m / 0.3048
  return get_lot_value_ft(unit_value, frontage_ft, depth_ft)


def get_depth_percent_ft(depth_ft: np.ndarray | float):
  """
  Given a depth in feet, return the relative value of this lot's depth compared to the unit lot (100 ft deep)
  :param depth_ft: The depth of the lot (in feet)
  :return: % value expressed in terms of 0.0 to 1.0 (and beyond). 0 ft yields 0.0, 100 ft yields 1.0, with diminishing returns
  """
  value = (133.6 * (1 - np.exp(-0.0326 * depth_ft**0.813)))/100
  value = np.round((value*1000)+0.5)/1000  # round to the nearest 0.001 -- ensures that 100 ft is *exactly* 100%
  return value


def get_depth_percent_m(depth_m: np.ndarray | float):
  """
  Given a depth in meters, return the relative value of this lot's depth compared to the unit lot (30.48 m deep)
  :param depth_m: The depth of the lot (in meters)
  :return: % value expressed in terms of 0.0 to 1.0 (and beyond). 0 m yields 0.0, 30.48 m yields 1.0, with diminishing returns
  """
  depth_ft = depth_m / 0.3048
  return get_depth_percent_ft(depth_ft)


def get_size_in_somers_units_ft(frontage_ft: np.ndarray | float, depth_ft: np.ndarray | float):
  # How big is a lot, in somers unit-feet?

  # Normalize the depth:
  depth_percent = get_depth_percent_ft(depth_ft)

  # Multiply by frontage:
  somers_units = depth_percent * frontage_ft

  return somers_units


def get_size_in_somers_units_m(frontage_m: np.ndarray | float, depth_m: np.ndarray | float, land_area_sqm: np.ndarray | float):
  frontage_ft = frontage_m / 0.3048
  depth_ft = depth_m / 0.3048
  land_area_sqft = land_area_sqm / 0.092903
  return get_size_in_somers_units_ft(frontage_ft, depth_ft, land_area_sqft)


