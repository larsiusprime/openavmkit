from __future__ import annotations
import numpy as np


def get_unit_ft(
    lot_value: np.ndarray | float,
    frontage_ft: np.ndarray | float,
    depth_ft: np.ndarray | float
) -> np.ndarray | float:
    """
    Calculate the Somers unit-foot value for a lot.

    The Somers unit-foot value is defined as the lot's total value divided by the
    product of its frontage in feet and a standard depth of 100 feet:
    Somers unit-foot = lot_value / (frontage_ft * 100).

    Parameters
    ----------
    lot_value : numpy.ndarray or float
        Total value of the lot.
    frontage_ft : numpy.ndarray or float
        Frontage of the lot in feet.
    depth_ft : numpy.ndarray or float
        Depth of the lot in feet (typically compared against a standard 100 ft).

    Returns
    -------
    numpy.ndarray or float
        Somers unit-foot value(s), matching the shape of the inputs.
    """
    # Calculate the Somers unit-foot value
    return lot_value / (get_depth_percent_ft(depth_ft) * frontage_ft)


def get_unit_m(
    lot_value: np.ndarray | float,
    frontage_m: np.ndarray | float,
    depth_m: np.ndarray | float
) -> np.ndarray | float:
    """
    Calculate the Somers unit-meter value for a lot.

    Given a lot's total value, frontage, and depth in meters, compute the Somers
    unit-meter value (value per meter of frontage × meter of depth).

    Parameters
    ----------
    lot_value : numpy.ndarray or float
        Total value of the lot.
    frontage_m : numpy.ndarray or float
        Frontage of the lot in meters.
    depth_m : numpy.ndarray or float
        Depth of the lot in meters.

    Returns
    -------
    numpy.ndarray or float
        Somers unit-meter value(s) for the lot(s).
    """
    # Convert to feet
    depth_ft = depth_m / 0.3048
    frontage_ft = frontage_m / 0.3048
    return get_unit_ft(lot_value, frontage_ft, depth_ft)


def get_lot_value_ft(
    unit_value: np.ndarray | float,
    frontage_ft: np.ndarray | float,
    depth_ft: np.ndarray | float
) -> np.ndarray | float:
    """
    Calculate the Somers system lot value from unit value and dimensions.

    Given a unit lot value (per 1 ft frontage × 100 ft depth), frontage, and depth,
    compute the total lot value.

    Parameters
    ----------
    unit_value : numpy.ndarray or float
        The value per Somers unit (1 ft of frontage × 100 ft of depth).
    frontage_ft : numpy.ndarray or float
        Frontage of the lot in feet.
    depth_ft : numpy.ndarray or float
        Depth of the lot in feet.

    Returns
    -------
    numpy.ndarray or float
        The total Somers system value of the lot.
    """

    # Calculate the value of the lot using the Somers formula
    return get_depth_percent_ft(depth_ft) * unit_value * frontage_ft


def get_lot_value_m(
    unit_value: np.ndarray | float,
    frontage_m: np.ndarray | float,
    depth_m: np.ndarray | float
) -> np.ndarray | float:
    """
    Calculate the Somers system lot value in metric units.

    Given a unit value (per 1 m of frontage × 1 m of depth), frontage, and depth in meters,
    compute the total lot value.

    Parameters
    ----------
    unit_value : numpy.ndarray or float
        The Somers unit-meter value (value per 1 m frontage × 1 m depth).
    frontage_m : numpy.ndarray or float
        Frontage of the lot in meters.
    depth_m : numpy.ndarray or float
        Depth of the lot in meters.

    Returns
    -------
    numpy.ndarray or float
        The total Somers system value of the lot in the same shape as the inputs.
    """
    depth_ft = depth_m / 0.3048
    frontage_ft = frontage_m / 0.3048
    return get_lot_value_ft(unit_value, frontage_ft, depth_ft)


def get_depth_percent_ft(depth_ft):
    """
    Calculate the relative depth of a lot compared to a standard 100 ft depth.

    Robust to scalars, numpy arrays, lists, and pandas Series/Index.
    Non-numeric inputs are coerced to NaN. Negative depths -> NaN.
    """
    # Optional pandas support (only used if pandas objects are passed)
    try:
        import pandas as pd  # type: ignore
        _HAS_PANDAS = True
    except Exception:
        pd = None
        _HAS_PANDAS = False

    # Preserve pandas Series output if that's what we got
    is_pandas_series = _HAS_PANDAS and isinstance(depth_ft, pd.Series)
    is_pandas_index = _HAS_PANDAS and isinstance(depth_ft, pd.Index)

    # Scalar detection (works for python + numpy scalars)
    is_scalar = np.isscalar(depth_ft) or isinstance(depth_ft, np.generic)

    # Coerce to a float ndarray (fixes bug)
    if depth_ft is None:
        arr = np.array(np.nan, dtype=float)
    elif is_pandas_series or is_pandas_index:
        # pd.to_numeric fixes object dtype, strings, None, etc.
        s = pd.to_numeric(depth_ft, errors="coerce")
        arr = s.to_numpy(dtype=float)
    else:
        # Handle lists/tuples/ndarrays/scalars; coerce to float
        arr = np.asarray(depth_ft)
        if arr.dtype == object:
            # Gracefully coerce mixed/object arrays to float (non-numeric -> NaN)
            if _HAS_PANDAS:
                arr = pd.to_numeric(arr.ravel(), errors="coerce").to_numpy(dtype=float).reshape(arr.shape)
            else:
                # Minimal fallback without pandas:
                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return np.nan
                arr = np.vectorize(_to_float, otypes=[float])(arr)
        else:
            arr = arr.astype(float, copy=False)

    # Negative depths would produce complex numbers for fractional powers
    arr = np.where(arr < 0, np.nan, arr)

    # --- Compute ---
    value = (133.6 * (1.0 - np.exp(-0.0326 * np.power(arr, 0.813)))) / 100.0

    # Round-half-up to nearest 0.001
    value = np.floor(value * 1000.0 + 0.5) / 1000.0

    # Ensure exactly 1.0 at 100 ft (even after float noise)
    value = np.where(np.isclose(arr, 100.0, rtol=0.0, atol=1e-12), 1.0, value)

    # --- Return in a friendly shape/type ---
    if is_pandas_series:
        return pd.Series(value, index=depth_ft.index, name=getattr(depth_ft, "name", None))
    if is_scalar:
        return float(np.asarray(value).reshape(()))  # scalar python float
    return value



def get_depth_percent_m(depth_m: np.ndarray | float) -> np.ndarray | float:
    """
    Calculate the relative depth of a lot compared to a standard 30.48 m depth.

    This function expresses the lot’s depth as a proportion of a 30.48 m standard lot depth.
    For example, 0 m → 0.0, 30.48 m → 1.0, and values beyond 30.48 m → >1.0.

    Parameters
    ----------
    depth_m : numpy.ndarray or float
        Depth of the lot in meters.

    Returns
    -------
    numpy.ndarray or float
        Relative depth proportion(s), calculated as `depth_m / 30.48`.
    """
    depth_ft = depth_m / 0.3048
    return get_depth_percent_ft(depth_ft)


def get_size_in_somers_units_ft(
    frontage_ft: np.ndarray | float, depth_ft: np.ndarray | float
):
    """
    Get the size of a parcel or parcels in somers unit-feet.

    Parameters
    ----------
    frontage_ft : np.ndarray | float
        The frontage of the parcel, in feet
    depth_ft : np.ndarray | float
        The depth of the parcel, in feet

    Returns
    -------
    np.ndarray | float
        The converted value in somers unit-feet
    """
    # How big is a lot, in somers unit-feet?

    # Normalize the depth:
    depth_percent = get_depth_percent_ft(depth_ft)

    # Multiply by frontage:
    somers_units = depth_percent * frontage_ft

    return somers_units


def get_size_in_somers_units_m(
    frontage_m: np.ndarray | float,
    depth_m: np.ndarray | float,
    land_area_sqm: np.ndarray | float,
):
    """
    Get the size of a parcel or parcels in somers unit-meters

    Parameters
    ----------
    frontage_m : np.ndarray | float
        The frontage of the parcel, in meters
    depth_m : np.ndarray | float
        The depth of the parcel, in meters

    Returns
    -------
    np.ndarray | float
        The converted value in somers unit-meters
    """
    frontage_ft = frontage_m / 0.3048
    depth_ft = depth_m / 0.3048
    land_area_sqft = land_area_sqm / 0.092903
    return get_size_in_somers_units_ft(frontage_ft, depth_ft, land_area_sqft)
