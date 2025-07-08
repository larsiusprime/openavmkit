import calendar
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from openavmkit.data import _get_sales
from openavmkit.utilities.data import div_df_z_safe


def calculate_time_adjustment(
    df_sales_in: pd.DataFrame, settings: dict, period: str = "M", verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate a time adjustment multiplier for sales data.

    Processes sales data to compute a median sale price per sqft over time (at a
    resolution determined dynamically), interpolates missing values, and returns a
    DataFrame with daily time adjustment multipliers.

    Parameters
    ----------
    df_sales_in : pandas.DataFrame
        Input sales DataFrame.
    settings : dict
        Settings dictionary.
    period : str, optional
        Initial period type ("M", "Q", or "Y"). Defaults to "M".
    verbose : bool, optional
        If True, print progress information. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with time adjustment values per day.
    """

    # We assume that all the sales we are presented with are valid sales, and for a single modeling group

    # We need at least 5 sales in a given time period to make a valid time adjustment
    min_sale_count = 5

    # We assume we have access to the following fields:
    essential_fields = [
        "sale_date",
        "sale_year",
        "sale_month",
        "sale_quarter",
        "sale_price",
        "bldg_area_finished_sqft",
        "land_area_sqft",
    ]
    for field in essential_fields:
        if field not in df_sales_in:
            raise ValueError(f"Field '{field}' not found in the sales data.")

    df_sales = df_sales_in.copy()

    if "sale_quarter" not in df_sales:
        df_sales["sale_quarter"] = (df_sales["sale_month"] - 1) // 3 + 1
        df_sales["sale_quarter"] = (
            df_sales["sale_year"].astype(str)
            + "-Q"
            + df_sales["sale_quarter"].astype(str)
        )

    df_sales["sale_price_per_impr_sqft"] = div_df_z_safe(
        df_sales, "sale_price", "bldg_area_finished_sqft"
    )
    df_sales["sale_price_per_land_sqft"] = div_df_z_safe(
        df_sales, "sale_price", "land_area_sqft"
    )

    # Determine whether land or improvement drives value the modeling group:
    per = _determine_value_driver(df_sales, settings)
    sale_field = f"sale_price_per_{per}_sqft"

    df_per = df_sales[df_sales[sale_field].gt(0)]

    # Determine the time resolution (Month, Quarter, Year) -- "M", "Q", or "Y":
    period = _determine_time_resolution(df_per, sale_field, min_sale_count, period)

    if verbose:
        print(f"--> Using period: {period}")
        print(f"--> Crunching time adjustment...")
    # Derive the time adjustment:
    df_crunch = _crunch_time_adjustment(df_per, sale_field, period, min_sale_count)
    if verbose:
        print(f"--> Flattening time adjustment...")
    # Flatten out the time adjustment to daily values:
    df_time = _flatten_periods_to_days(df_per, df_crunch, period, verbose)
    print(f"--> Time adjustment calculated for {len(df_time)} days.")

    return df_time


def apply_time_adjustment(
    df_sales_in: pd.DataFrame, settings: dict, period: str = "M", verbose: bool = False
) -> pd.DataFrame:
    """
    Compute time adjustment multipliers and apply them to adjust sale prices forward in time.

    Parameters
    ----------
    df_sales_in : pandas.DataFrame
        Input sales DataFrame.
    settings : dict
        Settings dictionary containing time adjustment parameters.
    period : str, optional
        Period type to use for adjustment ("M", "Q", or "Y"). Defaults to "M".
    verbose : bool, optional
        If True, print verbose output during computation. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Sales DataFrame with an added `sale_price_time_adj` column.
    """
    df_sales = df_sales_in.copy()
    df_time = calculate_time_adjustment(df_sales_in, settings, period, verbose)

    # df_time starts with 1.0 on the first day and ends with X.0 on the last day
    # if we were to divide by this value, we would time-adjust all sales BACKWARDS in time
    # what we want is to time-adjust all sales FORWARDS in time
    # we therefore normalize to the last day, then take the reciprocal to reverse the effect
    df_time["value"] = 1 / (df_time["value"] / df_time["value"].iloc[-1])

    # now we have a multiplier that we can straightforwardly multiply sales by, that will bring all sales FORWARDS in time

    # we merge the time adjustment back into the sales data
    df_time = df_time.rename(
        columns={"value": "time_adjustment", "period": "sale_date"}
    )

    # ensure both dtypes are datetime:
    dtype_time = df_time["sale_date"].dtype
    dtype_sales = df_sales["sale_date"].dtype
    if dtype_time != "datetime64[ns]":
        df_time["sale_date"] = pd.to_datetime(df_time["sale_date"])
    if dtype_sales != "datetime64[ns]":
        df_sales["sale_date"] = pd.to_datetime(df_sales["sale_date"])

    # now, ensure both are converted to YYYY-MM-DD format:
    df_time["sale_date"] = df_time["sale_date"].dt.strftime("%Y-%m-%d")
    df_sales["sale_date"] = df_sales["sale_date"].dt.strftime("%Y-%m-%d")

    df_sales = pd.merge(df_sales, df_time, how="left", on="sale_date")

    # we multiply the sale price by the time adjustment
    df_sales["sale_price_time_adj"] = (
        df_sales["sale_price"] * df_sales["time_adjustment"]
    )

    # we drop the time adjustment column
    df_sales = df_sales.drop(columns=["time_adjustment"])

    if "sale_price_per_impr_sqft" in df_sales:
        df_sales["sale_price_time_adj_per_impr_sqft"] = div_df_z_safe(
            df_sales, "sale_price_time_adj", "bldg_area_finished_sqft"
        )
    if "sale_price_per_land_sqft" in df_sales:
        df_sales["sale_price_time_adj_per_land_sqft"] = div_df_z_safe(
            df_sales, "sale_price_time_adj", "land_area_sqft"
        )

    return df_sales


def enrich_time_adjustment(
    df_in: pd.DataFrame, settings: dict, verbose: bool = False
) -> pd.DataFrame:
    """
    Enrich the sales data by generating time-adjusted sales if not already present.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input sales DataFrame.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Enriched sales DataFrame.
    """
    df = df_in.copy()

    # Gather settings
    ta = settings.get("data", {}).get("process", {}).get("time_adjustment", {})

    # Apply time adjustment if necessary
    if "sale_price_time_adj" not in df:
        if verbose:
            print("Applying time adjustment...")
        period = ta.get("period", "Q")
        df = apply_time_adjustment(df.copy(), settings, period=period, verbose=verbose)

    return df


#######################################
# PRIVATE
#######################################


def _generate_days(start_date: datetime, end_date: datetime):
    """Generate a list of days between two dates, inclusive."""
    days = []
    date = start_date
    while date <= end_date:
        days.append(date.strftime("%Y-%m-%d"))
        date = date + timedelta(days=1)
    return days


def _get_expected_periods(df: pd.DataFrame, period: str):
    """Determine the expected time periods from the DataFrame based on sale dates.

    Ensures the 'sale_date' column is of datetime type, then generates a list of periods
    (Year, Quarter, Month, or Day) spanning from the earliest to latest sale date.
    """
    if "sale_date" not in df:
        raise ValueError("Field 'sale_date' not found in the DataFrame.")
    else:
        dtype = df["sale_date"].dtype
        if dtype != "datetime64[ns]":
            df["sale_date"] = pd.to_datetime(df["sale_date"])

    if period not in ["Y", "Q", "M", "D"]:
        raise ValueError(f"Invalid period '{period}'.")

    # get the earliest and latest date:
    date_min: datetime = df["sale_date"].min()
    date_max: datetime = df["sale_date"].max()
    min_year = date_min.year
    max_year = date_max.year
    years = [year for year in range(min_year, max_year + 1)]
    periods = []
    if period == "Y":
        periods = years
    elif period == "Q":
        for year in years:
            for quarter in range(1, 5):
                periods.append(f"{year}Q{quarter}")
    elif period == "M":
        for year in years:
            for month in range(1, 13):
                periods.append(f"{year}-{month:02d}")
    elif period == "D":
        # start from date_min and go to date_max, inclusively:
        date = date_min
        while date <= date_max:
            periods.append(date.strftime("%Y-%m-%d"))
            date = date + timedelta(days=1)
    return periods


def _convert_periods_to_middle_days(periods: list[str], period_type: str):
    """Convert a list of periods into the middle day of each period.

    For each period (year, quarter, or month), the function computes the midpoint date,
    which represents the median day of the period.
    """
    days = []
    if period_type == "Q":
        for period in periods:
            year, quarter = period.split("Q")
            month = (int(quarter) - 1) * 3 + 1
            last_month = month + 2
            # get the first and last day of the quarter:
            first_day = datetime.strptime(f"{year}-{month:02}-01", "%Y-%m-%d")
            last_day = datetime.strptime(
                f"{year}-{last_month:02}-{calendar.monthrange(int(year), last_month)[1]}",
                "%Y-%m-%d",
            )
            # get the day halfway between first and last day:
            mid_day = first_day + (last_day - first_day) / 2
            days.append(mid_day.strftime("%Y-%m-%d"))
    elif period_type == "M":
        for period in periods:
            year, month = period.split("-")
            first_day = datetime.strptime(f"{year}-{month:02}-01", "%Y-%m-%d")
            last_day = datetime.strptime(
                f"{year}-{month:02}-{calendar.monthrange(int(year), int(month))[1]}",
                "%Y-%m-%d",
            )
            mid_day = first_day + (last_day - first_day) / 2
            days.append(mid_day.strftime("%Y-%m-%d"))
    elif period_type == "Y":
        for period in periods:
            year = period
            first_day = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
            last_day = datetime.strptime(f"{year}-12-31", "%Y-%m-%d")
            mid_day = first_day + (last_day - first_day) / 2
            days.append(mid_day.strftime("%Y-%m-%d"))
    return days


def _flatten_periods_to_days(
    df_in: pd.DataFrame, df_crunched: pd.DataFrame, period: str, verbose=False
):
    """Flatten crunched period data to daily resolution.

    Maps the crunched periods to their corresponding middle-day representations, sets
    'period' as the index, and interpolates missing daily values.
    """
    periods_expected = _get_expected_periods(df_in, "D")
    old_periods = df_crunched["period"].values
    periods_actual = _convert_periods_to_middle_days(
        df_crunched["period"].values, period
    )
    rename_period_map = dict(zip(old_periods, periods_actual))
    df_crunched = df_crunched.copy()
    df_crunched["period"] = df_crunched["period"].map(rename_period_map)

    # ensure that "period" is the index of df_crunched:
    df_crunched = df_crunched.set_index("period")
    df_crunched = df_crunched.rename(columns={"value": "median"})
    if verbose:
        print("---->interpolating missing periods...")

    interpolated = _interpolate_missing_periods(
        periods_expected, periods_actual, df_crunched
    )

    # wrap everything up in a dataframe matching periods_expected to interpolated values:
    interpolated = pd.DataFrame({"period": periods_expected, "value": interpolated})
    interpolated["period"] = pd.to_datetime(interpolated["period"])

    return interpolated


def _interpolate_missing_periods(periods_expected, periods_actual, df_median):
    """Interpolate missing median values for expected periods.

    Constructs a Series of median values aligned with the expected periods, interpolates
    missing values linearly, and fills edge NaNs.
    """
    periods_actual = [p for p in periods_actual if p in periods_expected]

    # Create a mapping of periods_actual to their corresponding median values
    period_to_value = df_median["median"].reindex(periods_actual)

    # Create an array for values aligned with periods_expected
    values = pd.Series(index=periods_expected, dtype=np.float64)

    # Map known values directly (filtering for valid indices)
    valid_periods = [p for p in periods_actual if p in values.index]
    values.loc[valid_periods] = period_to_value.loc[valid_periods].values

    # Interpolate missing values
    values.interpolate(method="linear", inplace=True, limit_direction="both")

    # Fill remaining NaN with closest non-NaN values (cap edges)
    values.ffill(inplace=True)
    values.bfill(inplace=True)

    # Return as a NumPy array for compatibility
    return values.values


def _crunch_time_adjustment(
    df_in: pd.DataFrame, field: str, period: str = "M", min_count: int = 5
):
    """Crunch sales data by computing median sale price per sqft over specified time
    periods.

    Groups the sales data by period (Year, Quarter, or Month) and computes the median
    value for the specified field, then filters groups with fewer than min_count sales and
    normalizes the values.
    """
    df = df_in[df_in[field].gt(0)].copy()

    if period == "Y":
        # group by year
        df_grouped = df.groupby("sale_year")
    elif period == "Q":
        # group by quarter
        df_grouped = df.groupby("sale_year_quarter")
    elif period == "M":
        # group by month
        df_grouped = df.groupby("sale_year_month")
    else:
        raise ValueError(f"Invalid period '{period}'.")

    # calculate the median field value:
    df_median = df_grouped[field].agg(["count", "median"])

    # filter df_median to only include periods with at least min_count sales:
    df_median = df_median[df_median["count"].ge(min_count)]

    # get unique periods:
    periods_actual = df_median.index.values
    periods_expected = _get_expected_periods(df, period)

    values = _interpolate_missing_periods(periods_expected, periods_actual, df_median)

    # normalize the values by dividing by the value from the first period:
    values = values / values[0]

    if period == "M":
        # perform a 3-period moving average to smooth the curve, but leave first and last unchanged:
        for idx in range(1, len(values) - 1):
            values[idx] = (values[idx - 1] + values[idx] + values[idx + 1]) / 3

    # generate a DataFrame with the index of periods_expected matched with values:
    df_result = pd.DataFrame({"period": periods_expected, "value": values})

    # unpack into daily values

    return df_result


def _determine_value_driver(df_in: pd.DataFrame, settings: dict):
    """Determine whether land or improvement drives the value in the modeling group.

    Compares median sale prices per sqft for improved and land properties and returns
    "land" if the relative difference is small and a significant portion of sales are
    land; otherwise returns "impr".
    """
    df = df_in.copy()

    df = _get_sales(df, settings)

    df_impr = df[df["bldg_area_finished_sqft"].gt(0)]
    df_land = df[df["land_area_sqft"].gt(0)]

    df_impr["sale_price_per_impr_sqft"] = div_df_z_safe(
        df_impr, "sale_price", "bldg_area_finished_sqft"
    )
    df_land["sale_price_per_land_sqft"] = div_df_z_safe(
        df_land, "sale_price", "land_area_sqft"
    )

    # get the median sale price per sqft for both land and improvement
    median_impr = df_impr["sale_price_per_impr_sqft"].median()
    median_land = df_land["sale_price_per_land_sqft"].median()

    perc_delta = abs(median_impr - median_land) / median_land

    if perc_delta < 0.15:
        # if the difference between the two values is small, we might favor land
        perc_land = len(df_land) / len(df)
        # if a significant portion of the modeling group is land, we favor land
        if perc_land >= 0.3:
            return "land"
    return "impr"


def _determine_time_resolution(df_per, sale_field, min_sale_count, period: str = "M"):
    """Determine the appropriate time resolution (Month, Quarter, or Year) based on sales
    counts.
    """
    months_total = 0
    months_missing = 0

    if period == "M":
        # Count how many months are below the minimum sale count
        df_counts = df_per.groupby("sale_year_month")[sale_field].agg(["count"])
        months_total = len(df_counts)
        months_above_min = len(df_counts[df_counts["count"].ge(min_sale_count)])
        months_missing = len(df_counts) - months_above_min

    # Tolerate up to 2 missing months a year
    if period != "M" or months_missing / months_total > 2 / 12:

        if period == "M":
            period = "Q"

        if period == "Q":
            # Count how many quarters are below the minimum sale count
            df_counts = df_per.groupby("sale_year_quarter")[sale_field].agg(["count"])
            quarters_total = len(df_counts)
            quarters_above_min = len(df_counts[df_counts["count"].ge(min_sale_count)])
            quarters_missing = len(df_counts) - quarters_above_min

            # Tolerate up to 1 missing quarter a year
            if quarters_missing / quarters_total > 1 / 4:
                period = "Y"

    return period
