"""
Time adjustment of sale prices.

Computes a per-day multiplier that adjusts historical sale prices to the
valuation date, using a rolling median of price per area unit. The multiplier
is applied via ``sale_price_time_adj = sale_price * multiplier`` and used
throughout downstream ratio studies, modeling, and reporting.

The built-in engine can be wholly replaced for any model group by setting
``data.process.time_adjustment.from_file.<model_group>`` in ``settings.json``
to point at a precomputed CSV — useful when a jurisdiction publishes its
own time-adjustment factors. The CSV is a price index (``period``, ``index``: a
day, a month ``2023-04``, a quarter ``2023-Q2`` or a year per row, at any base) or
a schedule as OpenAVMKit writes one; see :func:`read_schedule`.

A model group with too few sales of its own can use another group's time
adjustment: ``data.process.time_adjustment.borrow.<model_group> = <other group>``.
"""
import calendar
import os
import warnings
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from openavmkit.data import _get_sales
from openavmkit.utilities.data import div_df_z_safe, do_per_model_group
from openavmkit.utilities.settings import area_unit, get_time_adjustment_instructions


def calculate_time_adjustment(
    df_sales_in: pd.DataFrame, settings: dict, period: str = "M", verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate a time adjustment multiplier for sales data.

    Processes sales data to compute a median sale price per area unit over time (at a
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
    
    unit = area_unit(settings)
    
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
        f"bldg_area_finished_{unit}",
        f"land_area_{unit}",
    ]
    for field in essential_fields:
        if field not in df_sales_in.columns:
            raise ValueError(f"Field '{field}' not found in the sales data.")

    df_sales = df_sales_in.copy()

    if "sale_quarter" not in df_sales.columns:
        df_sales["sale_quarter"] = (df_sales["sale_month"] - 1) // 3 + 1
        df_sales["sale_quarter"] = (
            df_sales["sale_year"].astype(str)
            + "-Q"
            + df_sales["sale_quarter"].astype(str)
        )

    df_sales[f"sale_price_per_impr_{unit}"] = div_df_z_safe(
        df_sales, "sale_price", f"bldg_area_finished_{unit}"
    )
    df_sales[f"sale_price_per_land_{unit}"] = div_df_z_safe(
        df_sales, "sale_price", f"land_area_{unit}"
    )

    # Determine whether land or improvement drives value the modeling group:
    per = _determine_value_driver(df_sales, settings)
    sale_field = f"sale_price_per_{per}_{unit}"

    df_per_unfiltered = df_sales[df_sales[sale_field].gt(0)]
    # Exclude the wrong-side sales from the per-area median: vacant sales for an
    # impr-driven index, improved sales for a land-driven index. Without this,
    # vacant sales (land-only prices) divided by the parcel's currently-recorded
    # bldg_area pull the impr-PPSF median down by orders of magnitude.
    df_per = df_per_unfiltered
    if "vacant_sale" in df_per_unfiltered.columns and len(df_per_unfiltered) > 0:
        if per == "impr":
            df_per_filtered = df_per_unfiltered[~df_per_unfiltered["vacant_sale"].eq(True)]
        elif per == "land":
            df_per_filtered = df_per_unfiltered[df_per_unfiltered["vacant_sale"].eq(True)]
        else:
            df_per_filtered = df_per_unfiltered
        # If the V/I filter would empty the dataset, fall back to no filter rather
        # than producing a broken time-adjustment curve. This happens when a model
        # group has only one side (e.g. an apartment group with no vacant sales but
        # _determine_value_driver picked "land", or vice versa).
        if len(df_per_filtered) == 0:
            warnings.warn(
                f"Time-adjustment V/I filter for per={per!r} would leave 0 sales "
                f"(of {len(df_per_unfiltered)} candidates). Falling back to the "
                "unfiltered set — the resulting index may be biased."
            )
        else:
            df_per = df_per_filtered

    # If we still have no usable sales, return a flat (no-op) multiplier schedule
    # covering the sales date range. Downstream code applies the multiplier via
    # `sale_price * value`, so value=1.0 leaves prices unchanged.
    if len(df_per) == 0:
        warnings.warn(
            "Time adjustment: no sales with a positive "
            f"{sale_field} remain. Returning a flat multiplier=1.0 schedule."
        )
        if len(df_sales) > 0 and df_sales["sale_date"].notna().any():
            start = df_sales["sale_date"].min()
            end = df_sales["sale_date"].max()
            dates = pd.date_range(start=start, end=end, freq="D")
        else:
            dates = pd.DatetimeIndex([pd.Timestamp.today().normalize()])
        return pd.DataFrame({"period": dates, "value": np.ones(len(dates))})

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
    df_sales_in: pd.DataFrame,
    settings: dict,
    period: str = "M",
    write: bool = False,
    verbose: bool = False
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
    write : bool, optional
        Whether to write out the time adjustment data as a separate CSV file
    verbose : bool, optional
        If True, print verbose output during computation. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Sales DataFrame with an added `sale_price_time_adj` column.
    """
    # Groups that borrow another group's time adjustment: worked out from that group's sales (or its file).
    borrowed = {}
    for group, source in get_time_adjustment_instructions(settings).get("borrow", {}).items():
        if not source or source == group:
            continue
        if "model_group" not in df_sales_in or not df_sales_in["model_group"].eq(source).any():
            warnings.warn(
                f"Model group {group!r} borrows the time adjustment of {source!r}, which has no sales: "
                f"{group!r} works out its own."
            )
            continue
        df_source = df_sales_in[df_sales_in["model_group"].eq(source)]
        borrowed[group] = _schedule(df_source, settings, source, period, verbose)
    return do_per_model_group(
        df_sales_in,
        settings,
        apply_time_adjustment_per_model_group,
        {"settings": settings, "period": period, "write": write, "verbose": verbose, "borrowed": borrowed},
        key="key_sale"
    )


def _period_date(text):
    """A period as its first day: 2023-04-15, 2023-04 (a month), 2023-Q2 or 2023Q2 (a quarter), 2023 (a year)."""
    t = str(text).strip().upper()
    if not t or t in ("NAN", "NONE", "NAT"):
        return pd.NaT
    q = t.replace(" ", "").replace("-Q", "Q")
    if len(q) == 6 and q[4] == "Q" and q[:4].isdigit() and q[5] in "1234":
        return pd.Timestamp(year=int(q[:4]), month=3 * (int(q[5]) - 1) + 1, day=1)
    if len(t) == 4 and t.isdigit():
        return pd.Timestamp(year=int(t), month=1, day=1)
    if len(t) == 7 and t[4] in "-/" and t[:4].isdigit() and t[5:].isdigit():
        return pd.Timestamp(year=int(t[:4]), month=int(t[5:]), day=1)
    return pd.to_datetime(t, errors="coerce")


def read_schedule(path: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Read a time adjustment file and check it.

    Two kinds are accepted:

    * a price index: ``period`` (a day ``2023-04-15``, a month ``2023-04``, a quarter
      ``2023-Q2``, or a year ``2023``; one row each) and ``index`` (the price level then, at
      any base: 100, 1.0...; positive). Between periods it is interpolated day by day.
    * a schedule as OpenAVMKit writes one: ``period``, ``start_indexed``, ``end_indexed``,
      ``correction_factor`` (``start_indexed`` is the index).

    Parameters
    ----------
    path : str
        The CSV file.

    Returns
    -------
    tuple[pd.DataFrame | None, list[str]]
        The daily schedule (``period``, ``start_indexed``, ``end_indexed``, ``correction_factor``,
        prices brought to the last period), or None; and every problem found, in words.
    """
    try:
        # As written: problems are quoted as the file has them ("n/a", not NaN).
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception as e:
        return None, [f"It couldn't be read as a CSV file: {e}"]
    df.columns = [str(c).strip().lower() for c in df.columns]
    problems: list[str] = []
    value = "index" if "index" in df.columns else "start_indexed" if "start_indexed" in df.columns else None
    if "period" not in df.columns:
        problems.append("There's no 'period' column (the date each price level is for).")
    if value is None:
        problems.append("There's no 'index' column (the price level for each period).")
    if problems:
        return None, problems
    dates = df["period"].map(_period_date)
    levels = pd.to_numeric(df[value], errors="coerce")
    bad_dates = df.loc[dates.isna(), "period"].astype(str).tolist()
    if bad_dates:
        problems.append(
            f"{len(bad_dates)} period(s) aren't a date, a month (2023-04), a quarter (2023-Q2) or a year: "
            + ", ".join(bad_dates[:5])
        )
    bad_levels = [v or "(blank)" for v in df.loc[levels.isna() | levels.le(0), value].astype(str)]
    if bad_levels:
        problems.append(f"{len(bad_levels)} {value} value(s) aren't positive numbers: " + ", ".join(bad_levels[:5]))
    repeated = dates[dates.duplicated() & dates.notna()]
    if len(repeated):
        problems.append(
            f"{len(repeated)} period(s) appear more than once: "
            + ", ".join(sorted({d.strftime("%Y-%m-%d") for d in repeated})[:5])
        )
    ok = dates.notna() & levels.notna() & levels.gt(0)
    if int(ok.sum()) < 2:
        problems.append("It needs at least 2 periods with a price level.")
    if problems:
        return None, problems
    points = pd.Series(levels[ok].values, index=pd.DatetimeIndex(dates[ok])).sort_index()
    days = pd.date_range(points.index[0], points.index[-1], freq="D")
    daily = points.reindex(points.index.union(days)).interpolate(method="time").reindex(days)
    out = pd.DataFrame({"period": days.strftime("%Y-%m-%d"), "start_indexed": daily.values / daily.iloc[0]})
    out["end_indexed"] = out["start_indexed"] / out["start_indexed"].iloc[-1]
    out["correction_factor"] = 1 / out["end_indexed"]
    return out, []


def _cover(df_time: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """A schedule extended, flat, to the days of every sale: a sale before its first day gets the first day's
    correction, one after its last day the last day's (no trend is carried past what the schedule says)."""
    days = pd.to_datetime(df_time["period"])
    sold = pd.to_datetime(dates, errors="coerce").dropna()
    if len(sold) == 0:
        return df_time
    start, end = min(days.min(), sold.min().normalize()), max(days.max(), sold.max().normalize())
    full = pd.date_range(start, end, freq="D")
    out = df_time.copy()
    out.index = days
    out = out.reindex(full).ffill().bfill()
    out["period"] = full.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _schedule(df_sales: pd.DataFrame, settings: dict, model_group: str, period: str, verbose: bool) -> pd.DataFrame:
    """A group's daily schedule: from its file when it has one that can be used, else worked out from its sales."""
    path = get_time_adjustment_instructions(settings).get("from_file", {}).get(model_group)
    if path is not None:
        df_time, problems = read_schedule(path)
        if df_time is not None:
            print(f"Found a valid time adjustment file for model group {model_group}. Loading...")
            return df_time
        warnings.warn(
            f"Time adjustment file for model group {model_group} ({path}) can't be used: "
            + " ".join(problems)
            + " Working it out from the sales instead."
        )
    df_time = calculate_time_adjustment(df_sales, settings, period, verbose)
    df_time = df_time.rename(columns={"value": "start_indexed"})
    df_time["end_indexed"] = df_time["start_indexed"] / df_time["start_indexed"].iloc[-1]
    df_time["correction_factor"] = 1 / df_time["end_indexed"]
    return df_time


def read_time_adjustment_from_file(ta_settings: dict, model_group: str) -> pd.DataFrame|None:
    """A group's time adjustment file, as a daily schedule, or None (no file, or one that can't be used)."""
    override_path = ta_settings.get("from_file",{}).get(model_group)
    if override_path is None:
        return None
    df_time, problems = read_schedule(override_path)
    if df_time is None:
        warnings.warn(f"Time adjustment file for model group {model_group} can't be used: " + " ".join(problems))
    return df_time


def apply_time_adjustment_per_model_group(
    df_sales_in: pd.DataFrame,
    settings: dict,
    model_group: str,
    period: str = "M",
    write: bool = False,
    verbose: bool = False,
    borrowed: dict | None = None,
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
    write : bool, optional
        Whether to write out the time adjustment data as a separate CSV file
    verbose : bool, optional
        If True, print verbose output during computation. Defaults to False.
    borrowed : dict, optional
        Schedules borrowed from other model groups, by the group that borrows them.

    Returns
    -------
    pandas.DataFrame
        Sales DataFrame with an added `sale_price_time_adj` column.
    """
    if len(df_sales_in) == 0:
        return df_sales_in

    unit = area_unit(settings)

    df_sales = df_sales_in.copy()

    if borrowed and model_group in borrowed:
        # Another group's schedule, held flat outside its dates.
        df_time = _cover(borrowed[model_group], df_sales["sale_date"])
    elif get_time_adjustment_instructions(settings).get("from_file", {}).get(model_group) is not None:
        # The group's own file, held flat outside its dates.
        df_time = _cover(_schedule(df_sales_in, settings, model_group, period, verbose), df_sales["sale_date"])
    else:
        df_time = _schedule(df_sales_in, settings, model_group, period, verbose)

    os.makedirs(f"out/time_adjustment/{model_group}", exist_ok=True)
    df_time.to_csv(f"out/time_adjustment/{model_group}/time_adjustment_schedule.csv", index=False)

    # now we have a multiplier that we can straightforwardly multiply sales by, that will bring all sales FORWARDS in time
    # we merge the time adjustment back into the sales data
    df_time = df_time[["period","correction_factor"]].copy().rename(
        columns={"period": "sale_date"}
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
            df_sales["sale_price"] * df_sales["correction_factor"]
    )

    # When the time-adjustment schedule yielded no usable correction (e.g. insufficient
    # per-period variance for rural land sales), correction_factor is NaN and so is
    # sale_price_time_adj. Fall back to the unadjusted sale_price so downstream consumers
    # (canonical splits, spatial lag, _get_sales' positive-price filter) don't drop the row.
    n_missing = int(df_sales["sale_price_time_adj"].isna().sum())
    if n_missing > 0:
        df_sales["sale_price_time_adj"] = df_sales["sale_price_time_adj"].fillna(
            df_sales["sale_price"]
        )
        warnings.warn(
            f"Time adjustment unavailable for {n_missing:,} sales (model group "
            f"'{model_group}'); falling back to unadjusted sale_price for those rows."
        )

    # we drop the time adjustment column
    df_sales = df_sales.drop(columns=["correction_factor"])

    if f"sale_price_per_impr_{unit}" in df_sales:
        df_sales[f"sale_price_time_adj_per_impr_{unit}"] = div_df_z_safe(
            df_sales, "sale_price_time_adj", f"bldg_area_finished_{unit}"
        )
    if f"sale_price_per_land_{unit}" in df_sales:
        df_sales[f"sale_price_time_adj_per_land_{unit}"] = div_df_z_safe(
            df_sales, "sale_price_time_adj", f"land_area_{unit}"
        )

    return df_sales


def enrich_time_adjustment(
    df_in: pd.DataFrame, settings: dict, write: bool = False, verbose: bool = False
) -> pd.DataFrame:
    """
    Enrich the sales data by generating time-adjusted sales if not already present.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input sales DataFrame.
    settings : dict
        Settings dictionary.
    write : bool
        Whether to write out the time adjustment to a separate CSV file. Defaults to False.
    verbose : bool, optional
        If True, print verbose output. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Enriched sales DataFrame.
    """
    df = df_in.copy()

    # Gather settings
    ta = get_time_adjustment_instructions(settings)

    # Apply time adjustment if necessary
    if "sale_price_time_adj" not in df:
        if verbose:
            print("Applying time adjustment...")
        period = ta.get("period", "Q")
        df = apply_time_adjustment(df.copy(), settings, period=period, write=write, verbose=verbose)

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
    years = [year for year in range(int(min_year), int(max_year + 1))]
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
    """Crunch sales data by computing median sale price per area unit over specified time
    periods.

    Groups the sales data by period (Year, Quarter, or Month) and computes the median
    value for the specified field, then filters groups with fewer than min_count sales and
    normalizes the values.
    """
    df = df_in[df_in[field].gt(0)].copy()
    
    if len(df) == 0:
        return None
    
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


    return df_result


def _determine_value_driver(df_in: pd.DataFrame, settings: dict):
    """Determine whether land or improvement drives the value in the modeling group.

    Compares median sale prices per area unit for improved and land properties and returns
    "land" if the relative difference is small and a significant portion of sales are
    land; otherwise returns "impr".
    """
    unit = area_unit(settings)
    
    df = df_in.copy()

    df = _get_sales(df, settings)

    df_impr = df[df[f"bldg_area_finished_{unit}"].gt(0)]
    df_land = df[df[f"land_area_{unit}"].gt(0)]

    if len(df_land) == 0 and len(df_impr) !=0:
        return "impr"
    if len(df_impr) == 0 and len(df_land) !=0:
        return "land"
    if len(df_impr) == 0 and len(df_land) == 0:
        warnings.warn("No positive values for both land and improvement, default to improved value driver")
        return "impr"

    df_impr[f"sale_price_per_impr_{unit}"] = div_df_z_safe(
        df_impr, "sale_price", f"bldg_area_finished_{unit}"
    )
    df_land[f"sale_price_per_land_{unit}"] = div_df_z_safe(
        df_land, "sale_price", f"land_area_{unit}"
    )

    # get the median sale price per area unit for both land and improvement
    median_impr = df_impr[f"sale_price_per_impr_{unit}"].median()
    median_land = df_land[f"sale_price_per_land_{unit}"].median()

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
            if quarters_total == 0:
                warnings.warn("Unable to determine time resolution due to insufficient data")
                return period
            quarters_above_min = len(df_counts[df_counts["count"].ge(min_sale_count)])
            quarters_missing = len(df_counts) - quarters_above_min

            # Tolerate up to 1 missing quarter a year
            if quarters_missing / quarters_total > 1 / 4:
                period = "Y"

    return period
