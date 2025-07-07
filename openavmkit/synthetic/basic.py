import math
from shapely.geometry import box
from datetime import datetime as dt

import numpy as np
import pandas as pd
import geopandas as gpd

from openavmkit.income import derive_prices
from openavmkit.projection import project_trend
from openavmkit.time_adjustment import _generate_days


class SyntheticData:
    """A simple wrapper for holding generated data along with separate land/building inflation/depreciation curves

    Attributes
    ----------
    df_universe : pd.DataFrame
        The parcel universe
    df_sales : pd.DataFrame
        The sales observations
    time_land_mult : pd.DataFrame
        Land inflation curve over time
    time_bldg_mult : pd.DataFrame
        Building depreciation curve over time
    """

    def __init__(
        self,
        df_universe: pd.DataFrame,
        df_sales: pd.DataFrame,
        time_land_mult: pd.DataFrame,
        time_bldg_mult: pd.DataFrame,
    ):
        """Initialize a SyntheticData object

        Parameters
        ----------
        df_universe : pd.DataFrame
            The parcel universe
        df_sales : pd.DataFrame
            The sales observations
        time_land_mult : pd.DataFrame
            Land inflation curve over time
        time_bldg_mult : pd.DataFrame
            Building depreciation curve over time
        """
        self.df_universe = df_universe
        self.df_sales = df_sales
        self.time_land_mult = time_land_mult
        self.time_bldg_mult = time_bldg_mult


def generate_depreciation_curve(
    lifetime: int = 60,
    weight_linear: float = 0.2,
    weight_logistic: float = 0.8,
    steepness: float = 0.3,
    inflection_point: int = 20,
) -> np.ndarray:
    """Generates a depreciation curve that blends straight-line and logistic ("S-curve")
    methods.

    The function returns an array whose *i*-th element represents the remaining
    proportion of value after *i* years.

    A weighted average of straight-line
    depreciation and a logistic decay is used, giving you control over both the
    shape (via the logistic parameters) and the relative influence of each curve.

    Parameters
    ----------
    lifetime : int
        Total service life of the asset in years—the point at which the value
        is considered fully depreciated (zero).
    weight_linear : float
        Weight assigned to the straight-line component.
        Use 1.0 (and set ``weight_logistic`` to 0.0) for pure straight-line
        depreciation.
    weight_logistic : float
        Weight assigned to the logistic (sigmoid) component.
        Use 1.0 (and set ``weight_linear`` to 0.0) for a pure logistic curve.
    steepness : float
        The logistic steepness parameter *k*.
        Higher values make the "drop-off" around the inflection point sharper;
        lower values make the curve more gradual.
    inflection_point : int
        Year (zero-based index) at which the logistic curve crosses 50 % of its
        starting value.

        - For years **earlier** than ``inflection_point`` the logistic term is > 0.5,
          so the asset is still retaining more than half its value.
        - For years **later** than ``inflection_point`` the logistic term is < 0.5,
          so the asset value declines faster.

        Adjust this to shift the midpoint of rapid depreciation earlier or later
        in the asset’s life.

    Returns
    -------
    np.ndarray
        an array whose *i*-th element represents the remaining proportion of value after *i* years.
    """

    depreciation = np.zeros(lifetime)

    for i in range(0, lifetime):
        # linear depreciation
        linear = (lifetime - i) / lifetime

        # logistic depreciation
        logistic = 1 / (1 + np.exp(steepness * (i - inflection_point)))

        # combine the two curves
        y_combined = ((weight_linear * linear) + (weight_logistic * logistic)) / (
            weight_linear + weight_logistic
        )

        depreciation[i] = y_combined

    return depreciation


def generate_inflation_curve(
    start_year: int,
    end_year: int,
    annual_inflation_rate: float = 0.02,
    annual_inflation_rate_stdev: float = 0.01,
    seasonality_amplitude: float = 0.10,
    monthly_noise: float = 0.0,
    daily_noise: float = 0.0,
) -> np.ndarray:
    """
    Generate a pseudo-random daily price index covering one or more calendar years.

    The curve is built in three passes:

    1. **Annual step** – Each year’s inflation factor is drawn from a normal
       distribution *N*( ``annual_inflation_rate`` , ``annual_inflation_rate_stdev`` ).
    2. **Monthly step** – Values are linearly interpolated to month-ends, then
       modulated by a sinusoidal seasonal component that peaks in late spring
       and bottoms in mid-winter.  The seasonal deviation is bounded by
       ``seasonality_amplitude`` (e.g. 0.10 => +/- 10 % around the baseline),
       after which optional multiplicative monthly noise is applied.
    3. **Daily step** – Each month is linearly interpolated to daily resolution,
       and optional multiplicative daily noise is applied.

    Parameters
    ----------
    start_year : int
        First calendar year (January 1) included in the series.
    end_year : int
        Last calendar year (December 31) included in the series.
    annual_inflation_rate : float, default 0.02
        Mean annual inflation rate (e.g. 0.02 => 2 %).
    annual_inflation_rate_stdev : float, default 0.01
        Standard deviation of the *annual* inflation rate used in step 1.
    seasonality_amplitude : float, default 0.10
        Maximum proportional deviation caused by intra-year seasonality
        (positive in spring/summer, negative in winter).  Expressed as a
        fraction of the underlying price level.
    monthly_noise : float, default 0.0
        Standard deviation of multiplicative noise applied once per month:
        the month-end multiplier is drawn from *N*(1.0, ``monthly_noise``).
    daily_noise : float, default 0.0
        Standard deviation of multiplicative noise applied once per day:
        each daily multiplier is drawn from *N*(1.0, ``daily_noise``).

    Returns
    -------
    np.ndarray
        One-dimensional array of length equal to the number of days from
        ``start_year``-01-01 through ``end_year``-12-31 inclusive.  The first
        element is 1.0; subsequent elements represent the cumulative price
        index (≥ 0) after applying inflation, seasonality, and noise.
    """

    start_date = dt(year=start_year, month=1, day=1)
    end_date = dt(year=end_year, month=12, day=31)

    duration_years = (
        end_year - start_year
    ) + 1  # we add + 1 because we end in December of the end year
    duration_months = (duration_years * 12) + 1
    duration_days = (end_date - start_date).days + 1

    # First we generate a series of data points
    # +1 for the beginning value, then one for the end of each year:
    time_mult_years = np.array([1.0] * (duration_years + 1))

    # We increase each point after the first by the annual inflation rate:
    for i in range(1, duration_years + 1):
        curr_year_inflation_rate = np.random.normal(
            annual_inflation_rate, annual_inflation_rate_stdev
        )
        time_mult_years[i] = time_mult_years[i - 1] * (1 + curr_year_inflation_rate)

    # We subdivide each year into months, interpolating between the yearly values:
    # +1 for the beginning value, then one for the end of each month:
    time_mult_months = np.array([1.0] * duration_months)

    # We interpolate between the yearly values:
    # We start at 1.0, then each next value is for the end of that month
    month = 1
    year = 0
    for t in range(1, duration_months):
        curr_mult = time_mult_years[year]
        next_mult = time_mult_years[year + 1]
        time_mult_months[t] = curr_mult + ((next_mult - curr_mult) * (month / 12))
        month += 1
        if month > 12:
            month = 1
            year += 1

    # We prepare an array for seasonality:
    # +1 for the beginning value, then one for the end of each month:
    time_mult_season = np.array([1.0] * duration_months)

    # We add seasonality amplitude:
    # - prices peak in May/June
    # - prices bottom out in December/January
    # - we use a sine wave to model this:
    t_m = 0
    for t in range(0, duration_months):
        # t_n is the normalized time, ranging from 0 to 1
        t_n = t_m / 12
        # 1.4 * pi is the phase shift to peak in May/June
        time_mult_season[t] = 1.0 + (
            (math.sin((1.4 * math.pi) - (2 * math.pi * t_n))) * seasonality_amplitude
        )
        t_m += 1
        if t_m > 12:
            t_m = 1

    # We overlay the seasonality amplitude onto time_mult_months:
    time_mult_months = time_mult_months * time_mult_season

    # We add monthly noise:
    monthly_noise_values = np.random.normal(1.0, monthly_noise, duration_months)
    monthly_noise_values[0] = 1.0

    # We overlay the monthly noise onto time_mult:
    time_mult_months = time_mult_months * monthly_noise_values

    # Then we subdivide each month into days, interpolating between the monthly values:
    time_mult_days = np.array([1.0] * duration_days)

    curr_date = start_date
    curr_month = curr_date.month - 1
    curr_month_len_in_days = (curr_date + pd.DateOffset(months=1) - curr_date).days
    t_month = 0

    day_of_month = 1

    # We iterate over the days, interpolating between the monthly values:
    for t in range(0, duration_days):
        # add a time delta to curr_date of one day:
        t_month_next = t_month + 1
        mult_curr = time_mult_months[t_month]
        mult_next = time_mult_months[t_month_next]
        frac = day_of_month / curr_month_len_in_days
        time_mult_days[t] = mult_curr + (mult_next - mult_curr) * frac

        # add daily noise
        time_mult_days[t] *= np.random.normal(1.0, daily_noise)

        curr_date = curr_date + pd.DateOffset(days=1)
        new_month = curr_date.month - 1
        day_of_month += 1
        if new_month != curr_month:
            day_of_month = 1
            t_month += 1
            curr_month = new_month
            curr_month_len_in_days = (
                curr_date + pd.DateOffset(months=1) - curr_date
            ).days

    return time_mult_days


def create_rect(x: float, y: float, width: float, height: float):
    """Create a Shapely Polygon in the shape of a rectangle

    Parameters
    ----------
    x : float
        The x-center of the rectangle
    y : float
        The y-center of the rectangle
    width : float
        The width of the rectangle
    height : float
        The height of the rectangle

    Returns
    --------
    Polygon
        A Shapely polygon representing a rectangle
    """

    half_width = width / 2
    half_height = height / 2
    # Determine the bounds for the square
    minx = x - half_width
    maxx = x + half_width
    miny = y - half_height
    maxy = y + half_height
    return box(minx, miny, maxx, maxy)


def generate_basic(
    size: int,
    percent_sales: float = 0.1,
    percent_vacant: float = 0.1,
    noise_sales: float = 0.05,
    seed: int = 1337,
    land_inflation: dict = None,
    bldg_inflation: dict = None,
):
    """Build a synthetic real-estate data set of parcels and (optionally) sales.

    A square grid of ``size × size`` parcels is laid out around a notional
    CBD (central business district).  For each parcel the routine simulates—

    * **Land characteristics** (area, latitude/longitude, distance to CBD,
      land value).
    * **Improvement characteristics** (finished square footage,
      quality/condition scores, age, building type, depreciated value).
    * **Time-varying inflation factors** for land and improvements, generated
      with :func:`generate_inflation_curve`.
    * **Optional sale events.**  Each parcel is given a Bernoulli trial with
      success probability ``percent_sales``.  A successful trial produces one
      sale whose price is the sum of time-adjusted land and building values
      plus uniform noise ``+/- noise_sales``.

    A parcel may instead be vacant, controlled by ``percent_vacant``.  Vacant
    parcels have land value only.  All random draws are reproducible via the
    ``seed`` argument.

    Parameters
    ----------
    size : int
        Length of one side of the square study area.  The function creates
        ``size^2`` parcels.
    percent_sales : float, default ``0.1``
        Probability (0–1) that a parcel receives *one* valid sale event.
    percent_vacant : float, default ``0.1``
        Probability (0–1) that a parcel is vacant (no improvement).
        Vacant parcels may still transact if selected by ``percent_sales``.
    noise_sales : float, default ``0.05``
        Half-width of the uniform noise band applied to the simulated sale
        price: the multiplier is drawn from
        :math:`\\mathrm{U}(1-\\text{noise\\_sales},\\;1+\\text{noise\\_sales})`.
    seed : int, default ``1337``
        Seed passed to :pyfunc:`numpy.random.seed` for reproducibility.
    land_inflation : dict or None, optional
        Keyword arguments forwarded to :func:`generate_inflation_curve` to
        create a daily land-value index.  If *None*, a preset dict with
        10 % mean annual inflation (plus mild seasonality) is used.
    bldg_inflation : dict or None, optional
        Same as ``land_inflation`` but for building improvements.  Defaults to
        a preset dict with 2 % mean annual inflation and no seasonality.

    Returns
    -------
    SyntheticData
        An object with four public attributes

        ``parcels`` : geopandas.GeoDataFrame
            One record per parcel with geometry and static attributes
            (distance to CBD, quality/condition scores, etc.).

        ``sales`` : pandas.DataFrame
            One record per simulated sale (may be empty).  Includes sale price,
            unit-price metrics, sale date, and vacancy flag.

        ``land_index`` : pandas.DataFrame
            Daily land inflation multipliers (`period`, `value`).

        ``bldg_index`` : pandas.DataFrame
            Daily building inflation multipliers (`period`, `value`).

    Notes
    -----
    * The CBD is assumed to sit at latitude **29.760762° N**, longitude
      **95.361937° W** (roughly downtown Houston, TX).  Parcel coordinates are
      spread +/-0.25° lat / +/-0.20° lon from that center.
    * Land value decreases approximately exponentially with Euclidean
      (grid-based) distance from the CBD.
    * Building value per square foot depends on building type
      (“A”, “B”, “C”), quality, condition, and age depreciation
      (linear caps at 100 years).

    Examples
    --------
    >>> sd = generate_basic(size=25, percent_sales=0.2, seed=42)
    >>> sd.parcels.head()
    >>> sd.sales[['key', 'sale_price', 'sale_date']].sample(5)
    """
    data = {
        "key": [],
        "geometry": [],
        "neighborhood": [],
        "bldg_area_finished_sqft": [],
        "land_area_sqft": [],
        "bldg_type": [],
        "bldg_quality_num": [],
        "bldg_condition_num": [],
        "bldg_age_years": [],
        "land_value": [],
        "bldg_value": [],
        "total_value": [],
        "dist_to_cbd": [],
        "latitude": [],
        "longitude": [],
        "is_vacant": [],
    }

    data_sales = {
        "key": [],
        "key_sale": [],
        "valid_sale": [],
        "valid_for_ratio_study": [],
        "vacant_sale": [],
        "is_vacant": [],
        "sale_price": [],
        "sale_price_per_impr_sqft": [],
        "sale_price_per_land_sqft": [],
        "sale_age_days": [],
        "sale_date": [],
        "sale_year": [],
        "sale_month": [],
        "sale_quarter": [],
        "sale_year_month": [],
        "sale_year_quarter": [],
    }

    latitude_center = 29.760762
    longitude_center = -95.361937

    height = 0.5
    width = 0.4

    nw_lat = latitude_center - width / 2
    nw_lon = longitude_center - height / 2

    base_land_value = 5
    base_bldg_value = 50
    quality_value = 5

    # set a random seed:
    np.random.seed(seed)

    start_date = dt(year=2020, month=1, day=1)
    end_date = dt(year=2024, month=12, day=31)

    days_duration = (end_date - start_date).days

    # default time/bldg inflation parameters:
    if land_inflation is None:
        land_inflation = {
            "start_year": start_date.year,
            "end_year": end_date.year,
            "annual_inflation_rate": 0.1,
            "annual_inflation_rate_stdev": 0.01,
            "seasonality_amplitude": 0.025,
            "monthly_noise": 0.0125,
            "daily_noise": 0.0025,
        }
    if bldg_inflation is None:
        bldg_inflation = {
            "start_year": start_date.year,
            "end_year": end_date.year,
            "annual_inflation_rate": 0.02,
            "annual_inflation_rate_stdev": 0.005,
            "seasonality_amplitude": 0.00,
            "monthly_noise": 0.01,
            "daily_noise": 0.005,
        }

    # generate the time adjustment if so desired, using `land_inflation` as parameters:
    time_land_mult = generate_inflation_curve(**land_inflation)
    time_bldg_mult = generate_inflation_curve(**bldg_inflation)

    df_time_land_mult = pd.DataFrame(
        {"period": _generate_days(start_date, end_date), "value": time_land_mult}
    )
    df_time_bldg_mult = pd.DataFrame(
        {"period": _generate_days(start_date, end_date), "value": time_bldg_mult}
    )
    df_time_land_mult["period"] = pd.to_datetime(df_time_land_mult["period"])
    df_time_bldg_mult["period"] = pd.to_datetime(df_time_bldg_mult["period"])

    for y in range(0, size):
        for x in range(0, size):

            _x = x / size
            _y = y / size

            latitude = nw_lat + (width * _x)
            longitude = nw_lon + (height * _y)

            dist_x = abs(_x - 0.5)
            dist_y = abs(_y - 0.5)
            dist_center = (dist_x**2 + dist_y**2) ** 0.5

            valid_sale = False
            vacant_sale = False
            # roll for a sale:
            if np.random.rand() < percent_sales:
                valid_sale = True

            # base value with exponential falloff from center:
            _base_land_value = base_land_value - 1
            land_value_per_land_sqft = 1 + (_base_land_value * (1 - dist_center))

            key = f"{x}-{y}"
            land_area_sqft = np.random.randint(5445, 21780)
            land_value = land_area_sqft * land_value_per_land_sqft

            if np.random.rand() < percent_vacant:
                is_vacant = True
            else:
                is_vacant = False

            if not is_vacant:
                bldg_area_finished_sqft = np.random.randint(1000, 2500)
                bldg_quality_num = np.clip(np.random.normal(3, 1), 0, 6)
                bldg_condition_num = np.clip(np.random.normal(3, 1), 0, 6)
                bldg_age_years = np.clip(np.random.normal(20, 10), 0, 100)

                bldg_type = np.random.choice(["A", "B", "C"])

                bldg_type_mult = 1.0
                if bldg_type == "A":
                    bldg_type_mult = 0.5
                elif bldg_type == "B":
                    bldg_type_mult = 1.0
                elif bldg_type == "C":
                    bldg_type_mult = 2.0
            else:
                bldg_area_finished_sqft = 0
                bldg_quality_num = 0
                bldg_condition_num = 0
                bldg_age_years = 0
                land_value = 0
                bldg_type = ""
                bldg_type_mult = 0

            bldg_value_per_sqft = (
                base_bldg_value + (quality_value * bldg_quality_num)
            ) * bldg_type_mult

            depreciation_from_age = min(0.0, 1 - (bldg_age_years / 100))
            depreciation_from_condition = min(0.0, 1 - (bldg_condition_num / 6))

            total_depreciation = (
                depreciation_from_age + depreciation_from_condition
            ) / 2

            bldg_value_per_sqft = bldg_value_per_sqft * (1 - total_depreciation)
            bldg_value = bldg_area_finished_sqft * bldg_value_per_sqft

            total_value = land_value + bldg_value

            # TODO: properly evolve the city over time with sales in "real time" so we don't wind up with weird situations
            # such as the one we're in, where the sale version of the price doesn't take into account that the building was
            # younger than at the valuation date

            sale_price = 0
            sale_price_per_land_sqft = 0
            sale_price_per_impr_sqft = 0
            sale_age_days = 0

            sale_date = None
            sale_year = None
            sale_month = None
            sale_quarter = None
            sale_year_month = None
            sale_year_quarter = None

            if valid_sale:
                # account for time inflation:
                sale_age_days = np.random.randint(0, days_duration)
                land_value_per_land_sqft_sale = (
                    land_value_per_land_sqft * time_land_mult[sale_age_days]
                )
                # bldg_value_per_sqft_sale = bldg_value_per_sqft * time_bldg_mult[sale_age_days]
                bldg_value_per_sqft_sale = bldg_value_per_sqft

                # calculate total values:
                land_value_sale = land_area_sqft * land_value_per_land_sqft_sale
                bldg_value_sale = bldg_area_finished_sqft * bldg_value_per_sqft_sale
                total_value_sale = land_value_sale + bldg_value_sale

                # add some noise
                sale_price = total_value_sale * (
                    1 + np.random.uniform(-noise_sales, noise_sales)
                )

                sale_price_per_land_sqft = sale_price / land_area_sqft
                sale_price_per_impr_sqft = sale_price / bldg_area_finished_sqft

                sale_date = start_date + pd.DateOffset(days=sale_age_days)
                sale_year = sale_date.year
                sale_month = sale_date.month
                sale_quarter = (sale_month - 1) // 3 + 1
                sale_year_month = f"{sale_year:04}-{sale_month:02}"
                sale_year_quarter = f"{sale_year:04}Q{sale_quarter}"

                vacant_sale = is_vacant

            geometry = create_rect(longitude, latitude, height, width)

            data["key"].append(str(key))
            data["neighborhood"].append("")
            data["bldg_area_finished_sqft"].append(bldg_area_finished_sqft)
            data["land_area_sqft"].append(land_area_sqft)
            data["bldg_quality_num"].append(bldg_quality_num)
            data["bldg_condition_num"].append(bldg_condition_num)
            data["bldg_age_years"].append(bldg_age_years)
            data["bldg_type"].append(bldg_type)
            data["land_value"].append(land_value)
            data["bldg_value"].append(bldg_value)
            data["total_value"].append(total_value)
            data["dist_to_cbd"].append(dist_center)
            data["latitude"].append(latitude)
            data["longitude"].append(longitude)
            data["geometry"].append(geometry)
            data["is_vacant"].append(is_vacant)

            if valid_sale:
                sale_date_YYYY_MM_DD = sale_date.strftime("%Y-%m-%d")
                data_sales["key"].append(str(key))
                data_sales["key_sale"].append(str(key) + "---" + sale_date_YYYY_MM_DD)
                data_sales["valid_sale"].append(valid_sale)
                data_sales["valid_for_ratio_study"].append(valid_sale)
                data_sales["vacant_sale"].append(vacant_sale)
                data_sales["is_vacant"].append(vacant_sale)
                data_sales["sale_price"].append(sale_price)
                data_sales["sale_price_per_impr_sqft"].append(sale_price_per_impr_sqft)
                data_sales["sale_price_per_land_sqft"].append(sale_price_per_land_sqft)
                data_sales["sale_age_days"].append(sale_age_days)
                data_sales["sale_date"].append(sale_date)
                data_sales["sale_year"].append(sale_year)
                data_sales["sale_month"].append(sale_month)
                data_sales["sale_quarter"].append(sale_quarter)
                data_sales["sale_year_month"].append(sale_year_month)
                data_sales["sale_year_quarter"].append(sale_year_quarter)

    df = gpd.GeoDataFrame(data, geometry="geometry")
    df_sales = pd.DataFrame(data_sales)

    # Derive neighborhood:
    distance_quantiles = [0.0, 0.25, 0.75, 1.0]
    distance_bins = [np.quantile(df["dist_to_cbd"], q) for q in distance_quantiles]
    distance_labels = ["urban", "suburban", "rural"]
    df["neighborhood"] = pd.cut(
        df["dist_to_cbd"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )

    # Derive based on longitude/latitude what (NW, NE, SW, SE) quadrant a parcel is in:
    df["quadrant"] = ""
    df.loc[df["latitude"].ge(latitude_center), "quadrant"] += "s"
    df.loc[df["latitude"].lt(latitude_center), "quadrant"] += "n"
    df.loc[df["longitude"].ge(longitude_center), "quadrant"] += "e"
    df.loc[df["longitude"].lt(longitude_center), "quadrant"] += "w"

    df["neighborhood"] = (
        df["neighborhood"].astype(str) + "_" + df["quadrant"].astype(str)
    )

    sd = SyntheticData(df, df_sales, df_time_land_mult, df_time_bldg_mult)
    return sd


def generate_income_sales(
    count_per_year: int,
    start_year: int,
    end_year: int,
    base_noi: float = 50000,
    noi_growth: float = 0.02,
    base_cap_rate: float = 0.05,
    cap_rate_growth: float = 0.01,
    holding_period: int = 7,
    target_irr: float = 0.15,
):
    transactions = []
    hist_cap_rates = {}
    noi = base_noi
    hist_noi = {}
    cap_rate = base_cap_rate

    for year in range(start_year, end_year + 1):

        hist_cap_rates[year] = cap_rate
        hist_noi[year] = noi

        # x years before now, capped at start year
        start_lookback = max(start_year, year - holding_period)
        recent_caps = []
        recent_nois = []
        for y in range(start_lookback, year + 1):
            recent_caps.append(hist_cap_rates[y])
            recent_nois.append(hist_noi[y])
        recent_caps = np.array(recent_caps)
        recent_nois = np.array(recent_nois)

        # naively extrapolate exit cap rate along linear trend of last X years
        expected_exit_cap_rate = project_trend(
            recent_caps, len(recent_caps) + holding_period
        )

        # naively extrapolate noi along linear trend of last X years
        expected_exit_noi = project_trend(
            recent_nois, len(recent_nois) + holding_period
        )

        # get expected annual growth rate over the holding period:
        expected_noi_growth = (expected_exit_noi / noi) ** (1 / holding_period) - 1

        for i in range(0, count_per_year):
            entry_price, _ = derive_prices(
                target_irr,
                expected_exit_cap_rate,
                expected_exit_noi,
                expected_noi_growth,
                holding_period,
            )

            transactions.append(
                {
                    "sale_price": entry_price,
                    "sale_year": year,
                    "entry_noi": noi,
                    "holding_period": holding_period,
                    "expected_exit_noi": expected_exit_noi,
                    "expected_noi_growth": expected_noi_growth,
                    "expected_exit_cap_rate": expected_exit_cap_rate,
                }
            )

        # growth
        noi = noi * (1 + noi_growth)
        cap_rate = cap_rate * (1 + cap_rate_growth)

    df_transactions = pd.DataFrame(data=transactions)

    return {"transactions": df_transactions, "cap_rates": hist_cap_rates}
