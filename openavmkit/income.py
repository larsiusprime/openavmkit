import numpy as np
import pandas as pd
from scipy.optimize import brentq


def calculate_noi(price: float, cap_rate: float) -> float:
    """Calculate the Net Operating Income (NOI) given a property price and cap rate."""
    return price * cap_rate


def derive_prices(
    target_irr: float,
    exit_cap_rate: float,
    entry_noi: float,
    noi_growth: float,
    holding_period: int,
) -> tuple[float, float]:
    """Calculate the entry and exit prices based on a DCF model given:

    - **target_irr**: The target internal rate of return (IRR) (e.g., 0.10 for 10%)
    - **exit_cap_rate**: The exit cap rate used to value the property at sale (e.g., 0.06 for 6%)
    - **entry_noi**: NOI at purchase (NOI₀)
    - **noi_growth**: Expected annual NOI growth rate (g) (e.g., 0.03 for 3%)
    - **holding_period**: Holding period in years (H)

    The model assumes:

    - Annual NOI cash flows: `NOI₀ * (1 + noi_growth) ** t` for `t = 1, …, H`.
    - At exit, the property is sold at:

        Sale Price = `NOI₀ * (1 + noi_growth) ** (H + 1)` / `exit_cap_rate`

    - That sale price is discounted back to time 0.

    The DCF equation is:

    ```
    Asking Price = Σₜ₌₁ᴴ [ NOI₀ * (1 + noi_growth)^t / (1 + target_irr)^t ]
                      + [ NOI₀ * (1 + noi_growth)^(H + 1) / exit_cap_rate ]
                      / (1 + target_irr)^H
    ```

    Parameters
    ----------
    target_irr : float
        The target internal rate of return (IRR) as a decimal (e.g., 0.10 for 10%).
    exit_cap_rate : float
        The exit cap rate as a decimal (e.g., 0.06 for 6%).
    entry_noi : float
        The net operating income (NOI) at the time of purchase.
    noi_growth : float
        The expected annual growth rate of NOI as a decimal (e.g., 0.03 for 3%).
    holding_period : int
        The holding period in years.

    Returns
    -------
    tuple[float, float]
        The entry and exit prices
    """

    noi_mult = 1 + noi_growth
    irr_mult = 1 + target_irr

    # net operating income is present net operating income compounded for expected growth
    exit_noi = entry_noi * (noi_mult**holding_period)

    # exit price is expected net operating income @ exit time, divided by expected cap rate @ exit time
    exit_price = exit_noi / exit_cap_rate

    # now we get the net present value of all future cash flows

    # first, we get the sum of the discounted cash flows for every year but the last
    npv = sum(
        [entry_noi * (noi_mult**t) / (irr_mult**t) for t in range(1, holding_period)]
    )
    # then, for the last year, we add the exit price to the last year's net operating income, and discount them together
    npv += (exit_noi + exit_price) / (irr_mult**holding_period)

    # the entry price is the net present value of all future discounted cash flows
    entry_price = npv

    return entry_price, exit_price


def derive_irr(
    entry_price: float,
    exit_price: float,
    entry_cap_rate: float,
    noi_growth: float,
    holding_period: int,
) -> float:
    """Calculate the implied IRR given:

      - **entry_price**: Purchase price of the property.
      - **exit_price**: Observed sale price (terminal cash flow) at the end of the holding period.
      - **entry_cap_rate**: Entry cap rate (used to derive the initial NOI).
      - **noi_growth**: Annual growth rate of NOI (decimal form, e.g., 0.03 for 3%).
      - **holding_period**: Holding period in years.

    The model assumes:

      - NOI₀ = entry_price * entry_cap_rate
      - IRRM = (1+IRR)
      - NOIM = (1+noi_growth)

      And the DCF equation:

      ```
        NPV = ∑ₜ₌₁ᴴ [NOI₀ * (1 + noi_growth)ᵗ / (1 + IRR)ᵗ] + [sale_price / (1 + IRR)ᴴ] - entry_price = 0
        NPV = ∑ₜ₌₁ᴴ [NOI₀ * NOIMᵗ/IRRMᵗ] + [exit_price/IRRMᴴ] - entry_price = 0
      ```

    Parameters
    ----------
    entry_price : float
        The purchase price of the property.
    exit_price : float
        The observed sale price at the end of the holding period.
    entry_cap_rate : float
        The entry cap rate as a decimal (e.g., 0.06 for 6%).
    noi_growth : float
        The expected annual growth rate of NOI as a decimal (e.g., 0.03 for 3%).
    holding_period : int
        The holding period in years.

    Returns
    -------
    The IRR (as a decimal).
    """
    # Derive the initial NOI from the entry cap rate.
    entry_noi = entry_price * entry_cap_rate

    def dcf_equation(IRR: float) -> float:
        # Sum the present value of the NOI cash flows for each year.

        pv_noi = sum(
            [
                entry_noi * (1 + noi_growth) ** t / (1 + IRR) ** t
                for t in range(1, holding_period + 1)
            ]
        )
        # Discount the observed sale price to present value.
        pv_sale = exit_price / (1 + IRR) ** holding_period
        # The NPV should equal zero for the correct IRR.
        return pv_noi + pv_sale - entry_price

    # Use a numerical solver to find the IRR that zeros the equation.
    # We search in a reasonable range for IRR (here between -0.99 and 1.0, i.e. -99% to 100%).
    return brentq(dcf_equation, -0.99, 1.0, full_output=False)


def derive_noi_growth(
    target_irr: float,
    exit_cap_rate: float,
    entry_price: float,
    entry_cap_rate: float,
    holding_period: int,
    lower_bound=-0.10,
    upper_bound=0.2,
):
    entry_noi = entry_price * entry_cap_rate

    implied_growth = brentq(
        _npv_with_growth,
        a=lower_bound,
        b=upper_bound,
        args=(entry_price, entry_noi, target_irr, holding_period, exit_cap_rate),
    )
    return implied_growth


def calculate_cap_rate_growth(
    sale_price_growth: np.ndarray, noi_growth: np.ndarray
) -> np.ndarray:
    """Calculate the capitalization rate given the annual percentage changes in sale price
    ($/sqft) and net operating income (NOI).

    Given `NOI = Sale Price * Cap Rate`, the cap rate is `NOI / Sale Price`.
    Approximating the percentage change (assuming small changes):

    ```
    ΔCap Rate ≈ ΔNOI - ΔSale Price.
    ```

    Parameters
    ----------
    sale_price_growth : np.ndarray
        Annual percentage changes in sale price as decimals. (Example: [0.01, 0.01, 0.015, 0.02, 0.01] for 1%, 1%, 1.5%, 2%, 1%)
    noi_growth : np.ndarray
        Annual percentage changes in NOI as decimals. (Example: [0.01, 0.005, 0.0, -0.0025, 0.01])

    Returns
    -------
    np.ndarray :
        Capitalization rate as decimals
    """
    if len(sale_price_growth) != len(noi_growth):
        raise ValueError("Input arrays must have the same length.")

    cap_rates = []
    for s_growth, n_growth in zip(sale_price_growth, noi_growth):
        cap_rate = (n_growth + 1) / (1 + s_growth) - 1
        cap_rates.append(cap_rate)

    return np.array(cap_rates)


def calculate_noi_growth(
    sale_price_growth: np.ndarray, cap_rate_growth: np.ndarray
) -> np.ndarray:
    """Calculate the annual percentage change in NOI given the annual percentage changes
    in sale price ($/sqft) and cap rate.

    Given `NOI = Sale Price * Cap Rate`, the compounded growth in NOI is:
    `(1 + ΔSale Price) * (1 + ΔCap Rate) - 1`.


    Parameters
    ----------
    sale_price_growth : np.ndarray
        Annual percentage changes in sale price as decimals. (Example: [0.01, 0.01, 0.015, 0.02, 0.01] for 1%, 1%, 1.5%, 2%, 1%)
    cap_rate_growth : np.ndarray
        Annual percentage changes in cap rate as decimals. (Example: [0.01, 0.005, 0.0, -0.0025, 0.01])

    Returns
    -------
    np.ndarray :
        Annual percentage changes in NOI as decimals.

    """
    if len(sale_price_growth) != len(cap_rate_growth):
        raise ValueError("Input arrays must have the same length.")

    noi_growth = []
    for s_growth, c_growth in zip(sale_price_growth, cap_rate_growth):
        growth = (1 + s_growth) * (1 + c_growth) - 1
        noi_growth.append(growth)

    return np.array(noi_growth)


def derive_irr_df(
    df: pd.DataFrame, hist_cap_rates: dict, hist_noi_growths: dict
) -> pd.DataFrame:
    """Given a DataFrame of paired sales with columns:

      - "key"
      - "entry_price"
      - "exit_price"
      - "entry_date"
      - "exit_date"

    And historical dictionaries:

      - "hist_cap_rates" : {year -> entry cap rate}
      - "hist_noi_growths" : {year -> NOI growth rate}

    This function computes for each row:

      - holding_period (in years, float)
      - entry_year (from entry_date)
      - implied_IRR (using the DCF method)

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame of paired sales containing the columns ["key", "entry_price", "exit_price", "entry_date", "exit_date"]
    hist_cap_rates: dict
        A Dictionary whose keys are years and whose values are entry cap rates
    hist_noi_growths: dict
        A Dictionary whose keys are years and whose values are NOI growth rates

    Returns
    -------
    pd.DataFrame:
        A new DataFrame that includes the original columns plus these computed fields.
    """

    def compute_row(row):
        # Convert dates to Timestamp
        entry_date = pd.to_datetime(row["entry_date"])
        exit_date = pd.to_datetime(row["exit_date"])
        # Compute holding period in years (as float)
        holding_period_years = (exit_date - entry_date).days / 365.25
        # For DCF, use an integer number of years
        holding_period_int = int(round(holding_period_years))
        entry_year = entry_date.year

        # Look up historical parameters for the entry year.
        entry_cap_rate = hist_cap_rates.get(entry_year)
        noi_growth = hist_noi_growths.get(entry_year)
        if entry_cap_rate is None or noi_growth is None or holding_period_int < 1:
            return pd.Series(
                {
                    "holding_period": holding_period_int,
                    "implied_IRR": None,
                    "entry_year": entry_year,
                    "entry_cap_rate": entry_cap_rate,
                }
            )

        try:
            implied_irr = derive_irr(
                row["entry_price"],
                row["exit_price"],
                entry_cap_rate,
                noi_growth,
                holding_period_int,
            )
        except Exception:
            implied_irr = None

        return pd.Series(
            {
                "holding_period": holding_period_int,
                "implied_irr": implied_irr,
                "entry_year": entry_year,
                "entry_cap_rate": entry_cap_rate,
                "entry_noi": row["entry_price"] * entry_cap_rate,
            }
        )

    computed = df.apply(compute_row, axis=1)
    # Concatenate the computed columns with the original DataFrame.
    return pd.concat([df, computed], axis=1)


#######################################
# PRIVATE
#######################################


def _npv_with_growth(
    noi_growth_rate, entry_price, entry_noi, target_irr, holding_period, exit_cap_rate
):
    noi_mult = 1 + noi_growth_rate
    irr_mult = 1 + target_irr

    dcfs = [
        (entry_noi * (noi_mult**t))  # compounded net operating income
        / (irr_mult**t)  # discounted by target IRR
        for t in range(1, holding_period + 1)
    ]

    npv_flows = sum(dcfs)

    exit_noi = entry_noi * (noi_mult**holding_period)
    exit_price = exit_noi / exit_cap_rate
    discounted_exit_price = exit_price / (irr_mult**holding_period)

    npv = discounted_exit_price - entry_price + npv_flows
    return npv
