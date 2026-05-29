import warnings

import numpy as np
import pandas as pd

from openavmkit.data import get_hydrated_sales_from_sup, SalesUniversePair
from openavmkit.synthetic.basic import generate_basic
from openavmkit.time_adjustment import _interpolate_missing_periods, calculate_time_adjustment, apply_time_adjustment
from openavmkit.utilities.assertions import lists_are_equal


def test_interpolate_missing_periods_days():
  print("")
  data = {
    "period": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "21", "30", "41"],
    "median": [ 1 ,  2,   3,   4,   5,   6,   7,   8,   9,   10,   21,   30,   41]
  }
  df_median = pd.DataFrame(data)
  df_median = df_median.groupby("period")["median"].agg(["count","median"])
  periods_actual = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "21", "30", "41"]
  periods_expected = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41"
  ]
  results = _interpolate_missing_periods(
    periods_expected,
    periods_actual,
    df_median
  ).tolist()
  expected = [
    1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
    11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
    31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
    41.
  ]
  assert(lists_are_equal(expected, results))


def test_interpolate_missing_periods():
  print("")
  data = {
    "period": ["2014", "2016", "2019", "2020", "2021", "2022", "2024"],
    "median": [     1,      3,      6,      7,      8,      9,     11]
  }
  df_median = pd.DataFrame(data)
  df_median = df_median.groupby("period")["median"].agg(["count","median"])
  periods_actual = ["2014", "2015", "2016", "2019", "2020", "2021", "2022", "2023", "2024"]
  periods_expected = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
  results = _interpolate_missing_periods(
    periods_expected,
    periods_actual,
    df_median
  ).tolist()
  expected = [1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 11.]
  assert(lists_are_equal(expected, results))


def test_time_adjustment():
  print("")
  sd = generate_basic(100)

  sup = SalesUniversePair(sd.df_sales, sd.df_universe)
  df = get_hydrated_sales_from_sup(sup)

  df.loc[df["sale_year_quarter"].eq("2024-Q3"), "sale_price_per_impr_sqft"] = None

  # TODO: replace with proper sales subset function
  df = df[df["sale_price"].gt(0) & df["valid_sale"].ge(1)]

  df_time_m = calculate_time_adjustment(df, settings={}, period="M")
  df_time_q = calculate_time_adjustment(df, settings={}, period="Q")
  df_time_y = calculate_time_adjustment(df, settings={}, period="Y")

  time_land_mult = sd.time_land_mult.copy()

  time_land_mult["value"] = time_land_mult["value"] / time_land_mult["value"].iloc[0]

  df_norm = df.copy()
  df_norm = df_norm[df_norm["sale_price_per_impr_sqft"].gt(0)]
  df_norm["period"] = pd.to_datetime(df_norm["sale_year_quarter"])
  first_period = df_norm["period"].min()
  df_norm["sale_price_per_impr_sqft"] = df_norm["sale_price_per_impr_sqft"] / df_norm[df_norm["period"].eq(first_period)]["sale_price_per_impr_sqft"].median()

  # plt.plot(df_time_m["period"], df_time_m["value"])
  # plt.plot(df_time_q["period"], df_time_q["value"])
  # plt.plot(df_time_y["period"], df_time_y["value"])
  # plt.plot(time_land_mult["period"], time_land_mult["value"])
  #
  # #scatterplot df norm:
  # plt.scatter(df_norm["period"], df_norm["sale_price_per_impr_sqft"], color="gray", s=1)
  #
  # plt.show()


def test_apply_time_adjustment():
  print("")
  sd = generate_basic(100)

  sup = SalesUniversePair(sd.df_sales, sd.df_universe)
  df = get_hydrated_sales_from_sup(sup)
  settings = {
    "modeling":{
      "model_groups":{
        "residential_single_family":{}
      }
    }
  }

  # TODO: replace with proper sales subset function
  df = df[df["sale_price"].gt(0) & df["valid_sale"].ge(1)]

  for period, color, color2 in [("M","red", "pink"), ("Q","blue", "skyblue"), ("Y","black", "lightgray")]:
    df = apply_time_adjustment(df, settings=settings, period=period, write=False, verbose=True)

    df_median = df.groupby("sale_year_month")["sale_price_time_adj_per_impr_sqft"].agg(["count", "median"])
    df_median["period"] = pd.to_datetime(df_median.index)

    # plt.plot(df_median["period"], df_median["median"], color=color)
    # plt.scatter(df["sale_date"], df["sale_price_per_impr_sqft"], s=1, color=color2)
    # plt.scatter(df["sale_date"], df["sale_price_time_adj_per_impr_sqft"], s=1, color=color2)

  # plt.show()


# ---------------------------------------------------------------------------
# Robustness: when the V/I filter would empty the dataset, calculate_time_adjustment
# should fall back gracefully rather than crash in downstream period-derivation.
# ---------------------------------------------------------------------------


def _build_minimal_sales_df(n_rows: int, start_year: int = 2023, all_vacant: bool = False,
                             all_improved: bool = False, all_zero_bldg: bool = False) -> pd.DataFrame:
  """Build a tiny synthetic sales DF with the columns calculate_time_adjustment expects.

  Spreads sales across quarters so that per-period grouping has multiple buckets.

  ``all_vacant``: every sale has vacant_sale=True (no improved sales).
  ``all_improved``: every sale has vacant_sale=False.
  ``all_zero_bldg``: every sale has bldg_area_finished_sqft=0 (no usable per-impr signal).
  """
  rows = []
  for i in range(n_rows):
    # Spread across multiple quarters in 2023-2024.
    sale_date = pd.Timestamp(f"{start_year}-01-15") + pd.Timedelta(days=i * 30)
    q = ((sale_date.month - 1) // 3) + 1
    rows.append({
      "key_sale": f"k{i}",
      "sale_date": sale_date,
      "sale_year": sale_date.year,
      "sale_month": sale_date.month,
      "sale_year_month": f"{sale_date.year:04d}-{sale_date.month:02d}",
      "sale_quarter": q,
      "sale_year_quarter": f"{sale_date.year:04d}Q{q}",
      "sale_price": 100000.0 + i * 1000,
      "bldg_area_finished_sqft": 0.0 if (all_zero_bldg or all_vacant) else 1500.0,
      "land_area_sqft": 5000.0,
      "vacant_sale": True if all_vacant else False,
    })
  return pd.DataFrame(rows)


def test_calculate_time_adjustment_falls_back_when_all_vacant_and_per_impr():
  # Construct a dataset where every sale is vacant_sale=True but improved-area
  # column has values (some assessor data does this — the parcel currently has a
  # building but the sale itself was for a vacant parcel). _determine_value_driver
  # picks "impr" because bldg_area > 0, then our V/I filter would empty df_per.
  # The function should fall back to the unfiltered set and still produce a schedule.
  rows = []
  for i in range(20):
    sale_date = pd.Timestamp("2023-01-15") + pd.Timedelta(days=i * 30)
    q = ((sale_date.month - 1) // 3) + 1
    rows.append({
      "key_sale": f"k{i}",
      "sale_date": sale_date,
      "sale_year": sale_date.year,
      "sale_month": sale_date.month,
      "sale_year_month": f"{sale_date.year:04d}-{sale_date.month:02d}",
      "sale_quarter": q,
      "sale_year_quarter": f"{sale_date.year:04d}Q{q}",
      "sale_price": 100000.0 + i * 1000,
      "bldg_area_finished_sqft": 1500.0,   # NOT zero, so per-impr would be picked
      "land_area_sqft": 5000.0,
      "vacant_sale": True,                  # but every sale is vacant
    })
  df = pd.DataFrame(rows)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = calculate_time_adjustment(df, settings={}, period="Q")
  # Function returned a non-empty schedule
  assert len(result) > 0
  assert "value" in result.columns
  # And it warned the user about the fallback
  msgs = [str(w.message) for w in caught]
  assert any("V/I filter" in m and "0 sales" in m for m in msgs), \
    f"Expected fallback warning, got: {msgs}"


def test_calculate_time_adjustment_returns_flat_schedule_when_no_usable_sales():
  # Every sale has bldg_area_finished_sqft=0 → sale_price_per_impr_sqft is NaN/0,
  # so df_per is empty even before V/I filtering. The function should return a
  # flat schedule (value=1.0) covering the sales date range rather than crashing
  # in _get_expected_periods on NaT.
  df = _build_minimal_sales_df(5, all_zero_bldg=True)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = calculate_time_adjustment(df, settings={}, period="Q")
  assert len(result) > 0
  # All multipliers should be 1.0 (no adjustment)
  assert (result["value"] == 1.0).all()
  # And it warned about returning a flat schedule
  msgs = [str(w.message) for w in caught]
  assert any("flat multiplier" in m for m in msgs), \
    f"Expected flat-schedule warning, got: {msgs}"


def test_calculate_time_adjustment_handles_mixed_v_i_normally():
  # Sanity check: with a normal mix of vacant + improved sales, V/I filter retains
  # only the relevant side and no fallback warnings are emitted.
  df = _build_minimal_sales_df(30, all_vacant=False, all_improved=False)
  # Flag every 10th sale as vacant (10% vacant) so the filter has something to do.
  df.loc[df.index % 10 == 0, "vacant_sale"] = True
  df.loc[df.index % 10 == 0, "bldg_area_finished_sqft"] = 0.0
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = calculate_time_adjustment(df, settings={}, period="Q")
  assert len(result) > 0
  # No fallback warnings should fire
  msgs = [str(w.message) for w in caught]
  assert not any("V/I filter" in m for m in msgs), \
    f"Did not expect fallback warning for healthy mixed data; got: {msgs}"
  assert not any("flat multiplier" in m for m in msgs), \
    f"Did not expect flat-schedule warning for healthy data; got: {msgs}"
