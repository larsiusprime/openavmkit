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


def _resample_to(series_df, value_col, period_col, freq):
  """Collapse a daily period/value frame onto the given period grain."""
  s = series_df.copy()
  s[period_col] = pd.to_datetime(s[period_col])
  return s.set_index(period_col)[value_col].resample(freq).mean()


def _index_vs_truth(df_derived, df_truth, freq):
  """Normalise a derived index and the generator's ground-truth curve to their
  first common period and return (derived, truth) aligned on that grain."""
  d = _resample_to(df_derived, "value", "period", freq)
  t = _resample_to(df_truth, "value", "period", freq)
  common = d.index.intersection(t.index)
  d = d.loc[common]
  t = t.loc[common]
  return d / d.iloc[0], t / t.iloc[0]


# Thresholds below were calibrated by running this fixture across seeds
# 1 / 42 / 777 / 1337 / 2024. The observed correlation floor against
# time_land_mult was +0.63 (M), +0.67 (Q), +0.86 (Y); 0.45 leaves margin.
#
# A tight bound on relative error is NOT achievable here: across those seeds the
# median relative error ranged 5%-24% and the max reached 34%, because the
# generator's per-sale noise dominates at monthly and quarterly grain. Shape
# agreement (correlation) is the stable signal, so that is what these assert,
# with a loose error ceiling to catch gross breakage.
_CORR_FLOOR = 0.45
_MEDIAN_REL_ERR_CEILING = 0.40


def test_time_adjustment():
  print("")
  sd = generate_basic(100)

  sup = SalesUniversePair(sd.df_sales, sd.df_universe)
  df = get_hydrated_sales_from_sup(sup)

  # punch a hole so the interpolation path is exercised too
  df.loc[df["sale_year_quarter"].eq("2024-Q3"), "sale_price_per_impr_sqft"] = None

  # TODO: replace with proper sales subset function
  df = df[df["sale_price"].gt(0) & df["valid_sale"].ge(1)]

  df_time_m = calculate_time_adjustment(df, settings={}, period="M")
  df_time_q = calculate_time_adjustment(df, settings={}, period="Q")
  df_time_y = calculate_time_adjustment(df, settings={}, period="Y")

  for label, df_time, freq in (
    ("M", df_time_m, "MS"),
    ("Q", df_time_q, "QS"),
    ("Y", df_time_y, "YS"),
  ):
    assert len(df_time) > 0, f"{label}: no index produced"
    assert set(["period", "value"]).issubset(df_time.columns), f"{label}: bad columns"
    assert df_time["value"].notna().all(), f"{label}: index has NaN values"
    assert df_time["value"].gt(0).all(), f"{label}: index has non-positive values"

    derived, truth = _index_vs_truth(df_time, sd.time_land_mult, freq)
    assert len(derived) >= 3, f"{label}: only {len(derived)} common periods"

    corr = derived.corr(truth)
    assert corr > _CORR_FLOOR, (
      f"{label}: derived index does not track the generator's true land curve "
      f"(corr={corr:.4f}, floor={_CORR_FLOOR})"
    )

    rel_err = ((derived - truth).abs() / truth).dropna()
    assert rel_err.median() < _MEDIAN_REL_ERR_CEILING, (
      f"{label}: median relative error {rel_err.median():.4f} exceeds "
      f"{_MEDIAN_REL_ERR_CEILING}"
    )


def test_apply_time_adjustment():
  print("")
  settings = {
    "modeling":{
      "model_groups":{
        "residential_single_family":{}
      }
    }
  }

  # The point of the adjustment is to remove the time trend, so the per-period
  # median of price-per-sqft should come out FLATTER than it went in. Each period
  # grain is measured on its own fresh frame -- adjusting an already-adjusted
  # frame in a loop would compound the corrections.
  for period, max_ratio in (("M", 0.85), ("Q", 0.95), ("Y", 1.0)):
    sd = generate_basic(100)
    sup = SalesUniversePair(sd.df_sales, sd.df_universe)
    df = get_hydrated_sales_from_sup(sup)
    # TODO: replace with proper sales subset function
    df = df[df["sale_price"].gt(0) & df["valid_sale"].ge(1)]

    before = df.groupby("sale_year_month")["sale_price_per_impr_sqft"].median()
    cv_before = before.std() / before.mean()

    out = apply_time_adjustment(df, settings=settings, period=period,
                                write=False, verbose=True)

    assert "sale_price_time_adj" in out.columns, f"{period}: no adjusted price written"
    adj = out["sale_price_time_adj"]
    assert adj.notna().all(), f"{period}: adjusted price has NaN"
    assert adj.gt(0).all(), f"{period}: adjusted price has non-positive values"

    after = out.groupby("sale_year_month")["sale_price_time_adj_per_impr_sqft"].median()
    cv_after = after.std() / after.mean()

    # Worst observed ratios across seeds 1/42/777/1337/2024 were
    # M 0.706, Q 0.859, Y 0.932; the ceilings here leave margin.
    assert cv_after < cv_before, (
      f"{period}: adjustment did not flatten the series "
      f"(CV {cv_before:.4f} -> {cv_after:.4f})"
    )
    assert cv_after < max_ratio * cv_before, (
      f"{period}: adjustment flattened less than expected "
      f"(CV {cv_before:.4f} -> {cv_after:.4f}, ratio {cv_after / cv_before:.3f} "
      f"but wanted < {max_ratio})"
    )


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
