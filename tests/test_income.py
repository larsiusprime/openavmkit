import numpy as np
import pandas as pd

from openavmkit.income import calculate_noi, calculate_noi_growth, calculate_cap_rate_growth, derive_irr_df, \
  derive_prices, derive_noi_growth
from openavmkit.projection import project_trend
from openavmkit.synthetic.basic import generate_income_sales
from openavmkit.utilities.assertions import dfs_are_equal, lists_are_equal
from IPython.display import display

def test_noi():
  print("")
  m = 1000000
  prices = [1.00*m, 1.25*m, 1.50*m, 2.00*m]
  cap_rates = [0.05, 2/30, 0.08, 0.10]

  results = {
    "cap_rates": cap_rates
  }

  for price in prices:
    values = []
    for cap_rate in cap_rates:
      noi = calculate_noi(price, cap_rate)
      values.append(int(round(noi)))
      results[f"{int(price)}"] = values

  expected = {
    "cap_rates": cap_rates,
    "1000000": [ 50000,  66667,  80000, 100000],
    "1250000": [ 62500,  83333, 100000, 125000],
    "1500000": [ 75000, 100000, 120000, 150000],
    "2000000": [100000, 133333, 160000, 200000]
  }
  df_expected = pd.DataFrame(data=expected)
  df_actual = pd.DataFrame(data=results)

  assert dfs_are_equal(df_expected, df_actual)


def test_noi_growth():
  cap_rate_growth = np.array([
    0.01,
    0.00,
   -0.01,
    0.01,
    0.01,
    0.01
  ])

  price_growth = np.array([
    0.01,
    0.01,
    0.01,
    0.01,
    0.00,
    -0.01
  ])

  noi_growth = np.array([
    0.02010,
    0.01000,
    -0.0001,
    0.02010,
    0.01000,
    -0.0001,
  ])

  noi_growth_derived = calculate_noi_growth(price_growth, cap_rate_growth)

  assert lists_are_equal(noi_growth.tolist(), noi_growth_derived.tolist())

  cap_rate_growth_derived = calculate_cap_rate_growth(price_growth, noi_growth)

  assert lists_are_equal(cap_rate_growth.tolist(), cap_rate_growth_derived.tolist())


def test_implied_irr():

  m = 1000000
  d5y = "2020-01-01"
  d7y = "2018-01-01"
  d10y = "2015-01-01"
  dx = "2025-01-01"

  data = {
    "key":         ["1", "2", "3"],
    "entry_price": [1*m, 1*m, 1*m],
    "exit_price":  [2*m, 2*m, 2*m],
    "entry_date":  [d5y, d7y, d10y],
    "exit_date":   [dx, dx, dx]
  }
  df = pd.DataFrame(data=data)

  expected = {
    "key":         ["1", "2", "3"],
    "entry_price": [1*m, 1*m, 1*m],
    "exit_price":  [2*m, 2*m, 2*m],
    "entry_date":  [d5y, d7y, d10y],
    "exit_date":   [dx, dx, dx],
    "holding_period": [5.0, 7.0, 10.0],
    "implied_irr": [0.195618, 0.153147, 0.124912],
    "entry_year": [2020.0, 2018.0, 2015.0],
    "entry_cap_rate": [0.05, 0.05, 0.05],
    "entry_noi": [50000.0, 50000.0, 50000.0]
  }
  df_expected = pd.DataFrame(data=expected)

  entry_cap_rate = 0.05

  noi_growth_rate = 0.07
  cap_growth_rate = 0.00

  hist_noi_growths = {}
  hist_cap_rates = {}

  i = 0
  cap_rate = entry_cap_rate
  for year in range(2014, 2025+1):
    hist_noi_growths[year] = noi_growth_rate
    hist_cap_rates[year] = cap_rate
    cap_rate = cap_rate * (1 + cap_growth_rate)
    i += 1

  df_result = derive_irr_df(
    df=df,
    hist_cap_rates=hist_cap_rates,
    hist_noi_growths=hist_noi_growths
  )

  assert dfs_are_equal(df_result, df_expected)


def test_implied_growth():
  m = 1000000

  data = {
    "key":         ["1", "2", "3"],
    "entry_price": [1*m, 1*m, 1*m],
    "entry_cap_rate": [0.05, 0.05, 0.05],
    "target_irr": [0.195618, 0.153147, 0.124912],
    "holding_period": [5, 7, 10]
  }
  df = pd.DataFrame(data=data)

  # derive an exit cap rate that matches an NOI growth rate of 7%
  df["exit_noi"] = 50000 * (1.07 ** df["holding_period"])
  df["exit_cap_rate"] = df["exit_noi"] / 2000000

  df["noi_growth"] = df.apply(lambda x: derive_noi_growth(
    x["target_irr"],
    x["exit_cap_rate"],
    x["entry_price"],
    x["entry_cap_rate"],
    x["holding_period"]
  ), axis=1)

  # We expect an implied NOI growth of 7%
  diff = df["noi_growth"] - 0.07
  assert (np.abs(diff) < 1e-6).all()


def test_asking_price():

  # Attempt to reverse the results from above

  m = 1000000
  noi_growth = 0.07
  noi_mult = 1 + noi_growth

  exit_price = 2*m

  data = {
    "key":           ["1", "2", "3"],
    "entry_noi":     [50000.0, 50000.0, 50000.0],
    "entry_cap_rate":[0.05, 0.05, 0.05],
    "holding_period":[   5,    7,   10],
    "target_irr":    [0.195618, 0.153147, 0.124912]
  }
  df = pd.DataFrame(data=data)

  df["exit_noi"] = df["entry_noi"] * (noi_mult ** df["holding_period"])
  df["exit_cap_rate"] = df["exit_noi"] / exit_price

  expected = {
    "key":        ["1", "2", "3"],
    "entry_price":[1*m, 1*m, 1*m],
    "exit_price": [2*m, 2*m, 2*m]
  }
  df_expected = pd.DataFrame(data=expected)

  prices = df.apply(lambda x: derive_prices(
    x["target_irr"],
    x["exit_cap_rate"],
    x["entry_noi"],
    noi_growth,
    x["holding_period"]
  ), axis=1)

  df["entry_price"] = prices.apply(lambda x: x[0])
  df["exit_price"] = prices.apply(lambda x: x[1])

  df_diff = df[["entry_price", "exit_price"]] - df_expected[["entry_price", "exit_price"]]

  df_diff = df_diff / df_expected[["entry_price", "exit_price"]]

  # Final prices should be off by no more than 0.001%:
  assert (np.abs(df_diff) < 1e-5).all().all()


def test_stuff():
  # empirically observed median market preferences
  target_irr = 0.15
  holding_period = 7

  # cap rates derived from observation & research
  hist_cap_rates = {
    2008: 0.05,
    2009: 0.055,
    2010: 0.06,
    2011: 0.05,
    2012: 0.045,
    2013: 0.048,
    2014: 0.045,
    2015: 0.041,
    2016: 0.040,
    2017: 0.035,
    2018: 0.058,
    2019: 0.050,
    2020: 0.048,
    2021: 0.045,
    2022: 0.041,
    2023: 0.040,
    2024: 0.035,
    2025: 0.058,
  }

  # market transactions
  sale_years = [
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025
  ]
  sale_prices = [
    1000000,
    1022806,
    1036320,
    1047245,
    1070542,
    1093929,
    1125418,
    1138439,
    1156979,
    1188347,
    1214314
  ]

  data = {
    "sale_year": sale_years,
    "sale_price": sale_prices
  }
  df = pd.DataFrame(data=data)
  df["entry_cap_rate"] = df["sale_year"].apply(lambda x: hist_cap_rates[x])

  # exit cap rate (use cap rate as of valuation date)
  exit_cap_rate = hist_cap_rates[2025]

  expected_growth = [
    0.151148,
    0.155785,
    0.181030,
    0.086957,
    0.114218,
    0.121768,
    0.133752,
    0.151148,
    0.155785,
    0.181030,
    0.086957
  ]

  df["noi_growth"] = df.apply(lambda x: derive_noi_growth(
    target_irr=target_irr,
    exit_cap_rate=exit_cap_rate,
    entry_price=x["sale_price"],
    entry_cap_rate=x["entry_cap_rate"],
    holding_period=holding_period
  ), axis=1)

  diff = df["noi_growth"] - expected_growth
  assert (np.abs(diff) < 1e-6).all()


def test_blah():

  stuff = generate_income_sales(
    count_per_year=1,
    start_year=2010,
    end_year=2025
  )

  pd.set_option('display.max_columns', None)
  df = stuff["transactions"][["sale_price", "sale_year"]]
  caps = stuff["cap_rates"]

  df_med = df.groupby("sale_year")["sale_price"].agg(["median"]).reset_index()
  first_year = df_med["sale_year"].min()
  sale_price_denominator = df_med[df_med["sale_year"].eq(first_year)].iloc[0]["median"]
  sale_price_growth = df_med["median"] / sale_price_denominator

  cap_rates = np.array([caps[year] for year in df_med["sale_year"]])
  cap_rate_denominator = cap_rates[0]
  cap_rate_growth = cap_rates / cap_rate_denominator

  noi_growth = calculate_noi_growth(sale_price_growth, cap_rate_growth)

  # df["cap_rate"] = df["sale_year"].apply(lambda x: caps[x])
  # df["entry_noi"] = df["sale_price"] * df["cap_rate"]
  # df["exit_cap_rate"] =
  #
  # df["guess_price"] = df.apply(lambda x: derive_prices(
  #   target_irr = 0.15,
  #   exit_cap_rate = df["exit_cap_rate"],
  #   entry_noi = df["entry_noi"],
  #   noi_growth = noi_growth,
  #   holding_period = 7
  # ), axis=1)
