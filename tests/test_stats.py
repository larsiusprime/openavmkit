import numpy as np

from openavmkit.utilities.assertions import objects_are_equal
from openavmkit.utilities.stats import calc_cod_bootstrap
from openavmkit.utilities.timing import TimingData


def test_cod_bootstrap():

  np.random.seed(777)

  # generate an array of random values, 10000 items:
  values = np.random.normal(1, 0.20, 1000000)

  iterations = [10, 100, 1000, 10000, 100000, 1000000]

  t = TimingData()

  results = {}
  expected = {
    '10': {'low': 15.977491270587132, 'med': 15.952782920658212, 'hi': 15.998632312530669},
    '100': {'low': 15.977491270587132, 'med': 15.952782920658212, 'hi': 15.998632312530669},
    '1000': {'low': 15.977491270587132, 'med': 15.952782920658212, 'hi': 15.998632312530669},
    '10000': {'low': 15.977491270587132, 'med': 15.952782920658212, 'hi': 15.998632312530669},
    '100000': {'low': 15.977491270587132, 'med': 15.952782920658212, 'hi': 15.998632312530669},
    '1000000': {'low': 15.977491270587132, 'med': 15.952782920658212, 'hi': 15.998632312530669}
  }

  for iteration in iterations:
    low, med, hi = calc_cod_bootstrap(values, iterations=10)
    results[str(iteration)] = {"low": low, "med": med, "hi": hi}

  print("")
  print("results=")
  print(results)
  print("expected=")
  print(expected)
  print("***")

  assert objects_are_equal(results, expected)

def test_calc_prb_returns_slope():
  import numpy as np
  import statsmodels.api as sm
  from openavmkit.utilities.stats import calc_prb

  n = 500
  truth = np.linspace(100_000, 1_000_000, n)
  rank = np.linspace(0.0, 1.0, n)
  preds = truth * (0.9 + 0.2 * rank)  # ratios rise with parcel value

  prb, lo, hi = calc_prb(preds, truth)

  # fit the identically transformed regression by hand
  ratios = preds / truth
  med = np.median(ratios)
  left = (ratios - med) / med
  right = sm.add_constant(np.log2(preds / med + truth), has_constant="add")
  slope = sm.OLS(left, right).fit().params[1]

  assert abs(prb - slope) < 1e-9
  assert prb > 0
  assert lo <= prb <= hi

  # unbiased control: a flat multiplier of ground truth gives PRB near zero
  prb0, _, _ = calc_prb(truth * 0.95, truth)
  assert abs(prb0) < 1e-6
