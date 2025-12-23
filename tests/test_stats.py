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