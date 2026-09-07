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

def test_cross_validation_score_with_nan():
  """A feature column with genuine missing values must still yield a finite score.

  Without imputation, sklearn raises inside individual CV folds and the mean of the
  surviving scores is nan -- which silently disables the caller's `cv_score <
  best_score` refinement, since `nan < x` is always False.
  """

  import pandas as pd
  from openavmkit.utilities.stats import calc_cross_validation_score

  rng = np.random.default_rng(0)
  n = 2000
  X = pd.DataFrame({
    "bldg_area_finished_sqft": rng.normal(1500, 400, n),
    "bldg_rooms_bed": rng.integers(1, 6, n).astype(float),
    "latitude": rng.normal(39.95, 0.05, n),
  })
  y = pd.Series(
    2.0 * X["bldg_area_finished_sqft"]
    + 5000 * X["bldg_rooms_bed"]
    + rng.normal(0, 1000, n)
  )

  score_clean = calc_cross_validation_score(X, y)
  assert np.isfinite(score_clean)

  # Genuine missing building attributes, spread across folds.
  X_nan = X.copy()
  X_nan.loc[rng.choice(n, 200, replace=False), "bldg_rooms_bed"] = np.nan
  score_nan = calc_cross_validation_score(X_nan, y)
  assert np.isfinite(score_nan), "NaN in a feature column must not produce a nan score"

  # An all-NaN column has no median to impute from; it must be dropped, not propagated.
  X_dead = X.copy()
  X_dead["never_recorded"] = np.nan
  assert np.isfinite(calc_cross_validation_score(X_dead, y))

  # The numpy path is supported by the signature too.
  assert np.isfinite(calc_cross_validation_score(X_nan.to_numpy(), y.to_numpy()))
