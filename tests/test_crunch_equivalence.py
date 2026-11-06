"""_crunch works on NumPy (one percentile call, searchsorted for the bins) for speed: these hold it to exactly what
the pandas version it replaced gave, on every kind of column. _crunch_reference below is that version, verbatim."""

import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities import clustering
from openavmkit.utilities.clustering import _crunch, make_clusters


# The pandas _crunch as it was (openavmkit master, 2026-11-06), the reference the NumPy one must match.
def _crunch_reference(_df, field, min_count):
    """Crunch a field into a smaller number of bins, each with at least min_count
    elements.

    Dynamically adapts to find the best number of bins to use.
    """
    crunch_levels = [
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # 5 clusters
        (0.0, 0.25, 0.75, 1.0),  # 3 clusters (high, medium, low)
        (0.0, 0.5, 1.0),  # 2 clusters (high & low)
    ]
    good_series = None
    too_small = False

    # Cache the column to avoid repeated attribute lookups
    field_values = _df[field]

    # Boolean fast path
    if pd.api.types.is_bool_dtype(field_values):
        bool_series = field_values.astype(int)
        if bool_series.value_counts().min() < min_count:
            return None
        return bool_series

    # Precompute all unique quantiles required by all crunch levels
    unique_qs = {q for level in crunch_levels for q in level}
    quantile_values = {q: field_values.quantile(q) for q in unique_qs}

    def _value_in_list(value, lst):
        for v in lst:
            delta = abs(value - v)
            if delta < 1e-6:
                return True
        return False

    # Iterate over each crunch level
    for crunch_level in crunch_levels:
        test_bins = []
        for q in crunch_level:
            bin_val = quantile_values[q]
            # Only add non-NaN and new bin values to test_bins
            if not pd.isna(bin_val) and not _value_in_list(bin_val, test_bins):
                test_bins.append(bin_val)

        if len(test_bins) > 1:
            labels = test_bins[1:]
            series = pd.cut(
                field_values, bins=test_bins, labels=labels, include_lowest=True
            )
        else:
            # if we only have one bin, this crunch is pointless
            too_small = True
            break

        if series.value_counts().min() < min_count:
            # If any bin at this granularity is too small, fall through to the next
            # (coarser) crunch level rather than abandoning the search: `break` here
            # made the 3-bin and 2-bin levels unreachable under every input, so the
            # "dynamically adapts" promise in the docstring never held.
            too_small = True
            continue
        else:
            # if all bins are big enough, return this series
            return series

    return None


def _same(a, b):
  if a is None or b is None:
    assert a is None and b is None, (a, b)
  else:
    pd.testing.assert_series_equal(a, b, check_exact=True)


def _columns(seed: int):
  """Columns of the kinds _crunch meets: sizes, ages, grades, with ties, blanks, nullable kinds, a constant."""
  rng = np.random.default_rng(seed)
  n = int(rng.choice([1, 2, 7, 15, 16, 31, 60, 200, 1000]))
  out = {
    "float": rng.lognormal(7.5, 0.5, n),
    "int": rng.integers(0, 3_000, n),
    "grades": rng.integers(1, 6, n),  # quality 1-5: heavy ties
    "few": rng.choice([1.0, 2.0], n),
    "constant": np.full(n, 1950.0),
    "negative": rng.normal(0, 1, n),
    "rounded": np.round(rng.normal(100, 0.000001, n), 9),  # edges within 1e-6 of each other
  }
  blanks = rng.random(n) < 0.2
  with_nan = out["float"].copy()
  with_nan[blanks] = np.nan
  out["float_blanks"] = with_nan
  out["all_blank"] = np.full(n, np.nan)
  cols = {k: pd.Series(v, name=k) for k, v in out.items()}
  cols["Int64"] = pd.Series(pd.array(np.where(blanks, None, out["int"]), dtype="Int64"), name="Int64")
  cols["Float64"] = pd.Series(pd.array(np.where(blanks, None, out["float"]), dtype="Float64"), name="Float64")
  cols["UInt16"] = pd.Series(pd.array(out["grades"], dtype="UInt16"), name="UInt16")
  cols["float32"] = pd.Series(out["float"].astype(np.float32), name="float32")  # the pandas path
  cols["bool"] = pd.Series(rng.random(n) < 0.5, name="bool")
  cols["dates"] = pd.Series(pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 2000, n), unit="D"), name="dates")
  cols["huge_int"] = pd.Series(rng.integers(2**60, 2**61, n), name="huge_int")  # beyond float64: the pandas path
  # an index that isn't 0..n-1, as a cluster's rows are
  idx = pd.Index(rng.permutation(n * 3)[:n])
  return {k: v.set_axis(idx) for k, v in cols.items()}


@pytest.mark.parametrize("seed", range(60))
def test_crunch_matches_the_pandas_version_on_every_kind_of_column(seed):
  for name, s in _columns(seed).items():
    df = s.to_frame()
    for min_count in (1, 2, 5, 15, 50):
      try:
        want = _crunch_reference(df, name, min_count)
      except Exception as e:  # the reference raising: the new one raises the same
        with pytest.raises(type(e)):
          _crunch(df, name, min_count)
        continue
      _same(_crunch(df, name, min_count), want)


def test_infinite_values_still_go_the_pandas_way():
  s = pd.Series([1.0, 2.0, np.inf, np.inf, np.inf, 3.0] * 10, name="f")
  for m in (1, 5, 30):
    try:
      want = _crunch_reference(s.to_frame(), "f", m)
    except ValueError:
      with pytest.raises(ValueError):
        _crunch(s.to_frame(), "f", m)
      continue
    _same(_crunch(s.to_frame(), "f", m), want)


def test_make_clusters_gives_the_same_clusters_as_with_the_pandas_crunch(monkeypatch):
  rng = np.random.default_rng(1)
  n = 5_000
  df = pd.DataFrame(
    {
      "neighborhood": rng.choice([f"N{i}" for i in range(40)], n),
      "is_vacant": rng.random(n) < 0.1,
      "bldg_type": rng.choice(["ranch", "colonial", "cape", "split"], n),
      "land_area_sqft": rng.lognormal(9, 0.6, n),
      "bldg_area_finished_sqft": np.where(rng.random(n) < 0.1, np.nan, rng.lognormal(7.5, 0.4, n)),
      "bldg_quality_num": pd.array(rng.integers(1, 6, n), dtype="Int64"),
      "bldg_age_years": rng.integers(0, 120, n).astype(float),
      "bldg_condition_num": rng.integers(1, 6, n).astype(float),
    },
    index=pd.Index(rng.permutation(n * 2)[:n]),
  )
  args = dict(field_location="neighborhood", fields_categorical=["bldg_type"], fields_numeric=None, min_cluster_size=15)
  got = make_clusters(df, **args)
  monkeypatch.setattr(clustering, "_crunch", _crunch_reference)
  want = make_clusters(df, **args)
  pd.testing.assert_series_equal(got[0], want[0])
  assert got[1] == want[1]
  pd.testing.assert_series_equal(got[2], want[2])
  assert got[0].nunique() > 200  # it did split: the test isn't vacuous
