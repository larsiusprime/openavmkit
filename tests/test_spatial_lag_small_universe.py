"""Spatial lag must degrade on thin samples, not crash.

Unlike a NaN in `ind_vars` -- which is a missing fill rule and therefore the user's
problem (see test_linear_model_nan_guard.py) -- a model group with fewer parcels than
`k` is a fact about the jurisdiction. `k` is a library default (5), and rural,
commercial or heavily-filtered vacant groups legitimately fall below it.
advanced_settings.md 4.8 documents the intended behaviour: "If a model group has fewer
training sales than `k+1`, the sale-price lag is skipped for that group rather than
failing."

The sale-price path already honours that. The universe-field path and the shared
`calc_spatial_lag` primitive did not: cKDTree pads a short neighbour list with
``index == len(coords_train)``, so the value lookup raised
``IndexError: index N is out of bounds``.

Both now write a defined value instead. The column is always produced -- omitting it
would surface later as a KeyError deep inside the model runner, taking down every
other model group with it.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities.data import calc_spatial_lag


def _frames(n_sample, n_univ=6, seed=0):
    rng = np.random.default_rng(seed)
    sample = pd.DataFrame({
        "latitude": rng.uniform(29.7, 29.8, n_sample),
        "longitude": rng.uniform(-95.4, -95.3, n_sample),
        "sale_price": rng.uniform(100_000, 500_000, n_sample),
    })
    universe = pd.DataFrame({
        "latitude": rng.uniform(29.7, 29.8, n_univ),
        "longitude": rng.uniform(-95.4, -95.3, n_univ),
    })
    return sample, universe


@pytest.mark.parametrize("n_sample", [4, 3, 2, 1])
def test_sample_smaller_than_k_does_not_raise(n_sample):
    """This is the IndexError the guard exists to prevent."""
    sample, universe = _frames(n_sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = calc_spatial_lag(sample, universe, ["sale_price"], neighbors=5)
    assert "spatial_lag_sale_price" in out.columns
    assert len(out) == len(universe)


@pytest.mark.parametrize("n_sample", [4, 2, 1, 0])
def test_degraded_column_is_finite(n_sample):
    """A NaN lag column would be fatal to the linear engines downstream."""
    sample, universe = _frames(n_sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = calc_spatial_lag(sample, universe, ["sale_price"], neighbors=5)
    assert np.isfinite(out["spatial_lag_sale_price"]).all()


def test_degraded_value_is_the_sample_median():
    sample, universe = _frames(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = calc_spatial_lag(sample, universe, ["sale_price"], neighbors=5)
    assert out["spatial_lag_sale_price"].nunique() == 1
    assert out["spatial_lag_sale_price"].iloc[0] == pytest.approx(
        sample["sale_price"].median()
    )


def test_empty_sample_falls_back_to_zero():
    """With no sample there is no median; zeros match the sale-price lag's handling."""
    sample, universe = _frames(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = calc_spatial_lag(sample, universe, ["sale_price"], neighbors=5)
    assert (out["spatial_lag_sale_price"] == 0).all()


def test_degradation_is_announced():
    """Silently substituting a constant for a spatial surface must be visible."""
    sample, universe = _frames(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc_spatial_lag(sample, universe, ["sale_price"], neighbors=5)
    mine = [w for w in caught if "calc_spatial_lag" in str(w.message)]
    assert len(mine) == 1
    assert "sale_price" in str(mine[0].message)


def test_adequate_sample_is_untouched():
    """The guard must not fire, or alter results, when there are enough neighbours."""
    sample, universe = _frames(40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = calc_spatial_lag(sample, universe, ["sale_price"], neighbors=5)
    assert not [w for w in caught if "calc_spatial_lag" in str(w.message)]
    lag = out["spatial_lag_sale_price"]
    assert np.isfinite(lag).all()
    # a real surface varies; the degraded path would be constant
    assert lag.nunique() > 1


def test_self_exclusion_needs_one_extra_neighbour():
    """With exclude_self_in_sample the first neighbour is dropped, so k+1 are needed."""
    sample, universe = _frames(5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = calc_spatial_lag(sample, sample.drop(columns=["sale_price"]),
                               ["sale_price"], neighbors=5,
                               exclude_self_in_sample=True)
    assert np.isfinite(out["spatial_lag_sale_price"]).all()
    assert [w for w in caught if "calc_spatial_lag" in str(w.message)], (
        "5 samples with k=5 and self-exclusion leaves only 4 usable neighbours "
        "and should have degraded"
    )
