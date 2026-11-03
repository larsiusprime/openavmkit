"""calc_elastic_net_regularization leaves out a column with no values at all, rather than failing on it."""

import warnings

import numpy as np
import pandas as pd

from openavmkit.utilities.stats import calc_elastic_net_regularization


def test_an_empty_column_is_left_out():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"area": rng.uniform(800, 4000, 100), "days_to_reno": np.nan})
    y = pd.Series(50_000 + 40 * X["area"] + rng.normal(0, 5000, 100))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = calc_elastic_net_regularization(X, y)
    assert list(out["initial"]["variable"]) == ["area"]
    assert any("days_to_reno" in str(w.message) for w in caught)
