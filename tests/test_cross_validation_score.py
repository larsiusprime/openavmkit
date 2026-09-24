"""calc_cross_validation_score must survive NaN in X.

Real assessment data carries genuinely missing building attributes -- a few hundred
null bedroom counts in a 500k-parcel universe is ordinary. Without imputation,
sklearn raises "Input X contains NaN" inside each affected fold. cross_val_score does
not re-raise that; it records NaN for the failed folds, so the function returned NaN
and the `except ValueError` guard never fired.

That mattered because of what the caller does with the result
(``model_runner.get_variable_recommendations``)::

    cv_score = calc_cross_validation_score(X, y)
    if cv_score < best_score:      # nan < anything is False
        best_score = cv_score
        best_variables = curr_variables.copy()

so ``best_variables`` was never updated and the cross-validation refinement of the
variable set was inert on any dataset with a single NaN anywhere in X.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities.stats import calc_cross_validation_score


@pytest.fixture
def xy():
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({
        "area": rng.uniform(800, 4000, n),
        "age": rng.uniform(0, 100, n),
        "beds": rng.integers(1, 6, n).astype(float),
    })
    y = pd.Series(50_000 + 40 * X["area"] - 300 * X["age"] + rng.normal(0, 5000, n))
    return X, y


def test_score_is_finite_without_nan(xy):
    X, y = xy
    score = calc_cross_validation_score(X, y)
    assert np.isfinite(score) and score > 0


def test_a_few_nan_cells_do_not_poison_the_whole_score(xy):
    """Four NaN cells out of 600 used to return NaN for the entire score."""
    X, y = xy
    X = X.copy()
    X.loc[X.index[:4], "beds"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = calc_cross_validation_score(X, y)
    assert np.isfinite(score), "NaN in X must not make the whole score NaN"


def test_the_caller_comparison_now_updates(xy):
    """`nan < best_score` is always False -- that is how the failure stayed silent."""
    X, y = xy
    X = X.copy()
    X.loc[X.index[:4], "beds"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = calc_cross_validation_score(X, y)
    assert score < float("inf"), (
        "score must compare as less-than so get_variable_recommendations can "
        "actually update best_variables"
    )


def test_imputation_is_announced(xy):
    """Silently altering the data would be worse than the bug."""
    X, y = xy
    X = X.copy()
    X.loc[X.index[:4], "beds"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc_cross_validation_score(X, y)
    mine = [w for w in caught if "calc_cross_validation_score" in str(w.message)]
    assert len(mine) == 1, "expected exactly one imputation warning"
    assert "beds" in str(mine[0].message), "the warning should name the affected column"


def test_no_warning_when_there_is_nothing_to_impute(xy):
    X, y = xy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc_cross_validation_score(X, y)
    assert not [w for w in caught if "calc_cross_validation_score" in str(w.message)]


def test_ndarray_input_is_handled(xy):
    """X is documented as array-like, not only DataFrame."""
    X, y = xy
    X = X.copy()
    X.loc[X.index[:4], "beds"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = calc_cross_validation_score(X.to_numpy(), y.to_numpy())
    assert np.isfinite(score)


def test_all_nan_column_does_not_reintroduce_the_silent_nan(xy):
    """A column with no median at all must not put us back where we started."""
    X, y = xy
    X = X.copy()
    X["dead_field"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = calc_cross_validation_score(X, y)
    assert np.isfinite(score), "an all-NaN column must not make the score NaN"


def test_imputation_barely_moves_a_clean_score(xy):
    """Median imputation of a few cells should perturb, not transform, the result."""
    X, y = xy
    clean = calc_cross_validation_score(X, y)
    X2 = X.copy()
    X2.loc[X2.index[:4], "beds"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = calc_cross_validation_score(X2, y)
    assert imputed == pytest.approx(clean, rel=0.05)
