"""Smoke tests that each tuner entry point actually executes a trial end to end.

These exist because of a real regression: a commit whose stated purpose was renaming
"rolling origin" CV to "k-fold" (`f2e2b805`) accidentally deleted the `def evaluate(...)`
header out of both `_tune_xgboost` and `_tune_lightgbm` while rewriting the line beneath
it. The orphaned body was left inside `suggest` — unreachable dead code after the `return`
in the XGBoost case, and live-but-discarded in the LightGBM case, where it also made
`suggest` return `None`. Both tuners then hit `NameError: name 'evaluate' is not defined`
on the `_run_batched(...)` call and could not run at all.

Nothing caught it. `test_tuning_resume.py` covers the journal/resume machinery and the
inner-split helpers, but never calls a tuner's entry point, so the breakage survived two
commits and eight days before being repaired by `b199f5ba`.

The guard is deliberately shallow: one trial, two inner folds, tiny numeric-only data. It
asserts nothing about tuning *quality* — only that the plumbing from entry point through
`suggest` / `evaluate` / `_run_batched` to a completed Optuna trial is intact. Optuna's
``best_params`` raises when no trial completes, so a non-empty return is itself proof that
a trial ran and produced a finite objective value.

Features are numeric-only on purpose. Categorical encoding differs per engine and is
covered elsewhere; mixing it in here would trade flakiness for coverage this file is not
trying to provide.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from openavmkit.tuning import (
    _tune_xgboost,
    _tune_lightgbm,
    _tune_catboost,
    _tune_ngboost,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost and LightGBM split sampling (`suggest`) from evaluation (`evaluate`) so trials can
# be scored in parallel; CatBoost and NGBoost use a single Optuna `objective`. Both shapes are
# covered so a future refactor of either one cannot silently stop running.
TUNERS = {
    "xgboost": _tune_xgboost,
    "lightgbm": _tune_lightgbm,
    "catboost": _tune_catboost,
    "ngboost": _tune_ngboost,
}


def _toy_xy(n=120, seed=0):
    """A small, well-conditioned regression problem — strictly positive y, since every
    tuner here optimizes MAPE and would divide by zero on a zero target.

    ``y`` stays a Series, matching what the tuners are really handed (``ds.y_train``): the
    XGBoost and LightGBM CV helpers index it with ``.iloc``, so a bare ndarray raises
    ``AttributeError`` there even though NGBoost coerces with ``np.asarray`` and accepts one.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "land_area_sqft": rng.uniform(3000, 20000, n),
            "bldg_area_finished_sqft": rng.uniform(800, 4000, n),
            "bldg_age_years": rng.integers(0, 120, n).astype(float),
        }
    )
    y = (
        50000.0
        + 40.0 * X["bldg_area_finished_sqft"]
        + 2.0 * X["land_area_sqft"]
        + rng.normal(0, 10000, n)
    ).clip(lower=1000.0)
    return X, y


@pytest.mark.parametrize("name", list(TUNERS))
def test_tuner_entry_point_completes_a_trial(name):
    tuner = TUNERS[name]
    X, y = _toy_xy()

    best = tuner(
        X,
        y,
        n_trials=1,
        n_splits=2,
        random_state=42,
        cat_vars=[],
        verbose=False,
        storage_path=None,
    )

    # Non-empty => at least one trial COMPLETED; Optuna raises on best_params otherwise.
    assert isinstance(best, dict), f"{name} tuner returned {type(best).__name__}, not a dict"
    assert best, f"{name} tuner returned no hyperparameters"

    # Every tuner samples a learning rate, so its absence means `suggest`/`objective` was
    # bypassed rather than merely unlucky.
    assert "learning_rate" in best, f"{name} tuner did not sample learning_rate: {sorted(best)}"
    assert np.isfinite(best["learning_rate"])
