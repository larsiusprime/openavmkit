"""Guards for the CatBoost performance/correctness fixes.

Background (diagnosed on us-va-petersburgcity and us-nc-guilford single-family):

- The tuner optimized CatBoost's MAPE *loss*, which converges very slowly on price targets,
  so trials rarely early-stopped and the search drifted to Depthwise/Lossguide depth-10
  trees -- the slowest corner of the space (~80% of Guilford's tuning time).
- ``_tune_catboost`` returns Optuna ``best_params`` (searched keys only), and ``run_catboost``
  built the final model from that alone, so the shipped model trained on CatBoost defaults
  rather than the tuned configuration.
- CatBoost's ``cv()`` always creates ``./catboost_info``; parallel CV fold workers raced on it
  (``FileExistsError``) in a fresh locality directory.
- ``_calc_shap`` (the ``do_shaps`` summary path) ran shap's generic TreeExplainer on CatBoost
  instead of CatBoost's native SHAP -- ~25 minutes on Guilford.
"""

import os
import time

import numpy as np
import optuna
import pandas as pd
import catboost
import pytest
from catboost import CatBoostRegressor, Pool

from openavmkit.modeling import _catboost_final_params
from openavmkit.shap_analysis import _calc_shap
from openavmkit.tuning import _CATBOOST_FIXED_PARAMS, _tune_catboost
from openavmkit.utilities.modeling import CatBoostModel, TreeBasedCategoricalData

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _toy(n=300, seed=0, with_cat=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "land_area_sqft": rng.uniform(3000, 20000, n),
            "bldg_area_finished_sqft": rng.uniform(800, 4000, n),
            "bldg_age_years": rng.integers(0, 120, n).astype(float),
        }
    )
    if with_cat:
        X["neighborhood"] = pd.Categorical(rng.choice(["a", "b", "c", "d"], n))
    y = (50000.0 + 40.0 * X["bldg_area_finished_sqft"] + 2.0 * X["land_area_sqft"]
         + rng.normal(0, 10000, n)).clip(lower=1000.0)
    return X, y


def test_tuner_searches_symmetric_trees_only_and_leaves_no_catboost_info(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _toy()
    seen = []
    # The tuner reads cv() from catboost when it runs (CatBoost is an optional extra).
    real_cv = catboost.cv

    def spy(pool, params, **kw):
        seen.append(dict(params))
        return real_cv(pool, params, **kw)

    monkeypatch.setattr(catboost, "cv", spy)
    best = _tune_catboost(X, y, n_trials=3, n_splits=2, random_state=42,
                          cat_vars=["neighborhood"], use_gpu=False)

    assert "grow_policy" not in best and "max_leaves" not in best
    assert seen, "cv() was never called"
    for p in seen:
        for k, v in _CATBOOST_FIXED_PARAMS.items():
            assert p[k] == v, (k, p[k])
        # cv() must write to a private dir, never the shared working directory
        assert os.path.abspath(p["train_dir"]) != os.path.abspath("catboost_info")
    assert not os.path.exists(tmp_path / "catboost_info")
    assert not os.path.exists(seen[0]["train_dir"]), "private train_dir was not cleaned up"


def test_final_params_carry_the_tuned_configuration():
    tuned = {"iterations": 50, "learning_rate": 0.1, "depth": 4, "border_count": 32,
             "random_strength": 1.0, "reg_lambda": 1.0, "bagging_temperature": 3.0}
    params = _catboost_final_params(tuned, seed=7)
    for k, v in _CATBOOST_FIXED_PARAMS.items():
        assert params[k] == v
    assert params["random_seed"] == 7 and params["verbose"] is False

    X, y = _toy(n=200)
    m = CatBoostRegressor(**params, allow_writing_files=False)
    m.fit(Pool(X, y, cat_features=["neighborhood"]))
    ap = m.get_all_params()
    # Before the fix the final model fell back to CatBoost defaults (MVS bootstrap, which
    # silently discards the tuned bagging_temperature).
    assert ap["bootstrap_type"] == "Bayesian"
    assert ap["bagging_temperature"] == pytest.approx(3.0)
    assert ap["grow_policy"] == "SymmetricTree"
    assert ap["max_ctr_complexity"] == 1


def test_final_params_keep_saved_keys_on_conflict():
    # Saved params from an older search (e.g. a Lossguide pick) must still form a valid config.
    params = _catboost_final_params({"grow_policy": "Lossguide", "max_leaves": 40, "depth": 6})
    assert params["grow_policy"] == "Lossguide" and params["max_leaves"] == 40


def test_calc_shap_uses_native_catboost_shap():
    X, y = _toy(n=400)
    reg = CatBoostRegressor(iterations=200, depth=10, grow_policy="Depthwise", verbose=False,
                            random_seed=1, allow_writing_files=False)
    reg.fit(Pool(X, y, cat_features=["neighborhood"]))
    model = CatBoostModel(
        regressor=reg,
        cat_data=TreeBasedCategoricalData.from_training_data(X, categorical_cols=["neighborhood"]),
    )
    Xe = X.head(100)

    t = time.perf_counter()
    expl = _calc_shap(model, X, Xe)
    dt = time.perf_counter() - t

    assert expl.values.shape == (len(Xe), X.shape[1])
    assert expl.data.shape == expl.values.shape
    native = reg.get_feature_importance(Pool(Xe, cat_features=["neighborhood"]),
                                        type="ShapValues", shap_calc_type="Approximate")
    np.testing.assert_allclose(expl.values, native[:, :-1])
    # Native SHAP on 100 rows of a deep model is well under a second; shap's generic
    # TreeExplainer on the same model takes several seconds.
    assert dt < 5, f"_calc_shap took {dt:.1f}s -- is it back on the generic TreeExplainer path?"
