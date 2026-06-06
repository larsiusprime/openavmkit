"""Tests for exact path-dependent tree-SHAP on the LayeredComp model.

The fast EXTEND/UNWIND algorithm in ``openavmkit.shap_analysis`` is validated
against an independent brute-force Shapley oracle: for every feature that
appears in a (folded) tree we enumerate all coalitions and average the marginal
contribution, using the tree's cover-weighted conditional expectation. The two
must agree, and both must satisfy additivity:

    base_value + sum(shap over features) == model.predict(row)
"""

import os
import tempfile
from itertools import combinations
from math import factorial

import matplotlib
matplotlib.use("Agg")  # headless: no display needed for the beeswarm smoke test

import numpy as np
import pandas as pd

from layeredcompmodel import LayeredCompBaggingModel

from openavmkit.utilities.modeling import LayeredCompModel
from openavmkit.shap_analysis import (
    _layeredcomp_shap,
    _calc_shap,
    _expected_value,
    get_full_layeredcomp_shaps,
    plot_full_beeswarm,
)


# --- brute-force oracle ----------------------------------------------------

def _tree_feats(node, acc):
    if node.is_leaf:
        return
    acc.add(node.feat)
    for c in node.children:
        _tree_feats(c, acc)


def _cond_exp(node, xvals, S, hot_index):
    """Tree cover-weighted E[f | x_S] -- follow known features, average the rest."""
    if node.is_leaf:
        return node.value
    if node.feat in S:
        return _cond_exp(node.children[hot_index(node, xvals)], xvals, S, hot_index)
    if node.cover <= 0:
        return node.value
    return sum(
        (c.cover / node.cover) * _cond_exp(c, xvals, S, hot_index)
        for c in node.children
    )


def _brute_tree_phi(root, xvals, n_features, hot_index):
    feats = set()
    _tree_feats(root, feats)
    feats = sorted(feats)
    M = len(feats)
    phi = np.zeros(n_features)
    if M == 0:
        return phi
    for i in feats:
        others = [f for f in feats if f != i]
        total = 0.0
        for k in range(len(others) + 1):
            w = factorial(k) * factorial(M - k - 1) / factorial(M)
            for S in combinations(others, k):
                S = set(S)
                with_i = _cond_exp(root, xvals, S | {i}, hot_index)
                without = _cond_exp(root, xvals, S, hot_index)
                total += w * (with_i - without)
        phi[i] = total
    return phi


def _brute_shap(explainer, X):
    n = len(X)
    out = np.zeros((n, explainer.n_features))
    for r in range(n):
        xvals = [X[f].iloc[r] if f in X.columns else np.nan for f in explainer.feature_names]
        phi = np.zeros(explainer.n_features)
        for root in explainer.trees:
            phi += _brute_tree_phi(root, xvals, explainer.n_features, explainer._hot_index)
        out[r] = phi / (len(explainer.trees) or 1)
    return out


# --- fixtures --------------------------------------------------------------

def _fit_bag(X, y, tree_count=1, seed=0):
    bag = LayeredCompBaggingModel(
        tree_count=tree_count, sample_pct=0.8, random_state=seed, n_jobs=1
    )
    bag.fit(X, y)
    return bag


def _numeric_frame(n=80, seed=1):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "a": rng.uniform(0, 100, n),
        "b": rng.uniform(0, 50, n),
        "c": rng.uniform(-10, 10, n),
    })
    y = pd.Series(2.0 * X["a"] - 1.5 * X["b"] + 5.0 * X["c"] + rng.normal(0, 1, n))
    return X, y


def _mixed_frame(n=90, seed=2):
    rng = np.random.RandomState(seed)
    cls = rng.choice(["RES", "COM", "IND"], size=n)
    X = pd.DataFrame({
        "size": rng.uniform(800, 4000, n),
        "klass": cls,
        "age": rng.uniform(0, 120, n),
    })
    base = {"RES": 100.0, "COM": 250.0, "IND": 180.0}
    y = pd.Series(
        [base[c] for c in cls] + 0.05 * X["size"] - 0.4 * X["age"] + rng.normal(0, 3, n)
    )
    return X, y


# --- tests -----------------------------------------------------------------

def test_fold_reconstructs_prediction_numeric():
    X, y = _numeric_frame()
    bag = _fit_bag(X, y, tree_count=3)
    expl = _layeredcomp_shap(LayeredCompModel(bag))

    shap_vals = expl.shap_values(X)
    recon = expl.expected_value + shap_vals.sum(axis=1)
    np.testing.assert_allclose(recon, bag.predict(X), atol=1e-6)


def test_matches_brute_force_oracle_numeric():
    X, y = _numeric_frame()
    bag = _fit_bag(X, y, tree_count=2)
    expl = _layeredcomp_shap(LayeredCompModel(bag))

    fast = expl.shap_values(X.head(15))
    brute = _brute_shap(expl, X.head(15))
    np.testing.assert_allclose(fast, brute, atol=1e-8)


def test_matches_brute_force_oracle_mixed_categorical():
    X, y = _mixed_frame()
    bag = _fit_bag(X, y, tree_count=2, seed=7)
    expl = _layeredcomp_shap(LayeredCompModel(bag))

    fast = expl.shap_values(X.head(15))
    brute = _brute_shap(expl, X.head(15))
    np.testing.assert_allclose(fast, brute, atol=1e-8)

    # additivity against the real ensemble prediction
    recon = expl.expected_value + fast.sum(axis=1)
    np.testing.assert_allclose(recon, bag.predict(X.head(15)), atol=1e-6)


def test_nan_early_stop_is_exact():
    """A NaN at a numeric split makes the model stop early; SHAP must stay exact."""
    X, y = _mixed_frame()
    bag = _fit_bag(X, y, tree_count=2, seed=7)
    expl = _layeredcomp_shap(LayeredCompModel(bag))

    X_nan = X.head(12).copy()
    X_nan.loc[X_nan.index[:6], "size"] = np.nan
    X_nan.loc[X_nan.index[3:9], "age"] = np.nan

    fast = expl.shap_values(X_nan)
    brute = _brute_shap(expl, X_nan)
    np.testing.assert_allclose(fast, brute, atol=1e-8)

    # base + sum(shap) must still reconstruct the model's (early-stopped) output
    recon = expl.expected_value + fast.sum(axis=1)
    np.testing.assert_allclose(recon, bag.predict(X_nan), atol=1e-6)


def test_expected_value_is_mean_of_tree_bases():
    X, y = _numeric_frame()
    bag = _fit_bag(X, y, tree_count=4)
    expl = _layeredcomp_shap(LayeredCompModel(bag))

    manual = float(np.mean([_expected_value(t) for t in expl.trees]))
    assert abs(expl.expected_value - manual) < 1e-12


def test_calc_shap_and_beeswarm_render():
    """The path _quick_shap uses: _calc_shap -> a plottable beeswarm Explanation."""
    X, y = _mixed_frame()
    bag = _fit_bag(X, y, tree_count=2, seed=5)
    model = LayeredCompModel(bag)

    shaps = _calc_shap(model, X, X.head(20))
    assert shaps is not None
    assert shaps.values.shape == (20, len(bag.feature_names_in_))
    # beeswarm needs a numeric `data` matrix aligned to the feature names
    assert shaps.data.shape == shaps.values.shape
    assert list(shaps.feature_names) == list(bag.feature_names_in_)

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "beeswarm.png")
        plot_full_beeswarm(shaps, title="lcomp", save_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0


def test_numba_and_pure_python_paths_agree(monkeypatch):
    """The JIT kernel and the pure-Python fallback must produce identical SHAP."""
    import openavmkit.shap_analysis as sa

    X, y = _mixed_frame()
    bag = _fit_bag(X, y, tree_count=3, seed=11)
    expl = _layeredcomp_shap(LayeredCompModel(bag))

    fast = expl.shap_values(X.head(25))                  # numba path (if available)
    monkeypatch.setattr(sa, "_HAVE_NUMBA", False)
    slow = expl.shap_values(X.head(25))                  # forced pure-Python path
    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_get_full_layeredcomp_shaps_all_subsets():
    X, y = _mixed_frame()
    bag = _fit_bag(X, y, tree_count=2, seed=3)
    model = LayeredCompModel(bag)

    shaps = get_full_layeredcomp_shaps(
        model, X.head(20), X.head(8), X.head(10), X.head(30)
    )
    for subset in ("train", "test", "sales", "universe"):
        expl = shaps[subset]
        assert expl is not None
        assert expl.values.shape[1] == len(bag.feature_names_in_)
        assert list(expl.feature_names) == list(bag.feature_names_in_)
