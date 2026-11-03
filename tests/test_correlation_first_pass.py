"""calc_correlations' opt-in first pass: no correlation with the target, dropped before the combined scores."""

import numpy as np
import pandas as pd

from openavmkit.utilities.stats import calc_correlations


def _x():
    rng = np.random.default_rng(3)
    n = 400
    area = rng.uniform(800, 4000, n)
    noise = rng.normal(0, 1, n)  # nothing to do with price
    age = rng.uniform(0, 100, n)
    price = 40 * area - 300 * age + rng.normal(0, 5000, n)
    return pd.DataFrame({"price": price, "area": area, "age": age, "noise": noise})


def test_without_it_nothing_changes():
    out = calc_correlations(_x(), 0.1)
    assert out["weak"] == [] and "noise" in set(out["initial"]["variable"])
    assert list(out["matrix"].columns) == ["price", "area", "age", "noise"]


def test_first_pass_drops_no_correlation():
    out = calc_correlations(_x(), 0.1, min_strength=0.05)
    assert out["weak"] == ["noise"]
    assert "noise" not in set(out["initial"]["variable"]) and "area" in set(out["final"]["variable"])


def test_nothing_left():
    out = calc_correlations(_x()[["price", "noise"]], 0.1, min_strength=0.05)
    assert out["weak"] == ["noise"] and len(out["initial"]) == 0


def test_without_overlap_the_score_is_the_correlation_with_the_target_alone():
    """overlap=False: a near-duplicate of a strong variable keeps its own score (VIF judges overlap); only what's
    weakly related to the target goes."""
    rng = np.random.default_rng(1)
    n = 500
    a = rng.normal(size=n)
    twin = a + rng.normal(scale=0.05, size=n)  # nearly a
    noise = rng.normal(size=n)
    y = 2 * a + rng.normal(scale=0.5, size=n)
    x = pd.DataFrame({"y": y, "a": a, "twin": twin, "noise": noise})
    out = calc_correlations(x, 0.1, overlap=False)
    kept = set(out["final"]["variable"])
    assert {"a", "twin"} <= kept and "noise" not in kept
    assert (out["initial"]["corr_clarity"] == 1.0).all()
