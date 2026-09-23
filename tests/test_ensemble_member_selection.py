"""Ensemble members must be ranked on the held-out metric, not the in-sample one.

``_optimize_ensemble_iteration`` repeatedly evicts the worst member. Ranking on
``utility_sales_lookback`` -- the error over the sales lookback window -- scores each
model with a model that trained on those same rows (under cross-validation, Phase 2 is
refit on every trainable sale, so it is wholly in-sample). That systematically favours
whichever member overfit the sales hardest.

``utility_test`` is the leakage-free alternative living on the same results object:
under CV it is the stitched out-of-fold frame covering 100% of sales, otherwise the
holdout split.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from openavmkit.model_runner import _worst_ensemble_member


def _results(**by_name):
    """by_name: model -> (utility_test, utility_sales_lookback)"""
    return SimpleNamespace(model_results={
        name: SimpleNamespace(utility_test=ut, utility_sales_lookback=usl)
        for name, (ut, usl) in by_name.items()
    })


def test_evicts_the_worst_on_the_held_out_metric():
    """The two metrics disagree here; the held-out one must decide.

    `memoriser` looks best in-sample (1.0) but is worst held out (90.0) -- exactly the
    overfit member the old ranking would have kept.
    """
    all_results = _results(
        generaliser=(20.0, 18.0),
        middling=(30.0, 25.0),
        memoriser=(90.0, 1.0),
    )
    members = ["generaliser", "middling", "memoriser"]
    assert _worst_ensemble_member(members, all_results) == "memoriser"

    # and for contrast: ranking on the in-sample metric would have evicted the
    # generaliser instead, which is the behaviour this fix removes
    by_lookback = max(members, key=lambda m: all_results.model_results[m].utility_sales_lookback)
    assert by_lookback == "middling"
    assert by_lookback != "memoriser"


def test_picks_the_single_worst_when_metrics_agree():
    all_results = _results(a=(10.0, 10.0), b=(50.0, 50.0), c=(20.0, 20.0))
    assert _worst_ensemble_member(["a", "b", "c"], all_results) == "b"


def test_ignores_members_with_no_results():
    all_results = _results(a=(10.0, 1.0), b=(40.0, 2.0))
    # "ghost" has no entry in model_results and must simply be skipped
    assert _worst_ensemble_member(["a", "ghost", "b"], all_results) == "b"


def test_returns_none_when_nothing_is_scored():
    all_results = SimpleNamespace(model_results={})
    assert _worst_ensemble_member(["a", "b"], all_results) is None


def test_nan_scores_do_not_hijack_the_eviction():
    """A member whose held-out score is NaN must not be treated as the worst.

    NaN never compares greater, so it is skipped and a genuinely-scored member wins.
    """
    all_results = _results(a=(np.nan, 1.0), b=(15.0, 2.0), c=(35.0, 3.0))
    assert _worst_ensemble_member(["a", "b", "c"], all_results) == "c"


def test_single_member_is_returned_but_caller_guards_removal():
    """With one member left the helper still names it; the caller refuses to empty
    the ensemble (``len(ensemble_list) > 1``)."""
    all_results = _results(only=(12.0, 3.0))
    assert _worst_ensemble_member(["only"], all_results) == "only"
