"""_calc_variable_recommendations with only some of the tests run (tests_to_run without "r2")."""

from types import SimpleNamespace

import pandas as pd

from openavmkit.model_runner import _calc_variable_recommendations


def _final(**cols):
    return {"final": pd.DataFrame(cols)}


def test_without_r2():
    df = _calc_variable_recommendations(
        ds=SimpleNamespace(one_hot_descendants={}),
        settings={},
        rep_results={},
        correlation_results=_final(variable=["area", "age"], corr_score=[0.6, 0.3]),
        enr_results=None,
        r2_values_results=None,
        p_values_results=_final(variable=["area"], p_value=[0.001]),
        t_values_results=None,
        vif_results=None,
    )
    assert list(df["variable"]) == ["area", "age"]
    assert list(df["weighted_score"]) == [2, 1]
