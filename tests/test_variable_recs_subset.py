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


def test_without_correlation_with_a_report():
    """tests_to_run of VIF alone: the report's correlation table says N/A rather than failing."""
    from openavmkit.reports import MarkdownReport

    report = MarkdownReport("variables")
    vif = {
        "initial": pd.DataFrame({"variable": ["area", "age"], "vif": [1.5, 12.0]}),
        "final": pd.DataFrame({"variable": ["area"], "vif": [1.2]}),
    }
    df = _calc_variable_recommendations(
        ds=SimpleNamespace(one_hot_descendants={}),
        settings={},
        rep_results={},
        correlation_results=None,
        enr_results=None,
        r2_values_results=None,
        p_values_results=None,
        t_values_results=None,
        vif_results=vif,
        report=report,
    )
    assert list(df["variable"]) == ["area"]
    assert report.get_var("table_corr_final") == "N/A"
