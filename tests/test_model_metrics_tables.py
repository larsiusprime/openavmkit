"""The benchmark metrics tables must be internally consistent.

``_model_performance_metrics`` renders two tables, UNTRIMMED and TRIMMED. Each one
carries its own count / MAPE / MSE / RMSE, and within a table RMSE must be the square
root of that same table's MSE.

It used to write the trimmed RMSE into the variable holding the untrimmed one, so the
untrimmed table reported the trimmed RMSE beside the untrimmed MSE, and the trimmed
table reported the untrimmed MSE beside the trimmed RMSE. Neither table was
self-consistent, and these are the tables people read to compare models.
"""
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openavmkit.model_runner import _model_performance_metrics


def _make_result(y_true, y_pred):
    """Minimal stand-in for SingleModelResults as this report consumes it."""
    df_test = pd.DataFrame({
        "sale_date": pd.to_datetime(["2024-01-01"] * len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    ratio_study = SimpleNamespace(
        median_ratio=1.0, mean_ratio=1.0,
        median_ratio_trim=1.0, mean_ratio_trim=1.0,
    )
    return SimpleNamespace(
        df_test=df_test,
        df_sales=df_test,
        pred_test=SimpleNamespace(y_pred=np.asarray(y_pred, dtype=float),
                                  y=np.asarray(y_true, dtype=float),
                                  ratio_study=ratio_study),
        ve_test={"vei": 1.0, "vei_significance": 0.5},
    )


def _parse_markdown_table(block):
    """Pull {model: {column: value}} out of one rendered markdown table."""
    lines = [l for l in block.strip().splitlines() if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    out = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        row = dict(zip(header[1:], cells[1:]))
        out[cells[0]] = row
    return out


def _tables(text):
    untrimmed = text.split("UNTRIMMED")[1].split("TRIMMED")[0]
    trimmed = text.split("\nTRIMMED")[1]
    return _parse_markdown_table(untrimmed), _parse_markdown_table(trimmed)


def _num(s):
    return float(s.replace(",", ""))


@pytest.fixture
def report():
    rng = np.random.default_rng(3)
    y_true = np.concatenate([rng.uniform(100_000, 400_000, 80), [1_000_000.0]])
    # a deliberate outlier so trimming actually removes something
    y_pred = y_true * rng.uniform(0.9, 1.1, len(y_true))
    y_pred[-1] = y_true[-1] * 4.0

    all_results = SimpleNamespace(
        model_results={"modelA": _make_result(y_true, y_pred)}
    )
    return _model_performance_metrics("test_group", all_results, "Test", max_trim=0.05)


def test_untrimmed_rmse_is_sqrt_of_untrimmed_mse(report):
    untrimmed, _ = _tables(report)
    row = untrimmed["modelA"]
    mse_txt, rmse = row["MSE"], _num(row["RMSE"])
    # MSE goes through fancy_format, so recover it from RMSE and check the magnitude
    # agrees rather than parsing the abbreviated string exactly.
    assert rmse > 0
    assert _magnitude_matches(mse_txt, rmse ** 2), (
        f"untrimmed RMSE {rmse:,.0f} does not square to the untrimmed MSE {mse_txt}"
    )


def test_trimmed_rmse_is_sqrt_of_trimmed_mse(report):
    _, trimmed = _tables(report)
    row = trimmed["modelA"]
    mse_txt, rmse = row["MSE"], _num(row["RMSE"])
    assert rmse > 0
    assert _magnitude_matches(mse_txt, rmse ** 2), (
        f"trimmed RMSE {rmse:,.0f} does not square to the trimmed MSE {mse_txt}"
    )


def test_trimming_actually_changes_the_two_tables(report):
    """Guard the fixture: if trimming removed nothing, the tables would agree
    trivially and the two tests above would pass even with the values crossed."""
    untrimmed, trimmed = _tables(report)
    assert untrimmed["modelA"]["count"] != trimmed["modelA"]["count"], (
        "fixture did not trim anything, so the consistency checks prove nothing"
    )
    assert _num(untrimmed["modelA"]["RMSE"]) != _num(trimmed["modelA"]["RMSE"]), (
        "trimmed and untrimmed RMSE are identical -- the crossed-values bug would "
        "not be detectable with this fixture"
    )


_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}


def _magnitude_matches(mse_text, rmse_squared, tol=0.02):
    """fancy_format abbreviates (e.g. '1.23B'); compare within a rounding tolerance."""
    m = re.match(r"^([\d.]+)\s*([KMBTQ]?)$", mse_text.strip())
    if not m:
        pytest.fail(f"could not parse MSE cell {mse_text!r}")
    value = float(m.group(1)) * _SUFFIX.get(m.group(2), 1.0)
    if value == 0:
        return rmse_squared == 0
    return abs(value - rmse_squared) / value < tol
