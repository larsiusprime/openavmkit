"""The benchmark's frames carry every trimmed figure there is, while its printed tables stay as they were."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openavmkit.model_runner as mr
from openavmkit.modeling import PredictionResults
from openavmkit.vertical_equity_study import get_vertical_equity_scores


def _model():
    rng = np.random.default_rng(2)
    price = rng.uniform(100_000, 300_000, 80)
    pred = price * rng.uniform(0.8, 1.2, 80)
    pred[:4] *= 3  # a few outlying ratios, for the trim to take
    df = pd.DataFrame({"key_sale": [str(i) for i in range(80)], "sale_price": price, "prediction": pred,
                       "valid_for_ratio_study": True})
    pr = PredictionResults("sale_price", [], "prediction", df, 0.1, False)
    ve = get_vertical_equity_scores(pr.df, "sale_price", "prediction")
    ve_trim = get_vertical_equity_scores(pr.df_trim, "sale_price", "prediction")
    return SimpleNamespace(
        pred_test=pr, pred_sales_lookback=pr, ve_test=ve, ve_test_trim=ve_trim, ve_sales_lookback=ve,
        ve_sales_lookback_trim=ve_trim, utility_test=1.0, utility_study=1.0, utility_train=1.0,
        df_universe=pd.DataFrame(index=range(100)), chd=1.0, timing=SimpleNamespace(results={}),
    )


def test_the_frames_carry_the_trimmed_level_and_vei(monkeypatch):
    m = _model()
    monkeypatch.setattr(mr, "_get_post_valuation_smr", lambda r: r)
    b = mr._calc_benchmark({"mra": m})
    rs = m.pred_test.ratio_study
    assert rs.count_trim < rs.count  # the trim took something
    for df, ve in ((b.df_stats_test, m.ve_test_trim), (b.df_stats_full, m.ve_sales_lookback_trim)):
        row = df.loc["mra"]
        assert row["median_ratio_trim"] == pytest.approx(rs.median_ratio_trim)
        assert row["vei_trim"] == pytest.approx(ve["vei"])
        assert row["vei_sig_trim"] == pytest.approx(ve["vei_significance"])
    assert b.df_stats_test.loc["mra", "vei_trim"] != pytest.approx(b.df_stats_test.loc["mra", "vei"])
    # The printed tables are as they were.
    text = b.print()
    for c in ("median_ratio_trim", "vei_trim", "vei_sig_trim"):
        assert c not in text
    assert "cod_trim" in text
