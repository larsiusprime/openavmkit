"""The study set's figures, on every prediction (the default) or on the holdout's (study_set_predictions)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openavmkit.modeling import SingleModelResults
from openavmkit.utilities.settings import get_study_set_predictions


LOOK_BACK = int(pd.date_range("2024-01-01", periods=60, freq="12D").year.isin([2025]).sum())


def _model(mode):
    settings = {"modeling": {"metadata": {"valuation_date": "2026-01-01"}}}
    if mode:
        settings["analysis"] = {"ratio_study": {"study_set_predictions": mode}}
    rng = np.random.default_rng(1)
    price = rng.uniform(100_000, 300_000, 60)
    dates = pd.date_range("2024-01-01", periods=60, freq="12D")  # about half of them in 2025, the look-back
    frame = lambda pred: pd.DataFrame(  # noqa: E731
        {"key_sale": [str(i) for i in range(60)], "sale_date": dates, "sale_price": price,
         "prediction": pred, "valid_for_ratio_study": True}
    )
    # In sample: every sale near its price. On the holdout: off by up to a fifth.
    in_sample = frame(price * rng.uniform(0.99, 1.01, 60))
    holdout = frame(price * rng.uniform(0.8, 1.2, 60))
    m = SimpleNamespace(
        ds=SimpleNamespace(settings=settings, model_group="sf"),
        df_test=holdout,
        dep_var_test="sale_price",
        ind_vars=[],
        field_prediction="prediction",
        is_land_predictions=False,
    )
    SingleModelResults._set_study_set(m, in_sample)
    return m


def test_the_study_set_on_every_prediction_by_default():
    m = _model(None)
    assert len(m.df_sales_lookback) == LOOK_BACK and m.df_sales_lookback.sale_date.dt.year.eq(2025).all()
    assert m.pred_sales_lookback.ratio_study.cod < 1.5
    assert m.utility_study is None  # the benchmark keeps its training utility


def test_the_study_set_on_the_holdout():
    m = _model("holdout")
    assert len(m.df_sales_lookback) == LOOK_BACK  # the same sales
    assert m.pred_sales_lookback.ratio_study.cod > 5  # judged on predictions made without them
    assert m.utility_study == pytest.approx(m.pred_sales_lookback.mape * 100)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        get_study_set_predictions({"analysis": {"ratio_study": {"study_set_predictions": "test"}}})
