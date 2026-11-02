"""The trimmed vertical-equity score must come from the trimmed rows.

``PredictionResults`` computes a trim mask and derives mse/rmse/mape/r2/slope from it,
but used to discard the mask -- so nothing downstream could compute a trimmed VEI, and
the TRIMMED benchmark table repeated the untrimmed score under a trimmed heading.

``PredictionResults`` now keeps ``trim_mask`` and ``df_trim``, and
``SingleModelResults`` exposes ``ve_test_trim`` alongside ``ve_test``.
"""
import numpy as np
import pandas as pd
import pytest

from openavmkit.modeling import PredictionResults
from openavmkit.vertical_equity_study import get_vertical_equity_scores


def _frame(n=400, seed=4, outlier_frac=0.08):
    """Sales with a block of wild ratios, so trimming has something to remove."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(80_000, 600_000, n)
    pred = price * rng.uniform(0.92, 1.08, n)
    n_out = int(n * outlier_frac)
    # push the outliers onto the CHEAP end, so trimming them shifts vertical equity
    cheap = np.argsort(price)[:n_out]
    pred[cheap] = price[cheap] * 6.0
    return pd.DataFrame({
        "key_sale": [f"k{i}" for i in range(n)],
        "sale_price": price,
        "prediction": pred,
        "valid_for_ratio_study": True,
        "valid_for_land_ratio_study": True,
    })


@pytest.fixture
def pred():
    return PredictionResults(
        "sale_price", [], "prediction", _frame(), max_trim=0.10,
        is_land_predictions=False,
    )


def test_trim_mask_and_frame_are_exposed(pred):
    assert hasattr(pred, "trim_mask"), "PredictionResults must keep the trim mask"
    assert hasattr(pred, "df_trim"), "PredictionResults must keep the trimmed frame"
    assert pred.trim_mask.dtype == bool
    assert len(pred.trim_mask) == len(pred.df)


def test_trimmed_frame_is_a_strict_subset(pred):
    assert len(pred.df_trim) < len(pred.df), "fixture did not trim anything"
    assert len(pred.df_trim) == int(pred.trim_mask.sum())
    assert set(pred.df_trim["key_sale"]).issubset(set(pred.df["key_sale"]))


def test_trimmed_frame_matches_the_rows_the_trim_stats_used(pred):
    """df_trim must select the SAME rows the scalar _trim statistics came from."""
    y = pred.df_trim["sale_price"].to_numpy()
    y_pred = pred.df_trim["prediction"].to_numpy()
    from sklearn.metrics import mean_squared_error
    assert np.isclose(mean_squared_error(y, y_pred), pred.mse_trim), (
        "df_trim does not correspond to the rows mse_trim was computed on"
    )


def test_trimmed_vei_differs_from_untrimmed(pred):
    """Trimming wild low-end ratios must move the vertical-equity score.

    If these matched, the trimmed column would be indistinguishable from the untrimmed
    one and this whole exercise would be pointless.
    """
    ve_full = get_vertical_equity_scores(pred.df, "sale_price", "prediction")
    ve_trim = get_vertical_equity_scores(pred.df_trim, "sale_price", "prediction")
    assert np.isfinite(ve_full["vei"]), "fixture too small to score untrimmed"
    assert np.isfinite(ve_trim["vei"]), "fixture too small to score trimmed"
    assert not np.isclose(ve_full["vei"], ve_trim["vei"]), (
        f"trimmed VEI {ve_trim['vei']:.4f} equals untrimmed {ve_full['vei']:.4f}; "
        f"the trimmed score is not actually being computed on trimmed rows"
    )


def test_empty_frame_still_exposes_the_attributes():
    """A PredictionResults over no usable rows must not blow up on attribute access."""
    empty = pd.DataFrame({
        "key_sale": pd.Series([], dtype=str),
        "sale_price": pd.Series([], dtype=float),
        "prediction": pd.Series([], dtype=float),
        "valid_for_ratio_study": pd.Series([], dtype=bool),
        "valid_for_land_ratio_study": pd.Series([], dtype=bool),
    })
    pr = PredictionResults("sale_price", [], "prediction", empty, max_trim=0.10,
                           is_land_predictions=False)
    assert len(pr.df_trim) == 0
    assert len(pr.trim_mask) == 0
    # the scalar trim stats are all NaN, including mape_trim, which the empty branch
    # previously forgot to set at all
    for attr in ("mse_trim", "rmse_trim", "mape_trim", "r2_trim", "slope_trim"):
        assert hasattr(pr, attr), f"{attr} missing on an empty PredictionResults"
        assert np.isnan(getattr(pr, attr))
