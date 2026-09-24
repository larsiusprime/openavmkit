"""Linear models must refuse NaN loudly, not impute it away.

Doctrine (advanced_settings.md, `data.process.fill`): "every field in a model's
`ind_vars` must be NaN-free, or the linear models crash... whenever you add an
enrichment numeric to `ind_vars`, add it to `data.process.fill` too."

A partially-populated ind_var means a missing fill rule, which is the user's
data-preparation decision. The library must not guess a value for it. What it *should*
do is say which field, how bad, and what to change -- statsmodels' bare
``MissingDataError("exog contains inf or nans")`` does none of that.

`drop_all_nan_ind_vars` remains the one exception: a column with NO finite values
carries no signal and is dropped with a warning.
"""
import numpy as np
import pandas as pd
import pytest

from openavmkit.modeling import DataSplit, run_mra, run_multi_mra


def _dataset(nan_rows=(1, 3, 13), field="bldg_parking_spaces", all_nan=False):
    n = 20
    keys = [str(i) for i in range(n)]
    df = pd.DataFrame({
        "key": keys,
        "key_sale": keys,
        "bldg_area_finished_sqft": np.array([1000, 1500, 2000, 2500, 3000] * 4, float),
        "land_area_sqft": np.array([10000, 20000, 30000, 40000, 50000] * 4, float),
        "location_A": np.array([1, 0] * 10, float),
        "neighborhood": ["n1", "n2"] * 10,
        "model_group": ["a"] * n,
    })
    df["sale_price"] = (20.0 * df["bldg_area_finished_sqft"]
                        + 1.5 * df["land_area_sqft"] + 10000.0)
    df["valid_sale"] = True
    df["vacant_sale"] = False
    df["is_vacant"] = False
    df["valid_for_ratio_study"] = True
    df["sale_date"] = pd.to_datetime("2025-01-01")
    df["sale_age_days"] = 0

    ind_vars = ["bldg_area_finished_sqft", "land_area_sqft", "location_A"]
    if field is not None:
        df[field] = np.nan if all_nan else 1.0
        if not all_nan:
            df.loc[list(nan_rows), field] = np.nan
        ind_vars = ind_vars + [field]

    df_sales = df.copy()
    df_universe = df[["key", "is_vacant", "neighborhood"] + ind_vars].copy()
    test_keys = [str(i) for i in range(6)]
    train_keys = [k for k in keys if k not in test_keys]
    return df_sales, df_universe, ind_vars, test_keys, train_keys


def _split(df_sales, df_universe, ind_vars, test_keys, train_keys):
    return DataSplit("", df_sales, df_universe, "a", {}, "sale_price", "sale_price",
                     ind_vars, ["neighborhood"], {}, test_keys, train_keys)


def test_mra_raises_on_partially_nan_ind_var():
    ds = _split(*_dataset())
    with pytest.raises(ValueError) as exc:
        run_mra(ds, intercept=True)
    assert "mra" in str(exc.value)


def test_multi_mra_raises_on_partially_nan_ind_var(tmp_path):
    ds = _split(*_dataset())
    with pytest.raises(ValueError) as exc:
        run_multi_mra(ds, str(tmp_path), ["neighborhood"],
                      optimize_vars=False, intercept=True)
    assert "multi_mra" in str(exc.value)


def test_the_message_is_actually_actionable():
    """The whole point of raising rather than imputing is that the user can act."""
    ds = _split(*_dataset())
    with pytest.raises(ValueError) as exc:
        run_mra(ds, intercept=True)
    msg = str(exc.value)

    # names the offending field
    assert "bldg_parking_spaces" in msg
    # quantifies the damage
    assert "/14" in msg or "/20" in msg
    # names the remedy and where it lives
    assert "data.process.fill" in msg
    assert "median" in msg
    # says re-running the modeling notebook alone is not enough
    assert "clean" in msg.lower()


def test_it_does_not_silently_impute():
    """Guard against the tempting wrong fix."""
    ds = _split(*_dataset())
    with pytest.raises(ValueError):
        run_mra(ds, intercept=True)


def test_infinities_are_caught_too():
    """OLS rejects inf as well as NaN; so must the guard."""
    df_sales, df_universe, ind_vars, test_keys, train_keys = _dataset(nan_rows=())
    for frame in (df_sales, df_universe):
        frame.loc[2, "bldg_parking_spaces"] = np.inf
    ds = _split(df_sales, df_universe, ind_vars, test_keys, train_keys)
    with pytest.raises(ValueError) as exc:
        run_mra(ds, intercept=True)
    assert "bldg_parking_spaces" in str(exc.value)


def test_clean_data_still_fits():
    """The guard must not fire on fully-populated ind_vars."""
    ds = _split(*_dataset(field=None))
    res = run_mra(ds, intercept=True)
    assert np.isfinite(np.asarray(res.pred_univ, dtype=float)).all()


def test_all_nan_column_is_still_dropped_not_raised():
    """A column with NO finite values carries no signal -- drop_all_nan_ind_vars
    handles it with a warning, and that behaviour must survive this guard."""
    ds = _split(*_dataset(all_nan=True))
    with pytest.warns(UserWarning, match="no finite values"):
        res = run_mra(ds, intercept=True)
    assert np.isfinite(np.asarray(res.pred_univ, dtype=float)).all()


def test_every_offending_field_is_listed_not_just_the_first():
    """A user fixing one field at a time, re-running the clean stage each round,
    would be a miserable loop."""
    df_sales, df_universe, ind_vars, test_keys, train_keys = _dataset()
    for frame in (df_sales, df_universe):
        frame["census_median_income"] = 50000.0
        frame.loc[[7], "census_median_income"] = np.nan
    ind_vars = ind_vars + ["census_median_income"]
    ds = _split(df_sales, df_universe, ind_vars, test_keys, train_keys)

    with pytest.raises(ValueError) as exc:
        run_mra(ds, intercept=True)
    msg = str(exc.value)
    assert "bldg_parking_spaces" in msg
    assert "census_median_income" in msg
