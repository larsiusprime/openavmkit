"""Tests for residual categorical/boolean auto-fill (openavmkit.cleaning)."""
import numpy as np
import pandas as pd

from openavmkit.cleaning import _fill_unknown_values


def _settings(cat_fields):
    return {
        "field_classification": {
            "important": {},
            "other": {"categorical": cat_fields},
        }
    }


def _fill(df, cat_fields):
    from openavmkit.utilities.settings import get_fields_categorical

    settings = _settings(cat_fields)
    # guard the fixture: if the settings shape ever drifts, these tests would
    # silently pass by filling nothing
    resolved = get_fields_categorical(settings, df, include_boolean=False) or []
    assert set(cat_fields).issubset(set(resolved)), (
        f"fixture did not register categorical fields: got {resolved}"
    )
    return _fill_unknown_values(df, settings)


def test_missing_categoricals_become_UNKNOWN_not_the_string_nan():
    """astype("str") renders NaN as the literal "nan", which made the fillna a no-op.

    Models then received a "nan" category instead of "UNKNOWN".
    """
    df = pd.DataFrame({
        "key": ["a", "b", "c", "d"],
        "style": ["ranch", np.nan, "colonial", None],
    })
    out = _fill(df, ["style"])
    assert out["style"].tolist() == ["ranch", "UNKNOWN", "colonial", "UNKNOWN"]
    assert "nan" not in out["style"].tolist()
    assert "None" not in out["style"].tolist()


def test_fill_works_on_categorical_dtype():
    """Filling a category dtype with an unseen label needs the object round-trip."""
    df = pd.DataFrame({
        "key": ["a", "b"],
        "style": pd.Series(["ranch", None], dtype="category"),
    })
    out = _fill(df, ["style"])
    assert out["style"].tolist() == ["ranch", "UNKNOWN"]


def test_existing_values_are_untouched():
    """Only missing values change; real categories keep their labels and dtype str."""
    df = pd.DataFrame({
        "key": ["a", "b", "c"],
        "style": ["ranch", "ranch", "colonial"],
    })
    out = _fill(df, ["style"])
    assert out["style"].tolist() == ["ranch", "ranch", "colonial"]


def test_genuine_nan_string_is_left_alone():
    """A column that really contains the text "nan" is data, not a missing value."""
    df = pd.DataFrame({
        "key": ["a", "b"],
        "style": ["nan", np.nan],
    })
    out = _fill(df, ["style"])
    # the real string survives; only the true NaN becomes UNKNOWN
    assert out["style"].tolist() == ["nan", "UNKNOWN"]
