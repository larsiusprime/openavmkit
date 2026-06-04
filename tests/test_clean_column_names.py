"""clean_column_names must produce unique labels even when distinct source names clean
to the same string (many punctuation chars all map to '_'). Duplicate labels otherwise
break the one-hot reindex/alignment in modeling."""
import pandas as pd

from openavmkit.utilities.data import clean_column_names


def test_punctuation_collision_is_deduplicated():
    # "neighborhood_FOO-BAR" and "neighborhood_FOO BAR" both clean to the same name.
    df = pd.DataFrame(
        [[1, 2, 3]],
        columns=["neighborhood_FOO-BAR", "neighborhood_FOO BAR", "land_area_sqft"],
    )
    out = clean_column_names(df)
    assert out.columns.is_unique
    assert list(out.columns) == ["neighborhood_FOO_BAR", "neighborhood_FOO_BAR_1", "land_area_sqft"]


def test_no_collision_unchanged_up_to_cleaning():
    df = pd.DataFrame([[1, 2]], columns=["a b", "c"])
    out = clean_column_names(df)
    assert list(out.columns) == ["a_b", "c"]
    assert out.columns.is_unique


def test_deterministic_across_identical_frames():
    cols = ["x-y", "x y", "x.y", "z"]
    a = clean_column_names(pd.DataFrame([[0, 0, 0, 0]], columns=cols))
    b = clean_column_names(pd.DataFrame([[1, 1, 1, 1]], columns=cols))
    assert list(a.columns) == list(b.columns)  # same names => splits stay aligned
    assert a.columns.is_unique
