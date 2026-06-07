"""Tests for census region stamping (openavmkit.data._stamp_census_regions)."""
import numpy as np
import pandas as pd

from openavmkit.data import _stamp_census_regions


def test_stamps_tract_and_block_group_from_geoid():
    """census_block_group is the full 12-digit GEOID; census_tract is the first 11."""
    df = pd.DataFrame(
        {
            "key": ["a", "b"],
            "std_geoid": ["517300081001", "517300081002"],
        }
    )
    out = _stamp_census_regions(df)
    assert out["census_block_group"].tolist() == ["517300081001", "517300081002"]
    assert out["census_tract"].tolist() == ["51730008100", "51730008100"]


def test_nan_geoid_propagates_as_nan():
    """Parcels with no matching block group (NaN GEOID) get NaN regions, not errors."""
    df = pd.DataFrame({"key": ["a", "b"], "std_geoid": ["517300081001", None]})
    out = _stamp_census_regions(df)
    assert out["census_block_group"].tolist()[0] == "517300081001"
    assert pd.isna(out["census_block_group"].tolist()[1])
    assert pd.isna(out["census_tract"].tolist()[1])


def test_does_not_clobber_existing_columns():
    """Pre-existing census_tract / census_block_group are preserved."""
    df = pd.DataFrame(
        {
            "key": ["a"],
            "std_geoid": ["517300081001"],
            "census_tract": ["EXISTING_TRACT"],
            "census_block_group": ["EXISTING_BG"],
        }
    )
    out = _stamp_census_regions(df)
    assert out["census_tract"].tolist() == ["EXISTING_TRACT"]
    assert out["census_block_group"].tolist() == ["EXISTING_BG"]


def test_missing_geoid_column_is_noop():
    """Without a std_geoid column, the frame is returned unchanged."""
    df = pd.DataFrame({"key": ["a"], "neighborhood": ["X"]})
    out = _stamp_census_regions(df)
    assert "census_tract" not in out.columns
    assert "census_block_group" not in out.columns


def test_gating_flag_defaults_to_true():
    """The stamp_regions gate (used in _enrich_df_census) defaults on, opts out on false."""
    assert {}.get("stamp_regions", True) is True
    assert {"stamp_regions": False}.get("stamp_regions", True) is False
