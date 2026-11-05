"""Tests for census region stamping (openavmkit.data._stamp_census_regions)."""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from openavmkit.data import _stamp_census_regions
from openavmkit.utilities.census import match_to_census_blockgroups


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


# ---------------------------------------------------------------------------
# Block-group matching: the boundary tie-break
# ---------------------------------------------------------------------------

# An already-PROJECTED crs on purpose. Given a geographic crs the matcher
# reprojects to an equal-area crs first, and that reprojection nudges a centroid
# off a shared boundary, so the multi-match these tests are about never happens.
_CRS = "EPSG:3857"


def _boundary_fixture():
    small = Polygon([(1000, 0), (1000, 1000), (2000, 1000), (2000, 0)])
    big = Polygon([(-9000, -9000), (-9000, 10000), (1000, 10000), (1000, -9000)])
    # BIG first, so "whichever row the join emitted first" and "smallest polygon"
    # give different answers
    census = gpd.GeoDataFrame(
        {"std_geoid": ["BIG", "SMALL"]}, geometry=[big, small], crs=_CRS
    )
    parcels = gpd.GeoDataFrame(
        {"key": ["onEdge", "inSmall", "inBig", "outside"]},
        geometry=[
            # centroid lands exactly on the shared edge -> intersects both
            Polygon([(900, 400), (1100, 400), (1100, 600), (900, 600)]),
            Polygon([(1400, 400), (1600, 400), (1600, 600), (1400, 600)]),
            Polygon([(-5100, 400), (-4900, 400), (-4900, 600), (-5100, 600)]),
            Polygon([(50000, 50000), (50200, 50000), (50200, 50200), (50000, 50200)]),
        ],
        crs=_CRS,
    )
    return parcels, census


def test_blockgroup_match_does_not_duplicate_boundary_parcels():
    """A centroid on a shared edge matches two block groups; only one row may survive.

    The tie-break used to compute area on the joined frame's geometry -- the centroid
    POINTS, whose area is always 0.0 -- and then select with .loc on an index that has
    duplicate labels for exactly those parcels, so the boundary parcel came back twice.
    """
    parcels, census = _boundary_fixture()
    out = match_to_census_blockgroups(parcels.copy(), census.copy(), join_type="left")
    assert len(out) == len(parcels)
    assert out["key"].is_unique


def test_blockgroup_match_picks_the_smallest_containing_block_group():
    """Of the block groups a boundary centroid touches, the smallest polygon wins."""
    parcels, census = _boundary_fixture()
    out = match_to_census_blockgroups(parcels.copy(), census.copy(), join_type="left")
    got = dict(zip(out["key"], out["std_geoid"]))
    assert got["onEdge"] == "SMALL"
    # unambiguous parcels are unaffected
    assert got["inSmall"] == "SMALL"
    assert got["inBig"] == "BIG"


def test_blockgroup_match_keeps_unmatched_parcels():
    """A parcel outside every block group survives the left join with a NaN geoid."""
    parcels, census = _boundary_fixture()
    out = match_to_census_blockgroups(parcels.copy(), census.copy(), join_type="left")
    got = dict(zip(out["key"], out["std_geoid"]))
    assert pd.isna(got["outside"])
    # original polygon geometry is restored, and the helper column is cleaned up
    assert out.geometry.geom_type.eq("Polygon").all()
    assert "_census_area" not in out.columns


# ---------------------------------------------------------------------------
# The ACS's codes for figures it doesn't give; a block group's figure from its tract; the nearest block group
# ---------------------------------------------------------------------------

from openavmkit.utilities.census import fill_from_tracts, null_jam_values


def test_every_acs_jam_value_is_missing():
    """Each of the ACS's codes (-666666666 and the others) is a missing figure, not a hugely negative one."""
    df = pd.DataFrame(
        {
            "median_g_rent": [1500, -666666666, -999999999, -888888888, -555555555, -333333333, -222222222],
            "NAME": ["a"] * 7,
        }
    )
    out = null_jam_values(df, ["median_g_rent", "not_there"])
    assert out["median_g_rent"].tolist()[0] == 1500
    assert out["median_g_rent"].iloc[1:].isna().all()
    assert out["NAME"].tolist() == ["a"] * 7  # other columns untouched


def test_a_withheld_block_group_figure_is_its_tracts():
    """A block group the ACS gives no rent for takes its tract's; one with its own keeps it."""
    bg = pd.DataFrame(
        {
            "std_geoid": ["517300081001", "517300081002", "517300099001"],
            "median_g_rent": [1500.0, np.nan, np.nan],
        }
    )
    tracts = pd.DataFrame({"tract_geoid": ["51730008100", "51730009900"], "median_g_rent": [1400.0, np.nan]})
    out = fill_from_tracts(bg, tracts, ["median_g_rent", "median_income"])
    assert out["median_g_rent"].tolist()[:2] == [1500.0, 1400.0]
    assert pd.isna(out["median_g_rent"].tolist()[2])  # no tract figure either: still missing


def test_a_parcel_in_a_gap_takes_the_nearest_block_group_when_asked():
    """With nearest_m, a centroid just outside every block group takes the nearest within it; far away, none."""
    parcels, census = _boundary_fixture()
    near = Polygon([(2050, 400), (2150, 400), (2150, 600), (2050, 600)])  # 100 m east of SMALL
    parcels = pd.concat(
        [parcels, gpd.GeoDataFrame({"key": ["nearGap"]}, geometry=[near], crs=_CRS)], ignore_index=True
    )
    out = match_to_census_blockgroups(parcels.copy(), census.copy(), join_type="left", nearest_m=250)
    got = dict(zip(out["key"], out["std_geoid"]))
    assert got["nearGap"] == "SMALL" and pd.isna(got["outside"])
    plain = match_to_census_blockgroups(parcels.copy(), census.copy(), join_type="left")
    assert pd.isna(dict(zip(plain["key"], plain["std_geoid"]))["nearGap"])  # off by default


def test_tract_figures_come_keyed_by_tract_with_jam_values_missing():
    """The tracts' figures, from the ACS by tract, keyed by their 11-digit GEOID; a code for a withheld figure is
    missing."""
    from types import SimpleNamespace

    from openavmkit.utilities.census import CensusService

    rows = [
        {"NAME": "t1", "B19013_001E": 70000, "B01003_001E": 900, "B25064_001E": 1400, "B25058_001E": -666666666,
         "state": "51", "county": "685", "tract": "010100"},
    ]
    service = object.__new__(CensusService)
    service.census_client = SimpleNamespace(acs5=SimpleNamespace(state_county_tract=lambda *a, **k: rows))
    out = service.get_census_tract_data("51685", 2022, {})
    assert out["tract_geoid"].tolist() == ["51685010100"]
    assert out["median_g_rent"].tolist() == [1400] and pd.isna(out["median_c_rent"].iloc[0])
