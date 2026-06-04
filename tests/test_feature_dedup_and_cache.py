"""Tests for the OSM same-named-feature collapse (no duplicate distance columns,
no information lost) and the cache duplicate-column guard."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openavmkit.data import _collapse_features_by_name
from openavmkit.utilities.cache import write_cached_df


def test_collapse_features_by_name_unions_same_name():
    # Two disjoint polygons share name "A" (e.g. one golf course OSM-mapped as two
    # elements); "B" is distinct.
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "A", "B"], "area": [1.0, 1.0, 1.0]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(0, 2, 1, 3)],
        crs="EPSG:4326",
    )
    out = _collapse_features_by_name(gdf, "name")
    # One row per name (no duplicate that would become a duplicate column)...
    assert sorted(out["name"]) == ["A", "B"]
    # ...and "A" is the UNION of both elements (information preserved, not dropped):
    a_geom = out.loc[out["name"] == "A"].geometry.iloc[0]
    assert a_geom.covers(box(0, 0, 1, 1).centroid)
    assert a_geom.covers(box(2, 0, 3, 1).centroid)
    assert a_geom.area == pytest.approx(2.0)


def test_collapse_features_by_name_noop_when_unique():
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B"]}, geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs="EPSG:4326"
    )
    out = _collapse_features_by_name(gdf, "name")
    assert len(out) == 2


def test_write_cached_df_rejects_duplicate_columns_clearly():
    # A frame with two columns named "dup" (what an un-deduped feature loop produced).
    df = pd.DataFrame([[1, 10, 99], [2, 20, 88]], columns=["key", "dup", "dup"])
    with pytest.raises(ValueError, match="duplicate column"):
        write_cached_df(df, df, "test_dup_guard", "key")
