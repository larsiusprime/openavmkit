"""Tests for the DuckDB-backed Overture buildings fetch (ENG-3033).

`_stream_building_dfs` reads Overture's global buildings theme with the bbox-overlap
predicate pushed down to Parquet row-group statistics, so peak memory is bounded to
the matching rows instead of scanning the whole theme. These tests mock DuckDB so they
are fully network-free; the real S3 fetch and its bit-for-bit equivalence to the prior
PyArrow path are exercised by the avm-python-service scorecard.
"""
import geopandas as gpd
import pandas as pd
import shapely.geometry as sgeom
import shapely.wkb as swkb
from unittest.mock import MagicMock, patch

from openavmkit.utilities.overture import OvertureService

# A small built-up parcel near Santa Cruz, CA (real lon/lat so UTM estimation works).
_POLY = sgeom.box(-122.0263, 36.9741, -122.0260, 36.9744)
_BBOX = (-122.0283, 36.9721, -122.0243, 36.9761)


def _make_service():
    with patch.object(
        OvertureService, "_resolve_latest_buildings_prefix",
        return_value="release/2099-01-01.0/theme=buildings/type=building/",
    ), patch("openavmkit.utilities.overture.fs.S3FileSystem", MagicMock()):
        return OvertureService({"overture": {"enabled": True}})


def _result_df():
    return pd.DataFrame({
        "id": ["bldg-1"],
        "geometry": [swkb.dumps(_POLY)],
        "bbox": [{"xmin": -122.0263, "xmax": -122.0260,
                  "ymin": 36.9741, "ymax": 36.9744}],
        "height": [6.5],
        "num_floors": [2],
        "sources": [[{"property": "/properties/height", "confidence": 0.9}]],
    })


def _fake_connection(describe_cols, result_df, captured):
    con = MagicMock()

    def _execute(sql, params=None):
        captured.append((sql, params))
        res = MagicMock()
        if sql.strip().upper().startswith("DESCRIBE"):
            res.fetchall.return_value = [(c,) for c in describe_cols]
        # fetch_df_chunk streams: yield the frame once, then an empty frame.
        res.fetch_df_chunk.side_effect = [result_df, result_df.iloc[0:0]]
        return res

    con.execute.side_effect = _execute
    return con


def test_stream_pushes_bbox_predicate_and_drops_unavailable_columns():
    svc = _make_service()
    captured = []
    # est_height / num_floors_underground / subtype / class are absent from this
    # release and must be dropped from the projection (mirrors prior PyArrow behavior).
    describe_cols = ["id", "geometry", "bbox", "height", "num_floors", "sources"]
    con = _fake_connection(describe_cols, _result_df(), captured)
    with patch("openavmkit.utilities.overture.duckdb.connect", return_value=con):
        chunks = list(svc._stream_building_dfs(
            _BBOX, OvertureService.DEFAULT_COLUMNS.copy()))
    assert len(chunks) == 1 and len(chunks[0]) == 1
    select_calls = [c for c in captured if c[0].strip().upper().startswith("SELECT")]
    assert len(select_calls) == 1
    sql, params = select_calls[0]
    assert "bbox.xmin < ?" in sql and "bbox.xmax > ?" in sql
    assert "bbox.ymin < ?" in sql and "bbox.ymax > ?" in sql
    xmin, ymin, xmax, ymax = _BBOX
    assert params == [xmax, xmin, ymax, ymin]
    assert '"est_height"' not in sql and '"class"' not in sql
    assert '"id"' in sql and '"geometry"' in sql


def test_get_buildings_builds_geodataframe_with_footprint():
    svc = _make_service()
    describe_cols = ["id", "geometry", "bbox", "height", "num_floors", "sources"]
    con = _fake_connection(describe_cols, _result_df(), [])
    with patch("openavmkit.utilities.overture.duckdb.connect", return_value=con):
        gdf = svc.get_buildings(_BBOX, use_cache=False)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].equals(_POLY)
    assert gdf["bldg_area_footprint_sqft"].iloc[0] > 0
    assert gdf["height_m_best"].iloc[0] == 6.5
    assert gdf["floors_best"].iloc[0] == 2


def test_building_batches_yields_geodataframes_from_stream():
    svc = _make_service()
    describe_cols = ["id", "geometry", "bbox", "height", "num_floors", "sources"]
    con = _fake_connection(describe_cols, _result_df(), [])
    with patch("openavmkit.utilities.overture.duckdb.connect", return_value=con):
        frames = list(svc._building_batches(_BBOX))
    assert len(frames) == 1
    assert isinstance(frames[0], gpd.GeoDataFrame)
    assert frames[0].geometry.iloc[0].equals(_POLY)
