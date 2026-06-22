"""Tests for the DuckDB-backed Overture buildings fetch (ENG-3033).

`_stream_building_dfs` reads Overture's global buildings theme with the bbox-overlap
predicate pushed down to Parquet row-group statistics, so peak memory is bounded to
the matching rows instead of scanning the whole theme. The tests are network-free:
most mock DuckDB, and one uses a local Parquet file to exercise DuckDB's real struct
predicate handling.
"""
import geopandas as gpd
import pandas as pd
import pytest
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


def _building_result_df():
    return pd.DataFrame({
        "id": ["bldg-1"],
        "geometry": [swkb.dumps(_POLY)],
        "bbox": [{"xmin": -122.0263, "xmax": -122.0260,
                  "ymin": 36.9741, "ymax": 36.9744}],
        "height": [6.5],
        "num_floors": [2],
        "sources": [[{"property": "/properties/height", "confidence": 0.9}]],
    })


def _fake_connection(describe_cols, building_rows, captured):
    con = MagicMock()

    def _execute(sql, params=None):
        captured.append((sql, params))
        cursor = MagicMock()
        if sql.strip().upper().startswith("DESCRIBE"):
            cursor.fetchall.return_value = [(c,) for c in describe_cols]
        # fetch_df_chunk streams: yield the frame once, then an empty frame.
        cursor.fetch_df_chunk.side_effect = [building_rows, building_rows.iloc[0:0]]
        return cursor

    con.execute.side_effect = _execute
    return con


def _patched_overture_stream(describe_cols=None, captured=None):
    svc = _make_service()
    captured = [] if captured is None else captured
    describe_cols = describe_cols or ["id", "geometry", "bbox", "height", "num_floors", "sources"]
    con = _fake_connection(describe_cols, _building_result_df(), captured)
    return svc, patch("openavmkit.utilities.overture.duckdb.connect", return_value=con)


def test_stream_pushes_bbox_predicate_and_drops_unavailable_columns():
    captured = []
    # est_height / num_floors_underground / subtype / class are absent from this
    # release and must be dropped from the projection (mirrors prior PyArrow behavior).
    svc, duckdb_connect = _patched_overture_stream(captured=captured)
    with duckdb_connect:
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
    assert "read_parquet('s3://overturemaps-us-west-2/" in sql


def test_get_buildings_builds_geodataframe_with_footprint():
    svc, duckdb_connect = _patched_overture_stream()
    with duckdb_connect:
        gdf = svc.get_buildings(_BBOX, use_cache=False)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].equals(_POLY)
    expected_area = gpd.GeoSeries([_POLY], crs="EPSG:4326").to_crs(
        gdf.estimate_utm_crs()
    ).area.iloc[0] * 10.764
    assert gdf["bldg_area_footprint_sqft"].iloc[0] == pytest.approx(expected_area)
    assert gdf["height_m_best"].iloc[0] == 6.5
    assert gdf["floors_best"].iloc[0] == 2


def test_building_batches_yields_geodataframes_from_stream():
    svc, duckdb_connect = _patched_overture_stream()
    with duckdb_connect:
        frames = list(svc._building_batches(_BBOX))
    assert len(frames) == 1
    assert isinstance(frames[0], gpd.GeoDataFrame)
    assert frames[0].geometry.iloc[0].equals(_POLY)


def test_stream_uses_real_duckdb_bbox_predicate_on_local_parquet(tmp_path):
    path = tmp_path / "buildings.parquet"
    matching = _building_result_df()
    outside = pd.DataFrame(
        {
            "id": ["outside"],
            "geometry": [swkb.dumps(sgeom.box(-123, 37, -122.9, 37.1))],
            "bbox": [{"xmin": -123.0, "xmax": -122.9, "ymin": 37.0, "ymax": 37.1}],
            "height": [4.0],
            "num_floors": [1],
            "sources": [[]],
        }
    )
    pd.concat([matching, outside], ignore_index=True).to_parquet(path)

    svc = _make_service()
    with patch.object(svc, "_buildings_parquet_path", return_value=str(path)):
        chunks = list(svc._stream_building_dfs(_BBOX, OvertureService.DEFAULT_COLUMNS.copy()))

    fetched_buildings = pd.concat(chunks, ignore_index=True)
    assert fetched_buildings["id"].tolist() == ["bldg-1"]
