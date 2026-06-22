from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from shapely.geometry import Polygon

from openavmkit.utilities.overture import OvertureService


FOOTPRINT = "bldg_area_footprint_sqft"
HEIGHT = "bldg_height_ft"


def _svc(tmp_path):
    service = OvertureService.__new__(OvertureService)
    service.cache_dir = str(tmp_path)
    service.settings = {}
    return service


def _parcel_gdf():
    return gpd.GeoDataFrame(
        {
            "key": ["p1", "p2", "p3"],
            "address": ["1 Main", "2 Main", "3 Main"],
        },
        geometry=[
            Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)]),
            Polygon([(0.003, 0), (0.005, 0), (0.005, 0.002), (0.003, 0.002)]),
            Polygon([(0.006, 0), (0.008, 0), (0.008, 0.002), (0.006, 0.002)]),
        ],
        crs="EPSG:4326",
    )


def _building_frame(records):
    rows = [
        {key: value for key, value in record.items() if key != "geometry"}
        for record in records
    ]
    return gpd.GeoDataFrame(
        rows,
        geometry=[record["geometry"] for record in records],
        crs="EPSG:4326",
    )


def _building_one_record(extra=None):
    record = {
        "id": "b1",
        "height": 6.0,
        "num_floors": 2,
        "geometry": Polygon(
            [(0.0005, 0.0005), (0.0015, 0.0005), (0.0015, 0.0015), (0.0005, 0.0015)]
        ),
    }
    if extra:
        record.update(extra)
    return record


def _single_building_batch(service):
    return service._derive_height_and_floors(_building_frame([_building_one_record()]))


def _assert_stat_columns_equal(old, streamed):
    assert list(streamed["address"]) == ["1 Main", "2 Main", "3 Main"]
    for column in (FOOTPRINT, HEIGHT, "bldg_stories"):
        assert column in streamed.columns
        assert_series_equal(
            old[column].reset_index(drop=True),
            streamed[column].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
            rtol=1e-9,
            atol=1e-6,
        )


def _assert_streaming_stats_match_legacy(
    service, parcels, legacy_buildings, streaming_frames
):
    old = service.calculate_building_stats(
        parcels.copy(),
        legacy_buildings,
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
    )
    streamed = service._calculate_building_stats_from_frames(
        parcels.copy(),
        streaming_frames,
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
        use_cache=False,
    )

    _assert_stat_columns_equal(old, streamed)


def test_building_batches_splits_large_arrow_batches(tmp_path, monkeypatch):
    service = _svc(tmp_path)
    service.settings = {"stream_batch_rows": 2}
    geom = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])
    fetched = pd.DataFrame(
        {
            "id": [f"b{i}" for i in range(5)],
            "geometry": [geom.wkb for _ in range(5)],
        }
    )

    # The DuckDB fetch streams pandas chunks; _building_batches must re-slice them
    # into stream_batch_rows-sized GeoDataFrames for the per-parcel stats aggregation.
    def fake_stream(bbox, proj_cols, verbose=False):
        yield fetched

    monkeypatch.setattr(service, "_stream_building_dfs", fake_stream)

    chunks = list(service._building_batches((0, 0, 1, 1)))

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert all(chunk.crs == "EPSG:4326" for chunk in chunks)


def test_streaming_stats_match_all_at_once_for_split_batches(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    batch_1 = _building_frame(
        [
            _building_one_record({"est_height": None}),
            {
                "id": "b2",
                "height": None,
                "est_height": 9.0,
                "num_floors": None,
                "geometry": Polygon(
                    [(0.0034, 0.0004), (0.0046, 0.0004), (0.0046, 0.0016), (0.0034, 0.0016)]
                ),
            },
        ]
    )
    batch_2 = _building_frame(
        [
            {
                "id": "b3",
                "height": 12.0,
                "est_height": None,
                "num_floors": 4,
                "geometry": Polygon(
                    [(0.001, 0.001), (0.004, 0.001), (0.004, 0.003), (0.001, 0.003)]
                ),
            },
        ]
    )
    batches = [
        service._derive_height_and_floors(batch_1.copy()),
        service._derive_height_and_floors(batch_2.copy()),
    ]
    all_buildings = pd.concat(batches, ignore_index=True)

    _assert_streaming_stats_match_legacy(
        service,
        parcels,
        all_buildings,
        [batch.copy() for batch in batches],
    )


def test_streaming_stats_match_all_at_once_for_empty_buildings(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    empty_buildings = gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")

    _assert_streaming_stats_match_legacy(service, parcels, empty_buildings, [])


def test_streaming_stats_match_all_at_once_when_heights_are_absent(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    raw_buildings = _building_frame(
        [
            _building_one_record({"height": None, "num_floors": None}),
            {
                "id": "b2",
                "geometry": Polygon(
                    [(0.0035, 0.0005), (0.0045, 0.0005), (0.0045, 0.0015), (0.0035, 0.0015)]
                ),
            },
        ]
    )
    derived_buildings = service._derive_height_and_floors(raw_buildings.copy())

    _assert_streaming_stats_match_legacy(
        service,
        parcels,
        derived_buildings,
        [raw_buildings],
    )


def test_streaming_stats_handles_duplicate_parcel_keys(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    baseline = service._calculate_building_stats_from_frames(
        parcels.copy(),
        [_single_building_batch(service)],
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
        use_cache=False,
    )
    parcels = pd.concat([parcels, parcels.iloc[[0]].copy()], ignore_index=True)
    batch = _single_building_batch(service)

    with patch.object(service, "_stats_cache_load") as stats_cache_load, patch.object(
        service, "_stats_cache_save"
    ) as stats_cache_save:
        streamed = service._calculate_building_stats_from_frames(
            parcels.copy(),
            [batch],
            "sqft",
            FOOTPRINT,
            "ft",
            HEIGHT,
        )

    assert len(streamed) == len(parcels)
    stats_cache_load.assert_not_called()
    stats_cache_save.assert_not_called()
    duplicated = streamed[streamed["key"].eq("p1")]
    expected = baseline.loc[baseline["key"].eq("p1")].iloc[0]
    assert duplicated[FOOTPRINT].tolist() == pytest.approx(
        [expected[FOOTPRINT], expected[FOOTPRINT]]
    )
    assert duplicated[HEIGHT].tolist() == pytest.approx([expected[HEIGHT], expected[HEIGHT]])
    assert duplicated["bldg_stories"].tolist() == [
        expected["bldg_stories"],
        expected["bldg_stories"],
    ]
