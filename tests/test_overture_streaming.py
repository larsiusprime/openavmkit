import geopandas as gpd
import pandas as pd
from pandas.testing import assert_series_equal
from shapely.geometry import Polygon

from openavmkit.utilities.overture import OvertureService


FOOTPRINT = "bldg_area_footprint_sqft"
HEIGHT = "bldg_height_ft"


def _svc(tmp_path):
    service = OvertureService.__new__(OvertureService)
    service.cache_dir = str(tmp_path)
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


def test_streaming_stats_match_all_at_once_for_split_batches(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    batch_1 = _building_frame(
        [
            {
                "id": "b1",
                "height": 6.0,
                "est_height": None,
                "num_floors": 2,
                "geometry": Polygon(
                    [(0.0005, 0.0005), (0.0015, 0.0005), (0.0015, 0.0015), (0.0005, 0.0015)]
                ),
            },
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

    old = service.calculate_building_stats(
        parcels.copy(),
        all_buildings,
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
    )
    streamed = service._calculate_building_stats_from_frames(
        parcels.copy(),
        [batch.copy() for batch in batches],
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
        use_cache=False,
    )

    _assert_stat_columns_equal(old, streamed)


def test_streaming_stats_match_all_at_once_for_empty_buildings(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    empty_buildings = gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")

    old = service.calculate_building_stats(
        parcels.copy(),
        empty_buildings,
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
    )
    streamed = service._calculate_building_stats_from_frames(
        parcels.copy(),
        [],
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
        use_cache=False,
    )

    _assert_stat_columns_equal(old, streamed)


def test_streaming_stats_match_all_at_once_when_heights_are_absent(tmp_path):
    service = _svc(tmp_path)
    parcels = _parcel_gdf()
    raw_buildings = _building_frame(
        [
            {
                "id": "b1",
                "geometry": Polygon(
                    [(0.0005, 0.0005), (0.0015, 0.0005), (0.0015, 0.0015), (0.0005, 0.0015)]
                ),
            },
            {
                "id": "b2",
                "geometry": Polygon(
                    [(0.0035, 0.0005), (0.0045, 0.0005), (0.0045, 0.0015), (0.0035, 0.0015)]
                ),
            },
        ]
    )
    derived_buildings = service._derive_height_and_floors(raw_buildings.copy())

    old = service.calculate_building_stats(
        parcels.copy(),
        derived_buildings,
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
    )
    streamed = service._calculate_building_stats_from_frames(
        parcels.copy(),
        [raw_buildings],
        "sqft",
        FOOTPRINT,
        "ft",
        HEIGHT,
        use_cache=False,
    )

    _assert_stat_columns_equal(old, streamed)
