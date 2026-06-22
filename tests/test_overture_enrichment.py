from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from openavmkit.data import _enrich_df_overture


def _parcel_gdf(keys=None):
    keys = ["p1"] if keys is None else keys
    return gpd.GeoDataFrame(
        {"key": keys},
        geometry=[
            Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])
            for _ in keys
        ],
        crs="EPSG:4326",
    )


def _settings():
    return {"locality": {"units": "imperial"}}


def _enrich_settings(footprint_units="sqft", footprint_field="footprint_sqft", cache=True):
    return {
        "overture": {
            "enabled": True,
            "cache": cache,
            "footprint": {"units": footprint_units, "field": footprint_field},
            "height": {"units": "ft", "field": "height_ft"},
        }
    }


def test_enrich_df_overture_calls_streaming_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = _parcel_gdf()
    service = MagicMock()
    service.calculate_building_stats_streaming.return_value = parcels.assign(
        footprint_sqft=123.0,
        height_ft=6.0,
    )

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ), patch("openavmkit.data.init_service_overture", return_value=service):
        out = _enrich_df_overture(
            parcels,
            _enrich_settings(cache=False),
            {},
            _settings(),
        )

    assert out["footprint_sqft"].tolist() == [123.0]
    args, kwargs = service.calculate_building_stats_streaming.call_args
    assert args[2:6] == ("sqft", "footprint_sqft", "ft", "height_ft")
    assert kwargs == {"use_cache": False, "verbose": False}


def test_enrich_df_overture_fails_open_on_service_init_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = _parcel_gdf()

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch(
        "openavmkit.data.init_service_overture",
        side_effect=RuntimeError("prefix lookup failed"),
    ):
        with pytest.warns(UserWarning, match="Failed to calculate Overture building stats"):
            out = _enrich_df_overture(parcels, _enrich_settings(), {}, _settings())

    assert list(out.columns) == list(parcels.columns)
    assert out.geometry.equals(parcels.geometry)
    write_cached_df.assert_not_called()


def test_enrich_df_overture_stats_failure_does_not_write_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = _parcel_gdf()
    service = MagicMock()
    service.calculate_building_stats_streaming.side_effect = RuntimeError("duckdb failed")

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch("openavmkit.data.init_service_overture", return_value=service):
        with pytest.warns(UserWarning, match="Failed to calculate Overture building stats"):
            out = _enrich_df_overture(parcels, _enrich_settings(), {}, _settings())

    assert list(out.columns) == list(parcels.columns)
    assert out.geometry.equals(parcels.geometry)
    write_cached_df.assert_not_called()


def test_enrich_df_overture_skips_outer_cache_for_duplicate_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = _parcel_gdf(["p1", "p1"])
    service = MagicMock()
    service.calculate_building_stats_streaming.return_value = parcels.assign(
        footprint_sqft=[100.0, 100.0],
        height_ft=[10.0, 10.0],
    )

    with patch("openavmkit.data.get_cached_df") as get_cached_df, patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch("openavmkit.data.init_service_overture", return_value=service):
        out = _enrich_df_overture(parcels, _enrich_settings(), {}, _settings())

    assert out["footprint_sqft"].tolist() == [100.0, 100.0]
    get_cached_df.assert_not_called()
    write_cached_df.assert_not_called()


def test_enrich_df_overture_invalid_units_fail_fast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = _parcel_gdf()
    service = MagicMock()
    service.calculate_building_stats_streaming.side_effect = ValueError("Unsupported footprint units")

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch("openavmkit.data.init_service_overture", return_value=service):
        with pytest.raises(ValueError, match="Unsupported footprint units"):
            _enrich_df_overture(
                parcels,
                _enrich_settings("acres", "footprint_acres"),
                {},
                _settings(),
            )

    write_cached_df.assert_not_called()
