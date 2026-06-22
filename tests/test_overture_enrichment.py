from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from openavmkit.data import _enrich_df_overture


def test_enrich_df_overture_fails_open_on_service_init_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = gpd.GeoDataFrame(
        {"key": ["p1"]},
        geometry=[Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])],
        crs="EPSG:4326",
    )
    settings = {"locality": {"units": "imperial"}}
    enrich_settings = {
        "overture": {
            "enabled": True,
            "cache": True,
            "footprint": {"units": "sqft", "field": "footprint_sqft"},
            "height": {"units": "ft", "field": "height_ft"},
        }
    }

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch(
        "openavmkit.data.init_service_overture",
        side_effect=RuntimeError("prefix lookup failed"),
    ):
        with pytest.warns(UserWarning, match="Failed to calculate Overture building stats"):
            out = _enrich_df_overture(parcels, enrich_settings, {}, settings)

    assert list(out.columns) == list(parcels.columns)
    assert out.geometry.equals(parcels.geometry)
    write_cached_df.assert_not_called()


def test_enrich_df_overture_skips_outer_cache_for_duplicate_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = gpd.GeoDataFrame(
        {"key": ["p1", "p1"]},
        geometry=[
            Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]),
            Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]),
        ],
        crs="EPSG:4326",
    )
    settings = {"locality": {"units": "imperial"}}
    enrich_settings = {
        "overture": {
            "enabled": True,
            "cache": True,
            "footprint": {"units": "sqft", "field": "footprint_sqft"},
            "height": {"units": "ft", "field": "height_ft"},
        }
    }
    service = MagicMock()
    service.calculate_building_stats_streaming.return_value = parcels.assign(
        footprint_sqft=[100.0, 100.0],
        height_ft=[10.0, 10.0],
    )

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch("openavmkit.data.init_service_overture", return_value=service):
        out = _enrich_df_overture(parcels, enrich_settings, {}, settings)

    assert out["footprint_sqft"].tolist() == [100.0, 100.0]
    write_cached_df.assert_not_called()


def test_enrich_df_overture_invalid_units_fail_fast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parcels = gpd.GeoDataFrame(
        {"key": ["p1"]},
        geometry=[Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])],
        crs="EPSG:4326",
    )
    settings = {"locality": {"units": "imperial"}}
    enrich_settings = {
        "overture": {
            "enabled": True,
            "cache": True,
            "footprint": {"units": "acres", "field": "footprint_acres"},
            "height": {"units": "ft", "field": "height_ft"},
        }
    }
    service = MagicMock()
    service.calculate_building_stats_streaming.side_effect = ValueError("Unsupported footprint units")

    with patch("openavmkit.data.get_cached_df", return_value=None), patch(
        "openavmkit.data.write_cached_df"
    ) as write_cached_df, patch("openavmkit.data.init_service_overture", return_value=service):
        with pytest.raises(ValueError, match="Unsupported footprint units"):
            _enrich_df_overture(parcels, enrich_settings, {}, settings)

    write_cached_df.assert_not_called()
