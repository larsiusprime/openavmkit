from unittest.mock import patch

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
