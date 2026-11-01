"""`data.process.enrich.spatial_lag.only`: compute exactly the spatial lags listed (`sales`, `fields`), no others."""

import numpy as np
import pandas as pd
import pytest

from openavmkit.data import SPATIAL_LAG_SALES, _enrich_universe_spatial_lag, spatial_lag_plan
from openavmkit.model_runner import _require_spatial_lag


def _universe(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "key": [str(i) for i in range(n)],
            "model_group": ["sf"] * n,
            "latitude": rng.uniform(36.0, 36.1, n),
            "longitude": rng.uniform(-79.9, -79.8, n),
            "bldg_area_finished_sqft": rng.uniform(1000, 3000, n),
            "land_area_sqft": rng.uniform(4000, 9000, n),
            "bldg_age_years": rng.uniform(0, 80, n),
            "bldg_quality_num": rng.uniform(1, 5, n),
        }
    )


def _lags(settings):
    df = _enrich_universe_spatial_lag(
        _universe(), pd.DataFrame({"key": []}), model_group="sf", sample_from_mgs=["sf"], settings=settings
    )
    return sorted(c for c in df.columns if c.startswith("spatial_lag_"))


def _settings(**sl):
    return {"locality": {"units": "imperial"}, "data": {"process": {"enrich": {"spatial_lag": sl}}}}


def test_without_only_every_lag_as_before():
    assert spatial_lag_plan({}) == (False, set(SPATIAL_LAG_SALES))
    # `sales` means nothing without `only`.
    assert spatial_lag_plan({"sales": ["price"]}) == (False, set(SPATIAL_LAG_SALES))
    lags = _lags(_settings(fields={"bldg_quality_num": 5}))
    assert "spatial_lag_floor_area_ratio" in lags and "spatial_lag_bldg_age_years" in lags
    assert "spatial_lag_bldg_quality_num" in lags


def test_only_computes_exactly_what_is_listed():
    assert spatial_lag_plan({"only": True}) == (True, set())
    assert spatial_lag_plan({"only": True, "sales": ["price", "price_impr"]}) == (True, {"price", "price_impr"})
    assert _lags(_settings(only=True, fields={"bldg_age_years": 5})) == ["spatial_lag_bldg_age_years"]
    # A derived input is worked out only when asked for.
    assert _lags(_settings(only=True, fields={"floor_area_ratio": 5})) == ["spatial_lag_floor_area_ratio"]
    assert _lags(_settings(only=True)) == []


def test_unknown_names_are_said():
    with pytest.raises(ValueError, match="price_impr"):
        spatial_lag_plan({"only": True, "sales": ["price_per_sqft"]})
    with pytest.warns(UserWarning, match="nowhere"):
        _lags(_settings(only=True, fields={"nowhere": 5}))


def test_a_spatial_lag_model_names_what_to_add():
    universe = pd.DataFrame({"spatial_lag_sale_price": [1.0]})
    _require_spatial_lag("sl", universe, {"spatial_lag_sale_price": "price"})
    with pytest.raises(ValueError, match="price_impr"):
        _require_spatial_lag("sla", universe, {"spatial_lag_sale_price_impr_sqft": "price_impr"})
