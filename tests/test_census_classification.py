"""The ACS census figures are facts about the block group a parcel is in: land fields, not the building's, so a
vacant parcel keeps them (OpenAVMKit sets a building's fields to 0 on vacant parcels)."""

import pandas as pd

from openavmkit.data import _enrich_vacant
from openavmkit.pipeline import load_settings
from openavmkit.utilities.settings import get_fields_impr_as_list, get_fields_land_as_list

CENSUS = ["median_income", "median_c_rent", "median_g_rent", "total_population"]


def _settings() -> dict:
	return load_settings("", {"locality": {"name": "Test", "valuation_date": "2025-01-01"}})


def test_census_figures_are_land_fields():
	s = _settings()
	df = pd.DataFrame({c: [1.0] for c in CENSUS})
	assert sorted(get_fields_land_as_list(s, df)) == sorted(CENSUS)
	assert get_fields_impr_as_list(s, df) == []


def test_a_vacant_parcel_keeps_its_block_groups_population():
	df = pd.DataFrame(
		{
			"bldg_area_finished_sqft": [1800.0, 0.0],
			"bldg_quality_num": [4.0, 3.0],
			"total_population": [1200.0, 1200.0],
			"median_income": [91000.0, 91000.0],
		}
	)
	out = _enrich_vacant(df, _settings())
	assert out["is_vacant"].tolist() == [False, True]
	assert out["bldg_quality_num"].tolist() == [4.0, 0.0]  # a building's field: 0 with no building
	assert out["total_population"].tolist() == [1200.0, 1200.0]
	assert out["median_income"].tolist() == [91000.0, 91000.0]
