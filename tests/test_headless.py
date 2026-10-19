import json
import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from openavmkit import progress
from openavmkit.headless import run_stage
from openavmkit.synthetic.basic import generate_basic


def _write_locality(root, size=20):
	"""A tiny synthetic locality: one model group, one LightGBM model."""
	sd = generate_basic(size, percent_sales=0.3, percent_vacant=0.1)
	univ = sd.df_universe.copy()
	cell_lat, cell_lon = 0.4 / size, 0.5 / size
	univ["geometry"] = [
		box(lon - cell_lon / 2, lat - cell_lat / 2, lon + cell_lon / 2, lat + cell_lat / 2)
		for lat, lon in zip(univ["latitude"], univ["longitude"])
	]
	univ["bldg_year_built"] = np.where(univ["is_vacant"], 0, (2025 - univ["bldg_age_years"]).round())
	univ["assr_market_value"] = univ["total_value"].round()
	fields = [
		"key", "neighborhood", "bldg_area_finished_sqft", "land_area_sqft", "bldg_quality_num",
		"bldg_condition_num", "bldg_year_built", "assr_market_value",
	]
	in_dir = root / "in"
	in_dir.mkdir(parents=True)
	gpd.GeoDataFrame(univ[fields + ["geometry"]], geometry="geometry", crs="EPSG:4326").to_parquet(in_dir / "parcels.parquet")
	sd.df_sales[["key", "key_sale", "valid_sale", "vacant_sale", "sale_price", "sale_date"]].to_parquet(in_dir / "sales.parquet")
	settings = {
		"locality": {
			"name": "Headless test", "slug": "us-zz-headless", "units": "imperial",
			"center": {"latitude": float(univ["latitude"].mean()), "longitude": float(univ["longitude"].mean())},
		},
		"data": {
			"load": {
				"geo_parcels": {
					"filename": "parcels.parquet",
					"load": {f: ([f, "string"] if f == "key" else f) for f in fields},
					"dupes": {"subset": ["key"], "sort_by": ["key", "asc"], "drop": True},
				},
				"sales": {
					"filename": "sales.parquet",
					"load": {
						"key": ["key", "string"], "key_sale": ["key_sale", "string"],
						"valid_sale": ["valid_sale", "bool"], "vacant_sale": ["vacant_sale", "bool"],
						"sale_price": ["sale_price", "float"], "sale_date": ["sale_date", "datetime"],
					},
					"dupes": {"subset": ["key_sale"], "sort_by": ["key_sale", "asc"], "drop": True},
				},
			},
			"process": {
				"merge": {"universe": ["geo_parcels"], "sales": ["sales"]},
				"enrich": {"spatial_lag": {"model_groups": {"all": {"sample_from": ["all"]}}}},
			},
		},
		"modeling": {
			"metadata": {"modeler": "test", "valuation_date": "2025-01-01", "use_sales_from": 2020},
			"model_groups": {"all": {"name": "All", "filter": [">=", "land_area_sqft", 0]}},
			"instructions": {"main": {"run": ["lightgbm"]}, "cv_folds": 1},
			"models": {"main": {"default": {
				"n_trials": 2,
				"ind_vars": ["bldg_area_finished_sqft", "land_area_sqft", "bldg_age_years", "latitude", "longitude"],
			}}},
		},
		"field_classification": {"important": {"fields": {"loc_neighborhood": "neighborhood"}, "locations": ["neighborhood"]}},
	}
	(in_dir / "settings.json").write_text(json.dumps(settings))
	return root


def test_run_stage_end_to_end_reports_progress(tmp_path):
	root = _write_locality(tmp_path / "us-zz-headless")
	cwd = os.getcwd()
	events = []
	for stage in ("assemble", "clean", "model"):
		run_stage(root, stage, events.append, run_vacant=False)
		assert os.getcwd() == cwd, "run_stage must restore the working directory"

	stages = {e.stage for e in events}
	assert stages == {"assemble", "clean", "model"}
	for stage in stages:
		done = [e for e in events if e.stage == stage and e.step == "done"]
		assert len(done) == 1 and done[0].fraction == 1.0
	assert any(e.step.startswith("model:all:lightgbm") for e in events)
	assert (root / "out" / "1-assemble-sup.pickle").exists()
	assert (root / "out" / "2-clean-sup.pickle").exists()
	assert (root / "out" / "models" / "all" / "main" / "model_ensemble.pickle").exists()


def test_run_stage_cancel(tmp_path):
	root = _write_locality(tmp_path / "us-zz-headless")
	cwd = os.getcwd()
	with pytest.raises(progress.Cancelled):
		run_stage(root, "assemble", cancel=lambda: True)
	assert os.getcwd() == cwd


def test_run_stage_rejects_unknown_stage(tmp_path):
	with pytest.raises(ValueError):
		run_stage(tmp_path, "nope")


def test_report_is_noop_without_listener():
	progress.report("anything", 0.5)
