import json

import numpy as np
import pandas as pd

from openavmkit.data import SalesUniversePair
from openavmkit.synthetic.basic import generate_basic
from openavmkit.value_studies import run_value_field_studies


def _sup(with_clusters: bool):
	sd = generate_basic(40, percent_sales=0.5)
	univ = sd.df_universe.copy()
	univ["model_group"] = np.where(univ["is_vacant"], "vacant_land", "residential")
	univ["assr_market_value"] = univ["total_value"]
	if with_clusters:
		univ["he_id"] = univ["neighborhood"].astype(str)
	sales = sd.df_sales.drop(columns=["model_group"]).copy()
	sales["valid_for_ratio_study"] = True
	return SalesUniversePair(sales=sales, universe=univ)


SETTINGS = {
	"modeling": {
		"metadata": {"valuation_date": "2025-01-01"},
		"model_groups": {
			"residential": {"filter": ["==", "is_vacant", False]},
			"vacant_land": {"filter": ["==", "is_vacant", True]},
		},
	},
	"analysis": {"ratio_study": {"look_back_years": 1}},
	"field_classification": {"important": {"locations": ["neighborhood"]}},
}


def test_value_field_studies_all_groups_and_overall(tmp_path):
	out = run_value_field_studies(_sup(True), SETTINGS, ["assr_market_value"], outpath=str(tmp_path))
	r = out["assr_market_value"]
	assert set(r["groups"]) == {"residential", "vacant_land"}
	assert r["window"] == {"start": "2024-01-01", "end": "2025-01-01"}
	overall = r["overall"]
	assert overall["count_sales"] == sum(g["count_sales"] for g in r["groups"].values())
	assert overall["count_sales"] > 0
	med = overall["ratio"]["median_ratio"]
	assert med["low"] <= med["value"] <= med["high"]
	assert overall["vertical"]["prb"]["value"] is not None
	assert overall["horizontal"]["clusters"] > 0
	written = json.loads((tmp_path / "assr_market_value.json").read_text())
	assert written["overall"]["count_sales"] == overall["count_sales"]


def test_value_field_studies_without_clusters_skips_horizontal():
	out = run_value_field_studies(_sup(False), SETTINGS, ["assr_market_value"], outpath=None)
	assert out["assr_market_value"]["overall"]["horizontal"] is None
