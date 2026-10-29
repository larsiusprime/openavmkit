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


def test_trimmed_ratios_are_left_out_listed_and_judged(tmp_path):
	from openavmkit.value_studies import IAAO_STANDARDS, _vei_verdict

	sup = _sup(True)
	# A nominal deed among the residential sales: its ratio is in the thousands.
	when = pd.to_datetime(sup.sales["sale_date"])
	i = sup.sales.index[(when >= "2024-01-01") & (when <= "2025-01-01")][0]
	sup.sales.loc[i, "sale_price"] = 10.0
	mine = sup.sales.loc[i, "key_sale"]
	key = sup.sales.loc[i, "key"]
	sup.universe.loc[sup.universe["key"] == key, "model_group"] = "residential"
	plain = run_value_field_studies(sup, SETTINGS, ["assr_market_value"], outpath=None)["assr_market_value"]
	out = run_value_field_studies(
		sup, SETTINGS, ["assr_market_value"], outpath=str(tmp_path), trim_outliers=True, median_ci="iaao",
		standards=IAAO_STANDARDS, classes={"residential": "residential"},
	)["assr_market_value"]
	res = out["groups"]["residential"]
	assert mine in [t["key_sale"] for t in res["trimmed"]]
	assert res["ratio"]["cod"]["value"] < plain["groups"]["residential"]["ratio"]["cod"]["value"]
	assert out["overall"]["count_trimmed"] == sum(len(g["trimmed"]) for g in out["groups"].values())
	# The IAAO's order-statistic interval: bounds are observed ratios around the median.
	med = res["ratio"]["median_ratio"]
	assert med["low"] <= med["value"] <= med["high"]
	# Judged against the targets; a group with no class isn't.
	assert res["verdict"]["judged"] and res["verdict"]["cod_target"] == 15.0
	assert set(res["verdict"]) >= {"level", "cod", "prd", "prb", "vei"}
	assert out["groups"]["vacant_land"]["verdict"]["judged"] is False
	# Every sale studied, with its ratio and whether it was trimmed.
	sales = pd.read_parquet(tmp_path / "assr_market_value.sales.parquet")
	assert sales.loc[sales["key_sale"] == mine, "trimmed"].all() and sales["trimmed"].sum() == out["overall"]["count_trimmed"]
	# The VEI's reading.
	assert _vei_verdict(4.0, -3.0) == "compliant"
	assert _vei_verdict(-25.0, 12.0) == "regressive"
	assert _vei_verdict(25.0, 2.0) == "inconclusive"
	assert _vei_verdict(None, None) == "insufficient"


def test_a_time_adjusted_price_can_be_studied():
	sup = _sup(False)
	sup.sales["sale_price_time_adj"] = sup.sales["sale_price"] * 2
	a = run_value_field_studies(sup, SETTINGS, ["assr_market_value"], outpath=None)["assr_market_value"]
	b = run_value_field_studies(
		sup, SETTINGS, ["assr_market_value"], outpath=None, sale_field="sale_price_time_adj"
	)["assr_market_value"]
	ma, mb = (x["overall"]["ratio"]["median_ratio"]["value"] for x in (a, b))
	assert abs(mb - ma / 2) < 1e-9 and b["sale_field"] == "sale_price_time_adj"


def test_trimming_takes_the_furthest_out_and_never_more_than_a_tenth():
	from openavmkit.value_studies import _trimmed

	def group(ratios):
		return pd.DataFrame({"sale_price": [100.0] * len(ratios), "v": [100.0 * r for r in ratios]})

	assert not _trimmed(group([1.0, 1.0, 5.0, 0.1]), "v", 1.5).any()  # under 10 sales: none
	ratios = [1.0 + 0.01 * i for i in range(18)] + [9.0, 40.0]
	t = _trimmed(group(ratios), "v", 1.5)
	assert t.sum() == 2 and t[-1] and t[-2]
	t = _trimmed(group(ratios[:-1] + [1.05] + [0.0001]), "v", 1.5)
	assert t.sum() == 2  # 2 of 20: the cap
