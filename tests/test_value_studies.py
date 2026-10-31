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
	# Every sale, as before; the trimmed study beside it, without the nominal deed.
	assert res["ratio"]["cod"]["value"] == plain["groups"]["residential"]["ratio"]["cod"]["value"]
	assert res["trimmed_study"]["ratio"]["cod"]["value"] < res["ratio"]["cod"]["value"]
	assert res["trimmed_study"]["count_sales"] == res["count_sales"] - len(res["trimmed"])
	assert out["overall"]["count_trimmed"] == sum(len(g["trimmed"]) for g in out["groups"].values())
	# The IAAO's order-statistic interval: bounds are observed ratios around the median.
	med = res["ratio"]["median_ratio"]
	assert med["low"] <= med["value"] <= med["high"]
	# Judged against the targets, both ways; a group with no class isn't.
	assert res["verdict"]["judged"] and res["verdict"]["cod_target"] == [5.0, 15.0]
	assert set(res["verdict"]) >= {"level", "cod", "cod_low", "prb", "vei"} and "verdict" in res["trimmed_study"]
	assert res["vertical"]["mki"] is not None
	assert out["groups"]["vacant_land"]["verdict"]["judged"] is False
	# Every sale studied, with its ratio and whether it was trimmed.
	sales = pd.read_parquet(tmp_path / "assr_market_value.sales.parquet")
	assert sales.loc[sales["key_sale"] == mine, "trimmed"].all() and sales["trimmed"].sum() == out["overall"]["count_trimmed"]
	# The VEI's reading, as the exposure draft's Appendix E: within 10%, beyond it but the end groups' intervals
	# overlapping, significance within 10%, or significantly outside.
	assert _vei_verdict(4.0, -3.0) == "compliant"
	assert _vei_verdict(-25.0, 12.0, overlap=True) == "not_significant"
	assert _vei_verdict(-25.0, 12.0, overlap=False) == "regressive"
	assert _vei_verdict(25.0, 2.0, overlap=False) == "not_significant"
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


def test_neighborhood_price_levels():
	sup = _sup(False)
	out = run_value_field_studies(sup, SETTINGS, ["assr_market_value"], outpath=None, region="neighborhood")
	reg = out["assr_market_value"]["overall"]["regions"]
	assert reg["field"] == "neighborhood" and reg["regions"] >= 2
	if reg["sales"] >= 20:
		assert len(reg["strata"]) >= 2 and sum(s["n"] for s in reg["strata"]) == reg["sales"]
		levels = [s["level"] for s in reg["strata"]]
		assert levels == sorted(levels)
		assert reg["vei_verdict"] in ("compliant", "inconclusive", "regressive", "progressive")


def test_grouping_keeps_ties_together():
	from openavmkit.value_studies import _group_by_sort_key

	groups = _group_by_sort_key(np.array([1, 1, 1, 2, 3, 4, 5, 6.0]), 4)
	assert [sorted(g.tolist()) for g in groups][0] == [0, 1, 2]
	assert sum(len(g) for g in groups) == 8


def test_trimming_reproduces_the_exposure_drafts_example():
	"""Appendix C.4, Tables 9 and 10: 14 ratios, 1.5 x IQR with the Standard's quartile positions; 1.850 and 2.500
	are outliers (2 of 14: a fifth allowed, "the most extreme cases"), leaving median 0.917 and COD 15.649."""
	from openavmkit.utilities.stats import calc_cod
	from openavmkit.value_studies import _trimmed

	ratios = [0.611, 0.756, 0.762, 0.853, 0.867, 0.909, 0.925, 0.944, 1.014, 1.052, 1.178, 1.367, 1.850, 2.500]
	df = pd.DataFrame({"sale_price": [100.0] * 14, "v": [100.0 * r for r in ratios]})
	t = _trimmed(df, "v", 1.5, max_share=0.2)
	assert [r for r, x in zip(ratios, t) if x] == [1.850, 2.500]
	kept = np.array([r for r, x in zip(ratios, t) if not x])
	assert round(float(np.median(kept)), 3) == 0.917 and round(calc_cod(kept), 3) == 15.649
	assert _trimmed(df, "v", 1.5).sum() == 1  # at the usual tenth, only the furthest


def test_the_modified_kakwani_index():
	from openavmkit.utilities.stats import calc_mki

	price = np.array([100.0, 200, 300, 400, 500])
	assert abs(calc_mki(price * 0.9, price) - 1.0) < 1e-12  # one ratio for all: equitable
	assert calc_mki(np.array([150.0, 220, 300, 360, 420]), price) < 1  # cheaper over-assessed: regressive
	assert calc_mki(np.array([60.0, 160, 300, 440, 600]), price) > 1  # progressive


def test_each_class_can_have_its_own_targets():
	"""standards.by_class: a class's own level, PRB, VEI and minimum sales, in place of the ones for every class."""
	from openavmkit.value_studies import _verdict

	g = {
		"ratio": {"count": 40, "median_ratio": {"value": 0.85, "low": 0.80, "high": 0.88}, "cod": {"value": 12.0}},
		"vertical": {
			"prb": {"value": 0.07},
			"vei": 15.0,
			"vei_significance": 12.0,
			"vei_verdict": "progressive",
			"vei_strata": [{"low": 0.8, "high": 0.9}, {"low": 0.95, "high": 1.05}],
		},
	}
	everyone = {"level": [0.90, 1.10], "prb": [-0.05, 0.05], "prb_unacceptable": [-0.06, 0.06], "vei": 10.0,
		"cod": {"state": [5, 15]}}
	plain = _verdict(g, everyone, "state")
	assert (plain["level"], plain["prb"], plain["vei"]) == (False, False, False)
	own = everyone | {"by_class": {"state": {"level": [0.80, 1.20], "prb": [-0.10, 0.10], "vei": 20.0}}}
	mine = _verdict(g, own, "state")
	assert (mine["level"], mine["prb"], mine["vei"], mine["cod"]) == (True, True, True, True)
	assert _verdict(g, everyone | {"by_class": {"state": {"min_sales": 50}}}, "state")["judged"] is False
