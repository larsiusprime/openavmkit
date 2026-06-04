from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openavmkit.benchmark as benchmark
from openavmkit.benchmark import (
	_aggregate_ensemble,
	_perform_ensemble,
	_write_ensemble_contributions,
)
from openavmkit.utilities.settings import get_ensemble_instructions


def _make_settings(ensemble_block: dict) -> dict:
	"""Wrap an ensemble config block in the settings structure get_ensemble_instructions expects."""
	return {
		"modeling": {
			"instructions": {
				"main": {"ensemble": ensemble_block},
			}
		}
	}


def test_aggregate_ensemble_mean_vs_median():
	# Three model prediction columns where the mean and median differ.
	df = pd.DataFrame(
		{
			"key": ["a", "b"],
			"m1": [10.0, 0.0],
			"m2": [20.0, 0.0],
			"m3": [60.0, 30.0],
		}
	)
	cols = ["m1", "m2", "m3"]

	median = _aggregate_ensemble(df, cols, "median")
	mean = _aggregate_ensemble(df, cols, "mean")

	# median of (10, 20, 60) = 20; mean = 30 -> the two methods must diverge
	np.testing.assert_allclose(median.to_numpy(), [20.0, 0.0])
	np.testing.assert_allclose(mean.to_numpy(), [30.0, 10.0])
	assert not np.allclose(median.to_numpy(), mean.to_numpy())


def test_aggregate_ensemble_rejects_unknown_method():
	df = pd.DataFrame({"m1": [1.0], "m2": [2.0]})
	with pytest.raises(ValueError):
		_aggregate_ensemble(df, ["m1", "m2"], "geomean")


def test_get_ensemble_instructions_mean():
	settings = _make_settings({"type": "mean", "models": ["mra", "xgboost"]})
	inst = get_ensemble_instructions(settings, "main")
	assert inst["type"] == "mean"
	assert inst["models"] == ["mra", "xgboost"]


def test_get_ensemble_instructions_default_normalizes_to_median():
	# "default" is an alias: it must normalize to "median" so downstream only
	# ever sees the real aggregation method.
	default_inst = get_ensemble_instructions(
		_make_settings({"type": "default", "models": ["mra"]}), "main"
	)
	median_inst = get_ensemble_instructions(
		_make_settings({"type": "median", "models": ["mra"]}), "main"
	)
	assert default_inst == median_inst
	assert default_inst["type"] == "median"
	assert default_inst["models"] == ["mra"]


def test_get_ensemble_instructions_omitted_type_defaults_to_median():
	# Omitting "type" entirely falls back to the historical default (median).
	inst = get_ensemble_instructions(_make_settings({"models": ["mra"]}), "main")
	assert inst["type"] == "median"


@pytest.mark.parametrize(
	"ensemble_type,expected_agg",
	[("default", "median"), ("median", "median"), ("mean", "mean")],
)
def test_perform_ensemble_dispatches_correct_aggregation(
	monkeypatch, ensemble_type, expected_agg
):
	"""_perform_ensemble must map ensemble type -> aggregation method and pass it through.

	We monkeypatch the heavy default-ensemble runner so no model machinery executes;
	we only assert the wiring (type -> agg) is correct.
	"""
	captured = {}

	def _fake_default_ensemble(*args, **kwargs):
		captured["agg"] = kwargs.get("agg")
		captured["ensemble_list"] = kwargs.get("ensemble_list")
		return "sentinel-result"

	monkeypatch.setattr(benchmark, "_perform_default_ensemble", _fake_default_ensemble)

	settings = _make_settings({"type": ensemble_type, "models": ["mra", "xgboost"]})
	result = _perform_ensemble(
		df_sales=None,
		df_universe=None,
		model_group="mg",
		vacant_only=False,
		outpath="unused",
		dep_var="dep",
		dep_var_test="dep_test",
		all_results=None,
		settings=settings,
	)

	assert result == "sentinel-result"
	assert captured["agg"] == expected_agg
	assert captured["ensemble_list"] == ["mra", "xgboost"]


# ---------------------------------------------------------------------------
# Ensemble contributions / params assembly (_write_ensemble_contributions)
# ---------------------------------------------------------------------------


def _write_member_univ_contribs(tmp_path, m_key, filename, rows):
	"""Write a member's universe contributions CSV. `rows` maps column -> list."""
	member_dir = tmp_path / m_key
	member_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(rows).to_csv(member_dir / filename, index=False)


def _univ_member(keys, preds):
	"""Minimal SingleModelResults-like stub for the universe subset."""
	return SimpleNamespace(
		df_universe=pd.DataFrame({"key": keys}),
		pred_univ=np.asarray(preds, dtype=float),
	)


def _ensemble_results(df_universe):
	"""Minimal ensemble SingleModelResults-like stub (universe subset only)."""
	empty = pd.DataFrame(columns=["key", "key_sale", "prediction"])
	return SimpleNamespace(
		model_name="ensemble",
		df_test=empty,
		df_sales=empty,
		df_universe=df_universe,
	)


def test_mean_ensemble_contributions_assembly(tmp_path):
	keys = ["p1", "p2"]
	# Linear-style member: base column named "intercept", features feat_a/feat_b.
	_write_member_univ_contribs(
		tmp_path, "mra", "contributions_universe.csv",
		{
			"key": keys,
			"intercept": [100.0, 200.0],
			"feat_a": [10.0, 30.0],
			"feat_b": [20.0, 40.0],
			"contribution_sum": [130.0, 270.0],
			"prediction": [130.0, 270.0],
			"check_delta": [0.0, 0.0],
		},
	)
	# Tree-style member: base column "base_value", DIFFERENT feature set + "univ" filename.
	_write_member_univ_contribs(
		tmp_path, "xgboost", "contributions_univ.csv",
		{
			"key": keys,
			"base_value": [50.0, 60.0],
			"feat_a": [5.0, 7.0],
			"feat_c": [15.0, 33.0],
			"contribution_sum": [70.0, 100.0],
			"prediction": [70.0, 100.0],
			"check_delta": [0.0, 0.0],
		},
	)

	# Ensemble prediction is the row-wise mean of member predictions.
	df_universe = pd.DataFrame({
		"key": keys,
		"feat_a": [2.0, 5.0],
		"feat_b": [4.0, 8.0],
		"feat_c": [3.0, 11.0],
		"prediction": [100.0, 185.0],  # mean(130,70)=100 ; mean(270,100)=185
	})
	results = _ensemble_results(df_universe)
	all_results = SimpleNamespace(model_results={
		"mra": _univ_member(keys, [130.0, 270.0]),
		"xgboost": _univ_member(keys, [70.0, 100.0]),
	})

	_write_ensemble_contributions(
		results, str(tmp_path), {}, ["mra", "xgboost"], all_results, mode="mean"
	)

	con = pd.read_csv(tmp_path / "ensemble" / "contributions_universe.csv").set_index("key")
	# (i) feature contribs are the per-row mean (absent feature -> 0 for that member)
	np.testing.assert_allclose(con.loc["p1", "feat_a"], (10 + 5) / 2)
	np.testing.assert_allclose(con.loc["p2", "feat_a"], (30 + 7) / 2)
	np.testing.assert_allclose(con.loc["p1", "feat_b"], 20 / 2)   # xgboost lacks feat_b
	np.testing.assert_allclose(con.loc["p1", "feat_c"], 15 / 2)   # mra lacks feat_c
	# base == mean of member bases (intercept / base_value)
	np.testing.assert_allclose(con.loc["p1", "base_value"], (100 + 50) / 2)
	np.testing.assert_allclose(con.loc["p2", "base_value"], (200 + 60) / 2)
	# (ii) reconstruction is exact
	np.testing.assert_allclose(con.loc["p1", "contribution_sum"], 100.0)
	np.testing.assert_allclose(con.loc["p2", "contribution_sum"], 185.0)
	np.testing.assert_allclose(con["check_delta"].to_numpy(), [0.0, 0.0], atol=1e-9)

	# (iii) params are per-unit: ensemble feature contribution / feature value
	par = pd.read_csv(tmp_path / "ensemble" / "params_universe.csv").set_index("key")
	np.testing.assert_allclose(par.loc["p1", "feat_a"], 7.5 / 2.0)
	np.testing.assert_allclose(par.loc["p2", "feat_a"], 18.5 / 5.0)

	# A member whose only file uses the legacy "univ" name (xgboost, above) is
	# still picked up via the back-compat fallback; canonical output is "universe".
	assert (tmp_path / "ensemble" / "params_universe.csv").exists()


def test_mean_ensemble_folds_non_decomposable_member_into_base(tmp_path):
	keys = ["p1", "p2"]
	_write_member_univ_contribs(
		tmp_path, "mra", "contributions_universe.csv",
		{
			"key": keys,
			"intercept": [100.0, 200.0],
			"feat_a": [10.0, 30.0],
			"feat_b": [20.0, 40.0],
			"contribution_sum": [130.0, 270.0],
			"prediction": [130.0, 270.0],
			"check_delta": [0.0, 0.0],
		},
	)
	_write_member_univ_contribs(
		tmp_path, "xgboost", "contributions_univ.csv",
		{
			"key": keys,
			"base_value": [50.0, 60.0],
			"feat_a": [5.0, 7.0],
			"feat_c": [15.0, 33.0],
			"contribution_sum": [70.0, 100.0],
			"prediction": [70.0, 100.0],
			"check_delta": [0.0, 0.0],
		},
	)
	# local_area is in the ensemble but produces NO contributions file.
	df_universe = pd.DataFrame({
		"key": keys,
		"feat_a": [2.0, 5.0],
		"feat_b": [4.0, 8.0],
		"feat_c": [3.0, 11.0],
		"prediction": [200.0, 290.0],  # mean(130,70,400) ; mean(270,100,500)
	})
	results = _ensemble_results(df_universe)
	all_results = SimpleNamespace(model_results={
		"mra": _univ_member(keys, [130.0, 270.0]),
		"xgboost": _univ_member(keys, [70.0, 100.0]),
		"local_area": _univ_member(keys, [400.0, 500.0]),
	})

	with pytest.warns(UserWarning, match="local_area"):
		_write_ensemble_contributions(
			results, str(tmp_path), {}, ["mra", "xgboost", "local_area"],
			all_results, mode="mean",
		)

	con = pd.read_csv(tmp_path / "ensemble" / "contributions_universe.csv").set_index("key")
	# Reconstruction stays exact: local_area's prediction is absorbed into base.
	np.testing.assert_allclose(con.loc["p1", "contribution_sum"], 200.0)
	np.testing.assert_allclose(con.loc["p2", "contribution_sum"], 290.0)
	np.testing.assert_allclose(con["check_delta"].to_numpy(), [0.0, 0.0], atol=1e-9)
	# base == mean of (intercept, base_value, local_area prediction)
	np.testing.assert_allclose(con.loc["p1", "base_value"], (100 + 50 + 400) / 3)
	# features only attribute the decomposable members (divided by full N=3)
	np.testing.assert_allclose(con.loc["p1", "feat_a"], (10 + 5) / 3)


def test_local_ensemble_contributions_passthrough(tmp_path):
	keys = ["p1", "p2"]
	_write_member_univ_contribs(
		tmp_path, "mra", "contributions_universe.csv",
		{
			"key": keys,
			"intercept": [100.0, 200.0],
			"feat_a": [10.0, 30.0],
			"feat_b": [20.0, 40.0],
			"contribution_sum": [130.0, 270.0],
			"prediction": [130.0, 270.0],
			"check_delta": [0.0, 0.0],
		},
	)
	_write_member_univ_contribs(
		tmp_path, "xgboost", "contributions_univ.csv",
		{
			"key": keys,
			"base_value": [50.0, 60.0],
			"feat_a": [5.0, 7.0],
			"feat_c": [15.0, 33.0],
			"contribution_sum": [70.0, 100.0],
			"prediction": [70.0, 100.0],
			"check_delta": [0.0, 0.0],
		},
	)
	# Local ensemble: p1 picks mra, p2 picks xgboost. Prediction is the selected model's.
	df_universe = pd.DataFrame({
		"key": keys,
		"feat_a": [2.0, 5.0],
		"feat_b": [4.0, 8.0],
		"feat_c": [3.0, 11.0],
		"prediction": [130.0, 100.0],
	})
	results = _ensemble_results(df_universe)
	all_results = SimpleNamespace(model_results={
		"mra": SimpleNamespace(),
		"xgboost": SimpleNamespace(),
	})
	local_selection = {
		"universe": pd.Series(["mra", "xgboost"], index=pd.Index(keys, name="key")),
	}

	_write_ensemble_contributions(
		results, str(tmp_path), {}, ["mra", "xgboost"], all_results,
		mode="local", local_selection=local_selection,
	)

	con = pd.read_csv(tmp_path / "ensemble" / "contributions_universe.csv").set_index("key")
	# p1 == mra's contribs exactly (feat_c absent -> 0); p2 == xgboost's (feat_b absent -> 0)
	np.testing.assert_allclose(con.loc["p1", "feat_a"], 10.0)
	np.testing.assert_allclose(con.loc["p1", "feat_b"], 20.0)
	np.testing.assert_allclose(con.loc["p1", "feat_c"], 0.0)
	np.testing.assert_allclose(con.loc["p1", "base_value"], 100.0)
	np.testing.assert_allclose(con.loc["p2", "feat_a"], 7.0)
	np.testing.assert_allclose(con.loc["p2", "feat_c"], 33.0)
	np.testing.assert_allclose(con.loc["p2", "feat_b"], 0.0)
	np.testing.assert_allclose(con.loc["p2", "base_value"], 60.0)
	np.testing.assert_allclose(con["check_delta"].to_numpy(), [0.0, 0.0], atol=1e-9)
