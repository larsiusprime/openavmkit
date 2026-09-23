from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openavmkit.model_runner as benchmark
from openavmkit.model_runner import (
	_aggregate_ensemble,
	_perform_ensemble,
	_perform_default_ensemble,
	_validate_ensemble_models,
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
# Manual model selection: the `models` whitelist + `optimize` flag
# ---------------------------------------------------------------------------


def test_get_ensemble_instructions_optimize_defaults():
	# models given -> optimize defaults to False (whitelist, use as-is)
	inst = get_ensemble_instructions(
		_make_settings({"type": "median", "models": ["mra", "xgboost"]}), "main"
	)
	assert inst["models"] == ["mra", "xgboost"]
	assert inst["optimize"] is False

	# models omitted -> optimize defaults to True (historical: optimize over all)
	inst = get_ensemble_instructions(_make_settings({"type": "median"}), "main")
	assert inst["models"] == []
	assert inst["optimize"] is True


def test_get_ensemble_instructions_optimize_explicit():
	# explicit optimize=True with a list -> optimize *from* the whitelist
	inst = get_ensemble_instructions(
		_make_settings(
			{"type": "mean", "models": ["mra", "xgboost"], "optimize": True}
		),
		"main",
	)
	assert inst["models"] == ["mra", "xgboost"]
	assert inst["optimize"] is True

	# explicit optimize=False with no list -> combine everything, no pruning
	inst = get_ensemble_instructions(
		_make_settings({"type": "median", "optimize": False}), "main"
	)
	assert inst["models"] == []
	assert inst["optimize"] is False


def _fake_all_results(model_keys):
	"""MultiModelResults-like stub exposing only `model_results` keys."""
	return SimpleNamespace(model_results={k: SimpleNamespace() for k in model_keys})


def test_validate_ensemble_models_drops_unknown_with_warning():
	all_results = _fake_all_results(["mra", "xgboost"])
	with pytest.warns(UserWarning, match="bogus"):
		kept = _validate_ensemble_models(["mra", "bogus", "xgboost"], all_results)
	assert kept == ["mra", "xgboost"]  # order preserved, unknown dropped


def test_validate_ensemble_models_empty_passthrough():
	all_results = _fake_all_results(["mra", "xgboost"])
	assert _validate_ensemble_models([], all_results) == []
	assert _validate_ensemble_models(None, all_results) == []


@pytest.mark.parametrize(
	"ensemble_list,optimize,expect_optimize_called,expected_run_list",
	[
		# whitelist, no optimization -> exact list passed straight to _run_ensemble
		(["mra", "xgboost"], False, False, ["mra", "xgboost"]),
		# whitelist + optimize -> optimizer runs over the whitelist
		(["mra", "xgboost"], True, True, ["mra"]),
		# no list + optimize -> optimizer runs over everything
		([], True, True, ["mra"]),
	],
)
def test_perform_default_ensemble_optimize_branch(
	monkeypatch, ensemble_list, optimize, expect_optimize_called, expected_run_list
):
	"""_perform_default_ensemble must honor the optimize flag and feed the right list to _run_ensemble."""
	calls = {"optimize": False, "run_list": None, "optimize_input": None}

	def _fake_optimize(*args, **kwargs):
		calls["optimize"] = True
		calls["optimize_input"] = kwargs.get("ensemble_list")
		# Pretend the optimizer pruned down to a single best model.
		return ["mra"]

	def _fake_run(*args, **kwargs):
		calls["run_list"] = kwargs.get("ensemble_list")
		return "ran"

	monkeypatch.setattr(benchmark, "_optimize_ensemble", _fake_optimize)
	monkeypatch.setattr(benchmark, "_run_ensemble", _fake_run)

	all_results = _fake_all_results(["mra", "xgboost"])
	result = _perform_default_ensemble(
		df_sales=None,
		df_universe=None,
		model_group="mg",
		vacant_only=False,
		outpath="unused",
		dep_var="dep",
		dep_var_test="dep_test",
		all_results=all_results,
		settings={},
		ensemble_list=list(ensemble_list),
		optimize=optimize,
	)

	assert result == "ran"
	assert calls["optimize"] is expect_optimize_called
	assert calls["run_list"] == expected_run_list


def test_perform_default_ensemble_defaults_to_whitelist_when_models_given(monkeypatch):
	"""With optimize=None and a non-empty list, the list is used as-is (no optimizer)."""
	calls = {"optimize": False, "run_list": None}
	monkeypatch.setattr(
		benchmark, "_optimize_ensemble",
		lambda *a, **k: calls.__setitem__("optimize", True) or ["should-not-be-used"],
	)
	monkeypatch.setattr(
		benchmark, "_run_ensemble",
		lambda *a, **k: calls.__setitem__("run_list", k.get("ensemble_list")) or "ran",
	)

	_perform_default_ensemble(
		df_sales=None, df_universe=None, model_group="mg", vacant_only=False,
		outpath="unused", dep_var="dep", dep_var_test="dep_test",
		all_results=_fake_all_results(["mra", "xgboost", "lightgbm"]),
		settings={}, ensemble_list=["mra", "lightgbm"], optimize=None,
	)
	assert calls["optimize"] is False
	assert calls["run_list"] == ["mra", "lightgbm"]


def test_perform_ensemble_passes_optimize_through(monkeypatch):
	"""_perform_ensemble must forward the resolved optimize flag to the default runner."""
	captured = {}
	monkeypatch.setattr(
		benchmark, "_perform_default_ensemble",
		lambda *a, **k: captured.update(k) or "sentinel",
	)
	# models present, optimize unspecified -> resolves to False
	_perform_ensemble(
		df_sales=None, df_universe=None, model_group="mg", vacant_only=False,
		outpath="unused", dep_var="dep", dep_var_test="dep_test", all_results=None,
		settings=_make_settings({"type": "median", "models": ["mra", "xgboost"]}),
	)
	assert captured["ensemble_list"] == ["mra", "xgboost"]
	assert captured["optimize"] is False


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


def test_median_ensemble_odd_passthrough(tmp_path):
	# Odd present-count (3) -> the central member wins per row; the winner varies
	# across rows, so contributions are an exact per-row pass-through.
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
	_write_member_univ_contribs(
		tmp_path, "lightgbm", "contributions_universe.csv",
		{
			"key": keys,
			"base_value": [40.0, 200.0],
			"feat_a": [35.0, 50.0],
			"feat_d": [25.0, 50.0],
			"contribution_sum": [100.0, 300.0],
			"prediction": [100.0, 300.0],
			"check_delta": [0.0, 0.0],
		},
	)
	# Per-row medians: p1 -> sorted [70,100,130] -> 100 (lightgbm wins)
	#                  p2 -> sorted [100,270,300] -> 270 (mra wins)
	df_universe = pd.DataFrame({
		"key": keys,
		"feat_a": [2.0, 5.0],
		"feat_b": [4.0, 8.0],
		"feat_c": [3.0, 11.0],
		"feat_d": [6.0, 9.0],
		"prediction": [100.0, 270.0],
	})
	results = _ensemble_results(df_universe)
	all_results = SimpleNamespace(model_results={
		"mra": _univ_member(keys, [130.0, 270.0]),
		"xgboost": _univ_member(keys, [70.0, 100.0]),
		"lightgbm": _univ_member(keys, [100.0, 300.0]),
	})

	_write_ensemble_contributions(
		results, str(tmp_path), {}, ["mra", "xgboost", "lightgbm"],
		all_results, mode="median",
	)

	con = pd.read_csv(tmp_path / "ensemble" / "contributions_universe.csv").set_index("key")
	# p1 is exactly lightgbm's decomposition; features it lacks are 0.
	np.testing.assert_allclose(con.loc["p1", "feat_a"], 35.0)
	np.testing.assert_allclose(con.loc["p1", "feat_d"], 25.0)
	np.testing.assert_allclose(con.loc["p1", "feat_b"], 0.0)
	np.testing.assert_allclose(con.loc["p1", "feat_c"], 0.0)
	np.testing.assert_allclose(con.loc["p1", "base_value"], 40.0)
	np.testing.assert_allclose(con.loc["p1", "contribution_sum"], 100.0)
	# p2 is exactly mra's decomposition.
	np.testing.assert_allclose(con.loc["p2", "feat_a"], 30.0)
	np.testing.assert_allclose(con.loc["p2", "feat_b"], 40.0)
	np.testing.assert_allclose(con.loc["p2", "feat_d"], 0.0)
	np.testing.assert_allclose(con.loc["p2", "base_value"], 200.0)
	np.testing.assert_allclose(con.loc["p2", "contribution_sum"], 270.0)
	np.testing.assert_allclose(con["check_delta"].to_numpy(), [0.0, 0.0], atol=1e-9)


def test_median_ensemble_even_equals_mean_of_two(tmp_path):
	# Even present-count (2) -> median == mean of the two central members, so the
	# decomposition is the per-row mean of both members' contributions.
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
	df_universe = pd.DataFrame({
		"key": keys,
		"feat_a": [2.0, 5.0],
		"feat_b": [4.0, 8.0],
		"feat_c": [3.0, 11.0],
		"prediction": [100.0, 185.0],  # median of two == mean of two
	})
	results = _ensemble_results(df_universe)
	all_results = SimpleNamespace(model_results={
		"mra": _univ_member(keys, [130.0, 270.0]),
		"xgboost": _univ_member(keys, [70.0, 100.0]),
	})

	_write_ensemble_contributions(
		results, str(tmp_path), {}, ["mra", "xgboost"], all_results, mode="median",
	)

	con = pd.read_csv(tmp_path / "ensemble" / "contributions_universe.csv").set_index("key")
	np.testing.assert_allclose(con.loc["p1", "feat_a"], (10 + 5) / 2)
	np.testing.assert_allclose(con.loc["p1", "feat_b"], 20 / 2)
	np.testing.assert_allclose(con.loc["p1", "feat_c"], 15 / 2)
	np.testing.assert_allclose(con.loc["p1", "base_value"], (100 + 50) / 2)
	np.testing.assert_allclose(con.loc["p1", "contribution_sum"], 100.0)
	np.testing.assert_allclose(con.loc["p2", "contribution_sum"], 185.0)
	np.testing.assert_allclose(con["check_delta"].to_numpy(), [0.0, 0.0], atol=1e-9)


# --- Ensemble beeswarm plotting -------------------------------------------

import os
import matplotlib

matplotlib.use("Agg")  # headless: beeswarm smoke tests need no display

from openavmkit.model_runner import _ensemble_beeswarm, _model_shaps
from openavmkit.shap_analysis import (
	explanation_from_contributions,
	plot_full_beeswarm,
)


def _contrib_and_features():
	"""A tiny contributions table + matching raw feature values (by key_sale)."""
	df_contrib = pd.DataFrame({
		"key": ["k1", "k2", "k3"],
		"key_sale": ["s1", "s2", "s3"],
		"base_value": [100.0, 100.0, 100.0],
		"feat_a": [10.0, -5.0, 2.0],
		"feat_b": [-3.0, 4.0, 1.0],
		"contribution_sum": [107.0, 99.0, 103.0],
	})
	df_features = pd.DataFrame({
		"key_sale": ["s3", "s1", "s2"],  # deliberately out of order
		"feat_a": [20.0, 200.0, 50.0],
		"feat_b": [7.0, 70.0, 35.0],
	})
	return df_contrib, df_features


def test_explanation_from_contributions_shapes_and_alignment():
	df_contrib, df_features = _contrib_and_features()
	expl = explanation_from_contributions(df_contrib, df_features, key_col="key_sale")

	assert list(expl.feature_names) == ["feat_a", "feat_b"]
	assert expl.values.shape == (3, 2)
	assert expl.data.shape == expl.values.shape
	np.testing.assert_allclose(expl.base_values, [100.0, 100.0, 100.0])
	# values come straight from the contribution columns
	np.testing.assert_allclose(expl.values[:, 0], [10.0, -5.0, 2.0])
	# data is aligned by key_sale, not row order: s1->200, s2->50, s3->20
	np.testing.assert_allclose(expl.data[:, 0], [200.0, 50.0, 20.0])


def test_explanation_from_contributions_missing_feature_is_nan():
	df_contrib, df_features = _contrib_and_features()
	# Drop a feature from the raw values: its data column should be all-NaN.
	expl = explanation_from_contributions(
		df_contrib, df_features.drop(columns=["feat_b"]), key_col="key_sale"
	)
	b = expl.feature_names.index("feat_b")
	assert np.isnan(expl.data[:, b]).all()
	# values for the dropped-from-data feature are still present
	np.testing.assert_allclose(expl.values[:, b], [-3.0, 4.0, 1.0])


def test_explanation_from_contributions_renders_beeswarm(tmp_path):
	df_contrib, df_features = _contrib_and_features()
	expl = explanation_from_contributions(df_contrib, df_features, key_col="key_sale")
	out = tmp_path / "beeswarm.png"
	plot_full_beeswarm(expl, title="ensemble", save_path=str(out))
	assert out.exists() and out.stat().st_size > 0


def test_ensemble_beeswarm_filters_train_and_renders(tmp_path):
	"""_ensemble_beeswarm reads contributions_sales.csv, keeps train rows, renders."""
	path = tmp_path / "ensemble"
	os.makedirs(path, exist_ok=True)
	df_contrib = pd.DataFrame({
		"key": ["k1", "k2", "k3"],
		"key_sale": ["s1", "s2", "s3"],
		"base_value": [100.0, 100.0, 100.0],
		"feat_a": [10.0, -5.0, 2.0],
		"contribution_sum": [110.0, 95.0, 102.0],
	})
	df_contrib.to_csv(path / "contributions_sales.csv", index=False)

	# Train == sales minus the test row (s2). df_train carries raw feature values.
	df_train = pd.DataFrame({"key_sale": ["s1", "s3"], "feat_a": [200.0, 20.0]})
	smr = SimpleNamespace(
		model_name="ensemble",
		model_engine="ensemble",
		ds=SimpleNamespace(df_train=df_train),
	)
	# Should not raise; renders the 2 train rows only.
	_ensemble_beeswarm(smr, str(tmp_path), title="grp/ensemble")


def test_ensemble_beeswarm_missing_file_is_noop(tmp_path):
	smr = SimpleNamespace(
		model_name="ensemble",
		model_engine="ensemble",
		ds=SimpleNamespace(df_train=pd.DataFrame({"key_sale": ["s1"]})),
	)
	# No contributions_sales.csv on disk -> quietly returns.
	_ensemble_beeswarm(smr, str(tmp_path), title="grp/ensemble")
