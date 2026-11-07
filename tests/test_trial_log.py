"""The trial log (OPENAVMKIT_TRIAL_LOG): each finished tuning trial appended as a JSON line, for seeing where tuning
time goes. Nothing is written without it."""
import json

import numpy as np
import pandas as pd
import pytest

from openavmkit import tuning


def _data(n=120, seed=0):
	rng = np.random.default_rng(seed)
	X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y"], size=n)})
	X["c"] = X["c"].astype("category")
	y = 100_000 + 20_000 * X["a"] + 5_000 * X["b"] + rng.normal(scale=1000, size=n)
	return X, y


def _lines(path):
	return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_nothing_is_written_without_it(tmp_path, monkeypatch):
	monkeypatch.delenv(tuning.TRIAL_LOG_ENV, raising=False)
	tuning._trial_log("LightGBM", 0, 1.0, {}, 0.1)
	assert list(tmp_path.iterdir()) == []


def test_batched_trials_are_logged_each_with_its_own_time(tmp_path, monkeypatch):
	"""LightGBM's trials (asked in batches): one line each, its params, score and the data it searched on."""
	log = tmp_path / "trials.jsonl"
	monkeypatch.setenv(tuning.TRIAL_LOG_ENV, str(log))
	X, y = _data()
	tuning._tune_lightgbm(X, y, n_trials=3, n_splits=2, cat_vars=["c"])
	lines = _lines(log)
	assert len(lines) == 3
	first = lines[0]
	assert first["model"] == "LightGBM" and first["state"] == "COMPLETE"
	assert first["seconds"] >= 0 and first["value"] > 0 and "learning_rate" in first["params"]
	assert (first["rows"], first["features"], first["cat_features"], first["folds"]) == (120, 3, 1, 2)
	assert sorted(x["trial"] for x in lines) == [0, 1, 2]


def test_catboost_trials_are_logged(tmp_path, monkeypatch):
	"""CatBoost's trials (one at a time, study.optimize): Optuna's own duration, the threads it was given."""
	pytest.importorskip("catboost")
	log = tmp_path / "trials.jsonl"
	monkeypatch.setenv(tuning.TRIAL_LOG_ENV, str(log))
	monkeypatch.setenv("OPENAVMKIT_TUNE_WORKERS", "2")
	X, y = _data()
	tuning._tune_catboost(X, y, n_trials=2, n_splits=2, cat_vars=["c"], use_gpu=False)
	lines = _lines(log)
	assert [x["model"] for x in lines] == ["CatBoost", "CatBoost"]
	assert lines[0]["seconds"] > 0 and "depth" in lines[0]["params"]
	assert (lines[0]["cat_features"], lines[0]["folds"], lines[0]["thread_count"]) == (1, 2, 2)
