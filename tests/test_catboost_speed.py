"""CatBoost's speed (OAK box's trial log: 25-45 s a trial on an assessor's 500-sale group, one thread per fold, 19 of
24 threads idle): it takes the cores parallel folds leave idle (its results are the same at any thread count), and
its depth is searched up to 8, not 10. Only CatBoost's saved tuning is stale for it."""
import numpy as np
import pandas as pd
import pytest

from openavmkit import tuning


def test_catboost_threads_a_folds_share_then_the_trial_cap_then_its_default(monkeypatch):
	monkeypatch.delenv(tuning.CATBOOST_THREADS_ENV, raising=False)
	monkeypatch.delenv("OPENAVMKIT_TUNE_WORKERS", raising=False)
	assert tuning.catboost_thread_count() is None  # CatBoost's own default
	monkeypatch.setenv("OPENAVMKIT_TUNE_WORKERS", "1")
	assert tuning.catboost_thread_count() == 1
	monkeypatch.setenv(tuning.CATBOOST_THREADS_ENV, "4")  # folds in parallel: the fold's share wins
	assert tuning.catboost_thread_count() == 4


def test_catboost_depth_is_searched_up_to_8(tmp_path, monkeypatch):
	pytest.importorskip("catboost")
	log = tmp_path / "trials.jsonl"
	monkeypatch.setenv(tuning.TRIAL_LOG_ENV, str(log))
	monkeypatch.setenv(tuning.CATBOOST_THREADS_ENV, "2")
	rng = np.random.default_rng(1)
	X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
	y = 100_000 + 20_000 * X["a"] + rng.normal(scale=1000, size=100)
	tuning._tune_catboost(X, y, n_trials=6, n_splits=2, use_gpu=False)
	import json

	lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
	assert lines and all(x["params"]["depth"] <= 8 for x in lines)
	assert lines[0]["thread_count"] == 2


def test_only_catboosts_saved_tuning_is_stale():
	"""CatBoost's fingerprint carries its own space version; LightGBM's is as before."""
	from openavmkit.tuning import _MODEL_SPACE_VERSION, _study_fingerprint

	assert _MODEL_SPACE_VERSION == {"CatBoost": 2} and "LightGBM" not in _MODEL_SPACE_VERSION
	cols = ["a", "b"]
	plain = _study_fingerprint(cols, 100, 50, seed=42)
	with_space = _study_fingerprint(cols, 100, 50, seed=42, extra="space_CatBoost=v2")
	assert plain != with_space
