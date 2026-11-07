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


def test_catboost_categories_auto_is_one_hot_on_windows_only():
	"""On Windows CatBoost's CTRs cost ~20 ms a round whatever the data, so "auto" one-hot encodes categories there
	(up to 255 values); elsewhere CatBoost's own. "one_hot" and "catboost" are the same everywhere."""
	f = tuning.catboost_one_hot_max_size
	assert f("auto", platform="win32") == 255 and f(None, platform="win32") == 255
	assert f("auto", platform="linux") is None and f("auto", platform="darwin") is None
	assert f("one_hot", platform="linux") == 255
	assert f("catboost", platform="win32") is None
	with pytest.raises(ValueError, match="one_hot"):
		f("onehot")


def test_catboost_one_hot_reaches_its_tuning_trials(tmp_path, monkeypatch):
	pytest.importorskip("catboost")
	log = tmp_path / "trials.jsonl"
	monkeypatch.setenv(tuning.TRIAL_LOG_ENV, str(log))
	rng = np.random.default_rng(2)
	X = pd.DataFrame({"a": rng.normal(size=120), "nbhd": [f"n{i}" for i in rng.integers(0, 20, 120)]})
	X["nbhd"] = X["nbhd"].astype("category")
	y = 100_000 + 20_000 * X["a"] + rng.normal(scale=1000, size=120)
	tuning._tune_catboost(X, y, cat_vars=["nbhd"], n_trials=2, n_splits=2, use_gpu=False, one_hot_max_size=255)
	import json

	lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
	assert lines and all(x["one_hot_max_size"] == 255 for x in lines)


def test_catboost_shap_is_per_variable_and_adds_up_with_one_hot():
	"""One-hot is inside CatBoost: its SHAP values (contributions) stay one column per variable, the neighbourhood one
	column, and still add up to the prediction."""
	catboost = pytest.importorskip("catboost")
	rng = np.random.default_rng(3)
	X = pd.DataFrame({"a": rng.normal(size=300), "nbhd": [f"n{i}" for i in rng.integers(0, 40, 300)]})
	y = 100_000 + 20_000 * X["a"] + rng.normal(scale=1000, size=300)
	m = catboost.CatBoostRegressor(
		iterations=100, depth=4, verbose=False, allow_writing_files=False, one_hot_max_size=255
	).fit(catboost.Pool(X, y, cat_features=["nbhd"]))
	pool = catboost.Pool(X.iloc[:30], cat_features=["nbhd"])
	for kind in ("Regular", "Approximate"):
		sv = m.get_feature_importance(pool, type="ShapValues", shap_calc_type=kind)
		assert sv.shape == (30, 3)  # a, nbhd, the base value
		assert np.allclose(sv.sum(axis=1), m.predict(X.iloc[:30]))


def test_shap_numeric_matrix_with_a_category_listed_twice():
	"""A field listed twice among the categories (a location that's also a variable) failed CatBoost's SHAP at
	Finalize: the second pass found it already converted."""
	from openavmkit.utilities.modeling import TreeBasedCategoricalData

	X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "nbhd": pd.Categorical(["x", "y", "x"])})
	cd = TreeBasedCategoricalData.from_training_data(X, ["nbhd", "nbhd"])
	assert cd.categorical_cols == ["nbhd"]
	cd.categorical_cols = ["nbhd", "nbhd"]  # as a model saved before the fix carries it
	assert cd.to_numeric_matrix(X)[:, 1].tolist() == [0.0, 1.0, 0.0]
