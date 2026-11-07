"""Skipping one model while it runs (progress.in_part / check_part, OPENAVMKIT_SKIP_FILE): the caller names the part
("model:<group>:<model>") in the skip file; the model's tuning stops at its next trial, in a worker process too, and
the run goes on without it."""
import numpy as np
import pandas as pd
import pytest

from openavmkit import model_runner, progress, tuning


def test_a_part_named_in_the_skip_file_is_skipped(tmp_path, monkeypatch):
	f = tmp_path / "skip.txt"
	monkeypatch.setenv(progress.SKIP_FILE_ENV, str(f))
	progress.check_part("model:sf:catboost")  # no file yet: nothing
	f.write_text("model:sf:catboost\n", encoding="utf-8")
	with progress.in_part("model:sf:lightgbm"):
		progress.check_part()  # another model: goes on
	with progress.in_part("model:sf:catboost"):
		with pytest.raises(progress.Skipped):
			progress.check_part()
	progress.check_part()  # outside any part: nothing to skip


def test_the_callers_own_skip_check_counts_too():
	with progress.reporting(skip=lambda part: part == "model:sf:mra"), progress.in_part("model:sf:mra"):
		with pytest.raises(progress.Skipped):
			progress.check_part()


def test_tuning_stops_between_trials(tmp_path, monkeypatch):
	"""Batched trials (LightGBM): skipped mid-search, it stops at the next trial; on its pool's threads too."""
	f = tmp_path / "skip.txt"
	monkeypatch.setenv(progress.SKIP_FILE_ENV, str(f))
	rng = np.random.default_rng(0)
	X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
	y = 100_000 + 20_000 * X["a"] + rng.normal(scale=500, size=80)
	seen = []
	real = tuning._lightgbm_kfold_cv

	def counting(*args, **kwargs):
		seen.append(1)
		if len(seen) == 2:
			f.write_text("model:sf:lightgbm\n", encoding="utf-8")  # asked while the second trial runs
		return real(*args, **kwargs)

	monkeypatch.setattr(tuning, "_lightgbm_kfold_cv", counting)
	with progress.in_part("model:sf:lightgbm"), pytest.raises(progress.Skipped):
		tuning._tune_lightgbm(X, y, n_trials=40, n_splits=2)
	assert len(seen) < 40


def test_a_skipped_fold_says_so_and_the_cv_stops(tmp_path, monkeypatch):
	"""A fold (in a worker process, or not) skipped returns a marker, not an exception across processes."""
	f = tmp_path / "skip.txt"
	f.write_text("model:sf:catboost\n", encoding="utf-8")
	monkeypatch.setenv(progress.SKIP_FILE_ENV, str(f))
	out = model_runner._run_cv_fold(
		0, ["a"], ["b"], None, None, None, False, "sf", "catboost", {}, {}, "y", "y", [], [], str(tmp_path), False, False
	)
	assert out == model_runner._FOLD_SKIPPED
