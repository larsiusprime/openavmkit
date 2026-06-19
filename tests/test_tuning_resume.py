"""Tests for resumable (crash-safe) Optuna tuning and its auto-cleanup.

Covers the helpers in :mod:`openavmkit.tuning` and the journal lifecycle that
:func:`openavmkit.modeling._get_params` drives: a journal-backed study persists trials
incrementally, an interrupted run resumes from disk, and a clean finish writes the final
``{slug}_params.json`` and deletes the journal.
"""
import glob
import json
import os
import types

import numpy as np
import optuna
import pandas as pd
import pytest

from openavmkit.tuning import (
    _resumable_study,
    _remaining_trials,
    _study_fingerprint,
    _discard_stale_studies,
    _cleanup_study_files,
    _seeded_sampler,
    _is_plateaued,
    _run_batched,
)
from openavmkit.modeling import _get_params
from openavmkit.utilities.settings import get_model_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _quadratic(trial):
    x = trial.suggest_float("x", 0.0, 1.0)
    return (x - 0.5) ** 2


def _fake_ds(cols=("a", "b"), n=6):
    """Minimal stand-in for a DataSplit exposing only what _get_params reads."""
    df = pd.DataFrame({c: list(range(n)) for c in cols})
    return types.SimpleNamespace(
        categorical_vars=[],
        X_train=df,
        y_train=list(range(n)),
        train_sizes=None,
        train_he_ids=None,
    )


# --------------------------------------------------------------------------- helpers


def test_remaining_trials_inmemory_runs_full_target():
    study = optuna.create_study()
    assert _remaining_trials(study, 5, None) == 5


def test_fingerprint_is_order_stable_and_context_sensitive():
    assert _study_fingerprint(["b", "a"], 10, 5) == _study_fingerprint(["a", "b"], 10, 5)
    assert _study_fingerprint(["a", "b"], 10, 5) != _study_fingerprint(["a", "b"], 11, 5)
    assert _study_fingerprint(["a", "b"], 10, 5) != _study_fingerprint(["a", "b"], 10, 6)
    assert _study_fingerprint(["a"], 10, 5) != _study_fingerprint(["a", "b"], 10, 5)
    # Seed is part of the search context: a different seed must not resume an old journal.
    assert _study_fingerprint(["a"], 10, 5, seed=42) != _study_fingerprint(["a"], 10, 5, seed=7)
    assert _study_fingerprint(["a"], 10, 5, seed=42) != _study_fingerprint(["a"], 10, 5, seed=None)


def test_seeded_sampler():
    # A seed -> seeded TPE sampler; None -> default sampler.
    assert isinstance(_seeded_sampler(42), optuna.samplers.TPESampler)
    assert isinstance(_seeded_sampler(42, constant_liar=True), optuna.samplers.TPESampler)
    assert _seeded_sampler(None) is None


def test_get_model_seed_always_returns_int():
    assert get_model_seed({}) == 42  # default
    assert get_model_seed({"modeling": {"metadata": {"seed": 7}}}) == 7
    # Determinism is always on: a null/absent seed falls back to the default.
    assert get_model_seed({"modeling": {"metadata": {"seed": None}}}) == 42


def test_run_batched_is_deterministic_and_parallel():
    """Batched ask-and-tell gives identical results run-to-run despite parallel eval."""

    def make_study():
        return optuna.create_study(
            direction="minimize", sampler=_seeded_sampler(42, constant_liar=True)
        )

    def suggest(trial):
        return {"x": trial.suggest_float("x", -5.0, 5.0), "y": trial.suggest_float("y", -5.0, 5.0)}

    def evaluate(p):
        return (p["x"] - 1.3) ** 2 + (p["y"] + 0.7) ** 2

    s1 = make_study()
    _run_batched(s1, suggest, evaluate, n_trials=24, storage_path=None, verbose=False, batch_size=8)
    s2 = make_study()
    _run_batched(s2, suggest, evaluate, n_trials=24, storage_path=None, verbose=False, batch_size=8)
    # Reproducible: identical best params AND identical stop point (plateau triggers
    # deterministically), even though the batch was evaluated in parallel.
    n1 = len([t for t in s1.trials if t.value is not None])
    n2 = len([t for t in s2.trials if t.value is not None])
    assert s1.best_params == s2.best_params
    assert n1 == n2
    assert 0 < n1 <= 24  # may stop early via plateau; never exceeds the target


def test_resumable_study_persists_and_resumes(tmp_path):
    sp = str(tmp_path / "m_study_abc.journal")
    study = _resumable_study("minimize", study_name="m", storage_path=sp)
    study.optimize(_quadratic, n_trials=3)
    assert len(study.trials) == 3

    # Reattaching to the same journal sees the prior trials and only needs the remainder.
    study2 = _resumable_study("minimize", study_name="m", storage_path=sp)
    assert len(study2.trials) == 3
    assert _remaining_trials(study2, 5, sp) == 2


def test_cleanup_removes_journal_and_sidecars(tmp_path):
    sp = str(tmp_path / "m_study_abc.journal")
    study = _resumable_study("minimize", study_name="m", storage_path=sp)
    study.optimize(_quadratic, n_trials=1)
    assert glob.glob(sp + "*")
    _cleanup_study_files(sp)
    assert glob.glob(sp + "*") == []
    # Idempotent / tolerant of already-gone files.
    _cleanup_study_files(sp)


def test_discard_stale_studies_keeps_matching(tmp_path):
    out = str(tmp_path)
    # One stale journal (old fingerprint) and one matching the keep fingerprint.
    open(os.path.join(out, "mdl_study_stale0.journal"), "w").close()
    keep_path = os.path.join(out, "mdl_study_keep01.journal")
    open(keep_path, "w").close()

    _discard_stale_studies(out, "mdl", keep="keep01")

    assert not os.path.exists(os.path.join(out, "mdl_study_stale0.journal"))
    assert os.path.exists(keep_path)


# ------------------------------------------------------------------ _get_params lifecycle


def _stub_tune(records_trials=2, then_raise=False):
    """Build a stub tune_func that exercises the journal exactly like a real tuner."""

    def tune_func(X, y, sizes=None, he_ids=None, verbose=False, cat_vars=None,
                  storage_path=None, study_name=None, **kwargs):
        if storage_path is not None:
            study = _resumable_study("minimize", study_name=study_name,
                                     storage_path=storage_path)
            study.optimize(_quadratic, n_trials=records_trials)
        if then_raise:
            raise RuntimeError("simulated interruption")
        return {"x": 0.5}

    return tune_func


def test_get_params_writes_final_and_removes_journal(tmp_path):
    out = str(tmp_path)
    params = _get_params(
        "Stub", "mymodel", _fake_ds(), _stub_tune(), out,
        save_params=True, use_saved_params=False, verbose=False, n_trials=4,
    )
    assert params == {"x": 0.5}
    assert os.path.exists(os.path.join(out, "mymodel_params.json"))
    # Final params written -> journal cleaned up.
    assert glob.glob(os.path.join(out, "mymodel_study_*.journal*")) == []


def test_get_params_interrupted_keeps_journal_and_writes_no_params(tmp_path):
    out = str(tmp_path)
    with pytest.raises(RuntimeError):
        _get_params(
            "Stub", "mymodel", _fake_ds(), _stub_tune(then_raise=True), out,
            save_params=True, use_saved_params=False, verbose=False, n_trials=4,
        )
    # Interruption leaves the journal on disk (resume trigger) and writes no final params.
    assert glob.glob(os.path.join(out, "mymodel_study_*.journal*"))
    assert not os.path.exists(os.path.join(out, "mymodel_params.json"))


def test_get_params_no_storage_when_not_saving(tmp_path):
    out = str(tmp_path)
    seen = {}

    def tune_func(X, y, sizes=None, he_ids=None, verbose=False, cat_vars=None,
                  storage_path=None, study_name=None, **kwargs):
        seen["storage_path"] = storage_path
        return {"x": 1.0}

    params = _get_params(
        "Stub", "mymodel", _fake_ds(), tune_func, out,
        save_params=False, use_saved_params=False, verbose=False, n_trials=4,
    )
    assert params == {"x": 1.0}
    # Ephemeral tuning: no journal requested, nothing written to disk.
    assert seen["storage_path"] is None
    assert glob.glob(os.path.join(out, "mymodel_study_*")) == []
    assert not os.path.exists(os.path.join(out, "mymodel_params.json"))


def _counting_tune(result=None):
    """A tune_func that records how many times it's actually invoked (i.e. re-tuned)."""
    calls = {"n": 0}

    def tune_func(X, y, sizes=None, he_ids=None, verbose=False, cat_vars=None,
                  storage_path=None, study_name=None, **kwargs):
        calls["n"] += 1
        return dict(result if result is not None else {"x": 0.5})

    return tune_func, calls


def test_get_params_embeds_fingerprint_and_reuses_on_match(tmp_path):
    out = str(tmp_path)
    tune, calls = _counting_tune()
    p1 = _get_params("Stub", "m", _fake_ds(), tune, out,
                     save_params=True, use_saved_params=False, verbose=False, n_trials=4)
    assert calls["n"] == 1
    # Saved file carries the fingerprint; the returned params do NOT (model never sees it).
    saved = json.load(open(os.path.join(out, "m_params.json")))
    assert "__fingerprint" in saved
    assert "__fingerprint" not in p1 and p1 == {"x": 0.5}

    # Same context -> reused, tuner NOT called again, and the fingerprint key is stripped.
    tune2, calls2 = _counting_tune()
    p2 = _get_params("Stub", "m", _fake_ds(), tune2, out,
                     save_params=True, use_saved_params=True, verbose=False, n_trials=4)
    assert calls2["n"] == 0
    assert "__fingerprint" not in p2 and p2 == {"x": 0.5}


def test_get_params_retunes_on_fingerprint_mismatch(tmp_path):
    out = str(tmp_path)
    tune, _ = _counting_tune()
    _get_params("Stub", "m", _fake_ds(), tune, out,
                save_params=True, use_saved_params=False, verbose=False, n_trials=4)
    # A changed trial budget (part of the fingerprint) must invalidate the saved params.
    tune2, calls2 = _counting_tune()
    _get_params("Stub", "m", _fake_ds(), tune2, out,
                save_params=True, use_saved_params=True, verbose=False, n_trials=8)
    assert calls2["n"] == 1


def test_get_params_retunes_on_legacy_params_without_fingerprint(tmp_path):
    out = str(tmp_path)
    os.makedirs(out, exist_ok=True)
    # A params.json saved before this guard existed (no "__fingerprint") is treated as stale.
    json.dump({"x": 9.9}, open(os.path.join(out, "m_params.json"), "w"))
    tune, calls = _counting_tune(result={"x": 0.5})
    p = _get_params("Stub", "m", _fake_ds(), tune, out,
                    save_params=True, use_saved_params=True, verbose=False, n_trials=4)
    assert calls["n"] == 1 and p == {"x": 0.5}


def test_seeded_tuning_is_reproducible():
    """The core determinism guarantee: same seed + same data -> identical best params."""
    from openavmkit.tuning import _tune_lightgbm

    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.rand(120), "b": rng.rand(120)})
    y = pd.Series(3 * X["a"] - 2 * X["b"] + rng.rand(120) * 0.1)

    p1 = _tune_lightgbm(X, y, sizes=None, he_ids=None, n_trials=6, random_state=42)
    p2 = _tune_lightgbm(X, y, sizes=None, he_ids=None, n_trials=6, random_state=42)
    assert p1 == p2


def test_get_params_resumes_to_target_then_cleans(tmp_path):
    """A journal left by an interrupted run is resumed, reaches the target, and is removed."""
    out = str(tmp_path)
    # First run is interrupted after recording 2 trials.
    with pytest.raises(RuntimeError):
        _get_params(
            "Stub", "mdl", _fake_ds(), _stub_tune(records_trials=2, then_raise=True), out,
            save_params=True, use_saved_params=False, verbose=False, n_trials=5,
        )
    journals = glob.glob(os.path.join(out, "mdl_study_*.journal"))
    assert len(journals) == 1
    study = _resumable_study("minimize", study_name="mdl", storage_path=journals[0])
    assert len(study.trials) == 2  # only the pre-interruption trials survived

    # Second run resumes the same journal (same fingerprint -> same filename) and finishes.
    params = _get_params(
        "Stub", "mdl", _fake_ds(), _stub_tune(records_trials=3), out,
        save_params=True, use_saved_params=False, verbose=False, n_trials=5,
    )
    assert params == {"x": 0.5}
    assert os.path.exists(os.path.join(out, "mdl_params.json"))
    assert glob.glob(os.path.join(out, "mdl_study_*.journal*")) == []
