"""What a run of the models keeps: a try lets each group's results go unless asked to return them."""

import pandas as pd

import openavmkit.model_runner as mr
from openavmkit import pipeline


def _run(monkeypatch, **kwargs):
    made = []

    def fake(sup, model_group, *args, **kw):
        made.append(model_group)
        return f"results of {model_group}"

    monkeypatch.setattr(mr, "_run_models", fake)
    monkeypatch.setattr(mr, "write_out_all_results", lambda *a, **k: None)
    settings = {"modeling": {"instructions": {"model_groups": ["sf", "condo"]}}}
    sup = {"universe": pd.DataFrame({"key": ["1"]})}
    out = mr.run_models(sup, settings, run_vacant=False, **kwargs)
    assert made == ["sf", "condo"]
    return out


def test_a_try_keeps_nothing_unless_asked(monkeypatch):
    assert _run(monkeypatch) == {}
    expected = {"sf": "results of sf", "condo": "results of condo"}
    assert _run(monkeypatch, return_results=True) == expected
    # Results being written are kept to write them, and returned.
    assert _run(monkeypatch, save_results=True) == expected


def test_try_models_returns_results_only_when_asked(monkeypatch):
    seen = {}

    def fake_run_models(*args, **kwargs):
        seen.update(kwargs)
        return {"sf": "results"} if kwargs.get("return_results") else {}

    monkeypatch.setattr(mr, "run_models", fake_run_models)
    assert pipeline.try_models(None, {}) is None and seen["return_results"] is False
    assert pipeline.try_models(None, {}, return_results=True) == {"sf": "results"}
