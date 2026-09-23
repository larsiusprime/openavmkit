"""Tests for assessor-holdout handling: the `assessor_holdout` mode, user-supplied
test keys (`modeling.instructions.test_keys_file`), and the benchmark drop/keep gating."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openavmkit.model_runner as mr
from openavmkit.data import _do_write_canonical_split, _read_provided_test_keys
from openavmkit.utilities.settings import get_assessor_holdout_mode


def test_assessor_holdout_mode_default_and_values():
    assert get_assessor_holdout_mode({}) == "unknown"
    assert (
        get_assessor_holdout_mode(
            {"analysis": {"ratio_study": {"assessor_holdout": "shared"}}}
        )
        == "shared"
    )
    # Case-insensitive.
    assert (
        get_assessor_holdout_mode(
            {"analysis": {"ratio_study": {"assessor_holdout": "Shared"}}}
        )
        == "shared"
    )


def test_read_provided_test_keys_key_sale_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()
    pd.DataFrame({"key_sale": ["a", "b", "c"], "other": [1, 2, 3]}).to_csv(
        tmp_path / "in" / "keys.csv", index=False
    )
    assert _read_provided_test_keys("keys.csv") == {"a", "b", "c"}


def test_read_provided_test_keys_single_column_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()
    pd.DataFrame({"whatever": ["x", "y"]}).to_csv(
        tmp_path / "in" / "keys.csv", index=False
    )
    assert _read_provided_test_keys("keys.csv") == {"x", "y"}


def test_read_provided_test_keys_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        _read_provided_test_keys("nope.csv")


def test_custom_test_keys_define_the_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()

    n = 20
    keys = [f"k{i}" for i in range(n)]
    # First 15 are pre-valuation (sale_age_days >= 0), last 5 are post-valuation (< 0).
    sale_age = [10] * 15 + [-10] * 5
    df_sales = pd.DataFrame(
        {
            "key_sale": keys,
            "model_group": ["sf"] * n,
            "sale_age_days": sale_age,
            "vacant_sale": [False] * n,
        }
    )

    # User supplies a holdout of three pre-valuation keys.
    provided = ["k0", "k1", "k2"]
    pd.DataFrame({"key_sale": provided}).to_csv(
        tmp_path / "in" / "holdout.csv", index=False
    )

    settings = {"modeling": {"instructions": {"test_keys_file": "holdout.csv"}}}
    _do_write_canonical_split("sf", df_sales, settings, verbose=False)

    test_keys = set(
        pd.read_csv(tmp_path / "out/models/sf/_data/test_keys.csv")["key_sale"].astype(str)
    )
    train_keys = set(
        pd.read_csv(tmp_path / "out/models/sf/_data/train_keys.csv")["key_sale"].astype(str)
    )

    # Test set == exactly the provided keys.
    assert test_keys == set(provided)
    # Train set == all other PRE-valuation sales; post-valuation sales never train and
    # weren't listed, so they appear in neither set.
    expected_train = {f"k{i}" for i in range(3, 15)}
    assert train_keys == expected_train
    post_val = {f"k{i}" for i in range(15, 20)}
    assert not (train_keys & post_val)


def _fake_smr():
    """Duck-typed stand-in exposing only the attributes `_calc_benchmark` reads."""
    rs = SimpleNamespace(
        count=100,
        median_ratio=1.0,
        cod=10.0,
        prd=1.0,
        prb=0.0,
        count_trim=95,
        cod_trim=8.0,
        prd_trim=1.0,
        prb_trim=0.0,
    )
    pred = SimpleNamespace(ratio_study=rs)
    return SimpleNamespace(
        pred_test=pred,
        pred_sales_lookback=pred,
        utility_test=1.0,
        utility_train=1.0,
        df_universe=pd.DataFrame({"x": [1, 2, 3]}),
        ve_test={"vei": 0.0, "vei_significance": 0.0},
        ve_sales_lookback={"vei": 0.0, "vei_significance": 0.0},
        chd=5.0,
        timing=SimpleNamespace(results={}),
    )


def test_calc_benchmark_drops_assessor_from_test_when_flagged(monkeypatch):
    # Identity post-valuation transform so we don't need a real DataSplit.
    monkeypatch.setattr(mr, "_get_post_valuation_smr", lambda smr, verbose=False: smr)
    results = {"main": _fake_smr(), "assessor": _fake_smr()}

    dropped = mr._calc_benchmark(results, drop_assessor_from_test=True)
    assert "assessor" not in dropped.df_stats_test.index  # off the random holdout
    assert "assessor" in dropped.df_stats_test_post_val.index  # kept on post-valuation
    assert "assessor" in dropped.df_stats_full.index  # kept on study set
    assert dropped.assessor_in_test is False

    kept = mr._calc_benchmark(results, drop_assessor_from_test=False)
    assert "assessor" in kept.df_stats_test.index
    assert kept.assessor_in_test is True


def test_calc_benchmark_default_keeps_assessor(monkeypatch):
    # Default flag is False (keep) -- correct for the post-valuation benchmark path.
    monkeypatch.setattr(mr, "_get_post_valuation_smr", lambda smr, verbose=False: smr)
    results = {"main": _fake_smr(), "assessor": _fake_smr()}
    b = mr._calc_benchmark(results)
    assert "assessor" in b.df_stats_test.index
    assert b.assessor_in_test is True


def test_add_model_preserves_assessor_drop(monkeypatch):
    # Regression: adding the ensemble model recomputes the benchmark; the assessor-drop
    # choice must persist, or the ensemble step silently re-introduces the assessor into
    # the Test-set comparison (observed on Petersburg).
    monkeypatch.setattr(mr, "_get_post_valuation_smr", lambda smr, verbose=False: smr)
    results = {"main": _fake_smr(), "assessor": _fake_smr()}
    bench = mr._calc_benchmark(results, drop_assessor_from_test=True)
    mmr = mr.MultiModelResults(
        results,
        bench,
        df_univ=pd.DataFrame({"x": [1]}),
        df_sales=pd.DataFrame({"x": [1]}),
        drop_assessor_from_test=True,
    )
    assert "assessor" not in mmr.benchmark.df_stats_test.index

    mmr.add_model("ensemble", _fake_smr())
    assert "assessor" not in mmr.benchmark.df_stats_test.index
    assert mmr.benchmark.assessor_in_test is False


def test_benchmark_print_caveat_varies_with_assessor_in_test(monkeypatch):
    # Stub the table formatter so we test only the caveat logic, not numeric formatting.
    monkeypatch.setattr(mr, "_format_benchmark_df", lambda df, transpose=True: "")
    df = pd.DataFrame({"count_sales": [10]}, index=["main"])
    shown = mr.BenchmarkResults(df, df, df, df, assessor_in_test=True).print()
    hidden = mr.BenchmarkResults(df, df, df, df, assessor_in_test=False).print()
    assert "Assessor shown here" in shown
    assert "Assessor not shown here" in hidden
