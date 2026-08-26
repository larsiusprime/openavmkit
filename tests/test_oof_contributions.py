"""Tests for out-of-fold contribution artifacts under cross-validation.

The property under test is the one that was silently broken: in a CV run, every row of
``contributions_test.csv`` must be explained by a model that did not train on it, and its
``contribution_sum`` must reconstruct the ``prediction`` sitting beside it. Before the
out-of-fold stitch, the holdout frame carried per-fold predictions but production-model
attributions, so ``check_delta`` ran 8-35% of the prediction.

The fold models here are real (LayeredComp fit on disjoint two-thirds splits) rather than
mocked, because the whole point is that different models produce different base values.
"""

import os
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from layeredcompmodel import LayeredCompBaggingModel

from openavmkit.utilities.modeling import LayeredCompModel
from openavmkit.modeling import write_model_parameters
from openavmkit.model_runner import _model_artifact_dir, _stitch_oof_contributions
from openavmkit.shap_analysis import (
    _CONTRIB_NON_FEATURE_COLS,
    explanation_from_contributions,
)


N_FOLDS = 3


def _frame(n=90, seed=7):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "size": rng.uniform(800, 4000, n),
        "age": rng.uniform(0, 120, n),
        "quality": rng.uniform(1, 10, n),
    })
    y = pd.Series(0.05 * X["size"] - 0.4 * X["age"] + 20 * X["quality"] + rng.normal(0, 3, n))
    return X, y


def _fit(X, y, seed=0):
    bag = LayeredCompBaggingModel(tree_count=2, sample_pct=0.8, random_state=seed, n_jobs=1)
    bag.fit(X, y)
    return LayeredCompModel(bag)


def _smr(model, df, feats):
    """The slice of SingleModelResults that `write_model_parameters` actually reads."""
    return SimpleNamespace(
        ds=SimpleNamespace(ind_vars=feats, X_test=None, X_sales=None, X_univ=None),
        df_train=df,
        df_test=df,
        df_sales=df,
        df_universe=df,
    )


def _build_cv_layout(tmp_path, model_name="lcomp"):
    """Reproduce what `_run_cv_fold` leaves on disk: one holdout artifact per fold.

    Returns (outpath, prod_outpath, per-fold holdout frames, the production model).
    """
    X, y = _frame()
    feats = list(X.columns)
    fold_of = np.arange(len(X)) % N_FOLDS

    outpath = str(tmp_path / "models" / "res_sf" / "main")
    prod_outpath = f"{outpath}/cv_prod"

    holdouts = {}
    for k in range(N_FOLDS):
        is_holdout = fold_of == k
        # Fold model: trained on everything EXCEPT fold k.
        fold_model = _fit(X[~is_holdout], y[~is_holdout], seed=k)
        Xh = X[is_holdout].reset_index(drop=True)

        df = Xh.copy()
        df["key"] = [f"p{i}" for i in np.where(is_holdout)[0]]
        df["key_sale"] = df["key"] + "-a"
        # The holdout prediction comes from the fold's own model -- as it does in `pred_test`.
        df["prediction"] = fold_model.model.predict(Xh[feats])
        holdouts[k] = df

        d = _model_artifact_dir(f"{outpath}/cv_fold{k}", model_name)
        os.makedirs(d, exist_ok=True)
        write_model_parameters(fold_model, _smr(fold_model, df, feats), None, d,
                               subsets={"test"})

    prod_model = _fit(X, y, seed=99)
    return outpath, prod_outpath, holdouts, prod_model, X, feats


# --- the core guarantee ----------------------------------------------------

def test_stitched_oof_contributions_reconstruct_their_own_predictions(tmp_path):
    outpath, prod_outpath, holdouts, _prod, _X, _feats = _build_cv_layout(tmp_path)

    _stitch_oof_contributions(
        outpath, "lcomp", list(range(N_FOLDS)), prod_outpath, include_post_val=False
    )

    df = pd.read_csv(f"{outpath}/lcomp/contributions_test.csv")

    # Full coverage: every holdout row from every fold, exactly once.
    expected = {k for h in holdouts.values() for k in h["key_sale"]}
    assert set(df["key_sale"]) == expected
    assert len(df) == len(expected)

    # THE assertion: each row's attributions reconstruct the prediction beside it, because
    # both came from the same (fold) model. This is what ran 8-35% off before.
    rel = (df["check_delta"].abs() / df["prediction"].abs()).mean()
    assert rel < 0.01, f"stitched OOF contributions do not reconstruct: {rel:.1%}"


def test_production_model_would_not_reconstruct_the_same_frame(tmp_path):
    # The control. Explaining the same holdout rows with the production model -- the old
    # behavior -- leaves a large, silent gap. Without this, the test above proves nothing.
    outpath, _prod_outpath, holdouts, prod_model, _X, feats = _build_cv_layout(tmp_path)

    df_all = pd.concat(holdouts.values(), ignore_index=True)
    d = _model_artifact_dir(str(tmp_path / "prodwrite"), "lcomp")
    os.makedirs(d, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_model_parameters(prod_model, _smr(prod_model, df_all, feats), None, d,
                               subsets={"test"})

    got = pd.read_csv(f"{d}/contributions_test.csv")
    rel = (got["check_delta"].abs() / got["prediction"].abs()).mean()
    assert rel > 0.01, "control failed: production model reconstructs the fold predictions"
    # And P0's tripwire catches it.
    assert [w for w in caught if "do not reconstruct" in str(w.message)]


def test_oof_fold_column_tags_the_producing_model(tmp_path):
    outpath, prod_outpath, holdouts, _prod, _X, _feats = _build_cv_layout(tmp_path)
    _stitch_oof_contributions(
        outpath, "lcomp", list(range(N_FOLDS)), prod_outpath, include_post_val=False
    )
    df = pd.read_csv(f"{outpath}/lcomp/contributions_test.csv")

    assert set(df["oof_fold"]) == set(range(N_FOLDS))
    for k, holdout in holdouts.items():
        assert set(df.loc[df["oof_fold"] == k, "key_sale"]) == set(holdout["key_sale"])

    # base_value is per-model, so it must NOT be constant across folds -- that is precisely
    # why the file needs the oof_fold tag and why base_value isn't comparable row to row.
    per_fold_base = df.groupby("oof_fold")["base_value"].mean()
    assert per_fold_base.nunique() == N_FOLDS


# --- bookkeeping -----------------------------------------------------------

def test_oof_fold_is_not_mistaken_for_a_feature():
    assert "oof_fold" in _CONTRIB_NON_FEATURE_COLS
    df = pd.DataFrame({
        "key_sale": ["a", "b"],
        "base_value": [10.0, 10.0],
        "size": [1.0, 2.0],
        "age": [3.0, 4.0],
        "contribution_sum": [14.0, 16.0],
        "prediction": [14.0, 16.0],
        "check_delta": [0.0, 0.0],
        "oof_fold": [0, 1],
    })
    feats = pd.DataFrame({"key_sale": ["a", "b"], "size": [1.0, 2.0], "age": [3.0, 4.0]})
    expl = explanation_from_contributions(df, feats, key_col="key_sale")
    assert list(expl.feature_names) == ["size", "age"]
    assert expl.values.shape == (2, 2)


def test_model_artifact_dir_sanitizes_star():
    assert _model_artifact_dir("a/b", "plain") == "a/b/plain"
    assert _model_artifact_dir("a/b", "ens*mble") == "a/b/ens_starmble"


# --- stitcher edge cases ---------------------------------------------------

def _write_stub(path, keys, fold_tag_value=1.0, cols=("f1", "f2")):
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame({"key_sale": keys, "base_value": [fold_tag_value] * len(keys)})
    for c in cols:
        df[c] = 1.0
    df["contribution_sum"] = df["base_value"] + len(cols)
    df["prediction"] = df["contribution_sum"]
    df["check_delta"] = 0.0
    return df


def test_stitch_warns_and_continues_when_a_fold_is_missing(tmp_path):
    outpath = str(tmp_path / "main")
    for k in (0, 2):  # fold 1 never wrote anything
        d = _model_artifact_dir(f"{outpath}/cv_fold{k}", "m")
        _write_stub(d, [f"k{k}a", f"k{k}b"]).to_csv(f"{d}/contributions_test.csv", index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _stitch_oof_contributions(outpath, "m", [0, 1, 2], f"{outpath}/cv_prod",
                                  include_post_val=False)

    hits = [w for w in caught if "nothing written by fold(s) [1]" in str(w.message)]
    assert len(hits) == 1
    df = pd.read_csv(f"{outpath}/m/contributions_test.csv")
    assert set(df["oof_fold"]) == {0, 2}


def test_stitch_includes_post_val_rows_when_present(tmp_path):
    outpath = str(tmp_path / "main")
    prod_outpath = f"{outpath}/cv_prod"
    for k in (0, 1):
        d = _model_artifact_dir(f"{outpath}/cv_fold{k}", "m")
        _write_stub(d, [f"k{k}a"]).to_csv(f"{d}/contributions_test.csv", index=False)
    pv = _model_artifact_dir(prod_outpath, "m")
    _write_stub(pv, ["pv1", "pv2"]).to_csv(f"{pv}/contributions_test.csv", index=False)

    _stitch_oof_contributions(outpath, "m", [0, 1], prod_outpath, include_post_val=True)
    df = pd.read_csv(f"{outpath}/m/contributions_test.csv")
    # Post-valuation rows are tagged -1: explained by the Phase-2 refit, which never trained
    # on them either.
    assert set(df.loc[df["oof_fold"] == -1, "key_sale"]) == {"pv1", "pv2"}
    assert len(df) == 4


def test_stitch_carries_prefixed_variants(tmp_path):
    # log_ (MRA log-space) and std_ (NGBoost uncertainty) artifacts must come along without
    # being enumerated by name in the stitcher.
    outpath = str(tmp_path / "main")
    for k in (0, 1):
        d = _model_artifact_dir(f"{outpath}/cv_fold{k}", "m")
        for fn in ("contributions_test.csv", "log_contributions_test.csv",
                   "std_contributions_test.csv", "params_test.csv"):
            _write_stub(d, [f"k{k}a"]).to_csv(f"{d}/{fn}", index=False)

    _stitch_oof_contributions(outpath, "m", [0, 1], f"{outpath}/cv_prod",
                              include_post_val=False)
    for fn in ("contributions_test.csv", "log_contributions_test.csv",
               "std_contributions_test.csv", "params_test.csv"):
        assert os.path.exists(f"{outpath}/m/{fn}"), fn
        assert len(pd.read_csv(f"{outpath}/m/{fn}")) == 2


def test_stitch_warns_when_fold_columns_disagree(tmp_path):
    outpath = str(tmp_path / "main")
    d0 = _model_artifact_dir(f"{outpath}/cv_fold0", "m")
    _write_stub(d0, ["a"], cols=("f1", "f2")).to_csv(f"{d0}/contributions_test.csv", index=False)
    d1 = _model_artifact_dir(f"{outpath}/cv_fold1", "m")
    _write_stub(d1, ["b"], cols=("f1", "f3")).to_csv(f"{d1}/contributions_test.csv", index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _stitch_oof_contributions(outpath, "m", [0, 1], f"{outpath}/cv_prod",
                                  include_post_val=False)
    assert [w for w in caught if "differing columns" in str(w.message)]


def test_stitch_is_a_noop_when_there_is_nothing_to_stitch(tmp_path):
    # Engines with no per-subset artifacts (LocalAreaModel) legitimately write nothing.
    outpath = str(tmp_path / "main")
    os.makedirs(outpath, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _stitch_oof_contributions(outpath, "m", [0, 1], f"{outpath}/cv_prod",
                                  include_post_val=False)
    assert not caught
    assert not os.path.exists(f"{outpath}/m/contributions_test.csv")


# --- P4: Phase 2 must not explain the out-of-fold holdout frame ------------

def _fake_results(model_name="lcomp"):
    df = pd.DataFrame({"key": ["a"], "key_sale": ["a-1"], "prediction": [1.0]})
    return SimpleNamespace(
        model_name=model_name,
        model="stub",
        df_sales=df,
        df_universe=pd.DataFrame({"key": ["a"], "prediction": [1.0]}),
        pred_test={"stub": 1},
        pred_sales={"stub": 2},
        pred_univ={"stub": 3},
    )


def _patch_heavy(monkeypatch, recorder):
    import openavmkit.model_runner as mr
    monkeypatch.setattr(mr, "_assemble_model_results", lambda results, settings: {})
    monkeypatch.setattr(mr, "_write_open_ratio_study", lambda results, path, settings: None)
    monkeypatch.setattr(
        mr, "write_model_parameters",
        lambda model, smr, location, outpath, verbose=False, subsets=None: recorder.append(subsets),
    )


def test_write_model_results_forwards_subsets(tmp_path, monkeypatch):
    import openavmkit.model_runner as mr
    monkeypatch.chdir(tmp_path)
    seen = []
    _patch_heavy(monkeypatch, seen)

    # CV holds back "test" so the stitched out-of-fold file is not clobbered by a Phase-2
    # explanation of the same rows.
    mr._write_model_results(
        _fake_results(), str(tmp_path / "out"), {}, None,
        subsets={"train", "sales", "universe"},
    )
    assert seen == [{"train", "sales", "universe"}]
    assert "test" not in seen[0]


def test_write_model_results_defaults_to_every_subset(tmp_path, monkeypatch):
    # The single-split path is unchanged: no `subsets` argument means write them all.
    import openavmkit.model_runner as mr
    monkeypatch.chdir(tmp_path)
    seen = []
    _patch_heavy(monkeypatch, seen)
    mr._write_model_results(_fake_results(), str(tmp_path / "out"), {}, None)
    assert seen == [None]


def test_cv_phase2_write_subsets_rule():
    from openavmkit.model_runner import _cv_phase2_write_subsets

    # Normal CV: Phase 2 writes everything EXCEPT test, which the stitcher supplies.
    assert _cv_phase2_write_subsets(True, 5) == {"train", "sales", "universe"}
    # Feature switched off -> legacy behavior, Phase 2 writes all four.
    assert _cv_phase2_write_subsets(False, 5) is None
    # Every fold failed -> there is no OOF frame; Phase 2's own holdout stands, so it must
    # write test or the file would simply be missing.
    assert _cv_phase2_write_subsets(True, 0) is None
    assert _cv_phase2_write_subsets(False, 0) is None
