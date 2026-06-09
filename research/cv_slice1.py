"""Temporal rolling-origin CV — proof harness (Slices 1, 2a, 2b).

Proves, end-to-end, for ONE backdated origin tᵢ on Petersburg `single_family_suburban`:

  Slice 1  — pipeline runs FROM the 1-assemble checkpoint with a backdated valuation date, and the
             ≤tᵢ pre-clean filter makes the fold leakage-clean (time adjustment + the whole clean
             stage only ever see sales ≤ tᵢ).
  Slice 2a — the held-out future window (tᵢ, tᵢ+1yr] is scored against RAW sale_price after
             validity screening (`valid_for_ratio_study`) — the IAAO §4.4 convention.
  Slice 2b — the fold runs in an ISOLATED working dir (out/cv/fold_<tᵢ>/) so it never clobbers the
             jurisdiction's real out/; fold-local cache (no cross-fold contamination).

NOT yet: the K-fold loop + mean±CI aggregator (Slice 3), parallelism / orchestrator promotion (4).

Usage:
    python research/cv_slice1.py                # default origin 2024-07-01
    python research/cv_slice1.py 2024-04-01
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

try:  # Windows consoles default to cp1252; our status lines use unicode (tᵢ, ≤, ∅)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SLUG = "us-va-petersburgcity"
GROUP = "single_family_suburban"
RUN_LIST = ["assessor", "mra", "lcomp"]   # fast (no Optuna tuning) — proof, not the final menu
TEST_WINDOW_DAYS = 365                      # ≤1yr held-out future window (IAAO §4.4)


def _ok(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    return bool(cond)


def _v(x):
    # RatioStudyBootstrapped returns ConfidenceStat (value + CI); plain RatioStudy returns floats.
    return x.value if hasattr(x, "value") else x


def _link_dir(src, dst):
    """Make `dst` point at directory `src` without copying. Junction on Windows (no admin),
    symlink elsewhere."""
    if os.path.exists(dst):
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True, capture_output=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def _snapshot_mtimes(root):
    """Map of path -> mtime for every file under `root` (for the isolation assertion)."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                out[p] = os.path.getmtime(p)
            except OSError:
                pass
    return out


def run_fold(pl, settings, sup_assemble, origin, locality_dir):
    """Run one leakage-clean temporal fold in an isolated working dir; return its
    {group: MultiModelResults}. The fold sees ONLY sales ≤ origin through clean+model."""
    fold_dir = os.path.join(locality_dir, "out", "cv", f"fold_{origin.date()}")
    os.makedirs(os.path.join(fold_dir, "out"), exist_ok=True)
    _link_dir(os.path.join(locality_dir, "in"), os.path.join(fold_dir, "in"))

    settings.setdefault("modeling", {}).setdefault("metadata", {})["valuation_date"] = str(origin.date())

    fold = sup_assemble.copy()
    sales = fold.sales.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    fold.set("sales", sales[sales["sale_date"].le(origin)])

    prev_cwd = os.getcwd()
    os.chdir(fold_dir)
    try:
        # clean stage (notebook 02), valuation_date = origin
        fold = pl.fill_unknown_values_sup(fold, settings)
        fold = pl.mark_horizontal_equity_clusters_per_model_group_sup(
            fold, settings, verbose=False, do_land_clusters=True, do_impr_clusters=True)
        fold = pl.process_sales(fold, settings, verbose=False)
        fold = pl.run_sales_scrutiny(
            fold, settings, drop_cluster_outliers=False, drop_heuristic_outliers=False, verbose=False)
        fold = pl.collapse_sparse_categories_sup(fold, settings)
        # model stage (notebook 03)
        pl.write_canonical_splits(fold, settings, verbose=False)
        fold = pl.enrich_sup_spatial_lag(fold, settings, verbose=False)
        try:
            fold = pl.enrich_sup_area_stats(fold, settings, verbose=False)
        except Exception as e:
            print(f"[fold] (area_stats skipped: {type(e).__name__}: {str(e)[:80]})")
        results = pl.run_models(
            fold, settings,
            save_params=False, use_saved_params=False, save_results=True,
            verbose=False, run_main=True, run_vacant=False, run_ensemble=False,
            do_shaps=False, do_plots=False)
    finally:
        os.chdir(prev_cwd)
    return results, fold, fold_dir


def make_origins(train_dates, screened_test_dates, window_days, min_train, min_test):
    """Data-driven fold-spec generator: semi-annual (Jan 1 / Jul 1) origins that satisfy both the
    min-train (expanding window ≤ tᵢ) and min-test (screened ≤1yr window after tᵢ) guards."""
    dmin, dmax = train_dates.min(), train_dates.max()
    window = pd.Timedelta(days=window_days)
    origins = []
    for yr in range(dmin.year, dmax.year + 1):
        for mo in (1, 7):
            t = pd.Timestamp(yr, mo, 1)
            if t <= dmin or t >= dmax:
                continue
            n_train = int((train_dates <= t).sum())
            n_test = int(((screened_test_dates > t) & (screened_test_dates <= t + window)).sum())
            if n_train >= min_train and n_test >= min_test:
                origins.append((t, n_train, n_test))
    return origins


def main():
    import openavmkit.pipeline as pl
    from openavmkit.ratio_study import RatioStudy

    MIN_TRAIN, MIN_TEST = 100, 30
    locality_dir = os.path.join(REPO_ROOT, "notebooks", "pipeline", "data", SLUG)
    os.chdir(locality_dir)

    settings = pl.load_settings("in/settings.json")
    instr = settings.setdefault("modeling", {}).setdefault("instructions", {})
    instr["model_groups"] = [GROUP]
    instr.setdefault("main", {})["run"] = RUN_LIST

    print("[cv] loading checkpoints (1-assemble for training, 2-clean for screened held-out) ...")
    sup_assemble = pl.read_pickle("out/1-assemble-sup")
    clean = pl.read_pickle("out/2-clean-sup")

    # GROUP-SCOPE everything (guards + test pool) to single_family_suburban parcels — counting
    # jurisdiction-wide sales lets thin early folds (where the group had ~no sales) slip past the
    # min-train guard and train garbage models. Link sales->group via the universe's model_group.
    group_keys = set(clean.universe.loc[clean.universe["model_group"].eq(GROUP), "key"])
    print(f"[cv] {GROUP}: {len(group_keys)} parcels in group")

    # Count CLEAN/VALID group sales for the guards — clean drops invalid sales (cleaning.py:142),
    # so clean.sales is the train-eligible set. (1-assemble includes invalids that never train, which
    # would let thin early folds slip the min-train guard and train garbage models.)
    cg = clean.sales.copy()
    cg["sale_date"] = pd.to_datetime(cg["sale_date"], errors="coerce")
    cg = cg[cg["key"].isin(group_keys)]                    # group-scoped, valid (clean-filtered)
    train_dates = cg["sale_date"].dropna()                 # train-count proxy = valid group sales

    cs = cg.copy()
    if "valid_for_ratio_study" in cs.columns:
        cs = cs[cs["valid_for_ratio_study"].eq(True)]      # screened held-out pool (IAAO)
    cs = cs[cs["sale_price"].gt(0)]

    origins = make_origins(train_dates, cs["sale_date"], TEST_WINDOW_DAYS, MIN_TRAIN, MIN_TEST)
    print(f"[cv] {SLUG}/{GROUP}: {len(origins)} qualifying origins "
          f"(min_train={MIN_TRAIN}, min_test={MIN_TEST}, window={TEST_WINDOW_DAYS}d):")
    for t, ntr, nte in origins:
        print(f"      tᵢ={t.date()}  train≤tᵢ≈{ntr}  screened-test≈{nte}")
    if not origins:
        print("[cv] no qualifying origins — widen the data or relax guards."); return

    prod_out = os.path.join(locality_dir, "out", "models", GROUP)
    before = _snapshot_mtimes(prod_out)

    per_fold = []          # (origin, model, n_test, median_ratio, cod, cod_trim)
    all_pass = True
    for t, _ntr, _nte in origins:
        window_end = t + pd.Timedelta(days=TEST_WINDOW_DAYS)
        print(f"\n[cv] === fold tᵢ={t.date()} ===")
        results, fold, fold_dir = run_fold(pl, settings, sup_assemble, t, locality_dir)

        # per-fold leakage assertions
        tr_max = pd.to_datetime(fold.sales["sale_date"], errors="coerce").max()
        all_pass &= _ok(tr_max <= t, f"max(train sale_date)={tr_max.date()} ≤ tᵢ")
        ta_path = os.path.join(fold_dir, "out", "time_adjustment", GROUP, "time_adjustment_schedule.csv")
        if os.path.exists(ta_path):
            ta_max = pd.to_datetime(pd.read_csv(ta_path)["period"], errors="coerce").max()
            all_pass &= _ok(ta_max <= t, f"fold time-adjustment max period={ta_max.date()} ≤ tᵢ")

        # Score the held-out window by REUSING the formal ratio study (pipeline.run_ratio_study):
        # it natively group-filters, date-windows, applies valid_for_ratio_study (so vacant /
        # improved-mismatch sales are excluded by the library, not hand-filtered), drops
        # non-positives, and scores raw sale_price (IAAO §4.4). We overlay each model's universe
        # prediction onto the clean SUP and let the library do the rest.
        mm = results.get(GROUP) if isinstance(results, dict) else results
        w_end = t + pd.Timedelta(days=TEST_WINDOW_DAYS)
        for name, smr in mm.model_results.items():
            up = getattr(smr, "df_universe", None)
            if up is None or "prediction" not in up.columns:
                continue
            su = clean.copy()
            su.set("universe", su.universe.drop(columns=["prediction"], errors="ignore").merge(
                up[["key", "prediction"]], on="key", how="left"))
            rs = pl.run_ratio_study(su, GROUP, "prediction", "sale_price",
                                    start_date=str((t + pd.Timedelta(days=1)).date()),
                                    end_date=str(w_end.date()), land_only=False, max_trim=0.15)
            per_fold.append((t.date(), name, rs.count, _v(rs.median_ratio), _v(rs.cod), _v(rs.cod_trim)))

    # isolation across all folds
    print("\n[cv] ISOLATION ASSERTION (Slice 2b)")
    all_pass &= _ok(before == _snapshot_mtimes(prod_out),
                    f"jurisdiction out/models/{GROUP} untouched across all folds")

    pf = pd.DataFrame(per_fold, columns=["origin", "model", "n_test", "median_ratio", "cod", "cod_trim"])
    print("\n[cv] PER-FOLD (held-out, via pipeline.run_ratio_study — valid_for_ratio_study applied natively):")
    print(pf.round(3).to_string(index=False))

    # across-fold aggregate: mean ± std per model (Slice 3 headline)
    agg = pf.groupby("model").agg(
        folds=("cod", "size"),
        median_ratio=("median_ratio", "mean"),
        cod_mean=("cod", "mean"), cod_std=("cod", "std"),
        cod_trim_mean=("cod_trim", "mean"), cod_trim_std=("cod_trim", "std"),
    ).reset_index()
    disp = pd.DataFrame({"model": agg["model"], "folds": agg["folds"],
                         "median_ratio": agg["median_ratio"].round(3)})
    disp["cod (mean±std)"] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(agg["cod_mean"], agg["cod_std"].fillna(0))]
    disp["cod_trim (mean±std)"] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(agg["cod_trim_mean"], agg["cod_trim_std"].fillna(0))]
    print(f"\n[cv] ACROSS-FOLD AGGREGATE ({len(origins)} folds) — rolling-origin CV (run_ratio_study):")
    print(disp.to_string(index=False))

    print(f"\n[cv] {'ALL ASSERTIONS PASSED' if all_pass else 'SOME ASSERTIONS FAILED'}")


if __name__ == "__main__":
    main()
