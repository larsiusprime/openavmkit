"""Temporal rolling-origin cross-validation orchestrator for OpenAVMKit (research / publish mode).

Promoted from the cv_slice1 proof harness (Slices 1-3). For each backdated origin tᵢ it runs a
leakage-clean, isolated fold and scores the held-out future window with the library's own ratio
study, then aggregates across folds to mean ± std.

Guarantees, per fold:
  * Leakage-clean: the sup is filtered to sale_date ≤ tᵢ BEFORE the clean stage, so time-adjustment,
    variable-selection and scrutiny only ever see the past (the held-out window is reserved for
    scoring). Verified by assertions (train max date ≤ tᵢ; fold time-adjustment schedule ≤ tᵢ).
  * Isolated: the fold runs in out/cv/fold_<tᵢ>/ (own out/ + cache/), so it never touches the
    jurisdiction's real out/. `in/` is shared via a directory junction (Windows) / symlink.
  * IAAO-faithful scoring: pipeline.run_ratio_study (native valid_for_ratio_study / vacant split /
    raw sale_price / trimming) over (tᵢ, tᵢ+window].

Group selection, train/test guards, and the held-out pool all use the library's `get_sup_model_group`
(single source of truth for "what's in the group").

Usage:
    python research/cross_validation.py us-va-petersburgcity --group single_family_suburban
    python research/cross_validation.py us-va-petersburgcity --group single_family_suburban \
        --run assessor mra lcomp xgboost lightgbm catboost ngboost --jobs 4
    python research/cross_validation.py us-co-eagle --group single_family --run assessor mra lcomp
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# --------------------------------------------------------------------------- helpers
def _v(x):
    """RatioStudyBootstrapped stats are ConfidenceStat (value + CI); plain RatioStudy are floats."""
    return x.value if hasattr(x, "value") else x


def _link_dir(src, dst):
    """Point `dst` at directory `src` without copying: junction on Windows (no admin), symlink else."""
    if os.path.exists(dst):
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True, capture_output=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def _group_sales(pl, sup, group):
    """Group-scoped sales via the library's own scoping (get_sup_model_group)."""
    return pl.get_sup_model_group(sup, group).sales


def make_origins(train_dates, test_dates, window_days, min_train, min_test, step_months=6):
    """Semi-annual origins satisfying both guards (group-scoped counts).

    train_dates : valid group sale dates (expanding-window training pool, ≤ tᵢ)
    test_dates  : screened group sale dates eligible for the held-out window
    """
    dmin, dmax = train_dates.min(), train_dates.max()
    window = pd.Timedelta(days=window_days)
    months = list(range(1, 13, max(1, step_months)))
    origins = []
    for yr in range(dmin.year, dmax.year + 1):
        for mo in months:
            t = pd.Timestamp(yr, mo, 1)
            if t <= dmin or t >= dmax:
                continue
            n_train = int((train_dates <= t).sum())
            n_test = int(((test_dates > t) & (test_dates <= t + window)).sum())
            if n_train >= min_train and n_test >= min_test:
                origins.append((t, n_train, n_test))
    return origins


def _skip_shap_contributions():
    """The CV scores predictions only (via run_ratio_study) and never uses per-feature SHAP
    contributions. Computing them (write_model_parameters -> write_shaps, then the contributions
    map) is the slowest part of each fold on big universes, so no-op it. Predictions are unaffected
    (they're computed in run_one_model, before this write step)."""
    import openavmkit.model_runner as _mr
    _mr.write_model_parameters = lambda *a, **k: None


def _set_n_trials(node, n):
    """Recursively stamp n_trials on every dict under modeling.models (harmless where ignored).
    Reaches per-model entries AND per-model-group override blocks so the cap actually applies."""
    if isinstance(node, dict):
        node["n_trials"] = n
        for v in node.values():
            _set_n_trials(v, n)


def run_fold(pl, settings, sup_assemble, origin, locality_dir, group, run_list, n_trials=None):
    """One leakage-clean, isolated temporal fold. Returns {group: MultiModelResults}, fold sup, dir."""
    fold_dir = os.path.join(locality_dir, "out", "cv", f"fold_{origin.date()}")
    os.makedirs(os.path.join(fold_dir, "out"), exist_ok=True)
    _link_dir(os.path.join(locality_dir, "in"), os.path.join(fold_dir, "in"))

    settings.setdefault("modeling", {}).setdefault("metadata", {})["valuation_date"] = str(origin.date())
    instr = settings["modeling"].setdefault("instructions", {})
    instr["model_groups"] = [group]
    instr.setdefault("main", {})["run"] = list(run_list)
    if n_trials is not None:
        _set_n_trials(settings["modeling"].setdefault("models", {}).setdefault("main", {}), n_trials)
    _skip_shap_contributions()   # CV needs predictions only — skip the costly SHAP params/contribs

    fold = sup_assemble.copy()
    sales = fold.sales.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    fold.set("sales", sales[sales["sale_date"].le(origin)])   # ≤tᵢ pre-clean filter (the leak fix)

    prev_cwd = os.getcwd()
    os.chdir(fold_dir)
    try:
        fold = pl.fill_unknown_values_sup(fold, settings)
        fold = pl.mark_horizontal_equity_clusters_per_model_group_sup(
            fold, settings, verbose=False, do_land_clusters=True, do_impr_clusters=True)
        fold = pl.process_sales(fold, settings, verbose=False)
        fold = pl.run_sales_scrutiny(
            fold, settings, drop_cluster_outliers=False, drop_heuristic_outliers=False, verbose=False)
        fold = pl.collapse_sparse_categories_sup(fold, settings)
        pl.write_canonical_splits(fold, settings, verbose=False)
        fold = pl.enrich_sup_spatial_lag(fold, settings, verbose=False)
        try:
            fold = pl.enrich_sup_area_stats(fold, settings, verbose=False)
        except Exception as e:
            print(f"[fold {origin.date()}] (area_stats skipped: {type(e).__name__}: {str(e)[:80]})")
        results = pl.run_models(
            fold, settings, save_params=False, use_saved_params=False, save_results=True,
            verbose=False, run_main=True, run_vacant=False, run_ensemble=False,
            do_shaps=False, do_plots=False)
    finally:
        os.chdir(prev_cwd)
    return results, fold, fold_dir


def score_fold(pl, clean, results, group, origin, window_days, max_trim):
    """Score the held-out (tᵢ, tᵢ+window] window per model via the library's run_ratio_study.
    Returns (rows, leakage_ok)."""
    mm = results.get(group) if isinstance(results, dict) else results
    w_end = origin + pd.Timedelta(days=window_days)
    rows = []
    for name, smr in mm.model_results.items():
        up = getattr(smr, "df_universe", None)
        if up is None or "prediction" not in up.columns:
            continue
        su = clean.copy()
        su.set("universe", su.universe.drop(columns=["prediction"], errors="ignore").merge(
            up[["key", "prediction"]], on="key", how="left"))
        rs = pl.run_ratio_study(su, group, "prediction", "sale_price",
                                start_date=str((origin + pd.Timedelta(days=1)).date()),
                                end_date=str(w_end.date()), land_only=False, max_trim=max_trim)
        rows.append((origin.date(), name, rs.count, _v(rs.median_ratio), _v(rs.cod), _v(rs.cod_trim)))
    return rows


def _fold_worker(args):
    """Picklable entry for ProcessPoolExecutor: loads its own data, runs+scores one fold."""
    slug, group, run_list, origin_str, window_days, max_trim, n_trials = args
    import openavmkit.pipeline as pl
    origin = pd.Timestamp(origin_str)
    locality_dir = os.path.join(REPO_ROOT, "notebooks", "pipeline", "data", slug)
    os.chdir(locality_dir)
    settings = pl.load_settings("in/settings.json")
    sup_assemble = pl.read_pickle("out/1-assemble-sup")
    clean = pl.read_pickle("out/2-clean-sup")
    results, fold, fold_dir = run_fold(pl, settings, sup_assemble, origin, locality_dir, group, run_list, n_trials)
    tr_max = pd.to_datetime(fold.sales["sale_date"], errors="coerce").max()
    ta_path = os.path.join(fold_dir, "out", "time_adjustment", group, "time_adjustment_schedule.csv")
    ta_max = None
    if os.path.exists(ta_path):
        ta_max = pd.to_datetime(pd.read_csv(ta_path)["period"], errors="coerce").max()
    leak_ok = (tr_max <= origin) and (ta_max is None or ta_max <= origin)
    rows = score_fold(pl, clean, results, group, origin, window_days, max_trim)
    return origin_str, rows, bool(leak_ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("--group", required=True)
    ap.add_argument("--run", nargs="+", default=["assessor", "mra", "lcomp"], help="model run list")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--min-train", type=int, default=100)
    ap.add_argument("--min-test", type=int, default=30)
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--max-origins", type=int, default=5,
                    help="cap the number of folds; if more qualify, keep the most recent N")
    ap.add_argument("--max-trim", type=float, default=0.15)
    ap.add_argument("--n-trials", type=int, default=None,
                    help="cap Optuna trials for tunable models (per fold); e.g. 3 for a fast CV")
    ap.add_argument("--jobs", type=int, default=1, help="parallel folds (separate processes)")
    args = ap.parse_args()

    import openavmkit.pipeline as pl

    locality_dir = os.path.join(REPO_ROOT, "notebooks", "pipeline", "data", args.slug)
    os.chdir(locality_dir)
    settings = pl.load_settings("in/settings.json")
    clean = pl.read_pickle("out/2-clean-sup")
    sup_assemble = pl.read_pickle("out/1-assemble-sup")

    # group-scoped guard inputs via the library's own scoping (#1 cleanup)
    train_dates = pd.to_datetime(_group_sales(pl, clean, args.group)["sale_date"], errors="coerce").dropna()
    test_pool = _group_sales(pl, clean, args.group).copy()
    test_pool["sale_date"] = pd.to_datetime(test_pool["sale_date"], errors="coerce")
    if "valid_for_ratio_study" in test_pool.columns:
        test_pool = test_pool[test_pool["valid_for_ratio_study"].eq(True)]
    test_pool = test_pool[test_pool["sale_price"].gt(0)]

    origins = make_origins(train_dates, test_pool["sale_date"], args.window_days,
                           args.min_train, args.min_test, args.step_months)
    n_qualified = len(origins)
    if n_qualified > args.max_origins:           # keep the most recent N (most relevant / data-rich)
        origins = origins[-args.max_origins:]
    print(f"[cv] {args.slug}/{args.group}: using {len(origins)} of {n_qualified} qualifying origins "
          f"(cap={args.max_origins}; min_train={args.min_train}, min_test={args.min_test}, "
          f"window={args.window_days}d, run={args.run})")
    for t, ntr, nte in origins:
        print(f"      tᵢ={t.date()}  train≤tᵢ={ntr}  test={nte}")
    if not origins:
        print("[cv] no qualifying origins."); return

    prod_out = os.path.join(locality_dir, "out", "models", args.group)

    def _snapshot(root):
        out = {}
        for dp, _d, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                try:
                    out[p] = os.path.getmtime(p)
                except OSError:
                    pass
        return out
    before = _snapshot(prod_out)

    per_fold, all_leak_ok = [], True
    work = [(args.slug, args.group, args.run, str(t.date()), args.window_days, args.max_trim, args.n_trials)
            for t, _a, _b in origins]
    if args.jobs > 1:
        print(f"[cv] running {len(work)} folds across {args.jobs} processes ...")
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for origin_str, rows, leak_ok in ex.map(_fold_worker, work):
                per_fold.extend(rows); all_leak_ok &= leak_ok
                print(f"[cv] fold {origin_str} done (leakage_ok={leak_ok})")
    else:
        for w in work:
            print(f"[cv] === fold tᵢ={w[3]} ===")
            origin_str, rows, leak_ok = _fold_worker(w)
            per_fold.extend(rows); all_leak_ok &= leak_ok

    pf = pd.DataFrame(per_fold, columns=["origin", "model", "n_test", "median_ratio", "cod", "cod_trim"])
    agg = pf.groupby("model").agg(
        folds=("cod", "size"), median_ratio=("median_ratio", "mean"),
        cod_mean=("cod", "mean"), cod_std=("cod", "std"),
        cod_trim_mean=("cod_trim", "mean"), cod_trim_std=("cod_trim", "std")).reset_index()
    disp = pd.DataFrame({"model": agg["model"], "folds": agg["folds"],
                         "median_ratio": agg["median_ratio"].round(3)})
    disp["cod (mean±std)"] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(agg["cod_mean"], agg["cod_std"].fillna(0))]
    disp["cod_trim (mean±std)"] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(agg["cod_trim_mean"], agg["cod_trim_std"].fillna(0))]

    out_dir = os.path.join(locality_dir, "out", "cv")
    os.makedirs(out_dir, exist_ok=True)
    pf.to_csv(os.path.join(out_dir, f"per_fold_{args.group}.csv"), index=False)
    disp.to_csv(os.path.join(out_dir, f"aggregate_{args.group}.csv"), index=False)

    print("\n[cv] PER-FOLD (held-out via run_ratio_study):"); print(pf.round(3).to_string(index=False))
    print(f"\n[cv] ACROSS-FOLD AGGREGATE ({len(origins)} folds) — rolling-origin CV:")
    print(disp.to_string(index=False))
    iso_ok = (before == _snapshot(prod_out))
    print(f"\n[cv] leakage_ok(all folds)={all_leak_ok} | isolation_ok(prod out untouched)={iso_ok}")
    print(f"[cv] wrote {out_dir}/per_fold_{args.group}.csv + aggregate_{args.group}.csv")


if __name__ == "__main__":
    main()
