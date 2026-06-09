"""Reproducible benchmark harness for the OpenAVMKit research-paper program.

Backbone goal #2 (see ../../C:/Users/Lars/.claude/plans and research/README.md): drive the existing
``openavmkit.model_runner`` machinery from a frozen clean checkpoint and persist the *uniform per-model
comparison table* (`MultiModelResults.benchmark`) as a durable artifact, so every paper (P1-P5) draws
from the same experimental substrate.

What it does, for one locality:
  1. ``cd`` into ``notebooks/pipeline/data/<slug>`` (notebooks use locality-relative paths).
  2. ``load_settings("in/settings.json")``.
  3. Optionally splice a *research run list* into ``modeling.instructions.main.run`` so the flagship
     ``lcomp`` (and ``ngboost`` for P2) are always present and the comparison is apples-to-apples
     across jurisdictions. Eagle, e.g., defines ``lcomp`` but doesn't run it by default.
  4. Mirror notebook 03's pre-model sequence per fold: ``load_cleaned_data_for_modeling`` ->
     ``write_canonical_splits`` -> ``enrich_sup_spatial_lag`` -> ``run_models``.
  5. ``run_models(..., use_saved_params=True, save_results=True)`` -- reuse tuned hyperparameters but
     write per-model outputs under ``out/models/`` (required: run_models only returns its results dict
     when save_results=True). Reproducible, not read-only.
  6. Persist the per-model comparison (``benchmark.df_stats_test`` / ``df_stats_full`` / ``df_time``)
     to ``research/benchmark/<slug>/`` as CSV + Markdown, plus a settings+input manifest.

Evaluation modes:
  * Single split (default): one canonical split -> one benchmark table. Fast iteration.
  * Flavor A repeated-holdout CV (``--cv-repeats N``): re-draw the split N times (same valuation
    date, varying seed), aggregate per-model COD/PRD/PRB/etc. to ``mean ± std`` under
    ``research/benchmark/<slug>/cv/``. Hyperparameters stay fixed (reused) -- honest per-fold
    re-tuning and true rolling-origin are Flavor B (publish mode), deferred.

Potentially slow (re-fits every model per fold; GWR / kernel / NGBoost especially). Run one
jurisdiction at a time; Petersburg is the smaller / faster of the two.

Usage:
    python research/benchmark/run_benchmark.py us-va-petersburgcity
    python research/benchmark/run_benchmark.py us-va-petersburgcity --cv-repeats 5
    python research/benchmark/run_benchmark.py us-co-eagle --add-lcomp --cv-repeats 5
"""

import argparse
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# A sensible research run list: the flagship lcomp + probabilistic ngboost, the production tree
# models, the linear/spatial references, and the assessor benchmark. Used by --add-lcomp / --research.
RESEARCH_RUN_LIST = [
    "assessor",        # incumbent benchmark (always include)
    "mra",             # linear baseline
    "multi_mra",       # per-location linear
    "local_area",      # interpretable per-area baseline
    "lightgbm",        # production tree
    "xgboost",         # production tree
    "lcomp",           # FLAGSHIP: interpretable layered comps (P1)
    "ngboost",         # probabilistic AVM (P2)
]


def _file_digest(path: str, limit_mb: float = 200.0) -> str:
    """sha256 of a file, or '(skipped: too large)' for very large inputs."""
    try:
        if os.path.getsize(path) > limit_mb * 1024 * 1024:
            return "(skipped: >%.0fMB)" % limit_mb
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        return "(error: %s)" % e


def _write_manifest(slug: str, locality_dir: str, settings: dict, run_list, out_dir: str):
    in_dir = os.path.join(locality_dir, "in")
    inputs = []
    if os.path.isdir(in_dir):
        for name in sorted(os.listdir(in_dir)):
            p = os.path.join(in_dir, name)
            if os.path.isfile(p):
                inputs.append({
                    "file": name,
                    "bytes": os.path.getsize(p),
                    "sha256": _file_digest(p),
                })
    manifest = {
        "slug": slug,
        "valuation_date": settings.get("locality", {}).get("valuation_date")
        or settings.get("modeling", {}).get("metadata", {}).get("valuation_date"),
        "run_list_used": list(run_list),
        "model_defs": list(settings.get("modeling", {}).get("models", {}).get("main", {}).keys()),
        "inputs": inputs,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    # snapshot the raw settings file too
    raw = os.path.join(in_dir, "settings.json")
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as src:
            with open(os.path.join(out_dir, "settings.snapshot.json"), "w", encoding="utf-8") as dst:
                dst.write(src.read())


def _persist_one_group(bench, out_dir: str, slug: str, group: str):
    """Persist the df_test/df_full/df_time tables for a single model group."""
    g_dir = os.path.join(out_dir, group)
    os.makedirs(g_dir, exist_ok=True)
    parts = []
    # BenchmarkResults attribute names: df_stats_test (holdout), df_stats_full (study/universe),
    # df_stats_test_post_val (holdout post-valuation-date only), df_time.
    for attr, label in [("df_stats_test", "Holdout set"),
                        ("df_stats_test_post_val", "Holdout set (post-valuation-date only)"),
                        ("df_stats_full", "Study set (universe)"),
                        ("df_time", "Timing (s)")]:
        df = getattr(bench, attr, None)
        if df is None or len(df) == 0:
            continue
        df.to_csv(os.path.join(g_dir, f"benchmark_{attr}.csv"))
        parts.append(f"### {label}\n\n{df.to_markdown()}\n")
    with open(os.path.join(g_dir, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write(f"# Benchmark — {slug} / {group}\n\n")
        f.write("Per-model comparison from `MultiModelResults.benchmark` "
                "(`openavmkit.model_runner._calc_benchmark`). "
                "Columns: utility_score, counts, median_ratio, COD/PRD/PRB/VEI (+trimmed), CHD.\n\n")
        f.write("\n".join(parts))


def _persist_benchmark(results, out_dir: str, slug: str):
    """``run_models`` returns a dict {model_group: MultiModelResults}; persist each group's table."""
    if hasattr(results, "benchmark"):           # single MultiModelResults (defensive)
        results = {"_all": results}
    groups = []
    for group, mm in results.items():
        bench = getattr(mm, "benchmark", None)
        if bench is None:
            continue
        _persist_one_group(bench, out_dir, slug, group)
        groups.append(group)
    # an index pointing at the per-group tables
    with open(os.path.join(out_dir, "benchmark_index.md"), "w", encoding="utf-8") as f:
        f.write(f"# Benchmark index — {slug}\n\n")
        f.write("Model groups with persisted comparison tables (see each `<group>/benchmark.md`):\n\n")
        for g in groups:
            f.write(f"- [`{g}`]({g}/benchmark.md)\n")
    return groups


def _run_fold(pl, settings, seed, slug, verbose=False):
    """One repeated-holdout fold: reload cleaned data, re-split with ``seed``, re-enrich
    spatial-lag (which depends on the training split), run the model menu, and return the
    ``{model_group: MultiModelResults}`` dict.

    Reloading fresh each fold avoids state carryover; setting ``random_seed`` re-draws the
    canonical split (the post-valuation Tier-1 sales are always in test, so only the sampled
    portion varies). Hyperparameters are reused from saved params (use_saved_params=True) --
    honest per-fold re-tuning is Flavor B, not Flavor A.
    """
    settings.setdefault("modeling", {}).setdefault("instructions", {})["random_seed"] = seed
    print(f"[{slug}] fold seed={seed}: load cleaned -> split -> spatial-lag -> run_models")
    sup = pl.load_cleaned_data_for_modeling(settings)
    pl.write_canonical_splits(sup, settings, verbose=False)
    sup = pl.enrich_sup_spatial_lag(sup, settings, verbose=False)
    return pl.run_models(
        sup, settings,
        save_params=True, use_saved_params=True, save_results=True,
        verbose=verbose, run_main=True, run_vacant=False, run_ensemble=True,
        do_shaps=False, do_plots=False,
    )


def _aggregate_cv(per_fold, out_dir, slug, seeds):
    """Aggregate per-model benchmark stats across folds to mean/std/CI, per group & subset."""
    import numpy as np
    import pandas as pd

    cv_root = os.path.join(out_dir, "cv")
    os.makedirs(cv_root, exist_ok=True)
    subsets = [("df_stats_test", "holdout"), ("df_stats_full", "study")]
    # group -> label -> list[(fold_seed, df)]
    store = {}
    for seed, res in zip(seeds, per_fold):
        if hasattr(res, "benchmark"):
            res = {"_all": res}
        for group, mm in (res or {}).items():
            bench = getattr(mm, "benchmark", None)
            if bench is None:
                continue
            for attr, label in subsets:
                df = getattr(bench, attr, None)
                if df is None or len(df) == 0:
                    continue
                store.setdefault(group, {}).setdefault(label, []).append((seed, df.copy()))

    # headline metrics to surface as "mean ± std" in the markdown
    headline = ["cod", "prd", "prb", "median_ratio", "cod_trim", "prd_trim", "prb_trim", "vei"]
    groups = []
    for group, labels in store.items():
        g_dir = os.path.join(cv_root, group)
        os.makedirs(g_dir, exist_ok=True)
        md = [f"# Cross-validated benchmark (Flavor A: repeated holdout) — {slug} / {group}\n",
              f"{len(seeds)} folds (seeds {min(seeds)}–{max(seeds)}); same valuation date, "
              "re-drawn canonical split each fold; hyperparameters fixed (reused). "
              "Cells are `mean ± std` across folds.\n"]
        for attr, label in subsets:
            if label not in labels:
                continue
            dfs = labels[label]
            big = pd.concat([df.assign(_seed=s) for s, df in dfs])
            big.index.name = "model"
            big.to_csv(os.path.join(g_dir, f"{label}_raw.csv"))  # every fold, every model
            # Coerce every non-_seed column to numeric (BenchmarkResults stores some metric
            # columns as object dtype, and "N/A" -> NaN); select_dtypes alone would drop them.
            num = big.drop(columns=["_seed"], errors="ignore").apply(pd.to_numeric, errors="coerce")
            num_cols = [c for c in num.columns if num[c].notna().any()]
            grp = num[num_cols].groupby(level=0)
            mean_df = grp.mean()
            std_df = grp.std().fillna(0.0)
            n_df = num.groupby(level=0).size().rename("n_folds")
            mean_df.join(n_df).to_csv(os.path.join(g_dir, f"{label}_mean.csv"))
            std_df.to_csv(os.path.join(g_dir, f"{label}_std.csv"))
            # build a compact mean±std display for headline metrics
            cols = [c for c in headline if c in mean_df.columns]
            disp = pd.DataFrame(index=mean_df.index)
            disp["n"] = n_df
            for c in cols:
                disp[c] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean_df[c], std_df[c])]
            md.append(f"### {label.capitalize()} set\n\n{disp.to_markdown()}\n")
        with open(os.path.join(g_dir, "benchmark_cv.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        groups.append(group)

    with open(os.path.join(cv_root, "cv_index.md"), "w", encoding="utf-8") as f:
        f.write(f"# Cross-validated benchmark index — {slug}\n\n")
        f.write(f"Flavor A repeated holdout, {len(seeds)} folds (seeds {list(seeds)}).\n\n")
        for g in groups:
            f.write(f"- [`{g}`]({g}/benchmark_cv.md)\n")
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug", help="locality slug, e.g. us-va-petersburgcity")
    ap.add_argument("--add-lcomp", action="store_true",
                    help="ensure lcomp+ngboost are appended to the jurisdiction's existing run list")
    ap.add_argument("--research", action="store_true",
                    help="replace the main run list with the canonical RESEARCH_RUN_LIST")
    ap.add_argument("--run", nargs="+", default=None,
                    help="explicit main run list (overrides --research/--add-lcomp)")
    ap.add_argument("--cv-repeats", type=int, default=1,
                    help="Flavor A repeated-holdout folds (N>1 runs the CV loop and aggregates "
                         "to mean ± std). Default 1 = single split.")
    ap.add_argument("--cv-seed", type=int, default=1337,
                    help="base random seed; fold k uses cv_seed + k")
    ap.add_argument("--model-groups", nargs="+", default=None,
                    help="restrict the run to these model group(s) (sets "
                         "modeling.instructions.model_groups). E.g. --model-groups single_family")
    ap.add_argument("--checkpoint", default="out/2-clean-sup.pickle",
                    help="locality-relative path to the cleaned SalesUniversePair pickle")
    args = ap.parse_args()

    import openavmkit.pipeline as pl

    locality_dir = os.path.join(REPO_ROOT, "notebooks", "pipeline", "data", args.slug)
    if not os.path.isdir(locality_dir):
        ap.error(f"locality dir not found: {locality_dir}")

    out_dir = os.path.join(os.path.dirname(__file__), args.slug)
    os.makedirs(out_dir, exist_ok=True)

    prev_cwd = os.getcwd()
    os.chdir(locality_dir)
    try:
        settings = pl.load_settings("in/settings.json")
        instr = settings.setdefault("modeling", {}).setdefault("instructions", {}).setdefault("main", {})
        existing = list(instr.get("run", []))

        if args.run is not None:
            run_list = list(args.run)
        elif args.research:
            run_list = list(RESEARCH_RUN_LIST)
        elif args.add_lcomp:
            run_list = existing + [m for m in ("lcomp", "ngboost") if m not in existing]
        else:
            run_list = existing
        instr["run"] = run_list
        print(f"[{args.slug}] main run list -> {run_list}")

        if args.model_groups:
            # run_models reads instructions.model_groups (a sibling of .main/.vacant) and only
            # iterates those groups when present; restrict to e.g. just single-family.
            settings["modeling"]["instructions"]["model_groups"] = list(args.model_groups)
            print(f"[{args.slug}] restricting to model group(s) -> {args.model_groups}")

        ckpt = args.checkpoint
        if not os.path.isfile(ckpt):
            raise SystemExit(f"clean checkpoint not found: {ckpt} "
                             f"(run 01-assemble + 02-clean for this locality first)")

        # NOTE: each fold mirrors notebook 03's pre-model sequence (load cleaned ->
        # write_canonical_splits -> enrich_sup_spatial_lag -> run_models). run_models only
        # returns its per-group dict when save_results=True, so it writes per-model outputs
        # under out/models/ exactly as notebook 03 does -- reproducible, not read-only.
        if args.cv_repeats > 1:
            seeds = [args.cv_seed + k for k in range(args.cv_repeats)]
            print(f"[{args.slug}] Flavor A repeated-holdout CV: {args.cv_repeats} folds, seeds {seeds}")
            per_fold = []
            for k, seed in enumerate(seeds, 1):
                print(f"[{args.slug}] === fold {k}/{len(seeds)} ===")
                per_fold.append(_run_fold(pl, settings, seed, args.slug, verbose=False))
            groups = _aggregate_cv(per_fold, out_dir, args.slug, seeds)
            _write_manifest(args.slug, locality_dir, settings, run_list, out_dir)
            print(f"[{args.slug}] wrote CV (mean ± std) for {len(groups)} group(s) to "
                  f"{os.path.join(out_dir, 'cv')}: {groups}")
        else:
            results = _run_fold(pl, settings, args.cv_seed, args.slug, verbose=True)
            groups = _persist_benchmark(results, out_dir, args.slug)
            _write_manifest(args.slug, locality_dir, settings, run_list, out_dir)
            print(f"[{args.slug}] wrote benchmark artifacts for {len(groups)} group(s) "
                  f"to {out_dir}: {groups}")
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
