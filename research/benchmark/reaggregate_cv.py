"""Rebuild CV mean/std summaries from the per-fold raw CSVs.

The in-run aggregator originally dropped object-dtype metric columns, leaving only n_folds in
benchmark_cv.md. The raw per-fold tables (`<subset>_raw.csv`) are intact, so this rebuilds
`<subset>_mean.csv`, `<subset>_std.csv`, and `benchmark_cv.md` from them with proper numeric
coercion. Idempotent; safe to re-run.

Usage:
    python research/benchmark/reaggregate_cv.py us-va-petersburgcity single_family_suburban
    python research/benchmark/reaggregate_cv.py us-co-eagle single_family
"""
import os
import sys

import pandas as pd

HEADLINE = ["cod", "prd", "prb", "median_ratio", "cod_trim", "prd_trim", "prb_trim", "vei"]


def reaggregate(slug, group, root="research/benchmark"):
    g_dir = os.path.join(root, slug, "cv", group)
    md = [f"# Cross-validated benchmark (Flavor A: repeated holdout) — {slug} / {group}\n",
          "Repeated holdout, same valuation date, re-drawn canonical split each fold; "
          "hyperparameters fixed (reused). Cells are `mean ± std` across folds.\n"]
    found = False
    for raw_name, label in [("holdout_raw.csv", "Holdout set"), ("study_raw.csv", "Study set")]:
        raw_path = os.path.join(g_dir, raw_name)
        if not os.path.isfile(raw_path):
            continue
        found = True
        raw = pd.read_csv(raw_path)
        key = "model" if "model" in raw.columns else raw.columns[0]
        num = raw.drop(columns=[c for c in ("_seed",) if c in raw.columns])
        num_cols = [c for c in num.columns if c != key]
        num[num_cols] = num[num_cols].apply(pd.to_numeric, errors="coerce")
        grp = num.groupby(key)
        mean_df = grp[num_cols].mean()
        std_df = grp[num_cols].std().fillna(0.0)
        n = grp.size().rename("n_folds")
        stem = raw_name.replace("_raw.csv", "")
        mean_df.join(n).to_csv(os.path.join(g_dir, f"{stem}_mean.csv"))
        std_df.to_csv(os.path.join(g_dir, f"{stem}_std.csv"))
        cols = [c for c in HEADLINE if c in mean_df.columns]
        disp = pd.DataFrame(index=mean_df.index)
        disp["n"] = n
        for c in cols:
            disp[c] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean_df[c], std_df[c])]
        md.append(f"### {label}\n\n{disp.to_markdown()}\n")
    if not found:
        print(f"[{slug}/{group}] no *_raw.csv found in {g_dir}")
        return
    with open(os.path.join(g_dir, "benchmark_cv.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[{slug}/{group}] rebuilt benchmark_cv.md + mean/std CSVs")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: reaggregate_cv.py <slug> <model_group>")
    reaggregate(sys.argv[1], sys.argv[2])
