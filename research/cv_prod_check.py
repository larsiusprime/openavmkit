"""Q2 diagnostic: is lcomp's ~0.90 temporal median ratio real forward-bias, or model under-prediction?

Decisive comparison:
  (A) PRODUCTION study set = train on all sales ≤ production valuation date, score the lookback
      window (val-1yr, val] via run_ratio_study. This is contemporaneous (test ≈ training period),
      so it should sit ~1.0 if the model is unbiased in-period.
  (B) TEMPORAL CV (already measured) = train ≤tᵢ, score the FUTURE window — ~0.90.

If (A) ≈ 1.0 while (B) ≈ 0.90, the gap is forward-generalization bias (what rolling-origin surfaces
and production hides). Also reports the real improved-SF market trend, to see if the market actually
moved (and whether the flat time-adjustment failed to capture it).
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))  # for `import cv_slice1`
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv_slice1 as cv  # reuse run_fold + constants (main() is guarded by __main__)

SLUG, GROUP = cv.SLUG, cv.GROUP


def main():
    import openavmkit.pipeline as pl
    from openavmkit.utilities.settings import get_valuation_date

    locality_dir = os.path.join(REPO_ROOT, "notebooks", "pipeline", "data", SLUG)
    os.chdir(locality_dir)
    settings = pl.load_settings("in/settings.json")
    instr = settings.setdefault("modeling", {}).setdefault("instructions", {})
    instr["model_groups"] = [GROUP]
    instr.setdefault("main", {})["run"] = ["assessor", "lcomp"]

    sup_assemble = pl.read_pickle("out/1-assemble-sup")
    clean = pl.read_pickle("out/2-clean-sup")
    prod_date = pd.Timestamp(get_valuation_date(settings))
    lookback_start = prod_date - pd.Timedelta(days=365)
    print(f"[prodcheck] production valuation_date={prod_date.date()}  lookback=({lookback_start.date()}, {prod_date.date()}]")

    # (A) production-style: run_fold at the production valuation date trains on ALL sales ≤ prod_date;
    # then score the (in-period) lookback window via run_ratio_study.
    results, fold, fold_dir = cv.run_fold(pl, settings, sup_assemble, prod_date, locality_dir)
    mm = results.get(GROUP) if isinstance(results, dict) else results
    print("\n[prodcheck] PRODUCTION study set (lookback window, in-period):")
    rows = []
    for name, smr in mm.model_results.items():
        up = getattr(smr, "df_universe", None)
        if up is None or "prediction" not in up.columns:
            continue
        su = clean.copy()
        su.set("universe", su.universe.drop(columns=["prediction"], errors="ignore").merge(
            up[["key", "prediction"]], on="key", how="left"))
        rs = pl.run_ratio_study(su, GROUP, "prediction", "sale_price",
                                start_date=str(lookback_start.date()), end_date=str(prod_date.date()),
                                land_only=False, max_trim=0.15)
        rows.append((name, rs.count, cv._v(rs.median_ratio), cv._v(rs.cod), cv._v(rs.cod_trim)))
    print(pd.DataFrame(rows, columns=["model", "n", "median_ratio", "cod", "cod_trim"]).round(3).to_string(index=False))

    # real improved-SF market trend (vacant excluded): did the market actually move?
    s = clean.sales.copy()
    gk = set(clean.universe.loc[clean.universe["model_group"].eq(GROUP), "key"])
    s = s[s["key"].isin(gk)]
    if "vacant_sale" in s.columns:
        s = s[~s["vacant_sale"].eq(True)]               # improved sales only
    s["yr"] = pd.to_datetime(s["sale_date"], errors="coerce").dt.year
    if "bldg_area_finished_sqft" in s.columns:
        s["ppsf"] = s["sale_price"] / s["bldg_area_finished_sqft"]
        s = s[s["ppsf"].between(5, 1000)]
        print("\n[prodcheck] improved-SF median $/finished-sqft by year (real market trend):")
        print(s.groupby("yr")["ppsf"].agg(["count", "median"]).round(1).to_string())
    if "sale_price_time_adj" in clean.sales.columns:
        s2 = clean.sales[clean.sales["key"].isin(gk)].copy()
        s2["yr"] = pd.to_datetime(s2["sale_date"], errors="coerce").dt.year
        s2["adj_over_raw"] = s2["sale_price_time_adj"] / s2["sale_price"]
        print("\n[prodcheck] time-adjustment factor (adj/raw) by year — if ~1.0, index is flat:")
        print(s2.groupby("yr")["adj_over_raw"].median().round(3).to_string())


if __name__ == "__main__":
    main()
