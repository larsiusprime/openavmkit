"""Proper teardown classification (per the buildable-land definition):
  1. sold WITH a building on it  (not a vacant/land sale: vacant_sale == False)
  2. a new building was built shortly after  (current bldg newer than sale: age_at_sale < 0)
  3. price per LAND sqft is SUBSTANTIALLY BELOW the neighborhood's developed/finished-home
     $/land-sqft  -> the price was essentially for the lot, the old structure was ~worthless.

A finished-new-home sale fails (3): its price/land-sqft sits at the developed level (it includes
the new building). Only genuine teardowns price like land.

Run from repo root:  python research/lvi_teardown_explore.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod
import lvi_anchors as A

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
NB = "neighborhood_filled"; DEP = "sale_price_time_adj"; MG = "single_family"
TEARDOWN_FRAC = 0.50      # price/land-sqft below this fraction of the nbhd developed level = teardown


def main():
    os.chdir(DATA)
    sup = read_pickle("out/2-clean-sup"); u = sup.universe; ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy(); s["key"] = s["key"].astype(str)
    for c in ["assr_land_value", "assr_impr_value", "land_area_sqft", "bldg_area_finished_sqft", "zoning"]:
        s[c] = s["key"].map(ui[c])
    pr = pd.to_numeric(s[DEP], errors="coerce"); s["price"] = pr.where(pr > 0, pd.to_numeric(s["sale_price"], errors="coerce"))
    s["la"] = pd.to_numeric(s["land_area_sqft"], errors="coerce")
    s["price_psf"] = s["price"] / s["la"]
    s["sale_yr"] = pd.to_datetime(s["sale_date"], errors="coerce").dt.year
    s["age_at_sale"] = s["sale_yr"] - pd.to_numeric(s["bldg_year_built"], errors="coerce")
    s["qualified"] = A.qualified_sale_mask(s)
    s = s[(s["la"] > 0) & (s["price"] > 0)]

    # neighborhood DEVELOPED benchmark: median price/land-sqft of IMPROVED sales (the finished level)
    improved = s[(s["vacant_sale"] != True) & (pd.to_numeric(s["bldg_area_finished_sqft"], errors="coerce") > 0)]
    dev_psf = improved.groupby(NB)["price_psf"].median()
    dev_n = improved.groupby(NB)["price_psf"].count()

    # candidates: sold-with-building, new building shortly after
    cand = s[(s["vacant_sale"] != True) & (s["age_at_sale"] < 0)].copy()
    cand["dev_psf"] = cand[NB].map(dev_psf); cand["dev_n"] = cand[NB].map(dev_n).fillna(0).astype(int)
    cand["frac_of_dev"] = cand["price_psf"] / cand["dev_psf"]
    cov = cand[(cand["dev_n"] >= 5) & cand["frac_of_dev"].notna()]
    print(f"candidates (sold-with-building, age_at_sale<0): {len(cand)}  with nbhd benchmark: {len(cov)}")
    print("\ndistribution of price/land-sqft AS A FRACTION of neighborhood developed level:")
    for q in [10, 25, 50, 75, 90]:
        print(f"  p{q:>2}: {np.nanpercentile(cov['frac_of_dev'], q):.2f}")

    teardown = cov[cov["frac_of_dev"] < TEARDOWN_FRAC]
    finished = cov[cov["frac_of_dev"] >= TEARDOWN_FRAC]
    print(f"\nsplit at {TEARDOWN_FRAC:.0%} of developed level:")
    print(f"  TEARDOWN (price~land):     {len(teardown):>4}   qualified(A/C): {int(teardown['qualified'].sum())}")
    print(f"  finished-new-home (price~land+bldg): {len(finished):>4}   qualified(A/C): {int(finished['qualified'].sum())}")

    def a3(d, obs_is_price=True):
        land = pd.to_numeric(d["assr_land_value"], errors="coerce")
        obs = d["price"] if obs_is_price else (d["price"] - pd.to_numeric(d["assr_impr_value"], errors="coerce"))
        r = (land / obs).replace([np.inf, -np.inf], np.nan).dropna(); r = r[r > 0]
        if len(r) < 5: return f"n={len(r)} (thin)"
        return f"n={len(r)}  median={r.median():.3f}  COD={calc_cod(r.values):.1f}  land$/sf={(obs/d['la']).median():.2f}"

    print("\n=== how each subset behaves as a LAND anchor (assr_land / observed) ===")
    print(f"  TEARDOWN, observed_land = price        : {a3(teardown, True)}")
    print(f"  TEARDOWN(qualified only)               : {a3(teardown[teardown['qualified']], True)}")
    print(f"  finished-new-home, observed = price    : {a3(finished, True)}   <- WRONG (incl. new bldg)")
    print(f"  finished-new-home, observed = price-impr: {a3(finished, False)}   <- correct (residual)")

    print("\n=== sample TEARDOWNS (lowest frac_of_dev) ===")
    cols = ["key", NB, "zoning", "la", "price", "price_psf", "dev_psf", "frac_of_dev", "assr_land_value", "qualified"]
    print(teardown.sort_values("frac_of_dev")[cols].head(12).to_string(index=False))

    os.makedirs("out/lvi", exist_ok=True)
    teardown[cols].to_csv("out/lvi/teardowns.csv", index=False)
    print("\nwrote out/lvi/teardowns.csv")


if __name__ == "__main__":
    main()
