"""Second-layer smell test: land implied by IMPROVED sales (residual land) as an independent
yardstick to flag non-market vacant sales.

For an improved sale:  implied_land = sale_price_time_adj - building_cost,  where building_cost =
assr_impr_value (the assessor's cost-book RCN-less-depreciation). This is NON-CIRCULAR for the
LAND test: building value comes from cost tables, the land schedule (assr_land_value, what we're
testing) never enters.

Depreciation matters, so we analyze the residual separately by percent-good (bldg_condition_pct):
  - LOW/NO depreciation (new builds): building_cost ~ RCN regardless of the assessor's
    depreciation judgment, so the residual is the CLEANEST land estimate.
  - HIGH depreciation (old builds): building_cost depends heavily on the (uncertain) depreciation
    factor, so the residual is noisier / potentially biased.

We then compare vacant-land sale $/sqft to the neighborhood residual-land level (from low-dep
improved sales) and flag vacant sales that sit far below it — a market-based, non-circular
disqualifier.

Run from repo root:  python research/lvi_residual_smell.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
import lvi_anchors as A

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
NB = "neighborhood_filled"; DEP = "sale_price_time_adj"; MG = "single_family"


def _q(x, p): return np.nanpercentile(x, p)


def main():
    os.chdir(DATA)
    sup = read_pickle("out/2-clean-sup"); u = sup.universe; ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    for c in ["assr_impr_value", "assr_land_value", "land_area_sqft", "bldg_area_finished_sqft",
              "bldg_condition_pct"]:
        s[c] = pd.to_numeric(s["key"].map(ui[c]), errors="coerce")
    s["price"] = pd.to_numeric(s[DEP], errors="coerce")

    # ---- IMPROVED sales: residual land = sale - cost-book building (RCNLD) ----
    imp = s[(s.get("vacant_sale", False) != True) & (s["bldg_area_finished_sqft"] > 0) &
            (s["assr_impr_value"] > 0) & (s["price"] > 0) & (s["land_area_sqft"] > 0)].copy()
    imp["resid_land"] = imp["price"] - imp["assr_impr_value"]
    imp["resid_psf"] = imp["resid_land"] / imp["land_area_sqft"]
    imp["pct_good"] = imp["bldg_condition_pct"]   # 0..~155 (percent good; higher = less depreciation)

    # depreciation buckets
    def bucket(p):
        if p >= 90: return "low/no-dep (pct>=90)"
        if p >= 75: return "moderate (75-90)"
        return "high-dep (<75)"
    imp["dep_bucket"] = imp["pct_good"].apply(bucket)

    print(f"IMPROVED single-family sales with cost-book building: n={len(imp)}")
    print("\n=== residual land $/sqft by depreciation bucket ===")
    print(f"{'bucket':<22}{'n':>6}{'%neg':>7}{'median$/sf':>12}{'p25':>8}{'p75':>8}{'IQR/med':>9}")
    order = ["low/no-dep (pct>=90)", "moderate (75-90)", "high-dep (<75)"]
    levels = {}
    for b in order:
        g = imp[imp["dep_bucket"] == b]
        if len(g) < 10: continue
        pos = g[g["resid_land"] > 0]["resid_psf"]
        pneg = 100 * np.mean(g["resid_land"] <= 0)
        med, p25, p75 = pos.median(), _q(pos, 25), _q(pos, 75)
        levels[b] = med
        print(f"{b:<22}{len(g):>6}{pneg:>6.0f}%{med:>12.2f}{p25:>8.2f}{p75:>8.2f}{(p75-p25)/med:>9.2f}")

    # ---- vacant sales: how do they compare to the residual-implied land level? ----
    v = s[s.get("vacant_sale", False) == True].copy()
    v["price"] = pd.to_numeric(v[DEP], errors="coerce")
    v["psf"] = v["price"] / v["land_area_sqft"]
    v["qualified"] = A.qualified_sale_mask(v)
    bands = A.neighborhood_size_bands(u, NB, MG)
    v = A.add_prime_flags(v, bands, NB, residential_zoning=True)
    print("\n=== vacant land sale $/sqft vs residual-implied land level ===")
    print(f"  low/no-dep residual median land $/sqft : {levels.get('low/no-dep (pct>=90)', float('nan')):.2f}")
    for label, d in [("all vacant", v), ("prime vacant", v[v["prime"]]),
                     ("qualified vacant (A/C)", v[v["qualified"]]),
                     ("prime+qualified vacant", v[v["prime"] & v["qualified"]])]:
        dd = d[d["psf"].notna() & (d["psf"] > 0)]
        print(f"  {label:<26} n={len(dd):>3}  median $/sqft={dd['psf'].median():>7.2f}")

    # ---- neighborhood residual-land yardstick (low-dep) -> flag low vacant sales ----
    lowdep = imp[(imp["dep_bucket"] == "low/no-dep (pct>=90)") & (imp["resid_land"] > 0)]
    nb_resid = lowdep.groupby(NB)["resid_psf"].median()
    nb_n = lowdep.groupby(NB)["resid_psf"].count()
    v["nb_resid_psf"] = v[NB].map(nb_resid)
    v["nb_resid_n"] = v[NB].map(nb_n).fillna(0).astype(int)
    v["resid_ratio"] = v["psf"] / v["nb_resid_psf"]   # vacant sale $/sqft vs local residual land level
    cov = v[(v["nb_resid_n"] >= 5) & v["resid_ratio"].notna()]
    print(f"\n=== vacant sales vs neighborhood low-dep residual land level (>=5 residual comps) ===")
    print(f"  coverage: {len(cov)} of {len(v)} vacant sales have a residual yardstick")
    for thr in [0.25, 0.5, 0.75]:
        print(f"  vacant sale below {int(thr*100)}% of local residual land level: "
              f"{100*np.mean(cov['resid_ratio'] < thr):.0f}%  "
              f"(qualified-only: {100*np.mean(cov[cov['qualified']]['resid_ratio'] < thr):.0f}%)")
    print(f"  median resid_ratio: all={cov['resid_ratio'].median():.2f}  "
          f"prime={cov[cov['prime']]['resid_ratio'].median():.2f}  "
          f"qualified={cov[cov['qualified']]['resid_ratio'].median():.2f}")

    os.makedirs("out/lvi", exist_ok=True)
    cov_cols = ["key", NB, "psf", "nb_resid_psf", "nb_resid_n", "resid_ratio", "prime", "qualified"]
    cov[cov_cols].to_csv("out/lvi/residual_smell.csv", index=False)
    print("\nwrote out/lvi/residual_smell.csv")


if __name__ == "__main__":
    main()
