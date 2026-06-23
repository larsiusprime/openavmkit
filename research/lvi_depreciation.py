"""Test Wake's depreciation schedule (percent-good) using the market.

Logic (RCN treated as frozen/validated; percent-good is the assessor judgment under test):
  Within a fine neighborhood (VCS) the LAND is ~constant per sqft regardless of building age.
  So residual land  = sale - assr_impr ( = sale - RCN*percent_good )  per sqft should be FLAT
  with building age. If it DECLINES with age, assr_impr is too high for old homes -> the
  schedule UNDER-depreciates (percent_good too high at old ages). If it RISES, over-depreciates.

Two views:
  1. within-neighborhood (FWL-demeaned) residual land $/sqft by age bin — the robust signal.
  2. an INDEPENDENT depreciation curve: calibrate RCN/sqft by quality on NEW builds (percent_good
     ~ 1, so assr_impr/area = RCN/area, clean), apply to all homes to get RCN free of the old
     home's percent_good, then market-implied percent_good = (sale - neighborhood land) / RCN,
     compared to the assessor's percent_good by age.

Run from repo root:  python research/lvi_depreciation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"
AGE_BINS = [-1, 5, 15, 30, 50, 1000]
AGE_LBL = ["0-5", "6-15", "16-30", "31-50", "50+"]


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    for c in ["assr_impr_value", "land_area_sqft", "bldg_area_finished_sqft", "bldg_condition_pct",
              "bldg_quality_num", "bldg_year_built"]:
        s[c] = pd.to_numeric(s["key"].map(ui[c]), errors="coerce")
    pr = pd.to_numeric(s["sale_price_time_adj"], errors="coerce")
    s["price"] = pr.where(pr > 0, pd.to_numeric(s["sale_price"], errors="coerce"))
    s["sale_yr"] = pd.to_datetime(s["sale_date"], errors="coerce").dt.year
    s["age"] = s["sale_yr"] - s["bldg_year_built"]
    s["pg"] = s["bldg_condition_pct"] / 100.0
    s["qualified"] = A.qualified_sale_mask(s)

    imp = s[(s.get("vacant_sale", False) != True) & (s["bldg_area_finished_sqft"] > 0) &
            (s["assr_impr_value"] > 0) & (s["price"] > 0) & (s["land_area_sqft"] > 0) &
            (s["age"].between(0, 200)) & s["qualified"]].copy()
    imp["resid_psf"] = (imp["price"] - imp["assr_impr_value"]) / imp["land_area_sqft"]
    imp["age_bin"] = pd.cut(imp["age"], bins=AGE_BINS, labels=AGE_LBL)
    # within-neighborhood demean (removes the legit between-VCS land gradient)
    imp["resid_demean"] = imp["resid_psf"] - imp.groupby(NB)["resid_psf"].transform("median")

    print(f"qualified improved SF sales: {len(imp)}")
    print("\n=== View 1: residual land $/sqft by building age (land should be FLAT if depreciation is right) ===")
    print(f"{'age':<8}{'n':>6}{'resid$/sf(raw)':>16}{'resid$/sf(within-nbhd)':>24}{'assessor %good':>16}")
    g = imp.groupby("age_bin")
    for lbl in AGE_LBL:
        d = imp[imp["age_bin"] == lbl]
        if len(d) < 10: continue
        print(f"{lbl:<8}{len(d):>6}{d['resid_psf'].median():>16.1f}{d['resid_demean'].median():>24.1f}"
              f"{100*d['pg'].median():>15.0f}%")

    # FWL slope: within-neighborhood residual_psf on age
    dd = imp.dropna(subset=["resid_psf", "age", NB])
    yw = dd["resid_psf"] - dd.groupby(NB)["resid_psf"].transform("mean")
    aw = dd["age"] - dd.groupby(NB)["age"].transform("mean")
    slope = float(np.sum(aw * yw) / np.sum(aw * aw))
    print(f"\n  within-neighborhood slope: {slope:.3f} $/sqft per year of building age")
    print(f"  -> a 50-yr-older building sits on residual land {50*slope:+.0f} $/sqft different "
          f"(should be ~0 if depreciation is right)")

    # ---- View 2: independent depreciation curve ----
    # calibrate RCN/sqft by quality on NEW builds (age<=3, pg~1 so assr_impr/area = RCN/area)
    newb = imp[(imp["age"] <= 3) & (imp["pg"] >= 0.95)].copy()
    newb["rcn_psf"] = newb["assr_impr_value"] / newb["bldg_area_finished_sqft"]
    base = newb.groupby("bldg_quality_num")["rcn_psf"].median()
    imp["rcn_indep"] = imp["bldg_quality_num"].map(base) * imp["bldg_area_finished_sqft"]
    # neighborhood land level from new-build residuals (clean: sale - RCN at dep~0)
    land0 = newb.assign(lp=(newb["price"] - newb["assr_impr_value"]) / newb["land_area_sqft"]) \
                .groupby(NB)["lp"].median()
    n0 = newb.groupby(NB).size()
    imp["land0_psf"] = imp[NB].map(land0); imp["land0_n"] = imp[NB].map(n0).fillna(0)
    cov = imp[(imp["land0_n"] >= 3) & imp["rcn_indep"].notna() & (imp["rcn_indep"] > 0)].copy()
    cov["mkt_bldg"] = cov["price"] - cov["land0_psf"] * cov["land_area_sqft"]
    cov["pg_market"] = cov["mkt_bldg"] / cov["rcn_indep"]
    print(f"\n=== View 2: assessor percent-good vs MARKET-implied percent-good by age "
          f"(independent RCN; n={len(cov)}) ===")
    print(f"{'age':<8}{'n':>6}{'assessor %good':>16}{'market %good':>16}{'gap (assr-mkt)':>16}")
    for lbl in AGE_LBL:
        d = cov[cov["age_bin"] == lbl]
        d = d[d["pg_market"].between(0, 2)]   # drop wild outliers
        if len(d) < 10: continue
        ap, mp = 100 * d["pg"].median(), 100 * d["pg_market"].median()
        print(f"{lbl:<8}{len(d):>6}{ap:>15.0f}%{mp:>15.0f}%{ap-mp:>+15.0f}")
    print("\n  gap > 0 at old ages = assessor percent-good too HIGH = UNDER-depreciation")


if __name__ == "__main__":
    main()
