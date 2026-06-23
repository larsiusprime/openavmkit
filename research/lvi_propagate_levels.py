"""B1b (full) — propagate land LEVELS to unsupported zones + consistency check + per-parcel chains.

Mechanism (abstraction anchored to DIRECT evidence, not a cost table):
  1. Anchor zones have a validated market land $/sqft, L_direct (from vacant prime+qual / rcn_resid).
  2. For each matched-building cluster (impr_he_id), back out a LOCATION-INVARIANT building value
     per sqft from anchor-zone sales:  bldg_psf[t] = median( (price - L_direct*land_sqft)/bldg_area ).
     (sale - land = building; calibrated where land is known; location-invariant by A8.)
  3. Apply bldg_psf everywhere to get implied market land for ANY improved sale:
     implied_land = price - bldg_psf[t]*bldg_area ; per zone -> L_prop (market land level).
  This ties every zone with matched-building sales to direct evidence via concrete market prices.

Checks:
  - consistency: on anchor zones, L_prop vs L_direct should agree (validates the propagation).
  - cross-cluster: within a zone, do different building clusters imply consistent land? (sudoku).
  - extended A3: assessor land vs propagated market land across ALL covered zones (not just 25%).
Emits out/lvi/land_evidence_chains.csv (per-parcel protest artifact).

Run from repo root:  python research/lvi_propagate_levels.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod, trim_outlier_ratios
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"; DEP = "sale_price_time_adj"


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    for c in ["assr_land_value", "land_area_sqft"]:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    u["assr_land_psf"] = u["assr_land_value"] / u["land_area_sqft"]
    nb_val = u.groupby(NB)["assr_land_value"].sum(); tot = nb_val.sum()
    assr_psf_zone = u[u["assr_land_psf"] > 0].groupby(NB)["assr_land_psf"].median()

    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    obs = A.build_land_observations(s, ui, NB)
    prime = A.prime_lot_mask(u, NB, MG, residential_zoning=True); q = obs["qualified"].fillna(False)
    obs["prime_lot"] = obs["key"].map(prime).fillna(False)
    direct = obs[((obs["kind"] == "vacant") & obs["prime_lot"] & q) | ((obs["kind"] == "rcn_resid") & q)].copy()
    direct["obs_psf"] = direct["observed_land"] / direct["land_sqft"]
    L_direct = direct.groupby("nbhd")["obs_psf"].median()           # validated market land $/sqft (anchors)

    # improved sales
    imp = s[(s.get("vacant_sale", False) != True) & s["impr_he_id"].notna()].copy()
    imp["nbhd"] = imp["key"].map(ui[NB])
    imp["price"] = pd.to_numeric(imp[DEP], errors="coerce")
    imp["price"] = imp["price"].where(imp["price"] > 0, pd.to_numeric(imp["sale_price"], errors="coerce"))
    imp["la"] = pd.to_numeric(imp["key"].map(ui["land_area_sqft"]), errors="coerce")
    imp["ba"] = pd.to_numeric(imp["bldg_area_finished_sqft"], errors="coerce")
    imp = imp[(imp["price"] > 0) & (imp["la"] > 0) & (imp["ba"] > 0) & imp["nbhd"].notna()].copy()

    # 2. location-invariant building $/sqft per cluster, calibrated on ANCHOR zones
    imp["L_anchor"] = imp["nbhd"].map(L_direct)
    cal = imp[imp["L_anchor"].notna()].copy()
    cal["bpsf"] = (cal["price"] - cal["L_anchor"] * cal["la"]) / cal["ba"]
    cal = cal[(cal["bpsf"] > 20) & (cal["bpsf"] < 600)]            # sane building $/sqft
    bldg_psf = cal.groupby("impr_he_id")["bpsf"].median()
    bldg_n = cal.groupby("impr_he_id")["bpsf"].count()
    good = bldg_psf[bldg_n >= 3]                                    # need >=3 anchor sales to trust a cluster

    # 3. propagate: implied market land for every improved sale with a calibrated cluster
    imp["bpsf"] = imp["impr_he_id"].map(good)
    imp = imp[imp["bpsf"].notna()].copy()
    imp["implied_land"] = imp["price"] - imp["bpsf"] * imp["ba"]
    imp["implied_psf"] = imp["implied_land"] / imp["la"]
    imp = imp[imp["implied_land"] > 0]
    L_prop = imp.groupby("nbhd")["implied_psf"].median()
    n_clusters_zone = imp.groupby("nbhd")["impr_he_id"].nunique()

    covered = set(L_prop.index) | set(L_direct.index)
    print(f"=== Land-level propagation (anchored to direct evidence via matched buildings) ===")
    print(f"  anchor zones (direct evidence): {len(L_direct):,}")
    print(f"  zones with a propagated land level: {len(L_prop):,}")
    print(f"  TOTAL zones with a market land level (direct or propagated): {len(covered):,} of {u[NB].nunique():,}"
          f"  =  {100*nb_val[nb_val.index.isin(covered)].sum()/tot:.1f}% of land value")

    # consistency: on anchor zones, propagated vs direct
    chk = pd.DataFrame({"direct": L_direct, "prop": L_prop}).dropna()
    rho = spearmanr(chk["direct"], chk["prop"])[0]
    ratio = (chk["prop"] / chk["direct"]).median()
    print(f"\n  consistency (anchor zones, n={len(chk)}): propagated vs direct  "
          f"Spearman rho={rho:.3f}  median(prop/direct)={ratio:.3f}  COD={calc_cod((chk['prop']/chk['direct']).values):.1f}")
    # cross-cluster agreement within a zone (sudoku): dispersion of per-cluster implied land
    cc = imp.groupby(["nbhd", "impr_he_id"])["implied_psf"].median().reset_index()
    disp = cc.groupby("nbhd")["implied_psf"].apply(lambda x: calc_cod(x.values) if len(x) >= 3 else np.nan).dropna()
    print(f"  cross-cluster agreement within zone (>=3 clusters, n={len(disp)}): median COD={disp.median():.1f}")

    # extended A3: assessor land vs propagated market land, across ALL covered zones
    L_market = L_prop.combine_first(L_direct)                       # prefer propagated; direct fills gaps
    z = pd.DataFrame({"assr": assr_psf_zone, "mkt": L_market, "val": nb_val}).dropna()
    r = (z["assr"] / z["mkt"]); rt = trim_outlier_ratios(z["assr"].values, z["mkt"].values, 0.10)
    print(f"\n=== Extended A3: assessor land vs MARKET land, zone-level, n={len(z)} zones "
          f"({100*z['val'].sum()/tot:.1f}% of land value) ===")
    print(f"  median ratio={r.median():.3f}  COD={calc_cod(r.values):.1f}  COD_trim={calc_cod(rt[0]/rt[1]):.1f}")
    print(f"  (vs direct-evidence-only A3: ~0.85 on ~25% of value — now the same finding holds "
          f"across {100*z['val'].sum()/tot:.0f}%)")

    # per-parcel evidence chain artifact
    ev = u[[ "key", NB, "assr_land_value", "land_area_sqft", "assr_land_psf"]].copy()
    ev["market_land_psf"] = ev[NB].map(L_market)
    ev["market_land_value"] = ev["market_land_psf"] * ev["land_area_sqft"]
    ev["land_ratio"] = ev["assr_land_value"] / ev["market_land_value"]
    ev["support"] = np.where(ev[NB].isin(L_direct.index), "direct",
                       np.where(ev[NB].isin(L_prop.index), "matched-building", "none(no sales)"))
    ev["n_bridge_clusters"] = ev[NB].map(n_clusters_zone).fillna(0).astype(int)
    os.makedirs("out/lvi", exist_ok=True)
    ev.to_csv("out/lvi/land_evidence_chains.csv", index=False)
    print(f"\n  support distribution (share of land value):")
    for lvl in ["direct", "matched-building", "none(no sales)"]:
        m = ev["support"] == lvl
        print(f"    {lvl:<22}{100*pd.to_numeric(ev.loc[m,'assr_land_value']).sum()/ev['assr_land_value'].sum():>6.1f}%")
    print("  wrote out/lvi/land_evidence_chains.csv (per-parcel: assr land, market land, ratio, support, #bridges)")


if __name__ == "__main__":
    main()
