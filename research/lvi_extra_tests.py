"""The deferred land-integrity tests that close the gaps in the thesis:

  A5  desirability gradient   — does assessor LAND $/sqft rise with location desirability the way
                                the market does? (Spearman rho + slope, neighborhood level)
  A8g building-location test  — does assessor BUILDING $/sqft rise with location? (should NOT) —
                                the contrast A5 >> A8g de-tautologizes A2/A8: it shows the
                                assessor routes location into LAND, not building, against an
                                INDEPENDENT axis (market desirability), not its own clusters.
  B3  local spatial uniformity— is LAND $/sqft locally smooth? (Moran's I + local-outlier share)
  A6  sales chasing on land   — are vacant-land assessments silently set to the sale price?
                                (openavmkit detect_sales_chasing)

Desirability proxy = neighborhood median improved-sale price (market signal; non-circular).
Run from repo root:  python research/lvi_extra_tests.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.sales_chasing import detect_sales_chasing
import lvi_anchors as A

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
NB = "neighborhood_filled"; DEP = "sale_price_time_adj"; MG = "single_family"


def main():
    os.chdir(DATA)
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str)
    for c in ["assr_land_value", "assr_impr_value", "land_area_sqft", "bldg_area_finished_sqft"]:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    u["land_psf"] = u["assr_land_value"] / u["land_area_sqft"]
    u["impr_psf"] = u["assr_impr_value"] / u["bldg_area_finished_sqft"]
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy(); s["key"] = s["key"].astype(str)
    pr = pd.to_numeric(s[DEP], errors="coerce"); s["price"] = pr.where(pr > 0, pd.to_numeric(s["sale_price"], errors="coerce"))

    # ---- desirability proxies (market, independent of the assessment) ----
    #  proxy 1: neighborhood median sale $/sqft (home-price level; building-dominated).
    #  proxy 2 (cleanest): neighborhood median QUALIFIED VACANT-sale $/sqft = the actual MARKET
    #           LAND gradient. NOTE: total sale price is NOT used — it confounds home size/quality.
    imp = s[(s["vacant_sale"] != True) & (pd.to_numeric(s["bldg_area_finished_sqft"], errors="coerce") > 0)].copy()
    imp["sale_psf"] = imp["price"] / pd.to_numeric(imp["bldg_area_finished_sqft"], errors="coerce")
    vac = s[(s["vacant_sale"] == True)].copy()
    vac["la"] = pd.to_numeric(vac["key"].map(u.set_index("key")["land_area_sqft"]), errors="coerce")
    vac = vac[A.qualified_sale_mask(vac) & (vac["la"] > 0) & (vac["price"] > 0)]
    vac["vland_psf"] = vac["price"] / vac["la"]
    nb = pd.DataFrame({
        "desir_psf": imp.groupby(NB)["sale_psf"].median(),
        "nimp": imp.groupby(NB)["sale_psf"].count(),
        "mkt_land_psf": vac.groupby(NB)["vland_psf"].median(),
        "nvac": vac.groupby(NB)["vland_psf"].count(),
        "land_psf": u[u["land_psf"] > 0].groupby(NB)["land_psf"].median(),
        "impr_psf": u[u["impr_psf"] > 0].groupby(NB)["impr_psf"].median(),
    })
    a = nb[(nb["nimp"] >= 10)].dropna(subset=["desir_psf", "land_psf"])
    b = nb[(nb["nvac"] >= 3)].dropna(subset=["mkt_land_psf", "land_psf"])
    rho_la, _ = spearmanr(a["desir_psf"], a["land_psf"])
    rho_im, _ = spearmanr(a["desir_psf"], a["impr_psf"])
    rho_mkt, _ = spearmanr(b["mkt_land_psf"], b["land_psf"])
    print(f"=== A5 desirability / land gradient ===")
    print(f"  vs home $/sqft (n={len(a)} nbhds, building-dominated proxy): "
          f"land rho={rho_la:.3f}  bldg rho={rho_im:.3f}")
    print(f"  vs MARKET LAND $/sqft from qualified vacant sales (n={len(b)} nbhds, cleanest): "
          f"land rho={rho_mkt:.3f}  <- A5 verdict")
    print(f"     median assr land $/sqft {b['land_psf'].median():.1f} vs market {b['mkt_land_psf'].median():.1f}")
    print(f"     A5 {'PASS' if rho_mkt >= 0.6 else 'WARN'}: assessor land tracks the market land gradient")

    # ---- B3: local spatial uniformity of LAND $/sqft (Moran's I + local-outlier share) ----
    b = u[(u["land_psf"] > 0) & u["latitude"].notna() & u["longitude"].notna()].copy()
    b["z"] = np.log(b["land_psf"])
    # de-mean within neighborhood so we test LOCAL smoothness, not the (legit) between-nbhd gradient
    b["z"] = b["z"] - b.groupby(NB)["z"].transform("mean")
    xy = b[["latitude", "longitude"]].to_numpy()
    tree = cKDTree(xy)
    k = 8
    _, idx = tree.query(xy, k=k + 1)           # +1 = self
    z = b["z"].to_numpy()
    neigh_mean = z[idx[:, 1:]].mean(axis=1)     # mean of k neighbors (exclude self)
    moran = float(np.sum(z * neigh_mean) / np.sum(z * z))
    local_dev = np.abs(z - neigh_mean)          # |parcel - local mean| in log space
    hot = float(np.mean(local_dev > np.log(2)))  # >2x off its local neighbors
    print(f"\n=== B3 local spatial uniformity (LAND $/sqft, within-nbhd detrended, n={len(b)}, k={k}) ===")
    print(f"  Moran's I (row-standardized) = {moran:.3f}   (>0 = locally smooth; ~0 = random)")
    print(f"  median local deviation = {np.median(local_dev):.3f} log-units "
          f"({100*(np.exp(np.median(local_dev))-1):.0f}% off local neighbors)")
    print(f"  'hot pixel' share (>2x off local neighbors) = {100*hot:.1f}%")

    # ---- A6: sales chasing on vacant land ----
    v = s[(s.get("vacant_sale", False) == True) & (s["price"] > 0)].copy()
    v["assr_land_value"] = pd.to_numeric(v["key"].map(u.set_index("key")["assr_land_value"]), errors="coerce")
    for c in ["land_he_id", "sale_age_days"]:
        if c not in v.columns: v[c] = np.nan
    v = v[v["assr_land_value"].notna() & (v["assr_land_value"] > 0)]
    print(f"\n=== A6 sales chasing on land (vacant-land sales n={len(v)}) ===")
    try:
        res = detect_sales_chasing(v, suspect_field="assr_land_value", sale_price_field="price",
                                   cluster_field="land_he_id", sale_age_field="sale_age_days")
        # spike share near ratio==1.0
        r = v["assr_land_value"] / v["price"]
        spike = float(np.mean((r - 1.0).abs() <= 0.02))
        verdict = getattr(res, "verdict", None) or getattr(res, "summary", str(res))
        print(f"  ratio-at-1.0 spike share (|ratio-1|<=2%): {100*spike:.1f}%  (median ratio {r.median():.3f})")
        print(f"  detector verdict: {verdict}")
    except Exception as e:
        r = v["assr_land_value"] / v["price"]
        spike = float(np.mean((r - 1.0).abs() <= 0.02))
        print(f"  detector errored ({e}); manual spike test: {100*spike:.1f}% near 1.0, median ratio {r.median():.3f}")


if __name__ == "__main__":
    main()
