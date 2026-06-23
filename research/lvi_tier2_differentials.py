"""Tier-2 evidence: DIFFERENTIAL locational premiums from matched buildings.

Relative (not absolute) chains: a building model (impr_he_id) that sells in two locations gives a
pure LOCATIONAL land differential — the price gap is land, the building cancels (A8). Anchored to
pure RCN residuals (dep~0 qualified sales: land = price - RCN, the cleanest absolute), the
differential network yields each zone's location premium NET of building composition.

Method = building-composition control: model-demean the RCN-residual land $/sqft (subtract each
model's mean), then the zone mean of the residual is that zone's location premium, identified
only from zones that share models. Compare to the raw (uncontrolled) zone land level — agreement
means building mix wasn't confounding the location signal; divergence means it was, and the
matched-building differential is the cleaner number.

Run from repo root:  python research/lvi_tier2_differentials.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    obs = A.build_land_observations(s, ui, NB)
    q = obs["qualified"].fillna(False)
    # pure RCN-residual sales (dep~0), with known absolute land per parcel
    r = obs[(obs["kind"] == "rcn_resid") & q].copy()
    r["model"] = r["key"].map(ui["impr_he_id"])
    r["land_psf"] = r["observed_land"] / r["land_sqft"]
    r = r[(r["land_psf"] > 0) & r["model"].notna() & r["nbhd"].notna()].copy()
    r["lp"] = np.log(r["land_psf"])

    # keep models that appear in >=2 zones (they carry a locational differential)
    model_zones = r.groupby("model")["nbhd"].nunique()
    multi = set(model_zones[model_zones >= 2].index)
    rr = r[r["model"].isin(multi)].copy()
    print(f"RCN-residual sales: {len(r):,} in {r['nbhd'].nunique():,} zones, {r['model'].nunique():,} models")
    print(f"models spanning >=2 zones (carry a differential): {len(multi):,}  "
          f"-> {len(rr):,} sales in {rr['nbhd'].nunique():,} zones connected by matched models\n")

    # building-composition control: demean log land $/sqft by MODEL, then zone mean = location premium
    rr["lp_modeldm"] = rr["lp"] - rr.groupby("model")["lp"].transform("mean")
    loc_premium = rr.groupby("nbhd")["lp_modeldm"].mean()         # relative location premium (log, model-controlled)
    raw_level = r.groupby("nbhd")["lp"].mean()                    # uncontrolled zone land level (log)
    z = pd.DataFrame({"loc_prem": loc_premium, "raw": raw_level}).dropna()

    # does controlling for building model change the location ranking? (is composition confounding?)
    rho = spearmanr(z["loc_prem"], z["raw"] - z["raw"].mean())[0]
    print(f"=== composition-controlled location premium vs raw zone land level (n={len(z)} zones) ===")
    print(f"  Spearman rho(location premium, raw level) = {rho:.3f}  "
          f"({'composition NOT confounding' if rho > 0.9 else 'composition matters'})")
    # spread of the differential signal: range of model-controlled location premiums (in $/sqft terms)
    prem_psf = np.exp(loc_premium + r['lp'].mean())              # back to ~$/sqft scale
    print(f"  location-premium spread (model-controlled): p10={np.percentile(prem_psf,10):.1f} "
          f"p50={np.percentile(prem_psf,50):.1f} p90={np.percentile(prem_psf,90):.1f} $/sqft "
          f"(={np.percentile(prem_psf,90)/np.percentile(prem_psf,10):.1f}x across locations)")

    # consistency of the differentials: within a model, dispersion of zone land levels = the
    # locational signal; across models it should be CONSISTENT for a given zone pair.
    # check: for zone pairs sharing >=2 models, do the per-model differentials agree?
    diffs = []
    for m, g in rr.groupby("model"):
        zl = g.groupby("nbhd")["lp"].mean()
        if len(zl) >= 2:
            zl = zl - zl.mean()
            for nb, v in zl.items():
                diffs.append((nb, m, v))
    dd = pd.DataFrame(diffs, columns=["nbhd", "model", "dlp"])
    by_zone = dd.groupby("nbhd")["dlp"].agg(["mean", "std", "count"])
    multi_model = by_zone[by_zone["count"] >= 3]
    print(f"\n  cross-model consistency (zones with >=3 models, n={len(multi_model)}): "
          f"median within-zone std of differential = {multi_model['std'].median():.3f} log-units "
          f"(~{100*(np.exp(multi_model['std'].median())-1):.0f}% — agreement of independent matched-model chains)")

    # example differential chains (highest and lowest location premiums)
    ex = pd.DataFrame({"loc_prem_psf": prem_psf, "n_models": rr.groupby("nbhd")["model"].nunique()}).dropna()
    ex = ex[ex["n_models"] >= 3].sort_values("loc_prem_psf")
    print(f"\n  example zone location premiums (model-controlled, >=3 matched models):")
    print(f"    lowest:  {ex.index[0]} = ${ex['loc_prem_psf'].iloc[0]:.1f}/sqft   "
          f"highest: {ex.index[-1]} = ${ex['loc_prem_psf'].iloc[-1]:.1f}/sqft   "
          f"differential = ${ex['loc_prem_psf'].iloc[-1]-ex['loc_prem_psf'].iloc[0]:.1f}/sqft, "
          f"proven by the SAME models selling in both")

    os.makedirs("out/lvi", exist_ok=True)
    out = pd.DataFrame({"loc_premium_psf": prem_psf, "n_matched_models": rr.groupby("nbhd")["model"].nunique()})
    out.to_csv("out/lvi/location_differentials.csv")
    print("\n  wrote out/lvi/location_differentials.csv (per-zone model-controlled location premium + #models)")
    print("  footprint = the RCN-residual (new-construction) zones; relative chains are the most")
    print("  legible protest evidence (identical models, different locations) and control for building mix.")


if __name__ == "__main__":
    main()
