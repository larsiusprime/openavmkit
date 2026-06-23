"""Pooled, validated land-value evidence — four non-circular streams — with per-source A3.

Streams (all 'validated' = explicit qualified deed code A/C; positive land; residential):
  A vacant       : genuinely-vacant PRIME lots. observed_land = sale_price_time_adj.
  B teardown     : structure present at sale then demolished (age_at_sale<0). observed_land = sale.
  C rcn_resid    : improved, depreciation ~ 0 (percent_good >= 0.95) -> RCNLD == RCN, so
                   observed_land = sale - assr_impr ( = sale - RCN, the cleanest residual ).
  D rcnld_resid  : improved, low/moderate depreciation (0.75 <= percent_good < 0.95).
                   observed_land = sale - assr_impr ( = sale - RCNLD ).
High depreciation (percent_good < 0.75) is excluded (residual too noisy/biased — §11 finding).

percent_good = bldg_condition_pct/100 is the reverse-depreciation factor (RCN = assr_impr /
percent_good); it both gates the streams and tells us how clean each residual is.

Run from repo root:  python research/lvi_evidence_pool.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod, trim_outlier_ratios
import lvi_anchors as A

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
NB = "neighborhood_filled"; DEP = "sale_price_time_adj"; MG = "single_family"


def a3(land, obs):
    land = pd.to_numeric(land, errors="coerce").values
    obs = pd.to_numeric(obs, errors="coerce").values
    m = np.isfinite(land) & np.isfinite(obs) & (obs > 0) & (land > 0)
    land, obs = land[m], obs[m]
    if len(land) < 5:
        return len(land), np.nan, np.nan, np.nan
    r = land / obs
    lt, ot = trim_outlier_ratios(land, obs, max_percent=0.10)
    return len(land), float(np.median(r)), float(calc_cod(lt / ot)), float(np.nanmedian(obs / 1.0))


def main():
    os.chdir(DATA)
    sup = read_pickle("out/2-clean-sup"); u = sup.universe; ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    for c in ["assr_impr_value", "assr_land_value", "land_area_sqft", "bldg_area_finished_sqft",
              "bldg_condition_pct", "zoning", "geom_rectangularity_num", "is_vacant"]:
        s[c] = s["key"].map(ui[c])
    price = pd.to_numeric(s[DEP], errors="coerce")
    s["price"] = price.where(price > 0, pd.to_numeric(s["sale_price"], errors="coerce"))
    s["la"] = pd.to_numeric(s["land_area_sqft"], errors="coerce")
    s["assr_impr"] = pd.to_numeric(s["assr_impr_value"], errors="coerce")
    s["assr_land"] = pd.to_numeric(s["assr_land_value"], errors="coerce")
    s["pg"] = pd.to_numeric(s["bldg_condition_pct"], errors="coerce") / 100.0
    s["bldg_area"] = pd.to_numeric(s["bldg_area_finished_sqft"], errors="coerce").fillna(0)
    s["sale_yr"] = pd.to_datetime(s["sale_date"], errors="coerce").dt.year
    s["age_at_sale"] = s["sale_yr"] - pd.to_numeric(s["bldg_year_built"], errors="coerce")
    s["qualified"] = A.qualified_sale_mask(s)
    s["res_zone"] = s["zoning"].map(A.classify_zoning) == "residential"

    # PRIME-lot mask (parcel-level) for vacant/teardown comparability
    prime = A.prime_lot_mask(u, NB, MG, residential_zoning=True)
    s["prime_lot"] = s["key"].map(prime).fillna(False)

    base = s["qualified"] & s["res_zone"] & (s["la"] > 0) & (s["price"] > 0)
    vac = s["vacant_sale"] == True

    streams = {}
    # A vacant (genuinely-vacant prime lots)
    A_ = s[base & vac & s["prime_lot"]].copy(); A_["obs"] = A_["price"]
    streams["A vacant (prime+qual)"] = A_
    # B teardown (structure demolished; not vacant_sale; residential+size/shape comparable via prime_lot proxy)
    B_ = s[base & ~vac & (s["age_at_sale"] < 0) & s["price"].between(3000, 1e9)].copy()
    B_["obs"] = B_["price"]
    streams["B teardown (qual)"] = B_
    # C RCN residual (dep~0)
    C_ = s[base & ~vac & (s["bldg_area"] > 0) & (s["assr_impr"] > 0) & (s["pg"] >= 0.95)].copy()
    C_["obs"] = C_["price"] - C_["assr_impr"]
    streams["C rcn_resid (dep~0)"] = C_[C_["obs"] > 0]
    # D RCNLD residual (low/moderate dep)
    D_ = s[base & ~vac & (s["bldg_area"] > 0) & (s["assr_impr"] > 0) &
           (s["pg"] >= 0.75) & (s["pg"] < 0.95)].copy()
    D_["obs"] = D_["price"] - D_["assr_impr"]
    streams["D rcnld_resid (lo/mod dep)"] = D_[D_["obs"] > 0]

    print("=== A3 (assr_land / observed_land) by validated evidence stream ===")
    print(f"{'stream':<30}{'n':>6}{'median':>9}{'COD_tr':>8}{'land$/sf':>10}")
    print("-" * 63)
    pooled = []
    for name, d in streams.items():
        n, med, cod, _ = a3(d["assr_land"], d["obs"])
        psf = (d["obs"] / d["la"]).median()
        ms = f"{med:.3f}" if np.isfinite(med) else "n/a"
        cs = f"{cod:.1f}" if np.isfinite(cod) else "n/a"
        print(f"{name:<30}{n:>6}{ms:>9}{cs:>8}{psf:>10.2f}")
        pooled.append(d[["key", "assr_land", "obs", "la"]].assign(source=name))

    P = pd.concat(pooled, ignore_index=True).drop_duplicates(["key", "source"])
    n, med, cod, _ = a3(P["assr_land"], P["obs"])
    print("-" * 63)
    print(f"{'POOLED (A+B+C+D)':<30}{n:>6}{med:>9.3f}{cod:>8.1f}{(P['obs']/P['la']).median():>10.2f}")
    # pooled deduped to one obs per parcel (prefer direct vacant/teardown over residual)
    pref = {"A vacant (prime+qual)": 0, "B teardown (qual)": 1, "C rcn_resid (dep~0)": 2,
            "D rcnld_resid (lo/mod dep)": 3}
    P["pref"] = P["source"].map(pref)
    Pdd = P.sort_values("pref").drop_duplicates("key", keep="first")
    n, med, cod, _ = a3(Pdd["assr_land"], Pdd["obs"])
    print(f"{'POOLED (1 obs/parcel)':<30}{n:>6}{med:>9.3f}{cod:>8.1f}{(Pdd['obs']/Pdd['la']).median():>10.2f}")

    os.makedirs("out/lvi", exist_ok=True)
    Pdd.to_csv("out/lvi/evidence_pool.csv", index=False)
    print("\nwrote out/lvi/evidence_pool.csv")


if __name__ == "__main__":
    main()
