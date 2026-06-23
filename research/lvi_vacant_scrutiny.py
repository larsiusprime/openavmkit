"""Scrutinize vacant-land sales used as A3 gold-standard anchors.

Principle: only GENUINELY BUILDABLE "PRIME" lots are good local reflections of land value.
A vacant sale is disqualified if it is really a teardown (a structure was sold), sits on
non-residential / restrictive zoning, is acreage / rural, is weirdly shaped, or is not size-
comparable to the BUILT lots in its own neighborhood.

Reports the disqualification funnel and shows how the assessor's A3 land ratio (assr_land_value
vs vacant sale price) moves as the anchor set is cleaned to prime lots.

Run from repo root:  python research/lvi_vacant_scrutiny.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
NB = "neighborhood_filled"
DEP = "sale_price_time_adj"
MG = "single_family"

# tunable prime-lot thresholds
RECT_MIN = 0.40          # geom_rectangularity_num below this = weird shape
MIN_PEERS = 10           # neighborhood needs this many built SF lots to define a size band
SIZE_LO, SIZE_HI = 0.05, 0.95   # vacant lot must fall within built-peer size percentiles


def _ratio_stats(land, price):
    r = (pd.to_numeric(land, errors="coerce") / pd.to_numeric(price, errors="coerce")).replace(
        [np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if len(r) < 5:
        return dict(n=len(r), median=np.nan, cod=np.nan)
    return dict(n=len(r), median=float(r.median()), cod=float(calc_cod(r.values)))


def main():
    os.chdir(DATA)
    sup = read_pickle("out/2-clean-sup")
    u = sup.universe
    s = get_hydrated_sales_from_sup(sup)

    # ---- neighborhood BUILT-lot size bands (the comparability yardstick) ----
    built = u[(u["model_group"] == MG) & (u["is_vacant"] == False) &
              (pd.to_numeric(u["bldg_area_finished_sqft"], errors="coerce") > 0)].copy()
    built["la"] = pd.to_numeric(built["land_area_sqft"], errors="coerce")
    band = built.groupby(NB)["la"].agg(
        peer_n="count",
        peer_lo=lambda x: x.quantile(SIZE_LO),
        peer_hi=lambda x: x.quantile(SIZE_HI),
        peer_med="median").reset_index()

    # ---- vacant sales (the anchor candidates) ----
    v = s[(s["valid_sale"] == True) & (s.get("vacant_sale", False) == True)].copy()
    v["la"] = pd.to_numeric(v["land_area_sqft"], errors="coerce")
    v["rect"] = pd.to_numeric(v["geom_rectangularity_num"], errors="coerce")
    v["bldg_area"] = pd.to_numeric(v["bldg_area_finished_sqft"], errors="coerce").fillna(0)
    v["zoning"] = v["zoning"].astype(str)
    v = v.merge(band, on=NB, how="left")

    print(f"=== vacant sales: {len(v)} total (all model groups) ===")
    print(v.groupby("model_group").size().sort_values(ascending=False).to_dict())

    # focus on residential land: SF + UNKNOWN (unclassified residential lots)
    v = v[v["model_group"].isin([MG, "UNKNOWN"])].copy()
    print(f"\nfocusing on single_family + UNKNOWN: {len(v)}")

    # ---- disqualification flags ----
    res_zone = v["zoning"].str.match(r"^(R-?\d|PUD)", na=False)  # R*/PUD; excludes RA, RR, ag, etc.
    v["dq_teardown"]   = (v["is_vacant"] == False) | (v["bldg_area"] > 0)   # a structure is/was present
    v["dq_nonres"]     = ~res_zone
    v["dq_no_peers"]   = v["peer_n"].fillna(0) < MIN_PEERS
    v["dq_size"]       = ~v["la"].between(v["peer_lo"], v["peer_hi"]) & v["peer_n"].notna()
    v["dq_shape"]      = v["rect"] < RECT_MIN

    flags = ["dq_teardown", "dq_nonres", "dq_no_peers", "dq_size", "dq_shape"]
    labels = {"dq_teardown": "structure present (teardown/mislabeled)",
              "dq_nonres": "non-residential / restrictive zoning",
              "dq_no_peers": f"neighborhood has <{MIN_PEERS} built SF peers",
              "dq_size": f"lot size outside built-peer [{int(SIZE_LO*100)},{int(SIZE_HI*100)}] pctile",
              "dq_shape": f"weird shape (rectangularity < {RECT_MIN})"}

    print("\n=== disqualification funnel (each flag, standalone counts) ===")
    for f in flags:
        print(f"  {labels[f]:<52} {int(v[f].sum()):>5}")

    v["any_dq"] = v[flags].any(axis=1)
    prime = v[~v["any_dq"]]
    print(f"\nPRIME survivors (pass ALL filters): {len(prime)} of {len(v)} "
          f"({100*len(prime)/len(v):.0f}%)")

    # sequential funnel (how many remain as we stack filters)
    print("\n=== sequential funnel ===")
    remain = v.copy()
    print(f"  start                                  {len(remain):>5}")
    for f in flags:
        remain = remain[~remain[f]]
        print(f"  after removing {labels[f]:<38} {len(remain):>5}")

    # ---- A3 assessor land ratio at each cleaning stage ----
    def a3(df, name):
        st = _ratio_stats(df["key"].map(u.drop_duplicates("key").set_index("key")["assr_land_value"]),
                          df[DEP])
        print(f"  {name:<34} n={st['n']:>4}  median={st['median']:.3f}  COD={st['cod']:.1f}")

    print("\n=== A3 assessor land ratio (assr_land_value / vacant sale price) ===")
    a3(v, "all SF+UNKNOWN vacant")
    a3(v[~v["dq_teardown"]], "genuinely vacant only")
    a3(v[~v["dq_teardown"] & ~v["dq_nonres"]], "+ residential zoning")
    a3(prime, "PRIME (all filters)")

    os.makedirs("out/lvi", exist_ok=True)
    keep = ["key", "model_group", NB, "zoning", "la", "rect", "peer_n", "peer_lo", "peer_hi",
            DEP] + flags + ["any_dq"]
    v[keep].to_csv("out/lvi/vacant_scrutiny.csv", index=False)
    print("\nwrote out/lvi/vacant_scrutiny.csv")


if __name__ == "__main__":
    main()
