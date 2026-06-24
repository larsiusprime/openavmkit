"""Evidence coverage, level-propagation, and differential location premiums — all cfg-driven."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from openavmkit.utilities.stats import calc_cod

MI_PER_DEG_LAT = 69.0


def coverage_map(u, obs, cfg):
    """B1 — coverage by evidence stream: distance-to-nearest-evidence bands + in-zone share,
    weighted by the series land value. Returns (summary DataFrame, per-parcel DataFrame)."""
    F = cfg.fields
    u = u.copy()
    u["_lv"] = pd.to_numeric(u[F.land_value], errors="coerce")
    u = u[u[F.latitude].notna() & u[F.longitude].notna() & (u["_lv"] > 0)].copy()
    lat0 = np.radians(u[F.latitude].mean())

    def to_xy(df):
        return np.column_stack([df[F.longitude].to_numpy() * MI_PER_DEG_LAT * np.cos(lat0),
                                df[F.latitude].to_numpy() * MI_PER_DEG_LAT])

    uxy = to_xy(u); tot = u["_lv"].sum()
    lo, mid, hi = cfg.support.distance_bands_mi
    evsets = {"direct": obs[obs["kind"] == "direct"]["key"].unique(),
              "cost_residual": obs[obs["kind"] == "cost_residual"]["key"].unique(),
              "pooled": obs["key"].unique()}
    rows, per_parcel = [], u[["key", F.neighborhood, "_lv"]].copy()
    for label, keys in evsets.items():
        ev = u[u["key"].isin(keys)]
        if len(ev) < 1:
            continue
        d, _ = cKDTree(to_xy(ev)).query(uxy, k=1)
        in_zone = u[F.neighborhood].isin(ev[F.neighborhood].unique())
        rows.append(dict(evidence=label, points=len(ev), med_mi=float(np.median(d)),
                         p90_mi=float(np.quantile(d, 0.9)),
                         pct_val_near=100 * u.loc[d < mid, "_lv"].sum() / tot,
                         pct_val_far=100 * u.loc[d > hi, "_lv"].sum() / tot,
                         pct_val_in_zone=100 * u.loc[in_zone, "_lv"].sum() / tot,
                         n_zones=ev[F.neighborhood].nunique()))
        if label == "pooled":
            per_parcel["d_near_mi"] = d
    return pd.DataFrame(rows), per_parcel


def propagate_land_levels(u, s, obs, cfg):
    """B1b — propagate a market land $/sqft surface from direct-evidence zones to unsupported zones
    via matched buildings (location-invariant building value), with a consistency check."""
    F = cfg.fields
    ui = u.drop_duplicates("key").set_index("key")
    # anchor on the GOLD streams: direct, plus cost_residual when the cost basis is frozen/trusted
    kinds = ["direct", "cost_residual"] if cfg.evidence.frozen_sov else ["direct"]
    anchors = obs[obs["kind"].isin(kinds)].copy()
    anchors["psf"] = anchors["observed_land"] / anchors["land_sqft"]
    L_direct = anchors.groupby("nbhd")["psf"].median()

    s = s.copy(); s["key"] = s["key"].astype(str)
    imp = s[s[F.impr_he_id].notna()].copy() if F.impr_he_id in s.columns else s.copy()
    imp[F.impr_he_id] = imp["key"].map(ui[F.impr_he_id])
    imp["nbhd"] = imp["key"].map(ui[F.neighborhood])
    pr = pd.to_numeric(imp[cfg.dep], errors="coerce")
    imp["price"] = pr.where(pr > 0, pd.to_numeric(imp[F.sale_price], errors="coerce"))
    imp["la"] = pd.to_numeric(imp["key"].map(ui[F.land_area]), errors="coerce")
    imp["ba"] = pd.to_numeric(imp["key"].map(ui[F.bldg_area]), errors="coerce")
    imp = imp[(imp["price"] > 0) & (imp["la"] > 0) & (imp["ba"] > 0) & imp["nbhd"].notna() & imp[F.impr_he_id].notna()].copy()

    imp["L_anchor"] = imp["nbhd"].map(L_direct)
    cal = imp[imp["L_anchor"].notna()].copy()
    cal["bpsf"] = (cal["price"] - cal["L_anchor"] * cal["la"]) / cal["ba"]
    blo, bhi = cfg.support.bldg_psf_bounds
    cal = cal[(cal["bpsf"] > blo) & (cal["bpsf"] < bhi)]
    bldg_psf = cal.groupby(F.impr_he_id)["bpsf"].median()
    good = bldg_psf[cal.groupby(F.impr_he_id)["bpsf"].count() >= cfg.support.min_anchor_sales]

    imp["bpsf"] = imp[F.impr_he_id].map(good)
    imp = imp[imp["bpsf"].notna()].copy()
    imp["implied_psf"] = (imp["price"] - imp["bpsf"] * imp["ba"]) / imp["la"]
    imp = imp[imp["implied_psf"] > 0]
    L_prop = imp.groupby("nbhd")["implied_psf"].median()
    n_clusters = imp.groupby("nbhd")[F.impr_he_id].nunique()
    L_market = L_prop.combine_first(L_direct)

    chk = pd.DataFrame({"direct": L_direct, "prop": L_prop}).dropna()
    rho = spearmanr(chk["direct"], chk["prop"])[0] if len(chk) >= 5 else np.nan
    ratio = float((chk["prop"] / chk["direct"]).median()) if len(chk) else np.nan
    cc = imp.groupby(["nbhd", F.impr_he_id])["implied_psf"].median().reset_index()
    disp = cc.groupby("nbhd")["implied_psf"].apply(lambda x: calc_cod(x.values) if len(x) >= 3 else np.nan).dropna()
    return dict(L_market=L_market, L_direct=L_direct, L_prop=L_prop, n_clusters=n_clusters,
                consistency=dict(n=len(chk), rho=float(rho) if rho == rho else np.nan, ratio=ratio,
                                 cross_cluster_cod=float(disp.median()) if len(disp) else np.nan))


def location_differentials(u, obs, cfg):
    """Tier-2 — composition-controlled relative location premiums from matched building models
    (cost-residual sales). Returns (per-zone DataFrame, summary dict)."""
    F = cfg.fields
    ui = u.drop_duplicates("key").set_index("key")
    r = obs[obs["kind"] == "cost_residual"].copy()
    if not len(r):
        return pd.DataFrame(), dict(n=0, note="no cost-residual stream")
    r["model"] = r["key"].map(ui[F.impr_he_id])
    r["land_psf"] = r["observed_land"] / r["land_sqft"]
    r = r[(r["land_psf"] > 0) & r["model"].notna() & r["nbhd"].notna()].copy()
    r["lp"] = np.log(r["land_psf"])
    multi = set(r.groupby("model")["nbhd"].nunique().loc[lambda x: x >= 2].index)
    rr = r[r["model"].isin(multi)].copy()
    if len(rr) < 20:
        return pd.DataFrame(), dict(n=len(rr), note="too few matched-model sales")
    rr["lp_dm"] = rr["lp"] - rr.groupby("model")["lp"].transform("mean")
    loc_premium = rr.groupby("nbhd")["lp_dm"].mean()
    raw = r.groupby("nbhd")["lp"].mean()
    z = pd.DataFrame({"loc": loc_premium, "raw": raw}).dropna()
    rho = spearmanr(z["loc"], z["raw"])[0] if len(z) >= 5 else np.nan
    prem = np.exp(loc_premium + r["lp"].mean())
    diffs = []
    for m, g in rr.groupby("model"):
        zl = g.groupby("nbhd")["lp"].mean()
        if len(zl) >= 2:
            zl = zl - zl.mean(); diffs += [(i, v) for i, v in zl.items()]
    dd = pd.DataFrame(diffs, columns=["nbhd", "dlp"])
    cons = dd.groupby("nbhd")["dlp"].agg(["std", "count"])
    cons_std = float(cons[cons["count"] >= 3]["std"].median()) if len(cons) else np.nan
    out = pd.DataFrame({"loc_premium_psf": prem, "n_matched_models": rr.groupby("nbhd")["model"].nunique()}).dropna()
    return out, dict(n_zones=len(out), n_models=len(multi), rho_vs_raw=float(rho) if rho == rho else np.nan,
                     premium_p10=float(np.percentile(prem, 10)), premium_p90=float(np.percentile(prem, 90)),
                     cross_model_consistency_std=cons_std)
