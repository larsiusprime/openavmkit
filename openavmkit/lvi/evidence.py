"""Land-value evidence streams.

LVI does NO classification of its own — *which* sales count as evidence is decided by filters the
user owns (``resolve_filter``, the same engine model groups / valid_sale use). This module just
resolves those filters and computes the observed land value per stream:

  - ``direct``        sales matching ``land_evidence_filter`` — the gold-standard direct land
                      evidence (vacancy + qualification + zone relevance, all in the user's filter).
                      observed_land = sale price.
  - ``cost_residual`` sales matching ``cost_residual_filter`` (optional) — improved sales whose
                      building cost is trustworthy. observed_land = sale price − cost_bldg_value.

The one built-in screen is ``prime_comp``: size-comparability to the neighborhood's built lots +
shape sanity. It is jurisdiction-agnostic (no vocabulary), can't be a static filter (the size
bound is a per-neighborhood percentile), and is tunable / disable-able via config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from openavmkit.filters import resolve_filter


def reconstruct_rcn(df, impr_field="assr_impr_value", pctgood_field="bldg_condition_pct",
                    clip=(0.10, 1.20)):
    """Replacement-cost-new from the cost book: RCN = RCNLD / percent_good. NaN where percent-good
    or impr is non-positive; percent-good clipped to ``clip``."""
    pct = pd.to_numeric(df[pctgood_field], errors="coerce") / 100.0
    impr = pd.to_numeric(df[impr_field], errors="coerce")
    pct = pct.where((pct >= clip[0]) & (pct <= clip[1]))
    return impr.where(impr > 0) / pct


def neighborhood_size_bands(u, cfg):
    """Per-neighborhood built-lot size band for this model group (the comparability yardstick)."""
    F, P = cfg.fields, cfg.prime
    b = u[(u[F.model_group] == cfg.model_group) & (u[F.is_vacant] == False) &
          (pd.to_numeric(u[F.bldg_area], errors="coerce") > 0)].copy()
    b["__la"] = pd.to_numeric(b[F.land_area], errors="coerce")
    return b.groupby(F.neighborhood)["__la"].agg(
        peer_n="count",
        peer_lo=lambda x: x.quantile(P.size_lo),
        peer_hi=lambda x: x.quantile(P.size_hi)).reset_index()


def prime_comp(d, u, cfg):
    """General, jurisdiction-agnostic comparability mask over direct-evidence rows ``d``: the lot is
    size-typical for its neighborhood's built peers and not weirdly shaped. Returns a numpy bool
    array aligned to ``d``. No vocabulary; the size bound is a per-neighborhood percentile."""
    F, P = cfg.fields, cfg.prime
    bands = neighborhood_size_bands(u, cfg).set_index(F.neighborhood)
    ucols = u.drop_duplicates(F.key).set_index(F.key)
    la = pd.to_numeric(d["land_sqft"], errors="coerce")
    rect = pd.to_numeric(d["key"].map(ucols[F.rectangularity]), errors="coerce")
    peer_n = d["nbhd"].map(bands["peer_n"])
    lo, hi = d["nbhd"].map(bands["peer_lo"]), d["nbhd"].map(bands["peer_hi"])
    ok = (rect >= P.rect_min) & (peer_n.fillna(0) >= P.min_peers) & la.between(lo, hi) & peer_n.notna()
    return ok.fillna(False).to_numpy()


def build_land_price_index(u, cfg, s=None):
    """A land price index for time-adjusting old transfers, as **two-way (year x neighborhood)
    fixed effects** on log($/sqft) over a pool of land sales — the prior_land_xfer transfers plus
    (when given) the current vacant sales, which anchor the recent end. Returns (index Series keyed
    by year, target_year) or (None, None) if too few. Non-circular: land prices + dates only.

    The year coefficients (relative to the latest year = 0) control for which neighborhoods
    transacted in which years (composition). A simple demean-then-median index instead flattens
    long-run appreciation; FE recovers it (and its residual level bias is corrected at use time by
    recentering to recent vacant sales)."""
    F, E = cfg.fields, cfg.evidence
    if F.prior_xfer_price not in u.columns:
        return None, None
    px = pd.to_numeric(u[F.prior_xfer_price], errors="coerce")
    yr = pd.to_datetime(u[F.prior_xfer_date], errors="coerce").dt.year
    la = pd.to_numeric(u[F.land_area], errors="coerce")
    disq = u[F.prior_xfer_disq].astype(str) if F.prior_xfer_disq in u.columns else pd.Series("", index=u.index)
    psf = px / la
    ok = (px > 0) & (la > 0) & yr.notna() & disq.isin(list(E.prior_xfer_disq)) & psf.between(*E.prior_xfer_psf_bounds)
    pool = pd.DataFrame({"yr": yr, "nb": u[F.neighborhood].astype(str), "psf": psf})[ok.to_numpy()]
    if s is not None:                                  # anchor the recent end with current vacant sales
        sp = pd.to_numeric(s[cfg.dep], errors="coerce")
        sp = sp.where(sp > 0, pd.to_numeric(s[F.sale_price], errors="coerce"))
        sla = pd.to_numeric(s[F.key].astype(str).map(u.drop_duplicates(F.key).set_index(F.key)[F.land_area]), errors="coerce")
        syr = pd.to_datetime(s[F.sale_date], errors="coerce").dt.year
        spsf = sp / sla
        vmask = resolve_filter(s, cfg.land_evidence_filter) & syr.notna() & spsf.between(*E.prior_xfer_psf_bounds)
        vv = pd.DataFrame({"yr": syr, "nb": s[F.neighborhood].astype(str), "psf": spsf})[vmask.to_numpy()]
        pool = pd.concat([pool, vv], ignore_index=True)
    pool = pool[pool["psf"] > 0]
    if len(pool) < 100:
        return None, None
    pool["yr"] = pool["yr"].astype(int)
    pool["lpsf"] = np.log(pool["psf"])
    target = int(pool["yr"].max())
    yd = pd.get_dummies(pool["yr"], prefix="y").astype(float)
    nb = pool["nb"]
    yw = (pool["lpsf"] - pool.groupby(nb)["lpsf"].transform("mean")).to_numpy()
    ydw = (yd - yd.groupby(nb.values).transform("mean")).to_numpy()
    cols = list(yd.columns)
    keep = [i for i, c in enumerate(cols) if c != f"y_{target}"]   # latest year = baseline (0)
    beta, *_ = np.linalg.lstsq(ydw[:, keep], yw, rcond=None)
    idx = {int(cols[i].split("_")[1]): float(beta[j]) for j, i in enumerate(keep)}
    idx[target] = 0.0
    return pd.Series(idx).sort_index(), target


def build_land_observations(s, u, cfg):
    """Return observed land values across the evidence streams selected by the config's filters.

    Columns: key, kind {direct, cost_residual}, observed_land, land_sqft, observed_land_sqft, nbhd.
    One row per parcel per kind (deduped on key)."""
    F, E = cfg.fields, cfg.evidence
    ucols = u.drop_duplicates(F.key).set_index(F.key)
    s = s.copy()
    s[F.key] = s[F.key].astype(str)
    price = pd.to_numeric(s[cfg.dep], errors="coerce")
    price = price.where(price > 0, pd.to_numeric(s[F.sale_price], errors="coerce"))  # coalesce raw
    land_sqft = s[F.key].map(ucols[F.land_area])
    nbhd = s[F.key].map(ucols[F.neighborhood])
    cost_bldg = pd.to_numeric(s[F.key].map(ucols[F.cost_bldg_value]), errors="coerce")
    bldg_area = pd.to_numeric(s[F.key].map(ucols[F.bldg_area]), errors="coerce")

    def _pack(mask, observed_land, kind):
        d = pd.DataFrame({"key": s[F.key], "kind": kind, "observed_land": observed_land,
                          "land_sqft": land_sqft, "nbhd": nbhd})[mask.to_numpy()]
        d = d[d["observed_land"].notna() & (d["observed_land"] > 0) & (d["land_sqft"] > 0)]
        return d.drop_duplicates("key")

    frames = []
    # --- direct (vacant-sale) stream + a-priori validity gates V1/V4/S3 ---
    dmask = resolve_filter(s, cfg.land_evidence_filter)
    if E.exclude_teardowns and "is_teardown_sale" in s.columns:        # V1
        dmask = dmask & ~(s["is_teardown_sale"] == True)
    if E.token_price_floor > 0:                                        # S3
        dmask = dmask & (price >= E.token_price_floor)
    if E.vacant_psf_floor > 0:                                         # V4
        dmask = dmask & ((price / land_sqft) >= E.vacant_psf_floor)
    direct = _pack(dmask, price, "direct")
    if cfg.prime.enabled and len(direct):                             # V3
        direct = direct[prime_comp(direct, u, cfg)]
    frames.append(direct)

    # --- sale-RCN residual stream + a-priori validity gates C4/C5 (+ S3) ---
    if cfg.cost_residual_filter is not None:
        land_resid = price - cost_bldg
        land_share = land_resid / price
        cmask = resolve_filter(s, cfg.cost_residual_filter)
        if E.land_share_lo > 0:                                       # C4 lower
            cmask = cmask & (land_share >= E.land_share_lo)
        if E.land_share_hi < 1.0:                                     # C4 upper
            cmask = cmask & (land_share <= E.land_share_hi)
        if E.rcn_psf_lo > 0:                                          # C5 lower
            cmask = cmask & ((cost_bldg / bldg_area) >= E.rcn_psf_lo)
        if E.rcn_psf_hi > 0:                                          # C5 upper
            cmask = cmask & ((cost_bldg / bldg_area) <= E.rcn_psf_hi)
        if E.token_price_floor > 0:
            cmask = cmask & (price >= E.token_price_floor)
        frames.append(_pack(cmask, land_resid, "cost_residual"))

    # --- prior_land_xfer stream: time-adjusted historical vacant-land transfers (coverage lever) ---
    if E.use_prior_xfer and F.prior_xfer_price in u.columns:
        idx, target = build_land_price_index(u, cfg, s=s)
        if idx is not None:
            px = pd.to_numeric(u[F.prior_xfer_price], errors="coerce")
            yr = pd.to_datetime(u[F.prior_xfer_date], errors="coerce").dt.year
            la_u = pd.to_numeric(u[F.land_area], errors="coerce")
            disq = u[F.prior_xfer_disq].astype(str) if F.prior_xfer_disq in u.columns else pd.Series("", index=u.index)
            psf = px / la_u
            adj = px * np.exp(-yr.map(idx).fillna(0.0))            # bring each transfer to target year
            keep = ((px > 0) & (la_u > 0) & yr.notna() & disq.isin(list(E.prior_xfer_disq))
                    & psf.between(*E.prior_xfer_psf_bounds) & (yr >= target - E.prior_xfer_max_age))
            pf = pd.DataFrame({"key": u[F.key].astype(str), "kind": "prior_xfer", "observed_land": adj,
                               "land_sqft": la_u, "nbhd": u[F.neighborhood]})[keep.to_numpy()]
            pf = pf[(pf["observed_land"] > 0) & (pf["land_sqft"] > 0)].drop_duplicates("key")
            pf = pf[~pf["key"].isin(set(direct["key"]))]          # prefer a recent direct sale
            # recenter to the recent vacant level (corrects the FE index's residual level bias),
            # using neighborhoods that carry both — non-circular (vs recent direct evidence, not model)
            if len(direct) and len(pf):
                vnb = (direct["observed_land"] / direct["land_sqft"]).groupby(direct["nbhd"].values).median()
                pnb = (pf["observed_land"] / pf["land_sqft"]).groupby(pf["nbhd"].values).median()
                sh = pd.DataFrame({"v": vnb, "p": pnb}).dropna()
                if len(sh) >= 5:
                    pf["observed_land"] = pf["observed_land"] * float((sh["v"] / sh["p"]).median())
            if cfg.prime.enabled and len(pf):
                pf = pf[prime_comp(pf, u, cfg)]
            frames.append(pf)

    obs = pd.concat(frames, ignore_index=True)
    obs["observed_land_sqft"] = obs["observed_land"] / obs["land_sqft"]
    return obs


def summarize(obs):
    counts = obs.groupby("kind").size().to_dict()
    return (f"land observations: total={len(obs)}  " +
            "  ".join(f"{k}={counts.get(k, 0)}" for k in ("direct", "cost_residual", "prior_xfer")))
