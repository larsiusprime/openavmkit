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


def build_land_observations(s, u, cfg):
    """Return observed land values across the evidence streams selected by the config's filters.

    Columns: key, kind {direct, cost_residual}, observed_land, land_sqft, observed_land_sqft, nbhd.
    One row per parcel per kind (deduped on key)."""
    F = cfg.fields
    ucols = u.drop_duplicates(F.key).set_index(F.key)
    s = s.copy()
    s[F.key] = s[F.key].astype(str)
    price = pd.to_numeric(s[cfg.dep], errors="coerce")
    price = price.where(price > 0, pd.to_numeric(s[F.sale_price], errors="coerce"))  # coalesce raw
    land_sqft = s[F.key].map(ucols[F.land_area])
    nbhd = s[F.key].map(ucols[F.neighborhood])

    def _pack(mask, observed_land, kind):
        d = pd.DataFrame({"key": s[F.key], "kind": kind, "observed_land": observed_land,
                          "land_sqft": land_sqft, "nbhd": nbhd})[mask.to_numpy()]
        d = d[d["observed_land"].notna() & (d["observed_land"] > 0) & (d["land_sqft"] > 0)]
        return d.drop_duplicates("key")

    frames = []
    direct = _pack(resolve_filter(s, cfg.land_evidence_filter), price, "direct")
    if cfg.prime.enabled and len(direct):
        direct = direct[prime_comp(direct, u, cfg)]
    frames.append(direct)

    if cfg.cost_residual_filter is not None:
        cost = pd.to_numeric(s[F.key].map(ucols[F.cost_bldg_value]), errors="coerce")
        frames.append(_pack(resolve_filter(s, cfg.cost_residual_filter), price - cost, "cost_residual"))

    obs = pd.concat(frames, ignore_index=True)
    obs["observed_land_sqft"] = obs["observed_land"] / obs["land_sqft"]
    return obs


def summarize(obs):
    counts = obs.groupby("kind").size().to_dict()
    return (f"land observations: total={len(obs)}  " +
            "  ".join(f"{k}={counts.get(k, 0)}" for k in ("direct", "cost_residual")))
