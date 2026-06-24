"""Rendering: the headline scorecard and the per-parcel evidence packet."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openavmkit.lvi.evidence import reconstruct_rcn

# scorecard rows: (label, test-key, field, format)
_ROWS = [
    ("Step1 total: median ratio", "total", "median_ratio", "{:.3f}"),
    ("Step1 total: COD_trim",     "total", "cod",          "{:.1f}"),
    ("A1 improvement partial-R^2","A1",    "partial_r2",   "{:.3f}"),
    ("A2 land $/sqft CHD",        "A2",    "land_chd",     "{:.1f}"),
    ("A2 impr $/sqft CHD",        "A2",    "impr_chd",     "{:.1f}"),
    ("A3 land vs anchors: median","A3",    "median_ratio", "{:.3f}"),
    ("A3 land vs anchors: COD",   "A3",    "cod",          "{:.1f}"),
    ("A5 desirability rho",       "A5",    "rho_market",   "{:.3f}"),
    ("A6 sales-chasing spike",    "A6",    "spike",        "{:.2f}"),
    ("A7 sanity violations %",    "A7",    "pct_violations","{:.2f}"),
    ("A8 impr loc-invariance CHD","A8",    "impr_chd",     "{:.1f}"),
    ("B3 local Moran's I",        "B3",    "moran",        "{:.3f}"),
    ("Vertical equity VEI",       "VE",    "vei",          "{:.1f}"),
]


def _fmt(results, tkey, field, fmt):
    v = results.get(tkey, {}).get(field)
    return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"


def scorecard(results_list, title="LAND VALUE INTEGRITY SCORECARD"):
    """results_list: list of (series_name, results_dict). Returns a printable string."""
    names = [n for n, _ in results_list]
    w = 80 + 14 * max(0, len(names) - 1)
    out = ["=" * w, f"  {title}",
           "  " + f"{'metric':<32}" + "".join(f"{n:>14}" for n in names) + "   verdict", "-" * w]
    for label, tkey, field, fmt in _ROWS:
        cells = "".join(f"{_fmt(r, tkey, field, fmt):>14}" for _, r in results_list)
        verdicts = "/".join(r.get(tkey, {}).get("verdict", "?") for _, r in results_list)
        out.append(f"  {label:<32}{cells}   {verdicts}")
    out.append("=" * w)
    for tkey in ("total", "A1", "A2", "A3", "A5", "A6", "A7", "A8", "B3", "VE"):
        for nm, r in results_list:
            if tkey in r:
                out.append(f"  [{tkey}] {nm}: {r[tkey].get('detail', '')}")
    return "\n".join(out)


def evidence_packet(u, obs, flags, cfg, prop=None):
    """Per-parcel defensibility table for one land series. Attaches the propagated market land +
    support type when `prop` (from support.propagate_land_levels) is given."""
    F, C = cfg.fields, cfg.confidence
    d = u[["key", F.land_value, F.impr_value, F.total_value, F.land_area, F.neighborhood, F.land_he_id]].copy()
    d["land_psf"] = pd.to_numeric(d[F.land_value], errors="coerce") / pd.to_numeric(d[F.land_area], errors="coerce")
    d["cluster_median_psf"] = d.groupby(F.land_he_id)["land_psf"].transform("median")
    d["cluster_ratio"] = d["land_psf"] / d["cluster_median_psf"]
    ui = u.set_index("key")
    if F.pctgood and F.pctgood in u.columns:
        d["cost_rcn"] = reconstruct_rcn(ui, F.cost_bldg_value, F.pctgood).reindex(d["key"]).values
    d["residual_land"] = pd.to_numeric(d[F.total_value], errors="coerce").values - \
        pd.to_numeric(ui.loc[d["key"], F.cost_bldg_value], errors="coerce").values
    d["residual_ratio"] = pd.to_numeric(d[F.land_value], errors="coerce") / d["residual_land"].where(d["residual_land"] > 0)
    nb_psf = (obs["observed_land"] / obs["land_sqft"]).groupby(obs["nbhd"]).median()
    nb_n = obs.groupby("nbhd")["observed_land"].count()
    d["nbhd_anchor_psf"] = d[F.neighborhood].map(nb_psf)
    d["nbhd_anchor_n"] = d[F.neighborhood].map(nb_n).fillna(0).astype(int)
    d["anchor_ratio"] = d["land_psf"] / d["nbhd_anchor_psf"]
    if prop is not None:
        d["market_land_psf"] = d[F.neighborhood].map(prop["L_market"])
        d["support"] = np.where(d[F.neighborhood].isin(prop["L_direct"].index), "direct",
                         np.where(d[F.neighborhood].isin(prop["L_prop"].index), "matched-building", "none"))
        d["n_bridge_clusters"] = d[F.neighborhood].map(prop["n_clusters"]).fillna(0).astype(int)
    d = d.merge(flags, on="key", how="left")
    flag_cols = [c for c in d.columns if c.startswith("flag_")]
    d["n_flags"] = d[flag_cols].fillna(False).astype(bool).sum(axis=1)
    rr = pd.to_numeric(d["residual_ratio"], errors="coerce")
    ar = pd.to_numeric(d["anchor_ratio"], errors="coerce")
    cr = pd.to_numeric(d["cluster_ratio"], errors="coerce")
    ok_resid = rr.isna() | rr.between(*C.residual_band)
    ok_anchor = (d["nbhd_anchor_n"] >= C.min_anchor_n) & ar.between(*C.anchor_band)
    ok_cluster = cr.between(*C.cluster_band)
    d["land_integrity_confidence"] = np.where(
        d["n_flags"] > 0, "low", np.where(ok_cluster & (ok_anchor | ok_resid), "high", "med"))
    return d
