"""Evidence-calibrated land painter — the production land model (Phase 3).

Keeps the proven *math* of the ``docs_land`` painter — the 3-tier marginal-rate size curve
(full rate to the local median lot, half to p90, quarter beyond: the depth-curve / excess-land
insight) and the base-lot-vs-size-curve decision — but replaces its calibration **target**. The
``docs_land`` version balanced each cell to ``sum(prediction − assr_impr)``; here each cell's rate is
calibrated **directly to observed land evidence** (`lvi.evidence.build_land_observations`: direct
vacant sales + frozen-SOV cost residuals). That makes the model:

  * **held-out-able** — calibrate on the train fold, score on the test fold (the harness contract);
  * **non-circular** — land is a function of location + lot only, never improvement features;
  * **reconciled forward** — total = painted land + frozen-SOV building (locked decision #1).

Coverage uses a simple cascade: a neighborhood with enough evidence gets its own table; everything
else falls back to a global table. The smarter borrow cascade (matched-building, paired-sale,
similarity) is Phase 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from openavmkit.land.baselines import _finalize

DEFAULT_TIER_DECAYS = (1.0, 0.5, 0.25)   # marginal-rate decay: full / half / quarter
BASE_LOT_CV_THRESHOLD = 0.20             # lot-size CV below this => uniform tract => base_lot
TIER_PERCENTILES = (0.50, 0.90)          # breakpoints at local p50 / p90 of built lot sizes
MIN_OBS_PER_CELL = 5                     # evidence needed for a cell to get its own table
GLOBAL = ("__global__", "all")


@dataclass
class Tier:
    max_sqft: float | None               # None = unbounded top tier
    rate: float                          # marginal $/sqft within this bracket


@dataclass
class LandTable:
    cell_key: str
    cell_level: str
    approach: str                        # 'base_lot' | 'size_curve'
    base_lot_value: float | None = None
    tiers: list = field(default_factory=list)
    median_lot_sqft: float | None = None
    cv_lot_sqft: float | None = None
    n_built: int = 0
    n_obs: int = 0


def _apply_size_curve(land_areas, tiers):
    """Vectorized 3-tier marginal-rate evaluation (ported verbatim from docs_land — good math)."""
    if not tiers:
        return np.zeros_like(land_areas, dtype=float)
    cum = np.zeros_like(land_areas, dtype=float)
    last = 0.0
    for t in tiers:
        top = t.max_sqft if t.max_sqft is not None else np.inf
        cum += np.clip(land_areas - last, 0, top - last) * t.rate
        last = top
    return cum


def _weighted_area(sizes, p50, p90, decays):
    """Tier-weighted area: the size-curve shape with tier-1 rate factored out (for calibration)."""
    t1 = np.minimum(sizes, p50)
    t2 = np.clip(sizes - p50, 0, p90 - p50)
    t3 = np.maximum(sizes - p90, 0)
    return t1 * decays[0] + t2 * decays[1] + t3 * decays[2]


def _fit_cell(cell_key, level, built_sizes, obs, cfg, decays=DEFAULT_TIER_DECAYS):
    """Build a LandTable for one cell, calibrating the rate/level to the cell's observed land.

    ``obs`` rows carry ``land_sqft`` and ``observed_land`` (direct + cost-residual evidence)."""
    obs = obs[(obs["land_sqft"] > 0) & (obs["observed_land"] > 0)]
    if len(obs) < MIN_OBS_PER_CELL:
        return None
    sizes = pd.to_numeric(pd.Series(built_sizes), errors="coerce").dropna()
    med = float(sizes.median()) if len(sizes) else float(obs["land_sqft"].median())
    cv = float(sizes.std() / med) if (len(sizes) >= 3 and med > 0) else None
    t = LandTable(cell_key=str(cell_key), cell_level=level, approach="size_curve",
                  median_lot_sqft=med, cv_lot_sqft=cv, n_built=int(len(sizes)), n_obs=int(len(obs)))

    # uniform tract subdivision => flat per-lot value (the standard-lot rate)
    if cv is not None and cv < BASE_LOT_CV_THRESHOLD:
        t.approach = "base_lot"
        t.base_lot_value = float(obs["observed_land"].median())
        return t

    # size curve: breakpoints from built-lot distribution; tier-1 rate = ratio estimator on evidence
    p50 = med
    p90 = float(sizes.quantile(0.90)) if len(sizes) >= 5 else med * 1.5
    if p90 <= p50:
        p90 = p50 * 1.5
    w = _weighted_area(obs["land_sqft"].to_numpy(float), p50, p90, decays)
    # robust calibration: tier-1 rate = MEDIAN per-parcel unit rate (centers the ratio distribution
    # for COD), not the aggregate sum/sum (which big lots leverage).
    unit = obs["observed_land"].to_numpy(float) / np.where(w > 0, w, np.nan)
    unit = unit[np.isfinite(unit) & (unit > 0)]
    rate1 = float(np.median(unit)) if len(unit) else 0.0
    t.tiers = [Tier(p50, rate1 * decays[0]), Tier(p90, rate1 * decays[1]), Tier(None, rate1 * decays[2])]
    return t


def build_tables(u, train_obs, cfg, levels=None):
    """Build per-cell land tables at each level in ``levels`` (finest first) + a global fallback."""
    F = cfg.fields
    levels = levels or [F.neighborhood]
    built = u[pd.to_numeric(u[F.bldg_area], errors="coerce").fillna(0) > 0]
    tables = {}
    for level in levels:
        if level not in u.columns:
            continue
        built_by = {c: g[F.land_area] for c, g in built.groupby(level, dropna=False)}
        obs_by = dict(tuple(train_obs.groupby("nbhd", dropna=False))) if level == F.neighborhood else {}
        for cell, cell_obs in obs_by.items():
            tbl = _fit_cell(cell, level, built_by.get(cell, pd.Series(dtype=float)), cell_obs, cfg)
            if tbl is not None:
                tables[(level, str(cell))] = tbl
    # global fallback table over all evidence
    g = _fit_cell("all", GLOBAL[0], built[F.land_area], train_obs, cfg)
    if g is not None:
        tables[GLOBAL] = g
    return tables


def _paint_cell(land_areas, t):
    if t.approach == "base_lot":
        return np.full(len(land_areas), t.base_lot_value or 0.0, dtype=float)
    return _apply_size_curve(land_areas, t.tiers)


SPATIAL_K = 12                  # neighbors for the spatial borrow
SPATIAL_MIN = 4                 # need at least this many evidence points to borrow spatially
DIRECT_WEIGHT = 1.0             # extra weight on direct (vacant-sale) evidence in the borrow
BORROW_NODES = "nbhd"           # "nbhd" = denoised per-neighborhood nodes; "obs" = raw points


def _global_shape(u, train_obs, cfg, decays=DEFAULT_TIER_DECAYS):
    """Global size-curve breakpoints (p50/p90 of built lots) used by the spatial fallback so that
    borrowed parcels still get big-lot decay; the *level* is supplied per-parcel by the kNN borrow."""
    F = cfg.fields
    sizes = pd.to_numeric(u.loc[pd.to_numeric(u[F.bldg_area], errors="coerce").fillna(0) > 0,
                                F.land_area], errors="coerce").dropna()
    p50 = float(sizes.median()) if len(sizes) else float(train_obs["land_sqft"].median())
    p90 = float(sizes.quantile(0.90)) if len(sizes) >= 5 else p50 * 1.5
    return p50, (p90 if p90 > p50 else p50 * 1.5)


def _spatial_borrow(u_target, train_obs, cfg, p50, p90, decays=DEFAULT_TIER_DECAYS,
                    k=SPATIAL_K):
    """Phase-4 borrow: each target parcel gets a spatially-smoothed tier-1 rate from the k nearest
    evidence observations (distance-weighted, log space), applied through the global size-curve shape.
    Replaces a single global rate with a local one — location is the dominant land signal."""
    from scipy.spatial import cKDTree
    F = cfg.fields
    ev = train_obs.copy()
    ev["la"] = pd.to_numeric(ev["land_sqft"], errors="coerce")
    ev["lat"] = ev["key"].map(_UCOLS[F.latitude]); ev["lon"] = ev["key"].map(_UCOLS[F.longitude])
    ev = ev[(ev["la"] > 0) & (ev["observed_land"] > 0) & ev["lat"].notna() & ev["lon"].notna()]
    w = _weighted_area(ev["la"].to_numpy(float), p50, p90, decays)
    ev = ev.assign(tier1=ev["observed_land"].to_numpy(float) / np.where(w > 0, w, np.nan))
    ev = ev[np.isfinite(ev["tier1"]) & (ev["tier1"] > 0)]
    if len(ev) < SPATIAL_MIN:
        return None
    if BORROW_NODES == "nbhd":
        # denoise: one node per neighborhood = its obs centroid + MEDIAN tier-1 rate. Interpolating
        # between stable per-neighborhood rates beats interpolating over noisy per-parcel points.
        ev["__d"] = (ev["kind"] == "direct")
        g = ev.groupby("nbhd").agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                   tier1=("tier1", "median"), n=("tier1", "size"),
                                   anyd=("__d", "max"))
        ev = g.reset_index()
    ev_xy = ev[["lat", "lon"]].to_numpy(float)
    ev_log = np.log(ev["tier1"].to_numpy(float))
    is_direct = (ev["anyd"].to_numpy() if BORROW_NODES == "nbhd" else (ev["kind"].to_numpy() == "direct"))
    ev_kw = np.where(is_direct, DIRECT_WEIGHT, 1.0)  # gold-evidence upweight
    tree = cKDTree(ev_xy)
    tlat = pd.to_numeric(u_target[F.latitude], errors="coerce").to_numpy(float)
    tlon = pd.to_numeric(u_target[F.longitude], errors="coerce").to_numpy(float)
    kk = min(k, len(ev))
    dist, idx = tree.query(np.column_stack([tlat, tlon]), k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    wgt = ev_kw[idx] / (dist + 1e-9)
    tier1 = np.exp(np.sum(ev_log[idx] * wgt, axis=1) / np.sum(wgt, axis=1))
    la = pd.to_numeric(u_target[F.land_area], errors="coerce").fillna(0).to_numpy(float)
    return tier1 * _weighted_area(la, p50, p90, decays)


# module-level handle to the universe-key lookup, set per paint call (avoids re-deriving in helpers)
_UCOLS = None


def paint_size_curve(u, train_obs, cfg, levels=None):
    """Phase-3/4 painter (baseline signature): evidence-calibrated neighborhood tables where the
    evidence supports them, a spatial kNN borrow everywhere else, forward-composed with frozen
    building. The spatial borrow replaces the old single-global-rate fallback (the vacant-COD fix)."""
    global _UCOLS
    F = cfg.fields
    levels = levels or [F.neighborhood]
    _UCOLS = u.drop_duplicates(F.key).set_index(F.key)
    tables = build_tables(u, train_obs, cfg, levels=levels)
    land = pd.to_numeric(u[F.land_area], errors="coerce").fillna(0).to_numpy(float)
    out = np.full(len(u), np.nan)
    src = np.array(["none"] * len(u), dtype=object)   # provenance: table / borrow / global
    used = np.zeros(len(u), dtype=bool)
    for level in levels:
        if level not in u.columns:
            continue
        cells = u[level].astype(str).to_numpy()
        for (lv, ck), t in tables.items():
            if lv != level:
                continue
            m = (~used) & (cells == ck)
            if m.any():
                out[m] = _paint_cell(land[m], t)
                src[m] = "table"
                used[m] = True
    if (~used).any():                                 # spatial borrow for unanchored parcels
        p50, p90 = _global_shape(u, train_obs, cfg)
        borrowed = _spatial_borrow(u[~used], train_obs, cfg, p50, p90)
        if borrowed is not None:
            out[~used] = borrowed
            src[~used] = "borrow"
        elif GLOBAL in tables:                         # last resort
            out[~used] = _paint_cell(land[~used], tables[GLOBAL])
            src[~used] = "global"
    painted = _finalize(u, cfg, out)
    painted["land_source"] = src
    painted.attrs["land_tables"] = tables
    return painted
