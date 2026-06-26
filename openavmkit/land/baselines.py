"""Baseline land painters — the floor the real model must beat.

Each painter takes ``(u, train_obs, cfg)`` and returns a copy of the universe with the three series
columns filled. They calibrate only on ``train_obs`` (the train fold of the land-observation pool),
so the harness can score them out-of-fold. These are deliberately crude — pure direct-evidence
allocation with no borrowing, reconciliation, size curve, or adjustments — to establish a baseline
and exercise the harness before the ported painter lands.

``land`` is predicted from location + lot ONLY (never improvement features) — the non-circularity
rule that the whole exercise rests on. ``impr`` is set to ``total - land`` (reconcile-to-total) using
the assessor total as a stand-in until an independent total series is wired (Phase 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _finalize(u, cfg, land):
    """Attach land/impr/total series columns by FORWARD composition: land = the painted land model,
    impr = the trusted frozen-SOV building (``assr_impr_value``, locked decision #1), total =
    land + impr. No clamp-to-total — clamping land at an external total couples it back to building
    value and re-breaks A1 (the very failure we exist to avoid). Reconciliation against an
    *independent* total is Phase 5, once such a total series is wired."""
    F = cfg.fields
    m = u.copy()
    land = pd.to_numeric(pd.Series(np.asarray(land), index=m.index), errors="coerce").clip(lower=0)
    impr = pd.to_numeric(m[F.cost_bldg_value], errors="coerce").clip(lower=0)
    m[F.land_value] = land
    m[F.impr_value] = impr
    m[F.total_value] = land + impr
    return m


def paint_flat(u, train_obs, cfg):
    """Rung 0: ONE global $/sqft (median of direct evidence) painted on every parcel.

    The crudest possible allocation — no spatial differentiation, so it should fail A5 (desirability
    gradient). The floor of floors."""
    F = cfg.fields
    direct = train_obs[train_obs["kind"] == "direct"]
    rate = float(direct["observed_land_sqft"].median())
    land = pd.to_numeric(u[F.land_area], errors="coerce") * rate
    return _finalize(u, cfg, land)


def paint_neighborhood_rate(u, train_obs, cfg, min_n=3):
    """Rung 1: one $/sqft per neighborhood (median direct-evidence rate), global-median fallback for
    neighborhoods with too few train observations. The simplest spatially-aware land model."""
    F = cfg.fields
    direct = train_obs[train_obs["kind"] == "direct"].copy()
    by_nbhd = direct.groupby("nbhd")["observed_land_sqft"].agg(["median", "count"])
    rate = by_nbhd.loc[by_nbhd["count"] >= min_n, "median"]
    global_rate = float(direct["observed_land_sqft"].median())
    nbhd_rate = u[F.neighborhood].map(rate).fillna(global_rate)
    land = pd.to_numeric(u[F.land_area], errors="coerce") * nbhd_rate.to_numpy()
    return _finalize(u, cfg, land)


# registry — imported at module end so painter.py can import _finalize from here first
from openavmkit.land.painter import paint_size_curve  # noqa: E402

PAINTERS = {"flat": paint_flat, "neighborhood_rate": paint_neighborhood_rate,
            "size_curve": paint_size_curve}
