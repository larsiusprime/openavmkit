"""
LYCD uniform-rate land painter — "Least You Can Do" / Rung 1.

Implements the user's *Valuing Land: The Simplest Viable Method* essay
allocation rule (the so-called "Right Way", contrasted with the "Wrong Way"
``pct × per_parcel_total_value`` that ties land value to improvements):

  Within each neighborhood, derive ONE ``$/sqft`` rate from the prevailing
  improved-property value, and paint that uniform rate on every parcel in
  the cell.

This is the simplest viable painter — Rung 1.0 uses a single global
allocation %, Rung 1.1 calibrates per-neighborhood. Both share this code.
The production-grade :mod:`openavmkit.land.tables` painter (Rung 1.5+)
extends the same allocation rule with per-cell tables and zoning-anchored
size curves.

Inputs:

* A parcel universe with ``land_area_sqft``, neighborhood code, and
  optional ``zoning_emp_min_lot_sqft`` from
  :func:`openavmkit.zoning.join_empirical_zoning`.
* A witness pool from :mod:`openavmkit.land.evidence`. Used to derive
  the local allocation % when the caller wants per-neighborhood
  calibration; ignored if the caller passes a fixed global allocation.
* A :class:`HierarchySpec` from :mod:`openavmkit.neighborhoods` for
  cascading fallback when the finest-grained cell has too few witnesses.

Outputs (added to a copy of the universe):

* ``land_value`` — the per-parcel painted land value.
* ``land_value_per_sqft`` — per-sqft rate applied (uniform within cell).
* ``land_value_cell_level`` — the cascade level whose rate was used.
* ``land_value_excess_flag`` — True if size > 2× zoning min; receives
  a future-work hook for excess vs. surplus differentiation.

See Also
--------
openavmkit.land.evidence : Provides the witness pool used to derive
    allocation percentages and direct land rates.
openavmkit.neighborhoods : Provides the cascade walked when
    a neighborhood is evidence-thin.
openavmkit.zoning : Provides the de-facto zoning floor and FAR ceiling
    used in the surplus-discount split.
openavmkit.land.tests : Lars-Tests harness used to score the output.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openavmkit.neighborhoods import (
    HierarchySpec,
    cascade_aggregate,
    cascade_lookup,
)


# Defaults tied to Wake's Schedule of Values guidance: "typically 15-25%
# in newer residential, may be higher in older" (SOV p.89).
DEFAULT_GLOBAL_ALLOCATION_PCT = 0.20
DEFAULT_ALLOCATION_PRIOR_RANGE = (0.10, 0.40)


@dataclass
class LycdConfig:
    """
    Knobs for the LYCD painter.

    Parameters
    ----------
    global_allocation_pct : float, default 0.20
        Used by Rung 1 (LYCD): single county-wide allocation % applied
        uniformly per cell. Wake's SOV-implied prior is ~0.20 (15-25% range).
    allocation_pct_clip : tuple[float, float], default (0.10, 0.40)
        Clip the per-neighborhood allocation % to this range (used in
        Rung 2 calibration to avoid numerical blowups in evidence-thin
        cells).
    min_witnesses_per_cell : int, default 30
        Minimum witnesses for a cascade level to be considered
        self-supporting.
    excess_size_multiplier : float, default 2.0
        Threshold for flagging excess-vs-surplus (relative to local
        empirical zoning min).
    surplus_discount : float, default 0.50
        Marginal-rate discount applied to land area beyond the
        excess threshold when ``apply_surplus_discount=True``.
    apply_surplus_discount : bool, default True
        If True, multiply the *marginal* $/sqft for area above
        ``excess_size_multiplier × zoning_min`` by ``surplus_discount``.
        This implements a simple version of the surplus-vs-excess
        distinction (the user's PDFs and Wake SOV both call out
        diminishing returns to size).
    """
    global_allocation_pct: float = DEFAULT_GLOBAL_ALLOCATION_PCT
    allocation_pct_clip: tuple = DEFAULT_ALLOCATION_PRIOR_RANGE
    min_witnesses_per_cell: int = 30
    excess_size_multiplier: float = 2.0
    surplus_discount: float = 0.50
    apply_surplus_discount: bool = True


def _per_cell_prevailing(
    universe: pd.DataFrame,
    *,
    spec: HierarchySpec,
    market_value_col: str,
    land_area_col: str,
    bldg_area_col: str,
    min_built_per_cell: int,
) -> dict:
    """
    For each cascade level, compute the *prevailing* improved-property
    median total value AND median improved-property land size. The
    LYCD local rate is then::

        local_rate = (allocation_pct × prevailing_total_value) / prevailing_land_size

    Returns a dict keyed by level name → DataFrame indexed by level
    values with columns ``prevailing_value``, ``prevailing_size``, ``n``.
    """
    df = universe.copy()
    is_built = df[bldg_area_col].fillna(0) > 0
    df = df[is_built & (df[market_value_col] > 0) & (df[land_area_col] > 0)]
    out = {}
    for level in spec.levels:
        if level not in df.columns:
            continue
        g = df.groupby(level, dropna=False)
        prev_value = g[market_value_col].median().rename("prevailing_value")
        prev_size = g[land_area_col].median().rename("prevailing_size")
        n = g.size().rename("n")
        cell = pd.concat([prev_value, prev_size, n], axis=1)
        out[level] = cell
    return out


def derive_allocation_pct(
    witnesses: pd.DataFrame,
    *,
    universe: pd.DataFrame,
    spec: HierarchySpec,
    market_value_col: str = "assr_market_value",
    land_area_col: str = "land_area_sqft",
    bldg_area_col: str = "bldg_area_finished_sqft",
    cfg: LycdConfig | None = None,
) -> tuple:
    """
    Derive a single global allocation % from the witness pool.

    Method: take the median ``land_value_per_sqft`` across all witnesses
    in each neighborhood with adequate coverage, divide by the median
    improved-property ``$/sqft`` in the same neighborhoods, then take the
    median of those ratios as the global allocation %. Clipped to
    ``cfg.allocation_pct_clip``.

    Parameters
    ----------
    witnesses : pandas.DataFrame
        Output of :func:`openavmkit.land.evidence.curate_witnesses`. Must contain
        ``parcel_key``, ``land_value_per_sqft``, ``weight``.
    universe : pandas.DataFrame
        Parcel universe. Must contain ``market_value_col``,
        ``land_area_col``, ``bldg_area_col``, and the cascade levels.
    spec : HierarchySpec
        Cascade ladder. Used here only to identify which level is
        "neighborhood" (level 0).
    market_value_col, land_area_col, bldg_area_col : str
    cfg : LycdConfig

    Returns
    -------
    pct : float
        The derived global allocation %, clipped to the prior range.
    diagnostics : dict
        Keys: ``raw_pct`` (unclipped), ``n_neighborhoods``,
        ``n_witnesses``, ``per_neighborhood_pct`` (Series).
    """
    cfg = cfg or LycdConfig()
    nbhd_col = spec.levels[0]

    # Wire witnesses' parcel-level neighborhood from universe
    universe_keyed = universe.set_index(universe.columns[0]) if False else None
    # Fast join: derive neighborhood for each witness via merge
    if nbhd_col not in witnesses.columns:
        join = universe[[nbhd_col]].copy()
        join.index.name = "parcel_key"
        join = join.reset_index().rename(columns={"index": "parcel_key"})
        # In practice the universe already has its parcel-key as the index or a column
        # called "key"; make this robust:
        if "key" in universe.columns:
            join = universe[["key", nbhd_col]].rename(columns={"key": "parcel_key"})
        elif universe.index.name == "key":
            join = universe.reset_index()[["key", nbhd_col]].rename(
                columns={"key": "parcel_key"}
            )
        else:
            warnings.warn(
                "derive_allocation_pct: cannot find parcel-key column in universe; "
                "using witness $/sqft directly without neighborhood aggregation"
            )
            raw_pct = float(np.nanmedian(witnesses["land_value_per_sqft"]))
            return _clip_alloc(raw_pct, cfg), {
                "raw_pct": raw_pct,
                "n_neighborhoods": 0,
                "n_witnesses": int(len(witnesses)),
                "per_neighborhood_pct": pd.Series(dtype=float),
            }
        wits = witnesses.merge(join, on="parcel_key", how="left")
    else:
        wits = witnesses.copy()

    # Median land $/sqft per neighborhood (witness side)
    wnbhd = wits.groupby(nbhd_col, dropna=False).agg(
        land_per_sqft=("land_value_per_sqft", "median"),
        n_witnesses=("land_value_per_sqft", "size"),
    )
    wnbhd = wnbhd[wnbhd["n_witnesses"] >= 5]

    # Median market $/sqft per neighborhood (improved parcels only)
    df = universe.copy()
    is_built = df[bldg_area_col].fillna(0) > 0
    df = df[is_built & (df[market_value_col] > 0) & (df[land_area_col] > 0)]
    df["__market_per_sqft__"] = df[market_value_col] / df[land_area_col]
    mnbhd = df.groupby(nbhd_col, dropna=False).agg(
        market_per_sqft=("__market_per_sqft__", "median"),
        n_built=("__market_per_sqft__", "size"),
    )
    mnbhd = mnbhd[mnbhd["n_built"] >= 30]

    joined = wnbhd.join(mnbhd, how="inner")
    if len(joined) == 0:
        raw_pct = float(np.nanmedian(wits["land_value_per_sqft"])) / max(
            float(df["__market_per_sqft__"].median()), 1.0
        )
        return _clip_alloc(raw_pct, cfg), {
            "raw_pct": raw_pct,
            "n_neighborhoods": 0,
            "n_witnesses": int(len(wits)),
            "per_neighborhood_pct": pd.Series(dtype=float),
        }

    joined["pct"] = joined["land_per_sqft"] / joined["market_per_sqft"]
    raw_pct = float(joined["pct"].median())
    return _clip_alloc(raw_pct, cfg), {
        "raw_pct": raw_pct,
        "n_neighborhoods": int(len(joined)),
        "n_witnesses": int(len(wits)),
        "per_neighborhood_pct": joined["pct"],
    }


def _clip_alloc(pct: float, cfg: LycdConfig) -> float:
    if not np.isfinite(pct):
        return cfg.global_allocation_pct
    lo, hi = cfg.allocation_pct_clip
    return float(min(max(pct, lo), hi))


def paint_lycd(
    universe: pd.DataFrame,
    *,
    spec: HierarchySpec,
    allocation_pct: float,
    cfg: LycdConfig | None = None,
    market_value_col: str = "assr_market_value",
    land_area_col: str = "land_area_sqft",
    bldg_area_col: str = "bldg_area_finished_sqft",
    zoning_emp_min_col: str = "zoning_emp_min_lot_sqft",
    out_value_col: str = "land_value",
    out_per_sqft_col: str = "land_value_per_sqft",
    out_level_col: str = "land_value_cell_level",
    out_excess_col: str = "land_value_excess_flag",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Paint each parcel with a uniform $/sqft rate derived from the
    prevailing improved-property value in its cascade cell.

    Parameters
    ----------
    universe : pandas.DataFrame
    spec : HierarchySpec
        Cascade ladder. The first level is the finest neighborhood.
    allocation_pct : float
        Land-allocation percentage to apply: multiplied by the prevailing
        improved-property value at each cascade level (uniform per cell),
        not per parcel.
    cfg : LycdConfig
    market_value_col : str
    land_area_col, bldg_area_col, zoning_emp_min_col : str
    out_value_col, out_per_sqft_col, out_level_col, out_excess_col : str
    verbose : bool

    Returns
    -------
    pandas.DataFrame
        Copy of ``universe`` with land-value columns added.
    """
    cfg = cfg or LycdConfig()

    # Build the per-cell prevailing aggregates
    prevailing = _per_cell_prevailing(
        universe,
        spec=spec,
        market_value_col=market_value_col,
        land_area_col=land_area_col,
        bldg_area_col=bldg_area_col,
        min_built_per_cell=cfg.min_witnesses_per_cell,
    )

    # Convert to per-cell $/sqft = pct * prev_value / prev_size
    # so we can reuse cascade_lookup directly.
    rate_aggregates = {}
    for level, cell in prevailing.items():
        rate = (allocation_pct * cell["prevailing_value"]) / cell["prevailing_size"].replace(0, np.nan)
        rate.name = "rate"
        rate.attrs["n"] = cell["n"].rename("n")
        rate_aggregates[level] = rate

    rate_per_sqft, levels_used = cascade_lookup(
        universe,
        spec=spec,
        aggregates=rate_aggregates,
        min_n=cfg.min_witnesses_per_cell,
    )

    out = universe.copy()
    out[out_per_sqft_col] = rate_per_sqft.values
    out[out_level_col] = levels_used.values

    # Compute land value, with optional surplus discount on the marginal area
    base_size = out[land_area_col].fillna(0)
    if cfg.apply_surplus_discount and zoning_emp_min_col in out.columns:
        zmin = out[zoning_emp_min_col].fillna(0)
        excess_threshold = cfg.excess_size_multiplier * zmin
        full_part = np.where(base_size > excess_threshold, excess_threshold, base_size)
        surplus_part = np.where(
            base_size > excess_threshold, base_size - excess_threshold, 0.0
        )
        out[out_excess_col] = base_size > excess_threshold
        rate_full = out[out_per_sqft_col].fillna(0)
        out[out_value_col] = (
            full_part * rate_full + surplus_part * rate_full * cfg.surplus_discount
        )
    else:
        out[out_excess_col] = False
        out[out_value_col] = base_size * out[out_per_sqft_col].fillna(0)

    if verbose:
        n_painted = out[out_value_col].notna().sum()
        n_zero = (out[out_value_col] == 0).sum()
        print(
            f"paint_lycd: painted {n_painted:,}/{len(out):,} parcels; "
            f"{n_zero:,} got 0 (no rate available)"
        )
        if "land_value_cell_level" in out.columns:
            level_counts = out[out_level_col].value_counts(dropna=False)
            print("  cells used (top):")
            for lvl, n in level_counts.head(8).items():
                print(f"    {lvl!r:30s} {n:>7,}")

    return out
