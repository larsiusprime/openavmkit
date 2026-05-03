"""
Per-neighborhood land tables — the production-grade output of the
Right-Way painter.

Each neighborhood gets one of three rule structures:

* ``base_lot`` — flat ``$/lot`` value, used in tract subdivisions where
  the within-cell lot-size CV is small (Wake SOV's "Base-Lot method").
* ``size_curve`` — a 3-tier marginal ``$/sqft`` schedule, like an
  income-tax bracket, with breakpoints fit to the local lot-size
  distribution (Wake SOV's "Comparative Unit method"). Tier rates
  decay by the user-PDF's depth-curve insight: full rate up to the
  local median, half rate to local p90, quarter rate beyond.
* ``puv`` — state-set Present-Use-Value rate schedule for
  agricultural / horticultural / forestry parcels.

Adjustments are layered on top: multiplicative factors per
parcel-level feature (PUV deferred, historic deferred, partial
utilities, etc.). For v1 we apply a small set of empirically-derived
factors; future work would derive more from paired-sales analysis or
SHAP contributions from tree-based models.

Reconciliation: each neighborhood's table is calibrated so that the
*sum* of painted-land-from-table over that neighborhood's improved
parcels equals the *sum* of cost-residual (``prediction_main −
assr_impr_value``) over the same parcels. The aggregate is preserved;
per-parcel values can differ from a direct cost-residual subtraction
(by design — the table is a smoothed, rule-based approximation).

The artifacts written by the painter (driven from the run script):

* ``land_tables.parquet`` — one row per neighborhood with full
  table + adjustment summary.
* ``land_universe.parquet`` — per-parcel painted land value,
  ``table_used``, ``approach``, adjustment factors, and a
  human-readable explanation string.
* ``evidence_packets.parquet`` — long table of
  (neighborhood, witness_id, $/sqft) so any per-parcel land value
  can be traced back to its supporting witnesses.
* ``ratio_study.md`` — three reconciliation studies of the painted
  total (= table land + assr_impr) vs (a) ensemble prediction,
  (b) assr_market_value, (c) time-adjusted improved sales.

See Also
--------
openavmkit.land_lycd : Simpler uniform-rate painter (this module is
    the production-grade replacement when audit-ability is required).
openavmkit.land_evidence : Witness curation feeding the calibration.
openavmkit.neighborhood_hierarchy : Cascade walked for evidence-thin
    neighborhoods.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from openavmkit.neighborhood_hierarchy import HierarchySpec
from openavmkit.utilities.stats import calc_cod


# Tier-rate decay factors (multiplied against tier-1 rate).
DEFAULT_TIER_DECAYS = (1.0, 0.5, 0.25)
DEFAULT_TIER_PERCENTILES = (0.50, 0.90)  # breakpoints
BASE_LOT_CV_THRESHOLD = 0.20


@dataclass
class Tier:
    """One bracket of a marginal-rate size curve."""
    max_sqft: float | None  # None for unbounded top tier
    marginal_rate_per_sqft: float


@dataclass
class LandTable:
    """
    Per-neighborhood land valuation rule.

    One of three approaches:
    * ``approach == "base_lot"`` → use ``base_lot_value`` directly per parcel.
    * ``approach == "size_curve"`` → step through ``tiers``, applying each
      tier's ``marginal_rate_per_sqft`` to the area in that bracket.
    * ``approach == "puv"`` → use ``puv_rate_per_sqft`` × land area.

    Cell-level metadata (median lot size, CV, witness count) is preserved
    so downstream code can audit the choice.
    """
    cell_key: str             # cascade-level value, e.g. "01RA017" or "RA"
    cell_level: str           # cascade level name, e.g. "neighborhood"
    approach: str             # 'base_lot' | 'size_curve' | 'puv'
    base_lot_value: float | None = None
    puv_rate_per_sqft: float | None = None
    tiers: list = field(default_factory=list)
    median_lot_sqft: float | None = None
    cv_lot_sqft: float | None = None
    n_built_parcels: int = 0
    n_witnesses: int = 0
    median_residual_psf: float | None = None
    cost_residual_total: float | None = None
    confidence: str = "low"   # 'high' | 'medium' | 'low'
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tiers"] = [{"max_sqft": t.max_sqft, "rate": t.marginal_rate_per_sqft}
                      for t in self.tiers]
        return d


# --------------------------------------------------------------------- builders


def _decide_approach(
    cv: float | None,
    n_built: int,
    cv_threshold: float,
    is_puv: bool,
) -> str:
    if is_puv:
        return "puv"
    if cv is None or n_built < 10:
        # default to size_curve when uncertain (more general)
        return "size_curve"
    if cv < cv_threshold:
        return "base_lot"
    return "size_curve"


def _confidence(n_witnesses: int, witness_cod: float | None) -> str:
    if n_witnesses >= 30 and (witness_cod is None or witness_cod < 25):
        return "high"
    if n_witnesses >= 10:
        return "medium"
    return "low"


def _fit_size_curve(
    cell_built_universe: pd.DataFrame,
    cost_residual_total: float,
    *,
    land_area_col: str = "land_area_sqft",
    tier_percentiles: tuple = DEFAULT_TIER_PERCENTILES,
    tier_decays: tuple = DEFAULT_TIER_DECAYS,
) -> tuple[list, dict]:
    """
    Fit a 3-tier marginal-rate size curve so the painted total over the
    cell's built parcels equals ``cost_residual_total``.

    Strategy:
    1. Choose breakpoints at local percentiles (default p50 and p90 of the
       built-parcel land-size distribution).
    2. For each parcel compute its tier-weighted area =
       ``tier1_area * decay1 + tier2_area * decay2 + tier3_area * decay3``.
    3. Sum across the cell. Set the tier-1 rate ``X`` such that
       ``X * sum_weighted_area == cost_residual_total``.
    4. Return tiers with rates ``X * decay_i``.

    Returns
    -------
    tiers : list of Tier
    diag : dict with intermediate quantities
    """
    sizes = cell_built_universe[land_area_col].dropna().values
    if len(sizes) < 3 or cost_residual_total <= 0:
        # Degenerate — return uniform single-tier
        med = float(np.median(sizes)) if len(sizes) else 0.0
        rate = max(cost_residual_total / max(sizes.sum(), 1.0), 0.0)
        return [Tier(max_sqft=None, marginal_rate_per_sqft=rate)], {
            "p50": med, "p90": med, "tier1_rate": rate, "fallback": True
        }

    p50 = float(np.percentile(sizes, tier_percentiles[0] * 100))
    p90 = float(np.percentile(sizes, tier_percentiles[1] * 100))
    if p90 <= p50:
        p90 = p50 * 1.5  # avoid degenerate breakpoints

    tier1_area = np.minimum(sizes, p50)
    tier2_area = np.clip(sizes - p50, 0, p90 - p50)
    tier3_area = np.maximum(sizes - p90, 0)
    weighted = (
        tier1_area * tier_decays[0]
        + tier2_area * tier_decays[1]
        + tier3_area * tier_decays[2]
    )
    sum_weighted = float(weighted.sum())
    if sum_weighted <= 0:
        x = 0.0
    else:
        x = cost_residual_total / sum_weighted

    tiers = [
        Tier(max_sqft=p50, marginal_rate_per_sqft=x * tier_decays[0]),
        Tier(max_sqft=p90, marginal_rate_per_sqft=x * tier_decays[1]),
        Tier(max_sqft=None, marginal_rate_per_sqft=x * tier_decays[2]),
    ]
    return tiers, {
        "p50": p50,
        "p90": p90,
        "tier1_rate": x * tier_decays[0],
        "tier2_rate": x * tier_decays[1],
        "tier3_rate": x * tier_decays[2],
        "sum_weighted_area": sum_weighted,
    }


def build_neighborhood_table(
    cell_key: str,
    cell_level: str,
    cell_universe: pd.DataFrame,
    cell_witnesses: pd.DataFrame,
    *,
    cost_residual_total: float,
    land_area_col: str = "land_area_sqft",
    bldg_area_col: str = "bldg_area_finished_sqft",
    is_puv_cell: bool = False,
    cv_threshold: float = BASE_LOT_CV_THRESHOLD,
) -> LandTable:
    """Build the appropriate land table for a single cascade cell."""
    built = cell_universe[cell_universe[bldg_area_col].fillna(0) > 0]
    sizes = built[land_area_col].dropna()
    if len(sizes) >= 3:
        med = float(sizes.median())
        cv = float(sizes.std() / med) if med > 0 else None
    else:
        med = float(sizes.median()) if len(sizes) else None
        cv = None

    approach = _decide_approach(cv, len(built), cv_threshold, is_puv_cell)
    n_w = int(len(cell_witnesses))
    median_psf = (
        float(cell_witnesses["land_value_per_sqft"].median())
        if n_w > 0
        else None
    )
    witness_cod = None
    if n_w >= 5:
        try:
            witness_cod = float(
                calc_cod(cell_witnesses["land_value_per_sqft"].dropna().values)
            )
        except Exception:
            pass

    table = LandTable(
        cell_key=str(cell_key),
        cell_level=cell_level,
        approach=approach,
        median_lot_sqft=med,
        cv_lot_sqft=cv,
        n_built_parcels=int(len(built)),
        n_witnesses=n_w,
        median_residual_psf=median_psf,
        cost_residual_total=float(cost_residual_total),
        confidence=_confidence(n_w, witness_cod),
    )

    if approach == "base_lot":
        # Per-lot value = cost_residual_total / n_built_parcels
        if len(built) > 0:
            table.base_lot_value = float(cost_residual_total / len(built))
        else:
            table.base_lot_value = 0.0
        table.notes.append(
            f"base_lot: lot-size CV {cv:.2f} < threshold {cv_threshold}"
        )
    elif approach == "puv":
        # Distributing the cost-residual budget by area
        total_sqft = float(cell_universe[land_area_col].sum())
        if total_sqft > 0:
            table.puv_rate_per_sqft = float(cost_residual_total / total_sqft)
        else:
            table.puv_rate_per_sqft = 0.0
        table.notes.append("puv cell — flat rate across deferred parcels")
    else:
        # size_curve
        tiers, diag = _fit_size_curve(built, cost_residual_total, land_area_col=land_area_col)
        table.tiers = tiers
        table.notes.append(
            f"size_curve: breakpoints p50={diag.get('p50',0):.0f}, "
            f"p90={diag.get('p90',0):.0f} sqft; rates "
            f"${diag.get('tier1_rate',0):.2f} → "
            f"${diag.get('tier2_rate',0):.2f} → "
            f"${diag.get('tier3_rate',0):.2f}"
        )
    return table


def build_all_tables(
    universe: pd.DataFrame,
    witnesses: pd.DataFrame,
    *,
    spec: HierarchySpec,
    prediction_col: str = "prediction",
    impr_value_col: str = "assr_impr_value",
    land_area_col: str = "land_area_sqft",
    bldg_area_col: str = "bldg_area_finished_sqft",
    puv_flag_col: str = "land_deferred_code",
    min_built_per_cell: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Build one ``LandTable`` per cascade-cell that meets the minimum
    coverage threshold. Cells below the threshold inherit from the next
    coarser level (parent in the cascade).

    Parameters
    ----------
    universe : pandas.DataFrame
        Parcel universe with cascade-level columns, ``land_area_col``,
        ``bldg_area_col``, ``prediction_col``, ``impr_value_col``,
        and (optionally) ``puv_flag_col``.
    witnesses : pandas.DataFrame
        From :func:`land_evidence.curate_witnesses`. Must have
        ``parcel_key`` and ``land_value_per_sqft``.
    spec : HierarchySpec
        Cascade ladder, finest first.
    prediction_col, impr_value_col : str
        Used to compute the calibration target
        ``cost_residual = prediction − impr_value``.
    land_area_col, bldg_area_col, puv_flag_col : str
    min_built_per_cell : int, default 30
        Minimum built-parcel count for a cell to get its own table.
    verbose : bool

    Returns
    -------
    dict
        Mapping ``(cell_level, cell_key) -> LandTable``.
    """
    # Wire neighborhood column onto witnesses
    nbhd_col = spec.levels[0]
    if nbhd_col not in witnesses.columns:
        # Map via universe's parcel-key
        if "key" in universe.columns:
            join = universe[["key", nbhd_col]].rename(columns={"key": "parcel_key"})
            witnesses = witnesses.merge(join, on="parcel_key", how="left")
        else:
            warnings.warn(
                "build_all_tables: cannot wire neighborhood onto witnesses; "
                "results may be incomplete"
            )

    # Compute cost residual per parcel (only for parcels we'll calibrate from)
    df = universe.copy()
    df["__cost_residual__"] = (
        df[prediction_col].fillna(0) - df[impr_value_col].fillna(0)
    )
    is_built = df[bldg_area_col].fillna(0) > 0
    is_puv = (
        df.get(puv_flag_col, pd.Series([""] * len(df)))
        .astype("string").fillna("").str.strip().ne("")
    )
    df["__is_built__"] = is_built
    df["__is_puv__"] = is_puv

    # Pre-group witnesses by each cascade level once (O(N total) per level
    # instead of O(N × cells)).
    witness_groups_by_level = {}
    for level in spec.levels:
        if level in witnesses.columns:
            witness_groups_by_level[level] = {
                k: v for k, v in witnesses.groupby(level, dropna=False)
            }

    tables = {}
    built_only = df[is_built]
    for level in spec.levels:
        if level not in df.columns:
            continue
        # Single groupby — O(N total). Iterate the groups in this level.
        groups = built_only.groupby(level, dropna=False)
        wg = witness_groups_by_level.get(level, {})
        n_groups = 0
        for cell, cell_universe in groups:
            cell_key = (level, str(cell))
            if level != spec.levels[0] and len(cell_universe) < min_built_per_cell:
                continue
            cell_residual_total = float(cell_universe["__cost_residual__"].sum())
            cell_witnesses = wg.get(cell, pd.DataFrame())
            is_puv_cell = bool(cell_universe["__is_puv__"].any())
            try:
                table = build_neighborhood_table(
                    cell_key=str(cell),
                    cell_level=level,
                    cell_universe=cell_universe,
                    cell_witnesses=cell_witnesses,
                    cost_residual_total=cell_residual_total,
                    land_area_col=land_area_col,
                    bldg_area_col=bldg_area_col,
                    is_puv_cell=is_puv_cell,
                )
                tables[cell_key] = table
                n_groups += 1
            except Exception as e:
                warnings.warn(f"build_all_tables: failed for {cell_key}: {e}")
        if verbose:
            print(f"  built {n_groups:,} tables at level {level!r}")

    if verbose:
        n_by_level = {}
        for (lvl, _), _ in tables.items():
            n_by_level[lvl] = n_by_level.get(lvl, 0) + 1
        print(f"build_all_tables: built {len(tables):,} tables")
        for lvl in spec.levels:
            print(f"  {lvl:25s}  {n_by_level.get(lvl, 0):>5}")

    return tables


# --------------------------------------------------------------------- adjustments


@dataclass
class AdjustmentSpec:
    """Multiplicative factor applied per-parcel based on a feature."""
    feature: str
    factor: float
    description: str
    n_supporting_pairs: int = 0
    source: str = "manual"  # 'manual' | 'paired_sales' | 'shap'


def default_wake_adjustments() -> list:
    """
    Wake-specific v1 adjustment factors. These are conservative defaults
    based on the SOV's published influence codes and Wake's data
    coverage. Refinement via paired-sales or SHAP is future work.
    """
    return [
        AdjustmentSpec(
            feature="puv",
            factor=0.05,
            description="Land in PUV deferral (agriculture/horticulture/forestry)",
            source="manual",
        ),
        AdjustmentSpec(
            feature="historic_deferred",
            factor=0.50,
            description="Historic-preservation deferred parcel",
            source="manual",
        ),
        AdjustmentSpec(
            feature="utilities_partial",
            factor=0.85,
            description="Partial utilities (E only, no water/sewer) — well/septic rural",
            source="manual",
        ),
    ]


def apply_adjustments(
    df: pd.DataFrame,
    *,
    adjustments: list,
    base_value_col: str,
    out_factor_col: str = "land_value_adjustment_factor",
    out_features_col: str = "land_value_adjustments_applied",
) -> pd.DataFrame:
    """
    Apply multiplicative adjustments to the table-painted base value.

    Parameters
    ----------
    df : pandas.DataFrame
    adjustments : list[AdjustmentSpec]
    base_value_col : str
        The pre-adjustment land value column to scale.
    out_factor_col, out_features_col : str

    Returns
    -------
    pandas.DataFrame
        Copy with two added columns: the cumulative multiplicative
        factor and a string listing which features fired.
    """
    out = df.copy()
    factor = pd.Series(1.0, index=out.index, dtype=float)
    features_applied = pd.Series([""] * len(out), index=out.index, dtype="object")

    for adj in adjustments:
        feat_mask = _evaluate_feature(out, adj.feature)
        if feat_mask is None or not feat_mask.any():
            continue
        factor.loc[feat_mask] *= adj.factor
        features_applied.loc[feat_mask] = features_applied.loc[feat_mask].apply(
            lambda existing: existing + (";" if existing else "")
                              + f"{adj.feature}×{adj.factor:.2f}"
        )

    out[out_factor_col] = factor
    out[out_features_col] = features_applied
    out[base_value_col] = out[base_value_col] * factor
    return out


def _evaluate_feature(df: pd.DataFrame, feature: str) -> pd.Series | None:
    """Return a boolean mask for parcels triggering ``feature``."""
    if feature == "puv":
        if "land_deferred_code" not in df.columns:
            return None
        s = df["land_deferred_code"].astype("string").fillna("").str.strip()
        return s.ne("")
    if feature == "historic_deferred":
        if "historic_deferred_code" not in df.columns:
            return None
        s = df["historic_deferred_code"].astype("string").fillna("").str.strip()
        return s.ne("")
    if feature == "utilities_partial":
        if "utilities_code" not in df.columns:
            return None
        s = df["utilities_code"].astype("string").fillna("").str.strip()
        # 'ALL' is full; everything else is partial
        return s.ne("ALL") & s.ne("")
    return None


# --------------------------------------------------------------------- paint


def paint_from_tables(
    universe: pd.DataFrame,
    *,
    tables: dict,
    spec: HierarchySpec,
    adjustments: list | None = None,
    land_area_col: str = "land_area_sqft",
    out_value_col: str = "land_value",
    out_psf_col: str = "land_value_per_sqft",
    out_table_col: str = "table_used",
    out_approach_col: str = "table_approach",
    out_explanation_col: str = "land_value_explanation",
    explanations: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Paint per-parcel land value using neighborhood land tables.

    For each parcel, walks the cascade until it finds a table, then
    applies that table's rule (base_lot / size_curve / puv) to the
    parcel's land area. Multiplicative adjustments are applied last.

    Parameters
    ----------
    explanations : bool, default False
        If True, populate ``out_explanation_col`` with a per-parcel
        human-readable string. Disabled by default — generating these for
        all 377K Wake parcels takes ~30s of Python looping, and they're
        only useful for audit/spot-check workflows. Use the small
        :func:`build_explanations_for_subset` helper to generate strings
        for a subset of parcels (e.g. appeals candidates) on demand.
    """
    out = universe.copy()
    out[out_value_col] = 0.0
    out[out_table_col] = None
    out[out_approach_col] = None
    if explanations:
        out[out_explanation_col] = ""

    unfilled = pd.Series(True, index=out.index)
    for level in spec.levels:
        if not unfilled.any():
            break
        if level not in out.columns:
            continue
        # Per-cell groupby on the unfilled subset
        for cell, sub in out[unfilled].groupby(level, dropna=False):
            key = (level, str(cell))
            if key not in tables:
                continue
            t = tables[key]
            idx = sub.index
            land = sub[land_area_col].fillna(0).values
            if t.approach == "base_lot":
                vals = np.full(len(idx), t.base_lot_value or 0.0, dtype=float)
            elif t.approach == "puv":
                vals = land * (t.puv_rate_per_sqft or 0.0)
            else:
                vals = _apply_size_curve_vec(land, t.tiers)
            out.loc[idx, out_value_col] = vals
            out.loc[idx, out_table_col] = key[1]
            out.loc[idx, out_approach_col] = t.approach
            unfilled.loc[idx] = False

    out[out_psf_col] = out[out_value_col] / out[land_area_col].replace(0, np.nan)

    if adjustments:
        out = apply_adjustments(
            out,
            adjustments=adjustments,
            base_value_col=out_value_col,
        )
        out[out_psf_col] = out[out_value_col] / out[land_area_col].replace(0, np.nan)

    if explanations:
        out[out_explanation_col] = build_explanations_for_subset(
            out, tables, level_col_priority=spec.levels,
            land_area_col=land_area_col,
        )

    if verbose:
        n_painted = (out[out_value_col].fillna(0) > 0).sum()
        n_unfilled = unfilled.sum()
        print(
            f"paint_from_tables: painted {n_painted:,}/{len(out):,} parcels; "
            f"{n_unfilled:,} unfilled (no table covered them)"
        )
        ac = out[out_approach_col].value_counts(dropna=False)
        print("  approach distribution:")
        for k, v in ac.items():
            print(f"    {k!r:15s} {v:>7,}")

    return out


def _apply_size_curve_vec(land_areas: np.ndarray, tiers: list) -> np.ndarray:
    """
    Vectorized size-curve evaluation. Applies each tier's marginal rate
    to the slice of land area within that bracket and sums.

    Parameters
    ----------
    land_areas : np.ndarray
        Per-parcel land areas in sqft.
    tiers : list of Tier
        Tier brackets. The last tier's ``max_sqft`` may be None
        (interpreted as +inf).

    Returns
    -------
    np.ndarray
        Per-parcel land value.
    """
    if len(tiers) == 0:
        return np.zeros_like(land_areas)
    cum = np.zeros_like(land_areas, dtype=float)
    last = 0.0
    for t in tiers:
        top = t.max_sqft if t.max_sqft is not None else np.inf
        tier_area = np.clip(land_areas - last, 0, top - last)
        cum += tier_area * t.marginal_rate_per_sqft
        last = top
    return cum


def build_explanations_for_subset(
    df: pd.DataFrame,
    tables: dict,
    *,
    level_col_priority: list,
    land_area_col: str = "land_area_sqft",
) -> pd.Series:
    """
    Build a human-readable land-value explanation string per parcel.

    Use after ``paint_from_tables`` to attach explanations to a subset
    of parcels (e.g. appeals candidates, spot-checks). For 377K parcels
    this runs in ~30s, which is too slow to do unconditionally during
    full painting — call it on demand.
    """
    out = pd.Series([""] * len(df), index=df.index, dtype=object)
    if "table_used" not in df.columns or "table_approach" not in df.columns:
        return out

    # Identify which level each parcel resolved to by scanning the
    # cascade from finest -> coarsest and matching its key
    table_used = df["table_used"].astype(str)
    table_approach = df["table_approach"].astype(str)

    explanations = []
    for i, row in enumerate(df.itertuples(index=False)):
        approach = getattr(row, "table_approach", None)
        cell = getattr(row, "table_used", None)
        if not approach or not cell or cell == "None":
            explanations.append("")
            continue
        # Find the cell's level by checking which level column matches
        cell_level = None
        for level in level_col_priority:
            if level in df.columns and getattr(row, level, None) == cell:
                cell_level = level
                break
        key = (cell_level, str(cell)) if cell_level else None
        t = tables.get(key) if key else None
        if t is None:
            explanations.append("")
            continue
        a = getattr(row, land_area_col, 0) or 0
        if t.approach == "base_lot":
            expl = (
                f"{cell_level}={cell} (base_lot, n={t.n_built_parcels}). "
                f"$/lot = ${t.base_lot_value:,.0f}."
            )
        elif t.approach == "puv":
            v = a * (t.puv_rate_per_sqft or 0.0)
            expl = (
                f"{cell_level}={cell} (puv). "
                f"${t.puv_rate_per_sqft:.4f}/sqft × {a:,.0f} sqft = ${v:,.0f}."
            )
        else:
            bits = []
            cum = 0.0
            last = 0.0
            for j, tier in enumerate(t.tiers):
                top = tier.max_sqft if tier.max_sqft is not None else float("inf")
                if a <= last:
                    break
                tier_area = min(a, top) - last
                tier_value = tier_area * tier.marginal_rate_per_sqft
                cum += tier_value
                top_lbl = f"≤{int(top):,}" if top != float("inf") else f">{int(last):,}"
                bits.append(
                    f"T{j+1} ({top_lbl}) ${tier.marginal_rate_per_sqft:.2f}/sqft "
                    f"× {int(tier_area):,} = ${tier_value:,.0f}"
                )
                last = top
            expl = (
                f"{cell_level}={cell} (size_curve). " + "; ".join(bits)
                + f" → ${cum:,.0f}."
            )
        explanations.append(expl)
    return pd.Series(explanations, index=df.index, dtype=object)


# --------------------------------------------------------------------- artifacts


def write_land_tables_artifact(tables: dict, out_path: str) -> None:
    """Serialize tables to parquet (one row per cell)."""
    rows = []
    for (level, cell), t in tables.items():
        d = t.to_dict()
        d["cell_level"] = level
        d["cell_key"] = cell
        # Flatten tiers into stringified list for parquet-friendliness
        d["tiers_json"] = json.dumps(d.pop("tiers"))
        d["notes"] = "; ".join(d.pop("notes"))
        rows.append(d)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


def write_evidence_packets(
    witnesses: pd.DataFrame,
    out_path: str,
    *,
    nbhd_col: str = "neighborhood",
) -> None:
    """
    Write the long-form evidence parquet — one row per
    (neighborhood, witness, $/sqft). Used by audit / appeals to trace any
    per-parcel land value back to its supporting market evidence.
    """
    if nbhd_col not in witnesses.columns:
        warnings.warn("write_evidence_packets: witnesses missing neighborhood column")
        return
    cols = [
        nbhd_col,
        "parcel_key",
        "witness_kind",
        "land_value",
        "land_area_sqft",
        "land_value_per_sqft",
        "weight",
        "event_date",
        "notes",
    ]
    have = [c for c in cols if c in witnesses.columns]
    witnesses[have].to_parquet(out_path, index=False)
