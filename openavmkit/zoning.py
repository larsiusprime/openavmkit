"""
Empirical zoning summary tables.

Most US assessor data exposes a raw zoning code per parcel (e.g. ``R-4``,
``CMX-12``, ``B1-CZ``) but does not expose the structured limits the underlying
ordinance imposes (minimum lot size, maximum floor area ratio, height caps,
permitted-use tables). Reconstructing those from the actual UDOs across many
municipalities is expensive.

This module derives a *de facto* lookup from the parcel data itself:

* For each ``(jurisdiction, zoning)`` pair with enough built parcels, take
  the 10th-percentile of ``land_area_sqft`` among built parcels as the de
  facto minimum lot size, and the 90th-percentile of observed
  ``floor_area_ratio`` as the de facto FAR ceiling.
* Pairs with too few built parcels fall back to a jurisdiction-level
  aggregate, then to a global aggregate.

The resulting table feeds three downstream uses: the L5 Lars-Test
(density-FAR ordering, via ``zoning_emp_max_far``), the L6 Lars-Test
(per-cell size decay, via ``zoning_emp_min_lot_sqft``), and the W1
vacant-sale contamination filter (drop sales of parcels smaller than
the local de facto minimum).

The known limitation: empirical values bake in current built-out reality, so
they understate the legal maximum where the market hasn't built up to the
zoning cap. They are a substitute for, not equivalent to, real UDO data.

See Also
--------
openavmkit.land.evidence : Uses this table to filter vacant-sale witnesses.
openavmkit.land.tests : Uses this table for L5 (density-FAR ordering)
    and L6 (per-cell size decay).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


_SENTINEL_JURISDICTION = "__GLOBAL__"


def build_empirical_zoning_table(
    df: pd.DataFrame,
    *,
    jurisdiction_col: str = "planning_jurisdiction",
    zoning_col: str = "zoning",
    land_area_col: str = "land_area_sqft",
    bldg_area_col: str = "bldg_area_finished_sqft",
    far_col: str = "floor_area_ratio",
    min_built_parcels_per_pair: int = 30,
    min_built_parcels_per_jurisdiction: int = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a per ``(jurisdiction, zoning)`` empirical summary table.

    Each row records the de facto minimum lot size and de facto FAR ceiling
    for a zoning code in a given jurisdiction, derived from the actual built
    parcels in that group. Pairs with too few built parcels are recorded but
    flagged ``fallback_level`` non-zero so downstream code can decide whether
    to use them, fall back to the jurisdiction aggregate (also in the table),
    or fall back to the global aggregate (also in the table).

    Parameters
    ----------
    df : pandas.DataFrame
        Parcel universe. Must contain ``jurisdiction_col``, ``zoning_col``,
        ``land_area_col``, and ``bldg_area_col``. ``far_col`` is computed
        on the fly from ``bldg_area_col / land_area_col`` if missing.
    jurisdiction_col : str, default "planning_jurisdiction"
        Column identifying the planning jurisdiction (Wake's
        ``planning_jurisdiction`` is used as the canonical example: 17
        municipalities + unincorporated county).
    zoning_col : str, default "zoning"
        Column with the raw zoning code.
    land_area_col : str, default "land_area_sqft"
        Land area in square feet.
    bldg_area_col : str, default "bldg_area_finished_sqft"
        Heated/finished building area in square feet. Used to determine
        whether a parcel is "built" (>0).
    far_col : str, default "floor_area_ratio"
        Floor area ratio. If absent or all-NaN, computed as
        ``bldg_area_col / land_area_col``.
    min_built_parcels_per_pair : int, default 30
        Minimum number of built parcels for a ``(jurisdiction, zoning)``
        row to be considered self-supporting (``fallback_level=0``).
    min_built_parcels_per_jurisdiction : int, default 50
        Minimum number of built parcels for a jurisdiction-level aggregate
        row (``fallback_level=1``) to be considered self-supporting.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(jurisdiction, zoning)``. Columns:
        ``min_lot_sqft_p10``, ``min_lot_sqft_p25``, ``median_lot_sqft``,
        ``max_far_p90``, ``median_far``, ``n_parcels``, ``n_built_parcels``,
        ``fallback_level`` (0=self, 1=jurisdiction, 2=global). Rows for
        jurisdiction-aggregates use zoning value ``__JURISDICTION_AGG__``;
        the global aggregate uses jurisdiction ``__GLOBAL__`` and zoning
        ``__GLOBAL__``.
    """
    required = [jurisdiction_col, zoning_col, land_area_col, bldg_area_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_empirical_zoning_table: missing required columns: {missing}"
        )

    work = df[required].copy()
    if far_col in df.columns:
        work[far_col] = df[far_col]
    else:
        work[far_col] = np.nan
    # Compute FAR for any rows where it's missing
    needs_far = work[far_col].isna() & (work[land_area_col] > 0)
    work.loc[needs_far, far_col] = (
        work.loc[needs_far, bldg_area_col] / work.loc[needs_far, land_area_col]
    )

    # Strip rows that can't contribute
    work = work[work[land_area_col] > 0]
    work[jurisdiction_col] = work[jurisdiction_col].fillna("").astype(str).str.strip()
    work[zoning_col] = work[zoning_col].fillna("").astype(str).str.strip()
    work = work[(work[jurisdiction_col] != "") & (work[zoning_col] != "")]

    is_built = work[bldg_area_col].fillna(0) > 0
    work["__is_built"] = is_built

    if verbose:
        print(
            f"build_empirical_zoning_table: {len(work):,} usable parcels, "
            f"{is_built.sum():,} built; "
            f"{work[jurisdiction_col].nunique()} jurisdictions, "
            f"{work[[jurisdiction_col, zoning_col]].drop_duplicates().shape[0]} pairs"
        )

    # Per-pair rows
    pair_rows = _summarize(
        work,
        group_cols=[jurisdiction_col, zoning_col],
        land_area_col=land_area_col,
        far_col=far_col,
        is_built_mask=is_built,
        min_built=min_built_parcels_per_pair,
        fallback_level=0,
    )

    # Per-jurisdiction aggregates (across all zoning codes within a jurisdiction)
    juris_work = work.copy()
    juris_work["__zoning_agg__"] = "__JURISDICTION_AGG__"
    juris_rows = _summarize(
        juris_work,
        group_cols=[jurisdiction_col, "__zoning_agg__"],
        land_area_col=land_area_col,
        far_col=far_col,
        is_built_mask=juris_work["__is_built"],
        min_built=min_built_parcels_per_jurisdiction,
        fallback_level=1,
    )
    juris_rows = juris_rows.rename(columns={"__zoning_agg__": zoning_col})

    # Global aggregate (single row across all parcels)
    global_work = work.copy()
    global_work["__juris_agg__"] = _SENTINEL_JURISDICTION
    global_work["__zoning_agg__"] = "__GLOBAL__"
    global_rows = _summarize(
        global_work,
        group_cols=["__juris_agg__", "__zoning_agg__"],
        land_area_col=land_area_col,
        far_col=far_col,
        is_built_mask=global_work["__is_built"],
        min_built=1,  # always self-supporting
        fallback_level=2,
    )
    global_rows = global_rows.rename(
        columns={"__juris_agg__": jurisdiction_col, "__zoning_agg__": zoning_col}
    )

    out = pd.concat([pair_rows, juris_rows, global_rows], ignore_index=True)
    out = out.set_index([jurisdiction_col, zoning_col])

    if verbose:
        n_self = (out["fallback_level"] == 0).sum()
        n_juris = (out["fallback_level"] == 1).sum()
        print(
            f"build_empirical_zoning_table: {n_self} self-supporting pairs "
            f"(>= {min_built_parcels_per_pair} built), "
            f"{n_juris} jurisdiction aggregates, 1 global row"
        )

    return out


def _summarize(
    work: pd.DataFrame,
    *,
    group_cols: list,
    land_area_col: str,
    far_col: str,
    is_built_mask: pd.Series,
    min_built: int,
    fallback_level: int,
) -> pd.DataFrame:
    """Aggregate land-size and FAR statistics for a grouping."""
    g_all = work.groupby(group_cols, dropna=False)
    n_all = g_all.size().rename("n_parcels")

    built = work[is_built_mask]
    g_built = built.groupby(group_cols, dropna=False)
    n_built = g_built.size().rename("n_built_parcels")

    # Lot-size percentiles among built parcels (with fallback to all parcels
    # if a group has zero built — shouldn't happen often but guard anyway)
    if len(built) > 0:
        lot_p10 = g_built[land_area_col].quantile(0.10).rename("min_lot_sqft_p10")
        lot_p25 = g_built[land_area_col].quantile(0.25).rename("min_lot_sqft_p25")
        lot_med = g_built[land_area_col].median().rename("median_lot_sqft")
        far_p90 = g_built[far_col].quantile(0.90).rename("max_far_p90")
        far_med = g_built[far_col].median().rename("median_far")
    else:
        # Should not happen given how we call this, but be safe
        lot_p10 = pd.Series(dtype=float, name="min_lot_sqft_p10")
        lot_p25 = pd.Series(dtype=float, name="min_lot_sqft_p25")
        lot_med = pd.Series(dtype=float, name="median_lot_sqft")
        far_p90 = pd.Series(dtype=float, name="max_far_p90")
        far_med = pd.Series(dtype=float, name="median_far")

    out = pd.concat(
        [n_all, n_built, lot_p10, lot_p25, lot_med, far_p90, far_med], axis=1
    ).reset_index()
    out["n_built_parcels"] = out["n_built_parcels"].fillna(0).astype(int)
    out["fallback_level"] = fallback_level
    out["self_supporting"] = out["n_built_parcels"] >= min_built
    return out


def join_empirical_zoning(
    df: pd.DataFrame,
    zoning_table: pd.DataFrame,
    *,
    jurisdiction_col: str = "planning_jurisdiction",
    zoning_col: str = "zoning",
    out_prefix: str = "zoning_emp_",
) -> pd.DataFrame:
    """
    Add empirical-zoning columns to a parcel DataFrame using cascading lookup.

    For each row, look up the ``(jurisdiction, zoning)`` pair. If absent or
    not self-supporting, fall back to the jurisdiction aggregate. If that's
    also absent or not self-supporting, fall back to the global aggregate.

    Parameters
    ----------
    df : pandas.DataFrame
        Parcel DataFrame to enrich.
    zoning_table : pandas.DataFrame
        Output of :func:`build_empirical_zoning_table`, indexed by
        ``(jurisdiction, zoning)``.
    jurisdiction_col, zoning_col : str
        Column names in ``df``.
    out_prefix : str, default "zoning_emp_"
        Prefix for the new columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with added columns ``{prefix}min_lot_sqft``,
        ``{prefix}max_far``, ``{prefix}median_lot_sqft``,
        ``{prefix}median_far``, ``{prefix}fallback_level`` (0=pair-level,
        1=jurisdiction, 2=global, 3=no-match).
    """
    out = df.copy()
    juris = out[jurisdiction_col].fillna("").astype(str).str.strip()
    zone = out[zoning_col].fillna("").astype(str).str.strip()

    out_min = f"{out_prefix}min_lot_sqft"
    out_far = f"{out_prefix}max_far"
    out_lot_med = f"{out_prefix}median_lot_sqft"
    out_far_med = f"{out_prefix}median_far"
    out_fb = f"{out_prefix}fallback_level"

    out[out_min] = np.nan
    out[out_far] = np.nan
    out[out_lot_med] = np.nan
    out[out_far_med] = np.nan
    out[out_fb] = 3

    src_to_dst = [
        ("min_lot_sqft_p10", out_min),
        ("max_far_p90", out_far),
        ("median_lot_sqft", out_lot_med),
        ("median_far", out_far_med),
    ]

    # Pre-slice the table by fallback level, keep only self-supporting rows.
    def _level_table(level):
        sub = zoning_table[
            (zoning_table["fallback_level"] == level)
            & (zoning_table["self_supporting"])
        ]
        if sub.index.duplicated().any():
            sub = sub[~sub.index.duplicated(keep="first")]
        return sub

    pair_table = _level_table(0)
    juris_table = _level_table(1)
    global_idx = (_SENTINEL_JURISDICTION, "__GLOBAL__")
    global_row = (
        zoning_table.loc[global_idx]
        if global_idx in zoning_table.index
        else None
    )

    # Build a lookup key Series the same length as out
    pair_keys = pd.MultiIndex.from_arrays([juris, zone])
    juris_keys = pd.MultiIndex.from_arrays(
        [juris, pd.Series(["__JURISDICTION_AGG__"] * len(out), index=out.index)]
    )

    # Pair-level fill
    matched_pair = pair_table.reindex(pair_keys)
    has_pair = matched_pair[src_to_dst[0][0]].notna().values
    if has_pair.any():
        for src, dst in src_to_dst:
            out.loc[has_pair, dst] = matched_pair[src].values[has_pair]
        out.loc[has_pair, out_fb] = 0

    # Jurisdiction-level fill for what's still unfilled
    unfilled = out[out_fb].values == 3
    if unfilled.any():
        matched_juris = juris_table.reindex(juris_keys)
        has_juris = matched_juris[src_to_dst[0][0]].notna().values
        apply = unfilled & has_juris
        if apply.any():
            for src, dst in src_to_dst:
                out.loc[apply, dst] = matched_juris[src].values[apply]
            out.loc[apply, out_fb] = 1

    # Global fill for the rest
    unfilled = out[out_fb].values == 3
    if unfilled.any() and global_row is not None:
        for src, dst in src_to_dst:
            out.loc[unfilled, dst] = global_row[src]
        out.loc[unfilled, out_fb] = 2

    return out
