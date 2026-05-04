"""
Neighborhood-hierarchy derivation and cascading lookups.

When a finest-grained neighborhood (e.g. an assessor VCS, a CAMA neighborhood
code, a census block) has too few sales to support a credible local estimate,
we want to fall back to a coarser geography that's still locally meaningful —
ideally one that respects real administrative or physical boundaries rather
than an arbitrary clustering.

This module provides two things:

* :func:`probe_neighborhood_naming` — heuristically detects whether a
  neighborhood column's codes encode a hierarchy in their *naming*
  (e.g. Wake County's ``15RA093`` decomposes into
  ``[township][jurisdiction][sub-area]``). Returns a description.
* :func:`build_neighborhood_hierarchy` — given a DataFrame and a list of
  natural levels (column names + optional name-prefix splits), returns the
  DataFrame enriched with derived hierarchy columns plus an ordered cascade
  list that downstream code can walk.

The cascade is **ordered finest-first**: the first level is the most local,
each subsequent level a coarser fallback. Downstream painters (Rung 1+)
walk this list per-cell when the local witness count is below threshold.

This module deliberately does **not** invent geographic clusters via K-means
or similar. Real administrative subdivisions and substring-encoded hierarchies
already live in most assessor data; if they don't, a queen-contiguity polygon
merge respects real borders better than flat Euclidean clustering.

See Also
--------
openavmkit.land.lycd : LYCD uniform-rate painter that walks the cascade.
openavmkit.land.evidence : Witness curation that pools across the cascade.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class HierarchySpec:
    """
    Ordered cascade specification.

    ``levels`` is finest-first: walking it climbs from the most local cell
    upward to broader fallbacks. Each entry is a column name in the enriched
    DataFrame.

    Parameters
    ----------
    levels : list[str]
        Ordered cascade column names, finest first.
    derived : dict[str, dict]
        Description of any derived columns, keyed by their name. Each value
        is a small dict describing the derivation (e.g.
        ``{"source": "neighborhood", "kind": "prefix", "chars": "0:4"}``).
    notes : list[str]
        Free-form notes about the cascade (used in reports).
    """
    levels: list = field(default_factory=list)
    derived: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def probe_neighborhood_naming(
    df: pd.DataFrame,
    neighborhood_col: str = "neighborhood",
    sample_size: int = 5000,
) -> dict:
    """
    Detect whether a neighborhood column's codes share a common length and
    appear to encode positional substructure.

    A common-length finding is a *necessary* signal that prefix splits are
    meaningful — variable-length codes are unlikely to encode a positional
    hierarchy. The function reports the modal length and the % of values
    that share it. It does not attempt to label the substring meanings;
    that's locality-specific and is supplied by the caller via
    :func:`build_neighborhood_hierarchy`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input.
    neighborhood_col : str, default "neighborhood"
        Column to probe.
    sample_size : int, default 5000
        Sample size for the probe (full pass for diversity stats but only
        the sample is shown in the report).

    Returns
    -------
    dict
        Keys: ``n_codes`` (distinct), ``modal_length``, ``modal_length_pct``,
        ``length_distribution`` (Series), ``samples`` (list of strings).
    """
    s = df[neighborhood_col].dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return {
            "n_codes": 0,
            "modal_length": None,
            "modal_length_pct": 0.0,
            "length_distribution": pd.Series(dtype=int),
            "samples": [],
        }
    lengths = s.str.len()
    dist = lengths.value_counts().sort_index()
    modal_len = int(dist.idxmax())
    modal_pct = float((lengths == modal_len).mean())
    rng = np.random.default_rng(0)
    uniq = sorted(s.unique().tolist())
    n_samp = min(sample_size, len(uniq))
    samples = list(rng.choice(uniq, size=n_samp, replace=False))
    return {
        "n_codes": int(s.nunique()),
        "modal_length": modal_len,
        "modal_length_pct": modal_pct,
        "length_distribution": dist,
        "samples": samples[:20],
    }


def derive_prefix_columns(
    df: pd.DataFrame,
    neighborhood_col: str,
    splits: list,
    out_col_prefix: str = "vcs_",
) -> pd.DataFrame:
    """
    Add derived prefix-substring columns from a fixed-format neighborhood code.

    Parameters
    ----------
    df : pandas.DataFrame
    neighborhood_col : str
        Source column.
    splits : list of dict
        Each dict has keys:

        * ``name`` — output column name (without prefix)
        * ``start`` — 0-based start index (inclusive)
        * ``end`` — 0-based end index (exclusive); use ``None`` for "to end"
    out_col_prefix : str, default "vcs_"
        Prefix prepended to each split's ``name``.

    Returns
    -------
    pandas.DataFrame
        Copy with new columns. Empty/NaN source values produce NaN outputs.
    """
    out = df.copy()
    src = out[neighborhood_col].astype("string")
    for spec in splits:
        name = f"{out_col_prefix}{spec['name']}"
        start = spec.get("start", 0)
        end = spec.get("end", None)
        if end is None:
            piece = src.str.slice(start)
        else:
            piece = src.str.slice(start, end)
        # Treat empty strings as NaN
        piece = piece.where(piece.str.len().fillna(0) > 0)
        out[name] = piece
    return out


def build_neighborhood_hierarchy(
    df: pd.DataFrame,
    *,
    neighborhood_col: str = "neighborhood",
    splits: list | None = None,
    extra_levels: list | None = None,
    add_county: bool = True,
    verbose: bool = False,
) -> tuple:
    """
    Enrich a DataFrame with hierarchy columns and return a cascade spec.

    Parameters
    ----------
    df : pandas.DataFrame
    neighborhood_col : str, default "neighborhood"
        Finest-grained neighborhood column. This becomes cascade level 0.
    splits : list of dict, optional
        Prefix splits to derive. If omitted, no prefix derivation runs.
        Each split: ``{"name": str, "start": int, "end": int|None}``. The
        derived columns become cascade levels (in order) right after the
        finest neighborhood.
    extra_levels : list of str, optional
        Additional column names already on ``df`` to append to the cascade
        in order (typical: ``["planning_jurisdiction", "township"]``).
    add_county : bool, default True
        If True, append a constant ``"__county__"`` column as the last
        (broadest) level. Ensures every cell has a fallback.
    verbose : bool, default False
        Print cascade summary.

    Returns
    -------
    enriched : pandas.DataFrame
        Copy of ``df`` with derived columns added.
    spec : HierarchySpec
        Cascade specification (finest first).
    """
    splits = splits or []
    extra_levels = extra_levels or []
    enriched = derive_prefix_columns(df, neighborhood_col, splits) if splits else df.copy()

    levels = [neighborhood_col]
    derived_meta = {}
    for spec in splits:
        col = f"vcs_{spec['name']}"
        levels.append(col)
        derived_meta[col] = {
            "source": neighborhood_col,
            "kind": "prefix",
            "start": spec.get("start", 0),
            "end": spec.get("end", None),
        }
    for col in extra_levels:
        if col not in enriched.columns:
            warnings.warn(
                f"build_neighborhood_hierarchy: extra level '{col}' not in DataFrame, skipping"
            )
            continue
        if col in levels:
            continue  # don't duplicate
        levels.append(col)
    if add_county:
        enriched["__county__"] = "ALL"
        levels.append("__county__")

    # Validation: log granularity at each level.
    notes = []
    for col in levels:
        n = enriched[col].nunique(dropna=False)
        notes.append(f"{col}: {n} distinct values")
    if verbose:
        print("Cascade (finest first):")
        for note in notes:
            print(f"  {note}")

    return enriched, HierarchySpec(levels=levels, derived=derived_meta, notes=notes)


def validate_hierarchy_consistency(
    df: pd.DataFrame,
    derived_col: str,
    canonical_col: str,
    *,
    name: str = "",
    threshold: float = 0.95,
) -> dict:
    """
    Compare a derived hierarchy column to a canonical column for consistency.

    Useful sanity check when a prefix-derived column should match a
    pre-existing column (e.g. Wake's VCS chars 3-4 should match
    ``planning_jurisdiction``). Returns a small report.

    Parameters
    ----------
    df : pandas.DataFrame
    derived_col : str
        The derived column (e.g. ``"vcs_juris"``).
    canonical_col : str
        The canonical column (e.g. ``"planning_jurisdiction"``).
    name : str
        Human-readable label for the comparison.
    threshold : float, default 0.95
        Match-rate threshold below which a warning is issued.

    Returns
    -------
    dict
        ``{"name", "match_rate", "n_compared", "top_mismatches"}``.
    """
    pair_present = df[derived_col].notna() & df[canonical_col].notna()
    sub = df[pair_present]
    if len(sub) == 0:
        return {
            "name": name,
            "match_rate": None,
            "n_compared": 0,
            "top_mismatches": pd.Series(dtype=int),
        }
    # Compare via numeric values when both sides parse as numeric, else string.
    # This avoids the "1.0" vs "1" mismatch when one side is float and the
    # other int after NaN-coerced casting.
    left_num = pd.to_numeric(sub[derived_col], errors="coerce")
    right_num = pd.to_numeric(sub[canonical_col], errors="coerce")
    both_numeric = left_num.notna() & right_num.notna()
    matches = pd.Series(False, index=sub.index)
    matches.loc[both_numeric] = left_num[both_numeric] == right_num[both_numeric]
    # For non-numeric rows, fall back to string comparison
    non_num = ~both_numeric
    matches.loc[non_num] = (
        sub.loc[non_num, derived_col].astype(str)
        == sub.loc[non_num, canonical_col].astype(str)
    )
    rate = float(matches.mean())
    if rate < threshold:
        warnings.warn(
            f"{name}: derived '{derived_col}' matches canonical '{canonical_col}' "
            f"at only {rate*100:.2f}% (threshold {threshold*100:.0f}%)."
        )
    mism = sub[~matches]
    top = (
        mism.groupby([derived_col, canonical_col]).size().sort_values(ascending=False).head(10)
        if len(mism) > 0
        else pd.Series(dtype=int)
    )
    return {
        "name": name,
        "match_rate": rate,
        "n_compared": int(len(sub)),
        "top_mismatches": top,
    }


def cascade_aggregate(
    base_df: pd.DataFrame,
    *,
    spec: HierarchySpec,
    value_col: str,
    weight_col: str | None = None,
    agg: str = "median",
) -> dict:
    """
    Aggregate a value column at every cascade level.

    Returns a dict mapping cascade level name to a Series indexed by that
    level's distinct values. Used by Rung-1 painters that need, for each
    level, the median (or other aggregate) of a witness-derived rate.

    Parameters
    ----------
    base_df : pandas.DataFrame
        Witness-level data — one row per piece of evidence (e.g. one row
        per vacant sale).
    spec : HierarchySpec
        Output of :func:`build_neighborhood_hierarchy`.
    value_col : str
        Column to aggregate.
    weight_col : str, optional
        If given, compute weighted aggregate; otherwise unweighted.
    agg : str, default "median"
        One of ``"median"``, ``"mean"``, ``"weighted_mean"``.

    Returns
    -------
    dict[str, pandas.Series]
        Keyed by level column name; each Series indexed by that level's
        values, with the aggregated value plus an ``n`` attribute via
        ``.attrs``.
    """
    out = {}
    for level in spec.levels:
        if level not in base_df.columns:
            continue
        g = base_df.groupby(level, dropna=False)
        if agg == "median":
            agg_series = g[value_col].median()
        elif agg == "mean":
            agg_series = g[value_col].mean()
        elif agg == "weighted_mean":
            if weight_col is None:
                raise ValueError("weighted_mean requires weight_col")
            num = (base_df[value_col] * base_df[weight_col]).groupby(base_df[level]).sum()
            den = base_df[weight_col].groupby(base_df[level]).sum()
            agg_series = num / den
        else:
            raise ValueError(f"Unknown agg: {agg}")
        n_series = g.size().rename("n")
        agg_series.attrs["n"] = n_series
        out[level] = agg_series
    return out


def cascade_lookup(
    df: pd.DataFrame,
    *,
    spec: HierarchySpec,
    aggregates: dict,
    min_n: int = 30,
) -> tuple:
    """
    For each row in ``df``, walk the cascade and return the first-level
    aggregate that meets the minimum-n threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Target rows to enrich (one row per parcel).
    spec : HierarchySpec
        Cascade ladder (finest first).
    aggregates : dict[str, pandas.Series]
        Output of :func:`cascade_aggregate`. Each series should have an
        ``n`` attribute set on ``.attrs``.
    min_n : int, default 30
        Minimum count required to use a level. Levels failing the threshold
        are skipped; the next coarser level is tried.

    Returns
    -------
    values : pandas.Series
        Length matches ``df``; aggregated value chosen at the first level
        that met ``min_n`` for that row.
    levels_used : pandas.Series
        Length matches ``df``; the level column name actually used (or
        ``None`` if no level matched — should not happen if county-level
        is included).
    """
    values = pd.Series(np.nan, index=df.index, dtype=float)
    levels_used = pd.Series([None] * len(df), index=df.index, dtype=object)
    unfilled = pd.Series(True, index=df.index)

    for level in spec.levels:
        if not unfilled.any():
            break
        if level not in df.columns or level not in aggregates:
            continue
        agg_series = aggregates[level]
        n_series = agg_series.attrs.get("n", None)
        # Build a key->(value, n) lookup
        if n_series is None:
            # Treat as unbounded n — use everywhere available
            keys = df[level]
            looked = keys.map(agg_series)
            ok = looked.notna() & unfilled
        else:
            # Restrict to keys with sufficient n
            usable_keys = n_series[n_series >= min_n].index
            agg_subset = agg_series.loc[agg_series.index.isin(usable_keys)]
            keys = df[level]
            looked = keys.map(agg_subset)
            ok = looked.notna() & unfilled

        values.loc[ok] = looked.loc[ok].values
        levels_used.loc[ok] = level
        unfilled.loc[ok] = False

    return values, levels_used


