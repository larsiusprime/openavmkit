"""
Neighborhood-cascade infrastructure.

When a finest-grained neighborhood (e.g. an assessor VCS, a CAMA neighborhood
code, a census block) has too few sales to support a credible local estimate,
we want to fall back to a coarser geography that's still locally meaningful
— ideally one that respects real administrative or physical boundaries
rather than an arbitrary clustering.

This module provides the **cascade walker** infrastructure:

* :class:`HierarchySpec` — an ordered list of column names, finest first.
* :func:`build_neighborhood_hierarchy` — thin convenience wrapper that takes
  a list of pre-existing column names and returns a :class:`HierarchySpec`.
  Optionally appends a constant ``__county__`` fallback level.
* :func:`cascade_aggregate` — aggregate a witness-level value column at every
  cascade level.
* :func:`cascade_lookup` — for each parcel, walk the cascade and return the
  first level whose aggregate meets a minimum-n threshold.
* :func:`validate_hierarchy_consistency` — sanity-check helper to compare two
  columns that ought to agree (e.g. a derived prefix column against a
  canonical administrative column).

The columns that make up the cascade are **expected to already exist on the
DataFrame**. Where they come from is a locality concern:

* Existing assessor columns (``planning_jurisdiction``, ``township``,
  ``city``, ``census_tract``, ...) — declare them in ``data.load`` in your
  ``settings.json``.
* Substring-encoded prefixes of a neighborhood code (e.g. Wake's 7-char VCS
  ``[area:2][juris:2][sub:3]`` decomposes into ``vcs_area_juris`` etc.) —
  use the ``substr`` calc operator in ``data.load.<id>.calc``::

      "vcs_area_juris": ["substr", "neighborhood", {"left": 0, "right": 4}]

  See ``docs/docs/advanced_settings.md`` for the calc grammar.

This module deliberately does **not** invent geographic clusters via
K-means or similar. Real administrative subdivisions and substring-encoded
hierarchies already live in most assessor data; if they don't, a
queen-contiguity polygon merge respects real borders better than flat
Euclidean clustering.

See Also
--------
openavmkit.land.lycd : LYCD uniform-rate painter that walks the cascade.
openavmkit.land.evidence : Witness curation that pools across the cascade.
openavmkit.calculations.perform_calculations : Settings-driven calc engine
    that derives substring-prefix columns during data load.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HierarchySpec:
    """
    Ordered cascade specification.

    ``levels`` is finest-first: walking it climbs from the most local cell
    upward to broader fallbacks. Each entry is a column name that must be
    present on the DataFrame the cascade is applied to.

    Parameters
    ----------
    levels : list[str]
        Ordered cascade column names, finest first.
    notes : list[str]
        Free-form notes about the cascade (used in reports).
    """
    levels: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def build_neighborhood_hierarchy(
    df: pd.DataFrame,
    *,
    levels: list,
    add_county: bool = True,
    verbose: bool = False,
) -> tuple:
    """
    Validate a list of cascade column names and return a :class:`HierarchySpec`.

    All columns in ``levels`` must already exist on ``df`` — derive them
    upstream via ``data.load.<id>.calc`` (for substring prefixes) or by
    declaring them in ``data.load.<id>.load`` (for existing assessor
    columns). See the module docstring for an example.

    Parameters
    ----------
    df : pandas.DataFrame
        Parcel universe.
    levels : list of str
        Cascade column names, finest first. Missing columns are dropped
        from the cascade with a warning.
    add_county : bool, default True
        If True, append a constant ``"__county__"`` column as the last
        (broadest) level. Ensures every cell has a fallback. The column is
        added to ``df`` if not already present.
    verbose : bool, default False
        Print cascade summary.

    Returns
    -------
    enriched : pandas.DataFrame
        Copy of ``df``. Identical to the input unless ``add_county=True``
        added a ``__county__`` column.
    spec : HierarchySpec
        Cascade specification (finest first).
    """
    enriched = df.copy()

    valid_levels = []
    for col in levels:
        if col not in enriched.columns:
            warnings.warn(
                f"build_neighborhood_hierarchy: level '{col}' not in DataFrame, skipping"
            )
            continue
        if col in valid_levels:
            continue  # don't duplicate
        valid_levels.append(col)

    if add_county:
        if "__county__" not in enriched.columns:
            enriched["__county__"] = "ALL"
        if "__county__" not in valid_levels:
            valid_levels.append("__county__")

    notes = []
    for col in valid_levels:
        n = enriched[col].nunique(dropna=False)
        notes.append(f"{col}: {n} distinct values")
    if verbose:
        print("Cascade (finest first):")
        for note in notes:
            print(f"  {note}")

    return enriched, HierarchySpec(levels=valid_levels, notes=notes)


def validate_hierarchy_consistency(
    df: pd.DataFrame,
    derived_col: str,
    canonical_col: str,
    *,
    name: str = "",
    threshold: float = 0.95,
) -> dict:
    """
    Compare two columns expected to agree, and return a small report.

    Useful sanity check when a derived (e.g. substring-decomposed) column
    should match a pre-existing canonical column. For example, Wake's VCS
    chars 2-4 (``vcs_juris``) should match ``planning_jurisdiction`` for
    98%+ of parcels; rows where they disagree expose data-quality issues
    worth investigating.

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
        Cascade specification.
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
        values, with the count vector attached via ``.attrs["n"]``.
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
        ``None`` if no level matched — should not happen if a county-level
        constant is included).
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
        if n_series is None:
            keys = df[level]
            looked = keys.map(agg_series)
            ok = looked.notna() & unfilled
        else:
            usable_keys = n_series[n_series >= min_n].index
            agg_subset = agg_series.loc[agg_series.index.isin(usable_keys)]
            keys = df[level]
            looked = keys.map(agg_subset)
            ok = looked.notna() & unfilled

        values.loc[ok] = looked.loc[ok].values
        levels_used.loc[ok] = level
        unfilled.loc[ok] = False

    return values, levels_used
