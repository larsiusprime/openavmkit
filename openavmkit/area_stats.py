"""
Area-statistic ("neighborhood enrichment") feature generation.

Computes per-location summary statistics and stamps them onto every parcel as new
``area_stat_<location>_<field>_<stat>`` features (for example
``area_stat_neighborhood_bldg_area_finished_sqft_mean``). This is a quantized,
group-based counterpart to spatial lag (:func:`openavmkit.data.enrich_sup_spatial_lag`):
instead of a smooth k-nearest-neighbor surface, it summarizes discrete location groups
(neighborhood, census tract, ...) at one or more granularities.

Two rules keep the features honest:

- **Leakage:** sale-derived fields (the sale price and its variants) are aggregated over
  the *training* set of valid sales only, so test-set prices never enter a feature.
  Characteristic fields (building area, lot size, quality, zoning, ...) are aggregated
  over the full universe, since those are known at prediction time. An optional
  ``exclude_test_keys`` flag drops test-key parcels from *all* aggregation for shops that
  want strict out-of-sample hygiene.
- **Small samples:** a ``min_count`` floor blanks a stat to ``NaN`` when its group has too
  few observations, with *no* fallback. When locations are configured as a hierarchy
  (coarsest → finest), the coarser levels are simply separate ``area_stat_*`` columns the
  model can lean on where a finer one is missing.

The companion :func:`report_area_stats` ranks the generated features by their correlation
with sale price and optionally writes a Markdown report.
"""
import warnings

import numpy as np
import pandas as pd

from openavmkit.data import (
    SalesUniversePair,
    get_hydrated_sales_from_sup,
    get_sale_field,
    get_train_test_keys,
)
from openavmkit.reports import MarkdownReport, finish_report
from openavmkit.utilities.data import div_series_z_safe
from openavmkit.utilities.settings import (
    AREA_STAT_PREFIX,
    AREA_STATS_CATEGORICAL_DEFAULT,
    AREA_STATS_NUMERIC_DEFAULT,
    area_unit,
    expand_area_stats_fields,
    get_area_stats_config,
    get_fields_categorical,
    get_fields_numeric,
    get_valuation_date,
    is_sale_derived_field,
    make_area_stat_count_field_name,
    make_area_stat_field_name,
)
from openavmkit.utilities.stats import calc_correlations


def enrich_sup_area_stats(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """Enrich sales and universe with per-location area-statistic features.

    Reads the ``data.process.enrich.area_stats`` configuration and, for each configured
    ``location × field × stat`` combination, computes the statistic within each location
    group and stamps it onto every parcel. A per-location ``count`` (group size) column is
    always emitted. If the feature is not configured, ``sup`` is returned unchanged.

    Parameters
    ----------
    sup : SalesUniversePair
        SalesUniversePair containing sales and universe DataFrames.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    SalesUniversePair
        Enriched SalesUniversePair with new ``area_stat_*`` columns.
    """
    cfg = get_area_stats_config(settings)
    if not cfg:
        if verbose:
            print("area_stats: no configuration found; skipping.")
        return sup

    locations = cfg.get("locations", []) or []
    # Bare sale-price fields auto-expand into the full per-area family (level +
    # improved $/bldg-sqft + vacant/improved $/land-sqft), for both raw and time-adjusted.
    explicit_fields = set(cfg.get("fields", []) or [])
    fields = expand_area_stats_fields(settings, cfg.get("fields", []) or [])
    num_stats = cfg.get("stats", AREA_STATS_NUMERIC_DEFAULT) or []
    cat_stats = cfg.get("categorical_stats", AREA_STATS_CATEGORICAL_DEFAULT) or []
    min_count = int(cfg.get("min_count", 0) or 0)
    exclude_test_keys = bool(cfg.get("exclude_test_keys", False))

    df_sales = sup.sales.copy()
    df_universe = sup.universe.copy()

    # Split configured fields into sale-derived (train-only) and characteristic (universe).
    sale_fields = [f for f in fields if is_sale_derived_field(settings, f)]

    # Resolve base-field kinds once (numeric vs categorical) to pick the stat family.
    num_set = set(get_fields_numeric(settings, include_boolean=True))
    cat_set = set(get_fields_categorical(settings))
    unit = area_unit(settings)

    # Source frames -----------------------------------------------------------------
    # The sales frame is always built: sale-derived stats use it AND the per-location
    # sales counts (total / improved / vacant) are derived from it. It is restricted to
    # training valid sales so nothing here leaks the target.
    df_univ_src = df_universe
    df_hydrated = get_hydrated_sales_from_sup(sup)
    try:
        train_keys, test_keys = get_train_test_keys(df_hydrated, settings)
    except KeyError:
        # No model_group column / canonical splits available (e.g. run before the split
        # stage, or minimal frames): we can't build a leakage-safe sales source, so
        # sale-derived stats and sales counts are skipped rather than risk leakage.
        if sale_fields:
            warnings.warn(
                "area_stats: no train/test split available (missing 'model_group' or "
                "canonical splits); sale-derived fields and sales counts will be empty. "
                "Run after write_canonical_splits."
            )
        train_keys, test_keys = np.array([], dtype=str), np.array([], dtype=str)
    train_keys = set(np.asarray(train_keys).astype(str))
    test_keys = set(np.asarray(test_keys).astype(str))

    sale_mask = df_hydrated["key_sale"].astype(str).isin(train_keys)
    if "valid_sale" in df_hydrated.columns:
        sale_mask &= df_hydrated["valid_sale"].eq(True)
    df_sale_src = df_hydrated.loc[sale_mask].copy()
    if sale_fields:
        # Synthesize area-unit-normalized sale fields (e.g. $/finished-sqft) on the
        # train-only frame so they stay leakage-guarded, mirroring spatial lag.
        df_sale_src = _synthesize_sale_unit_fields(df_sale_src, sale_fields, unit)

    if exclude_test_keys:
        test_parcels = set(
            df_hydrated.loc[
                df_hydrated["key_sale"].astype(str).isin(test_keys), "key"
            ].astype(str)
        )
        df_univ_src = df_universe[~df_universe["key"].astype(str).isin(test_parcels)]

    # Compute and stamp --------------------------------------------------------------
    new_cols: list[str] = []

    for location in locations:
        if location not in df_universe.columns:
            warnings.warn(
                f"area_stats: location '{location}' not found in universe; skipping."
            )
            continue

        # Per-location counts (always emitted, never masked by min_count):
        #   count                -> universe parcels in the area
        #   sales_count          -> training valid sales in the area
        #   sales_count_improved -> of those, improved sales
        #   sales_count_vacant   -> of those, vacant sales
        parcels_name = make_area_stat_count_field_name(location, "count")
        df_universe[parcels_name] = df_universe[location].map(
            df_univ_src.groupby(location).size()
        )
        new_cols.append(parcels_name)

        total, improved, vacant = _sales_counts_by_group(df_sale_src, location, unit)
        for kind, series in (
            ("sales_count", total),
            ("sales_count_improved", improved),
            ("sales_count_vacant", vacant),
        ):
            cname = make_area_stat_count_field_name(location, kind)
            df_universe[cname] = df_universe[location].map(series).fillna(0)
            new_cols.append(cname)

        for field in fields:
            is_sale = field in sale_fields
            src = df_sale_src if is_sale else df_univ_src
            if src is None or field not in src.columns or location not in src.columns:
                # Only warn for fields the user listed explicitly; auto-expanded
                # sale-rate variants that don't apply here are skipped silently.
                if field in explicit_fields:
                    warnings.warn(
                        f"area_stats: field '{field}' unavailable for location "
                        f"'{location}'; skipping."
                    )
                continue

            kind = _base_field_kind(settings, field, src[field], num_set, cat_set)
            stats_list = num_stats if kind == "numeric" else cat_stats

            # Non-null observations per group drive the min_count floor.
            obs_count = src.groupby(location)[field].count()
            small_groups = (
                obs_count.index[obs_count < min_count] if min_count > 0 else None
            )

            for stat in stats_list:
                colname = make_area_stat_field_name(location, field, stat)
                series = _aggregate(src, location, field, stat, kind)
                if small_groups is not None and len(small_groups) > 0:
                    series = series.copy()
                    series.loc[series.index.isin(small_groups)] = np.nan
                df_universe[colname] = df_universe[location].map(series)
                new_cols.append(colname)

    # Propagate the universe-level features onto sales by parcel key (matches the
    # universe -> sales pattern used by spatial lag).
    for col in new_cols:
        df_sales = _fill_col_from_universe(df_sales, df_universe, col)

    if verbose:
        print(
            f"area_stats: added {len(new_cols)} column(s) across "
            f"{len(locations)} location(s)."
        )

    return SalesUniversePair(df_sales, df_universe)


def report_area_stats(
    sup: SalesUniversePair,
    settings: dict,
    outpath: str = None,
    threshold: float = 0.1,
    do_plots: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Rank area-stat features by their correlation with sale price.

    Computes the correlation of every numeric ``area_stat_*`` column with the sale price
    (over valid sales), returning a DataFrame ranked by correlation strength. When
    ``outpath`` is provided, also writes a Markdown report (and PDF/HTML per
    ``analysis.report.formats``).

    Parameters
    ----------
    sup : SalesUniversePair
        SalesUniversePair already enriched via :func:`enrich_sup_area_stats`.
    settings : dict
        Settings dictionary.
    outpath : str, optional
        Output path (without extension) for the Markdown report. If None, no file is
        written and only the ranked DataFrame is returned.
    threshold : float, optional
        Correlation score threshold passed to :func:`calc_correlations`. Defaults to 0.1.
    do_plots : bool, optional
        If True, render correlation heatmaps. Defaults to False.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    pandas.DataFrame
        Columns ``variable``, ``corr_strength``, ``corr_clarity``, ``corr_score``, sorted
        by ``corr_strength`` descending.
    """
    empty = pd.DataFrame(
        columns=["variable", "corr_strength", "corr_clarity", "corr_score"]
    )

    df = get_hydrated_sales_from_sup(sup)
    if "valid_sale" in df.columns:
        df = df[df["valid_sale"].eq(True)]

    sale_field = get_sale_field(settings, df)
    # get_sale_field returns the time-adjusted field by default even if it isn't
    # present; fall back to raw sale_price so the report still works pre-time-adjustment.
    if sale_field not in df.columns and "sale_price" in df.columns:
        sale_field = "sale_price"
    if sale_field not in df.columns:
        warnings.warn(
            f"area_stats report: sale field '{sale_field}' not found; skipping report."
        )
        return empty

    area_cols = [
        c
        for c in df.columns
        if c.startswith(AREA_STAT_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not area_cols:
        warnings.warn(
            "area_stats report: no numeric area_stat columns found; "
            "run enrich_sup_area_stats first."
        )
        return empty

    x_corr = df[[sale_field] + area_cols].copy()
    corr = calc_correlations(x_corr, threshold=threshold, do_plots=do_plots)

    ranked = corr["initial"].copy()
    ranked = ranked[ranked["variable"] != sale_field]
    ranked = ranked.sort_values(
        "corr_strength", ascending=False, na_position="last"
    ).reset_index(drop=True)

    if verbose:
        print(f"area_stats report: ranked {len(ranked)} feature(s) vs '{sale_field}'.")

    if outpath is not None:
        _write_area_stats_report(ranked, settings, outpath, sale_field)

    return ranked


#######################################
# PRIVATE
#######################################


def _synthesize_sale_unit_fields(
    df: pd.DataFrame, requested_fields: list, unit: str
) -> pd.DataFrame:
    """Create area-unit-normalized sale rates (e.g. ``$/finished-sqft``) where missing.

    Per-unit sale fields are not persisted columns — they're synthesized here on the
    (already train-only, valid-sale) frame by dividing the base sale value by an area, with
    a sample mask that makes the *which sales* explicit. The recognized suffixes are:

    - ``_impr_<unit>``        — $/finished-building-area, improved sales (``bldg_area > 0``;
      vacant sales have no building area and are excluded).
    - ``_vacant_land_<unit>`` — $/land-area, **vacant** sales only. A vacant sale's price is
      pure land, so this is a clean land-value rate.
    - ``_impr_land_<unit>``   — $/land-area, **improved** sales only. A developed-density rate
      (total price over land), distinct from the vacant land rate.
    - ``_land_<unit>``        — alias of ``_vacant_land_<unit>`` (kept so spatial-lag-style
      names don't silently fail; spatial lag's ``_land`` rate is likewise vacant-sampled).

    Vacant vs improved is decided by :func:`_vacant_sale_mask`. Rows failing a mask are left
    NaN so they're excluded from aggregation rather than skewing it. The more specific
    ``_*_land_<unit>`` suffixes are matched before ``_land_<unit>``/``_impr_<unit>``.
    """
    # Ordered most-specific suffix first (both `_vacant_land_<unit>` and `_impr_land_<unit>`
    # also end in `_land_<unit>`). sample: "vacant", "improved", or "any".
    specs = [
        (f"_vacant_land_{unit}", f"land_area_{unit}", "vacant"),
        (f"_impr_land_{unit}", f"land_area_{unit}", "improved"),
        (f"_impr_{unit}", f"bldg_area_finished_{unit}", "any"),
        (f"_land_{unit}", f"land_area_{unit}", "vacant"),  # alias of _vacant_land_<unit>
    ]
    vac_mask = _vacant_sale_mask(df, unit)
    for field in requested_fields:
        if field in df.columns:
            continue
        for suffix, area_col, sample in specs:
            if not field.endswith(suffix):
                continue
            base = field[: -len(suffix)]
            if base in df.columns and area_col in df.columns:
                mask = df[area_col] > 0
                if sample in ("vacant", "improved"):
                    if vac_mask is None:
                        warnings.warn(
                            f"area_stats: cannot split vacant vs improved sales (no "
                            f"'vacant_sale' or building-area column); '{field}' is computed "
                            f"from all {area_col}>0 sales."
                        )
                    elif sample == "vacant":
                        mask &= vac_mask
                    else:  # improved
                        mask &= ~vac_mask
                df[field] = div_series_z_safe(df[base], df[area_col]).where(mask)
            break
    return df


def _vacant_sale_mask(df: pd.DataFrame, unit: str):
    """Boolean mask of vacant sales, or None if vacancy can't be determined.

    Prefers the explicit ``vacant_sale`` flag (what spatial lag uses); falls back to
    "no finished building area" if that column is absent.
    """
    if "vacant_sale" in df.columns:
        return df["vacant_sale"].eq(True)
    area_col = f"bldg_area_finished_{unit}"
    if area_col in df.columns:
        return ~(df[area_col].fillna(0) > 0)
    return None


def _sales_counts_by_group(df_sales: pd.DataFrame, location: str, unit: str):
    """Return (total, improved, vacant) training-sale counts per location group.

    Each is a Series indexed by location value (empty where unavailable). Improved vs
    vacant split uses :func:`_vacant_sale_mask`; if vacancy can't be determined, all
    sales are reported as improved (with a warning) and vacant is left empty.
    """
    empty = pd.Series(dtype="float64")
    if df_sales is None or len(df_sales) == 0 or location not in df_sales.columns:
        return empty, empty, empty
    total = df_sales.groupby(location).size()
    vac_mask = _vacant_sale_mask(df_sales, unit)
    if vac_mask is None:
        warnings.warn(
            "area_stats: cannot determine vacant vs improved sales (no 'vacant_sale' "
            "or building-area column); reporting all sales as improved."
        )
        return total, total, empty
    vacant = df_sales.loc[vac_mask].groupby(location).size()
    improved = df_sales.loc[~vac_mask].groupby(location).size()
    return total, improved, vacant


def _base_field_kind(
    settings: dict,
    field: str,
    series: pd.Series,
    num_set: set,
    cat_set: set,
) -> str:
    """Decide whether a base field should be summarized with numeric or categorical stats."""
    if is_sale_derived_field(settings, field):
        return "numeric"
    if field in cat_set:
        return "categorical"
    if field in num_set:
        return "numeric"
    # Unclassified: infer from the data.
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def _aggregate(
    src: pd.DataFrame, location: str, field: str, stat: str, kind: str
) -> pd.Series:
    """Compute one statistic of ``field`` grouped by ``location``, indexed by group value."""
    g = src.groupby(location)[field]
    if kind == "numeric":
        if stat == "mean":
            return g.mean()
        if stat == "median":
            return g.median()
        if stat == "std":
            return g.std()
        if stat == "min":
            return g.min()
        if stat == "max":
            return g.max()
        if stat == "sum":
            return g.sum()
        if stat == "count":
            return g.count()
        if stat == "cv":
            mean = g.mean()
            return g.std() / mean.replace(0, np.nan)
        if stat == "p25":
            return g.quantile(0.25)
        if stat == "p75":
            return g.quantile(0.75)
        if stat == "iqr":
            return g.quantile(0.75) - g.quantile(0.25)
        raise ValueError(f"Unknown numeric area-stat: '{stat}'")

    # categorical base field
    if stat == "mode":
        return g.agg(_mode)
    if stat == "mode_frac":
        return g.agg(_mode_frac)
    if stat == "nunique":
        return g.nunique(dropna=True)
    if stat == "entropy":
        return g.agg(_entropy)
    raise ValueError(f"Unknown categorical area-stat: '{stat}'")


def _mode(s: pd.Series):
    """Most frequent non-null value (ties broken by value_counts order); NaN if empty."""
    vc = s.dropna().value_counts()
    return vc.index[0] if len(vc) else np.nan


def _mode_frac(s: pd.Series) -> float:
    """Share of the dominant category among non-null values; NaN if empty."""
    s = s.dropna()
    n = len(s)
    if n == 0:
        return np.nan
    return float(s.value_counts().iloc[0]) / n


def _entropy(s: pd.Series) -> float:
    """Shannon entropy (natural log) of the category distribution; NaN if empty."""
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    p = s.value_counts(normalize=True)
    return float(-(p * np.log(p)).sum())


def _fill_col_from_universe(
    df_sales: pd.DataFrame, df_universe: pd.DataFrame, col: str
) -> pd.DataFrame:
    """Copy ``col`` from universe onto sales by parcel ``key`` (one parcel -> many sales)."""
    merged = df_sales.merge(
        df_universe[["key", col]], on="key", how="left", suffixes=("", "___u___")
    )
    if f"{col}___u___" in merged.columns:
        merged[col] = merged[col].fillna(merged[f"{col}___u___"])
        merged = merged.drop(columns=f"{col}___u___")
    return merged


def _write_area_stats_report(
    ranked: pd.DataFrame, settings: dict, outpath: str, sale_field: str
) -> None:
    """Render the area-stats Markdown report from the ranked correlation table."""
    report = MarkdownReport("area_stats")
    report.set_var("locality", settings.get("locality", {}).get("name"))
    try:
        report.set_var("val_date", get_valuation_date(settings).strftime("%Y-%m-%d"))
    except (ValueError, AttributeError):
        report.set_var("val_date", "")
    report.set_var("sale_field", sale_field)
    report.set_var("num_features", len(ranked))
    report.set_var("correlation_table", _ranked_to_markdown(ranked))
    finish_report(report, outpath, "default", settings)


def _ranked_to_markdown(ranked: pd.DataFrame) -> str:
    """Format the ranked correlation DataFrame as a Markdown table."""
    if ranked.empty:
        return "_No area-stat features to report._"

    def fmt(value) -> str:
        return "—" if pd.isna(value) else f"{value:.3f}"

    lines = [
        "| Rank | Feature | Corr. strength | Corr. clarity | Corr. score |",
        "|------|---------|----------------|---------------|-------------|",
    ]
    for i, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.variable} | {fmt(row.corr_strength)} | "
            f"{fmt(row.corr_clarity)} | {fmt(row.corr_score)} |"
        )
    return "\n".join(lines)
