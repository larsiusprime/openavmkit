"""
Lars-Tests for land and improvement valuations.

Eleven tests that score a candidate per-parcel land-value column on
incentive-and-equity criteria. The framing comes from the
``Valuing Land: The Simplest Viable Method`` essay and the
``How Assessors Value Land Right Now`` interview. These are deliberately
*not* IAAO ratio studies in the standard sense — they test whether a
valuation generates the economic incentives a Land Value Tax is supposed
to produce, *as well as* whether the decomposition agrees with ground
truth (vacant + improved sales) and is geographically smooth.

The suite has three layers:

* **L1-L6 — Internal-consistency (land side).** Properties the
  decomposition has to satisfy for the LVT incentives to flow through.
  Pure functions of the candidate column + universe.
* **L7 — Foundation soundness (improvement side).** Tests whether the
  cost approach is applied uniformly across geography; depends only on
  ``assr_impr_value`` and is identical across candidates that share the
  same underlying assessment data.
* **L8-L11 — Ground truth + smoothness.** Match the decomposition
  against vacant-sale evidence (L8), improved-sale reconciliation (L10),
  Moran's I on residuals (L11), and check that adjacent parcels across
  cell boundaries don't show artificial $/sqft jumps (L9).

The eleven tests:

* **L1 - Improvement-neutrality.** For matched pairs of nearby parcels
  with similar size and zoning where one is improved and one is not, the
  candidate land values should be the same. Score = % of pairs within
  tolerance ±ε. *Improvements should not move land value.*
* **L2 - Side-by-side uniformity.** Within the existing land-equity
  clusters, the COD of ``$/sqft`` should be low. *Similarly-situated
  parcels should pay similar $/sqft.*
* **L3 - Vacant-burden flip.** Under a revenue-neutral LVT computed from
  these land values, the median vacant parcel's tax bill should go up
  relative to the current property tax. *An LVT actually shifts burden
  onto the underutilized parcels it claims to.*
* **L4 - Desirability tracking.** Spearman correlation of neighborhood-
  aggregate land $/sqft with assessor and predicted market $/sqft.
  *Location desirability flows through to land value.*
* **L5 - Density-FAR ordering.** Within neighborhoods with multiple
  zoning classes, the higher-``zoning_emp_max_far`` zone should have
  higher median land $/sqft. Random baseline = 50%. *Zoning that allows
  more buildable floor area should command more $/sqft.*
* **L6 - Per-cell size decay.** Large-lot parcels should be ≥10%
  cheaper per sqft than the same neighborhood's typical-lot median.
  *Extra land beyond the typical lot is worth less per sqft.*
* **L7 - Improvement-cost-table consistency.** Cluster parcels by
  building characteristics ignoring location, compute COD of
  ``assr_impr_value / bldg_area_finished_sqft`` within clusters. *Same
  building + same depreciation should yield the same improvement value
  regardless of location.*
* **L8 - Held-out vacant-sale prediction accuracy.** Re-paint the
  universe holding out a fold of W1 vacant sales, then check painter vs.
  the held-out sale prices. *Catches systematic level-bias in the land
  decomposition that the internal tests can't see.*
* **L9 - Boundary discontinuity.** For pairs of nearby parcels (within
  ~200 m) that sit in different finest-cascade cells, score the share
  whose $/sqft differs by less than 25%. Compared to within-cell pairs
  as a control. *Cells are a modeling artifact; cross-cell pairs should
  be roughly as smooth as within-cell pairs.*
* **L10 - Improved-sale reconciliation.** For improved sales, the median
  ratio of (painted_land + assr_impr) / sale_price should be ~1.0 with
  low COD. *Bridges to ground truth on the improved side: the sum of the
  decomposition has to match real sales.*
* **L11 - Residual spatial autocorrelation (Moran's I).** Compute
  Moran's I of percent residuals (sale - painted_total) / sale across
  improved sales using k-nearest spatial weights. *I close to 0 means
  the painter has absorbed location signal into the cells; I > 0 means
  whole neighborhoods are systematically over- or under-painted, which
  is an equity concern even when overall accuracy looks fine.*

Higher is better for L1, L3, L4, L5, L6, L9. Lower is better for L2, L7,
L11 (Moran's I is two-tailed; closer to 0 is better). For L8 and L10 the
median ratio target is ~1.0 with low COD.

Sample-size caveats:

* L5's pair count is limited by how many neighborhoods have multiple
  zoning classes.
* L6 only considers cells with at least 10 typical-sized parcels.
* L7 requires improvement characteristics; each band is added only if
  the underlying column is populated and not too-tied-to-bin.
* L8 is computed by :func:`run_holdout_vacant_test` (which needs a
  painter callback) and passed into :func:`run_lars_tests` via
  ``holdout_result``. Not run unconditionally because it's the only
  test that requires re-painting the universe.
* L9, L11 require parcel ``latitude`` / ``longitude`` (centroids).
* L10, L11 require an ``improved_sales`` DataFrame with a sale-price
  column (typically ``sale_price_time_adj``).

The harness runs all available tests and writes a single Markdown report
comparing multiple candidates side-by-side.

See Also
--------
openavmkit.zoning : Provides L5's ``zoning_emp_max_far`` and L6's
    ``zoning_emp_min_lot_sqft``.
openavmkit.land.evidence : Provides the witness pool used by the painter
    whose output is being scored, and the W1 vacant-sale fold for L8.
openavmkit.horizontal_equity_study : Provides L2's clusters.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openavmkit.utilities.stats import calc_cod


@dataclass
class LarsTestsResult:
    """Aggregate result from running the Lars-Tests on one candidate.

    L1-L6 are land-side (depend on the candidate column). L7 is improvement-side
    and is identical across candidates that share the same ``assr_impr_value`` —
    it's a property of the underlying assessment, not the painter.
    """
    candidate: str
    L1_neutrality_pct: float | None = None
    L1_n_pairs: int = 0
    L2_uniformity_cod: float | None = None
    L2_n_clusters: int = 0
    L3_vacant_burden_flip_pct: float | None = None
    L3_n_vacant_parcels: int = 0
    L4_spearman_assr: float | None = None
    L4_spearman_pred: float | None = None
    L4_n_neighborhoods: int = 0
    L5_density_pair_pct: float | None = None
    L5_n_pairs: int = 0
    L6_size_decay_pct: float | None = None
    L6_n_large_lots: int = 0
    L7_impr_uniformity_cod: float | None = None
    L7_n_clusters: int = 0
    # L8 — held-out vacant-sale prediction accuracy. Populated externally by
    # `run_holdout_vacant_test` (requires a painter callback) and stuffed into
    # the result for unified reporting.
    L8_holdout_vacant_median_ratio: float | None = None
    L8_holdout_vacant_cod: float | None = None
    L8_n_holdout: int = 0
    # L9 — boundary discontinuity. Cross-cell pairs of nearby parcels.
    L9_cross_cell_smooth_pct: float | None = None
    L9_within_cell_smooth_pct: float | None = None
    L9_n_cross_pairs: int = 0
    # L10 — improved-sale reconciliation: median painted_total / sale_price
    # plus COD around the median.
    L10_recon_median_ratio: float | None = None
    L10_recon_cod: float | None = None
    L10_n_improved_sales: int = 0
    # L11 — Moran's I on (sale_price − painted_land − assr_impr) for improved
    # sales. Lower magnitude = less spatial structure left in residuals.
    L11_residual_morans_i: float | None = None
    L11_morans_p_value: float | None = None
    L11_n_residuals: int = 0
    # L12 — Depreciation localization. Within (neighborhood, size, quality)
    # clusters, building $/sqft should fall with age (L12a strongly negative)
    # but candidate land $/sqft should NOT (L12b near zero). The score is the
    # differential — depreciation signal localizes in buildings, doesn't bleed
    # into land.
    L12a_bldg_age_spearman: float | None = None
    L12b_land_age_spearman_abs: float | None = None
    L12_localization_score: float | None = None
    L12_n_clusters: int = 0
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def run_lars_tests(
    universe: pd.DataFrame,
    *,
    candidate_col: str,
    candidate_name: str | None = None,
    sales_with_far: pd.DataFrame | None = None,
    parcel_key: str = "key",
    land_area_col: str = "land_area_sqft",
    bldg_area_col: str = "bldg_area_finished_sqft",
    neighborhood_col: str = "neighborhood",
    zoning_col: str = "zoning",
    assr_market_col: str = "assr_market_value",
    assr_land_col: str = "assr_land_value",
    assr_impr_col: str = "assr_impr_value",
    pred_market_col: str | None = "prediction",
    cluster_col: str = "land_he_id",
    zoning_emp_min_col: str = "zoning_emp_min_lot_sqft",
    zoning_emp_max_far_col: str = "zoning_emp_max_far",
    bldg_year_built_col: str = "bldg_year_built",
    bldg_age_col: str = "bldg_age_years",
    bldg_quality_col: str = "bldg_quality_num",
    bldg_condition_col: str = "bldg_condition_num",
    improved_sales: pd.DataFrame | None = None,
    sale_price_col: str = "sale_price_time_adj",
    sales_join_key: str = "key",
    parcel_lat_col: str = "latitude",
    parcel_lon_col: str = "longitude",
    holdout_result: dict | None = None,
    moran_k: int = 8,
    moran_n_permutations: int = 99,
    eps_pct: float = 0.10,
    verbose: bool = False,
) -> LarsTestsResult:
    """
    Score a candidate land-value column on the seven Lars-Tests.

    L1-L6 evaluate the candidate land-value column. L7 evaluates the
    underlying improvement-cost-table consistency and is identical across
    candidates that share the same ``assr_impr_value``.

    Parameters
    ----------
    universe : pandas.DataFrame
        Full parcel universe with the candidate column populated.
    candidate_col : str
        The land-value column to score.
    candidate_name : str, optional
        Friendly name for reports. Defaults to ``candidate_col``.
    sales_with_far : pandas.DataFrame, optional
        Sales DataFrame with ``floor_area_ratio`` for L4's market-side
        Spearman. If absent, L4 falls back to whatever's in ``universe``.
    parcel_key, land_area_col, bldg_area_col : str
    neighborhood_col, zoning_col : str
    assr_market_col, assr_land_col, pred_market_col, cluster_col, zoning_emp_min_col : str
    eps_pct : float, default 0.10
        L1 tolerance: a pair is "neutral" if predicted land values agree
        within ±ε × pair-mean.
    verbose : bool

    Returns
    -------
    LarsTestsResult
    """
    candidate_name = candidate_name or candidate_col
    res = LarsTestsResult(candidate=candidate_name)

    if candidate_col not in universe.columns:
        res.notes.append(f"candidate column '{candidate_col}' missing from universe")
        return res

    valid_mask = (
        universe[candidate_col].notna()
        & (universe[candidate_col] >= 0)
        & (universe[land_area_col].fillna(0) > 0)
    )
    work = universe[valid_mask].copy()
    work["__cand_per_sqft__"] = work[candidate_col] / work[land_area_col]

    # ---- L1: Improvement-neutrality ----
    res.L1_neutrality_pct, res.L1_n_pairs = _l1_improvement_neutrality(
        work,
        candidate_col=candidate_col,
        neighborhood_col=neighborhood_col,
        zoning_col=zoning_col,
        bldg_area_col=bldg_area_col,
        land_area_col=land_area_col,
        eps_pct=eps_pct,
    )

    # ---- L2: Side-by-side uniformity ----
    res.L2_uniformity_cod, res.L2_n_clusters = _l2_uniformity(
        work,
        cluster_col=cluster_col,
        per_sqft_col="__cand_per_sqft__",
    )

    # ---- L3: Vacant-burden flip ----
    res.L3_vacant_burden_flip_pct, res.L3_n_vacant_parcels = _l3_vacant_burden_flip(
        work,
        candidate_col=candidate_col,
        bldg_area_col=bldg_area_col,
        assr_market_col=assr_market_col,
    )

    # ---- L4: Desirability tracking ----
    res.L4_spearman_assr, res.L4_spearman_pred, res.L4_n_neighborhoods = _l4_desirability(
        work,
        per_sqft_col="__cand_per_sqft__",
        assr_market_col=assr_market_col,
        pred_market_col=pred_market_col,
        land_area_col=land_area_col,
        neighborhood_col=neighborhood_col,
    )

    # ---- L5: Density-aware land valuation ----
    res.L5_density_pair_pct, res.L5_n_pairs = _l5_density_pair(
        work,
        per_sqft_col="__cand_per_sqft__",
        neighborhood_col=neighborhood_col,
        zoning_col=zoning_col,
        zoning_emp_max_far_col=zoning_emp_max_far_col,
    )

    # ---- L6: Per-cell size decay ----
    res.L6_size_decay_pct, res.L6_n_large_lots = _l6_excess_surplus(
        work,
        per_sqft_col="__cand_per_sqft__",
        neighborhood_col=neighborhood_col,
        zoning_emp_min_col=zoning_emp_min_col,
        land_area_col=land_area_col,
    )

    # ---- L7: Improvement-cost-table consistency ----
    # Independent of candidate_col — depends only on assr_impr_value and the
    # building characteristics. Reported per-candidate row for convenience;
    # the value is identical across candidates that share the same
    # underlying assessment data.
    res.L7_impr_uniformity_cod, res.L7_n_clusters = _l7_improvement_uniformity(
        universe,
        bldg_area_col=bldg_area_col,
        assr_impr_col=assr_impr_col,
        bldg_year_built_col=bldg_year_built_col,
        bldg_quality_col=bldg_quality_col,
        bldg_condition_col=bldg_condition_col,
    )

    # ---- L8: Held-out vacant-sale prediction accuracy ----
    # Computed by `run_holdout_vacant_test` (which needs a painter callback)
    # and passed in via ``holdout_result``. We just stuff it into the result
    # for unified reporting.
    if holdout_result is not None:
        res.L8_holdout_vacant_median_ratio = holdout_result.get("median_ratio")
        res.L8_holdout_vacant_cod = holdout_result.get("cod")
        res.L8_n_holdout = int(holdout_result.get("n", 0))

    # ---- L9: Boundary discontinuity ----
    if parcel_lat_col in universe.columns and parcel_lon_col in universe.columns:
        (
            res.L9_cross_cell_smooth_pct,
            res.L9_within_cell_smooth_pct,
            res.L9_n_cross_pairs,
        ) = _l9_boundary_discontinuity(
            work,
            per_sqft_col="__cand_per_sqft__",
            neighborhood_col=neighborhood_col,
            lat_col=parcel_lat_col,
            lon_col=parcel_lon_col,
        )

    # ---- L10 / L11: improved-sale reconciliation + spatial residuals ----
    if improved_sales is not None and len(improved_sales) > 0:
        (
            res.L10_recon_median_ratio,
            res.L10_recon_cod,
            res.L10_n_improved_sales,
        ) = _l10_improved_sale_reconciliation(
            universe,
            improved_sales,
            candidate_col=candidate_col,
            assr_impr_col=assr_impr_col,
            sale_price_col=sale_price_col,
            sales_join_key=sales_join_key,
            parcel_key=parcel_key,
        )
        if parcel_lat_col in universe.columns and parcel_lon_col in universe.columns:
            (
                res.L11_residual_morans_i,
                res.L11_morans_p_value,
                res.L11_n_residuals,
            ) = _l11_residual_morans_i(
                universe,
                improved_sales,
                candidate_col=candidate_col,
                assr_impr_col=assr_impr_col,
                sale_price_col=sale_price_col,
                sales_join_key=sales_join_key,
                parcel_key=parcel_key,
                lat_col=parcel_lat_col,
                lon_col=parcel_lon_col,
                k=moran_k,
                n_permutations=moran_n_permutations,
            )

    # ---- L12: Depreciation localization ----
    # "Buildings depreciate, land doesn't." Within (neighborhood, size,
    # quality) clusters, implied building $/sqft should fall with age (L12a
    # strongly negative) but candidate land $/sqft should not (L12b near
    # zero). Score = |L12a| - L12b.
    (
        res.L12a_bldg_age_spearman,
        res.L12b_land_age_spearman_abs,
        res.L12_localization_score,
        res.L12_n_clusters,
    ) = _l12_depreciation_localization(
        universe,
        candidate_col=candidate_col,
        neighborhood_col=neighborhood_col,
        bldg_area_col=bldg_area_col,
        land_area_col=land_area_col,
        bldg_age_col=bldg_age_col,
        bldg_quality_col=bldg_quality_col,
        assr_market_col=assr_market_col,
    )

    if verbose:
        _print_summary(res)

    return res


# ------------------------------------------------------------------ L1

def _l1_improvement_neutrality(
    work: pd.DataFrame,
    *,
    candidate_col: str,
    neighborhood_col: str,
    zoning_col: str,
    bldg_area_col: str,
    land_area_col: str,
    eps_pct: float,
    max_pairs: int = 20000,
    size_band_pct: float = 0.20,
):
    """
    Within each (neighborhood, zoning, size-band) cell, sample matched pairs
    of (one improved, one vacant) and check whether their candidate land
    values agree within ±eps_pct.
    """
    df = work.copy()
    is_built = df[bldg_area_col].fillna(0) > 0
    df["__is_built__"] = is_built
    # Size band: log-quartiles within neighborhood for stability
    df["__size_band__"] = df.groupby(neighborhood_col, dropna=False)[land_area_col] \
        .transform(lambda s: pd.qcut(s, q=4, labels=False, duplicates="drop"))

    rng = np.random.default_rng(42)
    matches = 0
    n_pairs = 0
    grouped = df.groupby([neighborhood_col, zoning_col, "__size_band__"], dropna=False)
    for _, sub in grouped:
        if len(sub) < 2:
            continue
        built = sub[sub["__is_built__"]]
        vacant = sub[~sub["__is_built__"]]
        if len(built) == 0 or len(vacant) == 0:
            continue
        # Sample up to 5 pairs per cell (cap to keep runtime bounded)
        n_take = min(5, len(built), len(vacant))
        bi = rng.choice(built.index, size=n_take, replace=False)
        vi = rng.choice(vacant.index, size=n_take, replace=False)
        for b, v in zip(bi, vi):
            lb = built.loc[b, candidate_col]
            lv = vacant.loc[v, candidate_col]
            if pd.isna(lb) or pd.isna(lv):
                continue
            mean = (lb + lv) / 2.0
            if mean <= 0:
                continue
            if abs(lb - lv) <= eps_pct * mean:
                matches += 1
            n_pairs += 1
            if n_pairs >= max_pairs:
                break
        if n_pairs >= max_pairs:
            break

    if n_pairs == 0:
        return None, 0
    return matches / n_pairs, n_pairs


# ------------------------------------------------------------------ L2

def _l2_uniformity(
    work: pd.DataFrame,
    *,
    cluster_col: str,
    per_sqft_col: str,
    min_per_cluster: int = 5,
):
    """
    Compute the median per-cluster COD of $/sqft. Lower = more uniform.
    """
    if cluster_col not in work.columns:
        return None, 0
    cods = []
    for cid, sub in work.groupby(cluster_col, dropna=True):
        if len(sub) < min_per_cluster:
            continue
        vals = sub[per_sqft_col].dropna().values
        if len(vals) < min_per_cluster:
            continue
        try:
            cod = calc_cod(vals)
        except Exception:
            continue
        if cod is None or not np.isfinite(cod):
            continue
        cods.append(cod)
    if not cods:
        return None, 0
    return float(np.median(cods)), len(cods)


# ------------------------------------------------------------------ L3

def _l3_vacant_burden_flip(
    work: pd.DataFrame,
    *,
    candidate_col: str,
    bldg_area_col: str,
    assr_market_col: str,
):
    """
    Compute revenue-neutral LVT vs. property-tax burden change for vacant
    parcels. Returns % of vacant parcels whose tax goes UP.

    Formula (revenue-neutral LVT calibrated against current tax):
        rate_pt = total_tax_revenue / sum(market_value)
        rate_lvt = total_tax_revenue / sum(land_value)
    For a parcel: tax_pt = rate_pt * market_value
                  tax_lvt = rate_lvt * land_value
    Vacant-burden flip ratio: tax_lvt > tax_pt for the vacant parcels.
    """
    is_vacant = work[bldg_area_col].fillna(0) <= 0
    n_vac = int(is_vacant.sum())
    if n_vac == 0:
        return None, 0
    if assr_market_col not in work.columns:
        return None, 0

    total_market = work[assr_market_col].sum()
    total_land = work[candidate_col].sum()
    if total_market <= 0 or total_land <= 0:
        return None, 0

    # Set total tax = 1.0 (units cancel; only the per-parcel comparison matters)
    rate_pt = 1.0 / total_market
    rate_lvt = 1.0 / total_land

    sub = work[is_vacant]
    tax_pt = rate_pt * sub[assr_market_col].fillna(0)
    tax_lvt = rate_lvt * sub[candidate_col].fillna(0)
    flipped = (tax_lvt > tax_pt).sum()
    return float(flipped) / n_vac, n_vac


# ------------------------------------------------------------------ L4

def _l4_desirability(
    work: pd.DataFrame,
    *,
    per_sqft_col: str,
    assr_market_col: str,
    pred_market_col: str | None,
    land_area_col: str,
    neighborhood_col: str,
    min_per_neighborhood: int = 30,
):
    """
    Aggregate to neighborhood medians; Spearman-correlate land $/sqft
    against assessor market $/sqft and our predicted market $/sqft.
    """
    if assr_market_col not in work.columns:
        return None, None, 0
    df = work.copy()
    df["__assr_per_sqft__"] = df[assr_market_col] / df[land_area_col]
    if pred_market_col and pred_market_col in df.columns:
        df["__pred_per_sqft__"] = df[pred_market_col] / df[land_area_col]
    grp = df.groupby(neighborhood_col, dropna=False)
    counts = grp.size()
    qualifying = counts[counts >= min_per_neighborhood].index
    if len(qualifying) == 0:
        return None, None, 0
    df = df[df[neighborhood_col].isin(qualifying)]
    nbhd = df.groupby(neighborhood_col, dropna=False).median(numeric_only=True)
    if len(nbhd) < 3:
        return None, None, 0

    def _spear(x, y):
        try:
            return float(pd.Series(x).rank().corr(pd.Series(y).rank()))
        except Exception:
            return None

    s_assr = _spear(nbhd[per_sqft_col], nbhd["__assr_per_sqft__"])
    s_pred = (
        _spear(nbhd[per_sqft_col], nbhd["__pred_per_sqft__"])
        if "__pred_per_sqft__" in nbhd.columns
        else None
    )
    return s_assr, s_pred, len(nbhd)


# ------------------------------------------------------------------ L5

def _l5_density_pair(
    work: pd.DataFrame,
    *,
    per_sqft_col: str,
    neighborhood_col: str,
    zoning_col: str,
    zoning_emp_max_far_col: str,
    min_parcels_per_zoning: int = 10,
):
    """
    Density-aware land valuation. Tests whether the candidate respects the
    economic principle that — within a neighborhood — land in a higher-density
    zone (greater allowed FAR / buildable floor area per sqft of land) should
    command a higher $/sqft than land in a lower-density zone, since you can
    build more on it.

    For each neighborhood with >=2 zoning classes (each with at least
    ``min_parcels_per_zoning`` parcels), enumerate all pairs of zoning classes.
    For each pair, compare the median land $/sqft of the higher-FAR class
    against the lower-FAR class. A correct ordering scores 1; ties are
    skipped. Score = matches / n_pairs.

    Random baseline = 50%. A density-aware painter should score 65-80%+.

    Caveat: in jurisdictions with mostly zoning-homogeneous neighborhoods
    (Wake, much of suburban USA), the qualifying-neighborhood pool will be
    small, and high-FAR zones can correlate with non-residential or otherwise
    different uses that confound the comparison. Read this score directionally,
    not as a hard threshold.
    """
    if zoning_emp_max_far_col not in work.columns or zoning_col not in work.columns:
        return None, 0

    df = work[
        work[zoning_emp_max_far_col].notna()
        & (work[zoning_emp_max_far_col] > 0)
        & work[zoning_col].notna()
    ]
    if len(df) == 0:
        return None, 0

    # Per (nbhd, zoning) median psf + FAR + count. Within a (juris, zoning)
    # pair the empirical FAR is constant, so any aggregation works for FAR.
    agg = df.groupby([neighborhood_col, zoning_col], dropna=False).agg(
        median_psf=(per_sqft_col, "median"),
        max_far=(zoning_emp_max_far_col, "median"),
        n=(per_sqft_col, "size"),
    ).reset_index()
    agg = agg[agg["n"] >= min_parcels_per_zoning]
    if len(agg) == 0:
        return None, 0

    matches = 0
    n_pairs = 0
    for _, sub in agg.groupby(neighborhood_col, dropna=False):
        if len(sub) < 2:
            continue
        rows = sub[["max_far", "median_psf"]].values
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                far_i, psf_i = rows[i]
                far_j, psf_j = rows[j]
                if not np.isfinite(far_i) or not np.isfinite(far_j):
                    continue
                if far_i == far_j:
                    continue
                higher_psf = psf_i if far_i > far_j else psf_j
                lower_psf = psf_j if far_i > far_j else psf_i
                if not (np.isfinite(higher_psf) and np.isfinite(lower_psf)):
                    continue
                if higher_psf > lower_psf:
                    matches += 1
                n_pairs += 1

    if n_pairs == 0:
        return None, 0
    return matches / n_pairs, n_pairs


# ------------------------------------------------------------------ L6

def _l6_excess_surplus(
    work: pd.DataFrame,
    *,
    per_sqft_col: str,
    neighborhood_col: str,
    zoning_emp_min_col: str,
    land_area_col: str,
    min_typical_per_cell: int = 10,
):
    """
    Per-cell size decay test. For each large-lot parcel
    (``size_ratio = land_area / zoning_emp_min >= 2.0``), check whether its
    $/sqft is at least 10% below the *typical-lot* median $/sqft in the same
    neighborhood (typical = ``size_ratio in [0.9, 1.5]``, with at least
    ``min_typical_per_cell`` typical parcels for the cell to provide a usable
    baseline).

    Score = fraction of large-lot parcels where the in-cell norm_psf is < 0.9.

    Caveat: this version doesn't distinguish excess (separately developable)
    from surplus (just-more-grass-to-mow); it tests whether the painter
    exhibits *some* per-cell size-decay, which is the basic diminishing-returns
    signal called out by the user's PDFs and Wake's SOV. A future refinement
    would split on whether geometry + zoning permits lot subdivision.

    Note: an earlier version pooled the typical-lot median globally across all
    cells, which conflated location effects (rural lots are cheaper than
    urban) with the size-decay signal we actually want to measure. This
    per-cell version isolates size-decay cleanly.
    """
    if zoning_emp_min_col not in work.columns or neighborhood_col not in work.columns:
        return None, 0

    df = work[work[zoning_emp_min_col] > 0].copy()
    df["__size_ratio__"] = df[land_area_col] / df[zoning_emp_min_col]

    typical_mask = df["__size_ratio__"].between(0.9, 1.5)
    typical = df[typical_mask].groupby(neighborhood_col, dropna=False).agg(
        typical_median=(per_sqft_col, "median"),
        n=(per_sqft_col, "size"),
    )
    typical = typical[typical["n"] >= min_typical_per_cell]
    if len(typical) == 0:
        return None, 0

    large = df[df["__size_ratio__"] >= 2.0].merge(
        typical[["typical_median"]],
        left_on=neighborhood_col,
        right_index=True,
        how="inner",
    )
    n_large = len(large)
    if n_large == 0:
        return None, 0

    norm_psf = large[per_sqft_col] / large["typical_median"].replace(0, np.nan)
    correct = int((norm_psf < 0.9).sum())
    return correct / n_large, n_large


# ------------------------------------------------------------------ L7

def _l7_improvement_uniformity(
    universe: pd.DataFrame,
    *,
    bldg_area_col: str,
    assr_impr_col: str,
    bldg_year_built_col: str,
    bldg_quality_col: str,
    bldg_condition_col: str,
    valuation_year: int = 2024,
    min_per_cluster: int = 5,
    size_quantiles: int = 10,
    age_quantiles: int = 10,
    quality_quantiles: int = 5,
    condition_quantiles: int = 5,
):
    """
    Improvement-cost-table consistency. Mirror image of L1 / L2: clusters
    parcels by improvement characteristics *ignoring location* (size band, age
    band, quality band, condition band), then computes the COD of
    ``assr_impr_value / bldg_area_finished_sqft`` within each cluster.

    Score = median per-cluster COD. Lower = improvements applied more
    consistently across geography, consistent with a frozen cost-table approach
    (same building + same depreciation -> same value, no matter where it sits).

    The test depends only on the assessor's improvement values and the building
    characteristics; it does NOT depend on the candidate land-value column.
    L7 will therefore be identical across candidates that share the same
    underlying assessment data. It's reported in each candidate row for
    convenience.

    Optional bands fall through gracefully — if a band column is missing or
    too-tied-to-bin (e.g. condition all 1.0), L7 still runs on whatever bands
    are available.
    """
    if bldg_area_col not in universe.columns or assr_impr_col not in universe.columns:
        return None, 0

    df = universe[
        (universe[bldg_area_col].fillna(0) > 0)
        & (universe[assr_impr_col].fillna(0) > 0)
    ].copy()
    if len(df) == 0:
        return None, 0

    df["__impr_psf__"] = df[assr_impr_col] / df[bldg_area_col]

    band_cols: list = []

    def _qband(col: str, q: int, name: str) -> bool:
        if col not in df.columns:
            return False
        try:
            df[name] = pd.qcut(df[col], q=q, labels=False, duplicates="drop")
            band_cols.append(name)
            return True
        except (ValueError, TypeError):
            return False

    _qband(bldg_area_col, size_quantiles, "__size_band__")
    if bldg_year_built_col in df.columns:
        age = valuation_year - pd.to_numeric(df[bldg_year_built_col], errors="coerce")
        df["__age_raw__"] = age
        _qband("__age_raw__", age_quantiles, "__age_band__")
    _qband(bldg_quality_col, quality_quantiles, "__quality_band__")
    _qband(bldg_condition_col, condition_quantiles, "__condition_band__")

    if not band_cols:
        return None, 0

    cods = []
    for _, sub in df.groupby(band_cols, dropna=True):
        if len(sub) < min_per_cluster:
            continue
        vals = sub["__impr_psf__"].dropna().values
        if len(vals) < min_per_cluster:
            continue
        try:
            cod = calc_cod(vals)
        except Exception:
            continue
        if cod is None or not np.isfinite(cod):
            continue
        cods.append(cod)

    if not cods:
        return None, 0
    return float(np.median(cods)), len(cods)


# ------------------------------------------------------------------ L9

def _l9_boundary_discontinuity(
    work: pd.DataFrame,
    *,
    per_sqft_col: str,
    neighborhood_col: str,
    lat_col: str,
    lon_col: str,
    distance_threshold_meters: float = 200.0,
    smooth_jump_threshold: float = 0.25,
    max_neighbors: int = 5,
):
    """
    Boundary discontinuity test. For each parcel, find its nearest neighbors
    within ``distance_threshold_meters``; classify each (parcel, neighbor) pair
    by whether the two parcels share a finest-cascade cell. For each pair,
    compute the relative $/sqft jump:

        jump = |psf_i - psf_j| / mean(psf_i, psf_j)

    A "smooth" pair is one where the jump is below ``smooth_jump_threshold``
    (default 25%). Score = (cross-cell smooth %, within-cell smooth %, n).

    Interpretation: cross-cell pairs across an arbitrary cell boundary should
    be roughly as smooth as within-cell pairs. Big gap (cross-cell smooth %
    much lower than within-cell) means the painter is creating discontinuities
    at modeling-artifact boundaries.

    Implementation: KDTree on lat/lon, converted to meters via small-angle
    approximation at the dataset's mean latitude. Distance accuracy is fine
    for the 200 m scale.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, None, 0

    df = work.dropna(subset=[lat_col, lon_col, per_sqft_col, neighborhood_col]).copy()
    df = df[df[per_sqft_col] > 0]
    if len(df) < 2:
        return None, None, 0

    # Convert lat/lon to local meters: 1 deg lat ~ 111,320 m; 1 deg lon ~ 111,320 * cos(lat) m.
    mean_lat = df[lat_col].mean()
    deg_to_m_lat = 111_320.0
    deg_to_m_lon = 111_320.0 * float(np.cos(np.deg2rad(mean_lat)))
    coords = np.column_stack([
        df[lat_col].values * deg_to_m_lat,
        df[lon_col].values * deg_to_m_lon,
    ])

    tree = cKDTree(coords)
    # query_ball_tree style is heavy; use k-nearest with k = max_neighbors+1 (excl self),
    # then filter by distance.
    dists, idxs = tree.query(coords, k=max_neighbors + 1, distance_upper_bound=distance_threshold_meters)

    nbhd = df[neighborhood_col].astype(str).values
    psf = df[per_sqft_col].values

    cross_smooth = 0
    cross_total = 0
    within_smooth = 0
    within_total = 0

    n = len(df)
    for i in range(n):
        for k in range(1, max_neighbors + 1):
            j = idxs[i, k]
            d = dists[i, k]
            if j >= n or not np.isfinite(d):
                continue
            if i >= j:
                continue  # only count each unordered pair once
            psf_i, psf_j = psf[i], psf[j]
            mean_psf = 0.5 * (psf_i + psf_j)
            if mean_psf <= 0:
                continue
            jump = abs(psf_i - psf_j) / mean_psf
            same_cell = nbhd[i] == nbhd[j]
            if same_cell:
                within_total += 1
                if jump < smooth_jump_threshold:
                    within_smooth += 1
            else:
                cross_total += 1
                if jump < smooth_jump_threshold:
                    cross_smooth += 1

    cross_pct = cross_smooth / cross_total if cross_total else None
    within_pct = within_smooth / within_total if within_total else None
    return cross_pct, within_pct, cross_total


# ------------------------------------------------------------------ L10

def _l10_improved_sale_reconciliation(
    universe: pd.DataFrame,
    improved_sales: pd.DataFrame,
    *,
    candidate_col: str,
    assr_impr_col: str,
    sale_price_col: str,
    sales_join_key: str,
    parcel_key: str,
):
    """
    Improved-sale reconciliation: for qualified improved sales, compute the
    median ratio of ``(painted_land + assr_impr) / sale_price`` plus its COD.

    Tests whether the decomposition's *sum* matches actual market evidence.
    L1-L7 test internal consistency; this one bridges to ground truth on the
    improved side. (L8 does the same on the vacant side via held-out
    cross-validation.)

    Returns (median_ratio, cod, n).
    """
    needed_universe = [parcel_key, candidate_col, assr_impr_col]
    needed_sales = [sales_join_key, sale_price_col]
    for c in needed_universe:
        if c not in universe.columns:
            return None, None, 0
    for c in needed_sales:
        if c not in improved_sales.columns:
            return None, None, 0

    sales = improved_sales[needed_sales].copy()
    sales = sales[sales[sale_price_col].fillna(0) > 0]
    if len(sales) == 0:
        return None, None, 0

    u = universe[needed_universe].drop_duplicates(parcel_key)
    merged = sales.merge(u, left_on=sales_join_key, right_on=parcel_key, how="inner")
    if len(merged) == 0:
        return None, None, 0

    painted_total = merged[candidate_col].fillna(0) + merged[assr_impr_col].fillna(0)
    ratio = (painted_total / merged[sale_price_col]).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    if len(ratio) == 0:
        return None, None, 0

    median = float(ratio.median())
    try:
        cod = float(calc_cod(ratio.values))
    except Exception:
        cod = None
    return median, cod, int(len(ratio))


# ------------------------------------------------------------------ L11

def _l11_residual_morans_i(
    universe: pd.DataFrame,
    improved_sales: pd.DataFrame,
    *,
    candidate_col: str,
    assr_impr_col: str,
    sale_price_col: str,
    sales_join_key: str,
    parcel_key: str,
    lat_col: str,
    lon_col: str,
    k: int = 8,
    n_permutations: int = 99,
    rng_seed: int = 42,
):
    """
    Moran's I on percent residuals of improved-sale reconciliation.

    For each improved sale we compute::

        residual_pct = (sale_price - painted_land - assr_impr) / sale_price

    Then build a row-normalized k-nearest spatial weights matrix on parcel
    centroids (lat/lon -> approximate meters) and compute Moran's I plus a
    permutation p-value (default 99 permutations).

    Interpretation:

    * I close to 0: residuals are spatially uncorrelated; the painter has
      absorbed the location signal into the cells.
    * I substantially > 0: spatial structure remains in residuals — entire
      neighborhoods are systematically over- or under-painted. Indicates
      missed location signal that the painter could capture with finer
      cells or a different cascade.
    * I < 0 (rare): checkerboard pattern, usually a sign of overfitting to
      a noisy variable.

    Attribution: by L7's logic improvements are location-invariant, so
    spatial structure in total-value residuals is most likely a *land-side*
    failure to capture local variation.

    Returns (morans_i, p_value, n).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, None, 0

    needed_universe = [parcel_key, candidate_col, assr_impr_col, lat_col, lon_col]
    needed_sales = [sales_join_key, sale_price_col]
    for c in needed_universe:
        if c not in universe.columns:
            return None, None, 0
    for c in needed_sales:
        if c not in improved_sales.columns:
            return None, None, 0

    u = universe[needed_universe].drop_duplicates(parcel_key)
    merged = improved_sales[needed_sales].merge(
        u, left_on=sales_join_key, right_on=parcel_key, how="inner"
    )
    merged = merged.dropna(subset=[lat_col, lon_col, sale_price_col, candidate_col, assr_impr_col])
    merged = merged[merged[sale_price_col] > 0]
    if len(merged) < (k + 2):
        return None, None, 0

    painted_total = merged[candidate_col].fillna(0) + merged[assr_impr_col].fillna(0)
    residual_pct = (merged[sale_price_col] - painted_total) / merged[sale_price_col]
    residual_pct = residual_pct.replace([np.inf, -np.inf], np.nan)
    valid = residual_pct.notna()
    if valid.sum() < (k + 2):
        return None, None, 0

    sub = merged.loc[valid].copy()
    x = residual_pct.loc[valid].values.astype(float)

    mean_lat = sub[lat_col].mean()
    deg_to_m_lat = 111_320.0
    deg_to_m_lon = 111_320.0 * float(np.cos(np.deg2rad(mean_lat)))
    coords = np.column_stack([
        sub[lat_col].values * deg_to_m_lat,
        sub[lon_col].values * deg_to_m_lon,
    ])

    tree = cKDTree(coords)
    # k+1 because the closest point is itself; drop column 0.
    _, idxs = tree.query(coords, k=k + 1)
    neighbors = idxs[:, 1:]

    n = len(x)
    x_mean = x.mean()
    z = x - x_mean
    z2_sum = float((z * z).sum())
    if z2_sum <= 0:
        return None, None, n

    # Row-normalized weights: each row sums to 1, so each w_ij = 1/k.
    # Numerator: sum_i z_i * (1/k) * sum_j_in_N(i) z_j  =  (1/k) * sum_i z_i * sum_neighbors(z)
    neighbor_z_sum = z[neighbors].sum(axis=1)  # shape (n,)
    numer = float((z * neighbor_z_sum).sum()) / k
    # W = sum of all w_ij = n (each row sums to 1; n rows).
    W = float(n)
    morans_i = (n / W) * (numer / z2_sum)

    # Permutation test
    rng = np.random.default_rng(rng_seed)
    ge_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(z)
        perm_neighbor_sum = perm[neighbors].sum(axis=1)
        perm_numer = float((perm * perm_neighbor_sum).sum()) / k
        perm_i = (n / W) * (perm_numer / z2_sum)
        if abs(perm_i) >= abs(morans_i):
            ge_count += 1
    p_value = (ge_count + 1) / (n_permutations + 1)

    return float(morans_i), float(p_value), int(n)


# ------------------------------------------------------------------ L8 (standalone)

def run_holdout_vacant_test(
    universe: pd.DataFrame,
    witnesses: pd.DataFrame,
    paint_callback,
    *,
    holdout_pct: float = 0.20,
    rng_seed: int = 42,
    vacant_kind: str = "W1_vacant",
    witness_key_col: str = "parcel_key",
    universe_key_col: str = "key",
    candidate_col: str = "land_value",
    witness_value_col: str = "land_value",
) -> dict:
    """
    Held-out vacant-sale prediction accuracy (the L8 test).

    Splits the W1 vacant-sale witnesses into train (1 - holdout_pct) and
    held-out (holdout_pct) folds, re-paints the universe using only the train
    fold via ``paint_callback``, then evaluates the painter's predicted land
    value against the held-out witnesses' actual time-adjusted sale prices.

    The painter cannot memorize the held-out witnesses — they were never in
    the calibration set — so this is the cleanest available test of land-value
    prediction accuracy.

    Parameters
    ----------
    universe, witnesses : pandas.DataFrame
        Same as fed to the painter normally.
    paint_callback : callable
        Function accepting ``train_witnesses`` (a filtered copy of
        ``witnesses``) and returning a painted universe DataFrame with at
        least ``parcel_key_col`` and ``candidate_col``. Typically a closure
        around :func:`openavmkit.pipeline.run_land_painter_tables`.
    holdout_pct : float, default 0.20
    rng_seed : int, default 42
    vacant_kind : str, default ``"W1_vacant"``
        Witness kind to hold out from. Other kinds stay in the train fold.
    witness_key_col : str, default ``"parcel_key"``
        Key column on the witnesses DataFrame.
    universe_key_col : str, default ``"key"``
        Key column on the painter's output (the universe).
    candidate_col : str, default ``"land_value"``
        Column on the painter's output that holds the painted per-parcel
        land value.
    witness_value_col : str, default ``"land_value"``
        Column on the witnesses DataFrame that holds the time-adjusted sale
        price; in the witness pool this is itself called ``land_value``.

    Returns
    -------
    dict with keys ``median_ratio``, ``cod``, ``n``, ``p10``, ``p90``.
    Pass straight into :func:`run_lars_tests` via its ``holdout_result`` arg.
    """
    rng = np.random.default_rng(rng_seed)
    vacant = witnesses[witnesses["witness_kind"] == vacant_kind].copy()
    if len(vacant) < 10:
        return {"median_ratio": None, "cod": None, "n": 0, "p10": None, "p90": None}

    n_holdout = max(1, int(len(vacant) * holdout_pct))
    holdout_idx = rng.choice(vacant.index, size=n_holdout, replace=False)
    holdout = vacant.loc[holdout_idx]
    train_witnesses = witnesses.drop(index=holdout_idx)

    painted = paint_callback(train_witnesses)

    # Compare painted (keyed on universe_key_col) vs held-out witnesses
    # (keyed on witness_key_col, value in witness_value_col)
    p = painted[[universe_key_col, candidate_col]].rename(columns={universe_key_col: "__pk__"})
    h = holdout[[witness_key_col, witness_value_col]].rename(
        columns={witness_key_col: "__pk__", witness_value_col: "__sale__"}
    )
    merged = h.merge(p, on="__pk__", how="inner")
    if len(merged) == 0:
        return {"median_ratio": None, "cod": None, "n": 0, "p10": None, "p90": None}

    ratio = (merged[candidate_col] / merged["__sale__"]).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    if len(ratio) == 0:
        return {"median_ratio": None, "cod": None, "n": 0, "p10": None, "p90": None}

    try:
        cod = float(calc_cod(ratio.values))
    except Exception:
        cod = None
    return {
        "median_ratio": float(ratio.median()),
        "cod": cod,
        "n": int(len(ratio)),
        "p10": float(ratio.quantile(0.10)),
        "p90": float(ratio.quantile(0.90)),
    }


def run_vacant_ratio_study(
    painted: pd.DataFrame,
    witnesses: pd.DataFrame,
    *,
    witness_kinds: list | None = None,
    universe_key_col: str = "key",
    candidate_col: str = "land_value",
    witness_key_col: str = "parcel_key",
    witness_value_col: str = "land_value",
) -> dict:
    """L8 — no-holdout variant. Sales-ratio study of the painted land value
    against the actual sale price for all vacant + teardown witnesses.

    Less rigorous than :func:`run_holdout_vacant_test` because the witness
    pool was used to calibrate the painter — but on thin witness pools
    (e.g. jurisdictions with very few clean vacant sales) the holdout
    variant isn't feasible and this is the next-best signal.

    Returns the same dict shape as :func:`run_holdout_vacant_test` so the
    result can be passed straight to :func:`run_lars_tests` via the
    ``holdout_result`` arg.

    Parameters
    ----------
    painted : DataFrame
        The painted universe with per-parcel ``candidate_col`` populated.
    witnesses : DataFrame
        The witness pool. ``witness_value_col`` carries each witness's
        time-adjusted sale price.
    witness_kinds : list[str], optional
        Witness kinds to include. Defaults to ``["W1_vacant", "W2_teardown"]``
        — sales where the building had ~zero value at the time of sale,
        making the sale price effectively a land-only transaction.
    """
    if witness_kinds is None:
        witness_kinds = ["W1_vacant", "W2_teardown"]

    empty = {"median_ratio": None, "cod": None, "n": 0, "p10": None, "p90": None}

    if "witness_kind" not in witnesses.columns:
        return empty
    w = witnesses[witnesses["witness_kind"].isin(witness_kinds)].copy()
    if len(w) == 0:
        return empty

    p = painted[[universe_key_col, candidate_col]].rename(
        columns={universe_key_col: "__pk__"}
    )
    h = w[[witness_key_col, witness_value_col]].rename(
        columns={witness_key_col: "__pk__", witness_value_col: "__sale__"}
    )
    merged = h.merge(p, on="__pk__", how="inner")
    if len(merged) == 0:
        return empty

    ratio = (merged[candidate_col] / merged["__sale__"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    ratio = ratio[ratio > 0]
    if len(ratio) == 0:
        return empty

    try:
        cod = float(calc_cod(ratio.values))
    except Exception:
        cod = None
    return {
        "median_ratio": float(ratio.median()),
        "cod": cod,
        "n": int(len(ratio)),
        "p10": float(ratio.quantile(0.10)),
        "p90": float(ratio.quantile(0.90)),
    }


# ------------------------------------------------------------------ L12

def _l12_depreciation_localization(
    universe: pd.DataFrame,
    *,
    candidate_col: str,
    neighborhood_col: str,
    bldg_area_col: str,
    land_area_col: str,
    bldg_age_col: str,
    bldg_quality_col: str,
    assr_market_col: str,
    size_quantiles: int = 3,
    min_per_cluster: int = 5,
):
    """
    Depreciation localization — "buildings depreciate, land doesn't."

    Within (neighborhood, size_bucket, quality_bucket) clusters — i.e.
    controlling for everything except age — check that:

    * the **implied building $/sqft** is negatively correlated with age
      (depreciation is real and shows up where expected); and
    * the **candidate land $/sqft** is uncorrelated with age (no leakage
      of the depreciation signal into the land component).

    For each cluster with >= ``min_per_cluster`` parcels we compute two
    Spearman rank correlations. Aggregating across clusters:

    * **L12a** — median Spearman(age, bldg_implied_psf). Expected strongly
      negative; this is the sanity check.
    * **L12b** — median |Spearman(age, candidate_land_psf)|. Expected near
      zero; this is the discriminating metric.
    * **L12 score** — |L12a| - L12b. Higher = depreciation localizes in
      buildings; lower (or negative) = candidate land values are
      contaminated by building-age signal.

    Returns ``(L12a, L12b, score, n_clusters)``. Any value can be ``None``
    if data is insufficient (missing age column, < 1 valid cluster, etc.).
    """
    needed = [
        candidate_col, bldg_age_col, bldg_area_col, land_area_col,
        assr_market_col, neighborhood_col,
    ]
    for c in needed:
        if c not in universe.columns:
            return None, None, None, 0

    df = universe.copy()
    df = df[
        df[bldg_area_col].fillna(0).gt(0)
        & df[land_area_col].fillna(0).gt(0)
        & df[bldg_age_col].notna()
        & df[assr_market_col].notna()
        & df[candidate_col].notna()
    ]
    if len(df) < min_per_cluster:
        return None, None, None, 0

    df["__bldg_implied__"] = df[assr_market_col] - df[candidate_col]
    df = df[df["__bldg_implied__"].gt(0)]
    if len(df) < min_per_cluster:
        return None, None, None, 0

    df["__bldg_psf__"] = df["__bldg_implied__"] / df[bldg_area_col]
    df["__land_psf__"] = df[candidate_col] / df[land_area_col]

    # Build cluster key: neighborhood × size_bin × quality_bin
    df["__size_bin__"] = df.groupby(neighborhood_col, dropna=False)[bldg_area_col] \
        .transform(lambda s: pd.qcut(s, q=size_quantiles, labels=False, duplicates="drop"))
    if bldg_quality_col in df.columns:
        df["__qual_bin__"] = df[bldg_quality_col]
        cluster_cols = [neighborhood_col, "__size_bin__", "__qual_bin__"]
    else:
        cluster_cols = [neighborhood_col, "__size_bin__"]

    bldg_rhos: list[float] = []
    land_rhos: list[float] = []

    for _, sub in df.groupby(cluster_cols, dropna=False):
        if len(sub) < min_per_cluster:
            continue
        age = sub[bldg_age_col].astype(float)
        # Spearman is undefined if either side has zero variance
        if age.nunique() < 2:
            continue
        # Use scipy if available for a proper Spearman; otherwise fall back to
        # numpy via pandas rank-then-pearson, which is equivalent.
        bldg_psf = sub["__bldg_psf__"].astype(float)
        land_psf = sub["__land_psf__"].astype(float)
        if bldg_psf.nunique() >= 2:
            rho_b = age.rank().corr(bldg_psf.rank())
            if pd.notna(rho_b):
                bldg_rhos.append(float(rho_b))
        if land_psf.nunique() >= 2:
            rho_l = age.rank().corr(land_psf.rank())
            if pd.notna(rho_l):
                land_rhos.append(float(rho_l))

    if not bldg_rhos and not land_rhos:
        return None, None, None, 0

    # Use building-side cluster count as the headline n (the test only makes
    # sense when both rhos are present per cluster, but in practice they
    # almost always are).
    n_clusters = min(len(bldg_rhos), len(land_rhos)) if (bldg_rhos and land_rhos) else 0
    l12a = float(np.median(bldg_rhos)) if bldg_rhos else None
    l12b_abs = float(np.median(np.abs(land_rhos))) if land_rhos else None
    if l12a is not None and l12b_abs is not None:
        score = abs(l12a) - l12b_abs
    else:
        score = None

    return l12a, l12b_abs, score, n_clusters


# ------------------------------------------------------------------ output

def _print_summary(res: LarsTestsResult) -> None:
    print(f"\nLars-Tests on '{res.candidate}':")
    print(f"  L1  improvement-neutrality: "
          f"{_pctfmt(res.L1_neutrality_pct)} (n={res.L1_n_pairs:,} matched pairs)")
    print(f"  L2  within-cluster COD:     "
          f"{_numfmt(res.L2_uniformity_cod)} (n={res.L2_n_clusters} clusters)")
    print(f"  L3  vacant-burden flip:     "
          f"{_pctfmt(res.L3_vacant_burden_flip_pct)} (n={res.L3_n_vacant_parcels:,} vacant)")
    print(f"  L4  desirability Spearman:  "
          f"vs assr {_numfmt(res.L4_spearman_assr)}, "
          f"vs pred {_numfmt(res.L4_spearman_pred)} "
          f"(n={res.L4_n_neighborhoods} neighborhoods)")
    print(f"  L5  density-FAR ordering:   "
          f"{_pctfmt(res.L5_density_pair_pct)} (n={res.L5_n_pairs:,} zoning pairs)")
    print(f"  L6  per-cell size decay:    "
          f"{_pctfmt(res.L6_size_decay_pct)} (n={res.L6_n_large_lots:,} large lots)")
    print(f"  L7  impr-cost-table COD:    "
          f"{_numfmt(res.L7_impr_uniformity_cod)} (n={res.L7_n_clusters:,} bldg-similarity clusters)")
    if res.L8_n_holdout > 0:
        print(f"  L8  held-out vacant ratio:  "
              f"median={_numfmt(res.L8_holdout_vacant_median_ratio)}, "
              f"COD={_numfmt(res.L8_holdout_vacant_cod)} (n={res.L8_n_holdout:,} held out)")
    if res.L9_n_cross_pairs > 0:
        print(f"  L9  boundary smoothness:    "
              f"cross-cell {_pctfmt(res.L9_cross_cell_smooth_pct)} vs "
              f"within-cell {_pctfmt(res.L9_within_cell_smooth_pct)} "
              f"(n={res.L9_n_cross_pairs:,} cross-cell pairs)")
    if res.L10_n_improved_sales > 0:
        print(f"  L10 reconciliation:         "
              f"median={_numfmt(res.L10_recon_median_ratio)}, "
              f"COD={_numfmt(res.L10_recon_cod)} (n={res.L10_n_improved_sales:,} improved sales)")
    if res.L11_n_residuals > 0:
        print(f"  L11 residual Moran's I:     "
              f"I={_numfmt(res.L11_residual_morans_i)}, "
              f"p={_numfmt(res.L11_morans_p_value)} (n={res.L11_n_residuals:,} residuals)")
    if res.L12_n_clusters > 0:
        print(f"  L12 depreciation localiz.:  "
              f"score={_numfmt(res.L12_localization_score)} "
              f"(bldg rho={_numfmt(res.L12a_bldg_age_spearman)}, "
              f"land |rho|={_numfmt(res.L12b_land_age_spearman_abs)}, "
              f"n={res.L12_n_clusters:,} clusters)")
    # Side-by-side normalized + COD comparison tables are rendered separately
    # by `build_normalized_scores_table` / `build_cod_scores_table` (called
    # from the pipeline wrapper). The per-candidate text printout would just
    # be a one-column slice of those tables, so we don't duplicate it here.


def _pctfmt(x):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def _numfmt(x):
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def _normalize_lars_metrics(res: LarsTestsResult) -> tuple[list, list]:
    """
    Compute the unified "0% bad / 100% perfect" view alongside the raw COD
    group. Returns (normalized_rows, cod_rows) where each row is a
    ``(label, value)`` tuple. ``value`` is float in [0, 1] for normalized
    rows, or float COD for COD rows; either may be ``None`` when not
    available.

    Normalization rules — see the design table; in short: native % stays
    native, Spearman shifts/flips into [0, 1], Moran's I uses 1-|I|,
    median ratios use 1-|r-1| (clamped). L7 is candidate-invariant so it
    lives in the COD group only; L11 p-value is not normalized.
    """
    def _shift(rho):
        if rho is None:
            return None
        return max(0.0, min(1.0, (rho + 1.0) / 2.0))

    def _flip_shift(rho):  # target rho = -1 -> 1.0
        if rho is None:
            return None
        return max(0.0, min(1.0, (1.0 - rho) / 2.0))

    def _moran(i):
        if i is None:
            return None
        return max(0.0, min(1.0, 1.0 - abs(i)))

    def _ratio_to_one(r):
        if r is None:
            return None
        return max(0.0, min(1.0, 1.0 - abs(r - 1.0)))

    def _inv(v):
        if v is None:
            return None
        return max(0.0, min(1.0, 1.0 - v))

    normalized = [
        ("L1  improvement-neutrality",     res.L1_neutrality_pct),
        ("L3  vacant-burden flip",         res.L3_vacant_burden_flip_pct),
        ("L4a desirability vs assr",       _shift(res.L4_spearman_assr)),
        ("L4b desirability vs pred",       _shift(res.L4_spearman_pred)),
        ("L5  density-FAR ordering",       res.L5_density_pair_pct),
        ("L6  per-cell size decay",        res.L6_size_decay_pct),
        ("L8  holdout vacant ratio",       _ratio_to_one(res.L8_holdout_vacant_median_ratio)),
        ("L9  boundary smoothness cross",  res.L9_cross_cell_smooth_pct),
        ("L9  boundary smoothness within", res.L9_within_cell_smooth_pct),
        ("L10 reconciliation ratio",       _ratio_to_one(res.L10_recon_median_ratio)),
        ("L11 spatial-residual cleanness", _moran(res.L11_residual_morans_i)),
        ("L12a bldg depreciation",         _flip_shift(res.L12a_bldg_age_spearman)),
        ("L12b land temporal stability",   _inv(res.L12b_land_age_spearman_abs)),
        ("L12 localization score",         _shift(res.L12_localization_score)),
    ]

    cods = [
        ("L2  within-cluster",  res.L2_uniformity_cod,        ""),
        ("L7  impr-cost-table", res.L7_impr_uniformity_cod,   "(candidate-invariant)"),
        ("L8  holdout vacant",  res.L8_holdout_vacant_cod,    ""),
        ("L10 reconciliation",  res.L10_recon_cod,            ""),
    ]

    return normalized, cods


def build_normalized_scores_table(results: list) -> pd.DataFrame:
    """Side-by-side normalized scores. Rows = stats, columns = candidates.

    Values rendered as ``"NN.N%"`` strings (or ``"n/a"``). Pass straight
    to ``IPython.display.display(...)`` to render as a notebook table.
    """
    if not results:
        return pd.DataFrame()
    all_norm = [_normalize_lars_metrics(r)[0] for r in results]
    labels = [label for label, _ in all_norm[0]]
    data = {
        r.candidate: [_pctfmt(v) for _, v in rows]
        for r, rows in zip(results, all_norm)
    }
    return pd.DataFrame(data, index=pd.Index(labels, name="stat"))


def build_cod_scores_table(results: list) -> pd.DataFrame:
    """Side-by-side COD scores (lower = better). Rows = stats, columns = candidates.

    Values rendered as ``"NN.NNN"`` strings (or ``"n/a"``). L7 is
    candidate-invariant — the same value appears in every column.
    """
    if not results:
        return pd.DataFrame()
    all_cods = [_normalize_lars_metrics(r)[1] for r in results]
    labels = [label for label, _, _ in all_cods[0]]
    data = {
        r.candidate: [_numfmt(v) for _, v, _ in rows]
        for r, rows in zip(results, all_cods)
    }
    return pd.DataFrame(data, index=pd.Index(labels, name="stat (lower = better)"))


def write_lars_tests_report(
    results: list,
    out_path: str,
) -> None:
    """
    Write a Markdown report comparing multiple candidates side-by-side.

    Parameters
    ----------
    results : list[LarsTestsResult]
    out_path : str
    """
    lines = ["# Lars-Tests Report\n"]
    lines.append(
        "Twelve incentive-and-equity tests on a candidate land-value column. "
        "L1-L6 evaluate the land-side decomposition. L7 evaluates the "
        "underlying improvement-cost-table consistency. L8-L11 evaluate the "
        "decomposition against ground truth (vacant + improved sales) and "
        "geographic smoothness. L12 checks that depreciation localizes in "
        "buildings rather than bleeding into land.\n"
    )
    lines.append("## Internal-consistency tests (L1-L7)\n")
    lines.append(
        "**L1** improvement-neutrality % (higher better) - matched (improved, vacant) pairs whose land values agree within +/-10%. "
        "**L2** within-cluster COD (lower better) - uniformity of $/sqft across same-zoning, similar-size, same-cluster parcels. "
        "**L3** vacant-burden flip % (higher better) - share of vacant parcels whose tax goes UP under a revenue-neutral LVT. "
        "**L4a/L4b** Spearman vs assessor / predicted total $/sqft - desirability tracking. "
        "**L5** density-FAR ordering % (higher better) - share of zoning-class pairs (within multi-zoning neighborhoods) where the higher-FAR zone has the higher median land $/sqft. Random baseline = 50%. "
        "**L6** per-cell size decay % (higher better) - share of large-lot parcels (>=2x zoning_emp_min) whose $/sqft is at least 10% below the same neighborhood's typical-lot median. "
        "**L7** impr-cost-table COD (lower better) - per-cluster COD of `assr_impr_value / bldg_area` clustered by (size, age, quality, condition) ignoring location. Identical across candidates sharing the same `assr_impr_value`.\n"
    )
    lines.append("\n| Candidate | L1 % | L2 COD | L3 % | L4a | L4b | L5 % | L6 % | L7 COD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| " + " | ".join([
                r.candidate,
                _pctfmt(r.L1_neutrality_pct),
                _numfmt(r.L2_uniformity_cod),
                _pctfmt(r.L3_vacant_burden_flip_pct),
                _numfmt(r.L4_spearman_assr),
                _numfmt(r.L4_spearman_pred),
                _pctfmt(r.L5_density_pair_pct),
                _pctfmt(r.L6_size_decay_pct),
                _numfmt(r.L7_impr_uniformity_cod),
            ]) + " |"
        )

    # Skip L8-L11 section entirely if no candidate has any of those populated.
    has_external = any(
        (r.L8_n_holdout > 0)
        or (r.L9_n_cross_pairs > 0)
        or (r.L10_n_improved_sales > 0)
        or (r.L11_n_residuals > 0)
        for r in results
    )
    if has_external:
        lines.append("\n## Ground-truth and smoothness tests (L8-L11)\n")
        lines.append(
            "**L8** vacant-sale ratio - median painted_land / sale_price on vacant + teardown sales (target ~1.0; COD lower better). Catches systematic level-bias the internal tests can't. Two modes: a rigorous holdout variant (re-paints with a held-out fold of W1 vacants, requires a paint_callback and >=10 W1 vacants) or a no-holdout ratio study against the full W1+W2 witness pool (less rigorous but works on thin pools — Petersburg uses this). "
            "**L9** boundary smoothness % - share of nearby cross-cell parcel pairs whose $/sqft differs by less than 25%. Compared to within-cell pairs as a control. Cross-cell << within-cell means the cells create artificial discontinuities. "
            "**L10** reconciliation - median (painted_land + assr_impr) / sale_price on improved sales (target ~1.0; COD lower better). The improved-side mirror of L8. "
            "**L11** residual Moran's I (closer to 0 better) - spatial autocorrelation of `(sale - painted_total) / sale` across improved sales. Significantly > 0 means missed location signal clusters in identifiable pockets - an equity concern even when overall accuracy looks fine.\n"
        )
        lines.append(
            "\n| Candidate | L8 ratio | L8 COD | L9 cross-cell % | L9 within-cell % | L10 ratio | L10 COD | L11 I | L11 p |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            lines.append(
                "| " + " | ".join([
                    r.candidate,
                    _numfmt(r.L8_holdout_vacant_median_ratio),
                    _numfmt(r.L8_holdout_vacant_cod),
                    _pctfmt(r.L9_cross_cell_smooth_pct),
                    _pctfmt(r.L9_within_cell_smooth_pct),
                    _numfmt(r.L10_recon_median_ratio),
                    _numfmt(r.L10_recon_cod),
                    _numfmt(r.L11_residual_morans_i),
                    _numfmt(r.L11_morans_p_value),
                ]) + " |"
            )

    # L12 — depreciation-localization section. Always shown if any candidate
    # has L12 populated; depends only on universe data, so usually present.
    has_l12 = any(r.L12_n_clusters > 0 for r in results)
    if has_l12:
        lines.append("\n## Depreciation localization (L12)\n")
        lines.append(
            "**L12** depreciation localization (higher better) - within "
            "(neighborhood, size, quality) clusters, building $/sqft should "
            "fall with age (L12a strongly negative) while candidate land "
            "$/sqft should NOT (L12b near zero). Score = |L12a| - L12b. "
            "A high score means the candidate correctly localizes the "
            "depreciation signal in buildings; a low/negative score means "
            "the candidate is contaminated by building-age signal "
            "(typical of `sale - depreciated_cost` style land valuations).\n"
        )
        lines.append("\n| Candidate | L12 score | L12a bldg rho | L12b land &#124;rho&#124; |")
        lines.append("|---|---:|---:|---:|")
        for r in results:
            lines.append(
                "| " + " | ".join([
                    r.candidate,
                    _numfmt(r.L12_localization_score),
                    _numfmt(r.L12a_bldg_age_spearman),
                    _numfmt(r.L12b_land_age_spearman_abs),
                ]) + " |"
            )

    lines.append("\n## Sample sizes")
    lines.append(
        "| Candidate | L1 pairs | L2 clusters | L3 vacant | L4 nbhds | L5 zoning-pairs | L6 large lots | L7 clusters | L8 held out | L9 cross-pairs | L10 sales | L11 residuals | L12 clusters |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| " + " | ".join([
                r.candidate,
                f"{r.L1_n_pairs:,}",
                f"{r.L2_n_clusters:,}",
                f"{r.L3_n_vacant_parcels:,}",
                f"{r.L4_n_neighborhoods:,}",
                f"{r.L5_n_pairs:,}",
                f"{r.L6_n_large_lots:,}",
                f"{r.L7_n_clusters:,}",
                f"{r.L8_n_holdout:,}",
                f"{r.L9_n_cross_pairs:,}",
                f"{r.L10_n_improved_sales:,}",
                f"{r.L11_n_residuals:,}",
                f"{r.L12_n_clusters:,}",
            ]) + " |"
        )

    # ---- Normalized scores section ----
    # Every metric on a unified "0% bad -> 100% perfect" scale (except CODs,
    # which live in their own section below).
    all_norm = [_normalize_lars_metrics(r) for r in results]
    # Use the first candidate's labels as the column order (they're identical
    # across candidates).
    norm_labels = [label for label, _ in all_norm[0][0]]
    lines.append("\n## Normalized scores (0% bad -> 100% perfect)\n")
    lines.append(
        "Every test that maps cleanly to a single quality axis, rescaled to "
        "[0%, 100%] with 100% = perfect. CODs are reported separately below.\n"
    )
    header = "| Candidate | " + " | ".join(norm_labels) + " |"
    lines.append(header)
    lines.append("|---" + "|---:" * len(norm_labels) + "|")
    for r, (norm_rows, _) in zip(results, all_norm):
        cells = [r.candidate] + [_pctfmt(v) for _, v in norm_rows]
        lines.append("| " + " | ".join(cells) + " |")

    # ---- COD scores section ----
    lines.append("\n## COD scores (lower = better)\n")
    lines.append(
        "Coefficient of Dispersion: median absolute deviation of a "
        "per-cluster ratio from the cluster's median, expressed as a "
        "percent. IAAO residential ratio-study benchmark is roughly 15 or "
        "below. L7 is candidate-invariant (depends only on the assessor's "
        "improvement values, not on the candidate land column).\n"
    )
    cod_labels = [label for label, _, _ in all_norm[0][1]]
    cod_header = "| Candidate | " + " | ".join(cod_labels) + " |"
    lines.append(cod_header)
    lines.append("|---" + "|---:" * len(cod_labels) + "|")
    for r, (_, cod_rows) in zip(results, all_norm):
        cells = [r.candidate] + [_numfmt(v) for _, v, _ in cod_rows]
        lines.append("| " + " | ".join(cells) + " |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
