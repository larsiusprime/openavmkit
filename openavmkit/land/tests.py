"""
Lars-Tests for land valuations.

Six tests that score a candidate per-parcel land-value column on the
incentive-and-equity criteria from the ``Valuing Land: The Simplest
Viable Method`` essay and the ``How Assessors Value Land Right Now``
interview. These are deliberately *not* IAAO ratio studies — they
test whether a land valuation generates the economic incentives a Land
Value Tax is supposed to produce.

The six tests:

* **L1 — Improvement-neutrality.** For matched pairs of nearby parcels
  with similar size and zoning where one is improved and one is not (or
  is much-cheaper-improved), the predicted land values should be the
  same. Score = % of pairs within tolerance ε.
* **L2 — Side-by-side uniformity.** Within the assessor's land-equity
  clusters (typically location + zoning + size band), the COD of
  ``$/sqft`` should be low. Lower is better.
* **L3 — Vacant-burden flip.** Under a revenue-neutral LVT computed from
  these land values, the median vacant/underutilized parcel's tax bill
  should go *up* relative to the current property tax. Score = % of
  vacant parcels with a tax increase.
* **L4 — Desirability tracking.** Spearman correlation of land $/sqft
  with both (a) assessor's market $/sqft and (b) our model's market
  $/sqft, computed at the neighborhood-aggregate level so individual
  noise doesn't dominate. Should be strongly positive and monotonic.
* **L5 — Theory of Consistent Use.** Flag every parcel where the
  implied land highest-and-best-use disagrees with the improvement
  use (e.g. residential improvement on commercial-zoned land where
  land was valued at the commercial rate). Score = count of violators.
* **L6 — Excess vs Surplus correctness.** For parcels whose deeded
  acreage exceeds the local zoning minimum-lot-size by ≥ another full
  minimum-lot-size, the *excess* portion (separately developable) should
  trade at near-full rate; mere *surplus* (extra-but-not-divisible)
  should trade at lower marginal $/sqft. Score = % of large-lot parcels
  whose marginal $/sqft curve fits this expectation.

The harness runs all six and writes a single Markdown report.

See Also
--------
openavmkit.zoning : Provides L6's empirical zoning floor.
openavmkit.land.evidence : Provides the witness pool for L1/L4 baselines.
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
    """Aggregate result from running the Lars-Tests on one candidate."""
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
    L5_consistent_use_violators: int = 0
    L5_n_checked: int = 0
    L6_excess_surplus_correct_pct: float | None = None
    L6_n_large_lot_parcels: int = 0
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
    pred_market_col: str | None = "prediction",
    cluster_col: str = "land_he_id",
    zoning_emp_min_col: str = "zoning_emp_min_lot_sqft",
    eps_pct: float = 0.10,
    verbose: bool = False,
) -> LarsTestsResult:
    """
    Score a candidate land-value column on all six Lars-Tests.

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

    # ---- L5: Theory of Consistent Use ----
    res.L5_consistent_use_violators, res.L5_n_checked = _l5_consistent_use(
        work,
        candidate_col=candidate_col,
        assr_land_col=assr_land_col,
        bldg_area_col=bldg_area_col,
    )

    # ---- L6: Excess vs Surplus correctness ----
    res.L6_excess_surplus_correct_pct, res.L6_n_large_lot_parcels = _l6_excess_surplus(
        work,
        per_sqft_col="__cand_per_sqft__",
        zoning_emp_min_col=zoning_emp_min_col,
        land_area_col=land_area_col,
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

def _l5_consistent_use(
    work: pd.DataFrame,
    *,
    candidate_col: str,
    assr_land_col: str,
    bldg_area_col: str,
):
    """
    Theory of Consistent Use (Holding interview PDF): land and
    improvements should be valued under the same highest-and-best-use.
    Operationally: flag parcels where the *candidate land $/sqft* is
    much higher than the local norm for parcels with the same building
    use (proxy: same zoning + improved). Those are the "we valued this
    for redevelopment but it's currently improved as something else"
    cases that the Holding PDF cites as overtaxing the citizen.

    The implementation here is a v1 placeholder: it flags improved
    parcels whose candidate $/sqft exceeds 2× the median $/sqft of
    other improved parcels in the same neighborhood. Returns
    (n_violators, n_checked).

    Note: this test makes the most sense for our candidate; running it
    on the assessor as baseline measures the assessor against itself
    and is therefore ill-defined for direct comparison. Both rows
    appear in the report for transparency.
    """
    if "neighborhood" not in work.columns:
        return 0, 0
    is_built = work[bldg_area_col].fillna(0) > 0
    base = work[is_built & (work[candidate_col] > 0) & (work["land_area_sqft"] > 0)].copy()
    if len(base) == 0:
        return 0, 0
    base["__cand_per_sqft__"] = base[candidate_col] / base["land_area_sqft"]
    nbhd_median = base.groupby("neighborhood", dropna=False)["__cand_per_sqft__"].transform("median")
    rel = base["__cand_per_sqft__"] / nbhd_median.replace(0, np.nan)
    violators = int((rel > 2.0).sum())
    return violators, int(len(base))


# ------------------------------------------------------------------ L6

def _l6_excess_surplus(
    work: pd.DataFrame,
    *,
    per_sqft_col: str,
    zoning_emp_min_col: str,
    land_area_col: str,
):
    """
    For parcels with land_area >= 2 × zoning min lot, check whether the
    candidate's marginal $/sqft is *lower* than the parcel's normal
    $/sqft by at least 10% — consistent with diminishing returns to size
    that the user's PDFs and Wake's SOV both call out.

    Caveat: this version doesn't distinguish excess (separately
    developable) from surplus (just-more-grass-to-mow); it tests whether
    the painter exhibits *some* size-decay, which is the basic signal.
    A future refinement would split on whether geometry + zoning permits
    lot subdivision.
    """
    if zoning_emp_min_col not in work.columns:
        return None, 0
    df = work[work[zoning_emp_min_col] > 0].copy()
    df["__size_ratio__"] = df[land_area_col] / df[zoning_emp_min_col]
    large = df[df["__size_ratio__"] >= 2.0].copy()
    n_large = len(large)
    if n_large == 0:
        return None, 0
    # Compute neighborhood-baseline $/sqft from "normal-sized" parcels
    # (size_ratio in [0.9, 1.5]) and compare large lots against it.
    base = df[(df["__size_ratio__"] >= 0.9) & (df["__size_ratio__"] <= 1.5)]
    if len(base) == 0:
        return None, n_large
    base_median = base[per_sqft_col].median()
    large_median = large[per_sqft_col].median()
    if not np.isfinite(base_median) or base_median <= 0:
        return None, n_large
    # Score: are the large-lot $/sqft values < base * 0.90?
    correct = int((large[per_sqft_col] <= base_median * 0.90).sum())
    return correct / n_large, n_large


# ------------------------------------------------------------------ output

def _print_summary(res: LarsTestsResult) -> None:
    print(f"\nLars-Tests on '{res.candidate}':")
    print(f"  L1 improvement-neutrality:  "
          f"{_pctfmt(res.L1_neutrality_pct)} (n={res.L1_n_pairs:,} matched pairs)")
    print(f"  L2 side-by-side uniformity: "
          f"COD={_numfmt(res.L2_uniformity_cod)} (n={res.L2_n_clusters} clusters)")
    print(f"  L3 vacant-burden flip:      "
          f"{_pctfmt(res.L3_vacant_burden_flip_pct)} (n={res.L3_n_vacant_parcels:,} vacant)")
    print(f"  L4 desirability Spearman:   "
          f"vs assr {_numfmt(res.L4_spearman_assr)}, "
          f"vs pred {_numfmt(res.L4_spearman_pred)} "
          f"(n={res.L4_n_neighborhoods} neighborhoods)")
    print(f"  L5 consistent-use violators:"
          f" {res.L5_consistent_use_violators:,}/{res.L5_n_checked:,}")
    print(f"  L6 excess/surplus signal:   "
          f"{_pctfmt(res.L6_excess_surplus_correct_pct)} (n={res.L6_n_large_lot_parcels:,} large lots)")


def _pctfmt(x):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def _numfmt(x):
    if x is None:
        return "n/a"
    return f"{x:.3f}"


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
    lines.append("Compared candidates and their incentive-test scores.\n")
    lines.append("Columns: L1=improvement-neutrality (higher better), "
                 "L2=within-cluster COD (lower better), "
                 "L3=vacant-burden flip (higher better), "
                 "L4a=Spearman vs assessor $/sqft, "
                 "L4b=Spearman vs predicted $/sqft, "
                 "L5=consistent-use violators count, "
                 "L6=excess/surplus signal % (higher better).\n")
    lines.append("\n| Candidate | L1 % | L2 COD | L3 % | L4a | L4b | L5 viol | L6 % |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            "| " + " | ".join([
                r.candidate,
                _pctfmt(r.L1_neutrality_pct),
                _numfmt(r.L2_uniformity_cod),
                _pctfmt(r.L3_vacant_burden_flip_pct),
                _numfmt(r.L4_spearman_assr),
                _numfmt(r.L4_spearman_pred),
                f"{r.L5_consistent_use_violators}/{r.L5_n_checked}",
                _pctfmt(r.L6_excess_surplus_correct_pct),
            ]) + " |"
        )
    lines.append("\n## Sample sizes")
    lines.append("| Candidate | L1 pairs | L2 clusters | L3 vacant | L4 nbhds | L6 large lots |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            "| " + " | ".join([
                r.candidate,
                f"{r.L1_n_pairs:,}",
                f"{r.L2_n_clusters:,}",
                f"{r.L3_n_vacant_parcels:,}",
                f"{r.L4_n_neighborhoods:,}",
                f"{r.L6_n_large_lot_parcels:,}",
            ]) + " |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
