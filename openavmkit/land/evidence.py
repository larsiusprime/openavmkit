"""
Land-value witness curation.

Pulls together five complementary streams of land-value evidence from the
parcels and sales data, applies contamination filters, and stamps each
witness with a confidence weight. Downstream rungs (LYCD painter, Lars-Tests)
consume the unified witness DataFrame.

The witness streams (per the planning document):

* **W1 — clean vacant sales:** ``valid_sale=True AND vacant_sale=True``,
  with extra contamination pruning.
* **W2 — teardown sales:** sales already flipped to ``vacant_sale=True``
  by the permits enrichment (sale followed by demolition permit). Per
  the user's framing these are the gold for *prime buildable* land —
  proven economically usable.
* **W3 — extraction land residuals:** for sales where the building
  component value can be trusted (recent + frozen Schedule of Values
  cost basis), land = sale_price − depreciated_cost. Wake's frozen
  SOV cost+depreciation tables make this reliable.
* **W4 — low-FAR proxy:** sales of small old houses on big lots with
  ``floor_area_ratio`` in the bottom decile of the local distribution
  often trade near land value (the teardown play).
* **W5 — per-parcel prior land transfer:** Wake's parcels.csv records
  the most recent vacant-land transfer per parcel
  (``prior_land_xfer_price/date``); usable as a witness even when the
  parcel is now improved.

Each witness contributes one row keyed by parcel and event date with a
confidence weight. The default weights ranking (W2 > W3 > W5 > W1 > W4)
reflects the user's stated preference for prime-buildable evidence over
"vacant for a reason" sales.

Contamination filters applied across all streams:

* Drop PUV parcels (``land_deferred_code`` non-empty) — assessed at
  state-set rates, not market.
* Drop sales of parcels smaller than the local empirical zoning
  minimum.
* Drop sales of parcels larger than ``max_size_multiplier`` ×
  neighborhood-median land size.
* Drop disq flags D/E/F/G that leaked through.

See Also
--------
openavmkit.zoning : Provides the de-facto zoning floor.
openavmkit.land.lycd : LYCD uniform-rate painter that pools the witnesses.
openavmkit.land.tests : Lars-Tests that reuse the witness pool.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Default per-witness confidence weights. The unit is dimensionless;
# only relative magnitudes matter when pooling. Tunable.
DEFAULT_WEIGHTS = {
    "W1_vacant": 1.0,
    "W2_teardown": 1.5,
    "W3_extraction": 1.2,
    "W4_low_far": 0.8,         # raised after the v1 double-counting bug fix
    "W5_prior_xfer": 0.7,
    "W6_pred_residual": 1.0,   # full-universe synthetic from ensemble residual
}

LEAKED_DISQ_FLAGS = ("D", "E", "F", "G")


# -----------------------------------------------------------------------------
# Anomaly-filter constants
# -----------------------------------------------------------------------------
#
# These thresholds gate whether an improved-sale witness is treated as
# arms-length market evidence. Each is calibrated against a physical or
# logical lower bound rather than against statistical outlier-ness, so the
# rules are defensible without reference to our own model's predictions.
#
# Mild negative residuals (sale 0–25% below depreciated cost) are NOT
# anomalies — they reflect the normal market spread for older homes
# trading slightly under cost basis (~5–8% per the Holding-interview PDF
# cited in the planning doc). The diagnostic_findings.md report shows
# 169 of 237 negative residuals fall into this benign band; only the
# severe + very_neg tails need filtering. See also SUMMARY.md, "Known
# noise patterns".
TOKEN_PRICE_THRESHOLD_USD = 5000
"""Sales priced below this are token-consideration recordings for
non-arms-length transfers (gift deeds, family transfers, $1 conveyances).
Convention rooted in real-estate recording practice."""

MIN_PHYSICAL_BUILDING_PSF_USD = 30
"""Wake's 2024 Schedule of Values publishes residential base prices
ranging from ~$36/sqft (Class D, lowest finished construction) up to
~$280/sqft (A+25 luxury). The lowest plausible residential sale-per-sqft
for a real arms-length transaction in Wake should clear ~$30/sqft —
this is below the cost floor for any habitable construction and
indicates non-arms-length conveyance. Used by the anomaly filter to
flag sales below the physical cost basis."""

BELOW_COST_FLOOR_RATIO = 0.5
"""Sale-to-depreciated-cost ratio below which a newish good-condition
sale is considered anomalous. A 30-year-old home in 0.7+ condition can't
rationally sell for less than half its replacement cost in a market like
Wake — both the land and the (still-functional) structure are worth
something."""

PRIOR_XFER_INCONSISTENT_RATIO = 0.5
"""Sale price below this fraction of the time-adjusted prior xfer is
flagged as inconsistent. Wake-area land doesn't fall 50% in absolute
nominal price across a 10-year window."""

PRIOR_XFER_LOOKBACK_YEARS = 10

BELOW_COST_FLOOR_MAX_AGE_YEARS = 30
BELOW_COST_FLOOR_MIN_CONDITION = 0.7


@dataclass
class WitnessConfig:
    """
    Knobs governing witness curation.

    Parameters
    ----------
    weights : dict
        Confidence weights per witness kind. Keys must match
        ``DEFAULT_WEIGHTS``.
    max_size_multiplier : float, default 5.0
        Drop sales where parcel land area exceeds this multiple of the
        local-neighborhood median.
    min_size_multiplier : float, default 1.0
        Drop sales where parcel land area is below this multiple of the
        local empirical zoning minimum.
    w3_max_age_years : int, default 100
        Maximum building age (at sale) for W3 extraction. Originally
        capped at 2 (new construction only) on the theory that cost
        basis is more trustworthy for new buildings; cost-residual
        investigation on Wake (out/land/rung1/cost_residual_investigation.md)
        showed the assessor's frozen Schedule of Values produces
        consistent grade-premium $/sqft across all ages, so the cap was
        relaxed. 100 effectively means "any age".
    w4_far_percentile : float, default 0.10
        Bottom-percentile of FAR within a neighborhood that qualifies for
        the W4 low-FAR proxy stream.
    w5_max_age_years : int, default 5
        Maximum age (at valuation date) of a prior-land-transfer record
        to admit as a W5 witness. Older records risk poor time
        adjustment.
    drop_puv : bool, default True
        Drop tax-shelter (PUV) parcels.
    drop_historic_deferred : bool, default True
        Drop historic-preservation deferred parcels.
    """
    weights: dict = None
    max_size_multiplier: float = 5.0
    min_size_multiplier: float = 1.0
    w3_max_age_years: int = 100
    w4_far_percentile: float = 0.10
    w5_max_age_years: int = 5
    drop_puv: bool = True
    drop_historic_deferred: bool = True

    def __post_init__(self):
        if self.weights is None:
            self.weights = dict(DEFAULT_WEIGHTS)


def _compute_neighborhood_size_stats(
    universe: pd.DataFrame,
    *,
    neighborhood_col: str = "neighborhood",
    land_area_col: str = "land_area_sqft",
) -> pd.Series:
    """Median land area per neighborhood, used for the upper-size filter."""
    s = universe.copy()
    s = s[s[land_area_col] > 0]
    return s.groupby(neighborhood_col, dropna=False)[land_area_col].median()


def _attach_contamination_columns(
    df: pd.DataFrame,
    *,
    universe: pd.DataFrame,
    parcel_key: str,
    join_key: str,
    nbhd_median_size: pd.Series,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
) -> pd.DataFrame:
    """Add columns the filters need: zoning_min_lot_sqft, nbhd_median_size."""
    universe_keyed = universe.set_index(parcel_key)
    out = df.copy()
    out["__nbhd_median_size__"] = out[neighborhood_col].map(nbhd_median_size)
    if zoning_min_col in universe.columns:
        out["__zoning_min__"] = out[join_key].map(universe_keyed[zoning_min_col])
    else:
        out["__zoning_min__"] = np.nan
    return out


def _apply_baseline_filters(
    df: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    land_area_col: str,
    flag_col: str | None,
    log_prefix: str,
    verbose: bool,
) -> pd.DataFrame:
    """
    Drop rows that fail PUV / disq / size-bound checks. Operates on a
    DataFrame already enriched with ``__zoning_min__`` and
    ``__nbhd_median_size__``.
    """
    n0 = len(df)
    log = []
    out = df.copy()

    # Disq leakage
    if flag_col and flag_col in out.columns:
        leaked = out[flag_col].astype("string").isin(LEAKED_DISQ_FLAGS)
        n_drop = int(leaked.fillna(False).sum())
        if n_drop:
            out = out[~leaked.fillna(False)]
            log.append(f"dropped {n_drop} for disq flag in {LEAKED_DISQ_FLAGS}")

    # PUV
    if cfg.drop_puv and "land_deferred_code" in out.columns:
        is_puv = out["land_deferred_code"].astype("string").str.strip().fillna("") != ""
        n_drop = int(is_puv.sum())
        if n_drop:
            out = out[~is_puv]
            log.append(f"dropped {n_drop} PUV parcels")

    if cfg.drop_historic_deferred and "historic_deferred_code" in out.columns:
        is_hd = (
            out["historic_deferred_code"].astype("string").str.strip().fillna("") != ""
        )
        n_drop = int(is_hd.sum())
        if n_drop:
            out = out[~is_hd]
            log.append(f"dropped {n_drop} historic-deferred parcels")

    # Below zoning min
    has_zmin = out["__zoning_min__"].notna() & (out["__zoning_min__"] > 0)
    too_small = has_zmin & (
        out[land_area_col] < cfg.min_size_multiplier * out["__zoning_min__"]
    )
    n_drop = int(too_small.fillna(False).sum())
    if n_drop:
        out = out[~too_small.fillna(False)]
        log.append(
            f"dropped {n_drop} below {cfg.min_size_multiplier}x zoning min lot size"
        )

    # Above neighborhood-median × multiplier
    has_med = out["__nbhd_median_size__"].notna() & (out["__nbhd_median_size__"] > 0)
    too_big = has_med & (
        out[land_area_col] > cfg.max_size_multiplier * out["__nbhd_median_size__"]
    )
    n_drop = int(too_big.fillna(False).sum())
    if n_drop:
        out = out[~too_big.fillna(False)]
        log.append(
            f"dropped {n_drop} above {cfg.max_size_multiplier}x neighborhood-median size"
        )

    if verbose:
        print(f"  {log_prefix}: kept {len(out):,}/{n0:,}")
        for entry in log:
            print(f"    - {entry}")

    return out


def curate_w1_clean_vacant(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    parcel_key: str,
    sales_join_key: str,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
    nbhd_median_size: pd.Series,
    verbose: bool = False,
) -> pd.DataFrame:
    """W1: clean vacant sales, with contamination pruning."""
    sales = sales.copy()
    keep = (
        (sales["valid_sale"].fillna(False))
        & (sales["vacant_sale"].fillna(False))
        & (sales["sale_price"].fillna(0) > 0)
        & (sales[land_area_col].fillna(0) > 0)
    )
    base = sales[keep].copy()

    base = _attach_contamination_columns(
        base, universe=universe, parcel_key=parcel_key, join_key=sales_join_key,
        nbhd_median_size=nbhd_median_size, neighborhood_col=neighborhood_col,
        land_area_col=land_area_col, zoning_min_col=zoning_min_col,
    )
    # Carry deferred-code columns from universe so PUV filter works
    universe_keyed = universe.set_index(parcel_key)
    for col in ("land_deferred_code", "historic_deferred_code"):
        if col in universe.columns:
            base[col] = base[sales_join_key].map(universe_keyed[col])

    base = _apply_baseline_filters(
        base, cfg=cfg, land_area_col=land_area_col,
        flag_col="disq_flag", log_prefix="W1 vacant", verbose=verbose,
    )

    out = pd.DataFrame({
        "parcel_key": base[sales_join_key].values,
        "witness_kind": "W1_vacant",
        "land_value": base.get("sale_price_time_adj", base["sale_price"]).values,
        "land_area_sqft": base[land_area_col].values,
        "event_date": pd.to_datetime(base.get("sale_date"), errors="coerce").values,
        "weight": cfg.weights["W1_vacant"],
        "notes": "clean vacant sale",
    })
    out["land_value_per_sqft"] = out["land_value"] / out["land_area_sqft"]
    return out


def curate_w2_teardown(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    parcel_key: str,
    sales_join_key: str,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
    nbhd_median_size: pd.Series,
    verbose: bool = False,
) -> pd.DataFrame:
    """W2: teardown sales — already flipped to vacant_sale=True by permits enrichment."""
    candidate_flags = ("is_teardown_sale", "is_teardown", "teardown")
    flag = next((c for c in candidate_flags if c in sales.columns), None)
    if flag is None:
        if verbose:
            print(
                f"  W2 teardown: none of {candidate_flags} present in sales, skipping"
            )
        return _empty_witness_frame()

    sales = sales.copy()
    keep = (
        (sales["valid_sale"].fillna(False))
        & (sales[flag].fillna(False))
        & (sales["sale_price"].fillna(0) > 0)
        & (sales[land_area_col].fillna(0) > 0)
    )
    base = sales[keep].copy()
    if len(base) == 0:
        return _empty_witness_frame()

    base = _attach_contamination_columns(
        base, universe=universe, parcel_key=parcel_key, join_key=sales_join_key,
        nbhd_median_size=nbhd_median_size, neighborhood_col=neighborhood_col,
        land_area_col=land_area_col, zoning_min_col=zoning_min_col,
    )
    universe_keyed = universe.set_index(parcel_key)
    for col in ("land_deferred_code", "historic_deferred_code"):
        if col in universe.columns:
            base[col] = base[sales_join_key].map(universe_keyed[col])
    base = _apply_baseline_filters(
        base, cfg=cfg, land_area_col=land_area_col,
        flag_col="disq_flag", log_prefix="W2 teardown", verbose=verbose,
    )

    out = pd.DataFrame({
        "parcel_key": base[sales_join_key].values,
        "witness_kind": "W2_teardown",
        "land_value": base.get("sale_price_time_adj", base["sale_price"]).values,
        "land_area_sqft": base[land_area_col].values,
        "event_date": pd.to_datetime(base.get("sale_date"), errors="coerce").values,
        "weight": cfg.weights["W2_teardown"],
        "notes": "teardown — sale followed by demolition permit",
    })
    out["land_value_per_sqft"] = out["land_value"] / out["land_area_sqft"]
    return out


def curate_w3_extraction(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    parcel_key: str,
    sales_join_key: str,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
    nbhd_median_size: pd.Series,
    flagged_out: list | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    W3: extraction land residual = sale_price − depreciated_cost,
    restricted to sales where the building is at most ``w3_max_age_years``
    years old (cost basis most trustworthy).
    """
    sales = sales.copy()
    if "bldg_year_built" not in sales.columns or "sale_date" not in sales.columns:
        if verbose:
            print("  W3 extraction: missing bldg_year_built or sale_date, skipping")
        return _empty_witness_frame()

    sale_date = pd.to_datetime(sales["sale_date"], errors="coerce")
    yb = pd.to_numeric(sales["bldg_year_built"], errors="coerce")
    age_at_sale = sale_date.dt.year - yb
    keep = (
        (sales["valid_sale"].fillna(False))
        & (~sales["vacant_sale"].fillna(False))
        & (sales["sale_price"].fillna(0) > 0)
        & (sales[land_area_col].fillna(0) > 0)
        & (age_at_sale <= cfg.w3_max_age_years)
        & (age_at_sale >= 0)
    )
    base = sales[keep].copy()
    if len(base) == 0:
        if verbose:
            print(f"  W3 extraction: no eligible sales (max age {cfg.w3_max_age_years})")
        return _empty_witness_frame()

    universe_keyed = universe.set_index(parcel_key)

    # Use universe-side assr_impr_value as the depreciated-cost proxy
    if "assr_impr_value" not in universe.columns:
        if verbose:
            print("  W3 extraction: universe lacks assr_impr_value, skipping")
        return _empty_witness_frame()

    base["__assr_impr_value__"] = base[sales_join_key].map(
        universe_keyed["assr_impr_value"]
    )
    base["assr_impr_value"] = base["__assr_impr_value__"]
    # Bring building age + condition for anomaly evaluation
    for c in ("bldg_age_years", "bldg_condition_num", "prior_land_xfer_price",
              "prior_land_xfer_date"):
        if c in universe.columns and c not in base.columns:
            base[c] = base[sales_join_key].map(universe_keyed[c])
    sp = base.get("sale_price_time_adj", base["sale_price"])
    base["__land_residual__"] = sp - base["__assr_impr_value__"].fillna(0)
    base = base[base["__land_residual__"] > 0]

    # Anomaly filter — drop flagged sales but capture for audit trail
    anom = evaluate_sale_anomaly_flags(base)
    base = base.join(anom)
    flagged_mask = base["anomaly_flags"].astype("string").str.len() > 0
    n_flagged = int(flagged_mask.fillna(False).sum())
    if n_flagged and flagged_out is not None:
        flagged_audit = base.loc[flagged_mask].copy()
        flagged_audit["__source_witness__"] = "W3_extraction"
        flagged_out.append(flagged_audit)
    base = base[~flagged_mask.fillna(False)]
    if verbose and n_flagged:
        rule_counts = (
            anom.loc[flagged_mask, "anomaly_flags"].str.split(";")
            .explode().value_counts()
        )
        print(f"  W3 extraction: anomaly filter dropped {n_flagged} sales:")
        for rule, n in rule_counts.items():
            print(f"    - {rule}: {n}")

    base = _attach_contamination_columns(
        base, universe=universe, parcel_key=parcel_key, join_key=sales_join_key,
        nbhd_median_size=nbhd_median_size, neighborhood_col=neighborhood_col,
        land_area_col=land_area_col, zoning_min_col=zoning_min_col,
    )
    for col in ("land_deferred_code", "historic_deferred_code"):
        if col in universe.columns:
            base[col] = base[sales_join_key].map(universe_keyed[col])
    base = _apply_baseline_filters(
        base, cfg=cfg, land_area_col=land_area_col,
        flag_col="disq_flag", log_prefix="W3 extraction", verbose=verbose,
    )

    out = pd.DataFrame({
        "parcel_key": base[sales_join_key].values,
        "witness_kind": "W3_extraction",
        "land_value": base["__land_residual__"].values,
        "land_area_sqft": base[land_area_col].values,
        "event_date": pd.to_datetime(base["sale_date"], errors="coerce").values,
        "weight": cfg.weights["W3_extraction"],
        "notes": f"extraction: sale_price - assr_impr_value (bldg <= {cfg.w3_max_age_years}y)",
    })
    out["land_value_per_sqft"] = out["land_value"] / out["land_area_sqft"]
    out.index = base.index
    return out


def curate_w4_low_far(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    parcel_key: str,
    sales_join_key: str,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
    nbhd_median_size: pd.Series,
    far_col: str = "floor_area_ratio",
    flagged_out: list | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    W4: improved sales with FAR in the bottom decile of their VCS — these
    are good extraction candidates because the building contributes
    relatively little to total value, so the cost-basis subtraction is
    less sensitive to errors in ``assr_impr_value``.

    Implements the same residual subtraction as W3
    (``sale_price − assr_impr_value``); the difference is W4's filter
    selects sales where the residual is dominated by land. (An earlier
    v1 of this function returned the full sale price as the land value,
    which double-counted the building. Cost-residual investigation
    surfaced the bug — see out/land/rung1/cost_residual_investigation.md.)
    """
    sales = sales.copy()
    if far_col not in sales.columns:
        if "bldg_area_finished_sqft" in sales.columns and land_area_col in sales.columns:
            sales[far_col] = (
                pd.to_numeric(sales["bldg_area_finished_sqft"], errors="coerce")
                / pd.to_numeric(sales[land_area_col], errors="coerce").replace(0, np.nan)
            )
        else:
            if verbose:
                print(f"  W4 low_far: no {far_col} and can't compute, skipping")
            return _empty_witness_frame()

    keep = (
        (sales["valid_sale"].fillna(False))
        & (~sales["vacant_sale"].fillna(False))
        & (sales["sale_price"].fillna(0) > 0)
        & (sales[land_area_col].fillna(0) > 0)
        & (sales[far_col].fillna(0) > 0)
    )
    base = sales[keep].copy()
    if len(base) == 0:
        return _empty_witness_frame()

    grp = base.groupby(neighborhood_col, dropna=False)[far_col]
    base["__far_p10__"] = grp.transform(lambda s: s.quantile(cfg.w4_far_percentile))
    base = base[base[far_col] <= base["__far_p10__"]]
    if len(base) == 0:
        return _empty_witness_frame()

    universe_keyed = universe.set_index(parcel_key)
    if "assr_impr_value" not in universe.columns:
        if verbose:
            print("  W4 low_far: universe lacks assr_impr_value, skipping")
        return _empty_witness_frame()
    base["__assr_impr_value__"] = base[sales_join_key].map(universe_keyed["assr_impr_value"])
    base["assr_impr_value"] = base["__assr_impr_value__"]
    for c in ("bldg_age_years", "bldg_condition_num", "prior_land_xfer_price",
              "prior_land_xfer_date"):
        if c in universe.columns and c not in base.columns:
            base[c] = base[sales_join_key].map(universe_keyed[c])
    sp = base.get("sale_price_time_adj", base["sale_price"])
    base["__land_residual__"] = sp - base["__assr_impr_value__"].fillna(0)
    base = base[base["__land_residual__"] > 0]
    if len(base) == 0:
        return _empty_witness_frame()

    # Anomaly filter — same rules as W3
    anom = evaluate_sale_anomaly_flags(base)
    base = base.join(anom)
    flagged_mask = base["anomaly_flags"].astype("string").str.len() > 0
    n_flagged = int(flagged_mask.fillna(False).sum())
    if n_flagged and flagged_out is not None:
        flagged_audit = base.loc[flagged_mask].copy()
        flagged_audit["__source_witness__"] = "W4_low_far"
        flagged_out.append(flagged_audit)
    base = base[~flagged_mask.fillna(False)]
    if verbose and n_flagged:
        rule_counts = (
            anom.loc[flagged_mask, "anomaly_flags"].str.split(";")
            .explode().value_counts()
        )
        print(f"  W4 low_far: anomaly filter dropped {n_flagged} sales:")
        for rule, n in rule_counts.items():
            print(f"    - {rule}: {n}")

    base = _attach_contamination_columns(
        base, universe=universe, parcel_key=parcel_key, join_key=sales_join_key,
        nbhd_median_size=nbhd_median_size, neighborhood_col=neighborhood_col,
        land_area_col=land_area_col, zoning_min_col=zoning_min_col,
    )
    for col in ("land_deferred_code", "historic_deferred_code"):
        if col in universe.columns:
            base[col] = base[sales_join_key].map(universe_keyed[col])
    base = _apply_baseline_filters(
        base, cfg=cfg, land_area_col=land_area_col,
        flag_col="disq_flag", log_prefix="W4 low_far", verbose=verbose,
    )

    out = pd.DataFrame({
        "parcel_key": base[sales_join_key].values,
        "witness_kind": "W4_low_far",
        "land_value": base["__land_residual__"].values,
        "land_area_sqft": base[land_area_col].values,
        "event_date": pd.to_datetime(base.get("sale_date"), errors="coerce").values,
        "weight": cfg.weights["W4_low_far"],
        "notes": f"low-FAR extraction (FAR <= local p{int(cfg.w4_far_percentile*100)})",
    })
    out["land_value_per_sqft"] = out["land_value"] / out["land_area_sqft"]
    return out


def curate_w5_prior_xfer(
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    parcel_key: str,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
    nbhd_median_size: pd.Series,
    valuation_date: pd.Timestamp,
    annual_growth_rate: float = 0.04,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    W5: per-parcel ``prior_land_xfer_price/date`` — the most recent
    vacant-land transfer recorded on the parcel itself. Time-adjust
    forward to ``valuation_date`` using a simple compounded annual
    growth rate (a placeholder for a proper price index).

    Records older than ``cfg.w5_max_age_years`` from valuation_date are
    dropped — too long a horizon makes the simple growth-rate adjustment
    unreliable.
    """
    if "prior_land_xfer_price" not in universe.columns:
        if verbose:
            print("  W5 prior_xfer: no prior_land_xfer_price, skipping")
        return _empty_witness_frame()
    if "prior_land_xfer_date" not in universe.columns:
        if verbose:
            print("  W5 prior_xfer: no prior_land_xfer_date, skipping")
        return _empty_witness_frame()

    base = universe.copy()
    price = pd.to_numeric(base["prior_land_xfer_price"], errors="coerce")
    date = pd.to_datetime(base["prior_land_xfer_date"], errors="coerce")
    keep = (
        (price > 0)
        & (date.notna())
        & (base[land_area_col].fillna(0) > 0)
    )
    base = base[keep].copy()
    if len(base) == 0:
        return _empty_witness_frame()

    age_years = (
        (valuation_date - pd.to_datetime(base["prior_land_xfer_date"]))
        .dt.days / 365.25
    )
    base = base[age_years <= cfg.w5_max_age_years]
    if len(base) == 0:
        return _empty_witness_frame()

    # Drop disq leakage on prior xfer flag
    if "prior_land_xfer_disq_flag" in base.columns:
        leaked = base["prior_land_xfer_disq_flag"].astype("string").isin(LEAKED_DISQ_FLAGS)
        base = base[~leaked.fillna(False)]

    # Time-adjust: simple compounded
    age_years = (
        (valuation_date - pd.to_datetime(base["prior_land_xfer_date"]))
        .dt.days / 365.25
    )
    factor = (1.0 + annual_growth_rate) ** age_years
    adj_price = pd.to_numeric(base["prior_land_xfer_price"], errors="coerce") * factor

    base = _attach_contamination_columns(
        base, universe=universe, parcel_key=parcel_key, join_key=parcel_key,
        nbhd_median_size=nbhd_median_size, neighborhood_col=neighborhood_col,
        land_area_col=land_area_col, zoning_min_col=zoning_min_col,
    )
    base = _apply_baseline_filters(
        base, cfg=cfg, land_area_col=land_area_col,
        flag_col=None, log_prefix="W5 prior_xfer", verbose=verbose,
    )
    # Realign factor/adj_price after any filter drops
    adj_price = adj_price.loc[base.index]

    out = pd.DataFrame({
        "parcel_key": base[parcel_key].values,
        "witness_kind": "W5_prior_xfer",
        "land_value": adj_price.values,
        "land_area_sqft": base[land_area_col].values,
        "event_date": pd.to_datetime(base["prior_land_xfer_date"]).values,
        "weight": cfg.weights["W5_prior_xfer"],
        "notes": f"prior land xfer, time-adj at {annual_growth_rate*100:.1f}%/yr",
    })
    out["land_value_per_sqft"] = out["land_value"] / out["land_area_sqft"]
    return out


def curate_w6_pred_residual(
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig,
    parcel_key: str,
    neighborhood_col: str,
    land_area_col: str,
    zoning_min_col: str,
    nbhd_median_size: pd.Series,
    prediction_col: str = "prediction",
    impr_value_col: str = "assr_impr_value",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    W6: full-universe synthetic land-value witness from the cost-residual
    of our ensemble market-value prediction.

    For every improved parcel, ``land_value = prediction − assr_impr_value``.
    Cost-residual investigation showed this is correlated with actual
    sale-residuals (R1 in the report) at 0.984 Spearman per neighborhood,
    so it's a reliable proxy at full coverage.

    Coverage: every improved parcel in the universe with positive
    prediction. This fills evidence gaps in cells where W1–W5 are thin.

    Caveat: W6 doesn't represent an actual transaction; it's a
    model-derived synthetic. It should NOT be used to validate the
    painter's calibration (that would be circular). Used as
    fallback evidence when the witness pool is sparse, and as a
    full-universe sanity check for the painter's own output.
    """
    if prediction_col not in universe.columns or impr_value_col not in universe.columns:
        if verbose:
            print(
                f"  W6 pred_residual: missing {prediction_col} or "
                f"{impr_value_col}, skipping"
            )
        return _empty_witness_frame()

    base = universe.copy()
    is_built = base.get("bldg_area_finished_sqft", pd.Series([], dtype=float)).fillna(0) > 0
    keep = (
        is_built
        & (base[prediction_col] > 0)
        & (base[impr_value_col].fillna(0) >= 0)
        & (base[land_area_col].fillna(0) > 0)
    )
    base = base[keep].copy()
    if len(base) == 0:
        return _empty_witness_frame()

    base["__land_residual__"] = base[prediction_col] - base[impr_value_col].fillna(0)
    base = base[base["__land_residual__"] > 0]
    if len(base) == 0:
        return _empty_witness_frame()

    base = _attach_contamination_columns(
        base, universe=universe, parcel_key=parcel_key, join_key=parcel_key,
        nbhd_median_size=nbhd_median_size, neighborhood_col=neighborhood_col,
        land_area_col=land_area_col, zoning_min_col=zoning_min_col,
    )
    base = _apply_baseline_filters(
        base, cfg=cfg, land_area_col=land_area_col,
        flag_col=None, log_prefix="W6 pred_residual", verbose=verbose,
    )

    out = pd.DataFrame({
        "parcel_key": base[parcel_key].values,
        "witness_kind": "W6_pred_residual",
        "land_value": base["__land_residual__"].values,
        "land_area_sqft": base[land_area_col].values,
        "event_date": pd.NaT,
        "weight": cfg.weights["W6_pred_residual"],
        "notes": "ensemble prediction minus depreciated cost",
    })
    out["land_value_per_sqft"] = out["land_value"] / out["land_area_sqft"]
    return out


def evaluate_sale_anomaly_flags(
    sales_with_universe: pd.DataFrame,
    *,
    sale_price_col: str = "sale_price_time_adj",
    impr_value_col: str = "assr_impr_value",
    bldg_age_col: str = "bldg_age_years",
    bldg_condition_col: str = "bldg_condition_num",
    bldg_area_col: str = "bldg_area_finished_sqft",
    prior_xfer_price_col: str = "prior_land_xfer_price",
    prior_xfer_date_col: str = "prior_land_xfer_date",
    sale_date_col: str = "sale_date",
    annual_growth_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Evaluate the four anomaly-flag rules for each row.

    Each rule corresponds to a physically or logically implausible sale
    pattern, not a statistical outlier. A row triggering ANY rule is an
    anomaly candidate; downstream code should drop it from calibration
    pools but log the firing rules to a flagged-sales artifact for
    audit / human review.

    Returns a DataFrame indexed identically to the input with one column:
    ``anomaly_flags`` — a semicolon-separated string of rule names that
    fired (empty string if none).
    """
    df = sales_with_universe
    out = pd.DataFrame(index=df.index)
    flags = pd.Series([""] * len(df), index=df.index, dtype="object")

    def _add_flag(mask: pd.Series, name: str) -> None:
        nonlocal flags
        mask = mask.fillna(False)
        flags.loc[mask] = flags.loc[mask].apply(
            lambda existing: f"{existing};{name}" if existing else name
        )

    sale = pd.to_numeric(df.get(sale_price_col), errors="coerce")
    impr = pd.to_numeric(df.get(impr_value_col), errors="coerce")
    age = pd.to_numeric(df.get(bldg_age_col), errors="coerce")
    cond = pd.to_numeric(df.get(bldg_condition_col), errors="coerce")
    area = pd.to_numeric(df.get(bldg_area_col), errors="coerce")
    prior_p = pd.to_numeric(df.get(prior_xfer_price_col), errors="coerce")
    prior_d = pd.to_datetime(df.get(prior_xfer_date_col), errors="coerce")
    sale_d = pd.to_datetime(df.get(sale_date_col), errors="coerce")

    # Rule 1: BELOW_COST_FLOOR
    rule_1 = (
        (sale < BELOW_COST_FLOOR_RATIO * impr)
        & (age < BELOW_COST_FLOOR_MAX_AGE_YEARS)
        & (cond >= BELOW_COST_FLOOR_MIN_CONDITION)
        & impr.notna()
    )
    _add_flag(rule_1, "BELOW_COST_FLOOR")

    # Rule 2: PRIOR_XFER_INCONSISTENT (time-adjusted)
    age_yrs = (sale_d - prior_d).dt.days / 365.25
    in_window = (age_yrs > 0) & (age_yrs <= PRIOR_XFER_LOOKBACK_YEARS)
    prior_adj = prior_p * ((1.0 + annual_growth_rate) ** age_yrs)
    rule_2 = (
        (prior_p > 0)
        & in_window
        & (sale < PRIOR_XFER_INCONSISTENT_RATIO * prior_adj)
    )
    _add_flag(rule_2, "PRIOR_XFER_INCONSISTENT")

    # Rule 3: TOKEN_PRICE
    rule_3 = sale < TOKEN_PRICE_THRESHOLD_USD
    _add_flag(rule_3, "TOKEN_PRICE")

    # Rule 4: BELOW_PHYSICAL_PSF (per-sqft of building below physical floor)
    psf = sale / area.replace(0, np.nan)
    rule_4 = (area > 0) & (psf < MIN_PHYSICAL_BUILDING_PSF_USD)
    _add_flag(rule_4, "BELOW_PHYSICAL_PSF")

    out["anomaly_flags"] = flags
    return out


def _empty_witness_frame() -> pd.DataFrame:
    """Return an empty witness DataFrame with the canonical schema."""
    return pd.DataFrame(
        columns=[
            "parcel_key",
            "witness_kind",
            "land_value",
            "land_area_sqft",
            "event_date",
            "weight",
            "notes",
            "land_value_per_sqft",
        ]
    )


def curate_witnesses(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: WitnessConfig | None = None,
    parcel_key: str = "key",
    sales_join_key: str = "key",
    neighborhood_col: str = "neighborhood",
    land_area_col: str = "land_area_sqft",
    zoning_min_col: str = "zoning_emp_min_lot_sqft",
    valuation_date: pd.Timestamp | None = None,
    flagged_out: list | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run all five witness streams and concatenate into a unified pool.

    Parameters
    ----------
    sales : pandas.DataFrame
        Cleaned sales with columns ``valid_sale``, ``vacant_sale``,
        ``sale_price``, ``sale_date``, ``land_area_sqft``,
        ``bldg_area_finished_sqft``, optional ``sale_price_time_adj``,
        ``disq_flag``, ``is_teardown``, ``floor_area_ratio``,
        ``bldg_year_built``.
    universe : pandas.DataFrame
        Parcel universe. Should already have empirical-zoning columns
        joined (typically by calling :func:`openavmkit.zoning.join_empirical_zoning`)
        and ``floor_area_ratio`` computed.
    cfg : WitnessConfig, optional
        Knobs. Defaults to ``WitnessConfig()``.
    parcel_key : str
        Parcel-key column name in ``universe``.
    sales_join_key : str
        Column in ``sales`` that joins to ``universe[parcel_key]``.
    neighborhood_col : str
        Column with the finest-grained neighborhood code.
    land_area_col : str
    zoning_min_col : str
        Empirical zoning-floor column on ``universe`` (typically
        ``"zoning_emp_min_lot_sqft"``).
    valuation_date : pandas.Timestamp, optional
        Defaults to ``universe`` settings or "now". Used by W5.
    verbose : bool

    Returns
    -------
    pandas.DataFrame
        Concatenated witnesses. Columns: ``parcel_key``, ``witness_kind``,
        ``land_value``, ``land_area_sqft``, ``land_value_per_sqft``,
        ``event_date``, ``weight``, ``notes``.
    """
    cfg = cfg or WitnessConfig()
    if valuation_date is None:
        valuation_date = pd.Timestamp.utcnow().tz_localize(None)

    nbhd_median_size = _compute_neighborhood_size_stats(
        universe, neighborhood_col=neighborhood_col, land_area_col=land_area_col
    )

    common_kwargs = dict(
        cfg=cfg,
        parcel_key=parcel_key,
        sales_join_key=sales_join_key,
        neighborhood_col=neighborhood_col,
        land_area_col=land_area_col,
        zoning_min_col=zoning_min_col,
        nbhd_median_size=nbhd_median_size,
        verbose=verbose,
    )

    if verbose:
        print("Curating witnesses:")

    parts = []
    parts.append(curate_w1_clean_vacant(sales, universe, **common_kwargs))
    parts.append(curate_w2_teardown(sales, universe, **common_kwargs))
    parts.append(curate_w3_extraction(sales, universe, flagged_out=flagged_out, **common_kwargs))
    parts.append(curate_w4_low_far(sales, universe, flagged_out=flagged_out, **common_kwargs))
    parts.append(
        curate_w5_prior_xfer(
            universe,
            cfg=cfg,
            parcel_key=parcel_key,
            neighborhood_col=neighborhood_col,
            land_area_col=land_area_col,
            zoning_min_col=zoning_min_col,
            nbhd_median_size=nbhd_median_size,
            valuation_date=valuation_date,
            verbose=verbose,
        )
    )
    parts.append(
        curate_w6_pred_residual(
            universe,
            cfg=cfg,
            parcel_key=parcel_key,
            neighborhood_col=neighborhood_col,
            land_area_col=land_area_col,
            zoning_min_col=zoning_min_col,
            nbhd_median_size=nbhd_median_size,
            verbose=verbose,
        )
    )

    out = pd.concat(parts, ignore_index=True)
    if verbose:
        summary = out.groupby("witness_kind").size()
        print("\nWitness pool by kind:")
        for kind, n in summary.items():
            print(f"  {kind:18s} {n:>7,}")
        print(f"  {'TOTAL':18s} {len(out):>7,}")

    return out
