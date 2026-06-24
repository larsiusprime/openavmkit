"""Settings-driven configuration for the LVI battery.

`land_value_integrity` in settings.json is a MAP KEYED BY MODEL-GROUP NAME; groups not listed are
skipped. Each group's "which sales count" decisions are FILTERS resolved by
``openavmkit.filters.resolve_filter`` (the same engine model groups / valid_sale use) — LVI does no
classification of its own. The only built-in, jurisdiction-agnostic screen is ``prime_comp``
(size-comparability to neighborhood built peers + shape), which can't be a static filter (the bound
is a per-neighborhood percentile) and carries no vocabulary; it is tunable and disable-able.

Every key is optional; defaults below reproduce the Wake battery. An optional ``__defaults`` block
is merged under every group before its own overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as _dc_fields


# ----- canonical column names (the pipeline produces these; override only if different) -----
@dataclass
class Fields:
    key: str = "key"
    neighborhood: str = "neighborhood_filled"
    model_group: str = "model_group"
    latitude: str = "latitude"
    longitude: str = "longitude"
    land_he_id: str = "land_he_id"
    impr_he_id: str = "impr_he_id"
    land_value: str = "land_value"
    impr_value: str = "impr_value"
    total_value: str = "total_value"
    cost_bldg_value: str = "assr_impr_value"     # cost-book RCNLD (for cost residuals / RCN / depreciation)
    pctgood: str | None = "bldg_condition_pct"   # percent-good; None => RCN-based diagnostics degrade
    land_area: str = "land_area_sqft"
    bldg_area: str = "bldg_area_finished_sqft"
    rectangularity: str = "geom_rectangularity_num"
    is_vacant: str = "is_vacant"
    bldg_year_built: str = "bldg_year_built"
    sale_price: str = "sale_price"
    sale_price_time_adj: str = "sale_price_time_adj"
    sale_date: str = "sale_date"
    sale_age_days: str = "sale_age_days"
    impr_feats: tuple = ("bldg_area_finished_sqft", "bldg_age_years", "bldg_quality_num", "bldg_condition_num")


@dataclass
class PrimeComp:
    """General, jurisdiction-agnostic comparability screen on the direct-evidence stream."""
    enabled: bool = True
    rect_min: float = 0.40       # geom_rectangularity below this = irregular, dropped
    min_peers: int = 10          # neighborhood needs this many built peers to define a size band
    size_lo: float = 0.05        # lot must sit within [size_lo, size_hi] of built-peer land-area
    size_hi: float = 0.95


@dataclass
class Evidence:
    frozen_sov: bool = True      # promote cost-residual stream into the A3 gold standard


@dataclass
class Verdicts:                  # (pass_at, warn_at); IAAO defaults
    total_cod: tuple = (15, 25)
    a1_partial_r2: tuple = (0.05, 0.15)
    a2_chd: tuple = (15, 25)
    a3_cod: tuple = (15, 25)
    a5_rho: tuple = (0.6, 0.4)
    a7_pct: tuple = (1.0, 5.0)
    a8_chd: tuple = (15, 25)
    b3_hot: tuple = (0.05, 0.10)
    ve_vei_abs: tuple = (10, 25)
    depreciation_drift50: tuple = (5, 15)


@dataclass
class Support:
    bldg_psf_bounds: tuple = (20, 600)
    min_anchor_sales: int = 3
    distance_bands_mi: tuple = (0.5, 1.0, 2.0)


@dataclass
class Confidence:
    residual_band: tuple = (0.6, 1.6)
    anchor_band: tuple = (0.6, 1.6)
    cluster_band: tuple = (0.5, 2.0)
    min_anchor_n: int = 3


@dataclass
class GroupConfig:
    model_group: str
    land_evidence_filter: list                          # required: gold-standard direct land evidence
    total_sale_filter: list = field(default_factory=lambda: ["==", "valid_for_ratio_study", True])
    cost_residual_filter: list | None = None            # optional: improved sales for the cost residual
    fields: Fields = field(default_factory=Fields)
    prime: PrimeComp = field(default_factory=PrimeComp)
    evidence: Evidence = field(default_factory=Evidence)
    verdicts: Verdicts = field(default_factory=Verdicts)
    support: Support = field(default_factory=Support)
    confidence: Confidence = field(default_factory=Confidence)
    dep: str = "sale_price_time_adj"


def _sub(cls, raw):
    """Build a small dataclass from a dict, ignoring unknown / __comment keys."""
    if raw is None:
        return cls()
    known = {f.name for f in _dc_fields(cls)}
    kw = {k: v for k, v in raw.items() if k in known}
    # tuples come through JSON as lists; coerce pair/threshold fields back to tuple
    for f in _dc_fields(cls):
        if isinstance(f.default, tuple) and f.name in kw and isinstance(kw[f.name], list):
            kw[f.name] = tuple(kw[f.name])
    return cls(**kw)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _group_from_dict(model_group: str, raw: dict) -> GroupConfig:
    if "land_evidence_filter" not in raw:
        raise ValueError(
            f"land_value_integrity['{model_group}'] needs a 'land_evidence_filter' "
            f"(e.g. [\"==\", \"valid_for_land_ratio_study\", true]).")
    fld = _sub(Fields, raw.get("fields"))
    if "fields" in raw and "impr_feats" in raw["fields"]:
        fld.impr_feats = tuple(raw["fields"]["impr_feats"])
    return GroupConfig(
        model_group=model_group,
        land_evidence_filter=raw["land_evidence_filter"],
        total_sale_filter=raw.get("total_sale_filter", ["==", "valid_for_ratio_study", True]),
        cost_residual_filter=raw.get("cost_residual_filter"),
        fields=fld,
        prime=_sub(PrimeComp, raw.get("prime")),
        evidence=_sub(Evidence, raw.get("evidence")),
        verdicts=_sub(Verdicts, raw.get("verdicts")),
        support=_sub(Support, raw.get("support")),
        confidence=_sub(Confidence, raw.get("confidence")),
        dep=raw.get("dep", fld.sale_price_time_adj),
    )


def load_lvi_configs(settings: dict) -> dict[str, GroupConfig]:
    """Parse settings['land_value_integrity'] into {model_group: GroupConfig}. Groups not present
    are skipped (no entry => not scored). Keys beginning with '__' are reserved (e.g. __defaults)."""
    block = settings.get("land_value_integrity", {}) or {}
    defaults = block.get("__defaults", {})
    out = {}
    for mg, raw in block.items():
        if mg.startswith("__"):
            continue
        out[mg] = _group_from_dict(mg, _deep_merge(defaults, raw))
    return out
