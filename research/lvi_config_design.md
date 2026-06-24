# LVI config design — making `openavmkit.lvi` jurisdiction-agnostic

*Proposal for review. No code written yet. Goal: lift every Wake-specific constant out of the
code into one settings-driven config, with Wake values as the defaults so current behavior is
byte-for-byte unchanged until a jurisdiction overrides.*

## Principles

1. **Defaults = Wake.** `LVIConfig()` with no args reproduces today's results exactly.
2. **Settings-driven.** `LVIConfig.from_settings(settings)` reads a new `land_value_integrity`
   block in a jurisdiction's `settings.json` (the existing openavmkit convention). Precedence:
   jurisdiction settings > config defaults.
3. **Minimal API churn.** One `cfg` object threads from `run_land_value_integrity(...)` down to
   evidence/battery/support; every function takes `cfg=None` → `cfg or LVIConfig()`.
4. **Two kinds of knob, kept separate:** *column mappings* (which the pipeline already normalizes,
   so defaults rarely change) vs *vocabularies & thresholds* (genuinely per-jurisdiction).
5. **Canonical names stay assumed where the pipeline guarantees them** (he_id clusters, lat/long)
   — configurable but not something a normal onboarding touches.

## The config object (nested dataclasses, `openavmkit/lvi/config.py`)

```python
@dataclass
class ColumnMap:
    # identity / location
    key="key"; neighborhood="neighborhood_filled"; model_group="model_group"
    latitude="latitude"; longitude="longitude"
    land_he_id="land_he_id"; impr_he_id="impr_he_id"
    # series under test (per-series universe carries these three)
    land_value="land_value"; impr_value="impr_value"; total_value="total_value"
    # cost basis (residuals / RCN / depreciation)
    cost_bldg_value="assr_impr_value"      # RCNLD from the cost book
    pctgood="bldg_condition_pct"           # percent-good; None => RCN streams degrade
    # physical
    land_area="land_area_sqft"; bldg_area="bldg_area_finished_sqft"
    rectangularity="geom_rectangularity_num"; is_vacant="is_vacant"
    bldg_year_built="bldg_year_built"
    # sales
    sale_price="sale_price"; sale_price_time_adj="sale_price_time_adj"
    sale_date="sale_date"; sale_age_days="sale_age_days"
    vacant_sale="vacant_sale"; qualification="disq_flag"
    valid_sale="valid_sale"; valid_for_ratio_study="valid_for_ratio_study"
    valid_for_land_ratio_study="valid_for_land_ratio_study"
    # A1 improvement features
    impr_feats=("bldg_area_finished_sqft","bldg_age_years","bldg_quality_num","bldg_condition_num")

@dataclass
class Vocab:
    qualified_sale_codes=("A","C")                 # WHITELIST (Wake Disq_and_Qual A/C)
    residential_model_groups=("single_family","multifamily","apartment","UNKNOWN")
    # zoning classifier — see "the one real fork" below
    classify_field="zoning"                        # column to classify on
    classify_mode="regex"                          # "regex" | "set"
    residential_codes=()                           # used when mode="set"
    zoning_regex=DEFAULT_WAKE_ZONING_RULES         # ordered [(bucket,[patterns])], when mode="regex"

@dataclass
class PrimeLot:
    rect_min=0.40; min_peers=10; size_lo=0.05; size_hi=0.95

@dataclass
class EvidenceCfg:
    frozen_sov=True                                # promote dep~0 RCN residuals into A3 gold
    rcn_pctgood_min=0.95                           # dep~0 gate for rcn_resid
    rcnld_pctgood_lo=0.75; rcnld_pctgood_hi=0.95   # rcnld_resid band
    teardown_frac=0.50                             # price/land-sqft ceiling vs developed level
    vacant_price_floor=2000
    pctgood_clip=(0.10,1.20)                       # RCN reconstruction guardrail

@dataclass
class Verdicts:                                    # (pass_at, warn_at); IAAO defaults
    total_cod=(15,25); a1_partial_r2=(0.05,0.15); a2_chd=(15,25); a3_cod=(15,25)
    a5_rho=(0.6,0.4); a7_pct=(1.0,5.0); a8_chd=(15,25); b3_hot=(0.05,0.10)
    ve_vei_abs=(10,25); depreciation_drift50=(5,15)

@dataclass
class SupportCfg:
    bldg_psf_bounds=(20,600); min_anchor_sales=3; distance_bands_mi=(0.5,1.0,2.0)

@dataclass
class ConfidenceCfg:                              # per-parcel evidence-packet grading
    residual_band=(0.6,1.6); anchor_band=(0.6,1.6); cluster_band=(0.5,2.0); min_anchor_n=3

@dataclass
class LVIConfig:
    columns: ColumnMap = ColumnMap()
    vocab: Vocab = Vocab()
    prime: PrimeLot = PrimeLot()
    evidence: EvidenceCfg = EvidenceCfg()
    verdicts: Verdicts = Verdicts()
    support: SupportCfg = SupportCfg()
    confidence: ConfidenceCfg = ConfidenceCfg()

    @classmethod
    def from_settings(cls, settings: dict) -> "LVIConfig": ...   # reads settings["land_value_integrity"]
```

## The one real fork — the zoning classifier

`classify_zoning` is the only piece that can't be solved by column-renaming. Two supported modes:

- **`mode="set"` (recommended for most, incl. FL Lee):** `classify_field` points at a clean code
  column (e.g. FL DOR **use code**), `residential_codes` is the explicit allowlist; everything
  else is non-residential. Simple, exhaustive, auditable.
- **`mode="regex"` (Wake default):** `zoning_regex` is an ordered `[(bucket, [patterns])]` list
  evaluated top-down (rural → nonres → residential → other) — needed where zoning codes are messy
  (Raleigh's 341-code UDO+county mix). Default = today's Wake patterns, verbatim.

Recommendation: keep Wake on `regex` (defaults), onboard FL Lee on `set` against DOR use codes.

## What this maps to in `settings.json`

```json
"land_value_integrity": {
  "model_group": "single_family",
  "columns": { "qualification": "DOR_QUAL_CODE", "pctgood": "PCT_GOOD",
               "cost_bldg_value": "BLDG_JUST_VALUE", "land_value": "LAND_JUST_VALUE" },
  "vocab": { "qualified_sale_codes": ["01","02","03"], "classify_mode": "set",
             "classify_field": "dor_use_code", "residential_codes": ["0100","0200"] },
  "evidence": { "frozen_sov": false },
  "verdicts": { "a3_cod": [15, 25] }
}
```

Only the keys a jurisdiction overrides appear; everything else falls back to the Wake/IAAO defaults.

## Threading through the code (mechanical, low-risk)

- `run_land_value_integrity(..., cfg=None)` builds `cfg = cfg or LVIConfig()` and passes it down.
- `evidence.*`, `battery.*`, `support.*`, `report.evidence_packet` gain `cfg=None`; they read
  `cfg.columns.X` / `cfg.verdicts.Y` instead of the literals. `classify_zoning(code, cfg)`.
- `run.py` gains `LVIConfig.from_settings(load_settings("in/settings.json"))`.
- Backward-compatible: existing call sites that pass no cfg get Wake behavior.

## What stays hard-coded (and why)

- The **test math** and the **non-circularity discipline** — methodology, not jurisdiction.
- **he_id cluster names / lat-long** — produced by the openavmkit pipeline for every jurisdiction;
  configurable via `ColumnMap` but not normally touched.
- The **IAAO threshold *defaults*** — overridable via `Verdicts`, but the defaults are the standard.

## Open questions for review

1. **Verdict thresholds as config?** I propose yes (with IAAO defaults) so standards/use-cases can
   tune and the pass/warn/fail is auditable. Push back if you'd rather freeze them.
2. **Zoning fork** — OK to support both `set` and `regex`, Wake stays `regex`, Lee uses `set`?
3. **Config granularity** — one flat `land_value_integrity` block (as above) vs nesting under each
   model group? (Wake is single-group; multi-group jurisdictions may want per-group overrides.)
4. **Degradation policy** — when `pctgood`/`qualification` columns are absent, auto-degrade
   silently (vacant-only gold) with a logged note, or require an explicit opt-in flag?
5. **Where defaults live** — in `config.py` dataclass defaults (proposed), or mirrored into a
   shipped `resources/settings/lvi.defaults.json`? Dataclass is simpler; JSON is more discoverable.
```
