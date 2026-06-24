# `openavmkit.lvi` — Land Value Integrity

A **config-driven, method-agnostic battery that scores an existing per-parcel land series** — "are
these land values good?" — and produces a comparative scorecard plus per-parcel evidence packets
defensible in a tax protest. It only *validates* a land series; it does not create one.

> Separate by design from `openavmkit.land` (the land *creation* / painter system). Reconcile
> later — see `research/land_reconciliation_map.md`.

## Principle: the pipeline rules, LVI tests

LVI does **no classification of its own** — no zoning regex, no deed-code lists, no bespoke
filtering. *Which* sales count as evidence is decided by **filters you own**, resolved by
`openavmkit.filters.resolve_filter` (the same engine model groups and `valid_sale` use). The only
built-in screen is `prime_comp` (size-comparability to neighborhood built peers + shape) — a
general, vocabulary-free helper that can't be a static filter; it's tunable and disable-able.

## Configuration (settings.json)

`land_value_integrity` is a **map keyed by model group**; groups not listed are skipped. Every key
is optional and falls back to a Wake/IAAO default (see `config.py`).

```jsonc
"land_value_integrity": {
  "single_family": {
    "land_evidence_filter": ["and",
      ["==","vacant_sale",true], ["==","valid_for_land_ratio_study",true],
      ["isin","disq_flag",["A","C"]]],          // A3/A5 gold gate — you own what "good" means
    "cost_residual_filter": ["and", ["==","vacant_sale",false], [">=","bldg_condition_pct",95]],
    "evidence": { "frozen_sov": true },          // dep~0 cost-residuals join the gold
    "fields": { "land_value":"land_value", "cost_bldg_value":"assr_impr_value" },  // override only what differs
    "verdicts": { "a3_cod":[15,25] }             // IAAO defaults; override to retune
  }
  // "multifamily": { "land_evidence_filter": [...] }, ...
}
```

A second model group is often a one-line `land_evidence_filter`. An optional `__defaults` block is
merged under every group.

## Quick start

```python
from openavmkit.lvi import run_land_value_integrity, load_lvi_configs
cfg = load_lvi_configs(settings)["single_family"]
# each universe carries land_value/impr_value/total_value (the series under test) + features
res = run_land_value_integrity({"assessor": assessor_universe}, sales, cfg)
print(res.scorecard())
res.packets["assessor"].to_csv("evidence.csv", index=False)
# res.diagnostics (A0, depreciation), res.support (coverage, propagation, differentials)
```

Driver / CLI (loops every configured group for a jurisdiction):

```bash
python -m openavmkit.lvi.run                       # default: notebooks/pipeline/data/us-nc-wake
python -m openavmkit.lvi.run path/to/jurisdiction  # writes out/lvi/scorecard_<group>.txt, evidence_<group>_<series>.csv
```

## The battery

| | test | what it checks |
|---|---|---|
| **Step 1** | total ratio study | total values are good (precondition) |
| **A1** | improvement-independence | land $/sqft uncorrelated with what's built |
| **A2** | uniformity | like-land→like land value; like-building→like building value |
| **A3** | land vs anchors (gold) | land vs `land_evidence_filter` evidence (+ cost-residuals if `frozen_sov`) |
| **A5** | desirability gradient | land tracks the market land gradient |
| **A6** | sales-chasing | land not silently set to the sale price |
| **A7** | summation & sanity | land+impr=total; bounds |
| **A8** | building location-invariance | matched buildings → same building value across locations |
| **B3** | local spatial uniformity | land $/sqft locally smooth |
| **VE** | vertical equity | regressivity (VEI) on the direct evidence |

Diagnostics (once): **A0** unit selection ($/sqft vs $/lot), **depreciation** schedule calibration.
Coverage (`support.py`): **B1** support map, **B1b** level-propagation + per-parcel evidence chains,
**Tier-2** composition-controlled location premiums.

## Modules

`config.py` (GroupConfig + `load_lvi_configs`) · `evidence.py` (filter-driven streams +
`prime_comp` + `reconstruct_rcn`) · `battery.py` (per-series tests + A0/depreciation diagnostics) ·
`support.py` (coverage / propagation / differentials) · `report.py` (scorecard + evidence_packet) ·
`run.py` (driver / CLI).

## Multi-jurisdiction

Adding a county = onboard it to the openavmkit pipeline (canonical columns) + write its
`land_value_integrity` block (the filters for its qualified/relevant sales, and `frozen_sov`). See
`research/lvi_settings_example.json` for a fully-commented reference and a minimal Florida example.

## Methodology

`research/land_value_integrity_spec.md` (§10-21) and `research/land_value_integrity_RESULTS.md`.
