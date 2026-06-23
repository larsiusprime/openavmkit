# Land Value Integrity Tests — Specification

**Status:** Phase 1 — specification / framework (no implementation yet)
**Scope:** A method-agnostic battery of tests that answers *"Are these land values good?"*
**Eventual home:** implemented as `openavmkit/land_value_integrity.py` + a
`resources/reports/land_value_integrity.md` report template, mirroring
`openavmkit/market_basket_validate.py`.

---

## 1. Context & motivation

We can now produce per-parcel land values — primarily via the market-basket / Fasteen
counterfactual in `openavmkit/market_basket.py` — but we have no principled, repeatable way to
answer the question a stakeholder actually asks: **"Are these land values good?"**

"Good" has three distinct meanings, and a defensible answer has to address all three:

1. **Correct economic incentives** (would these land values, used as an LVT base, produce the
   right incentives?).
2. **Survivability under tax protest** (are they evidence-based, explainable, market-anchored,
   and built on accepted IAAO practice?).
3. **Common-sense plausibility** (unbuildable < buildable; locally uniform; spikes where
   theory predicts).

Today we have only scattered checks in `market_basket_validate.py`, all tied to one valuation
method. This spec defines a fixed battery we can run against **any** per-parcel land series —
our market-basket output, the **assessor's published land split**, or any alternative — and
get back a comparable **scorecard**. The most useful answer to "are these good?" is
comparative: run the same battery on two land series and read the contrast.

This document is the source of truth that the eventual code will implement. Each test is
defined rigorously enough (formula, data required, reused utility, thresholds, objectivity
rating, per-parcel flag) to be reviewed against IAAO standards before any code is written.

### Reference grounding

- **IAAO Course 201 — Appraisal of Land (2025)**, especially Ch 4 (single-property methods),
  Ch 5 (mass-appraisal statistics), Ch 6 (mass-appraisal methods, depth/shape/corner/size
  adjustments, quality control), Ch 7 (soils), and the Glossary.
- **IAAO Standard on Ratio Studies (2010)** for level / uniformity / vertical-equity
  thresholds (reproduced in Course 201 Ch 5–6).
- Existing in-repo machinery: `utilities/stats.py`, `horizontal_equity_study.py`,
  `vertical_equity_study.py`, `ratio_study.py`, `sales_chasing.py`, `area_stats.py`,
  `market_basket.py`, `market_basket_validate.py`.

---

## 2. Cross-cutting principles

These four principles shape every test and should be read before the battery.

### P1 — Analyze land value *per unit*, and choose the unit empirically (see A0)

Raw (total) land value confounds lot size: a big lot and a small lot side by side *should*
have different total land values. Every test normalizes to a per-unit rate. **The unit is not
assumed — it is selected per area / model group by test A0.** The five IAAO units of
comparison:

| Unit | Typical fit | Field(s) |
|------|-------------|----------|
| **$/front-foot** | commercial / waterfront; depth-adjusted | `frontage_ft_1`, `land_area_somers_ft` |
| **$/sqft** | most urban land | `land_area_sqft` |
| **$/acre** | industrial / agricultural / rural | `land_area_sqft` → acres |
| **$/lot (site value)** | subdivision lots where size doesn't move price | lot count |
| **$/buildable-unit** | multifamily / zoned-density | zoning density × area |

Per IAAO the optimal unit is the one with the **least variation** across land sales (Glossary,
"Units of Comparison": *"the optimal unit of comparison… is typically the one with the lowest
coefficient of variation"*; the procedure is Course 201 Ch 6 Practical Application 6-1). Every
downstream per-unit test reports in the unit A0 selected for that area, with **$/sqft as the
universal fallback**.

### P2 — Land value = highest-and-best-use, as though vacant

Per IAAO, land is valued at its **highest and best use as if vacant**, applying the four HBU
tests in order: *legally permissible → physically possible → financially feasible → maximally
productive*. This is the conceptual basis of the whole battery:

- Market-basket's archetype is the "typical improvement at HBU." Its **archetype-divergence**
  diagnostic (`market_basket_validate.archetype_divergence`) is exactly the *current-use vs
  HBU gap* — a parcel whose land value swings widely across low/mid/high archetypes is
  HABU-sensitive and should be flagged.
- The **consistent-use** principle forbids valuing the land on one use basis and the building
  on another. Flag parcels where the implied land use ≠ the improvement's use.

### P3 — "Uniformity" is always *conditional*

Definition 3b ("locally uniform") and definition 3c ("spikes at waterfront / frontage") only
coexist if uniformity is measured **after** controlling for the legitimate land-value drivers
IAAO enumerates (Course 201 Ch 4 §I.B, Ch 5 §III.C). Before any uniformity test, condition on
or adjust for:

- location / neighborhood,
- **frontage & depth** — depth tables / **4-3-2-1 rule** (front-foot value rises with depth but
  not proportionately),
- **shape** — **65-35 rule** for triangles/irregulars,
- **corner influence**,
- **size economies** — $/sqft *declines* with parcel size (Ch 6 Practical App 6-6); a
  size-adjustment factor is required *before* clustering or neighbors will be flagged
  spuriously,
- topography / slope, soil & drainage,
- view / waterfront / golf-frontage.

Uniformity tests cluster on these drivers; spike tests verify the drivers are actually
reflected in the values.

### P4 — Method-agnostic input contract

The battery consumes a generic land series, so it scores the assessor's roll identically to
our market-basket output:

```text
run_land_value_integrity(
    land:         per-parcel Series/DataFrame keyed by `key`, column `land_value`
                  (+ optional `land_value_raw`),
    sup:          SalesUniversePair (universe + sales) — features and anchors,
    settings:     dict,
    model_group:  str,
    land_compare: optional second land series (e.g. assessor) for side-by-side scoring,
) -> LandValueIntegrityResult
```

No dependence on market-basket internals.

---

## 3. Input data (verified against the cleaned `SalesUniversePair`)

`key`, `land_value`, `land_value_raw`, `land_area_sqft` / `_sqm`, `land_area_somers_ft`,
`is_vacant`, `vacant_sale`, `valid_sale`, `sale_price` / `sale_price_time_adj`, `sale_date`,
`sale_age_days`, building fields (`bldg_area_finished_sqft`, age, quality), `neighborhood` /
other locations, `census_tract`, `he_id` / `land_he_id` / `impr_he_id`, `latitude` /
`longitude`, `spatial_lag_sale_price*` *(present only where enriched — absent in Guilford; see
§7)*, DEM (**`slope_mean`, `slope_stdev`, `elevation_mean`, `elevation_stdev`** — note these
names, not `_deg`/`_ft`), `frontage_ft_1..4` / `depth_ft_1..4` + `osm_frontage_{class}_ft`, and
the assessor's published land/total split — **`assr_land_value`** (the `land_compare` series),
**`assr_impr_value`**, **`assr_market_value`** (total). Area unit via `area_unit(settings)`.

---

## 4. The battery

Every test specifies: **definition → metric/formula → data required → reused utility →
pass/warn/fail thresholds → objectivity rating → per-parcel flag.** Tiered so the objective
core leads; Tier B is more data-dependent and partly interpretive.

### Tier A — Objective core (economic incentives + market-anchored IAAO)

---

#### A0 — Land-unit appropriateness *(foundational; runs first, per area / "split allocation")*

**Definition.** Determine *which* of the five IAAO units best fits each area / model group,
rather than assuming one. (IAAO calls per-neighborhood land units **"split allocations."**)

**Metric / formula.**
- **Step 1 — normalized-range pre-screen** (the selector taught in Course 201). For each
  candidate unit, compute the per-unit rate for every land sale (vacant + teardown-implied) in
  the area, then the **normalized range**:

  ```text
  normalized_range(unit) = (max_rate − min_rate) / min_rate
  ```

  Pick the unit with the **smallest** normalized range (least variation), e.g. compare
  `(maxFF − minFF)/minFF` vs `(maxSQFT − minSQFT)/minSQFT` and take the smaller.
- **Step 2 — COD / COV confirmation.** Confirm with the more robust statistic — lowest **COD**
  (median-rate ratio study) and/or lowest **COV** — since the normalized range is sensitive to
  a single outlier. Flag when Step 1 and Step 2 disagree (the choice is outlier-driven).

**Frontage caveat (when front-foot wins — partly manual).** "Front-foot" is only meaningful
against the *value-relevant* frontage, and a parcel can have more than one. If front-foot is
selected, the procedure must identify **which** frontage drives value for that use:

- waterfront / lake / river lots → **water frontage**, not road frontage;
- commercial / retail → **commercial street frontage** (and the *higher-value* street on a
  corner);
- golf-course lots → **golf-course frontage**.

The repo exposes `frontage_ft_1..4` / `depth_ft_1..4` (a parcel genuinely can have several
frontages) **and** OSM frontage *by road class*: `osm_frontage_{motorway, trunk, primary,
secondary, tertiary, residential, service, unclassified}_ft` + `osm_total_frontage_ft`
(confirmed present in Guilford). So frontage-*type* selection is **partly automatable**:

- **Street frontage by class is derivable** from the `osm_frontage_*` family — e.g. commercial
  / retail front-foot can key off arterial classes (primary/secondary/trunk) and pick the
  higher-value street on a corner.
- **Waterfront / golf / view frontage is not** captured by the OSM road classes — those remain
  a **manual review / data-gap item** (need a water/feature-distance layer) unless such a layer
  exists for the jurisdiction.

The spec must record *which* frontage measurement maps to value per model group. A front-foot
unit chosen against the wrong frontage will produce a low normalized range yet still be wrong
— so this frontage-type confirmation **gates** A0's front-foot result.

**Depth note (front-foot rates must be depth-adjusted).** Front-foot value rises with lot
depth but *not* proportionately (principle of contribution), so raw $/front-foot is not
comparable across lots of different depth — it must be normalized to a **standard depth** via
a **depth table** (Course 201 Ch 6 §V.A). A depth table is a schedule of **depth factors** =
`$/front-foot at depth D ÷ $/front-foot at the standard depth` (the standard-depth factor =
1.000; deeper > 1, shallower < 1).

- **Empirically derive the depth table from sales** (Course 201 Ch 6 Practical App 6-4 /
  Exercise 6-4) when enough vacant land sales of varying depth exist:
  1. compute $/front-foot for each vacant land sale,
  2. plot $/front-foot against depth and fit the curve,
  3. set the standard-depth value (1.000), and
  4. for each depth benchmark, `depth_factor = $/FF(depth) ÷ $/FF(standard depth)`, interpolating
     and rounding.
- **Fallback — the 4-3-2-1 rule of thumb** (Glossary; Ch 6 §V.A) when sales are too thin to
  derive a table: the front quarter of depth holds **40%** of lot value, the next **30%**, the
  third **20%**, the rear **10%**.
- The repo's **`land_area_somers_ft`** is effectively a frontage×depth-normalized measure (the
  Somers depth-table approach); an empirically-derived depth table is how you'd *construct or
  validate* that normalization for the area. The spec should record which depth table /
  standard depth applies per model group, and treat a missing/assumed table as a data-gap item.

**Data.** Land sales; `land_area_sqft`, `land_area_somers_ft`, `frontage_ft_1` (with
frontage-type confirmation, see caveat) and `depth_ft_1` (for depth adjustment), lot count;
zoning density for buildable-unit.

**Reused utility.** `calc_cod`, `calc_ratio_stats_bootstrap` (`utilities/stats.py`); the
ratio-study pattern from `market_basket_validate.ratio_study_vs_anchors`.

**Thresholds.** Report winning unit + margin over runner-up. *Warn* if the normalized ranges
of the top two units are within ~10% (ambiguous); *warn* if Step 1 ≠ Step 2.

**Objectivity.** Objective. **Underpins all per-unit tests (A1–A5, A7, B2–B4).**

**Per-parcel flag.** none (area-level result), but the selected unit is attached to every
parcel for downstream normalization.

> Expected pattern: dense urban → front-foot / site; suburban → per-lot or sqft; rural /
> large-acreage → per-acre; multifamily → buildable-unit.

---

#### A1 — Improvement independence *(cornerstone)*

**Definition.** Land value per unit must be **uncorrelated with what is built on the lot**,
after controlling for location and lot. Folds definitions 1a (identical lots, different
buildings → same land value) and 1b (improving your property doesn't raise your land bill).

**Metric / formula.** Two complementary views:
1. Regress land $/unit on improvement features (`bldg_area_finished_sqft`, improvement value,
   age, quality) **with a location control**: neighborhood fixed effects (`neighborhood_filled`)
   as the primary/portable control, or a `spatial_lag` control where enriched. *(spatial_lag is
   absent in Guilford — §7 — so neighborhood FE is the default.)* The partial effect / partial
   R² of the improvement block should be ≈ 0.
2. Within `land_he_id` clusters (which group on land attributes), correlation of land $/unit
   vs building $/unit should be ≈ 0.

**Data.** `land_value`, improvement fields, `neighborhood`/locations, `land_he_id`,
`spatial_lag_sale_price`.

**Reused utility.** `land_he_id` from `horizontal_equity_study.py`; standard regression.

**Thresholds.** *Pass* partial-R²(improvements) ≲ 0.05; *warn* 0.05–0.15; *fail* > 0.15
(values pending review against in-sample noise on Guilford).

**Objectivity.** Objective; most diagnostic test in the battery.

**Per-parcel flag.** `flag_impr_dependent` for parcels whose land value is best explained by
their improvement.

> **Named villain.** IAAO's **allocation (land-ratio) method** sets land = a fixed % of total,
> making land mechanically proportional to total (hence to the improvement). An assessor roll
> built that way fails A1 hard — this is the headline market-basket-vs-assessor contrast.

---

#### A2 — Horizontal land equity

**Definition.** Identically-featured lots pay the same: land $/unit should be uniform within
clusters defined on **land** attributes (def 1a).

**Metric / formula.** Coefficient of Horizontal Dispersion (CHD) of land $/unit within
`land_he_id` clusters, summarized as the median CHD across clusters. **Size-adjust first**
(P3) so size economies don't inflate CHD.

**Data.** `land_value`, `land_he_id`, `land_area_*`.

**Reused utility.** `calc_chds` / `quick_median_chd_pl` (`utilities/stats.py`),
`HorizontalEquityStudy` (`horizontal_equity_study.py`).

**Thresholds.** Treat as a land-uniformity COD; reuse the A3 vacant-land COD bands as a guide.

**Objectivity.** Objective; IAAO-standard.

**Per-parcel flag.** `flag_he_outlier` for parcels far from their cluster median.

---

#### A3 — Land ratio study vs vacant-land sales *(gold standard)*

**Definition.** Compare predicted land to observed land-sale prices — the most defensible,
IAAO-blessed check (def 2).

**Metric / formula.** `ratio = predicted_land / land_sale_price`, then **median ratio**
(level), **COD** (uniformity). **Vertical equity is measured by VEI + decile charts only — no
PRD/PRB** (see note). Teardown-implied and new-construction-implied land as secondary anchors.

**Data.** valid vacant land sales, teardown / new-construction anchors, `land_value`.

**Reused utility.** `market_basket_validate.ratio_study_vs_anchors` (line 47),
`calc_ratio_stats_bootstrap`, `calc_cod` (`utilities/stats.py`); vertical equity via
`vertical_equity_study.{VerticalEquityStudy, get_vertical_equity_scores}`.

**Thresholds (pinned to IAAO).**
- **Median ratio (level): 0.90 – 1.10.**
- **COD bands (2010 Standard on Ratio Studies):**
  - residential vacant land — **≤ 15 / ≤ 20 / ≤ 25** (very-large+active / mid / rural-depressed),
  - other non-ag vacant land — **≤ 20 / ≤ 25 / ≤ 30**.
- **Vertical equity:** VEI ≈ 1.0 with its 90% CI overlapping 1.0; **both** decile charts (see
  below) should be **flat** (no monotone trend). Exact VEI band pending review on Guilford.

> **Vertical-equity policy (project decision).** We dispense with **PRD and PRB entirely** and
> use **only the Vertical Equity Index (VEI)** plus **decile charts** (median ratio across
> value deciles). VEI/deciles are more legible than a single PRD/PRB scalar and show *where* in
> the value range any regressivity/progressivity occurs. Reuse `get_vertical_equity_scores`
> (VEI per quantile with 90% CI) and `VerticalEquityStudy` for the decile breakdowns.
>
> **Two decile charts — global vs neighborhood-relative.** Vertical equity is tested **two
> ways**, because *"are we over-valuing low-priced homes?"* and *"are we over-valuing poor
> people's homes?"* are different questions:
>
> 1. **Price-level decile chart (global).** Bin all parcels into deciles by **absolute** price
>    level (sale price / market value), plot median ratio per decile.
> 2. **Neighborhood price-level decile charts.** Bin by price level **relative to the parcel's
>    neighborhood** (e.g. within-neighborhood price decile, or price ÷ neighborhood median),
>    plot median ratio per relative decile.
>
> **Diagnostic logic:**
> - If the two **agree** (same regressive/progressive shape) → genuine vertical inequity that
>   holds *within* neighborhoods — i.e. we are systematically over-valuing **poor people's
>   homes**. This is the serious finding.
> - If they **disagree** — global (1) shows regressivity but neighborhood-relative (2) is flat
>   → the apparent regressivity is a **between-neighborhood / level artifact**, not
>   within-neighborhood inequity. Homes may be valued *correctly relative to their
>   neighborhood*, with the global slope driven by anomalously low prices on a few outliers
>   (often **invalid sales**). Treat global-only regressivity as a prompt to re-screen the
>   low-price tail for sale validity before concluding inequity.
>
> Both charts are emitted to the scorecard; the agree/disagree verdict is the headline
> vertical-equity result.
>
> **Sample for the decile/VEI study:** pool *all* land observations — vacant + teardown-implied
> + new-construction-implied land — not just vacant sales. On Guilford that's ≈ 192 obs vs 39
> vacant alone (§7); 39 is too thin for a stable 10-decile chart.

**Objectivity.** Most objective / most defensible.

**Per-parcel flag.** none (study-level), but per-anchor ratios are retained.

---

#### A4 — Tax-incidence / vacant-burden *(LVT vs revenue-neutral PT)*

**Definition.** Under a revenue-neutral LVT, vacant and under-improved parcels must pay *more*
than under a total-value property tax (def 1d). Pure arithmetic — needs no market data.

**Metric / formula.** With target revenue `R`:

```text
t_L = R / Σ land_value          # revenue-neutral LVT rate
t_T = R / Σ total_value         # revenue-neutral property-tax rate
Δ(parcel) = t_L · land_value − t_T · total_value     # burden shift to LVT
```

Assert: (a) for vacant parcels, `Δ > 0` (share with `Δ>0` ≈ all); (b) `Δ` rises monotonically
as the improvement-to-total ratio falls (negative correlation of `Δ` with `impr/total`).

**Data.** `land_value`, `total_value`, `is_vacant`. *(Guilford: no `market_value` column — use
`assr_market_value` as total, or `assr_land_value + assr_impr_value`.)*

**Reused utility.** none (arithmetic).

**Thresholds.** *Pass* if ≥ 99% of vacant parcels have `Δ>0` **and** corr(`Δ`,
`impr_ratio`) < −0.5; *warn / fail* otherwise.

**Objectivity.** Fully objective.

**Per-parcel flag.** `flag_incidence_anomaly` for vacant/under-improved parcels with `Δ ≤ 0`.

---

#### A5 — Desirability gradient

**Definition.** More desirable / productive areas pay more per unit of land (def 1c).

**Metric / formula.** Spearman rank correlation of neighborhood-median land $/unit vs a
neighborhood desirability proxy; should be strongly positive. Plus the paired-sales /
paired-neighborhood gradient: observed Δprice ≈ model Δland (slope ≈ 1, high Pearson r).
**Desirability proxy:** `spatial_lag_sale_price` where enriched, else **neighborhood-median
sale price / total $/sqft via `area_stats`** (Guilford fallback — spatial_lag absent, §7).

**Data.** `land_value`, neighborhood desirability proxy (spatial_lag *or* area_stats),
`neighborhood_filled`, matched sale pairs (`impr_he_id`).

**Reused utility.** `market_basket_validate.paired_neighborhood_gradient` (line 106),
`paired_sales_gradient` (line 62).

**Thresholds.** Spearman ρ ≳ 0.6 *pass*; gradient slope ∈ [0.8, 1.2] *pass*.

**Objectivity.** Objective-ish.

**Per-parcel flag.** none (area/pair-level).

---

#### A6 — No sales chasing on land

**Definition.** Land values must not be silently copied from the one nearby sale (def 2) — a
chased roll collapses under "show me how you'd value the lot that *didn't* sell."

**Metric / formula.** Run the existing three-signal detector against vacant-land sales: ratio
spike at 1.0, COD–CHD divergence, in/out-of-sample COD jump across the valuation date.

**Data.** `land_value`, vacant `sale_price`, `sale_age_days`, `he_id`/`land_he_id`.

**Reused utility.** `detect_sales_chasing` (`sales_chasing.py`).

**Thresholds.** Verdict `likely` (≥ 2 signals) *fail*; `possible` (1) *warn*; `no signal`
*pass*.

**Objectivity.** Objective.

**Per-parcel flag.** `flag_chased` for parcels at ratio ≈ 1.0 against their own sale.

---

#### A7 — Sanity bounds *(hygiene)*

**Definition.** Cheap, absolute correctness checks.

**Metric / formula.** Per-parcel bound checks (improvement derived as
`impr = total_value − land_value`):
- **`0 ≤ land_value ≤ total_value`** — land can't exceed total market value;
- **`0 ≤ impr ≤ total_value`** — improvement can't exceed total market value;
- (`land + impr = total` should hold by construction; flag any residual.)

Plus % floored negatives (`land_value_raw < 0`); no NaN / negative in the final series; land
$/unit within sane absolute bounds anchored to local land sales.

**Data.** `land_value`, `land_value_raw`, `total_value` (`assr_market_value` on Guilford),
local sale bounds.

**Reused utility.** market-basket `diagnostics` (`pct_floored_negative`).

**Thresholds.** *Warn* if floored share > 5%; *fail* > 10% (high flooring ⇒
improvement×location leakage / RCN mismatch). Any `land > total`, `impr > total`, negative
land/impr, or NaN ⇒ *fail*.

**Objectivity.** Objective.

**Per-parcel flag.** `flag_floored`, `flag_land_exceeds_total`, `flag_impr_exceeds_total`,
`flag_negative`, `flag_nan`.

---

#### A8 — Improvement-value location invariance *(building-side complement to A1)*

**Definition.** The same building costs about the same to build regardless of *where* it sits;
the location premium belongs in **land**, not in the improvement. So for **matched buildings in
different locations**, improvement value per sqft should be ≈ equal. This is the mirror image
of the A1 cornerstone — A1 says land is independent of the improvement; A8 says improvement is
independent of location. Together they confirm the split routes *location → land* and
*structure → improvement* (defs 1a, 1b, 2).

**Metric / formula.** Improvement derived as `impr = total_value − land_value` (or the
assessor's `assr_impr_value`). Form matched-building pairs across neighborhoods using
`impr_he_id` (improvement-only HE clusters) with a ±sqft / ±quality tolerance, then regress
Δ(impr $/sqft) on Δ(neighborhood desirability) through the origin — **slope ≈ 0, low
correlation** (contrast with A5, where Δland *should* track Δprice). Equivalently: CHD of impr
$/sqft *within* `impr_he_id` clusters should be low **even though those parcels span
neighborhoods**.

**Data.** `total_value`, `land_value` (→ `impr`), `impr_he_id`, `bldg_area_finished_sqft`,
quality, `neighborhood_filled`.

**Reused utility.** `impr_he_id` from `horizontal_equity_study.py`; the paired-matching logic
in `market_basket_validate.paired_sales_gradient` (built on `impr_he_id`), run on impr instead
of price.

**Thresholds.** Δimpr-vs-Δlocation slope ≈ 0 (≪ the A5 land slope); *warn/fail* if improvement
value carries a location gradient (⇒ the split is leaking location into improvements — the
opposite failure mode to A1).

**Objectivity.** Objective.

**Per-parcel flag.** `flag_impr_location_dependent`.

> A1 and A8 are a matched pair: a healthy split passes **both** (land carries location,
> improvement carries structure). An allocation/ratio split tends to fail both at once.

---

### Tier B — Common-sense / spatial *(second tier, more data-dependent)*

---

#### B1 — Market support / coverage

**Definition.** How much of the land value is *tied to evidence* vs extrapolated — the
protest-risk map (def 2).

**Metric / formula.** % of parcels **and** % of total land *value* on-manifold (near
comparable sales) vs extrapolated, via support-tier / k-NN distance to the nearest land sale.

**Data.** land sales, parcel `latitude`/`longitude` (normalized), `neighborhood`.

**Reused utility.** support-tier concept from `research/mb_support_map.py`; `spatial_lag`
confidence weights.

**Thresholds.** Report the distribution; *warn* if a large share of total land *value* sits in
the lowest support tier.

**Objectivity.** Semi-objective.

**Per-parcel flag.** `flag_unsupported` (off-manifold).

---

#### B2 — Developability ordering

**Definition.** Unbuildable / constrained land is worth less per unit than buildable land, all
else equal (def 3a).

**Metric / formula.** Compare land $/unit of constrained vs unconstrained peers in the same
area, where "constrained" = steep slope / erratic elevation (DEM **`slope_mean`,
`slope_stdev`, `elevation_mean`, `elevation_stdev`** — present in Guilford), sub-minimum lot
size, poor shape (**65-35 rule**), conservation `zoning`, floodplain, poor soil/drainage
(USDA-NRCS drainage classes, Course 201 Ch 7).

**Data.** DEM (`slope_mean` etc.), `zoning`, shape (rectangularity — confirm field name),
floodplain / soil/drainage (**not in Guilford** — data-conditional).

**Reused utility.** `area_stats` for peer baselines; geometry utilities for shape.

**Thresholds.** Constrained parcels should sit *below* unconstrained peers; *warn* on
inversions.

**Objectivity.** Semi-objective, **data-conditional** (needs DEM / zoning / soil).

**Per-parcel flag.** `flag_developability_inversion`.

---

#### B3 — Local spatial uniformity

**Definition.** Land $/unit should be locally smooth among similar nearby parcels (def 3b).

**Metric / formula.** Local dispersion / Moran's-I-style spatial autocorrelation among k-NN
neighbors; flag "hot pixels" far off their neighbors. **Must first apply IAAO site
adjustments** (P3) so legitimate variation isn't flagged: depth (depth table / 4-3-2-1),
**size economies** (size-adjustment factor), corner influence.

**Data.** `land_value`, coordinates, `frontage_ft_1`/`depth_ft_1`, `land_area_*`.

**Reused utility.** `spatial_lag` machinery (`data.py`).

**Thresholds.** Report local-dispersion distribution; flag the worst pixels for review.

**Objectivity.** Semi-objective.

**Per-parcel flag.** `flag_hot_pixel`.

---

#### B4 — Expected spikes

**Definition.** Land $/unit should spike where theory predicts (def 3c).

**Metric / formula.** Top-quantile land $/unit should co-locate with waterfront / view, prime
frontage, corners, and commercial corridors at rates above chance.

**Data.** distance-to-amenity / waterfront, `frontage_ft_1`, corner flag, corridor proximity.

**Reused utility.** distance / spatial enrichment (`data.py`).

**Thresholds.** Interpretive; report co-location lift vs baseline.

**Objectivity.** Interpretive, **data-conditional**.

**Per-parcel flag.** none (descriptive).

---

## 5. Scorecard & comparative output

- Each test emits a **headline metric + pass / warn / fail** vs the documented thresholds, plus
  per-parcel flags where applicable.
- A per-parcel **land-integrity confidence** aggregates the flags (mirrors market-basket's
  `land_confidence`: high / med / low).
- **Comparative scorecard.** Run the identical battery on `land` and `land_compare` (the
  assessor's published land split) and print results side by side. The answer to "are these
  good?" is the contrast — most sharply on **A1** (improvement independence).
- **Defensibility checklist (def 2 / Cat 2)**, following IAAO Course 201 Ch 6 §VIII *Support of
  Land Values*: document the selected unit (A0), the market anchors (A3), and the adjustment
  factors (depth / size / corner) per model group — these are exactly what survives a protest.
  Note IAAO's caution against defending land and building separately on improved parcels, and
  that an LVT use case deliberately inverts that — surface it as a **known tension, not a
  failure**.
- The eventual report renders markdown → PDF following the `reports.py` convention
  (`start_report` / `set_var` / `finish_report`); no template is built in this phase.

---

## 6. Definition → test traceability

| "Good" definition | Tests |
|---|---|
| 1a — identical lots pay the same regardless of what's built | A1, A2, A8 |
| 1b — improving your property doesn't raise your land bill | A1, A8 |
| 1c — productive areas pay more per unit land | A5 |
| 1d — vacant lots pay more under LVT than revenue-neutral PT | A4 |
| 2 — survives protest: evidence-based, market-anchored, IAAO practice | A3, A6, A8, B1, defensibility checklist |
| 3a — unbuildable < buildable | B2 |
| 3b — locally uniform (conditional) | A2, B3 |
| 3c — spikes where expected (waterfront, frontage, corners) | B4 |
| (foundation) — right unit of measure | A0 |
| (foundation) — HBU as-though-vacant; consistent use | P2, archetype-divergence |
| (hygiene) | A7 |

---

## 7. Guilford (NC) validation — computability check (executed)

Ran `research/lvi_verify_guilford.py guilford_subset` (read-only; loads `out/2-clean-sup`,
filters to `single_family`, builds anchors via `mb_config.prepare`). Result:
**universe 10,715 × 182, sales 1,877 × 202.**

**Anchor counts:** vacant land sales **39**, teardown **5**, new-construction (RCN-implied)
**148**.

| Test | Computable on Guilford? | Notes |
|---|---|---|
| A0 unit / frontage / depth | ✅ | `land_area_sqft`, `land_area_somers_ft`, `frontage_ft_1..4`, `depth_ft_1..4`, `zoning` all present; `osm_frontage_{class}_ft` enables auto road-class frontage typing |
| A1 improvement independence | ✅ *(fallback)* | improvement fields + `land_he_id` present; **`spatial_lag` absent** → use neighborhood FE (`neighborhood_filled`) |
| A2 horizontal land equity | ✅ | `land_he_id` present |
| A3 ratio study / VE | ✅ *(power-limited)* | anchors + RCN (`bldg_value_replacement`) present; **39 vacant sales is thin for 10-decile VE** — pool land obs (vacant+teardown+NC-implied ≈ 192) for the decile/VEI sample; teardown (5) too thin to lean on |
| A4 tax incidence | ✅ | total = `assr_market_value` (no `market_value`); `is_vacant` present |
| A5 desirability gradient | ✅ *(fallback)* | `spatial_lag` absent → `area_stats` neighborhood medians; `impr_he_id` present for paired-sales |
| A6 sales chasing | ✅ | `he_id`/`land_he_id`/`sale_age_days` present |
| A7 sanity bounds | ✅ | `land_value_raw`, `assr_market_value` present (land/impr ≤ total) |
| A8 impr location invariance | ✅ | `impr_he_id`, `assr_market_value`/`assr_impr_value` present |
| B1 market support | ✅ | coords + sales present |
| B2 developability | ✅ *(partial)* | DEM present as `slope_mean`/`elevation_mean` (+stdev), `zoning` present; **floodplain / soil-drainage absent** |
| B3 local uniformity | ✅ | `latitude`/`longitude`(+`_norm`) present |
| B4 expected spikes | ❌ data-conditional | no water-distance / corner layer in Guilford (could partly proxy via `osm_frontage_*`) |

**Comparative scorecard feasible on Guilford:** `assr_land_value` is the assessor land split
(`land_compare`), `assr_market_value` the total — A1/A3/A4 can be run head-to-head
market-basket vs assessor.

**Key gaps to carry forward:** (1) `spatial_lag` not enriched → neighborhood-FE / area_stats
fallbacks are the default, not the exception; (2) vacant-land sample (39) limits vertical-equity
decile power — pool all land observations; (3) waterfront/corner data (B4) and floodplain/soil
(B2 tail) absent. Re-run `lvi_verify_guilford.py petersburg` to check DEM-rich coverage.

---

## 8. Open items

- IAAO Course 201 (incorporated) supplied: level (0.90–1.10), vacant-land COD bands, the five
  units of comparison + least-variation selection rule, depth / 65-35 / corner / size
  adjustments, and the support-of-land-values defensibility checklist.
- Vertical equity uses **VEI + decile charts only** (PRD/PRB dropped, project decision) —
  calibrate the VEI pass band on Guilford.
- Still worth pulling the primary **Standard on Ratio Studies** and **Standard on Mass
  Appraisal of Real Property** for any land-specific trim guidance.
- Confirm the assessor land/total column names per jurisdiction for the comparison series.

---

## 9. Verification of this spec (review + computability, not test runs)

1. ✅ **Done** — `research/lvi_verify_guilford.py` maps every test's data requirements to the
   cleaned Guilford columns and reports anchor counts; results in §7. Gaps recorded
   (spatial_lag, floodplain/soil, B4 layers); fallbacks specified inline.
2. ✅ **Done** — reused-utility references confirmed to exist:
   `market_basket_validate.{ratio_study_vs_anchors, paired_sales_gradient,
   paired_neighborhood_gradient, archetype_divergence}`, `utilities/stats.{calc_cod,
   calc_chds, calc_ratio_stats_bootstrap}`, `sales_chasing.detect_sales_chasing`,
   `vertical_equity_study.{VerticalEquityStudy, get_vertical_equity_scores}`,
   `horizontal_equity_study` (`land_he_id`).
3. ⏳ **Pending implementation** — calibrate open thresholds (A1 partial-R², A2 CHD, VEI band)
   against in-sample noise once run on Guilford.
4. ⏳ **Pending implementation** — sanity-check the comparative framing by running A1 / A3 / A4
   on the market-basket land series vs the assessor `assr_land_value` series for Guilford
   (data confirmed available, §7).

---

## 10. Wake County (NC) MVP run — executed 2026-06-20

First working implementation: `research/lvi_anchors.py` (ground-truth extractor) +
`research/lvi_battery.py` (Step1/A1/A2/A3/A7/A8) + `research/lvi_wake.py` (runner). Testing the
**assessor** roll (`assr_land_value`) on `single_family` (universe 377,518; 28,077 valid sales).

**Correction to §9.2:** `market_basket_validate.{ratio_study_vs_anchors, paired_sales_gradient,
…}` **do not exist** in the package or `/research` (uncommitted / other branch). The anchor
builder and paired-sales were written net-new. Confirmed-real reuse:
`utilities/stats.{calc_cod, calc_ratio_stats_bootstrap, trim_outlier_ratios}`,
`vertical_equity_study.get_vertical_equity_scores`, `horizontal_equity_study` (`land_he_id`/
`impr_he_id`).

**Anchors:** vacant 603, teardown 14, new-construction 10,941. RCN reconstructed from the cost
book: `RCN = assr_impr_value / (bldg_condition_pct/100)`; new-construction land = `sale_price_
time_adj − assr_impr_value` (dep≈0).

**Assessor scorecard (current data):**

| Test | Result | Verdict |
|---|---|---|
| Step1 total median / COD / VEI | 0.983 / 6.3 / −6.8 | pass (matches existing ratio_study.md 0.96 / 7.2) |
| A1 improvement partial-R² | **0.093** | warn (0.05–0.15 band; land moderately tracks improvements) |
| A2 land $/sqft CHD / impr CHD | 8.9 / 6.3 | pass |
| A3 GOLD (vacant+teardown) median / COD | **0.819 / 29.7** | fail (land ~18% below market vacant-land; high dispersion) |
| A3 new-construction (semi-circular) median / COD | 0.913 / 25.2 | (reported only) |
| A7 sanity violations | 0.03% | pass |
| A8 impr loc-invariance CHD | 6.3 (clusters span ~85 nbhds) | pass |

Per-parcel confidence: high 362,215 / med 15,175 / low 128.

**Methodological findings (carry into promotion):**
1. **Residual cross-check (land = total − RCNLD) is degenerate for the assessor** — `total −
   assr_impr_value ≡ assr_land_value` by construction, so `residual_ratio` ≡ 1.0. It is an
   **AVM-only** diagnostic (the silver top-down direction). For the assessor, corroboration
   comes from the vacant-sale `anchor_ratio` (median 0.867) and cluster uniformity.
2. **New-construction anchor is semi-circular for the assessor** (uses `assr_impr_value`) — it
   really re-tests total-value level on new builds. Reported but excluded from the gold verdict;
   gold = vacant (+teardown). It IS a clean silver cross-check for the AVM.
3. **Headline story:** Wake's *totals* and *buildings* are excellent (COD 6–7, uniform,
   location-invariant), but *land* is moderately improvement-dependent (A1 warn) and runs ~18%
   below direct vacant-land market evidence with high dispersion (A3 fail) — a land-schedule
   level/uniformity problem, not a total-value problem.

**Calibration (replaces §9.3 'pending'):** observed A1 warn at 0.093 and A2 land-CHD 8.9 / impr-
CHD 6.3 confirm the draft bands are sensibly placed; A3 uses the IAAO vacant-land COD bands
(≤15/20/25) directly. VEI band still needs a multi-jurisdiction sweep before pinning.

**AVM comparison not available — multi_mra emits no land split (finalize re-run 2026-06-21).**
After refreshing the models, the head-to-head still can't run: in Wake's `multi_mra` output
`prediction_land_sqft == prediction / land_area` for 100% of rows (and `prediction_impr_sqft ==
prediction / bldg_area`), i.e. these are *total value per unit*, NOT a land/improvement
decomposition. Constructing "AVM land" from them just relabels the total (symptoms: A7 91%
summation failure, A3 new-construction ratio ~4.6). The runner now reads both series from the
model dir's own `universe.csv`+`pred_universe.csv` (100% consistent, key-join issue gone) and
**auto-suppresses the AVM column** via `_has_real_land_split()` until a model exports a genuine
`land_value`/`impr_value`. The intended A1 contrast (assessor warn vs AVM pass) needs that
decomposition wired into the modeling export first.

> Earlier note about "stale models / 48.5% key2 overlap" was a red herring from re-loading
> `2-clean-sup.pickle` separately; the finalize works off a more-processed snapshot. Reading
> straight from the model dir sidesteps it entirely.

**Assessor flagship verdict (stable):** totals excellent (median 0.983, COD 6.3), buildings
uniform + location-invariant (A2/A8 CHD ~6), land is the weak link — improvement-dependent (A1
0.093 warn) and ~16% below direct vacant-land market evidence with high dispersion (A3 gold
prime-vacant median 0.836 / COD 28.8, n=217). Per-parcel confidence high 362k / med 15k / low 128.

---

## 11. Anchor scrutiny — "prime buildable" vacant lots (2026-06-21)

Only genuinely buildable, locally-comparable lots are valid land-value evidence. Added
`research/lvi_vacant_scrutiny.py` (standalone funnel) + a prime-lot filter in `lvi_anchors`
(`classify_zoning`, `neighborhood_size_bands`, `add_prime_flags`, `prime_lot_mask`), now baked
into the battery's A3 gold standard.

**Disqualifiers** (a vacant sale is dropped if any hold): a structure is/was present (teardown
or mislabeled "vacant"); non-residential / rural zoning (`classify_zoning`: keep R-*/RX/UR/SR/
GR/RMD/PUD incl. watershed R-*W; drop rural RA/RR and commercial/office/industrial/mixed);
neighborhood has < `MIN_PEERS`=10 built peers; lot size outside the neighborhood built-peer
[5,95] pctile band; weird shape (`geom_rectangularity_num` < `RECT_MIN`=0.40).

**Funnel (960 SF+UNKNOWN vacant sales):** structure-present 191, non-res/rural 302 (overcount —
crude before allowlist), <10 peers 104, size-outlier 377, weird-shape 178 → **PRIME = 241 (25%)**
(median lot 20.5k sqft, p10–p90 7k–58k, rectangularity 0.88).

**Key result — cleaning does NOT rescue the assessor's land** (so the A3 fail is real, not an
anchor artifact). Single-family assessor land ratio by stage: all-vacant 0.837/COD_trim 29.0 →
genuinely-vacant 0.822/31.6 → prime 0.840/28.9. (Battery, on the more-filtered model-dir sales:
prime 0.836/28.8, n=217.)

**Land ratio study BY MODEL GROUP** (`research/lvi_land_ratio_by_group.py`; assessor land vs
vacant sale, prime stage, COD_trim):

| model_group | n_prime | median | COD_trim | note |
|---|---|---|---|---|
| single_family | 233 | 0.840 | 28.9 | only group with a trustworthy sample |
| UNKNOWN | 65 | 0.766 | 36.4 | unmodeled; no time-adj price → raw price used |
| mobile_manufactured | 18 | 0.741 | 48.3 | thin |
| public | 11 | 0.521 | 74.2 | thin, noisy |
| agricultural | 7 | 0.742 | 50.3 | thin |
| commercial | 7 | 0.760 | 1116 | n too small / wild dispersion — unusable |

**Every** group shows assessor land *under*-assessed vs vacant sales (medians 0.52–0.84, all <
0.90); only single_family has the sample to trust. UNKNOWN has `assr_land_value` but no
`sale_price_time_adj` (time-adjustment skips unmodeled groups) — runner coalesces to raw
`sale_price`.

---

## 12. Non-circular sale-validity filters (2026-06-21)

**Circularity problem.** The A3 metric is `assr_land / sale_price`. Any filter that consults the
*assessment* (or flags "this sale gives a high ratio") is circular — it mechanically suppresses
detectable over-assessment. A valid filter must use information NOT in the ratio.

**The two non-circular signals (both confirmed to work on Wake):**

1. **Deed qualification code — `disq_flag` WHITELIST (primary).** Wake's Disq_and_Qual code:
   A/C = qualified arm's-length; E/F/D/G/T/L = family/fractional/non-warranty/life-estate/etc.
   Uses neither price nor assessment. *How it was being missed:* `valid_sale` applied it as a
   *deny-list* — `isempty(disq_flag) OR isin(disq_flag,[A,C])` — so unstamped (NaN) sales passed
   by default, and vacant land sales are mostly NaN (the flag is patched from `parcels.csv`'s
   most-recent-sale, which rarely lines up with a land transfer). The fix for the small,
   high-stakes land gold standard is a **whitelist** (`lvi_anchors.qualified_sale_mask`): require
   explicit A/C. Keep the permissive deny-list for the 27k modeling sales (robust to stragglers);
   require the whitelist only for the ~hundreds of land anchors.

2. **Residual-land market floor (secondary/corroborating).** Land implied by *improved* sales =
   `sale − assr_impr_value` (cost-book RCNLD; independent of the land schedule). Where ≥5 such
   comps exist in a neighborhood, flag vacant sales far below the local residual level.
   (`lvi_residual_smell.py`.) Depreciation split: **low/no-dep (pct≥90)** residuals are the
   cleanest (building ≈ RCN, IQR/med 1.16); **high-dep (<75)** are noisier (IQR/med 1.43) and
   run lower — use low-dep as the yardstick.

**The definitive convergence.** Two fully independent non-circular estimates of the prime SF land
level agree at **~$22–26/sqft**: low-dep improved-sale residual **$22.27/sqft** ≈ prime+qualified
vacant **$26.49/sqft**. Uncleaned vacant sets sit at **$11–12/sqft (≈ half)** — i.e. raw vacant
anchors understate land ~2×, which is exactly why naïve A3 looked so bad. Among *qualified* sales,
the residual floor flags only ~4% below 50% of local level — so **qualification alone is nearly
sufficient; the residual test independently confirms it** rather than adding much filtering.

**Defensible exclusion rule set (final, all non-circular w.r.t. the land assessment):**
- Tier 1 (deed, primary): keep only `disq_flag ∈ {A,C}`.
- Tier 2 (lot attributes): genuinely vacant; residential zoning; size within neighborhood
  built-peer [5,95] band; rectangularity ≥ 0.40; ≥10 built peers.
- Tier 3 (market backstop): drop vacant sales < ~50% of the neighborhood low-dep residual land
  level (where ≥5 comps). Replaces the earlier circular flat `$/sqft < 5` floor.

**A3 single-family by stage (median | COD_trim):**

| stage | n | median | COD_trim | verdict |
|---|---|---|---|---|
| all vacant | 633 | 0.837 | 29.0 | fail |
| prime (lot attrs only) | 233 | 0.840 | 28.9 | fail |
| **qualified (A/C) — primary** | **231** | **0.828** | **21.9** | **pass** |
| prime + qualified (purest) | 75 | 0.828 | 20.5 | pass |

**Qualification alone does the work and keeps the sample** (231 vs 633; COD 29→22, passes IAAO
≤25). The lot-attribute filter on top barely tightens COD (20.5) at heavy sample cost — use
qualified-only as the headline gold, prime+qualified as the conservative confirmation.

**Refined flagship verdict.** Wake single-family land is **uniformly assessed** (COD_trim ~21–22,
passes IAAO once non-circular filters are applied — the earlier "fail" was anchor contamination)
but **systematically ~17% under-assessed at the level** (median ratio ~0.83), with the building
side correspondingly over-weighted (ties to A1 = 0.093). Totals and buildings remain excellent.

---

## 13. Pooled evidence streams, residual masking, and teardowns (2026-06-21)

`build_land_observations` now emits four validated streams (all gated on qualified A/C deed
code): **vacant**, **teardown**, **rcn_resid** (dep≈0, percent_good≥0.95: price−assr_impr=price−RCN),
**rcnld_resid** (low/mod dep 0.75–0.95: price−assr_impr=price−RCNLD). Per-source A3 (assr_land /
observed) on Wake SF:

| stream | n | A3 median | COD_tr | land $/sqft |
|---|---|---|---|---|
| vacant (prime+qual) | 73 | **0.831** | 20.9 | 26.49 |
| rcn_resid (dep≈0) | 4,272 | 0.881 | 25.3 | 24.88 |
| rcnld_resid (lo/mod dep) | 15,713 | 0.960 | 23.2 | 16.26 |

**Residual masking (key finding).** For the assessor's OWN series the residual streams are
semi-circular: `observed_land = price − assr_impr` and `assr_land = assr_market − assr_impr` share
`assr_impr`, so if the assessor over-weights buildings (the flip side of the land under-assessment)
the same error depresses both numerator and denominator, pulling the ratio toward the total ratio
(~0.98) and **masking** the under-assessment. The drift is monotone in depreciation reliance:
vacant 0.831 → rcn_resid 0.881 → rcnld_resid 0.937–0.960. ⇒ **Grade the assessor on vacant
(prime+qualified) only.** The residual streams are *fully non-circular for an EXTERNAL land series
(AVM)* — that's their designated use. The $/sqft *levels* still corroborate: vacant $26.49 ≈
rcn_resid (dep≈0) $24.88 (two independent methods → true prime land ≈ $25/sqft).

**Teardowns: essentially none in Wake.** The naive `age_at_sale<0` rule (current building newer
than the sale) caught **new-construction sales**, not teardowns — the price includes a brand-new
building, so `observed_land=price` gave a bogus 0.227 ratio / $84/sqft. Proper definition (sold
WITH a building · new build shortly after · **price/land-sqft substantially below the neighborhood
developed level**) yields **1 of 14** candidates — the other 13 priced at ~97% of the developed
level (i.e. finished homes), and as residuals give a sensible median 1.067 / $24.73/sqft.
`build_land_observations` now uses the proper definition (`teardown_frac=0.50`); on Wake teardown
n=1. Net: there is no teardown evidence stream here; those sales correctly flow into `rcn_resid`.

**Final defensible anchor design.** Assessor land gold standard = **vacant, prime, qualified**
(direct, fully non-circular). rcn_resid (dep≈0) corroborates the *level* only. rcnld_resid +
teardown reserved/empty. Verdict unchanged and now maximally defensible: **~17% under-assessed
(median 0.833), uniform (COD_trim 21.1, passes IAAO ≤25), n=67.**

---

## 14. Deferred tests run — A5, B3, A6 + de-tautologization (`lvi_extra_tests.py`, 2026-06-22)

- **A5 desirability gradient — PASS (strong).** Against the *cleanest* proxy — neighborhood
  median **qualified vacant-sale $/sqft** (the actual market land gradient) — assessor land
  $/sqft tracks it at **Spearman ρ = 0.911** (n=15 nbhds with ≥3 qualified vacant sales; median
  assr $20.7 vs market $26.5, consistent with the ~17% level shortfall). CAUTION: against
  neighborhood median *home* $/sqft the land ρ is only 0.134 — but home-$/sqft is a
  building-dominated proxy, not a land proxy. **An earlier run using median *total* sale price
  gave a spurious ρ ≈ −0.05 ("inversion") — a pure proxy artifact; total price confounds home
  size/quality.** Use the market-land proxy.
- **A6 sales chasing on land — PASS (clean).** `detect_sales_chasing` verdict = *no signal*;
  ratio-at-1.0 spike share 5.2% (median ratio 0.837, nowhere near a 1.0 pile-up). The assessor
  is not copying land values from the sale.
- **B3 local spatial uniformity — moderate pass.** Within-neighborhood-detrended log land $/sqft:
  Moran's I = 0.150 (positive → locally smooth beyond the legit between-nbhd gradient), median
  local deviation 11%, hot-pixel share 2.4%. Locally smooth, modestly autocorrelated.
- **De-tautologization (the important part).** A2/A8's low CHDs partly reflect the assessor's own
  schedules. A5b settles it for LAND: assessor land tracks **independent market vacant-land
  prices** at ρ=0.911 — it is not merely self-consistent, it matches the market gradient. For
  BUILDING, matched-building CHD (A8 = 6.3) holds; the across-neighborhood building-$/sqft
  gradient (ρ=0.335 vs home $/sqft) is **quality composition** (nicer areas → higher-quality
  homes → higher cost/sqft), not location leakage — so A8's location-independence stands.

**Net effect on the thesis:** the three deferred tests resolve *in favor* of the thesis. Land
tracks the market gradient (ρ=0.911), is locally smooth (B3), and is not sales-chased (A6); the
de-tautologized land gradient is the strongest single piece of independent support. Remaining
caveats: A5b rests on 15 neighborhoods; single jurisdiction / model group / valuation date; the
building quality-vs-location confound can't be fully separated without external cost data.

---

## 15. Frozen-SOV assumption → residuals become non-circular; expanded gold (2026-06-22)

**Stipulation (modeling assumption):** derived RCN comes from Wake's **frozen, publicly-adopted,
finalized Schedule of Values** cost table — i.e. building cost is EXOGENOUS, not a free parameter
the assessor tunes against land. This dissolves the §13 masking objection: with RCN externally
fixed, `observed_land = price − RCN` is a legitimate, accurate land estimate, and the residual A3
ratio is the *true* land ratio (not pulled toward the total ratio). The **rcn_resid (dep≈0)**
stream — which uses the frozen RCN with ~no depreciation judgment — is therefore promoted from
"external-AVM-only" to **bona-fide non-circular land gold evidence.**

**Expanded gold standard = prime+qualified vacant ∪ qualified rcn_resid(dep≈0):**

| stream | n | median ratio | COD_trim | land $/sqft |
|---|---|---|---|---|
| vacant prime+qual (purest) | 67 | 0.833 | 21.1 | $25.1 |
| rcn_resid dep≈0 (frozen RCN) | 4,481 | 0.877 | 25.4 | $25.2 |
| **EXPANDED GOLD** | **4,548** | **0.850** [0.842,0.860] | **25.3** | **$25.2** |

The two independent methods agree on the land *level* to within **0.4%** ($25.1 vs $25.2) and on
the *ratio* (0.83 vs 0.88) — land is **~15% under-assessed**, now backed by **4,548 observations,
not 67.** COD ~25 sits right at the IAAO acceptable boundary (the pure vacant is tighter at 21).

**Two cautions carried forward:**
- **rcnld_resid stays OUT of gold.** At low/moderate depreciation the ratio drifts to 0.961 and
  land $/sqft falls to $16.6 — because the *depreciation* factor (percent_good → RCNLD) is a
  separate assessor judgment, NOT part of the frozen RCN. The drift hints the depreciation
  schedule may **under-depreciate** older homes (RCNLD too high → residual land too low),
  confounded with older-home/cheaper-land composition. A separate, testable question.
- **Vertical equity.** On the large sample, VEI ≈ −60 (expensive land under-assessed more than
  cheap land — regressive), vs −10.5 on pure vacant. The residual mechanics likely inflate the
  large-sample VEI; treat the pure-vacant VEI (mild regressivity) as the trustworthy read, and
  flag land vertical equity as the next thing to pin down.

---

## 16. Land vertical equity — resolved (`lvi_land_vertical.py`, 2026-06-23)

Ran VEI + decile charts on every stream, clean-vacant vs residual side by side, to separate real
regressivity from residual mechanics. **Result: the severe regressivity is an artifact; a mild
real tilt survives.**

| stream | n | VEI | global deciles (low→high land value) |
|---|---|---|---|
| prime+qual vacant (purest) | 67 | **−10.5** | 0.91 0.93 0.83 0.82 |
| qualified vacant | 218 | −8.2 | ~flat, noisy |
| all-vacant | 602 | +9.9 | no trend |
| rcn_resid (frozen RCN) | 4,481 | **−62.8** | 1.32→0.77 (steep) |
| rcnld_resid | 16,149 | −52.4 | 1.29→0.79 (steep) |

**The VEI −60 is a residual artifact.** Clean vacant evidence (no RCN subtraction) shows only
VEI −8 to −10; the residual streams show −52 to −63. At the low-value end `observed_land =
price − RCN` is a small difference of large numbers, so RCN error inflates the ratio — the
residual's cheap-land decile reads 1.32 ("over-assessed"), which is untrustworthy. **Drop the
severe-regressivity reading.**

**A mild, genuine regressive tilt survives.** Pure-vacant VEI ≈ −10.5, and the
**neighborhood-relative** decile trend is consistently negative across ALL streams incl. clean
vacant (prime+qual rho −0.92, all-vacant −0.81): within a neighborhood, premium lots (high
$/sqft) are under-assessed relative to ordinary lots. This is the signature of **schedule-based
land valuation** — a fairly uniform $/sqft per neighborhood zone that misses lot-level premiums
(corner / view / frontage / position). Modest (~10% gap), but consistent.

**Verdict:** no severe land regressivity (the −60 was mechanical). Mild real regressive tilt
(VEI ≈ −10) from within-neighborhood under-differentiation — so the ~15% under-assessment is
somewhat *worse* for premium lots. Methodological win: running clean-vs-residual in parallel is
what exposed the artifact.

---

## 17. Depreciation schedule — tested, well-calibrated (`lvi_depreciation.py`, 2026-06-23)

Hypothesis (from the §15 rcnld_resid drift): the percent-good schedule under-depreciates old
homes. **Not supported.**

- **View 1 (robust, n=26,726):** within a fine neighborhood (VCS) land is ~constant, so residual
  land ($/sqft) = sale − assr_impr should be flat with building age. It is: within-neighborhood
  FWL slope = **−0.035 $/sqft per year** (≈ −$2/sqft over 50 yrs on a ~$15–18 base — negligible).
  The raw between-age decline ($18.7→$14.9) is **between-neighborhood composition** (older homes
  in lower-land-value VCS zones), which within-neighborhood demeaning removes entirely.
- **View 2 (independent percent-good curve, n=4,754):** reconstruct RCN independently (cost/sqft
  by quality, calibrated on new builds where pg≈1 → free of the old home's pg), then market %good
  = (sale − neighborhood-new-build land)/RCN. Assessor pg tracks market pg within ±10 pts,
  non-monotone, no systematic under-depreciation (faint over-depreciation at 50+, but n=54/229,
  noisy).

**Verdict:** depreciation schedule is well-calibrated; the rcnld_resid drift was composition, not
a schedule error. Confirms keeping rcnld_resid out of the gold standard, and means there is no
building-side depreciation problem accompanying the land under-assessment.

---

## 18. A0 (unit selection) + B1 (market support) — run 2026-06-23

**A0 — $/lot (site value) marginally beats $/sqft for Wake SF.** Within-neighborhood size
elasticity beta = dlog(land)/dlog(area) ≈ **0.37–0.46** (vacant 0.46, rcn_resid 0.37): land value
scales strongly *sub-linearly* with lot size (strong size economies). Consequence: within-
neighborhood COD is lower for $/lot (20.6) than $/sqft (22.1) — site value captures Wake SF land
marginally better. Margin is small and rests on residual evidence (vacant too thin per-nbhd for
COD, but its beta corroborates). Rigorous IAAO reading: $/sqft over-charges big lots; use $/lot or
$/sqft-with-size-adjustment. Does NOT change prior results — A3 is a total-value ratio
(unit-invariant); the ~15% under-assessment stands. ($/acre ≡ $/sqft rescaled; $/front-foot
uncomputable — no frontage; $/buildable-unit ≈ $/lot for SF.) `lvi_a0_unit.py`.

**B1 — good spatial coverage, sparse per-zone coverage (the protest-risk surface).** Evidence =
4,548 qualified land obs (1.2% of parcels). By DISTANCE, coverage is good: median 0.44 mi to
nearest evidence, 99% of parcels (99.5% of land value) within 2 mi, only 0.9% extrapolated (>2
mi). By VCS NEIGHBORHOOD, it's sparse: **81.7% of parcels / 75.1% of land value sit in zones with
ZERO internal land evidence** (only 559 of 4,699 neighborhoods have any). Reconciliation: land is
calibrated per VCS, so the real risk isn't distance but "no sale in your own valuation zone" —
3/4 of land value rests on cross-zone inference. The decent spatial coverage *depends on the
frozen-SOV residual stream*; pure-vacant-only (67 pts) would collapse it. Per-parcel map:
`out/lvi/support_map.csv`. `lvi_b1_support.py`.

**Net:** A0 refines the recommended unit (site value) without disturbing findings; B1 quantifies
the recurring "thin evidence" constraint — it's a per-zone, not spatial, sparsity, and it is the
honest boundary on how defensible any individual parcel's land value is.

---

## 19. B1b — analogical support / evidence propagation ("sudoku" coverage) — 2026-06-23

For protest defensibility, tie every zone lacking DIRECT land evidence to a supported zone via
MATCHED BUILDINGS. Licensed by two validated tests: building value is location-invariant (A8,
CHD 6.3), so matched buildings (same impr_he_id) in zones A,B give land_B − land_A = price_B −
price_A (building cancels); and that transfer gradient is real (A5, rho=0.911). `lvi_b1b_propagate.py`.

**Coverage cascade (share of $56.9B SF assessed land value):**

| support level | zones | % land value |
|---|---|---|
| Direct (in-zone evidence) | 559 | 24.9% |
| +1 hop, ≥3 matched-building bridges (defensible) | 1,337 | **58.0%** |
| +1 hop, ≥1 bridge | 3,137 | 87.5% |
| +full propagation (bridge chains) | 3,601 | 93.3% |
| unreachable WITH sales | 0 | 0.0% |
| no improved sales (needs spatial/area fallback) | 1,138 | 6.8% |

**Result:** the per-zone sparsity (75% of value lacking direct evidence, §18) is largely
resolved. Strict, protest-grade tie (≥3 independent matched-building bridges to direct evidence)
covers **58%** of land value; any-link 87.5%; chains 93.3%. Every zone with improved sales
connects — only 6.8% (zones with no improved sales) is genuinely un-bridgeable and needs a
spatial/area-level fallback.

**Wield it tiered:** a single bridge (cluster spans ~85 zones) is weak; ≥3 one-hop bridges is the
defensible threshold; multi-hop chains compound error. Per-link evidence is concrete (matched-
building paired-sale price difference), not assumption — three passing tests composed.

**Next (the actual protest artifact, not yet built):** propagate land LEVELS (land_B = land_A +
Σ Δprice along bridges), run the sudoku consistency check (independent paths to a zone must
agree), and emit per-parcel evidence chains tying each unsupported parcel to a validated zone.

---

## 20. B1b (full) — land-level propagation + per-parcel evidence chains — 2026-06-23

Propagate land LEVELS (not just connectivity): calibrate a location-invariant building $/sqft per
matched-building cluster on the DIRECT-evidence zones (where market land is known), apply it
everywhere to back out market land (abstraction anchored to direct evidence, not a cost table).
`lvi_propagate_levels.py` → `out/lvi/land_evidence_chains.csv`.

**Mechanism validated (unbiased):** on 545 anchor zones, propagated vs direct land:
median(prop/direct)=**1.007**, Spearman **rho=0.903**, COD 38.3 (unbiased + correlated, but noisy
per zone). Cross-cluster (sudoku) agreement within a zone: median COD 24.9.

**Coverage 25% → 74% of land value:** direct 24.9% + matched-building 49.1% = 74.0%; remaining
26.0% has no CALIBRATED bridge. NOTE this is lower than the §19 connectivity (87–93%): a *level*
needs the bridging cluster calibrated (≥3 anchor sales), which is stricter than mere connection.
The 26% needs the cost-table abstraction (rcnld_resid, §17) or a spatial fallback.

**Extended A3 (zone-level, all 2,311 covered zones, 74% of value):** median assr/market land =
0.955, COD_trim 36.2. The under-assessment **direction holds far beyond the original 25%**, but
propagation is **noisier and milder** (zone-level 0.955 vs parcel-level direct 0.85) — it
*corroborates the direction, does not sharpen the magnitude*. The clean ~15% stays the
direct-evidence figure.

**Per-parcel artifact (`land_evidence_chains.csv`):** per parcel — assr land, propagated market
land + $/sqft, land ratio, support type (direct/matched-building/none), and `n_bridge_clusters`
(confidence tier). This is the protest packet: each parcel's land tied to a market level with its
evidence basis and strength.

**Verdict:** the sudoku works and is validated unbiased — more than doubles land value tied to
concrete market evidence (25%→74%) and gives every covered parcel an evidence chain. Candor: per-
zone estimates are noisy (use the bridge-count tier), coverage is 74% not 93% (calibration is
stricter than connectivity), and it corroborates rather than re-estimates the under-assessment.

---

## 21. Tier-2 — differential locational premiums (matched buildings) — 2026-06-23

Relative (not absolute) evidence: a building model (impr_he_id) selling in two zones gives a pure
locational land differential (price gap = land, building cancels by A8), anchored to pure RCN
residuals. Implemented as a building-composition control: model-demean RCN-residual log land
$/sqft, then the zone mean = location premium net of what's built. `lvi_tier2_differentials.py`.

**Findings:**
- Network: 4,481 RCN-residual sales in just **95 models**; 80 span >=2 zones, connecting **524
  zones**.
- **Building composition WAS partly confounding the raw location signal:** composition-controlled
  premium vs raw zone level rho=**0.836** (not ~1) — controlling for what's built re-ranks zones,
  isolating pure location (removes expensive-models-on-expensive-land selection).
- Strong location signal: model-controlled premiums $6.5->$64/sqft (p10->p90), **~10x** across
  zones (inside-beltline 01RA244 $99/sqft vs rural 09WC900 $1.3/sqft, via the same models).

**Honest caveat — per-chain noise:** cross-model consistency is poor — median within-zone std of
the differential ~**51%** (independent matched-model chains for one zone disagree ~±50%). Causes:
only 95 coarse model bins + RCN-residual noise amplified at the high end. So a SINGLE matched-pair
chain is too noisy to stand alone; the reliable product is the **multi-model zone aggregate**.

**Role:** a relative-evidence REFINEMENT, not a coverage extension. Footprint = RCN-residual
(new-construction) zones (524) — matched pure-RCN anchors are new builds. Adds over §20/A5:
(1) composition control (isolates location from building mix), (2) legibility ("same model, two
subdivisions, $X premium" — the most concrete protest evidence), (3) relative robustness (the
location ranking holds even where absolutes are noisy). Per-zone premiums in
`out/lvi/location_differentials.csv`. Older-only zones still need §20's depreciation-handling
abstraction.
