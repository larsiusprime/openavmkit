# Land Value Integrity — Wake County Flagship: Methods, Results & Generalization

*Companion retrospective to `land_value_integrity_spec.md` (§10–17 hold the chronological detail).
Scope: Wake County NC, single-family, assessor roll (`assr_land_value`/`assr_impr_value`/
`assr_market_value`), 2026 valuation cycle. Goal: a non-circular, defensible answer to "are these
land values good?" plus headline statistics and per-parcel evidence packets.*

---

## 0. The organizing idea

Everything hangs off the identity **Total = Land + Building**. Land is the only term nobody
observes directly, so we corroborate it from **three independent directions** and check they
agree:

1. **Top-down (accounting):** if Total is right and Building is right, Land = Total − Building.
2. **Bottom-up (direct evidence):** Land vs vacant-land sales — the gold standard.
3. **Internal coherence (no external data):** land independent of buildings; building independent
   of location; uniform within like clusters; sanity bounds.

The central methodological discipline that emerged: **a validity filter must never consult the
assessment being tested** (else it's circular). This one rule reshaped half the analysis.

---

## 1. Ideas that turned out to be useful

| Idea | Why it mattered |
|---|---|
| **Three-direction corroboration** | Organizing frame; agreement = defensible, disagreement localizes the fault. |
| **Method-agnostic battery** | Scores assessor or AVM identically; the contrast is the result. |
| **Non-circularity discipline** | The single most important principle. A filter using price or the assessment can't detect over/under-assessment. |
| **Deed qualification WHITELIST** (`disq_flag ∈ {A,C}`) | The best filter: non-circular (deed code, not price), and it *preserves sample* (231 vs 633) while halving COD. |
| **"Prime buildable lot" filter** | Genuinely vacant + residential zoning + size-comparable to neighborhood built peers + sane shape + enough peers. Operationalizes "good land comp." |
| **RCN reconstruction from the cost book** (`assr_impr / percent_good`) | Unlocked the new-construction & residual evidence streams. |
| **Residual land from improved sales** (`sale − building cost`), split by depreciation | The dep≈0 stream became gold-grade under the frozen-SOV assumption — 60× the vacant sample. |
| **Two-method convergence** | Vacant $25.1/sqft ≈ rcn-residual $25.2/sqft — independent methods agreeing is the strongest validation. |
| **Clean-vs-residual in parallel** | Caught the VEI −60 artifact; is the general tool for separating real signal from mechanics. |
| **Within-neighborhood (FWL) demeaning** | Powered A1 (absorb 4,734 FEs cheaply) and the depreciation test (land flat with age within a VCS). Separates within- vs between-location effects. |
| **Market-LAND desirability proxy** (not home price) | A5 = ρ 0.911; the home-price proxy nearly produced a false inversion. |
| **Treating the frozen public SOV cost table as exogenous** | Dissolved the residual-circularity objection for the dep≈0 stream. |
| **Reading from one consistent source** | After the key-join saga, all signals from one snapshot. |
| **Reused openavmkit machinery** | `calc_cod`, `calc_ratio_stats_bootstrap`, `get_vertical_equity_scores`, `detect_sales_chasing`, `land_he_id`/`impr_he_id`. |

---

## 2. Ideas we tried and discarded (and why)

| Discarded | Why |
|---|---|
| **Naive teardown rule** (`age_at_sale < 0` alone) | Catches *new-construction* sales (price includes the new building) → bogus 0.227 ratio. Replaced by price-vs-developed-level definition (which found ~0 real teardowns in Wake). |
| **Flat `$/sqft < 5` price floor** | **Circular** — uses the sale price, the denominator under test. Replaced by the deed-code whitelist. |
| **Total-sale-price desirability proxy** (A5) | Confounded by home size/quality; nearly produced a false "land is regressive/inverted" finding (ρ −0.05). Replaced by market-land $/sqft. |
| **Large-sample VEI = −60 "regressivity"** | Residual-mechanics artifact (cheap-land decile reads an impossible 1.32). Trusted clean-vacant VEI (−10). |
| **rcnld_resid (moderate-dep) in the gold standard** | Composition (older homes in cheaper-land zones), not representative land; also depends on the depreciation judgment. |
| **Residual streams as assessor gold (pre-frozen-SOV)** | Semi-circular/masking when building cost is endogenous. Promoted only the dep≈0 stream, only under the frozen-cost stipulation. |
| **Per-parcel residual cross-check for the assessor** | Degenerate (≡1.0 by construction); meaningful only for an external series. |
| **The pickle→model-output key2 join** | 48% overlap (different/stale snapshot). Switched to one consistent source. |
| **A0 / A4 / B2 / B4 (not run)** | A0 unit-selection unnecessary for SF $/sqft; A4 is LVT-policy arithmetic, not a quality test; B2/B4 need flood/water/corner layers Wake lacks. |
| **Assessor-vs-AVM head-to-head (parked)** | `multi_mra` emits no genuine land split (`prediction_land_sqft ≡ prediction/land_area`). Needs a real `land_value`/`impr_value` export. |

---

## 3. Results of everything tested

### Final scorecard — Wake single-family assessor roll

| Test | Metric | Result | Verdict |
|---|---|---|---|
| **Step 1 — Total** | median ratio / COD / VEI | 0.983 / 6.3 / −6.8 | ✅ excellent |
| **A1 — Improvement independence** | partial-R²(impr \| nbhd FE) | 0.093 | ⚠️ mild dependence |
| **A2 — Land uniformity** | CHD within `land_he_id` | 8.9 | ✅ |
| **A2 — Building uniformity** | CHD within `impr_he_id` | 6.3 | ✅ |
| **A3 — Land vs market (GOLD)** | median / COD / $sqft / n | 0.85 / ~25 / $25.2 / 4,548 | under ~15% |
| &nbsp;&nbsp;↳ purest (vacant prime+qual) | median / COD / n | 0.833 / 21.1 / 67 | under ~17% |
| **A5 — Desirability gradient** | ρ(assr land, market land) | 0.911 | ✅ tracks market |
| **A6 — Sales chasing on land** | detector | no signal | ✅ |
| **A7 — Summation & sanity** | % violations | 0.03% | ✅ |
| **A8 — Building location-invariance** | CHD within `impr_he_id` | 6.3 (spans ~85 nbhds) | ✅ |
| **B3 — Local spatial uniformity** | Moran's I / hot-pixels | 0.15 / 2.4% | ✅ smooth |
| **Vertical equity** | VEI (clean vacant) | ≈ −10 | ⚠️ mild regressive tilt |
| **Depreciation schedule** | within-nbhd resid-land/age slope | −0.035 $/sqft·yr | ✅ well-calibrated |

### Supporting findings

- **Anchor scrutiny:** only **25%** of vacant sales are "prime" (241/960); the rest fail
  genuinely-vacant / zoning / size-comparability / shape. Qualification (deed code) does most of
  the cleaning while preserving sample.
- **Convergence:** vacant ($25.1/sqft) and dep≈0 residual ($25.2/sqft) agree to **0.4%** → true
  prime land ≈ **$25/sqft**.
- **Over-assessed tail** is dominated by **non-market low-priced sales**, not real
  over-valuation (the over-assessment is in the denominator).
- **By model group:** every group shows land under-assessed (medians 0.52–0.84); only
  single-family has a trustworthy sample.
- **VEI −60 → artifact:** the severe regressivity was residual mechanics; the mild real tilt
  (VEI ≈ −10) is within-neighborhood under-differentiation (premium lots under-assessed a bit
  more — the signature of uniform $/sqft schedules missing lot-level premiums).
- **Depreciation:** the `rcnld_resid` drift ($25→$16.6) was **composition** (older homes in
  cheaper-land zones), not under-depreciation — within-VCS, residual land is flat with age.

### The settled thesis

> Wake's single-family **totals and buildings are excellent** — uniform, location-invariant,
> correctly depreciated. **Land is uniformly applied and tracks the market gradient tightly
> (ρ=0.911), is not sales-chased, and is correctly depreciated — but is systematically ~15%
> under-assessed**, backed by 4,548 observations and two independent methods converging at
> $25/sqft, **with a mild regressive tilt** (premium lots under-assessed slightly more). The
> location premium is correctly routed to land (not building). Every exclusion rule is
> non-circular.

---

## 4. Generalizing to other jurisdictions (same data availability)

**Assumed data:** assessor land/impr/total split; cost-book fields (RCNLD + percent-good →
reconstructable RCN); deed validity/qualification codes; `vacant_sale` flag; zoning; lot geometry
(area, rectangularity); coordinates; HE clusters; neighborhood/VCS; time-adjusted sale prices.

**Transfers unchanged (the portable core):** the three-direction logic; non-circularity
discipline; prime-lot filter (size/shape/peers); two-method convergence check; clean-vs-residual
parallel runs; within-VCS depreciation test; FWL A1; by-group breakdown; the reused openavmkit
stats/equity/chasing machinery.

**Needs per-jurisdiction configuration (the seams):**
1. **Residential zoning allowlist** — `classify_zoning` is Wake/Raleigh-tuned (R-*/RX/UR/SR/GR
   incl. watershed; rural RA/RR and commercial/industrial out). Each jurisdiction needs its own
   code map (push to `settings.json`).
2. **Qualified-sale code map** — `{A,C}` is Wake's `disq_flag`; other rolls use different validity
   coding (NCDOR vs local vs deed-type). The whitelist concept transfers; the codes don't.
3. **Cost-basis fields** — RCN reconstruction needs RCNLD + percent-good. Where present (and the
   cost table is a frozen, adopted SOV), the residual streams expand the gold sample hugely; where
   absent, fall back to **vacant-only gold** (smaller, but still the purest signal).
4. **`vacant_sale` semantics** — confirm it means "sold as land" per jurisdiction.

**Coverage-dependent power:**
- **A5 (market-land gradient)** needs enough qualified vacant sales *per neighborhood* — Wake had
  only 15 such neighborhoods. Sparser markets get a weaker A5 (lean harder on the residual
  gradient and the cluster-CHD).
- **A3 gold sample** scales with vacant + (cost-book ⇒ residual) availability.
- **B2/B4** only run where flood / water-distance / corner layers exist.

**The cross-jurisdiction payoff:** the comparative scorecard. Run the identical battery across N
counties and read the contrasts — under-assessment levels, regressivity, and especially the
**A1/A5 allocation signature** (does the county route the location premium into land or building?).
That comparison is where this becomes a research instrument, not just an audit.

**Generalization risks to state honestly:** the non-circular validity (deed codes) and the cost
basis (frozen SOV) are jurisdiction-specific privileges. Without trustworthy deed codes you lose
the cleanest filter; without an adopted cost table you can't promote the residual streams to gold.
The method degrades gracefully — to vacant-only, prime-lot-filtered evidence — but the sample and
confidence shrink.

---

## 5. How to wrap up a comprehensive test suite / results

**Productize (per the spec's intent):**
- Promote the stable core (`lvi_anchors` + `lvi_battery`) into
  `openavmkit/land_value_integrity.py` with the §P4 method-agnostic signature
  `run_land_value_integrity(land, sup, settings, model_group, land_compare=None)`.
- Per-jurisdiction config (zoning allowlist, qualified codes, cost-field names) in `settings.json`.
- A rendered report (`resources/reports/land_value_integrity.md`, via the `reports.py`
  `start_report`/`set_var`/`finish_report` convention).

**Structure the suite in tiers** so the objective core leads and the data-dependent tests follow:
- **Tier 0 — Precondition:** total ratio study (don't trust a land split if the total is wrong).
- **Tier 1 — Objective core:** A1 (independence), A2 (uniformity), A3 (vs validated anchors),
  A7 (summation/sanity), A8 (building location-invariance).
- **Tier 2 — Spatial / equity:** A5 (gradient), A6 (chasing), B3 (local uniformity), vertical
  equity.
- **Tier 3 — Diagnostics:** depreciation calibration, by-model-group breakdown, anchor scrutiny.

**Two deliverables, both required:**
1. **Headline scorecard** — one metric per test, pass/warn/fail vs documented IAAO thresholds,
   *with the evidence stream and sample size attached to every number*.
2. **Per-parcel evidence packet** (the original goal / defensibility artifact) — for each parcel:
   land value, corroborating anchors (nearest qualified vacant + residual cross-check), cluster
   position, per-parcel flags, and an aggregated confidence grade (high/med/low).

**Design principles to bake in (the hard-won lessons):**
- **Non-circularity is the contract** — every filter is independent of the value under test;
  state it explicitly for each.
- **Report the evidence, not just the number** — sample size, stream, and CI travel with every
  statistic.
- **Run clean-vs-noisy in parallel** — it's how you catch artifacts (VEI −60).
- **Require two independent methods to converge** before calling a result "confident."
- **Separate within-location from between-location** (FWL/VCS demeaning) — composition masquerades
  as signal otherwise.
- **Prefer codes over prices, and the right proxy** — deed codes beat price floors; market-land
  beats home-price for desirability.
- **State caveats as first-class output** — sample sizes, confounds, single-jurisdiction scope.

**Meta-lessons (the five that generalize beyond appraisal):**
1. Circularity is the central threat in self-referential data; design every test against it.
2. Proxy choice can *invert* a finding — validate the proxy before believing the result.
3. Composition vs. within-group effects must be separated or you'll attribute the wrong cause.
4. Artifacts hide in derived quantities (residuals, ratios) — sanity-check the extremes.
5. The strongest evidence is *convergence of independent methods*, not the precision of one.

---

## Appendix — runnable scripts (research/)

`lvi_anchors.py` (evidence streams + filters) · `lvi_battery.py` (A1/A2/A3/A7/A8 + total) ·
`lvi_wake.py` (runner + scorecard + per-parcel packet) · `lvi_land_ratio_by_group.py` ·
`lvi_vacant_scrutiny.py` · `lvi_residual_smell.py` · `lvi_evidence_pool.py` ·
`lvi_teardown_explore.py` · `lvi_extra_tests.py` (A5/A6/B3) · `lvi_land_vertical.py` ·
`lvi_depreciation.py`. Outputs in `out/lvi/`.
