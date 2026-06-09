# P0 — OpenAVMKit: An Open-Source, Auditable, Reproducible Toolkit for Mass Appraisal

> **Status:** first draft (anchor / tool paper). Target venue: IAAO *Journal of Property Tax
> Assessment & Administration (JPTAA)*. This is the citable reference every later paper (P1–P5)
> points at. Effort kept low-medium by design — it synthesizes existing documentation,
> [AGENTS.md](../../AGENTS.md), and the pipeline notebooks rather than introducing new method.

---

## Abstract (draft)

Computer-assisted mass appraisal (CAMA) increasingly relies on machine-learning automated valuation
models (AVMs), yet the production systems that set taxable values are typically proprietary, opaque,
and non-reproducible. This opacity is not merely an academic concern: documented assessment
regressivity persists in part because the public, taxpayers, and even oversight bodies cannot
inspect, re-run, or audit the models that produce assessed values. We present **OpenAVMKit**, an
open-source Python toolkit for mass appraisal whose central design commitment is *auditability*. Its
pipeline is organized into three deliberately separated stages — **assemble** (establish facts),
**clean** (state assumptions), and **model** (produce predictions) — each checkpointed so that the
factual record, the analyst's discretionary choices, and the resulting valuations are inspectable in
isolation. A single declarative `settings.json` file fully specifies a run, so a third party can
reproduce a jurisdiction's valuation from published inputs. The toolkit ships roughly twenty
modeling engines (linear MRA, multi-MRA, GWR, kernel regression, gradient-boosted trees, a
probabilistic NGBoost model, an interpretable layered-comps model, spatial-lag predictors,
pass-through/assessor references, and deliberately weak baselines) under a uniform interface, plus a
full IAAO-standard equity-analysis suite (COD, PRD, PRB, with horizontal- and vertical-equity
studies). We argue that an open, auditable, settings-reproducible pipeline is itself a methodological
contribution to the assessment field, and we position OpenAVMKit as shared research infrastructure
on which reproducible mass-appraisal studies can be built.

---

## 1. Motivation: opacity as an equity problem

- The mass-appraisal field has converged on ML AVMs (tree ensembles especially) for production
  valuation, but the models that determine taxable value are overwhelmingly proprietary.
- Tie to the regressivity literature (Berry, *Reassessing the Property Tax*; Atuahene & Berry 2018):
  when assessments are systematically regressive, the inability to audit the model is part of why the
  problem is hard to diagnose and harder to litigate.
- The IAAO *Standard on Mass Appraisal of Real Property* and *Standard on Ratio Studies* prescribe
  what good assessment looks like (uniformity statistics, ratio studies) but the tooling to *produce
  and check* those statistics reproducibly is fragmented and often in-house.
- Thesis: **transparency, auditability, and reproducibility are first-class methodological goals**,
  not just software-engineering niceties. A toolkit that bakes them into its architecture is a
  contribution in its own right.

## 2. Design principles

The argument of the paper is that three architectural commitments, taken together, make a mass-
appraisal pipeline auditable. Each maps to a concrete feature of OpenAVMKit.

### 2.1 Separation of facts, assumptions, and predictions

The pipeline is split into three checkpointed stages with distinct epistemic character (this framing
is canonical in the repository, [notebooks/README.md](../../notebooks/README.md)):

| Stage | Notebook | Epistemic character | Output |
|---|---|---|---|
| **Assemble** | `01-assemble.ipynb` | *Facts* — where parcels are, what features they have, what sold for how much | a `SalesUniversePair` of factual assertions |
| **Clean** | `02-clean.ipynb` | *Opinions* — how to fill missing values, which sales to trust, how to time-adjust | a `SalesUniversePair` of reasoned assumptions over the facts |
| **Model** | `03-model.ipynb` | *Predictions* — educated guesses derived from facts + assumptions | per-group predictions, params/contribs, reports |
| **Assessment quality** | `assessment_quality.ipynb` | *Evaluation* — predictions vs. observed sales | quality/ratio/equity reports |

Why this matters for auditability: a reviewer can challenge an *assumption* (e.g. a sales-validity
rule or a missing-value fill) without re-litigating the *facts*, and can inspect the facts without
being entangled in the model. Each stage resumes from the prior stage's checkpoint, so the boundary
is enforced, not just conventional.

### 2.2 Declarative, reproducible configuration

- A run is driven by one `settings.json` per locality (`notebooks/pipeline/data/<slug>/in/`).
- The settings preprocessor ([openavmkit/utilities/settings.py](../../openavmkit/utilities/settings.py))
  adds the features a real audit needs without leaving JSON: `__`-prefixed comment keys, `$$path`
  variable references (one source of truth for repeated thresholds/column lists), template merging
  with `!` (stomp) and `+` (extend) operators. Documented in [AGENTS.md §2](../../AGENTS.md) and
  [docs/docs/advanced_settings.md](../../docs/docs/advanced_settings.md).
- Consequence: **the settings file plus the input manifest fully determines the output.** A third
  party can re-run a jurisdiction's valuation and get the same numbers — the reproducibility claim
  that anchors the whole paper program.

### 2.3 A uniform model interface with built-in baselines and references

- ~20 engines behind one dispatch ([openavmkit/model_runner.py](../../openavmkit/model_runner.py),
  classes in [openavmkit/utilities/modeling.py](../../openavmkit/utilities/modeling.py)); see the
  full menu in [docs/docs/models_reference.md](../../docs/docs/models_reference.md).
- Crucially, the menu includes **reference** models (`assessor`, `pass_through`, `ground_truth`) and
  **deliberately weak baselines** (`naive_area`, `mean`/`median`, `garbage`). Auditability means you
  always evaluate a candidate model against the incumbent assessor *and* against a floor — the
  framework makes that the default, not an afterthought.
- Every engine emits the same two per-feature artifacts per subset — `params_<subset>.csv`
  ("per-unit effect of each feature") and `contributions_<subset>.csv` ("how much each feature
  contributed to this row") — so interpretability is uniform across linear, tree, GWR, and ensemble
  models ([AGENTS.md §7](../../AGENTS.md)).

## 3. The toolkit in depth

### 3.1 Data assembly and enrichment
- Loading/joining raw tabular + geospatial inputs; sales↔parcel join; model-group tagging.
- Optional enrichment sources (census, streets, distances, Overture, DEM/elevation, spatial lag),
  each opt-in and gated in settings — relevant to the auditability story because enrichment is
  explicit and reproducible, never hidden.
- The assessor-vs-GIS land-area precedence rule as a worked example of "facts" discipline
  ([AGENTS.md §4](../../AGENTS.md)).

### 3.2 Cleaning: where assumptions live
- Missing-value handling, equity clustering, sales-validity processing, sales-scrutiny heuristic
  (forward-reference to P3), time adjustment (forward-reference to its own engine).
- Emphasize the **flag-vs-exclude** discipline: questionable sales are marked, not silently dropped,
  preserving the audit trail.

### 3.3 Modeling: the engine menu
- Walk the categories from `models_reference.md`: production predictive (MRA, multi-MRA, GWR, kernel,
  XGBoost/LightGBM/CatBoost, NGBoost, layered-comps, spatial-lag), references, baselines, ensembles
  (median/mean greedy-selected, or per-location `local`).
- Forward-reference the novel engines that get their own papers: **layered comps** (P1),
  **NGBoost + uncertainty attribution** (P2). Here they are presented only as menu items.

### 3.4 Evaluation and equity analysis
- Ratio studies and the IAAO uniformity suite: COD, PRD, PRB (and CHD/VEI) with bootstrap CIs
  ([openavmkit/ratio_study.py](../../openavmkit/ratio_study.py),
  [openavmkit/utilities/stats.py](../../openavmkit/utilities/stats.py),
  [openavmkit/horizontal_equity_study.py](../../openavmkit/horizontal_equity_study.py),
  [openavmkit/vertical_equity_study.py](../../openavmkit/vertical_equity_study.py)).
- Rolling-origin cross-validation ([openavmkit/tuning.py](../../openavmkit/tuning.py)) as the honest
  temporal evaluation protocol — no peeking across the valuation date.

## 4. Illustration (scoped: Eagle CO + Petersburg VA)

> Per the program plan, the anchor paper uses **light** data — one or two wired-up example
> jurisdictions for illustration only; no large benchmark is needed here (that is P1's job).

- **Eagle County, CO** — mountain resort market (Vail); showcases DEM/elevation enrichment and a
  high-variance, amenity-driven value surface.
- **Petersburg City, VA** — small urban market; showcases `collapse_sparse_categories` handling and
  a modest, more typical assessment jurisdiction.
- For each: show the three-stage outputs, a representative model-menu run with the assessor and a
  naive baseline included, and the IAAO uniformity table. The point is *reproducibility and
  auditability demonstrated end-to-end*, not a model bake-off.

## 5. Related work
- CAMA / AVM practice and the shift to ML ensembles.
- Reproducibility and open science in applied econometrics / real estate (the gap: open-source,
  end-to-end auditable AVM tooling is nearly absent).
- IAAO standards as the normative backdrop.

## 6. Discussion
- What auditability buys: appeals defensibility, oversight, cross-jurisdiction comparability,
  and a substrate for reproducible research (P1–P5 all build on this).
- Limitations: not a hosted service; requires data assembly; the toolkit encodes opinions in its
  defaults (which is *why* the facts/assumptions split matters).
- The paper program: this anchor → interpretable comps (P1) → uncertainty attribution (P2) →
  automated sales scrutiny (P3) → condo resolution (P4) → cross-jurisdiction equity audit (P5).

## 7. Reproducibility appendix (template)
- Published `settings.json` for each illustrated jurisdiction + input manifest.
- "Re-run these cells" note mapping to the three notebooks.
- Toolkit version / commit hash; `pytest tests/` green.

---

## Draft notes / open items

- **Co-authorship & venue norms:** confirm JPTAA submission norms; decide co-authors (Center for
  Land Economics; potential Lee County FL team for later papers).
- **Figure list:** (1) three-stage architecture diagram (facts→assumptions→predictions, with
  checkpoints); (2) a `settings.json` excerpt annotated with preprocessor features; (3) Eagle +
  Petersburg uniformity tables; (4) the uniform params/contribs output schema.
- **Citations to pull at write time:** IAAO *Standard on Mass Appraisal*, *Standard on Ratio
  Studies*; Berry (*Reassessing the Property Tax*); Atuahene & Berry 2018; CAMA/AVM survey refs;
  reproducibility-in-science refs.
- **Keep it light:** resist turning this into a methods paper — its job is to be the clean, citable
  description of the instrument. Depth goes in P1–P5.
