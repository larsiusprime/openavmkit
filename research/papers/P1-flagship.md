# P1 (FLAGSHIP) — Interpretable Comparable-Sales Valuation: Layered Comps with Exact Tree-SHAP

> **Status:** full draft skeleton with method detail grounded in code. Target venue: lead in
> *Journal of Real Estate Finance & Economics (JREF)* or *Journal of Real Estate Research (JRER)*;
> optional condensed applied companion in IAAO *JPTAA*. This is the technical centerpiece of the
> paper program. Verification numbers are in
> [../verification/shap_correctness.md](../verification/shap_correctness.md); the empirical table is
> produced by [../benchmark/run_benchmark.py](../benchmark/run_benchmark.py) on Petersburg + Eagle.

---

## Abstract (draft)

The comparable-sales (comps) approach is the valuation paradigm assessors trust and appeals boards
demand, but in automated mass appraisal it has historically been either *manual* (an analyst picks
and adjusts comps) or *implicit and opaque* (buried inside black-box gradient-boosted ensembles
whose feature importances are at best post-hoc). We present **Layered Comps (lcomp)**, a regression
model that formalizes the appraiser's comparable-sales intuition as a bagged ensemble of hierarchical
trees: each parcel's value is a falloff-weighted average of robust (Wilson-trimmed) sale-price means
along a root-to-terminal path of progressively-narrower comparable cohorts. Crucially, layered comps
admit an **exact, additive, per-feature SHAP decomposition**. We show that, although a layered-comp
tree predicts from a *path blend* rather than a single leaf, the blend can be *folded* into a
per-terminal-node value, after which the exact path-dependent tree-SHAP algorithm of Lundberg et al.
(2019, Algorithm 2) applies directly. Our construction additionally handles two cases the standard
SHAP tree format cannot represent: **categorical one-vs-rest splits** and **early stopping** on
missing values. We verify exactness against a brute-force Shapley oracle (agreement to ~1e-14) and
confirm additivity (`base + Σφ = prediction`) to machine epsilon. Empirically, on diverse U.S.
jurisdictions, layered comps are competitive on accuracy and IAAO uniformity (COD/PRD/PRB) with
leading black-box ensembles while remaining *natively interpretable in comps terms* — each prediction
reconstructs as an explicit, weighted set of comparable cohorts and an additive set of feature
contributions, directly usable in assessment appeals.

---

## Core claims (each defended in the paper)

- **C1 — Layered comps formalize the comparable-sales intuition as a bagged tree ensemble.** A tree
  is built by recursively choosing the split (numeric threshold via binary search, ≥8 samples; or
  categorical one-vs-rest) that minimizes the weighted child-to-parent error ratio (MAE or MSE).
  Each node stores a **Wilson-trimmed mean** of its cohort's sale prices (trim the top/bottom 2.5%,
  average the middle 95%; `calculate_wilson_mean`). A parcel's prediction is the **falloff-weighted
  average of the Wilson means along its root→terminal path**: node *i* of an *n*-node path gets
  weight `(1 − x_i)^falloff` with `x_i = (n−1−i)/(n−1)`, normalized (`_predict_row`,
  [layeredcompmodel/model.py]; replicated for SHAP in `_fold_path_value`,
  [openavmkit/shap_analysis.py:217](../../openavmkit/shap_analysis.py#L217)). Higher `weight_falloff`
  concentrates weight on the narrowest (leaf) cohort; lower falloff blends toward broad-market means.
  Bagging averages over bootstrap-sampled trees.
- **C2 — The model admits an *exact* path-dependent tree-SHAP decomposition.** Because the path-blend
  is a deterministic function of *which* terminal node a row reaches, we **fold** it into a single
  value per terminal node (`_fold`, `_FoldedNode` in
  [openavmkit/shap_analysis.py](../../openavmkit/shap_analysis.py)), turning each layered-comp tree
  into an ordinary regression tree on which Lundberg et al. (2019) Algorithm 2 (the EXTEND/UNWIND
  recursion) is exact. The construction natively handles **categorical one-vs-rest splits** (hot/cold
  child by equality) and **early stopping** (a NaN at a numeric split, or a routed-to child whose
  partition was empty, emits the *current* node's folded value) via synthetic cover-weighted "stop"
  leaves — neither expressible in SHAP's numeric-threshold-only tree format. Additivity holds to
  floating point.
- **C3 — Verified correctness.** Against an independent brute-force Shapley oracle, the fast SHAP
  agrees to **max ~4.6e-14** including the categorical and NaN-early-stop cases, and
  `base_value + Σφ` reconstructs the ensemble prediction to **max ~5.7e-14**; the numba kernel and
  the pure-Python fallback agree to ~1.4e-14. (Test suite:
  [tests/test_lcomp_shap.py](../../tests/test_lcomp_shap.py); numbers in
  [../verification/shap_correctness.md](../verification/shap_correctness.md).)
- **C4 — Under honest *temporal* evaluation, lcomp matches or beats the production GBMs.** On
  **rolling-origin CV at production-realistic tuning (n_trials = 5)**, lcomp has the **best-or-tied
  trimmed COD of any model** on both jurisdictions (Petersburg 14.2 vs LightGBM 15.3 / XGBoost 15.7 /
  MRA 16.1; Eagle 21.2 vs LightGBM 23.0 / MRA 24.6 / **XGBoost 28.7, unstable ±12.4**), with
  well-centered median ratios (0.90 / 0.99). **Crucially, this overturns the in-period picture**: a
  contemporaneous study set ranks GBMs far ahead (study-set COD ≈ 7 vs lcomp ≈ 13), but that edge is
  *overfitting that doesn't survive forward prediction* — every model degrades from in-period to
  temporal, the GBMs most. So the defensible claim is **not** "lcomp trades accuracy for
  interpretability" but "**lcomp is competitive-to-best on forward generalization, and the GBMs'
  apparent superiority is an artifact of in-period evaluation**" (the protocol-reordering point — a
  methods contribution in its own right). *Caveats:* within fold-to-fold noise vs LightGBM; one group
  per jurisdiction; XGBoost+LightGBM so far (CatBoost/NGBoost + the n_trials=5 record run will
  finalize).
- **C5 — Decision-useful for appeals.** The model is interpretable two complementary ways: (i) the
  native `explain_value` trace returns the explicit comparable cohorts, their counts, Wilson means,
  and path weights, plus the exact arithmetic that produced the value ("these comps, these
  weights"); and (ii) exact SHAP gives an additive per-feature attribution that reconstructs the
  prediction. Both are per-parcel and reproduce the model output exactly.
- **C6 — Misses are diagnosable and actionable (errors carry a built-in next step).** Every lcomp
  prediction terminates at a node with a known comp-cohort size (`CompNode.count`) and a
  Wilson-trimmed mean, so a large residual is *mechanically classifiable from the parcel's
  root→terminal path* into exactly two regimes, and the path tells you which:
  - **Thin/no comps** — the terminal node has few sales, or the path *early-stopped* (NaN at a numeric
    split, or a routed-to child whose partition was empty). The estimate rests on a small or broad
    cohort. → *Action:* acquire/borrow comps; treat as high-uncertainty.
  - **Anomalous-in-cluster** — the path reached a *well-populated* node (ample comps) but the
    subject's own sale lies outside the cohort's 2.5–97.5 trimmed band (precisely what
    `calculate_wilson_mean` trims). → *Action:* review the sale (atypical / non-arm's-length),
    re-check recorded characteristics, or suspect an unobserved variable / data error.

  Black-box ensembles emit a residual but not this triage. Two testable corollaries: (a) part of
  lcomp's COD gap is *honest exposure of bad sales* — large residuals at high-`count` nodes are
  candidate non-arm's-length / mis-keyed sales that GBMs silently absorb and train on; checkable by
  cross-referencing against the **sales-scrutiny flags (P3)**. (b) The thin-comps regime *is* a
  high-predictive-uncertainty region — lcomp surfaces it structurally, NGBoost quantifies it
  (**P2**); the two are complementary.

---

## Section skeleton (draft)

### 1. Introduction
- Comps are the dominant, trusted, legally-defensible valuation paradigm; assessment appeals are
  argued in comps terms.
- The accuracy–interpretability tension: production AVMs have moved to tree ensembles, whose
  explanations are post-hoc and whose internal logic is not a comps narrative.
- Contribution: a model that is *natively* a comps method **and** admits *exact* additive
  attribution — not an approximation, not a surrogate.

### 2. Related work
- Gradient-boosting AVMs (saturated); SHAP-for-importance applied to AVMs (common).
- Automated comparable-sales selection: **sparse**; *interpretable* comps automation: essentially
  unpublished — state this thinness explicitly; it is the gap the paper fills.
- Assessment regressivity & appeals (Berry; Atuahene & Berry 2018) as the motivating application.
- SHAP foundations: Lundberg & Lee 2017; Lundberg et al. 2019 (consistent tree attribution).

### 3. Method: Layered Comps
- Tree construction: split criterion (weighted child/parent error ratio, MAE/MSE; numeric
  binary-search thresholds with ≥8-sample guard; categorical one-vs-rest, NaN as its own category).
- Wilson-trimmed node means (robustness to outlier sales).
- Falloff path-weighting; interpretation of `weight_falloff` (leaf-concentration vs. market-blend).
- Bagging over bootstrap samples.
- Contrast with: plain CART (leaf value only), kNN/kernel comps (no learned hierarchy),
  black-box GBMs (no comps semantics).

### 4. Method: Exact Tree-SHAP for Layered Comps
- The folding argument: path-blend → per-terminal-node folded value → ordinary regression tree.
- Statement: with folding, Lundberg et al. (2019) Alg. 2 yields exact Shapley values for lcomp;
  proof sketch via the cover-weighted conditional-expectation game on the folded tree.
- Categorical one-vs-rest handling (feature-identity bookkeeping is split-rule-agnostic).
- Early-stop handling: synthetic stop leaves with cover = `parent.count − Σ child.count`, keeping
  both the prediction and the cover-weighted base value exact.
- Ensemble: mean of per-tree SHAP, base = mean of per-tree bases (mirrors bagging `predict`).
- Complexity; the numba kernel; the pure-Python reference path.

### 5. Verification
- Brute-force Shapley oracle (enumerate coalitions, cover-weighted conditional expectation).
- Results table lifted from the test suite (numeric, categorical, NaN-early-stop, numba-vs-python,
  expected-value identity) — agreement and additivity at ~1e-14 (see verification note).
- Frame as: "exact" is meant literally, to machine precision, not approximately.

### 6. Empirical study
- Backbone: Petersburg VA + Eagle CO (single-family model group), versioned settings.
- **Evaluation protocol: temporal rolling-origin CV** (claim-grade Flavor B) via the orchestrator
  [research/cross_validation.py](../cross_validation.py): per backdated origin tᵢ, filter to sales
  ≤ tᵢ *before* the clean stage (leakage-clean — time-adjustment + variable-selection see only the
  past, asserted per fold), train, then score the held-out (tᵢ, tᵢ+1yr] window against **raw
  `sale_price`** via the library's `run_ratio_study` (native `valid_for_ratio_study` / vacant-improved
  split; IAAO §4.4). Per-fold output isolation; origins capped at the most-recent 5; tuning at
  **n_trials = 5 (production-faithful — diminishing returns beyond 5, per maintainer)**.
- Models (this run): assessor (incumbent roll), MRA, lcomp, XGBoost, LightGBM. CatBoost/NGBoost + the
  n_trials=5 record run pending.
- Metrics: MAPE/R² and IAAO COD/PRD/PRB (trimmed + untrimmed), VEI, per model group.
- Ablations: `weight_falloff` sweep; Wilson-trim on/off; bagging tree count.
- *(Tables: [research/benchmark/<slug>/cv/<group>/benchmark_cv.md](../benchmark/).)*

> **HEADLINE RESULT — temporal rolling-origin CV, mean ± std COD_trim (lower = better):**
>
> | model | Petersburg `single_family_suburban` (4 folds) | Eagle `single_family` (5 folds) |
> |---|---|---|
> | **lcomp** | **14.2 ± 3.5** | **21.2 ± 1.8** |
> | LightGBM | 15.3 ± 1.1 | 23.0 ± 3.0 |
> | XGBoost | 15.7 ± 2.9 | 28.7 ± 12.4 |
> | MRA | 16.1 ± 4.4 | 24.6 ± 2.4 |
> | assessor (in-period roll †) | 15.7 ± 3.1 | 11.5 ± 2.3 |
>
> **lcomp is the best-or-tied model on both jurisdictions** — across a small urban market (Petersburg)
> and a high-variance resort market (Eagle). On Eagle, **XGBoost is the *worst* model and wildly
> unstable forward (28.7 ± 12.4)**; lcomp's Wilson-trimming + bagging is most robust exactly where the
> trees overfit. Median ratios are well-centered (lcomp 0.90 / 0.99).
>
> **C4, restated and supported:** under *honest temporal evaluation at production-realistic tuning
> (n_trials = 5)*, lcomp **matches or beats the production GBMs on forward prediction** — it does not
> trade accuracy for interpretability here.
>
> **The protocol-reordering finding (ties to §8 / the Q2 result).** This *overturns* the naive
> in-period picture. A contemporaneous repeated-holdout study set (time-adjusted target) ranks the GBMs
> far ahead — study-set COD_trim ≈ 7 vs lcomp ≈ 13 on Petersburg — but that edge is **overfitting that
> does not survive forward prediction**: every model degrades from in-period to temporal, the GBMs
> *most* (Petersburg XGBoost ≈ 7 → 15.7; lcomp ≈ 13 → 14.2). **In-period evaluation flatters black
> boxes; rolling-origin reveals the truth.** That methods point holds regardless of the exact ranking,
> and is itself a contribution (P0/P1).
>
> † *assessor* is the existing in-period assessment roll (stale valuation date), **not** a model
> trained ≤ tᵢ; its low COD is the incumbent baseline, not a forward-prediction competitor.
>
> **Caveats:** lcomp-vs-LightGBM trimmed overlaps within fold-to-fold noise (lcomp's lower mean +
> tighter std favor it — "best-or-tied," not a blowout); one model group per jurisdiction; XGBoost +
> LightGBM only so far (CatBoost/NGBoost pending); Eagle's most-recent fold (2026-01) is effectively
> in-period (train = all sales). The n_trials=5 record run + CatBoost/NGBoost will finalize this table.

### 7. Interpretability in practice
- A worked per-parcel explanation from `explain_value`: the comparable cohorts, counts, Wilson means,
  weights, and the reconstructing arithmetic — presented as an appeals exhibit.
- The same parcel's SHAP waterfall (additive feature attribution) alongside it.
- Global view: SHAP beeswarm (`plot_full_beeswarm`, [openavmkit/shap_analysis.py]) across the
  jurisdiction.
- Discussion: the two explanation modes answer different questions ("which comps?" vs. "which
  features, how much?") and are mutually consistent (both reconstruct the prediction).

#### 7.1 Actionable misses — error triage (C6)
- Every miss is classifiable from the parcel's path into **thin/no comps** (low terminal `count` /
  early-stop) vs. **anomalous-in-cluster** (well-populated node, subject outside the Wilson-trimmed
  band) — and the path tells you which, with a concrete next action for each.
- Figure: per-parcel **residual vs. terminal-node `count`**, points colored by within-/outside-the
  trimmed band — visually partitions misses into the two regimes (thin-comps = low count; anomalous =
  high count + outside band).
- **Validation against P3:** cross-reference the anomalous-in-cluster misses with the independent
  sales-scrutiny flags; quantify overlap (precision/recall). If large lcomp residuals at high-count
  nodes are disproportionately flagged sales, that *empirically* supports "the COD gap is partly
  honest bad-sale exposure," not modeling weakness.
- **Link to P2:** the thin-comps set is a structurally-identified high-uncertainty region; compare it
  to NGBoost's per-parcel `prediction_std` (do thin-comp parcels carry higher predicted std?).
- Assessor-workflow framing: lcomp turns "the model was wrong here" into a *triaged worklist* —
  get-more-comps vs. review-this-sale/characteristics — which black-box residuals do not provide.

### 8. Discussion / limitations / reproducibility
- When lcomp wins vs. when a GBM edges it; the interpretability premium and who pays for it.
- **The accuracy gap reconsidered (C6):** raw COD understates lcomp because part of its residual is
  *diagnostic* — anomalous-in-cluster misses flag bad sales/data rather than modeling failure, and
  GBMs hide these by absorbing them. The right comparison for mass appraisal may be "accuracy + an
  actionable error worklist," not COD alone.
- **Honest error & the data-quality flywheel.** lcomp is *architecturally* resistant to "heroically"
  fitting anomalies: Wilson trimming removes a cohort's tails from the value it's compared against, and
  falloff path-blending regularizes toward broader-market means — so lcomp predicts robust central
  tendencies and leaves a large, honest residual on the outlier rather than memorizing it. A GBM can
  spend capacity carving out and fitting such points, which inflates apparent (especially in-sample)
  accuracy. **Decompose lcomp's COD gap into (a) honest exposure** (residuals that *should* be large —
  bad/atypical sales) **and (b) genuine expressiveness shortfall** (hierarchical comps capture
  high-order interactions less flexibly than GBMs). Do *not* claim the whole gap is (a); C6's triage
  lets us *measure* the (a)/(b) split. The flywheel: lcomp not only errs honestly, it *says what to
  fix* (bad sale vs. thin comps) — fixing it cleans the data and lifts **every** model, GBMs included
  (they stop wasting capacity on the corrected sale). Defensible claim: lcomp is the better
  *diagnostic*; the GBM may remain the better *predictor*; lcomp's diagnostics raise the ceiling for
  both. Caveat: cleaning shrinks (a) for everyone but (b) persists, so the gap need not close — "both
  improve" ≠ "gap closes." Tested by the clean-and-retrain ablation (Figure 8).
- Limitations: tree instability mitigated by bagging; falloff/trim are hyperparameters; comps
  semantics assume a meaningful similarity hierarchy.
- Reproducibility appendix: settings files + manifest (from the benchmark harness) + `pytest`.

---

## Figures / tables
1. Folded-tree schematic (path-blend → folded leaf value → regression tree SHAP applies).
2. Accuracy × uniformity benchmark table (lcomp vs. baselines; Petersburg + Eagle).
3. `weight_falloff` / Wilson-trim ablation curves.
4. Oracle-vs-fast SHAP scatter (on the diagonal, residual ~1e-14).
5. Per-parcel `explain_value` comps exhibit + SHAP waterfall (same parcel).
6. Global SHAP beeswarm.
7. Error-triage scatter: residual vs. terminal-node `count`, colored by inside/outside the Wilson
   band — partitions misses into thin-comps vs. anomalous-in-cluster regimes (C6).
8. Clean-and-retrain ablation: COD before/after removing lcomp-flagged + P3-confirmed bad sales,
   for lcomp *and* each GBM (does cleaning lift all models?).

## Key citations (verify/expand at write time)
Lundberg et al. 2019 (consistent individualized tree attribution); Lundberg & Lee 2017 (SHAP);
Berry (*Reassessing the Property Tax*); Atuahene & Berry 2018; IAAO *Standard on Mass Appraisal* and
*Standard on Ratio Studies*; recent JREF/JRER interpretable-ML AVM work (2023–2025). The
comps-automation reference list is deliberately thin — that thinness *is* the gap, and the paper
should say so.

## Open items / next steps
- ✅ §6 table filled from 5-fold repeated-holdout CV on Petersburg `single_family_suburban` + Eagle
  `single_family` (consistent ordering GBMs < lcomp < linear/incumbent).
- Promote evaluation to **Flavor B (true rolling-origin)** for the claim-grade table (per-fold
  time-adjustment recompute); current numbers are Flavor A (repeated holdout).
- **Build the C6 error-triage analysis** (residual vs. terminal `count` + Wilson-band membership) and
  the **clean-and-retrain ablation** — the strongest empirical payload; bridges to P3.
- Pick the worked-example parcel for §7 (instructive comp path + clean additivity check).
- Decide co-authors / which jurisdiction leads the empirical narrative.
