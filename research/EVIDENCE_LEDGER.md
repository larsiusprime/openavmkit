# Research Evidence Ledger

*Last updated: 2026-06-09.* Maps each paper/claim to the evidence actually produced, with status and
source. Durable record so we don't re-litigate "what's backed by what" each session. Status tags:
**VERIFIED** (claim-grade, reproducible) · **STRONG** (solid but lean/one-shot) · **DIRECTIONAL**
(established trend, needs finalizing) · **GAP** (claimed, not yet evidenced) · **UNTOUCHED**.

---

## Evidence assets produced

| Asset | Location | What it is |
|---|---|---|
| Exact tree-SHAP verification | [verification/shap_correctness.md](verification/shap_correctness.md), [tests/test_lcomp_shap.py](../tests/test_lcomp_shap.py) | lcomp SHAP ≡ brute-force Shapley oracle to ~1e-14; additivity ~1e-14; categorical + NaN-early-stop + numba≡pure-python; 8/8 pass |
| Temporal rolling-origin CV harness | [cross_validation.py](cross_validation.py) | Leakage-clean (≤tᵢ pre-clean filter, asserted), per-fold isolated, IAAO-faithful scoring via `run_ratio_study`; params: `--run`, `--n-trials`, `--max-origins`, `--jobs` |
| Benchmark harness (single-split) | [benchmark/run_benchmark.py](benchmark/run_benchmark.py) | Reuses `model_runner` to emit the uniform COD/PRD table; Flavor-A repeated-holdout prototype |
| Two-jurisdiction temporal CV results | `notebooks/pipeline/data/<slug>/out/cv/` (gitignored, regenerable) | Petersburg `single_family_suburban` (4 folds) + Eagle `single_family` (5 folds), lean menu, n_trials=3 |
| Paper drafts | [papers/P0-anchor.md](papers/P0-anchor.md), [papers/P1-flagship.md](papers/P1-flagship.md) | P0 anchor skeleton; P1 flagship w/ C1–C6 + §6 temporal-CV result |

**Key empirical numbers (temporal rolling-origin, mean ± std COD_trim, lower=better):**

| model | Petersburg (4 folds) | Eagle (5 folds) |
|---|---|---|
| **lcomp** | **14.2 ± 3.5** | **21.2 ± 1.8** |
| LightGBM | 15.3 ± 1.1 | 23.0 ± 3.0 |
| XGBoost | 15.7 ± 2.9 | 28.7 ± 12.4 |
| MRA | 16.1 ± 4.4 | 24.6 ± 2.4 |
| assessor (in-period roll, not a forward competitor) | 15.7 ± 3.1 | 11.5 ± 2.3 |

Tuning: `n_trials=3 ≈ production 5` (maintainer: diminishing returns beyond 5). Menu: xgb+lgbm only.

---

## P1 — Interpretable Comparable-Sales (lcomp + exact tree-SHAP) [FLAGSHIP]

| Claim | Evidence | Status |
|---|---|---|
| C1 — lcomp formalizes comps (falloff-weighted Wilson-trimmed path means, bagged) | Method description from code (`layeredcompmodel`, `shap_analysis.py`) | Ready (method section) |
| C2 — exact path-dependent tree-SHAP (categorical + early-stop) | Verified vs oracle to ~1e-14 | **VERIFIED** |
| C3 — verified correctness + additivity | Test suite, machine-epsilon agreement | **VERIFIED** |
| C4 — competitive accuracy vs production GBMs | Temporal CV, 2 jurisdictions, lean menu (xgb+lgbm, n_trials=3≈5) | **DIRECTIONAL** → record run finalizes |
| C5 — decision-useful for appeals (`explain_value` exhibit) | None produced yet | **GAP (quick win)** |
| C6 — diagnosable misses (thin-comps vs anomalous-in-cluster) | Mechanism argued; triage analysis not built | **GAP (the real hole)** |
| *Protocol-reordering* (emergent) — in-period flatters GBMs; honest temporal CV reorders, lcomp best-or-tied | 2 jurisdictions; Q2 (in-period 0.96/COD13 vs temporal 0.90/COD20); IAAO App E.3 = split-sample | **STRONG — likely the headline** |

**Reading:** the technical core (C2/C3) is bulletproof; the empirical hook (protocol-reordering + C4)
is strong and honest but lean (xgb+lgbm, single group/jurisdiction, n_trials=3); C5/C6 are the two
gaps between this and a complete P1.

**C6 splits into two claims with different data setups** (clarified 2026-06-09):
- *C6a — misses are diagnosable (taxonomy).* Run on **cleaned** sales (production setting):
  per-parcel residual vs terminal-cohort `count` + Wilson-band membership → thin-comps (low count /
  early-stop) vs anomalous-in-cluster (high count, outside band). Descriptive exhibit, no ground truth.
- *C6b — the anomalous signal is a valid bad-sale detector.* Needs bad sales **present** — cannot use
  cleaned sales, because `cleaning.py:142` physically drops `valid_sale==False` rows, so every
  scrutiny-caught sale is already gone (cross-ref against scrutiny flags on cleaned data is empty by
  construction). Setups: (1) **agreement** — keep flagged sales (the `flagged` col is set at
  sales_scrutiny_study.py:166 *before* the drop; intercept it or re-run scrutiny on `1-assemble`),
  label = scrutiny `flagged`; the *interesting* result is the complement `lcomp-flagged ∧
  scrutiny-missed` = P1↔P3 complementarity (cluster-scrutiny is blind without a cluster). (2)
  **injection** — inject synthetic bad sales into cleaned data, refit, measure precision/recall; the
  cleanest self-contained number, independent of scrutiny correctness. (3) external deed/validity
  codes (data-dependent). Use **out-of-bag residuals** (score each sale from trees that didn't train
  on it) so a bad sale doesn't explain itself; Wilson 2.5% trim already limits its leverage.

---

## P0 — Open/Auditable/Reproducible toolkit [ANCHOR]

- Draft skeleton exists ([papers/P0-anchor.md](papers/P0-anchor.md)); 3-stage facts→assumptions→
  predictions architecture, model menu, equity suite. **Ready (synthesis).**
- **Strengthened this session:** the reproducible leakage-clean rolling-origin CV harness +
  IAAO §4.4 / App E.3 grounding (rolling-origin = IAAO's own sales-chasing split-sample test) +
  the documented "two evaluation paths" and all-sales-preprocessing-optimism caveats (AGENTS.md §4).
  These make the auditability thesis concrete. **STRONG support.**

---

## P2–P5

| Paper | Status | Notes |
|---|---|---|
| P2 — uncertainty attribution (NGBoost log-std SHAP) | **UNTOUCHED** | no ngboost CV / no uncertainty-attribution analysis run. The record run would put ngboost into the CV (bonus for P2). |
| P3 — automated sales scrutiny | **GAP (leads only)** | anecdotal: scrutiny is blind to sparse vacant sales (no clusters → no flag); bad/atypical sales survive into training. No precision/recall study, no assessor ground truth. |
| P4 — condo resolution | **UNTOUCHED** | — |
| P5 — cross-jurisdiction equity | **GAP (infra only)** | 2-jurisdiction CV harness is a seed; no equity-specific (COD/PRD/PRB regressivity) study run. |

---

## Open methodological findings (cross-cutting, paper-relevant)

- **Forward-bias is real (Q2):** in-period evaluation overstates AVM quality; rolling-origin reveals
  the forward number. Drift-invariant for COD (dispersion), shifts median ratio (level).
- **Petersburg time-adjustment index is ~flat** (`sale_price_time_adj ≈ sale_price` all years) — a
  likely under-powered time-adjustment, implicated in forward under-prediction. A concrete improvement
  lever and a P0/P1 talking point.
- **Inner tuning CV is shuffled k-fold, not temporal** (`_*_kfold_cv`) — fine for selection; noted.
- **`--jobs > 1` oversubscribes** (xgboost Optuna `n_jobs=-1` × folds) — run folds sequentially, or
  add an inner-`n_jobs=1` guard before enabling fold parallelism. (Harness TODO.)

---

## Recommended next moves (when resumed), highest-leverage first

1. **Build the C6 triage analysis** — biggest evidence gap, most distinctive lcomp claim, runnable on
   data in hand; bridges to P3. (medium effort)
2. **Produce the C5 worked exhibit** — one `explain_value` per-parcel + SHAP waterfall. (quick)
3. **Record run** — `--n-trials 5`, + catboost/ngboost, default harness to 5; finalizes C4 + the §6
   table; expected to confirm, not change. (slow / overnight)
4. Then: P0 prose pass (well-positioned); P2–P5 remain future.
