# Verification evidence — Exact tree-SHAP for Layered Comps (P1 claims C2/C3)

**Date captured:** 2026-06-06
**Source:** [tests/test_lcomp_shap.py](../../tests/test_lcomp_shap.py); residuals computed via a one-off
driver over the same fixtures. All 8 tests pass (`pytest tests/test_lcomp_shap.py`, 8 passed, ~13s,
Python 3.11.9, win32).

## What is being verified

The fast EXTEND/UNWIND path-dependent tree-SHAP implementation in
[openavmkit/shap_analysis.py](../../openavmkit/shap_analysis.py) is checked against an **independent
brute-force Shapley oracle** that enumerates all feature coalitions and averages marginal
contributions using the tree's cover-weighted conditional expectation (`_brute_tree_phi`,
`_cond_exp` in the test file). Two properties must hold:

1. **Oracle agreement** — `fast_shap == brute_force_shap` for every feature/row.
2. **Additivity** — `base_value + Σ(shap over features) == ensemble.predict(row)`.

## Achieved residuals (max absolute deviation)

These are the *actual* numbers, not the (looser) tolerances asserted in the test file. They sit at
the floating-point floor — i.e. the decomposition is exact to machine epsilon.

| Case | Oracle agreement (max \|fast − brute\|) | Additivity (max \|base+Σφ − predict\|) | Asserted test tol. |
|---|---|---|---|
| Numeric features (3 feat, 2 trees) | 2.84e-14 | 3.73e-14 | 1e-8 / 1e-6 |
| **Mixed categorical** (one-vs-rest splits) | 4.62e-14 | 5.68e-14 | 1e-8 / 1e-6 |
| **NaN early-stop** (numeric split → stop leaf) | 4.09e-14 | 5.68e-14 | 1e-8 / 1e-6 |
| Numba kernel vs pure-Python fallback | 1.42e-14 (max diff) | — | 1e-9 |
| `expected_value` == mean of per-tree bases | < 1e-12 | — | 1e-12 |

## Why this is paper-grade

- The two hardest cases for standard SHAP tooling — **categorical one-vs-rest splits** and
  **early-stopping on NaN** — agree with the brute-force oracle to ~1e-14, the same floor as the
  plain-numeric case. Standard SHAP's numeric-threshold tree format cannot represent these at all;
  the folded-tree construction handles them natively and exactly.
- Additivity holds to ~1e-14 against the *real ensemble prediction*, so per-parcel explanations
  literally reconstruct the model output — the property appeals framing (C5) rests on this.
- The optimized numba kernel and the reference pure-Python path agree to ~1e-14, so the speedup
  introduces no numerical drift worth reporting.

These figures are ready to drop into P1 §5 (Verification) as the proof that "exact" is meant
literally, and into the additivity statement in §4.
