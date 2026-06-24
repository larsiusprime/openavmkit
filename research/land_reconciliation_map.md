# Land work reconciliation map: `docs_land` package vs `lvi` research

*Honest assessment per "check before tossing." Verdict up front: **docs_land is NOT inferior — in
most respects it is the more complete, more mature system.** The lvi research independently
re-derived and validated much of it on current data, found one real defect, and adds ~5 net-new
tests + the non-circularity discipline. So the move is: keep docs_land as the base, fold lvi's
fixes/additions in — not the reverse.*

## The two bodies of work

| | `docs_land` (`openavmkit/land/`, ~3,900 LOC) | `lvi` research (`research/lvi_*.py`, ~1,300 LOC) |
|---|---|---|
| age | 73 commits behind develop; built+run on an OLD snapshot | current develop; run on the latest `2-clean-sup` |
| evidence | `evidence.py`: **6 witness streams** W1–W6 + non-circular **anomaly filter** + PUV/historic/zoning-floor drops + weights | `lvi_anchors.py`: 4 streams (vacant/teardown/rcn/rcnld) + prime-lot + qualification |
| tests | `tests.py`: **L1–L11** | `lvi_battery` + extras: A0–A8 subset + VE + depreciation + B1/B1b + Tier-2 |
| **creation (painters)** | **`tables.py` (base_lot/size_curve/puv) + `lycd.py` (uniform rate)** | **none — lvi only validates existing series** |
| deps | needs `openavmkit.zoning` + `openavmkit.neighborhoods` (**both MISSING on current develop**) | only existing utils |
| validated on current data? | **no** (stale base) | **yes** |

## Component verdicts

- **`tables.py` + `lycd.py` (painters) — KEEP, irreplaceable.** This is the land *creation* side; lvi
  has nothing comparable. The biggest reason not to toss docs_land. (And A0's β≈0.4 finding directly
  informs `tables.py`'s base_lot-vs-size_curve choice — site value leans favored.)
- **`evidence.py` — KEEP, with one fix.** 6 streams (W1–W6) is a superset of lvi's; the anomaly
  filter (TOKEN_PRICE / BELOW_PHYSICAL_PSF / BELOW_COST_FLOOR / PRIOR_XFER_INCONSISTENT) is the
  non-circular price-signal layer we wanted, done well. **DEFECT to fix:** deed filter is a
  deny-list (`LEAKED_DISQ_FLAGS=D,E,F,G`, keeps NaN) — lvi §12 showed it must be a **whitelist**
  (keep only A/C) for the gold standard. lvi's prime-lot *shape* filter (rectangularity) is a small
  add; lvi's streams otherwise SUBSUMED by W1–W6.
- **`tests.py` (L1–L11) — KEEP, merge in lvi's net-new tests.** See mapping below.
- **`zoning.py` + `neighborhoods.py` — must come forward too** (land package imports them; missing
  on develop). Decide whether they're wanted as standalone modules.

## Test mapping (L1–L11 ↔ lvi)

| docs_land | lvi equivalent | verdict |
|---|---|---|
| L1 improvement-neutrality | A1 | duplicate (both fine) |
| L2 within-cluster uniformity | A2 | duplicate |
| L3 vacant-burden flip | A4 (I'd deferred) | **docs_land has it** |
| L4 desirability Spearman | A5 | duplicate (lvi cleaner proxy: market-land not home-price) |
| L5 density-FAR ordering | B2-ish | **docs_land has it** |
| L6 per-cell size decay | A0 | duplicate (lvi quantified β≈0.4) |
| L7 impr-cost-table COD | A8 | duplicate |
| L8 **held-out** vacant prediction | A3 | **docs_land's holdout is better methodology** than lvi's in-sample A3 |
| L9 boundary discontinuity | B3-ish | **docs_land has it** |
| L10 improved-sale reconciliation | (Step1-ish) | docs_land has it |
| L11 Moran's I residual | B3 | duplicate |
| — | **A6 sales-chasing** | **lvi-only → add** |
| — | **A7 summation/sanity** | **lvi-only → add** |
| — | **vertical equity (VEI + clean-vs-residual)** | **lvi-only → add** (caught the −60 artifact) |
| — | **B1 coverage / protest-risk map** | **lvi-only → add** |
| — | **B1b support propagation + per-parcel evidence chains** | **lvi-only → add (the protest artifact)** |
| — | **Tier-2 composition-controlled differentials** | **lvi-only → add** |

## lvi's lasting contributions (fold into docs_land)

1. **deed-code whitelist** (fixes evidence.py's deny-list defect).
2. **6 net-new tests/capabilities** not in L1–L11: A6, A7, vertical equity, B1, B1b (+per-parcel
   evidence chains), Tier-2.
3. **non-circularity discipline** as an explicit contract + the **clean-vs-residual** parallel-run
   pattern that catches artifacts.
4. **fresh empirical validation on current data** — lvi confirms the L-tests' methods still produce
   sane numbers on the latest `2-clean-sup` (docs_land's were last run on a stale snapshot).

## Strategic fork (your call)

docs_land is richer but **73 commits behind with missing deps** (`zoning`, `neighborhoods`), so a
full rebase of the land+zoning+neighborhoods subsystem onto develop is a real lift. Two paths:

- **(A) Rebase docs_land forward.** Bring `land/` + `zoning.py` + `neighborhoods.py` onto current
  develop, fix conflicts, re-validate on current data, then fold in lvi's whitelist + 6 new tests.
  Best if `zoning`/`neighborhoods` are wanted as durable modules. Heavier merge.
- **(B) Current develop as base, cherry-pick the keepers (recommended).** Port docs_land's
  irreplaceable pieces forward onto the current base: the **painters** (tables/lycd), the
  **anomaly filter** + W4/W5/W6 streams, and the **L3/L5/L8/L9/L10/L11** tests — adapting column
  names to current develop. Add lvi's whitelist + 6 new tests on top. Avoids dragging a 73-commit
  subsystem; everything lands validated on current data. Requires deciding the fate of
  `zoning`/`neighborhoods` (port or inline what the streams actually need — the empirical zoning
  floor + VCS cascade).

**Recommendation: (B).** Current develop + lvi is already validated on live data; cherry-pick
docs_land's painters/anomaly-filter/extra-streams/extra-tests forward rather than rebasing the old
subsystem. Net result: one `openavmkit/land/` package = docs_land's structure (painters + 6
streams + anomaly filter + L1–L11) **minus the deny-list defect, plus lvi's 6 tests + non-circular
discipline**, all running on current data.

## Concrete next step

Decide A vs B. Then first task either way: stand up `openavmkit/land/` on the `lvi` branch with
`evidence.py` + the painters, get it *importing and running on current Wake data* (resolve the
`zoning`/`neighborhoods` dependency — port or inline), with the whitelist fix applied. That's the
foundation; the test merge (L-tests + lvi's 6 additions) follows.
