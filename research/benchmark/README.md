# Benchmark backbone — reproducible multi-jurisdiction substrate for the paper program

This is the **cross-cutting backbone** of the OpenAVMKit research-paper program: one reproducible
benchmark that every paper (P1–P5) draws from, so all empirical claims rest on the same experimental
substrate. Per the approved plan it is **scoped to two jurisdictions for now** — Eagle County CO and
Petersburg VA — with Guilford/Philadelphia/Maricopa/Lee/Beckham/Skagit deferred as additive
expansion.

## What's here

- **`run_benchmark.py`** — the harness. Given a locality slug, it mirrors notebook 03's pre-model
  sequence — load cleaned data (`load_cleaned_data_for_modeling`), write canonical train/test splits,
  enrich spatial-lag features — then runs the full model menu via `openavmkit.pipeline.run_models`
  (reusing tuned hyperparameters), and persists the uniform per-model comparison table plus a
  reproducibility manifest under `research/benchmark/<slug>/`. It optionally splices a research run
  list first.
- **`<slug>/`** — per-jurisdiction outputs: `benchmark_df_test.csv`, `benchmark_df_full.csv`,
  `benchmark_df_time.csv`, a combined `benchmark.md`, `manifest.json` (run list + input file
  hashes/sizes + valuation date), and `settings.snapshot.json`.
- **`<slug>_run.log`** — full console log of a run (per-model-group benchmark tables are printed
  here even before the final persisted artifact).

## The uniform comparison table

Built by `openavmkit.model_runner._calc_benchmark` (`MultiModelResults.benchmark`). Per model, per
subset (holdout/test, study, universe): `utility_score`, sale/universe counts, `median_ratio`,
**COD / PRD / PRB / VEI** (and trimmed variants), and **CHD** — i.e. the IAAO uniformity suite plus
a utility score, all in one apples-to-apples table. This is exactly what P1 §6 (accuracy ×
uniformity) and P5 (cross-jurisdiction equity) need.

## How to run

```bash
# Petersburg already runs lcomp + ngboost in its own run list — just run it:
python research/benchmark/run_benchmark.py us-va-petersburgcity

# Eagle DEFINES lcomp but doesn't run it by default — add it (and ngboost) for an
# apples-to-apples flagship comparison:
python research/benchmark/run_benchmark.py us-co-eagle --add-lcomp

# Or force the canonical research run list on any jurisdiction:
python research/benchmark/run_benchmark.py us-co-eagle --research

# Flavor A repeated-holdout CV — stabilizes the noisy single-split numbers into mean ± std:
python research/benchmark/run_benchmark.py us-va-petersburgcity --cv-repeats 5
```

## Evaluation modes

A single 80/20 holdout at ~150 test sales is too noisy to rank models (re-drawing the split
reshuffles the order). So the harness supports two modes:

- **Single split** (default): one canonical split → one comparison table per group under
  `<slug>/<group>/benchmark.md`. Fast iteration.
- **Flavor A repeated-holdout CV** (`--cv-repeats N`): re-draws the canonical split N times at the
  *same* valuation date (varying `random_seed`), re-runs the full per-fold sequence each time, and
  aggregates per-model COD/PRD/PRB/median-ratio/VEI to **`mean ± std`** under `<slug>/cv/<group>/`
  (`benchmark_cv.md`, plus `*_mean.csv` / `*_std.csv` / `*_raw.csv`). This kills sampling noise.
  Hyperparameters are held fixed (reused from saved params); honest per-fold re-tuning and **true
  rolling-origin** (the temporally-correct, claim-grade protocol — needs per-fold recompute of time
  adjustment, since its global end-date normalization embeds a future price level) are **Flavor B /
  publish mode**, deferred to a later first-class `evaluation.mode: fast|publish` switch in
  `pipeline.py` + notebook 03.

Note: the inner rolling-origin CV in [openavmkit/tuning.py](../../openavmkit/tuning.py) only tunes
hyperparameters; it does not stabilize the *reported* metric. That's what these outer modes do.

Prerequisite: the locality must have a cleaned checkpoint (`out/2-clean-sup.pickle`), i.e.
`01-assemble` + `02-clean` have been run. Both scoped jurisdictions already have one.

Notes:
- The harness **writes per-model outputs under `out/models/` exactly as notebook 03 does** (this is
  required — `run_models` only returns its results dict when `save_results=True`). It is reproducible,
  not read-only; it also produces the per-model `contributions_*.csv` / `params_*.csv` the papers use
  for worked SHAP examples. It reuses saved tuned params (`use_saved_params=True`).
- It re-fits every model in the run list even with saved params; GWR/kernel/NGBoost are the slow
  ones. Petersburg (1,341 sales / 14,460 parcels) takes ~3–5 min per model group; Eagle is larger.
- Run lists, per run, are recorded in `manifest.json` so a result is always traceable to the exact
  model set that produced it.

## Frozen jurisdiction set (current scope)

| Slug | Market | Runs `lcomp`? | Runs `ngboost`? | Notes |
|---|---|---|---|---|
| `us-va-petersburgcity` | small urban (VA) | **yes** (in run list) | **yes** (ensembled w/ lcomp) | ideal P1 **and** P2 site; ready as-is |
| `us-co-eagle` | mountain resort (CO, Vail) | defined, **not run** by default | defined, not run | use `--add-lcomp`; DEM/elevation showcase |

## Reproducibility appendix template (per paper)

Each paper ships, from this directory: the jurisdiction's `settings.snapshot.json` + `manifest.json`
(input hashes), the relevant `benchmark_*.csv`, and a note that `python research/benchmark/run_benchmark.py <slug>`
plus `pytest tests/` reproduces the numbers. This operationalizes the P0 auditability thesis.
