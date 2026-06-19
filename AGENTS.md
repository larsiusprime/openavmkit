# AGENTS.md

A working guide for coding agents (Claude, Cursor, Copilot, etc.) and human contributors landing in OpenAVMKit. Read this first before changing code.

This file is **a living document.** Append to it when you discover a non-obvious *general* pattern. Keep jurisdiction-specific findings in your own memory system, not here.

---

## 1. Public API surface

[openavmkit/pipeline.py](openavmkit/pipeline.py) is the public surface **for the notebooks**. Its module docstring states the rule:

> Every public function should be called from at least one notebook. The primary openavmkit notebooks should only call functions from this module. This module imports from other modules, but no other modules import from it.

If you're using OpenAVMKit as a Python library outside the notebooks, you're free to import directly from any module — `openavmkit.ratio_study`, `openavmkit.utilities.stats`, etc. The notebook constraint exists to keep notebook code legible and stable.

When adding a new notebook-facing capability:

1. Implement it in the appropriate domain module (`modeling.py`, `cleaning.py`, `data.py`, etc.).
2. Add a thin wrapper in `pipeline.py` that the notebooks call.
3. The wrapper is where you put the user-friendly docstring (NumPy style).

If a notebook reaches into a non-pipeline module directly, that's a sign the wrapper is missing — add one rather than working around it.

---

## 2. Settings.json — preprocessor features

Before any code reads `settings.json`, [openavmkit/utilities/settings.py](openavmkit/utilities/settings.py) (`load_settings`) runs it through a preprocessor. These features are not standard JSON — they're project-specific, and they're widely used in real settings files. Know them before editing settings.

### `__`-prefixed keys are comments

Any key starting with double underscore is stripped before settings are used. JSON has no comment syntax; this is the project's workaround.

```json
{
  "__comment": "This is a note for humans, not for the loader",
  "__commented_out_field": { "old": "config" },
  "locality": { "name": "Example" }
}
```

Real example: [notebooks/pipeline/data/us-nc-guilford/in/settings.json](notebooks/pipeline/data/us-nc-guilford/in/settings.json) has dozens of `__comment`, `__comment1`, `__commented_out_characteristics`, etc.

**Don't confuse with single underscore.** A single-underscore key (e.g. `"_run"`) is *not* stripped — it's just a key that doesn't match what the loader looks for. That can have side effects, but it's not a documented disable convention.

### `$$path.to.value` resolves variable references

A string value beginning with `$$` is replaced by looking up the dotted path within the same settings tree. Resolution is recursive — chains of references work. See `_replace_variables` in [openavmkit/utilities/settings.py](openavmkit/utilities/settings.py).

```json
{
  "ref": {
    "default_dupes": ["key", "sale_date"]
  },
  "data": {
    "load": {
      "sales": { "dupes": "$$ref.default_dupes" }
    }
  }
}
```

Use this to keep one source of truth for repeated values (column lists, thresholds, valuation dates).

### Settings are merged with a built-in template

The user's `settings.json` is merged with [openavmkit/resources/settings/settings.template.json](openavmkit/resources/settings/settings.template.json), so users only need to specify overrides. Look in the template for defaults before assuming a key is missing.

### `!key` stomps the template

Prefix a key with `!` to overwrite the template entirely instead of merging:

```json
{ "!field_classification": { "land": { "numeric": ["only_this_one"] } } }
```

Without the `!`, your value would be merged into the template's much longer list.

### `+key` extends a template list

Prefix with `+` to *add* your items to the template's list instead of replacing it:

```json
{ "+field_classification": { "+land": { "+numeric": ["my_extra_field"] } } }
```

The merge logic skips items already in the template (set-union semantics).

---

## 3. Settings.json — content conventions

### Read settings via helpers, not raw `.get()` chains

[openavmkit/utilities/settings.py](openavmkit/utilities/settings.py) provides typed accessors: `get_valuation_date`, `get_model_group_ids`, `get_fields_categorical`, `area_unit`, etc. Use them. They handle defaults, validate types, and centralize what would otherwise be brittle nested `.get(...).get(...)` chains.

If a helper for the setting you need doesn't exist, consider adding one rather than reaching in directly.

### Where example settings live

`notebooks/pipeline/data/<locality>/in/settings.json` — every supported locality has one. They're the best learning resource.

The **canonical examples** to read first are:

- [us-nc-guilford](notebooks/pipeline/data/us-nc-guilford/in/settings.json) — Guilford County, NC
- [us-va-petersburgcity](notebooks/pipeline/data/us-va-petersburgcity/in/settings.json) — Petersburg City, VA
- [us-pa-philadelphia](notebooks/pipeline/data/us-pa-philadelphia/in/settings.json) — Philadelphia, PA

These three cover the breadth of supported features and idioms. Reach for them before writing a settings file from scratch.

### Adding a new setting

1. Pick a path that fits the existing tree (`data.process.*`, `modeling.*`, `analysis.*`).
2. Add the default to [openavmkit/resources/settings/settings.template.json](openavmkit/resources/settings/settings.template.json) if there's a sensible default.
3. Read it via a helper in `utilities/settings.py` (add one if needed).
4. Document it in [docs/docs/advanced_settings.md](docs/docs/advanced_settings.md) so it doesn't become buried.

---

## 4. General gotchas

Patterns, not jurisdictions. Jurisdiction-specific findings belong in agent memory.

- **Assessor data is the default source of truth.** When two fields cover the same concept (e.g. assessor-recorded `land_area_sqft` vs. GIS-derived `land_area_gis_sqft`), prefer the assessor field. Assessors typically encode information that's invisible to a naive GIS polygon read — easements, unusable land, surveyed corrections, etc. The GIS-derived value exists primarily as an automatic backfill: `_basic_geo_enrichment` in [openavmkit/data.py](openavmkit/data.py) substitutes GIS area whenever the assessor area is `0`, negative, or `NaN`, and exposes the deviation as `land_area_gis_delta_<unit>` and `land_area_gis_delta_percent` for diagnostic use. **Only override and prefer the GIS value when you have reason to believe the assessor field is unreliable for a specific jurisdiction or model group.** Record those overrides in your agent memory, not in this file.
- **Many enrichment steps silently skip if not opted in.** `data.process.enrich.streets.enabled` defaults to `false` (see `enrich_df_streets` in [openavmkit/data.py](openavmkit/data.py) — the code prints a hint when it skips). Other enrichments (`census`, `distances`, `permits`, `overture`) need **both** their section to be present in `data.process.enrich` *and* `enabled: true` set on the section. The orchestrator at `_enrich_data` only invokes each function when the section key is present; the function itself then early-returns unless `enabled` is true.
- **DEM enrichment depends on optional raster packages and warns (no longer silently) if they're missing.** `data.process.enrich.dem` needs `rasterio` and `seamless-3dep` (both in `requirements.txt`, imported lazily inside `DEMService`). A venv that hasn't been synced to `requirements.txt` will add no elevation columns — `_enrich_df_dem` in [openavmkit/data.py](openavmkit/data.py) now checks for them up front and emits a loud, actionable warning (`pip install -r requirements.txt`) instead of letting an opaque `ImportError` get swallowed by its catch-all.
- **Street enrichment is computationally expensive but only needs to run once.** It can take a long time the first time, but the result is cached and does not need to be regenerated unless the locality's geometry changes.
- **`data.process.invalid_sales` defaults to off.** See `filter_invalid_sales` in [openavmkit/cleaning.py](openavmkit/cleaning.py) — it early-returns when `enabled` is not set. Set `data.process.invalid_sales.enabled = true` to actually run validation checks. The filter runs on the **hydrated** sales frame, so it can reference universe (CAMA) fields like `assr_market_value`. For **relative** rules the filter DSL can't express inline (it has no scalar arithmetic — e.g. `sale_price < 0.5 * assr_market_value`), add a `data.process.invalid_sales.calc` block (same DSL as `data.process.calc`, incl. zero-safe `/0`); it computes derived columns like `sale_to_assr_ratio` on the hydrated frame just before the filter resolves. See [advanced_settings.md § 5.3](docs/docs/advanced_settings.md).
- **Time adjustment can be wholly overridden.** `data.process.time_adjustment.from_file.<model_group>` (see `read_time_adjustment_from_file` in [openavmkit/time_adjustment.py](openavmkit/time_adjustment.py)) replaces the built-in engine for that model group with a CSV. Check this before debugging unexpected time-adjustment output.
- **Notebooks accumulate large outputs.** `notebooks/pipeline/03-model.ipynb` is over 1 MB. Strip outputs (`jupyter nbconvert --clear-output --inplace …`) before committing.
- **Caching is designed to self-invalidate but isn't perfect.** If you change `settings.json` (or any other input) and the next run looks suspiciously like the last one — or if enrichment output is missing fields it should have — **nuke the cache.** Use `delete_checkpoints("<prefix>")` for notebook checkpoints, or delete the locality's `cache/` folder for the enrichment cache. See [docs/docs/advanced_settings.md § 8](docs/docs/advanced_settings.md#8-caching-checkpoints) for full detail. **A specific, loud failure mode:** when new settings *add columns* to the universe (a new `load` mapping, a new `calc`, a new ref-table field), `write_cached_df` in [openavmkit/utilities/cache.py](openavmkit/utilities/cache.py) **raises** `ValueError: Cached DataFrame does not match the original DataFrame` mid-enrichment rather than silently recomputing — the stored `.cols.parquet` diff can't reconstruct the new schema. Fix is the same: delete `cache/` and let enrichment regenerate.
- **Saved model parameters are a separate cache layer with different semantics.** Tunable models (XGBoost / LightGBM / CatBoost / NGBoost via Optuna; GWR / kernel regression via bandwidth search) save their tuned hyperparameters under `<locality>/out/models/<model_group>/.../` as `<slug>_params.json`, `<model_name>_bw.json`, or `kernel_bw.pkl`. **Deleting these forces a fresh hyperparameter search** on the next run. **Keeping them skips the search.** As of 2026-06-18 the Optuna `<slug>_params.json` is **fingerprint-guarded** (`_get_params` in [openavmkit/modeling.py](openavmkit/modeling.py) embeds a `__fingerprint` from feature columns + row count + `n_trials` + seed + `_SEARCH_SPACE_VERSION`): if any of those change — different `ind_vars`, sales window, trial budget, or a bumped tuner search space — the saved params are detected as stale and the model **re-tunes automatically** (no manual delete needed); a loud "saved params … are stale … re-tuning" line prints. **A change the fingerprint can't see** (e.g. you edit raw data values without changing columns/row count) still needs a manual delete. Note `<model_name>_bw.json` / `kernel_bw.pkl` (GWR/kernel) are **not** fingerprint-guarded — delete those manually if their context shifts. **Bump `_SEARCH_SPACE_VERSION` in [openavmkit/tuning.py](openavmkit/tuning.py) whenever you change any tuner's search space** so existing params/journals invalidate. See [advanced_settings.md § 8.4](docs/docs/advanced_settings.md#84-saved-model-parameters-different-semantics).
  - **Optuna tuning is crash-resumable (when `save_params=True`).** During tuning the Optuna study is journal-backed at `<model_group>/.../<slug>_study_<fingerprint>.journal` (`_resumable_study` in [openavmkit/tuning.py](openavmkit/tuning.py)), so an interrupted run (crash / Ctrl-C / OOM) resumes from the trials already on disk instead of restarting at trial 0. On a clean finish, `_get_params` ([openavmkit/modeling.py](openavmkit/modeling.py)) writes the final `<slug>_params.json` and **deletes the journal** — so a leftover `*_study_*.journal` means "interrupted, will resume." The `<fingerprint>` (hash of feature columns + row count + `n_trials`) scopes the journal to its search context; change `ind_vars` or the sales window and the stale journal is discarded, not resumed. Journal backend (not SQLite) + an open()-based lock are used so concurrent (`n_jobs=-1`) trials work and Windows doesn't hit the symlink-privilege error. Tuning stays fully in-memory when `save_params=False`. This applies to the four Optuna tree tuners only — GWR/kernel bandwidth searches are single-shot and just persist their one final artifact.
  - **`lcomp` (LayeredComp) caches its learned per-tree `weight_falloff` (when `save_params=True`).** lcomp does no hyperparameter search — its only learned state is one `weight_falloff` per bagging tree (the `minimize_scalar` step, ~60% of fit time). `run_layeredcomp` ([openavmkit/modeling.py](openavmkit/modeling.py)) saves those floats to `<model_group>/.../lcomp/lcomp_falloffs.json`; on the next run with `use_saved_params=True` it rebuilds the ensemble injecting them and **skips the search** (the comp-tree structure is still rebuilt — that part isn't cached). Same staleness rules as the params files: a fingerprint (feature columns + row count + hyperparams) and the pinned `layeredcompmodel` version guard it, and ANY mismatch falls back to a full fit (never a wrong model). Reconstruction replicates `LayeredCompBaggingModel.fit()` minus the search; it's verified bit-for-bit against a normal fit by `tests/test_modeling.py::test_lcomp_reconstruct_with_falloffs_matches_full_fit`. The clean long-term replacement is an upstream `LayeredCompBaggingModel.fit(weight_falloffs=...)` hook.
- **Never `collapse_sparse_categories` a *location* field in place.** Collapsing rare values into an `"Other"` bucket is fine for a generic model feature, but location fields (anything in `field_classification.important.locations` / `.fields.loc_*`, `analysis.*.location`, a ratio-study `<loc_*>` breakdown, a model/ensemble `locations` list, or `land.lycd.*.location`) are assumed geographically coherent — equity clustering, ratio-study breakdowns, local-ensemble selection, and sales-scrutiny clusters all *group by* them. Collapsing in place merges unrelated zones into one `"Other"` bucket and corrupts those analyses. Instead set `output_field` on the collapse config to write a `<field>_collapsed` modeling variant, classify *that* categorical, and use it only as a model feature; leave the raw location intact. The code warns loudly (config-based, via `get_location_fields` / `get_collapsed_fields` / `warn_if_location_collapsed` in [openavmkit/utilities/settings.py](openavmkit/utilities/settings.py)) both at collapse time and at each grouping site; `"strict": true` in the collapse block escalates to a hard error. See [advanced_settings.md § 5.4](docs/docs/advanced_settings.md).
- **In model-group (and other DSL) filters, use `isin`/`notin` — not `==`/`not` — on any field that can be null.** `resolve_filter` ([openavmkit/filters.py](openavmkit/filters.py)) maps `==`/`!=` to `Series.eq`/`.ne`, which on a *nullable* dtype (pandas `string`/`Int64`/`boolean`) return `<NA>` for null cells, not `False`. That `<NA>` then propagates through `and`/`not` by Kleene logic, so a parcel with a null in the compared field matches **no** group and silently lands in `UNKNOWN`. `isin`/`notin` treat null as `False`/`True` (no propagation), so they're null-safe. Real example: a suburban parcel with null `style` evaluated `["not", ["and", ["==","use","010..."], ["==","style","16 OLD STYLE"]]]` to `<NA>` and dropped out of every single-family group; rewriting the residual as `["or", ["notin","use",[...]], ["notin","style",[...]]]` fixed it. When two groups are meant to be exact complements, build them from `isin`/`notin` so they partition exhaustively regardless of nulls. (`>`/`<`/`>=`/`<=` are already null-safe — they `fillna(0)` for numeric fields.)
- **Enrichment numerics fed to `ind_vars` need a fill rule — `mra`/`multi_mra` hard-crash on NaN.** The linear models go through `statsmodels` OLS, which raises `MissingDataError: exog contains inf or nans` on any NaN/inf in the design matrix; tree engines (lightgbm/xgboost/catboost) tolerate NaN, so a missing fill rule only surfaces when a *linear* model runs (and `lcomp` may too). Enrichment outputs frequently have partial coverage — census (block-group misses), DEM (coverage gaps), `ref_tables` (unmatched keys) — and `data.process.fill` ([openavmkit/cleaning.py](openavmkit/cleaning.py) `_fill_unknown_values`) only fills the fields you list. So whenever you add an enrichment field to a model's `ind_vars`, add it to `data.process.fill` (usually `median` for continuous). `proximity_to_*` is already 0-filled by the distance enricher; basic-geo fields are safe. Fill runs in the **clean** stage, so the fix needs a notebook-2 re-run, not just a model re-run. (Allegheny: `median_income` ~3% NaN and `elevation_mean_ft`/`slope_mean_deg` ~1% crashed `mra` until added to `fill.median`.) See [advanced_settings.md § 5.1](docs/docs/advanced_settings.md).
- **Log-target is a per-model `log` flag on the linear models — NOT a `dep_var` change.** To stop `mra`/`multi_mra` from extrapolating negative predictions, set `"log": true` on that model's entry under `modeling.models.<main|vacant>.<group>.<model>` (read via `entry.get("log")` in `run_one_model`, consumed by `run_mra`/`run_multi_mra` in [openavmkit/modeling.py](openavmkit/modeling.py)). The model fits on `np.log(target)` and **exponentiates its own predictions back to price space**, so the model boundary is always price-space and nothing downstream (metrics, ratio study, ensemble, variable selection) needs log awareness — that containment is the whole point. Only `mra`/`multi_mra` read the flag; other engines ignore it. Do NOT resurrect the `dep_var = "log_..."` approach: `dep_var` is a per-group setting that leaks the log target into every model in the group (echo models like `assessor` overflow on `exp`, nearest-neighbor/`spatial_lag` predict in the wrong space, the ensemble double-exps) — that blast radius is exactly why the contained per-model flag exists. See [advanced_settings.md](docs/docs/advanced_settings.md) "log".
- **Resolving a model's settings entry is per-model-group — always narrow to the group first.** `modeling.models.<main|vacant>` is keyed by **model group**, and each group holds the per-model entries (`<group>.<model_name>` / `<group>.default`). The canonical three-step resolution (see `run_one_model` / `_run_models` in [openavmkit/model_runner.py](openavmkit/model_runner.py)) is: `me = settings["modeling"]["models"][vacant_status]` → `me = me.get(model_group, me)` (narrow to the group; the `, me` fallback preserves the legacy/global layout) → `entry = me.get(model_name, me.get("default", {}))`. Code that reads `models.main.<model_name>` (or `models.main.lcomp`) **without** the middle `me.get(model_group, me)` step silently gets `{}` under the per-group layout and the feature degrades to a no-op. This bit the outlier comp-analysis path in [openavmkit/pipeline.py](openavmkit/pipeline.py) (it printed "Skipping comp-analysis: missing ind_vars" because `main.lcomp` resolved to `{}` once models were nested per group). When adding any new read of model settings, mirror the three-step pattern.
- **`use_sales_from` can be set per model group.** `modeling.metadata.use_sales_from` ([openavmkit/utilities/settings.py](openavmkit/utilities/settings.py)) accepts a scalar, a legacy `{improved, vacant}` per-type dict, or a per-group `{default: <entry>, by_model_group: {<group>: <entry>}}` form (each `<entry>` itself a scalar or `{improved, vacant}`). Two-layer application: the cleaning/clip stages call `use_sales_from_floor` (the widest/oldest window across all groups) because they permanently drop too-old sales *before* the per-group split — dropping to a narrow window there would starve a longer-reach group (e.g. data-poor commercial reaching back further than recent-only residential). `get_data_split_for` ([openavmkit/data.py](openavmkit/data.py)) then narrows each group to its own window via `resolve_use_sales_from(settings, model_group=...)`. So set `default` to your common window and override only the groups that differ; the floor widens automatically. The IAAO ratio-study eval window (`analysis.ratio_study.look_back_years`) is separate and unaffected.
- **Two evaluation paths with different conventions — don't conflate them.** (1) The **benchmark model-comparison holdout** (`_calc_benchmark`, scored against `dep_var_test`, default `sale_price_time_adj` — [openavmkit/model_runner.py](openavmkit/model_runner.py) ~line 3188) ranks models against each other on **time-adjusted** prices over the three-tier split. (2) The **formal ratio study** (`_run_ratio_study_breakdowns` — [openavmkit/ratio_study.py](openavmkit/ratio_study.py) ~line 452) is the IAAO-style `assessor`-vs-`openavmkit` report, and it scores against **raw `sale_price`** restricted to the **lookback window** (`look_back_years`, default 1yr — line 421). Per IAAO *Standard on Ratio Studies* §4.4 (window "ideally no more than one year"; longer only if "adjusted for time as necessary"), raw-price-within-1yr is the standard-endorsed choice, so the formal ratio study needs **no** time-adjustment of its test sales.
- **The benchmark model-comparison metric (path 1) is mildly optimistic: two preprocessing steps are fit on *all* sales, not train-only.** Time adjustment (`enrich_time_adjustment`, `process_sales`, clean stage — [openavmkit/time_adjustment.py](openavmkit/time_adjustment.py)) and variable auto-reduction (`get_variable_recommendations` on `ds.X_sales`/`ds.df_sales` — [openavmkit/model_runner.py](openavmkit/model_runner.py) ~line 612) run across the **full sales set, before the split exists**, so held-out test rows inform the time-adjustment index and the retained feature set → that COD/PRD is slightly optimistic. Magnitude is small (test ≈ 20% of rows; one sale barely moves a bucket median, a zero-representation var rarely flips). (The formal ratio study, path 2, scores against raw price so the time-adjustment leak doesn't touch its ground truth.)
  - **This is *not* "sales chasing"**, and the production roll is fine. Per the IAAO *Standard on Ratio Studies* (App. E), chasing means *selectively* reappraising sold parcels toward their sale prices; a global, uniformly-applied time-adjustment index treats sold and unsold parcels identically, satisfying §4.5's "value the sample parcels the same as the population" condition. So calibrating on all sales is standard CAMA practice for the **production valuation**.
  - **But IAAO does NOT bless *evaluating* accuracy on calibration sales** — Appendix E exists precisely because that makes appraisals "appear more uniform than they are." IAAO's recommended check is **E.3, the Split Sample Technique**: compare a ratio study on *pre-appraisal-date* sales vs one on *post-appraisal-date* sales; if the pre-date study is consistently better, that signals chasing and "the first study should be rejected." That temporal pre/post split is exactly the planned **publish-mode rolling-origin CV** (post-date = the honest, un-chaseable holdout). So: leave the everyday all-sales calibration as-is (correct for the roll); get leakage-correct accuracy from the rolling-origin evaluation, which is IAAO-aligned.
  - **Spatial lag, by contrast, is already train-only** (`df_sub_train` in [openavmkit/data.py](openavmkit/data.py) ~line 882) — it *must* be, because it builds a model *feature* from neighbor sale prices.
- **Modeling is always deterministic, and parallelism is NOT sacrificed for it — `modeling.metadata.seed` (default 42) is the one knob.** MRA was always deterministic; the tree models were not, because the Optuna sampler was entropy-seeded. `get_model_seed` ([openavmkit/utilities/settings.py](openavmkit/utilities/settings.py)) is the single source of truth — it seeds the Optuna TPE sampler (`_seeded_sampler`), the CV folds, and the final XGBoost/LightGBM/CatBoost/NGBoost/lcomp fits.
  - **The perf-vs-determinism tradeoff is overcome (not chosen) for XGBoost/LightGBM via synchronous batched ask-and-tell** (`_run_batched` in [openavmkit/tuning.py](openavmkit/tuning.py)): params are *sampled sequentially* in the ask phase (so the sampler RNG advances in a deterministic order — never inside the parallel section, which would race), then a batch is *evaluated in parallel*, then *told back in ask order* (so study state after each generation is identical regardless of completion order). Result: reproducible **and** parallel. The non-determinism in plain `study.optimize(n_jobs=-1)` came from telling results in completion order, not from parallelism per se.
  - **Why each fit is single-threaded inside the batch.** `nthread=1` (xgb) / `num_threads=1,deterministic=True,force_row_wise=True` (lgbm) make each CV fit bit-reproducible and avoid core oversubscription; parallelism comes from running `_TUNING_BATCH_SIZE` evaluations concurrently. Batch size is a fixed constant (not core count) so the TPE adaptivity cadence — and the result — is machine-independent; only how many run at once scales with cores.
  - **CatBoost/NGBoost tune serially** (CatBoost prunes via `trial.report`, which is hard to make parallel-deterministic; NGBoost isn't thread-safe). They're deterministic via the seeded sampler + serial `optimize`. The batched path is xgb/lgbm only.
  - There is **no nondeterministic mode** — a `null` seed falls back to 42. If you add a tuner/final fit, thread `seed` through `_get_params`/the `run_*` signature, and note the seed is part of the tuning-journal fingerprint (changing it discards a stale resume journal rather than mixing seeds). Caveat: a study *interrupted and resumed* via the journal is not guaranteed bit-identical to a single-process run (the sampler re-seeds on reattach); clean runs are reproducible.
- **The inner hyperparameter-tuning CV is shuffled k-fold, not temporal.** `_xgb_kfold_cv` / `_lightgbm_kfold_cv` and the CatBoost/NGBoost tuners in [openavmkit/tuning.py](openavmkit/tuning.py) use `KFold(shuffle=True)` to *select* hyperparameters (on training data only — correctly nested). They are **not** temporal. This is fine for selection and does not leak, but don't mistake it for an outer rolling-origin evaluation protocol.

---

## 5. Code style

- **Docstrings: NumPy style.** Matches existing code (see the `RatioStudy` class in [openavmkit/ratio_study.py](openavmkit/ratio_study.py)) and `mkdocs.yml`'s `mkdocstrings` config. Don't write Google-style docstrings — they will render inconsistently.
- **PEP 8** per [CONTRIBUTING.md](CONTRIBUTING.md).
- **Module-level docstrings** are surfaced as the page header by `mkdocstrings`. New modules should have one.
- **Don't add comments that restate the code.** The existing code is light on comments by intent — only add one when the *why* is non-obvious.

---

## 6. Testing

Tests live under [tests/](tests/). Run them with:

```bash
pytest tests/
```

Tests use real settings fixtures from `tests/data/<slug>/`. When adding a feature that depends on settings, prefer extending an existing fixture over creating a new locality.

---

## 7. Extending common patterns

### Adding a new model

Models live in [openavmkit/utilities/modeling.py](openavmkit/utilities/modeling.py) (e.g. `XGBoostModel`, `LightGBMModel`, `MRAModel`). To add one:

1. Subclass the appropriate base class in `utilities/modeling.py`.
2. Wire it into the dispatch in [openavmkit/model_runner.py](openavmkit/model_runner.py) (search for the existing tree-based model handling).
3. Add a config entry under `modeling.models.<main|vacant>` in the template.
4. Add a public wrapper to `pipeline.py` if it should be runnable directly from a notebook.
5. **Wire up params and contribs.** Every model must produce two outputs per subset (`test`, `sales`, `universe`):
   - **`params_<subset>.csv`** — per-feature **parameters**. For linear models these are regression coefficients; for tree-based models they are SHAP values normalized by value size. Conceptually: "what is each feature's per-unit effect on the prediction?"
   - **`contributions_<subset>.csv`** — per-feature **contributions**. For linear models these are coefficients × the feature's value for that row; for tree-based models they are raw SHAP contributions. Conceptually: "how much did each feature actually contribute to this row's prediction?"

   Existing implementations to follow: [`write_mra_params`](openavmkit/modeling.py) (linear), [`write_tree_based_params`](openavmkit/modeling.py) (SHAP), [`write_gwr_params`](openavmkit/modeling.py) (per-location coefficients), [`write_local_area_params`](openavmkit/modeling.py), [`write_multi_mra_params`](openavmkit/modeling.py). Pick the one that most closely matches your model's structure and follow the same file-output contract. The dispatch that calls these per-model writers is the common interface; if your model is fundamentally new in shape, add a new writer in `modeling.py` and a dispatch case alongside the existing ones.

   Use the canonical subset names `test` / `sales` / `universe` in those filenames (`contributions_universe.csv`, not the legacy `univ`). `_write_model_results` reads `contributions_universe.csv` to build `contributions_map.parquet`; a writer that emits a different universe name silently skips the map.

   **Ensembles also produce params/contribs.** `_write_ensemble_contributions` in [openavmkit/model_runner.py](openavmkit/model_runner.py) reassembles them from the member models rather than from a fitted model object: mean/median ensembles are per-row convex combinations of members (so the ensemble contribution is the weighted mean of member contributions, with the base term taken as the residual `prediction − Σ contributions` so reconstruction is exact), and local passes through the selected member. A new model engine automatically participates as an ensemble member as long as it emits the standard per-row `contributions_<subset>.csv`; engines that can't (e.g. `local_area`, the naive baselines) fold cleanly into the ensemble base.

### Adding a new equity study

See `horizontal_equity_study.py`, `vertical_equity_study.py` as templates. The shape is: a `*Study` class with statistics in `__init__`, a runner function that handles the SalesUniversePair plumbing, and a `pipeline.py` wrapper.

### Adding a new enrichment source

Add a `_enrich_df_<source>` function in [openavmkit/data.py](openavmkit/data.py), gate it on a settings key under `data.process.enrich.<source>`, and call it from `_enrich_data` alongside the other `_enrich_df_*` calls.

---

## 8. Memory hooks (for AI agents)

If you have a persistent memory system:

- **Save jurisdiction-specific quirks to memory**, not to this file. Examples: "for `<slug>`, prefer field X over Y because ..."
- **Save user preferences to memory.** Things like "this user wants terse PRs" are agent-specific.
- **Append to this file only when you've found a *general* pattern** that applies repo-wide and is non-obvious from the code alone.

---

## 9. Pending removals (tech debt with deadlines)

Time-boxed cleanups that must happen before a named release. Delete the entry when done.

- **Remove the `openavmkit.benchmark` compatibility shim before 0.7.0.** The module was renamed to
  [openavmkit/model_runner.py](openavmkit/model_runner.py); [openavmkit/benchmark.py](openavmkit/benchmark.py)
  is now a thin re-export that emits a `DeprecationWarning`. Once downstream imports are updated, delete
  the shim and this note.

---

## 10. Living document footer

Sections worth growing over time:

- **Section 4 (gotchas)** — when you hit one and figure it out, add it here.
- **Section 7 (extending patterns)** — when you add a new kind of thing (a new study type, a new data source), describe the recipe so the next agent can copy it.

Keep entries short. Link to the source line that proves the claim. If a claim becomes stale because the code changed, fix or remove the entry instead of leaving it.
