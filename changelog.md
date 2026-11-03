# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

### ⚠️ Behavior changes

- **CatBoost tuning is much faster, and the shipped CatBoost model now matches what was tuned.** The tuner now optimizes an RMSE loss (still scored on MAPE) over symmetric trees only, with categorical feature combinations off (`max_ctr_complexity=1`). Previously it optimized CatBoost's MAPE loss, which converges so slowly on price targets that trials rarely stopped early and the search drifted into Depthwise/Lossguide depth-10 trees. On the public localities: Guilford single-family went from 3,418s to 462s for the CatBoost group run (out-of-fold RMSE $76.9k → $57.8k); Petersburg single-family CatBoost models ran 2–3× faster. `_SEARCH_SPACE_VERSION` is bumped, so saved CatBoost params and resume journals are invalidated and re-tuned once. CatBoost predictions will change.
- **Sales scrutiny now catches multi-parcel ("bulk") deeds whose other rows were invalid, and stops dropping unrelated sales that share a date and price.** A bulk deed stamps one bundle price onto every parcel it covers, so none of those sales is usable. Which sales `run_heuristics` drops changes in three ways:
  - **New heuristic, `flag_bulk_deed`.** `process_data` now records `sale_parcel_count` (parcels on each deed) *before* invalid sales are filtered out, and any sale whose deed covered more than one parcel is dropped. This catches an "orphan" bulk deed reduced to one surviving row, which the duplicate heuristics can't see. Deeds are identified by `analysis.sales_scrutiny.deed_id` (plus sale date and `jurisdiction`) when it's set, otherwise by `key_sale`. Report: `out/sales_scrutiny/multi_parcel_bulk_deeds.xlsx`. On Philadelphia, sales like these had inflated the vacant-land median price per sqft about 4×.
  - **Repeated date + price is narrowed only when deed IDs are available.** Without `deed_id`, every repeated (date, price) is still flagged, because that's the only trace a bulk deed leaves. With `deed_id`, sales on separate, known deeds that happen to share a date and price (a round price, or a builder selling identical lots on separate deeds) are kept. Rows in the group that share a deed, have a missing deed, or are the same parcel twice are still flagged.
  - **Repeated deed + date no longer treats missing deed IDs as one shared deed.** Previously every sale with a blank deed on the same date was flagged as a duplicate of the others.

  Synthesizes [#349](https://github.com/larsiusprime/openavmkit/pull/349) and [#351](https://github.com/larsiusprime/openavmkit/pull/351). See [Sales scrutiny heuristics](docs/docs/advanced_settings.md#analysissales_scrutinydeed_id-and-analysissales_scrutinyjurisdiction).
- **Ratio-study quantile breakdowns now have exactly the number of bins you ask for.** Bins used to start at 0, so the first slice ran from 0 to the column's minimum and held a single sale: `quantiles: 10` (the default `sale_price` breakdown) produced 11 slices, the first one empty in practice. Bins now start at the minimum, and their edges are actual data values, so boundaries in the labels can move by one sale's worth (e.g. `239,500` → `239,000`). If many sales share the minimum value (e.g. 1-story homes), they still get a bin of their own.

### New features

- **Explicit `bins` for ratio-study breakdowns.** `{"by": "pct_vacant", "bins": [0, 0.1, 0.25, 0.5, 1.0], "bin_labels": ["0-10%", "10-25%", "25-50%", "50-100%"]}` slices on value edges you choose, for columns where equal-count quantiles or integer `slice_size` labels don't work (e.g. 0–1 shares). Bad edges or labels raise a clear error; sales outside the edges are left out with a warning. See [Ratio study breakdowns](docs/docs/advanced_settings.md#analysisratio_studybreakdowns). From [#366](https://github.com/larsiusprime/openavmkit/pull/366).

### Fixed

- **Ratio-study breakdowns no longer crash the whole ratio study** on a column with any NaN values (common for enrichment fields with partial coverage) or with negative values ("bins must increase monotonically"). NaN rows are left out of that breakdown; negative values are binned normally. A column with no numeric values at all is skipped with a warning. From [#366](https://github.com/larsiusprime/openavmkit/pull/366).

- The final CatBoost fit dropped the tuner's fixed settings (Optuna's `best_params` only holds searched keys), so the shipped model trained with CatBoost's defaults — RMSE loss and MVS bootstrap, which silently discarded the tuned `bagging_temperature`. The fixed settings are now merged into the final fit.
- CatBoost tuning could crash with `FileExistsError: 'catboost_info'` when parallel CV folds started together in a fresh locality directory. Tuning now uses a private temporary `train_dir`, and no longer leaves `catboost_info/` in the locality folder.
- The `do_shaps` summary for CatBoost used shap's generic `TreeExplainer` instead of CatBoost's native SHAP, which took ~25 minutes on Guilford single-family. It now uses the same native (Approximate) path as the contribution files.
- The Overture building fetch used ~4.4 GB of memory even for a 1 km bounding box, enough to get killed on small workers. PyArrow doesn't use Parquet statistics on nested fields like `bbox.xmin` to skip data, so it was decoding row groups from the whole planet. The fetch now reads each file's footer first and scans only the row groups whose bbox statistics overlap the request. On a 1 km box in Greensboro, NC: 4.4 GB → 0.56 GB and 32s → 7s; on a 10 km box (46k buildings): 4.6 GB → 0.67 GB and 47s → 7s. Results are identical. Diagnosis credit to [#368](https://github.com/larsiusprime/openavmkit/pull/368).

## [0.7.0] - 2026-09-22

### ⚠️ Breaking / behavior changes

- **Models are now evaluated with nested cross-validation by default.** `modeling.instructions.cv_folds` defaults to **5**, replacing the single 20% holdout. **Your reported COD / PRD / PRB will change**, because they are now computed over a different (and much larger) sample: each fold trains on the other folds and predicts its own held-out slice, so the stitched out-of-fold predictions cover **every** sale instead of a noisy ~20% slice. Each fold re-tunes hyperparameters from scratch on its own training data, so the holdout is leakage-free with respect to both training and hyperparameter selection. Folds are grouped by parcel (`key`), not by sale, so a repeat-sale parcel can never appear in both a fold's training set and its holdout. Values that ship come from a separate model refit on all trainable sales. Set `cv_folds: 1` to restore the old single-split behavior. See [Cross-validation](docs/docs/advanced_settings.md#cross-validation-nested-holdout).
  - Holdout scores are an **interpolation** estimate, not a forecast. Post-valuation-date sales never train under any scenario and are still broken out separately as the forecast number.
- **Ratio studies are now apples-to-apples against the assessor.** The formal ratio study scores against raw `sale_price` within the lookback window (IAAO §4.4) rather than time-adjusted prices, and applies a sales-chasing check. Reported ratio-study figures will differ from 0.6.0 for the same data.
- Renamed module `openavmkit.benchmark` → `openavmkit.model_runner` (it orchestrates the whole model run, not just the benchmark comparison). A deprecating compatibility shim remains at `openavmkit.benchmark` — it re-exports everything and emits a `DeprecationWarning`. **The shim is scheduled for removal in 0.8.0**, giving this release as the deprecation window. The `BenchmarkResults` class keeps its name.
- `layeredcompmodel` is now pinned to `==0.3.0` (was `==0.2.1`), required for the lcomp model cache below.

### New features

- **Cross-validation controls** under `modeling.instructions`: `cv_folds` (fold count), `cv_inner` (inner hyperparameter-tuning validation; `"auto"` picks a single grouped split for large training sets, else 3 grouped folds), `cv_production_params` (`aggregate` (default) / `refit` / `best_fold` — how the shipped model gets its hyperparameters), and `cv_max_workers` (fold-level parallelism, defaulting to `min(cv_folds, cores − 2)`). Parallel workers pin single-threaded math and a fixed `PYTHONHASHSEED`, so parallel runs are bit-reproducible run to run.
- Ensembles are CV-aware for all types (`median`, `mean`, `local`): they combine the base models' full-coverage out-of-fold predictions, so the ensemble's holdout covers the same sales its base models do.
- **SHAP over subsets, with additivity cross-checks**, plus out-of-fold contributions and per-fold parameter artifacts.
- **`drop_fields`** — discard scratch columns created by `calc` expressions, both at load time (alongside `filename` / `load` / `calc`) and at enrich time (`data.process.enrich.drop_fields.universe` / `.sales`). Participates in the same ordered operation queue as `calc` and `tweak`. See [Advanced settings § 2.6](docs/docs/advanced_settings.md#26-discarding-scratch-columns--drop_fields).
- **Resumable, deterministic hyperparameter tuning.** Tuning studies are journal-backed, so an interrupted run resumes from the trials already on disk, and a seeded run reproduces exactly.
- **lcomp model cache.** `run_layeredcomp` serializes the whole fitted ensemble to portable JSON and, on a later run with `use_saved_params=True`, deserializes and predicts instead of refitting — skipping both the tree build and the per-tree `weight_falloff` search. Guarded by a data + hyperparameter fingerprint *and* the writing library's version; any mismatch falls back to a full fit.
- Log-target MRA (`log=True`), handled end to end under CV including the out-of-fold back-transform.
- openratiostudy.com-formatted exports, including location fields.

### Fixed

- Independent variables are now validated against **both** the sales and universe frames before modeling, naming the offending fields instead of failing far downstream inside LightGBM with no column named. Two cases are distinguished:
  - **Fatal** — a variable present in sales but missing from the universe. The model would train on a feature it cannot predict with. Fields that `DataSplit` synthesizes onto the universe (`sale_date` and its derivatives, `sale_age_days`, the validity/price flags — the universe is scored as "every parcel sold on the valuation date") are correctly treated as available and are not flagged.
  - **Warning** — a variable absent from *both* frames. `DataSplit` drops these symmetrically, so the model stays coherent and simply loses a feature; the run continues. This keeps an unavailable optional enrichment from killing the pipeline (census-derived variables do not exist without a `CENSUS_API_KEY`), while making the previously silent omission visible. Both paths suggest near-miss column names for typos.
- Spatial lag now respects CV fold boundaries instead of leaking across them.
- Crash guards on the Vertical Equity Index and MRA models.
- Substantially faster data enrichment, and a more efficient COD calculation.
- Renamed the misleadingly-named inner tuning-CV helpers `_xgb_rolling_origin_cv` / `_lightgbm_rolling_origin_cv` / `_catboost_rolling_origin_cv` → `_xgb_kfold_cv` / `_lightgbm_kfold_cv` / `_catboost_kfold_cv`. They use `KFold(shuffle=True)` (random k-fold for hyperparameter selection), **not** temporal/rolling-origin CV. No behavior change. (`_catboost_kfold_cv` is unused — the live CatBoost tuner uses CatBoost's built-in `cv()`.)

### Documentation

- New website and documentation site at [openavmkit.com](https://www.openavmkit.com/), with an API reference, `llms.txt`, and a test that catches broken in-document anchors under both GitHub's and MkDocs' differing slug rules.
- Documented in [AGENTS.md](AGENTS.md) §4 that the benchmark model-comparison metric remains mildly optimistic because time adjustment and variable auto-reduction are fit on all sales (correct for production valuation; a small evaluation-only bias). The formal ratio study is unaffected, as it scores against raw price.

### Known issues

- The variable-representation check (`calc_representation`) prints *"You should NOT model with these"* for variables with zero representation in either frame, but does **not** remove them from the model. It is a diagnostic: it is not one of the `tests_to_run` feature-selection tests, contributes nothing to the variable score, and is absent from the variable report. Whether it should become a real selection test is unresolved; for now, treat its output as advice and drop such variables yourself via `ind_vars`.

## [0.6.0] - 2026-06-05

### New test jurisdictions
- Added `us-co-eagle` (Eagle County / Vail, CO) as a new end-to-end example jurisdiction, built specifically as an elevation/DEM showcase (ski-resort terrain where elevation strongly drives price). Pulls parcel geometry from the county ArcGIS endpoint, ingests xlsx account + sales extracts, and runs all "free" enrichments (DEM headline, census, OSM distances, spatial lag, Overture footprints)
- Added `us-va-petersburgcity` (Petersburg, VA) as a new end-to-end example jurisdiction; it now drives the CI smoke/docker container

### Major features
- Add condo modeling pathway — an opt-in, settings-driven affordance (`data.process.condos` + new `openavmkit/condos.py`) for jurisdictions where condo units are their own accounts but have no parcel geometry. Links each unit to its building polygon (`id_prefix` / `parent_id` / `spatial`), borrows that geometry so units flow through every spatial enrichment, groups units (`condo_group`), and allocates a per-unit land share (legible `field` or `floor_area` pro-rate). New template/data-dictionary fields: `condo_group`, `land_area_alloc_sqft`, `geometry_borrowed`
- Add layered comparables (`lcomp`) model — a bagged comparable-sales model engine
- Add support for different independent variables per model group
- Add support for loading your own time adjustments per model group, plus a start-indexed time-adjustment export and additional file reporting
- Add categorical collapse (`collapse_sparse_categories`) and USGS 3DEP DEM elevation enrichment
- Add Vertical Equity Index (VEI) statistics
- Add ensemble contributions/parameters output and a SHAP contributions map to the finalize-models flow
- Add standard errors to MRA parameter output
- Add OSM coastline support for distance/proximity enrichment
- Add local ensembling as a model option
- Add CSV export option for end-of-notebook "look" files
- Add debug information for piecewise data fills

### Major bug fixes
- Fix bug where all boolean fields were filled with True during fill-missing
- Fix invalid cache from duplicate columns in enrichment
- Fix Overture cache bug that cached/returned a stale full input frame (now caches computed stats only, bbox-keyed, and merges)
- Remove caching from basic geo enrichment where it caused errors
- Fix one-hot / duplicate column-name collisions; collapsed-category output fields are now correctly classified as categorical
- Fix a batch of crash conditions that stopped notebook 3, including crash in identify_outliers, variable-selection crashes on degenerate model groups, GWR/SHAP crashes, and beeswarm/prettify on empty data
- Impute NaN before variable-selection steps in stats
- Guard against n_splits > n_samples in rolling-origin CV
- Guard against all-NA scores in calc_correlations
- Skip non-numeric columns in calc_r2
- Fix ArrowNotImplementedError in SalesScrutinyStudy with pyarrow >= 22
- Cap LightGBM num_leaves/min_data_in_leaf search space for thin datasets
- Fix vertical equity to gracefully handle a missing location field
- Fix assessment quality calculation
- Fix memory use in CHD calculation; guard land_area log fields against infinities
- Numerous column-existence and explicit-truthiness fixes

### Breaking / behavioral changes
- Remove worthless "triangular" parcel detection entirely
- Remove old land/deploy notebooks (crufty, unused) — but the land notebook will be back soon, new and improved!
- Add new opt-in settings blocks: `data.process.condos`, `collapse_sparse_categories`, per-model-group variables, and per-model-group time adjustments (existing settings files are unaffected unless they opt in)
- Change Overture cache format (stats-only / bbox-keyed) — old Overture caches will be recomputed
- Make `readme` packaging dynamic: `setup.py` rewrites the README's repo-relative links to absolute GitHub URLs so they render correctly on PyPI (the in-repo README stays relative for GitHub and the docs site)

### Dependencies & infrastructure
- Numerous dependency bumps: numpy 2.3.5, xgboost 3.2.0, scikit-learn 1.8.0, polars 1.38.1, rich 15.0.0, huggingface-hub 1.17.0, scipy <1.17, statsmodels 0.14.6, matplotlib 3.10.9, and others
- CI: GitHub Actions version bumps, CLA workflow updates, and docker CI fixes

## [0.5.1] - 2025-12-04
- Fix bug in examine_df/examine_df_in_ridiculous_detail

## [0.5.0] - 2025-12-04
- Move to Python 3.11+
- Add metric unit support
- Add multi-mra model
- Add writing out model parameters (coefficients/SHAPs)
- Add support for named models
- Add custom pass-through models
- Add docker container deployment to CI
- Add more/better warnings/errors/feedback
- Optimize memory use in model runs
- Optimize GWR training
- Optimize catboost training
- Optimize performance by removing redundant copy() calls
- Remove stacked ensemble code
- Fix notebook bug with to_parquet (use write_parquet instead)
- Fix formatting in examine_df
- Fix bug with fill missing
- Fix triangular parcel detection
- Fix bug with hedonic ensembles
- Fix various export bugs
- Fix casting regression bug in MRA
- Update dependency versions
- Cleanup caching logic

## [0.4.5] - 2025-11-07
- Fix aggregation logic
- Fix duplicate handling
- Fix depencency issue

## [0.4.4] - 2025-11-06
- Fix broken geometry in _write_model_results
- Fix enrichment regression
- Version bumps for dependencies
- Updated documentation to explain pipeline module
- Updates to default dockerfile
- Modify pipeline to handle dataframe loading better in 01-assemble
- Cleanup + type annotations for utilities
- Fixed missing imports

## [0.4.3] - 2025-10-29
- Allow anoynmous read-only access to public Azure repositories
- Add "cloud.json" workflow
- Remove "bootstrap_cloud" notebook variable
- Move public data test repository to Azure
- Update documentation to reflect the change

## [0.4.2] - 2025-10-28
- Add "make_simple_scrutiny_sheet" function
- Rename "validate_arms_length_sales" to "filter_invalid_sales" and update its functionality
- Add "limit_sales_to_keys" function in SalesUniversePair
- First steps of calculating building height via overture enrichment
- Auto-calculate "assr_date_age_days" if "assr_date" is present
- Add "lake" and "airport" as open street map shortcut words
- Speed up clustering/caching
- Fixed spatial lag enrichment to not explode when inputs are length 0
- Fixed bootstrap ratio studies to not explode when inputs are length 0
- Fix street enrichment data reading
- Better error handling for missing census key

## [0.4.1] - 2025-10-09
- Fixed geometry CRS errors
- Removed obsolete "local_somers" predictive model
- Removed some unnecessary warnings
- Fixed a bug with "append" logic in dataframes not working correctly
- Added basic dockerfile

## [0.4.0] - 2025-10-06
- Moved .env file loading out of cloud_sync() and into init_notebook()
- Removed need to manually specify location of .env file -- system finds it automatically
- Routine dependabot updates to libraries and automated actions

## [Unreleased]