# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Renamed module `openavmkit.benchmark` → `openavmkit.model_runner` (it orchestrates the whole model run, not just the benchmark comparison, and the old name collided with the new research `benchmark/` harness). A deprecating compatibility shim remains at `openavmkit.benchmark` (re-exports everything, emits a `DeprecationWarning`) and **will be removed before 0.7.0** (tracked in AGENTS.md §9). The `BenchmarkResults` class keeps its name. No behavior change.

### Fixed / clarity
- Renamed the misleadingly-named inner tuning-CV helpers `_xgb_rolling_origin_cv` / `_lightgbm_rolling_origin_cv` / `_catboost_rolling_origin_cv` → `_xgb_kfold_cv` / `_lightgbm_kfold_cv` / `_catboost_kfold_cv` in `openavmkit/tuning.py`. They use `KFold(shuffle=True)` (random k-fold for hyperparameter selection), **not** temporal/rolling-origin CV; docstrings now say so. No behavior change. (`_catboost_kfold_cv` is currently unused — the live CatBoost tuner uses CatBoost's built-in `cv()`.)
- Documented in [AGENTS.md](AGENTS.md) §4 that reported holdout metrics are mildly optimistic because time adjustment and variable auto-reduction are fit on all sales (correct for production valuation; a small evaluation-only bias). Spatial lag is already train-only. A leakage-correct evaluation is the job of the planned publish-mode rolling-origin CV, not the everyday path.

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