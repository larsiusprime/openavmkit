# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

### ⚠️ Behavior changes

- **Census: every code the ACS writes for a figure it doesn't give is now missing.** Only -666666666 was caught; the others (-999999999, -888888888, -555555555, -333333333, -222222222) went in as a real, hugely negative income, rent or population. They're now missing, like the first.
- **Elevation is fetched with a margin round the parcels** (8 cells, at least 100 m). Reprojecting the DEM to UTM leaves empty cells along its edges, and a cell's slope needs its neighbours, so parcels on the outer edge of the locality got no slope (and, right on the edge, no elevation). They now get both. The DEM's cache key includes the bbox, so it's fetched once again.

- **The XGBoost and CatBoost models are optional extras too.** `xgboost` and `catboost` (about 560 MB with what they bring, plotly and graphviz among them) are no longer installed with OpenAVMKit, and nothing loads them until a model uses them. **What to do:** for the `xgboost` model, `pip install "openavmkit[xgboost]"`; for `catboost`, `pip install "openavmkit[catboost]"`. Without them, a run whose settings name one stops before it trains anything, with one ImportError naming every missing engine and the command that installs them all (e.g. `pip install "openavmkit[catboost,xgboost]"`), instead of after the models before it have trained; inference experiments (`infer.model.experiment`) try the other models, leaving out `xgboost` and `ensemble` and saying so. `[dev]` and the Docker image include both. See [Getting Started](docs/docs/getting_started.md#what-each-install-gives-you).
- **Cloud storage and the NGBoost model are optional extras.** `azure-storage-blob`, `azure-core`, `huggingface-hub`, `paramiko` and `ngboost` are no longer installed with OpenAVMKit, and nothing loads them until it's used: importing the pipeline no longer loads every cloud SDK. **What to do:** for `cloud_sync` (Azure, Hugging Face or SFTP), `pip install "openavmkit[cloud]"`; for the `ngboost` model, `pip install "openavmkit[ngboost]"`. Without them, `cloud_sync` or an `ngboost` model raises an ImportError naming the extra to install; everything else works as before. `[dev]` includes both. See [Getting Started](docs/docs/getting_started.md#what-each-install-gives-you).
- **CatBoost tuning is much faster, and the shipped CatBoost model now matches what was tuned.** The tuner now optimizes an RMSE loss (still scored on MAPE) over symmetric trees only, with categorical feature combinations off (`max_ctr_complexity=1`). Previously it optimized CatBoost's MAPE loss, which converges so slowly on price targets that trials rarely stopped early and the search drifted into Depthwise/Lossguide depth-10 trees. On the public localities: Guilford single-family went from 3,418s to 462s for the CatBoost group run (out-of-fold RMSE $76.9k → $57.8k); Petersburg single-family CatBoost models ran 2–3× faster. `_SEARCH_SPACE_VERSION` is bumped, so saved CatBoost params and resume journals are invalidated and re-tuned once. CatBoost predictions will change.
- **Sales scrutiny now catches multi-parcel ("bulk") deeds whose other rows were invalid, and stops dropping unrelated sales that share a date and price.** A bulk deed stamps one bundle price onto every parcel it covers, so none of those sales is usable. Which sales `run_heuristics` drops changes in three ways:
  - **New heuristic, `flag_bulk_deed`.** `process_data` now records `sale_parcel_count` (parcels on each deed) *before* invalid sales are filtered out, and any sale whose deed covered more than one parcel is dropped. This catches an "orphan" bulk deed reduced to one surviving row, which the duplicate heuristics can't see. Deeds are identified by `analysis.sales_scrutiny.deed_id` (plus sale date and `jurisdiction`) when it's set, otherwise by `key_sale`. Report: `out/sales_scrutiny/multi_parcel_bulk_deeds.xlsx`. On Philadelphia, sales like these had inflated the vacant-land median price per sqft about 4×.
  - **Repeated date + price is narrowed only when deed IDs are available.** Without `deed_id`, every repeated (date, price) is still flagged, because that's the only trace a bulk deed leaves. With `deed_id`, sales on separate, known deeds that happen to share a date and price (a round price, or a builder selling identical lots on separate deeds) are kept. Rows in the group that share a deed, have a missing deed, or are the same parcel twice are still flagged.
  - **Repeated deed + date no longer treats missing deed IDs as one shared deed.** Previously every sale with a blank deed on the same date was flagged as a duplicate of the others.

  Synthesizes [#349](https://github.com/larsiusprime/openavmkit/pull/349) and [#351](https://github.com/larsiusprime/openavmkit/pull/351). See [Sales scrutiny heuristics](docs/docs/advanced_settings.md#analysissales_scrutinydeed_id-and-analysissales_scrutinyjurisdiction).
- **Ratio-study quantile breakdowns now have exactly the number of bins you ask for.** Bins used to start at 0, so the first slice ran from 0 to the column's minimum and held a single sale: `quantiles: 10` (the default `sale_price` breakdown) produced 11 slices, the first one empty in practice. Bins now start at the minimum, and their edges are actual data values, so boundaries in the labels can move by one sale's worth (e.g. `239,500` → `239,000`). If many sales share the minimum value (e.g. 1-story homes), they still get a bin of their own.
- **VEI figures change.** The VEI ranks sales into percentile groups by a market value proxy; it was half the sale price plus the *whole* median-adjusted value, and is now the average of the two (½ each), as the IAAO's May 2026 exposure draft (Appendix E) and openratiostudy.com define it, and as the PRB's proxy already was. Every VEI moves where the reweighting moves sales between groups: the benchmark's (holdout and study set, trimmed and untrimmed) and the assessment-quality studies'. Usually a small shift at the group edges, but a VEI near its threshold can change its verdict. (The draft's worked example gives the same groups either way.)
- **A blank in a yes/no text column is now missing, not True.** Lower-casing turned a missing value into `"<na>"` or `""`, neither of which counted as missing, so blanks became True whatever `na_handling` said. Now `na_handling` decides: `"true"` makes them True; `"false"`, or unset, makes them False. **This can remove sales:** a sale with a blank `valid_sale` used to count as valid and now doesn't, and a blank `vacant_sale` now reads as improved. To keep the old reading, set `na_handling: "true"` on the field. Columns already stored as booleans (Parquet) are unaffected.
- **A distance feature from your own file that fails now stops the run**, with the reason, instead of printing an error and silently leaving the feature out (OpenStreetMap features still only warn: a download can fail for reasons outside your data). And a distance feature's `enabled` now defaults to `true`, as documented: before, a feature listed without `enabled` was picked but fetched as nothing. **Distances listed without `enabled` are now computed** (for OpenStreetMap ones, downloaded); set `enabled: false` to leave one out.
- **`ipython`, `pytest` and `pipreqs` are no longer OpenAVMKit's runtime requirements.** Only notebooks, the test suite and maintainers use them. OpenAVMKit no longer imports IPython when it loads (`openavmkit.utilities.notebook` shows tables richly inside a notebook and prints them elsewhere), so it runs in scripts and apps without it. **`pytest` and `pipreqs` are no longer installed with OpenAVMKit.** (IPython still is, for now, through `layeredcompmodel`, which lists `ipykernel` as a requirement.) **What to do:** to run the notebooks, `pip install jupyter`, as before; to run the tests from a Git checkout, `pip install -e ".[dev]"` (adds `pytest`, `ruff` and `ipython`). See [Getting Started](docs/docs/getting_started.md#what-each-install-gives-you).

### New features

- **The middle of the parcels without the dissolve:** `locality.center_method: "area_weighted"` finds the center that polar coordinates are measured from (when `locality.center` isn't set) as the area-weighted mean of the distinct parcels' centroids, instead of the centroid of all of them dissolved into one (`"union"`, still the default). It's the same point where parcels don't overlap (exact duplicates, such as stacked condo units, count once in both), in milliseconds instead of seconds: the dissolve took about 5 s at 50,000 parcels. Where shapes only partly overlap, the two can differ slightly. See [Basic geometric enrichment](docs/docs/advanced_settings.md#41-basic-geometric-enrichment--dataprocessenrichbasic).

- **Census: a block group's withheld figure from its tract, and a parcel between block groups matched to the nearest.** `data.process.enrich.census.tract_fallback: true` fills a block group's missing figure (the ACS withholds median rent where there are too few renters, often) with its census tract's, at the cost of one more Census API call. `nearest_m` (meters) gives a parcel whose centroid is in no block group the nearest one within that distance. Both off by default. See the settings template's `__census`.

- **Explicit `bins` for ratio-study breakdowns.** `{"by": "pct_vacant", "bins": [0, 0.1, 0.25, 0.5, 1.0], "bin_labels": ["0-10%", "10-25%", "25-50%", "50-100%"]}` slices on value edges you choose, for columns where equal-count quantiles or integer `slice_size` labels don't work (e.g. 0–1 shares). Bad edges or labels raise a clear error; sales outside the edges are left out with a warning. See [Ratio study breakdowns](docs/docs/advanced_settings.md#analysisratio_studybreakdowns). From [#366](https://github.com/larsiusprime/openavmkit/pull/366).
- **Running a stage without Jupyter:** `openavmkit.headless.run_stage(project_dir, stage, on_progress, cancel)` runs the assemble, clean and model sequences from a script or an app, restoring the working directory afterwards. **Progress events** (`openavmkit.progress`: `reporting()`, `report()`, cooperative cancellation) say which step, model group and model it's on, including inside enrichments (each OpenStreetMap feature, Overture batch, elevation stage and census download), and an optional part can be skipped without stopping the run. Nobody listening: nothing changes, as in a notebook.
- **GeoPackage, shapefile and GeoJSON files load** in `data.load`, as the tutorial already said (through GDAL/pyogrio, with the same column selection as CSV and Parquet; a GeoPackage entry can name its `layer`; extensions in any case, `PARCELS.SHP`). A shapefile's sidecar files must sit next to it. A distance feature from your own file no longer needs a `name` column.
- **The data dictionary says where each field comes from:** every entry has a `"source"`: `"input"` (only ever from your data), `"derived"` (always computed by OpenAVMKit: a loaded column of that name is overwritten), `"either"` (computed when missing, kept when given) or `"reserved"` (a name OpenAVMKit means to compute but nothing writes yet). `key_sale`, the sale ID the sales merge joins on, has an entry.
- **Calcs: `upper`**, for IDs written in lower case in one file and upper in another (`["upper", "PARCEL_ID"]`).
- **`drop_fields` patterns:** a name with `*` matches every column it fits, for columns named as they're made.
- **Condo land shares:** `land_share.method: "equal"` (the building's land in equal shares, for rolls with no land or finished area on unit rows), `"floor_area"` when the units' finished area comes from another file joined by the merge, and `"none"` for units whose land is the shared parcel's (mobile-home parks, leased land). After the merge, every unit's `land_area_sqft` is its share.
- **Only the spatial lags you list:** `data.process.enrich.spatial_lag.only: true` computes exactly the sale-price lags named in `sales` (`"price"`, `"price_impr"`, `"price_land"`, `"price_vacant"`) and the characteristic lags in `fields`, none of the built-in ones besides; the derived floor area ratio and bedroom density only when listed. The `spatial_lag` and `spatial_lag_area` models say which `sales` entry to add when theirs is left out. Unset: every lag, as before. See [Advanced settings § 4.8](docs/docs/advanced_settings.md#48-spatial-lag--dataprocessenrichspatial_lag).
- **Sales scrutiny for review.** `analysis.sales_scrutiny.drop.clusters` / `.drop.heuristics` set whether flagged sales are dropped; `keep_file` lists sales that are flagged but never dropped; with `write_flags: true`, every flagged sale, with its check, numbers and whether it was kept or dropped, is written to `out/sales_scrutiny/flags.parquet` (off by default: nothing is written); `passes: 2` finds outliers another outlier hid. Each heuristic can be turned off on its own (`heuristics.<name>: false`). See [Advanced settings § 7](docs/docs/advanced_settings.md#analysissales_scrutiny--drops-keeps-passes-and-the-flags-file).
- **Filled values can be recorded:** with `data.process.write_filled: true`, `fill_unknown_values_sup` writes `out/filled.parquet`, which parcels had each field filled rather than found in the data. Off by default: nothing is written.
- **Time adjustment files as a price index** (`period`, `index`, by day, month, quarter or year; interpolated to days), checked with `read_schedule`, which says every problem; and **borrowing another model group's time adjustment** (`data.process.time_adjustment.borrow`). Before, a file had to be a day-by-day schedule, and anything coarser matched no sale dates and left prices unadjusted. See [Advanced settings § 3](docs/docs/advanced_settings.md#3-time-adjustment).
- **Sales scrutiny as a diagnosis.** `analysis.sales_scrutiny.patterns: "strict"` rates each flagged sale's price, price per area and size against its cluster (low, in range, high, by the thresholds sales are flagged at) and gives it exactly one of the five patterns, or `other` (flagged only, never dropped); the ratings and the cluster's median price and size are in the flags file. New opt-in checks: `nominal_price`, `built_after_sale`, `vacant_priced_improved`, `same_surname` (with `surnames()` and the `deed_grantor_surname` / `deed_grantee_surname` fields). See [Advanced settings § 7](docs/docs/advanced_settings.md#analysissales_scrutiny--drops-keeps-passes-and-the-flags-file).
- **Value-field studies for checking a roll** (`run_value_field_studies`): `trim_outliers` (the study again without IQR outliers, beside the study of every sale; the trimmed sales listed and a per-sale table written), `median_ci="iaao"`, `sale_field`, `standards` / `classes` judging each model group (`IAAO_STANDARDS`), VEI percentile groups and its reading as the May 2026 exposure draft's Appendix E, the Modified Kakwani Index (`calc_mki`), and neighborhood price-level statistics (`region`). Judged as the May 2026 exposure draft judges: COD ranges by property category, the level by its confidence interval, the PRB's good and unacceptable ranges, the PRD reported only. Trimming locates quartiles as the Standard's Appendix C. All opt-in.
- **A first pass for the correlation test:** `feature_selection.thresholds.correlation_strength` (`calc_correlations(..., min_strength=)`) drops a variable whose correlation with price is weaker than it right away, before the combined scores, so it doesn't lower the others' clarity. `calc_correlations` also returns the correlation matrix and the variables the first pass dropped. Unset: as before.
- **Variable tests one by one:** `get_variable_recommendations(..., return_tests=True)` also returns each test's results for every variable (representation, correlation, elastic net, R², p-value, t-value, VIF; before and after their drops), with the thresholds, weights and one-hot columns, so a front end can show the tests one at a time. `calc_representation` keeps its shares as numbers (`raw`) beside the formatted table. Opt-in.
- **Model results returned:** `finalize_models` returns each model group's results (benchmarks included), which it already held to write them. `try_models(..., return_results=True)` (and `run_models(..., return_results=True)`) returns them from a try too; they're then held in memory until every group has run. Unset, a try keeps nothing and returns None, as before, so each group's results are let go as the next starts.
- **The study set on the holdout's predictions:** `analysis.ratio_study.study_set_predictions: "holdout"` judges each model on the study set by its holdout predictions (each sale valued by a model that never saw it) rather than by its predictions on every sale, most of which it trained on; so the models and the assessor are compared there like for like. Default `"all"`, as before. See [Advanced settings](docs/docs/advanced_settings.md#analysisratio_studystudy_set_predictions).
- **The academic figures as numbers:** `academic_metrics(all_results, max_trim)` returns the benchmark's academic table (count, MAPE, MSE, RMSE, median and mean ratio, VEI and its significance, slope), untrimmed and trimmed, as data frames; before, they existed only as the printout's text.
- **Every trimmed figure in the benchmark:** the benchmark's frames (`df_stats_test`, `df_stats_test_post_val`, `df_stats_full`) also carry `median_ratio_trim`, `vei_trim` and `vei_sig_trim`, on the same trimmed sales as `cod_trim`, so a caller can show a complete trimmed table. Its printed tables leave them out, as before.
- **Condo units take their parcel's fields:** `data.process.condos.inherit` lists fields (zoning, a school district…) each unit takes from the parcel it's on, after the merge, where its own is blank (`apply_inherited_fields`). Before, a unit had only its shape from its parcel. Unset: as before. See [Advanced settings § 4.12](docs/docs/advanced_settings.md#412-condo-resolution--dataprocesscondos).
- **Outside values judged like the assessor:** `analysis.ratio_study.outside_models` lists models whose values were made outside openavmkit (a vendor's AVM as a `pass_through` model, say); they're left off the random holdout with the assessor, since they may have seen those sales. Unset: as before. See [Advanced settings](docs/docs/advanced_settings.md#analysisratio_studyoutside_models).
- **What the time adjustment index is of, per group:** `data.process.time_adjustment.per.<group>` (`"impr"` or `"land"`) sets whether a group's index is of price per finished area or per land area (`calculate_time_adjustment(..., per=)`). Unset: worked out from the sales, as before. Condo units, whose land is a small share of their parcel's, could be indexed on land and get a meaningless index.
- **Each standard its own targets:** `run_value_field_studies(standards={..., "by_class": {cls: {...}}})` gives a class its own `level`, `prb`, `prb_unacceptable`, `vei` and `min_sales` in place of the ones for every class (a state's standards, an office's), and a class can have any name `standards["cod"]` has. Unset: as before.
- **The correlation test without overlap:** `feature_selection.correlation_overlap: false` (`calc_correlations(..., overlap=False)`) judges each variable by its correlation with price alone, and drops none for overlapping another, leaving overlap to the VIF test. Default `true`, as before.
- **Progress per model group:** `do_per_model_group` reports which model group it's on (`progress.detail`), so time adjustment, equity clusters and scrutiny say where they are.

### Performance

- **Sales scrutiny's clustering is several times faster, with the same clusters.** Splitting each cluster on a numeric field (`_crunch`, called once per cluster per field: thousands of times on a county) took nine pandas `quantile` calls and up to three `pd.cut`s, each mostly pandas' own overhead on a small cluster. It now takes one `np.percentile` call and `np.searchsorted`, reproducing pandas' arithmetic and types exactly (columns NumPy can't reproduce pandas on, such as dates and float32, still go through pandas). On a 50,000-parcel test county, `make_clusters` went from 13.9 s to 3.6 s and the whole sales scrutiny run from 25.6 s to 15.6 s, and every output (clusters, flags, the sales set aside, the hand-off to the next stage) was identical. `tests/test_crunch_equivalence.py` holds it to the pandas version on every kind of column.

### Fixed

- **Ratio-study breakdowns no longer crash the whole ratio study** on a column with any NaN values (common for enrichment fields with partial coverage) or with negative values ("bins must increase monotonically"). NaN rows are left out of that breakdown; negative values are binned normally. A column with no numeric values at all is skipped with a warning. From [#366](https://github.com/larsiusprime/openavmkit/pull/366).

- The final CatBoost fit dropped the tuner's fixed settings (Optuna's `best_params` only holds searched keys), so the shipped model trained with CatBoost's defaults — RMSE loss and MVS bootstrap, which silently discarded the tuned `bagging_temperature`. The fixed settings are now merged into the final fit.
- CatBoost tuning could crash with `FileExistsError: 'catboost_info'` when parallel CV folds started together in a fresh locality directory. Tuning now uses a private temporary `train_dir`, and no longer leaves `catboost_info/` in the locality folder.
- The `do_shaps` summary for CatBoost used shap's generic `TreeExplainer` instead of CatBoost's native SHAP, which took ~25 minutes on Guilford single-family. It now uses the same native (Approximate) path as the contribution files.
- The Overture building fetch used ~4.4 GB of memory even for a 1 km bounding box, enough to get killed on small workers. PyArrow doesn't use Parquet statistics on nested fields like `bbox.xmin` to skip data, so it was decoding row groups from the whole planet. The fetch now reads each file's footer first and scans only the row groups whose bbox statistics overlap the request. On a 1 km box in Greensboro, NC: 4.4 GB → 0.56 GB and 32s → 7s; on a 10 km box (46k buildings): 4.6 GB → 0.67 GB and 47s → 7s. Results are identical. Diagnosis credit to [#368](https://github.com/larsiusprime/openavmkit/pull/368).
- **Silent parcel loss from the default de-duplication key.** A geometry file with no `dupes` setting was de-duplicated on "the first non-geometry column", whose position varied between runs; on a parcel layer it could pick an attribute such as building type and drop almost every parcel. It now uses the file's key (`key`, `key2`, `key3`), and warns only when there's none. Columns load in the file's own order, so any fallback is the same every run.
- **CSV columns cast to `bool` load:** they're read as text and converted after loading (y/n, yes/no, true/false, 1/0), instead of failing in the CSV reader.
- **Polar coordinates with `locality.center` on a projected file** (State Plane, UTM): the centre, given in degrees, was placed in the file's own coordinates, far away, so every `polar_radius` and `polar_angle` was wrong. It's placed in degrees now. Re-run if you use a centre with a projected parcel file.
- **Point and line layers** (schools, roads) loaded empty: every geometry file was cleaned to polygons. Layers of only points or lines keep them now, so distances to them are no longer silently missing.
- **A field two files carry:** the first file's blanks are filled from the second when exactly one later file clashes, as they already were with two or more; before, the second file's values were dropped. The leftover `<field>_<file id>` column no longer appears.
- **Elevation and slope for every parcel:** parcels sharing a shape (condo units) and small parcels could get none (each grid cell keeps one parcel's label). Each distinct shape is now measured once and given to every parcel with it; a shape with no cells takes the cell under its centre.
- **Permits on sales, sale by sale:** every sale of a parcel got the result of one of them (a 2016 sale marked a teardown because of a 2020 demolition). And `calc_effective_age` no longer stops when the sales record has no year built (the sales take the universe's).
- The settings template listed `reno_date` among improvement categories, which filled its blanks with "UNKNOWN"; it's with the dates now.
- `read_schedule` quotes a schedule file's problems as the file has them (`n/a`, a blank).
- Layered comps (`lcomp`) is treated as an ML model, like the gradient-boosted trees, when variables are reduced: a run with lcomp no longer starts the variable recommendations because of it, and lcomp with the default variables gets the full list, not the reduced one.
- `get_variable_recommendations(..., tests_to_run=...)` without `"corr"` (VIF alone, say) failed while writing its report's correlation table; that table now reads N/A.
- Running the `assessor` model beside tree models only (say `run: ["lightgbm", "assessor"]`) ran the variable recommendations, as if a linear model needed them, and could fail there. The assessor takes no variables: like `pass_through`, it no longer asks for them.
- `BenchmarkResults.print()` formatted its own tables in place: after the benchmark was printed (`try_models` prints it even with `verbose=False`), `df_stats_test`, `df_stats_full` and `df_time` held display text such as "5,408", "-130" and "N/A" instead of numbers, and a second `print()` failed. It formats a copy now.
- `get_variable_recommendations(..., tests_to_run=...)` without `"r2"` failed while combining the scores (it merged a blank R² table and scored `adj_r2` regardless). The R² points are left out when R² wasn't run.
- `calc_elastic_net_regularization` failed on a column with no values at all (its median is blank, so the fill left it blank): `get_variable_recommendations` then failed for the whole model group. Such a column is left out of the elastic net now, with a warning.
- Dates the settings template lists as categories (`demo_date`, `reno_date`, `sale_date`) were turned into text and their blanks filled with "UNKNOWN", leaving columns of dates and strings that Parquet couldn't write: the clean stage's output then failed (with only a warning). Dates stay dates, a missing one stays missing.
- A sales scrutiny model group with no flagged sales lost its sales from the scrutiny output and report counts.
- The settings template's `analysis.sales_scrutiny.fields_categorical_improved` was never read; it's `fields_categorical`.
- The cluster scrutiny output now includes each cluster's `median`.
- Sales scrutiny's sold-as-vacant check no longer fails when there's no `bldg_year_built`: it's skipped, with a warning.

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