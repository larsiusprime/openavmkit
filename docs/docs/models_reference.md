# Models reference

OpenAVMKit ships with around 20 prediction models, ranging from production-grade ML algorithms (XGBoost, LightGBM, CatBoost, GWR) to deliberately bad baselines (garbage, mean) used as sanity-check floors for evaluation.

This page is the authoritative reference for **what each model is**, **how to invoke it**, **what settings it takes**, and **when to use it**. It also explains the model-naming and dispatch system, including how to run multiple variants of the same engine.

For the broader modeling workflow, see [tutorial.md § B.7](tutorial.md#b7-run-notebook-3-model). For where the modeling settings fit in the larger settings tree, see [advanced_settings.md § 6](advanced_settings.md#6-modeling-control).

---

## 1. How model invocation works

Models live in `settings.json` under `modeling.models.<stage>`, where `<stage>` is one of `main` or `vacant`. The list of which models actually run for each stage is configured separately under `modeling.instructions.<stage>.run`.

Two layers, related but distinct:

```json
{
    "modeling": {
        "instructions": {
            "main": {
                "run": ["mra", "xgboost", "gwr"]
            }
        },
        "models": {
            "main": {
                "default": { "n_trials": 50, "ind_vars": [...] },
                "mra": { "ind_vars": [...] },
                "xgboost": { "n_trials": 100 },
                "gwr": { "ind_vars": [...] }
            }
        }
    }
}
```

- **`modeling.instructions.<stage>.run`** — the **list of model names** to actually invoke for that stage.
- **`modeling.models.<stage>.<name>`** — the **configuration** for each named model (independent variables, hyperparameters, etc.).

A model only runs if its name appears in the `run` list. Models defined in `modeling.models` but not listed in `run` sit dormant — useful for keeping configurations on hand without invoking them.

For configuring the `run` list, including per-model-group skip lists, see [advanced_settings.md § 6](advanced_settings.md#6-modeling-control).

### 1.1 Model name vs. engine

Each entry under `modeling.models.<stage>` is keyed by a **unique model name** — a string of your choosing. The entry can either:

1. **Use the name itself as the engine.** If the name matches a recognized engine (e.g. `"mra"`, `"xgboost"`, `"gwr"`), no `model` field is needed:

    ```json
    "xgboost": {
        "n_trials": 100,
        "ind_vars": ["latitude_norm", "longitude_norm", "bldg_area_finished_sqft"]
    }
    ```

2. **Specify the engine explicitly.** Set the `model` field to the engine's name. The model name and the engine name can differ:

    ```json
    "xgboost_full": {
        "model": "xgboost",
        "n_trials": 100,
        "ind_vars": [/* big list */]
    }
    ```

> **Terminology** — in this doc, "engine" means the underlying algorithm (`xgboost`, `mra`, `gwr`, etc.), and "model name" means the user-chosen key in `modeling.models.<stage>`. The `model` field on an entry selects which engine it uses; if absent, the model name is interpreted as the engine.

### 1.2 Multiple variants of the same engine

The `model` field is what lets you run **multiple variants of the same engine** — e.g. two XGBoost runs with different variable lists, two GWR runs with different bandwidth strategies, etc. Each variant gets its own unique name in `modeling.models`, declares the shared engine via `model`, and overrides whichever settings differ:

```json
{
    "modeling": {
        "instructions": {
            "main": {
                "run": ["xgboost_full", "xgboost_lite", "mra"]
            }
        },
        "models": {
            "main": {
                "default": { "n_trials": 50 },
                "xgboost_full": {
                    "model": "xgboost",
                    "n_trials": 100,
                    "ind_vars": [
                        "latitude_norm", "longitude_norm",
                        "bldg_area_finished_sqft", "land_area_sqft",
                        "bldg_age_years", "bldg_quality_num", "bldg_condition_num",
                        "neighborhood", "polar_radius", "polar_angle",
                        "spatial_lag_sale_price"
                    ]
                },
                "xgboost_lite": {
                    "model": "xgboost",
                    "n_trials": 50,
                    "ind_vars": [
                        "latitude_norm", "longitude_norm",
                        "bldg_area_finished_sqft", "land_area_sqft"
                    ]
                },
                "mra": {
                    "ind_vars": ["bldg_area_finished_sqft", "land_area_sqft", "bldg_age_years"]
                }
            }
        }
    }
}
```

Both XGBoost variants run, write outputs under their own model-name folders (`out/models/<model_group>/main/xgboost_full/`, `.../xgboost_lite/`), and contribute independently to ensemble averaging if it's enabled.

This is especially useful for:

- **Ablation studies** — "how does the model perform with vs. without the spatial-lag features?"
- **Comparing variable selections** — full feature set vs. carefully-pruned set
- **A/B-testing tuning depth** — `n_trials: 50` vs `n_trials: 200`
- **Trying different location encodings** for the same algorithm

### 1.3 The `default` entry

Every `modeling.models.<stage>` block can include a `default` entry. It's special:

- Its values fill in fields that other entries omit (think of it as inherited settings)
- It is **not itself run**, even if `"default"` appears in the `run` list

This is the cleanest way to share `n_trials`, `ind_vars`, or `interactions` across many entries:

```json
"main": {
    "default": {
        "n_trials": 50,
        "ind_vars": ["latitude_norm", "longitude_norm", "bldg_area_finished_sqft", "land_area_sqft"]
    },
    "xgboost": {},
    "lightgbm": {},
    "catboost": {}
}
```

All three tree-based models pick up the default `n_trials` and `ind_vars`.

### 1.4 The `*` suffix — sales chasing toggle

Putting an asterisk in the `model` value (e.g. `"model": "xgboost*"`) enables **sales chasing** — predictions on sold parcels deliberately copy the observed sale price with a small amount of random noise, simulating leakage. This is for **analytical purposes only**; it lets you measure how much a model's reported accuracy comes from genuinely good predictions versus inadvertent leakage. **Never** use sales chasing in production.

### 1.5 Per-model-group overrides

By default, the entries under `modeling.models.<stage>` apply to **every** model group. If you need a different model configuration for a specific model group — different `ind_vars`, different `n_trials`, even a different set of models entirely — nest the overrides under a key that matches the model group's id:

```json
{
    "modeling": {
        "models": {
            "main": {
                "default": { "n_trials": 50, "ind_vars": ["bldg_area_finished_sqft", "land_area_sqft", "bldg_age_years"] },
                "mra": {},
                "xgboost": {},

                "single_family_residential": {
                    "default": { "n_trials": 100 },
                    "mra": { "ind_vars": ["bldg_area_finished_sqft", "land_area_sqft", "bldg_age_years", "bldg_quality_num", "neighborhood"] },
                    "xgboost": { "ind_vars": ["latitude_norm", "longitude_norm", "bldg_area_finished_sqft", "land_area_sqft", "bldg_age_years", "spatial_lag_sale_price"] }
                }
            }
        }
    }
}
```

Resolution: for each model group, OpenAVMKit first checks whether `modeling.models.<stage>.<model_group_id>` exists. If it does, that nested dict is used in place of the top-level one (same shape — `default` plus model-name entries). If it doesn't, the top-level entries apply. There is **no merging** between the override block and the top level: the override replaces it wholesale for that model group, so include every model entry you want to run there.

- **Source** — see `_run_models`, `_prepare_ds`, `get_variable_recommendations`, and `get_model_location` in [openavmkit/model_runner.py](https://github.com/larsiusprime/openavmkit/blob/master/openavmkit/model_runner.py); all four do `model_entries.get(model_group, model_entries)`.
- **When to use** — different model groups need substantively different feature sets (e.g. single-family wants neighborhood encodings; vacant-land wants only land features), or you want to tune trees harder on one group than another.
- **When not to use** — small per-entry tweaks; just override `ind_vars` on the specific model entry at the top level if every group is otherwise the same.

---

## 2. Common entry fields

These fields can appear on any model entry under `modeling.models.<stage>.<name>`. Most are optional.

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `model` | string | (model name) | Which engine to use. See [§ 1.1](#11-model-name-vs-engine). |
| `ind_vars` | list of strings | (from `default`) | Independent variables to feed the model. |
| `interactions` | dict | empty | Variable-interaction config (mostly relevant for MRA). |
| `locations` | list of strings | from `field_classification.important.locations` | Location field names. Required for `local_area` and `multi_mra`. **Must NOT be a cardinality-collapsed field** — collapsing a location merges unrelated zones into `"Other"` and corrupts per-location fits; see [advanced_settings.md § 5.4](advanced_settings.md#54-dataprocesscollapse_sparse_categories). |
| `dep_var` | string | sale price field | Override the dependent variable. |
| `dep_var_test` | string | same as `dep_var` | Override the dependent variable used for test-set evaluation. |
| `n_trials` | int | 50 | Number of Optuna trials for tree-based hyperparameter tuning. |
| `use_gpu` | bool | true | (CatBoost) use GPU acceleration if available. |
| `intercept` | bool | true | (MRA, multi-MRA) include constant term. |
| `optimize_vars` | bool | false | (multi-MRA) run per-location variable optimization. |
| `field` | string | — | (`pass_through` engine only) the column to use as the prediction. |

Engine-specific quirks are documented in [§ 3](#3-engine-reference) below.

---

## 3. Engine reference

Engines are grouped by category. For each engine: name, one-line description, accepted settings, when to use, when not to use.

### 3.1 Production-grade predictive models

#### `mra` — Multiple Regression Analysis

Standard linear regression. Fast, interpretable, produces clean coefficients.

- **Accepts**: `ind_vars`, `interactions`, `intercept`
- **Native spatial awareness**: no — feed location via `latitude_norm`/`longitude_norm`, polar coords, or categorical region fields.
- **When to use**: simple, well-understood relationships; baseline against which more complex models are compared; when interpretability matters.
- **When not to use**: highly nonlinear value surfaces; jurisdictions where price interacts strongly with categorical fields without obvious linear encoding.

#### `multi_mra` — Multi-MRA (per-location linear regressions)

Fits separate MRA models for each unique value of one or more location fields. Captures geographic heterogeneity that a single global MRA misses.

- **Accepts**: `ind_vars`, `interactions`, `intercept`, `locations` (required), `optimize_vars`
- **Native spatial awareness**: yes — partitions on user-supplied region fields. Cannot run without `locations`.
- **When to use**: jurisdictions with strong submarket effects where coefficients should differ by neighborhood / market area.
- **When not to use**: locations are too granular (every region has too few sales for stable per-location fits).

#### `xgboost`, `lightgbm`, `catboost` — Gradient-boosted tree models

Production-grade tree-based ensembles. Handle nonlinearities, interactions, and missing data well; fit categorical variables natively in OpenAVMKit's wrappers.

- **Accepts**: `ind_vars`, `n_trials`. CatBoost also accepts `use_gpu`.
- **Hyperparameter tuning**: yes, via Optuna. Tuned parameters cached at `<outpath>/<slug>_params.json` (see [advanced_settings.md § 8.4](advanced_settings.md#84-saved-model-parameters-different-semantics)).
- **Native spatial awareness**: no — feed location via `latitude_norm`/`longitude_norm`, polar coords, or categorical region fields.
- **When to use**: most production AVM workloads. Often the strongest single-model performers.
- **When not to use**: very small training sets; when interpretability is a hard requirement.

#### `ngboost` — Probabilistic gradient boosting (NGBoost)

Natural-gradient boosting that predicts a full probability distribution per parcel, not just a point value. The distribution **mean** is used as the prediction (so it slots in like the other engines), and the per-parcel predictive **standard deviation** is written out as an extra `prediction_std` column on the universe output (and merged onto sales by `key`), following the same per-parcel pattern as `spatial_lag` confidence.

- **Accepts**: `ind_vars`, `n_trials`.
- **Hyperparameter tuning**: yes, via Optuna (`learning_rate`, `n_estimators`, `minibatch_frac`, base-learner `max_depth`). Tuned parameters cached at `<outpath>/<slug>_params.json`. NGBoost's natural-gradient boosting is **notably slower** than xgboost/lightgbm/catboost — keep `n_trials` small (e.g. 5).
- **Categoricals**: not native. Its sklearn tree base learner is numeric-only, so categoricals are encoded to integer codes internally (same encoding used for SHAP).
- **Native spatial awareness**: no — feed location via `latitude_norm`/`longitude_norm`, polar coords, or categorical region fields (as with the other tree engines).
- **Params / contributions**: emitted via **exact tree-SHAP**. Although SHAP's `TreeExplainer` has no native NGBoost support, NGBoost's mean is an additive ensemble of per-stage trees, so its SHAP is computed exactly as the weighted sum of per-tree explanations. Standard `params_<subset>.csv` / `contributions_<subset>.csv` describe the **point estimate** (so NGBoost is a full ensemble member). A parallel `params_std_<subset>.csv` / `contributions_std_<subset>.csv` describe what drives the **predictive uncertainty** — note these are in **log-std space** (`base + Σ contributions = log(std)`), since the decomposition is additive on `logscale`, not on the raw std.
- **When to use**: when you want calibrated per-parcel uncertainty alongside the point estimate.
- **When not to use**: when SHAP attributions are required, or when training time is tight.

#### `gwr` — Geographic Weighted Regression

Linear regression where each prediction is weighted by spatial proximity to training points. The weighting kernel uses lat/lon directly.

- **Accepts**: `ind_vars`
- **Hyperparameter tuning**: bandwidth search. Cached at `<outpath>/<model_name>_bw.json`.
- **Native spatial awareness**: **strictly native** — lat/lon enter the algorithm via the kernel, not as feature columns. Don't include `latitude`/`longitude` in `ind_vars` — they're auto-stripped to avoid collinearity.
- **When to use**: jurisdictions with strong, smooth spatial gradients; when you want spatially-varying coefficients.
- **When not to use**: very large datasets (GWR scales poorly); jurisdictions with sharp geographic discontinuities (better captured by categorical regions or multi-MRA).

#### `kernel` — Kernel regression

Nonparametric regression using local-window weighting. As OpenAVMKit invokes it, longitude and latitude are automatically prepended to the variable matrix, so the kernel weights by geographic proximity in addition to feature similarity.

- **Accepts**: `ind_vars`
- **Hyperparameter tuning**: per-variable bandwidth search. Cached at `<outpath>/kernel_bw.pkl`.
- **Native spatial awareness**: yes — `longitude` and `latitude` are auto-injected. Like GWR, raw lat/lon and `latitude_norm`/`longitude_norm` are stripped from `ind_vars`.
- **When to use**: smooth nonlinear value surfaces with a moderate number of features.
- **When not to use**: high-dimensional feature spaces (curse of dimensionality); large datasets (slow).

#### `spatial_lag` and `spatial_lag_area`

Use spatial-lag features as the predictor — the average sale price (or price-per-area) of a parcel's neighbors becomes the prediction. Requires `data.process.enrich.spatial_lag` to have run.

- **`spatial_lag`** — predicts using a single spatial-lag-of-price feature.
- **`spatial_lag_area`** — predicts using lagged price-per-area, multiplied by the parcel's own area.
- **Accepts**: nothing model-specific (variables are fixed)
- **When to use**: when neighborhood-average pricing is the dominant signal in the jurisdiction; as a strong sanity-check baseline.
- **When not to use**: when within-neighborhood variation is the main thing you want to capture.

#### `local_area` — Local-area average pricing

Computes per-area value averages keyed by user-supplied location fields, then applies them at predict time. "Houses in River Heights average $X per sqft."

- **Accepts**: `locations` (required)
- **Native spatial awareness**: yes — through user-supplied categorical region fields, not coordinates. Cannot be invoked without `locations`.
- **When to use**: simple, interpretable baseline for residential modeling; when assessor neighborhoods are well-drawn.
- **When not to use**: feature-rich modeling where building characteristics vary widely within each region.

### 3.2 Reference / pass-through models

Not "predictive" in the algorithmic sense — they pass through an existing field (or the ground-truth target) as the prediction. Used to anchor evaluation against a known reference.

#### `assessor`

Uses the assessor's recorded value (`assr_market_value` for main, `assr_land_value` for vacant) as the prediction. Lets you compare your model's accuracy to the existing assessor's.

- **Accepts**: nothing model-specific
- **When to use**: always, in fact — it's the natural benchmark.
- **When not to use**: when you don't have assessor values in your data.

#### `pass_through`

Generalized assessor — uses any user-specified column as the prediction.

- **Accepts**: `field` (required)
- **When to use**: comparing your model against any external valuation (a vendor's AVM, a previous OpenAVMKit run, etc.).

```json
"vendor_avm": { "model": "pass_through", "field": "vendor_avm_value" }
```

#### `ground_truth`

Uses the dependent variable itself as the prediction (`true_market_value` or `true_land_value`). Predictions are perfect by construction.

- **Accepts**: nothing model-specific
- **When to use**: synthetic-data testing where ground truth is known; establishing an upper-bound on achievable accuracy.
- **When not to use**: real production runs (it's not a real model).

### 3.3 Naive baselines

Deliberately simple models that establish the *floor* of acceptable performance. If your real models can't beat these, you have a problem.

#### `naive_area`

`prediction = (global average price per sqft) × (parcel's sqft)`. Assumes uniform per-area pricing across the jurisdiction.

- **When to use**: minimum baseline for area-driven modeling; sanity check.

#### `mean` / `median`

Predicts the global mean (or median) sale price for every parcel.

- **When to use**: absolute floor baseline. Real models should crush these.

#### `garbage` / `garbage_normal`

Random predictions (uniform or normal-distributed). Establishes what "literally noise" performance looks like.

- **When to use**: sanity-check that your evaluation pipeline correctly identifies bad models. Never as a real prediction.

### 3.4 Special: ensemble

`ensemble` is not invoked through `modeling.models` — it runs automatically after all the configured models, combining their predictions. Configure under `modeling.instructions.<stage>.ensemble`:

```json
"main": {
    "run": ["mra", "xgboost", "gwr"],
    "ensemble": { "type": "median" }
}
```

Three combination strategies:

- **`type: "median"`** (aka `"default"`, the value used when `type` is omitted) — global greedy backward-elimination, then combines the surviving subset via per-row **median** (robust to a single model going wild on a parcel).
- **`type: "mean"`** — identical greedy selection, but combines via per-row **mean** (every surviving model pulls proportionally).
- **`type: "local"`** (only for `main`) — picks the *single best* model per location at predict time; no combining, one model wins per neighborhood.

For `median`/`mean` you can **manually pick the ensemble members** with an explicit `models` list — by default it is a whitelist (those exact models, no pruning). Add `"optimize": true` to instead greedily prune from that list, or omit `models` to optimize over every model that ran. See [advanced_settings.md → `modeling.instructions.<stage>.ensemble`](advanced_settings.md#modelinginstructionsmainvacantensemble).

All three also **reassemble per-feature `params_<subset>.csv` / `contributions_<subset>.csv`** for the ensemble itself (and stamp `ensemble_meta.json`), so the ensemble is as interpretable as the individual models — see [tutorial.md § per-model output](tutorial.md). See [advanced_settings.md → `modeling.instructions.<stage>.ensemble`](advanced_settings.md#modelinginstructionsmainvacantensemble) for full configuration including the `locations` list.

---

## 4. Worked example: ablation study

Compare a "rich" XGBoost with a "lean" XGBoost to see how much each variable group contributes.

```json
{
    "modeling": {
        "instructions": {
            "main": {
                "run": ["assessor", "xgboost_rich", "xgboost_lean", "xgboost_no_spatial"],
                "ensemble": { "type": "default" }
            }
        },
        "models": {
            "main": {
                "default": {
                    "n_trials": 50
                },
                "xgboost_rich": {
                    "model": "xgboost",
                    "ind_vars": [
                        "latitude_norm", "longitude_norm", "polar_radius", "polar_angle",
                        "neighborhood", "market_area",
                        "bldg_area_finished_sqft", "land_area_sqft",
                        "bldg_age_years", "bldg_quality_num", "bldg_condition_num",
                        "spatial_lag_sale_price",
                        "proximity_to_parks", "proximity_to_water_bodies"
                    ]
                },
                "xgboost_lean": {
                    "model": "xgboost",
                    "ind_vars": [
                        "latitude_norm", "longitude_norm",
                        "bldg_area_finished_sqft", "land_area_sqft",
                        "bldg_age_years", "bldg_quality_num", "bldg_condition_num"
                    ]
                },
                "xgboost_no_spatial": {
                    "model": "xgboost",
                    "ind_vars": [
                        "bldg_area_finished_sqft", "land_area_sqft",
                        "bldg_age_years", "bldg_quality_num", "bldg_condition_num"
                    ]
                }
            }
        }
    }
}
```

After a run, compare the test-set metrics across `xgboost_rich`, `xgboost_lean`, and `xgboost_no_spatial` to see how much spatial features and proximity features actually contribute.

---

## 5. See also

- [Tutorial § B.7 → Modeling best practices](tutorial.md#modeling-best-practices) — variable selection, big five, location encoding
- [Tutorial § B.7 → `try_models` vs `finalize_models`](tutorial.md#try_models-vs-finalize_models-iterate-fast-then-commit) — workflow for iterating
- [Advanced settings § 6 — Modeling control](advanced_settings.md#6-modeling-control) — `run` lists, per-group skip, feature-selection thresholds
- [Advanced settings § 8.4 — Saved model parameters](advanced_settings.md#84-saved-model-parameters-different-semantics) — caching tuned hyperparameters
- [AGENTS.md § 7 → Adding a new model](https://github.com/larsiusprime/openavmkit/blob/master/AGENTS.md) — for contributors wiring up a new engine
- [openavmkit/utilities/modeling.py](https://github.com/larsiusprime/openavmkit/blob/master/openavmkit/utilities/modeling.py) — model class definitions
- [openavmkit/model_runner.py](https://github.com/larsiusprime/openavmkit/blob/master/openavmkit/model_runner.py) — dispatch and orchestration
