# Build a jurisdiction from scratch

This is the end-to-end walkthrough for taking a real jurisdiction from raw data to a working AVM. It comes in two parts:

- **Part A — Smoke test with sample data.** A 10–15 minute exercise on a public dataset that confirms your install works and shows you what successful output looks like at every stage. **Do this first.**
- **Part B — Onboard your own jurisdiction.** The real walkthrough: from "I have an assessor extract" to "I have predictions, ratio studies, and equity reports."

Each section ends with pointers into the reference docs ([advanced_settings.md](advanced_settings.md), [recipe.md](recipe.md), [config.md](config.md)) — the tutorial introduces what you need to make decisions, the reference docs cover every option in detail.

**For domain experts (assessors, IAAO-trained appraisers):** skim Part A to see the rhythm, then jump to Part B section 4 ("Author a minimum viable settings.json"). The IAAO terminology callouts are for generalists; you can ignore them.

**For technical generalists (data scientists, engineers):** read both parts in order. The blockquoted callouts explain mass-appraisal terminology (model groups, ratio studies, COD/PRD/PRB, equity) that the rest of the doc assumes.

---

## Part A — Smoke test with sample data

The Center for Land Economics publishes a small public dataset for **Guilford County, North Carolina** (`us-nc-guilford`). You can pull it down without an account and run all four pipeline notebooks against it.

### A.1 Pre-flight checklist — install and activate

Before anything else, make sure your install is complete and your virtual environment is **active**. The pipeline notebooks won't run otherwise. If you're new to OpenAVMKit, work through these sections of [Getting Started](getting_started.md) in order:

1. **[Install Python 3.11](getting_started.md#2-install-python)** — OpenAVMKit is tested on 3.11 specifically. Older or newer versions may produce import errors or subtle bugs.
2. **[Clone the repo](getting_started.md#1-clone-the-repository)** (if installing from Git) or `pip install openavmkit` (if installing from [PyPI](getting_started.md#option-1---install-from-pypi)).
3. **[Set up a virtual environment](getting_started.md#3-set-up-a-virtual-environment)** with `python -m venv venv` and activate it (`source venv/bin/activate` on macOS/Linux, `venv\Scripts\activate` on Windows). **You must activate the venv every time you open a new terminal** — your prompt should show `(venv)` when it's active.
4. **[Install dependencies](getting_started.md#4-install-dependencies)** with `pip install -r requirements.txt` and **[install openavmkit itself](getting_started.md#5-install-openavmkit)** with `pip install -e .`.
5. **[Install Jupyter](getting_started.md#running-jupyter-notebooks)** with `pip install jupyter` if you haven't already — the pipeline runs as Jupyter notebooks.
6. **[Run the test suite](getting_started.md#running-tests)** with `pytest` to confirm everything imports and the install is healthy.

Confirm before continuing: your terminal prompt shows `(venv)`, `pytest` passes, and `jupyter notebook` opens a browser tab listing your repo files.

### A.2 Set up the sample locality

1. Inside your OpenAVMKit checkout, navigate to `notebooks/pipeline/data/`.
2. Create a folder named `us-nc-guilford`.
3. Inside it, create a file named `cloud.json` with this content:

    ```json
    {
        "type": "azure",
        "azure_storage_container_url": "https://landeconomics.blob.core.windows.net/localities-public"
    }
    ```

That's it. No credentials needed — it's a public read-only container.

### A.3 Run

**Make sure your venv is active** (your prompt should show `(venv)`). If you opened a fresh terminal since A.1, re-run the activate command.

Launch Jupyter (`jupyter notebook` from the `notebooks/` directory) and open [`pipeline/01-assemble.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/01-assemble.ipynb).

In the second cell, set:

```python
locality = "us-nc-guilford"
```

Run all cells from the top. The `cloud_sync` cell will download the input data to `notebooks/pipeline/data/us-nc-guilford/in/`. Subsequent cells load, enrich, and tag model groups. Watch for the `examine_sup` output near the end — every numeric and categorical field should have sensible non-null counts.

Now run, in order:

- [`02-clean.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/02-clean.ipynb) — fills missing values, runs sales scrutiny, computes time-adjusted prices.
- [`03-model.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/03-model.ipynb) — trains models per model group and writes predictions and reports under `data/us-nc-guilford/out/`.
- [`assessment_quality.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/assessment_quality.ipynb) — generates ratio and equity reports.

### A.4 What success looks like

After all four notebooks run cleanly:

- `data/us-nc-guilford/out/` has parquet files for the universe, sales, and predictions
- `data/us-nc-guilford/out/models/<model_group>/` has per-model output: predictions, `params_<subset>.csv`, `contributions_<subset>.csv` — including an `ensemble/` folder with its own reassembled params/contributions
- `data/us-nc-guilford/out/reports/` has ratio study and equity reports
- The `examine_sup` output shows non-null fields for every parcel, sales correctly partitioned into model groups

If any stage errors out: check the install (re-run `pytest` from the repo root), confirm Python is 3.11, and confirm `cloud.json` is in the right place.

**For full reference**, see [notebooks/README.md](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/README.md) for the run-order map and [recipe.md](recipe.md) for what each public function does.

---

## Part B — Onboard your own jurisdiction

Now the real work. You have an assessor extract for a jurisdiction OpenAVMKit doesn't already support, and you want to get to a working AVM.

### B.1 Before you start: what you need

**Required:**

- **Parcel records** with a stable parcel ID (key) and geometry. Geometry can be a separate shapefile / GeoPackage joined on the key, or a single GeoDataFrame containing both.
- **Sales records** with a parcel ID (foreign key to parcels), sale date, sale price, and ideally a validity / arms-length flag.
- **Property characteristics** — building square footage/meters, year built, quality, condition, room counts, land area, zoning, etc. These can live on the parcels file, the sales file, or separate tables joined by ID.

**Optional but recommended:**

- **Building permits** — for capturing post-sale renovations that explain price differences.
- **Reference shapefiles** for distance/proximity features — CBD polygon, school district boundaries, airport, employment centers, etc. (See [advanced_settings.md § 4.5](advanced_settings.md#45-distance-proximity-enrichment-dataprocessenrichdistances) for what these unlock.)
- **Time adjustment factors** if your jurisdiction publishes its own — see [advanced_settings.md § 3](advanced_settings.md#3-time-adjustment).

**File formats** — CSV, parquet, shapefile, geopackage. OpenAVMKit reads them all. Parquet is fastest; shapefiles work fine for geometry.

**Units — imperial or metric, your choice.** OpenAVMKit fully supports both unit systems. You declare which one applies via `locality.units` in `settings.json` (covered in § B.4 below), and the rest of the pipeline follows. Your raw data can be in either system as long as it's consistent within itself — you just map the appropriate columns to the matching canonical field names (`land_area_sqft` vs. `land_area_sqm`, `bldg_area_finished_sqft` vs. `bldg_area_finished_sqm`, `frontage_ft` vs. `frontage_m`, etc.). If your data is in one system but you want to work in the other, use a `calc` to convert at load time (e.g. `["*", "AREA_ACRES", 43560]` for acres → sqft, or `["*", "AREA_HECTARES", 10000]` for ha → sqm).

### B.2 Set up the locality folder

Pick a slug following the format documented in [the_basics.md → Creating a new locality](the_basics.md#creating-a-new-locality) — typically `<country>-<state>-<locality>`, all lowercase with underscores instead of hyphens inside the locality name. Then:

```text
notebooks/pipeline/data/<your-slug>/
├── in/
│   ├── settings.json
│   └── (your raw data files)
└── out/   (created automatically on first run)
```

Place your raw data files (CSV, parquet, shapefile) directly in `in/`. They don't have to be cleaned or pre-processed — that's what OpenAVMKit is for.

### B.3 Profile your raw data

Before writing a settings file, know what you have. Open a blank Jupyter notebook in your locality folder and look:

```python
import pandas as pd
import geopandas as gpd

parcels = gpd.read_file("in/parcels.shp")  # or pd.read_csv, etc.
sales = pd.read_csv("in/sales.csv")

# What columns are there, and what dtypes?
print(parcels.dtypes)
print(sales.dtypes)

# How nullable is each column?
print(parcels.isna().sum())
print(sales.isna().sum())

# What are the cardinalities of likely categorical fields?
for col in ["LAND_USE", "ZONING", "GRADE", "SALE_STATUS"]: # use your actual fields, these are examples
    if col in parcels.columns:
        print(col, parcels[col].value_counts().head(10))
```

You're trying to answer:

- Which column is the parcel key? Is it stable across parcels and sales?
- What date format is the sale date in?
- What categorical values mean "valid sale"?
- Which fields are mostly null and shouldn't be modeled on?
- Does the geometry have a CRS, and is it reasonable?

Write the answers down. They're the inputs to your settings file.

### B.4 Author a minimum viable `settings.json`

This is the heart of Part B. We'll build up a settings file just complete enough to run notebook 1 successfully. Add advanced features later.

A minimum viable settings file has four sections: `locality`, `data.load`, `modeling.metadata`, and `modeling.model_groups`.

#### locality

```json
{
    "locality": {
        "name": "Imaginary County",
        "county": "Imaginary",
        "state": "TX",
        "slug": "us-tx-imaginarycounty",
        "units": "imperial",
        "center": {
            "latitude": 29.7604,
            "longitude": -95.3698
        }
    }
}
```

`center` is used for polar-coordinate enrichment; an approximate jurisdiction centroid is fine.

#### About `locality.units`

This single setting determines the unit system used everywhere downstream. Choose `"imperial"` or `"metric"` (default if omitted: `"imperial"`).

| Quantity | Imperial | Metric |
| --- | --- | --- |
| Small length | `ft` | `m` |
| Big length | `mi` | `km` |
| Small area | `sqft` | `sqm` |
| Big area | `acre` | `ha` |

The setting affects:

- **Which canonical field names you map your raw columns to.** When `units = "imperial"`, the modeling code looks for `land_area_sqft`, `bldg_area_finished_sqft`, `frontage_ft`, `depth_ft`, etc. When `units = "metric"`, it looks for `land_area_sqm`, `bldg_area_finished_sqm`, `frontage_m`, `depth_m`, etc. **Map your raw columns to the names that match your configured unit system** — if your assessor data reports building area in square feet and you've set `units = "imperial"`, map it to `bldg_area_finished_sqft`; if you've set `units = "metric"`, convert at load time and map to `bldg_area_finished_sqm`.
- **Enrichment outputs.** GIS-derived land area is written to `land_area_gis_sqft` or `land_area_gis_sqm` depending on the setting. Distance enrichment defaults to `km` regardless, but you can override `unit` per feature (see [advanced_settings.md § 4.5](advanced_settings.md#45-distance-proximity-enrichment-dataprocessenrichdistances)).
- **Modeling features.** Spatial-lag, density, and Somers-units calculations all use the configured small-area unit.
- **Reports.** Ratio studies and equity studies display areas and distances in the configured units.

**The conversion principle**: pick the unit system that matches your data (or that you prefer to think in), then map your raw columns into the matching canonical field names. If your raw data is in the *other* system, use a `calc` block at load time to convert: `["*", "TOTAL_ACRES", 43560]` produces `land_area_sqft` from acres, `["*", "AREA_HECTARES", 10000]` produces `land_area_sqm` from hectares, `["/", "AREA_SQM", 0.092903]` would produce sqft from sqm if you really need to swap, and so on. See [The `calc` expression language](calc_reference.md) for the full operator reference.

#### data.load

This is where most onboarding work happens. Each subkey under `data.load` declares a file, its column-to-canonical-field mapping, and any computed columns. A real example from [us-nc-guilford](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-nc-guilford/in/settings.json):

```json
{
    "data": {
        "load": {
            "parcels": {
                "filename": "parcels.csv",
                "load": {
                    "key": ["GDV_REID", "string"],
                    "address": "LOCATION_ADDR",
                    "land_class": "LAND_CLASS",
                    "zoning": "ZONING",
                    "neighborhood": "NEIGHBORHOOD",
                    "assr_land_value": "TOTAL_LAND_VALUE_ASSESSED",
                    "assr_impr_value": "TOTAL_BLDG_VALUE_ASSESSED"
                },
                "calc": {
                    "land_area_sqft": ["*", "TOTAL_ACRES", 43560]
                }
            },
            "sales": {
                "filename": "sale.csv",
                "load": {
                    "key": ["REID", "string"],
                    "sale_date": ["SALE_DATE", "datetime", "%m/%d/%Y %H:%M"],
                    "sale_price": ["SALE_PRICE", "float"],
                    "bldg_area_finished_sqft": "HEATED_AREA",
                    "bldg_year_built": "YEAR_BUILT"
                }
            }
        }
    }
}
```

Three patterns to notice:

- **Plain string** — `"address": "LOCATION_ADDR"` — rename the source column to the canonical name. No type coercion.
- **Two-element list** — `"key": ["GDV_REID", "string"]` — rename **and** coerce dtype. Useful when the source is e.g. an integer that should be treated as a string (parcel keys are notoriously prone to this).
- **Three-element list** — `"sale_date": ["SALE_DATE", "datetime", "%m/%d/%Y %H:%M"]` — for dates, the third element is a `strftime`-style format string.

The `calc` block lets you derive new columns at load time. `["*", "TOTAL_ACRES", 43560]` produces `land_area_sqft = TOTAL_ACRES * 43560`. The expression language is small but expressive — every entry is a list whose first element is an operator and whose remaining elements are operands. Common operators: arithmetic (`+`, `-`, `*`, `/`, `/0` for z-safe), comparison (`==`, `!=`), filters (`?`), string operations (`split_before`, `split_after`, `replace`, `join`, `substr`), type coercion (`asint`, `asfloat`, `asstr`), conditionals (`where`), dictionary lookup (`map`), date parsing (`datetime`), and area-from-geometry (`geo_area`). **For the full operator reference with worked examples, see [The `calc` expression language](calc_reference.md).**

**The canonical field names** (`key`, `sale_date`, `sale_price`, `bldg_area_finished_sqft`, `land_area_sqft`, `bldg_year_built`, `bldg_quality_num`, `bldg_condition_num`, `neighborhood`, etc.) are what OpenAVMKit's modeling and analysis code looks for. The renaming step in `data.load` is how you bridge from your source schema to OpenAVMKit's. Read [the_basics.md → Terminology](the_basics.md#terminology) for the conceptual model.

**Year built vs. age — load year, model on age.** Map your raw year-built columns to `bldg_year_built` (and `bldg_effective_year_built` if you have it) here in `data.load`. OpenAVMKit's cleaning step automatically derives `bldg_age_years` and `bldg_effective_age_years` by subtracting `bldg_year_built` from the *year* of your `valuation_date` (year-precision, not date-precision). **Model on the `_age_years` fields, never on the `_year_built` fields.** See [§ B.7 → Age variables](#age-variables-use-age-not-year-built) for the full rationale.

#### modeling.metadata

```json
{
    "modeling": {
        "metadata": {
            "modeler": "Your Name",
            "valuation_date": "2026-01-01"
        }
    }
}
```

`valuation_date` is the date predictions are anchored to — typically January 1 of the assessment year. Used by time adjustment.

#### modeling.model_groups

**Model group**: a named partition of the parcels in your jurisdiction that share similar characteristics, similar buyers and sellers, and should therefore be modeled together. Single-family residential, commercial, agricultural, and townhomes/condos are typical model groups. The choice of how to split is a real decision — usage and buyer pool matter more than zoning code.

Each model group has a name and a filter expression. Filters are nested-list expressions evaluated against universe rows; here's a simple one:

```json
{
    "modeling": {
        "model_groups": {
            "single_family": {
                "name": "Residential single-family",
                "filter": ["==", "land_class", "str:RES1"]
            },
            "commercial": {
                "name": "Commercial",
                "filter": ["in", "land_class", ["COMM", "INDUSTRIAL"]]
            }
        }
    }
}
```

For more sophisticated splits — handling vacant vs. improved sub-types, common-area exclusions, or filter reuse via `$$ref` — see the [us-nc-guilford settings](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-nc-guilford/in/settings.json) `model_groups` block.

**For everything else** — preprocessor syntax (`__` comments, `$$` variable refs, `!` and `+` flags), the full enrichment menu, modeling overrides, ratio study tuning — see [advanced_settings.md](advanced_settings.md). The minimum viable settings above is enough to run notebook 1; layer on advanced features once that's working.

### B.5 Run notebook 1 (Assemble)

Open [`01-assemble.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/01-assemble.ipynb), set `locality = "<your-slug>"`, and run cells from the top.

**What to watch for:**

- **`load_dataframes` row counts.** If you expected 50,000 parcels and you're seeing 12, your `filename` is wrong or the file is malformed.
- **Date parsing.** If `sale_date` shows `NaT` for many rows, your format string in `data.load.sales.load.sale_date` is wrong. Open the source file and check the actual format.
- **`process_data` enrichment.** Look at the output: did Census, distances, OSM steps run as expected? (Each is an opt-in. If you didn't configure `data.process.enrich.distances`, distances won't run — that's fine.)
- **`tag_model_groups_sup` output.** It prints how many parcels landed in each model group. If you see "0 parcels" for a group you expected, your filter doesn't match. If parcels are landing in `Exempt` you didn't intend, your filters aren't catching them.
- **`examine_sup`**. Spot-check non-null counts for every field you mapped. Fields that are mostly null might be wrongly mapped or might genuinely be sparse — investigate before relying on them in a model.
- **The `out/look/*.parquet` files**. Drop them into QGIS, ArcGIS, or [Felt](https://felt.com/) and confirm parcels render in the right place with sensible attributes.
- **Don't trust class/use-code labels at face value — verify them empirically.** Assessor classification codes routinely mislead: a code named `…-LAND` may actually sit on an *improved* parcel; a "qualified" sale may be a bulk transfer; a per-unit field may be a share or the whole-parcel figure repeated. Before you key model groups, vacancy, or land size off a coded field, **sample it**: check its distinct values, and within natural groups (building, subdivision, deed) check sums and correlations against physical fields. *Example:* to confirm a per-unit land figure is a real pro-rated share, verify it sums to the parent parcel and correlates with floor area — rather than assuming the column name is honest. Trust the data's behavior, not its label.

**Common failure modes:**

| Symptom | Likely cause |
| --- | --- |
| `KeyError` on a column name | The source column isn't where you said it was; check spelling and case |
| All `sale_date` values are `NaT` | Wrong format string |
| `0 parcels` in a model group | Filter doesn't match anything; check categorical values |
| All parcels render at lat/lon (0, 0) | Geometry has no CRS or wrong CRS — set the CRS on import |
| `valid_sale` is empty | You forgot to define a `valid_sale` calc on the sales load |

When something is wrong, the fix is **almost always in `settings.json`**. Edit the settings, then re-run — but see § B.10 below before re-running.

### B.6 Run notebook 2 (Clean)

Open [`02-clean.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/02-clean.ipynb).

**What this notebook is doing:**

- **Filling missing values.** OpenAVMKit cannot model on null data, so `data.process.fill.<method>` rules in your settings get applied. Choose methods deliberately: `zero` for "missing means absent" (e.g., building area on vacant parcels — `zero_vacant`), `mode` for categorical defaults, `median` for skewed numerics, `mean` for symmetric numerics. The `_impr` and `_vacant` suffixes scope a fill to improved-only or vacant-only parcels. **Full reference**: [advanced_settings.md § 5.1](advanced_settings.md#51-filling-missing-values-dataprocessfillmethod).
- **Equity clusters.** Parcels are grouped by similar characteristics + similar location for later horizontal-equity analysis.
- **Sales scrutiny.** A clustering heuristic flags suspect sales — outlier prices within a peer group, suspicious clusters of identical-looking sales. Your `analysis.sales_scrutiny` settings drive the cluster definitions.
- **Time adjustment.** Computes a per-day price-adjustment multiplier so historical sales can be compared at the valuation date. The default engine fits a rolling median; if your jurisdiction publishes its own time factors, use `data.process.time_adjustment.from_file.<model_group>` to load them — see [advanced_settings.md § 3](advanced_settings.md#3-time-adjustment).

**What to watch for:** the `examine_sup` output near the end should show no remaining nulls in fields you intend to use. Time-adjusted prices should be sensible — if every adjustment factor is 1.0, time adjustment didn't run.

### B.7 Run notebook 3 (Model)

Open [`03-model.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/03-model.ipynb).

**What this notebook is doing:**

- **Train/test split.** `write_canonical_splits` writes a deterministic 80/20 split so all models in this run train and evaluate on the same data.
- **Variable selection.** `try_variables` runs a more thorough variable-importance experiment than the inline auto-reduction. Worth using when you have hypotheses about which characteristics matter; skippable when you trust the defaults.
- **Model training.** Per model group, the configured models train and predict. Default lineup: MRA (linear regression), GWR (geographic weighted regression), XGBoost, LightGBM, CatBoost, plus several baselines. **For the full catalog of available models, how to invoke them, how to run multiple variants of the same engine (e.g. two XGBoost configurations side-by-side), and what settings each accepts, see [Models reference](models_reference.md).**
- **Per-model output.** Each model produces three artifacts per subset (test/sales/universe):
    - **Predictions** — the central output.
    - **`params_<subset>.csv`** — per-feature parameters (regression coefficients for linear; SHAP-normalized for tree-based). "What is each feature's per-unit effect?" For MRA the file carries two columns — `coefficient` and `error` (the regression standard error) — so you can read both the effect and its uncertainty.
    - **`contributions_<subset>.csv`** — per-feature contributions (coef × value for linear; raw SHAP for tree-based). "How much did each feature contribute to this row's prediction?"
- **Ensemble output.** The `ensemble/` folder reassembles its own `params_<subset>.csv` / `contributions_<subset>.csv` from the member models (mean and median ensembles average the members' contributions per row; local passes through the selected model's), so the combined prediction stays interpretable. It also writes `ensemble_meta.json` recording which type (`mean`/`median`/`local`) and members produced it. Alongside the on-disk artifacts, each run prints a per-model summary table with `count`, `MAPE`, `MSE`, `RMSE`, `m.ratio`, `avg.ratio`, `VEI`, `Slope`. `VEI` (Vertical Equity Index) flags regressive vs. progressive valuation; see the glossary in [§ B.8](#b8-run-notebook-4-assessment-quality).

#### `try_models` vs `finalize_models` — iterate fast, then commit

The notebook offers two functions for running models, and the choice matters a lot when you're iterating:

- **`try_models(sup, settings, ...)`** — trains the configured models and computes test-set metrics, but **does not write predictions, reports, or artifacts to disk**. Optimized for speed and iteration.
- **`finalize_models(sup, settings, ...)`** — same training, but **also writes everything to disk**: per-model predictions, `params_*.csv`, `contributions_*.csv`, plots, ratio-study breakdowns, ensemble outputs, etc.

The reason these are separate functions is purely **performance**: writing all the modeling artifacts to disk turned out to be a substantial portion of total runtime, and most of the time when you're iterating — adjusting variables, tweaking model groups, testing fill rules — you don't need the on-disk results yet. You're just looking at the in-memory metrics to decide what to change next. Calling `finalize_models` on every iteration would burn a lot of time on disk I/O you're going to throw away.

**Practical workflow:**

1. Use `try_models` repeatedly while you're tuning. Read the test-set metrics, decide what to change, edit settings, re-run.
2. Once you're happy with the results, call `finalize_models` once to commit everything to disk for downstream analysis (notebook 4, reports, hand-off).

If your goal is reproducibility (the same input produces the same output and you want it persisted), call `finalize_models`. If you're still in the explore-and-tweak phase, use `try_models` to keep the cycle fast.

For configuring which models run for which model groups, see [advanced_settings.md § 6](advanced_settings.md#6-modeling-control). For wiring up a new model class, see [AGENTS.md § 7](https://github.com/larsiusprime/openavmkit/blob/master/AGENTS.md).

#### Modeling best practices

The single biggest determinant of model quality is **what variables you feed it.** Picking the right ones — and *not* picking the wrong ones — usually matters more than which model algorithm you use.

##### Variable selection: linear models hate stuffing — tree models tolerate it

There is no single right answer here; the right number of features depends on the model family. Linear/parametric models (MRA, multi-MRA, kernel, GWR) and tree-based models (XGBoost, LightGBM, CatBoost) react to extra features in opposite directions.

**For linear/parametric models**, more variables hurts you:

1. **Multicollinearity destabilizes coefficients.** When two predictors are highly correlated, the linear system has no unique solution and the per-feature coefficients become noisy and uninterpretable. The prediction may still be okay but you lose any ability to read off effects, and small data changes can flip signs.
2. **Weakly predictive variables add noise.** Linear regression will spend coefficient on anything you give it; if a variable is mostly random with respect to price, the fitted coefficient absorbs whatever spurious correlation exists in your training sample.
3. **Mostly-empty variables introduce fill-rule bias.** Whatever you put in for missing values becomes a constant disguised as a feature.

The right discipline for linear models is **highly predictive but minimally redundant**: each variable carries information the others don't. Three good variables almost always beat ten mediocre ones. **`try_variables`** is calibrated for this case — it runs a battery of tests then prunes via greedy backward elimination, returning the smallest set that does the job:

- **Predictive power** — correlation with the target, R², t-values, p-values, elastic-net regularization (ENR) coefficients.
- **Cross-correlation** — Variance Inflation Factor (VIF) for multicollinearity. A variable can be highly predictive on its own yet redundant with another you're already using; keeping both costs you with no benefit. **For linear models, follow `try_variables`'s recommendation directly** — its CSV output (`out/try/<model_group>/<vacant_status>.csv`) already represents the minimum-sufficient set.

**For tree-based models, the picture is different — and the same `try_variables` advice would mislead you.** XGBoost, LightGBM, and CatBoost split on one feature at a time per node, so multicollinearity doesn't destabilize anything; the model just picks whichever correlated feature gives a better split at each junction. The implementations have built-in regularization (`lambda_l1`, `lambda_l2`, `feature_fraction`, `min_data_in_leaf`) that handles weakly predictive features automatically — anything that doesn't help training gets near-zero weight at that split or gets pruned. And they can extract value from interactions that no linear model would find on its own (e.g. "homes built before 1950 in neighborhood X" via two splits).

What this means in practice for tree models:

1. **Include all moderately predictive features**, even if they look correlated. Don't let `try_variables`'s post-VIF survivors set your tree-model's ind_vars — that's the linear-model recipe.
2. **Include both raw and derived versions of the same concept** when both have predictive power. For example, `bldg_age_years` (linear scale) AND `bldg_year_built` (categorical-cohort feel) — trees often find non-linear relationships in one that the other can't express.
3. **Include all flavors of `spatial_lag_sale_price_time_adj`** (the absolute lag, the per-land-sqft lag, the per-impr-sqft lag) — they encode different aspects of local market and the tree picks at each split.
4. **Include high-cardinality categoricals natively.** LightGBM and CatBoost handle categoricals without one-hot encoding (just declare them as categorical in your dataset). Neighborhood / VCS / school district can be passed as-is.
5. **Cap is set by training time, not predictive power.** Around 30-100 numeric features is a comfortable working range for typical AVMs. Past that, start by dropping features with near-zero feature-importance after a first fit.
6. **Diagnose with model-derived importance, not pre-modeling correlation.** Train a model with everything plausible, look at SHAP / gain / split-count importance from the fitted model, and only THEN consider pruning. The fitted model's view of "which features matter" is far more reliable than any correlation test.

How to act on this in OpenAVMKit:

- The `default` ind_vars under `modeling.models.<main|vacant>.default` apply to every model that doesn't have its own override. Set this to the lean linear-friendly set returned by `try_variables`.
- Per-model ind_vars overrides go under `modeling.models.<stage>.<model_name>.ind_vars`. Use this to give tree-based models the broader set. Real example from the Wake County smoke test: `mra` and `multi_mra` use the 3-var default; `lightgbm` overrides with ~25 features and consistently outperforms the default-fed version.
- After a first `try_models` pass, look at the per-model `params.csv` (linear) or `contributions.csv` (tree). Variables that contribute nothing to predictions in the tree case are pruning candidates. Variables with unstable signs across folds in the linear case should be removed.

The takeaway: **`try_variables` is for the linear-model defaults; tree-based models want more.** A good production setup uses both — small focused defaults plus per-model overrides for trees.

##### What makes a variable useful

A variable is genuinely worth including only when all three of these are true:

1. **Strong association with price** — positive *or* negative is fine, but it has to move.
2. **Well recorded in the data** — what's in the column accurately reflects reality on the ground. A variable that's wrong half the time is worse than no variable at all.
3. **Well formatted** — consistent dtypes, sensible categorical values, no leading/trailing whitespace, no mixed encodings of the same concept (e.g. `"YES"`/`"yes"`/`"Y"`/`1` for the same boolean).

If a candidate variable fails any of these, fix the underlying issue (with `calc` operators or a better data source) or leave it out.

##### The "big three" — location, location, location

Real estate is famously about location — but how you encode location depends on which model you're using:

- **GWR (geographic weighted regression)** is *natively* spatially aware in the strictest sense. It uses lat/lon internally as part of its weighting kernel; the spatial structure is built into the algorithm itself, not just supplied as variables.
- **Kernel regression** is also natively spatially aware as OpenAVMKit invokes it. The runner automatically prepends `longitude` and `latitude` to the variable matrix before fitting, so the kernel always weights by geographic proximity in addition to your other features. You don't need to add them as model variables — they're injected for you.
- **`LocalAreaModel`** is a special case: it isn't spatial via coordinates, but it's "natively aware" in the sense that you literally cannot invoke it without giving it `location_fields` at construction (e.g. `neighborhood`, `market_region`, `census_tract`). It computes per-area value averages keyed by those region fields and applies them at predict time. Spatial through user-supplied categorical regions, not lat/lon.
- **Everything else** — MRA (linear regression), XGBoost, LightGBM, CatBoost, and the various baselines — needs location given to them as feature columns. Several options, often used in combination:
    - **`latitude_norm` / `longitude_norm`** — automatically created by basic-geo enrichment (see [advanced_settings.md § 4.1](advanced_settings.md#41-basic-geometric-enrichment-dataprocessenrichbasic)). These are min-max-normalized to `[0, 1]` over your jurisdiction's bounding box and often perform better in non-spatial models than raw lat/lon (which is on a much larger numeric scale and tends to be hard for tree splits to use efficiently).
    - **`polar_radius` / `polar_angle`** — also auto-created. Polar coordinates relative to your `locality.center`. Useful when value gradients are roughly radial (distance-from-CBD effects, ring suburbs).
    - **Categorical region fields** — neighborhoods, market areas, school districts, ZIP codes (see "Categorical variables" below).

Test all of these with `try_variables` before committing — different jurisdictions favor different encodings, and using all four spatial representations at once is usually overkill.

##### The "big five"

Five variables drive most of the predictive power for residential parcels. Get these right before worrying about anything else:

| Variable | What it captures | Notes |
| --- | --- | --- |
| `bldg_area_finished_sqft` / `bldg_area_finished_sqm` | Building size | Almost always the single strongest predictor. |
| `land_area_sqft` / `land_area_sqm` | Land size | Especially important for vacant and large-lot parcels. |
| `bldg_age_years` *or* `bldg_effective_age_years` | Building age | See "Age variables" below — pick exactly one. |
| `bldg_quality_num` | Construction quality (materials, workmanship) | Encode as ordinal numeric. |
| `bldg_condition_num` | Physical condition / depreciation | Encode as ordinal numeric. |

Quality and condition are different things: a brand-new poorly-built house is high-condition / low-quality; a well-built but dilapidated house is high-quality / low-condition. Both matter independently.

**Encoding quality and condition: accuracy beats precision.** A 4-tier scale that's reliably coded ("poor / fair / good / excellent") beats a 16-tier scale where assessors disagree on the boundaries. Aim for ordinal numeric (e.g. 1–4 or a 0–100 scale) so models can interpolate; don't proliferate tiers just to look granular.

##### Categorical variables — use them, but judiciously

Categorical variables (zoning, building style, neighborhood, exterior material, heating type, etc.) need careful handling.

**How they're treated under the hood:**

- **Tree-based models** (XGBoost, LightGBM, CatBoost) handle categoricals natively in OpenAVMKit's wrappers — they understand a category as a category and split on it directly.
- **Linear and kernel models** (MRA, GWR, kernel regression) require **one-hot encoding**: a categorical with N unique values becomes N (or N−1) boolean columns under the hood. This means a categorical with 50 categories adds 50 columns to your training matrix.

**The cost of high-cardinality categoricals:**

A categorical with N unique values is roughly equivalent to adding N variable columns. This:

- Slows training significantly for non-tree-based models
- Can dilute predictive power if many categories are sparsely populated
- Often doesn't add proportional accuracy — five well-distributed categories often beat fifty long-tail ones

So use categoricals **deliberately**, not reflexively.

**Where categoricals shine — model-group segmentation.** A categorical's most powerful use is often *not* as a model variable but as a **filter for partitioning into model groups**. Splitting your jurisdiction into "single-family residential," "townhomes/condos," "commercial," and "agricultural" — each modeled separately — captures the categorical's effect more cleanly than feeding it to a single mega-model. Trying to model agricultural land and high-rise condos with the same MRA is a losing battle no matter how good your variables are.

**Use categoricals as model variables when** they're well-defined, modest in cardinality, and capture meaningful within-group variation that the model groups don't already isolate.

**Best categoricals to consider:** *well-drawn* assessor neighborhoods, market areas, or land economic areas (LEAs). When these reflect real submarket boundaries — areas where buyers and sellers genuinely treat properties as substitutes — they're some of the most powerful categoricals you can use. **But:** if your jurisdiction has thousands of micro-neighborhoods, or the boundaries are arbitrary administrative artifacts, the same field becomes a liability. Cardinality and quality both matter.

##### Age variables — use age, not year built

This one's important enough to call out in bold:

**Model on `bldg_age_years` or `bldg_effective_age_years`. Never on `bldg_year_built` or `bldg_effective_year_built`.**

The flow is: **load** the year-built field from your raw data (in `data.load.<id>.load`) → OpenAVMKit's cleaning step automatically derives `bldg_age_years` from it by subtracting `bldg_year_built` from the year of your `valuation_date` (see `_fill_unknown_values` in [openavmkit/cleaning.py](https://github.com/larsiusprime/openavmkit/blob/master/openavmkit/cleaning.py)) → **model** on the derived `bldg_age_years`.

Why year-built is wrong as a model variable:

- Year-built is on an arbitrary numeric scale (1923, 1987, 2024) that the model has to learn the meaning of. Age is on a directly meaningful scale (0 = brand new, larger = older).
- Year-built ties your model to a specific calendar moment. If you re-run next year with a new valuation date, year-built doesn't change but age does — so the model trained on year-built learns relationships that drift over time.
- Both `bldg_year_built` and the derived `bldg_age_years` carry the same information; using both wastes a slot and creates collinearity.

**Effective vs. regular age — pick one, not both.** `bldg_age_years` is calendar age (valuation year minus year built). `bldg_effective_age_years` is the appraiser's judgment of how old the building "feels" given recent renovations, condition, and modernization. Both can be useful, but they measure overlapping concepts and using both as model variables makes them compete — often degrading both their coefficients. Pick whichever is better-recorded in your data: effective age if the assessor actively maintains it, calendar age otherwise.

**Summary of the age rule:** `_year_built` fields belong in `data.load`. `_age_years` fields belong in your modeling variables. Never reverse this.

**What to watch for:** large prediction errors on a particular model group usually mean either (a) the model group is too heterogeneous and should be split, or (b) the variables you're feeding the model don't capture what's driving prices in that group.

### B.8 Run notebook 4 (Assessment quality)

Open [`assessment_quality.ipynb`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/assessment_quality.ipynb).

**What this notebook is doing:**

- **Ratio study.** Computes IAAO statistics on predicted-to-sale-price ratios.
    > **Ratio study glossary** (for generalists):
    > - **COD** (Coefficient of Dispersion) — overall variability of ratios. Lower is better; IAAO standard is < 15.0 for single-family residential.
    > - **PRD** (Price-Related Differential) — ratio of mean to weighted-mean ratio. Should be ~1.0; > 1.03 suggests regressive valuation (low-priced properties over-assessed).
    > - **PRB** (Price-Related Bias) — alternative vertical-equity measure. Should be near zero; outside ±0.05 is concerning.
    > - **VEI** (Vertical Equity Index) — `100 × (top-percentile-group median ratio − bottom-percentile-group median ratio) / overall median ratio`. Number of percentile groups scales with sample size (2 / 4 / 10 for 20–50 / 51–500 / >500 sales; `NaN` below 20). Zero means high- and low-value parcels are valued with the same accuracy; positive means regressive (low-priced over-assessed), negative means progressive. Reported alongside `VEI_sig`, a 90%-CI version: if `VEI_sig` and `VEI` share a sign, the gap is statistically significant.
- **Horizontal equity study.** Within clusters of similar properties, do similar parcels get similar predictions? Reports CHD (Coefficient of Horizontal Dispersion) per cluster.
- **Vertical equity study.** Across price quantiles, does the model treat high-value and low-value parcels with the same accuracy? Reports PRD/PRB and per-quantile median ratios.

**What to watch for:** if the assessor's existing values look better than yours by these measures, your model has work to do. Common causes: bad variable selection, model groups that are too coarse, or untreated outliers in sales scrutiny. Iterate.

#### When untrimmed COD is much worse than trimmed COD

If the untrimmed COD is several multiples (5×–20×) of the trimmed COD, you have a small number of extreme sale-vs-prediction mismatches dominating the means. Trimmed metrics tell you the model is mostly fine; untrimmed metrics tell you *something* is rotting the tails. **Investigate before you tune.**

Strong signal: if **all** your models — including the assessor baseline — show the same untrimmed/trimmed gap, the issue is almost certainly in the *data*, not in any one model. The outliers are real records that no honest model can fit, so investigate them as data first.

**Different signal: only your models show the gap; the assessor's untrimmed COD looks fine.** Two readings of this, and they have opposite implications:

1. **Your model genuinely has a problem.** Your features, fill rules, or target definition are letting some sales blow up your tails while the assessor (with more domain knowledge) handles them. Action: look at the same outliers, but in the modeling lens — what does your model see vs. what the assessor sees that you don't?
2. **The assessor is sales-chasing.** A jurisdiction that revalues parcels to match observed sale prices will look great on a ratio study run against those sales — *because they fit the sales by construction* — even when the underlying mass-appraisal model is no better than yours. You'd be measuring honest predictions against a baseline that has the answer key. Run the sales-chasing diagnostic below before concluding your model is the problem.

##### Sub-diagnostic: is the assessor sales-chasing?

"Sales chasing" means the assessor saw the sale price and revalued the parcel to match. The assessor's ratio statistics will look great on the sold parcels — because they're the parcels whose values got tweaked — but the unsold parcels in the same neighborhood drift away. Signs to look for:

- **COD suspiciously low (< ~4–5).** IAAO standards target COD < 15 for SFR; healthy real-world models land in 5–12. A sustained sub-5 COD on real sales is rare without sales chasing or genuinely homogeneous housing stock.
- **Median ratio extremely close to 1.00 with a tight spread** (small inter-quartile range on the ratio distribution). Honest predictions have unbiased noise around 1.0; sales-chased predictions look almost too good.
- **CHD (Coefficient of Horizontal Dispersion) high while COD is low.** This is the smoking gun: similar properties get similar values *if they all sold*, but among the horizontal-equity peer group, the sold parcels match their sales and the unsold ones don't, so within-cluster dispersion balloons. Honest mass appraisal keeps both COD and CHD in range; sales chasing pulls them apart.
- **Year-over-year value changes concentrated on recently-sold parcels.** If sold parcels jumped 15% YoY while their never-sold neighbors moved 1%, the assessor is rewriting around sales rather than running a uniform model.
- **`(assessor_value - sale_price)` distribution suspiciously spiked at zero.** A genuine valuation is a noisy estimate; sales chasing produces unnaturally many exact matches.
- **Tight slope ≈ 1.0 paired with a high `prb` (PRB).** Slope close to 1 looks great, but if PRB is also far from zero, the assessor is hitting the sale on the dollar without actually being uniformly accurate across the price spectrum.

If your assessor baseline is sales-chasing, the right comparison is the assessor's metrics on **prior-year sales the assessor hadn't seen yet at the time of valuation**, not the current cycle. Several jurisdictions also publish the prior-cycle assessed value separately — comparing your model to *that* avoids the chase confound.

**Automated check.** The ratio study report runs several of these signals for you in its **"Sales-chasing check"** section (ratio spike at 1.0, the COD-vs-CHD divergence above, and a pre- vs. post-valuation COD gap), comparing the assessor baseline against your own model. Thresholds are configurable under `analysis.ratio_study.sales_chasing` (see [advanced settings](advanced_settings.md)). It reports *likely*/*possible* as a context cue, not a verdict, and is not a substitute for the manual investigation above. Relatedly, openavmkit by default **does not show the assessor on the random pre-valuation holdout** — not because the assessor did anything wrong, but because we can't know the holdout status of values we didn't generate, so it wouldn't be a like-for-like comparison. The assessor is shown on the post-valuation holdout and the full study set; the post-valuation comparison assumes your `valuation_date` is aligned with the roll-close date of the values being compared. If you *are* the assessor and know the holdout status, you can opt back into the holdout comparison via `analysis.ratio_study.assessor_holdout` (see [the basics](the_basics.md#when-you-are-the-assessor)).

##### Diagnostic flow for outlier investigation

Regardless of whether sales chasing is in play, the actual outliers driving the tails need to be looked at:

1. Pull `out/models/<mg>/main/ensemble/outliers.csv` (auto-written by `identify_outliers`).
2. Sort by `prediction_ratio`. The top tail (`> 2.0`) is over-prediction (sale was suspiciously low); the bottom tail (`< 0.5`) is under-prediction (sale was suspiciously high).
3. For each tail, join back to your raw `parcels.csv` and `sales.csv` to retrieve fields that didn't survive the canonical-name renaming (e.g. raw deed flags, sale-type indicators, card numbers).
4. Look for these patterns:
    - **Over-prediction tail (ratio > 2)** — sale was lower than the parcel "should be" worth. Common causes:
        - **Mislabeled vacant**: a vacant sale tagged as improved. Check your jurisdiction's sale-type field.
        - **Invalid sale that scrutiny missed**: family transfer, forced sale, quitclaim deed. Check the raw deed/qualification flag in the assessor file — many jurisdictions publish flags ("disqualified — life estate reservation", "non-warranty deed", etc.) that a separately-published "qualified sales" file may nonetheless leak. **Cross-validate the qualified-sales file against the parcel-level disqualification flag.**
        - **Distressed sale not flagged**: foreclosure, short sale.
        - **Token-consideration transfer**: $1, $10, "ten dollars and other valuable consideration." A simple price floor (e.g. exclude sales below $10K for SFR) catches these without harming legitimate transactions.
        - **Price disagrees with the recorded transfer tax**: where the jurisdiction records a documentary / transfer-tax fee proportional to consideration (e.g. Colorado's $0.01 per $100 → `doc_fee = sale_price × 0.0001`), cross-check `sale_price` against it. A fee of zero or one wildly inconsistent with the stated price flags a nominal/exempt transfer or a data-entry error — a cheap, jurisdiction-agnostic sanity check independent of any qualification flag.
    - **Under-prediction tail (ratio < 0.5)** — sale was higher than the parcel "should be" worth. Common causes:
        - **Multi-parcel sale**: one sale_price covering N parcels, recorded against one. Check whether `deed_book + deed_page` is shared with other sales — and, if the assessor publishes a free-text sale-remark field, scan it: bulk deeds are frequently labeled there outright (e.g. "MULTI-SALE INCLUDES…", "SALE INCLUDES SCHEDULES …"). A remark scan often catches bulk sales that share no obvious deed key.
        - **Misclassification**: parcel actually commercial / multifamily / mixed-use but tagged single_family.
        - **Genuine luxury**: high-end home with features the model can't see (custom finishes, view, prestige). Consider a luxury model_group or an explicit indicator (assessor grade letter, age + lot size, neighborhood premium).
    - **Both tails simultaneously** + no obvious pattern: possibly a bad fill rule converting plausible NaN into a constant that the model anchors on.
5. **Don't silently exclude.** Each outlier exclusion needs a reason you'd defend in a hearing: "this was a $1 family transfer," not "this kills my COD."

What to fix:

- **Bad data with a clear cause** → tighten `data.process.invalid_sales`, add to `in/invalid_sales.csv`, or fix your `valid_sale` / `vacant_sale` calc in `data.load.sales.calc`.
- **Genuine luxury / unique** → either split into a model_group or add a high-end indicator variable to your tree-based model's `ind_vars`. **Do not use the assessor's market value (or anything derived from it) as a luxury indicator** — that invites circularity, since the assessor's value is what your ratio study is measured against. Use raw structural features instead: GRADE letter, year built + recent remodel year, premium neighborhood codes, lot size relative to neighbors.
- **Genuine but unfittable** → document and accept. Keep an eye on whether the trimmed metrics also show drift — that's when it stops being just-tail noise.

### B.9 Iterate

The work doesn't stop after one pass. Typical cycle:

- "Examine" output reveals a bad characteristic → revise `settings.json` (column rename, calc, or fill rule) → re-run from notebook 1 (after caching note below)
- Modeling output is poor for one model group → revisit model group definitions or fill rules → re-run from notebook 3
- Ratio study fails → look at outliers, revisit sales scrutiny config → re-run from notebook 2
- Enrichment looks wrong → nuke `cache/` to force a fresh remote pull → re-run

### B.10 Caching: when to trust it, when to nuke it

OpenAVMKit caches expensive intermediate results in three places:

- **Notebook checkpoints** at `<locality>/out/checkpoints/` — every `from_checkpoint(...)` call in the notebooks writes its result here. On re-run, the cell loads the saved result instead of re-executing.
- **Enrichment cache** at `<locality>/cache/` — used internally by expensive enrichment steps (OpenStreetMap, Census, Overture, distance calculations, street networks).
- **Saved model parameters** at `<locality>/out/models/<model_group>/.../` — tuned hyperparameters and bandwidths from previous model runs (XGBoost / LightGBM / CatBoost Optuna results, GWR bandwidth, kernel regression bandwidth).

The first two layers are *designed* to self-invalidate when the relevant inputs change, so that you can iterate on `settings.json` and see your changes take effect on the next run.

**But edge cases happen.** A signature comparison can miss a subtle change, a partial write can leave a corrupt file, a remote source can drift. **If you're getting weird behavior — your settings change doesn't seem to do anything, output looks suspiciously similar to a previous run, an enrichment is missing fields you know it should have — nuke the cache to be safe.**

The third layer (saved model parameters) is different — see below the table.

| Layer | Path | Cost to rebuild | When to clear |
| --- | --- | --- | --- |
| Notebook checkpoints | `<locality>/out/checkpoints/` | Seconds to minutes per notebook | Whenever changes seem stuck or you want a clean re-run |
| Enrichment cache | `<locality>/cache/` | Minutes to hours (streets / Overture / large OSM bboxes) | When enrichment output looks wrong or stale |
| Saved model parameters | `<locality>/out/models/<model_group>/.../*_params.json`, `*_bw.json`, `kernel_bw.pkl` | Minutes to hours per tuning run (Optuna with many trials, GWR bandwidth search) | When training data has meaningfully changed and previous tuning is no longer appropriate |

**About saved model parameters specifically.** Tunable models (XGBoost, LightGBM, CatBoost, GWR, kernel regression) save their tuned hyperparameters / bandwidths after a successful tuning run, so subsequent runs can skip the search:

- **Delete the file → forces a fresh hyperparameter search.** Slow, but adapts to changes in your training data.
- **Keep the file → skips the parameter search.** Fast, but the model is constrained by the *previous* run's tuning. The model still re-fits on whatever training data it sees — what gets cached is the *tuning step* (which hyperparameters to use), not the predictions themselves.

So if you've changed your training data meaningfully (different sales window, different features, different model group definitions), delete the saved params for the affected models so the next run re-tunes. If you're just iterating on downstream analysis and want fast re-runs, keep them.

**How to nuke** (mostly harmless; you'll just pay re-run cost):

- `delete_checkpoints("<prefix>")` from a notebook clears specific notebook checkpoints (e.g. `delete_checkpoints("1-assemble")` for notebook 1's intermediate state).
- Set `clear_checkpoints = True` at the top of a notebook before running it for a clean re-run.
- Delete the locality's `cache/` folder to wipe the enrichment cache entirely.
- Delete `out/checkpoints/` to wipe all notebook checkpoints for the locality.
- Delete the relevant `*_params.json`, `*_bw.json`, or `kernel_bw.pkl` under `out/models/<model_group>/.../` to force a fresh hyperparameter search.

**Don't nuke prophylactically** — OSM streets and tuning runs are expensive ([advanced_settings.md § 4.7](advanced_settings.md#47-openstreetmap-streets-dataprocessenrichstreetsenabled), § 8.4). Nuke when something feels off.

For the full reference, see [advanced_settings.md § 8](advanced_settings.md#8-caching-checkpoints).

---

## Where to go from here

You now have a working AVM. To go further:

- **[advanced_settings.md](advanced_settings.md)** — full settings reference: preprocessor, enrichment menu, modeling control, ratio study tuning, caching reference.
- **[calc_reference.md](calc_reference.md)** — the full `calc` expression language: every operator with worked examples.
- **[models_reference.md](models_reference.md)** — every model engine: invocation, name-vs-engine dispatch, multiple variants of the same engine, settings, when to use each.
- **[recipe.md](recipe.md)** — every public function organized by pipeline stage.
- **[config.md](config.md)** — environment-level config: cloud storage credentials, Census API key, PDF report generation.
- **[AGENTS.md](https://github.com/larsiusprime/openavmkit/blob/master/AGENTS.md)** — extending OpenAVMKit (new models, new equity studies, new enrichment sources).
- **Canonical examples to learn from** — read these settings files when you want to see how something is done in practice:
    - [`us-nc-guilford`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-nc-guilford/in/settings.json) — Guilford County, NC
    - [`us-va-petersburgcity`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-va-petersburgcity/in/settings.json) — Petersburg City, VA
    - [`us-pa-philadelphia`](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-pa-philadelphia/in/settings.json) — Philadelphia, PA
