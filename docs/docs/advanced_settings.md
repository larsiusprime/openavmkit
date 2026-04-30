# Advanced Settings Reference

`settings.json` is where almost every modeling, cleaning, and analysis decision in OpenAVMKit lives. This page documents features that are real and used but not covered in the introductory docs — the syntax of the settings preprocessor itself, plus a reference to high-impact keys you'd otherwise only discover by reading source.

If you haven't read [The Basics](the_basics.md) yet, do that first. For environment-level configuration (cloud credentials, `.env`, PDF generation), see [Configuration](config.md).

## How to read this page

Each setting entry follows the same shape:

- **Path** — the dotted path inside `settings.json`
- **Default** — what happens if you don't set it
- **Effect** — what changing it does
- **Source** — file (and function or class name) where the code reads it
- **When to use** — practical guidance

---

## 1. The settings.json preprocessor

Before any code reads `settings.json`, [openavmkit/utilities/settings.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/utilities/settings.py) runs your file through a preprocessor. These features are not standard JSON and are not interchangeable with raw JSON tools — but they're widely used in real settings files.

### 1.1 Comments via `__`-prefixed keys

JSON has no comment syntax. OpenAVMKit's preprocessor strips any key that starts with double underscore (`__`) before the settings dict is consumed.

```json
{
  "__comment": "Notes for humans, ignored by the loader",
  "__commented_out_field": { "old": "config" },
  "locality": { "name": "Example" }
}
```

You'll see this all over real settings files — `__comment`, `__comment1`, `__commented_out_characteristics`, `__notes`, etc. Use whatever name makes the surrounding section readable.

> **Single underscore is not the same.** A key like `"_run"` is *not* stripped. It's just a key that doesn't match `"run"`, so the loader won't find it. If you want to comment out a key, use double underscore.

### 1.2 Variable references via `$$path.to.value`

Any string value beginning with `$$` is replaced by looking up the dotted path inside the same settings tree. Resolution is recursive — chains of references work, until the tree stops changing.

```json
{
  "ref": {
    "default_dupes": ["key", "sale_date"],
    "valuation_year": 2026
  },
  "data": {
    "load": {
      "sales": { "dupes": "$$ref.default_dupes" },
      "parcels": { "dupes": "$$ref.default_dupes" }
    }
  }
}
```

Use this to keep one source of truth for repeated values — column lists, thresholds, valuation dates — instead of duplicating them across every section that needs them.

> **Tip.** If you see a `ref` block at the top of a real settings file, that's almost certainly a target for `$$` references elsewhere in the file.

### 1.3 Template merging

Your `settings.json` is *merged* with a built-in template at [openavmkit/resources/settings/settings.template.json](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/resources/settings/settings.template.json). You only need to specify keys you want to override.

The template defines defaults for things like:

- `data.process.fill.zero` — list of fields to fill with zero
- `modeling.instructions.feature_selection.thresholds` — variable selection cutoffs
- `analysis.ratio_study.look_back_years` — how far back to consider sales
- `analysis.horizontal_equity.fields_numeric` — default fields used in equity studies
- `field_classification` — categorical/numeric/boolean field lists

Read the template before assuming a key is missing or unsupported.

### 1.4 `!key` — stomp the template

By default, dicts are merged recursively. If you want to *replace* the template's value entirely, prefix your key with `!`:

```json
{
  "!field_classification": {
    "land": { "numeric": ["only_this_one"] }
  }
}
```

Without the `!`, your value would be merged into the template's much longer field list.

### 1.5 `+key` — extend a template list

By default, lists are *replaced* by your value. If you want to *append* your items to the template list (set-union — duplicates are skipped), prefix with `+`:

```json
{
  "+field_classification": {
    "+land": {
      "+numeric": ["my_extra_field", "another_one"]
    }
  }
}
```

The `+` flag must be on every level of the path that should extend rather than replace.

---

## 2. Time adjustment

### `data.process.time_adjustment.from_file.<model_group>`

Replace OpenAVMKit's built-in time-adjustment engine with a precomputed CSV for a specific model group.

- **Default** — not set; built-in engine runs
- **Effect** — when set, OpenAVMKit reads the CSV at the configured path and uses those multipliers verbatim. The internal model is not run for that model group.
- **Source** — `read_time_adjustment_from_file` in [openavmkit/time_adjustment.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/time_adjustment.py)
- **When to use** — your jurisdiction publishes its own time-adjustment factors; you want reproducible adjustments across runs; you're debugging modeling and want to hold time-adjustment constant.

---

## 3. Data enrichment

Enrichment runs after data is loaded and before modeling. The orchestrator `_enrich_data` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py) calls each enrichment step in sequence. The presence (or `enabled` flag) of each subsection of `data.process.enrich` controls what runs.

### 3.1 Basic geometric enrichment — `data.process.enrich.basic`

Default-on. Computes from parcel geometry:

- **Lat/lon and normalized lat/lon** — `latitude`, `longitude`, `latitude_norm`, `longitude_norm` (parcel centroids in WGS84, plus min-max normalization)
- **GIS-derived land area** — `land_area_gis_<unit>`. When the assessor's `land_area_<unit>` is `0`, negative, or `NaN`, GIS area is automatically substituted; the original assessor value is preserved as `land_area_given_<unit>`, and the deviation is exposed as `land_area_gis_delta_<unit>` and `land_area_gis_delta_percent`. Assessor values are preferred by default — see the gotchas section in [AGENTS.md](https://github.com/landeconomics/openavmkit/blob/master/AGENTS.md).
- **Shape metrics** — `geom_rectangularity_num`, `aspect_ratio`, `geom_vertices`
- **Polar coordinates** — `polar_angle`, `polar_radius` (relative to the locality center)

Sub-flags (all default `true`): `latlon`, `area`, `shape`, `polar`. Set to `false` to skip individual steps. Set the parent `basic.enabled = false` to skip the entire stage.

- **Source** — `_basic_geo_enrichment` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)

### 3.2 Spatial joins — `data.process.enrich.spatial_joins`

Joins user-provided shapefiles (neighborhoods, school districts, zoning, etc.) onto the universe by spatial intersection.

- **Source** — `_enrich_df_spatial_joins` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — your locality has area-based reference layers that aren't in the parcel data and you want them as parcel-level fields.

### 3.3 Overture building footprints — `data.process.enrich.overture`

Pulls building footprints from the Overture Maps dataset and aggregates them onto each parcel.

- **Source** — `_enrich_df_overture` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — assessor data lacks building footprint counts/areas, or you want an external check on what's there.

### 3.4 Census enrichment — `data.process.enrich.census`

Spatial-joins parcels to US Census block groups and pulls demographic and income variables.

- **Activation** — set both the section and `census.enabled = true`
- **Requires** — a Census API key (see [config.md](config.md#configuring-census-api-access))
- **Source** — `_enrich_df_census` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — you want neighborhood demographics (median income, etc.) as model features.

### 3.5 Distance & proximity enrichment — `data.process.enrich.distances`

For each parcel, computes how close it is to features such as parks, water bodies, schools, transportation, the CBD, or individual landmarks. This is one of the most useful enrichments — and the most worth understanding in detail.

- **Activation** — presence of the `distances` key, **plus** `distances.enabled = true` (defaults to `false` at the inner level)
- **Source** — `_enrich_df_distances` and `_do_perform_distance_calculations_osm` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)

#### Two ways to source the geometry

Each feature class gets its geometry from one of two sources, chosen per feature:

1. **OpenStreetMap (`osm: true`)** — OpenAVMKit downloads the feature geometries from OSM automatically, using the bounding box of your locality. Good for generic, well-tagged feature classes (parks, water bodies, schools, transportation networks, etc.) where OSM coverage is reliable.
2. **A user-supplied shapefile (`source: <id>`)** — Reference a dataframe you've already loaded under `data.load.<id>` (any GeoDataFrame with a `geometry` column will do). Use this when:
    - OSM coverage is poor or wrong for your locality
    - You have an authoritative jurisdiction-published shapefile (downtown polygons, school district boundaries, named landmarks)
    - The feature you care about isn't a standard OSM tag (specific employment centers, custom amenity zones, your own neighborhood polygons)

Set **either** `osm: true` **or** `source: "<id>"` per feature — they're alternatives, not combinable. If you provide neither, the enrichment will raise an error for that feature.

##### Worked example: mixing OSM and user shapefiles

```json
{
  "data": {
    "load": {
      "cbd": {
        "filename": "cbd_polygon.shp"
      },
      "employment_centers": {
        "filename": "employment_centers.shp"
      }
    },
    "process": {
      "enrich": {
        "distances": {
          "enabled": true,
          "__comment": "OSM-sourced features:",
          "parks": {
            "enabled": true,
            "osm": true,
            "max_distance": 2.0,
            "unit": "km",
            "store_top": true,
            "top_n": 3
          },
          "water_bodies": {
            "enabled": true,
            "osm": true,
            "max_distance": 5.0,
            "unit": "km"
          },
          "__comment2": "User-shapefile-sourced features:",
          "cbd": {
            "enabled": true,
            "source": "cbd",
            "max_distance": 8.0,
            "unit": "km"
          },
          "employment_centers": {
            "enabled": true,
            "source": "employment_centers",
            "max_distance": 5.0,
            "unit": "km",
            "store_top": true,
            "top_n": 5
          }
        }
      }
    }
  }
}
```

A real example of the user-shapefile pattern lives in [notebooks/pipeline/data/us-fl-broward/in/settings.json](https://github.com/landeconomics/openavmkit/blob/master/notebooks/pipeline/data/us-fl-broward/in/settings.json) — Broward County loads local shapefiles for `cbd`, `airport`, `colleges`, `universities`, `golf_courses`, `lakes`, and `parks` and references them via `source`.

#### What gets produced

For each configured feature class (e.g. `parks`), every parcel gets **three** columns:

| Column                         | Meaning                                                                                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `dist_to_<feature>`            | Raw distance to the nearest instance, in the configured unit (default `km`). Past `max_distance`, clipped to `max_distance + 1`.       |
| `within_<feature>`             | Boolean: is the parcel within `max_distance` of any instance?                                                                          |
| `proximity_to_<feature>`       | `max(dist_to_<feature>) - dist_to_<feature>`. Past `max_distance`, falls to `0.0`. **Higher value = closer.**                          |

Plus a corresponding `log_dist_to_<feature>` (log transform of the raw distance) for use in models that benefit from log-distance.

#### Distance vs. proximity — which to regress on?

**Prefer `proximity_to_<feature>` over `dist_to_<feature>` for regression.** Two reasons:

1. **Direction matches intuition.** A higher proximity means the parcel is closer to the feature. A higher distance means the parcel is farther. Most "amenity" effects (parks, transit, water) move *with* proximity and *against* distance, so the sign of the regression coefficient on proximity is positive in the natural direction. Easier to read, easier to debug.
2. **Built-in saturation past the "I no longer care" threshold.** With `max_distance` set, proximity falls to `0.0` for any parcel beyond that distance — but distance keeps climbing. In real markets, a park 50 ft away matters a lot; a park 5 miles away does not — and the *difference* between 5 miles and 10 miles matters even less. A linear regression on distance has to fit that flat tail; a regression on proximity gets it for free because the tail is already clipped to zero.

You can use either column (or both, or `log_dist_to_*`) in your modeling — the choice is yours. But proximity is almost always the better default.

#### Per-feature options

| Key            | Default          | Effect                                                                                                          |
| -------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
| `enabled`      | `true`           | Toggle this specific feature.                                                                                   |
| `osm`          | `false`          | **Geometry source — option A.** Pull this feature's geometry from OpenStreetMap automatically.                  |
| `source`       | (none)           | **Geometry source — option B.** Reference the ID of a dataframe you've loaded under `data.load.<id>` (must contain a `geometry` column). Use this for jurisdiction-supplied shapefiles or anything OSM doesn't cover well. Mutually exclusive with `osm`. |
| `max_distance` | (none)           | Beyond this distance (in `unit`), parcels are clipped: `dist_to` saturates at `max_distance + 1`, `proximity_to` falls to `0.0`. Strongly recommended — sets the "no longer care" threshold. |
| `unit`         | `km`             | Distance unit. Affects every `dist_to` / `proximity_to` value for this feature.                                 |
| `store_top`    | `false`          | If `true`, also compute distances to the **top N individual named instances** (see below).                      |
| `top_n`        | `0`              | How many top instances to single out when `store_top` is true.                                                  |
| `sort_field`   | feature-specific | Field used to rank instances when picking the top N (e.g. `area` for parks, `length` for transportation).       |
| `type_field`   | feature-specific | Field used as a fallback name when an OSM feature has no `name` tag.                                            |

> Specify **either** `osm: true` **or** `source: "<id>"` per feature — not both, and not neither.

#### Distance to specific named features — `store_top` + `top_n`

By default, `dist_to_parks` is the distance to the *nearest* park — any park. But often you want to know about specific parks individually: "how far to Central Park" is a very different signal from "how far to the nearest pocket park."

Set `store_top: true` and `top_n: N` to additionally produce **per-named-feature** columns:

```json
"parks": {
  "enabled": true,
  "osm": true,
  "max_distance": 5.0,
  "store_top": true,
  "top_n": 3
}
```

This produces, in addition to the standard `dist_to_parks` / `within_parks` / `proximity_to_parks`:

- `dist_to_parks_<name>` and `proximity_to_parks_<name>` for each of the top 3 parks (ranked by `sort_field`, default `area` for parks, so largest first)

Names come from the OSM `name` tag when present (cleaned for use as a column name); otherwise they fall back to `<type_field_value>_<index>` (e.g. `park_5`).

Use this when individual landmarks are likely to drive value differently from each other — flagship parks, major employers, named transit stations — and you want the model to fit a separate coefficient per landmark.

#### Default feature classes

Six feature classes ship with sensible defaults baked in (you can still override). When you list one of these names, you don't need to specify `sort_field` or `type_field`:

| Feature        | `sort_field` | `type_field` | `store_top` default |
| -------------- | ------------ | ------------ | ------------------- |
| `coastline`    | `length`     | `natural`    | `false`             |
| `water_bodies` | `area`       | `water`      | `true`              |
| `transportation` | `length`   | `highway`    | `false`             |
| `educational`  | `area`       | `amenity`    | `true`              |
| `parks`        | `area`       | `leisure`    | `true`              |
| `golf_courses` | `area`       | `leisure`    | `true`              |

Any other feature name (e.g. `cbd`, `airport`, `university`) is also supported — just provide the configuration explicitly.

#### When to use

- Your value model treats proximity to landmarks (parks, water, transit, the CBD) as a real driver of price.
- You have a small set of known specific landmarks that matter individually (use `store_top` + `top_n`).
- You want a regression-friendly proximity feature that saturates past a sensible "I no longer care" threshold (set `max_distance`).

### 3.6 Building permits — `data.process.enrich.permits`

Joins permit records onto sales by parcel ID and date, producing fields like recent renovation activity that explain post-sale price differences.

- **Activation** — presence of the `permits` key
- **Source** — `_enrich_permits` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — your jurisdiction publishes a permit feed and you want to control for permitted improvements.

### 3.7 OpenStreetMap streets — `data.process.enrich.streets.enabled`

Adds OSM-derived street-network features to each parcel: frontage broken down by street class (motorway, primary, residential, …), street speed limits, lane counts, plus a Somers-unit-normalized land area derived from frontage and depth.

- **Default** — `false`
- **Source** — `enrich_df_streets` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **Performance** — this step is **computationally expensive**. It can take a long time to run on a large locality the first time. However, the result is cached and does not need to be regenerated unless the locality's parcel geometry changes. Plan accordingly: budget time for the first run, and don't worry about subsequent runs.

### 3.8 Spatial lag — `data.process.enrich.spatial_lag`

For each parcel, computes neighborhood averages of selected fields (sale price, building age, floor-area ratio, bedroom density, etc.). Produces dozens of `spatial_lag_*` columns that capture local context not encoded in categorical location fields.

- **Source** — `enrich_sup_spatial_lag` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — when you want models to see neighborhood smoothing of key signals, especially price.

### 3.9 Spatial inference (gap fill) — `data.process.enrich.infer`

Fills missing values for selected fields using geospatial patterns from nearby parcels. Runs after all other enrichments so it can use enriched fields as predictors.

- **Activation** — presence of the `infer` key
- **Source** — `_enrich_spatial_inference` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — you have systematically missing data on subsets of parcels and "this parcel probably looks like its neighbors" is a defensible assumption.

---

## 4. Data cleaning & validation

### 4.1 Filling missing values — `data.process.fill.<method>`

`data.process.fill` is a dict whose **keys are fill methods** and whose **values are lists of fields** to apply that method to. The cleaner walks each method/field-list pair and applies the named method to each field's missing values. See `_fill_unknown_values` in [openavmkit/cleaning.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/cleaning.py).

Available fill methods:

| Method      | Replaces missing with                                | When to use                                                                 |
| ----------- | ---------------------------------------------------- | --------------------------------------------------------------------------- |
| `zero`      | `0`                                                  | Building area on vacant parcels; count fields where missing means absent    |
| `unknown`   | `"UNKNOWN"`                                          | Categorical fields with a meaningful "data not provided" bucket             |
| `none`      | `"NONE"`                                             | Categorical fields where missing means "this attribute does not apply"      |
| `false`     | `False` (or `"False"` for string columns)            | Boolean attributes where missing means "not present"                        |
| `mode`      | The modal value of that field (most-common non-null) | Categorical fields where missing should default to the population mode      |
| `median`    | The median of non-null values                        | Numeric fields with skewed distributions (more robust than mean)            |
| `mean`      | The mean of non-null values                          | Numeric fields with roughly symmetric distributions                         |
| `max`       | The maximum non-null value                           | Rare; use when "missing should not penalize relative to peers"              |
| `min`       | The minimum non-null value                           | Rare; mirror case of `max`                                                  |
| `custom`    | A user-specified value                               | One-off field/value pairs that don't fit a standard method                  |

For `custom`, the list contains dicts instead of plain field names:

```json
{
  "data": {
    "process": {
      "fill": {
        "custom": [
          { "field": "bldg_style", "value": "single_family" }
        ]
      }
    }
  }
}
```

#### Conditional suffixes — `_impr` and `_vacant`

Any fill method can be scoped to improved or vacant parcels by suffixing the method name. The cleaner strips the suffix and applies the underlying method to the matching subset only:

| Method key       | Applied to                              |
| ---------------- | --------------------------------------- |
| `zero_vacant`    | Only parcels where `is_vacant == True`  |
| `zero_impr`      | Only parcels where `is_vacant == False` |
| `unknown_vacant` | Only vacant parcels                     |
| `unknown_impr`   | Only improved parcels                   |
| `mode_impr`      | Only improved parcels                   |
| (etc.)           | (same pattern for any method)           |

Example — fill `bldg_style` with the modal value, but only for improved parcels (vacant parcels keep their NaN, which other logic handles):

```json
{
  "data": {
    "process": {
      "fill": {
        "mode_impr": ["bldg_style", "bldg_exterior"]
      }
    }
  }
}
```

#### Automatic post-fill cleanup

After your configured fills run, the cleaner does a few things automatically:

1. **Year-built / age-years reconciliation.** If `bldg_year_built` exists, `bldg_age_years` is recomputed as `valuation_year - bldg_year_built` (and clamped to `0` for non-positive year-built values). If only `bldg_age_years` exists, `bldg_year_built` is derived from it. The same logic applies to `bldg_effective_year_built` / `bldg_effective_age_years`. After reconciliation, all four year/age fields get a final `zero` fill.
2. **Categorical auto-fill.** Any categorical field configured via `field_classification.categorical` that still has NaN after all explicit fills is filled with `"UNKNOWN"`. Boolean fields are similarly normalized.
3. **Per-model-group execution.** All fills are applied per model group, so a `mode` or `median` fill uses the model group's distribution rather than the global one — see `_fill_unknown_values_per_model_group` in [openavmkit/cleaning.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/cleaning.py).

### 4.2 `data.process.reconcile`

Post-merge reconciliation rules. After sales and universe data are merged, these rules let you resolve conflicts (e.g. when both sides have a value for the same field) by ID.

- **Default** — empty dict `{}`
- **Source** — `_merge_dict_of_dfs` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — your sales and universe both carry overlapping fields and you need deterministic precedence rules.

### 4.3 `data.validation.enabled`

Run validation checks after data processing — value ranges, required fields, expected types, etc.

- **Default** — `false`
- **Effect** — when `true`, validation checks run and report issues. When `false`, they're skipped silently.
- **Source** — `filter_invalid_sales` in [openavmkit/cleaning.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/cleaning.py)
- **When to use** — turn on whenever you're onboarding a new locality or have changed cleaning logic. Leave off only for fast inner-loop iteration once you trust the inputs.

---

## 5. Modeling control

> For the **full catalog of model engines** (XGBoost, LightGBM, CatBoost, MRA, GWR, kernel, SLICE, baselines, etc.), the **model-name-vs-engine dispatch** mechanism, and how to run **multiple variants of the same engine** (e.g. two XGBoost configurations side-by-side), see **[Models reference](models_reference.md)**. The settings on this page are the orchestration layer; that page documents each model.

### `modeling.instructions.<main|vacant|hedonic>.run`

Explicit list of model names to run for the main, vacant, or hedonic stages. Without it, all models defined under `modeling.models` are run.

- **Source** — `_run_models` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — you want a fast iteration on a single model, or you want to skip slow models (e.g. `gwr`) for a quick run.

### `modeling.instructions.<main|vacant|hedonic>.skip.<model_group>`

Per-model-group skip list. For the named model group, the listed models are skipped even if they're in `run`.

- **Source** — `_run_models` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — a particular model is unstable on a particular model group (low sample count, rank-deficient features) and you want to exclude it from that group only.

### `modeling.instructions.feature_selection.thresholds` and `.weights`

Fine-tune the variable-selection scoring used during model setup. The thresholds gate inclusion (correlation, VIF, p-value, t-value, ENR coefficient, adjusted R²); the weights control how each test contributes to the composite score.

- **Default** (from template) —

  ```json
  {
    "thresholds": {
      "correlation": 0.1, "vif": 10, "p_value": 0.05,
      "t_value": 2, "enr_coef": 0.01, "adj_r2": 0.05
    },
    "weights": {
      "vif": 3, "p_value": 3, "t_value": 2,
      "enr_coef": 2, "corr_score": 2, "coef_sign": 2, "adj_r2": 1
    }
  }
  ```

- **Source** — `modeling.instructions.feature_selection` in [openavmkit/resources/settings/settings.template.json](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/resources/settings/settings.template.json), consumed in `benchmark.py`.
- **When to use** — your standard variable-selection results don't reflect domain knowledge. Loosen `correlation` to keep weak-but-meaningful features, or tighten `vif` to drop more multicollinear ones.

### `modeling.try_variables.variables`

Run a dedicated variable-importance experiment over a custom set of candidate variables before main modeling. Surfaced via [openavmkit.pipeline.try_variables](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/pipeline.py).

- **Source** — `try_variables` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — you have hypotheses about which variables matter and want a slower, more thorough comparison than the auto-reduction step does inline.

---

## 6. Analysis & QA

### `analysis.outliers.skip`

List of model groups to exclude from outlier analysis entirely.

- **Source** — `identify_outliers` in [openavmkit/pipeline.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/pipeline.py)
- **When to use** — small or unusual model groups where the outlier heuristics generate too many false positives.

### `analysis.outliers.model_groups.<id>` and `analysis.outliers.default`

Per-model-group outlier detection config; if no entry exists for a model group, `default` is used. Each entry can specify which model type's predictions to use for each of `main`, `vacant`, `hedonic_land`.

- **Source** — `identify_outliers` in [openavmkit/pipeline.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/pipeline.py)

### `analysis.ratio_study.trim.<model_group>.max_percent`

Maximum fraction of records the ratio study is allowed to trim as outliers when computing trimmed statistics (COD-trim, PRD-trim, etc.).

- **Default** — `0.1` (10%)
- **Source** — `_get_max_ratio_study_trim` in [openavmkit/utilities/settings.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/utilities/settings.py)
- **When to use** — your sales data is unusually noisy and the default 10% trim is masking real volatility, or unusually clean and a tighter cap is appropriate.

### `analysis.ratio_study.look_back_years`

How many years before the valuation date to include sales from when running the ratio study.

- **Default** — `1` (from template)
- **Source** — `get_look_back_dates` in [openavmkit/utilities/settings.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/utilities/settings.py)
- **When to use** — your jurisdiction expects a multi-year ratio-study window, or you want to widen the sample for low-volume model groups.

---

## 7. Caching & checkpoints

### 7.1 Three cache layers

OpenAVMKit caches expensive intermediate results in three places:

- **Notebook checkpoints** at `<locality>/out/checkpoints/` — per-notebook intermediate state, written by every `from_checkpoint(...)` call.
- **Enrichment cache** at `<locality>/cache/` — used internally by enrichment steps that pull from remote sources (OpenStreetMap, Census, Overture) or do heavy local computation (street networks, distance calculations).
- **Saved model parameters** at `<locality>/out/models/<model_group>/.../` — tuned hyperparameters and bandwidths from previous model runs (XGBoost / LightGBM / CatBoost Optuna results, GWR bandwidth, kernel regression bandwidth).

The first two layers are *designed* to self-invalidate when the relevant inputs or settings change. The third layer (saved model parameters) has different semantics — see § 7.4 below.

### 7.2 The notebook checkpoint system

Every cell that wraps a function call in `from_checkpoint(...)` saves its result to disk:

```python
sup = from_checkpoint(
    "1-assemble-02-process_data",
    process_dataframes,
    {"dataframes": dataframes, "settings": settings}
)
```

On re-run, the cell reads `out/checkpoints/1-assemble-02-process_data.parquet` (or `.pickle`) and **skips the function call entirely**. This is the mechanism that lets you stop and resume between notebooks.

**Public API:**

- `from_checkpoint(prefix, func, params)` — load if cached, otherwise run and save
- `delete_checkpoints("<prefix>")` — delete all checkpoints starting with the given prefix (e.g. `delete_checkpoints("1-assemble")` clears all of notebook 1's intermediate state)
- `clear_checkpoints = True` at the top of a notebook — convention used in the pipeline notebooks; combined with a `delete_checkpoints("<this_notebook>")` call, gives you a one-flag "start fresh" toggle

**Source** — [openavmkit/checkpoint.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/checkpoint.py).

### 7.3 The enrichment cache

Enrichment steps that pull from remote sources or do heavy computation (streets, OSM features, Census joins, Overture footprints, distance calculations) cache their intermediate results under `<locality>/cache/`. Each cached entry is written alongside a "signature" — typically the relevant settings subtree — and the cache is invalidated automatically when the signature changes.

Cost characteristics:

| Enrichment | Cold-cache cost |
| --- | --- |
| OSM streets | Minutes to **hours** for a large bbox |
| Overture footprints | Minutes |
| Distance / proximity (per feature) | Seconds to minutes |
| OSM feature pulls (parks, water, transit, etc.) | Seconds to minutes |
| Census | Seconds |

Streets in particular benefit from caching — a fresh run on a large jurisdiction can take hours, but cached re-runs complete in seconds.

**Source** — [openavmkit/utilities/cache.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/utilities/cache.py).

### 7.4 Saved model parameters — different semantics

This is a separate concern from the other two layers. Tunable models (XGBoost, LightGBM, CatBoost via Optuna; GWR via bandwidth search; kernel regression via bandwidth search) can save their tuned parameters to disk after a successful tuning run, so subsequent runs can skip the (expensive) parameter search.

**Files written** (under `<locality>/out/models/<model_group>/<subdir>/`):

| File | Produced by | Contains |
| --- | --- | --- |
| `<slug>_params.json` | XGBoost / LightGBM / CatBoost (Optuna) | Best tuned hyperparameters from the last search |
| `<model_name>_bw.json` | GWR | Optimal bandwidth from the last search |
| `kernel_bw.pkl` | Kernel regression | Optimal per-variable bandwidth from the last search |

**Mechanism** — controlled by two flags passed to the model runners (`run_xgboost`, `run_lightgbm`, `run_catboost`, `run_gwr`, `run_kernel`):

- `save_params=True` — write the tuned params after a successful tuning run
- `use_saved_params=True` — on a re-run, read the saved file and skip the tuning search

**The trade-off** — different from the other two cache layers:

- **Delete the file → forces a fresh hyperparameter search.** The next run re-tunes from scratch. Slow, but adapts to changes in your training data.
- **Keep the file → skips the parameter search.** The next run uses the saved hyperparameters and trains the model with them. Fast, but the model is constrained by the *previous* run's tuning even if your training data has shifted.

> **In other words**: the saved params don't cache *predictions*. The model still re-fits on whatever training data it sees. They cache the *tuning step* — the search for which hyperparameters to use. That's a much bigger deal for tree-based models with deep search spaces (Optuna trials can take minutes to hours) than for fast-fitting linear models.

**When to delete saved params:**

- You've meaningfully changed the training data (different sales, different features, different model group definitions, different fill rules) and want the tuning to adapt.
- You suspect the previous tuning was over-fit to a transient quirk of the data.
- Your jurisdiction-specific characteristics changed in a way that should affect which hyperparameters are best.

**When to keep them:**

- You want fast, reproducible re-runs (e.g. for iterating on downstream analysis without paying the tuning cost again).
- You're confident the previous tuning is still appropriate — incremental data changes that aren't likely to shift the optimal hyperparameters.

### 7.5 When self-invalidation isn't enough

The notebook checkpoint and enrichment cache layers are *designed* to invalidate automatically, but it isn't perfect. Edge cases happen:

- A signature comparison can miss a subtle change in inputs
- A partial write (interrupted run, full disk) can leave a corrupt cache file
- A remote source can change shape without changing what the local signature sees

**If you're getting weird behavior — your settings change doesn't seem to do anything, output looks suspiciously similar to a previous run, or an enrichment is missing fields you know it should have — nuke the cache to be safe.**

Symptoms to watch for:

- You changed a value in `settings.json` and the next run produced identical output
- A model group's predictions look identical to the last run despite settings changes
- An enrichment step says it "found cached data" when you expected it to re-fetch
- Distance / proximity columns are missing for features you've added

**How to nuke** (mostly harmless; you'll just pay re-run cost):

| Command | Effect |
| --- | --- |
| `delete_checkpoints("<prefix>")` from a notebook | Clear specific notebook checkpoints (e.g. `"1-assemble"` clears all of notebook 1's state) |
| Set `clear_checkpoints = True` at top of notebook | Convention in pipeline notebooks; clears that notebook's checkpoints on re-run |
| `rm -rf <locality>/out/checkpoints/` | Wipe all notebook checkpoints for the locality |
| `rm -rf <locality>/cache/` | Wipe the enrichment cache entirely |
| Delete `<outpath>/<slug>_params.json`, `<model_name>_bw.json`, or `kernel_bw.pkl` | Force a fresh hyperparameter search on next model run (see § 7.4) |

**Don't nuke prophylactically.** OSM streets and Overture pulls are expensive on a fresh cache — clearing them every run will cost you hours over a project. Hyperparameter searches can also be expensive (Optuna with 100+ trials on a tree-based model can take a long time). Nuke when something feels off, then re-run.

---

## See also

- [Models reference](models_reference.md) — full catalog of model engines, invocation patterns, multi-variant runs
- [The `calc` expression language](calc_reference.md) — the full operator reference for `calc` blocks (used in `data.load.<id>.calc`, `data.process.calc`, etc.)
- [Configuration](config.md) — `.env`, cloud storage, PDF generation
- [The Basics](the_basics.md) — locality folder structure, terminology
- [Recipe](recipe.md) — public function reference and notebook map
- Real settings examples: `notebooks/pipeline/data/<locality>/in/settings.json`
- The settings template: [openavmkit/resources/settings/settings.template.json](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/resources/settings/settings.template.json)
