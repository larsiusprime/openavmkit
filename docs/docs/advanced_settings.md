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

## 2. Data load: `data.load.<id>`

Each subkey under `data.load` declares one source file: where to read it, how to map source columns onto canonical OpenAVMKit field names, and what to do about duplicate rows. The basic mapping patterns (scalar rename, two-element list with dtype, three-element list with date format, plus the `calc` block) are introduced in [tutorial.md § B.4](tutorial.md#b4-author-a-minimum-viable-settingsjson). This section covers the parts that aren't there: the **`dupes` rule** and its full schema, which is how you handle source files where one parcel spans multiple rows.

### 2.1 The `dupes` rule

A `dupes` value goes alongside `filename` and `load`:

```json
"parcels": {
  "filename": "parcels.csv",
  "dupes": "auto",
  "load": { "key": ["REID", "string"], "...": "..." }
}
```

It can take three forms:

| Value | Meaning |
| --- | --- |
| `"auto"` (or omitted for geometry) | OpenAVMKit picks the first usable key column (`key_sale`, then `key`, `key2`, `key3`) for non-geometry data, or the first non-`geometry` column for shapefiles. Sorts ascending by that column and drops later duplicates. |
| `"allow"` | Pass-through. No dedupe, no aggregation. Use when downstream code is supposed to see the duplicates (rare). |
| Object | Custom dedupe rule, with optional aggregation. Detailed below. |

For non-geometry dataframes, **omitting `dupes` entirely** is equivalent to `{}` — duplicates are *not* checked. That's almost never what you want; pick `"auto"` unless you're writing a custom rule.

- **Source** — `get_dupes` in [openavmkit/utilities/settings.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/utilities/settings.py), `_handle_duplicated_rows` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py).

### 2.2 Custom dedupe rules

The object form has four keys:

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `subset` | string or list of strings | `"key"` | Columns whose combined value defines a duplicate. |
| `sort_by` | `[col, "asc"\|"desc"]` or list of such pairs | none | Sort applied **before** dedupe — controls which row "wins" when `drop: true`. |
| `drop` | `true`, `false`, `"all"` | `true` | `true` keeps the first row per group; `"all"` drops every row that has a duplicate; `false` keeps everything (use this when you only want aggregation). |
| `agg` | object | none | Per-field aggregation rules — see § 2.3. |

Subset names are matched against **canonical names** (the post-`load` names), not your source file's column headers, because the `load` mapping has already run by the time `dupes` is applied.

**Worked example.** A jurisdiction's parcels file uses `(REID, CARD_NUMBER)` to identify rows where one parcel has multiple buildings. Keep the lowest card number for each parcel:

```json
"parcels": {
  "filename": "parcels.csv",
  "dupes": {
    "subset": ["key"],
    "sort_by": [["card_number", "asc"]],
    "drop": true
  },
  "load": {
    "key": ["REID", "string"],
    "card_number": "CARD_NUMBER",
    "...": "..."
  }
}
```

This is fine when card 1 is the primary building and the per-card fields on cards 2+ are not needed. But if the parcel-level fields (e.g. `assr_market_value`) are repeated identically on every card while per-card fields (`HEATED_AREA`, `YEAR_BUILT`, `BATH_FIXTURES`) carry distinct values per building, plain dedupe throws away real information. That's where `agg` comes in.

### 2.3 Aggregation across duplicate rows — `dupes.agg`

The `agg` block tells OpenAVMKit to compute a per-group summary for one or more fields **before** dedupe-and-merge. The aggregated values overwrite whatever was in those columns on the surviving row.

```json
"parcels": {
  "filename": "parcels.csv",
  "dupes": {
    "subset": ["key"],
    "sort_by": [["card_number", "asc"]],
    "drop": true,
    "agg": {
      "bldg_area_finished_sqft":   { "field": "bldg_area_finished_sqft",   "op": "sum" },
      "bldg_year_built":           { "field": "bldg_year_built",           "op": "min" },
      "bldg_effective_year_built": { "field": "bldg_effective_year_built", "op": "max" },
      "bldg_rooms_bath_full":      { "field": "bldg_rooms_bath_full",      "op": "sum" },
      "bldg_units":                { "field": "bldg_units",                "op": "sum" }
    }
  }
}
```

How each agg entry resolves:

| Sub-key | Meaning |
| --- | --- |
| (the entry's name) | Output column. Usually identical to `field` so the aggregated value lands back in the canonical column. Make it different if you want both raw and aggregated values side-by-side — but note the merge step drops the original column when names collide and warns. |
| `field` | Source column to aggregate. Canonical name (post-`load`). |
| `op` | Pandas aggregation function — any string accepted by `DataFrame.groupby(...).agg({col: op})`. Common choices: `"sum"`, `"mean"`, `"median"`, `"min"`, `"max"`, `"first"`, `"last"`, `"count"`, `"nunique"`. |
| `sort_by` | Optional per-aggregation sort order (same shape as the outer `sort_by`). Only useful with `"first"` / `"last"`, where you want a different sort than the dedupe sort — for example, dedupe by `card_number asc` but pull `"first"` `bldg_quality_txt` from the row with the largest `bldg_area_finished_sqft`. |

Mechanics, in order:

1. The full pre-dedupe dataframe is sorted by the outer `sort_by`.
2. Duplicates are dropped per `drop`. The result is the **base** — one row per `subset` value.
3. For each `agg` entry, the **original** (pre-dedupe) dataframe is grouped by `subset` and aggregated. Result column is renamed to the entry's name.
4. All aggregated tables are merged together on `subset`, then merged onto the base with a left join.
5. Where aggregated column names collide with base column names, the base columns are dropped first (and a warning is emitted) so the aggregated values win.

**Picking the right `op` per field.** As a rough rule:

- **Building size, room counts, fixtures, units** → `"sum"` (totals across all buildings on the parcel).
- **Year built** → `"min"` (oldest building governs depreciation curves; a remodel addition shouldn't make a 1920 farmhouse look 1995).
- **Effective year built / remodel year** → `"max"` (most recent improvement).
- **Quality / condition / style** when stored as ordinal numerics → `"max"` if you want the best building to set the parcel level, `"first"` with a `sort_by` on building size if you want the largest building to set it.
- **Categorical fields** with no obvious total — quality letter grades, foundation type, style — usually take `"first"` and live with whatever card 1 reports, unless you're prepared to define a tie-break sort.
- **Parcel-level fields** that repeat identically across rows (assessed values, sale price, deeded acreage) don't need `agg` at all — `drop: true` will keep the surviving row's copy, which is correct.

### 2.4 Dedupe for sales

Sales tables follow the same shape, but the convention is to dedupe on `key_sale` (a synthesized per-transaction identifier) rather than `key`:

```json
"sales": {
  "filename": "sales.csv",
  "dupes": {
    "subset": ["key_sale"],
    "sort_by": [["key_sale", "asc"]],
    "drop": true
  }
}
```

If a sales dataframe ever lacks `key_sale` in its dedupe `subset`, OpenAVMKit emits a warning at load time — see `_handle_duplicated_rows` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py) — because deduping sales by parcel `key` alone collapses multiple legitimate transactions on the same parcel into one row.

### 2.5 `dupes` at the merged level too

`data.process.dupes.universe` and `data.process.dupes.sales` apply the same schema **after** all per-source tables are merged. This is the right place for cross-table rules — for example, deduping the joined universe by `key` once parcels and a separate building file have been combined. The Guilford locality file uses this pattern via `$$ref.dupes_universe` and `$$ref.dupes_sales`.

---

## 3. Time adjustment

### `data.process.time_adjustment.from_file.<model_group>`

Replace OpenAVMKit's built-in time-adjustment engine with a precomputed CSV for a specific model group.

- **Default** — not set; built-in engine runs
- **Effect** — when set, OpenAVMKit reads the CSV at the configured path and uses those multipliers verbatim. The internal model is not run for that model group.
- **Source** — `read_time_adjustment_from_file` in [openavmkit/time_adjustment.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/time_adjustment.py)
- **When to use** — your jurisdiction publishes its own time-adjustment factors; you want reproducible adjustments across runs; you're debugging modeling and want to hold time-adjustment constant.

---

## 4. Data enrichment

Enrichment runs after data is loaded and before modeling. The orchestrator `_enrich_data` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py) calls each enrichment step in sequence. The presence (or `enabled` flag) of each subsection of `data.process.enrich` controls what runs.

### 4.1 Basic geometric enrichment — `data.process.enrich.basic`

Default-on. Computes from parcel geometry:

- **Lat/lon and normalized lat/lon** — `latitude`, `longitude`, `latitude_norm`, `longitude_norm` (parcel centroids in WGS84, plus min-max normalization)
- **GIS-derived land area** — `land_area_gis_<unit>`. When the assessor's `land_area_<unit>` is `0`, negative, or `NaN`, GIS area is automatically substituted; the original assessor value is preserved as `land_area_given_<unit>`, and the deviation is exposed as `land_area_gis_delta_<unit>` and `land_area_gis_delta_percent`. Assessor values are preferred by default — see the gotchas section in [AGENTS.md](https://github.com/landeconomics/openavmkit/blob/master/AGENTS.md).
- **Shape metrics** — `geom_rectangularity_num`, `geom_aspect_ratio`, `geom_vertices`
- **Polar coordinates** — `polar_angle`, `polar_radius` (relative to the locality center)

Sub-flags (all default `true`): `latlon`, `area`, `shape`, `polar`. Set to `false` to skip individual steps. Set the parent `basic.enabled = false` to skip the entire stage.

- **Source** — `_basic_geo_enrichment` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)

### 4.2 Spatial joins — `data.process.enrich.spatial_joins`

Joins user-provided shapefiles (neighborhoods, school districts, zoning, etc.) onto the universe by spatial intersection.

- **Source** — `_enrich_df_spatial_joins` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — your locality has area-based reference layers that aren't in the parcel data and you want them as parcel-level fields.

### 4.3 Overture building footprints — `data.process.enrich.overture`

Pulls building footprints from the Overture Maps dataset and aggregates them onto each parcel.

- **Source** — `_enrich_df_overture` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — assessor data lacks building footprint counts/areas, or you want an external check on what's there.

### 4.4 Census enrichment — `data.process.enrich.census`

Spatial-joins parcels to US Census block groups and pulls demographic and income variables.

- **Activation** — set both the section and `census.enabled = true`
- **Requires** — a Census API key (see [config.md](config.md#configuring-us-census-api-access))
- **Source** — `_enrich_df_census` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — you want neighborhood demographics (median income, etc.) as model features.

### 4.5 Distance & proximity enrichment — `data.process.enrich.distances`

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

> **OSM-sourced features get an `osm_` prefix.** When `osm: true`, the column names are `dist_to_osm_<feature>`, `within_osm_<feature>`, `proximity_to_osm_<feature>`, `log_dist_to_osm_<feature>`. Only `source:`-supplied features get the bare name. Reference the prefixed names in your models when the feature was pulled from OSM.

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

This produces, in addition to the standard `dist_to_osm_parks` / `within_osm_parks` / `proximity_to_osm_parks` (note the `osm_` prefix because `osm: true`):

- `dist_to_osm_parks_<name>` and `proximity_to_osm_parks_<name>` for each of the top 3 parks (ranked by `sort_field`, default `area` for parks, so largest first). For `source:`-supplied features the prefix is omitted (e.g. `dist_to_parks_<name>`).

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

### 4.6 Building permits — `data.process.enrich.permits`

Joins permit records onto parcels and sales to detect (a) teardown sales — sales where the buyer demolishes the existing structure shortly after — and (b) recent renovations that explain otherwise-anomalous prices. Solves a class of outliers no model-side trick can: an old small house in an expensive area selling for "land + future construction" money rather than "house" money.

- **Activation** — `data.process.enrich.permits.sources` is a non-empty list, AND each named source must be present in `data.load`.
- **Source** — `_enrich_permits` / `_process_permits_sales` / `_process_permits_univ` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py).

#### Required columns in each permits source

The dataframe loaded under `data.load.<source_id>` (after column renaming) must contain:

| Field | Type | Meaning |
| --- | --- | --- |
| `key` | string | Parcel key — must match the universe `key`. |
| `date` | datetime | Permit issue date. Must be a real datetime dtype, not a string. |
| `is_teardown` | bool | True for demolition permits. |
| `is_renovation` | bool | True for additions / alterations / major remodels. |
| `renovation_txt` | string (optional) | Human-readable label for the renovation (logged for diagnostics). |
| `renovation_num` | int (optional) | Severity score for the renovation. **`3` = major** (significant enough to reset effective year built); lower values mean less. |

Multiple sources can be listed under `sources`; they're concatenated before processing.

#### What it produces

Sales-side fields (added to the sales dataframe):

| Field | Type | Set by |
| --- | --- | --- |
| `is_teardown_sale` | bool | True if a demolition permit was issued within `max_days_to_demo` (default 365) **after** the sale date — i.e. the buyer purchased the parcel and then demolished. |
| `demo_date` | datetime | The matched demolition date (must be *after* the sale). |
| `days_to_demo` | int | Days from sale to demolition (positive). |
| `vacant_sale` | bool | **Automatically set to True for `is_teardown_sale` rows.** This is the key behavior — teardown sales get reclassified as vacant-land sales for training, since the buyer paid for the lot, not the house. |

Sales of *already-demolished* parcels (sale_date > demo_date) are intentionally **not** flagged here — those are pre-cleared lot sales, which most jurisdictions already label as `Land` or `Vacant` in their sale-type field. They reach `vacant_sale=True` through that path, not through this enrichment.

Universe-side fields (added to the universe dataframe):

| Field | Type | Set by |
| --- | --- | --- |
| `last_permit_was_teardown` | bool | True if the parcel's most recent permit (before the valuation date) was a demolition. |
| `demo_date` | datetime | Date of that demolition. |
| `reno_date` | datetime | Date of the most recent significant renovation before the valuation date. |
| `days_to_reno` | int | Days from valuation date back to that renovation (negative). |
| `renovation_num` | int | Severity of that renovation. |
| `renovation_txt` | string | Label of that renovation. |

If `calc_effective_age: true`, the universe-side step ALSO recomputes `bldg_effective_year_built`: parcels with a `renovation_num == 3` (major) renovation have their effective year set to the renovation year. Use this when you trust the permit data more than the assessor's effective-year field.

#### Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `sources` | `[]` | List of dataframe IDs (must each be loaded under `data.load`). |
| `max_days_to_demo` | `365` | A sale is flagged as a teardown only if a demolition permit follows within this many days. Loosen for slow markets, tighten if you have lots of demos that aren't sale-related. |
| `calc_effective_age` | `false` | Recompute `bldg_effective_year_built` from major renovations. |

#### Worked example

```json
{
  "data": {
    "load": {
      "permits": {
        "filename": "permits.csv",
        "dupes": "auto",
        "load": {
          "key": "REID",
          "date": ["ISSUE_DATE", "datetime", "%Y-%m-%d"],
          "is_teardown": "is_teardown",
          "is_renovation": "is_renovation",
          "renovation_txt": "WORK_CLASS",
          "renovation_num": "renovation_num"
        }
      }
    },
    "process": {
      "enrich": {
        "permits": {
          "sources": ["permits"],
          "max_days_to_demo": 540,
          "calc_effective_age": true
        }
      }
    }
  }
}
```

#### Common pitfalls

- **PIN ≠ REID.** Many jurisdictions key permits on `PIN` (the parcel identifier number) but assessor data on `REID` (a sequential real-estate ID). You'll likely need to join PIN → REID via your parcels file before producing the permits CSV. Once produced, the `key` field in the permits dataframe must match whatever your universe uses.
- **`is_teardown` / `is_renovation` are NOT in raw permit data.** Your jurisdiction publishes `PERMIT_TYPE`, `WORK_CLASS`, or text descriptions. You must classify these into the booleans before loading. A small preprocessor that reads the raw permits and emits a clean CSV is the path of least resistance.
- **Date alignment.** `date` must be parsed as a real datetime by pandas before reaching the enricher. Spell out the format string in `data.load.<source>.load.date` (e.g. `["ISSUE_DATE", "datetime", "%Y-%m-%d %H:%M:%S"]`) — incorrect parsing silently produces `NaT` and the enricher will skip those rows.
- **Effective age leakage risk.** If you set `calc_effective_age: true` AND your training data already had `bldg_effective_year_built` from the assessor, you may be replacing one signal with another — verify the new field is sensibly distributed before keeping the override.

- **When to use** — your jurisdiction publishes a permit feed and (a) you want to detect teardown sales for training, or (b) you want to control for permitted renovations.

### 4.7 OpenStreetMap streets — `data.process.enrich.streets.enabled`

Adds OSM-derived street-network features to each parcel: frontage broken down by street class (motorway, primary, residential, …), street speed limits, lane counts, plus a Somers-unit-normalized land area derived from frontage and depth.

- **Default** — `false`
- **Source** — `enrich_df_streets` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **Performance** — this step is **computationally expensive**. It can take a long time to run on a large locality the first time. However, the result is cached and does not need to be regenerated unless the locality's parcel geometry changes. Plan accordingly: budget time for the first run, and don't worry about subsequent runs.

### 4.8 Spatial lag — `data.process.enrich.spatial_lag`

For each parcel, computes neighborhood averages of selected fields (sale price, building age, floor-area ratio, bedroom density, etc.). Produces dozens of `spatial_lag_*` columns that capture local context not encoded in categorical location fields.

- **Source** — `enrich_sup_spatial_lag` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — when you want models to see neighborhood smoothing of key signals, especially price.

### 4.9 Spatial inference (gap fill) — `data.process.enrich.infer`

Fills missing values for selected fields using geospatial patterns from nearby parcels. Runs after all other enrichments so it can use enriched fields as predictors.

- **Activation** — presence of the `infer` key
- **Source** — `_enrich_spatial_inference` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py); model orchestration in [openavmkit/inference.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/inference.py)
- **When to use** — you have systematically missing data on subsets of parcels and "this parcel probably looks like its neighbors" is a defensible assumption.

#### Per-field model config

Each entry under `infer` is keyed by the field to be filled, and its value is the model config for that field. The supported sub-keys are:

| Key | Type | Effect |
| --- | --- | --- |
| `model.type` | string | Required. One of `ratio_proxy`, `random_forest`, `lightgbm`, `xgboost`, `ensemble`. |
| `model.proxies` | list of strings | Predictor fields used by the inference model. |
| `model.locations` | list of strings | Location fields to group by (e.g. `["neighborhood", "market_area"]`). If omitted, OpenAVMKit emits a warning and runs inference as a single global group. To run global-only without the warning, set this to `[]` explicitly. |
| `model.group_by` | list of strings | (`ratio_proxy` only) Additional grouping fields combined with each location. |
| `model.interactions` | list | (tree-based and ensemble) Variable-interaction config. |
| `filters` | list | Filter expressions limiting which rows to predict for. |
| `fill` | list | List of follow-up fields to fill from the inferred values. |

```json
{
  "data": {
    "process": {
      "enrich": {
        "infer": {
          "bldg_area_finished_sqft": {
            "model": {
              "type": "lightgbm",
              "proxies": ["bldg_area_footprint_sqft", "land_area_sqft"],
              "locations": ["neighborhood", "market_area"]
            }
          }
        }
      }
    }
  }
}
```

### 4.10 Reference table joins — `data.process.enrich.ref_tables`

Joins a separately-loaded "reference" dataframe onto your universe (or sales) by a key match. Unlike a spatial join, this is a plain SQL-style left join on a column you specify on each side. Use it when one of your loaded `data.load.<id>` files is a small lookup table — code → description, code → category, zoning short-name → long-name — and you want to add one or more of its columns to every parcel that matches.

- **Activation** — presence of the `ref_tables` key with a non-empty `universe` or `sales` list. No `enabled` flag.
- **Source** — `_perform_ref_tables` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py); invoked from `_enrich_df_basic` for both the universe and sales dataframes.
- **When to use** — your assessor data uses coded values (1-character `LAND_CLASS`, numeric `TYPE_AND_USE`, jurisdiction-specific zoning codes) and you want a readable label or a higher-level category as a separate column, especially when the lookup adds **more than one field per code** (where a `calc` block's `map` operator would need one calc per added field). Also when the same lookup is shared between universe and sales.

#### Schema

```json
{
  "data": {
    "process": {
      "enrich": {
        "ref_tables": {
          "universe": [
            {
              "id": "ref_land_class",
              "key_ref_table": "land_class_code",
              "key_target": "land_class",
              "add_fields": ["land_class_desc", "land_class_category"]
            }
          ],
          "sales": []
        }
      }
    }
  }
}
```

The `universe` and `sales` keys are **lists** of ref-table entries — one entry per join you want to perform — applied to the universe and sales dataframes respectively. Either or both may be present; an empty/missing list means "no ref-table joins for that side." Each list element has these keys:

| Key | Required | Effect |
| --- | --- | --- |
| `id` | yes | The `data.load.<id>` key of the reference dataframe. The dataframe must already be loaded. |
| `key_ref_table` | yes | Column in the reference dataframe to match on. |
| `key_target` | yes | Column in the universe (or sales) dataframe to match on. May be the same as `key_ref_table` (in which case the join is on a single shared column name). |
| `add_fields` | yes | List of column names from the reference dataframe to add. Must be non-empty. None of these names may already exist in the target — this is enforced and raises a `ValueError`. |

#### Mechanics

1. The reference dataframe is loaded like any other through `data.load.<id>` — give it a `filename`, `dupes`, and `load` mapping. It does not need a `key` column; it only needs `key_ref_table` and the `add_fields`.
2. During basic enrichment, OpenAVMKit pulls just `[key_ref_table] + add_fields` from the reference frame and does a left merge onto the target on `key_target == key_ref_table`. When the two key column names match, a single-column merge is used; when they differ, the `key_ref_table` column is dropped from the result so only the renamed `key_target` remains.
3. Unmatched rows in the target keep `NaN` in the new `add_fields` columns. Decide a fill rule for them in `data.process.fill` if those fields will feed modeling.

#### Worked example

A jurisdiction's parcels file uses 1-character land class codes (`R`, `V`, `N`, `C`, …). You want both a readable description and a higher-level category for filtering and reporting. Build a small CSV at `in/ref_land_class.csv`:

```csv
land_class_code,land_class_desc,land_class_category
R,Residential <10ac homesite,Residential
V,Vacant,Vacant
N,Condo,Residential
C,Commercial,Commercial
```

Load it as a regular table and reference it:

```json
{
  "data": {
    "load": {
      "parcels": {
        "filename": "parcels.csv",
        "load": {
          "key": ["REID", "string"],
          "land_class": "Land_classification"
        }
      },
      "ref_land_class": {
        "filename": "ref_land_class.csv",
        "dupes": "auto",
        "load": {
          "land_class_code": "land_class_code",
          "land_class_desc": "land_class_desc",
          "land_class_category": "land_class_category"
        }
      }
    },
    "process": {
      "enrich": {
        "ref_tables": {
          "universe": [
            {
              "id": "ref_land_class",
              "key_ref_table": "land_class_code",
              "key_target": "land_class",
              "add_fields": ["land_class_desc", "land_class_category"]
            }
          ]
        }
      }
    }
  }
}
```

Every parcel now gets `land_class_desc` and `land_class_category` filled in from the matching ref row.

#### `ref_tables` vs `calc` `map`

These overlap: both can convert a code to a label. Pick by table size and number of added fields.

- **`calc` with `map`** ([calc_reference.md](calc_reference.md)) is best when the mapping is small and inline-readable, and you only need *one* derived column. Each `map` invocation adds one column; doing several requires several `calc` entries that all repeat the same mapping dictionary.
- **`ref_tables`** is best when the lookup is large enough that a CSV is more maintainable than inline JSON, when **multiple columns** come from the same lookup (e.g. both `_desc` and `_category` from the same code), or when the same mapping needs to apply to both `universe` and `sales`.

The two mechanisms are not mutually exclusive — use whichever fits per case in the same settings file.

#### Common errors

| Error | Cause |
| --- | --- |
| `ValueError: No 'id' / 'key_ref_table' / 'key_target' / 'add_fields' found in ref table.` | One of the four required entry keys is missing from a list element. |
| `ValueError: Ref table '<id>' not found in dataframes.` | The `id` doesn't match any loaded `data.load.<id>` entry. Check spelling and that the table is actually being loaded. |
| `ValueError: Key field '<col>' not found in ref table '<id>'.` | `key_ref_table` doesn't exist in the loaded reference dataframe. Likely you didn't map it through that table's `load` block. |
| `ValueError: Target field '<col>' not found in base dataframe` | `key_target` doesn't exist on the universe/sales after merge. Common cause: the field is created later by enrichment or `calc`, but `ref_tables` runs *first* during basic enrichment, so the column has to exist by load-and-merge time. |
| `ValueError: Field '<col>' already exists in base dataframe.` | One of `add_fields` collides with an existing column. Either remove the conflict from `add_fields` or rename it in the reference table's `load` block. |

---

## 5. Data cleaning & validation

### 5.1 Filling missing values — `data.process.fill.<method>`

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
3. **Per-model-group execution (universe only).** Universe fills are applied per model group, so a `mode` or `median` fill uses the model group's distribution rather than the global one — see `_fill_unknown_values_per_model_group` in [openavmkit/cleaning.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/cleaning.py). **Sales-side fills are restricted to sale-metadata fields** (sale price, sale date, sale conditions, etc.) and applied globally — characteristic blanks on sales rows are deliberate overlays on top of the universe and are left alone, so they don't go through per-model-group fills either. See `fill_unknown_values_sup` for the universe-vs-sales split.

### 5.2 `data.process.reconcile`

Post-merge reconciliation rules. After sales and universe data are merged, these rules let you resolve conflicts (e.g. when both sides have a value for the same field) by ID.

- **Default** — empty dict `{}`
- **Source** — `_merge_dict_of_dfs` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)
- **When to use** — your sales and universe both carry overlapping fields and you need deterministic precedence rules.

### 5.3 `data.process.invalid_sales.enabled`

Filter out non-arms-length sales after data processing, using the conditions defined in `data.process.invalid_sales.filter`.

- **Default** — `false`
- **Effect** — when `true`, sales matching the filter are excluded. When `false`, the step is skipped silently.
- **Source** — `filter_invalid_sales` in [openavmkit/cleaning.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/cleaning.py)
- **When to use** — if you have a set of sales you know are invalid and can exclude by rule, that aren't covered by your existing sales validity codes

---

## 6. Modeling control

> For the **full catalog of model engines** (XGBoost, LightGBM, CatBoost, MRA, GWR, kernel, SLICE, baselines, etc.), the **model-name-vs-engine dispatch** mechanism, and how to run **multiple variants of the same engine** (e.g. two XGBoost configurations side-by-side), see **[Models reference](models_reference.md)**. The settings on this page are the orchestration layer; that page documents each model.

### `modeling.instructions.<main|vacant|hedonic>.run`

Explicit list of model names to run for the main, vacant, or hedonic stages. Without it, all models defined under `modeling.models.<main|vacant|hedonic>` are run.

- **Source** — `_run_models` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — you want a fast iteration on a single model, or you want to skip slow models (e.g. `gwr`) for a quick run.

### `modeling.instructions.<main|vacant|hedonic>.skip.<model_group>`

Per-model-group skip list. For the named model group, the listed models are skipped even if they're in `run`.

- **Source** — `_run_models` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — a particular model is unstable on a particular model group (low sample count, rank-deficient features) and you want to exclude it from that group only.

### Train / test split rules

The canonical train/test split lives in `_perform_canonical_split` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py). It splits each model group's valid sales into a test set (default 20%) and a training set (default 80%), maintaining vacant/improved balance and respecting three constraints:

1. **No leakage.** Post-valuation-date sales never appear in the training set.
2. **Sufficient lookback representation in test, without overrepresentation.** The lookback period (sales within `analysis.ratio_study.look_back_years` of the valuation date) gets a hard floor in the test set so the resulting ratio study has a defensible IAAO-aligned sample size, and a cap that prevents the lookback period from dominating the test set when other years are available.
3. **Stratified random sampling** within each tier. Vacant sales are stratified by `sale_year` only; improved sales are stratified by age, finished area, and `sale_year` (user-configurable). Stratification uses `sklearn.model_selection.train_test_split` with graceful fallback when strata are too thin.

#### `modeling.instructions.test_train_frac`

Fraction of total sales that go to **training** (the test set is the complement).

- **Default** — `0.8` (80% train, 20% test)
- **Source** — `_write_canonical_splits` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)

#### `modeling.instructions.test_lookback_cap_ratio`

The lookback period's test-share is capped at this multiple of the non-lookback test-share. Prevents the test set from being dominated by lookback sales when there are also older sales available to draw from.

- **Default** — `2.0` (lookback can be at most 2× overrepresented in test relative to the rest)
- **Set to `null`** to disable the cap.
- **Worked example** — with `test_count=53`, `lookback_size=75`, `non_lookback_size=188` (the Petersburg single-family-suburban shape), the cap allows at most `2 × 53 × 75 / (188 + 150) = 23` lookback sales in test. Training keeps the remaining 52 lookback sales — substantially more recent training signal than the legacy "fill test first" approach would give.
- **Disabled when there is no non-lookback** — if all sales are inside the lookback window there's nothing to overrepresent against, so the cap is silently disabled and the function falls back to filling the test set from lookback.
- **Source** — `compute_lookback_test_size` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)

#### `modeling.instructions.test_lookback_floor`

Hard floor: never put fewer than this many lookback sales in the test set (subject to availability — if lookback only has 10 sales, the floor is silently 10). When the cap above would push the lookback count below the floor, the cap is overridden — the floor is treated as a hard requirement for a usable ratio-study sample, and a slight overrepresentation is preferred to a too-thin holdout.

- **Default** — `15` (rule-of-thumb minimum for IAAO-style COD confidence intervals)
- **Set to `null`** to disable the floor.

The floor doesn't act as a ceiling: when the cap allows more lookback than the floor (the common case), the function takes as much lookback as cap and availability allow — well above the floor.

#### `modeling.instructions.test_strat_fields_improved`

List of fields to stratify improved-sales splits by. `sale_year` is always appended automatically. Numeric continuous fields are quantile-binned to 4 strata before being used as labels.

- **Default** — `null`, which auto-resolves to:
  - `bldg_effective_age_years` if the column exists, otherwise `bldg_age_years`
  - `bldg_area_finished_<unit>` where `<unit>` is `sqft` or `sqm` per the locality's `units` setting
- **User override example** —

  ```json
  "test_strat_fields_improved": ["neighborhood", "bldg_quality_num"]
  ```

  The defaults are *not* added when an explicit list is provided. `sale_year` is still appended.

- **Source** — `_resolve_strat_fields_improved` in [openavmkit/data.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/data.py)

#### Disabling the rule entirely

To revert to the pre-rule behavior ("test set is filled from the lookback period first, then from older sales") for a single jurisdiction, set both knobs to `null`:

```json
"modeling": {
  "instructions": {
    "test_lookback_floor": null,
    "test_lookback_cap_ratio": null
  }
}
```

This is rarely useful in production but is the right setting for synthetic-data tests where there are no non-lookback sales to draw from.

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

### `modeling.instructions.<main|vacant|hedonic>.ensemble`

Configure how individual model predictions get combined into an `ensemble` prediction. Two types are supported.

#### `type: "default"` — global greedy ensemble

```json
"main": {
    "run": ["mra", "multi_mra", "local_area", "lightgbm"],
    "ensemble": { "type": "default" }
}
```

Runs a greedy backward-elimination over the candidate models: starts with all of them combined via per-row **median**, then drops the model whose removal *improves* (lowers) the test MAPE, and repeats until removing any further model would only hurt. The surviving subset is combined element-wise via median (not mean — median is robust to a single model going wild on a particular parcel) and the result is named `ensemble`. Useful when one or two models are weakening the combination and you want them auto-pruned.

- **Optional** `models` — explicit candidate list. Defaults to all models that ran except `assessor` and `ground_truth`.
- **Source** — `_perform_default_ensemble` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)

#### `type: "local"` — best-model-per-location

```json
"main": {
    "run": ["mra", "multi_mra", "local_area", "lightgbm"],
    "ensemble": {
        "type": "local",
        "locations": ["neighborhood_filled", "city"]
    }
}
```

For each unique value of each location field, finds the single best-MAPE model on the **training** set and assigns that model's prediction to parcels in that location. When a parcel matches multiple location levels (e.g. a specific neighborhood AND a city), the more specific match wins. Parcels whose location values weren't seen in training fall back to a global best-model pick.

This is **not averaging** — at each parcel, exactly one model's prediction is used. The choice varies across the locality.

- **`locations`** — list of categorical fields to partition by, ordered specific → general (the painter walks the list and the *most specific* match wins). If omitted, falls back to `field_classification.important.locations`.
- **Only valid for `main`** — vacant and hedonic stages only support `default`.
- **Source** — `_perform_local_ensemble` and `_run_local_ensemble_test_and_paint` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — different sub-markets favor different models (e.g. tree-based dominates dense urban neighborhoods where it has plenty of sales, but multi-MRA wins in rural areas where signals are sparser). Local ensemble lets each neighborhood pick its own winner. Avoid when (a) you have very few sales per location (many locations will pick a model based on noise), or (b) you want a single coherent global prediction for explainability.
- **Pairs naturally with** — model engines that themselves vary by location (`multi_mra`, `local_area`, `gwr`), since they often dominate in different parts of the locality.

### `modeling.try_variables.variables`

Run a dedicated variable-importance experiment over a custom set of candidate variables before main modeling. Surfaced via [openavmkit.pipeline.try_variables](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/pipeline.py).

- **Source** — `try_variables` in [openavmkit/benchmark.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/benchmark.py)
- **When to use** — you have hypotheses about which variables matter and want a slower, more thorough comparison than the auto-reduction step does inline.

---

## 7. Analysis & QA

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
- **Diagnostic** — if your **untrimmed COD is several multiples of your trimmed COD** (5×–20×), you have a small number of extreme sale-vs-prediction mismatches dragging the means. Don't just raise `max_percent` to silence them — that hides bad data and can mask sales-chasing in the assessor baseline. See [tutorial.md → "When untrimmed COD is much worse than trimmed COD"](tutorial.md#when-untrimmed-cod-is-much-worse-than-trimmed-cod) for the full diagnostic flow including the sales-chasing sub-check.

### `analysis.ratio_study.look_back_years`

How many years before the valuation date to include sales from when running the ratio study.

- **Default** — `1` (from template)
- **Source** — `get_look_back_dates` in [openavmkit/utilities/settings.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/utilities/settings.py)
- **When to use** — your jurisdiction expects a multi-year ratio-study window, or you want to widen the sample for low-volume model groups.

---

## 8. Caching & checkpoints

### 8.1 Three cache layers

OpenAVMKit caches expensive intermediate results in three places:

- **Notebook checkpoints** at `<locality>/out/checkpoints/` — per-notebook intermediate state, written by every `from_checkpoint(...)` call.
- **Enrichment cache** at `<locality>/cache/` — used internally by enrichment steps that pull from remote sources (OpenStreetMap, Census, Overture) or do heavy local computation (street networks, distance calculations).
- **Saved model parameters** at `<locality>/out/models/<model_group>/.../` — tuned hyperparameters and bandwidths from previous model runs (XGBoost / LightGBM / CatBoost Optuna results, GWR bandwidth, kernel regression bandwidth).

The first two layers are *designed* to self-invalidate when the relevant inputs or settings change. The third layer (saved model parameters) has different semantics — see § 8.4 below.

### 8.2 The notebook checkpoint system

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

- `from_checkpoint(path, func, params, use_checkpoint=True)` — load if cached, otherwise run and save. Pass `use_checkpoint=False` to bypass the cache (always re-run and overwrite the saved result).
- `delete_checkpoints("<prefix>")` — delete all checkpoints starting with the given prefix (e.g. `delete_checkpoints("1-assemble")` clears all of notebook 1's intermediate state)
- `clear_checkpoints = True` at the top of a notebook — convention used in the pipeline notebooks; combined with a `delete_checkpoints("<this_notebook>")` call, gives you a one-flag "start fresh" toggle

**Source** — [openavmkit/checkpoint.py](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/checkpoint.py).

### 8.3 The enrichment cache

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

### 8.4 Saved model parameters — different semantics

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

### 8.5 When self-invalidation isn't enough

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
| Delete `<outpath>/<slug>_params.json`, `<model_name>_bw.json`, or `kernel_bw.pkl` | Force a fresh hyperparameter search on next model run (see § 8.4) |

**Don't nuke prophylactically.** OSM streets and Overture pulls are expensive on a fresh cache — clearing them every run will cost you hours over a project. Hyperparameter searches can also be expensive (Optuna with 100+ trials on a tree-based model can take a long time). Nuke when something feels off, then re-run.

---

## See also

- [Models reference](models_reference.md) — full catalog of model engines, invocation patterns, multi-variant runs
- [The `calc` expression language](calc_reference.md) — the full operator reference for `calc` blocks (used in `data.load.<id>.calc`, `data.process.enrich.calc`, etc.)
- [Configuration](config.md) — `.env`, cloud storage, PDF generation
- [The Basics](the_basics.md) — locality folder structure, terminology
- [Recipe](recipe.md) — public function reference and notebook map
- Real settings examples: `notebooks/pipeline/data/<locality>/in/settings.json`
- The settings template: [openavmkit/resources/settings/settings.template.json](https://github.com/landeconomics/openavmkit/blob/master/openavmkit/resources/settings/settings.template.json)
