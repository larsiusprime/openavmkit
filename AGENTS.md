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
- **`data.process.invalid_sales` defaults to off.** See `filter_invalid_sales` in [openavmkit/cleaning.py](openavmkit/cleaning.py) — it early-returns when `enabled` is not set. Set `data.process.invalid_sales.enabled = true` to actually run validation checks.
- **Time adjustment can be wholly overridden.** `data.process.time_adjustment.from_file.<model_group>` (see `read_time_adjustment_from_file` in [openavmkit/time_adjustment.py](openavmkit/time_adjustment.py)) replaces the built-in engine for that model group with a CSV. Check this before debugging unexpected time-adjustment output.
- **Notebooks accumulate large outputs.** `notebooks/pipeline/03-model.ipynb` is over 1 MB. Strip outputs (`jupyter nbconvert --clear-output --inplace …`) before committing.
- **Caching is designed to self-invalidate but isn't perfect.** If you change `settings.json` (or any other input) and the next run looks suspiciously like the last one — or if enrichment output is missing fields it should have — **nuke the cache.** Use `delete_checkpoints("<prefix>")` for notebook checkpoints, or delete the locality's `cache/` folder for the enrichment cache. See [docs/docs/advanced_settings.md § 8](docs/docs/advanced_settings.md#8-caching-checkpoints) for full detail. **A specific, loud failure mode:** when new settings *add columns* to the universe (a new `load` mapping, a new `calc`, a new ref-table field), `write_cached_df` in [openavmkit/utilities/cache.py](openavmkit/utilities/cache.py) **raises** `ValueError: Cached DataFrame does not match the original DataFrame` mid-enrichment rather than silently recomputing — the stored `.cols.parquet` diff can't reconstruct the new schema. Fix is the same: delete `cache/` and let enrichment regenerate.
- **Saved model parameters are a separate cache layer with different semantics.** Tunable models (XGBoost / LightGBM / CatBoost via Optuna; GWR / kernel regression via bandwidth search) save their tuned hyperparameters under `<locality>/out/models/<model_group>/.../` as `<slug>_params.json`, `<model_name>_bw.json`, or `kernel_bw.pkl`. **Deleting these forces a fresh hyperparameter search** on the next run. **Keeping them skips the search** but constrains the model to the previous run's tuning even though the model still re-fits on current training data. If your training data has meaningfully shifted (different sales window, different features, different model groups), delete the saved params so the next run re-tunes. See [advanced_settings.md § 8.4](docs/docs/advanced_settings.md#84-saved-model-parameters-different-semantics).
- **Never `collapse_sparse_categories` a *location* field in place.** Collapsing rare values into an `"Other"` bucket is fine for a generic model feature, but location fields (anything in `field_classification.important.locations` / `.fields.loc_*`, `analysis.*.location`, a ratio-study `<loc_*>` breakdown, a model/ensemble `locations` list, or `land.lycd.*.location`) are assumed geographically coherent — equity clustering, ratio-study breakdowns, local-ensemble selection, and sales-scrutiny clusters all *group by* them. Collapsing in place merges unrelated zones into one `"Other"` bucket and corrupts those analyses. Instead set `output_field` on the collapse config to write a `<field>_collapsed` modeling variant, classify *that* categorical, and use it only as a model feature; leave the raw location intact. The code warns loudly (config-based, via `get_location_fields` / `get_collapsed_fields` / `warn_if_location_collapsed` in [openavmkit/utilities/settings.py](openavmkit/utilities/settings.py)) both at collapse time and at each grouping site; `"strict": true` in the collapse block escalates to a hard error. See [advanced_settings.md § 5.4](docs/docs/advanced_settings.md).

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
2. Wire it into the dispatch in [openavmkit/benchmark.py](openavmkit/benchmark.py) (search for the existing tree-based model handling).
3. Add a config entry under `modeling.models.<main|vacant>` in the template.
4. Add a public wrapper to `pipeline.py` if it should be runnable directly from a notebook.
5. **Wire up params and contribs.** Every model must produce two outputs per subset (`test`, `sales`, `universe`):
   - **`params_<subset>.csv`** — per-feature **parameters**. For linear models these are regression coefficients; for tree-based models they are SHAP values normalized by value size. Conceptually: "what is each feature's per-unit effect on the prediction?"
   - **`contributions_<subset>.csv`** — per-feature **contributions**. For linear models these are coefficients × the feature's value for that row; for tree-based models they are raw SHAP contributions. Conceptually: "how much did each feature actually contribute to this row's prediction?"

   Existing implementations to follow: [`write_mra_params`](openavmkit/modeling.py) (linear), [`write_tree_based_params`](openavmkit/modeling.py) (SHAP), [`write_gwr_params`](openavmkit/modeling.py) (per-location coefficients), [`write_local_area_params`](openavmkit/modeling.py), [`write_multi_mra_params`](openavmkit/modeling.py). Pick the one that most closely matches your model's structure and follow the same file-output contract. The dispatch that calls these per-model writers is the common interface; if your model is fundamentally new in shape, add a new writer in `modeling.py` and a dispatch case alongside the existing ones.

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

## 9. Living document footer

Sections worth growing over time:

- **Section 4 (gotchas)** — when you hit one and figure it out, add it here.
- **Section 7 (extending patterns)** — when you add a new kind of thing (a new study type, a new data source), describe the recipe so the next agent can copy it.

Keep entries short. Link to the source line that proves the claim. If a claim becomes stale because the code changed, fix or remove the entry instead of leaving it.
