# Notebooks

> **For a full end-to-end walkthrough** — from raw data to a working AVM — see the [tutorial](../docs/docs/tutorial.md). This README is a quick reference for run order and per-notebook intent.

This directory contains the Jupyter notebooks that drive an OpenAVMKit workflow. There are three subfolders:

- **[pipeline/](pipeline/)** — the standard end-to-end workflow. Most users start here.
- **[examples/](examples/)** — meant to eventually demonstrate specific features in a standalone fashion; experimental, not part of the standard workflow.
- **[research/](research/)** — exploratory notebooks used during research; not part of the standard workflow.

## The pipeline workflow

The pipeline notebooks are designed to be run in order against a single locality. Each notebook starts from the previous notebook's output via the checkpoint system, so you can stop and resume between stages.

Set up a locality folder under `pipeline/data/<slug>/` first — see [The Basics → Creating a new locality](../docs/docs/the_basics.md#creating-a-new-locality).

### Run order

| Order | Notebook | What it does | Input | Output |
| ----- | -------- | ------------ | ----- | ------ |
| 1 | [`01-assemble.ipynb`](pipeline/01-assemble.ipynb) | Loads raw tabular and geospatial files, joins sales to parcels, performs basic enrichment (lat/lon, GIS area, optional census/streets/distances), and tags model groups. | `data/<slug>/in/` (raw files + `settings.json`) | A `SalesUniversePair` of *factual* assertions about parcels and sales. |
| 2 | [`02-clean.ipynb`](pipeline/02-clean.ipynb) | Fills missing values, computes equity clusters, processes sales validity, runs a sales scrutiny heuristic, and produces time-adjusted sale prices. | Output of `01-assemble`. | A `SalesUniversePair` ready for modeling. Adds reasoned assumptions on top of the raw facts. |
| 3 | [`03-model.ipynb`](pipeline/03-model.ipynb) | Trains predictive models per model group (MRA, GWR, XGBoost, LightGBM, CatBoost, ensembles, etc.), runs analysis on predictions, and produces statistical reports. | Output of `02-clean`. | Per-model-group predictions, params/contribs CSVs, and reports under `data/<slug>/out/`. |
| 4 | [`04-land.ipynb`](pipeline/04-land.ipynb) | Produces a per-parcel **land** value layer separately from full market value. Curates a witness pool (W1-W6), paints zoning-anchored per-cell tables, and scores the result on the L1-L7 Lars-Tests (LVT-incentive checks plus an improvement-cost-table consistency check). Locality-specific cascade splits and adjustment factors are picked up automatically from `data/<slug>/in/land_config.py` if present. | Output of `02-clean` plus an ensemble prediction from `03-model`. | Per-parcel land values, per-cell tables, evidence packets, and Lars-Tests report under `data/<slug>/out/land/`. |
| 5 | [`assessment_quality.ipynb`](pipeline/assessment_quality.ipynb) | Standalone quality checks on already-prepared data: ratio studies, equity studies, etc. Does not retrain models — analyzes existing predictions (yours or the assessor's). | Output of `01-assemble` (basic checks) or `02-clean` (full checks including horizontal equity). | Quality reports. |

### Why the workflow is split into stages

Each stage has a distinct character, and keeping them separate keeps your work auditable:

- **Assemble** establishes *facts*: where things are, which parcels have what features, what sold for how much. The output should reflect ground truth as faithfully as possible.
- **Clean** introduces *opinions*: how to fill missing values, which sales to trust, how to time-adjust. Reasonable people may disagree on these choices, so the cleaned output is kept in its own checkpoint.
- **Model** introduces *predictions*: educated guesses derived from the facts and assumptions established above.
- **Land** uses the model's predictions plus a curated witness pool to derive a separate land-value layer; output is structured around LVT-incentive-aligned tests rather than IAAO ratio studies.
- **Assessment quality** evaluates predictions (yours or others') against observed sales.

## Tips

- **Strip notebook outputs before committing.** `03-model.ipynb` in particular accumulates large outputs. Run `jupyter nbconvert --clear-output --inplace notebooks/pipeline/03-model.ipynb` before staging.
- **Use the checkpoint system.** The pipeline notebooks save intermediate state via `from_checkpoint(...)`. Set `clear_checkpoints = True` at the top of a notebook to start fresh; leave it `False` to resume from the last good state.
- **The notebooks call only `openavmkit.pipeline`.** If you find yourself reaching into another module from a notebook cell, that's a sign a `pipeline.py` wrapper is missing — see [AGENTS.md](../AGENTS.md) for the convention.

## Related docs

- [Getting started](../docs/docs/getting_started.md) — installation
- [The basics](../docs/docs/the_basics.md) — locality structure, terminology
- [Recipe](../docs/docs/recipe.md) — public function reference, organized by pipeline stage
- [Advanced settings reference](../docs/docs/advanced_settings.md) — settings.json preprocessor and high-impact keys
