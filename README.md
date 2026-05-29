# OpenAVMKit

[![PyPI version](https://img.shields.io/pypi/v/openavmkit?color=blue)](https://pypi.org/project/openavmkit/)
![Python versions](https://img.shields.io/pypi/pyversions/openavmkit.svg)
[![CI](https://github.com/larsiusprime/openavmkit/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/larsiusprime/openavmkit/actions/workflows/ci.yml)

OpenAVMKit is a python library for real estate mass appraisal. It includes modules for data cleaning, data enrichment, modeling, and statistical evaluation of predictive models. It also includes Jupyter notebooks that model typical workflows.

See the [Changelog](changelog.md)

# Table of Contents

## 1. Getting started
  - [Install from PyPI](docs/docs/getting_started.md#option-1---install-from-pypi)
  - [Install from Git](docs/docs/getting_started.md#option-2---install-from-git)
  - [Running Tests](docs/docs/getting_started.md#running-tests)
  - [Smoke test with sample data](docs/docs/getting_started.md#smoke-test-with-sample-data)
## 2. Build a jurisdiction from scratch (tutorial)
  - [Smoke test with sample data](docs/docs/tutorial.md#part-a-smoke-test-with-sample-data)
  - [Onboard your own jurisdiction](docs/docs/tutorial.md#part-b-onboard-your-own-jurisdiction)
## 3. The basics
  - [Creating a new locality](docs/docs/the_basics.md#creating-a-new-locality)
  - [Code modules](docs/docs/the_basics.md#code-modules)
  - [Jupyter Notebooks](docs/docs/the_basics.md#using-the-jupyter-notebooks)
  - [Terminology](docs/docs/the_basics.md#terminology)
## 4. Configuration
  - [Cloud storage](docs/docs/config.md#configuring-cloud-storage)
  - [PDF report generation](docs/docs/config.md#configuring-pdf-report-generation)
  - [US Census API](docs/docs/config.md#configuring-census-api-access)
  - [Open Street Map API](docs/docs/config.md#configuring-openstreetmap-enrichment)
## 5. Advanced settings reference
  - [The settings.json preprocessor](docs/docs/advanced_settings.md#1-the-settingsjson-preprocessor) — comments, `$$` references, template merging, `!`/`+` flags
  - [Data load: `data.load.<id>`](docs/docs/advanced_settings.md#2-data-load-dataloadid) — `dupes` rules and per-field aggregation across duplicate rows
  - [Time adjustment overrides](docs/docs/advanced_settings.md#3-time-adjustment)
  - [Data enrichment toggles](docs/docs/advanced_settings.md#4-data-enrichment) — basic geo, spatial joins, Overture, Census, distance & proximity, OSM streets, spatial lag, spatial inference
  - [Data cleaning & validation](docs/docs/advanced_settings.md#5-data-cleaning-validation) — full fill-method reference
  - [Modeling control](docs/docs/advanced_settings.md#6-modeling-control)
  - [Analysis & QA](docs/docs/advanced_settings.md#7-analysis-qa)
  - [Caching & checkpoints](docs/docs/advanced_settings.md#8-caching-checkpoints) — when to nuke the cache
  - [The `calc` expression language](docs/docs/calc_reference.md) — full operator reference for derived columns and filters
  - [Models reference](docs/docs/models_reference.md) — every model engine, name-vs-engine dispatch, multi-variant runs
## 6. Notebooks & API
  - [Pipeline notebooks workflow](notebooks/README.md) — which notebook to run when
  - [Recipe — public functions](docs/docs/recipe.md)
  - [API reference](docs/docs/api/) (auto-generated from docstrings)
## 7. For contributors & coding agents
  - [AGENTS.md](AGENTS.md) — repo conventions, settings preprocessor, gotchas, extending patterns
  - [CONTRIBUTING.md](CONTRIBUTING.md) — PR workflow, style guide
## 8. Legal
  - [License](LICENSE)
  - [License Philosophy](LICENSE-PHILOSOPHY.md)