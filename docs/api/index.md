# API Reference

Auto-generated from the docstrings in the `openavmkit` package. Every module that
defines at least one public function or class gets a page here.

New to the library? Start with [Getting Started](../docs/getting_started.md) and
[The Basics](../docs/the_basics.md) — this section is a reference, not a tutorial.

## Core

The main pipeline and analysis modules.

| Module | What it covers |
| --- | --- |
| [`pipeline`](Core/pipeline.md) | End-to-end orchestration of a run |
| [`data`](Core/data.md) | Loading, merging, and the `SalesUniversePair` |
| [`cleaning`](Core/cleaning.md) | Fill rules and data validation |
| [`filters`](Core/filters.md) | The filter DSL |
| [`calculations`](Core/calculations.md) | The `calc` expression language |
| [`inference`](Core/inference.md) | Spatial inference |
| [`condos`](Core/condos.md) | Condo modeling pathway |
| [`modeling`](Core/modeling.md) | Model primitives and `DataSplit` |
| [`model_runner`](Core/model_runner.md) | Running and comparing model groups |
| [`projection`](Core/projection.md) | Projections |
| [`time_adjustment`](Core/time_adjustment.md) | Sale-date time adjustment and market indices |
| [`income`](Core/income.md) | Income approach |
| [`ratio_study`](Core/ratio_study.md) | IAAO ratio studies |
| [`horizontal_equity_study`](Core/horizontal_equity_study.md) | Horizontal equity |
| [`vertical_equity_study`](Core/vertical_equity_study.md) | Vertical equity / VEI |
| [`sales_scrutiny_study`](Core/sales_scrutiny_study.md) | Anomalous-sale detection |
| [`sales_chasing`](Core/sales_chasing.md) | Sales-chasing detection |
| [`shap_analysis`](Core/shap_analysis.md) | SHAP contributions |
| [`area_stats`](Core/area_stats.md) | Area-level statistics |
| [`reports`](Core/reports.md) | PDF and HTML report generation |
| [`checkpoint`](Core/checkpoint.md) | Caching and checkpoints |

## Cloud

Remote storage backends: [`cloud`](cloud/cloud.md), [`base`](cloud/base.md),
[`azure`](cloud/azure.md), [`huggingface`](cloud/huggingface.md), [`sftp`](cloud/sftp.md).

## Synthetic

Synthetic data generation for testing and demos:
[`synthetic`](synthetic/synthetic.md), [`basic`](synthetic/basic.md),
[`city`](synthetic/city.md), [`generate`](synthetic/generate.md).

## Utilities

Supporting helpers — geometry, enrichment sources, formatting, and stats:
[`geometry`](utilities/geometry.md), [`census`](utilities/census.md),
[`openstreetmap`](utilities/openstreetmap.md), [`overture`](utilities/overture.md),
[`dem`](utilities/dem.md), [`clustering`](utilities/clustering.md),
[`somers`](utilities/somers.md), [`stats`](utilities/stats.md),
[`settings`](utilities/settings.md), [`format`](utilities/format.md),
[`plotting`](utilities/plotting.md), [`excel`](utilities/excel.md),
[`cache`](utilities/cache.md), [`timing`](utilities/timing.md),
[`assertions`](utilities/assertions.md), [`data`](utilities/data.md),
[`modeling`](utilities/modeling.md).
