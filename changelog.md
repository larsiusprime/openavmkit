# Changelog
All notable changes to this project will be documented in this file.

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