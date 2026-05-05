# Saint-Quentin fixture

This directory contains the lightweight OpenAVMKit settings fixture for the
Saint-Quentin, Aisne, France metric jurisdiction.

The raw and generated data are intentionally not committed. Regeneration writes
to the ignored working directory:

```text
notebooks/pipeline/data/fr-02-saint_quentin/
```

Public data sources:

- Cadastre parcels and buildings: `https://services1.arcgis.com/5nIW6mZeb2YNJ7np/ArcGIS/rest/services/SIG_CADASTRE/FeatureServer`
- DVF sales data API: `https://www.data.gouv.fr/api/1/datasets/demandes-de-valeurs-foncieres/`

Local full preparation command:

```bash
python scripts/prepare_saint_quentin.py --years 2021 2022 2023 2024 2025
```

The fixture uses metric units, monthly time adjustment, a single `all` model
group, and main/vacant `naive_area`, `local_area`, and `lightgbm` runs. It
exercises the European metric data path without requiring large raw files in
git.
