"""Sales scrutiny's multi-parcel heuristic: sales sharing a deed ID and a sale date."""

import os

import pandas as pd

from openavmkit.data import SalesUniversePair
from openavmkit.sales_scrutiny_study import run_heuristics


def test_blank_deed_ids_are_not_one_deed(tmp_path):
    os.chdir(tmp_path)
    universe = pd.DataFrame({"key": [f"p{i}" for i in range(6)], "bldg_year_built": [1990] * 6})
    sales = pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(6)],
            "key_sale": [f"s{i}" for i in range(6)],
            "sale_date": pd.to_datetime(["2020-01-02"] * 6),
            "sale_year": [2020] * 6,
            "sale_price": [100_000, 200_000, 300_000, 400_000, 500_000, 600_000],
            "valid_sale": [True] * 6,
            "vacant_sale": [False] * 6,
            # Two sales on one deed; three with no deed at all; one on a deed of its own.
            "deed_id": ["D1", "D1", None, None, "", "D2"],
        }
    )
    settings = {"analysis": {"sales_scrutiny": {"deed_id": "deed_id"}}, "locality": {"units": "imperial"}}
    out = run_heuristics(SalesUniversePair(sales=sales, universe=universe), settings, drop=True)
    # The two on one deed are dropped as a sale of several parcels; the blank ones stay.
    assert sorted(out.sales["key_sale"]) == ["s2", "s3", "s4", "s5"]
