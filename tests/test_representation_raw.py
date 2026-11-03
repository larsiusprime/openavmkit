"""calc_representation keeps the shares as numbers beside the formatted table, for a front end to chart."""

import pandas as pd

from openavmkit.utilities.stats import calc_representation


def test_raw_shares_are_numbers():
    sales = pd.DataFrame({"pool": [1, 0, 0, 0], "area": [1000.0, 1200.0, None, 900.0]})
    univ = pd.DataFrame({"pool": [0, 0, 0, 0], "area": [1000.0, 0.0, 800.0, 900.0]})
    out = calc_representation(sales, univ)
    raw = out["raw"].set_index("name")
    assert raw.loc["area", "sales_pct_gt0"] == 0.75 and raw.loc["area", "univ_pct_gt0"] == 0.75
    assert raw.loc["pool", "univ_pct_gt0"] == 0.0 and out["bad_vars"] == ["pool"]
    assert isinstance(out["df"].iloc[0]["sales_pct_gt0"], str)  # the display table is unchanged
