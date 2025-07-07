import warnings

import pandas as pd

from openavmkit.utilities.data import div_series_z_safe


def check_land_values(df_in: pd.DataFrame, model_group: str) -> pd.DataFrame:
    """Perform various sanity checks on land values

    Parameters
    ----------
    df_in : pd.DataFrame
        The dataframe you want to check
    model_group: str
        The model group you want to check

    Returns
    -------
    pd.DataFrame
        A copy of the original dataframe, with any necessary amendments to land value
    """

    df = df_in.copy()

    # Perform basic sanity checks / error correction to land values

    # Here are all the checks we will perform:
    # 1. land value is not negative
    # 2. land value is not greater than market value
    # 3. if a building exists:
    #    3.1. land allocation is not 1.0
    # 4. if no building exists:
    #    4.1. land allocation is 1.0

    counts = {
        "market_lt_land": 0,
        "negative_market": 0,
        "negative_land": 0,
        "negative_impr": 0,
        "land_gt_market": 0,
        "land_gt_market_vacant": 0,
        "land_gt_market_improved": 0,
        "bldg_yes_land_alloc_ge_1": 0,
        "bldg_no_land_alloc_ne_1": 0,
    }

    labels = {
        "market_lt_land": "Market value less than land value",
        "negative_market": "Negative market value",
        "negative_land": "Negative land value",
        "negative_impr": "Negative improvement value",
        "land_gt_market": "Land value greater than market value",
        "land_gt_market_vacant": "Land value greater than market value (vacant)",
        "land_gt_market_improved": "Land value greater than market value (improved)",
        "bldg_yes_land_alloc_ge_1": "Building exists but land allocation is 1.0",
        "bldg_no_land_alloc_ne_1": "No building exists but land allocation not 1.0",
    }

    _perform_land_checks(df, counts, do_remedy=True)

    #######################

    n = len(df)

    if any(counts.values()):
        warnings.warn(f"Land value sanity check failed for model group {model_group}.")
        for key, value in counts.items():
            if value:
                label = labels[key]
                perc = value / n
                print(f"  {perc:6.2%} -- {label}: {value}/{n} rows")

    # Derive the final improvement values and make sure everything is consistent
    df.loc[df["model_land_value"].lt(0), "model_land_value"] = 0.0

    df["model_land_alloc"] = div_series_z_safe(
        df["model_land_value"], df["model_market_value"]
    )
    df["model_impr_value"] = df["model_market_value"] - df["model_land_value"]
    df["model_impr_alloc"] = div_series_z_safe(
        df["model_impr_value"], df["model_market_value"]
    )

    derived_land_alloc = div_series_z_safe(
        df["model_land_value"], df["model_market_value"]
    )
    assert derived_land_alloc.equals(df["model_land_alloc"])

    # Re-count all the checks

    _perform_land_checks(df, counts, do_remedy=True)
    _perform_land_checks(df, counts, do_remedy=False)

    print("")
    if any(counts.values()):
        warnings.warn(
            f"Remaining issues after error correction for model group {model_group}:"
        )
        for key, value in counts.items():
            if value:
                label = labels[key]
                perc = value / n
                print(f"  {perc:6.2%} -- {label}: {value}/{n} rows")
    else:
        print(f"No issues after error correction {model_group}.")

    return df


def _perform_land_checks(df: pd.DataFrame, counts: dict, do_remedy: bool) -> dict:

    # Check 0: market values must be >= 0:
    idx_negative_market = df["model_market_value"].lt(0)
    counts["negative_market"] = idx_negative_market.sum()
    if do_remedy:
        # Remedy 0: set negative market values to 0
        df.loc[idx_negative_market, "model_market_value"] = 0.0
        df.loc[idx_negative_market, "model_land_value"] = 0.0
        df.loc[idx_negative_market, "model_land_alloc"] = 1.0
        df.loc[idx_negative_market, "model_impr_value"] = 0.0

    # Check 1: land values must be >= 0
    idx_negative_land = df["model_land_value"].lt(0)
    counts["negative_land"] = idx_negative_land.sum()
    if do_remedy:
        # Remedy 1: set negative land values to 0
        df.loc[idx_negative_land, "model_land_value"] = 0.0
        df.loc[idx_negative_land, "model_land_alloc"] = 0.0
        df.loc[idx_negative_land, "model_impr_value"] = df.loc[
            idx_negative_land, "model_market_value"
        ]

    # Check 2: market value < land value:
    idx_market_lt_land = df["model_market_value"].lt(df["model_land_value"])
    counts["market_lt_land"] = idx_market_lt_land.sum()
    if do_remedy:
        # Remedy 2: set land value to market value
        df.loc[idx_market_lt_land, "model_land_value"] = df.loc[
            idx_market_lt_land, "model_market_value"
        ]
        df.loc[idx_market_lt_land, "model_land_alloc"] = 1.0
        df.loc[idx_market_lt_land, "model_impr_value"] = 0.0

    # Check 3: land value must not be greater than market value
    idx_land_gt_market = df["model_land_value"].gt(df["model_market_value"])
    idx_land_gt_market_vacant = df["model_land_value"].gt(
        df["model_market_value"]
    ) & df["bldg_area_finished_sqft"].eq(0)
    idx_land_gt_market_improved = df["model_land_value"].gt(
        df["model_market_value"]
    ) & df["bldg_area_finished_sqft"].ge(1)
    counts["land_gt_market"] = idx_land_gt_market.sum()
    counts["land_gt_market_vacant"] = idx_land_gt_market_vacant.sum()
    counts["land_gt_market_improved"] = idx_land_gt_market_improved.sum()
    if do_remedy:
        # Remedy 3: set land values greater than market value to market value
        df.loc[idx_land_gt_market, "model_land_value"] = df.loc[
            idx_land_gt_market, "model_market_value"
        ]
        df.loc[idx_land_gt_market, "model_land_alloc"] = 1.0
        df.loc[idx_land_gt_market, "model_impr_value"] = 0.0

    # Check 4: If a building exists...
    # land allocation must not be 1.0:
    idx_bldg_yes_land_alloc_ge_1 = df["bldg_area_finished_sqft"].ge(1) & df[
        "model_land_alloc"
    ].ge(1)
    counts["bldg_yes_land_alloc_ge_1"] = idx_bldg_yes_land_alloc_ge_1.sum()
    # Remedy 4: no action taken, but warn the user

    # Check 5: If no building exists:
    # land allocation must be 1.0:
    idx_bldg_no_land_alloc_ne_1 = df["bldg_area_finished_sqft"].eq(0) & df[
        "model_land_alloc"
    ].ne(1)
    counts["bldg_no_land_alloc_ne_1"] = idx_bldg_no_land_alloc_ne_1.sum()
    if do_remedy:
        # Remedy 4: set land value to market value
        df.loc[idx_bldg_no_land_alloc_ne_1, "model_land_value"] = df.loc[
            idx_bldg_no_land_alloc_ne_1, "model_market_value"
        ]
        df.loc[idx_bldg_no_land_alloc_ne_1, "model_land_alloc"] = 1.0
        df.loc[idx_bldg_no_land_alloc_ne_1, "model_impr_value"] = 0.0

    return counts
