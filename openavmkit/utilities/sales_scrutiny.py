import pandas as pd


def flag_dupe_date_price(df_sales: pd.DataFrame, jurisdiction=None) -> pd.DataFrame:
    """Flag same-parcel duplicate sales that share a date and price."""
    if jurisdiction is not None:
        date_price = (
            df_sales[jurisdiction].astype(str)
            + "---"
            + df_sales["sale_date"].astype(str)
            + "---"
            + df_sales["sale_price"].astype(str)
        )
    else:
        date_price = (
            df_sales["sale_date"].astype(str)
            + "---"
            + df_sales["sale_price"].astype(str)
        )
    # Distinct parcels can legitimately share one date/price in a multi-parcel
    # deed. Include the parcel key so only same-parcel repeats are flagged.
    if "key" in df_sales.columns:
        parcel_id = df_sales["key"].astype(str)
        missing_key = df_sales["key"].isna() | df_sales["key"].astype(str).str.strip().eq("")
        if missing_key.any() and "key_sale" in df_sales.columns:
            parcel_id = parcel_id.mask(missing_key, df_sales["key_sale"].astype(str))
        date_price = date_price + "---" + parcel_id
    elif "key_sale" in df_sales.columns:
        date_price = date_price + "---" + df_sales["key_sale"].astype(str)

    dupes = date_price.value_counts()
    dupe_keys = dupes[dupes > 1].index.values
    df_sales.loc[date_price.isin(dupe_keys), "flag_dupe_date_price"] = True
    return df_sales
