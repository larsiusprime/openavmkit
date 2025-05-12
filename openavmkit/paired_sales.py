import os

import numpy as np
import pandas as pd
from scipy.spatial._ckdtree import cKDTree

from openavmkit.data import SalesUniversePair, get_hydrated_sales_from_sup, get_sale_field
from openavmkit.filters import select_filter, resolve_filter
from openavmkit.utilities.data import div_z_safe
from openavmkit.utilities.geometry import get_crs
from openavmkit.utilities.plotting import plot_bar
from openavmkit.utilities.stats import calc_cod, trim_outliers


def spatial_paired_sales(
    sup: SalesUniversePair,
    settings: dict,
    model_group: str,
    verbose: bool = False,
    plot: bool = False,
    write: bool = False
):

  s_land = settings.get("land", {})
  s_entry = s_land.get(model_group, s_land.get("default", {}))
  land_filter = s_entry.get("prime_lot_filter")
  max_distance = s_entry.get("max_distance", 0.5) # miles

  distance_m = max_distance * 1609.34 # meters

  sale_field = get_sale_field(settings, sup.sales)

  # =============================================================
  # 0. Get sales data, reproject, and add coordinates
  # =============================================================

  df_sales = get_hydrated_sales_from_sup(sup).copy()

  if land_filter is not None:
    land_filter = ["and",
       ["==", "vacant_sale", True],
       land_filter
     ]
    mask_vacant = resolve_filter(df_sales, land_filter)
  else:
    mask_vacant = df_sales["vacant_sale"].eq(True)

  df_sales = df_sales[[
    "key_sale",
    "geometry",
    "model_group",
    "he_id",
    "land_he_id",
    "impr_he_id",
    "land_area_sqft",
    "bldg_area_finished_sqft",
    sale_field,
    "vacant_sale"
  ]].copy()

  crs = get_crs(df_sales, "equal_distance")
  df_sales = df_sales.to_crs(crs)

  df_sales = df_sales[df_sales["model_group"].eq(model_group)]

  df_sales["loc_x"] = df_sales.geometry.centroid.x
  df_sales["loc_y"] = df_sales.geometry.centroid.y
  df_sales = df_sales[df_sales["loc_x"].notna() & df_sales["loc_y"].notna()]

  df_sales = df_sales.drop(columns=["geometry"])

  count_vacant = df_sales["vacant_sale"].sum()

  df_v = df_sales[mask_vacant]
  df_i = df_sales[df_sales["vacant_sale"].eq(False)]

  print(f"Found {count_vacant} vacant sales and {len(df_i)} improved sales in {model_group} model group")

  print(f"--> {len(df_v)} vacant sales after applying land filter")

  # =============================================================
  # 1. For each vacant sale, find improved sales within N distance
  # =============================================================

  coords_v = df_v[['loc_x', 'loc_y']].values
  coords_i = df_i[['loc_x', 'loc_y']].values

  tree = cKDTree(coords_i)

  # Query the tree: for each vacant sale, find all improved sales within range
  # each entry in indices_within is an array of indices into coords_i, corresponding to matches with coords_v
  indices_within = tree.query_ball_point(coords_v, r=distance_m)

  # ===============================================================================================================
  # 2. Identify all the impr_he_id clusters near a vacant sale, calculate improved value as sale_price - land_value
  # ===============================================================================================================

  # Process the vacant sales
  df_v_base = df_v[["key_sale", "land_area_sqft", sale_field]].copy().rename(
    columns={
      "key_sale": "v_key",
      "land_area_sqft": "v_land_area_sqft",
      sale_field: "land_value"
    }
  )
  df_v_base["land_value_sqft"] = div_z_safe(df_v, sale_field, "land_area_sqft")

  df_results: pd.DataFrame | None = None

  for i, match_idx in enumerate(indices_within):

    # Get the corresponding local improved sales
    df_match = df_i.iloc[match_idx].copy()

    # Get the corresponding vacant sale key
    v_key = df_v_base.iloc[i]["v_key"]

    # Pair the vacant sale with each improved sale, generating a dataframe containing one record per pair
    df_match["v_key"] = v_key
    df_combined = df_match.merge(df_v_base, how="left", on="v_key")

    # Join the impr_he_id cluster with the vacant sale key to identify the *local* cluster
    df_combined["impr_he_id_x_v_key"] = df_combined["impr_he_id"].astype(str) + "_x_" + df_combined["v_key"].astype(str)

    # Paint each improved sale with the vacant sale's land value rate
    df_combined["land_value"] = df_combined["land_value_sqft"] * df_combined["land_area_sqft"]

    # Calculate imputed improved value by subtracting land value from sale price
    df_combined["impr_value"] = df_combined[sale_field] - df_combined["land_value"]

    # Calculate imputed improved value per sqft
    df_combined["impr_value_sqft"] = div_z_safe(df_combined, "impr_value", "bldg_area_finished_sqft")

    # Join all our results together
    if df_results is None:
      df_results = df_combined
    else:
      df_results = pd.concat([df_results, df_combined], ignore_index=True)

  # ===============================================================================================================
  # 3. Calculate median improved value per sqft and measure variation, locally & globally
  # ===============================================================================================================

  d_local = {
    "impr_he_id": [],
    "v_key": [],
    "count": [],
    "neg": [],
    "25th": [],
    "med": [],
    "75th": [],
    "stdev": [],
    "cov": [],
    "abs_cov": []
  }

  d_global = {
    "impr_he_id": [],
    "locs": [],
    "count": [],
    "neg": [],
    "25th": [],
    "med": [],
    "75th": [],
    "stdev": [],
    "cov": [],
    "abs_cov": []
  }

  # Calculate global variation per impr_he_id cluster
  for impr_he_id in df_results["impr_he_id"].unique():
    df_loc = df_results[df_results["impr_he_id"].eq(impr_he_id)]
    impr_values_global = df_loc["impr_value_sqft"].values
    count = len(impr_values_global)
    if count < 3:
      continue

    impr_values_global = impr_values_global[~np.isnan(impr_values_global)]
    if len(impr_values_global) == 0:
      continue
    count_neg = len(impr_values_global[impr_values_global < 0])
    perc_25 = np.round(np.percentile(impr_values_global, 25)).astype(int)
    median_global = np.round(np.median(impr_values_global)).astype(int)
    perc_75 = np.round(np.percentile(impr_values_global, 75)).astype(int)
    stdev = np.round(np.std(impr_values_global)).astype(int)
    cov = (np.round(stdev / np.mean(impr_values_global), 2)*100).astype(int)

    locations = len(df_loc["v_key"].unique())

    d_global["impr_he_id"].append(impr_he_id)
    d_global["locs"].append(locations)
    d_global["count"].append(count)
    d_global["neg"].append(count_neg)
    d_global["25th"].append(perc_25)
    d_global["med"].append(median_global)
    d_global["75th"].append(perc_75)
    d_global["stdev"].append(stdev)
    d_global["cov"].append(cov)
    d_global["abs_cov"].append(abs(cov))

  # Calculate local variation within each impr_he_id cluster x v_key combination
  for impr_he_id_x_v_key in df_results["impr_he_id_x_v_key"].unique():
    df_loc = df_results[df_results["impr_he_id_x_v_key"].eq(impr_he_id_x_v_key)]
    impr_he_id = df_loc["impr_he_id"].values[0]
    v_key = df_loc["v_key"].values[0]
    impr_values_local = df_loc["impr_value_sqft"].values
    count = len(impr_values_local)
    if count < 3:
      continue

    impr_values_local = impr_values_local[~np.isnan(impr_values_local)]
    if len(impr_values_local) == 0:
      continue
    count_neg = len(impr_values_local[impr_values_local < 0])
    perc_25 = np.round(np.percentile(impr_values_local, 25)).astype(int)
    median_local = np.round(np.median(impr_values_local)).astype(int)
    perc_75 = np.round(np.percentile(impr_values_local, 75)).astype(int)
    stdev = np.round(np.std(impr_values_local)).astype(int)
    cov = (np.round(stdev / np.mean(impr_values_local), 2)*100).astype(int)

    d_local["impr_he_id"].append(impr_he_id)
    d_local["v_key"].append(v_key)
    d_local["count"].append(len(impr_values_local))
    d_local["neg"].append(count_neg)
    d_local["25th"].append(perc_25)
    d_local["med"].append(median_local)
    d_local["75th"].append(perc_75)
    d_local["stdev"].append(stdev)
    d_local["cov"].append(cov)
    d_local["abs_cov"].append(abs(cov))

  df_local = pd.DataFrame(d_local)
  df_global = pd.DataFrame(d_global)

  df_local.sort_values(by=["impr_he_id", "abs_cov"], ascending=[True, True], inplace=True)
  df_global.sort_values(by=["abs_cov"], ascending=[True], inplace=True)

  df_local = df_local.drop(columns=["abs_cov"])
  df_global = df_global.drop(columns=["abs_cov"])

  df_payload = df_global[["impr_he_id", "med", "stdev", "cov"]].rename(columns={
    "med": "impr_value_sqft",
    "stdev": "impr_stdev",
    "cov": "impr_cov"
  })

  df_sales = get_hydrated_sales_from_sup(sup)
  df_site_values = df_sales[df_sales["model_group"].eq(model_group)].copy()
  df_site_values = df_site_values.merge(df_payload, on="impr_he_id", how="left")
  df_site_values["impr_value"] = df_site_values["impr_value_sqft"] * df_site_values["bldg_area_finished_sqft"]
  df_site_values["impr_value_low"] = df_site_values["impr_value_sqft"] - df_site_values["impr_stdev"]
  df_site_values["land_value"] = df_site_values[sale_field] - df_site_values["impr_value"]
  df_site_values["land_value_high"] = df_site_values[sale_field] - df_site_values["impr_value_low"]
  df_site_values["land_value_sqft"] = div_z_safe(df_site_values, "land_value", "land_area_sqft")
  df_site_values["land_value_high_sqft"] = div_z_safe(df_site_values, "land_value_high", "land_area_sqft")
  df_site_values["site_value_degree"] = 0

  df_site_values.loc[
    df_site_values["key_sale"].isin(df_v["key_sale"].unique()),
    "site_value_degree"
  ] = 1

  df_site_values.loc[
    df_site_values["key_sale"].isin(df_results["key_sale"].unique()) &
    df_site_values["key_sale"].isin(df_i["key_sale"].unique()) &
    df_site_values["site_value_degree"].eq(0),
    "site_value_degree"
  ] = 2

  df_site_values.loc[
    ~df_site_values["impr_value_sqft"].isna() &
    df_site_values["site_value_degree"].eq(0),
    "site_value_degree"
  ] = 3

  df_site_values = df_site_values[df_site_values["site_value_degree"].gt(0)]

  if verbose:

    print("")
    print(f"Derived site values for {len(df_site_values)} sales in {model_group} model group:")
    print(f"1st degree: {len(df_site_values[df_site_values['site_value_degree'].eq(1)])}")
    print(f"2nd degree: {len(df_site_values[df_site_values['site_value_degree'].eq(2)])}")
    print(f"3rd degree: {len(df_site_values[df_site_values['site_value_degree'].eq(3)])}")
    print("")

    print("==========================================================")
    print("          Global improvement values + variation")
    print("==========================================================")
    print(df_global.to_string(index=False))
    print("")

    print("==========================================================")
    print("   Local improvement values + variation, per impr_he_id")
    print("==========================================================")
    print(df_local.to_string(index=False))
    print("")

  if write:
    os.makedirs(f"out/paired_sales/{model_group}", exist_ok=True)
    df_local.to_csv(f"out/paired_sales/{model_group}/local.csv", index=False)
    df_global.to_csv(f"out/paired_sales/{model_group}/global.csv", index=False)
    df_site_values.to_parquet(f"out/paired_sales/{model_group}/site_values.parquet", index=False)

  if plot:
    for impr_he_id in df_results["impr_he_id"].unique():
      df_loc = df_results[df_results["impr_he_id"].eq(impr_he_id)]
      locations = len(df_loc["v_key"].unique())
      df_loc["impr_value_sqft_round"] = (np.round(df_loc["impr_value_sqft"]/5) * 5)

      all_vcs: pd.DataFrame | None = None

      for v_key in df_loc["v_key"].unique():
        df_loc_v = df_loc[df_loc["v_key"].eq(v_key)]
        vcs = df_loc_v["impr_value_sqft_round"].value_counts().reset_index()
        vcs["v_key"] = v_key
        if all_vcs is None:
          all_vcs = vcs
        else:
          all_vcs = pd.concat([all_vcs, vcs], ignore_index=True)

      os.makedirs(f"out/paired_sales/{model_group}/plots", exist_ok=True)
      plot_bar(
        all_vcs,
        "impr_value_sqft_round",
        height=all_vcs["count"].values,
        width=4,
        title=f"{impr_he_id} @ {locations} locations",
        style={
          "random_color_by": "v_key"
        },
        out_file=f"out/paired_sales/{model_group}/plots/{impr_he_id}.png" if write else None
      )