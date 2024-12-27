import os

import pandas as pd

from openavmkit.data import get_sales, get_sale_field, get_vacant
from openavmkit.horizontal_equity_study import HorizontalEquityStudy
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import div_z_safe, div_field_z_safe
from openavmkit.utilities.settings import get_fields_categorical, apply_dd_to_df_cols


class SalesScrutinyStudy:
  df_vacant: pd.DataFrame
  df_improved: pd.DataFrame
  settings: dict

  def __init__(
    self,
    df: pd.DataFrame,
    settings: dict
  ):

    df = get_sales(df, settings)

    df_vacant = get_vacant(df, settings)
    df_improved = get_vacant(df, settings, invert=True)

    stuff = {
      "i": df_improved,
      "v": df_vacant
    }

    sale_field = get_sale_field(settings)

    for key in stuff:
      df = stuff[key]
      df, cluster_fields = mark_sales_scrutiny_clusters(df, settings)
      df["ss_id"] = key + "_" + df["ss_id"]
      per_sqft = ""
      denominator = ""
      if key == "i":
        per_sqft = "per_impr_sqft"
        denominator = "bldg_area_finished_sqft"
      elif key == "v":
        per_sqft = "per_land_sqft"
        denominator = "land_area_sqft"

      sale_field_per = f"{sale_field}_{per_sqft}"
      df[sale_field_per] = div_z_safe(df, sale_field, denominator)

      df_cluster_fields = df[["key"] + cluster_fields]
      df = calc_sales_scrutiny(df, sale_field_per)
      df = df.merge(df_cluster_fields, on="key", how="left")

      stuff[key] = df

    self.df_vacant = stuff["v"]
    self.df_improved = stuff["i"]
    self.settings = settings

  def write(self, path: str):
    os.makedirs(f"{path}/sales_scrutiny", exist_ok=True)

    df_vacant = _prettify(self.df_vacant, self.settings)
    df_improved = _prettify(self.df_improved, self.settings)

    df_vacant.to_csv(f"{path}/sales_scrutiny/vacant.csv", index=False)
    df_improved.to_csv(f"{path}/sales_scrutiny/improved.csv", index=False)


def _prettify(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
  renames = {
    "key": "Primary key",
    "ss_id": "Sales scrutiny cluster",
    "count": "# of sales in cluster",
    "sale_price_time_adj_per_impr_sqft": "Sale price / improved sqft (time adjusted)",
    "sale_price_time_adj_per_land_sqft": "Sale price / land sqft (time adjusted)",
    "sale_price_per_impr_sqft": "Sale price / improved sqft",
    "sale_price_per_land_sqft": "Sale price / land sqft",
    "median": "Median",
    "chd": "CHD",
    "max": "Max",
    "min": "Min",
    "stdev": "Standard deviation",
    "relative_ratio": "Relative ratio",
    "med_dist_stdevs": "Median distance from median, in std. deviations",
    "flagged": "Flagged"
  }
  df = df.rename(columns=renames)
  df = apply_dd_to_df_cols(df, settings)
  return df


def calc_sales_scrutiny(df: pd.DataFrame, sales_field: str):
    df = df.copy()
    he_study = HorizontalEquityStudy(df, "ss_id", sales_field)
    summaries = he_study.cluster_summaries

    data = {
      "ss_id": [],
      "count": [],
      "median": [],
      "chd": [],
      "max": [],
      "min": [],
      "stdev": [],
    }

    for cluster in summaries:
      summary = summaries[cluster]
      count = summary.count
      median = summary.median
      max = summary.max
      min = summary.min
      chd = summary.chd

      data["ss_id"].append(cluster)
      data["count"].append(count)
      data["median"].append(median)
      data["chd"].append(chd)
      data["max"].append(max)
      data["min"].append(min)

      # get the slice:
      df_c = df[df["ss_id"].eq(cluster)]

      # calculate stdev:
      stdev = df_c[sales_field].std()

      if pd.isna(stdev):
        stdev = 0.0

      data["stdev"].append(stdev)

    df_cluster = pd.DataFrame(data)

    df = df[["key", "ss_id", sales_field]].copy()
    df = df.merge(df_cluster, on="ss_id", how="left")
    df["relative_ratio"] = div_z_safe(df, sales_field, "median")
    df["med_dist_stdevs"] = div_field_z_safe(df[sales_field] - df["median"], df["stdev"])

    df["low_thresh"] = -float('inf')
    df["high_thresh"] = float('inf')

    df.loc[
      ~df["median"].isna() &
      ~df["stdev"].isna(),
      "low_thresh"] = df["median"] - (2 * df["stdev"])
    df.loc[
      ~df["median"].isna() &
      ~df["stdev"].isna(),
      "high_thresh"] = df["median"] + (2 * df["stdev"])

    idx_low = df[sales_field].lt(df["low_thresh"])
    idx_high = df[sales_field].gt(df["high_thresh"])

    df["flagged"] = False
    df.loc[idx_low | idx_high, "flagged"] = True

    # drop low_thresh/high_thresh:
    df = df.drop(columns=["low_thresh", "high_thresh"])

    df = df[["key", "ss_id", "count", sales_field, "median", "max", "min", "chd", "stdev", "relative_ratio", "med_dist_stdevs", "flagged"]]

    return df


def mark_sales_scrutiny_clusters(df: pd.DataFrame, settings: dict, verbose: bool = False):
  df_sales = get_sales(df, settings)

  ss = settings.get("analysis", {}).get("sales_scrutiny", {})
  location = ss.get("location", "neighborhood")
  fields_categorical = ss.get("fields_categorical", [])
  fields_numeric = ss.get("fields_numeric", None)

  # check if this is a vacant dataset:
  if df_sales["is_vacant"].eq(1).all():
    # if so remove all improved categoricals
    impr_fields = get_fields_categorical(settings, df, include_boolean=False, types=["impr"])
    fields_categorical = [f for f in fields_categorical if f not in impr_fields]

  df_sales["ss_id"], fields_used = make_clusters(df_sales, location, fields_categorical, fields_numeric, verbose=verbose)
  return df_sales, fields_used