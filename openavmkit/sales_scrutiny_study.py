import os

import pandas as pd
from diptest import diptest

from openavmkit.data import get_sales, get_sale_field, get_vacant, get_important_fields, get_locations
from openavmkit.horizontal_equity_study import HorizontalEquityStudy
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import div_z_safe, div_field_z_safe, rename_dict
from openavmkit.utilities.excel import write_to_excel
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
    important_fields = get_important_fields(settings, df)
    location_fields = get_locations(settings, df)

    for key in stuff:
      df = stuff[key]
      df, cluster_fields = mark_sales_scrutiny_clusters(df, settings)
      df["ss_id"] = key + "_" + df["ss_id"]
      per_sqft = ""
      denominator = ""
      if key == "i":
        per_sqft = "impr_sqft"
        denominator = "bldg_area_finished_sqft"
      elif key == "v":
        per_sqft = "land_sqft"
        denominator = "land_area_sqft"

      sale_field_per = f"{sale_field}_{per_sqft}"
      df[sale_field_per] = div_z_safe(df, sale_field, denominator)

      other_fields = cluster_fields + location_fields + important_fields
      other_fields = list(dict.fromkeys(other_fields))
      other_fields += ["address", "sale_date", "valid_sale", "vacant_sale"]

      other_fields = [f for f in other_fields if f in df]

      df_cluster_fields = df[["key"] + other_fields]
      df = calc_sales_scrutiny(df, sale_field_per)
      df = df.merge(df_cluster_fields, on="key", how="left")

      stuff[key] = df

    self.df_vacant = stuff["v"]
    self.df_improved = stuff["i"]
    self.settings = settings

  def write(self, path: str):
    self._write(path, True)
    self._write(path, False)

  def _write(self, path: str, is_vacant: bool):
    os.makedirs(f"{path}/sales_scrutiny", exist_ok=True)

    if is_vacant:
      df = self.df_vacant
      path = f"{path}/sales_scrutiny/vacant"
    else:
      df = self.df_improved
      path = f"{path}/sales_scrutiny/improved"

    idx_flagged = df[df["flagged"].eq(True)].index
    idx_bimodal = df[df["bimodal"].eq(True)].index
    df = _prettify(df, self.settings)
    df.to_csv(f"{path}.csv", index=False)

    _curr_0 = {"num_format": "#,##0"}
    _curr_2 = {"num_format": "#,##0.00"}
    _dec_0 = {"num_format": "#,##0"}
    _dec_2 = {"num_format": "#,##0.00"}
    _float_2 = {"num_format": "0.00"}
    _float_0 = {"num_format": "#,##0"}
    _date = {"num_format": "yyyy-mm-dd"}
    _int = {"num_format": "0"}
    _bigint = {"num_format": "#,##0"}

    # Write to excel:
    columns = rename_dict({
      "sale_price": _curr_0,
      "sale_price_time_adj": _curr_0,
      "sale_price_impr_sqft": _curr_2,
      "sale_price_land_sqft": _curr_2,
      "sale_price_time_adj_impr_sqft": _curr_2,
      "sale_price_time_adj_land_sqft": _curr_2,
      "Median": _curr_2,
      "Max": _curr_2,
      "Min": _curr_2,
      "CHD": _float_2,
      "Standard deviation": _curr_2,
      "Relative ratio": _float_2,
      "Median distance from median, in std. deviations": _float_2
    }, _get_ss_renames())

    column_conditions = {
      "Flagged": {
        "type": "cell",
        "criteria": "==",
        "value": "TRUE",
        "format": {"bold": True, "font_color": "red"}
      },
      "Bimodal cluster": {
        "type": "cell",
        "criteria": "==",
        "value": "TRUE",
        "format": {"bold": True, "font_color": "red"}
      }
    }

    write_to_excel(df, f"{path}.xlsx", {
      "columns": {
        "formats": columns,
        "conditions": column_conditions
      }
    })


def _get_ss_renames():
  return {
    "key": "Primary key",
    "ss_id": "Sales scrutiny cluster",
    "count": "# of sales in cluster",
    "sale_price": "Sale price",
    "sale_price_impr_sqft": "Sale price / improved sqft",
    "sale_price_land_sqft": "Sale price / land sqft",
    "sale_price_time_adj": "Sale price (time adjusted)",
    "sale_price_time_adj_impr_sqft": "Sale price / improved sqft (time adjusted)",
    "sale_price_time_adj_land_sqft": "Sale price / land sqft (time adjusted)",
    "median": "Median",
    "chd": "CHD",
    "max": "Max",
    "min": "Min",
    "stdev": "Standard deviation",
    "relative_ratio": "Relative ratio",
    "med_dist_stdevs": "Median distance from median, in std. deviations",
    "flagged": "Flagged",
    "bimodal": "Bimodal cluster"
  }


def _prettify(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
  df = df.rename(columns=_get_ss_renames())
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

    base_sales_field = _get_base_sales_field(sales_field)

    df = df[["key", "ss_id", sales_field, base_sales_field]].copy()
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

    df["bimodal"] = False
    bimodal_clusters = _identify_bimodal_clusters(df, sales_field)
    df.loc[df["ss_id"].isin(bimodal_clusters), "bimodal"] = True

    # drop low_thresh/high_thresh:
    df = df.drop(columns=["low_thresh", "high_thresh"])

    df = df[["key", "ss_id", "count", sales_field, base_sales_field,   "median", "max", "min", "chd", "stdev", "relative_ratio", "med_dist_stdevs", "flagged", "bimodal"]]

    return df


def _get_base_sales_field(field: str):
  return "sale_price" if "time_adj" not in field else "sale_price_time_adj"


def _identify_bimodal_clusters(df, sales_field):
  bimodal_clusters = []

  for cluster_id, group in df.groupby('ss_id'):
    values = group[sales_field].values
    dip, p_value = diptest(values)
    if p_value < 0.05:  # Statistically significant deviation from unimodality
      bimodal_clusters.append(cluster_id)

  return bimodal_clusters


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