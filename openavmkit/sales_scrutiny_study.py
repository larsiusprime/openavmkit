import os

import pandas as pd
from diptest import diptest

from openavmkit.checkpoint import read_checkpoint, write_checkpoint
from openavmkit.data import get_sales, get_sale_field, get_important_fields, get_locations, \
  get_vacant_sales
from openavmkit.horizontal_equity_study import HorizontalEquityStudy
from openavmkit.reports import start_report, finish_report
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import div_z_safe, div_field_z_safe, rename_dict, do_per_model_group, combine_dfs
from openavmkit.utilities.excel import write_to_excel
from openavmkit.utilities.settings import get_fields_categorical, apply_dd_to_df_cols


class SalesScrutinyStudySummary:
  num_sales_flagged: int
  num_sales_total: int
  num_sales_total: int
  num_flagged_sales_by_type: dict[str : int]

  def __init__(self):
    self.num_sales_flagged = 0
    self.num_sales_total = 0
    self.num_flagged_sales_by_type = {}


class SalesScrutinyStudy:
  df_vacant: pd.DataFrame
  df_improved: pd.DataFrame
  settings: dict
  model_group: str
  summaries: dict[str, SalesScrutinyStudySummary]

  def __init__(
    self,
    df: pd.DataFrame,
    settings: dict,
    model_group: str
  ):

    self.model_group = model_group

    df = df[df["model_group"].eq(model_group)]
    df = get_sales(df, settings)

    df_vacant = get_vacant_sales(df, settings)
    df_improved = get_vacant_sales(df, settings, invert=True)

    stuff = {
      "i": df_improved,
      "v": df_vacant
    }

    sale_field = get_sale_field(settings)
    important_fields = get_important_fields(settings, df)
    location_fields = get_locations(settings, df)
    self.summaries = {
      "i": SalesScrutinyStudySummary(),
      "v": SalesScrutinyStudySummary()
    }

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

      total_anomalies = 0
      for i in range(1, 6):
        field = f"anomaly_{i}"
        count_anomaly = len(df[df[field].eq(True)])
        total_anomalies += count_anomaly
        self.summaries[key].num_flagged_sales_by_type[field] = count_anomaly

      self.summaries[key].num_sales_flagged = len(df[df["flagged"].eq(True)])
      self.summaries[key].num_sales_total = len(df)

      stuff[key] = df

    self.df_vacant = stuff["v"]
    self.df_improved = stuff["i"]
    self.settings = settings


  def write(self, path: str):
    self._write(path, True)
    self._write(path, False)


  def get_scrutinized(self, df: pd.DataFrame):
    df_v = self.df_vacant
    df_i = self.df_improved

    # remove flagged sales:
    keys_flagged = df_v[df_v["flagged"].eq(True)]["key"].tolist()
    keys_flagged += df_i[df_i["flagged"].eq(True)]["key"].tolist()

    # ensure unique:
    keys_flagged = list(dict.fromkeys(keys_flagged))

    df.loc[df["key"].isin(keys_flagged), "valid_sale"] = False

    # merge ss_id into df:
    df = combine_dfs(df, df_v[["key", "ss_id"]])
    df = combine_dfs(df, df_i[["key", "ss_id"]])

    return df


  def _write(self, path: str, is_vacant: bool):
    os.makedirs(f"{path}/sales_scrutiny", exist_ok=True)

    root_path = path

    if is_vacant:
      df = self.df_vacant
      path = f"{path}/sales_scrutiny/vacant"
    else:
      df = self.df_improved
      path = f"{path}/sales_scrutiny/improved"

    df = _prettify(df, self.settings)

    df = df.sort_values(by="CHD", ascending=False)

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

    key = "v" if is_vacant else "i"
    self._write_report(root_path, key=key, model_group=self.model_group)


  def _write_report(self, path: str, key: str, model_group: str):
    report = start_report("sales_scrutiny", self.settings, model_group)

    summary = self.summaries.get(key)

    num_sales_total = summary.num_sales_total
    num_sales_flagged = summary.num_sales_flagged

    for i in range(1, 6):
      field = f"anomaly_{i}"
      count = summary.num_flagged_sales_by_type.get(field, 0)
      percent = count / num_sales_total
      report.set_var(f"num_sales_flagged_type_{i}", f"{count:0,.0f}")
      report.set_var(f"pct_sales_flagged_type_{i}", f"{percent:0.2%}")

    pct_sales_flagged = num_sales_flagged / num_sales_total
    report.set_var("num_sales_flagged", f"{num_sales_flagged:0,.0f}")
    report.set_var("num_sales_total", f"{num_sales_total:0,.0f}")
    report.set_var("pct_sales_flagged", f"{pct_sales_flagged:0.2%}")

    vacant_type = "vacant" if key == "v" else "improved"
    outpath = f"{path}/reports/sales_scrutiny_{vacant_type}"

    finish_report(report, outpath, "sales_scrutiny")


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
    "bimodal": "Bimodal cluster",
    "anomaly_1": "Weird price/sqft & weird sqft",
    "anomaly_2": "Low price & low price/sqft",
    "anomaly_3": "High price & high price/sqft",
    "anomaly_4": "Normal price & high price/sqft",
    "anomaly_5": "Normal price & low price/sqft"
  }


def _prettify(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
  df = df.rename(columns=_get_ss_renames())
  df = apply_dd_to_df_cols(df, settings)
  return df


def calc_sales_scrutiny(df_in: pd.DataFrame, sales_field: str):
    df = df_in.copy()

    df = _apply_he_stats(df, "ss_id", sales_field)

    base_sales_field = _get_base_sales_field(sales_field)

    # Calculate standard deviation thresholds:
    df["low_thresh"] = -float('inf')
    df["high_thresh"] = float('inf')

    # Flag anything above or below 2 standard deviations from the median
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

    # Additionally, flag anything with a relative ratio >= 4.0
    df.loc[df["relative_ratio"].ge(4.0), "flagged"] = True

    # Additionally, flag anything with a relative ratio <= .35 AND a stdev distance of < -1.0
    df.loc[
      df["relative_ratio"].le(0.35) &
      df["med_dist_stdevs"].lt(-1.0),
    "flagged"] = True

    # Check for the five anomalies:
    df = _check_for_anomalies(df, df_in, sales_field)

    df["bimodal"] = False
    bimodal_clusters = _identify_bimodal_clusters(df, sales_field)
    df.loc[df["ss_id"].isin(bimodal_clusters), "bimodal"] = True

    # drop low_thresh/high_thresh:
    df = df.drop(columns=["low_thresh", "high_thresh"])

    df = df[["key", "ss_id", "count", sales_field, base_sales_field, "median", "max", "min", "chd", "stdev", "relative_ratio", "med_dist_stdevs", "flagged", "bimodal", "anomaly_1", "anomaly_2", "anomaly_3", "anomaly_4", "anomaly_5"]]

    return df


def _apply_he_stats(df: pd.DataFrame, cluster_id: str, sales_field: str):
  he_study = HorizontalEquityStudy(df, cluster_id, sales_field)
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

  if base_sales_field in df and base_sales_field != sales_field:
    df = df[["key", "ss_id", sales_field, base_sales_field]].copy()
  else:
    df = df[["key", "ss_id", sales_field]].copy()

  df = df.merge(df_cluster, on="ss_id", how="left")

  df["relative_ratio"] = div_z_safe(df, sales_field, "median")
  df["med_dist_stdevs"] = div_field_z_safe(df[sales_field] - df["median"], df["stdev"])
  return df


def _check_for_anomalies(df_in: pd.DataFrame, df_sales: pd.DataFrame, sales_field: str):

  # Limit search only to clusters with flagged sales
  df_flagged = df_in[df_in["flagged"].eq(True)]
  flagged_clusters = df_flagged["ss_id"].unique()

  df = df_in.copy()
  df = df[df["ss_id"].isin(flagged_clusters)]

  # land or sqft? Check sales_field:
  sqft = "land_area_sqft" if "land" in sales_field else "bldg_area_finished_sqft" if "impr" in sales_field else ""
  price = "sale_price" if "time_adj" not in sales_field else "sale_price_time_adj"

  if sqft == "":
    raise ValueError("expected `sales_field` to be suffixed with either `_impr_sqft` or `_land_sqft`")

  df_sqft = _apply_he_stats(df_sales, "ss_id", sqft)
  df_price = _apply_he_stats(df_sales, "ss_id", price)

  df_fl = df[df["flagged"].eq(True)]

  df_sqft_fl = df_sqft[df_sqft["key"].isin(df_fl["key"].values)]
  df_price_fl = df_price[df_sqft["key"].isin(df_fl["key"].values)]

  # Check for the symptoms

  # price/sqft low/high/in range (already done)
  # price low/high/in range
  # sqft low/high/in range

  idx_price_low = df_price_fl["relative_ratio"].le(1.0)
  idx_price_high = df_price_fl["relative_ratio"].ge(1.0)

  idx_price_low = df["key"].isin(df_price_fl[idx_price_low]["key"].values)
  idx_price_high = df["key"].isin(df_price_fl[idx_price_high]["key"].values)

  idx_price_not_low = df_price_fl["med_dist_stdevs"].ge(-1.0)
  idx_price_not_high = df_price_fl["med_dist_stdevs"].le(1.0)

  idx_price_not_low = df["key"].isin(df_price_fl[idx_price_not_low]["key"].values)
  idx_price_not_high = df["key"].isin(df_price_fl[idx_price_not_high]["key"].values)

  idx_sqft_low = df_sqft_fl["med_dist_stdevs"].le(-2.0)
  idx_sqft_high = df_sqft_fl["med_dist_stdevs"].ge(2.0)

  idx_sqft_low = df["key"].isin(df_sqft_fl[idx_sqft_low]["key"].values)
  idx_sqft_high = df["key"].isin(df_sqft_fl[idx_sqft_high]["key"].values)

  idx_sqft_not_low = df_sqft_fl["med_dist_stdevs"].ge(-1.0)
  idx_sqft_not_high = df_sqft_fl["med_dist_stdevs"].le(1.0)

  idx_sqft_not_low = df["key"].isin(df_sqft_fl[idx_sqft_not_low]["key"].values)
  idx_sqft_not_high = df["key"].isin(df_sqft_fl[idx_sqft_not_high]["key"].values)

  idx_price_sqft_low = df_fl["relative_ratio"].le(1.0)
  idx_price_sqft_high = df_fl["relative_ratio"].ge(1.0)

  idx_price_sqft_low = df["key"].isin(df_fl[idx_price_sqft_low]["key"].values)
  idx_price_sqft_high = df["key"].isin(df_fl[idx_price_sqft_high]["key"].values)

  # Check for the five anomalies:

  # 1. Price/sqft is high or low, sqft is high or low:
  df.loc[
    (idx_price_sqft_low | idx_price_sqft_high) &
    (idx_sqft_low | idx_sqft_high),
  "anomaly_1"] = True

  # 2. Low price, low price/sqft, sqft is in range
  df.loc[
    idx_price_low &
    idx_price_sqft_low &
    (idx_sqft_not_low | idx_sqft_not_high),
  "anomaly_2"] = True

  # 3. High price, high price/sqft, sqft is in range
  df.loc[
    idx_price_high &
    idx_price_sqft_high &
    (idx_sqft_not_low | idx_sqft_not_high),
  "anomaly_3"] = True

  # 4. Price in range, high price/sqft
  df.loc[
    (idx_price_not_low & idx_price_not_high) &
    idx_price_sqft_high &
    (idx_sqft_not_low | idx_sqft_not_high),
  "anomaly_4"] = True

  # 5. Price in range, low price/sqft
  df.loc[
    (idx_price_not_low & idx_price_not_high) &
    idx_price_sqft_low &
    (idx_sqft_not_low | idx_sqft_not_high),
  "anomaly_5"] = True

  df_out = df_in.copy()
  df_out["anomalies"] = 0
  for i in range(1, 6):
    field = f"anomaly_{i}"
    df_out[field] = False
    df_out[field] = df[field]
    df_out["anomalies"] = df_out["anomalies"] + df[field].fillna(0).astype("Int64")

  df_out.loc[
    df_out["anomalies"].le(0),
    "flagged"] = False

  df_out.drop(columns=["anomalies"], inplace=True)

  return df_out



def _get_base_sales_field(field: str):
  return "sale_price" if "time_adj" not in field else "sale_price_time_adj"


def _identify_bimodal_clusters(df, sales_field):
  bimodal_clusters = []

  for cluster_id, group in df.groupby('ss_id'):
    values = group[sales_field].values
    if len(values) > 3:
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

  df_sales["ss_id"], fields_used = make_clusters(df_sales, location, fields_categorical, fields_numeric, min_cluster_size=5, verbose=verbose)

  # TODO: we don't return fields_used at the moment
  return df_sales


def identify_suspicious_characteristics(df: pd.DataFrame, settings: dict, is_vacant: bool = False, verbose: bool = False):
  df_sales = get_sales(df, settings)
  df_sales = get_vacant_sales(df_sales, settings, not is_vacant)

  ss = settings.get("analysis", {}).get("sales_scrutiny", {})
  location = ss.get("location", "neighborhood")

  sale_field = get_sale_field(settings)
  per_field = "land_sqft" if is_vacant else "impr_sqft"
  sale_field_per = f"{sale_field}_{per_field}"

  df_sales.groupby("location")[[sale_field, sale_field_per]].agg(["count", "min", "median", "max", "std"])

  # What we are looking for is parcels where the sale_field is in line with the overall area but the sale_field_per is not


def _mark_ss_ids(df_in: pd.DataFrame, model_group: str, settings: dict, verbose: bool):
  df = mark_sales_scrutiny_clusters(df_in, settings, verbose)
  df["ss_id"] = model_group + "_" + df["ss_id"]
  return df

def run_sales_scrutiny_per_model_group(df_in: pd.DataFrame, settings: dict, verbose=False):
  return do_per_model_group(df_in, _mark_ss_ids, {"settings": settings, "verbose": verbose})


def run_sales_scrutiny(df_in: pd.DataFrame, settings: dict, model_group: str, verbose=False):
  # run sales validity:
  ss = SalesScrutinyStudy(df_in, settings, model_group=model_group)
  ss.write(f"out")

  # clean sales data:
  return ss.get_scrutinized(df_in)