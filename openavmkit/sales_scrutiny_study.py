"""
Sales scrutiny — flagging suspect sales.

Heuristic and statistical checks for identifying sales that may not reflect
arms-length, market-rate transactions: outlier prices within a cluster of
similar properties, suspicious price clusters (Gaussian mixture analysis),
manual exclusion lists, etc. Marks suspect sales so they can be excluded
from training data while remaining visible for audit.

Cluster definitions come from ``analysis.sales_scrutiny`` in ``settings.json``.
"""
import os
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from openavmkit.utilities.timing import TimingData

from openavmkit.data import (
  _get_sales,
  get_sale_field,
  get_important_fields,
  get_vacant_sales,
  SalesUniversePair,
  get_hydrated_sales_from_sup,
)
from openavmkit.horizontal_equity_study import HorizontalEquityStudy
from openavmkit.reports import start_report, finish_report
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import (
    div_df_z_safe,
    div_series_z_safe,
    rename_dict,
    do_per_model_group,
    combine_dfs,
)
from openavmkit.utilities.excel import write_to_excel
from openavmkit.utilities.sales_scrutiny import (
    SALE_PARCEL_COUNT,
    flag_bulk_deeds,
    flag_dupe_date_price,
    flag_dupe_deed_date,
)
from openavmkit.utilities.settings import get_fields_categorical, _apply_dd_to_df_cols, area_unit, get_locations, warn_if_location_collapsed


class SalesScrutinyStudySummary:
    """
    Summary statistics for a Sales scrutiny study

    Attributes
    ----------
    num_sales_flagged : int
        The number of sales flagged by the study
    num_sales_total : int
        The number of sales that were tested
    num_flagged_sales_by_type : dict[str:int]
        Dictionary breaking down number of flagged sales by anomaly type
    """

    num_sales_flagged: int
    num_sales_total: int
    num_flagged_sales_by_type: dict[str:int]

    def __init__(self):
        self.num_sales_flagged = 0
        self.num_sales_total = 0
        self.num_flagged_sales_by_type = {}


class SalesScrutinyStudy:
    """
    Sales scrutiny study object

    This class performs cluster-based analysis on sales to identify anomalies
    Anomalous sales detected by this method are more likely to be invalid sales
    This study helps the modeler narrow their focus on which sales should be scrutinized

    Attributes
    ----------
    df_vacant : pd.DataFrame
        DataFrame of sales that were allegedly vacant (no building) at time of sale
    df_improved : pd.DataFrame
        DataFrame of sales that were allegedly improved (had building) at time of sale
    settings : dict
        Settings dictionary
    model_group : str
        The model group to investigate
    summaries : dict[str, SalesScrutinyStudySummary]
        Dictionary in which the results are stored
    unit : str
        The area unit ("sqft" or "sqm")
    """

    def __init__(self, df: pd.DataFrame, settings: dict, model_group: str):
        """Initialize a SalesScrutinyStudy object

        Parameters
        ----------
        df : pd.DataFrame
            The data you wish to analyze
        settings : dict
            Settings dictionary
        model_group : str
            Model group to analyze
        """
        self.model_group = model_group
        
        self.unit = area_unit(settings)

        df = df[df["model_group"].eq(model_group)]
        df = _get_sales(df, settings)

        df_vacant = get_vacant_sales(df, settings)
        df_improved = get_vacant_sales(df, settings, invert=True)

        stuff = {"i": df_improved, "v": df_vacant}

        sale_field = get_sale_field(settings)
        important_fields = get_important_fields(settings, df)
        location_fields = get_locations(settings, df)
        self.summaries = {
            "i": SalesScrutinyStudySummary(),
            "v": SalesScrutinyStudySummary(),
        }

        for key in stuff:
            df = stuff[key]
            df, cluster_fields = _mark_sales_scrutiny_clusters(df, settings)
            df["ss_id"] = df["ss_id"].astype(object).fillna("").astype(str)
            df["ss_id"] = df["model_group"].astype(object).astype(str) + "_" + key + "_" + df["ss_id"]
            per_area = ""
            denominator = ""
            if key == "i":
                per_area = f"impr_{self.unit}"
                denominator = f"bldg_area_finished_{self.unit}"
            elif key == "v":
                per_area = f"land_{self.unit}"
                denominator = f"land_area_{self.unit}"

            sale_field_per = f"{sale_field}_{per_area}"
            df[sale_field_per] = div_df_z_safe(df, sale_field, denominator)

            other_fields = cluster_fields + location_fields + important_fields
            other_fields = list(dict.fromkeys(other_fields))
            other_fields += [
                "address",
                "deed_book",
                "deed_page",
                "sale_date",
                "valid_sale",
                "vacant_sale",
            ]

            other_fields = [f for f in other_fields if f in df.columns]

            df_cluster_fields = df[["key", "key_sale"] + other_fields]
            ss_settings = settings.get("analysis", {}).get("sales_scrutiny", {})
            passes = ss_settings.get("passes", 1)
            strict = ss_settings.get("patterns", "classic") == "strict"
            df = _calc_sales_scrutiny(df, sale_field_per, passes=passes, strict=strict)
            df["vacant"] = key == "v"
            df["area"] = per_area
            df = df.merge(df_cluster_fields, on="key_sale", how="left")

            total_anomalies = 0
            for i in range(1, 6):
                field = f"anomaly_{i}"
                if field in df:
                    count_anomaly = len(df[df[field].eq(True)])
                else:
                    count_anomaly = 0
                total_anomalies += count_anomaly
                self.summaries[key].num_flagged_sales_by_type[field] = count_anomaly

            if "flagged" in df:
                self.summaries[key].num_sales_flagged = len(df[df["flagged"].eq(True)])
            else:
                self.summaries[key].num_sales_flagged = 0
            self.summaries[key].num_sales_total = len(df)

            stuff[key] = df

        self.df_vacant = stuff["v"]
        self.df_improved = stuff["i"]
        self.settings = settings

    def write(self, path: str):
        """Writes the report to disk

        Parameters
        ----------
        path : str
            The root path
        """
        self._write(path, True)
        self._write(path, False)
        os.makedirs(path, exist_ok=True)
        self.flags().to_parquet(f"{path}/flags.parquet", index=False)

    def flags(self) -> pd.DataFrame:
        """Every sale this study flagged, one row each, with the numbers behind the flag.

        Returns
        -------
        pd.DataFrame
            Columns: ``key_sale``, ``key``, ``model_group``, ``check`` (``"cluster"``),
            ``vacant``, ``ss_id``, ``count`` (sales in the cluster it was compared with),
            ``price``, ``price_per_area``, ``area`` (what the price is per, e.g. ``"impr_sqft"``),
            ``median``, ``stdev``, ``relative_ratio``, ``med_dist_stdevs``, ``rules`` (which
            thresholds it crossed), ``pattern`` (which of the five anomalies, e.g. ``"2"``),
            ``scrutiny_pass`` and ``bimodal``.
        """
        return _flag_rows([self.df_improved, self.df_vacant], self.model_group)

    def get_scrutinized(self, df_in: pd.DataFrame, verbose: bool = False):
        """Remove flagged sales from the dataset and return the modified dataset

        Parameters
        ----------
        df_in : pd.DataFrame
            The dataframe you wish to clean
        verbose : bool, optional
            Whether to print verbose output. Default False.
        """
        df = df_in.copy()

        df_v = self.df_vacant
        df_i = self.df_improved

        keys_flagged = []

        if "flagged" in df_v:
            # remove flagged sales (not those flagged only: no pattern, the user decides):
            vacant_flags = df_v[df_v["flagged"].eq(True) & ~_flag_only(df_v)]["key_sale"].tolist()
            keys_flagged += vacant_flags
            if verbose:
                print(f"--> Flagged {len(vacant_flags)} vacant sales")

        if "flagged" in df_i:
            improved_flags = df_i[df_i["flagged"].eq(True) & ~_flag_only(df_i)]["key_sale"].tolist()
            keys_flagged += improved_flags
            if verbose:
                print(f"--> Flagged {len(improved_flags)} improved sales")

        # ensure unique, and leave the sales the user has kept:
        keys_flagged = list(dict.fromkeys(keys_flagged))
        keep = read_keep_keys(self.settings)
        if keep:
            kept = [k for k in keys_flagged if k in keep]
            keys_flagged = [k for k in keys_flagged if k not in keep]
            if verbose and kept:
                print(f"--> Kept {len(kept)} flagged sales listed in the keep file")

        if len(df) > 0:

            num_valid_sales_before = len(df[df["valid_sale"].eq(True)])

            df.loc[df["key_sale"].isin(keys_flagged), "valid_sale"] = False

            num_valid_sales_after = len(df[df["valid_sale"].eq(True)])

            if verbose:
                print(f"--> Unmarked sales before: {num_valid_sales_before}")
                print(f"--> Unmarked sales after: {num_valid_sales_after}")
                diff = num_valid_sales_before - num_valid_sales_after
                print(f"--> Marked {diff} new potentially invalid sales")

            # merge ss_id into df:
            df = combine_dfs(df, df_v[["key_sale", "ss_id"]], index="key_sale")
            df = combine_dfs(df, df_i[["key_sale", "ss_id"]], index="key_sale")

        return df

    def _write(self, path: str, is_vacant: bool):
        os.makedirs(f"{path}", exist_ok=True)

        root_path = path
        
        if is_vacant:
            df = self.df_vacant
            path = f"{path}/vacant"
        else:
            df = self.df_improved
            path = f"{path}/improved"

        df = _prettify(df, self.settings)

        if "CHD" in df:
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
        columns = rename_dict(
            {
                "sale_price": _curr_0,
                "sale_price_time_adj": _curr_0,
                f"sale_price_impr_{self.unit}": _curr_2,
                f"sale_price_land_{self.unit}": _curr_2,
                f"sale_price_time_adj_impr_{self.unit}": _curr_2,
                f"sale_price_time_adj_land_{self.unit}": _curr_2,
                "Median": _curr_2,
                "Max": _curr_2,
                "Min": _curr_2,
                "CHD": _float_2,
                "Standard deviation": _curr_2,
                "Relative ratio": _float_2,
                "Median distance from median, in std. deviations": _float_2,
            },
            _get_ss_renames(),
        )

        column_conditions = {
            "Flagged": {
                "type": "cell",
                "criteria": "==",
                "value": "TRUE",
                "format": {"bold": True, "font_color": "red"},
            },
            "Bimodal cluster": {
                "type": "cell",
                "criteria": "==",
                "value": "TRUE",
                "format": {"bold": True, "font_color": "red"},
            },
        }
        
        write_to_excel(
            df,
            f"{path}.xlsx",
            {"columns": {"formats": columns, "conditions": column_conditions}},
        )

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
            if num_sales_total > 0:
                percent = count / num_sales_total
                percent = f"{percent:0.2%}"
            else:
                percent = "N/A"
            report.set_var(f"num_sales_flagged_type_{i}", f"{count:0,.0f}")
            report.set_var(f"pct_sales_flagged_type_{i}", percent)

        if num_sales_total > 0:
            pct_sales_flagged = num_sales_flagged / num_sales_total
            pct_sales_flagged = f"{pct_sales_flagged:0.2%}"
        else:
            pct_sales_flagged = "N/A"

        report.set_var("num_sales_flagged", f"{num_sales_flagged:0,.0f}")
        report.set_var("num_sales_total", f"{num_sales_total:0,.0f}")
        report.set_var("pct_sales_flagged", pct_sales_flagged)

        vacant_type = "vacant" if key == "v" else "improved"
        outpath = f"{path}/sales_scrutiny_{vacant_type}"

        finish_report(report, outpath, "sales_scrutiny", self.settings)


# The heuristic checks, each on unless analysis.sales_scrutiny.heuristics.<name> is false.
HEURISTICS = ("dupe_deed_date", "dupe_date_price", "bulk_deed", "false_vacant")
# The checks that are off unless analysis.sales_scrutiny.heuristics.<name> is true.
OPT_IN_HEURISTICS = ("built_after_sale", "vacant_priced_improved", "same_surname")
FLAG_COLUMNS = [
    "key_sale", "key", "model_group", "check", "vacant", "ss_id", "count", "price", "price_per_area", "area",
    "median", "stdev", "relative_ratio", "med_dist_stdevs", "rules", "pattern", "scrutiny_pass", "bimodal",
    "price_rating", "per_area_rating", "size_rating", "price_median", "price_dist", "size", "size_median",
    "size_dist", "flag_only", "detail",
]
# How a flagged sale's price, price per area and size compare with its cluster: "low", "in" (range) or "high".
_RATINGS = ["price_rating", "per_area_rating", "size_rating"]
_SIDE = ["price_median", "price_dist", "size", "size_median", "size_dist"]
# The fewest sales a cluster is compared on in a later pass (as make_clusters' min_cluster_size).
_MIN_PASS_SALES = 5
_ANOMALIES = [f"anomaly_{i}" for i in range(1, 6)]
_PASS_COLS = (
    ["count", "median", "chd", "max", "min", "stdev", "relative_ratio", "med_dist_stdevs", "flagged", "rules"]
    + _ANOMALIES
    + ["anomaly_other"]
    + _RATINGS
    + _SIDE
)


def _flag_only(df: pd.DataFrame) -> pd.Series:
    """Flagged but fitting none of the five patterns (strict patterns): shown, never dropped."""
    if "anomaly_other" not in df:
        return pd.Series(False, index=df.index)
    return df["anomaly_other"].eq(True)


def _floats(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce").astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    return pd.Series(values, index=s.index)


def _threshold_flags(
    df: pd.DataFrame, sales_field: str, ratios_only: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Which sales cross a threshold, and which: more than 2 standard deviations below or above
    the median, 4x the median or more, 0.35x the median or less and more than one deviation
    below. A sale is flagged when it also shows one of the five anomalies. ratios_only: only the
    two ratio thresholds (later passes, where a cluster is smaller and its spread narrower)."""
    x, median, stdev, ratio, dist = (
        _floats(df[c]) for c in (sales_field, "median", "stdev", "relative_ratio", "med_dist_stdevs")
    )
    has = median.notna() & stdev.notna()
    checks = {
        "below_2sd": has & x.lt(median - 2 * stdev),
        "above_2sd": has & x.gt(median + 2 * stdev),
        "4x_median": ratio.ge(4.0),
        "0.35x_median": ratio.le(0.35) & dist.lt(-1.0),
    }
    if ratios_only:
        checks = {k: v for k, v in checks.items() if k.endswith("_median")}
    rules = pd.Series("", index=df.index, dtype=object)
    flagged = pd.Series(False, index=df.index)
    for name, idx in checks.items():
        idx = idx.fillna(False).astype(bool)
        flagged |= idx
        rules[idx] = rules[idx].where(rules[idx].eq(""), rules[idx] + ",") + name
    return flagged, rules


def _scrutinize(
    df_in: pd.DataFrame, sales_field: str, ratios_only: bool = False, strict: bool = False
) -> pd.DataFrame:
    df = _apply_he_stats(df_in, "ss_id", sales_field)
    if "median" not in df:
        return df
    df["flagged"], df["rules"] = _threshold_flags(df, sales_field, ratios_only)
    return _check_for_anomalies(df, df_in, sales_field, strict=strict)


def _calc_sales_scrutiny(df_in: pd.DataFrame, sales_field: str, passes: int = 1, strict: bool = False):
    """Cluster outliers. strict: the patterns as the five-pattern table reads (each of price, price per area and
    size low, in range or high at 2 standard deviations from the cluster's median, each flagged sale exactly one
    pattern); a flagged sale that fits none is "other", flagged only (never dropped)."""
    df = _scrutinize(df_in, sales_field, strict=strict)

    if "median" not in df:
        # If the median is not present, then we can't do anything
        # This is usually because there's nothing to analyze
        return df

    base_sales_field = _get_base_sales_field(sales_field)
    df["scrutiny_pass"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[df["flagged"].eq(True), "scrutiny_pass"] = 1

    # Later passes: one far-off sale stretches its cluster's spread, which can hide another.
    # Compare the rest of each cluster that had a sale flagged again, without the flagged ones,
    # on the ratio thresholds only: without its outlier a small cluster's spread is narrow, and
    # 2 deviations would set aside ordinary sales.
    for n in range(2, int(passes) + 1):
        out = df.loc[df["flagged"].eq(True), "key_sale"]
        changed = df.loc[df["scrutiny_pass"].eq(n - 1), "ss_id"].unique()
        rest = df_in[~df_in["key_sale"].isin(out) & df_in["ss_id"].isin(changed)]
        rest = rest[rest.groupby("ss_id")["ss_id"].transform("size").ge(_MIN_PASS_SALES)]
        if len(rest) == 0:
            break
        again = _scrutinize(rest, sales_field, ratios_only=True, strict=strict)
        if "median" not in again:
            break
        new = again[again["flagged"].eq(True)].set_index("key_sale")
        if len(new) == 0:
            break
        idx = df["key_sale"].isin(new.index)
        for col in _PASS_COLS:
            df[col] = df[col].astype(object)
            df.loc[idx, col] = df.loc[idx, "key_sale"].map(new[col]).values
        df.loc[idx, "scrutiny_pass"] = n

    for col in ["count", "median", "chd", "max", "min", "stdev", "relative_ratio", "med_dist_stdevs"]:
        df[col] = pd.to_numeric(df[col])
    df["flagged"] = df["flagged"].eq(True)

    df["bimodal"] = False
    bimodal_clusters = _identify_bimodal_clusters(df, sales_field)
    df.loc[df["ss_id"].isin(bimodal_clusters), "bimodal"] = True

    the_cols = [
        "key_sale",
        "ss_id",
        "count",
        sales_field,
        base_sales_field,
        "median",
        "chd",
        "stdev",
        "relative_ratio",
        "med_dist_stdevs",
        "flagged",
        "rules",
        "scrutiny_pass",
        "bimodal",
    ] + _ANOMALIES + ["anomaly_other"] + _RATINGS + _SIDE

    df = df[the_cols]

    return df


def _flag_rows(dfs: list[pd.DataFrame], model_group: str) -> pd.DataFrame:
    rows = []
    for df in dfs:
        if "flagged" not in df or len(df) == 0:
            continue
        f = df[df["flagged"].eq(True)].copy()
        if len(f) == 0:
            continue
        area = str(f["area"].iloc[0])
        per = next(c for c in f.columns if c.endswith(f"_{area}") and c.startswith("sale_price"))
        out = pd.DataFrame(
            {
                "key_sale": f["key_sale"].astype(str),
                "key": f["key"].astype(str) if "key" in f else pd.NA,
                "model_group": str(model_group),
                "check": "cluster",
                "vacant": f["vacant"].astype(bool),
                "ss_id": f["ss_id"].astype(str),
                "count": pd.to_numeric(f["count"]).astype("Int64"),
                "price": pd.to_numeric(f[_get_base_sales_field(per)]).astype(float),
                "price_per_area": pd.to_numeric(f[per]).astype(float),
                "area": area,
                "median": pd.to_numeric(f["median"]).astype(float),
                "stdev": pd.to_numeric(f["stdev"]).astype(float),
                "relative_ratio": pd.to_numeric(f["relative_ratio"]).astype(float),
                "med_dist_stdevs": pd.to_numeric(f["med_dist_stdevs"]).astype(float),
                "rules": f["rules"].astype(str),
                "pattern": f[_ANOMALIES].fillna(False).astype(bool).apply(
                    lambda r: ",".join(str(i + 1) for i, v in enumerate(r) if v), axis=1
                ).where(~_flag_only(f), "other"),
                "scrutiny_pass": pd.to_numeric(f["scrutiny_pass"]).astype("Int64"),
                "bimodal": f["bimodal"].astype(bool),
                **{c: f[c].astype(object) for c in _RATINGS},
                **{c: pd.to_numeric(f[c]).astype(float) for c in _SIDE},
                "flag_only": _flag_only(f).astype(bool),
            }
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _mark_sales_scrutiny_clusters(
    df: pd.DataFrame, settings: dict, verbose: bool = False
):
    df_sales = _get_sales(df, settings)

    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    location = ss.get("location", "neighborhood")
    warn_if_location_collapsed(
        settings, location, context="sales scrutiny clustering"
    )
    fields_categorical = ss.get("fields_categorical", [])
    fields_numeric = ss.get("fields_numeric", None)

    unit = area_unit(settings)

    # check if this is a vacant dataset:
    if df_sales["is_vacant"].eq(1).all():
        # if so remove all improved categoricals
        impr_fields = get_fields_categorical(
            settings, df, include_boolean=False, types=["impr"]
        )
        fields_categorical = [f for f in fields_categorical if f not in impr_fields]

    # Get cluster IDs and used fields
    cluster_ids, fields_used, _ = make_clusters(
        df_sales,
        location,
        fields_categorical,
        fields_numeric,
        settings,
        min_cluster_size=5,
        unit=unit,
        verbose=verbose,
    )

    # Ensure cluster IDs are strings
    df_sales["ss_id"] = cluster_ids.astype(str)

    return df_sales, fields_used


def _run_land_percentiles(sup: SalesUniversePair, settings: dict):
    df_sales = get_hydrated_sales_from_sup(sup)

    unit = area_unit(settings)

    df_sales = df_sales[df_sales["vacant_sale"].eq(True)]
    
    df_sales[f"sale_price_time_adj_land_{unit}"] = div_series_z_safe(
        df_sales["sale_price_time_adj"], df_sales[f"land_area_{unit}"]
    )
    locations = get_locations(settings, df_sales)
    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    deed_id = ss.get("deed_id", "deed_id")

    def _do_run_land_percentiles(df: pd.DataFrame, model_group: str):
        df.sort_values(
            by=f"sale_price_time_adj_land_{unit}", inplace=True, ascending=False
        )
        df[f"sale_price_time_adj_land_{unit}_percentile"] = df[
            f"sale_price_time_adj_land_{unit}"
        ].rank(pct=True)
        cols = [
            "key_sale",
            "model_group",
            "sale_date",
            "sale_price",
            "sale_price_time_adj",
            f"sale_price_time_adj_land_{unit}",
            f"sale_price_time_adj_land_{unit}_percentile",
            "deed_book",
            "deed_page",
            deed_id,
            "sale_year",
            "bldg_year_built",
            f"bldg_area_finished_{unit}",
            f"land_area_{unit}",
            "valid_sale",
            "vacant_sale",
            "address",
        ] + locations
        cols = [col for col in cols if col in df]
        df = df[cols]
        os.makedirs(f"out/sales_scrutiny/{model_group}", exist_ok=True)
        df.to_csv(
            f"out/sales_scrutiny/{model_group}/land_price_percentiles.csv", index=False
        )

    do_per_model_group(df_sales, settings, _do_run_land_percentiles, {}, key="key_sale")


def _read_key_file(path: str | None, what: str) -> list[str]:
    if path is None:
        return []
    if not os.path.exists(path):
        warnings.warn(f"--> {what} file {path} does not exist, skipping it")
        return []
    return pd.read_csv(path, dtype={"key_sale": str})["key_sale"].dropna().tolist()


def read_keep_keys(settings: dict) -> set[str]:
    """The sales the user has looked at and kept: sales scrutiny flags them but never drops them.

    Read from the CSV named in ``analysis.sales_scrutiny.keep_file`` (a ``key_sale`` column).

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    set[str]
        The kept sales' ``key_sale`` values; empty without a keep file.
    """
    path = settings.get("analysis", {}).get("sales_scrutiny", {}).get("keep_file", None)
    return set(_read_key_file(path, "Keep"))


def _write_flags(df: pd.DataFrame, name: str) -> None:
    os.makedirs(f"out/sales_scrutiny/{name}", exist_ok=True)
    df.to_parquet(f"out/sales_scrutiny/{name}/flags.parquet", index=False)


def drop_manual_exclusions(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """Drops sales that the user has individually marked as invalid

    Parameters
    ----------
    sup : SalesUniversePair
        The data you want to clean
    settings : dict
        Settings dictionary
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    Returns
    -------
    SalesUniversePair
        The original data with any modifications
    """
    invalid_key_file: str | None = (
        settings.get("analysis", {})
        .get("sales_scrutiny", {})
        .get("invalid_key_file", None)
    )
    if invalid_key_file is not None:
        bad_keys = _read_key_file(invalid_key_file, "Invalid key")
        df_sales = sup.sales.copy()
        len_before = len(df_sales)
        excluded = df_sales[df_sales["key_sale"].isin(bad_keys)]
        _write_flags(
            pd.DataFrame(
                {
                    "key_sale": excluded["key_sale"].astype(str),
                    "key": excluded["key"].astype(str),
                    "check": "excluded",
                }
            ),
            "excluded",
        )
        df_sales = df_sales[~df_sales["key_sale"].isin(bad_keys)]
        len_after = len(df_sales)
        num_dropped = len_before - len_after
        if verbose:
            print("")
            print(f"Dropping {num_dropped} manual exclusions from sales scrutiny")
        sup.sales = df_sales
    else:
        if verbose:
            print("")
            print(
                f"No manual exclusions file specified in settings, skipping manual exclusions"
            )
    return sup


# Words that mark a name as a company, trust, bank or government rather than a person.
SURNAME_SKIP_WORDS = (
    "LLC", "L.L.C.", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LP", "LLP", "LTD", "PLLC",
    "TRUST", "TR", "TRS", "TRUSTEE", "TRUSTEES", "ESTATE", "EST", "BANK", "MORTGAGE", "FINANCIAL", "CREDIT", "LENDING",
    "ASSOCIATION", "ASSN", "HOA", "CHURCH", "MINISTRIES", "COUNTY", "CITY", "STATE", "TOWN", "VILLAGE", "AUTHORITY",
    "DEPARTMENT", "DEPT", "HOUSING", "FEDERAL", "NATIONAL", "UNITED", "SECRETARY", "HUD", "PROPERTIES", "PROPERTY",
    "HOLDINGS", "INVESTMENTS", "INVESTORS", "PARTNERS", "PARTNERSHIP", "GROUP", "REALTY", "HOMES", "BUILDERS",
    "DEVELOPMENT", "ENTERPRISES", "FUND", "FOUNDATION", "UNIVERSITY", "SCHOOL", "DISTRICT", "VENTURES", "CAPITAL",
)
# The surname is the first word ("SMITH JOHN A"). Others: "^\\s*(?P<surname>[^,]+)," ("SMITH, JOHN"), and the last word
# before a suffix ("John A Smith Jr").
SURNAME_FIRST = r"^\s*(?P<surname>[A-Za-z][A-Za-z'\-]*)"
_JOINT = re.compile(r"\s*(?:&|;|/|\bAND\b)\s*", re.IGNORECASE)


def surnames(name, pattern: str = SURNAME_FIRST, skip_words=SURNAME_SKIP_WORDS) -> list[str]:
    """The surnames in a buyer's or seller's name, upper case.

    A name can hold several people, joined by ``&``, ``;``, ``/`` or ``AND`` ("SMITH JOHN & MARY JONES"); each is
    read with ``pattern``, a regular expression with a ``surname`` group. A part of one word is a given name sharing
    another part's surname ("SMITH JOHN & MARY", "JOHN & MARY SMITH": two Smiths), unless no part has more than one
    word ("SMITH"). A part with any of ``skip_words`` (a company, a trust, a bank, a government) has no surname.

    Parameters
    ----------
    name : str
        The name as recorded
    pattern : str, optional
        Where the surname is: a regular expression with a ``surname`` group. Default: the first word.
    skip_words : iterable of str, optional
        Words that mean it isn't a person.

    Returns
    -------
    list[str]
        The surnames found; empty for a blank name, a company, or a name the pattern doesn't match.
    """
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return []
    skip = {w.upper().strip(".") for w in skip_words}
    rx = re.compile(pattern, re.IGNORECASE)
    texts = [t.strip() for t in _JOINT.split(str(name))]
    parts = [(t, re.findall(r"[A-Za-z][A-Za-z'.\-]*", t)) for t in texts]
    parts = [(t, words) for t, words in parts if words]
    if any(w.upper().strip(".") in skip for _, words in parts for w in words):
        return []  # a company, trust, bank or government among them: not people
    longest = max((len(words) for _, words in parts), default=0)
    out: list[str] = []
    for text, words in parts:
        if len(words) == 1 and longest > 1:
            continue  # a given name: its surname is another part's
        m = rx.search(text)
        if not m or not m.groupdict().get("surname"):
            continue
        surname = re.sub(r"[^A-Z'\-]", "", m.group("surname").upper())
        if surname and surname not in out:
            out.append(surname)
    return out


def _same_surname(df: pd.DataFrame, ss: dict) -> tuple[pd.Series, pd.Series]:
    """Buyer and seller share a surname. Surnames from deed_grantor_surname / deed_grantee_surname where those are
    filled, else read from deed_grantor / deed_grantee with analysis.sales_scrutiny.same_surname.pattern."""
    cfg = ss.get("same_surname", {}) or {}
    pattern = cfg.get("pattern") or SURNAME_FIRST
    skip = cfg.get("skip_words", SURNAME_SKIP_WORDS)
    none = pd.Series(False, index=df.index), pd.Series(pd.NA, index=df.index, dtype=object)
    if not any(c in df for c in ("deed_grantor", "deed_grantor_surname")) or not any(
        c in df for c in ("deed_grantee", "deed_grantee_surname")
    ):
        warnings.warn("No buyer and seller names in the sales (deed_grantee, deed_grantor), so the same-surname check can't be run")
        return none

    def names(who: str) -> pd.Series:
        given = df.get(f"deed_{who}_surname")
        parsed = df[f"deed_{who}"].map(lambda n: surnames(n, pattern, skip)) if f"deed_{who}" in df else None
        if given is None:
            return parsed
        mine = given.map(lambda n: surnames(n, r"(?P<surname>.+)", skip))
        return mine.where(mine.map(len) > 0, parsed) if parsed is not None else mine

    seller, buyer = names("grantor"), names("grantee")
    shared = pd.Series([sorted(set(a or []) & set(b or [])) for a, b in zip(seller, buyer)], index=df.index)
    idx = shared.map(len).gt(0)
    return idx, shared.map(lambda x: ", ".join(x) if x else pd.NA)


def _vacant_priced_improved(df: pd.DataFrame, ss: dict) -> tuple[pd.Series, pd.Series]:
    """A vacant sale for at least `share` (default half) of the median improved sale in its model group and
    location (analysis.sales_scrutiny.vacant_priced_improved.location, else sales scrutiny's location), compared
    when there are at least `min_sales` (default 3) improved sales there."""
    cfg = ss.get("vacant_priced_improved", {}) or {}
    share = float(cfg.get("share", 0.5))
    min_sales = int(cfg.get("min_sales", 3))
    location = cfg.get("location", ss.get("location"))
    if isinstance(location, list):
        location = location[0] if location else None
    price = "sale_price_time_adj" if "sale_price_time_adj" in df else "sale_price"
    keys = [c for c in ("model_group", location) if c and c in df]
    if location and location not in df:
        warnings.warn(f"No `{location}` in the sales: vacant sales are compared with their whole model group")
    p = pd.to_numeric(df[price], errors="coerce")
    vacant = df["vacant_sale"].eq(1).fillna(False).astype(bool)
    improved = ~vacant & p.gt(0)
    group = df[keys].astype(str).agg("|".join, axis=1) if keys else pd.Series("", index=df.index)
    medians = p[improved].groupby(group[improved]).median()
    counts = p[improved].groupby(group[improved]).size()
    median = group.map(medians)
    enough = group.map(counts).fillna(0).ge(min_sales)
    idx = vacant & enough & p.ge(share * median)
    said = (p / median).map(lambda r: f"{r:.0%} of the median improved sale" if pd.notna(r) else pd.NA)
    return idx.fillna(False).astype(bool), said


def run_heuristics(
    sup: SalesUniversePair, settings: dict, drop: bool = True, verbose: bool = False
) -> SalesUniversePair:
    """
    Identifies and flags anomalous sales by heuristic. Drops them if the user specifies.

    Parameters
    ----------
    sup : SalesUniversePair
        The data you want to analyze/clean
    settings : dict
        Settings dictionary
    drop : bool, optional
        If True, drops all sales flagged by this method. Default is True.
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    Returns
    -------
    SalesUniversePair
        The original data with any modifications
    """
    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    deed_id = ss.get("deed_id", None)
    unit = area_unit(settings)
    df_sales = get_hydrated_sales_from_sup(sup)

    #### Multi-parcel sales detection heuristics
    # A multi-parcel ("bulk") deed stamps its one bundle price onto every parcel it covers,
    # so none of those per-parcel prices is usable. See openavmkit/utilities/sales_scrutiny.py.

    jurisdiction = ss.get("jurisdiction", None)
    has_deed_id = (deed_id is not None) and (deed_id in df_sales)
    # Each check can be turned off: analysis.sales_scrutiny.heuristics.<name> = false
    on = {name: ss.get("heuristics", {}).get(name, True) is not False for name in HEURISTICS}

    def _set_flag(mask: pd.Series, flag: str):
        if mask.any():
            df_sales.loc[mask, flag] = True

    # The newer checks are off unless turned on: analysis.sales_scrutiny.heuristics.<name> = true
    on |= {name: ss.get("heuristics", {}).get(name, False) is True for name in OPT_IN_HEURISTICS}
    detail = pd.Series(pd.NA, index=df_sales.index, dtype=object)
    
    # 1 -- Flag sales with identical deed IDs AND identical sale dates
    if not on["dupe_deed_date"]:
        pass
    elif has_deed_id:
        _set_flag(flag_dupe_deed_date(df_sales, deed_id, jurisdiction), "flag_dupe_deed_date")
    else:
        if deed_id is None:
            warnings.warn("You didn't provide a `deed_id` in `analysis.sales_scrutiny.deed_id`, so no deed-based sales validation heuristic can be run")
        else:
            warnings.warn(f"You provided a `deed_id`: \"{deed_id}\", but it wasn't found in in the sales dataframe, so no deed-based sales validation heuristic can be run")

    # 2 -- Flag sales made on the same date for the same price. Without deed IDs this is the
    # only bulk-deed detector, so every repeated (date, price) is flagged; with deed IDs,
    # coincidences between separate, known deeds are spared.
    if on["dupe_date_price"]:
        _set_flag(
            flag_dupe_date_price(df_sales, jurisdiction, deed_id if has_deed_id else None),
            "flag_dupe_date_price",
        )

    # 4 -- Flag deeds that covered several parcels at ingestion, even if only one of their
    # rows is left (the others were invalid sales and got filtered out).
    if on["bulk_deed"]:
        _set_flag(flag_bulk_deeds(df_sales), "flag_bulk_deed")

    #### Misclassified vacant sales detection heuristics

    # 3 -- Flag vacant sales with a building year built older than the sale year
    if on["false_vacant"] and "bldg_year_built" not in df_sales:
        warnings.warn("No `bldg_year_built` in the sales, so the sold-as-vacant-but-built-before check can't be run")
    elif on["false_vacant"]:
        idx_false_vacant = (
            df_sales["vacant_sale"].eq(1)
            & df_sales["bldg_year_built"].gt(0)
            & df_sales["bldg_year_built"].lt(df_sales["sale_year"])
        )
        df_sales.loc[idx_false_vacant, "flag_false_vacant"] = True

    # 4 -- Flag improved sales whose building was built after the sale year: likely land when sold
    if on["built_after_sale"] and "bldg_year_built" not in df_sales:
        warnings.warn("No `bldg_year_built` in the sales, so the built-after-the-sale check can't be run")
    elif on["built_after_sale"]:
        idx = (
            df_sales["vacant_sale"].ne(1)
            & df_sales["bldg_year_built"].gt(0)
            & df_sales["bldg_year_built"].gt(df_sales["sale_year"])
        )
        df_sales.loc[idx, "flag_built_after_sale"] = True
        detail[idx] = "built " + df_sales.loc[idx, "bldg_year_built"].astype(int).astype(str)

    # 5 -- Flag vacant sales priced like the improved sales around them: likely had a building
    if on["vacant_priced_improved"]:
        idx, said = _vacant_priced_improved(df_sales, ss)
        df_sales.loc[idx, "flag_vacant_priced_improved"] = True
        detail[idx] = said[idx]

    # 6 -- Flag sales whose buyer and seller share a surname: possibly between family
    if on["same_surname"]:
        idx, said = _same_surname(df_sales, ss)
        df_sales.loc[idx, "flag_same_surname"] = True
        detail[idx] = said[idx]

    files = {
        "flag_dupe_deed_date": "duplicated_deeds_and_dates",
        "flag_dupe_date_price": "duplicated_dates_and_prices",
        "flag_false_vacant": "classified_vacant_but_bldg_older_than_sale_year",
        "flag_bulk_deed": "multi_parcel_bulk_deeds",
        "flag_built_after_sale": "classified_improved_but_bldg_built_after_sale_year",
        "flag_vacant_priced_improved": "classified_vacant_but_priced_like_improved",
        "flag_same_surname": "buyer_and_seller_share_a_surname",
    }

    locations = get_locations(settings, df_sales)

    bad_keys = []

    if verbose:
        print(f"Validating sales by heuristic, {len(df_sales)} total sales")
    
    os.makedirs("out/sales_scrutiny/", exist_ok=True)
    
    for key in files:
        if key in df_sales.columns:
            df = df_sales[df_sales[key].eq(True)]
            if len(df) > 0:
                path = f"out/sales_scrutiny/{files[key]}.xlsx"

                cols = (
                    [
                        "key_sale",
                        "sale_date",
                        "sale_price",
                        "sale_price_time_adj",
                        "deed_book",
                        "deed_page",
                        deed_id,
                        SALE_PARCEL_COUNT,
                        "sale_year",
                        "bldg_year_built",
                        f"bldg_area_finished_{unit}",
                        f"land_area_{unit}",
                        "valid_sale",
                        "vacant_sale",
                        "address",
                    ]
                    + locations
                    + [key]
                )

                cols = [col for col in cols if col in df]
                df = df[cols]

                _bad_keys = df["key_sale"].tolist()

                if verbose:
                    print(f"--> {len(_bad_keys)} bad keys for heuristic: {key}")

                bad_keys = list(set(bad_keys + _bad_keys))

                df = df.rename(
                    columns={
                        "key": "Primary key",
                        "key_sale": "Sale key",
                        "sale_date": "Sale date",
                        "sale_price": "Sale price",
                        "sale_price_time_adj": "Sale price\n(Time adj.)",
                        "deed_book": "Deed book",
                        "deed_page": "Deed page",
                        deed_id: "Deed ID",
                        SALE_PARCEL_COUNT: "Parcels\non deed",
                        "sale_year": "Year sold",
                        "bldg_year_built": "Year built",
                        f"bldg_area_finished_{unit}": f"Impr {unit}",
                        f"land_area_{unit}": f"Land {unit}",
                        "valid_sale": "Valid sale",
                        "vacant_sale": "Vacant sale",
                        "flag_dupe_deed_date": "Repeated\ndeed & sale date",
                        "flag_dupe_date_price": "Repeated\nsale date & price",
                        "flag_false_vacant": "Bldg older\nthan sale year",
                        "flag_bulk_deed": "Multi-parcel\nbulk deed",
                        "flag_built_after_sale": "Bldg built\nafter sale year",
                        "flag_vacant_priced_improved": "Vacant, priced\nlike improved",
                        "flag_same_surname": "Buyer & seller\nshare a surname",
                    }
                )

                _curr_0 = {"num_format": "#,##0"}
                _date = {"num_format": "yyyy-mm-dd"}

                columns = {
                    "Sale date": _date,
                    "Sale price": _curr_0,
                    "Sale price\n(Time adj.)": _curr_0,
                }

                column_conditions = {
                    "Repeated\ndeed ID": {
                        "type": "cell",
                        "criteria": "==",
                        "value": "TRUE",
                        "format": {"bold": True, "font_color": "red"},
                    },
                    "Repeated\nsale date & price": {
                        "type": "cell",
                        "criteria": "==",
                        "value": "TRUE",
                        "format": {"bold": True, "font_color": "red"},
                    },
                    "Bldg older\nthan sale year": {
                        "type": "cell",
                        "criteria": "==",
                        "value": "TRUE",
                        "format": {"bold": True, "font_color": "red"},
                    },
                    "Multi-parcel\nbulk deed": {
                        "type": "cell",
                        "criteria": "==",
                        "value": "TRUE",
                        "format": {"bold": True, "font_color": "red"},
                    },
                }

                write_to_excel(
                    df,
                    path,
                    {"columns": {"formats": columns, "conditions": column_conditions}},
                )

    flag_rows = []
    for key in files:
        if key in df_sales.columns:
            df = df_sales[df_sales[key].eq(True)]
            flag_rows.append(
                pd.DataFrame(
                    {
                        "key_sale": df["key_sale"].astype(str),
                        "key": df["key"].astype(str),
                        "model_group": df["model_group"].astype(str) if "model_group" in df else pd.NA,
                        "check": key.removeprefix("flag_"),
                        "vacant": df["vacant_sale"].eq(1).fillna(False).astype(bool),
                        "price": pd.to_numeric(df["sale_price"]).astype(float),
                        "flag_only": False,
                        "detail": detail[df.index].astype(object),
                    }
                )
            )
    _write_flags(
        pd.concat(flag_rows, ignore_index=True) if flag_rows else pd.DataFrame(columns=["key_sale", "key", "check"]),
        "heuristics",
    )

    keep = read_keep_keys(settings)
    kept = [k for k in bad_keys if k in keep]
    bad_keys = [k for k in bad_keys if k not in keep]
    if kept:
        print(f"Kept {len(kept)} sales identified by heuristic that are listed in the keep file")

    if drop:
        print(f"Dropped {len(bad_keys)} invalid sales keys identified by heuristic")
        df_sales = sup.sales.copy()
        df_sales = df_sales[~df_sales["key_sale"].isin(bad_keys)]
        sup.set("sales", df_sales)
    else:
        print(
            f"Identified {len(bad_keys)} invalid sales keys identified by heuristic, but not dropping them"
        )

    #_run_land_percentiles(sup, settings)

    return sup


def mark_ss_ids_per_model_group(
    df_in: pd.DataFrame, settings: dict, verbose: bool = False
) -> pd.DataFrame:
    """
    Cluster parcels for a sales scrutiny study by assigning sales scrutiny IDs.

    This function processes each model group within the provided dataset,
    identifies clusters of parcels for scrutiny, and writes the cluster identifiers
    into a new field.

    Parameters
    ----------
    df_in : pd.DataFrame
        The data you want to mark
    settings : dict
        Configuration settings.
    verbose : bool, optional
        If True, prints verbose output during processing. Defaults to False.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with marked sales scrutiny IDs.
    """
    # Mark the sales scrutiny ID's
    df = do_per_model_group(
        df_in.copy(),
        settings,
        _mark_ss_ids,
        {"settings": settings, "verbose": verbose},
        key="key_sale",
        instructions={"just_stomp_columns": ["ss_id"]},
    )
    return df


def run_sales_scrutiny_per_model_group(
    df_in: pd.DataFrame, settings: dict, verbose=False
) -> pd.DataFrame:
    """
    Run sales scrutiny analysis for each model group within a SalesUniversePair.

    Parameters
    ----------
    df_in : pd.DataFrame
        The data that you want to analyze
    settings : dict
        Configuration settings.
    verbose : bool, optional
        If True, enables verbose logging. Defaults to False.

    Returns
    -------
    SalesUniversePair
        Updated DataFrame after sales scrutiny analysis.
    """
    # Run sales scrutiny for each model group
    df = do_per_model_group(
        df_in.copy(),
        settings,
        run_sales_scrutiny,
        {"settings": settings, "verbose": verbose},
        key="key_sale",
    )
    return df


def run_sales_scrutiny(
    df_in: pd.DataFrame, settings: dict, model_group: str, verbose=False
) -> pd.DataFrame:
    """
    Run sales scrutiny analysis on an individual model group

    Parameters
    ----------
    df_in : pd.DataFrame
        The data that you want to analyze
    settings : dict
        Configuration settings.
    model_group : str
        The model group you want to analyze
    verbose : bool, optional
        If True, enables verbose logging. Defaults to False.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame after sales scrutiny analysis.
    """
    # run sales validity:
    ss = SalesScrutinyStudy(df_in, settings, model_group=model_group)
    ss.write(f"out/sales_scrutiny/{model_group}")

    # clean sales data:
    return ss.get_scrutinized(df_in, verbose=verbose)


# Private


def _mark_ss_ids(
    df_in: pd.DataFrame, model_group: str, settings: dict, verbose: bool
) -> pd.DataFrame:
    df, _ = _mark_sales_scrutiny_clusters(df_in, settings, verbose)
    if pd.isna(model_group):
        model_group = "UNKNOWN"
    # Ensure both parts are strings before concatenation
    model_group_str = str(model_group)
    df["ss_id"] = model_group_str + "_" + df["ss_id"]
    return df


def _get_ss_renames():
    return {
        "key": "Primary key",
        "key_sale": "Sale key",
        "ss_id": "Sales scrutiny cluster",
        "count": "# of sales in cluster",
        "sale_price": "Sale price",
        "sale_price_impr_sqft": "Sale price / improved sqft",
        "sale_price_impr_sqm": "Sale price / improved sqm",
        "sale_price_land_sqft": "Sale price / land sqft",
        "sale_price_land_sqm": "Sale price / land sqm",
        "sale_price_time_adj": "Sale price (time adjusted)",
        "sale_price_time_adj_impr_sqft": "Sale price / improved sqft (time adjusted)",
        "sale_price_time_adj_impr_sqm": "Sale price / improved sqm (time adjusted)",
        "sale_price_time_adj_land_sqft": "Sale price / land sqft (time adjusted)",
        "sale_price_time_adj_land_sqm": "Sale price / land sqm (time adjusted)",
        "median": "Median",
        "chd": "CHD",
        "max": "Max",
        "min": "Min",
        "stdev": "Standard deviation",
        "relative_ratio": "Relative ratio",
        "med_dist_stdevs": "Median distance from median, in std. deviations",
        "flagged": "Flagged",
        "rules": "Thresholds crossed",
        "scrutiny_pass": "Found in pass",
        "bimodal": "Bimodal cluster",
        "anomaly_1": "Weird price/area & weird area",
        "anomaly_2": "Low price & low price/area",
        "anomaly_3": "High price & high price/area",
        "anomaly_4": "Normal price & high price/area",
        "anomaly_5": "Normal price & low price/area",
    }


def _prettify(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = df.rename(columns=_get_ss_renames())
    renames = _get_ss_renames()
    df = df.rename(columns=renames)
    if "Primary key" in df.columns:
        # rearrange columns so primary key comes first
        df = df[["Primary key"] + [col for col in df.columns.values if col != "Primary key"]]
    df = _apply_dd_to_df_cols(df, settings)
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
        df = df[["key", "key_sale", "ss_id", sales_field, base_sales_field]].copy()
    else:
        df = df[["key", "key_sale", "ss_id", sales_field]].copy()

    if len(df) > 0:
        df = df.merge(df_cluster, on="ss_id", how="left")
        df["relative_ratio"] = div_df_z_safe(df, sales_field, "median")
        df["med_dist_stdevs"] = div_series_z_safe(
            df[sales_field] - df["median"], df["stdev"]
        )

    return df


def _rate(dist: pd.Series, ratio: pd.Series) -> pd.Series:
    """low or high by the thresholds a sale is flagged at: more than 2 standard deviations from the median, or 4x the
    median or more (high), 0.35x or less and more than one deviation below (low); else in (range). The ratios matter
    in a small cluster, whose spread the outlier itself stretches: a $10 deed among $300,000 sales is low."""
    d, r = _floats(dist), _floats(ratio)
    low = d.lt(-2.0) | (r.le(0.35) & d.lt(-1.0))
    high = d.gt(2.0) | r.ge(4.0)
    return pd.Series(np.where(low, "low", np.where(high, "high", "in")), index=dist.index, dtype=object)


def _strict_patterns(price: pd.Series, per_area: pd.Series, size: pd.Series) -> dict[str, pd.Series]:
    """The five patterns from the three readings (low / in / high): 1 size and price per area off; 2-5 size in range,
    by price and price per area. A flagged sale matching none is "other"."""
    size_off = size.isin(["low", "high"])
    per_off = per_area.isin(["low", "high"])
    return {
        "anomaly_1": size_off & per_off,
        "anomaly_2": ~size_off & price.eq("low") & per_area.eq("low"),
        "anomaly_3": ~size_off & price.eq("high") & per_area.eq("high"),
        "anomaly_4": ~size_off & price.eq("in") & per_area.eq("high"),
        "anomaly_5": ~size_off & price.eq("in") & per_area.eq("low"),
    }


def _check_for_anomalies(df_in: pd.DataFrame, df_sales: pd.DataFrame, sales_field: str, strict: bool = False):

    # Limit search only to clusters with flagged sales
    df_flagged = df_in[df_in["flagged"].eq(True)]
    flagged_clusters = df_flagged["ss_id"].unique()

    df = df_in.copy()
    df = df[df["ss_id"].isin(flagged_clusters)]

    if len(df) == 0:
        # Nothing flagged: every sale stays, with no anomalies.
        df = df_in.copy()
        for i in range(1, 6):
            field = f"anomaly_{i}"
            df[field] = False
        df["anomaly_other"] = False
        for c in _RATINGS + _SIDE:
            df[c] = np.nan if c in _SIDE else None
        return df

    # land or bldg area? Check sales_field:
    
    area = ""
    
    if "land" in sales_field:
        if "sqft" in sales_field:
            area = "land_area_sqft"
        elif "sqm" in sales_field:
            area = "land_area_sqm"
    elif "impr" in sales_field:
        if "sqft" in sales_field:
            area = "bldg_area_finished_sqft"
        elif "sqm" in sales_field:
            area = "bldg_area_finished_sqm"
    
    price = "sale_price" if "time_adj" not in sales_field else "sale_price_time_adj"

    if area == "":
        raise ValueError(
            "expected `sales_field` to be suffixed with either `_impr_sqft`/`_land_sqft`, or `_impr_sqm`/`_land_sqm`"
        )

    df_area = _apply_he_stats(df_sales, "ss_id", area)
    df_price = _apply_he_stats(df_sales, "ss_id", price)

    df_fl = df[df["flagged"].eq(True)]

    df_area_fl = df_area[df_area["key_sale"].isin(df_fl["key_sale"].values)]
    df_price_fl = df_price[df_price["key_sale"].isin(df_fl["key_sale"].values)]

    # Check for the symptoms

    # price/area low/high/in range (already done)
    # price low/high/in range
    # area low/high/in range

    idx_price_low = df_price_fl["relative_ratio"].le(1.0)
    idx_price_high = df_price_fl["relative_ratio"].ge(1.0)

    idx_price_low = df["key_sale"].isin(df_price_fl[idx_price_low]["key_sale"].values)
    idx_price_high = df["key_sale"].isin(df_price_fl[idx_price_high]["key_sale"].values)

    idx_price_not_low = df_price_fl["med_dist_stdevs"].ge(-1.0)
    idx_price_not_high = df_price_fl["med_dist_stdevs"].le(1.0)

    idx_price_not_low = df["key_sale"].isin(
        df_price_fl[idx_price_not_low]["key_sale"].values
    )
    idx_price_not_high = df["key_sale"].isin(
        df_price_fl[idx_price_not_high]["key_sale"].values
    )

    idx_area_low = df_area_fl["med_dist_stdevs"].le(-2.0)
    idx_area_high = df_area_fl["med_dist_stdevs"].ge(2.0)

    idx_area_low = df["key_sale"].isin(df_area_fl[idx_area_low]["key_sale"].values)
    idx_area_high = df["key_sale"].isin(df_area_fl[idx_area_high]["key_sale"].values)

    idx_area_not_low = df_area_fl["med_dist_stdevs"].ge(-1.0)
    idx_area_not_high = df_area_fl["med_dist_stdevs"].le(1.0)

    idx_area_not_low = df["key_sale"].isin(
        df_area_fl[idx_area_not_low]["key_sale"].values
    )
    idx_area_not_high = df["key_sale"].isin(
        df_area_fl[idx_area_not_high]["key_sale"].values
    )

    idx_price_area_low = df_fl["relative_ratio"].le(1.0)
    idx_price_area_high = df_fl["relative_ratio"].ge(1.0)

    idx_price_area_low = df["key_sale"].isin(
        df_fl[idx_price_area_low]["key_sale"].values
    )
    idx_price_area_high = df["key_sale"].isin(
        df_fl[idx_price_area_high]["key_sale"].values
    )

    # Check for the five anomalies:

    # 1. Price/area is high or low, area is high or low:
    df.loc[
        (idx_price_area_low | idx_price_area_high) & (idx_area_low | idx_area_high),
        "anomaly_1",
    ] = True

    # 2. Low price, low price/area, area is in range
    df.loc[
        idx_price_low & idx_price_area_low & (idx_area_not_low | idx_area_not_high),
        "anomaly_2",
    ] = True

    # 3. High price, high price/area, area is in range
    df.loc[
        idx_price_high & idx_price_area_high & (idx_area_not_low | idx_area_not_high),
        "anomaly_3",
    ] = True

    # 4. Price in range, high price/area
    df.loc[
        (idx_price_not_low & idx_price_not_high)
        & idx_price_area_high
        & (idx_area_not_low | idx_area_not_high),
        "anomaly_4",
    ] = True

    # 5. Price in range, low price/area
    df.loc[
        (idx_price_not_low & idx_price_not_high)
        & idx_price_area_low
        & (idx_area_not_low | idx_area_not_high),
        "anomaly_5",
    ] = True

    # Each flagged sale's three readings, against its cluster: price, price per area, size.
    by_key = lambda d, col: df["key_sale"].map(d.set_index("key_sale")[col])  # noqa: E731
    ratings = pd.DataFrame(index=df.index)
    for col, d in (("price_rating", df_price_fl), ("per_area_rating", df_fl), ("size_rating", df_area_fl)):
        ratings[col] = _rate(by_key(d, "med_dist_stdevs"), by_key(d, "relative_ratio"))
    ratings["price_median"] = by_key(df_price_fl, "median")
    ratings["price_dist"] = by_key(df_price_fl, "med_dist_stdevs")
    ratings["size"] = by_key(df_area_fl, area)
    ratings["size_median"] = by_key(df_area_fl, "median")
    ratings["size_dist"] = by_key(df_area_fl, "med_dist_stdevs")
    is_flagged = df["key_sale"].isin(df_fl["key_sale"].values)
    for c in _RATINGS:
        ratings.loc[~is_flagged, c] = None
    df["anomaly_other"] = False
    if strict:
        for field, idx in _strict_patterns(*(ratings[c] for c in _RATINGS)).items():
            df[field] = is_flagged & idx.fillna(False)
        matched = pd.concat([df[f].eq(True) for f in _ANOMALIES], axis=1).any(axis=1)
        df["anomaly_other"] = is_flagged & ~matched

    df_out = df_in.copy()
    df_out["anomalies"] = 0
    df_out["anomalies"] = df_out["anomalies"].astype("Int64")
    for i in range(1, 6):
        field = f"anomaly_{i}"
        df_out[field] = False
        df_out[field] = df[field]
        df_out["anomalies"] = df_out["anomalies"] + df[field].astype("Int64").fillna(0)

    df_out["anomaly_other"] = df["anomaly_other"].reindex(df_out.index).eq(True)
    for c in _RATINGS + _SIDE:
        df_out[c] = ratings[c].reindex(df_out.index)
    # Classic: a sale fitting none of the five patterns isn't flagged. Strict: it's "other", flagged only.
    df_out.loc[df_out["anomalies"].le(0) & ~df_out["anomaly_other"], "flagged"] = False

    df_out.drop(columns=["anomalies"], inplace=True)

    return df_out


def _get_base_sales_field(field: str):
    return "sale_price" if "time_adj" not in field else "sale_price_time_adj"


def _identify_bimodal_clusters(df, sales_field):
    """
    Identify clusters whose distribution of `sales_field` is likely bimodal
    using Gaussian Mixture Models with information-criterion + separation checks.

    Criteria:
      - BIC(1) - BIC(2) >= 10  (2 components strongly preferred)
      - Ashman's D > 2.0       (components well-separated)
      - min component weight >= 0.15 (avoid tiny spurious modes)
    """
    bimodal_clusters = []

    for cluster_id, group in df.groupby("ss_id"):
        values = group[sales_field].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        # Need at least a few points to estimate two components sensibly
        if values.size < 8:
            continue

        X = values.reshape(-1, 1)

        # 1 vs 2 components BIC comparison
        gm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=0)
        gm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=0)

        gm1.fit(X)
        gm2.fit(X)

        bic1 = gm1.bic(X)
        bic2 = gm2.bic(X)
        delta_bic = bic1 - bic2  # positive favors 2 components

        if delta_bic < 10.0:
            continue  # not enough evidence for 2 modes

        # Separation: Ashman's D
        means = np.sort(gm2.means_.ravel())
        covars = gm2.covariances_.ravel()  # since full, but 1D -> shape (2,1,1) or (2,)
        # Guard in case of extremely small variances
        covars = np.maximum(covars, 1e-12)
        mu1, mu2 = means
        # Match covariances to sorted means: use argsort on original means
        order = np.argsort(gm2.means_.ravel())
        vars_sorted = covars[order]
        D = (np.sqrt(2.0) * abs(mu2 - mu1)) / np.sqrt(vars_sorted[0] + vars_sorted[1])

        if D <= 2.0:
            continue  # components not well separated

        # Component weights sanity check
        weights = gm2.weights_[order]
        if np.min(weights) < 0.15:
            continue  # one component too small -> likely a tail, not a mode

        bimodal_clusters.append(cluster_id)

    return bimodal_clusters


def make_simple_scrutiny_sheet(sup: SalesUniversePair, settings: dict):
    
    df = get_hydrated_sales_from_sup(sup)
    if "geometry" in df:
        df = df.drop(columns="geometry")
    
    sale_field = get_sale_field(settings)
    
    df_sales["pplf"] = div_df_z_safe(df_sales, sale_field, "land_area_sqft")
    df_sales["ppsf"] = div_df_z_safe(df_sales, sale_field, "bldg_area_finished_sqft")
    
    ids = ["he_id", "land_he_id", "impr_he_id", "ss_id"]
    
    for cluster in ids:
        if cluster in df_stuff:
            df_stuff = df_sales.groupby(cluster)[["pplf","ppsf"]].agg("median").reset_index().rename(columns={
                "pplf": f"{cluster}_pplf",
                "ppsf": f"{cluster}_ppsf"
            })
            df_sales = df_sales.merge(df_stuff, on=cluster, how="left")
            df_sales[f"{cluster}_pplf_ratio"] = div_df_z_safe(df_sales, "pplf", f"{cluster}_pplf")
            df_sales[f"{cluster}_ppsf_ratio"] = div_df_z_safe(df_sales, "ppsf", f"{cluster}_ppsf")
    
    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    location = ss.get("location", "neighborhood")
    
    cols = [
        "key", "key_sale", 
        "sale_date", "sale_year", "sale_age_days",
        "sale_price", "sale_price_time_adj", "pplf", "ppsf",
        "land_area_sqft", "bldg_area_finished_sqft",
        "bldg_age_years", "bldg_year_built",
        "model_group", location, 
        "vacant_sale"
    ] + [
        f"{cluster}_pplf" for cluster in ids
    ] + [
        f"{cluster}_ppsf" for cluster in ids
    ] + [
        f"{cluster}_pplf_ratio" for cluster in ids
    ] + [
        f"{cluster}_ppsf_ratio" for cluster in ids
    ]
    
    cols = [col for col in cols if col in df_sales]
    
    df_sales = df_sales[cols]
    return df_sales
