import os
import warnings
import pandas as pd
from diptest import diptest

from openavmkit.data import (
    get_sales,
    get_sale_field,
    get_important_fields,
    get_locations,
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
from openavmkit.utilities.settings import get_fields_categorical, _apply_dd_to_df_cols


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

    """

    df_vacant: pd.DataFrame
    df_improved: pd.DataFrame
    settings: dict
    model_group: str
    summaries: dict[str, SalesScrutinyStudySummary]

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

        df = df[df["model_group"].eq(model_group)]
        df = get_sales(df, settings)

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
            df["ss_id"] = df["model_group"] + "_" + key + "_" + df["ss_id"].astype(str)
            per_sqft = ""
            denominator = ""
            if key == "i":
                per_sqft = "impr_sqft"
                denominator = "bldg_area_finished_sqft"
            elif key == "v":
                per_sqft = "land_sqft"
                denominator = "land_area_sqft"

            sale_field_per = f"{sale_field}_{per_sqft}"
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

            other_fields = [f for f in other_fields if f in df]

            df_cluster_fields = df[["key_sale"] + other_fields]
            df = _calc_sales_scrutiny(df, sale_field_per)
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
            # remove flagged sales:
            vacant_flags = df_v[df_v["flagged"].eq(True)]["key_sale"].tolist()
            keys_flagged += vacant_flags
            if verbose:
                print(f"--> Flagged {len(vacant_flags)} vacant sales")

        if "flagged" in df_i:
            improved_flags = df_i[df_i["flagged"].eq(True)]["key_sale"].tolist()
            keys_flagged += improved_flags
            if verbose:
                print(f"--> Flagged {len(improved_flags)} improved sales")

        # ensure unique:
        keys_flagged = list(dict.fromkeys(keys_flagged))

        if verbose:
            print(f"--> Flagged {len(keys_flagged)} total sales")

        if len(df) > 0:

            num_valid_sales_before = len(df[df["valid_sale"].eq(True)])

            df.loc[df["key_sale"].isin(keys_flagged), "valid_sale"] = False

            num_valid_sales_after = len(df[df["valid_sale"].eq(True)])

            if verbose:
                print(f"--> Valid sales before: {num_valid_sales_before}")
                print(f"--> Valid sales after: {num_valid_sales_after}")
                diff = num_valid_sales_before - num_valid_sales_after
                print(f"--> Marked {diff} new invalid sales")

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


def _calc_sales_scrutiny(df_in: pd.DataFrame, sales_field: str):
    df = df_in.copy()

    df = _apply_he_stats(df, "ss_id", sales_field)

    if "median" not in df:
        # If the median is not present, then we can't do anything
        # This is usually because there's nothing to analyze
        return df

    base_sales_field = _get_base_sales_field(sales_field)

    # Calculate standard deviation thresholds:
    df["low_thresh"] = -float("inf")
    df["high_thresh"] = float("inf")
    df["low_thresh"] = df["low_thresh"].astype("Float64")
    df["high_thresh"] = df["high_thresh"].astype("Float64")

    # Flag anything above or below 2 standard deviations from the median
    df.loc[~df["median"].isna() & ~df["stdev"].isna(), "low_thresh"] = df["median"] - (
        2 * df["stdev"]
    )
    df.loc[~df["median"].isna() & ~df["stdev"].isna(), "high_thresh"] = df["median"] + (
        2 * df["stdev"]
    )

    idx_low = df[sales_field].lt(df["low_thresh"])
    idx_high = df[sales_field].gt(df["high_thresh"])

    df["flagged"] = False
    df.loc[idx_low | idx_high, "flagged"] = True

    # Additionally, flag anything with a relative ratio >= 4.0
    df.loc[df["relative_ratio"].ge(4.0), "flagged"] = True

    # Additionally, flag anything with a relative ratio <= .35 AND a stdev distance of < -1.0
    df.loc[
        df["relative_ratio"].le(0.35) & df["med_dist_stdevs"].lt(-1.0), "flagged"
    ] = True

    # Check for the five anomalies:
    df = _check_for_anomalies(df, df_in, sales_field)

    df["bimodal"] = False
    bimodal_clusters = _identify_bimodal_clusters(df, sales_field)
    df.loc[df["ss_id"].isin(bimodal_clusters), "bimodal"] = True

    # drop low_thresh/high_thresh:
    df = df.drop(columns=["low_thresh", "high_thresh"])

    the_cols = [
        "key_sale",
        "ss_id",
        "count",
        sales_field,
        base_sales_field,
        "chd",
        "stdev",
        "relative_ratio",
        "med_dist_stdevs",
        "flagged",
        "bimodal",
        "anomaly_1",
        "anomaly_2",
        "anomaly_3",
        "anomaly_4",
        "anomaly_5",
    ]

    df = df[the_cols]

    return df


def _mark_sales_scrutiny_clusters(
    df: pd.DataFrame, settings: dict, verbose: bool = False
):
    df_sales = get_sales(df, settings)

    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    location = ss.get("location", "neighborhood")
    fields_categorical = ss.get("fields_categorical", [])
    fields_numeric = ss.get("fields_numeric", None)

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
        min_cluster_size=5,
        verbose=verbose,
    )

    # Ensure cluster IDs are strings
    df_sales["ss_id"] = cluster_ids.astype(str)

    return df_sales, fields_used


def _run_land_percentiles(sup: SalesUniversePair, settings: dict):
    df_sales = get_hydrated_sales_from_sup(sup)

    df_sales = df_sales[df_sales["vacant_sale"].eq(True)]

    df_sales["sale_price_time_adj_land_sqft"] = div_series_z_safe(
        df_sales["sale_price_time_adj"], df_sales["land_area_sqft"]
    )
    locations = get_locations(settings, df_sales)

    def _do_run_land_percentiles(df: pd.DataFrame, model_group: str):
        df.sort_values(
            by="sale_price_time_adj_land_sqft", inplace=True, ascending=False
        )
        df["sale_price_time_adj_land_sqft_percentile"] = df[
            "sale_price_time_adj_land_sqft"
        ].rank(pct=True)
        cols = [
            "key_sale",
            "sale_date",
            "sale_price",
            "sale_price_time_adj",
            "sale_price_time_adj_land_sqft",
            "sale_price_time_adj_land_sqft_percentile",
            "deed_book",
            "deed_page",
            "deed_id",
            "sale_year",
            "bldg_year_built",
            "bldg_area_finished_sqft",
            "land_area_sqft",
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
        if os.path.exists(invalid_key_file):
            df_invalid_keys = pd.read_csv(invalid_key_file, dtype={"key_sale": str})
            bad_keys = df_invalid_keys["key_sale"].tolist()
        else:
            warnings.warn(
                f"--> Invalid key file {invalid_key_file} does not exist, skipping manual exclusions"
            )
            bad_keys = []
        df_sales = sup.sales.copy()
        len_before = len(df_sales)
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
    deed_id = ss.get("deed_id", "deed_id")

    df_sales = get_hydrated_sales_from_sup(sup)

    #### Multi-parcel sales detection heuristics

    # 1 -- Flag sales with identical deed ids
    vcs = df_sales[deed_id].value_counts()
    idx_dupe_deeds = vcs[vcs > 1].index.values
    df_sales.loc[df_sales[deed_id].isin(idx_dupe_deeds), "flag_dupe_deed"] = True

    # 2 -- Flag sales made on the same date for the same price
    vcs = df_sales["sale_date"].value_counts()
    idx_dupe_dates = vcs[vcs > 1].index.values
    vcs = df_sales["sale_price"].value_counts()
    idx_dupe_prices = vcs[vcs > 1].index.values
    df_sales.loc[
        df_sales["sale_date"].isin(idx_dupe_dates)
        & df_sales["sale_price"].isin(idx_dupe_prices),
        "flag_dupe_date_price",
    ] = True

    #### Misclassified vacant sales detection heuristics

    # 3 -- Flag vacant sales with a building year built older than the sale year
    idx_false_vacant = (
        df_sales["vacant_sale"].eq(1)
        & df_sales["bldg_year_built"].gt(0)
        & df_sales["bldg_year_built"].lt(df_sales["sale_year"])
    )
    df_sales.loc[idx_false_vacant, "flag_false_vacant"] = True

    files = {
        "flag_dupe_deed": "duplicated_deeds",
        "flag_dupe_date_price": "duplicated_dates_and_prices",
        "flag_false_vacant": "classified_vacant_but_bldg_older_than_sale_year",
    }

    locations = get_locations(settings, df_sales)

    bad_keys = []

    if verbose:
        print(f"Validating sales by heuristic, {len(df_sales)} total sales")

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
                        "deed_id",
                        "sale_year",
                        "bldg_year_built",
                        "bldg_area_finished_sqft",
                        "land_area_sqft",
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
                        "key_sale": "Sale key",
                        "sale_date": "Sale date",
                        "sale_price": "Sale price",
                        "sale_price_time_adj": "Sale price\n(Time adj.)",
                        "deed_book": "Deed book",
                        "deed_page": "Deed page",
                        "deed_id": "Deed ID",
                        "sale_year": "Year sold",
                        "bldg_year_built": "Year built",
                        "bldg_area_finished_sqft": "Impr sqft",
                        "land_area_sqft": "Land sqft",
                        "valid_sale": "Valid sale",
                        "vacant_sale": "Vacant sale",
                        "flag_dupe_deed": "Repeated\ndeed ID",
                        "flag_dupe_date_price": "Repeated\nsale date & price",
                        "flag_false_vacant": "Bldg older\nthan sale year",
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
                }

                write_to_excel(
                    df,
                    path,
                    {"columns": {"formats": columns, "conditions": column_conditions}},
                )

    if drop:
        print(f"Dropped {len(bad_keys)} invalid sales keys identified by heuristic")
        df_sales = sup.sales.copy()
        df_sales = df_sales[~df_sales["key_sale"].isin(bad_keys)]
        sup.df_sales = df_sales
    else:
        print(
            f"Identified {len(bad_keys)} invalid sales keys identified by heuristic, but not dropping them"
        )

    _run_land_percentiles(sup, settings)

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
        "anomaly_5": "Normal price & low price/sqft",
    }


def _prettify(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = df.rename(columns=_get_ss_renames())
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
        df = df[["key_sale", "ss_id", sales_field, base_sales_field]].copy()
    else:
        df = df[["key_sale", "ss_id", sales_field]].copy()

    if len(df) > 0:
        df = df.merge(df_cluster, on="ss_id", how="left")
        df["relative_ratio"] = div_df_z_safe(df, sales_field, "median")
        df["med_dist_stdevs"] = div_series_z_safe(
            df[sales_field] - df["median"], df["stdev"]
        )

    return df


def _check_for_anomalies(df_in: pd.DataFrame, df_sales: pd.DataFrame, sales_field: str):

    # Limit search only to clusters with flagged sales
    df_flagged = df_in[df_in["flagged"].eq(True)]
    flagged_clusters = df_flagged["ss_id"].unique()

    df = df_in.copy()
    df = df[df["ss_id"].isin(flagged_clusters)]

    if len(df) == 0:
        df["anomalies"] = 0
        for i in range(1, 6):
            field = f"anomaly_{i}"
            df[field] = False
        return df

    # land or sqft? Check sales_field:
    sqft = (
        "land_area_sqft"
        if "land" in sales_field
        else "bldg_area_finished_sqft" if "impr" in sales_field else ""
    )
    price = "sale_price" if "time_adj" not in sales_field else "sale_price_time_adj"

    if sqft == "":
        raise ValueError(
            "expected `sales_field` to be suffixed with either `_impr_sqft` or `_land_sqft`"
        )

    df_sqft = _apply_he_stats(df_sales, "ss_id", sqft)
    df_price = _apply_he_stats(df_sales, "ss_id", price)

    df_fl = df[df["flagged"].eq(True)]

    df_sqft_fl = df_sqft[df_sqft["key_sale"].isin(df_fl["key_sale"].values)]
    df_price_fl = df_price[df_price["key_sale"].isin(df_fl["key_sale"].values)]

    # Check for the symptoms

    # price/sqft low/high/in range (already done)
    # price low/high/in range
    # sqft low/high/in range

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

    idx_sqft_low = df_sqft_fl["med_dist_stdevs"].le(-2.0)
    idx_sqft_high = df_sqft_fl["med_dist_stdevs"].ge(2.0)

    idx_sqft_low = df["key_sale"].isin(df_sqft_fl[idx_sqft_low]["key_sale"].values)
    idx_sqft_high = df["key_sale"].isin(df_sqft_fl[idx_sqft_high]["key_sale"].values)

    idx_sqft_not_low = df_sqft_fl["med_dist_stdevs"].ge(-1.0)
    idx_sqft_not_high = df_sqft_fl["med_dist_stdevs"].le(1.0)

    idx_sqft_not_low = df["key_sale"].isin(
        df_sqft_fl[idx_sqft_not_low]["key_sale"].values
    )
    idx_sqft_not_high = df["key_sale"].isin(
        df_sqft_fl[idx_sqft_not_high]["key_sale"].values
    )

    idx_price_sqft_low = df_fl["relative_ratio"].le(1.0)
    idx_price_sqft_high = df_fl["relative_ratio"].ge(1.0)

    idx_price_sqft_low = df["key_sale"].isin(
        df_fl[idx_price_sqft_low]["key_sale"].values
    )
    idx_price_sqft_high = df["key_sale"].isin(
        df_fl[idx_price_sqft_high]["key_sale"].values
    )

    # Check for the five anomalies:

    # 1. Price/sqft is high or low, sqft is high or low:
    df.loc[
        (idx_price_sqft_low | idx_price_sqft_high) & (idx_sqft_low | idx_sqft_high),
        "anomaly_1",
    ] = True

    # 2. Low price, low price/sqft, sqft is in range
    df.loc[
        idx_price_low & idx_price_sqft_low & (idx_sqft_not_low | idx_sqft_not_high),
        "anomaly_2",
    ] = True

    # 3. High price, high price/sqft, sqft is in range
    df.loc[
        idx_price_high & idx_price_sqft_high & (idx_sqft_not_low | idx_sqft_not_high),
        "anomaly_3",
    ] = True

    # 4. Price in range, high price/sqft
    df.loc[
        (idx_price_not_low & idx_price_not_high)
        & idx_price_sqft_high
        & (idx_sqft_not_low | idx_sqft_not_high),
        "anomaly_4",
    ] = True

    # 5. Price in range, low price/sqft
    df.loc[
        (idx_price_not_low & idx_price_not_high)
        & idx_price_sqft_low
        & (idx_sqft_not_low | idx_sqft_not_high),
        "anomaly_5",
    ] = True

    df_out = df_in.copy()
    df_out["anomalies"] = 0
    df_out["anomalies"] = df_out["anomalies"].astype("Int64")
    for i in range(1, 6):
        field = f"anomaly_{i}"
        df_out[field] = False
        df_out[field] = df[field]
        df_out["anomalies"] = df_out["anomalies"] + df[field].astype("Int64").fillna(0)

    df_out.loc[df_out["anomalies"].le(0), "flagged"] = False

    df_out.drop(columns=["anomalies"], inplace=True)

    return df_out


def _get_base_sales_field(field: str):
    return "sale_price" if "time_adj" not in field else "sale_price_time_adj"


def _identify_bimodal_clusters(df, sales_field):
    bimodal_clusters = []

    for cluster_id, group in df.groupby("ss_id"):
        values = group[sales_field].values
        if len(values) > 3:
            dip, p_value = diptest(values)
            if p_value < 0.05:  # Statistically significant deviation from unimodality
                bimodal_clusters.append(cluster_id)

    return bimodal_clusters
