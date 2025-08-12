"""
Pipeline
---------
This module contains every public function that is called from the notebooks in the openavmkit project.

Rules:

- Every public function should be called from at least one notebook.
- The primary openavmkit notebooks should only call functions from this module.
- This module imports from other modules, but no other modules import from it.
"""

import os
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd

import openavmkit
import openavmkit.data
import openavmkit.benchmark
import openavmkit.land
import openavmkit.checkpoint
import openavmkit.ratio_study
import openavmkit.horizontal_equity_study
import openavmkit.cleaning

from openavmkit.cleaning import clean_valid_sales, validate_arms_length_sales
from openavmkit.cloud import cloud
from openavmkit.data import (
    _load_dataframe,
    process_data,
    SalesUniversePair,
    get_hydrated_sales_from_sup,
)
from openavmkit.sales_scrutiny_study import (
    run_sales_scrutiny_per_model_group,
    mark_ss_ids_per_model_group,
    run_heuristics,
    drop_manual_exclusions,
)
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import combine_dfs
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_fields_numeric,
    get_fields_boolean,
    get_fields_land,
    get_fields_impr,
    get_fields_other,
    get_valuation_date,
)


# Basic data stuff


class NotebookState:
    """Represents the state of a notebook session including the base path and locality.

    Attributes
    ----------
        base_path : str
            The base directory path for the notebook.
        locality : str
            The locality identifier (e.g., "us-nc-guilford").
    """

    base_path: str
    locality: str

    def __init__(self, locality: str, base_path: str = None):
        """Initialize a NotebookState instance.

        Attributes
        ----------
        locality: str
            The locality slug (e.g., "us-nc-guilford").
        base_path : str
            The base directory path. Defaults to the current working directory if not provided.
        """
        self.locality = locality
        if base_path is None:
            base_path = os.getcwd()
        self.base_path = base_path


def init_notebook(locality: str):
    """Initialize the notebook environment for a specific locality.

    This function sets up the notebook state by configuring the working directory and
    ensuring that the appropriate data directories exist.

    Attributes
    ----------
    locality : str
        The locality slug (e.g., "us-nc-guilford").
    """
    first_run = False
    if hasattr(init_notebook, "nbs"):
        nbs = init_notebook.nbs
    else:
        nbs = None
        first_run = True
    nbs = _set_locality(nbs, locality)
    if first_run:
        init_notebook.nbs = nbs

        # Fix warnings too
        oldformatwarning = warnings.formatwarning

        # Customize warning format
        def custom_formatwarning(msg, category, filename, lineno, line):
            # if it's a user warning:
            if issubclass(category, UserWarning):
                return f"UserWarning: {msg}\n"
            else:
                return oldformatwarning(msg, category, filename, lineno, line)

        warnings.formatwarning = custom_formatwarning


def load_settings(
    settings_file: str = "in/settings.json", settings_object: dict = None, error : bool = True, warning : bool = True
) -> dict:
    """
    Load and return the settings dictionary for the locality.

    This merges the user's settings for their specific locality with the default settings
    template and the default data dictionary. It also performs variable substitution.
    The result is a fully resolved settings dictionary.

    Parameters
    ----------
    settings_file : str, optional
        Path to the settings file. Defaults to "in/settings.json".
    settings_object : dict, optional
        Optional settings object to use instead of loading from a file.
    error : bool, optional
        If True, raises an error if the settings file cannot be loaded. Defaults to True.
    warning : bool, optional
        If True, raises a warning if the settings file cannot be loaded. Defaults to True.

    Returns
    -------
    dict
        The fully resolved settings dictionary.
    """

    return openavmkit.utilities.settings.load_settings(settings_file, settings_object, error, warning)


def examine_sup_in_ridiculous_detail(sup: SalesUniversePair, s: dict):
    """
    Print details of the sales and universe data from a SalesUniversePair,
    but in RIDICULOUS DETAIL.

    Parameters
    ----------
    sup : SalesUniversePair
        Object containing 'sales' and 'universe' DataFrames.
    s : dict
        Settings dictionary.
    """
    print("")
    print("EXAMINING UNIVERSE...")
    print("")
    examine_df_in_ridiculous_detail(sup["universe"], s)

    print("")
    print("EXAMINING SALES...")
    print("")
    examine_df_in_ridiculous_detail(sup["sales"], s)


def examine_sup(sup: SalesUniversePair, s: dict) -> None:
    """
    Print examination details of the sales and universe data from a SalesUniversePair.

    This function displays summary statistics and unique values for both the sales and
    universe DataFrames contained in the provided SalesUniversePair.

    Parameters
    ----------
    sup : SalesUniversePair
        Object containing 'sales' and 'universe' DataFrames.
    s : dict
        Settings dictionary.
    """

    print("")
    print("EXAMINING UNIVERSE...")
    print("")
    examine_df(sup["universe"], s)

    print("")
    print("EXAMINING SALES...")
    print("")
    examine_df(sup["sales"], s)


def examine_df_in_ridiculous_detail(df: pd.DataFrame, s: dict):
    """
    Print details of the dataframe, but in RIDICULOUS DETAIL

    Parameters
    ----------
    df : pd.DataFrame
        The data you wish to examine
    s : dict
        Settings dictionary.
    """

    def fill_str(char: str, size: int):
        text = ""
        for _i in range(0, size):
            text += char
        return text

    def fit_str(txt: str, size: int):
        if len(txt) >= size:
            len_first = int((size - 3) / 2)
            len_last = (size - 3) - len_first
            first_bit = txt[0:len_first]
            last_bit = txt[len(txt) - len_last :]
            txt = first_bit + "..." + last_bit
        return f"{txt:{size}}"

    def get_num_line(col):
        describe = df[col].describe()
        return f"DESCRIBE --> {describe}\n\n"

    def get_cat_line(col):
        value_counts = df[col].value_counts()
        return f"VALUE COUNTS --> {value_counts}\n\n"

    def get_line(
        col, dtype, count_non_zero, p, count_non_null, pnn, uniques: list or str
    ):
        dtype = f"{dtype}"
        if type(count_non_zero) != str:
            count_non_zero = f"{count_non_zero:,}"

        if type(count_non_null) != str:
            count_non_null = f"{count_non_null:,}"

        if isinstance(uniques, list):
            unique_str = str(uniques)
            if len(unique_str) > 40:
                uniques = f"{len(uniques):,}"
            else:
                uniques = unique_str

        return f"{fit_str(col, 30)} {dtype:^10} {count_non_zero:>10} {p:>5.0%} {count_non_null:>10} {pnn:>5.0%} {uniques:>40}"

    def print_horz_line(char: str):
        print(
            fill_str(char, 30)
            + " "
            + fill_str(char, 10)
            + " "
            + fill_str(char, 10)
            + " "
            + fill_str(char, 5)
            + " "
            + fill_str(char, 10)
            + " "
            + fill_str(char, 5)
            + " "
            + fill_str(char, 40)
        )

    print(
        f"{'FIELD':^30} {'TYPE':^10} {'NON-ZERO':^10} {'%':^5} {'NON-NULL':^10} {'%':^5} {'UNIQUE':^40}"
    )

    fields_land = get_fields_land(s, df)
    fields_impr = get_fields_impr(s, df)
    fields_other = get_fields_other(s, df)

    fields_noted = []

    stuff = {
        "land": {"name": "LAND", "fields": fields_land},
        "impr": {"name": "IMPROVEMENT", "fields": fields_impr},
        "other": {"name": "OTHER", "fields": fields_other},
    }

    i = 0

    for landimpr in stuff:
        entry = stuff[landimpr]
        name = entry["name"]

        fields = entry["fields"]
        nums = fields["numeric"]
        bools = fields["boolean"]
        cats = fields["categorical"]

        if (len(nums) + len(bools) + len(cats)) == 0:
            continue

        if i != 0:
            print("")

        print_horz_line("=")
        print(f"{name:^30}")
        print_horz_line("=")

        nums.sort()
        bools.sort()
        cats.sort()

        if len(nums) > 0:
            print_horz_line("-")
            print(f"{'NUMERIC':^30}")
            print_horz_line("-")
            for n in nums:
                fields_noted.append(n)
                df_non_null = df[~pd.isna(df[n])]
                non_zero = len(df_non_null[np.abs(df_non_null[n]).gt(0)])
                perc = non_zero / len(df)
                non_null = len(df_non_null)
                perc_non_null = non_null / len(df)
                print(
                    get_line(
                        n, df[n].dtype, non_zero, perc, non_null, perc_non_null, ""
                    )
                )
                print(get_num_line(n))

        if len(bools) > 0:
            print_horz_line("-")
            print(f"{'BOOLEAN':^30}")
            print_horz_line("-")
            for b in bools:
                fields_noted.append(b)
                df_non_null = df[~pd.isna(df[b])]
                non_zero = len(df_non_null[np.abs(df_non_null[b]).gt(0)])
                perc = non_zero / len(df)
                non_null = len(df_non_null)
                perc_non_null = non_null / len(df)
                print(
                    get_line(
                        b,
                        df[b].dtype,
                        non_zero,
                        perc,
                        non_null,
                        perc_non_null,
                        df[b].unique().tolist(),
                    )
                )

        if len(cats) > 0:
            print_horz_line("-")
            print(f"{'CATEGORICAL':^30}")
            print_horz_line("-")
            for c in cats:
                fields_noted.append(c)
                non_zero = (~pd.isna(df[c])).sum()
                perc = non_zero / len(df)
                print(
                    get_line(
                        c,
                        df[c].dtype,
                        non_zero,
                        perc,
                        non_zero,
                        perc,
                        df[c].unique().tolist(),
                    )
                )
                print(get_cat_line(c))

        i += 1

    fields_unclassified = []

    for column in df.columns:
        if column not in fields_noted:
            fields_unclassified.append(column)

    if len(fields_unclassified) > 0:
        fields_unclassified.sort()
        print("")
        print_horz_line("=")
        print(f"{'UNCLASSIFIED:':<30}")
        print_horz_line("=")
        for u in fields_unclassified:
            non_zero = (~pd.isna(df[u])).sum()
            perc = non_zero / len(df)
            perc_non_null = non_zero / len(df)
            print(
                get_line(
                    u, df[u].dtype, non_zero, perc, non_zero, perc, list(df[u].unique())
                )
            )


def examine_df(df: pd.DataFrame, s: dict):
    """
    Print examination details of the dataframe.
    This function displays summary statistics and unique values.

    Parameters
    ----------
    df : pd.DataFrame
        The data you wish to examine
    s : dict
        Settings dictionary.
    """

    def fill_str(char: str, size: int):
        text = ""
        for _i in range(0, size):
            text += char
        return text

    def fit_str(txt: str, size: int):
        if len(txt) >= size:
            len_first = int((size - 3) / 2)
            len_last = (size - 3) - len_first
            first_bit = txt[0:len_first]
            last_bit = txt[len(txt) - len_last :]
            txt = first_bit + "..." + last_bit
        return f"{txt:{size}}"

    def get_line(
        col, dtype, count_non_zero, p, count_non_null, pnn, uniques: list or str
    ):
        dtype = f"{dtype}"
        if type(count_non_zero) != str:
            count_non_zero = f"{count_non_zero:,}"

        if type(count_non_null) != str:
            count_non_null = f"{count_non_null:,}"

        if isinstance(uniques, list):
            unique_str = str(uniques)
            if len(unique_str) > 40:
                uniques = f"{len(uniques):,}"
            else:
                uniques = unique_str

        return f"{fit_str(col, 30)} {dtype:^10} {count_non_zero:>10} {p:>5.0%} {count_non_null:>10} {pnn:>5.0%} {uniques:>40}"

    buffer = ""
    lines = 0
    chunk_size = 3

    def print_horz_line(char: str):
        nonlocal buffer
        nonlocal lines
        if buffer != "":
            buffer += "\n"
        buffer += (
            fill_str(char, 30)
            + " "
            + fill_str(char, 10)
            + " "
            + fill_str(char, 10)
            + " "
            + fill_str(char, 5)
            + " "
            + fill_str(char, 10)
            + " "
            + fill_str(char, 5)
            + " "
            + fill_str(char, 40)
        )
        lines += 1
        if lines >= chunk_size:
            print(buffer)
            lines = 0
            buffer = ""

    def print_buffer(text: str):
        nonlocal buffer
        nonlocal lines
        if buffer != "":
            buffer += "\n"
        buffer += text
        lines += 1
        if lines >= chunk_size:
            print(buffer)
            buffer = ""
            lines = 0

    print(
        f"{'FIELD':^30} {'TYPE':^10} {'NON-ZERO':^10} {'%':^5} {'NON-NULL':^10} {'%':^5} {'UNIQUE':^40}"
    )

    fields_land = get_fields_land(s, df)
    fields_impr = get_fields_impr(s, df)
    fields_other = get_fields_other(s, df)

    fields_noted = []

    stuff = {
        "land": {"name": "LAND", "fields": fields_land},
        "impr": {"name": "IMPROVEMENT", "fields": fields_impr},
        "other": {"name": "OTHER", "fields": fields_other},
    }

    i = 0

    for landimpr in stuff:
        entry = stuff[landimpr]
        name = entry["name"]

        fields = entry["fields"]
        nums = fields["numeric"]
        bools = fields["boolean"]
        cats = fields["categorical"]

        if (len(nums) + len(bools) + len(cats)) == 0:
            continue

        if i != 0:
            print_buffer("")

        print_horz_line("=")
        print_buffer(f"{name:^30}")
        print_horz_line("=")

        nums.sort()
        bools.sort()
        cats.sort()

        if len(nums) > 0:
            print_horz_line("-")
            print_buffer(f"{'NUMERIC':^30}")
            print_horz_line("-")
            for n in nums:
                fields_noted.append(n)
                df_non_null = df[~pd.isna(df[n])]
                non_zero = len(df_non_null[np.abs(df_non_null[n]).gt(0)])
                perc = non_zero / len(df)
                non_null = len(df_non_null)
                perc_non_null = non_null / len(df)
                print_buffer(
                    get_line(
                        n, df[n].dtype, non_zero, perc, non_null, perc_non_null, ""
                    )
                )

        if len(bools) > 0:
            print_horz_line("-")
            print_buffer(f"{'BOOLEAN':^30}")
            print_horz_line("-")
            for b in bools:
                fields_noted.append(b)
                df_non_null = df[~pd.isna(df[b])]
                non_zero = len(df_non_null[np.abs(df_non_null[b]).gt(0)])
                perc = non_zero / len(df)
                non_null = len(df_non_null)
                perc_non_null = non_null / len(df)
                print_buffer(
                    get_line(
                        b,
                        df[b].dtype,
                        non_zero,
                        perc,
                        non_null,
                        perc_non_null,
                        df[b].unique().tolist(),
                    )
                )

        if len(cats) > 0:
            print_horz_line("-")
            print_buffer(f"{'CATEGORICAL':^30}")
            print_horz_line("-")
            for c in cats:
                fields_noted.append(c)
                non_zero = (~pd.isna(df[c])).sum()
                perc = non_zero / len(df)
                print_buffer(
                    get_line(
                        c,
                        df[c].dtype,
                        non_zero,
                        perc,
                        non_zero,
                        perc,
                        df[c].unique().tolist(),
                    )
                )
        i += 1

    fields_unclassified = []

    for column in df.columns:
        if column not in fields_noted:
            fields_unclassified.append(column)

    if len(fields_unclassified) > 0:
        fields_unclassified.sort()
        print_buffer("")
        print_horz_line("=")
        print_buffer(f"{'UNCLASSIFIED:':<30}")
        print_horz_line("=")
        for u in fields_unclassified:
            non_zero = (~pd.isna(df[u])).sum()
            perc = non_zero / len(df)
            perc_non_null = non_zero / len(df)
            print_buffer(
                get_line(
                    u, df[u].dtype, non_zero, perc, non_zero, perc, list(df[u].unique())
                )
            )

    if len(buffer) > 0:
        print(buffer)
        buffer = ""
        lines = 0


# Data loading & processing stuff


def load_dataframes(settings: dict, verbose: bool = False) -> dict:
    """
    Load dataframes based on the provided settings and return them in a dictionary.

    This function reads various data sources defined in the settings and loads them into
    pandas DataFrames. It performs validations to ensure required data, such as
    'geo_parcels', is present and correctly formatted.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints detailed logs during data loading. Defaults to False.

    Returns
    -------
    dict
        Dictionary mapping keys to loaded DataFrames.

    Raises
    ------
    ValueError
        If required dataframes or columns (e.g., 'geo_parcels' or its 'geometry' column) are missing.
    """

    s_data = settings.get("data", {})
    s_load = s_data.get("load", {})
    dataframes = {}

    fields_cat = get_fields_categorical(settings, include_boolean=False)
    fields_bool = get_fields_boolean(settings)
    fields_num = get_fields_numeric(settings, include_boolean=False)

    for key in s_load:
        entry = s_load[key]
        df = _load_dataframe(
            entry,
            settings,
            verbose=verbose,
            fields_cat=fields_cat,
            fields_bool=fields_bool,
            fields_num=fields_num,
        )
        if df is not None:
            dataframes[key] = df

    if "geo_parcels" not in dataframes:
        raise ValueError(
            "No 'geo_parcels' dataframe found in the dataframes. This layer is required, and it must contain parcel geometry."
        )

    if "geometry" not in dataframes["geo_parcels"].columns:
        raise ValueError(
            "The 'geo_parcels' dataframe does not contain a 'geometry' column. This layer must contain parcel geometry."
        )

    return dataframes


def load_and_process_data(settings: dict) -> SalesUniversePair:
    """
    Load and process data according to provided settings.

    This function first loads the dataframes, then merges and enriches the data,
    returning a SalesUniversePair.

    Parameters
    ----------
    settings : dict
        A dictionary of settings for data loading and processing.

    Returns
    -------
    SalesUniversePair
        A SalesUniversePair object containing the processed sales and universe data.
    """

    dataframes = load_dataframes(settings)
    results = process_data(dataframes, settings)
    return results


def tag_model_groups_sup(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """
    Tag model groups for a SalesUniversePair.

    This function applies user-specified filters that identify rows belonging to
    particular model groups, then writes the results to the `model_group` field.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    verbose : bool, optional
        If True, enables verbose output.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair with tagged model groups.
    """
    return openavmkit.data._tag_model_groups_sup(sup, settings, verbose)


def process_sales(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """
    Process sales data within a SalesUniversePair.

    This function cleans invalid sales, applies time adjustments, and updates the
    SalesUniversePair with the enriched sales DataFrame.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    verbose : bool, optional
        If True, prints verbose output during processing. Defaults to False.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair with processed sales data.
    """

    # select only valid sales
    sup = clean_valid_sales(sup, settings)

    print(f"len before validate = {len(sup['sales'])}")

    # validate arms length sales using filters
    sup = validate_arms_length_sales(sup, settings, verbose)

    print(f"len after validate = {len(sup['sales'])}")

    # make sure sales field has necessary fields for the next step
    df_sales_hydrated = get_hydrated_sales_from_sup(sup)

    print(f"len after hydrate = {len(sup['sales'])}")

    # enrich with time adjustment, and mark what fields were added
    df_sales_enriched = enrich_time_adjustment(df_sales_hydrated, settings, verbose)

    print(f"len after enrich = {len(df_sales_enriched)}")

    df_sales_clipped = _clip_sales_to_use(df_sales_enriched, settings, verbose)

    print(f"len after clip = {len(df_sales_clipped)}")

    # update the SUP sales
    sup.update_sales(df_sales_clipped, allow_remove_rows=True)

    return sup


def enrich_sup_spatial_lag(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
):
    """Enrich the sales and universe DataFrames with spatial lag features.

    This function calculates "spatial lag", that is, the spatially-weighted
    average, of the sale price and other fields, based on nearest neighbors.

    For sales, the spatial lag is calculated based on the training set of sales.
    For non-sale characteristics, the spatial lag is calculated based on the
    universe parcels.

    Parameters
    ----------
    sup : SalesUniversePair
        SalesUniversePair containing sales and universe DataFrames.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    SalesUniversePair
        Enriched SalesUniversePair with spatial lag features.
    """
    return openavmkit.data.enrich_sup_spatial_lag(sup, settings, verbose)


def enrich_sup_streets(sup: SalesUniversePair, settings: dict, verbose: bool = False):
    """Enrich a GeoDataFrame with street network data.

    This function enriches the input GeoDataFrame with street network data by calculating
    frontage, depth, distance to street, and many other related metrics, for every road vs.
    every parcel in the GeoDataFrame, using OpenStreetMap data.

    WARNING: This function can be VERY computationally and memory intensive for large datasets
    and may take a long time to run.

    We definitely need to work on its performance or make it easier to split into smaller chunks.

    Parameters
    ----------
    sup : SalesUniversePair
        The data you want to enrich
    settings : dict
        Settings dictionary
    verbose : bool, optional
        If True, prints verbose output during processing. Defaults to False.

    Returns
    -------
    gpd.GeoDataFrame
        Enriched GeoDataFrame with additional columns for street-related metrics.
    """
    df_univ = sup.universe
    df_univ = openavmkit.data.enrich_df_streets(df_univ, settings, verbose=verbose)
    sup.universe = df_univ
    return sup


def fill_unknown_values_sup(
    sup: SalesUniversePair, settings: dict
) -> SalesUniversePair:
    """Fill unknown values with default values as specified in settings.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    settings : dict
        The settings dictionary containing configuration for filling unknown values.

    Returns
    -------
    SalesUniversePair
        The updated SalesUniversePair with filled unknown values.
    """
    return openavmkit.cleaning.fill_unknown_values_sup(sup, settings)


# Clustering stuff


def mark_ss_ids_per_model_group_sup(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """
    Cluster parcels for a sales scrutiny study by assigning sales scrutiny IDs.

    This function processes each model group within the provided SalesUniversePair,
    identifies clusters of parcels for scrutiny, and writes the cluster identifiers
    into a new field on the universe DataFrame.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    verbose : bool, optional
        If True, prints verbose output during processing. Defaults to False.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair with marked sales scrutiny IDs.
    """
    df_sales_hydrated = get_hydrated_sales_from_sup(sup)
    df_marked = mark_ss_ids_per_model_group(df_sales_hydrated, settings, verbose)
    sup.update_sales(df_marked, allow_remove_rows=False)
    return sup


def mark_horizontal_equity_clusters_per_model_group_sup(
    sup: SalesUniversePair,
    settings: dict,
    verbose: bool = False,
    do_land_clusters: bool = True,
    do_impr_clusters: bool = True,
) -> SalesUniversePair:
    """
    Cluster parcels for a horizontal equity study by assigning horizontal equity cluster IDs.

    This is done for each model group within a SalesUniversePair. Marking IDs ahead of time
    allows for more efficient processing later. Delegates to
    `openavmkit.horizontal_equity_study.mark_horizontal_equity_clusters_per_model_group_sup`.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    verbose : bool, optional
        If True, prints verbose output. Defaults to False.
    do_land_clusters : bool, optional
        If True, enables land clustering. Defaults to True.
    do_impr_clusters : bool, optional
        If True, enables improvement clustering. Defaults to True.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair with horizontal equity clusters marked.
    """
    return openavmkit.horizontal_equity_study.mark_horizontal_equity_clusters_per_model_group_sup(
        sup,
        settings,
        verbose,
        do_land_clusters=do_land_clusters,
        do_impr_clusters=do_impr_clusters,
    )


def run_sales_scrutiny(
    sup: SalesUniversePair,
    settings: dict,
    drop_cluster_outliers: bool = False,
    drop_heuristic_outliers: bool = True,
    verbose: bool = False,
) -> SalesUniversePair:
    """
    Run sales scrutiny analysis for each model group within a SalesUniversePair.

    1. Performs basic sales validation heuristics
    2. Optionally drops manually excluded sales flagged by user
    3. Runs a cluster-based sales scrutiny analysis report

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    drop_cluster_outliers : bool, optional
        If True, drops invalid sales identified through cluster analysis. Defaults to False.
    drop_heuristic_outliers : bool, optional
        If True, drops invalid sales identified through heuristics. Defaults to True.
    verbose : bool, optional
        If True, enables verbose logging. Defaults to False.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair after sales scrutiny analysis.
    """
    sup = run_heuristics(sup, settings, drop_heuristic_outliers, verbose)
    sup = drop_manual_exclusions(sup, settings, verbose)
    sup = run_sales_scrutiny_per_model_group_sup(
        sup, settings, drop_cluster_outliers, verbose
    )
    return sup


def run_sales_scrutiny_per_model_group_sup(
    sup: SalesUniversePair, settings: dict, drop: bool = True, verbose: bool = False
) -> SalesUniversePair:
    """
    Run sales scrutiny analysis for each model group within a SalesUniversePair.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    drop : bool, optional
        If True, drops invalid sales after scrutiny. Defaults to True.
    verbose : bool, optional
        If True, enables verbose logging. Defaults to False.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair after sales scrutiny analysis.
    """

    df_sales_hydrated = get_hydrated_sales_from_sup(sup)
    df_scrutinized = run_sales_scrutiny_per_model_group(
        df_sales_hydrated, settings, verbose
    )

    if drop:
        # Drop all invalid sales
        df_scrutinized = df_scrutinized[df_scrutinized["valid_sale"].eq(True)]
        sup_num_valid_before = len(sup.sales[sup.sales["valid_sale"].eq(True)])

        sup.update_sales(df_scrutinized, allow_remove_rows=True)

        sup_num_valid_after = len(sup.sales[sup.sales["valid_sale"].eq(True)])

        if verbose:
            diff = sup_num_valid_before - sup_num_valid_after
            print("")
            print(
                f"Number of valid sales in SUP before scrutiny: {sup_num_valid_before}"
            )
            print(f"Number of valid sales in SUP after scrutiny: {sup_num_valid_after}")
            print(f"Difference in valid sales in SUP: {diff}")
    else:
        sup.update_sales(df_scrutinized, allow_remove_rows=False)

    return sup


# Read & write stuff


def from_checkpoint(path: str, func: callable, params: dict) -> pd.DataFrame:
    """
    Read cached data from a checkpoint file or generate it via a function.

    Wrapper that attempts to load a DataFrame from the given checkpoint path. If the file
    does not exist, it calls the provided function with the given parameters to generate
    the data, saves the result to the checkpoint, and returns it.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    func : callable
        Function to run if the checkpoint is not available. Should return a DataFrame.
    params : dict
        Parameters to pass to `func` when generating the data.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame, loaded from the checkpoint or generated.
    """
    return openavmkit.checkpoint.from_checkpoint(path, func, params)


def delete_checkpoints(prefix: str) -> None:
    """
    Delete all checkpoints that match the given prefix.

    Parameters
    ----------
    prefix : str
        The prefix used to identify checkpoints to delete.

    Returns
    -------
    None
        FILL_IN_HERE: Describe return value if any.
    """
    return openavmkit.checkpoint.delete_checkpoints(prefix)


def write_checkpoint(data: Any, path: str):
    """
    Write data to a checkpoint file.

    Saves a pandas DataFrame as Parquet if `data` is a DataFrame; otherwise, pickle-serializes `data`.

    Parameters
    ----------
    data : Any
        Data to be checkpointed.
    path : str
        File path for saving the checkpoint.
    """
    return openavmkit.checkpoint.write_checkpoint(data, path)


def write_notebook_output_sup(
    sup: SalesUniversePair, prefix: str = "1-assemble"
) -> None:
    """
    Write notebook output to disk.

    This function saves the SalesUniversePair as a pickle file and writes the
    corresponding 'universe' and 'sales' DataFrames to Parquet files.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    prefix : str, optional
        File prefix for naming output files. Defaults to "1-assemble".
    """

    with open(f"out/{prefix}-sup.pickle", "wb") as file:
        pickle.dump(sup, file)
    os.makedirs("out/look", exist_ok=True)

    # Handle geometry columns for both universe and sales
    def prepare_df_for_parquet(df):
        if "geometry" in df.columns:
            # Convert geometry to WKB for storage
            df = df.copy()
            if hasattr(df, "to_wkb"):
                # If it's a GeoDataFrame, use to_wkb() method
                df["geometry"] = df.geometry.to_wkb()
            else:
                # If it's a regular DataFrame with geometry column
                import shapely.wkb

                df["geometry"] = df["geometry"].apply(
                    lambda geom: geom.wkb if geom is not None else None
                )
        return df

    # Prepare and write universe DataFrame
    df_universe = prepare_df_for_parquet(sup["universe"])
    df_universe.to_parquet(f"out/look/{prefix}-universe.parquet", engine="pyarrow")

    # Prepare and write sales DataFrame

    df_sales = prepare_df_for_parquet(sup["sales"])
    df_sales.to_parquet(f"out/look/{prefix}-sales.parquet")

    df_hydrated = get_hydrated_sales_from_sup(sup)
    df_hydrated = prepare_df_for_parquet(df_hydrated)
    df_hydrated.to_parquet(
        f"out/look/{prefix}-sales-hydrated.parquet", engine="pyarrow"
    )

    print("Results written to:")
    print(f"...out/{prefix}-sup.pickle")
    print(f"...out/look/{prefix}-universe.parquet")
    print(f"...out/look/{prefix}-sales.parquet")
    print(f"...out/look/{prefix}-sales-hydrated.parquet")


def cloud_sync(
    locality: str,
    verbose: bool = False,
    env_path: str = "",
    bootstrap: str = "",
    dry_run: bool = False,
    ignore_paths: list = None,
) -> None:
    """
    Synchronize local files to cloud storage.

    This function initializes the cloud service and syncs files for the given locality.

    Parameters
    ----------
    locality : str
        The locality identifier used to form remote paths.
    verbose : bool, optional
        If True, prints detailed log messages. Defaults to False.
    env_path : str, optional
        Path to the environment configuration file. Defaults to an empty string.
    bootstrap: str, optional
        Which cloud service to bootstrap from on an initial run. Defaults to an empty string.
    dry_run : bool, optional
        If True, simulates the sync without performing any changes. Defaults to False.
    ignore_paths : list, optional
        List of file paths or patterns to ignore during sync. Defaults to None.
    """

    settings = load_settings(error = False, warning = False)
    if settings is None:
        if bootstrap == "":
            raise ValueError("No settings file found, please specify a cloud service to bootstrap from, or create a settings file from scratch.")
        else:
            print(f"No settings.json file found, bootstrapping from cloud service ({bootstrap})...")
            settings = {
                "cloud": {
                    "type": bootstrap
                }
            }

    cloud_service = cloud.init(verbose, env_path=env_path, settings=settings)
    if cloud_service is None:
        print("Cloud service not initialized, skipping...")
        return

    if ignore_paths is None:
        ignore_paths = []
    extra_ignore = settings.get("cloud", {}).get("ignore_paths", [])
    ignore_paths = ignore_paths + extra_ignore

    print(f"ignore_paths = {ignore_paths}")

    remote_path = locality.replace("-", "/") + "/"
    cloud_service.sync_files(
        locality,
        "in",
        remote_path,
        dry_run=dry_run,
        verbose=verbose,
        ignore_paths=ignore_paths,
    )


def load_cleaned_data_for_modeling(settings: dict):
    """
    Read and return the cleaned data from notebook 2 so notebook 3 can use it.
    Additionally, check the sales scrutiny settings for the invalid key file, and
    if it's defined, use that to exclude any recently marked invalid sales.
    
    (This saves having to do a full round trip through notebook 1&2 just to exclude a newly
    identified invalid sale)
    
    Parameters
    ----------
    settings : dict
        Configuration settings
        
    Returns
    -------
    SalesUniversePair
        The cleaned and ready SalesUniversePair
    
    """
    sales_univ_pair = read_pickle("out/2-clean-sup")
    s_sales_scrutiny = settings.get("analysis", {}).get("sales_scrutiny", {})
    invalid_key_file = s_sales_scrutiny.get("invalid_key_file")
    if invalid_key_file is not None:
        if os.path.exists(invalid_key_file):
            df_invalid_keys = pd.read_csv(invalid_key_file, dtype={"key_sale": str})
            bad_keys = df_invalid_keys["key_sale"].values
            df_sales = sales_univ_pair.sales
            df_sales = df_sales[~df_sales["key_sale"].isin(bad_keys)].copy()
            sales_univ_pair.sales = df_sales
    return sales_univ_pair


def read_pickle(path: str) -> Any:
    """
    Read and return data from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    Any
        The object loaded from the pickle file.
    """
    return openavmkit.checkpoint.read_pickle(path)


# Modeling stuff


def try_variables(
    sup: SalesUniversePair,
    settings: dict,
    verbose: bool = False,
    plot: bool = False,
    do_report: bool = False,
):
    """
    Run tests on variables to figure out which might be the most predictive.

    Parameters
    ----------
    sup : SalesUniversePair
        Your data
    settings : dict
        Settings dictionary
    verbose : bool, optional
        If True, prints detailed logs during data loading. Defaults to False.
    plot : bool, optional
        If True, prints visual plots. Defaults to False.
    do_report : bool, optional
        If True, generates PDF reports. Defaults to False.
    """
    sup = fill_unknown_values_sup(sup, settings)
    openavmkit.benchmark.try_variables(sup, settings, verbose, plot, do_report)


def try_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    verbose: bool = False,
    run_main: bool = True,
    run_vacant: bool = True,
    run_hedonic: bool = True,
    run_ensemble: bool = True,
    do_shaps: bool = False,
    do_plots: bool = False
) -> None:
    """
    Tries out predictive models on the given SalesUniversePair. Optimized for speed
    and iteration, doesn't finalize results or write anything to disk.

    This function takes detailed instructions from the provided settings dictionary and
    handles all the internal details like splitting the data, training the models, and
    saving the results. It performs basic statistic analysis on each model, and optionally
    combines results into an ensemble model.

    If "run_main" is true, it will run normal models as well as hedonic models (if the
    user so specifies), "hedonic" in this context meaning models that attempt to generate
    a land value and an improvement value separately. If "run_vacant" is true, it will run
    vacant models as well -- models that only use vacant models as evidence to generate
    land values.

    This function delegates the model execution to `openavmkit.benchmark.run_models`
    with the given settings.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    save_params : bool, optional
        Whether to save model parameters. Defaults to True.
    use_saved_params : bool, optional
        Whether to use saved model parameters. Defaults to True.
    verbose : bool, optional
        If True, enables verbose output. Defaults to False.
    run_main : bool, optional
        Flag to run main models. Defaults to True.
    run_vacant : bool, optional
        Flag to run vacant models. Defaults to True.
    run_hedonic : bool, optional
        Flag to run hedonic models. Defaults to True.
    run_ensemble : bool, optional
        Flag to run ensemble models. Defaults to True.
    do_shaps : bool, optional
        Flag to run SHAP analysis. Defaults to False.
    do_plots : bool, optional
        Flag to plot scatterplots. Defaults to False.
    """

    openavmkit.benchmark.run_models(
        sup,
        settings,
        save_params,
        use_saved_params,
        save_results=False,
        verbose=verbose,
        run_main=run_main,
        run_vacant=run_vacant,
        run_hedonic=run_hedonic,
        run_ensemble=run_ensemble,
        do_shaps=do_shaps,
        do_plots=do_plots
    )


def finalize_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    verbose: bool = False,
) -> None:
    """
    Tries out predictive models on the given SalesUniversePair, finalizes results and writes to disk.

    This function takes detailed instructions from the provided settings dictionary and handles all the internal
    details like splitting the data, training the models, and saving the results. It performs basic statistic analysis
    on each model, and optionally combines results into an ensemble model.

    This function iterates over model groups and runs models for main, hedonic and vacant cases.

    It delegates the model execution to `openavmkit.benchmark.run_models` with the given settings.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        The settings dictionary.
    save_params : bool, optional
        Whether to save model parameters.
    use_saved_params : bool, optional
        Whether to use saved model parameters.
    verbose : bool, optional
        If True, prints additional information.

    Returns
    -------
    MultiModelResults
        The MultiModelResults containing all model results and benchmarks.
    """

    openavmkit.benchmark.run_models(
        sup,
        settings,
        save_params,
        use_saved_params,
        save_results=True,
        verbose=verbose,
        run_main=True,
        run_vacant=True,
        run_hedonic=True,
        run_ensemble=True,
        do_shaps=False,
        do_plots=False
    )


def run_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    save_results: bool = True,
    verbose: bool = False,
    run_main: bool = True,
    run_vacant: bool = True,
    run_hedonic: bool = True,
    run_ensemble: bool = True,
    do_shaps: bool = False,
    do_plots: bool = False
):
    """
    Runs predictive models on the given SalesUniversePair.

    This function takes detailed instructions from the provided settings dictionary and handles all the internal
    details like splitting the data, training the models, and saving the results. It performs basic statistic analysis
    on each model, and optionally combines results into an ensemble model.

    If "run_main" is true, it will run normal models as well as hedonic models (if the user so specifies),
    "hedonic" in this context meaning models that attempt to generate a land value and an improvement value separately.
    If "run_vacant" is true, it will run vacant models as well -- models that only use vacant models as evidence
    to generate land values.

    This function iterates over model groups and runs models for both main and vacant cases.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        The settings dictionary.
    save_params : bool, optional
        Whether to save model parameters.
    use_saved_params : bool, optional
        Whether to use saved model parameters.
    save_results : bool, optional
        Whether to save model results.
    verbose : bool, optional
        If True, prints additional information.
    run_main : bool, optional
        Whether to run main (non-vacant) models.
    run_vacant : bool, optional
        Whether to run vacant models.
    run_hedonic : bool, optional
        Whether to run hedonic models.
    run_ensemble : bool, optional
        Whether to run ensemble models.
    do_shaps : bool, optional
        Whether to compute SHAP values.
    do_plots : bool, optional
        Whether to plot scatterplots

    Returns
    -------
    MultiModelResults
        The MultiModelResults containing all model results and benchmarks.
    """
    return openavmkit.benchmark.run_models(
        sup,
        settings,
        save_params,
        use_saved_params,
        save_results,
        verbose,
        run_main,
        run_vacant,
        run_hedonic,
        run_ensemble,
        do_shaps,
        do_plots
    )


def write_canonical_splits(sup: SalesUniversePair, settings: dict, verbose: bool = False) -> None:
    """
    Write canonical splits for the sales DataFrame.

    This separates the sales data into training and test sets and stores the keys to disk,
    ensuring consistent splits across multiple models for proper ensembling. Delegates to
    `openavmkit.data._write_canonical_splits`.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data.
    settings : dict
        Configuration settings.
    verbose : bool
        Whether to print verbose output.
    """

    openavmkit.data._write_canonical_splits(sup, settings, verbose)


def run_and_write_ratio_study_breakdowns(settings: dict) -> None:
    """
    Run ratio study breakdowns and write the results to disk.

    Parameters
    ----------
    settings : dict
        Configuration settings for the ratio study.
    """
    openavmkit.ratio_study.run_and_write_ratio_study_breakdowns(settings)


# PRIVATE:


def _clip_sales_to_use(
    df_sales: pd.DataFrame, settings: dict, verbose: bool = False
) -> pd.DataFrame:

    val_year = get_valuation_date(settings).year

    metadata = settings.get("modeling", {}).get("metadata", {})
    use_sales_from = metadata.get("use_sales_from", {})

    if isinstance(use_sales_from, int):
        use_sales_from_impr = use_sales_from
        use_sales_from_vacant = use_sales_from
    else:
        use_sales_from_impr = use_sales_from.get("improved", val_year - 5)
        use_sales_from_vacant = use_sales_from.get("vacant", val_year - 5)

    # mark which sales are to be used (only those that are valid and within the specified time frame)
    df_sales.loc[
        df_sales["sale_year"].lt(use_sales_from_impr)
        & df_sales["vacant_sale"].eq(False),
        "valid_sale",
    ] = False
    df_sales.loc[
        df_sales["sale_year"].lt(use_sales_from_vacant)
        & df_sales["vacant_sale"].eq(True),
        "valid_sale",
    ] = False

    df_sales = df_sales[df_sales["valid_sale"].eq(True)].copy()

    return df_sales


def _set_locality(nbs, locality: str):
    """Set or update the notebook state with a new locality.

    This function updates the NotebookState to reflect the specified locality, changes
    the working directory to the appropriate path, and ensures that the data directory
    exists.
    """
    base_path = None
    if nbs is not None:
        base_path = nbs.base_path
        if locality != nbs.locality:
            nbs = NotebookState(locality, base_path)
    if base_path is None:
        nbs = NotebookState(locality, None)

    if base_path is not None:
        os.chdir(nbs.base_path)

    os.makedirs(f"data/{locality}", exist_ok=True)

    os.chdir(f"data/{locality}")

    print(f"locality = {locality}")
    print(f"base path = {nbs.base_path}")
    print(f"current path = {os.getcwd()}")
    return nbs
