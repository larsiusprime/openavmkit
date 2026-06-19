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
import geopandas as gpd

import openavmkit
import openavmkit.data
import openavmkit.area_stats
import openavmkit.model_runner
import openavmkit.checkpoint
import openavmkit.ratio_study
import openavmkit.horizontal_equity_study
import openavmkit.vertical_equity_study
import openavmkit.cleaning

from dotenv import load_dotenv, find_dotenv

from openavmkit.cleaning import clean_valid_sales, filter_invalid_sales
from openavmkit.cloud import cloud
from openavmkit.data import (
    load_dataframe,
    process_data,
    SalesUniversePair,
    get_sup_model_group,
    get_hydrated_sales_from_sup,
    write_parquet,
    write_gpkg,
    write_zipped_shapefile,
    write_csv
)
from openavmkit.sales_scrutiny_study import (
    run_sales_scrutiny_per_model_group,
    mark_ss_ids_per_model_group,
    run_heuristics,
    drop_manual_exclusions,
)
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import combine_dfs, div_df_z_safe
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_fields_numeric,
    get_fields_boolean,
    get_fields_land,
    get_fields_impr,
    get_fields_other,
    get_valuation_date,
    get_model_group_ids,
    area_unit
)
from openavmkit.calculations import (
    div_series_z_safe
)
from openavmkit.utilities.plotting import (
    plot_scatterplot
)
from openavmkit.ratio_study import (
    RatioStudy,
    RatioStudyBootstrapped
)
from openavmkit.horizontal_equity_study import (
    HorizontalEquityStudy
)
from openavmkit.vertical_equity_study import (
    VerticalEquityStudy
)

from IPython.display import display

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
    
    load_dotenv(dotenv_path=find_dotenv())


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
    
    _ex_name_width = 60

    def fill_str(char: str, size: int):
        text = ""
        for _i in range(0, size):
            text += char
        return text

    def fit_str(txt: str, size: int):
        last_bit = ""
        first_bit = ""
        if len(txt) >= size:
            len_first = int((size - 3) / 2)
            len_last = (size - 3) - len_first
            first_bit = txt[0:len_first]
            last_bit = txt[len(txt) - len_last :]
            if len(last_bit) > 0:
                last_bit = "\n\n"+last_bit
            txt = first_bit
        return first_bit, last_bit

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
        
        fitted_str, extra_str = fit_str(col, _ex_name_width)
        return f"{fitted_str} {dtype:^10} {count_non_zero:>10} {p:>5.0%} {count_non_null:>10} {pnn:>5.0%} {uniques:>40}{extra_str}"

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
                    
                if len(df) != 0:
                    perc = non_zero / len(df)
                    non_null = len(df_non_null)
                    perc_non_null = non_null / len(df)
                else:
                    perc = float("nan")
                    perc_non_null = float('nan')
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
                if len(df) != 0:
                    perc = non_zero / len(df)
                else:
                    perc = float("nan")
                non_null = len(df_non_null)
                if len(df) != 0:
                    perc_non_null = non_null / len(df)
                else:
                    perc_non_null = float("nan")
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
            if len(df) != 0:
                perc = non_zero / len(df)
            else:
                perc = float("nan")
            if len(df) != 0:
                perc_non_null = non_zero / len(df)
            else:
                perc_non_null = float("nan")
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
    
    _ex_name_width = 60
    
    def fill_str(char: str, size: int):
        text = ""
        for _i in range(0, size):
            text += char
        return text

    def fit_str(txt: str, size: int):
        last_bit = ""
        first_bit = txt
        if len(txt) >= size:
            first_bit = txt[0:size]
            last_bit = txt[size:]
            if len(last_bit) > 0:
                last_bit = "\n" + last_bit
        return f"{first_bit:{size}}", last_bit

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
        
        fitted_str, extra_str = fit_str(col, _ex_name_width)
        
        return f"{fitted_str} {dtype:^10} {count_non_zero:>10} {p:>5.0%} {count_non_null:>10} {pnn:>5.0%} {uniques:>40}{extra_str}"

    buffer = ""
    lines = 0
    chunk_size = 3

    def print_horz_line(char: str):
        nonlocal buffer
        nonlocal lines
        if buffer != "":
            buffer += "\n"
        buffer += (
            fill_str(char, _ex_name_width)
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
        f"{'FIELD':^{_ex_name_width}} {'TYPE':^10} {'NON-ZERO':^10} {'%':^5} {'NON-NULL':^10} {'%':^5} {'UNIQUE':^40}"
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
        print_buffer(f"{name:^{_ex_name_width}}")
        print_horz_line("=")

        nums.sort()
        bools.sort()
        cats.sort()

        if len(nums) > 0:
            print_horz_line("-")
            print_buffer(f"{'NUMERIC':^{_ex_name_width}}")
            print_horz_line("-")
            for n in nums:
                fields_noted.append(n)
                df_non_null = df[~pd.isna(df[n])]
                non_zero = len(df_non_null[np.abs(df_non_null[n]).gt(0)])
                if len(df) != 0:
                    perc = non_zero / len(df)
                else:
                    perc = float("nan")
                non_null = len(df_non_null)
                if len(df) != 0:
                    perc_non_null = non_null / len(df)
                else:
                    perc_non_null = float("nan")
                print_buffer(
                    get_line(
                        n, df[n].dtype, non_zero, perc, non_null, perc_non_null, ""
                    )
                )

        if len(bools) > 0:
            print_horz_line("-")
            print_buffer(f"{'BOOLEAN':^{_ex_name_width}}")
            print_horz_line("-")
            for b in bools:
                fields_noted.append(b)
                df_non_null = df[~pd.isna(df[b])]
                non_zero = len(df_non_null[np.abs(df_non_null[b]).gt(0)])
                if len(df) != 0:
                    perc = non_zero / len(df)
                else:
                    perc = float("nan")
                
                non_null = len(df_non_null)
                if non_null != 0:
                    perc_non_null = non_null / len(df)
                else:
                    perc_non_null = float("nan")
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
            print_buffer(f"{'CATEGORICAL':^{_ex_name_width}}")
            print_horz_line("-")
            for c in cats:
                fields_noted.append(c)
                non_zero = (~pd.isna(df[c])).sum()
                if len(df) != 0:
                    perc = non_zero / len(df)
                else:
                    perc = float("nan")
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
        print_buffer(f"{'UNCLASSIFIED:':<{_ex_name_width}}")
        print_horz_line("=")
        for u in fields_unclassified:
            non_zero = (~pd.isna(df[u])).sum()
            if len(df) != 0:
                perc = non_zero / len(df)
                perc_non_null = non_zero / len(df)
            else:
                perc = float("nan")
                perc_non_null = float("nan")
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
        df = load_dataframe(
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


def process_dataframes(dataframes: dict[str, pd.DataFrame], settings: dict, verbose: bool = False) -> SalesUniversePair:
    """
    Load and process data according to provided settings.

    This function first loads the dataframes, then merges and enriches the data,
    returning a SalesUniversePair.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Dictionary of DataFrames.
    settings : dict
        A dictionary of settings for data loading and processing.
    verbose : bool, optional
        If True, prints detailed logs during data loading. Defaults to False.

    Returns
    -------
    SalesUniversePair
        A SalesUniversePair object containing the processed sales and universe data.
    """

    results = process_data(dataframes, settings, verbose)

    write_notebook_output_sup(results)

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
    sup: SalesUniversePair, settings: dict, write: bool = False, verbose: bool = False
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
    write : bool
        Whether to write out data during processing. Defaults to False.
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

    # validate sales using filters
    sup = filter_invalid_sales(sup, settings, verbose)

    print(f"len after validate = {len(sup['sales'])}")

    # make sure sales field has necessary fields for the next step
    df_sales_hydrated = get_hydrated_sales_from_sup(sup)

    print(f"len after hydrate = {len(sup['sales'])}")

    # enrich with time adjustment, and mark what fields were added
    df_sales_enriched = enrich_time_adjustment(df_sales_hydrated, settings, write, verbose)

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


def enrich_sup_area_stats(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
):
    """Enrich sales and universe with per-location area-statistic features.

    Area statistics are per-location summary statistics (mean, median, dispersion,
    dominant category, ...) of user-selected fields, stamped onto every parcel as
    ``area_stat_<location>_<field>_<stat>`` columns. This is a quantized, group-based
    counterpart to spatial lag and should be run after cleaning, sales scrutiny, and the
    canonical train/test split, so invalid sales and test-set prices never enter the
    statistics.

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
        Enriched SalesUniversePair with area-statistic features.
    """
    return openavmkit.area_stats.enrich_sup_area_stats(sup, settings, verbose)


def report_area_stats(
    sup: SalesUniversePair,
    settings: dict,
    outpath: str = None,
    threshold: float = 0.1,
    do_plots: bool = False,
    verbose: bool = False,
):
    """Rank area-stat features by their correlation with sale price.

    Returns a DataFrame ranking every numeric ``area_stat_*`` feature by its correlation
    with sale price (over valid sales). When ``outpath`` is provided, also writes a
    Markdown report.

    Parameters
    ----------
    sup : SalesUniversePair
        SalesUniversePair already enriched via :func:`enrich_sup_area_stats`.
    settings : dict
        Settings dictionary.
    outpath : str, optional
        Output path (without extension) for the Markdown report. If None, no file is
        written.
    threshold : float, optional
        Correlation score threshold. Defaults to 0.1.
    do_plots : bool, optional
        If True, render correlation heatmaps. Defaults to False.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    pandas.DataFrame
        Ranked correlation table.
    """
    return openavmkit.area_stats.report_area_stats(
        sup, settings, outpath, threshold, do_plots, verbose
    )


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


def collapse_sparse_categories_sup(
    sup: SalesUniversePair, settings: dict
) -> SalesUniversePair:
    """Collapse rare categorical values into a per-field replacement bucket.

    Reads ``data.process.collapse_sparse_categories`` from ``settings`` and,
    for each configured field, replaces any category whose row count falls
    below ``sales_min`` in the hydrated sales set OR below ``univ_min`` in
    the universe set with the configured ``replacement_value`` (default
    ``"Other"``).

    The same mapping is applied to both the sales and universe DataFrames
    so downstream modeling and ratio-study artifacts see a single, consistent
    vocabulary. Fields where fewer than two categories would be collapsed
    are left untouched (renaming a single category buys nothing).

    Settings shape::

        "data": {
          "process": {
            "collapse_sparse_categories": {
              "roof_material": {"sales_min": 2, "univ_min": 5},
              "roof_shape":    {"sales_min": 2, "univ_min": 5,
                                "replacement_value": "Other Shape"}
            }
          }
        }

    Per-field rules:
        - ``sales_min`` and ``univ_min`` are required (else ``ValueError``).
        - ``replacement_value`` is optional; defaults to ``"Other"``.
        - The field must be declared in ``field_classification.*.categorical``
          (else ``ValueError``).

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    settings : dict
        The settings dictionary.

    Returns
    -------
    SalesUniversePair
        The updated SalesUniversePair with sparse categories collapsed.

    Raises
    ------
    ValueError
        If a configured field is misconfigured or unknown.
    """
    return openavmkit.cleaning.collapse_sparse_categories_sup(sup, settings)


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
    
    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    clusters_enabled = ss.get("clusters_enabled", True)
    heuristics_enabled = ss.get("heuristics_enabled", True)
    
    os.makedirs("out/sales_scrutiny/", exist_ok=True)
    
    if heuristics_enabled:
        sup = run_heuristics(sup, settings, drop_heuristic_outliers, verbose)
    elif verbose:
        print(f"Skipping sales scrutiny heuristics...")
    
    sup = drop_manual_exclusions(sup, settings, verbose)
    
    if clusters_enabled:
        sup = run_sales_scrutiny_per_model_group_sup(
            sup, settings, drop_cluster_outliers, verbose
        )
    elif verbose:
        print(f"Skipping clustered sales scrutiny analysis...")
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
    sup: SalesUniversePair, 
    prefix: str = "1-assemble",
    parquet: bool = True,
    gpkg: bool = False,
    shp: bool = False,
    csv: bool = False
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
    parquet : bool, optional
        Whether to write to parquet format. Defaults to true.
    gpkg : bool, optional
        Whether to write to gpkg format. Defaults to false.
    shp : bool, optional
        Whether to write to ESRI shapefile format. Defaults to false.
    """

    try:
        os.makedirs("out/look", exist_ok=True)
        with open(f"out/{prefix}-sup.pickle", "wb") as file:
            pickle.dump(sup, file)
        
        # universe
        if parquet:
            write_parquet(sup["universe"], f"out/look/{prefix}-universe.parquet")
        if gpkg:
            write_gpkg(sup["universe"], f"out/look/{prefix}-universe.gpkg")
        if shp:
            write_zipped_shapefile(sup["universe"], f"out/look/{prefix}-universe.shp.zip")
        if csv:
            write_csv(sup["universe"], f"out/look/{prefix}-universe.csv")
        
        # sales
        if parquet:
            write_parquet(sup["sales"], f"out/look/{prefix}-sales.parquet")
        
        # sales (hydrated)
        df_hydrated = get_hydrated_sales_from_sup(sup)
        if parquet:
            write_parquet(df_hydrated, f"out/look/{prefix}-sales-hydrated.parquet")
        if gpkg:
            write_gpkg(df_hydrated, f"out/look/{prefix}-sales-hydrated.gpkg")
        if shp:
            write_zipped_shapefile(df_hydrated, f"out/look/{prefix}-sales-hydrated.shp.zip")
        if csv:
            write_csv(df_hydrated, f"out/look/{prefix}-sales-hydrated.csv")

        print(f"...out/{prefix}-sup.pickle")
        if parquet:
            print(f"...out/look/{prefix}-universe.parquet")
            print(f"...out/look/{prefix}-sales.parquet")
            print(f"...out/look/{prefix}-sales-hydrated.parquet")
        if gpkg:
            print(f"...out/look/{prefix}-universe.gpkg")
            print(f"...out/look/{prefix}-sales-hydrated.gpkg")
        if shp:
            print(f"...out/look/{prefix}-universe.shp.zip")
            print(f"...out/look/{prefix}-sales-hydrated.shp.zip")
        if csv:
            print(f"...out/look/{prefix}-universe.csv")
            print(f"...out/look/{prefix}-sales-hydrated.csv")
    except Exception as e:
        warnings.warn(f"Failed to output sup: {str(e)}")


def write_parquet(df: pd.DataFrame, path: str):
    """
    Write data to a parquet file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to be written
    path : str
        File path for saving the parquet.
    """
    
    openavmkit.data.write_parquet(df, path)


def cloud_sync(
    locality: str,
    verbose: bool = False,
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
    dry_run : bool, optional
        If True, simulates the sync without performing any changes. Defaults to False.
    ignore_paths : list, optional
        List of file paths or patterns to ignore during sync. Defaults to None.
    """
    
    cloud_settings = cloud.load_cloud_settings()
    
    if cloud_settings is None:
        warnings.warn("No cloud.json file found, cannot initialize cloud service.")
        return

    cloud_service = cloud.init(verbose, cloud_settings=cloud_settings)
    if cloud_service is None:
        print("Cloud service not initialized, skipping...")
        return

    if ignore_paths is None:
        ignore_paths = []
    extra_ignore = cloud_settings.get("ignore_paths", [])
    ignore_paths = ignore_paths + extra_ignore + ["cloud.json"]

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
    openavmkit.model_runner.try_variables(sup, settings, verbose, plot, do_report)


def try_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    verbose: bool = False,
    run_main: bool = True,
    run_vacant: bool = True,
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

    If "run_main" is true, it will run normal (full market value) models. If "run_vacant"
    is true, it will run vacant models as well -- models that only use vacant sales as
    evidence to generate land values.

    This function delegates the model execution to `openavmkit.model_runner.run_models`
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
    run_ensemble : bool, optional
        Flag to run ensemble models. Defaults to True.
    do_shaps : bool, optional
        Flag to run SHAP analysis. Defaults to False.
    do_plots : bool, optional
        Flag to plot scatterplots. Defaults to False.
    """

    openavmkit.model_runner.run_models(
        sup,
        settings,
        save_params,
        use_saved_params,
        save_results=False,
        verbose=verbose,
        run_main=run_main,
        run_vacant=run_vacant,
        run_ensemble=run_ensemble,
        do_shaps=do_shaps,
        do_plots=do_plots
    )


def _select_outlier_keys(
    dfm: pd.DataFrame,
    low_thresh: float = 0.75,
    high_thresh: float = 1.25,
    top_n: int = 10,
) -> pd.DataFrame:
    """Pick outliers from a prediction frame: top/bottom N by ratio plus everything outside the ratio thresholds."""
    if "prediction_ratio" not in dfm.columns or len(dfm) == 0:
        return dfm.iloc[0:0]
    s = dfm.sort_values("prediction_ratio")
    bottom = s.head(top_n)
    top = s.tail(top_n)
    extreme = dfm[(dfm["prediction_ratio"] < low_thresh) | (dfm["prediction_ratio"] > high_thresh)]
    out = pd.concat([bottom, top, extreme], ignore_index=False)
    return out.drop_duplicates(subset="key_sale")


def _fit_comp_models(
    df_full: pd.DataFrame,
    ind_vars: list[str],
    unit: str,
    is_vacant_only: bool,
):
    """Fit LayeredCompModel(s) on the full sales set for outlier comp-analysis.

    Returns (m_price, m_ppsf_impr, m_ppsf_land, feat_cols). Any model may be None
    if there isn't enough data on its slice."""
    from layeredcompmodel import LayeredCompModel
    from openavmkit.modeling import _coerce_categoricals_to_object

    feat_cols = [c for c in ind_vars if c in df_full.columns]
    if not feat_cols:
        return None, None, None, []

    df_full = df_full[df_full["sale_price_time_adj"].fillna(0).gt(0)].copy()
    if len(df_full) < 5:
        return None, None, None, feat_cols

    # Derive per-impr / per-land time-adj PPSF on the fly if not pre-computed.
    ppsf_impr_col = f"sale_price_time_adj_per_impr_{unit}"
    ppsf_land_col = f"sale_price_time_adj_per_land_{unit}"
    impr_size_col = f"bldg_area_finished_{unit}"
    land_size_col = f"land_area_{unit}"
    if ppsf_impr_col not in df_full.columns and impr_size_col in df_full.columns:
        df_full[ppsf_impr_col] = df_full["sale_price_time_adj"] / df_full[impr_size_col].replace(0, np.nan)
    if ppsf_land_col not in df_full.columns and land_size_col in df_full.columns:
        df_full[ppsf_land_col] = df_full["sale_price_time_adj"] / df_full[land_size_col].replace(0, np.nan)

    X_full = _coerce_categoricals_to_object(df_full[feat_cols])
    y_price = df_full["sale_price_time_adj"].astype(float)
    m_price = LayeredCompModel(split_metric="mae", n_jobs=1)
    m_price.fit(X_full, y_price)

    m_ppsf_impr = None
    if not is_vacant_only and ppsf_impr_col in df_full.columns:
        df_i = df_full[df_full["vacant_sale"].eq(False) & df_full[ppsf_impr_col].fillna(0).gt(0)]
        if len(df_i) >= 5:
            X_i = _coerce_categoricals_to_object(df_i[feat_cols])
            y_i = df_i[ppsf_impr_col].astype(float)
            m_ppsf_impr = LayeredCompModel(split_metric="mae", n_jobs=1)
            m_ppsf_impr.fit(X_i, y_i)

    m_ppsf_land = None
    if ppsf_land_col in df_full.columns:
        df_v = df_full[df_full["vacant_sale"].eq(True) & df_full[ppsf_land_col].fillna(0).gt(0)]
        if len(df_v) >= 5:
            X_v = _coerce_categoricals_to_object(df_v[feat_cols])
            y_v = df_v[ppsf_land_col].astype(float)
            m_ppsf_land = LayeredCompModel(split_metric="mae", n_jobs=1)
            m_ppsf_land.fit(X_v, y_v)

    return m_price, m_ppsf_impr, m_ppsf_land, feat_cols


def _build_comp_quality_table(
    df_outliers_full: pd.DataFrame,
    m_price,
    m_ppsf_impr,
    m_ppsf_land,
    feat_cols: list[str],
    unit: str,
) -> pd.DataFrame:
    """For each outlier, run explain_value() on the price + PPSF comp models and
    record leaf Wilson means, tree depth, leaf count, and the subject-to-leaf ratios."""
    from openavmkit.modeling import _coerce_categoricals_to_object

    if m_price is None or len(df_outliers_full) == 0:
        return pd.DataFrame()

    X_out = _coerce_categoricals_to_object(df_outliers_full[feat_cols])

    rows = []
    for idx, out_row in df_outliers_full.iterrows():
        x_row = X_out.loc[idx]
        is_v = bool(out_row.get("vacant_sale", False))
        actual_price = out_row.get("sale_price_time_adj", np.nan)
        try:
            actual_price = float(actual_price)
        except (TypeError, ValueError):
            actual_price = np.nan

        rec = {
            "key_sale": out_row["key_sale"],
            "vacant_sale": is_v,
            "ppsf_unit": "land" if is_v else "impr",
            "actual_price": actual_price,
        }

        try:
            exp_p = m_price.explain_value(x_row)
            path_p = exp_p.get("path", [])
            wm_price = float(path_p[-1]["wilson_mean"]) if path_p else np.nan
            depth_p = len(path_p)
            leaf_n_p = int(path_p[-1]["count"]) if path_p else 0
        except Exception:
            wm_price, depth_p, leaf_n_p = np.nan, np.nan, 0

        rec["leaf_wilson_price"] = wm_price
        rec["price_ratio"] = (actual_price / wm_price) if (wm_price and not np.isnan(wm_price) and not np.isnan(actual_price)) else np.nan
        rec["tree_depth_price"] = depth_p
        rec["leaf_count_price"] = leaf_n_p

        m_ppsf = m_ppsf_land if is_v else m_ppsf_impr
        size_col = f"land_area_{unit}" if is_v else f"bldg_area_finished_{unit}"
        size_val = out_row.get(size_col, np.nan)
        try:
            size_val = float(size_val)
        except (TypeError, ValueError):
            size_val = np.nan

        actual_ppsf = (actual_price / size_val) if (size_val and size_val > 0 and not np.isnan(actual_price)) else np.nan

        if m_ppsf is not None and not np.isnan(actual_ppsf):
            try:
                exp_pp = m_ppsf.explain_value(x_row)
                path_pp = exp_pp.get("path", [])
                wm_ppsf = float(path_pp[-1]["wilson_mean"]) if path_pp else np.nan
                depth_pp = len(path_pp)
                leaf_n_pp = int(path_pp[-1]["count"]) if path_pp else 0
            except Exception:
                wm_ppsf, depth_pp, leaf_n_pp = np.nan, np.nan, 0
        else:
            wm_ppsf, depth_pp, leaf_n_pp = np.nan, np.nan, 0

        rec["actual_ppsf"] = actual_ppsf
        rec["leaf_wilson_ppsf"] = wm_ppsf
        rec["ppsf_ratio"] = (actual_ppsf / wm_ppsf) if (wm_ppsf and not np.isnan(wm_ppsf) and not np.isnan(actual_ppsf)) else np.nan
        rec["tree_depth_ppsf"] = depth_pp
        rec["leaf_count_ppsf"] = leaf_n_pp

        rows.append(rec)

    return pd.DataFrame(rows)


def identify_outliers(
    sup: SalesUniversePair,
    settings: dict
):
    unit = area_unit(settings)
    outliers = settings.get("analysis", {}).get("outliers", {})
    df_sales = get_hydrated_sales_from_sup(sup)
    ids = get_model_group_ids(settings, df_sales)
    
    ss = settings.get("analysis", {}).get("sales_scrutiny", {})
    deed_id = ss.get("deed_id", None)
    location = ss.get("location", None)
    skip = outliers.get("skip", [])
    
    mgs = outliers.get("model_groups", {})
    
    default = outliers.get("default", {})
    
    for id in ids:
        if id in skip:
            continue
        df_sub = df_sales[df_sales["model_group"].eq(id)]
        entry = mgs.get(id, default)
        print("====================")
        print(f"MODEL GROUP = {id}")

        # Fit comp-analysis models once per model group (reused across mtypes / pred files).
        # Resolve lcomp ind_vars with the canonical per-group pattern: narrow models.main to
        # this model group (falling back to the top-level dict for the legacy/global layout),
        # then read lcomp's ind_vars (falling back to the group "default" entry).
        model_entries = settings.get("modeling", {}).get("models", {}).get("main", {})
        model_entries = model_entries.get(id, model_entries)
        lc_entry = model_entries.get("lcomp", model_entries.get("default", {}))
        ind_vars_lc = lc_entry.get("ind_vars", []) if isinstance(lc_entry, dict) else []
        if not isinstance(ind_vars_lc, list):
            ind_vars_lc = []
        comp_models = (None, None, None, [])
        if ind_vars_lc and "sale_price_time_adj" in df_sub.columns and "vacant_sale" in df_sub.columns:
            try:
                print(f"  Fitting comp-analysis LayeredCompModels on {len(df_sub):,} sales ({len(ind_vars_lc)} features)...")
                comp_models = _fit_comp_models(df_sub, ind_vars_lc, unit, is_vacant_only=False)
            except Exception as e:
                warnings.warn(f"Comp-analysis model fit failed for {id}: {e}")
        else:
            print(f"  Skipping comp-analysis: missing ind_vars / sale_price_time_adj / vacant_sale for {id}")
        m_price, m_ppsf_impr, m_ppsf_land, feat_cols = comp_models

        for mtype in ["main","vacant"]:
            model = entry.get("mtype", "ensemble")
            print(f"model type = {mtype}, model = {model}")
            
            path = f"out/models/{id}/{mtype}/{model}/pred_sales.csv"
            outdir = f"out/models/{id}/{mtype}/{model}/"
            outpath = f"out/models/{id}/{mtype}/{model}/outliers.csv"
            
            dfm : pd.DataFrame = None
                
            if os.path.exists(path):
                usecols = ["key_sale", "sale_price", "sale_date", "prediction", "prediction_ratio"]
                dtypes = {
                    "key_sale": "str",
                    "sale_price": "float",
                    "sale_date": "str",
                    "prediction": "float",
                    "prediction_ratio": "float"
                }
                dfm = pd.read_csv(path, dtype=dtypes, usecols=usecols)
                dfm["sale_date"] = pd.to_datetime(dfm["sale_date"])
                dfm = dfm[
                    dfm["prediction_ratio"].ge(0.75) |
                    dfm["prediction_ratio"].le(1.25)
                ]
                key_fields = [
                    "key_sale", 
                    "address", 
                    location, 
                    deed_id, 
                    f"bldg_area_finished_{unit}", 
                    f"land_area_{unit}", 
                    "assr_market_value", 
                    "assr_land_value", 
                    "assr_impr_value", 
                    "vacant_sale"
                ]
                key_fields = [field for field in key_fields if field is not None and field in df_sub]
                dfm = dfm.merge(df_sub[key_fields], on="key_sale", how="left")
            elif model == "assessor":
                # We can use the assessor fields directly
                the_field = "assr_market_value"
                if mtype == "vacant":
                    the_field = "assr_land_value"
                print(f"--> for assessor model, using \"{the_field}\" as prediction field...")
                dfm = df_sub.copy()
                dfm["prediction"] = dfm[the_field]
                dfm["prediction_ratio"] = div_df_z_safe(dfm, "prediction", "sale_price")
            
            if dfm is not None:
                if mtype != "main":
                    # it's a land model, only look at vacant sales:
                    dfm = dfm[dfm["vacant_sale"].eq(True)]
            
            if dfm is not None and "prediction" in dfm:
                print("")
                print("----------------------")
                value_fields = ["sale_price", "prediction", "assr_market_value", "assr_land_value", "assr_impr_value"]
                for v in value_fields:
                    if "impr" not in v:
                        dfm[f"{v}_land_{unit}"] = div_series_z_safe(dfm[v], dfm[f"land_area_{unit}"])
                    if "land" not in v:
                        dfm[f"{v}_impr_{unit}"] = div_series_z_safe(dfm[v], dfm[f"bldg_area_finished_{unit}"])
                
                dfm_i = dfm[dfm["vacant_sale"].eq(False)]
                dfm_v = dfm[dfm["vacant_sale"].eq(True)]
                
                df_loc_price_i = dfm_i.groupby(location)["sale_price"].agg(["count","median"]).reset_index().rename(columns={
                    "count":"local_impr_sales",
                    "median":"local_impr_price"
                })
                df_loc_price_is = dfm_i.groupby(location)[f"sale_price_impr_{unit}"].agg(["median"]).reset_index().rename(columns={
                    "median":f"local_impr_price_{unit}"
                })
                
                df_loc_price_v = dfm_v.groupby(location)["sale_price"].agg(["count","median"]).reset_index().rename(columns={
                    "count":"local_land_sales",
                    "median":"local_land_price"
                })
                df_loc_price_vs = dfm_v.groupby(location)[f"sale_price_land_{unit}"].agg(["median"]).reset_index().rename(columns={
                    "median":f"local_land_price_{unit}"
                })
                
                if mtype == "main":
                    dfm = dfm.merge(df_loc_price_i, on=location, how="left")
                    dfm = dfm.merge(df_loc_price_is, on=location, how="left")
                
                dfm = dfm.merge(df_loc_price_v, on=location, how="left")
                dfm = dfm.merge(df_loc_price_vs, on=location, how="left")
                
                #Re-arrange columns in a massively opinionated way
                cols = dfm.columns.tolist()
                put_at_front = ["key_sale", deed_id, "address", "prediction_ratio", "prediction", "sale_price"]
                if mtype == "main":
                    put_at_front += [
                        f"prediction_impr_{unit}", 
                        f"sale_price_impr_{unit}", 
                        f"local_impr_price_{unit}", 
                        f"local_impr_sales"
                    ]
                put_at_front += [
                    f"prediction_land_{unit}", 
                    f"sale_price_land_{unit}", 
                    f"local_land_price_{unit}", 
                    "local_land_sales"
                ]
                put_at_end = ["address", location]

                # Some of these opinionated front/end columns may be absent (most commonly
                # 'address', when no address field was mapped in settings). Warn loudly but
                # don't crash -- omit the missing columns and print the table anyway.
                desired = [c for c in (put_at_front + put_at_end) if c is not None]
                missing = [c for c in dict.fromkeys(desired) if c not in dfm.columns]
                if missing:
                    msg = (
                        f"identify_outliers: preferred columns missing from the data and omitted "
                        f"from {outpath}: {missing}."
                    )
                    if "address" in missing:
                        msg += (
                            " In particular 'address' is not present -- the outlier tables will have"
                            " no human-readable address column. Map an 'address' field in your"
                            " settings (data.load.<source>.load.address, or compose one in"
                            " preprocessing) to fix this."
                        )
                    warnings.warn(msg)
                put_at_front = [c for c in put_at_front if c in dfm.columns]
                put_at_end = [c for c in put_at_end if c in dfm.columns]

                cols = [col for col in cols if col not in put_at_front and col not in put_at_end and col in dfm and col is not None]
                cols = put_at_front + cols + put_at_end
                # de-duplicate, preserving order ('address' is intentionally listed in both
                # put_at_front and put_at_end).
                cols = list(dict.fromkeys(cols))

                dfm = dfm[cols]
                
                os.makedirs(outdir, exist_ok=True)
                
                dfm.to_csv(outpath, index=False)
                
                print("")
                print("Top 10 UNDER-predictions:")
                print("")
                dfm = dfm.sort_values(by="prediction_ratio", ascending=True)
                display(dfm.head(n=10))
                
                print("")
                print("Top 10 OVER-predictions:")
                print("")
                dfm = dfm.sort_values(by="prediction_ratio", ascending=False)
                display(dfm.head(n=10))

                # --- Comp-analysis (Phase 2): trace outliers through a LayeredCompModel
                # fit on the full sales set for this model group. We answer:
                # "What does the subject's lowest comp-tree leaf say its time-adjusted price
                # and PPSF should be?" Output is a per-outlier ratio + tree-depth diagnostic.
                if m_price is not None and feat_cols:
                    for pred_label, pred_fname in [("sales", "pred_sales.csv"), ("test", "pred_test.csv")]:
                        pred_path = f"out/models/{id}/{mtype}/{model}/{pred_fname}"
                        if not os.path.exists(pred_path):
                            continue
                        try:
                            df_pred = pd.read_csv(pred_path, dtype={"key_sale": "str"})
                            if "prediction_ratio" not in df_pred.columns:
                                continue
                            df_out = _select_outlier_keys(df_pred)
                            if mtype != "main":
                                # vacant: only score vacant sales
                                df_out = df_out.merge(
                                    df_sub[["key_sale", "vacant_sale"]].drop_duplicates("key_sale"),
                                    on="key_sale", how="left", suffixes=("", "_sub"),
                                )
                                df_out = df_out[df_out["vacant_sale"].eq(True)]
                            if len(df_out) == 0:
                                continue
                            # Merge in the features + V/I flag + time-adj price from df_sub
                            need_cols = list(set(feat_cols + [
                                "sale_price_time_adj", "vacant_sale",
                                f"bldg_area_finished_{unit}", f"land_area_{unit}",
                            ]) - set(df_out.columns))
                            need_cols = [c for c in need_cols if c in df_sub.columns]
                            df_out_full = df_out.merge(
                                df_sub[["key_sale"] + need_cols].drop_duplicates("key_sale"),
                                on="key_sale", how="left",
                            )
                            df_comps = _build_comp_quality_table(
                                df_out_full, m_price, m_ppsf_impr, m_ppsf_land, feat_cols, unit,
                            )
                            if len(df_comps) == 0:
                                continue
                            # Join the prediction_ratio back for sorting/context
                            df_comps = df_comps.merge(
                                df_out[["key_sale", "prediction_ratio"]],
                                on="key_sale", how="left",
                            )
                            df_comps = df_comps.sort_values("prediction_ratio")
                            comp_outpath = f"{outdir}outliers_comp_analysis_{pred_label}.csv"
                            df_comps.to_csv(comp_outpath, index=False)
                            print("")
                            print(f"Comp-analysis ({pred_label}): {len(df_comps)} outliers -> {comp_outpath}")
                            display_cols = [
                                "key_sale", "prediction_ratio", "vacant_sale", "ppsf_unit",
                                "actual_price", "leaf_wilson_price", "price_ratio",
                                "tree_depth_price", "leaf_count_price",
                                "actual_ppsf", "leaf_wilson_ppsf", "ppsf_ratio",
                                "tree_depth_ppsf", "leaf_count_ppsf",
                            ]
                            display_cols = [c for c in display_cols if c in df_comps.columns]
                            display(df_comps[display_cols])
                        except Exception as e:
                            warnings.warn(f"Comp-analysis failed for {id}/{mtype}/{pred_label}: {e}")

                print("")



def finalize_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    verbose: bool = False,
    run_main: bool = True,
    run_vacant: bool = True,
    run_ensemble: bool = True,
) -> None:
    """
    Tries out predictive models on the given SalesUniversePair, finalizes results and writes to disk.

    This function takes detailed instructions from the provided settings dictionary and handles all the internal
    details like splitting the data, training the models, and saving the results. It performs basic statistic analysis
    on each model, and optionally combines results into an ensemble model.

    This function iterates over model groups and runs models for main and vacant cases.

    It delegates the model execution to `openavmkit.model_runner.run_models` with the given settings.

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
    run_main : bool, optional
        Flag to run main models. Defaults to True.
    run_vacant : bool, optional
        Flag to run vacant models. Defaults to True.
    run_ensemble : bool, optional
        Flag to run ensemble models. Defaults to True.

    Returns
    -------
    MultiModelResults
        The MultiModelResults containing all model results and benchmarks.
    """

    openavmkit.model_runner.run_models(
        sup,
        settings,
        save_params,
        use_saved_params,
        save_results=True,
        verbose=verbose,
        run_main=run_main,
        run_vacant=run_vacant,
        run_ensemble=run_ensemble,
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
    run_ensemble: bool = True,
    do_shaps: bool = False,
    do_plots: bool = False
):
    """
    Runs predictive models on the given SalesUniversePair.

    This function takes detailed instructions from the provided settings dictionary and handles all the internal
    details like splitting the data, training the models, and saving the results. It performs basic statistic analysis
    on each model, and optionally combines results into an ensemble model.

    If "run_main" is true, it will run normal (full market value) models.
    If "run_vacant" is true, it will run vacant models as well -- models that only use vacant sales as evidence
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
    return openavmkit.model_runner.run_models(
        sup,
        settings,
        save_params,
        use_saved_params,
        save_results,
        verbose,
        run_main,
        run_vacant,
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


def read_sales_univ(path: str):
    return openavmkit.data.read_sales_univ(path)


def run_ratio_study(
    sup: SalesUniversePair,
    model_group: str,
    field_prediction: str,
    field_sales: str,
    start_date: str = None,
    end_date: str = None,
    land_only: bool = False,
    max_trim: float = 0.05
):
    # Filter to just the designated model group
    sup = get_sup_model_group(sup, model_group)
    
    # Merge universe characteristics onto sales to create one combined dataframe
    df_sales = get_hydrated_sales_from_sup(sup)

    # Select only sales between the start and end date
    if start_date is not None and end_date is not None:
        df_sales = df_sales[
            df_sales["sale_date"].ge(start_date) &
            df_sales["sale_date"].le(end_date)
    ]
    
    valid_field = "valid_for_ratio_study"
    if "valid_for_ratio_study" not in df_sales:
        valid_field = "valid_sale"
    
    if land_only:
        valid_field = "valid_for_land_ratio_study"
        if "valid_for_land_ratio_study" not in df_sales:
            valid_field = "vacant_sale"
        
    if valid_field in df_sales:
        df_sales = df_sales[df_sales[valid_field].eq(True)]

    # Ensure only non-null values are selected
    
    df_sales_clean = df_sales[
        ~df_sales[field_prediction].isna() &
        ~df_sales[field_sales].isna() &
        df_sales[field_sales].gt(0)
    ]
    
    # Get predictions and sales
    predictions = df_sales_clean[field_prediction]
    sales = df_sales_clean[field_sales]
    
    # Run the ratio study and return it
    return RatioStudyBootstrapped(predictions, sales, max_trim)


def run_horizontal_equity_study(
    sup: SalesUniversePair,
    model_group: str,
    field: str,
    cluster_id: str = "he_id",
):
    if cluster_id not in sup.universe:
        return None
    
    # Filter to just the designated model group
    sup = get_sup_model_group(sup, model_group)

    df = sup.universe

    he_study = HorizontalEquityStudy(df, cluster_id, field)
    return he_study


def run_vertical_equity_study(
    sup: SalesUniversePair,
    model_group: str,
    field_prediction: str,
    field_sales: str,
    field_location: str,
    start_date: str,
    end_date: str,
    max_trim: float = 0.05
):
    # Filter to just the designated model group
    sup = get_sup_model_group(sup, model_group)

    # Merge universe characteristics onto sales to create one combined dataframe
    df_sales = get_hydrated_sales_from_sup(sup)

    # Select only sales between the start and end date
    df_sales = df_sales[
        df_sales["sale_date"].ge(start_date) &
        df_sales["sale_date"].le(end_date)
    ]

    # If no usable rows survive (empty model-group / date window, or no finite
    # positive sales), skip the study — mirrors `run_horizontal_equity_study`.
    if len(df_sales) == 0:
        return None
    usable = (
        np.isfinite(df_sales[field_prediction])
        & np.isfinite(df_sales[field_sales])
        & df_sales[field_sales].gt(0)
    )
    if not usable.any():
        return None

    # Run the vertical equity study and print the results
    return VerticalEquityStudy(
        df_sales,
        field_sales,
        field_prediction,
        field_location
    )


def plot_prediction_vs_sales(
    sup: SalesUniversePair,
    model_group: str,
    field_prediction: str,
    field_truth: str,
    start_date: str,
    end_date: str,
    land_only: bool = False,
    max_prediction: float = None,
    max_truth: float = None
):
    # Filter to just the designated model group
    sup = get_sup_model_group(sup, model_group)
    
    # Merge universe characteristics onto sales to create one combined dataframe
    df_sales = get_hydrated_sales_from_sup(sup)

    # Select only sales between the start and end date
    df_sales = df_sales[
        df_sales["sale_date"].ge(start_date) &
        df_sales["sale_date"].le(end_date)
    ]
    
    if max_prediction is not None:
        df_sales = df_sales[df_sales[field_prediction].le(max_prediction)]
    
    if max_truth is not None:
        df_sales = df_sales[df_sales[field_truth].lt(max_truth)]
    
    valid_field = "valid_for_ratio_study"
    if "valid_for_ratio_study" not in df_sales:
        valid_field = "valid_sale"
    if land_only == True:
        valid_field = "valid_for_land_ratio_study"
        if "valid_for_land_ratio_study" not in df_sales:
            valid_field = "vacant_sale"
    
    if valid_field in df_sales:
        df_sales = df_sales[df_sales[valid_field].eq(True)]
    
    plot_scatterplot(
        df_sales, 
        field_truth, 
        field_prediction, 
        field_truth, 
        field_prediction, 
        "Sale price vs. Prediction",
        best_fit_line=True, 
        perfect_fit_line=True
    )
    
# PRIVATE:


def _clip_sales_to_use(
    df_sales: pd.DataFrame, settings: dict, verbose: bool = False
) -> pd.DataFrame:

    val_year = get_valuation_date(settings).year

    # Global clip uses the FLOOR (widest window any group needs); per-group narrowing
    # happens later at the train/test split. See use_sales_from_floor for rationale.
    from openavmkit.utilities.settings import use_sales_from_floor
    use_sales_from_impr, use_sales_from_vacant = use_sales_from_floor(settings)
    if use_sales_from_impr is None:
        use_sales_from_impr = val_year - 5
    if use_sales_from_vacant is None:
        use_sales_from_vacant = val_year - 5

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
