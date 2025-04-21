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
import openavmkit.benchmark
import openavmkit.land
import openavmkit.checkpoint
import openavmkit.ratio_study
import openavmkit.horizontal_equity_study
import openavmkit.cleaning
from openavmkit.benchmark import get_variable_recommendations

from openavmkit.cleaning import clean_valid_sales, validate_arms_length_sales
from openavmkit.cloud import cloud
from openavmkit.data import _load_dataframe, process_data, SalesUniversePair, get_hydrated_sales_from_sup
from openavmkit.sales_scrutiny_study import run_sales_scrutiny_per_model_group, mark_ss_ids_per_model_group
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import combine_dfs
from openavmkit.utilities.settings import get_fields_categorical, get_fields_numeric, get_fields_boolean, \
   get_fields_land, get_fields_impr, get_fields_other


# Basic data stuff

class NotebookState:
   """
   Represents the state of a notebook session including the base path and locality.

   Attributes:
       base_path (str): The base directory path for the notebook.
       locality (str): The locality identifier (e.g., "us-nc-guilford").
   """
   base_path: str
   locality: str

   def __init__(self, locality: str, base_path: str = None):
      """
      Initialize a NotebookState instance.

      :param locality: The locality slug (e.g., "us-nc-guilford").
      :type locality: str
      :param base_path: The base directory path. Defaults to the current working directory if not provided.
      :type base_path: str, optional
      """
      self.locality = locality
      if base_path is None:
         base_path = os.getcwd()
      self.base_path = base_path


def init_notebook(locality: str):
   """
   Initialize the notebook environment for a specific locality.

   This function sets up the notebook state by configuring the working directory
   and ensuring that the appropriate data directories exist.

   :param locality: The locality slug (e.g., "us-nc-guilford").
   :type locality: str
   :returns: None
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


def load_settings(settings_file: str = "in/settings.json", settings_object: dict = None):
   """
   Load and return the settings dictionary for the locality.

   This merges the user's settings for their specific locality with the default settings template and the default data dictionary.
   It also performs variable substitution. The result is a fully resolved settings dictionary.

   :param settings_file: Path to the settings file.
   :type settings_file: str, optional
   :param settings_object: Optional settings object to use instead of loading from a file.
   :type settings_object: dict, optional

   :returns: The settings dictionary.
   :rtype: dict
   """
   return openavmkit.utilities.settings.load_settings(settings_file, settings_object)


def examine_sup_in_ridiculous_detail(sup: SalesUniversePair, s: dict):
   print("")
   print("EXAMINING UNIVERSE...")
   print("")
   examine_df_in_ridiculous_detail(sup["universe"], s)

   print("")
   print("EXAMINING SALES...")
   print("")
   examine_df_in_ridiculous_detail(sup["sales"], s)


def examine_sup(sup: SalesUniversePair, s: dict):
   """
   Print examination details of the sales and universe data from a SalesUniversePair.

   This function displays summary statistics and unique values for both the sales
   and universe dataframes.

   :param sup: A dictionary-like object containing 'sales' and 'universe' DataFrames.
   :type sup: SalesUniversePair
   :param s: Settings dictionary
   :type s: dict
   :returns: None
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
         last_bit = txt[len(txt) - len_last:]
         txt = first_bit + "..." + last_bit
      return f"{txt:{size}}"

   def get_num_line(col):
      describe = df[col].describe()
      return f"DESCRIBE --> {describe}\n\n"

   def get_cat_line(col):
      value_counts = df[col].value_counts()
      return f"VALUE COUNTS --> {value_counts}\n\n"

   def get_line(col, dtype, count_non_zero, p, count_non_null, pnn, uniques: list or str):
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
      print(fill_str(char, 30) + " " + fill_str(char, 10) + " " + fill_str(char, 10) +
            " " + fill_str(char, 5) + " " + fill_str(char, 10) + " " + fill_str(char, 5) +
            " " + fill_str(char, 40))

   print(f"{'FIELD':^30} {'TYPE':^10} {'NON-ZERO':^10} {'%':^5} {'NON-NULL':^10} {'%':^5} {'UNIQUE':^40}")

   fields_land = get_fields_land(s, df)
   fields_impr = get_fields_impr(s, df)
   fields_other = get_fields_other(s, df)

   fields_noted = []

   stuff = {
      "land": {"name": "LAND", "fields": fields_land},
      "impr": {"name": "IMPROVEMENT", "fields": fields_impr},
      "other": {"name": "OTHER", "fields": fields_other}
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
            print(get_line(n, df[n].dtype, non_zero, perc, non_null, perc_non_null, ""))
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
            print(get_line(b, df[b].dtype, non_zero, perc, non_null, perc_non_null, df[b].unique().tolist()))

      if len(cats) > 0:
         print_horz_line("-")
         print(f"{'CATEGORICAL':^30}")
         print_horz_line("-")
         for c in cats:
            fields_noted.append(c)
            non_zero = (~pd.isna(df[c])).sum()
            perc = non_zero / len(df)
            print(get_line(c, df[c].dtype, non_zero, perc, non_zero, perc, df[c].unique().tolist()))
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
         print(get_line(u, df[u].dtype, non_zero, perc, non_zero, perc, list(df[u].unique())))


def examine_df(df: pd.DataFrame, s: dict):
   """
   Display detailed statistics for each column in a DataFrame.

   The function prints a formatted table including the field name, data type,
   non-zero counts, percentage of non-zero values, non-null counts, and unique values.

   :param df: The DataFrame to examine.
   :type df: pd.DataFrame
   :param s: Settings dictionary
   :type s: dict
   :returns: None

   :note: This function contains several helper functions to format the output.
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
         last_bit = txt[len(txt) - len_last:]
         txt = first_bit + "..." + last_bit
      return f"{txt:{size}}"

   def get_line(col, dtype, count_non_zero, p, count_non_null, pnn, uniques: list or str):
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
      print(fill_str(char, 30) + " " + fill_str(char, 10) + " " + fill_str(char, 10) +
            " " + fill_str(char, 5) + " " + fill_str(char, 10) + " " + fill_str(char, 5) +
            " " + fill_str(char, 40))

   print(f"{'FIELD':^30} {'TYPE':^10} {'NON-ZERO':^10} {'%':^5} {'NON-NULL':^10} {'%':^5} {'UNIQUE':^40}")

   fields_land = get_fields_land(s, df)
   fields_impr = get_fields_impr(s, df)
   fields_other = get_fields_other(s, df)

   fields_noted = []

   stuff = {
      "land": {"name": "LAND", "fields": fields_land},
      "impr": {"name": "IMPROVEMENT", "fields": fields_impr},
      "other": {"name": "OTHER", "fields": fields_other}
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
            print(get_line(n, df[n].dtype, non_zero, perc, non_null, perc_non_null, ""))

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
            print(get_line(b, df[b].dtype, non_zero, perc, non_null, perc_non_null, df[b].unique().tolist()))

      if len(cats) > 0:
         print_horz_line("-")
         print(f"{'CATEGORICAL':^30}")
         print_horz_line("-")
         for c in cats:
            fields_noted.append(c)
            non_zero = (~pd.isna(df[c])).sum()
            perc = non_zero / len(df)
            print(get_line(c, df[c].dtype, non_zero, perc, non_zero, perc, df[c].unique().tolist()))
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
         print(get_line(u, df[u].dtype, non_zero, perc, non_zero, perc, list(df[u].unique())))


# Data loading & processing stuff

def load_dataframes(settings: dict, verbose: bool = False) -> dict:
   """
   Load dataframes based on the provided settings and returns them in a dictionary.

   This function reads various data sources defined in the settings and loads them
   into pandas DataFrames. It performs validations to ensure required data, such as
   'geo_parcels', is present and correctly formatted.

   :param settings: Settings dictionary.
   :type settings: dict
   :param verbose: If True, prints detailed logs during data loading.
   :type verbose: bool, optional
   :returns: A dictionary mapping keys to loaded DataFrames.
   :rtype: dict
   :raises ValueError: If required dataframes or columns (e.g., 'geo_parcels' or its 'geometry' column) are missing.
   """
   s_data = settings.get("data", {})
   s_load = s_data.get("load", {})
   dataframes = {}

   fields_cat = get_fields_categorical(settings, include_boolean=False)
   fields_bool = get_fields_boolean(settings)
   fields_num = get_fields_numeric(settings, include_boolean=False)

   for key in s_load:
      entry = s_load[key]
      df = _load_dataframe(entry, settings, verbose=verbose,
         fields_cat=fields_cat, fields_bool=fields_bool, fields_num=fields_num)
      if df is not None:
         dataframes[key] = df

   if "geo_parcels" not in dataframes:
      raise ValueError("No 'geo_parcels' dataframe found in the dataframes. This layer is required, and it must contain parcel geometry.")

   if "geometry" not in dataframes["geo_parcels"].columns:
      raise ValueError("The 'geo_parcels' dataframe does not contain a 'geometry' column. This layer must contain parcel geometry.")

   return dataframes


def load_and_process_data(settings: dict) -> SalesUniversePair:
   """
   Load and process data according to provided settings.

   This function first loads the dataframes, then merges and enriches the data, returning a SalesUniversePair.

   :param settings: A dictionary of settings for data loading and processing.
   :type settings: dict
   :returns: A SalesUniversePair object containing the processed sales and universe data.
   :rtype: SalesUniversePair
   """
   dataframes = load_dataframes(settings)
   results = process_data(dataframes, settings)
   return results


def tag_model_groups_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False):
   """
   Tag model groups for a SalesUniversePair.

   This function applies user-specified filters that identify rows belonging to particular model groups, then writes the
   results to the "model_group" field.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param verbose: If True, enables verbose output.
   :type verbose: bool, optional
   :returns: Updated SalesUniversePair with tagged model groups.
   :rtype: SalesUniversePair
   """
   return openavmkit.data._tag_model_groups_sup(sup, settings, verbose)


def process_sales(sup: SalesUniversePair, settings: dict, verbose: bool = False):
   """
   Process sales data within a SalesUniversePair.

   This function cleans invalid sales, applies time adjustments, and updates the SalesUniversePair with the enriched
   sales DataFrame.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param verbose: If True, prints verbose output during processing.
   :type verbose: bool, optional
   :returns: Updated SalesUniversePair with processed sales data.
   :rtype: SalesUniversePair
   """
   # select only valid sales
   sup = clean_valid_sales(sup, settings)

   print(f"len before tag = {len(sup['sales'])}")

   # tag model groups before validation
   sup = tag_model_groups_sup(sup, settings, verbose)

   print(f"len before validate = {len(sup['sales'])}")

   # validate arms length sales using outlier detection
   sup = validate_arms_length_sales(sup, settings, verbose)

   print(f"len after validate = {len(sup['sales'])}")

   # clean again to remove newly marked invalid sales
   sup = clean_valid_sales(sup, settings)

   print(f"len after final clean = {len(sup['sales'])}")

   # make sure sales field has necessary fields for the next step
   df_sales_hydrated = get_hydrated_sales_from_sup(sup)

   print(f"len after hydrate = {len(sup['sales'])}")

   # enrich with time adjustment, and mark what fields were added
   df_sales_enriched = enrich_time_adjustment(df_sales_hydrated, settings, verbose)

   print(f"len after enrich = {len(df_sales_enriched)}")

   # update the SUP sales
   sup.update_sales(df_sales_enriched)

   # enrich sales w/ spatial lag fields
   sup = enrich_sup_spatial_lag(sup, settings, verbose=verbose)

   return sup


def enrich_sup_spatial_lag(sup: SalesUniversePair, settings: dict, verbose: bool=False):
   return openavmkit.data.enrich_sup_spatial_lag(sup, settings, verbose)


def fill_unknown_values_sup(sup: SalesUniversePair, settings: dict) -> SalesUniversePair:
   """
   Fill unknown or missing values in a SalesUniversePair.

   This function cleans the SalesUniversePair by replacing unknown or missing values based on the settings provided.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :returns: Updated SalesUniversePair with unknown values filled.
   :rtype: SalesUniversePair
   """
   return openavmkit.cleaning.fill_unknown_values_sup(sup, settings)


# Clustering stuff

def mark_ss_ids_per_model_group_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False) -> SalesUniversePair:
   """
   Clusters parcels for sales scrutiny study, making them with 'sales scrutiny ids.' This is done for each model group
   within a SalesUniversePair. Marking ids ahead of time allows for more efficient processing later.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param verbose: If True, prints verbose output during processing.
   :type verbose: bool, optional
   :returns: Updated SalesUniversePair with marked sales scrutiny IDs.
   :rtype: SalesUniversePair
   """
   df_sales_hydrated = get_hydrated_sales_from_sup(sup)
   df_marked = mark_ss_ids_per_model_group(df_sales_hydrated, settings, verbose)
   sup.update_sales(df_marked)
   return sup


def mark_horizontal_equity_clusters_per_model_group_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False):
   """
   Clusters parcels for horizontal equity study, marking them with 'horizontal equity cluster ids.' This is done for
   each model group within a SalesUniversePair. Marking ids ahead of time allows for more efficient processing later.

   Delegates the operation to the :func:`openavmkit.horizontal_equity_study.mark_horizontal_equity_clusters_per_model_group_sup` function.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param verbose: If True, prints verbose output.
   :type verbose: bool, optional
   :returns: Updated SalesUniversePair with horizontal equity clusters marked.
   :rtype: SalesUniversePair
   """
   return openavmkit.horizontal_equity_study.mark_horizontal_equity_clusters_per_model_group_sup(sup, settings, verbose)


def run_sales_scrutiny_per_model_group_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False) -> SalesUniversePair:
   """
   Run sales scrutiny analysis for each model group within a SalesUniversePair. Assumes that the SalesUniversePair has
   already been processed and marked with sales scrutiny ids.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param verbose: If True, enables verbose logging.
   :type verbose: bool, optional
   :returns: Updated SalesUniversePair after sales scrutiny analysis.
   :rtype: SalesUniversePair
   """
   df_sales_hydrated = get_hydrated_sales_from_sup(sup)
   df_scrutinized = run_sales_scrutiny_per_model_group(df_sales_hydrated, settings, verbose)
   sup.update_sales(df_scrutinized)
   return sup


# Read & write stuff

def from_checkpoint(path: str, func: callable, params: dict) -> pd.DataFrame:
   """
   Wrapper function -- read cached data from a checkpoint file if it exists, otherwise run the specified function to
   generate the data (and save it to a checkpoint file).

   :param path: Path to the checkpoint file.
   :type path: str
   :param func: Function to run if checkpoint is not available.
   :type func: callable
   :param params: Parameters for the function.
   :type params: dict
   :returns: The resulting DataFrame.
   :rtype: pd.DataFrame
   """
   return openavmkit.checkpoint.from_checkpoint(path, func, params)


def delete_checkpoints(prefix: str):
   """
   Delete all checkpoints that match the given prefix.

   :param prefix: The prefix used to identify checkpoints to delete.
   :type prefix: str
   :returns: FILL_IN_HERE: Describe return value if any.
   """
   return openavmkit.checkpoint.delete_checkpoints(prefix)


def write_checkpoint(data: Any, path: str):
   """
   Write data to a checkpoint file -- a parquet file for dataframes, and a pickle file for anything else.

   :param data: Data to be checkpointed.
   :type data: Any
   :param path: File path for saving the checkpoint.
   :type path: str
   :returns: FILL_IN_HERE: Describe return value if any.
   """
   return openavmkit.checkpoint.write_checkpoint(data, path)


def write_notebook_output_sup(sup: SalesUniversePair, prefix="1-assemble"):
   """
   Write notebook output to disk.

   This function saves the SalesUniversePair as a pickle file and writes the corresponding
   'universe' and 'sales' DataFrames to Parquet files.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param prefix: File prefix for naming output files.
   :type prefix: str, optional
   :returns: None
   """
   with open(f"out/{prefix}-sup.pickle", "wb") as file:
      pickle.dump(sup, file)
   os.makedirs("out/look", exist_ok=True)

   # Handle geometry columns for both universe and sales
   def prepare_df_for_parquet(df):
      if "geometry" in df.columns:
         # Convert geometry to WKB for storage
         df = df.copy()
         if hasattr(df, 'to_wkb'):
            # If it's a GeoDataFrame, use to_wkb() method
            df["geometry"] = df.geometry.to_wkb()
         else:
            # If it's a regular DataFrame with geometry column
            import shapely.wkb
            df["geometry"] = df["geometry"].apply(lambda geom: geom.wkb if geom is not None else None)
      return df

   # Prepare and write universe DataFrame
   df_universe = prepare_df_for_parquet(sup["universe"])
   df_universe.to_parquet(f"out/look/{prefix}-universe.parquet", engine="pyarrow")

   # Prepare and write sales DataFrame

   df_sales = prepare_df_for_parquet(sup["sales"])
   df_sales.to_parquet(f"out/look/{prefix}-sales.parquet")

   df_hydrated = get_hydrated_sales_from_sup(sup)
   df_hydrated = prepare_df_for_parquet(df_hydrated)
   df_hydrated.to_parquet(f"out/look/{prefix}-sales-hydrated.parquet", engine="pyarrow")

   print("Results written to:")
   print(f"...out/{prefix}-sup.pickle")
   print(f"...out/look/{prefix}-universe.parquet")
   print(f"...out/look/{prefix}-sales.parquet")
   print(f"...out/look/{prefix}-sales-hydrated.parquet")


def cloud_sync(locality: str, verbose: bool = False, env_path: str = "", settings: dict = None, dry_run: bool = False, ignore_paths: list = None):
   """
   Synchronize local files to the cloud storage.

   This function initializes the cloud service and syncs files for the given locality.

   :param locality: The locality identifier used to form remote paths.
   :type locality: str
   :param verbose: If True, prints detailed log messages.
   :type verbose: bool, optional
   :param dry_run: If True, simulates the sync without performing any changes.
   :type dry_run: bool, optional
   :param ignore_paths: List of file paths or patterns to ignore during sync.
   :type ignore_paths: list, optional
   :returns: None
   """
   cloud_service = cloud.init(verbose, env_path = env_path, settings = settings)
   if cloud_service is None:
      print("Cloud service not initialized, skipping...")
      return

   if ignore_paths is None:
      ignore_paths = []
   extra_ignore = settings.get("cloud",{}).get("ignore_paths", [])
   ignore_paths = ignore_paths + extra_ignore

   print(f"ignore_paths = {ignore_paths}")

   remote_path = locality.replace("-", "/") + "/"
   cloud_service.sync_files(locality, "in", remote_path, dry_run=dry_run, verbose=verbose, ignore_paths=ignore_paths)


def read_pickle(path: str):
   """
   Read and return data from a pickle file.

   :param path: Path to the pickle file.
   :type path: str
   :returns: FILL_IN_HERE: Describe the returned object.
   """
   return openavmkit.checkpoint.read_pickle(path)


# Modeling stuff


def try_variables(
    sup: SalesUniversePair,
    settings: dict,
    verbose: bool = False,
    plot: bool = False
):

   sup = fill_unknown_values_sup(sup, settings)
   openavmkit.benchmark.try_variables(
      sup,
      settings,
      verbose,
      plot
   )


def try_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    verbose: bool = False,
    run_main: bool = True,
    run_vacant: bool = True,
    run_hedonic: bool = True,
    run_ensemble: bool = True
):
   """
   Tries out predictive models on the given SalesUniversePair. Optimized for speed and iteration, doesn't finalize
   results or write anything to disk.

   This function takes detailed instructions from the provided settings dictionary and handles all the internal details
   like splitting the data, training the models, and saving the results. It performs basic statistic analysis on each
   model, and optionally combines results into an ensemble model.

   If "run_main" is true, it will run normal models as well as hedonic models (if the user so specifies), "hedonic" in
   this context meaning models that attempt to generate a land value and an improvement value separately. If "run_vacant"
   is true, it will run vacant models as well -- models that only use vacant models as evidence to generate land values.

   This function delegates the model execution to :func:`openavmkit.benchmark.run_models` with the given settings.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param save_params: Whether to save model parameters.
   :type save_params: bool, optional
   :param use_saved_params: Whether to use saved model parameters.
   :type use_saved_params: bool, optional
   :param verbose: If True, enables verbose output.
   :type verbose: bool, optional
   :param run_main: Flag to run main models.
   :type run_main: bool, optional
   :param run_vacant: Flag to run vacant models.
   :type run_vacant: bool, optional
   :param run_hedonic: Flag to run hedonic models.
   :type run_hedonic: bool, optional
   :param run_ensemble: Flag to run ensemble models.
   :type run_ensemble: bool, optional
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
      run_ensemble=run_ensemble
   )


def finalize_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = True,
    use_saved_params: bool = True,
    verbose: bool = False
):
   """
   Finalizes predictive models on the given SalesUniversePair. Generates final predictions and writes them to disk for
   the rest of the pipeline to use.

   This function takes detailed instructions from the provided settings dictionary and handles all the internal details
   like splitting the data, training the models, and saving the results. It performs basic statistic analysis on each
   model, and optionally combines results into an ensemble model.

   If "run_main" is true, it will run normal models as well as hedonic models (if the user so specifies), "hedonic" in
   this context meaning models that attempt to generate a land value and an improvement value separately. If "run_vacant"
   is true, it will run vacant models as well -- models that only use vacant models as evidence to generate land values.

   This function delegates the model execution to :func:`openavmkit.benchmark.run_models` with the given settings.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param save_params: Whether to save model parameters.
   :type save_params: bool, optional
   :param use_saved_params: Whether to use saved model parameters.
   :type use_saved_params: bool, optional
   :param verbose: If True, enables verbose output.
   :type verbose: bool, optional
   """
   print("OAIJSDOISAK")
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
      run_ensemble=True
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
    run_ensemble: bool = True
):
   """
   Runs predictive models on the given SalesUniversePair. This function takes detailed instructions from the provided
   settings dictionary and handles all the internal details like splitting the data, training the models, and saving the
   results. It performs basic statistic analysis on each model, and optionally combines results into an ensemble model.

   If "run_main" is true, it will run normal models as well as hedonic models (if the user so specifies), "hedonic" in
   this context meaning models that attempt to generate a land value and an improvement value separately. If "run_vacant"
   is true, it will run vacant models as well -- models that only use vacant models as evidence to generate land values.

   This function delegates the model execution to
   :func:`openavmkit.benchmark.run_models` with the given settings.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param save_params: Whether to save model parameters.
   :type save_params: bool, optional
   :param use_saved_params: Whether to use saved model parameters.
   :type use_saved_params: bool, optional
   :param save_results: Whether to save model results.
   :type save_results: bool, optional
   :param verbose: If True, enables verbose output.
   :type verbose: bool, optional
   :param run_main: Flag to run main models.
   :type run_main: bool, optional
   :param run_vacant: Flag to run vacant models.
   :type run_vacant: bool, optional
   :param run_hedonic: Flag to run hedonic models.
   :type run_hedonic: bool, optional
   :param run_ensemble: Flag to run ensemble models.
   :type run_ensemble: bool, optional
   :returns: FILL_IN_HERE: Describe the output.
   """
   return openavmkit.benchmark.run_models(sup, settings, save_params, use_saved_params, save_results, verbose, run_main, run_vacant, run_hedonic, run_ensemble)


def finalize_land_values_sup(sup: SalesUniversePair, settings: dict, generate_boundaries: bool = False, verbose: bool = False):
   """
   Finalize land values within a SalesUniversePair.

   This function is currently not implemented.

   :param sup: Sales and universe data.
   :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :param generate_boundaries: Flag to generate boundaries.
   :type generate_boundaries: bool, optional
   :param verbose: If True, enables verbose output.
   :type verbose: bool, optional
   :raises NotImplementedError: Always raised as the function is not yet implemented.
   """
   raise NotImplementedError("This function is not yet implemented.")
   # return openavmkit.land.finalize_land_values()


def write_canonical_splits(sup: SalesUniversePair, settings: dict):
   """
   Write canonical splits for the sales DataFrame. This separates the sales data into training and test sets and stores
   the keys to disk. This way, the same splits can be used consistently for multiple models, ensuring that the results
   can be properly ensembled.

   This function delegates the operation to :func:`openavmkit.data._write_canonical_splits`.
   :func:`openavmkit.data._write_canonical_splits`.

   :param sup: Sales and universe data.
    :type sup: SalesUniversePair
   :param settings: Configuration settings.
   :type settings: dict
   :returns: None
   """
   openavmkit.data._write_canonical_splits(sup, settings)


def run_and_write_ratio_study_breakdowns(settings: dict):
   """
   Run ratio study breakdowns and write the results to disk.

   :param settings: Configuration settings for the ratio study.
   :type settings: dict
   :returns: None
   """
   openavmkit.ratio_study.run_and_write_ratio_study_breakdowns(settings)


# PRIVATE:

def _set_locality(nbs, locality: str):
   """
   Set or update the notebook state with a new locality.

   This function updates the NotebookState to reflect the specified locality,
   changes the working directory to the appropriate path, and ensures that the data directory exists.

   :param nbs: An existing NotebookState instance or None.
   :type nbs: NotebookState or None
   :param locality: The new locality identifier.
   :type locality: str
   :returns: Updated NotebookState with the new locality.
   :rtype: NotebookState
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
