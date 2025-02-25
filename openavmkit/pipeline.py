"""
Pipeline
---------
This module contains every public function called from the notebooks. The rules for this module are:

- Every public function should be called from at least one notebook
- The primary openavmkit notebooks should not call any functions from openavmkit that are not in this module
- This module imports from other modules, but no other modules import from it

"""

import os
import pickle

import numpy as np
import pandas as pd

import openavmkit
import openavmkit.checkpoint
from openavmkit.cleaning import clean_valid_sales
from openavmkit.cloud import cloud
from openavmkit.data import load_dataframe, process_data, SalesUniversePair, get_hydrated_sales_from_sup
from openavmkit.sales_scrutiny_study import run_sales_scrutiny_per_model_group, mark_ss_ids_per_model_group
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import combine_dfs
from openavmkit.utilities.settings import get_fields_categorical, get_fields_numeric, get_fields_boolean, \
   get_fields_land, get_fields_impr, get_fields_other


class NotebookState:
   base_path: str
   locality: str

   def __init__(self, locality: str, base_path: str = None):
      self.locality = locality
      if base_path is None:
         base_path = os.getcwd()
         self.base_path = base_path


def init_notebook(locality: str):
   """
    Initialize the notebook for the specific locality. This function should be called at the beginning of every notebook.
   :param locality: The locality slug, e.g. "us-nc-guilford"
   :return: None
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


def examine_sup(sup: SalesUniversePair, s: dict):
   print("")
   print("EXAMINING SALES...")
   print("")
   examine_df(sup["sales"], s)

   print("")
   print("EXAMINING UNIVERSE...")
   print("")
   examine_df(sup["universe"], s)


def examine_df(df: pd.DataFrame, s: dict):

   def fill_str(char: str, size: int):
      text = ""
      for _i in range(0, size):
         text += char
      return text

   def fit_str(txt: str, size: int):
      if len(txt) >= size:
         len_first = int((size-3)/2)
         len_last = (size-3)-len_first
         first_bit = txt[0:len_first]
         last_bit = txt[len(txt)-len_last:]
         txt = first_bit + "..." + last_bit
      return f"{txt:{size}}"

   def get_line(col, dtype, count_non_zero, p, count_non_null, pnn, uniques: list | str):
      dtype = f"{dtype}"
      if type(count_non_zero) != str:
         count_non_zero = f"{count_non_zero:,}"

      if type(count_non_null) != str:
         count_non_null = f"{count_non_null:,}"

      if type(uniques) is list:
         unique_str = str(uniques)
         if len(unique_str) > 40:
            uniques = f"{len(uniques):,}"
         else:
            uniques = unique_str

      return f"{fit_str(col,30)} {dtype:^10} {count_non_zero:>10} {p:>5.0%} {count_non_null:>10} {pnn:>5.0%} {uniques:>40}"

   def print_horz_line(char: str):
      print(fill_str(char,30)+" "+fill_str(char,10)+" "+fill_str(char,10)+" "+fill_str(char, 5)+" "+fill_str(char,10)+" "+fill_str(char, 5)+" "+fill_str(char,40))

   print(f"{'FIELD':^30} {'TYPE':^10} {'NON-ZERO':^10} {'%':^5} {'NON-NULL':^10} {'%':^5} {'UNIQUE':^40}")

   fields_land = get_fields_land(s, df)
   fields_impr = get_fields_impr(s, df)
   fields_other = get_fields_other(s, df)

   fields_noted = []

   stuff = {
      "land": { "name": "LAND", "fields": fields_land},
      "impr": { "name": "IMPROVEMENT", "fields": fields_impr},
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
            non_zero = len(df[np.abs(df[n]).gt(0)])
            perc = non_zero / len(df)
            non_null = len(df[~pd.isna(df[n])])
            perc_non_null = non_null / len(df)
            print(get_line(n, df[n].dtype, non_zero, perc, non_null, perc_non_null,""))

      if len(bools) > 0:
         print_horz_line("-")
         print(f"{'BOOLEAN':^30}")
         print_horz_line("-")
         for b in bools:
            fields_noted.append(b)
            non_zero = len(df[np.abs(df[n]).gt(0)])
            perc = non_zero / len(df)
            non_null = len(df[~pd.isna(df[b])])
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


def load_dataframes(settings: dict, verbose: bool = False) -> dict[str : pd.DataFrame]:
   """
   Load the data from the settings.
   """
   s_data = settings.get("data", {})
   s_load = s_data.get("load", {})
   dataframes = {}

   # TODO: should we even make it optional to include_booleans? Or at least make it false by default?
   fields_cat = get_fields_categorical(settings, include_boolean=False)
   fields_bool = get_fields_boolean(settings)
   fields_num = get_fields_numeric(settings, include_boolean=False)

   for key in s_load:
      entry = s_load[key]
      df = load_dataframe(entry, settings, verbose=verbose, fields_cat=fields_cat, fields_bool=fields_bool, fields_num=fields_num)
      if df is not None:
         dataframes[key] = df

   if "geo_parcels" not in dataframes:
      raise ValueError("No 'geo_parcels' dataframe found in the dataframes. This layer is required, and it must contain parcel geometry.")

   if "geometry" not in dataframes["geo_parcels"].columns:
      raise ValueError("The 'geo_parcels' dataframe does not contain a 'geometry' column. This layer must contain parcel geometry.")

   return dataframes


def load_and_process_data(settings: dict):

   dataframes = load_dataframes(settings)
   results = process_data(dataframes, settings)

   return results


def tag_model_groups_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False):
   return openavmkit.data.tag_model_groups_sup(sup, settings, verbose)


def load_settings():
   return openavmkit.utilities.settings.load_settings()


def process_sales(sup: SalesUniversePair, settings: dict, verbose: bool = False):
   # select only valid sales
   sup = clean_valid_sales(sup, settings)

   # make sure sales field has necessary fields for the next step
   df_sales_hydrated = get_hydrated_sales_from_sup(sup)

   # enrich with time adjustment, and mark what fields were added
   df_sales_enriched = enrich_time_adjustment(df_sales_hydrated, settings, verbose)

   # update the SUP sales
   sup.update_sales(df_sales_enriched)

   return sup


def mark_ss_ids_per_model_group_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False):
   df_sales_hydrated = get_hydrated_sales_from_sup(sup)
   df_marked = mark_ss_ids_per_model_group(df_sales_hydrated, settings, verbose)
   sup.update_sales(df_marked)
   return sup


def run_sales_scrutiny_per_model_group_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False) -> SalesUniversePair:
   df_sales_hydrated = get_hydrated_sales_from_sup(sup)
   df_scrutinized = run_sales_scrutiny_per_model_group(df_sales_hydrated, settings, verbose)
   sup.update_sales(df_scrutinized)
   return sup


def cloud_sync(settings: dict, verbose: bool = False, dry_run: bool = False):
    """
    Syncs the data to the cloud.
    """
    cloud_service = cloud.init(verbose)
    if cloud_service is None:
      print("Cloud service not initialized, skipping...")
      return

    locality = settings.get("locality", {})
    slug : str | None = locality.get("slug", None)
    if slug is None:
        raise ValueError("No slug found in the locality settings.")

    remote_path = slug.replace("-", "/")
    cloud_service.sync_files(slug,"in", remote_path, dry_run=dry_run, verbose=verbose)


def from_checkpoint(path: str, func: callable, params: dict)->pd.DataFrame:
   return openavmkit.checkpoint.from_checkpoint(path, func, params)


def delete_checkpoints(prefix: str):
   return openavmkit.checkpoint.delete_checkpoints(prefix)


def write_out_assemble_results(sup: SalesUniversePair):
   with open (f"out/sales_univ.pickle", "wb") as file:
      pickle.dump(sup, file)
   os.makedirs("out/look", exist_ok=True)
   sup["universe"].to_parquet("out/look/1-assemble-universe.parquet")
   sup["sales"].to_parquet("out/look/1-assemble-sales.parquet")
   print("Results written to:")
   print("...out/sales_univ.pickle")
   print("...out/look/1-assemble-universe.parquet")
   print("...out/look/1-assemble-sales.parquet")


# PRIVATE:


def _set_locality(nbs, locality: str):
   base_path = None
   if nbs is not None:
      base_path = nbs.base_path
      if locality != nbs.locality:
         nbs = NotebookState(locality, base_path)
   if base_path is None:
      nbs = NotebookState(locality, None)

   if base_path is not None:
      os.chdir(nbs.base_path)

   os.chdir(f"data/{locality}")

   print(f"locality = {locality}")
   print(f"base path = {nbs.base_path}")
   print(f"current path = {os.getcwd()}")
   return nbs