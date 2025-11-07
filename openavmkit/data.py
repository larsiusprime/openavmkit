import gc
import math
import os
from datetime import date

import tempfile
import zipfile
import shutil

from pathlib import Path

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Tuple

import shapely

from shapely.geometry import Point
from osmnx import settings
from joblib import Parallel, delayed

import osmnx as ox

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd

from scipy.spatial._ckdtree import cKDTree
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import unary_union
import warnings
import traceback

from shapely.strtree import STRtree

from openavmkit.calculations import (
    _crawl_calc_dict_for_fields,
    perform_calculations,
    perform_tweaks,
)
from openavmkit.filters import resolve_filter, select_filter
from openavmkit.utilities.somers import get_size_in_somers_units_ft
from openavmkit.utilities.cache import get_cached_df, write_cached_df
from openavmkit.utilities.data import (
    combine_dfs,
    div_series_z_safe,
    merge_and_stomp_dfs,
    fill_from_df
)
from openavmkit.utilities.geometry import (
    get_crs,
    clean_geometry,
    identify_irregular_parcels,
    geolocate_point_to_polygon,
    ensure_geometries,
    is_likely_epsg4326,
    detect_crs_from_parquet
)
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_fields_boolean,
    get_fields_numeric,
    get_model_group_ids,
    get_fields_date,
    get_long_distance_unit,
    get_valuation_date,
    get_center,
    get_small_area_unit,
    get_short_distance_unit,
    _get_sales,
    _simulate_removed_buildings,
)

from openavmkit.utilities.census import (
    get_creds_from_env_census,
    init_service_census,
    match_to_census_blockgroups,
)
from openavmkit.utilities.openstreetmap import init_service_openstreetmap
from openavmkit.utilities.overture import init_service_overture
from openavmkit.inference import perform_spatial_inference
from openavmkit.utilities.timing import TimingData
from pyproj import CRS


def log_mem(stage):
    pass
    # mem_mb = process.memory_info().rss / 1024**2
    # logging.info(f"{stage}: {mem_mb:.1f} MB")


@dataclass
class SalesUniversePair:
    """A container for the sales and universe DataFrames, many functions operate on this
    data structure. This data structure is necessary because the sales and universe
    DataFrames are often used together and need to be passed around together. The sales
    represent transactions and any known data at the time of the transaction, while the
    universe represents the current state of all parcels. The sales dataframe specifically
    allows for duplicate primary parcel transaction keys, since an individual parcel may
    have sold multiple times. The universe dataframe forbids duplicate primary parcel keys.

    Attributes
    ----------
    sales : pd.DataFrame
        DataFrame containing sales data.
    universe : pd.DataFrame
        DataFrame containing universe (parcel) data.
    """

    sales: pd.DataFrame
    universe: pd.DataFrame

    def __getitem__(self, key):
        return getattr(self, key)

    def copy(self):
        """Create a copy of the SalesUniversePair object.

        Returns
        -------
        SalesUniversePair
            A new SalesUniversePair object with copied DataFrames.
        """
        return SalesUniversePair(self.sales.copy(), self.universe.copy())

    def set(self, key: str, value: pd.DataFrame):
        """Set the sales or universe DataFrame.

        Attributes
        ----------
        key : str
            Either "sales" or "universe".
        value : pd.DataFrame
            The new DataFrame to set for the specified key.

        Raises
        ------
        ValueError
            If an invalid key is provided
        """
        if key == "sales":
            self.sales = value
        elif key == "universe":
            self.universe = value
        else:
            raise ValueError(f"Invalid key: {key}")
    
    
    def limit_sales_to_keys(self, new_sale_keys: list[str]):
        """
        Update the sales DataFrame to only those that match a key in `new_sale_keys`
        
        Parameters
        ----------
        new_sale_keys : list[str]
            List of sale keys to filter to
        """
        
        s = self.sales.copy()
        s = s[s["key_sale"].isin(new_sale_keys)]
        self.sales = s
    
    
    def update_sales(self, new_sales: pd.DataFrame, allow_remove_rows: bool):
        """
        Update the sales DataFrame with new information as an overlay without redundancy.

        This function lets you push updates to "sales" while keeping it as an "overlay" that
        doesn't contain any redundant information.

        - First we note what fields were in sales last time.
        - Then we note what sales are in universe but were not in sales.
        - Finally, we determine the new fields generated in new_sales that are not in the
          previous sales or in the universe.
        - A modified version of df_sales is created with only two changes:
          - Reduced to the correct selection of keys.
          - Addition of the newly generated fields.

        Parameters
        ----------
        new_sales : pd.DataFrame
            New sales DataFrame with updates.
        allow_remove_rows : bool
            If True, allows the update to remove rows from sales. If False, preserves all
            original rows.
        """

        old_fields = self.sales.columns.values
        univ_fields = [
            field for field in self.universe.columns.values if field not in old_fields
        ]
        new_fields = [
            field
            for field in new_sales.columns.values
            if field not in old_fields and field not in univ_fields
        ]

        old_sales = self.sales.copy()
        return_keys = new_sales["key_sale"].values
        if not allow_remove_rows and len(return_keys) > len(old_sales):
            raise ValueError(
                "The new sales DataFrame contains more keys than the old sales DataFrame. update_sales() may only be used to shrink the dataframe or keep it the same size. Use set() if you intend to replace the sales dataframe."
            )

        if allow_remove_rows:
            old_sales = old_sales[old_sales["key_sale"].isin(return_keys)].reset_index(
                drop=True
            )
        reconciled = combine_dfs(
            old_sales,
            new_sales[["key_sale"] + new_fields].copy().reset_index(drop=True),
            index="key_sale",
        )
        self.sales = reconciled


SUPKey = Literal["sales", "universe"]


def get_hydrated_sales_from_sup(sup: SalesUniversePair):
    """
    Merge the sales and universe DataFrames to "hydrate" the sales data.

    The sales data represents transactions and any known data at the time of the transaction,
    while the universe data represents the current state of all parcels. When we merge the
    two sets, the sales data overrides any existing data in the universe data. This is useful
    for creating a "hydrated" sales DataFrame that contains all the information available at
    the time of the sale (it is assumed that any difference between the current state of the
    parcel and the state at the time of the sale is accounted for in the sales data).

    If the merged DataFrame contains a "geometry" column and the original sales did not,
    the result is converted to a GeoDataFrame.

    Parameters
    ----------
    sup : SalesUniversePair
        SalesUniversePair containing sales and universe DataFrames.

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        The merged (hydrated) sales DataFrame.
    """

    df_sales = sup["sales"]
    df_univ = sup["universe"].copy()
    df_univ = df_univ[df_univ["key"].isin(df_sales["key"].values)].reset_index(
        drop=True
    )
    df_merged = merge_and_stomp_dfs(df_sales, df_univ, df2_stomps=False)

    if "geometry" in df_merged and "geometry" not in df_sales:
        # convert df_merged to geodataframe:
        df_merged = gpd.GeoDataFrame(df_merged, geometry="geometry")

    return df_merged


def enrich_time(df: pd.DataFrame, time_formats: dict, settings: dict) -> pd.DataFrame:
    """
    Enrich the DataFrame by converting specified time fields to datetime and deriving additional fields.

    For each key in time_formats, converts the column to datetime. Then, if a field with
    the prefix "sale" exists, enriches the DataFrame with additional time fields (e.g.,
    "sale_year", "sale_month", "sale_age_days").

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    time_formats : dict
        Dictionary mapping field names to datetime formats.
    settings : dict
        Settings dictionary.

    Returns
    -------
    pandas.DataFrame
        DataFrame with enriched time fields.
    """

    for key in time_formats:
        time_format = time_formats[key]
        if key in df:
            df[key] = pd.to_datetime(df[key], format=time_format, errors="coerce")

    for prefix in ["sale"]:
        do_enrich = False
        for col in df.columns.values:
            if f"{prefix}_" in col:
                do_enrich = True
                break
        if do_enrich:
            df = _enrich_time_field(
                df, prefix, add_year_month=True, add_year_quarter=True
            )
            if prefix == "sale":
                df = _enrich_sale_age_days(df, settings)

    return df


def get_sale_field(settings: dict, df: pd.DataFrame = None) -> str:
    """
    Determine the appropriate sale price field ("sale_price" or "sale_price_time_adj")
    based on time adjustment settings.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        Optional DataFrame to check field existence.

    Returns
    -------
    str
        Field name to be used for sale price.
    """

    ta = settings.get("data", {}).get("process", {}).get("time_adjustment", {})
    use = ta.get("use", True)
    if use:
        sale_field = "sale_price_time_adj"
    else:
        sale_field = "sale_price"
    if df is not None:
        if sale_field == "sale_price_time_adj" and "sale_price_time_adj" in df:
            return "sale_price_time_adj"
    return sale_field


def get_vacant_sales(
    df_in: pd.DataFrame, settings: dict, invert: bool = False
) -> pd.DataFrame:
    """
    Filter the sales DataFrame to return only vacant (unimproved) sales.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame.
    settings : dict
        Settings dictionary.
    invert : bool, optional
        If True, return non-vacant (improved) sales.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an added `is_vacant` column.
    """

    df = df_in.copy()
    df = _boolify_column_in_df(df, "vacant_sale", "na_false")
    idx_vacant_sale = df["vacant_sale"].eq(True)
    if invert:
        idx_vacant_sale = ~idx_vacant_sale
    df_vacant_sales = df[idx_vacant_sale].copy()
    return df_vacant_sales


def get_vacant(
    df_in: pd.DataFrame, settings: dict, invert: bool = False
) -> pd.DataFrame:
    """
    Filter the DataFrame based on the 'is_vacant' column.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame.
    settings : dict
        Settings dictionary.
    invert : bool, optional
        If True, return non-vacant rows.

    Returns
    -------
    pandas.DataFrame
        DataFrame filtered by the `is_vacant` flag.

    Raises
    ------
    ValueError
        If the `is_vacant` column is not boolean.
    """

    df = df_in.copy()
    is_vacant_dtype = df["is_vacant"].dtype
    if is_vacant_dtype != bool:
        raise ValueError(
            f"The 'is_vacant' column must be a boolean type (found: {is_vacant_dtype})"
        )
    idx_vacant = df["is_vacant"].eq(True)
    if invert:
        idx_vacant = ~idx_vacant
    df_vacant = df[idx_vacant].copy()
    return df_vacant


def get_report_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Retrieve report location fields from settings.

    These are location fields that will be used in report breakdowns, such as for ratio studies.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        Optional DataFrame to filter available locations.

    Returns
    -------
    list[str]
        List of report location field names.
    """

    locations = (
        settings.get("field_classification", {})
        .get("important", {})
        .get("report_locations", [])
    )
    if df is not None:
        locations = [loc for loc in locations if loc in df]
    return locations


def get_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Retrieve location fields from settings. These are all the fields that are considered locations.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        Optional DataFrame to filter available locations.

    Returns
    -------
    list[str]
        List of location field names.
    """

    locations = (
        settings.get("field_classification", {})
        .get("important", {})
        .get("locations", [])
    )
    if df is not None:
        locations = [loc for loc in locations if loc in df]
    return locations


def get_important_fields(settings: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Retrieve important field names from settings.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        Optional DataFrame to filter fields.

    Returns
    -------
    list[str]
        List of important field names.
    """

    imp = settings.get("field_classification", {}).get("important", {})
    fields = imp.get("fields", {})
    list_fields = []
    if df is not None:
        for field in fields:
            other_name = fields[field]
            if other_name in df:
                list_fields.append(other_name)
    return list_fields


def get_important_field(
    settings: dict, field_name: str, df: pd.DataFrame = None
) -> str | None:
    """
    Retrieve the important field name for a given field alias from settings.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    field_name : str
        Identifier for the field.
    df : pandas.DataFrame, optional
        Optional DataFrame to check field existence.

    Returns
    -------
    str or None
        The mapped field name if found, else None.
    """

    imp = settings.get("field_classification", {}).get("important", {})
    other_name = imp.get("fields", {}).get(field_name, None)
    if df is not None:
        if other_name is not None and other_name in df:
            return other_name
        else:
            return None
    return other_name


def get_field_classifications(settings: dict) -> dict:
    """
    Retrieve a mapping of field names to their classifications (land, improvement or other)
    as well as their types (numeric, categorical, or boolean).

    Parameters
    ----------
    settings : dict
        Settings dictionary.

    Returns
    -------
    dict
        Dictionary mapping field names to type and class.
    """

    field_map = {}
    for ftype in ["land", "impr", "other"]:
        nums = get_fields_numeric(
            settings, df=None, include_boolean=False, types=[ftype]
        )
        cats = get_fields_categorical(
            settings, df=None, include_boolean=False, types=[ftype]
        )
        bools = get_fields_boolean(settings, df=None, types=[ftype])
        for field in nums:
            field_map[field] = {"type": ftype, "class": "numeric"}
        for field in cats:
            field_map[field] = {"type": ftype, "class": "categorical"}
        for field in bools:
            field_map[field] = {"type": ftype, "class": "boolean"}
    return field_map


def get_dtypes_from_settings(settings: dict) -> dict:
    """
    Generate a dictionary mapping fields to their designated data types based on settings.

    Parameters
    ----------
    settings : dict
        Settings dictionary.

    Returns
    -------
    dict
        Dictionary of field names to data type strings.
    """

    cats = get_fields_categorical(settings, include_boolean=False)
    bools = get_fields_boolean(settings)
    nums = get_fields_numeric(settings, include_boolean=False)
    dtypes = {}
    for c in cats:
        dtypes[c] = "string"
    for b in bools:
        dtypes[b] = "bool"
    for n in nums:
        dtypes[n] = "Float64"
    return dtypes


def process_data(
    dataframes: dict[str, pd.DataFrame], settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """
    Process raw dataframes according to settings and return a SalesUniversePair.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Dictionary mapping keys to DataFrames.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    SalesUniversePair
        A SalesUniversePair containing processed sales and universe data.

    Raises
    ------
    ValueError
        If required merge instructions or columns are missing.
    """

    s_data = settings.get("data", {})
    s_process = s_data.get("process", {})
    s_merge = s_process.get("merge", {})

    merge_univ: list | None = s_merge.get("universe", None)
    merge_sales: list | None = s_merge.get("sales", None)

    if merge_univ is None:
        raise ValueError(
            'No "universe" merge instructions found. data.process.merge must have exactly two keys: "universe", and "sales"'
        )
    if merge_sales is None:
        raise ValueError(
            'No "sales" merge instructions found. data.process.merge must have exactly two keys: "universe", and "sales"'
        )

    df_univ = _merge_dict_of_dfs(dataframes, merge_univ, settings, required_key="key")
    df_sales = _merge_dict_of_dfs(
        dataframes, merge_sales, settings, required_key="key_sale"
    )

    if "valid_sale" not in df_sales:
        raise ValueError("The 'valid_sale' column is required in the sales data. If you don't have anything to go on, you can just create that column and fill it with an assumption (i.e. all are valid), but ideally you should look for some kind of validation criteria for your sales.")
    if "vacant_sale" not in df_sales:
        raise ValueError("The 'vacant_sale' column is required in the sales data. If you don't have anything to go on, you can just create that column and fill it with an assumption (i.e. match vacant status in the universe), but ideally you should look for some kind of sales metadata on this.")
    # Print number and percentage of valid sales
    valid_count = df_sales["valid_sale"].sum()
    total_count = len(df_sales)
    valid_percent = (valid_count / total_count * 100) if total_count > 0 else 0
    print(f"Valid sales: {valid_count} ({valid_percent:.1f}% of {total_count} total)")
    df_sales = df_sales[df_sales["valid_sale"].eq(True)].copy().reset_index(drop=True)

    sup: SalesUniversePair = SalesUniversePair(universe=df_univ, sales=df_sales)
    
    sup = _enrich_data(
        sup, s_process.get("enrich", {}), dataframes, settings, verbose=verbose
    )

    dupe_univ: dict | None = s_process.get("dupes", {}).get("universe", None)
    dupe_sales: dict | None = s_process.get("dupes", {}).get("sales", None)
    if dupe_univ:
        sup.set(
            "universe",
            _handle_duplicated_rows(sup.universe, dupe_univ, verbose=verbose),
        )
    if dupe_sales:
        sup.set(
            "sales", _handle_duplicated_rows(sup.sales, dupe_sales, verbose=verbose)
        )

    return sup


def enrich_df_streets(
    df_in: gpd.GeoDataFrame,
    settings: dict,
    spacing: float = 1.0,  # in meters
    max_ray_length: float = 25.0,  # meters to shoot rays
    network_buffer: float = 500.0,  # buffer for street network
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """Enrich a GeoDataFrame with street network data.

    This function enriches the input GeoDataFrame with street network data by calculating
    frontage, depth, distance to street, and many other related metrics, for every road vs.
    every parcel in the GeoDataFrame, using OpenStreetMap data.

    WARNING: This function can be VERY computationally and memory intensive for large datasets
    and may take a long time to run.

    We definitely need to work on its performance or make it easier to split into smaller chunks.

    Parameters
    ----------
    df_in : gpd.GeoDataFrame
        Input GeoDataFrame containing parcels.
    settings : dict
        Settings dictionary containing configuration for the enrichment.
    spacing : float, optional
        Spacing in meters for ray casting to calculate distances to streets. Default is 1.0.
    max_ray_length : float, optional
        Maximum length of rays to shoot for distance calculations, in meters. Default is 25.0.
    network_buffer : float, optional
        Buffer around the street network to consider for distance calculations, in meters.
        Default is 500.0.
    verbose : bool, optional
        If True, prints progress information. Default is False.

    Returns
    -------
    gpd.GeoDataFrame
        Enriched GeoDataFrame with additional columns for street-related metrics.
    """
    e_streets = settings.get("data",{}).get("process", {}).get("enrich", {}).get("streets", {})
    do_streets = e_streets.get("enabled", False)
    
    if do_streets:
        df_out = _enrich_df_streets(
            df_in, settings, spacing, max_ray_length, network_buffer, verbose
        )

        # add somers unit land size normalization using frontage & depth
        df_out["land_area_somers_ft"] = get_size_in_somers_units_ft(
            df_out["frontage_ft_1"], df_out["depth_ft_1"]
        )
    else:
        df_out = df_in
        if verbose:
            print(f"Street enrichment disabled. To enable it, add `data.process.enrich.streets.enabled = true` to your settings file.")

    return df_out


def enrich_sup_spatial_lag(
    sup: SalesUniversePair, 
    settings: dict, 
    verbose: bool = False
) -> SalesUniversePair:
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
    
    mg_ids = get_model_group_ids(settings)
    
    df_sales = sup.sales
    df_universe = sup.universe
    
    # For each model group, calculate its spatial lag surface(s)
    for mg in mg_ids:
        sup_mg = _enrich_sup_spatial_lag_for_model_group(
            sup,
            settings,
            mg,
            verbose
        )
        if sup_mg is None:
            continue
        # For each spatial lag surface, copy it back to the master SalesUniversePair
        sl_cols = [field for field in sup_mg.universe.columns if field.startswith("spatial_lag_")]
        for col in sl_cols:
            # Only fill in values that haven't been set already
            if col in sup_mg.sales:
                df_sales = fill_from_df(df_sales, sup_mg.sales, "key_sale", col)
            if col in sup_mg.universe:
                df_universe = fill_from_df(df_universe, sup_mg.universe, "key", col)
    
    sup.sales = df_sales
    sup.universe = df_universe
    
    return sup


def _enrich_sup_spatial_lag_for_model_group(
    sup: SalesUniversePair, 
    settings: dict, 
    model_group: str,
    verbose: bool = False
) -> SalesUniversePair:

    s_sl = (
        settings.get("data", {})
        .get("process", {})
        .get("enrich", {})
        .get("spatial_lag", {})
    )
    
    s_mgs = s_sl.get("model_groups", {})
    if model_group not in s_mgs:
        warnings.warn(f"Could not find model entry \"{model_group}\" in process.enrich.spatial_lag.model_groups, skipping...")
        return None
    
    entry = s_mgs.get(model_group)
    sample_from_mgs = entry.get("sample_from", [])
    if len(sample_from_mgs) == 0:
        raise ValueError(f"'process.enrich.spatial_lag.model_groups.{model_group}' does not specify \"sample_from\"! Provide an explicit list of model groups you are sampling from. A safe default is just the same model group.")
    
    BANDWIDTH_MILES = 0.5  # distance at which confidence --> 0
    METRES_PER_MILE = 1609.344
    D_SCALE = BANDWIDTH_MILES * METRES_PER_MILE

    df_sales = sup.sales.copy()
    df_universe = sup.universe.copy()

    df_hydrated = get_hydrated_sales_from_sup(sup)
    train_keys, test_keys = get_train_test_keys(df_hydrated, settings)
    
    # only WRITE TO the given model group
    df_hydrated_mg = df_hydrated[df_hydrated["model_group"].eq(model_group)]
    df_sales = df_sales[df_sales["key_sale"].isin(df_hydrated_mg["key_sale"].values)]
    df_universe = df_universe[df_universe["model_group"].eq(model_group)]
    
    # only SAMPLE FROM the designated source model groups
    df_hydrated = df_hydrated[df_hydrated["model_group"].isin(sample_from_mgs)]
    
    sale_field = get_sale_field(settings)
    sale_field_vacant = f"{sale_field}_vacant"

    per_land_field = f"{sale_field}_land_sqft"
    per_impr_field = f"{sale_field}_impr_sqft"

    if per_land_field not in df_hydrated:
        df_hydrated[per_land_field] = div_series_z_safe(
            df_hydrated[sale_field], df_hydrated["land_area_sqft"]
        )
    if per_impr_field not in df_hydrated:
        df_hydrated[per_impr_field] = div_series_z_safe(
            df_hydrated[sale_field], df_hydrated["bldg_area_finished_sqft"]
        )
    if sale_field_vacant not in df_hydrated:
        df_hydrated[sale_field_vacant] = None
        df_hydrated[sale_field_vacant] = df_hydrated[sale_field].where(
            df_hydrated["bldg_area_finished_sqft"].le(0)
            & df_hydrated["land_area_sqft"].gt(0)
        )

    value_fields = [sale_field, sale_field_vacant, per_land_field, per_impr_field]

    for value_field in value_fields:

        if value_field == sale_field:
            df_sub = df_hydrated.loc[df_hydrated["valid_sale"].eq(True)].copy()
        elif (value_field == sale_field_vacant) or (value_field == per_land_field):
            df_sub = df_hydrated.loc[
                df_hydrated["valid_sale"].eq(True)
                & df_hydrated["vacant_sale"].eq(True)
                & df_hydrated["land_area_sqft"].gt(0)
            ].copy()
        elif value_field == per_impr_field:
            df_sub = df_hydrated.loc[
                df_hydrated["valid_sale"].eq(True)
                & df_hydrated["bldg_area_finished_sqft"].gt(0)
            ].copy()
        else:
            raise ValueError(f"Unknown value field: {value_field}")

        if df_sub.empty:
            df_universe[f"spatial_lag_{value_field}"] = 0
            df_sales[f"spatial_lag_{value_field}"] = 0
            continue
        
        df_sub = df_sub[~pd.isna(df_sub["latitude"]) & ~pd.isna(df_sub["longitude"])]

        # Choose the number of nearest neighbors to use
        k = 5  # adjust this number as needed

        df_sub_train = df_sub.loc[df_sub["key_sale"].isin(train_keys)].copy()
        
        if len(df_sub_train) <= 0:
            continue
        
        # Get the coordinates for the universe parcels
        crs_equal_distance = get_crs(df_universe, "equal_distance")
        df_proj = df_universe.to_crs(crs_equal_distance)

        # Use the projected coordinates for the universe parcels
        universe_coords = np.vstack(
            [df_proj.geometry.centroid.x.values, df_proj.geometry.centroid.y.values]
        ).T

        # Get the coordinates for the sales training parcels
        df_sub_train_proj = df_sub_train.to_crs(crs_equal_distance)

        sales_coords_train = np.vstack(
            [
                df_sub_train_proj.centroid.geometry.x.values,
                df_sub_train_proj.centroid.geometry.y.values,
            ]
        ).T

        # Build a cKDTree from df_sales coordinates -- but ONLY from the training set
        sales_tree = cKDTree(sales_coords_train)

        # count any NA coordinates in the universe
        n_na_coords = universe_coords.shape[0] - np.count_nonzero(
            pd.isna(universe_coords).any(axis=1)
        )

        # Query the tree: for each parcel in df_universe, find the k nearest sales
        # distances: shape (n_universe, k); indices: corresponding indices in df_sales
        distances, indices = sales_tree.query(universe_coords, k=k)

        # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        # For each universe parcel, compute sigma as the mean distance to its k neighbors.
        sigma = distances.mean(axis=1, keepdims=True)

        # Handle zeros in sigma
        sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

        # Compute Gaussian kernel weights for all neighbors
        weights = np.exp(-(distances**2) / (2 * sigma**2))

        # Normalize the weights so that they sum to 1 for each parcel
        weights_norm = weights / weights.sum(axis=1, keepdims=True)

        # Get the sales prices corresponding to the neighbor indices
        sales_prices = df_sub_train[value_field].values
        neighbor_prices = sales_prices[indices]  # shape (n_universe, k)

        # Compute the weighted average (spatial lag) for each parcel in the universe
        spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_prices)).sum(
            axis=1
        )

        # Add the spatial lag as a new column
        df_universe[f"spatial_lag_{value_field}"] = spatial_lag

        # Fill NaN values in the spatial lag with the median value of the original field
        median_value = df_sub_train[value_field].median()
        df_universe[f"spatial_lag_{value_field}"] = df_universe[
            f"spatial_lag_{value_field}"
        ].fillna(median_value)

        # Add the new field to sales:
        df_sales = df_sales.merge(
            df_universe[["key", f"spatial_lag_{value_field}"]], on="key", how="left"
        )

        # ------------------------------------------------
        # Calculate confidence:

        # Raw inverse-square information mass
        distances_safe = distances.copy()
        distances_safe[distances_safe == 0] = np.finfo(float).eps  # protect ÷ 0

        inv_sq = 1.0 / distances_safe**2  # shape (n_parcel, 5)
        info_mass = inv_sq.sum(axis=1)  # Σ 1/d²

        # Fixed-bandwidth confidence
        conf = 1.0 - (k / D_SCALE**2) / info_mass
        spatial_lag_confidence = np.clip(conf, 0.0, 1.0)  # keep in [0, 1]

        # store
        df_universe[f"spatial_lag_{value_field}_confidence"] = spatial_lag_confidence
        df_sales = df_sales.merge(
            df_universe[["key", f"spatial_lag_{value_field}_confidence"]],
            on="key",
            how="left",
        )
        # ------------------------------------------------

    df_test = df_sales.loc[df_sales["key_sale"].isin(test_keys)].copy()
    
    # we pass in the original sup.universe so we can re-select model groups properly within this function
    df_universe_enriched = _enrich_universe_spatial_lag(
        sup.universe,
        df_test,
        model_group=model_group,
        sample_from_mgs=sample_from_mgs,
        settings=settings
    )
    
    # we merge back all new spatial lag fields back into the universe
    sl_fields = [field for field in df_universe_enriched.columns if field.startswith("spatial_lag_")]
    for field in sl_fields:
        df_universe = fill_from_df(df_universe, df_universe_enriched, "key", field)
    
    # we return a new sup containing our modified sales & universe
    sup = SalesUniversePair(df_sales, df_universe)
    return sup


def get_train_test_keys(df_in: pd.DataFrame, settings: dict):
    """Get the training and testing keys for the sales DataFrame.

    This function gets the train/test keys for each model group defined in the settings,
    combines them into a single mask for the sales DataFrame, and returns the keys for
    training and testing as numpy arrays.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame containing sales data.
    settings : dict
        Settings dictionary

    Returns
    -------
    tuple
        A tuple containing two numpy arrays: keys_train and keys_test.
        - keys_train: keys for training set
        - keys_test: keys for testing set
    """

    model_group_ids = get_model_group_ids(settings, df_in)

    # an empty mask the same size as the input DataFrame
    mask_train = pd.Series(np.zeros(len(df_in), dtype=bool), index=df_in.index)
    mask_test = pd.Series(np.zeros(len(df_in), dtype=bool), index=df_in.index)

    for model_group in model_group_ids:
        # Read the split keys for the model group
        test_keys, train_keys = _read_split_keys(model_group)

        # Filter the DataFrame based on the keys
        mask_test |= df_in["key_sale"].isin(test_keys)
        mask_train |= df_in["key_sale"].isin(train_keys)

    keys_test = df_in.loc[mask_test, "key_sale"].values
    keys_train = df_in.loc[mask_train, "key_sale"].values

    return keys_train, keys_test


def get_train_test_masks(df_in: pd.DataFrame, settings: dict):
    """Get the training and testing masks for the sales DataFrame.

    This function gets the train/test masks for each model group defined in the settings,
    combines them into a single mask for the sales DataFrame, and returns the masks as pandas Series

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame containing sales data.
    settings : dict
        Settings dictionary

    Returns
    -------
    tuple
        A tuple containing two pandas Series: mask_train and mask_test.
        - mask_train: boolean mask for training set
        - mask_test: boolean mask for testing set
    """
    model_group_ids = get_model_group_ids(settings, df_in)

    # an empty mask the same size as the input DataFrame
    mask_train = pd.Series(np.zeros(len(df_in), dtype=bool), index=df_in.index)
    mask_test = pd.Series(np.zeros(len(df_in), dtype=bool), index=df_in.index)

    for model_group in model_group_ids:
        # Read the split keys for the model group
        test_keys, train_keys = _read_split_keys(model_group)

        # Filter the DataFrame based on the keys
        mask_test |= df_in["key_sale"].isin(test_keys)
        mask_train |= df_in["key_sale"].isin(train_keys)

    return mask_train, mask_test


#######################################
# PRIVATE
#######################################


def _enrich_data(
    sup: SalesUniversePair,
    s_enrich: dict,
    dataframes: dict[str, pd.DataFrame],
    settings: dict,
    verbose: bool = False,
) -> SalesUniversePair:
    """
    Enrich both sales and universe data based on enrichment instructions.

    Applies enrichment operations (e.g., spatial and basic enrichment) to both the
    "sales" and "universe" DataFrames.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    s_enrich : dict
        Enrichment instructions.
    dataframes : dict[str, pd.DataFrame]
        Dictionary of additional DataFrames.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    SalesUniversePair
        Enriched SalesUniversePair.
    """

    if verbose:
        print(f"Enriching data...")

    df_sales = sup.sales
    df_univ = sup.universe

    if s_enrich is not None:

        # do spatial joins on user data
        df_univ = _enrich_df_spatial_joins(
            df_univ, s_enrich, dataframes, settings, verbose=verbose
        )

        # add building footprints
        df_univ = _enrich_df_overture(
            df_univ, s_enrich, dataframes, settings, verbose=verbose
        )

        # add lat/lon/rectangularity etc.
        df_univ = _basic_geo_enrichment(df_univ, s_enrich, settings, verbose=verbose)

        # handle Census enrichment for universe if enabled
        if "census" in s_enrich:
            df_univ = _enrich_df_census(
                df_univ, s_enrich.get("census", {}), verbose=verbose
            )

        # handle distance enrichment for universe if enabled
        if "distances" in s_enrich:
            df_univ = _enrich_df_distances(
                df_univ,
                s_enrich.get("distances", {}),
                dataframes,
                verbose=verbose,
                use_cache=True,
            )

        if "permits" in s_enrich:
            df_sales = _enrich_permits(
                df_sales, s_enrich, dataframes, settings, is_sales=True, verbose=verbose
            )

        # fill in missing data based on geospatial patterns (should happen after all other enrichments have been done)
        if "infer" in s_enrich:
            df_univ = _enrich_spatial_inference(
                df_univ, s_enrich, dataframes, settings, verbose=verbose
            )

    # User calcs apply at the VERY end of enrichment, after all automatic enrichments have been applied
    if s_enrich is not None:
        df_univ = _enrich_df_basic(
            df_univ,
            s_enrich,
            dataframes,
            settings,
            is_sales=False,
            verbose=verbose,
        )
        
        df_sales = _enrich_df_basic(
            df_sales,
            s_enrich,
            dataframes,
            settings,
            is_sales=True,
            verbose=verbose
        )

    # Enforce vacant status
    df_univ = _enrich_vacant(df_univ, settings)
    df_sales = _enrich_vacant(df_sales, settings)

    sup.set("universe", df_univ)
    sup.set("sales", df_sales)

    return sup


def _enrich_df_census(
    df_in: pd.DataFrame | gpd.GeoDataFrame, census_settings: dict, verbose: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Enrich a DataFrame with Census data by performing a spatial join with Census block
    groups.
    """
    if not census_settings.get("enabled", False):
        if verbose:
            print("Census enrichment disabled, skipping...")
        return df_in

    if verbose:
        print("Enriching with Census data...")

    df_out = get_cached_df(df_in, "census", "key", census_settings)
    if df_out is not None:
        if verbose:
            print("--> found cached data")
        return df_out

    df = df_in.copy()

    # try:
    # Get Census credentials and initialize service
    creds = get_creds_from_env_census()
    if creds is None:
        warnings.warn("Failed to get census credentials, skipping census enrichment")
        return df_in
    census_service = init_service_census(creds)

    # Get FIPS code from settings
    fips_code = census_settings.get("fips", "")
    if not fips_code:
        warnings.warn(
            "Census enrichment enabled but no FIPS code provided in settings"
        )
        return df

    year = census_settings.get("year", 2022)
    if verbose:
        print("Getting Census Data...")
    
    # Get Census data with boundaries
    census_data, census_boundaries = census_service.get_census_data_with_boundaries(
        fips_code=fips_code, year=year, census_settings=census_settings
    )

    # Spatial join with universe data only
    if not isinstance(df, gpd.GeoDataFrame):
        warnings.warn("DataFrame is not a GeoDataFrame, skipping Census enrichment")
        return df
    
    # Get census columns to keep
    field_map = census_service.get_census_map(census_settings)
    census_cols_to_keep = ["std_geoid"] + [field_map[key] for key in field_map]
    
    # Ensure all census columns exist in the census_boundaries
    missing_cols = [
        col for col in census_cols_to_keep if col not in census_boundaries.columns
    ]
    if missing_cols:
        # Filter to only include columns that exist
        census_cols_to_keep = [
            col for col in census_cols_to_keep if col in census_boundaries.columns
        ]
    
    # Create a copy of census_boundaries with only the columns we need
    census_boundaries_subset = census_boundaries[
        ["geometry"] + census_cols_to_keep
    ].copy()
    
    # Replace all -666666666.0 sentinel values with None
    for col in census_cols_to_keep:
        if pd.api.types.is_numeric_dtype(census_boundaries_subset[col]):
            census_boundaries_subset.loc[
                abs(census_boundaries_subset[col] + 666666666.0).le(1e6), col
            ] = None
    
    if verbose:
        print("Performing spatial join with Census Data...")

    # Perform the spatial join
    df = match_to_census_blockgroups(
        gdf=df, census_gdf=census_boundaries_subset, join_type="left"
    )
    df = df.drop(columns="std_geoid")

    write_cached_df(df_in, df, "census", "key", census_settings)
    
    return df

    # except Exception as e:
        # warnings.warn(f"Failed to enrich with Census data: {str(e)}")
    #return df


def _enrich_df_distances(
    df_in: pd.DataFrame | gpd.GeoDataFrame,
    dist_settings: dict,
    dataframes: dict[str, pd.DataFrame],
    verbose: bool = False,
    use_cache: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Enrich a DataFrame with OpenStreetMap data by calculating distances to all
    features.
    """

    if verbose:
        print("Enriching with OpenStreetMap data...")

    if use_cache:
        df_out = get_cached_df(df_in, "osm/all", "key", dist_settings)
        if df_out is not None:
            if verbose:
                print("--> found cached data")
            return df_out

    df = df_in.copy()

    try:
        if not dist_settings.get("enabled", False):
            if verbose:
                print("OpenStreetMap enrichment disabled, skipping all OSM features")
            return df

        # Initialize OpenStreetMap service
        osm_service = init_service_openstreetmap(dist_settings)

        # Convert DataFrame to GeoDataFrame if it isn't already
        if not isinstance(df, gpd.GeoDataFrame):
            warnings.warn(
                "DataFrame is not a GeoDataFrame, skipping OpenStreetMap enrichment"
            )
            return df

        # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) before getting bounds
        original_crs = df.crs

        if original_crs is None:
            warnings.warn("GeoDataFrame has no CRS set, attempting to infer EPSG:4326")
            if is_likely_epsg4326(df):
                df.set_crs(epsg=4326, inplace=True)
            else:
                raise ValueError("Cannot determine CRS of input GeoDataFrame")
        elif not original_crs.equals(CRS.from_epsg(4326)):
            df = df.to_crs(epsg=4326)

        if "latitude" not in df or "longitude" not in df:
            raise ValueError(
                "DataFrame must contain 'latitude' and 'longitude' columns for OpenStreetMap enrichment"
            )

        north = df["latitude"].max()
        south = df["latitude"].min()
        east = df["longitude"].max()
        west = df["longitude"].min()

        bbox = [west, south, east, north]

        def get_feature(thing: str, _bbox: Tuple[float, float, float, float], s: dict, _use_cache: bool = True):
            use_osm = s.get("osm", False)
            source = s.get("source", None)
            if use_osm:
                return osm_service.get_features(thing, _bbox, s, _use_cache)
            elif source:
                df = None
                if source in dataframes:
                    df = dataframes[source]
                    if df is not None:
                        if "geometry" not in df.columns:
                            raise ValueError(f"Distance entry ({thing}) source dataframe with id \"{source}\" has no geometry!")
                if df is None:
                    raise ValueError(f"Could not find dataframe with id \"{source}\" matching distance entry ({thing}.source={source})")
                return osm_service.get_features(thing, _bbox, s, _use_cache, df)
            else:
                raise ValueError(f"Distance entry ({thing}) must have either \"osm\" or \"source\" field!")

        default_configs = {
            "water_bodies": {
                "verbose_label": "water bodies",
                "store_top": True,
                "error_method": "print",  # print error message with traceback
                "sort_field": "area",  # field to sort by for top features
                "type_field": "water",  # field containing feature type for unnamed features
            },
            "transportation": {
                "verbose_label": "transportation networks",
                "store_top": False,  # no top features for transportation
                "error_method": "warn",  # use warnings.warn
                "sort_field": "length",
                "type_field": "highway",
            },
            "educational": {
                "verbose_label": "educational institutions",
                "store_top": True,
                "error_method": "warn",
                "sort_field": "area",
                "type_field": "amenity",
            },
            "parks": {
                "verbose_label": "parks",
                "store_top": True,
                "error_method": "warn",
                "sort_field": "area",
                "type_field": "leisure",
            },
            "golf_courses": {
                "verbose_label": "golf courses",
                "store_top": True,
                "error_method": "warn",
                "sort_field": "area",
                "type_field": "leisure",
            },
        }

        features_config = {}

        for key in dist_settings:
            if key == "enabled":
                continue
            features_config[key] = dist_settings[key]

        # Loop through each feature configuration:
        for feature, config in features_config.items():
            # Check if feature is enabled in the osm_settings
            if config.get("enabled", True):

                default_config = default_configs.get(feature, {})
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]

                if verbose:
                    print(f"--> Getting {feature}...")
                try:
                    # Call the designated getter function
                    is_osm = config.get("osm", False)
                    result = get_feature(
                        thing=feature,
                        _bbox=bbox,
                        s=config,
                        _use_cache=use_cache
                    )
                    if verbose:
                        if result.empty:
                            print(f"    No {config.get('verbose_label', feature)} found")
                        else:
                            print(f"--> Found {len(result)} {config.get('verbose_label',feature)}")
                            pd.set_option("display.max_columns", None)
                            pd.set_option("display.max_rows", None)
                            pd.set_option("display.width", 1000)

                    if not result.empty:
                        # Get distance settings from distances configuration
                        feature_id = f"osm_{feature}" if is_osm else feature
                        max_distance = config.get("max_distance", None)
                        unit = config.get("unit", "km")

                        if verbose:
                            print(f"\nDistance settings for {feature_id}:")
                            print(f"max_distance: {max_distance}")
                            print(f"unit: {unit}")
                            print()

                        # Calculate distances to all features
                        df = _do_perform_distance_calculations_osm(
                            df, result, feature_id, max_distance=max_distance, unit=unit
                        )

                        # If store_top is enabled, calculate distances to top features
                        if config.get("store_top", False) and config.get("top_n", 0) > 0:
                            # Get top features based on configured sort field
                            sort_field = config.get("sort_field")
                            if sort_field in result.columns:
                                top_features = result.nlargest(
                                    config["top_n"], sort_field
                                )
                            else:
                                # Fallback to first numeric column or just take first N
                                numeric_cols = result.select_dtypes(
                                    include=[np.number]
                                ).columns
                                if len(numeric_cols) > 0:
                                    top_features = result.nlargest(
                                        config.get("top_n", 1), numeric_cols[0]
                                    )
                                else:
                                    top_features = result.head(
                                        config.get("top_n", 1)
                                    )

                            # Calculate distances to each top feature
                            for idx, top_feature in top_features.iterrows():
                                # Try to get name, fallback to type + index if no name
                                feature_name = None
                                if "name" in top_feature and pd.notna(
                                    top_feature["name"]
                                ):
                                    feature_name = str(top_feature["name"])
                                else:
                                    # Use type field if available
                                    type_field = config.get("type_field")
                                    if type_field in top_feature and pd.notna(
                                        top_feature[type_field]
                                    ):
                                        feature_type = str(top_feature[type_field])
                                        feature_name = f"{feature_type}_{idx}"
                                    else:
                                        feature_name = f"feature_{idx}"

                                # Clean the feature name
                                feature_name = _clean_series(pd.Series([feature_name]))[
                                    0
                                ]

                                # Create single-feature GeoDataFrame
                                feature_gdf = gpd.GeoDataFrame(
                                    geometry=[top_feature.geometry], crs=result.crs
                                )

                                # Calculate distance to this top feature using same distance settings
                                df = _do_perform_distance_calculations_osm(
                                    df,
                                    feature_gdf,
                                    f"{feature_id}_{feature_name}",
                                    max_distance=max_distance,
                                    unit=unit,
                                )

                except Exception as e:
                    err_msg = f"Failed to get {config.get('verbose_label', feature)}: {str(e)}"
                    if config.get("error_method", None) == "warn":
                        warnings.warn(err_msg)
                    else:
                        print("ERROR " + err_msg)
                        print("Traceback: " + traceback.format_exc())

        write_cached_df(df_in, df, "osm/all", "key", dist_settings)

        return df

    except Exception as e:
        warnings.warn(f"Failed to enrich with OpenStreetMap data: {str(e)}")
        return df


def _enrich_df_streets(
    df_in: gpd.GeoDataFrame,
    settings: dict,
    spacing: float = 1.0,  # in meters
    max_ray_length: float = 25.0,  # meters to shoot rays
    network_buffer: float = 500.0,  # buffer for street network
    verbose: bool = False,
) -> gpd.GeoDataFrame:

    bounds = df_in.total_bounds

    signature = {
        "rows": len(df_in),
        "bounds": {
            "minx": bounds[0],
            "miny": bounds[1],
            "maxx": bounds[2],
            "maxy": bounds[3],
        },
    }

    if os.path.exists("in/osm/streets.parquet"):
        df_streets = pd.read_parquet("in/osm/streets.parquet")
        if "key" in df_streets:
            df_out = df_in.copy()
            df_out = df_out.merge(df_streets, on="key", how="left")
            if verbose:
                print(
                    f"--> found streets in in/osm/streets.parquet, loading from disk!"
                )
            return df_out
    # ---- setup parcels ----

    t = TimingData()

    t.start("all")

    t.start("setup")

    df = df_in[["key", "geometry", "latitude", "longitude"]].copy()

    # drop invalid
    df = df[df.geometry.notna() & df.geometry.area.gt(0)]
    # project to equal-distance CRS
    crs_eq = get_crs(df, "conformal")
    df = df.to_crs(crs_eq)

    t.stop("setup")
    log_mem("setup")

    if verbose:
        print(f"T setup = {t.get('setup'):.0f}s")

    t.start("prepare")

    minx = df["longitude"].min()
    miny = df["latitude"].min()
    maxx = df["longitude"].max()
    maxy = df["latitude"].max()

    lat_buf = network_buffer / 111000
    lon_buf = network_buffer / (111000 * math.cos(math.radians((miny + maxy) / 2)))

    # DEBUG
    # lat_buf = 0
    # lon_buf = 0
    # pad_size = 0.25
    # size_x = maxx - minx
    # size_y = maxy - miny
    # minx += size_x * (pad_size)
    # miny += size_y * (pad_size)
    # maxx -= size_x * (pad_size)
    # maxy -= size_y * (pad_size)
    # DEBUG

    north, south = maxy + lat_buf, miny - lat_buf
    east, west = maxx + lon_buf, minx - lon_buf

    df = (
        df.loc[
            df["latitude"].ge(south)
            & df["latitude"].le(north)
            & df["longitude"].ge(west)
            & df["longitude"].le(east)
        ]
        .drop(columns=["latitude", "longitude"])
        .copy()
    )

    wanted = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "unclassified",
    ]
    highway_regex = "|".join(wanted)
    custom_filter = f'["highway"~"{highway_regex}"]'
    t.stop("prepare")
    log_mem("prepare")
    if verbose:
        print(f"T prepare = {t.get('prepare'):.0f}s")

    if verbose:
        print(f"Loading network within ({south},{west}) -> ({north},{east})")

    ox.settings.use_cache = True
    t.start("load street")
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north), network_type="all", custom_filter=custom_filter
    )
    t.stop("load street")
    log_mem("load street")
    if verbose:
        print(f"T load street = {t.get('load street'):.0f}s")

    t.start("edges")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)[
        ["geometry", "name", "highway", "osmid"]
    ]
    G = None

    edges = (
        edges.explode(index_parts=False)
        .dropna(subset=["geometry"])
        .to_crs(crs_eq)
        .reset_index(drop=True)
    )

    # unwrap lists to single values to avoid ArrowTypeError
    edges["road_name"] = edges["name"].apply(
        lambda v: v[0] if isinstance(v, (list, tuple)) else v
    )
    edges["road_type"] = edges["highway"].apply(
        lambda v: v[0] if isinstance(v, (list, tuple)) else v
    )
    edges["road_idx"] = edges.index
    t.stop("edges")
    log_mem("edges")
    if verbose:
        print(f"T edges = {t.get('edges'):.0f}s")

    # fill missing road names with the OSM id field:
    edges["road_name"] = edges["road_name"].fillna(edges["osmid"])

    # flatten lists
    edges["road_name"] = edges["road_name"].apply(
        lambda v: v if isinstance(v, str) else str(v)
    )

    # ---- helper for single-edge rays ----
    def _rays_from_edge(geom, rid, rname, rtype, spacing=spacing, max_ray_length=25.0):

        # 1) inject new vertices every `spacing` metres
        dens = shapely.segmentize(geom, spacing)

        # 2) pull out coords
        coords = list(dens.coords)
        if len(coords) < 3:
            return []

        _out = []
        # skip first & last point, so i in [1 .. len(coords)-2]
        for i in range(1, len(coords) - 1):
            (_ox, _oy) = coords[i]
            (x0, y0), (x1, y1) = coords[i - 1], coords[i + 1]
            # estimate tangent from prev->next
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy)
            nx, ny = -dy / norm, dx / norm  # unit-normal

            for sign in (+1, -1):
                ex = _ox + sign * nx * max_ray_length
                ey = _oy + sign * ny * max_ray_length
                _out.append(
                    {
                        "road_idx": rid,
                        "road_name": rname,
                        "road_type": rtype,
                        "geometry": LineString([(_ox, _oy), (ex, ey)]),
                        "angle": math.atan2(ey - _oy, ex - _ox),
                    }
                )
        return _out

    # ---- parallel ray generation ----
    args = list(zip(edges.geometry, edges.road_idx, edges.road_name, edges.road_type))
    edges = None

    t.start("rays_parallel")
    n_jobs = 8
    if verbose:
        print(f"Generating rays for {len(args)} edges with {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0)(
        delayed(_rays_from_edge)(*a) for a in args
    )

    # flatten & continue exactly as before
    rays = [r for sub in results for r in sub]
    args = None

    rays_gdf = gpd.GeoDataFrame(rays, geometry="geometry", crs=crs_eq)
    rays = None

    rays_gdf = rays_gdf.drop(columns=["origin"], errors="ignore")
    rays_gdf["road_name"] = rays_gdf["road_name"].astype(str)
    rays_gdf["road_type"] = rays_gdf["road_type"].astype(str)

    # Create out/temp directory if it doesn't exist
    os.makedirs("out/temp", exist_ok=True)

    # DEBUG
    # rays_gdf.to_parquet(f"out/temp/rays.parquet", index=False)

    t.stop("rays_parallel")
    log_mem("rays_parallel")

    gc.collect()

    if verbose:
        print(f"--> T rays_parallel = {t.get('rays_parallel'):.0f}s")

    # ---- block by first parcel ----
    t.start("block")
    # spatial join rays -> parcels
    gdf = df[["key", "geometry"]].rename(columns={"geometry": "parcel_geom"})
    gdf = gpd.GeoDataFrame(gdf, geometry="parcel_geom", crs=crs_eq)

    # DEBUG
    # gdf.to_file(f"out/temp/gdf.gpkg", driver="GPKG")

    # Build spatial index on the rays
    tree = STRtree(list(rays_gdf.geometry.values))

    # now use `query` on the array of geometries
    # returns a 2×N array: [ray_idx_array, parcel_idx_array]
    pairs = tree.query(
        gdf.geometry.values, predicate="intersects"  # array of parcel_geom
    )
    tree = None
    gc.collect()

    parcel_idxs = pairs[0]  # indices into gdf
    ray_idxs = pairs[1]  # indices into rays_gdf
    pairs = None
    gc.collect()

    # 3) Select only those matching rows, preserving order
    parcels_sel = gdf.iloc[parcel_idxs].reset_index(drop=True)
    rays_sel = rays_gdf.iloc[ray_idxs].reset_index(drop=False)
    rays_sel = rays_sel.rename(columns={"index": "ray_id"})
    rays_gdf = None
    gc.collect()

    # 4) Combine into ray_par DataFrame
    ray_par = rays_sel.copy()
    ray_par["key"] = parcels_sel["key"]
    ray_par["parcel_geom"] = parcels_sel["parcel_geom"]
    parcel_sel = None
    rays_sel = None
    gc.collect()

    # drop self if occurs
    ray_par = ray_par[ray_par.road_idx.notna()]
    gc.collect()
    t.stop("block")

    log_mem("block")
    if verbose:
        print(f"T block = {t.get('block'):.0f}s")

    if ray_par.empty:
        print(f"Ray par is empty, return early")
        return df_in

    t.start("dist")

    t.start("dist_0")
    # grab the raw Shapely geometries as simple arrays
    rays = ray_par.geometry.values
    parcels = ray_par.parcel_geom.values
    n = len(ray_par)

    t.stop("dist_0")
    log_mem("dist_0")
    if verbose:
        print(f"T dist_0 = {t.get('dist_0'):.0f}s")

    t.start("origins setup")
    # flat list of all coordinates (shape (total_pts, 2))
    coords = shapely.get_coordinates(rays)
    # get # of points in each LineString (shape (n_rays,))
    counts = shapely.get_num_coordinates(rays)
    # compute index of *first* point in each geometry
    offsets = np.empty_like(counts)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)[:-1]
    # index directly into coords to get the origins array (shape (n_rays, 2))
    origins_all = coords[offsets]
    coords = None
    offsets = None
    gc.collect()
    t.stop("origins setup")

    log_mem("origins setup")
    if verbose:
        print(f"T origins setup = {t.get('origins setup'):.0f}s")

    t.start("intersect")

    chunk_size = 100_000
    segs_list = []
    i = 0

    for start in range(0, len(rays), chunk_size):
        end = start + chunk_size
        if verbose:
            perc = start / len(rays)
            print(f"--> {perc:5.2%}: chunk from {start} to {end}")

        segs_chunk = shapely.intersection(rays[start:end], parcels[start:end])

        segs_list.append(segs_chunk)
        segs_chunk = None

    rays = None
    parcels = None

    i += 1
    segs = np.concatenate(segs_list)
    segs_list = None
    gc.collect()
    t.stop("intersect")
    log_mem("intersect")
    if verbose:
        print(f"T intersect = {t.get('intersect'):.0f}s")

    t.start("coords_counts")
    coords = shapely.get_coordinates(segs)
    counts = shapely.get_num_coordinates(segs)
    offsets = np.empty_like(counts)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)[:-1]
    t.stop("coords_counts")
    log_mem("coords_counts")
    if verbose:
        print(f"T coords_counts = {t.get('coords_counts'):.0f}s")

    t.start("entries")
    entries = coords[offsets]
    coors = None
    counts = None
    offsets = None
    t.stop("entries")
    log_mem("entries")
    if verbose:
        print(f"T entries = {t.get('entries'):.0f}s")

    t.start("distances")
    diffs = entries - origins_all
    distances = np.hypot(diffs[:, 0], diffs[:, 1])
    t.stop("distances")
    log_mem("distances")

    # stick it back on your GeoDataFrame
    ray_par["distance"] = distances
    diffs = None
    origins_all = None
    distances = None

    # keep only the closest-hit per ray
    first_hits = ray_par.loc[ray_par.groupby("ray_id")["distance"].idxmin()].copy()

    # now 'first_hits' has at most one row per ray (the nearest parcel)
    first_hits = first_hits.drop(columns=["ray_id"])

    ray_par = first_hits
    first_hits = None
    gc.collect()

    t.stop("dist")
    log_mem("dist")

    if verbose:
        print(f"T dist = {t.get('dist'):.0f}s")

    # Fill road name with road_idx if none
    ray_par["road_name"] = ray_par["road_name"].fillna(
        f"Unknown Road, ID: " + ray_par["road_idx"].astype(str)
    )

    # ---- aggregate frontages ----
    t.start("agg")
    agg = (
        ray_par.groupby(["key", "road_name", "road_type"])
        .agg(
            count_rays=("distance", "count"),
            min_distance=("distance", "min"),
            mean_angle=("angle", "mean"),
        )
        .reset_index()
    )
    ray_par = None

    agg["frontage"] = agg["count_rays"] * spacing

    # approximate depth via area/frontage
    areas = df[["key"]].copy()
    areas["area"] = df.geometry.area
    agg = agg.merge(areas, on="key", how="left")
    areas = None

    agg["depth"] = agg["area"] / agg["frontage"]
    t.stop("agg")
    log_mem("agg")
    if verbose:
        print(f"T agg = {t.get('agg'):.0f}s")

    # ---- rank, dedupe, slot & pivot up to 4 frontages ----
    t.start("pivot")

    # 1) assign type_rank once
    priority = {
        "motorway": 0,
        "trunk": 1,
        "primary": 2,
        "secondary": 3,
        "tertiary": 4,
        "residential": 5,
        "service": 6,
        "unclassified": 7,
    }
    agg["type_rank"] = agg["road_type"].map(priority).fillna(99).astype(int)

    # 2) sort then drop duplicates by (key,road_name), keeping best
    # NOTE: since we were aggregating on key/road_idx/road_name/road_type, but here only on key/road_name, we have to be careful
    # because it's possible that OTHER SEGMENTS of the same road that "front" on our parcel are still hanging around
    # we make sure to de-duplicate correctly here by sorting on the highest frontage for cases of the identical road names/types
    agg = agg.sort_values(
        ["key", "road_name", "type_rank", "frontage", "min_distance"],
        ascending=[True, True, True, False, True],
    ).drop_duplicates(subset=["key", "road_name"], keep="first")

    # per key, aggregate the min distance and the max frontage:
    agg2 = (
        agg.groupby("key")
        .agg(
            hits=("min_distance", "count"),
            max_distance=("min_distance", "max"),
            med_frontage=("frontage", "median"),
        )
        .reset_index()
    )

    agg = agg.merge(agg2, on="key", how="left")
    agg2 = None

    ######## Remove spurious hits: #######
    # Heuristic:
    # - For any parcel with more than two street hits
    # - If this hit's distance is the maximum distance of all hits, and is more than 10 meters away
    # - If this hit's frontage is less than half the median frontage of all hits
    agg["spurious"] = False

    agg.loc[
        agg["hits"].gt(2)
        & agg["max_distance"].gt(10)
        & abs(agg["min_distance"] - agg["max_distance"]).lt(1e-6)
        & agg["frontage"].lt(agg["med_frontage"] / 2),
        "spurious",
    ] = True

    # drop spurious hits:
    agg = agg[agg["spurious"].eq(False)]

    agg = agg.drop(columns=["hits", "max_distance", "med_frontage"], errors="ignore")

    ######

    distance_score = 1.0 - (agg["min_distance"] / max_ray_length)
    agg["sort_score"] = agg["frontage"] * distance_score

    # 3) now sort by overall priority & distance, assign slots, cap at 4
    agg = agg.sort_values(
        ["key", "type_rank", "sort_score", "frontage", "min_distance"],
        ascending=[True, True, False, False, True],
    )
    agg["slot"] = agg.groupby("key").cumcount() + 1
    agg = agg[agg["slot"] <= 4]

    agg = agg.rename(
        columns={"mean_angle": "road_angle", "min_distance": "dist_to_road"}
    )

    directions = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]

    agg["road_face"] = agg["road_angle"].apply(
        lambda x: directions[int((x + math.pi) / (2 * math.pi) * 8) % 8]
    )

    # 4) pivot into the _1 … _4 columns
    final = agg.pivot(
        index="key",
        columns="slot",
        values=[
            "road_name",
            "frontage",
            "road_type",
            "road_angle",
            "road_face",
            "depth",
            "dist_to_road",
        ],
    )
    agg = None

    # 5) flatten the MultiIndex and drop any all‑null columns
    final.columns = [f"{field}_{i}" for field, i in final.columns]
    final = final.reset_index().dropna(axis=1, how="all")

    t.stop("pivot")
    log_mem("pivot")
    if verbose:
        print(f"T pivot = {t.get('pivot'):.0f}s")

    # ---- merge back and add directions ----
    t.start("merge")
    out = df_in.merge(final, on="key", how="left")
    # compute compass dir for each angle if needed...

    t.stop("merge")
    log_mem("merge")
    if verbose:
        print(f"T merge = {t.get('merge'):.0f}s")

    t.stop("all")
    log_mem("all")

    if verbose:
        print("***ALL TIMING***")
        print(t.print())
        print("****************")

    df_out = gpd.GeoDataFrame(out, geometry="geometry", crs=df_in.crs)
    df_out = _finish_df_streets(df_out, settings)

    # Eliminate a bunch of things I'm not using anymore:
    agg = None
    final = None
    out = None

    net_columns = [col for col in df_out if col not in df_in.columns]
    df_net_streets = df_out[["key"] + net_columns]

    os.makedirs("in/osm", exist_ok=True)

    df_net_streets.to_parquet("in/osm/streets.parquet")

    return df_out


def _finish_df_streets(df: gpd.GeoDataFrame, settings: dict) -> gpd.GeoDataFrame:
    units = get_short_distance_unit(settings)

    if units == "ft":
        conversion_mult = 3.28084
        suffix = "_ft"
    else:
        conversion_mult = 1.0
        suffix = "_m"

    stubs = ["frontage", "depth", "dist_to_road"]
    for stub in stubs:
        for i in range(1, 5):
            col = f"{stub}_{i}"
            if col in df:
                df[col] = df[col].fillna(0.0) * conversion_mult
                df.rename(columns={col: f"{stub}{suffix}_{i}"}, inplace=True)
                print(f"renaming FROM: ({col}) TO: ({stub}{suffix}_{i})")

    df[f"osm_total_frontage{suffix}"] = (
        df[f"frontage{suffix}_1"].fillna(0.0)
        + df[f"frontage{suffix}_2"].fillna(0.0)
        + df[f"frontage{suffix}_3"].fillna(0.0)
        + df[f"frontage{suffix}_4"].fillna(0.0)
    )

    for road_type in [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "unclassified",
    ]:
        df[f"osm_frontage_{road_type}{suffix}"] = 0.0
        for i in range(1, 5):
            df[f"osm_frontage_{road_type}{suffix}"] += df[
                f"frontage{suffix}_{i}"
            ].where(df[f"road_type_{i}"] == road_type, 0.0)

    stubs_to_prefix = [
        "frontage",
        "road_name",
        "road_type",
        "road_face",
        "depth",
        "dist_to_road",
        "road_angle",
    ]

    renames = {}
    for stub in stubs_to_prefix:
        for i in range(1, 5):
            renames[f"{stub}_{i}"] = f"osm_{stub}_{i}"

    df = df.rename(columns=renames)

    return df


def _identify_parcels_with_holes(
    df: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Identify parcels with holes (interior rings) in their geometries.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        GeoDataFrame with parcel geometries.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        GeoDataFrame with parcels containing interior rings.
    """

    # Identify parcels with holes
    def has_holes(geom):
        if geom.is_valid:
            if geom.geom_type == "Polygon":
                return len(geom.interiors) > 0
            elif geom.geom_type == "MultiPolygon":
                return any(len(p.interiors) > 0 for p in geom.geoms)
        return False

    parcels_with_holes = df[df.geometry.apply(has_holes)]
    # Remove duplicates:
    parcels_with_holes = parcels_with_holes.drop_duplicates(subset="key")
    return parcels_with_holes


def _enrich_sale_age_days(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Enrich the DataFrame with a 'sale_age_days' column indicating the age in days since
    sale.
    """
    val_date = get_valuation_date(settings)
    # create a new field with dtype Int64
    df["sale_age_days"] = None
    df["sale_age_days"] = df["sale_age_days"].astype("Int64")
    sale_date_as_datetime = pd.to_datetime(
        df["sale_date"], format="%Y-%m-%d", errors="coerce"
    )
    
    if "assr_date" in df:
        df["assr_date_age_days"] = None
        df["assr_date_age_days"] = df["assr_date_age_days"].astype("Int64")
        sale_date_as_datetime = pd.to_datetime(
            df["sale_date"], format="%Y-%m-%d", errors="coerce"
        )
    
    df.loc[~sale_date_as_datetime.isna(), "sale_age_days"] = (
        val_date - sale_date_as_datetime
    ).dt.days
    return df


def _enrich_year_built(df: pd.DataFrame, settings: dict, is_sales: bool = False):
    """Enrich the DataFrame with building age information based on year built."""
    val_date = get_valuation_date(settings)
    for prefix in ["bldg", "bldg_effective"]:
        col = f"{prefix}_year_built"
        if col in df:
            new_col = f"{prefix}_age_years"
            df = _do_enrich_year_built(df, col, new_col, val_date, is_sales)
    return df


def _do_enrich_year_built(
    df: pd.DataFrame, col: str, new_col: str, val_date: datetime, is_sales: bool = False
) -> pd.DataFrame:
    """Calculate building age and add it as a new column."""
    if not is_sales:
        val_year = val_date.year
        df[new_col] = val_year - df[col]

        # Avoid 2000+ year old buildings whose year built is 0
        df.loc[df[col].isna() | df[col].le(0), new_col] = 0
    else:
        df.loc[df["sale_year"].notna(), new_col] = df["sale_year"] - df[col]

        df.loc[df["sale_year"].isna() | df["sale_year"].le(0), new_col] = 0
    return df


def _enrich_time_field(
    df: pd.DataFrame,
    prefix: str,
    add_year_month: bool = True,
    add_year_quarter: bool = True,
) -> pd.DataFrame:
    """Enrich a DataFrame with time-related fields based on a prefix."""
    if f"{prefix}_date" not in df:
        # Check if we have _year, _month, and _day:
        if f"{prefix}_year" in df and f"{prefix}_month" in df and f"{prefix}_day" in df:
            date_str_series = (
                df[f"{prefix}_year"].astype(str).str.pad(4, fillchar="0")
                + "-"
                + df[f"{prefix}_month"].astype(str).str.pad(2, fillchar="0")
                + "-"
                + df[f"{prefix}_day"].astype(str).str.pad(2, fillchar="0")
            )
            df[f"{prefix}_date"] = pd.to_datetime(
                date_str_series, format="%Y-%m-%d", errors="coerce"
            )
        else:
            raise ValueError(
                f"The dataframe does not contain a '{prefix}_date' column."
            )
    df[f"{prefix}_date"] = pd.to_datetime(
        df[f"{prefix}_date"], format="%Y-%m-%d", errors="coerce"
    )
    df[f"{prefix}_year"] = df[f"{prefix}_date"].dt.year
    df[f"{prefix}_month"] = df[f"{prefix}_date"].dt.month
    df[f"{prefix}_day"] = df[f"{prefix}_date"].dt.day
    df[f"{prefix}_quarter"] = df[f"{prefix}_date"].dt.quarter
    if add_year_month:
        df[f"{prefix}_year_month"] = (
            df[f"{prefix}_date"].dt.to_period("M").astype("str")
        )
    if add_year_quarter:
        df[f"{prefix}_year_quarter"] = (
            df[f"{prefix}_date"].dt.to_period("Q").astype("str")
        )
    checks = ["_year", "_month", "_day", "_year_month", "_year_quarter"]
    for check in checks:
        if f"{prefix}{check}" in df:
            if f"{prefix}_date" in df:
                if check in ["_year", "_month", "_day"]:
                    date_value = None
                    if check == "_year":
                        date_value = df[f"{prefix}_date"].dt.year.astype("Int64")
                    elif check == "_month":
                        date_value = df[f"{prefix}_date"].dt.month.astype("Int64")
                    elif check == "_day":
                        date_value = df[f"{prefix}_date"].dt.day.astype("Int64")
                    if not df[f"{prefix}{check}"].astype("Int64").equals(date_value):
                        n_diff = (
                            df[f"{prefix}{check}"].astype("Int64").ne(date_value).sum()
                        )
                        if n_diff > 0:
                            raise ValueError(
                                f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows."
                            )
                elif check in ["_year_month", "_year_quarter"]:
                    date_value = None
                    if check == "_year_month":
                        date_value = (
                            df[f"{prefix}_date"].dt.to_period("M").astype("str")
                        )
                    elif check == "_year_quarter":
                        date_value = (
                            df[f"{prefix}_date"].dt.to_period("Q").astype("str")
                        )
                    if not df[f"{prefix}{check}"].equals(date_value):
                        n_diff = df[f"{prefix}{check}"].ne(date_value).sum()
                        raise ValueError(
                            f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows."
                        )
    return df


def _boolify_series(series: pd.Series, na_handling: str = None):
    """Convert a series with potential string representations of booleans into actual
    booleans.
    """
    # Convert to string and clean if needed
    if series.dtype in ["object", "string", "str"]:
        series = series.astype(str).str.lower().str.strip()
        series = series.replace(["true", "t", "1", "y", "yes"], 1)
        series = series.replace(["false", "f", "0", "n", "no"], 0)
        # Convert common string representations of missing values to NaN
        none_patterns = ["none", "nan", "null", "na", "n/a", "-", "unknown"]
        series = series.replace(none_patterns, pd.NA)

    # Handle NA values before boolean conversion
    if na_handling == "true":
        series = series.fillna(True)
    elif na_handling == "false":
        series = series.fillna(False)
    else:
        series = series.fillna(False)

    # Convert to non-nullable boolean
    series = series.astype(bool)
    return series


def _boolify_column_in_df(df: pd.DataFrame, field: str, na_handling: str = None):
    """Convert a specified column in a DataFrame to boolean."""
    series = df[field]

    # Determine NA handling based on settings
    if na_handling == "na_false":
        na_handling = "false"
    elif na_handling == "na_true":
        na_handling = "true"
    elif na_handling is None:
        warnings.warn(
            f"No NA handling specified for boolean field '{field}'. Defaulting to 'na_false'."
        )
        na_handling = "false"
    else:
        raise ValueError(
            f"Invalid na_handling value: {na_handling}. Expected 'na_true', 'na_false', or None."
        )

    series = _boolify_series(series, na_handling)
    df[field] = series
    return df


def _enrich_universe_spatial_lag(
    df_univ_in: pd.DataFrame,
    df_test: pd.DataFrame,
    model_group : str,
    sample_from_mgs : list[str],
    settings: dict
) -> pd.DataFrame:

    df = df_univ_in.copy()

    s_sl = settings.get("data", {}).get("process", {}).get("enrich", {}).get("spatial_lag", {})

    if "floor_area_ratio" not in df:
        df["floor_area_ratio"] = div_series_z_safe(
            df["bldg_area_finished_sqft"], df["land_area_sqft"]
        )
    if "bedroom_density" not in df and "bldg_rooms_bed" in df:
        df["bedroom_density"] = div_series_z_safe(
            df["bldg_rooms_bed"], df["land_area_sqft"]
        )
    
    # only include samples not in the test set
    df_train_univ = df[~df["key"].isin(df_test["key"].values)].copy()
    # only sample from model groups explicitly listed
    df_train_univ = df_train_univ[df_train_univ["model_group"].isin(sample_from_mgs)]

    # only write to records in the model group:
    df = df[df["model_group"].eq(model_group)]

    # FAR, bedroom density, and big five:
    value_fields = {
        "floor_area_ratio": 5,
        "bedroom_density": 5,
        "bldg_age_years": 5,
        "bldg_effective_age_years": 5,
        "bldg_area_finished_sqft": 5,
        "land_area_sqft": 5,
        "bldg_quality_num": 5,
        "bldg_condition_num": 5,
    }

    extra_fields = s_sl.get("fields", {})
    for key in extra_fields:
        value_fields[key] = extra_fields[key]


    # Build a cKDTree from df_sales coordinates

    # we TRAIN on these coordinates -- coordinates that are NOT in the test set
    coords_train = df_train_univ[["latitude", "longitude"]].values
    tree = cKDTree(coords_train)

    # we PREDICT on these coordinates -- all the coordinates in the universe
    coords_all = df[["latitude", "longitude"]].values

    for value_field in value_fields:
        if value_field not in df:
            continue

        # Choose the number of nearest neighbors to use
        k = value_fields[value_field]

        # Query the tree: for each parcel in df_universe, find the k nearest parcels
        # distances: shape (n_universe, k); indices: corresponding indices in df_sales
        distances, indices = tree.query(coords_all, k=k)

        # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        # For each universe parcel, compute sigma as the mean distance to its k neighbors.
        sigma = distances.mean(axis=1, keepdims=True)

        # Handle zeros in sigma
        sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

        # Compute Gaussian kernel weights for all neighbors
        weights = np.exp(-(distances**2) / (2 * sigma**2))

        # Normalize the weights so that they sum to 1 for each parcel
        weights_norm = weights / weights.sum(axis=1, keepdims=True)

        # Get the values corresponding to the neighbor indices
        parcel_values = df_train_univ[value_field].values
        neighbor_values = parcel_values[indices]  # shape (n_universe, k)

        # Compute the weighted average (spatial lag) for each parcel in the universe
        spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_values)).sum(
            axis=1
        )

        # Add the spatial lag as a new column
        df[f"spatial_lag_{value_field}"] = spatial_lag

        median_value = df[value_field].median()
        df[f"spatial_lag_{value_field}"] = df[f"spatial_lag_{value_field}"].fillna(
            median_value
        )

    return df


def _enrich_df_basic(
    df_in: pd.DataFrame,
    s_enrich_this: dict,
    dataframes: dict[str, pd.DataFrame],
    settings: dict,
    is_sales: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform basic enrichment on a DataFrame including reference table joins,
    calculations, year built enrichment, and vacant status enrichment.
    """
    df = df_in.copy()

    supkey = "sales" if is_sales else "universe"
    
    for word in ["ref_tables", "calc", "tweak"]:
        if word in s_enrich_this:
            warnings.warn(f"Found `word` @ `data.process.enrich.word`, but it should be under `data.process.enrich.sales` or `data.process.enrich.universe`! Nothing will happen!")
    
    s_ref = s_enrich_this.get("ref_tables", {}).get(supkey, [])
    s_calc = s_enrich_this.get("calc", {}).get(supkey, {})
    s_tweak = s_enrich_this.get("tweak", {}).get(supkey, {})

    # reference tables:
    df = _perform_ref_tables(df, s_ref, dataframes, verbose=verbose)

    # calculations:
    df = perform_calculations(df, s_calc)

    # tweaks:
    df = perform_tweaks(df, s_tweak)

    # enrich year built:
    df = _enrich_year_built(df, settings, is_sales)

    return df


def _finesse_columns(
    df_in: pd.DataFrame | gpd.GeoDataFrame, suffix_left: str, suffix_right: str
):
    """Combine columns with matching base names but different suffixes into a single
    column.
    """
    df = df_in.copy()
    cols_to_finesse = []
    for col in df.columns.values:
        if col.endswith(suffix_left):
            base_col = col[: -len(suffix_left)]
            if base_col not in cols_to_finesse:
                cols_to_finesse.append(base_col)
    for col in cols_to_finesse:
        col_spatial = f"{col}{suffix_left}"
        col_data = f"{col}{suffix_right}"
        if col_spatial in df and col_data in df:
            df[col] = df[col_spatial].combine_first(df[col_data])
            df = df.drop(columns=[col_spatial, col_data], errors="ignore")
    return df


def _enrich_vacant(df_in: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Enrich the DataFrame by determining vacant properties based on finished building
    area.
    """

    df = df_in.copy()

    if "bldg_area_finished_sqft" in df_in:
        df["is_vacant"] = False

        df.loc[pd.isna(df["bldg_area_finished_sqft"]), "bldg_area_finished_sqft"] = 0
        df.loc[df["bldg_area_finished_sqft"].eq(0), "is_vacant"] = True

        idx_vacant = df["is_vacant"].eq(True)

        # Remove building characteristics from anything that is vacant:
        df = _simulate_removed_buildings(df, settings, idx_vacant)

    else:
        df = df_in

    return df


def _enrich_df_spatial_joins(
    df_in: pd.DataFrame,
    s_enrich_this: dict,
    dataframes: dict[str, pd.DataFrame],
    settings: dict,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """Perform basic geometric enrichment on a DataFrame by adding spatial features."""

    df = df_in.copy()
    s_geom = s_enrich_this.get("geometry", [])

    gdf: gpd.GeoDataFrame

    # geometry
    gdf = _perform_spatial_joins(s_geom, dataframes, verbose=verbose)

    gdf = gdf[gdf["key"].isin(df["key"].values)]

    # Merge everything together:
    try_keys = ["key", "key2", "key3"]
    success = False
    gdf_merged: gpd.GeoDataFrame | None = None
    for key in try_keys:
        if key in gdf and key in df:
            if verbose:
                print(f'Using "{key}" to merge shapefiles onto df')
            n_dupes_gdf = gdf.duplicated(subset=key).sum()
            n_dupes_df = df.duplicated(subset=key).sum()
            if n_dupes_gdf > 0 or n_dupes_df > 0:
                raise ValueError(
                    f'Found {n_dupes_gdf} duplicate keys for key "{key}" in the geo_parcels dataframe, and {n_dupes_df} duplicate keys in the base dataframe. Cannot perform spatial join. De-duplicate your dataframes and try again.'
                )
            gdf_merged = gdf.merge(
                df, on=key, how="left", suffixes=("_spatial", "_data")
            )
            gdf_merged = _finesse_columns(gdf_merged, "_spatial", "_data")
            success = True

            # count the number of times "key" appears in gdf_merged.columns:
            n_key = 0
            for col in gdf_merged:
                if col == "key":
                    n_key += 1

            if n_key > 1:
                print(
                    f'A Found {n_key} columns with "{key}" in the name. This may be a problem.'
                )
                print(f"Columns = {gdf_merged.columns}")

            break
    if not success:
        raise ValueError(
            f"Could not find a common key between geo_parcels and base dataframe. Tried keys: {try_keys}"
        )

    # drop null keys
    gdf_merged = gdf_merged[gdf_merged["key"].notna()]

    return gdf_merged


def _enrich_df_overture(
    gdf_in: gpd.GeoDataFrame,
    s_enrich_this: dict,
    dataframes: dict[str, pd.DataFrame],
    settings: dict,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    gdf_out = get_cached_df(gdf_in, "geom/overture", "key", s_enrich_this)
    if gdf_out is not None:
        if verbose:
            print("--> found cached data...")
        return gdf_out

    gdf = gdf_in.copy()

    s_overture = s_enrich_this.get("overture", {})
    # Enrich with Overture building data if enabled
    if s_overture.get("enabled", False):

        if verbose:
            print("Enriching with Overture building data...")

        # Initialize Overture service with the correct settings path
        overture_settings = {
            "overture": s_overture  # Pass the overture settings directly
        }
        overture_service = init_service_overture(overture_settings)

        # Get bounding box from data
        bbox = gdf.to_crs("EPSG:4326").total_bounds

        # Fetch building data
        buildings = overture_service.get_buildings(
            bbox, use_cache=s_overture.get("cache", True), verbose=verbose
        )

        if not buildings.empty:
            # Calculate building footprints
            sq_unit = get_small_area_unit(settings)
            s_footprint = s_overture.get("footprint", {})
            footprint_units = s_footprint.get("units", None)
            if footprint_units is None:
                warnings.warn(
                    f"`process.enrich.overture.footprint.units` not specified, defaulting to '{sq_unit}'"
                )
                footprint_units = sq_unit
            footprint_field = s_footprint.get("field", None)
            if footprint_field is None:
                warnings.warn(
                    f"`process.enrich.overture.footprint.field` not specified, defaulting to 'bldg_area_footprint_{sq_unit}'"
                )
                footprint_field = f"bldg_area_footprint_{sq_unit}"
            
            # Calculate building height
            len_unit = get_short_distance_unit(settings)
            s_height = s_overture.get("height", {})
            height_units = s_height.get("units", None)
            if height_units is None:
                warnings.warn(
                    f"`process.enrich.overture.height.units` not specified, defaulting to {len_unit}'"
                )
                height_units = len_unit
            height_field = s_height.get("field", None)
            if height_field is None:
                warnings.warn(
                    f"`process.enrich.overture.height.field` not specified, defaulting to 'bldg_height_{len_unit}'"
                )
                height_field = f"bldg_height_{len_unit}"
            
            gdf = overture_service.calculate_building_stats(
                gdf, 
                buildings,
                footprint_units,
                footprint_field,
                height_units,
                height_field,
                verbose=verbose
            )
            
            
        elif verbose:
            print("--> No buildings found in the area")

        write_cached_df(gdf_in, gdf, "geom/overture", "key", s_enrich_this)

    return gdf


def _enrich_spatial_inference(
    gdf_in: gpd.GeoDataFrame,
    s_enrich_this: dict,
    dataframes: dict[str, pd.DataFrame],
    settings: dict,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    gdf = gdf_in.copy()
    s_infer = s_enrich_this.get("infer", {})
    gdf = perform_spatial_inference(gdf, s_infer, "key", verbose=verbose)
    return gdf


def _enrich_permits(
    df_in: pd.DataFrame,
    s_enrich_this: dict,
    dataframes: dict[str, pd.DataFrame],
    settings: dict,
    is_sales: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    s_permits = s_enrich_this.get("permits", {})

    sources = s_permits.get("sources", [])
    if sources is None:
        return df_in

    df = df_in.copy()

    fields = [
        "key",
        "date",
        "is_teardown",
        "is_renovation",
        "renovation_txt",
        "renovation_num",
    ]

    df_all_permits: pd.DataFrame | None = None

    for source in sources:
        df_permits = dataframes[source]
        if df_permits is None:
            raise ValueError(f"Teardown source '{source}' not found in dataframes.")
        the_fields = [field for field in fields if field in df_permits.columns]
        if "key" not in the_fields:
            raise ValueError(f"Permits source '{source}' does not contain 'key' field.")
        if "date" not in the_fields:
            raise ValueError(
                f"Permits source '{source}' does not contain 'date' field."
            )
        if not pd.api.types.is_datetime64_any_dtype(df_permits["date"]):
            raise ValueError(
                f"Permits source '{source}' 'date' column must be of datetime type."
            )
        df_permits = df_permits[the_fields]
        if df_all_permits is None:
            df_all_permits = df_permits
        else:
            df_all_permits = pd.concat([df_all_permits, df_permits], ignore_index=True)

    if is_sales:
        df = _process_permits_sales(
            df, df_all_permits, s_permits, settings, verbose=verbose
        )
    else:
        df = _process_permits_univ(
            df, df_all_permits, s_permits, settings, verbose=verbose
        )
    return df


def _enrich_polar_coordinates(
    gdf_in: gpd.GeoDataFrame, settings: dict, verbose: bool = False
) -> gpd.GeoDataFrame:
    gdf = gdf_in[["key", "geometry"]].copy()

    longitude, latitude = get_center(settings, gdf)

    crs = get_crs(gdf, "equal_area")
    gdf = gdf.to_crs(crs)

    # convert longitude, latitude, to same point space as gdf:
    point = Point(longitude, latitude)
    single_point_gdf = gpd.GeoDataFrame({"geometry": [point]}, crs=gdf_in.crs)
    single_point_gdf = single_point_gdf.to_crs(crs)

    x_center = single_point_gdf.geometry.x.iloc[0]
    y_center = single_point_gdf.geometry.y.iloc[0]

    gdf["x_diff"] = gdf.geometry.centroid.x - x_center
    gdf["y_diff"] = gdf.geometry.centroid.y - y_center

    gdf["polar_radius"] = np.sqrt(gdf["x_diff"] ** 2 + gdf["y_diff"] ** 2)
    gdf["polar_angle"] = np.arctan2(gdf["y_diff"], gdf["x_diff"])
    gdf["polar_angle"] = np.degrees(gdf["polar_angle"])

    gdf_result = gdf_in.merge(
        gdf[["key", "polar_radius", "polar_angle"]], on="key", how="left"
    )
    return gdf_result


def _basic_geo_enrichment(
    gdf_in: gpd.GeoDataFrame, s_enrich: dict, settings: dict, verbose: bool = False
) -> gpd.GeoDataFrame:
    """Perform basic geometric enrichment on a GeoDataFrame by adding spatial features."""
    t = TimingData()
    
    s_basic = s_enrich.get("basic", {})
    
    do_anything = s_basic.get("enabled", True)
    if not do_anything:
        if verbose:
            print("Skipping basic geo enrichment...")
        return
    
    do_latlon = s_basic.get("latlon", True)
    do_area = s_basic.get("area", True)
    do_shape = s_basic.get("shape", True)
    do_polar = s_basic.get("polar", True)
    
    if verbose:
        print(f"Performing basic geometric enrichment...")
    gdf_out = get_cached_df(gdf_in, "geom/basic", "key")
    if gdf_out is not None:
        if verbose:
            print("--> found cached data...")

        parcels_with_no_land = gdf_out["land_area_sqft"].isna().sum()
        if parcels_with_no_land > 0:
            raise ValueError(
                f"Found '{parcels_with_no_land}' parcels with no land area in cached data. This should not be able to happen as they should be backfilled with GIS land area. Please check your data."
            )

        return gdf_out

    gdf = gdf_in.copy()

    
    if do_latlon:
        t.start("latlon")
        gdf_latlon = gdf.to_crs(get_crs(gdf, "latlon"))
        gdf["latitude"] = gdf_latlon.geometry.centroid.y
        gdf["longitude"] = gdf_latlon.geometry.centroid.x
        gdf["latitude_norm"] = (gdf["latitude"] - gdf["latitude"].min()) / (
            gdf["latitude"].max() - gdf["latitude"].min()
        )
        gdf["longitude_norm"] = (gdf["longitude"] - gdf["longitude"].min()) / (
            gdf["longitude"].max() - gdf["longitude"].min()
        )
        t.stop("latlon")
        
        if verbose:
            _t = t.get("latlon")
            print(f"--> added latitude/longitude...({_t:.2f}s)")
    else:
        if verbose:
            print(f"--> skipping latitude/longitude...")
    
    if do_area:
        t.start("area")
        gdf_area = gdf.to_crs(get_crs(gdf, "equal_area"))
        
        # we converted to a metric CRS, so we are in meters right now
        area_in_meters = gdf_area.geometry.area

        gdf["land_area_gis_sqft"] = area_in_meters * 10.7639

        if "land_area_sqft" not in gdf:
            gdf["land_area_sqft"] = gdf["land_area_gis_sqft"]
        else:
            gdf["land_area_given_sqft"] = gdf["land_area_sqft"]
        
            # Anywhere given land area is 0, negative, or NULL, use GIS area
            gdf["land_area_sqft"] = gdf["land_area_sqft"].combine_first(
                gdf["land_area_gis_sqft"]
            )
            gdf["land_area_sqft"] = np.round(
                gdf["land_area_sqft"].combine_first(gdf["land_area_gis_sqft"])
            ).astype(int)
            gdf.loc[
                gdf["land_area_given_sqft"].le(0) | gdf["land_area_given_sqft"].isna(),
                "land_area_sqft",
            ] = gdf["land_area_gis_sqft"]

            # Calculate difference
            gdf["land_area_gis_delta_sqft"] = gdf["land_area_gis_sqft"] - gdf["land_area_sqft"]
            gdf["land_area_gis_delta_percent"] = div_series_z_safe(
                gdf["land_area_gis_delta_sqft"], gdf["land_area_sqft"]
            )

        gdf["land_area_sqft_log"] = np.log(gdf["land_area_sqft"])

        t.stop("area")
    
        if verbose:
            _t = t.get("area")
            print(f"--> calculated GIS area of each parcel...({_t:.2f}s)")
    else:
        if verbose:
            print(f"--> skipping calculated area...")
    
    if do_shape:
        gdf = _calc_parcel_shape(gdf, verbose)
    else:
        if verbose:
            print(f"--> skipping calculated parcel shapes...")
    
    if do_polar:
        t.start("polar")
        gdf = _enrich_polar_coordinates(gdf, settings, verbose)
        t.stop("polar")
        if verbose:
            _t = t.get("polar")
            print(f"--> calculated polar coordinates...({_t:.2f}s)")
    else:
        if verbose:
            print(f"--> skipping polar coordinates...")
    
    parcels_with_no_land = gdf["land_area_sqft"].isna().sum()
    if parcels_with_no_land > 0:
        raise ValueError(
            f"Found '{parcels_with_no_land}' parcels with no land area. This should not be able to happen as they should be backfilled with GIS land area. Please check your data."
        )

    write_cached_df(gdf_in, gdf, "geom/basic", "key")

    return gdf


def _calc_parcel_shape(
    gdf_in: gpd.GeoDataFrame, verbose: bool = False
) -> gpd.GeoDataFrame:
    """Compute additional geometric properties for a GeoDataFrame, such as rectangularity"""

    gdf = get_cached_df(gdf_in, "geom/stuff", "key")
    if gdf is not None:
        return gdf

    t = TimingData()
    t.start("rectangularity")
    gdf = gdf_in.copy()
    min_rotated_rects = gdf.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    min_rotated_rects_area_delta = np.abs(min_rotated_rects.area - gdf.geometry.area)
    min_rotated_rects_area_delta_percent = div_series_z_safe(
        min_rotated_rects_area_delta, gdf.geometry.area
    )
    gdf["geom_rectangularity_num"] = 1.0 - min_rotated_rects_area_delta_percent
    coords = min_rotated_rects.apply(
        lambda rect: np.array(rect.exterior.coords[:-1])
    )  # Drop duplicate last point
    t.stop("rectangularity")
    if verbose:
        _t = t.get("rectangularity")
        print(f"--> calculated parcel rectangularity...({_t:.2f}s)")
    t.start("aspect_ratio")
    edge_lengths = coords.apply(
        lambda pts: np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    )
    dimensions = edge_lengths.apply(lambda lengths: np.sort(lengths)[:2])
    aspect_ratios = dimensions.apply(
        lambda dims: dims[1] / dims[0] if dims[0] != 0 else float("inf")
    )
    gdf["geom_aspect_ratio"] = aspect_ratios
    t.stop("aspect_ratio")
    if verbose:
        _t = t.get("aspect_ratio")
        print(f"--> calculated parcel aspect ratios...({_t:.2f}s)")
    gdf = identify_irregular_parcels(gdf, verbose)

    write_cached_df(gdf_in, gdf, "geom/stuff", "key")
    return gdf


def _perform_spatial_joins(
    s_geom: list, dataframes: dict[str, pd.DataFrame], verbose: bool = False
) -> gpd.GeoDataFrame:
    """Perform spatial joins based on a list of spatial join instructions.

    Strings in s_geom are interpreted as IDs of loaded shapefiles; dicts must contain an
    'id' and optionally a 'predicate' (default "contains_centroid").
    """
    if not isinstance(s_geom, list):
        s_geom = [s_geom]

    if "geo_parcels" not in dataframes:
        raise ValueError(
            "No 'geo_parcels' dataframe found in the dataframes. This layer is required, and it must contain parcel geometry."
        )

    gdf_parcels: gpd.GeoDataFrame = dataframes["geo_parcels"]
    gdf_merged = gdf_parcels.copy()
    
    if verbose:
        print(f"Performing spatial joins...")

    for geom in s_geom:
        if isinstance(geom, str):
            entry = {"id": str(geom), "predicate": "contains_centroid"}
        elif isinstance(geom, dict):
            entry = geom
        else:
            raise ValueError(f"Invalid geometry entry: {geom}")
        _id = entry.get("id")
        predicate = entry.get("predicate", "contains_centroid")
        if _id is None:
            raise ValueError("No 'id' found in geometry entry.")
        if verbose:
            if predicate != "contains_centroid":
                print(f"--> {_id} @ {predicate}")
            else:
                print(f"--> {_id}")
        gdf = dataframes[_id]
        fields_to_tag = entry.get("fields", None)
        if fields_to_tag is None:
            fields_to_tag = [field for field in gdf.columns if field != "geometry"]
        else:
            for field in fields_to_tag:
                if field not in gdf:
                    raise ValueError(
                        f"Field to tag '{field}' not found in geometry dataframe '{_id}'."
                    )
        gdf_merged = _perform_spatial_join(gdf_merged, gdf, predicate, fields_to_tag)

        n_keys = 0
        for col in gdf_merged.columns:
            if col == "key":
                n_keys += 1
        if n_keys > 1:
            print(
                f'Found {n_keys} columns with "key" in the name. This may be a problem.'
            )
            print(f"Columns = {gdf_merged.columns}")

    gdf_no_geometry = gdf_merged[gdf_merged["geometry"].isna()]
    if len(gdf_no_geometry) > 0:
        warnings.warn(
            f"Found {len(gdf_no_geometry)} parcels with no geometry. These parcels will be excluded from the analysis. You can find them in out/errors/"
        )
        os.makedirs("out/errors", exist_ok=True)
        gdf_no_geometry.to_parquet("out/errors/parcels_no_geometry.parquet")
        gdf_no_geometry.to_csv("out/errors/parcels_no_geometry.csv", index=False)
        gdf_no_geom_keys = gdf_no_geometry["key"].values
        with open("out/errors/parcels_no_geometry_keys.txt", "w") as f:
            for key in gdf_no_geom_keys:
                f.write(f"{key}\n")
        gdf_merged = gdf_merged.dropna(subset=["geometry"])

    return gdf_merged


def _perform_spatial_join_contains_centroid(
    gdf: gpd.GeoDataFrame, gdf_overlay: gpd.GeoDataFrame
):
    """Perform a spatial join where the centroid of geometries in gdf is within
    gdf_overlay.
    """
    # Compute centroids of each parcel
    gdf["geometry_centroid"] = gdf.geometry.centroid

    # Use within predicate for spatial join
    gdf = gpd.sjoin(
        gdf.set_geometry("geometry_centroid"),
        gdf_overlay,
        how="left",
        predicate="within",
    )
    # remove extra columns like "index_right":
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    return gdf


def _perform_spatial_join(
    gdf_in: gpd.GeoDataFrame,
    gdf_overlay: gpd.GeoDataFrame,
    predicate: str,
    fields_to_tag: list[str],
):
    """Perform a spatial join between two GeoDataFrames using the specified predicate."""
    gdf = gdf_in.copy()
    
    if gdf.crs is None:
        if is_likely_epsg4326(gdf):
            gdf.set_crs(epsg=4326, inplace=True)
            warnings.warn("The geodataframe was missing a CRS, but it looks like it is probably EPSG 4326, so we set that automatically. Make sure this is what you want!")
        else:
            raise ValueError("The geodataframe is missing a CRS, and it didn't look like EPSG 4326, so we couldn't set that automatically. Your source file is likely corrupted or missing a CRS.")
        
    if gdf_overlay.crs is None:
        if is_likely_epsg4326(gdf_overlay):
            gdf_overlay.set_crs(epsg=4326, inplace=True)
            warnings.warn("The overlay geodataframe was missing a CRS, but it looks like it is probably EPSG 4326, so we set that automatically. Make sure this is what you want!")
        else:
            raise ValueError("The overlay geodataframe is missing a CRS, and it didn't look like EPSG 4326, so we couldn't set that automatically. Your source file is likely corrupted or missing a CRS.")
    
    gdf_overlay = gdf_overlay.to_crs(gdf.crs)
    if "__overlay_id__" in gdf_overlay:
        raise ValueError(
            "The overlay GeoDataFrame already contains a '__overlay_id__' column. This column is used internally by the spatial join function, and must not be present in the overlay GeoDataFrame."
        )
    gdf_overlay["__overlay_id__"] = range(len(gdf_overlay))
    # TODO: add more predicates as needed
    if predicate == "contains_centroid":
        gdf = _perform_spatial_join_contains_centroid(gdf, gdf_overlay)
    else:
        raise ValueError(f"Invalid spatial join predicate: {predicate}")
    gdf = gdf.drop(columns=fields_to_tag, errors="ignore")
    gdf = gdf.merge(
        gdf_overlay[["__overlay_id__"] + fields_to_tag], on="__overlay_id__", how="left"
    )
    gdf.set_geometry("geometry", inplace=True)
    gdf = gdf.drop(columns=["geometry_centroid", "__overlay_id__"], errors="ignore")
    return gdf


def _do_perform_distance_calculations(
    df_in: gpd.GeoDataFrame,
    gdf_in: gpd.GeoDataFrame,
    _id: str,
    max_distance: float = None,
    unit: str = "km",
) -> pd.DataFrame:
    """Perform a divide-by-zero-safe nearest neighbor spatial join to calculate
    distances.
    """
    unit_factors = {"m": 1, "km": 0.001, "mile": 0.000621371, "ft": 3.28084}
    if unit not in unit_factors:
        raise ValueError(f"Unsupported unit '{unit}'")
    crs = get_crs(df_in, "equal_distance")

    # check for duplicate keys:
    if df_in.duplicated(subset="key").sum() > 0:
        # caching won't work if there's duplicate keys, and there shouldn't be any duplicate keys here anyways
        raise ValueError(
            f"Duplicate keys found before distance calculation for '{_id}.' This should not happen."
        )

    # construct a unique cache signature
    signature = {
        "crs": crs.name,
        "_id": _id,
        "max_distance": max_distance,
        "unit": unit,
        "df_in_len": len(df_in),
        "gdf_in_len": len(gdf_in),
        "df_cols": sorted(df_in.columns.tolist()),
        "gdf_cols": sorted(gdf_in.columns.tolist()),
    }

    # check if we already have this distance calculation
    df_out = get_cached_df(df_in, f"osm/do_distance_{_id}", "key", signature)
    if df_out is not None:
        return df_out

    df_projected = df_in.to_crs(crs).copy()
    gdf_projected = gdf_in.to_crs(crs).copy()

    # Initialize dictionary to store new columns
    new_columns = {
        f"within_{_id}": pd.Series(False, index=df_projected.index),
        f"dist_to_{_id}": pd.Series(np.nan, index=df_projected.index),
    }

    if max_distance is not None:
        # Create buffer around features we're measuring distance to
        gdf_buffer = gdf_projected.copy()
        gdf_buffer.geometry = gdf_buffer.geometry.buffer(
            max_distance / unit_factors[unit]
        )

        # Find parcels that intersect with the buffer
        parcels_within = gpd.sjoin(
            df_projected, gdf_buffer, how="inner", predicate="intersects"
        )

        # Clean up any index_right column from the spatial join
        parcels_within = parcels_within.drop(columns=["index_right"], errors="ignore")

        # Only calculate distances for parcels within buffer
        if len(parcels_within) > 0:
            nearest = gpd.sjoin_nearest(
                parcels_within, gdf_projected, how="left", distance_col=f"dist_to_{_id}"
            )

            # Clean up any index_right column from the spatial join
            nearest = nearest.drop(columns=["index_right"], errors="ignore")

            # Keep only the columns we need
            nearest = nearest[["key", f"dist_to_{_id}"]]

            nearest[f"dist_to_{_id}"] *= unit_factors[unit]

            # Mark these parcels as within distance
            new_columns[f"within_{_id}"] = pd.Series(False, index=df_projected.index)
            new_columns[f"within_{_id}"].loc[
                df_projected["key"].isin(parcels_within["key"])
            ] = True

            # Handle duplicates in nearest
            if nearest.duplicated(subset="key").sum() > 0:
                nearest = nearest.sort_values(
                    by=["key", f"dist_to_{_id}"], ascending=[True, True]
                )
                nearest = nearest.drop_duplicates(subset="key")

            # Add distance column
            distances_series = pd.Series(nearest.set_index("key")[f"dist_to_{_id}"])
            new_columns[f"dist_to_{_id}"] = distances_series.reindex(
                df_projected["key"]
            ).values

    else:
        # If no max_distance specified, calculate for all parcels
        nearest = gpd.sjoin_nearest(
            df_projected, gdf_projected, how="left", distance_col=f"dist_to_{_id}"
        )

        # Clean up any index_right column from the spatial join
        nearest = nearest.drop(columns=["index_right"], errors="ignore")

        # Keep only the columns we need
        nearest = nearest[["key", f"dist_to_{_id}"]]

        nearest[f"dist_to_{_id}"] *= unit_factors[unit]

        # Handle duplicates in nearest
        if nearest.duplicated(subset="key").sum() > 0:
            nearest = nearest.sort_values(
                by=["key", f"dist_to_{_id}"], ascending=[True, True]
            )
            nearest = nearest.drop_duplicates(subset="key")

        # All parcels considered "within distance" when no max_distance specified
        new_columns[f"within_{_id}"] = pd.Series(True, index=df_projected.index)

        # Add distance column
        distances_series = pd.Series(nearest.set_index("key")[f"dist_to_{_id}"])
        new_columns[f"dist_to_{_id}"] = distances_series.reindex(
            df_projected["key"]
        ).values

    # Create new DataFrame with all new columns
    new_df = pd.DataFrame(new_columns, index=df_projected.index)

    # Combine original DataFrame with new columns using concat
    df_out = pd.concat([df_in, new_df], axis=1)

    # Calculate log versions of all columns
    for col in new_columns:
        if col in df_out and "dist_to_" in col:
            log_dist_to = col.replace("dist_to_", "log_dist_to_", 1)
            df_out[log_dist_to] = np.log(df_out[col])

    write_cached_df(df_in, df_out, f"osm/do_distance_{_id}", "key", signature)

    return df_out


def _do_perform_distance_calculations_osm(
    df_in: gpd.GeoDataFrame,
    gdf_in: gpd.GeoDataFrame,
    _id: str,
    max_distance: float = None,
    unit: str = "km",
) -> pd.DataFrame:
    """Perform a divide-by-zero-safe nearest neighbor spatial join to calculate
    distances.
    """
    unit_factors = {"m": 1, "km": 0.001, "mile": 0.000621371, "ft": 3.28084}
    if unit not in unit_factors:
        raise ValueError(f"Unsupported unit '{unit}'")

    # Get appropriate CRS for distance calculations
    crs = get_crs(df_in, "equal_distance")

    # Check for duplicate keys
    if df_in.duplicated(subset="key").sum() > 0:
        raise ValueError(
            f"Duplicate keys found before distance calculation for '{_id}.' This should not happen."
        )

    if max_distance is not None:
        # Convert max_distance to meters (since our distances are in meters)
        max_distance_m = max_distance / unit_factors[unit]
    else:
        max_distance_m = None

    print(f"Calculating distance, id={_id}, max_distance={max_distance}, unit={unit}, max_distance (in meters)={max_distance_m}")

    # Construct cache signature
    signature = {
        "crs": crs.name,
        "_id": _id,
        "max_distance": max_distance,
        "unit": unit,
        "df_in_len": len(df_in),
        "gdf_in_len": len(gdf_in),
        "df_cols": sorted(df_in.columns.tolist()),
        "gdf_cols": sorted(gdf_in.columns.tolist())
    }

    # Check cache
    df_out = get_cached_df(df_in, f"osm/do_distance_{_id}", "key", signature)
    if df_out is not None:
        return df_out

    # Project geometries
    df_projected = df_in.to_crs(crs).copy()
    gdf_projected = gdf_in.to_crs(crs).copy()

    # Calculate distances for all parcels first
    nearest = gpd.sjoin_nearest(
        df_projected, gdf_projected, how="left", distance_col="distance", max_distance=max_distance_m
    )

    # Handle duplicates by keeping shortest distance
    if nearest.duplicated(subset="key").sum() > 0:
        nearest = nearest.sort_values("distance").drop_duplicates("key")

    # Create distance series (distances are in meters at this point)
    distance_series = pd.Series(nearest["distance"].values, index=nearest.index)

    # Initialize within flag
    within_series = pd.Series(False, index=df_projected.index)

    if max_distance is not None:
        # Mark parcels within max_distance
        within_series[distance_series <= max_distance_m] = True

        # Set distances beyond max_distance to max_distance + 1 (in the target unit)
        distance_series[distance_series > max_distance_m] = (
            max_distance + 1
        ) / unit_factors[unit]

        # Convert all distances to target unit
        distance_series = distance_series * unit_factors[unit]
    else:
        # If no max_distance, all parcels are considered "within"
        within_series[:] = True
        # Convert distances to target unit
        distance_series = distance_series * unit_factors[unit]

    proximity_series = np.max(distance_series) - distance_series

    # Create output DataFrame with new columns
    new_df = pd.DataFrame(
        {
            f"dist_to_{_id}": distance_series,
            f"within_{_id}": within_series,
            f"proximity_to_{_id}": proximity_series
        },
        index=df_projected.index,
    )

    # Fill null proximity with 0.0 (maximum distance)
    new_df[f"proximity_to_{_id}"] = new_df[f"proximity_to_{_id}"].fillna(0.0)

    # Combine with original DataFrame
    df_out = pd.concat([df_in, new_df], axis=1)

    # Cache results
    write_cached_df(df_in, df_out, f"osm/do_distance_{_id}", "key", signature)

    return df_out


def _perform_distance_calculations(
    df_in: gpd.GeoDataFrame,
    s_dist: list,
    dataframes: dict[str, pd.DataFrame],
    unit: str = "km",
    verbose: bool = False,
    cache_key: str = "geom/distance",
) -> gpd.GeoDataFrame:
    """Perform distance calculations based on enrichment instructions."""
    df = df_in.copy()
    if verbose:
        print(f"Performing distance calculations {cache_key}...")

    # Collect all distance calculations to apply at once
    all_distance_dfs = []

    # check for duplicate keys:
    if df_in.duplicated(subset="key").sum() > 0:
        # caching won't work if there's duplicate keys, and there shouldn't be any duplicate keys here anyways
        raise ValueError(
            f"Duplicate keys found before distance calculation. This should not happen."
        )

    signature = {
        "unit": unit,
        "crs": df_in.crs.name,
        "df_in_len": len(df_in),
        "df_cols": sorted(df_in.columns.tolist()),
        "s_dist": s_dist,
    }

    gdf_out = get_cached_df(df_in, cache_key, "key", signature)
    if gdf_out is not None:
        if verbose:
            print("--> found cached data...")
        return gdf_out

    for entry in s_dist:
        if isinstance(entry, str):
            entry = {"id": str(entry)}
        elif not isinstance(entry, dict):
            raise ValueError(f"Invalid distance entry: {entry}")

        _id = entry.get("id")

        source = entry.get("source", _id)

        max_distance = entry.get("max_distance")  # Get max_distance from settings
        entry_unit = entry.get("unit", unit)  # Allow overriding unit per feature

        if _id is None:
            raise ValueError("No 'id' found in distance entry.")
        if source not in dataframes:
            if verbose:
                print(
                    f"--> Skipping {_id} - not found in dataframes (likely disabled in settings)"
                )
            continue

        gdf = dataframes[source]
        field = entry.get("field", None)

        if verbose:
            print(f"--> {_id}")
            if max_distance is not None:
                print(f"    max_distance: {max_distance} {entry_unit}")

        if field is None:
            if verbose:
                print(f"--> {_id} field is None")

            # Calculate distances for this feature
            distance_df = _do_perform_distance_calculations(
                df, gdf, _id, max_distance, entry_unit
            )

            # Extract only the new columns
            new_cols = [col for col in distance_df.columns if col not in df.columns]

            all_distance_dfs.append(distance_df[new_cols])
            if verbose:
                print(f"--> {_id} done")
        else:
            if verbose:
                print(f"--> {_id} field is {field}")
            uniques = gdf[field].unique()
            for unique in uniques:
                if pd.isna(unique):
                    continue
                gdf_subset = gdf[gdf[field].eq(unique)]
                # Calculate distances for this subset
                distance_df = _do_perform_distance_calculations(
                    df, gdf_subset, f"{_id}_{unique}", max_distance, entry_unit
                )
                # Extract only the new columns
                new_cols = [col for col in distance_df.columns if col not in df.columns]

                all_distance_dfs.append(distance_df[new_cols])
            if verbose:
                print(f"--> {_id} done")

    # Apply all distance calculations at once
    if len(all_distance_dfs):
        # Combine all distance DataFrames
        combined_distances = pd.concat(all_distance_dfs, axis=1)
        # Combine with original DataFrame
        df = pd.concat([df, combined_distances], axis=1)

    new_cols = [col for col in df.columns if col not in df_in.columns]
    df_net_change = df[["key"] + new_cols].copy()
    # check for duplicate keys:

    if df_net_change.duplicated(subset="key").sum() > 0:
        raise ValueError(
            f"Duplicate keys found after distance calculation. This should not happen."
        )

    # save to cache:
    write_cached_df(df_in, df, cache_key, "key", signature)

    return df


def _perform_ref_tables(
    df_in: pd.DataFrame | gpd.GeoDataFrame,
    s_ref: list | dict,
    dataframes: dict[str, pd.DataFrame],
    verbose: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Perform reference table joins to enrich the input DataFrame."""
    df = df_in.copy()
    if not isinstance(s_ref, list):
        s_ref = [s_ref]

    if verbose:
        print(f"Performing reference table joins...")

    for ref in s_ref:
        _id = ref.get("id", None)
        key_ref_table = ref.get("key_ref_table", None)
        key_target = ref.get("key_target", None)
        add_fields = ref.get("add_fields", None)
        if verbose:
            print(f"--> {_id}")
        if _id is None:
            raise ValueError("No 'id' found in ref table.")
        if key_ref_table is None:
            raise ValueError("No 'key_ref_table' found in ref table.")
        if key_target is None:
            raise ValueError("No 'key_target' found in ref table.")
        if add_fields is None:
            raise ValueError("No 'add_fields' found in ref table.")
        if not isinstance(add_fields, list):
            raise ValueError("The 'add_fields' field must be a list of strings.")
        if len(add_fields) == 0:
            raise ValueError("The 'add_fields' field must contain at least one string.")
        if _id not in dataframes:
            raise ValueError(f"Ref table '{_id}' not found in dataframes.")
        df_ref = dataframes[_id]
        if key_ref_table not in df_ref:
            raise ValueError(
                f"Key field '{key_ref_table}' not found in ref table '{_id}'."
            )
        if key_target not in df:
            print(f"Target field '{key_target}' not found in base dataframe")
            print(f"base df columns = {df.columns.values}")
            raise ValueError(f"Target field '{key_target}' not found in base dataframe")
        for field in add_fields:
            if field not in df_ref:
                raise ValueError(f"Field '{field}' not found in ref table '{_id}'.")
            if field in df_in:
                raise ValueError(f"Field '{field}' already exists in base dataframe.")
        df_ref = df_ref[[key_ref_table] + add_fields]
        if key_ref_table == key_target:
            df = df.merge(df_ref, on=key_target, how="left")
        else:
            df = df.merge(
                df_ref, left_on=key_target, right_on=key_ref_table, how="left"
            )
            df = df.drop(columns=[key_ref_table])
    return df


def _get_calc_cols(settings: dict, exclude_loaded_fields: bool = False) -> list[str]:
    """Retrieve a list of calculated columns based on settings."""
    s_load = settings.get("data", {}).get("load", {})
    cols_found = []
    cols_base = []
    for key in s_load:
        entry = s_load[key]
        cols = _do_get_calc_cols(entry)
        cols_found += cols
        if exclude_loaded_fields:
            entry_load = entry.get("load", {})
            for load_key in entry_load:
                cols_base.append(load_key)

    cols_found = list(set(cols_found) - set(cols_base))
    return cols_found


def _do_get_calc_cols(df_entry: dict) -> list[str]:
    """Extract column names referenced in a calculation dictionary."""
    e_calc = df_entry.get("calc", {})
    fields_in_calc = _crawl_calc_dict_for_fields(e_calc)
    return fields_in_calc


def load_dataframe(
    entry: dict,
    settings: dict,
    verbose: bool = False,
    fields_cat: list = None,
    fields_bool: list = None,
    fields_num: list = None,
) -> pd.DataFrame | None:
    """Load a DataFrame from a file based on instructions and perform calculations and
    type adjustments.
    """
    filename = entry.get("filename", "")
    entry_key = entry.get("key", "")
    if filename == "":
        return None
    filename = f"in/{filename}"
    ext = str(filename).split(".")[-1]

    column_names = _snoop_column_names(filename)

    e_load = entry.get("load", {})

    # Get all calc and tweak operations in order they appear
    operation_order = []
    for key in entry:
        if "calc" in key or "tweak" in key:  # Match any key containing calc or tweak
            op_type = "calc" if "calc" in key else "tweak"
            operation_order.append({"type": op_type, "operations": entry[key]})
    
    # Get all fields used in aggregation operations
    dupes = entry.get("dupes", {})
    agg = dupes.get("agg", {})
    agg_fields = []
    for agg_key in agg:
        agg_entry = agg[agg_key]
        agg_field = agg_entry.get("field", "")
        if agg_field != "" and agg_field not in agg_fields:
            agg_fields.append(agg_field)
    
    
    if verbose:
        print(f'Loading "{filename}"...')

    rename_map = {}
    dtype_map = {}
    extra_map = {}
    cols_to_load = []

    for rename_key in e_load:
        original = e_load[rename_key]
        original_key = None
        if isinstance(original, list):
            if len(original) > 0:
                original_key = original[0]
                cols_to_load += [original_key]
                rename_map[original_key] = rename_key
            if len(original) > 1:
                dtype_map[original_key] = original[1]
            if len(original) > 2:
                extra_map[rename_key] = original[2]
        elif isinstance(original, str):
            cols_to_load += [original]
            rename_map[original] = rename_key

    # Only include fields from calcs that exist in the source data
    fields_in_calc = []
    for operation in operation_order:
        if operation["type"] == "calc":
            fields_in_calc.extend(_crawl_calc_dict_for_fields(operation["operations"]))
    fields_in_calc = [f for f in fields_in_calc if f in column_names]
    cols_to_load += fields_in_calc
    
    # Only include fields from aggs that exist in the source data
    fields_in_agg = [f for f in agg_fields if f in column_names]
    cols_to_load += fields_in_agg
    
    cols_to_load = list(set(cols_to_load))
    
    is_geometry = False
    if "geometry" in column_names and "geometry" not in cols_to_load:
        cols_to_load.append("geometry")
        is_geometry = True
    if is_geometry:
        is_geometry = entry.get("geometry", is_geometry)
    
    if ext == "parquet":
        try:
            df = gpd.read_parquet(filename, columns=cols_to_load)
            if "geometry" in df:
                crs, geom_col = detect_crs_from_parquet(filename, "geometry")
                df = ensure_geometries(df, geom_col=geom_col, crs=crs)
        except ValueError:
            df = pd.read_parquet(filename, columns=cols_to_load)
    elif ext == "csv":
        csv_dtype_map = {}
        for key in dtype_map:
            dtype_value = dtype_map[key]
            if dtype_value == "datetime":
                dtype_value = "string"
            csv_dtype_map[key] = dtype_value
        df = pd.read_csv(filename, usecols=cols_to_load, dtype=csv_dtype_map)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Enforce user's dtypes
    for col in df.columns:
        if col in dtype_map:
            target_dtype = dtype_map[col]
            if target_dtype == "bool" or target_dtype == "boolean":
                rename_key = rename_map.get(col, col)
                if rename_key in extra_map:
                    # if the user has specified a na_handling, we will manually boolify the column
                    na_handling = extra_map[rename_key]
                    df = _boolify_column_in_df(df, col, na_handling)
                else:
                    # otherwise, we use the exact dtype they specified with a warning and default to casting NA to false
                    warnings.warn(
                        f"Column '{col}' is being converted to boolean, but you didn't specify na_handling. All ambiguous values/NA's will be cast to false."
                    )
                    df[col] = df[col].astype(target_dtype)
                    df = _boolify_column_in_df(df, col, "na_false")
            elif target_dtype == "datetime":
                rename_key = rename_map.get(col, col)
                format_str = extra_map.get(rename_key)
                if rename_key in extra_map:
                    format_str = extra_map[rename_key]
                    try:
                        result = pd.to_datetime(df[col].astype(str), format=format_str)
                    except ValueError:
                        s = df[col].astype(str).replace({None: pd.NA, "None": pd.NA, "": pd.NA})
                        result = pd.to_datetime(s, format=format_str, errors="coerce", exact=True)
                    df[col] = result
                else:
                    warnings.warn(
                        f"Column '{col}' is being converted to datetime, but you didn't specify the format. Will attempt to auto-cast and coerce, which could be wrong!"
                    )
                    df[col] = pd.to_datetime(df[col].astype(str), errors="coerce")
            else:
                try:
                    df[col] = df[col].astype(target_dtype)
                except ValueError as e:
                    if target_dtype == "float":
                        # force lowercase since we've converting to float anyways
                        df[col] = df[col].astype(str).str.lower()
                        
                        # check for and clear various known problematic strings
                        for badvalue in [' ', '<na>', 'none', 'null', 'na']:
                            df.loc[df[col].eq(badvalue), col] = None
                        
                        warnings.warn(f"Column {col} had values that could not be cast to float, suppressed them to null")
                        df[col] = df[col].astype(target_dtype, errors="ignore")
                    else:
                        raise ValueError(f"Error casting column {col} to dtype {dtype_map[col]}: {e}")
    
    # Rename columns
    df = df.rename(columns=rename_map)

    # Perform operations in order they appear in settings
    for operation in operation_order:
        op_type = operation["type"]
        if op_type == "calc":
            df = perform_calculations(df, operation["operations"], rename_map)
        elif op_type == "tweak":
            df = perform_tweaks(df, operation["operations"], rename_map)

    if fields_cat is None:
        fields_cat = get_fields_categorical(settings, include_boolean=False)
    if fields_bool is None:
        fields_bool = get_fields_boolean(settings)
    if fields_num is None:
        fields_num = get_fields_numeric(settings, include_boolean=False)

    for col in df.columns:
        if col in fields_cat:
            if "date" not in col:
                df[col] = df[col].astype("string")
        elif col in fields_bool or df[col].dtype == "boolean":
            na_handling = None
            if col in extra_map:
                na_handling = extra_map[col]
            df = _boolify_column_in_df(df, col, na_handling)
        elif col in fields_num:
            mask_non_numeric = ~df[col].apply(lambda x: isinstance(x, (int, float)))
            if mask_non_numeric.sum() > 0:
                df.loc[mask_non_numeric, col] = np.nan
            df[col] = df[col].astype("Float64")
    
    date_fields = get_fields_date(settings, df)
    time_format_map = {}
    for xkey in extra_map:
        if xkey in date_fields:
            time_format_map[xkey] = extra_map[xkey]

    for dkey in date_fields:
        if dkey not in time_format_map:
            example_value = df[~df[dkey].isna()][dkey].iloc[0]
            dtype = df[dkey].dtype
            
            if not (
                pd.api.types.is_datetime64_any_dtype(df[dkey].dtype) or
                pd.api.types.is_datetime64_dtype(df[dkey].dtype)
            ):
                raise ValueError(
                    f"Date field '{dkey}' does not have a time format specified. Example value from {dkey}: \"{example_value}\""
                )
            
            s = df[dkey]
            if s.dt.tz is not None:
                s = s.dt.tz_localize(None)  # strips tz, keeps wall time
            # As strings 'YYYY-MM-DD'
            ymd = s.dt.strftime('%Y-%m-%d')
            df[dkey] = pd.to_datetime(ymd, format="%Y-%m-%d", errors="coerce")
            
    df = enrich_time(df, time_format_map, settings)

    dupes = entry.get("dupes", None)
    dupes_was_none = dupes is None
    if dupes is None:
        if is_geometry:
            dupes = "auto"
        else:
            dupes = {}
    if dupes == "auto":
        if is_geometry:
            cols = [col for col in df.columns.values if col != "geometry"]
            col = cols[0]
            dupes = {"subset": [col], "sort_by": [col, "asc"], "drop": True}
            if dupes_was_none:
                warnings.warn(
                    f"'dupes' not found for geo df '{entry_key}', defaulting to \"{col}\" as de-dedupe key. Set 'dupes:\"auto\" to remove this warning.'"
                )
        else:
            keys = ["key", "key2", "key3"]
            for key in keys:
                if key in df:
                    dupes = {"subset": [key], "sort_by": [key, "asc"], "drop": True}
                    break
    
    # If it's a sales dataframe, and we're not deduplicating on key_sale, something is probably wrong:
    if "key_sale" in df.columns.values:
        subset = dupes.get("subset", [])
        if dupes is not None and "key_sale" not in subset:
            warnings.warn(
                f"df '{entry_key}' contains field 'key_sale', indicating it is likely a sales dataframe. However, it's de-dupe subset is {subset}, which does not contain 'key_sale'. This could result in improper de-duplication of sales transactions."
            )

    df = _handle_duplicated_rows(df, dupes)
    
    if is_geometry:
        gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(df, geometry="geometry", crs=df.crs)
        
        pre_len = len(gdf)
        gdf = clean_geometry(gdf, ensure_polygon=True)
        post_len = len(gdf)
        
        perc_len = (pre_len-post_len)/pre_len
        if perc_len >= 0.25:
            warnings.warn(f"Dropped {perc_len:.0%} of rows from dataframe \"{entry_key}\" due to invalid/null geometry. If you don't care about geometry for this dataframe and want to retain all rows, then set '\"geometry\": false' in settings under this dataframe's 'data.load' entry")
        
        df = gdf
    
    drop = entry.get("drop", [])
    if len(drop) > 0:
        df = df.drop(columns=drop, errors="ignore")
    
    if verbose:
        print(f"--> rows = {len(df)}")
    
    return df


def _snoop_column_names(filename: str) -> list[str]:
    """Retrieve column names from a file without loading full data."""
    ext = str(filename).split(".")[-1]
    if ext == "parquet":
        parquet_file = pq.ParquetFile(filename)
        return parquet_file.schema.names
    elif ext == "csv":
        return pd.read_csv(filename, nrows=0).columns.tolist()
    raise ValueError(f'Unsupported file extension: "{ext}"')


def _get_sort_by(container: dict, default_value=None):
    sort_by = container.get("sort_by")
    if sort_by is None:
        return default_value
    # Handle main deduplications:
    if not isinstance(sort_by, list):
        raise ValueError(
            "sort_by must be a list of string pairs of the form [<field_name>, <asc|desc>]"
        )
    if len(sort_by) == 2:
        if isinstance(sort_by[0], str) and isinstance(sort_by[1], str):
            sort_by = [sort_by]
    else:
        for entry in sort_by:
            if not isinstance(entry, list):
                raise ValueError(
                    f"sort_by must be a list of string pairs, but found a non-list entry: {entry}"
                )
            elif len(entry) != 2:
                raise ValueError(
                    f"sort_by entry has {len(entry)} members: {entry}"
                )
            elif not isinstance(entry[0], str) or not isinstance(entry[1], str):
                raise ValueError(
                    f"sort_by entry has non-string members: {entry}"
                )
    bys = [x[0] for x in sort_by]
    ascendings = [x[1] == "asc" for x in sort_by]
    return {
        "bys": bys,
        "ascendings": ascendings
    }


def _handle_duplicated_rows(
    df_in: pd.DataFrame, dupes: str | dict, verbose: bool = False
) -> pd.DataFrame:
    """Handle duplicated rows in a DataFrame based on specified rules."""
    if dupes == "allow":
        return df_in
    subset = dupes.get("subset", "key")
    if not isinstance(subset, list):
        subset = [subset]
    for key in subset:
        if key not in df_in:
            return df_in
    do_drop = dupes.get("drop", True)
    agg: dict | None = dupes.get("agg", None)
    num_dupes = df_in.duplicated(subset=subset).sum()
    orig_len = len(df_in)
    if num_dupes > 0:
        sort_by = _get_sort_by(dupes, {"bys":["key"],"asc":["asc"]})
        bys = sort_by["bys"]
        ascendings = sort_by["ascendings"]
        
        df = df_in.copy()
        df = df.sort_values(by=bys, ascending=ascendings)
        if do_drop:
            if do_drop == "all":
                df = df.drop_duplicates(subset=subset, keep=False)
            else:
                df = df.drop_duplicates(subset=subset, keep="first")
            final_len = len(df)
            if verbose:
                print(
                    f"Dropped {orig_len - final_len} duplicate rows based on '{subset}'"
                )
        df_deduped = df.reset_index(drop=True)
        
        if agg is not None:
            
            df_agg : pd.DataFrame = None
            
            # Handle aggregations:
            for agg_key in agg:
                agg_entry = agg[agg_key]
                field = agg_entry.get("field")
                op = agg_entry.get("op")
                
                # Custom sort information per aggregation in case the user is relying on "first"/"last" agg
                agg_sort_by = _get_sort_by(agg_entry, sort_by)
                
                agg_bys = agg_sort_by["bys"]
                agg_ascendings = agg_sort_by["ascendings"]
                
                if field not in df_in:
                    raise ValueError(f"Field '{field}' not found in DataFrame.")
                
                df_sorted = df_in.sort_values(by=agg_bys, ascending=agg_ascendings)
                
                df_result = (
                    df_sorted.groupby(subset)
                    .agg({field: op})
                    .reset_index()
                    .rename(columns={field: agg_key})
                )
                if df_agg is None:
                    df_agg = df_result
                else:
                    df_agg = df_agg.merge(df_result, on=subset, how="outer")
            
            cols_in_common = [col for col in df_deduped if col in df_agg and col not in subset]
            if len(cols_in_common) > 0:
                df_deduped = df_deduped.drop(columns=cols_in_common)
                warnings.warn(f"{len(cols_in_common)} aggregated columns have names that conflict with existing base columns. The base columns have been dropped and overwritten by the aggregated columns.")
            
            # Merge the aggregated results back into the deduped base
            df_result = df_deduped.merge(df_agg, on=subset, how="left")
        else:
            df_result = df_deduped
    else:
        df_result = df_in
    
    the_key = "key_sale" if "key_sale" in df_result else "key"
    
    if the_key in df_result:
        df_final = df_result.sort_values(by=the_key, ascending=True)
    else:
        df_final = df_result
    
    df_final = df_final.reset_index(drop=True)
    
    return df_final


def _merge_dict_of_dfs(
    dataframes: dict[str, pd.DataFrame],
    merge_list: list,
    settings: dict,
    required_key="key",
) -> pd.DataFrame:
    """Merge multiple DataFrames according to merge instructions."""
    merges = []
    s_reconcile = settings.get("data", {}).get("process", {}).get("reconcile", {})

    # Generate instructions for merging, but don't merge just yet
    for entry in merge_list:
        df_id = None
        how = "left"
        on = required_key
        left_on = None
        right_on = None

        payload = {}

        if isinstance(entry, str):
            if entry not in dataframes:
                raise ValueError(f"Merge key '{entry}' not found in dataframes.")
            df_id = entry
        elif isinstance(entry, dict):
            df_id = entry.get("id", None)
            how = entry.get("how", how)
            on = entry.get("on", on)
            left_on = entry.get("left_on", left_on)
            right_on = entry.get("right_on", right_on)
            for key in entry:
                if key not in ["id", "df", "how", "on", "left_on", "right_on"]:
                    payload[key] = entry[key]
        if df_id is None:
            raise ValueError(
                "Merge entry must be either a string or a dictionary with an 'id' key."
            )
        if df_id not in dataframes:
            raise ValueError(f"Merge key '{df_id}' not found in dataframes.")

        payload["id"] = df_id
        payload["df"] = dataframes[df_id]
        payload["how"] = how
        payload["on"] = on
        payload["left_on"] = left_on
        payload["right_on"] = right_on

        merges.append(payload)

    df_merged: pd.DataFrame | None = None
    all_cols = []
    conflicts = {}
    all_suffixes = []

    # Generate suffixes and note conflicts, which we'll resolve further down
    for merge in merges:
        df_id = merge["id"]
        df = merge["df"]
        on = merge["on"]
        how = merge["how"]
        left_on = merge["left_on"]
        right_on = merge["right_on"]
        merge_keys = []
        if on is not None:
            merge_keys = [on] if not isinstance(on, list) else on
        if right_on is not None:
            merge_keys = right_on if isinstance(right_on, list) else [right_on]
        if how == "lat_long":
            merge_keys = ["latitude", "longitude"]

        suffixes = {}
        for col in df.columns.values:
            if col in merge_keys:
                continue
            if col not in all_cols:
                all_cols.append(col)
            else:
                if how != "append":
                    suffixed = f"{col}_{merge['id']}"
                    suffixes[col] = suffixed
                    if col not in conflicts:
                        conflicts[col] = []
                    conflicts[col].append(suffixed)
                    all_suffixes.append(suffixed)
        df = df.rename(columns=suffixes)
        merge["df"] = df

    # Perform the actual merges
    for merge in merges:
        _id = merge["id"]
        df = merge.get("df", None)
        how = merge.get("how", "left")
        on = merge.get("on", required_key)
        left_on = merge.get("left_on", None)
        right_on = merge.get("right_on", None)
        dupes = merge.get("dupes", None)

        if df_merged is None:
            df_merged = df
        elif how == "append":
            df_merged = pd.concat([df_merged, df], ignore_index=True)
        elif how == "lat_long":
            if not (
                isinstance(df_merged, gpd.GeoDataFrame) and "geometry" in df_merged
            ):
                raise ValueError(
                    "Cannot perform lat_long merge against a non-geodataframe. Make sure there is a geodataframe earlier in the merge chain."
                )
            if "latitude" not in df.columns and "longitude" not in df.columns:
                raise ValueError(
                    "Neither 'latitude' nor 'longitude' fields found in dataframe being merged with 'lat_long'"
                )
            if "latitude" not in df.columns:
                raise ValueError(
                    "No 'latitude' field found in dataframe being merged with 'lat_long'"
                )
            if "longitude" not in df.columns:
                raise ValueError(
                    "No 'longitude' field found in dataframe being merged with 'lat_long'"
                )
            # use geolocation to get the right keys
            parcel_id_field = on if on is not None else "key"
            df_with_key = geolocate_point_to_polygon(
                df_merged,
                df,
                lat_field="latitude",
                lon_field="longitude",
                parcel_id_field=parcel_id_field,
            )

            # de-duplicate
            dupe_rows = df_with_key[
                df_with_key.duplicated(subset=[parcel_id_field], keep=False)
            ]
            if len(dupe_rows) > 0:
                if dupes is None:
                    raise ValueError(
                        f"Found {len(dupe_rows)} duplicates in geolocation merge '{_id}' on field '{parcel_id_field}'. But, you have no 'dupes' policy to deal with them. If you're okay with duplicates (such as in a sales dataset), set dupes='allow' in the merge instructions."
                    )
                df_with_key = _handle_duplicated_rows(df_with_key, dupes, verbose=True)

            # merge the dataframes the conventional way
            df_merged = pd.merge(
                df_merged,
                df_with_key,
                how="left",
                on=parcel_id_field,
                suffixes=("", f"_{_id}"),
            )
        else:
            if left_on is not None and right_on is not None:
                # Verify that both columns exist before attempting merge
                if isinstance(left_on, list):
                    for col in left_on:
                        if col not in df_merged.columns:
                            raise ValueError(
                                f"Left merge column '{col}' not found in left dataframe. Available columns: {df_merged.columns.tolist()}"
                            )
                else:
                    if left_on not in df_merged.columns:
                        raise ValueError(
                            f"Left merge column '{left_on}' not found in left dataframe. Available columns: {df_merged.columns.tolist()}"
                        )

                if isinstance(right_on, list):
                    for col in right_on:
                        if col not in df.columns:
                            raise ValueError(
                                f"Right merge column '{col}' not found in right dataframe. Available columns: {df.columns.tolist()}"
                            )
                else:
                    if right_on not in df.columns:
                        raise ValueError(
                            f"Right merge column '{right_on}' not found in right dataframe. Available columns: {df.columns.tolist()}"
                        )

                df_merged = pd.merge(
                    df_merged,
                    df,
                    how=how,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=("", f"_{_id}"),
                )
            else:
                if on not in df_merged.columns:
                    raise ValueError(
                        f"Merge column '{on}' not found in left dataframe. Available columns: {df_merged.columns.tolist()}"
                    )
                if on not in df.columns:
                    raise ValueError(
                        f"Merge column '{on}' not found in right dataframe. Available columns: {df.columns.tolist()}"
                    )
                df_merged = pd.merge(
                    df_merged, df, how=how, on=on, suffixes=("", f"_{_id}")
                )

        # General case de-duplication
        if on in df_merged:
            dupe_rows = df_merged[df_merged.duplicated(subset=[on], keep=False)]
            if len(dupe_rows) > 0:
                if dupes is None:
                    raise ValueError(
                        f"Found {len(dupe_rows)} duplicates in geolocation merge id='{_id}' how='{how}' on='{on}'. But, you have no 'dupes' policy to deal with them. If you're okay with duplicates (such as in a sales dataset), set dupes='allow' in the merge instructions."
                    )
                df_merged = _handle_duplicated_rows(df_merged, dupes, verbose=True)

    # Reconcile conflicts
    for base_field in s_reconcile:
        df_ids = s_reconcile[base_field]
        if base_field not in all_cols:
            raise ValueError(
                f"Reconciliation field '{base_field}' not found in any of the dataframes."
            )
        child_fields = [f"{base_field}_{df_id}" for df_id in df_ids]
        if base_field in conflicts:
            old_child_fields = conflicts[base_field]
            old_child_fields = [
                field for field in old_child_fields if field not in child_fields
            ]
            child_fields = child_fields + old_child_fields
        conflicts[base_field] = child_fields
    for base_field in conflicts:
        if base_field not in df_merged:
            warnings.warn(
                f"Reconciliation field '{base_field}' not found in merged dataframe."
            )
            continue
        child_fields = conflicts[base_field]
        if len(child_fields) > 1:
            # TODO: remove this when this becomes default pandas behavior
            old_value = pd.get_option("future.no_silent_downcasting")
            pd.set_option("future.no_silent_downcasting", True)

            df_merged[base_field] = df_merged[base_field].fillna(
                df_merged[child_fields[0]]
            )
            for i in range(1, len(child_fields)):
                df_merged[base_field] = df_merged[base_field].fillna(
                    df_merged[child_fields[i]]
                )
            df_merged = df_merged.drop(columns=child_fields)

            # TODO: remove this when this becomes default pandas behavior
            pd.set_option("future.no_silent_downcasting", old_value)

    # Remove columns used as INGREDIENTS in calculations, but which the user never intends to load directly
    calc_cols = _get_calc_cols(settings, exclude_loaded_fields=True)
    for col in df_merged.columns.values:
        if col in calc_cols:
            df_merged = df_merged.drop(columns=[col])

    # Final checks
    if required_key is not None and required_key not in df_merged:
        raise ValueError(
            f"No '{required_key}' field found in merged dataframe. This field is required. Keys found = {df_merged.columns.values}"
        )
    len_old = len(df_merged)
    df_merged = df_merged.dropna(subset=[required_key])
    len_new = len(df_merged)
    if len_new < len_old:
        warnings.warn(f"Dropped {len_old - len_new} rows due to missing primary key.")

    all_suffixes = [col for col in all_suffixes if col in df_merged]
    df_merged = df_merged.drop(columns=all_suffixes)

    # ensure a clean index:
    df_merged = df_merged.reset_index(drop=True)

    fields_bool = get_fields_boolean(settings)
    fields_num = get_fields_numeric(settings, include_boolean=False)
    fields_cat = get_fields_categorical(settings, include_boolean=False)

    # enforce types post-merge:
    for col in df_merged.columns:
        if col in fields_bool:
            df_merged = _boolify_column_in_df(df_merged, col, "na_false")
        elif col in fields_num:
            df_merged[col] = df_merged[col].astype("Float64")
        elif col in fields_cat:
            if "date" not in col:
                df_merged[col] = df_merged[col].astype("string")

    # drop null keys
    df_merged = df_merged.dropna(subset=[required_key])

    return df_merged


def _write_canonical_splits(sup: SalesUniversePair, settings: dict, verbose: bool=False):
    """Write canonical split keys for sales data to disk."""
    df_sales_in = sup.sales
    df_univ = sup.universe

    if verbose:
        print(f"Write canonical splits...")
        print(f"Sales in = {len(df_sales_in)}")
    
    df_sales = _get_sales(df_sales_in, settings, df_univ=df_univ)
    model_groups = get_model_group_ids(settings, df_sales)
    
    if verbose:
        print(f"Get sales= {len(df_sales)}")
    
    instructions = settings.get("modeling", {}).get("instructions", {})
    test_train_frac = instructions.get("test_train_frac", 0.8)
    random_seed = instructions.get("random_seed", 1337)
    for model_group in model_groups:
        _do_write_canonical_split(
            model_group, df_sales, settings, test_train_frac, random_seed, verbose
        )


def _perform_canonical_split(
    model_group: str,
    df_sales_in: pd.DataFrame,
    settings: dict,
    test_train_fraction: float = 0.8,
    random_seed: int = 1337,
    verbose : bool = False
):
    """Perform a canonical split of the sales DataFrame for a given model group into test
    and training sets.
    """

    # High level goals
    # 1. Split the sales data into a TRAINING and a TEST set
    # 2. Maintain uniformity of sample size across VACANT and IMPROVED sales
    # 3. TEST set details:
    #   - sales no older than the lookback period (typically one year, user-configurable)
    #   - favor post-valuation-date sales if available
    #   - aim to be 20% of total sales (user-configurable). Exception: if we have more post-valuation sales, use them all for test
    #   - if not enough sales in the post-valuation period to hit 20%, use lookback period sales to fill
    #   - if not enough sales in the lookback period to hit 20%, use all pre-valuation sales to fill
    #   - sample randomly whenever possible
    # 4. TRAINING set details:
    #   - whatever is not in the TEST set
    #   - never use post-valuation-date sales, even if there's some left over
    #   - aim to be 80% of total sales
    
    if verbose:
        print("")
        print(f"Making canonical test/train split for model group {model_group}...")
    
    rs = settings.get("analysis", {}).get("ratio_study", {})
    look_back_years = rs.get("look_back_years", 1)
    
    md = settings.get("modeling", {}).get("metadata", {})
    use_sales_from = md.get("use_sales_from", None)
    
    val_date = get_valuation_date(settings)

    # Look back N years BEFORE the valuation date
    look_back_date = val_date - pd.DateOffset(years=look_back_years)

    # Get that time in terms of days (positive = days before the valuation date)
    look_back_days = (val_date - look_back_date).days

    # Get our sales dataframe for this model group and split it into vacant and improved sales
    df = df_sales_in[df_sales_in["model_group"].eq(model_group)].copy()
    
    # Only use sales on or after the use_sales_from year
    if use_sales_from is not None:
        df = df[df["sale_year"].ge(use_sales_from)]
    
    df_v = get_vacant_sales(df, settings)
    df_i = df.drop(df_v.index)
    
    df_check = df[df["vacant_sale"].eq(True)]

    df_v_pre_val = df_v[
        df_v["sale_age_days"].ge(0)
    ]  # Positive sale_age_days indicates days BEFORE the valuation date
    df_i_pre_val = df_i[df_i["sale_age_days"].ge(0)]
    
    # Find sales that occurred on or after the look_back_date (i.e., within 1 year of the valuation date, but not after it)
    df_v_look_back = df_v[
        df_v["sale_age_days"].le(look_back_days) & (df_v["sale_age_days"].ge(0))
    ]
    df_i_look_back = df_i[
        df_i["sale_age_days"].le(look_back_days) & (df_i["sale_age_days"].ge(0))
    ]

    # Find sales that occurred AFTER the valuation date
    # These will also be candidates for the holdout set
    df_v_post_val = df_v[
        df_v["sale_age_days"].lt(0)
    ]  # Negative sale_age_days indicates days AFTER the valuation date
    df_i_post_val = df_i[df_i["sale_age_days"].lt(0)]
    
    count_v = len(df_v)
    count_i = len(df_i)
    count_pre_val_v = len(df_v_pre_val)
    count_pre_val_i = len(df_i_pre_val)
    count_post_val_v = len(df_v_post_val)
    count_post_val_i = len(df_i_post_val)
    count_look_back_v = len(df_v_look_back)
    count_look_back_i = len(df_i_look_back)

    if verbose:
        print("All:")
        print(f"--> Vacant  : {count_v}")
        print(f"--> Improved: {count_i}")
        print("Pre-valuation: ")
        print(f"--> Vacant  : {count_pre_val_v}")
        print(f"--> Improved: {count_pre_val_i}") 
        print(f"Post-valuation: ")
        print(f"--> Vacant    : {count_post_val_v}")
        print(f"--> Improved  : {count_post_val_v}")
        print(f"In look back period: ")
        print(f"--> Vacant    : {count_look_back_v}")
        print(f"--> Improved  : {count_look_back_i}")
        print("")

    test_share = 1.0 - test_train_fraction

    # This is how many test sales we need
    test_set_count = np.ceil(len(df) * test_share).astype(int)
    if len(df) <= 30:
        test_set_count_v = np.ceil(len(df_v) * test_share).astype(int)
    else:
        test_set_count_v = max(15, np.ceil(len(df_v) * test_share).astype(int))
    test_set_count_i = test_set_count - test_set_count_v
    
    # Sales in general
    count_sales_all = len(df)
    count_sales_all_v = len(df_v)
    count_sales_all_i = len(df_i)

    # Sales after the val date
    count_post_val = len(df_v_post_val) + len(df_i_post_val)

    # Sales within the look back period
    count_look_back = len(df_v_look_back) + len(df_i_look_back)
    
    np.random.seed(random_seed)

    df_v_test: pd.DataFrame | None = None
    df_i_test: pd.DataFrame | None = None
    df_v_train: pd.DataFrame | None = None
    df_i_train: pd.DataFrame | None = None

    if count_post_val >= test_set_count:
        # The post-valuation set is GREATER than or EQUAL to the test set
        # We will use the whole post-valuation set exclusively for the test set, and the pre-valuation set exclusively for the training set
        # (We do not train on post-valuation sales, to maintain the integrity of post-valuation data)

        # The test set is exactly equal to the post-valuation set:
        df_v_test = df_v_post_val
        df_i_test = df_i_post_val

        # The train set is exactly equal to the pre-valuation set:
        df_v_train = df_v_pre_val
        df_i_train = df_i_pre_val

    elif count_post_val < test_set_count:
        # The post-valuation set is SMALLER than the test set
        # We consume the whole post-valuation set for the test set

        df_v_test = df_v_post_val
        df_i_test = df_i_post_val

        # How many more sales do we need to fill the test set?
        remaining_test_count = test_set_count - count_post_val
        remaining_test_frac = None

        if count_look_back > 0:
            # Express that as a fraction of the lookback sales if they exist
            remaining_test_frac = remaining_test_count / count_look_back

        if count_look_back > 0 and remaining_test_frac <= 1.0:
            # We can sample the lookback sales alone to fill the test set
            n_v = max(test_set_count_v, np.floor(len(df_v_look_back) * remaining_test_frac).astype(int))
            n_i = max(test_set_count_i, np.floor(len(df_i_look_back) * remaining_test_frac).astype(int))
            
            remaining_test_count_v = test_set_count_v - len(df_v_test)
            remaining_test_count_i = test_set_count_i - len(df_i_test)

            diff_v = remaining_test_count_v - n_v
            diff_i = remaining_test_count_i - n_i

            while diff_v > 0:
                old_diff = diff_v
                if len(df_v_look_back) > n_v:
                    n_v += 1
                    diff_v -= 1
                if diff_v == old_diff:
                    break
    
            while diff_i > 0:
                old_diff = diff_i
                if len(df_i_look_back) > n_i:
                    n_i += 1
                    diff_i -= 1
                if diff_i == old_dif:
                    break
            
            if n_v > 0 and len(df_v_look_back) > 0:
                n_v = min(n_v, len(df_v_look_back))
                _df_v_test = df_v_look_back.sample(n=n_v, random_state=random_seed)
                df_v_test = pd.concat([df_v_test, _df_v_test])
            if n_i > 0 and len(df_i_look_back) > 0:
                n_i = min(n_i, len(df_i_look_back))
                _df_i_test = df_i_look_back.sample(n=n_i, random_state=random_seed)
                df_i_test = pd.concat([df_i_test, _df_i_test])
        else:
            # We need to sample ALL the lookback sales to start with...
            df_v_test = pd.concat([df_v_test, df_v_look_back])
            df_i_test = pd.concat([df_i_test, df_i_look_back])
            
            if len(df_v_test) > test_set_count_v:
                # in case we already have enough vacant sales in the test set
                df_v_test = df_v_test.sample(n=test_set_count_v, random_state=random_seed)
           
            # ...and then we need to sample the pre-valuation sales to fill the remainder of the test set
            remaining_test_count_v = test_set_count_v - len(df_v_test)
            remaining_test_count_i = test_set_count_i - len(df_i_test)

            # Figure out how many sales we haven't touched yet
            untouched_sales_count_v = count_sales_all_v - len(df_v_test)
            untouched_sales_count_i = count_sales_all_i - len(df_i_test)

            df_v_untouched = df_v[~df_v["key_sale"].isin(df_v_test["key_sale"])]
            df_i_untouched = df_i[~df_i["key_sale"].isin(df_i_test["key_sale"])]

            # Express how many more we need as a fraction of the untouched sales
            remaining_test_frac_i = 0 if untouched_sales_count_i == 0 else remaining_test_count_i / untouched_sales_count_i
            remaining_test_frac_v = 0 if untouched_sales_count_v == 0 else remaining_test_count_v / untouched_sales_count_v

            n_v = np.floor(len(df_v_untouched) * remaining_test_frac_v).astype(int)
            n_i = np.floor(len(df_i_untouched) * remaining_test_frac_i).astype(int)
            
            diff_v = remaining_test_count_v - n_v
            diff_i = remaining_test_count_i - n_i
            
            while diff_v > 0:
                old_diff = diff_v
                if len(df_v_look_back) > n_v:
                    n_v += 1
                    diff_v -= 1
                if diff_v == old_diff:
                    break
    
            while diff_i > 0:
                old_diff = diff_i
                if len(df_i_look_back) > n_i:
                    n_i += 1
                    diff_i -= 1
                if diff_i == old_diff:
                    break

            n_v = min(n_v, len(df_v_untouched))
            n_i = min(n_i, len(df_i_untouched))

            # Sample the untouched sales to fill the test set
            # Then, add the untouched sales to the test set -- we're finally done
            if n_v > 0:
                _df_v_test = df_v_untouched.sample(n=n_v, random_state=random_seed)
                df_v_test = pd.concat([df_v_test, _df_v_test])
            
            if n_i > 0:
                _df_i_test = df_i_untouched.sample(n=n_i, random_state=random_seed)
                df_i_test = pd.concat([df_i_test, _df_i_test])
            
        # Training set is whatever is not in the test set
        df_v_train = df_v_pre_val[~df_v_pre_val["key_sale"].isin(df_v_test["key_sale"])]
        df_i_train = df_i_pre_val[~df_i_pre_val["key_sale"].isin(df_i_test["key_sale"])]

    df_test = pd.concat([df_v_test, df_i_test]).reset_index(drop=True)
    df_train = pd.concat([df_v_train, df_i_train]).reset_index(drop=True)
    
    df_v_test_look_back = df_v_test[
        df_v_test["sale_age_days"].le(look_back_days) & 
        df_v_test["sale_age_days"].ge(0)
    ]
    
    df_i_test_look_back = df_i_test[
        df_i_test["sale_age_days"].le(look_back_days) & 
        df_i_test["sale_age_days"].ge(0)
    ]
    
    df_v_train_look_back = df_v_train[
        df_v_train["sale_age_days"].le(look_back_days) & 
        df_v_train["sale_age_days"].ge(0)
    ]
    
    df_i_train_look_back = df_i_train[
        df_i_train["sale_age_days"].le(look_back_days) & 
        df_i_train["sale_age_days"].ge(0)
    ]     
    
    if verbose:
        print(f"--> Test set       : {len(df_test)}")
        print(f"------> Vacant     : {len(df_v_test)}")
        print(f"------> Improved   : {len(df_i_test)}")
        print(f"----> In lookback")
        print(f"------> Vacant     : {len(df_v_test_look_back)}")
        print(f"------> Improved   : {len(df_i_test_look_back)}")
        
        print(f"--> Train set      : {len(df_train)}")
        print(f"------> Vacant     : {len(df_v_train)}")
        print(f"------> Improved   : {len(df_i_train)}")
        print(f"----> In lookback")
        print(f"-------> Vacant    : {len(df_v_train_look_back)}")
        print(f"-------> Improved  : {len(df_i_train_look_back)}")
    
        keys_v_test = len(df_v_test["key_sale"].unique())
        keys_i_test = len(df_i_test["key_sale"].unique())
        keys_v_train = len(df_v_train["key_sale"].unique())
        keys_i_train = len(df_i_train["key_sale"].unique())
        
        print(f"")
        print(f"Unique keys:")
        print(f"Test set:")
        print(f"--> Vacant  : {keys_v_test}")
        print(f"--> Improved: {keys_i_test}")
        print(f"Train set:")
        print(f"--> Vacant  : {keys_v_train}")
        print(f"--> Improved: {keys_i_train}")
        
    
    return df_test, df_train


def _do_write_canonical_split(
    model_group: str,
    df_sales_in: pd.DataFrame,
    settings: dict,
    test_train_fraction: float = 0.8,
    random_seed: int = 1337,
    verbose: bool = False
):
    """Write the canonical split keys (train and test) for a given model group to disk.
    Also performs outlier detection on training data if enabled in settings.
    """
    # Get initial split
    df_test, df_train = _perform_canonical_split(
        model_group, df_sales_in, settings, test_train_fraction, random_seed, verbose
    )

    # Get initial keys
    train_keys = df_train["key_sale"].values
    test_keys = df_test["key_sale"].values

    # Create output directory and save keys
    outpath = f"out/models/{model_group}/_data"
    os.makedirs(outpath, exist_ok=True)
    pd.DataFrame({"key_sale": train_keys}).to_csv(
        f"{outpath}/train_keys.csv", index=False
    )
    pd.DataFrame({"key_sale": test_keys}).to_csv(
        f"{outpath}/test_keys.csv", index=False
    )



def _read_split_keys(model_group: str):
    """Read the train and test split keys for a model group from disk."""
    path = f"out/models/{model_group}/_data"
    train_path = f"{path}/train_keys.csv"
    test_path = f"{path}/test_keys.csv"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError("No split keys found.")
    train_keys = pd.read_csv(train_path)["key_sale"].astype(str).values
    test_keys = pd.read_csv(test_path)["key_sale"].astype(str).values
    return test_keys, train_keys


def _tag_model_groups_sup(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
):
    """Tag model groups for both sales and universe DataFrames based on settings.

    Hydrates sales data and assigns model groups to parcels and sales by applying filters
    from settings. Also prints summary statistics if verbose is True.
    """
    df_sales = sup["sales"].copy()
    df_univ = sup["universe"].copy()
    df_sales_hydrated = get_hydrated_sales_from_sup(sup)
    mg = settings.get("modeling", {}).get("model_groups", {})

    print(f"Len univ before = {len(df_univ)}")
    print(f"Len sales before = {len(df_sales)} after = {len(df_sales_hydrated)}")
    print(f"Overall")
    print(f"--> {len(df_univ):,} parcels")
    print(f"--> {len(df_sales):,} sales")

    df_univ["model_group"] = None
    df_sales_hydrated["model_group"] = None

    for mg_id in mg:
        # only apply model groups to parcels that don't already have one
        idx_no_model_group = df_univ["model_group"].isnull()
        entry = mg[mg_id]
        _filter = entry.get("filter", [])

        if len(_filter) == 0:
            raise ValueError(
                "No 'filter' entry found for model group '{mg_id}'. Check your spelling!"
            )

        univ_index = resolve_filter(df_univ, _filter)
        df_univ.loc[idx_no_model_group & univ_index, "model_group"] = mg_id

        idx_no_model_group = df_sales_hydrated["model_group"].isnull()
        sales_index = resolve_filter(df_sales_hydrated, _filter)
        df_sales_hydrated.loc[idx_no_model_group & sales_index, "model_group"] = mg_id

    os.makedirs("out/look", exist_ok=True)

    if not isinstance(df_univ, gpd.GeoDataFrame):
        df_univ = gpd.GeoDataFrame(df_univ, geometry="geometry")
    df_univ.to_parquet("out/look/tag-univ-0.parquet")

    if not isinstance(df_univ, gpd.GeoDataFrame):
        df_univ = gpd.GeoDataFrame(df_univ, geometry="geometry")
    df_univ.to_parquet("out/look/tag-univ-0.parquet", engine="pyarrow")
    old_model_group = df_univ[["key", "model_group"]]

    for mg_id in mg:
        entry = mg[mg_id]
        print(f"Assigning model group {mg_id}...")
        common_area = entry.get("common_area", False)
        print("common_area --> ", common_area)
        if not common_area:
            continue
        print(f"Assigning common areas for model group {mg_id}...")
        common_area_filters: list | None = None
        if isinstance(common_area, list):
            common_area_filters = common_area
        print(f"common area filters = {common_area_filters}")
        df_univ = _assign_modal_model_group_to_common_area(
            df_univ, mg_id, common_area_filters
        )

    df_univ.to_parquet("out/look/tag-univ-1.parquet", engine="pyarrow")
    index_changed = ~old_model_group["model_group"].eq(df_univ["model_group"])
    rows_changed = df_univ[index_changed]
    print(f" --> {len(rows_changed)} parcels had their model group changed.")

    # TODO: fix this
    # Update sales for any rows that changed due to common area assignment
    # df_sales = combine_dfs(df_sales, rows_changed, df2_stomps=True, index="key")

    for mg_id in mg:
        entry = mg[mg_id]
        name = entry.get("name", mg_id)
        _filter = entry.get("filter", [])
        univ_index = resolve_filter(df_univ, _filter)
        sales_index = resolve_filter(df_sales_hydrated, _filter)
        if verbose:
            valid_sales_index = sales_index & df_sales_hydrated["valid_sale"].eq(True)
            improved_sales_index = (
                sales_index
                & valid_sales_index
                & ~df_sales_hydrated["vacant_sale"].eq(True)
            )
            vacant_sales_index = (
                sales_index
                & valid_sales_index
                & df_sales_hydrated["vacant_sale"].eq(True)
            )
            print(f"{name}")
            print(f"--> {univ_index.sum():,} parcels")
            print(f"--> {valid_sales_index.sum():,} sales")
            print(f"----> {improved_sales_index.sum():,} improved sales")
            print(f"----> {vacant_sales_index.sum():,} vacant sales")
    df_univ.loc[df_univ["model_group"].isna(), "model_group"] = "UNKNOWN"
    sup.set("universe", df_univ)
    sup.set("sales", df_sales)
    return sup


def _assign_modal_model_group_to_common_area(
    df_univ_in: gpd.GeoDataFrame,
    model_group_id: str,
    common_area_filters: list | None = None,
) -> gpd.GeoDataFrame:
    """Assign the modal model_group of parcels inside an enveloping "COMMON AREA" parcel
    to that parcel.
    """
    df_univ = df_univ_in.copy()

    # Ensure geometry column is set
    if df_univ.geometry.name is None:
        raise ValueError("GeoDataFrame must have a geometry column.")

    # Reduce df_univ to ONLY those parcels that have holes in them:
    df = _identify_parcels_with_holes(df_univ)

    print(f" {len(df)} parcels with holes found.")
    df.to_parquet("out/look/common_area-0-holes.parquet", engine="pyarrow")
    df["has_holes"] = True

    if common_area_filters is not None:
        df_extra = select_filter(df_univ, common_area_filters).copy()
        df_extra["is_common_area"] = True
        print(f" {len(df_extra)} extra parcels found.")
        df = pd.concat([df, df_extra], ignore_index=True)
        # drop duplicate keys:
        df = df.drop_duplicates(subset="key")

    print(f" {len(df)} potential COMMON AREA parcels found.")
    df.to_parquet("out/look/common_area-1-common_area.parquet", engine="pyarrow")

    print(
        f"Assigning modal model_group to {len(df)}/{len(df_univ_in)} potential parcels..."
    )

    df["modal_tagged"] = None

    # Iterate over COMMON AREA parcels
    for idx, row in df.iterrows():
        # Get the envelope of the COMMON AREA parcel
        common_area_geom = row.geometry
        common_area_gs = gpd.GeoSeries([common_area_geom], crs=df.crs)
        common_area_envelope_geom = common_area_geom.envelope
        common_area_envelope_gs = gpd.GeoSeries([common_area_envelope_geom], crs=df.crs)

        geom = common_area_geom.buffer(0)
        if geom.geom_type == "Polygon":
            outer_polygon = Polygon(geom.exterior)
        elif geom.geom_type == "MultiPolygon":
            outer_polygons = [Polygon(poly.exterior) for poly in geom.geoms]
            outer_polygon = unary_union(outer_polygons)
        else:
            raise ValueError("Geometry must be a Polygon or MultiPolygon")
        # outer_polygon_gs = gpd.GeoSeries([outer_polygon], crs=df.crs)

        # Find parcels wholly inside the COMMON AREA envelope
        inside_parcels = df_univ_in[
            df_univ_in.geometry.within(common_area_envelope_geom)
        ].copy()

        # buffer 0 on inside parcel geometry
        inside_parcels["geometry"] = inside_parcels["geometry"].apply(
            lambda g: g.buffer(0)
        )

        count1 = len(inside_parcels)

        # Exclude the COMMON AREA parcel itself (if it is in df_univ)
        inside_parcels = inside_parcels[
            ~inside_parcels.geometry.apply(lambda g: g.equals(common_area_geom))
        ]
        count2 = len(inside_parcels)

        # Optionally use a tiny negative buffer to avoid boundary issues

        # Exclude parcels that are not wholly inside the COMMON AREA parcel (not just the envelope bounding box):
        if isinstance(outer_polygon, np.ndarray):
            if outer_polygon.size == 1:
                outer_polygon = outer_polygon[0]
            else:
                # If there are multiple elements, combine them into one geometry
                outer_polygon = unary_union(list(outer_polygon))
            print("outer_polygon type:", type(outer_polygon))
        inside_parcels = inside_parcels[
            inside_parcels.geometry.centroid.within(outer_polygon)
        ]
        count3 = len(inside_parcels)

        print(
            f" {idx} --> {count1} parcels inside the envelope, {count2} after excluding the COMMON AREA, {count3} after excluding those not wholly inside the COMMON AREA"
        )

        # If it's empty, continue:
        if inside_parcels.empty:
            continue

        # Check that at least one of the inside_parcels matches the target model_group_id, otherwise continue:
        if not inside_parcels["model_group"].eq(model_group_id).any():
            continue

        # Determine the modal model_group value
        modal_model_group = inside_parcels["model_group"].value_counts().index[0]
        if modal_model_group is not None and modal_model_group != "":
            print(
                f" {idx} --> modal model group = {modal_model_group} for {len(inside_parcels)} inside parcels"
            )
            # Apply the modal model_group to the COMMON AREA parcel
            df.at[idx, "model_group"] = modal_model_group
            df.at[idx, "modal_tagged"] = True
        else:
            print(
                f" {idx} --> XXX modal model group is {modal_model_group} for {len(inside_parcels)} inside parcels"
            )

    df.to_parquet("out/look/common_area-2-tagged.parquet", engine="pyarrow")
    df_return = df_univ_in.copy()
    # Update and return df_univ
    df_return = combine_dfs(
        df_return, df[["key", "model_group"]], df2_stomps=True, index="key"
    )
    df_return.to_parquet("out/look/common_area-3-return.parquet", engine="pyarrow")
    return df_return


def _clean_series(series: pd.Series) -> pd.Series:
    """Clean a pandas Series by converting to lowercase, replacing spaces with
    underscores, and removing special characters.
    """
    # Convert to string if not already
    series = series.astype(str)

    # Convert to lowercase
    series = series.str.lower()

    # Replace spaces and special characters with underscores
    series = series.str.replace(r"[^a-z0-9]", "_", regex=True)

    # Replace multiple underscores with single underscore
    series = series.str.replace(r"_+", "_", regex=True)

    # Remove leading/trailing underscores
    series = series.str.strip("_")

    return series


def _process_permits_univ(
    df_in: pd.DataFrame,
    df_permits: pd.DataFrame,
    s_permits: dict,
    settings: dict,
    verbose: bool = False,
):
    calc_effective_age = s_permits.get("calc_effective_age", False)

    # We might have multiple permits per key. We have multiple questions to answer:

    # 1. Do we have a demolition permit? When was the demolition date?
    df_demos = df_permits[df_permits["is_teardown"].eq(True)][["key", "date"]].copy()

    # 2. Do we have a renovation permit? When was the renovation date?
    df_renos = df_permits[df_permits["is_renovation"].eq(True)][["key", "date"]].copy()

    # ==========================================================================================#
    #                              Process teardown universe                                   #
    # ==========================================================================================#

    # We want to know -- was the most recent permit for this parcel a demolition permit?

    df_u = df_in[["key"]].copy()

    df_permits = df_permits.rename(columns={"date": "permit_date"})
    df_u = df_u.merge(df_permits, on="key", how="left")
    df_u["sale_date"] = get_valuation_date(settings)
    # Ignore permits that happened AFTER the valuation date
    df_u.loc[df_u["permit_date"].gt(df_u["sale_date"]), "permit_date"] = np.nan

    # we could have multiple hits, we need to de-duplicate.
    # find the permit date closest to the valuation date for each key
    df_u = df_u.sort_values(by=["permit_date"], descending=[True])
    df_u = df_u.drop_duplicates(subset=["key"], keep="first")

    df_u["last_permit_was_teardown"] = df_u[df_u["is_teardown"].eq(True)]
    df_u = df_u.rename(columns={"permit_date": "demo_date"})

    # Now we know, for each parcel, if it's last permit was for a teardown, and when it was torn down
    df_univ = df_in.merge(
        df_u[["key", "last_permit_was_teardown", "demo_date"]], on="key", how="left"
    )

    # ===========================================================================================#
    #                              Process renovation universe                                  #
    # ===========================================================================================#

    valuation_date = get_valuation_date(settings)

    df_u = df_in[["key"]].copy()

    df_u = df_u.merge(df_renos, on="key", how="left")
    df_u["sale_date"] = valuation_date
    df_u["days_to_reno"] = (df_u["reno_date"] - df_u["sale_date"]).dt.days
    # Ignore renovations that happened AFTER the valuation date
    df_u.loc[df_u["days_to_reno"].ge(0), "days_to_reno"] = np.nan

    # Find the most recent major renovation date:
    df_u = df_u.sort_values(by=["renovation_num", "days_to_reno"], ascending=[False])
    df_u = df_u.drop_duplicates(subset=["key"], keep="first")

    # Merge the results back onto df_universe
    df_univ = df_univ.merge(
        df_u[["key", "reno_date", "days_to_reno", "renovation_num", "renovation_txt"]],
        on="key",
        how="left",
    )

    if calc_effective_age:
        # Calculate effective year built based on last major renovation
        if "bldg_effective_year_built" in df_univ:
            warnings.warn(
                "bldg_effective_year_built already exists in df_univ, overwriting it."
            )
            df_univ["bldg_effective_year_built"] = df_univ["bldg_effective_year_built"]
        else:
            df_univ["bldg_effective_year_built"] = df_univ["bldg_year_built"]

        # Major renovations reset the date to the current year
        df_univ.loc[df_univ["renovation_num"].eq(3), "bldg_effective_year_built"] = (
            df_univ["reno_date"].dt.year
        )

    return df_univ


def _process_permits_sales(
    df_in: pd.DataFrame,
    df_permits: pd.DataFrame,
    s_permits: dict,
    settings: dict,
    verbose: bool = False,
):
    df_sales = df_in.copy()

    calc_effective_age = s_permits.get("calc_effective_age", False)

    # We might have multiple permits per key. We have multiple questions to answer:

    # =========================================================================================#
    #                              Process teardown sales                                     #
    # =========================================================================================#

    # 1. Do we have a demolition permit? When was the demolition date?
    if "is_teardown" in df_permits:
        df_demos = df_permits[df_permits["is_teardown"].eq(True)][
            ["key", "date"]
        ].copy()

        df = df_sales[
            ["key", "key_sale", "sale_date", "valid_sale", "vacant_sale", "sale_price"]
        ].copy()

        # Label the teardown sales (sales torn down shortly AFTER the sale date)
        df_demos = df_demos.rename(columns={"date": "demo_date"})
        df = df.merge(df_demos, on="key", how="left")
        df["days_to_demo"] = (df["sale_date"] - df["demo_date"]).dt.days
        # Ignore demolitions that happened BEFORE the sale
        df.loc[df["days_to_demo"].le(0), "days_to_demo"] = np.nan

        # we could have multiple hits, we need to de-duplicate.
        # find the demo date closest to the sale for each key
        df = df.sort_values(by=["days_to_demo"], ascending=[True])
        df = df.drop_duplicates(subset=["key"], keep="first")

        max_days_to_demo = s_permits.get("max_days_to_demo", 365)

        # Count it as a teardown if the demo date is within the max_days_to_demo
        df["is_teardown_sale"] = False
        df.loc[
            df["days_to_demo"].gt(0) & df["days_to_demo"].le(max_days_to_demo),
            "is_teardown_sale",
        ] = True

        # Merge the results back onto df_sales
        # Now we know, for each sale, if it's a likely teardown sale and when it was torn down
        df_sales = df_sales.merge(
            df[["key", "is_teardown_sale", "demo_date", "days_to_demo"]],
            on="key",
            how="left",
        )

        # Set is_vacant_sale to True for teardown sales
        df_sales.loc[df_sales["is_teardown_sale"].eq(True), "vacant_sale"] = True

        if verbose:
            teardown_sales = df[df["is_teardown_sale"]]
            print(f"Identified {len(teardown_sales)} teardown sales.")

    # =========================================================================================#
    #                              Process renovation sales                                   #
    # =========================================================================================#

    # 2. Do we have a renovation permit? When was the renovation date?
    if "is_renovation" in df_permits:

        if "renovation_num" not in df_permits:
            raise ValueError(
                "Missing field 'renovation_num' in df_permits. Cannot process renovation permits."
            )
        if "renovation_txt" not in df_permits:
            raise ValueError(
                "Missing field 'renovation_txt' in df_permits. Cannot process renovation permits."
            )

        df_renos = df_permits[df_permits["is_renovation"].eq(True)][
            ["key", "date", "renovation_num", "renovation_txt", "is_renovation"]
        ].copy()

        # Label sales with the most recent renovation data from BEFORE the sale date
        df_renos = df_renos.rename(
            columns={"date": "reno_date", "is_renovation": "is_renovated"}
        )
        df = df_in.merge(df_renos, on="key", how="left")
        df.loc[pd.isna(df["is_renovated"]), "is_renovated"] = False
        df["days_to_reno"] = (df["reno_date"] - df["sale_date"]).dt.days

        # Ignore renovations that happened AFTER the sale
        df.loc[df["days_to_reno"].ge(0), "days_to_reno"] = np.nan
        df.loc[pd.isna(df["days_to_reno"]), "reno_date"] = None
        df.loc[pd.isna(df["days_to_reno"]), "renovation_num"] = np.nan
        df.loc[pd.isna(df["days_to_reno"]), "renovation_txt"] = None
        df.loc[pd.isna(df["days_to_reno"]), "is_renovated"] = False

        # Find the most recent major renovation date:
        df = df.sort_values(
            by=["renovation_num", "days_to_reno"], ascending=[False, False]
        )

        df = df.drop_duplicates(subset=["key"], keep="first")

        # Merge the results back onto df_sales
        df_sales = df_sales.merge(
            df[
                [
                    "key",
                    "is_renovated",
                    "reno_date",
                    "days_to_reno",
                    "renovation_num",
                    "renovation_txt",
                ]
            ],
            on="key",
            how="left",
        )

        if calc_effective_age:
            # Calculate effective year built based on last major renovation
            if "bldg_effective_year_built" in df_sales:
                warnings.warn(
                    "bldg_effective_year_built already exists in df_sales, overwriting it."
                )
                df_sales["bldg_effective_year_built"] = df_sales[
                    "bldg_effective_year_built"
                ]
            else:
                df_sales["bldg_effective_year_built"] = df_sales["bldg_year_built"]

            # Major renovations reset the date to the current year
            df_sales.loc[
                df_sales["renovation_num"].eq(3), "bldg_effective_year_built"
            ] = df_sales["reno_date"].dt.year

            # TODO: Medium renovations reset the date partially, which requires knowing a bunch of stuff

            # Minor renovations do not reset the date

        if verbose:
            renovations = df[df["is_renovated"]]
            renovations_3 = df[df["renovation_num"].eq(3)]
            renovations_2 = df[df["renovation_num"].eq(2)]
            renovations_1 = df[df["renovation_num"].eq(1)]
            print(f"Identified {len(renovations)} renovated sales.")
            print(f"--> {len(renovations_3)} major renovations.")
            print(f"--> {len(renovations_2)} medium renovations.")
            print(f"--> {len(renovations_1)} minor renovations.")

    return df_sales


def read_sales_univ(path: str):
    sales_path = f"{path}sales.parquet"
    univ_path = f"{path}universe.parquet"
    if not os.path.exists(sales_path):
        raise ValueError(f"{sales_path} does not exist!")
    if not os.path.exists(univ_path):
        raise ValueError(f"{univ_path} does not exist!")
    df_sales = pd.read_parquet(sales_path)
    df_univ = _read_univ_parquet(univ_path)
    return SalesUniversePair(df_sales, df_univ)


def read_predictions(model: str, model_group: str, pred_type: str):
    df : pd.DataFrame = None
    key = "key" if pred_type == "universe" else "key_sale"
    for thing in ["main"]:
        try:
            _df = pd.read_parquet(f"out/models/{model_group}/{thing}/{model}/pred_{pred_type}.parquet")
            _df = _df[[key, "prediction"]]
            if df is None:
                df = _df
            else:
                df = pd.concat([df,_df]).reset_index(drop=True)
        except FileNotFoundError as e:
            print(f"Error reading {model_group}/{model} : {e}")
    return df


def _read_univ_parquet(path: str):
    df = pd.read_parquet(path)
    if "geometry" in df:
        crs, geom_col = detect_crs_from_parquet(path, "geometry")
        gdf = ensure_geometries(df, geom_col=geom_col, crs=crs)
        if gdf.crs is None:
            raise ValueError("No CRS found in parquet metadata")
        return gdf
    return df

def get_sup_model_group(sup: SalesUniversePair, model_group: str):
    df = get_hydrated_sales_from_sup(sup)
    df = df[df["model_group"].eq(model_group)]
    keys = df["key_sale"].unique()
    df_sales = sup.sales[sup.sales["key_sale"].isin(keys)]
    df_univ = sup.universe[sup.universe["model_group"].eq(model_group)].copy()
    return SalesUniversePair(df_sales, df_univ)
    

def write_parquet(df, path):
    """
    Write data to a parquet file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to be written
    path : str
        File path for saving the parquet.
    """
    
    if not path.endswith(".parquet"):
        raise ValueError("Path must end with .parquet!")
    
    # If it has a geometry column, write as GeoParquet
    if "geometry" in df.columns:
        # Ensure it's a GeoDataFrame
        gdf = df if isinstance(df, gpd.GeoDataFrame) else gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None))

        # You MUST have a CRS for it to be recorded in metadata
        if gdf.crs is None:
            raise ValueError(f"{path}: geometry has no CRS. Set it (e.g., gdf = gdf.set_crs('EPSG:4326')) before writing.")

        # GeoPandas writes WKB + GeoParquet metadata (including CRS)
        gdf.to_parquet(path, engine="pyarrow", index=False)
    else:
        # Regular table
        df.to_parquet(path, engine="pyarrow", index=False)


def write_gpkg(df, path):
    """
    Write data to a geopackage file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to be written
    path : str
        File path for saving the geopackage.
    """
    if not path.endswith(".gpkg"):
        raise ValueError("Path must end with .gpkg!")
    
    # If it has a geometry column, write as GeoParquet
    if "geometry" in df.columns:
        # Ensure it's a GeoDataFrame
        gdf = df if isinstance(df, gpd.GeoDataFrame) else gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None))
       
        # You MUST have a CRS for it to be recorded in metadata
        if gdf.crs is None:
            raise ValueError(f"{path}: geometry has no CRS. Set it (e.g., gdf = gdf.set_crs('EPSG:4326')) before writing.")
        
        gdf.to_file(path, driver='GPKG', layer='name', mode='w')
    else:
        raise ValueError("cannot write to gpkg without geometry")


def write_shapefile(df, path):
    """
    Write data to a shapefile file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to be written
    path : str
        File path for saving the shapefile.
    """
    
    if not path.endswith(".shp"):
        raise ValueError("Path must end with .shp!")
    
    # If it has a geometry column, write as GeoParquet
    if "geometry" in df.columns:
        # Ensure it's a GeoDataFrame
        gdf = df if isinstance(df, gpd.GeoDataFrame) else gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None))

        # You MUST have a CRS for it to be recorded in metadata
        if gdf.crs is None:
            raise ValueError(f"{path}: geometry has no CRS. Set it (e.g., gdf = gdf.set_crs('EPSG:4326')) before writing.")

        gdf.to_file(path)
    else:
        raise ValueError("cannot write to gpkg without geometry")


def write_zipped_shapefile(df, path: str) -> Path:
    """
    Write a zipped ESRI Shapefile. Produces a single {name}.shp.zip with the
    shapefile parts (name.shp, .shx, .dbf, .prj, .cpg, etc.) at the ZIP root.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        Data to be written (must include a 'geometry' column and a CRS).
    path : str
        Destination path ending with '.shp.zip' (e.g., 'out/roads.shp.zip').

    Returns
    -------
    pathlib.Path
        Path to the created .shp.zip
    """
    p = Path(path)

    # Require ".shp.zip" exactly, per your spec
    if p.suffixes[-2:] != [".shp", ".zip"]:
        raise ValueError("Path must end with .shp.zip (e.g., 'out/roads.shp.zip').")

    # layer name (strip .zip then .shp)
    layer = Path(p.stem).stem
    if not layer:
        raise ValueError("Could not derive layer name from path.")

    # Make sure parent directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    # Write shapefile into a temp dir, then zip and move atomically
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        shp_path = tmpdir / f"{layer}.shp"

        # Reuse your existing function (validates geometry + CRS)
        write_shapefile(df, str(shp_path))

        # Common shapefile sidecar extensions we may need to include if present
        sidecars = {
            ".shp", ".shx", ".dbf", ".prj", ".cpg",
            ".qix", ".sbn", ".sbx", ".fbn", ".fbx",
            ".ain", ".aih", ".ixs", ".mxs", ".atx",
            ".xml", ".qpj"
        }

        tmp_zip = tmpdir / f"{layer}.shp.zip"
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for ext in sorted(sidecars):
                f = tmpdir / f"{layer}{ext}"
                if f.exists():
                    # Store with just the filename at the ZIP root
                    zf.write(f, arcname=f.name)

        # Move the finished ZIP to the destination (overwrites if exists)
        shutil.move(str(tmp_zip), str(p))

    return p


def filter_df_by_date_range(df, start_date, end_date):
    """
    Filter df to rows where 'sale_date' is between start_date and end_date (inclusive).
    - start_date/end_date may be 'YYYY-MM-DD' strings or date/datetime/Timestamp.
    - Time-of-day and time zones are ignored.
    - Rows with missing/unparseable 'sale_date' are dropped.
    """
    import pandas as pd
    from datetime import date, datetime, timedelta
    from pandas.api.types import is_datetime64tz_dtype

    def _as_date(x):
        # If already a date (but not datetime), keep it
        if isinstance(x, date) and not isinstance(x, datetime):
            return x
        # Otherwise parse and take the calendar date
        return pd.to_datetime(x).date()

    start_d = _as_date(start_date)
    end_d   = _as_date(end_date)
    if start_d > end_d:
        raise ValueError("start_date cannot be after end_date.")

    # Coerce to datetime; tolerate bad/missing → NaT
    s = pd.to_datetime(df["sale_date"], errors="coerce")

    # Strip timezone info if present, preserving local wall time
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)

    # Build inclusive range using an exclusive upper bound
    start_ts = pd.Timestamp(start_d)                       # 00:00:00 on start day
    end_excl = pd.Timestamp(end_d) + pd.Timedelta(days=1)  # first moment after end day

    # NaT values compare as False and will be dropped
    mask = s.ge(start_ts) & s.lt(end_excl)
    return df.loc[mask].copy()
