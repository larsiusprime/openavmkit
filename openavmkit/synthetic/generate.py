import math
import os
import pickle
import random

import geopandas as gpd
import numpy as np
import pandas as pd

from openavmkit.benchmark import (
    run_one_model,
    MultiModelResults,
    _calc_benchmark,
    get_data_split_for,
    run_one_hedonic_model,
    _optimize_ensemble,
    _run_ensemble,
    run_ensemble,
    _format_benchmark_df,
)
from openavmkit.data import (
    SalesUniversePair,
    enrich_time,
    _perform_canonical_split,
    get_important_field,
    _basic_geo_enrichment,
    enrich_sup_spatial_lag,
    get_hydrated_sales_from_sup,
    get_sale_field,
)
from openavmkit.horizontal_equity_study import mark_horizontal_equity_clusters
from openavmkit.modeling import SingleModelResults, LandPredictionResults
from openavmkit.synthetic.synthetic import make_geo_blocks
from openavmkit.utilities.data import div_series_z_safe
from openavmkit.utilities.geometry import get_crs_from_lat_lon
from openavmkit.utilities.settings import area_unit


def add_polar_neighborhoods(
    gdf: gpd.GeoDataFrame, divisions: list[tuple]
) -> gpd.GeoDataFrame:
    max = gdf["polar_radius"].max()

    for loc_slice_x, loc_slice_y in divisions:
        gdf[f"loc_polar_{loc_slice_x}"] = (
            (gdf["polar_radius"] // (max / loc_slice_x)).astype(int).astype(str)
            + "x"
            + (gdf["polar_angle"] // (360 / loc_slice_y)).astype(int).astype(str)
        )
    return gdf


def simple_plane(params: dict) -> gpd.GeoDataFrame:
    latitude = params["latitude"]
    longitude = params["longitude"]
    units = params["units"]
    blocks_x = params["blocks_x"]
    blocks_y = params["blocks_y"]
    block_size_x = params["block_size_x"]
    block_size_y = params["block_size_y"]
    crs = params.get("crs", None)

    if crs is None:
        crs = get_crs_from_lat_lon(latitude, longitude, "equal_area", units)

    blocks = []

    for y in range(0, blocks_y):
        for x in range(0, blocks_x):
            blocks.append({"x": x, "y": y})

    gdf = make_geo_blocks(
        latitude, longitude, block_size_y, block_size_x, blocks, units, crs
    )
    gdf["key"] = gdf["x"].astype(str) + "-" + gdf["y"].astype(str)

    # get centroid:
    centroid = gdf["geometry"].unary_union.centroid

    # calculate distance between every parcel's geometry and the centroid
    gdf["dist_to_centroid"] = gdf.geometry.distance(centroid)

    # add land area and improvement area
    gdf[f"is_vacant"] = True  # everything's vacant
    gdf[f"land_area_sq{units}"] = gdf.geometry.area
    gdf[f"bldg_area_finished_sq{units}"] = 0.0

    for loc_slice in [2, 4, 8, 16, 32]:
        gdf[f"loc_{loc_slice}"] = (
            (gdf["x"] // (blocks_x / loc_slice)).astype(int).astype(str)
            + "x"
            + (gdf["y"] // (blocks_y / loc_slice)).astype(int).astype(str)
        )

    # clean up
    gdf = gdf.drop(columns=["x", "y"])

    return gdf


def simple_plane_w_buildings(params: dict) -> gpd.GeoDataFrame:
    random.seed(params.get("seed", 1337))

    gdf = simple_plane(params)

    perc_vacant = params["perc_vacant"]
    units = params["units"]

    gdf[f"is_vacant"] = gdf["key"].apply(lambda x: random.random() < perc_vacant)
    gdf.loc[gdf[f"is_vacant"].eq(False), f"bldg_area_finished_sq{units}"] = gdf[
        f"land_area_sq{units}"
    ].apply(lambda x: random.uniform(0.1, 0.5) * x)

    return gdf


def add_simple_transactions(gdf: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:

    perc_sales = params["perc_sales"]
    perc_sales = min(1.0, max(0.0, perc_sales))
    value_field = params["value_field"]
    error = params["perc_error"]
    error_low = 1.0 - (error / 2)
    error_high = 1.0 + (error / 2)

    valid_sales = np.zeros(len(gdf), dtype=bool)
    for i in range(len(gdf)):
        valid_sales[i] = random.random() < perc_sales

    gdf[f"valid_sale"] = valid_sales
    gdf["valid_for_ratio_study"] = gdf["valid_sale"]

    settings = {
        "modeling": {
            "metadata": {"valuation_date": pd.to_datetime("now").strftime("%Y-%m-%d")}
        }
    }

    gdf.loc[gdf[f"valid_sale"].eq(True), f"sale_date"] = pd.to_datetime("now").strftime(
        "%Y-%m-%d"
    )
    gdf = enrich_time(gdf, {}, settings)
    gdf.loc[gdf[f"valid_sale"].eq(True), f"sale_price"] = gdf[value_field].apply(
        lambda x: x * random.uniform(error_low, error_high)
    )
    gdf["sale_price_time_adj"] = gdf["sale_price"]
    gdf["key_sale"] = gdf["key"] + "_" + gdf["sale_date"].dt.strftime("%Y-%m-%d")

    return gdf


def add_simple_bldg_value(gdf: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    base_value = params["base_value"]
    size_field = params["size_field"]
    series = gdf[size_field] * base_value
    return series


def add_simple_land_value(gdf: gpd.GeoDataFrame, params: dict) -> gpd.GeoDataFrame:
    curve = params["curve"]
    base_value = params["base_value"]
    size_field = params["size_field"]

    dist_norm = gdf["dist_to_centroid"] / gdf["dist_to_centroid"].max()

    if curve == "linear":
        series = dist_norm.apply(linear_decrease)
    elif curve == "inverse_square":
        series = dist_norm.apply(inverse_square_decrease)
    elif curve == "exponential":
        series = dist_norm.apply(exponential_decrease)
    else:
        raise ValueError(f"Unknown curve type: {curve}")

    series *= base_value * gdf[size_field]
    return series


def linear_decrease(d):
    """
    Linear decrease: f(d) = 1 - d for 0 <= d <= 1.
    Beyond d = 1, the value is 0.
    """
    return max(0.0, 1 - d)


def inverse_square_decrease(d):
    """
    Inverse-square-like decrease:
    f(d) = 1 / (1 + d^2)
    This avoids a singularity at d = 0 and decreases the value with the square of d.
    """
    return 1.0 / (1 + d**2)


def exponential_decrease(d, alpha=5):
    """
    Exponential decrease:
    f(d) = exp(-alpha * d)

    Parameters:
      d     : normalized distance in [0, 1]
      alpha : decay constant (default 5)
    """
    return math.exp(-alpha * d)
