import os
import pickle
import warnings

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio import features

import pandas as pd
from IPython.core.display_functions import display
from geopandas import GeoDataFrame

from openavmkit.benchmark import MultiModelResults, _calc_benchmark, run_ensemble, \
  optimize_ensemble_allocation
from openavmkit.modeling import SingleModelResults, plot_value_surface
from openavmkit.quality_control import check_land_values
from openavmkit.utilities.data import div_field_z_safe
from openavmkit.utilities.geometry import select_grid_size_from_size_str, get_crs
from openavmkit.utilities.plotting import plot_histogram_df
from openavmkit.utilities.settings import get_model_group_ids


def run_land_analysis(
    df_in: pd.DataFrame,
    settings: dict,
    verbose: bool = False
):
  model_group_ids = get_model_group_ids(settings)
  for model_group in model_group_ids:
    _run_land_analysis(df_in, settings, model_group, verbose)


def finalize_land_values(
    settings: dict,
    verbose: bool = False
):
  model_group_ids = get_model_group_ids(settings)
  for model_group in model_group_ids:
    outpath = f"out/models/cache/{model_group}/land_analysis.pickle"
    if os.path.exists(outpath):
      df_finalize = pd.read_pickle(outpath)
      _finalize_land_values(
        df_finalize,
        model_group,
        verbose
      )


def _finalize_land_values(
    df_in: pd.DataFrame,
    model_group: str,
    verbose: bool = False
):
  df = df_in.copy()

  # Derive the final land values
  df["model_land_value"] = df["model_market_value"] * df["model_land_alloc"]
  df["model_impr_value"] = df["model_market_value"] - df["model_land_value"]

  # Apply basic sanity check / error correction to land values
  df = check_land_values(df, model_group)

  # Plot the final land values on a map
  plot_value_surface(
    "Land value",
    df["model_land_value"],
    x_coords=df["longitude"],
    y_coords=df["latitude"],
    gdf=df,
    center_on_zero=False
  )

  df["model_land_value_land_sqft"] = div_field_z_safe(df["model_land_value"], df["land_area_sqft"])
  plot_value_surface(
    "Land value per sqft",
    df["model_land_value_land_sqft"],
    x_coords=df["longitude"],
    y_coords=df["latitude"],
    gdf=df
  )

  # outpath = f"out/models/{model_group}/_images/"
  # os.makedirs(outpath, exist_ok=True)
  # generate_raster(df, outpath,"model_land_value", plot=True)


  # If necessary, apply a mild gaussian blur the final land values

  #


# Generate a raster from the final land values on a map
def generate_raster(
    gdf_in: GeoDataFrame,
    path: str,
    field: str,
    plot: bool = False
):
  crs = get_crs(gdf_in, projection_type="equal_area")
  gdf = gdf_in.to_crs(crs)
  height, width = select_grid_size_from_size_str(gdf, "10ft")
  minx, miny, maxx, maxy = gdf.total_bounds

  pixel_size_x = (maxx - minx) / width
  pixel_size_y = (maxy - miny) / height

  transform = rasterio.transform.from_origin(minx, maxy, pixel_size_x, pixel_size_y)

  shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[field]))

  raster = features.rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=np.nan,
    all_touched=True,
    dtype="float32"
  )

  if plot:
    # visualize the raster
    vmin = 0
    vmax = np.quantile(gdf[field], 0.95)
    plt.imshow(raster, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

  # Write out the raster:
  outpath = f"{path}raster_{field}.tif"
  with rasterio.open(outpath, 'w', driver='GTiff', height=height, width=width, count=1, dtype=rasterio.float32, crs=crs, transform=transform) as dst:
    dst.write(raster, 1)

  display(raster)

  return raster

def _run_land_analysis(
    df_in: pd.DataFrame,
    settings: dict,
    model_group: str,
    verbose: bool = False
):
  instructions = settings.get("modeling", {}).get("instructions", {})
  allocation = instructions.get("allocation", {})

  results_map = {
    "main": {},
    "hedonic": {},
    "vacant": {}
  }

  land_fields = []
  land_results : dict[str : SingleModelResults] = {}

  # STEP 1: Gather results from the main, hedonic, and vacant models

  for key in ["main", "hedonic", "vacant"]:
    short_key = key[0]
    if key == "main":
      models = instructions.get("main", {}).get("run", [])
      if "ensemble" not in models:
        models.append("ensemble")
    else:
      models = allocation.get(key, [])
    outpath = f"out/models/{model_group}/{key}"

    if verbose:
      print(f"key = {key}")
    if len(models) > 0:
      for model in models:
        if verbose:
          print(f"----> model = {model}")
        filepath = f"{outpath}/model_{model}.pickle"
        if os.path.exists(filepath):
          with open(filepath, "rb") as file:
            results = pickle.load(file)
            df = results.df_universe[["key"]].copy()
            df.loc[:, "prediction"] = results.pred_univ
            results_map[key][model] = df
            if key != "main":
              land_results[f"{short_key}_{model}"] = results
              land_fields.append(f"{short_key}_{model}")

  df_all_alloc = results_map["main"]["ensemble"].copy()
  df_all_land_values = df_all_alloc.copy()
  df_all_land_values = df_all_land_values[["key"]].merge(df_in, on="key", how="left")
  all_alloc_names = []

  bins = 400

  # STEP 2: Calculate land allocations for each model

  for key in ["hedonic", "vacant"]:
    short_key = key[0]
    df_alloc = results_map["main"]["ensemble"].copy()
    alloc_names = []
    entries = results_map[key]
    for model in entries:

      pred_main = None #results_map["main"].get(model)

      if pred_main is None:
        warnings.warn(f"No main model found for model: {model}, using ensemble instead")
        pred_main = results_map["main"].get("ensemble")

      pred_land = results_map[key].get(model).rename(columns={"prediction": "prediction_land"})
      df = pred_main.merge(pred_land, on="key", how="left")
      alloc_name = f"{short_key}_{model}"
      df.loc[:, alloc_name] = df["prediction_land"] / df["prediction"]

      df_alloc = df_alloc.merge(df[["key", alloc_name]], on="key", how="left")
      df_all_alloc = df_all_alloc.merge(df[["key", alloc_name]], on="key", how="left")

      df2 = df.copy().rename(columns={"prediction_land": alloc_name})
      df_all_land_values = df_all_land_values.merge(df2[["key", alloc_name]], on="key", how="left")

      alloc_names.append(alloc_name)
      all_alloc_names.append(alloc_name)

    df_alloc["allocation_ensemble"] = df_alloc[alloc_names].median(axis=1)

    plot_histogram_df(
      df=df_alloc,
      fields=alloc_names,
      xlabel="% of value attributable to land",
      ylabel="Number of parcels",
      title=f"Land allocation -- {key}",
      bins=bins,
      x_lim=(0.0, 1.0)
    )
    plot_histogram_df(
      df=df_alloc,
      fields=["allocation_ensemble"],
      xlabel="% of value attributable to land",
      ylabel="Number of parcels",
      title=f"Land allocation -- {key}, ensemble",
      bins=bins,
      x_lim=(0.0, 1.0)
    )

  plot_histogram_df(
    df=df_all_alloc,
    fields=all_alloc_names,
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- all",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  df_all_alloc["allocation_ensemble"] = df_all_alloc[all_alloc_names].median(axis=1)
  plot_histogram_df(
    df=df_all_alloc,
    fields=["allocation_ensemble"],
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- all, ensemble",
    bins=bins,
    x_lim=(0.0, 1.0)
  )



  # STEP 3: Optimize the ensemble allocation

  print(f"Putting it all together...")

  all_land_results = MultiModelResults(
    land_results,
    _calc_benchmark(land_results)
  )

  best_ensemble = optimize_ensemble_allocation(
    df=None,
    model_group=model_group,
    vacant_only=True,
    ind_var="sale_price_time_adj",
    ind_var_test="sale_price_time_adj",
    all_results=all_land_results,
    settings=settings,
    verbose=verbose
  )

  print(f"Best land ensemble --> {best_ensemble}")

  # Run the ensemble model
  ensemble_results = run_ensemble(
    df=df_in,
    model_group=model_group,
    vacant_only=True,
    hedonic=False,
    ind_var="sale_price_time_adj",
    ind_var_test="sale_price_time_adj",
    outpath=outpath,
    ensemble_list=best_ensemble,
    all_results=all_land_results,
    settings=settings,
    verbose=verbose
  )

  all_land_results.add_model("ensemble", ensemble_results)

  print("LAND ENSEMBLE BENCHMARK:")
  print(all_land_results.benchmark.print())

  drop_alloc_names = [name for name in all_alloc_names if name not in best_ensemble]
  df_all_alloc = df_all_alloc.drop(columns=drop_alloc_names)

  plot_histogram_df(
    df=df_all_alloc,
    fields=best_ensemble,
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- best ensemble (components)",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  df_all_alloc["allocation_ensemble"] = df_all_alloc[best_ensemble].median(axis=1)
  plot_histogram_df(
    df=df_all_alloc,
    fields=["allocation_ensemble"],
    xlabel="% of value attributable to land",
    ylabel="Number of parcels",
    title=f"Land allocation -- best ensemble",
    bins=bins,
    x_lim=(0.0, 1.0)
  )

  # STEP 4: Finalize the results
  df_finalize = df_all_alloc.drop(columns=best_ensemble)
  df_finalize = df_finalize.rename(columns={"allocation_ensemble": "model_land_alloc", "prediction": "model_market_value"})
  df_finalize = df_finalize.merge(df_in[["key", "geometry", "latitude", "longitude", "land_area_sqft", "bldg_area_finished_sqft"]], on="key", how="left")
  df_finalize = GeoDataFrame(df_finalize, geometry="geometry", crs=df_in.crs)

  outpath = f"out/models/cache/{model_group}/land_analysis.pickle"
  os.makedirs(os.path.dirname(outpath), exist_ok=True)
  df_finalize.to_pickle(outpath)
