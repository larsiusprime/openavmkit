import os
import pickle

from openavmkit.utilities.plotting import plot_histogram_df


def run_land_analysis(
    settings: dict,
    verbose: bool = False
):
  instructions = settings.get("modeling", {}).get("instructions", {})
  allocation = instructions.get("allocation", {})

  results_map = {
    "main": {},
    "hedonic": {},
    "vacant": {}
  }

  for key in ["main", "hedonic", "vacant"]:
    if key == "main":
      models = instructions.get("main", {}).get("run", [])
      if "ensemble" not in models:
        models.append("ensemble")
    else:
      models = allocation.get(key, [])
    outpath = f"out/models/{key}"
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

  df_all_alloc = results_map["main"]["ensemble"].copy()
  all_alloc_names = []

  bins = 400

  for key in ["hedonic", "vacant"]:
    df_alloc = results_map["main"]["ensemble"].copy()
    alloc_names = []
    entries = results_map[key]
    for model in entries:
      pred_main = results_map["main"].get(model)
      pred_land = results_map[key].get(model).rename(columns={"prediction": "prediction_land"})
      df = pred_main.merge(pred_land, on="key", how="left")
      alloc_name = f"{key}_{model}"
      df.loc[:, alloc_name] = df["prediction_land"] / df["prediction"]
      df.loc[df[alloc_name].gt(2.0), alloc_name] = None
      df.loc[df[alloc_name].lt(-0.5), alloc_name] = None
      df.loc[df[alloc_name].gt(1.0), alloc_name] = 1.0
      df.loc[df[alloc_name].lt(0.0), alloc_name] = 0.0
      df_alloc = df_alloc.merge(df[["key", alloc_name]], on="key", how="left")
      df_all_alloc = df_all_alloc.merge(df[["key", alloc_name]], on="key", how="left")
      alloc_names.append(alloc_name)
      all_alloc_names.append(alloc_name)

    df_alloc["allocation_ensemble"] = df_alloc[alloc_names].median(axis=1)

    plot_histogram_df(
      df=df_alloc,
      fields=(alloc_names),
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
    fields=(all_alloc_names),
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