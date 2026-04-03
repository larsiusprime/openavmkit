import os
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from sklearn.metrics import mean_absolute_percentage_error

from IPython.display import display
from openavmkit.data import (
    _get_sales,
    get_hydrated_sales_from_sup,
    SalesUniversePair,
    get_train_test_keys,
    get_sale_field,
)
from openavmkit.modeling import SingleModelResults, plot_value_surface, simple_ols
from openavmkit.quality_control import check_land_values
from openavmkit.utilities.data import (
    div_series_z_safe,
    add_area_fields,
)
from openavmkit.utilities.plotting import plot_histogram_df
from openavmkit.utilities.settings import area_unit, get_model_group_ids

from openavmkit.utilities.stats import calc_correlations, calc_cod, calc_r2, calc_mse_r2_adj_r2


def run_land_analysis(sup: SalesUniversePair, settings: dict, verbose: bool = False):
    df_sales = get_hydrated_sales_from_sup(sup)
    model_group_ids = get_model_group_ids(settings)
    for model_group in model_group_ids:
        _run_land_analysis(df_sales, sup.universe, settings, model_group, verbose)


def convolve_land_analysis(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
):
    df_sales = get_hydrated_sales_from_sup(sup)
    model_group_ids = get_model_group_ids(settings)
    for model_group in model_group_ids:
        _convolve_land_analysis(df_sales, sup.universe, settings, model_group, verbose)


def finalize_land_values(
    df_in: pd.DataFrame,
    settings: dict,
    verbose: bool = False,
) -> pd.DataFrame:
    model_group_ids = get_model_group_ids(settings)
    df_all_values: pd.DataFrame | None = None
    for model_group in model_group_ids:
        df_values = df_in[df_in["model_group"].eq(model_group)].copy()
        outpath = f"out/models/{model_group}/_cache/land_analysis.pickle"
        if os.path.exists(outpath):
            df_finalize = pd.read_pickle(outpath)
            df_finalize = _finalize_land_values(
                df_in, df_finalize, model_group, settings, verbose
            )
            df_values = df_values.merge(
                df_finalize[
                    [
                        "key",
                        "model_market_value",
                        "model_impr_value",
                        "model_land_value",
                    ]
                ],
                on="key",
                how="left",
            )
            df_values = add_area_fields(df_values, settings)
            if df_all_values is None:
                df_all_values = df_values
            else:
                df_all_values = pd.concat([df_all_values, df_values], ignore_index=True)
    df_all_values.reset_index(inplace=True, drop=True)
    new_fields = [field for field in df_all_values.columns.values if field != "key"]
    df_return = df_in.copy()
    df_return = df_return[[col for col in df_return if col not in new_fields]]
    df_return = df_return.merge(df_all_values, on="key", how="left")

    os.makedirs(f"out/models/", exist_ok=True)
    gdf = gpd.GeoDataFrame(df_return, geometry="geometry")
    gdf.to_parquet(f"out/models/predictions.parquet")

    return gdf


def _finalize_land_values(
    df_orig: pd.DataFrame,
    df_in: pd.DataFrame,
    model_group: str,
    settings: dict,
    verbose: bool = False,
):
    df = df_in.copy()
    
    unit = area_unit(settings)

    # Derive the final land values
    df["model_land_value"] = df["model_market_value"] * df["model_land_alloc"]
    df["model_impr_value"] = df["model_market_value"] - df["model_land_value"]

    # Apply basic sanity check / error correction to land values
    df = check_land_values(df, model_group)

    df[f"model_land_value_land_{unit}"] = div_series_z_safe(
        df["model_land_value"], df[f"land_area_{unit}"]
    )
    df[f"model_market_value_land_{unit}"] = div_series_z_safe(
        df["model_market_value"], df[f"land_area_{unit}"]
    )
    df[f"model_market_value_impr_{unit}"] = div_series_z_safe(
        df["model_market_value"], df[f"bldg_area_finished_{unit}"]
    )

    # Find variables correlated with land value

    df_subset = df_orig[df_orig["model_group"].eq(model_group)]
    df_sales = _get_sales(df_subset, settings)
    df_sales = df_sales.merge(
        df[
            [
                "key_sale",
                "key",
                "model_market_value",
                "model_land_value",
                f"model_land_value_land_{unit}",
            ]
        ],
        on="key",
        how="left",
    )
    df_sales[f"model_market_value_impr_{unit}"] = div_series_z_safe(
        df_sales["model_market_value"], df_sales[f"bldg_area_finished_{unit}"]
    )
    df_sales[f"model_market_value_land_{unit}"] = div_series_z_safe(
        df_sales["model_market_value"], df_sales[f"land_area_{unit}"]
    )

    ind_vars = (
        settings.get("modeling", {})
        .get("models", {})
        .get("main", {})
        .get("default", {})
        .get("ind_vars", [])
    )
    ind_vars = ind_vars + [
        "assr_market_value",
        "assr_land_value",
        "model_market_value",
        f"model_market_value_land_{unit}",
        f"model_market_value_impr_{unit}",
    ]

    print("LAND VALUE")
    X_corr = df_sales[["model_land_value"] + ind_vars]
    corrs = calc_correlations(X_corr)
    print("INITIAL")
    display(corrs["initial"])
    print("")
    print("FINAL)")
    display(corrs["final"])
    print("")
    print(f"LAND VALUE PER {unit.upper()}")
    X_corr = df_sales[[f"model_land_value_land_{unit}"] + ind_vars]
    corrs = calc_correlations(X_corr)
    print("INITIAL")
    display(corrs["initial"])
    print("")
    print("FINAL)")
    display(corrs["final"])

    # Super tiny slivers of land will have insane $/area values
    df[f"model_market_value_land_{unit}"] = div_series_z_safe(
        df["model_market_value"], df[f"land_area_{unit}"]
    )
    df[f"model_market_value_impr_{unit}"] = div_series_z_safe(
        df["model_market_value"], df[f"bldg_area_finished_{unit}"]
    )
    df_not_tiny = df[df[f"land_area_{unit}"].gt(5000)]

    plot_value_surface(
        f"Land value per {unit}",
        df_not_tiny[f"model_land_value_land_{unit}"],
        gdf=df,
        cmap="viridis",
        norm="log",
    )

    plot_value_surface(
        f"Market value per land {unit}",
        df_not_tiny[f"model_market_value_land_{unit}"],
        gdf=df,
        cmap="viridis",
        norm="log",
    )

    outpath = f"out/models/{model_group}/_images/"
    os.makedirs(outpath, exist_ok=True)

    return df


def _run_land_analysis(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    settings: dict,
    model_group: str,
    verbose: bool = False,
):
    instructions = settings.get("modeling", {}).get("instructions", {})
    allocation = instructions.get("allocation", {})

    sale_field = get_sale_field(settings, df_sales)

    results_map = {"main": {}, "hedonic": {}, "vacant": {}}
    sales_map = {"main": {}, "hedonic": {}, "vacant": {}}

    land_fields = []
    land_results: dict[str:SingleModelResults] = {}
    land_sales: dict[str:SingleModelResults] = {}

    # STEP 1: Gather results from the main, hedonic, and vacant models
    for key in ["main", "hedonic", "vacant"]:
        short_key = key[0]
        if key == "main":
            models = instructions.get("main", {}).get("run", [])
            skip = instructions.get("main", {}).get("skip", [])
            if model_group in skip:
                if "all" in skip[model_group]:
                    print(f"Skipping model group: {model_group}")
                    return
            if "ensemble" not in models:
                models.append("ensemble")
        else:
            models = allocation.get(key, [])
        
        path = key
        if key == "hedonic":
            path = "hedonic_land"

        outpath = f"out/models/{model_group}/{key}"

        if verbose:
            print(f"key = {key}")
        if len(models) > 0:
            for model in models:
                if verbose:
                    print(f"----> model = {model}")

                filepath = f"{outpath}/{model}"
                if os.path.exists(filepath):
                    fpred_univ = f"{filepath}/pred_universe.parquet"
                    fpred_sales = f"{filepath}/pred_sales.parquet"
                    if not os.path.exists(fpred_univ):
                        fpred_univ = f"{filepath}/pred_{model}_universe.parquet"
                        fpred_sales = f"{filepath}/pred_{model}_sales.parquet"
                    if os.path.exists(fpred_univ):
                        df_u = pd.read_parquet(fpred_univ)[["key", "prediction"]]
                        results_map[key][model] = df_u
                        df_s = pd.read_parquet(fpred_sales)
                        if "key_x" in df_s:
                            df_s = df_s.rename(columns={"key_x":"key"})
                        df_s = df_s[["key", "key_sale", "prediction"]]
                        sales_map[key][model] = df_s

                fpred_results = f"{filepath}/pred_universe.pkl"
                fpred_sales = f"{filepath}/pred_sales.pkl"
                if os.path.exists(fpred_results):
                    if key != "main":
                        with open(fpred_results, "rb") as file:
                            results = pickle.load(file)
                            land_results[f"{short_key}_{model}"] = results
                            land_fields.append(f"{short_key}_{model}")
                        with open(fpred_sales, "rb") as file:
                            results = pickle.load(file)
                            land_sales[f"{short_key}_{model}"] = results

    df_all_alloc = results_map["main"]["ensemble"].copy()
    df_all_alloc_sales = sales_map["main"]["ensemble"].copy()
    df_all_land_values = df_all_alloc.copy()
    df_all_land_values = df_all_land_values[["key"]].merge(
        df_universe, on="key", how="left"
    )
    df_all_land_sales = df_all_alloc_sales.copy()
    df_all_land_sales = df_all_land_sales[["key_sale"]].merge(
        df_sales, on="key_sale", how="left"
    )
    all_alloc_names = []

    bins = 400

    # STEP 2: Calculate land allocations for each model

    data_compare = {
        "type": [],
        "model": [],
        "count": [],
        "mape": [],
        "r2": [],
        "rmse": [],
        "alloc_median": [],
    }

    for key in ["hedonic", "vacant"]:
        short_key = key[0]
        df_alloc = results_map["main"]["ensemble"].copy()
        alloc_names = []
        entries = results_map[key]
        
        for model in entries:

            pred_main = results_map["main"].get(model).copy()
            pred_sales = sales_map["main"].get(model).copy()
            
            
            if pred_main is None:
                warnings.warn(
                    f"No main model found for model: {model}, using ensemble instead"
                )
                pred_main = results_map["main"].get("ensemble").copy()
                pred_sales = sales_map["main"].get("ensemble").copy()
            
            pred_land = (
                results_map[key]
                .get(model)
                .rename(columns={"prediction": "prediction_land"})
            )
            pred_sales_land = (
                sales_map[key]
                .get(model)
                .rename(columns={"prediction": "prediction_land"})
            )
            df = pred_main.merge(pred_land, on="key", how="left")
            dfs = pred_sales.merge(pred_sales_land[["key_sale", "prediction_land"]], on="key_sale", how="left")
            alloc_name = f"{short_key}_{model}"
            df.loc[:, alloc_name] = df["prediction_land"] / df["prediction"]
            dfs.loc[:, alloc_name] = dfs["prediction_land"] / df["prediction"]
            
            df_alloc = df_alloc.merge(df[["key", alloc_name]], on="key", how="left")
            df_all_alloc = df_all_alloc.merge(
                df[["key", alloc_name]], on="key", how="left"
            )
            df_all_alloc_sales = df_all_alloc_sales.merge(
                dfs[["key_sale", alloc_name]], on="key_sale", how="left"
            )

            df2 = df[["key","prediction_land"]].copy().rename(columns={"prediction_land": alloc_name})
            df2s = dfs[["key_sale","prediction_land"]].copy().rename(columns={"prediction_land": alloc_name})
            
            df_all_land_values = df_all_land_values.merge(
                df[["key", alloc_name]], on="key", how="left"
            )
            df_all_land_sales = df_all_land_sales.merge(
                df2s[["key_sale", alloc_name]], on="key_sale", how="left"
            )
            
            alloc_names.append(alloc_name)
            all_alloc_names.append(alloc_name)

            total_count = len(df)
            data_compare["type"].append(key)
            data_compare["model"].append(model)
            data_compare["count"].append(total_count)
            
            if sale_field not in df2s:
                df2s = df2s.merge(df_sales[["key_sale",sale_field]], on="key_sale", how="left")
            
            df2s_clean = df2s[~pd.isna(df2s[alloc_name]) & ~pd.isna(df2s[sale_field])]
            mape = mean_absolute_percentage_error(df2s_clean[alloc_name], df2s_clean[sale_field])
            mse, r2, _ = calc_mse_r2_adj_r2(df2s_clean[alloc_name], df2s_clean[sale_field], 1)
            rmse = np.sqrt(mse)

            data_compare["r2"].append(r2)
            data_compare["rmse"].append(rmse)
            data_compare["mape"].append(mape)
            data_compare["alloc_median"].append(
                np.round(100 * df[alloc_name].median()) / 100
            )

        df_compare = pd.DataFrame(data_compare)

        print("MODEL COMPARISON")
        display(df_compare)
        print("")

        df_alloc["allocation_ensemble"] = df_alloc[alloc_names].median(axis=1)

        plot_histogram_df(
            df=df_alloc,
            fields=alloc_names,
            xlabel="% of value attributable to land",
            ylabel="Number of parcels",
            title=f"({model_group}) Land allocation -- {key}",
            bins=bins,
            x_lim=(0.0, 1.0),
        )
        plot_histogram_df(
            df=df_alloc,
            fields=["allocation_ensemble"],
            xlabel="% of value attributable to land",
            ylabel="Number of parcels",
            title=f"({model_group}) Land allocation -- {key}, ensemble",
            bins=bins,
            x_lim=(0.0, 1.0),
        )

    plot_histogram_df(
        df=df_all_alloc,
        fields=all_alloc_names,
        xlabel="% of value attributable to land",
        ylabel="Number of parcels",
        title=f"({model_group}) Land allocation -- all",
        bins=bins,
        x_lim=(0.0, 1.0),
    )

    df_all_alloc["allocation_ensemble"] = df_all_alloc[all_alloc_names].median(axis=1)
    plot_histogram_df(
        df=df_all_alloc,
        fields=["allocation_ensemble"],
        xlabel="% of value attributable to land",
        ylabel="Number of parcels",
        title=f"({model_group}) Land allocation -- all, ensemble",
        bins=bins,
        x_lim=(0.0, 1.0),
    )

    # STEP 3: Optimize the ensemble allocation

    print(f"Putting it all together...")

    curr_ensemble = all_alloc_names
    best_score = float("inf")

    scores = {}

    for alloc_name in all_alloc_names:
        alloc = df_all_alloc[alloc_name]
        pct_neg = (alloc.lt(0)).sum() / len(alloc)
        pct_over = (alloc.gt(1)).sum() / len(alloc)
        score = pct_neg + (pct_over * 2.0)
        scores[alloc_name] = score

    print(f"Scores =\n{scores}")

    best_ensemble = None

    # Don't ensemble on assessor models:
    curr_ensemble = [col for col in curr_ensemble if "assessor" not in col]

    while len(curr_ensemble) > 0:
        alloc_ensemble = df_all_alloc[curr_ensemble].median(axis=1)
        pct_neg = (alloc_ensemble.lt(0)).sum() / len(alloc_ensemble)
        pct_over = (alloc_ensemble.gt(1)).sum() / len(alloc_ensemble)

        score = pct_neg + pct_over

        if score < best_score:
            best_score = score
            best_ensemble = curr_ensemble.copy()

        worst_score = -float("inf")
        worst_alloc = None
        for alloc_name in curr_ensemble:
            alloc_score = scores[alloc_name]
            if alloc_score > worst_score:
                worst_score = alloc_score
                worst_alloc = alloc_name

        if worst_alloc is not None:
            curr_ensemble.remove(worst_alloc)

        print(
            f"Ensemble score: {score:4.6f} (n:{pct_neg:4.2%} o:{pct_over:4.2%}), worst_score: {worst_score:4.2f}, eliminated: {worst_alloc}, best: {best_ensemble}"
        )

    if best_ensemble is None:
        print("No valid ensemble found, bailing...")
        return

    print(f"BEST ENSEMBLE = {best_ensemble}")
    print("LAND ENSEMBLE SCORE:")
    alloc_ensemble = df_all_alloc[best_ensemble].median(axis=1)
    pct_neg = (alloc_ensemble.lt(0)).sum() / len(alloc_ensemble)
    pct_over = (alloc_ensemble.gt(1)).sum() / len(alloc_ensemble)
    print(f"--> % neg : {pct_neg:4.2%}")
    print(f"--> % over: {pct_over:4.2%}")
    print(f"--> median: {alloc_ensemble.median():4.2%}")

    drop_alloc_names = [name for name in all_alloc_names if name not in best_ensemble]
    df_all_alloc = df_all_alloc.drop(columns=drop_alloc_names)

    plot_histogram_df(
        df=df_all_alloc,
        fields=best_ensemble,
        xlabel="% of value attributable to land",
        ylabel="Number of parcels",
        title=f"({model_group} Land allocation -- best ensemble (components)",
        bins=bins,
        x_lim=(0.0, 1.0),
    )

    df_all_alloc["allocation_ensemble"] = df_all_alloc[best_ensemble].median(axis=1)
    plot_histogram_df(
        df=df_all_alloc,
        fields=["allocation_ensemble"],
        xlabel="% of value attributable to land",
        ylabel="Number of parcels",
        title=f"{model_group} Land allocation -- best ensemble",
        bins=bins,
        x_lim=(0.0, 1.0),
    )

    # STEP 4: Finalize the results
    df_finalize = df_all_alloc.drop(columns=best_ensemble)
    df_finalize = df_finalize.rename(
        columns={
            "allocation_ensemble": "model_land_alloc",
            "prediction": "model_market_value",
        }
    )
    df_finalize = df_finalize.merge(
        df_universe[
            [
                "key",
                "geometry",
                "latitude",
                "longitude",
                f"land_area_{unit}",
                f"bldg_area_finished_{unit}",
            ]
        ],
        on="key",
        how="left",
    )

    df_finalize["model_land_value"] = (
        df_finalize["model_market_value"] * df_finalize["model_land_alloc"]
    )

    df_finalize = add_area_fields(df_finalize, settings)

    outpath = f"out/models/{model_group}/land_analysis.csv"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df_finalize.to_csv(outpath)

    gdf = GeoDataFrame(df_finalize, geometry="geometry", crs=df_universe.crs)
    gdf.to_parquet(f"out/models/{model_group}/land_analysis.parquet")


def _convolve_land_analysis(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    settings: dict,
    model_group: str,
    verbose: bool = False,
):
    instructions = settings.get("modeling", {}).get("instructions", {})
    allocation = instructions.get("allocation", {})

    results_map = {"main": {}, "hedonic": {}, "vacant": {}}

    land_fields = []
    land_results: dict[str:SingleModelResults] = {}

    train_keys, test_keys = get_train_test_keys(df_sales, settings)

    vacant_sales = (
        df_sales[df_sales["valid_for_land_ratio_study"].eq(True)]["key_sale"]
        .unique()
        .tolist()
    )
    df_vacant_sale = df_sales[
        df_sales["key_sale"].isin(vacant_sales)
        & df_sales["model_group"].eq(model_group)
    ].copy()

    # STEP 1: Gather results from the main, hedonic, and vacant models
    for key in ["main", "hedonic", "vacant"]:
        short_key = key[0]
        if key == "main":
            models = instructions.get("main", {}).get("run", [])
            skip = instructions.get("main", {}).get("skip", [])
            if model_group in skip:
                if "all" in skip[model_group]:
                    print(f"Skipping model group: {model_group}")
                    return
            if "ensemble" not in models:
                models.append("ensemble")
        else:
            models = allocation.get(key, [])
        outpath = f"out/models/{model_group}/{key}"

        if verbose:
            print(f"key = {key}")
        if len(models) > 0:
            for model in models:

                filepath = f"{outpath}/{model}"
                if os.path.exists(filepath):
                    fpred_univ = f"{filepath}/pred_universe.parquet"
                    if not os.path.exists(fpred_univ):
                        fpred_univ = f"{filepath}/pred_{model}_universe.parquet"

                    if os.path.exists(fpred_univ):
                        df_univ = pd.read_parquet(fpred_univ)[["key", "prediction"]]
                        results_map[key][model] = df_univ

                fpred_results = f"{filepath}/pred_universe.pkl"
                if os.path.exists(fpred_results):
                    if key != "main":
                        with open(fpred_results, "rb") as file:
                            results = pickle.load(file)
                            land_results[f"{short_key}_{model}"] = results
                            land_fields.append(f"{short_key}_{model}")

    sale_field = get_sale_field(settings)

    data_results = {
        "model": [],
        "r2_ols": [],
        "r2_y=x": [],
        "slope": [],
        "med_ratio": [],
        "cod": [],
    }

    data_results_test = {
        "model": [],
        "r2_ols": [],
        "r2_y=x": [],
        "slope": [],
        "med_ratio": [],
        "cod": [],
    }

    # STEP 2: Calculate smoothed values for each surface
    for full_or_test in ["full", "test"]:
        for key in ["hedonic", "vacant"]:
            entries = results_map[key]
            for model in entries:

                dfv = df_vacant_sale.copy()

                pred_main = None

                if pred_main is None:
                    pred_main = results_map["main"].get("ensemble")

                pred_land = (
                    results_map[key]
                    .get(model)
                    .rename(columns={"prediction": "prediction_land"})
                )
                df = pred_main.merge(pred_land, on="key", how="left")
                df = df.merge(
                    df_universe[
                        ["key", "latitude", "longitude", f"land_area_{unit}"]
                    ],
                    on="key",
                    how="left",
                )

                # Clamp land predictions to be non-negative and not exceed the main prediction
                df.loc[df["prediction_land"].lt(0), "prediction_land"] = 0.0
                df["prediction_land"] = df["prediction_land"].astype("Float64")
                df.loc[
                    df["prediction_land"].gt(df["prediction"]),
                    "prediction_land",
                ] = df["prediction"].astype("Float64")

                # Calculate land area per square foot of land
                df[f"prediction_land_{unit}"] = div_series_z_safe(
                    df["prediction_land"], df[f"land_area_{unit}"]
                )

                # Calculate the sale price per square foot of land
                sale_field_land_area = f"{sale_field}_land_{unit}"
                dfv[sale_field_land_area] = div_series_z_safe(
                    dfv[sale_field], dfv[f"land_area_{unit}"]
                )

                df["prediction_land_smooth"] = df["prediction_land"]

                df[f"prediction_land_smooth_{unit}"] = div_series_z_safe(
                    df["prediction_land_smooth"], df[f"land_area_{unit}"]
                )

                dfv = dfv.merge(
                    df[
                        [
                            "key",
                            "prediction_land_smooth",
                            f"prediction_land_smooth_{unit}",
                        ]
                    ],
                    on="key",
                    how="left",
                )
                dfv = dfv[
                    ~dfv["prediction_land_smooth"].isna()
                    & ~dfv[f"prediction_land_smooth_{unit}"].isna()
                    & ~dfv[sale_field_land_area].isna()
                ]

                if full_or_test == "test":
                    dfv = dfv[dfv["key_sale"].isin(test_keys)]
                if len(dfv) == 0:
                    continue

                dfv["sales_ratio"] = div_series_z_safe(
                    dfv["prediction_land_smooth"], dfv[sale_field]
                )

                median_ratio = dfv["sales_ratio"].median()
                cod = calc_cod(dfv["sales_ratio"].values)

                results = simple_ols(
                    dfv, "prediction_land_smooth", "sale_price", intercept=True
                )

                slope = results["slope"]
                r2 = results["r2"]

                y_true = dfv["sale_price"].values
                y_pred = dfv["prediction_land_smooth"].values

                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2_yx = 1 - (ss_res / ss_tot)

                if full_or_test == "full":
                    data_results["model"].append(
                        f"{key}_{model}"
                    )
                    data_results["slope"].append(f"{slope:.2f}")
                    data_results["r2_ols"].append(f"{r2:.2f}")
                    data_results["r2_y=x"].append(f"{r2_yx:.2f}")
                    data_results["med_ratio"].append(f"{median_ratio:.2f}")
                    data_results["cod"].append(f"{cod:.1f}")
                else:
                    data_results_test["model"].append(
                        f"{key}_{model}"
                    )
                    data_results_test["slope"].append(f"{slope:.2f}")
                    data_results_test["r2_ols"].append(f"{r2:.2f}")
                    data_results_test["r2_y=x"].append(f"{r2_yx:.2f}")
                    data_results_test["med_ratio"].append(f"{median_ratio:.2f}")
                    data_results_test["cod"].append(f"{cod:.1f}")

    df_results = pd.DataFrame(data_results)

    df_results["r2_"] = np.floor(df_results["r2_y=x"].astype("float").fillna(0.0) * 10)
    df_results["slope_"] = np.abs(1.0 - df_results["slope"].astype("float").fillna(0.0))

    df_results = df_results.sort_values(by=["r2_", "slope_"], ascending=False)
    df_results = df_results.drop(columns=["r2_", "slope_"])
    df_results = df_results.reset_index(drop=True)

    count = len(df_vacant_sale)
    print("=" * 80)
    print(f"FULL LAND RESULTS, MODEL GROUP : {model_group}, count: {count}")
    print("=" * 80)
    print(df_results.to_string())
    print("")

    df_results_test = pd.DataFrame(data_results_test)
    if len(df_results_test) > 0:
        df_results_test["r2_"] = np.floor(
            df_results_test["r2_y=x"].astype("float").fillna(0.0) * 10
        )
        df_results_test["slope_"] = np.abs(
            1.0 - df_results_test["slope"].astype("float").fillna(0.0)
        )
        df_results_test = df_results_test.sort_values(
            by=["r2_", "slope_"], ascending=False
        )
        df_results_test = df_results_test.drop(columns=["r2_", "slope_"])
        df_results_test = df_results_test.reset_index(drop=True)
        
        count = len(df_vacant_sale[df_vacant_sale["key_sale"].isin(test_keys)])
        print("=" * 80)
        print(f"TEST LAND RESULTS, MODEL GROUP : {model_group}, count: {count}")
        print("=" * 80)
        print(df_results_test.to_string())
        print("")


def calc_lycd_land_values(
    df: pd.DataFrame,
    settings: dict,
    land_alloc: "float | dict | None" = None,
    market_value_field: str = "model_market_value",
    subarea_field: "str | None" = None,
    min_improved_per_cell: int = 10,
) -> pd.DataFrame:
    """Compute land values using the "Least You Can Do" (LYCD) method.

    For each local area this method:

    1. Takes the median market value and median lot size of **improved**
       (non-vacant) properties in the area.
    2. Derives a uniform local land rate::

           local_land_rate = (median_market_value * land_alloc_pct) / median_lot_size

    3. Applies that rate to every parcel in the area::

           land_value = local_land_rate * parcel_lot_size

    Because every parcel's land value is driven by the *typical* parcel in its
    area rather than its own improvement value, the method avoids the absurd
    side-by-side disparities that arise from naively multiplying each parcel's
    market value by a fixed allocation fraction.

    When ``subarea_field`` is provided, rates are computed per
    ``(subarea, model_group)`` cell rather than per model group alone.  This
    captures the sharp neighbourhood-to-neighbourhood price jumps that a single
    county-wide rate per land-use type would miss.  Cells with fewer than
    ``min_improved_per_cell`` improved parcels fall back first to the
    model-group rate, then to the global rate.

    When ``land_alloc`` is ``None`` the allocation for each cell is derived
    automatically from the data.  The median per-unit value of **vacant**
    properties is divided by the median per-unit value of **improved**
    properties; that ratio is the implied land allocation.  Fallback order:
    cell → model group → global.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at minimum:

        - ``"key"``
        - ``"model_group"``
        - ``market_value_field`` (default ``"model_market_value"``)
        - ``"is_vacant"`` (bool)
        - ``f"land_area_{unit}"`` where *unit* is ``"sqft"`` or ``"sqm"``
        - ``subarea_field`` column, if provided

    settings : dict
        Settings dictionary (used to determine the area unit).
    land_alloc : float, dict, or None
        Fraction of market value attributable to land.

        - **float** – applied uniformly to all cells.
        - **dict** – keyed by ``model_group`` (``{mg: float}``) for per-group
          allocations; individual cells inherit their group's allocation.
        - **None** – derived automatically from the ratio of vacant-to-improved
          per-unit market values, at the finest available level.

    market_value_field : str
        Column name holding the market value estimate.  Defaults to
        ``"model_market_value"``.
    subarea_field : str or None
        Column name whose values define geographic subareas (e.g.
        ``"neighborhood"``, ``"census_tract"``).  When supplied, one local land
        rate is derived per ``(subarea, model_group)`` cell.  When ``None``
        (default), a single rate is derived per ``model_group``, which is
        equivalent to treating the entire county as one area per land-use type.
    min_improved_per_cell : int
        Minimum number of improved parcels required in a
        ``(subarea, model_group)`` cell before that cell gets its own rate.
        Cells below this threshold fall back to the model-group rate.
        Ignored when ``subarea_field`` is ``None``.  Default: 10.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with three additional columns:

        - ``"lycd_land_alloc"`` – land allocation fraction used for the cell.
        - ``"lycd_local_land_rate"`` – implied per-area-unit land rate for the cell.
        - ``"lycd_land_value"`` – resulting land value (clamped to ``[0, market_value]``).
    """
    unit = area_unit(settings)
    lot_area_field = f"land_area_{unit}"

    df = df.copy()
    model_groups = df["model_group"].unique()

    # ------------------------------------------------------------------
    # Resolve land_alloc to a per-model-group dict (used as fallback even
    # when subarea_field is set, and as the primary source when it is not).
    # ------------------------------------------------------------------
    if isinstance(land_alloc, dict):
        alloc_by_group = {mg: land_alloc.get(mg, np.nan) for mg in model_groups}
    elif land_alloc is not None:
        alloc_by_group = {mg: float(land_alloc) for mg in model_groups}
    else:
        alloc_by_group = _derive_lycd_alloc_from_data(
            df, lot_area_field, market_value_field, group_field="model_group"
        )

    # ------------------------------------------------------------------
    # Helper: compute one local land rate for a slice of the dataframe.
    # Returns np.nan if the slice is too small or the medians are invalid.
    # ------------------------------------------------------------------
    def _rate_for_mask(mask_improved, alloc):
        sub = df[mask_improved]
        if len(sub) == 0 or np.isnan(alloc):
            return np.nan
        median_mv = sub[market_value_field].median()
        median_lot = sub[lot_area_field].median()
        if median_lot <= 0 or np.isnan(median_lot) or np.isnan(median_mv):
            return np.nan
        return (median_mv * alloc) / median_lot

    # ------------------------------------------------------------------
    # Build per-model-group fallback rates (always needed).
    # ------------------------------------------------------------------
    group_rates = {}
    for mg in model_groups:
        mask = df["model_group"].eq(mg) & df["is_vacant"].eq(False)
        group_rates[mg] = _rate_for_mask(mask, alloc_by_group.get(mg, np.nan))

    # ------------------------------------------------------------------
    # Assign rates and allocs row-by-row into output columns.
    # ------------------------------------------------------------------
    df["lycd_land_alloc"] = np.nan
    df["lycd_local_land_rate"] = np.nan

    if subarea_field is None:
        # No geographic subdivision: one rate per model group.
        for mg in model_groups:
            idx = df["model_group"].eq(mg)
            df.loc[idx, "lycd_local_land_rate"] = group_rates[mg]
            df.loc[idx, "lycd_land_alloc"] = alloc_by_group.get(mg, np.nan)
    else:
        # Geographic subdivision: one rate per (subarea, model_group) cell,
        # with fallback to model-group rate for thin cells.
        subareas = df[subarea_field].unique()

        # Pre-derive per-cell allocations when land_alloc is None.
        if land_alloc is None:
            cell_allocs = _derive_lycd_alloc_from_data(
                df,
                lot_area_field,
                market_value_field,
                group_field="model_group",
                subarea_field=subarea_field,
                alloc_by_group_fallback=alloc_by_group,
            )
        else:
            cell_allocs = {}  # will use alloc_by_group for every cell

        for mg in model_groups:
            mask_mg = df["model_group"].eq(mg)
            for sa in subareas:
                mask_sa = df[subarea_field].eq(sa)
                mask_cell_improved = mask_mg & mask_sa & df["is_vacant"].eq(False)
                n_improved = mask_cell_improved.sum()
                idx = mask_mg & mask_sa

                if n_improved >= min_improved_per_cell:
                    cell_key = (sa, mg)
                    alloc = cell_allocs.get(cell_key, alloc_by_group.get(mg, np.nan))
                    rate = _rate_for_mask(mask_cell_improved, alloc)
                    if np.isnan(rate):
                        # Cell medians invalid despite enough rows; fall back.
                        rate = group_rates[mg]
                        alloc = alloc_by_group.get(mg, np.nan)
                else:
                    # Too few improved parcels in this cell; use group rate.
                    rate = group_rates[mg]
                    alloc = alloc_by_group.get(mg, np.nan)

                df.loc[idx, "lycd_local_land_rate"] = rate
                df.loc[idx, "lycd_land_alloc"] = alloc

    df["lycd_land_value"] = df["lycd_local_land_rate"] * df[lot_area_field]

    # Clamp: must be >= 0 and <= market value
    df["lycd_land_value"] = df["lycd_land_value"].clip(lower=0)
    exceeds_mv = df["lycd_land_value"].gt(df[market_value_field])
    df.loc[exceeds_mv, "lycd_land_value"] = df.loc[exceeds_mv, market_value_field]

    return df


def _derive_lycd_alloc_from_data(
    df: pd.DataFrame,
    lot_area_field: str,
    market_value_field: str,
    group_field: str = "model_group",
    subarea_field: "str | None" = None,
    alloc_by_group_fallback: "dict | None" = None,
) -> dict:
    """Derive land allocation fractions from vacant vs improved per-unit values.

    When ``subarea_field`` is None, returns ``{model_group: alloc}``.
    When ``subarea_field`` is provided, returns ``{(subarea, model_group): alloc}``,
    falling back to the group-level allocation for cells with no vacant parcels,
    and to the global allocation if the group also has none.
    """
    def _per_unit_median(mask):
        sub = df[mask].copy()
        sub = sub[sub[lot_area_field].gt(0)]
        rates = sub[market_value_field] / sub[lot_area_field]
        return rates.median()

    # Global fallback
    global_vacant_rate = _per_unit_median(df["is_vacant"].eq(True))
    global_improved_rate = _per_unit_median(df["is_vacant"].eq(False))
    if global_improved_rate > 0 and not (
        np.isnan(global_vacant_rate) or np.isnan(global_improved_rate)
    ):
        global_alloc = global_vacant_rate / global_improved_rate
    else:
        global_alloc = np.nan

    # Per-group allocations (always computed; used as fallback for cells).
    group_allocs = {}
    for mg in df[group_field].unique():
        mask_mg = df[group_field].eq(mg)
        vr = _per_unit_median(mask_mg & df["is_vacant"].eq(True))
        ir = _per_unit_median(mask_mg & df["is_vacant"].eq(False))
        if ir > 0 and not (np.isnan(vr) or np.isnan(ir)):
            group_allocs[mg] = vr / ir
        else:
            group_allocs[mg] = (
                alloc_by_group_fallback.get(mg, global_alloc)
                if alloc_by_group_fallback
                else global_alloc
            )

    if subarea_field is None:
        return group_allocs

    # Per-(subarea, group) allocations with group → global fallback.
    result = {}
    for mg in df[group_field].unique():
        mask_mg = df[group_field].eq(mg)
        for sa in df[subarea_field].unique():
            mask_sa = df[subarea_field].eq(sa)
            vr = _per_unit_median(mask_mg & mask_sa & df["is_vacant"].eq(True))
            ir = _per_unit_median(mask_mg & mask_sa & df["is_vacant"].eq(False))
            if ir > 0 and not (np.isnan(vr) or np.isnan(ir)):
                result[(sa, mg)] = vr / ir
            else:
                result[(sa, mg)] = group_allocs.get(mg, global_alloc)

    return result