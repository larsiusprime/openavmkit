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


def trial_simple_plane(params: dict):

    random.seed(params.get("seed", 1337))

    # Generate the plane
    gdf = simple_plane(params)
    gdf = _basic_geo_enrichment(gdf, {})
    gdf = add_polar_neighborhoods(gdf, [(2, 1), (4, 1), (8, 1), (16, 1), (32, 1)])

    # Add land values
    land_curve = params["land_value_curve"]
    base_value = params["land_value_base"]
    size_field = params["land_value_size_field"]
    perc_sales = params["perc_sales"]
    perc_error = params["perc_sales_error"]

    gdf["model_group"] = "test"

    print("************************")
    print(f"LAND CURVE = {land_curve}")
    print("************************")

    gdf["true_land_value"] = add_simple_land_value(
        gdf, {"curve": land_curve, "base_value": base_value, "size_field": size_field}
    )
    gdf["true_market_value"] = gdf["true_land_value"]

    df_univ = gdf.copy()

    # Add transactions
    gdf = add_simple_transactions(
        gdf,
        {
            "perc_sales": perc_sales,
            "perc_error": perc_error,
            "value_field": "true_market_value",
        },
    )

    df_sales = gdf[gdf["valid_sale"] == True].copy().reset_index(drop=True)
    # There's no buildings so every sale is a vacant sale
    df_sales["vacant_sale"] = True
    df_sales["valid_for_ratio_study"] = True
    df_sales["valid_for_land_ratio_study"] = True

    # Add horizontal equity ids
    he_settings = {
        "analysis": {
            "horizontal_equity": {
                "location": "loc_polar_16",
                "fields_numeric": ["dist_to_centroid"],
            }
        }
    }
    df_univ = mark_horizontal_equity_clusters(df_univ, he_settings, verbose=False)

    # Package as SUP
    sup = SalesUniversePair(df_sales, df_univ)

    # Enrich with spatial lag
    sup = enrich_sup_spatial_lag(sup, {})

    return sup


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


def trial_simple_plane_w_buildings(params: dict):

    random.seed(params.get("seed", 1337))

    # Generate the plane
    gdf = simple_plane_w_buildings(params)
    gdf = _basic_geo_enrichment(gdf, {})
    gdf = add_polar_neighborhoods(gdf, [(2, 1), (4, 1), (8, 1), (16, 1), (32, 1)])

    # Add land values
    land_curve = params["land_value_curve"]
    base_value = params["land_value_base"]
    size_field = params["land_value_size_field"]
    perc_sales = params["perc_sales"]
    perc_error = params["perc_sales_error"]
    units = params["units"]

    gdf["model_group"] = "test"

    gdf["true_land_value"] = add_simple_land_value(
        gdf, {"curve": land_curve, "base_value": base_value, "size_field": size_field}
    )
    gdf["true_impr_value"] = add_simple_bldg_value(
        gdf,
        {
            "base_value": params["bldg_value_base"],
            "size_field": f"bldg_area_finished_sq{units}",
        },
    )

    gdf["true_market_value"] = gdf["true_land_value"] + gdf["true_impr_value"]

    df_univ = gdf.copy()

    # Add transactions
    gdf = add_simple_transactions(
        gdf,
        {
            "perc_sales": perc_sales,
            "perc_error": perc_error,
            "value_field": "true_market_value",
        },
    )

    df_sales = gdf[gdf["valid_sale"] == True].copy().reset_index(drop=True)

    df_sales["vacant_sale"] = df_sales["valid_sale"].eq(True) & df_sales[
        "is_vacant"
    ].eq(True)
    df_sales["valid_for_ratio_study"] = df_sales["valid_sale"].eq(True)
    df_sales["valid_for_land_ratio_study"] = df_sales["valid_sale"].eq(True) & df_sales[
        "is_vacant"
    ].eq(True)

    # Add horizontal equity ids
    he_settings = {
        "analysis": {
            "horizontal_equity": {
                "location": "loc_polar_16",
                "fields_numeric": ["dist_to_centroid", f"bldg_area_finished_sq{units}"],
            },
            "land_equity": {
                "location": "loc_polar_16",
                "fields_numeric": ["dist_to_centroid", "polar_angle"],
            },
            "impr_equity": {"fields_numeric": [f"bldg_area_finished_sq{units}"]},
        }
    }

    df_univ = mark_horizontal_equity_clusters(df_univ, he_settings, verbose=False)

    df_univ = mark_horizontal_equity_clusters(
        df_univ,
        he_settings,
        verbose=False,
        settings_object="land_equity",
        id_name="land_he_id",
    )

    df_univ = mark_horizontal_equity_clusters(
        df_univ,
        he_settings,
        verbose=False,
        settings_object="impr_equity",
        id_name="impr_he_id",
    )

    # Package as SUP
    sup = SalesUniversePair(df_sales, df_univ)

    # Enrich with spatial lag
    sup = enrich_sup_spatial_lag(sup, {})

    return sup


def run_trials(sup_generator: callable, params: dict, variations: dict):

    verbose = params.get("verbose", False)

    if verbose:
        print(f"RUNNING TRIALS")

    trial_results = {}
    trial_vacant_results = {}
    trial_hedonic_results = {}
    sups = {}

    i = 0
    for variation in variations:
        if verbose:
            print("***************************")
            print(f"TRIAL {i} --> {variation['id']}")
            print("***************************")
        p = params.copy()
        id = variation["id"]
        for key in variation:
            p[key] = variation[key]

        outpath = p["outpath"]

        p["outpath"] = outpath + "/main/" + id
        p["hedonic_outpath"] = outpath + "/hedonic/" + id
        sup = sup_generator(p)

        # Calculate % of vacant sales:
        perc_vacant_sales = sup.sales["vacant_sale"].sum() / len(sup.sales)

        # Run the actual trial:
        results, hedonic_results = run_one_trial(sup, p)

        trial_results[id] = results
        if hedonic_results is not None:
            trial_hedonic_results[id] = hedonic_results

        if perc_vacant_sales < 1.0:
            if perc_vacant_sales > 0.0:
                p["vacant_only"] = True
                p["hedonic"] = False
                p["outpath"] = outpath + "/vacant/" + id + "_vacant"
                results, _ = run_one_trial(sup, p)
                trial_vacant_results[id] = results

        sups[id] = sup
        i += 1

    main_outpath = params["outpath"] + "/summary/main"
    hedonic_outpath = params["outpath"] + "/summary/hedonic"
    vacant_outpath = params["outpath"] + "/summary/vacant"

    verbose = True

    write_trial_results(trial_results, sups, main_outpath, "main", verbose=verbose)
    if len(trial_hedonic_results) > 0:
        write_trial_results(
            trial_hedonic_results, sups, hedonic_outpath, "hedonic", verbose=verbose
        )
    if len(trial_vacant_results) > 0:
        write_trial_results(
            trial_vacant_results, sups, vacant_outpath, "vacant", verbose=verbose
        )

    for id in trial_results:
        main_results = trial_results.get(id, None)
        vacant_results = trial_vacant_results.get(id, None)
        hedonic_results = trial_hedonic_results.get(id, None)

        if vacant_results is not None or hedonic_results is not None:

            sup = sups[id]
            sale_field = get_sale_field({}, sup.sales)
            evaluate_trial_land_results(
                id,
                sup,
                "test",
                {},
                main_results,
                vacant_results,
                hedonic_results,
                params["outpath"],
                sale_field,
                "true_market_value",
                "true_land_value",
                verbose,
            )


def evaluate_trial_one_land_result(
    model_id: str,
    sup: SalesUniversePair,
    total_results: SingleModelResults,
    land_results: SingleModelResults,
    sale_field: str,
    ground_truth_land_field: str,
    max_trim: float,
    verbose: bool = False,
):
    sup = sup.copy()

    df_univ = sup.universe
    df_univ["model_market_value"] = total_results.pred_univ
    df_univ["model_land_value"] = land_results.pred_univ
    df_univ["model_impr_value"] = (
        df_univ["model_market_value"] - df_univ["model_land_value"]
    )

    df_sales = sup.sales

    sup.set("universe", df_univ)

    scores = LandPredictionResults(
        "model_land_value",
        "model_impr_value",
        "model_market_value",
        sale_field,
        land_results.ind_vars,
        sup,
        max_trim
    )

    scores_truth = LandPredictionResults(
        "model_land_value",
        "model_impr_value",
        "model_market_value",
        ground_truth_land_field,
        land_results.ind_vars,
        sup,
        max_trim
    )

    return {
        "model": model_id,
        " ": "",
        "count": len(df_univ),
        "sales": len(df_sales[df_sales["valid_sale"].eq(True)]),
        "land_sales": len(df_sales[df_sales["valid_for_land_ratio_study"].eq(True)]),
        "utility_score": scores.utility_score,
        "  ": "",
        "true_r2": scores_truth.r2,
        "true_adj_r2": scores_truth.adj_r2,
        "true_prb": scores_truth.prb,
        "r2": scores.r2,
        "adj_r2": scores.adj_r2,
        "prb": scores.prb,
        "   ": "",
        "true_med_ratio": scores_truth.land_ratio_study.median_ratio,
        "true_cod": scores_truth.land_ratio_study.cod,
        "true_cod_trim": scores_truth.land_ratio_study.cod_trim,
        "med_ratio": scores.land_ratio_study.median_ratio,
        "cod": scores.land_ratio_study.cod,
        "cod_trim": scores.land_ratio_study.cod_trim,
        "    ": "",
        "chd_total": scores.total_chd,
        "chd_impr": scores.impr_chd,
        "chd_land": scores.land_chd,
        "     ": "",
        "null": scores.perc_land_null,
        "neg": scores.perc_land_negative,
        "bad_sum": scores.perc_dont_add_up,
        "land_over": scores.perc_land_overshoot,
        "vac_not_100": scores.perc_vacant_land_not_100,
    }


def evaluate_trial_land_results(
    trial_id: str,
    sup: SalesUniversePair,
    model_group: str,
    settings: dict,
    main: MultiModelResults,
    vacant: MultiModelResults | None,
    hedonic: MultiModelResults | None,
    outpath: str,
    sale_field: str,
    ground_truth_field: str,
    ground_truth_land_field: str,
    verbose: bool = False,
):
    main_outpath = f"{outpath}/main/{trial_id}"
    hedonic_outpath = f"{outpath}/hedonic/{trial_id}"
    vacant_outpath = f"{outpath}/vacant/{trial_id}"

    vacant_ensemble_list: dict[str] | None = None
    hedonic_ensemble_list: dict[str] | None = None
    vacant_ensemble: SingleModelResults | None = None
    hedonic_ensemble: SingleModelResults | None = None

    main_ensemble, main_ensemble_list = run_ensemble(
        sup.sales,
        sup.universe,
        model_group=model_group,
        vacant_only=False,
        dep_var=ground_truth_field,
        dep_var_test=ground_truth_field,
        outpath=main_outpath,
        all_results=main,
        settings=settings,
        verbose=verbose,
        hedonic=False,
    )

    rows_h = []
    rows_v = []

    if vacant is not None:
        for key in vacant.model_results:
            land_smr = vacant.model_results[key]
            if key in main.model_results:
                main_smr = main.model_results[key]
            else:
                main_smr = main_ensemble
                raise ValueError(
                    f"Couldn't find '{key}' in main model results, using ensemble instead"
                )
            if land_smr.pred_univ is None or main_smr.pred_univ is None:
                continue

            non_null_land_univ_count = len(
                land_smr.pred_univ[~pd.isna(land_smr.pred_univ)]
            )
            non_null_main_univ_count = len(
                main_smr.pred_univ[~pd.isna(main_smr.pred_univ)]
            )
            if non_null_land_univ_count == 0 or non_null_main_univ_count == 0:
                continue
            
            max_trim = _get_max_ratio_study_trim(settings, model_group)
            data = evaluate_trial_one_land_result(
                key,
                sup,
                main_smr,
                land_smr,
                sale_field,
                ground_truth_land_field,
                verbose,
                max_trim
            )
            rows_v.append(data)

    if hedonic is not None:
        for key in hedonic.model_results:
            land_smr = hedonic.model_results[key]
            main_smr = main.model_results[key]
            if land_smr.pred_univ is None or main_smr.pred_univ is None:
                continue

            non_null_land_univ_count = len(
                land_smr.pred_univ[~pd.isna(land_smr.pred_univ)]
            )
            non_null_main_univ_count = len(
                main_smr.pred_univ[~pd.isna(main_smr.pred_univ)]
            )

            if non_null_land_univ_count == 0 or non_null_main_univ_count == 0:
                continue
            
            max_trim = _get_max_ratio_study_trim(settings, model_group)
            data = evaluate_trial_one_land_result(
                key,
                sup,
                main_smr,
                land_smr,
                sale_field,
                ground_truth_land_field,
                verbose,
            )
            rows_h.append(data)

    for rows, m_type in [(rows_h, "hedonic"), (rows_v, "vacant")]:
        if len(rows) > 0:
            df_results = pd.DataFrame(rows)
            df_results.sort_values(by="utility_score", ascending=True, inplace=True)

            print("")
            print("************************")
            print(f"LAND RESULTS FOR: {trial_id}, {m_type}")
            print("************************")

            # set "model" as index:
            df_results = df_results.set_index("model")
            print(_format_benchmark_df(df_results))

    all_rows = rows_h + rows_v
    df_results = pd.DataFrame(all_rows)
    if len(df_results) > 0:
        df_results.sort_values(by="utility_score", ascending=True, inplace=True)
        df_results.to_csv(f"{main_outpath}/land_results.csv", index=False)


def write_trial_results(
    trial_results: dict, sups: dict, outpath: str, id: str, verbose: bool = False
):

    df_stats_full: pd.DataFrame | None = None
    df_stats_test: pd.DataFrame | None = None
    df_time: pd.DataFrame | None = None

    def col_first(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index(col_name)))
        return df[cols]

    # summarize the results
    for key in trial_results:
        results = trial_results[key]
        if results is None:
            continue
        sup = sups[key]

        if verbose:
            print("")
            print("************************")
            print(f"RESULTS FOR: {id} / {key}")
            print("************************")

            print(results.benchmark.print())

        if df_stats_full is None:
            df_stats_full = results.benchmark.df_stats_full
            df_stats_full = df_stats_full.reset_index()
            df_stats_full["variant"] = key
            df_stats_full = col_first(df_stats_full, "variant")
        else:
            other = results.benchmark.df_stats_full
            other = other.reset_index()
            other["variant"] = key
            other = col_first(other, "variant")
            df_stats_full = pd.concat([df_stats_full, other], ignore_index=True)

        if df_stats_test is None:
            df_stats_test = results.benchmark.df_stats_test
            df_stats_test = df_stats_test.reset_index()
            df_stats_test["variant"] = key
            df_stats_test = col_first(df_stats_test, "variant")
        else:
            other = results.benchmark.df_stats_test
            other = other.reset_index()
            other["variant"] = key
            other = col_first(other, "variant")
            df_stats_test = pd.concat([df_stats_test, other], ignore_index=True)

        if df_time is None:
            df_time = results.benchmark.df_time
            df_time = df_time.reset_index()
            df_time["variant"] = key
            df_time = col_first(df_time, "variant")
        else:
            other = results.benchmark.df_time
            other = other.reset_index()
            other["variant"] = key
            other = col_first(other, "variant")
            df_time = pd.concat([df_time, other], ignore_index=True)

        os.makedirs(outpath, exist_ok=True)
        df_stats_full.to_csv(f"{outpath}/stats_full.csv")
        df_stats_test.to_csv(f"{outpath}/stats_test.csv")
        df_time.to_csv(f"{outpath}/time.csv")

        os.makedirs(f"{outpath}/datasets", exist_ok=True)
        sup.sales.to_parquet(f"{outpath}/datasets/{key}_sales.parquet")
        sup.universe.to_parquet(f"{outpath}/datasets/{key}_universe.parquet")


def run_one_trial(sup: SalesUniversePair, params: dict):
    # Get universe and sales
    df_univ = sup.universe
    df_sales = sup.sales

    test_train_frac = params["train_frac"]

    # split test and train
    df_test, df_train = _perform_canonical_split(
        "test", df_sales, {}, test_train_fraction=test_train_frac
    )
    test_keys = df_test["key_sale"]
    train_keys = df_train["key_sale"]

    models = params["models"]
    ind_vars = params["ind_vars"]
    cat_vars = params["cat_vars"]

    model_entries = {"default": {"ind_vars": ind_vars}}

    outpath = params["outpath"]
    verbose = params["verbose"]
    dep_var_test = params.get("dep_var_test", "sale_price")
    dep_var_test_hedonic = params.get("dep_var_test_hedonic", "sale_price")
    model_results = {}

    settings = {
        "field_classification": {"important": {"locations": params["locations"]}}
    }

    vacant_only = params.get("vacant_only", False)
    hedonic = params.get("hedonic", False)
    hedonic_outpath = params["hedonic_outpath"]

    for model in models:
        results = run_one_model(
            df_sales=df_sales,
            df_universe=df_univ,
            vacant_only=vacant_only,
            model_group="test",
            model=model,
            model_entries=model_entries,
            settings=settings,
            dep_var="sale_price",
            dep_var_test=dep_var_test,
            best_variables=ind_vars,
            fields_cat=cat_vars,
            outpath=outpath,
            save_params=True,
            use_saved_params=True,
            save_results=False,
            verbose=verbose,
            hedonic=False,
            test_keys=test_keys,
            train_keys=train_keys,
        )
        if results is not None:
            model_results[model] = results

    hedonic_results = None
    if hedonic:
        hedonic_results = {}
        hedonic_test_against_vacant_sales = "sale_price" in dep_var_test_hedonic
        print(
            f"HEDONIC MODEL: test against : {dep_var_test_hedonic}, test against sales? {hedonic_test_against_vacant_sales}"
        )
        for model in models:
            smr = model_results[model]
            if smr is not None:
                results = run_one_hedonic_model(
                    df_sales=df_sales,
                    df_univ=df_univ,
                    settings=settings,
                    model=model,
                    smr=smr,
                    model_group="test",
                    dep_var="sale_price",
                    dep_var_test=dep_var_test_hedonic,
                    fields_cat=cat_vars,
                    outpath=hedonic_outpath,
                    hedonic_test_against_vacant_sales=hedonic_test_against_vacant_sales,
                    save_results=False,
                    verbose=verbose,
                )
                if results is not None:
                    hedonic_results[model] = results

    all_results = MultiModelResults(
        model_results=model_results, benchmark=_calc_benchmark(model_results)
    )

    all_hedonic_results = None
    if hedonic_results is not None:
        all_hedonic_results = MultiModelResults(
            model_results=hedonic_results, benchmark=_calc_benchmark(hedonic_results)
        )

    return all_results, all_hedonic_results


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


# def _do_perform_distance_calculations(df_in: gpd.GeoDataFrame, gdf_in: gpd.GeoDataFrame, _id: str, unit: str = "km") -> pd.DataFrame:
#   """
#   Perform a divide-by-zero-safe nearest neighbor spatial join to calculate distances.
#
#   :param df_in: Base GeoDataFrame.
#   :type df_in: geopandas.GeoDataFrame
#   :param gdf_in: Overlay GeoDataFrame.
#   :type gdf_in: geopandas.GeoDataFrame
#   :param _id: Identifier used for naming the distance column.
#   :type _id: str
#   :param unit: Unit for distance conversion (default "km").
#   :type unit: str, optional
#   :returns: DataFrame with an added distance column.
#   :rtype: pandas.DataFrame
#   :raises ValueError: If an unsupported unit is specified.
#   """
#   unit_factors = {"m": 1, "km": 0.001, "mile": 0.000621371, "ft": 3.28084}
#   if unit not in unit_factors:
#     raise ValueError(f"Unsupported unit '{unit}'")
#   crs = get_crs(df_in, "equal_distance")
#   df_projected = df_in.to_crs(crs).copy()
#   gdf_projected = gdf_in.to_crs(crs).copy()
#   nearest = gpd.sjoin_nearest(df_projected, gdf_projected, how="left", distance_col=f"dist_to_{_id}")[["key", f"dist_to_{_id}"]]
#   nearest[f"dist_to_{_id}"] *= unit_factors[unit]
#   n_duplicates_nearest = nearest.duplicated(subset="key").sum()
#   n_duplicates_df = df_in.duplicated(subset="key").sum()
#   if n_duplicates_df > 0:
#     raise ValueError(f"Found {n_duplicates_nearest} duplicate keys in the base dataframe, cannot perform distance calculations. Please de-duplicate your dataframes and try again.")
#   if n_duplicates_nearest > 0:
#     nearest = nearest.sort_values(by=["key", f"dist_to_{_id}"], ascending=[True, True])
#     nearest = nearest.drop_duplicates(subset="key")
#   df_out = df_in.merge(nearest, on="key", how="left")
#   return df_out
