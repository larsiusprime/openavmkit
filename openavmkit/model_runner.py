"""
Model-run orchestration (the high-level coordinator that drives modeling).

Runs the full configured model menu (MRA, GWR, XGBoost, LightGBM, CatBoost,
NGBoost, layered comps, kernel regression, ensembles, and several "naive"
baselines) across each model group, with optional variable-importance
experiments, then compares model outputs (the "benchmark" comparison) and
produces ensemble predictions.

This module is the high-level coordinator that drives :mod:`openavmkit.modeling`.
The main entry points (``run_models``, ``try_variables``, ``try_models``,
``finalize_models``, ensemble runners) are exposed as wrappers in
:mod:`openavmkit.pipeline`.

History
-------
Formerly named ``openavmkit.benchmark``; renamed to ``openavmkit.model_runner``
because the module orchestrates the whole model run, not only the benchmark
comparison (and to avoid confusion with the research ``benchmark/`` harness). A
deprecating compatibility shim remains at ``openavmkit.benchmark`` and is slated
for removal before the 0.7.0 release.

Notes
-----
The list of models to run for each main/vacant stage is configured in
``settings.json`` under ``modeling.instructions.<stage>.run``. Per-model-group
skip lists live under ``modeling.instructions.<stage>.skip.<model_group>``.
"""
import os
import json
import pickle
import warnings
import math

from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
from catboost import CatBoostRegressor
from lightgbm import Booster
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.nonparametric.kernel_regression import KernelReg
from xgboost import XGBRegressor
from IPython.display import display
import numpy as np
from openavmkit.reports import finish_report
from openavmkit.utilities.stats import calc_representation
from openavmkit.utilities.settings import get_ensemble_instructions, get_locations, warn_if_location_collapsed
from sklearn.linear_model import LinearRegression

from openavmkit.data import (
    get_important_field,
    _read_split_keys,
    SalesUniversePair,
    get_hydrated_sales_from_sup,
    get_report_locations,
    get_sale_field,
    filter_df_by_date_range
)
from openavmkit.modeling import (
    run_mra,
    run_multi_mra,
    run_gwr,
    run_xgboost,
    run_lightgbm,
    run_catboost,
    run_ngboost,
    run_layeredcomp,
    run_garbage,
    run_average,
    run_naive_area,
    run_kernel,
    run_local_area,
    run_pass_through,
    run_ground_truth,
    run_spatial_lag,
    SingleModelResults,
    predict_multi_mra,
    predict_garbage,
    predict_average,
    predict_naive_area,
    predict_local_area,
    predict_pass_through,
    predict_kernel,
    predict_gwr,
    predict_xgboost,
    predict_catboost,
    predict_lightgbm,
    predict_layeredcomp,
    predict_ground_truth,
    predict_spatial_lag,
    GarbageModel,
    AverageModel,
    DataSplit,
    write_model_parameters, get_shap_contributions_map,
    _add_prediction_to_contribution, _contrib_to_unit_values
)
from openavmkit.reports import MarkdownReport, _markdown_to_pdf
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import (
    div_df_z_safe,
    df_to_markdown,
    do_per_model_group,
    load_model_results,
)
from openavmkit.utilities.format import fancy_format, dig2_fancy_format
from openavmkit.utilities.modeling import (
    NaiveAreaModel,
    LocalAreaModel,
    PassThroughModel,
    GWRModel,
    MRAModel,
    MultiMRAModel,
    GroundTruthModel,
    SpatialLagModel, XGBoostModel, LightGBMModel, CatBoostModel
)
from openavmkit.utilities.plotting import plot_scatterplot, _simple_ols
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_variable_interactions,
    get_valuation_date,
    get_model_seed,
    get_model_group,
    _apply_dd_to_df_rows,
    get_model_group_ids,
    get_fields_boolean,
    _get_sales,
    _simulate_removed_buildings,
    _get_max_ratio_study_trim,
    get_look_back_dates,
    get_assessor_holdout_mode,
    area_unit,
    length_unit
)
from openavmkit.utilities.stats import (
    calc_vif_recursive_drop,
    calc_t_values_recursive_drop,
    calc_p_values_recursive_drop,
    calc_elastic_net_regularization,
    calc_correlations,
    calc_r2,
    calc_cross_validation_score,
    calc_cod,
    calc_mse,
    trim_outliers_mask,
)
from openavmkit.utilities.geometry import ensure_geometries
from openavmkit.utilities.timing import TimingData
from openavmkit.shap_analysis import (
    _calc_shap,
    plot_full_beeswarm,
    explanation_from_contributions,
)

#######################################
# PUBLIC
#######################################


class BenchmarkResults:
    """Container for benchmark results.

    Attributes
    ----------
    df_time : pd.DataFrame
        DataFrame containing timing information.
    df_stats_test :pd.DataFrame
        DataFrame with statistics for the test set.
    df_stats_test_post_val: pd.DataFrame
        DataFrame with statistics for the test set (post-valuation-date only).
    df_stats_full: pd.DataFrame
        DataFrame with statistics for the full universe.
    test_empty : bool
        Whether df_stats_test contains no records
    full_empty: bool
        Whether df_stats_full contains no records
    test_post_val_empty: bool
        Whether df_stats_test_post_val contains no records
    """

    def __init__(
        self,
        df_time: pd.DataFrame,
        df_stats_test: pd.DataFrame,
        df_stats_test_post_val: pd.DataFrame,
        df_stats_full: pd.DataFrame,
        assessor_in_test: bool = True,
    ):
        """
        Initialize a BenchmarkResults instance.

        Parameters
        ----------
        df_time : pandas.DataFrame
            DataFrame containing timing data.
        df_stats_test : pandas.DataFrame
            DataFrame with test set statistics.
        df_stats_test_post_val : pandas.DataFrame
            DataFrame with test set (post-valuation-date only) statistics.
        df_stats_full : pandas.DataFrame
            DataFrame with full universe statistics.
        """
        self.df_time = df_time
        self.df_stats_test = df_stats_test
        self.df_stats_test_post_val = df_stats_test_post_val
        self.df_stats_full = df_stats_full

        test_empty = False == (df_stats_test["count_sales"].sum() > 0)
        full_empty = False == (df_stats_full["count_sales"].sum() > 0)

        if df_stats_test_post_val is not None:
            test_post_val_empty = False == (df_stats_test_post_val["count_sales"].sum() > 0)
        else:
            test_post_val_empty = True

        self.test_empty = test_empty
        self.full_empty = full_empty
        self.test_post_val_empty = test_post_val_empty
        self.assessor_in_test = assessor_in_test

    def print(self) -> str:
        """
        Return a formatted string summarizing the benchmark results.

        Returns
        -------
        str
            A string that includes timings, test set stats, and universe set stats.
        """
        result = "Timings:\n"
        result += _format_benchmark_df(self.df_time)
        result += "\n\n"
        if (
            self.df_stats_test_post_val is not None
            and not self.test_post_val_empty
        ):
            result += "Holdout set (post-valuation-date only):\n"
            result += (
                "  (Like-for-like vs. the assessor: these sales postdate the valuation date,\n"
                "   so they are out-of-sample for both -- as long as valuation_date is aligned\n"
                "   with the assessor's roll-close date.)\n"
            )
            result += _format_benchmark_df(self.df_stats_test_post_val)
            result += "\n\n"
        result += "Holdout set:\n"
        if self.assessor_in_test:
            result += (
                "  (Assessor shown here because you've declared its values honor this same\n"
                "   holdout (analysis.ratio_study.assessor_holdout: shared). Otherwise it is\n"
                "   left off, since the holdout status of values we didn't generate is unknown.)\n"
            )
        else:
            result += (
                "  (Assessor not shown here: this is a random pre-valuation holdout we draw\n"
                "   ourselves. Our figures are out-of-sample, but we can't know whether values\n"
                "   we didn't generate were held out the same way, so the comparison wouldn't be\n"
                "   like-for-like. If you are the assessor and know the holdout status, see\n"
                "   analysis.ratio_study.assessor_holdout.)\n"
            )
        result += _format_benchmark_df(self.df_stats_test)
        result += "\n\n"
        result += "Study set:\n"
        result += (
            "  (Assessor shown as an audit of the finished roll over all sales -- the standard\n"
            "   IAAO frame, not a predictive holdout. See the sales-chasing check in the ratio\n"
            "   study report for context on interpreting a very tight assessor result.)\n"
        )
        result += _format_benchmark_df(self.df_stats_full)
        result += "\n\n"
        return result


class MultiModelResults:
    """Container for results from multiple models along with a benchmark.

    Attributes:
        model_results (dict[str, SingleModelResults]): Dictionary mapping model names to their results.
        benchmark (BenchmarkResults): Benchmark results computed from the model results.
    """

    model_results: dict[str, SingleModelResults]
    benchmark: BenchmarkResults
    df_univ_orig: pd.DataFrame
    df_sales_orig: pd.DataFrame
    drop_assessor_from_test: bool

    def __init__(
        self, model_results: dict[str, SingleModelResults], benchmark: BenchmarkResults, df_univ: pd.DataFrame, df_sales: pd.DataFrame, drop_assessor_from_test: bool = False
    ):
        """Initialize a MultiModelResults instance.

        Parameters
        ----------
        model_results: dict[str, SingleModelResults]
            Dictionary of individual model results.
        benchmark: BenchmarkResults
            Benchmark results.
        drop_assessor_from_test: bool
            Whether the assessor should be left off the pre-valuation "Test set". Stored so
            that ``add_model`` (which recomputes the benchmark, e.g. when the ensemble is
            added) preserves the same choice as the initial ``_calc_benchmark`` call.
        """
        self.model_results = model_results
        self.benchmark = benchmark
        self.df_univ_orig = df_univ
        self.df_sales_orig = df_sales
        self.drop_assessor_from_test = drop_assessor_from_test

    def add_model(self, model: str, results: SingleModelResults):
        """Add a new model's results and update the benchmark.

        Parameters
        ----------
        model: str
            The model name.
        results: SingleModelResults
            The results for the given model.
        """
        self.model_results[model] = results
        # Recalculate the benchmark based on updated model results. Preserve the assessor
        # drop choice -- otherwise adding the ensemble model would silently re-introduce the
        # assessor into the Test-set comparison.
        self.benchmark = _calc_benchmark(
            self.model_results, drop_assessor_from_test=self.drop_assessor_from_test
        )


def try_variables(
    sup: SalesUniversePair,
    settings: dict,
    verbose: bool = False,
    plot: bool = False,
    do_report: bool = False,
):
    """Experiment with variables to determine which are most useful for modeling.

    Parameters
    ----------
    sup: SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    settings: dict
        Settings dictionary
    verbose: bool
        Whether to print verbose output. Default is False.
    plot: bool
        Whether to generate plots. Default is False.
    do_report: bool
        Whether to generate a pdf report. Default is False.

    """

    df_hydrated = get_hydrated_sales_from_sup(sup)

    idx_vacant = df_hydrated["vacant_sale"].eq(True)

    df_vacant = df_hydrated[idx_vacant].copy()

    df_vacant = _simulate_removed_buildings(df_vacant, settings, idx_vacant)

    # update df_hydrated with *all* the characteristics of df_vacant where their keys match:
    df_hydrated.loc[idx_vacant, df_vacant.columns] = df_vacant.values

    all_best_variables = {}
    all_reports = {}

    try_vars = settings.get("modeling", {}).get("try_variables", {})
    model_groups_to_skip = try_vars.get("skip", [])

    def _try_variables(
        df_in: pd.DataFrame,
        model_group: str,
        df_univ: pd.DataFrame,
        do_report: bool,
        settings: dict,
        verbose: bool,
        results: dict,
        reports: dict
    ):
        bests = {}
        local_reports = {}

        for vacant_only in [False, True]:

            if vacant_only:
                if df_in["vacant_sale"].sum() == 0:
                    if verbose:
                        print("No vacant sales found, skipping...")
                    continue
            else:
                if df_in["valid_sale"].sum() == 0:
                    if verbose:
                        print("No valid sales found, skipping...")
                    continue

            try_vars = settings.get("modeling", {}).get("try_variables", {})
            variables_to_use = (
                try_vars.get("variables", [])
            )

            if len(variables_to_use) == 0:
                raise ValueError(
                    "No variables defined. Please check settings `modeling.try_variables.variables`"
                )

            df_univ = df_univ[df_univ["model_group"].eq(model_group)].copy()

            try:
                var_recs = get_variable_recommendations(
                    df_in,
                    df_univ,
                    vacant_only,
                    settings,
                    model_group,
                    variables_to_use=variables_to_use,
                    tests_to_run=["corr", "r2"],
                    do_report=True,
                    do_cross=True,
                    do_plots=plot,
                    verbose=verbose,
                )
            except Exception as e:
                # A model group can be too small or too degenerate (e.g. all-constant /
                # all-NaN features, too few sales) for the correlation/R2 recommendation
                # step to produce a result -- it would otherwise crash the whole
                # try_variables run. Warn and skip this group instead.
                warnings.warn(
                    f"try_variables: skipping model group '{model_group}' "
                    f"({'vacant' if vacant_only else 'main'}) -- could not compute variable "
                    f"recommendations ({type(e).__name__}: {e}). This usually means the group "
                    f"has too few or too-degenerate sales for the requested variables."
                )
                continue

            best_variables = var_recs["variables"]
            df_results = var_recs["df_results"]
            report = var_recs["report"]
            
            if vacant_only:
                bests["vacant_only"] = df_results
                local_reports["vacant_only"] = report
            else:
                bests["main"] = df_results
                local_reports["main"] = report

        results[model_group] = bests
        reports[model_group] = local_reports

    do_per_model_group(
        df_hydrated,
        settings,
        _try_variables,
        params={
            "settings": settings,
            "df_univ": sup.universe,
            "do_report": do_report,
            "verbose": verbose,
            "results": all_best_variables,
            "reports": all_reports
        },
        key="key_sale",
        skip=model_groups_to_skip
    )

    sale_field = get_sale_field(settings)

    print("")
    print("********** BEST VARIABLES ***********")
    for model_group in all_best_variables:
        entry = all_best_variables[model_group]
        report_entry = all_reports[model_group]
        for vacant_status in entry:
            print("")
            print(f"model group: {model_group} / {vacant_status}")
            results = entry[vacant_status]
            report = report_entry[vacant_status]
            results = results[~results["corr_strength"].isna()]

            styled = results.style.format(
                {
                    "corr_strength": "{:,.2f}",
                    "corr_clarity": "{:,.2f}",
                    "corr_score": "{:,.2f}",
                    "r2": "{:,.2f}",
                    "adj_r2": "{:,.2f}",
                    "coef_sign": "{:,.0f}"
                }
            )

            pd.set_option("display.max_rows", None)
            display(styled)
            pd.set_option("display.max_rows", 15)
            
            file_out = f"out/try/{model_group}/{vacant_status}.csv"
            report_out = f"out/try/{model_group}/{vacant_status}_report"
            if not os.path.exists(os.path.dirname(file_out)):
                os.makedirs(os.path.dirname(file_out))
            results.to_csv(file_out, index=False)
            
            if do_report:
                finish_report(report, report_out, "variable", settings)


def get_variable_recommendations(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    vacant_only: bool,
    settings: dict,
    model_group: str,
    variables_to_use: list[str] | None = None,
    tests_to_run: list[str] | None = None,
    do_cross: bool = True,
    do_report: bool = False,
    do_plots: bool = False,
    verbose: bool = False,
    t: TimingData = None
) -> dict:
    """Determine which variables are most likely to be meaningful in a model.

    This function examines sales and universe data, applies feature selection via
    correlations, elastic net regularization, R², p-values, t-values, and VIF, and
    produces a set of recommended variables along with a written report.

    Parameters
    ----------
    df_sales : pandas.DataFrame
        The sales data.
    df_universe : pandas.DataFrame
        The parcel universe data.
    vacant_only : bool
        Whether to consider only vacant sales.
    settings : dict
        The settings dictionary.
    model_group : str
        The model group to consider.
    variables_to_use : list[str] or None
        A list of variables to use for feature selection. If None, variables are pulled
        from modeling section
    tests_to_run : list[str] or None
        A list of tests to run. If None, all tests are run. Legal values are "corr",
        "r2", "p_value", "t_value", "enr", and "vif"
    do_report : bool
        If True, generates a report of the variable selection process.
    do_plots: bool, optional
        If True, prints correlation plots
    verbose : bool, optional
        If True, prints additional debugging information.
    t : TimingData or None
        TimingData object
    
    Returns
    -------
    dict
        A dictionary with keys "variables" (the best variables list) and "report"
        (the generated report).
    """    
    if t is None:
        t = TimingData()
    
    t.start("variables.markdown")
    report: MarkdownReport = MarkdownReport("variables")
    t.stop("variables.markdown")
    if tests_to_run is None:
        tests_to_run: list[str] = ["corr", "r2", "p_value", "t_value", "enr", "vif"]

    if "sale_price_time_adj" not in df_sales.columns:
        warnings.warn("Time adjustment was not found in sales data. Calculating now...")
        t.start("variables.time_adjustment")
        df_sales = enrich_time_adjustment(df_sales, settings, write=False, verbose=verbose)
        t.stop("variables.time_adjustment")
    
    t.start("variables.stuff")
    settings_model = settings.get("modeling", {})
    vacant_status = "vacant" if vacant_only else "main"
    model_entries = settings_model.get("models", {}).get(vacant_status, {})
    model_entries = model_entries.get(model_group, model_entries)
    entry: dict | None = model_entries.get("model", model_entries.get("default", {}))
    if variables_to_use is None:
        variables_to_use: list | None = entry.get("ind_vars", None)
    
    if variables_to_use is None or len(variables_to_use) == 0:
        raise ValueError("No independent variables provided! Please define some!")
    
    categoricals = get_fields_categorical(settings, df_sales, include_boolean=False)

    flagged = []
    categoricals_to_use = [x for x in variables_to_use if x in categoricals]
    for variable in categoricals_to_use:
        if df_sales[variable].nunique() > 50:
            warnings.warn(
                f"Variable '{variable}' has more than 50 unique values. No variable analysis will be done on it and it will not be auto-dropped. Hope you know what you're doing!"
            )
            flagged.append(variable)

    if len(flagged) > 0:
        variables_to_use = [
            variable for variable in variables_to_use if variable not in flagged
        ]
    
    # Check for duplicate variables in variables_to_use
    if variables_to_use is not None:
        seen_vars = set()
        duplicates = []
        deduped_vars = []

        for var in variables_to_use:
            if var in seen_vars:
                duplicates.append(var)
            else:
                seen_vars.add(var)
                deduped_vars.append(var)

        if duplicates:
            print(
                f"\n⚠️ WARNING: Found duplicate variables in variables_to_use: {duplicates}"
            )
            print(f"Using only the first occurrence of each variable for analysis.")
            variables_to_use = deduped_vars

    # Check for duplicate columns in DataFrame (could happen from merges)
    duplicate_cols = df_sales.columns[df_sales.columns.duplicated()].tolist()
    if duplicate_cols:
        print(
            f"\n⚠️ WARNING: Found duplicate columns in sales DataFrame: {duplicate_cols}"
        )
        print(
            f"This could cause errors in analysis. Keeping only first occurrence of each column."
        )
        df_sales = df_sales.loc[:, ~df_sales.columns.duplicated()]

    duplicate_cols_univ = df_universe.columns[df_universe.columns.duplicated()].tolist()
    if duplicate_cols_univ:
        print(
            f"\n⚠️ WARNING: Found duplicate columns in universe DataFrame: {duplicate_cols_univ}"
        )
        print(
            f"This could cause errors in analysis. Keeping only first occurrence of each column."
        )
        df_universe = df_universe.loc[:, ~df_universe.columns.duplicated()]
    
    t.stop("variables.stuff")
    
    t.start("variables.prepare_ds")
    ds = _prepare_ds(
        "var_recs", df_sales, df_universe, model_group, vacant_only, settings, variables_to_use
    )
    t.stop("variables.prepare_ds")
    t.start("variables.one_hot")
    ds = ds.encode_categoricals_with_one_hot()
    t.stop("variables.one_hot")
    t.start("variables.split")
    ds.split()
    t.stop("variables.split")
    feature_selection = (
        settings.get("modeling", {})
        .get("instructions", {})
        .get("feature_selection", {})
    )
    thresh = feature_selection.get("thresholds", {})

    X_sales = ds.X_sales[ds.ind_vars]
    y_sales = ds.y_sales
    
    X_univ = ds.X_univ[ds.ind_vars]
    
    t.start("variables.rep")
    rep_results = calc_representation(X_sales, X_univ, do_plots=do_plots)
    bad_vars = rep_results["bad_vars"]
    t.stop("variables.rep")
    
    # Remove bad variables
    ind_vars = [var for var in ds.ind_vars if var not in bad_vars]
    
    if "corr" in tests_to_run:
        # Correlation
        X_corr = ds.df_sales[[ds.dep_var] + ind_vars]
        t.start("variables.corr")
        corr_results = calc_correlations(X_corr, thresh.get("correlation", 0.1), do_plots=do_plots)
        
        # Remove bad variables
        ind_vars = [var for var in ds.ind_vars if var not in corr_results["bad_vars"]]
        t.stop("variables.corr")
    else:
        corr_results = None
    
    if "enr" in tests_to_run:
        # Elastic net regularization
        try:
            t.start("variables.enr")
            enr_coefs = calc_elastic_net_regularization(
                X_sales, y_sales, thresh.get("enr", 0.01)
            )
            t.stop("variables.enr")
        except ValueError as e:
            nulls_in_X = X_sales[X_sales.isna().any(axis=1)]
            print(f"Found {len(nulls_in_X)} rows with nulls in X:")
            # identify columns with nulls in them:
            cols_with_null = nulls_in_X.columns[nulls_in_X.isna().any()].tolist()
            print(f"Columns with nulls: {cols_with_null}")
            raise e
    else:
        enr_coefs = None

    if "r2" in tests_to_run:
        # R² values
        t.start("variables.r2")
        r2_values = calc_r2(ds.df_sales, ind_vars, y_sales)
        t.stop("variables.r2")
    else:
        r2_values = None

    if "p_value" in tests_to_run:
        # P Values
        t.start("variables.p")
        p_values = calc_p_values_recursive_drop(
            X_sales, y_sales, thresh.get("p_value", 0.05)
        )
        t.stop("variables.p")
    else:
        p_values = None

    if "t_value" in tests_to_run:
        # T Values
        t.start("variables.t")
        t_values = calc_t_values_recursive_drop(
            X_sales, y_sales, thresh.get("t_value", 2)
        )
        t.stop("variables.t")
    else:
        t_values = None

    if "vif" in tests_to_run:
        t.start("variables.vif")
        # VIF
        # Filter out boolean columns before VIF calculation
        bool_cols = []
        vif_X = X_sales.copy()

        for col in X_sales.columns:
            # Check if column is boolean or contains only 0/1 values
            if X_sales[col].dtype == bool or (
                X_sales[col].isin([0, 1, True, False]).all()
                and len(X_sales[col].unique()) <= 2
            ):
                bool_cols.append(col)

        if bool_cols:
            vif_X = vif_X.drop(columns=bool_cols)

        # Don't run VIF if we have no columns left or too few rows
        if 0 < vif_X.shape[1] < len(vif_X):
            vif = calc_vif_recursive_drop(vif_X, thresh.get("vif", 10), settings)

            # Add boolean columns back to the final VIF results with NaN VIF values
            if bool_cols and vif is not None and "final" in vif:
                for bool_col in bool_cols:
                    vif["final"] = pd.concat(
                        [
                            vif["final"],
                            pd.DataFrame(
                                {"variable": [bool_col], "vif": [float("nan")]}
                            ),
                        ],
                        ignore_index=True,
                    )
        else:
            if verbose:
                print(
                    "Skipping VIF calculation - not enough non-boolean variables or samples"
                )
            vif = {
                "initial": pd.DataFrame(columns=["variable", "vif"]),
                "final": pd.DataFrame(columns=["variable", "vif"]),
            }
        t.stop("variables.vif")
    else:
        vif = None
    
    t.start("variables.calc_recs")
    # Generate final results & recommendations
    df_results = _calc_variable_recommendations(
        ds=ds,
        settings=settings,
        rep_results=rep_results,
        correlation_results=corr_results,
        enr_results=enr_coefs,
        r2_values_results=r2_values,
        p_values_results=p_values,
        t_values_results=t_values,
        vif_results=vif,
        report=report
    )
    t.stop("variables.calc_recs")

    t.start("variables.final_stuff")
    curr_variables = df_results["variable"].tolist()
    best_variables = curr_variables.copy()
    best_score = float("inf")
    
    df_cross = df_results.copy()
    y = ds.y_sales
    
    t.start("variables.final_stuff.while")
    if do_cross:
        while len(curr_variables) > 0:
            X = ds.df_sales[curr_variables]
            t.start("variables.final_stuff.while.cross")
            cv_score = calc_cross_validation_score(X, y)
            t.stop("variables.final_stuff.while.cross")
            if cv_score < best_score:
                best_score = cv_score
                best_variables = curr_variables.copy()
            worst_idx = df_cross["weighted_score"].idxmin()
            worst_variable = df_cross.loc[worst_idx, "variable"]
            curr_variables.remove(worst_variable)
            # Remove the variable from the results dataframe.
            df_cross = df_cross[df_cross["variable"].ne(worst_variable)]
    t.stop("variables.final_stuff.while")
    
    # Create a table from the list of best variables.
    df_best = pd.DataFrame(best_variables, columns=["Variable"])
    df_best["Rank"] = range(1, len(df_best) + 1)
    df_best["Description"] = df_best["Variable"]
    
    t.start("variables.final_stuff.apply_dd")
    df_best = _apply_dd_to_df_rows(
        df_best, "Variable", settings, ds.one_hot_descendants, "name"
    )
    df_best = _apply_dd_to_df_rows(
        df_best, "Description", settings, ds.one_hot_descendants, "description"
    )
    t.stop("variables.final_stuff.apply_dd")
    df_best = df_best[["Rank", "Variable", "Description"]]
    df_best.loc[df_best["Variable"].eq(df_best["Description"]), "Description"] = ""
    df_best.set_index("Rank", inplace=True)

    if do_report:
        report.set_var("summary_table", df_best.to_markdown())
        report = generate_variable_report(report, settings, model_group, best_variables)
    else:
        report = None
    t.stop("variables.final_stuff")
    
    print(t.print())

    return {"variables": best_variables, "report": report, "df_results": df_results}


def generate_variable_report(
    report: MarkdownReport, settings: dict, model_group: str, best_variables: list[str]
):
    """
    Generate a variable selection report.

    This function updates the MarkdownReport with various threshold values, weights, and
    summary tables based on the best variables.

    Parameters
    ----------
    report : MarkdownReport
        The markdown report object.
    settings : dict
        The settings dictionary.
    model_group : str
        The model group identifier.
    best_variables : list[str]
        List of selected best variables.

    Returns
    -------
    MarkdownReport
        The updated markdown report.
    """
    locality = settings.get("locality", {})
    report.set_var("locality", locality.get("name", "...LOCALITY..."))

    mg = get_model_group(settings, model_group)
    report.set_var("val_date", get_valuation_date(settings).strftime("%Y-%m-%d"))
    report.set_var("model_group", mg.get("name", mg))

    instructions = settings.get("modeling", {}).get("instructions", {})
    feature_selection = instructions.get("feature_selection", {})
    thresh = feature_selection.get("thresholds", {})

    report.set_var("thresh_correlation", thresh.get("correlation", ".2f"))
    report.set_var("thresh_enr_coef", thresh.get("enr_coef", ".2f"))
    report.set_var("thresh_vif", thresh.get("vif", ".2f"))
    report.set_var("thresh_p_value", thresh.get("p_value", ".2f"))
    report.set_var("thresh_t_value", thresh.get("t_value", ".2f"))
    report.set_var("thresh_adj_r2", thresh.get("adj_r2", ".2f"))

    weights = feature_selection.get("weights", {})
    df_weights = pd.DataFrame(weights.items(), columns=["Statistic", "Weight"])
    df_weights["Statistic"] = df_weights["Statistic"].map(
        {
            "vif": "VIF",
            "p_value": "P-value",
            "t_value": "T-value",
            "corr_score": "Correlation",
            "enr_coef": "ENR",
            "coef_sign": "Coef. sign",
            "adj_r2": "R-squared",
        }
    )
    df_weights.set_index("Statistic", inplace=True)
    report.set_var("pre_model_weights", df_weights.to_markdown())

    # TODO: Construct summary and post-model tables as needed.
    post_model_table = "...POST MODEL TABLE..."
    report.set_var("post_model_table", post_model_table)

    return report


def run_models(
    sup: SalesUniversePair,
    settings: dict,
    save_params: bool = False,
    use_saved_params: bool = True,
    save_results: bool = False,
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

    t = TimingData()

    t.start("setup")
    s = settings
    s_model = s.get("modeling", {})
    s_inst = s_model.get("instructions", {})
    model_groups = s_inst.get("model_groups", [])

    df_univ = sup["universe"]

    if len(model_groups) == 0:
        model_groups = get_model_group_ids(settings, df_univ)

    dict_all_results = {}
    t.stop("setup")

    t.start("run model groups")
    for model_group in model_groups:
        t.start(f"model group: {model_group}")
        for main_vacant in ["main", "vacant"]:
            if main_vacant == "main" and not run_main:
                continue
            if main_vacant == "vacant" and not run_vacant:
                continue

            models_to_skip = s_inst.get(main_vacant, {}).get("skip", {}).get(model_group, [])

            if "all" in models_to_skip:
                if verbose:
                    print(
                        f"Skipping all models for model_group: {model_group}/{main_vacant}"
                    )
                continue

            if verbose:
                print("")
                print("")
                print("******************************************************")
                print(f"Running models for model_group: {model_group}")
                print("******************************************************")
                print("")
                print("")

            mg_results = _run_models(
                sup,
                model_group,
                settings,
                main_vacant,
                save_params,
                use_saved_params,
                save_results,
                verbose,
                run_ensemble,
                do_shaps=do_shaps,
                do_plots=do_plots
            )
            if mg_results is not None and save_results:
                dict_all_results[model_group] = mg_results
        t.stop(f"model group: {model_group}")
    t.stop("run model groups")

    if save_results:
        t.start("write")
        write_out_all_results(sup, dict_all_results)
        t.stop("write")

    print("**********TIMING FOR RUN ALL MODELS***********")
    print(t.print())
    print("***********************************************")

    return dict_all_results


def write_out_all_results(sup: SalesUniversePair, all_results: dict):
    """Write out all model results to CSV and Parquet files.

    This function collects predictions from all model groups and writes them to a single
    DataFrame, which is then saved to both CSV and Parquet formats. It also merges the
    predictions with the universe DataFrame to include all keys.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    all_results : dict
        A dictionary where keys are model group identifiers and values are MultiModelResults
        containing the results for each model group.
    """
    t = TimingData()
    df_all = None

    for model_group in all_results:
        t.start(f"model group: {model_group}")
        t.start("read")
        mm_results: MultiModelResults = all_results[model_group]

        # Skip if no results for this model group
        if mm_results is None:
            t.stop("read")
            t.stop(f"model group: {model_group}")
            continue

        # Collect all ensemble types to output
        output_models = []
        if "ensemble" in mm_results.model_results:
            output_models.append("ensemble")
        if not output_models:
            t.stop("read")
            t.stop(f"model group: {model_group}")
            continue

        # For each output model, extract predictions and add to df_univ_local
        df_univ_local = None
        for model_type in output_models:
            smr = mm_results.model_results[model_type]
            col_name = (
                f"market_value_{model_type}"
                if "ensemble" not in model_type
                else "market_value"
            )
            df_pred = smr.df_universe[["key", smr.field_prediction]].rename(
                columns={smr.field_prediction: col_name}
            )
            if df_univ_local is None:
                df_univ_local = df_pred
            else:
                df_univ_local = df_univ_local.merge(df_pred, on="key", how="outer")
        df_univ_local["model_group"] = model_group

        if df_all is None:
            df_all = df_univ_local
        else:
            t.start("concat")
            df_all = pd.concat([df_all, df_univ_local])
            t.stop("concat")

        t.stop(f"model group: {model_group}")

    # Only proceed with writing if we have results
    if df_all is not None:
        t.start("copy")
        df_univ = sup.universe.copy()
        t.stop("copy")
        t.start("merge")
        df_univ = df_univ.merge(df_all, on="key", how="left")
        t.stop("merge")

        outpath = "out/models/all_model_groups"
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        t.start("csv")
        df_univ.to_csv(f"{outpath}/universe.csv", index=False)
        t.stop("csv")
        t.start("parquet")
        df_univ.to_parquet(f"{outpath}/universe.parquet", engine="pyarrow")
        t.stop("parquet")


def get_data_split_for(
    model_name: str,
    model_engine: str,
    model_entry: dict,
    model_group: str,
    location_fields: list[str] | None,
    ind_vars: list[str],
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    settings: dict,
    dep_var: str,
    dep_var_test: str,
    fields_cat: list[str],
    interactions: dict,
    test_keys: list[str],
    train_keys: list[str],
    vacant_only: bool,
):
    """
    Prepare a DataSplit object for a given model.

    Parameters
    ----------
    model_name: str,
        Model unique identifier
    model_engine : str
        Model engine ("xgboost", "mra", etc.)
    model_entry : dict
        Model parameters
    model_group : str
        The model group identifier.
    location_fields : list[str] or None
        List of location fields.
    ind_vars : list[str]
        List of independent variables.
    df_sales : pandas.DataFrame
        Sales DataFrame.
    df_universe : pandas.DataFrame
        Universe DataFrame.
    settings : dict
        The settings dictionary.
    dep_var : str
        Dependent variable for training.
    dep_var_test : str
        Dependent variable for testing.
    fields_cat : list[str]
        List of categorical fields.
    interactions : dict
        Dictionary of variable interactions.
    test_keys : list[str]
        Keys for test split.
    train_keys : list[str]
        Keys for training split.
    vacant_only : bool
        Whether to consider only vacant sales.

    Returns
    -------
    DataSplit
        A DataSplit object.
    """

    unit = area_unit(settings)
    lenunit = length_unit(settings)

    if model_engine == "local_area":
        _ind_vars = location_fields + [f"bldg_area_finished_{unit}", f"land_area_{unit}"]
    elif model_engine == "multi_mra":
        _ind_vars = [v for v in ind_vars if v not in location_fields]
    elif model_engine == "assessor":
        _ind_vars = ["assr_market_value"]
    elif model_engine == "pass_through":
        field = model_entry.get("field")
        if field is None:
            raise ValueError("pass_through model \"{model_name}\" has no .field parameter!")
        _ind_vars = [field]
    elif model_engine == "ground_truth":
        _ind_vars = ["true_market_value"]
    elif model_engine == "spatial_lag":
        sale_field = get_sale_field(settings)
        field = f"spatial_lag_{sale_field}"
        if vacant_only:
            field = f"{field}_vacant"
        _ind_vars = [field]
    elif model_engine == "spatial_lag_area":
        sale_field = get_sale_field(settings)
        _ind_vars = [
            f"spatial_lag_{sale_field}_impr_{unit}",
            f"spatial_lag_{sale_field}_land_{unit}",
            f"bldg_area_finished_{unit}",
            f"land_area_{unit}",
        ]
    elif model_engine == "catboost":
        df_sales = _clean_categoricals(df_sales, fields_cat, settings)
        df_universe = _clean_categoricals(df_universe, fields_cat, settings)
        _ind_vars = ind_vars
    elif model_engine == "lcomp":
        _ind_vars = ind_vars
    else:
        _ind_vars = ind_vars
        if model_engine == "gwr" or model_engine == "kernel":
            exclude_vars = ["latitude", "longitude", "latitude_norm", "longitude_norm"]
            _ind_vars = [var for var in _ind_vars if var not in exclude_vars]

    return DataSplit(
        model_name,
        df_sales,
        df_universe,
        model_group,
        settings,
        dep_var,
        dep_var_test,
        _ind_vars,
        fields_cat,
        interactions,
        test_keys,
        train_keys,
        vacant_only=vacant_only,
    )


def run_one_model(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    vacant_only: bool,
    model_group: str,
    model_name: str,
    model_entries: dict,
    settings: dict,
    dep_var: str,
    dep_var_test: str,
    best_variables: list[str],
    fields_cat: list[str],
    outpath: str,
    save_params: bool,
    use_saved_params: bool,
    save_results: bool,
    verbose: bool = False,
    test_keys: list[str] | None = None,
    train_keys: list[str] | None = None,
) -> SingleModelResults | None:
    """
    Run a single model based on provided parameters and return its results.

    Parameters
    ----------
    df_sales : pandas.DataFrame
        Sales DataFrame.
    df_universe : pandas.DataFrame
        Universe DataFrame.
    vacant_only : bool
        Whether to use only vacant sales.
    model_group : str
        Model group identifier.
    model_name : str
        Model's unique identifier.
    model_entries : dict
        Dictionary of model configuration entries.
    settings : dict
        Settings dictionary.
    dep_var : str
        Dependent variable for training.
    dep_var_test : str
        Dependent variable for testing.
    best_variables : list[str]
        List of best variables selected.
    fields_cat : list[str]
        List of categorical fields.
    outpath : str
        Output path for saving results.
    save_params : bool
        Whether to save parameters.
    use_saved_params : bool
        Whether to use saved parameters.
    save_results : bool
        Whether to save results.
    verbose : bool, optional
        If True, prints additional information.
    test_keys : list[str] or None, optional
        Optional list of test keys (will be read from disk if not provided).
    train_keys : list[str] or None, optional
        Optional list of training keys (will be read from disk if not provided).

    Returns
    -------
    SingleModelResults or None
        SingleModelResults if successful, else None.
    """

    t = TimingData()

    t.start("setup")
    
    entry: dict | None = model_entries.get(model_name, None)
    default_entry: dict | None = model_entries.get("default", {})
    if entry is None:
        entry = default_entry
        if entry is None:
            raise ValueError(
                f"Model entry for {model_name} not found, and there is no default entry!"
            )
    model_engine = entry.get("model", model_name)
    if model_engine == "default":
        # this isn't a real model, just a settings object to fill in for others
        return None
    
    if "*" in model_engine:
        sales_chase = 0.01
        model_engine = model_engine.replace("*", "")
    else:
        sales_chase = False

    if verbose:
        print(f"------------------------------------------------")
        print(f"Running model {model_name} on {len(df_sales)} rows...")

    are_ind_vars_default = entry.get("ind_vars", None) is None
    ind_vars: list | None = entry.get("ind_vars", default_entry.get("ind_vars", None))
 
    # no duplicates!
    ind_vars = list(set(ind_vars))
    if ind_vars is None:
        raise ValueError(f"ind_vars not found for model {model_name}")

    if are_ind_vars_default:
        if (best_variables is not None) and (set(ind_vars) != set(best_variables)):
            if verbose:
                print(
                    f"--> using default variables, auto-optimized variable list: {best_variables}"
                )
            ind_vars = best_variables

    interactions = get_variable_interactions(entry, settings, df_sales)
    location_fields = entry.get("locations", get_locations(settings, df_sales))

    if test_keys is None or train_keys is None:
        test_keys, train_keys = _read_split_keys(model_group)
    t.stop("setup")

    t.start("data split")
    ds = get_data_split_for(
        model_name=model_name,
        model_engine=model_engine,
            model_entry=entry,
            model_group=model_group,
            location_fields=location_fields,
            ind_vars=ind_vars,
            df_sales=df_sales,
            df_universe=df_universe,
            settings=settings,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            fields_cat=fields_cat,
            interactions=interactions,
            test_keys=test_keys,
            train_keys=train_keys,
            vacant_only=vacant_only,
        )
        
    # safeguards against invalid splits
    n_sales = len(ds.df_sales) if ds.df_sales is not None else 0
    n_train = len(ds.df_train) if ds.df_train is not None else 0
    n_test  = len(ds.df_test)  if ds.df_test  is not None else 0
    p = ds.X_train.shape[1] if getattr(ds, "X_train", None) is not None else 0
    
    if n_train == 0 or p == 0:
        # compute some helpful diagnostics
        missing_vars = []
        if p == 0:
            # requested vars that are not present in train
            train_cols = set(ds.df_train.columns) if ds.df_train is not None else set()
            missing_vars = [v for v in ds.ind_vars if v not in train_cols]
    
        why = []
        if n_train == 0:
            if n_test == n_sales and n_sales > 0:
                why.append("all sales ended up in the test split (train set empty) — check split keys")
            else:
                why.append("filters/slicing removed all training rows")
        if p == 0:
            why.append(f"no usable features in X_train (missing ind_vars: {missing_vars[:20]}{'...' if len(missing_vars)>20 else ''})")
    
        warnings.warn(
            f"Skipping model {model_group}/{model_name} ({model_engine}): "
            f"sales={n_sales}, train={n_train}, test={n_test}, X_train_cols={p}. "
            + " ".join(why),
            RuntimeWarning
        )
        return None
    
    t.stop("data split")

    t.start("setup")
    if len(ds.y_sales) < 15:
        if verbose:
            print(f"--> model {model_name} has less than 15 sales. Skipping...")
        return None

    optimize_vars = entry.get("optimize_vars", False)
    intercept = entry.get("intercept", True)
    # Per-model opt-in to log-target training (mra / multi_mra only). Fitting on log(price) keeps
    # the linear models from extrapolating negative values; the model exponentiates its own
    # predictions back to price space, so this stays contained to the model (no dep_var changes).
    log = entry.get("log", False)
    n_trials = entry.get("n_trials", 50)
    use_gpu = entry.get("use_gpu", True)
    seed = get_model_seed(settings)
    t.stop("setup")

    t.start("run")
    if model_engine == "garbage":
        results = run_garbage(
            ds, normal=False, sales_chase=sales_chase, verbose=verbose
        )
    elif model_engine == "garbage_normal":
        results = run_garbage(ds, normal=True, sales_chase=sales_chase, verbose=verbose)
    elif model_engine == "mean":
        results = run_average(
            ds, average_type="mean", sales_chase=sales_chase, verbose=verbose
        )
    elif model_engine == "median":
        results = run_average(
            ds, average_type="median", sales_chase=sales_chase, verbose=verbose
        )
    elif model_engine == "naive_area":
        results = run_naive_area(ds, sales_chase=sales_chase, verbose=verbose)
    elif model_engine == "local_area":
        results = run_local_area(
            ds,
            location_fields=location_fields,
            sales_chase=sales_chase,
            verbose=verbose,
        )
    elif model_engine == "assessor" or model_engine == "pass_through":
        results = run_pass_through(ds, model_engine, verbose=verbose)
    elif model_engine == "ground_truth":
        results = run_ground_truth(ds, verbose=verbose)
    elif model_engine == "spatial_lag":
        results = run_spatial_lag(ds, per_area=False, verbose=verbose)
    elif model_engine == "spatial_lag_area":
        results = run_spatial_lag(ds, per_area=True, verbose=verbose)
    elif model_engine == "mra":
        results = run_mra(ds, intercept=intercept, verbose=verbose, log=log)
    elif model_engine == "multi_mra":
        results = run_multi_mra(ds, outpath, location_fields, optimize_vars=optimize_vars, intercept=intercept, verbose=verbose, log=log)
    elif model_engine == "kernel":
        results = run_kernel(
            ds, outpath, save_params, use_saved_params, verbose=verbose
        )
    elif model_engine == "gwr":
        results = run_gwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
    elif model_engine == "xgboost":
        results = run_xgboost(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose, seed=seed
        )
    elif model_engine == "lightgbm":
        results = run_lightgbm(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose, seed=seed
        )
    elif model_engine == "catboost":
        results = run_catboost(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose, use_gpu=use_gpu, seed=seed
        )
    elif model_engine == "ngboost":
        results = run_ngboost(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose, seed=seed
        )
    elif model_engine == "lcomp":
        results = run_layeredcomp(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose, seed=seed
        )
    else:
        raise ValueError(f"Model {model_engine} not found!")
    t.stop("run")
    
    if results is None:
        return None
    
    if ds.vacant_only:
        # If this is a vacant model, we attempt to load a corresponding "full value" model
        max_trim = _get_max_ratio_study_trim(settings, results.ds.model_group)

    if save_results:
        t.start("write")
        main_vacant = "vacant" if vacant_only else "main"
        location = get_model_location(settings, main_vacant, model_name, model_group)
        _write_model_results(results, outpath, settings, location, verbose=verbose)
        t.stop("write")

    return results


def run_ensemble(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    dep_var: str,
    dep_var_test: str,
    outpath: str,
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
) -> tuple[SingleModelResults, list[str]]:
    """Run an ensemble model based on the provided parameters.

    This function optimizes the ensemble model and runs it, returning the results and the list of models used in the ensemble.

    Parameters
    ----------
    df_sales : pandas.DataFrame or None
        Sales DataFrame. If None, it will be read from the MultiModelResults.
    df_universe : pandas.DataFrame or None
        Universe DataFrame. If None, it will be read from the MultiModelResults.
    model_group : str
        Model group identifier.
    vacant_only : bool
        Whether to use only vacant sales.
    dep_var : str
        Dependent variable for training.
    dep_var_test : str
        Dependent variable for testing.
    outpath : str
        Output path for saving results.
    all_results : MultiModelResults
        MultiModelResults containing all model results.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints additional information. Defaults to False.

    Returns
    -------
    tuple[SingleModelResults, list[str]]
        A tuple containing the SingleModelResults of the ensemble model and a list of models used in the ensemble.
    """
    if verbose:
        print("Optimizing ensemble...")

    ensemble_list = _optimize_ensemble(
        df_sales,
        df_universe,
        model_group,
        vacant_only,
        dep_var,
        dep_var_test,
        all_results,
        settings,
        verbose=verbose,
        ensemble_list=None,
    )
    if verbose:
        print("Running ensemble...")
    ensemble = _run_ensemble(
        df_sales,
        df_universe,
        model_group,
        vacant_only=vacant_only,
        dep_var=dep_var,
        dep_var_test=dep_var_test,
        outpath=outpath,
        ensemble_list=ensemble_list,
        all_results=all_results,
        settings=settings,
        verbose=verbose,
    )
    if verbose:
        print("Finished ensemble!")
    return ensemble, ensemble_list


#######################################
# PRIVATE
#######################################


def _calc_benchmark(
    model_results: dict[str, SingleModelResults], drop_assessor_from_test: bool = False
):
    """
    Calculate benchmark statistics from individual model results.

    Parameters
    ----------
    model_results : dict[str, SingleModelResults]
        Per-model results to summarize.
    drop_assessor_from_test : bool, optional
        When True, the assessor is left out of the pre-valuation "Test set" comparison.
        See the note in the body for why; controlled by ``analysis.ratio_study.assessor_holdout``
        at the main call site. Defaults to False (the assessor is kept), which is correct for
        the post-valuation benchmark and for incremental recomputation.
    """
    data_time = {
        "model": [],
        "total": [],
        "param": [],
        "train": [],
        "test": [],
        "univ": [],
        "chd": [],
    }

    data = {
        "model": [],
        "subset": [],
        "utility_score": [],
        "count_sales": [],
        "count_univ": [],
        "median_ratio": [],
        "cod": [],
        "prd": [],
        "prb": [],
        "vei": [],
        "vei_sig":[],
        "count_trim": [],
        "cod_trim": [],
        "prd_trim": [],
        "prb_trim": [],
        "chd": [],
    }
    for key in model_results:
        for kind in ["test", "test_post_val", "univ"]:
            results = model_results[key]
            if kind == "test":
                pred_results = results.pred_test
                subset = "Test set"
            elif kind == "test_post_val":
                results = _get_post_valuation_smr(results)
                pred_results = results.pred_test
                subset = "Test set (post-valuation date)"
            else:
                pred_results = results.pred_sales_lookback
                subset = "Universe set"

            data["model"].append(key)
            data["subset"].append(subset)
            if kind == "test" or kind == "test_post_val":
                data["utility_score"].append(results.utility_test)
            else:
                data["utility_score"].append(results.utility_train)
            data["count_sales"].append(pred_results.ratio_study.count)
            data["count_univ"].append(results.df_universe.shape[0])
            data["median_ratio"].append(pred_results.ratio_study.median_ratio)
            data["cod"].append(pred_results.ratio_study.cod)
            data["prd"].append(pred_results.ratio_study.prd)
            data["prb"].append(pred_results.ratio_study.prb)
            if kind == "test":
                data["vei"].append(results.ve_test["vei"])
                data["vei_sig"].append(results.ve_test["vei_significance"])
            elif kind == "test_post_val":
                # results here is the post-valuation SMR
                # which doesn't explicitly calculate ve_test in _get_post_valuation_smr yet
                # but let's see if we can get it or if it's fine to be nan
                data["vei"].append(np.nan)
                data["vei_sig"].append(results.ve_test["vei_significance"])
            else:
                data["vei"].append(results.ve_sales_lookback["vei"])
                data["vei_sig"].append(results.ve_sales_lookback["vei_significance"])
            data["count_trim"].append(pred_results.ratio_study.count_trim)
            data["cod_trim"].append(pred_results.ratio_study.cod_trim)
            data["prd_trim"].append(pred_results.ratio_study.prd_trim)
            data["prb_trim"].append(pred_results.ratio_study.prb_trim)

            chd_results = None
            if kind == "univ":
                chd_results = results.chd
                tim = results.timing.results
                data_time["model"].append(key)
                data_time["total"].append(tim.get("total"))
                data_time["param"].append(tim.get("parameter_search"))
                data_time["train"].append(tim.get("train"))
                data_time["test"].append(tim.get("predict_test"))
                data_time["univ"].append(tim.get("predict_univ"))
                data_time["chd"].append(tim.get("chd"))
            data["chd"].append(chd_results)

    df = pd.DataFrame(data)
    df_time = pd.DataFrame(data_time)
    df_test = df[df["subset"].eq("Test set")].drop(columns=["subset"])
    df_test_post_val = df[df["subset"].eq("Test set (post-valuation date)")].drop(
        columns=["subset"]
    )
    df_full = df[df["subset"].eq("Universe set")].drop(columns=["subset"])
    df_time = pd.DataFrame(data_time)

    # The pre-valuation "Test set" is a random holdout we draw ourselves. We have no way to
    # know whether the assessor's values were produced holding out these same sales, so a
    # head-to-head here would not be like-for-like: our figures are out-of-sample while the
    # assessor's may not be. By default we therefore leave the assessor off this set only --
    # it is still shown on the post-valuation holdout (out-of-sample for both, given an
    # aligned valuation_date) and on the full "Universe"/study set (an IAAO-style audit of the
    # finished roll, where evaluating on all sales is standard). If you are the assessor and
    # know the holdout status, `analysis.ratio_study.assessor_holdout: "shared"` keeps the
    # assessor here (see get_assessor_holdout_mode).
    assessor_in_test = "assessor" in model_results
    if drop_assessor_from_test:
        df_test = df_test[df_test["model"].ne("assessor")]
        assessor_in_test = False

    df_test.set_index("model", inplace=True)
    df_test_post_val.set_index("model", inplace=True)
    df_full.set_index("model", inplace=True)
    df_time.set_index("model", inplace=True)

    results = BenchmarkResults(
        df_time, df_test, df_test_post_val, df_full, assessor_in_test=assessor_in_test
    )
    return results


def _format_benchmark_df(df: pd.DataFrame, transpose: bool = True):
    """
    Format a benchmark DataFrame for display.
    """
    formats = {
        "utility_score": fancy_format,
        "count_sales": "{:,.0f}",
        "count_univ": "{:,.0f}",
        "count_trim": "{:,.0f}",
        "mse": fancy_format,
        "rmse": fancy_format,
        "mape": fancy_format,
        "r2": dig2_fancy_format,
        "adj_r2": dig2_fancy_format,
        "median_ratio": dig2_fancy_format,
        "vei": dig2_fancy_format,
        "vei_sig": dig2_fancy_format,
        "cod": dig2_fancy_format,
        "cod_trim": dig2_fancy_format,
        "true_mse": fancy_format,
        "true_rmse": fancy_format,
        "true_r2": dig2_fancy_format,
        "true_adj_r2": dig2_fancy_format,
        "true_median_ratio": dig2_fancy_format,
        "true_cod": dig2_fancy_format,
        "true_cod_trim": dig2_fancy_format,
        "true_prb": dig2_fancy_format,
        "prd": dig2_fancy_format,
        "prd_trim": dig2_fancy_format,
        "prb": dig2_fancy_format,
        "prb_trim": dig2_fancy_format,
        "total": fancy_format,
        "param": fancy_format,
        "train": fancy_format,
        "test": fancy_format,
        "univ": fancy_format,
        "chd": fancy_format,
        "med_ratio": dig2_fancy_format,
        "true_med_ratio": dig2_fancy_format,
        "chd_total": fancy_format,
        "chd_impr": fancy_format,
        "chd_land": fancy_format,
        "null": "{:.1%}",
        "neg": "{:.1%}",
        "bad_sum": "{:.1%}",
        "land_over": "{:.1%}",
        "vac_not_100": "{:.1%}",
    }

    for col in df.columns:
        if col.strip() == "":
            continue
        if col in formats:
            if callable(formats[col]):
                df[col] = df[col].apply(formats[col])
            else:
                df[col] = df[col].apply(lambda x: formats[col].format(x))
    if transpose:
        df = df.transpose()
    return df.to_markdown()

def _clamp_land_predictions(
    results: SingleModelResults, 
    model_group: str, 
    model_name: str, 
    model_engine: str,
    outpath: str, 
    max_trim: float
):
    """
    Clamp land value predictions based on the full market value predictions.
    This function ensures that land value predictions are non-negative and do not exceed the full market value predictions.
    """

    lookpath = "main"
    if "vacant" in outpath:
        lookpath = "main"

    # Look for the corresponding universe, sales, and test predictions for the land value model.
    df_univ = load_model_results(model_group, model_name, "universe", lookpath)
    
    if df_univ is not None:
        # There's a match for this model name (ex: "xgboost" or "lightgbm") in the set of main models
        df_sales = load_model_results(model_group, model_name, "sales", lookpath)
        df_test = load_model_results(model_group, model_name, "test", lookpath)
    else:
        # There's not a match for this model name, so we look for the ensemble as the baseline
        df_univ = load_model_results(model_group, "ensemble", "universe", lookpath)
        if df_univ is not None:
            df_sales = load_model_results(model_group, "ensemble", "sales", lookpath)
            df_test = load_model_results(model_group, "ensemble", "test", lookpath)
        else:
            warnings.warn(
                f"Couldn't find main baseline for {model_group}/{model_name} land value predictions, skipping clamping. Run finalize and try again!"
            )
            return results

    field_pred = results.field_prediction

    # Get our predictions and interpet as land value
    df_land_univ = (
        results.df_universe[["key", field_pred]]
        .copy()
        .rename(columns={field_pred: "land_value"})
    )
    df_land_sales = (
        results.df_sales[["key_sale", field_pred]]
        .copy()
        .rename(columns={field_pred: "land_value"})
    )
    df_land_test = (
        results.df_test[["key_sale", field_pred]]
        .copy()
        .rename(columns={field_pred: "land_value"})
    )

    # Merge the baseline (full market value) prediction onto our land value predictions
    df_land_univ = df_land_univ.merge(
        df_univ[["key", "prediction"]], on="key", how="left"
    )
    df_land_sales = df_land_sales.merge(
        df_sales[["key_sale", "prediction"]], on="key_sale", how="left"
    )
    df_land_test = df_land_test.merge(
        df_test[["key_sale", "prediction"]], on="key_sale", how="left"
    )

    # Clamp land value to the range of (0.0, prediction)
    # - No negative land values are allowed
    # - Land value cannot exceed the full market value prediction
    # - NOTE: this does *not* look at any sales data, so it's not cheating, it's just another step in the prediction algorithm
    #   we're just looking at another prediction we made earlier in the pipeline and using that to judge land value

    count_univ_clipped = df_land_univ[
        df_land_univ["land_value"].lt(0)
        | df_land_univ["land_value"].gt(df_land_univ["prediction"])
    ].shape[0]
    count_sales_clipped = df_land_sales[
        df_land_sales["land_value"].lt(0)
        | df_land_sales["land_value"].gt(df_land_sales["prediction"])
    ].shape[0]
    count_test_clipped = df_land_test[
        df_land_test["land_value"].lt(0)
        | df_land_test["land_value"].gt(df_land_test["prediction"])
    ].shape[0]

    df_land_univ["land_value"] = df_land_univ["land_value"].clip(
        lower=0.0, upper=df_land_univ["prediction"]
    )
    df_land_sales["land_value"] = df_land_sales["land_value"].clip(
        lower=0.0, upper=df_land_sales["prediction"]
    )
    df_land_test["land_value"] = df_land_test["land_value"].clip(
        lower=0.0, upper=df_land_test["prediction"]
    )

    # Extract the land value predictions
    y_pred_test = df_land_test["land_value"].values
    y_pred_sales = df_land_sales["land_value"].values
    y_pred_univ = df_land_univ["land_value"].values

    # turn to ndarray
    y_pred_test = np.asarray(y_pred_test)
    y_pred_sales = np.asarray(y_pred_sales)
    y_pred_univ = np.asarray(y_pred_univ)
    
    ds = results.ds.copy()
    
    # reconstruct dataframes
    
    ds.df_test = df_land_test.merge(ds.df_test[["key_sale"] + [f for f in ds.df_test if f not in df_land_test]], on="key_sale", how="left")
    ds.df_sales = df_land_sales.merge(ds.df_sales[["key_sale"] + [f for f in ds.df_sales if f not in df_land_sales]], on="key_sale", how="left")
    
    ds.df_universe = df_land_univ.merge(ds.df_universe[["key"] + [f for f in ds.df_universe if f not in df_land_univ]], on="key", how="left")
    
    # Create a new SingleModelResults object with the clamped land value predictions
    results = SingleModelResults(
        ds,
        field_pred,
        results.field_horizontal_equity_id,
        model_name,
        model_engine,
        results.model,
        y_pred_test,
        y_pred_sales,
        y_pred_univ,
        results.timing,
        results.verbose,
        results.sale_filter
    )

    count_univ = len(results.df_universe)
    count_sales = len(results.df_sales)
    count_test = len(results.df_test)

    print(f"--> univ  : {count_univ_clipped}/{count_univ} clamped land values")
    print(f"--> sales : {count_sales_clipped}/{count_sales} clamped land values")
    print(f"--> test  : {count_test_clipped}/{count_test} clamped land values")

    return results


def _clean_categoricals(df_in: pd.DataFrame, fields: list[str], settings: dict):
    """
    Clean categorical fields in the DataFrame.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame.
    fields : list[str]
        List of fields to clean.
    settings : dict
        The settings dictionary.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame.
    """

    fields_bool = get_fields_boolean(settings, df_in)
    fields_cat = get_fields_categorical(settings, df_in)

    for field in fields:
        if field in df_in.columns:
            if field in fields_bool:
                # Convert boolean fields to integers
                df_in[field] = df_in[field].astype(int)
            elif field in fields_cat:
                # Convert categorical fields to categoricals
                df_in[field] = df_in[field].astype("category")
            else:
                raise ValueError(
                    f"Field '{field}' is neither boolean nor categorical, but was indicated as a categorical field. Please classify it properly!"
                )

    return df_in


def _assemble_model_results(results: SingleModelResults, settings: dict):
    """
    Assemble model results into DataFrames for sales, universe, and test sets.
    """
    
    unit = area_unit(settings)
    
    locations = get_report_locations(settings)
    fields = [
        "key",
        "geometry",
        "prediction",
        "assr_market_value",
        "assr_land_value",
        "true_market_value",
        "true_land_value",
        f"bldg_area_finished_{unit}",
        f"land_area_{unit}",
        "sale_price",
        "sale_price_time_adj",
        "sale_date",
    ] + locations
    fields = [field for field in fields if field in results.df_sales.columns]

    dfs = {
        "sales": results.df_sales[["key_sale"] + fields].copy(),
        "universe": results.df_universe[fields].copy(),
        "test": results.df_test[["key_sale"] + fields].copy()
    }

    for key in dfs:
        df = dfs[key]
        df["prediction_ratio"] = div_df_z_safe(df, "prediction", "sale_price_time_adj")

        if f"bldg_area_finished_{unit}" in df:
            df[f"prediction_impr_{unit}"] = div_df_z_safe(
                df, "prediction", f"bldg_area_finished_{unit}"
            )
        if f"land_area_{unit}" in df:
            df[f"prediction_land_{unit}"] = div_df_z_safe(
                df, "prediction", f"land_area_{unit}"
            )

        if "assr_market_value" in df:
            df["assr_ratio"] = div_df_z_safe(
                df, "assr_market_value", "sale_price_time_adj"
            )
        else:
            df["assr_ratio"] = None
        if "true_market_value" in df:
            df["true_vs_sale_ratio"] = div_df_z_safe(
                df, "true_market_value", "sale_price_time_adj"
            )
            df["pred_vs_true_ratio"] = div_df_z_safe(
                df, "prediction", "true_market_value"
            )
        for location in locations:
            if location in df:
                df[f"prediction_cod_{location}"] = None
                df[f"assr_cod_{location}"] = None
                location_values = df[location].unique()
                for value in location_values:
                    predictions = df.loc[
                        df[location].eq(value), "prediction_ratio"
                    ].values
                    predictions = predictions[~pd.isna(predictions)]
                    df.loc[df[location].eq(value), f"prediction_cod_{location}"] = (
                        calc_cod(predictions)
                    )

                    if "assr_market_value" in df:
                        assr_ratios = df.loc[
                            df[location].eq(value), "assr_ratio"
                        ].values
                        assr_ratios = assr_ratios[~pd.isna(assr_ratios)]
                        df.loc[df[location].eq(value), f"assr_cod_{location}"] = (
                            calc_cod(assr_ratios)
                        )
                    if "true_market_value" in df:
                        true_vs_sales_ratios = df.loc[
                            df[location].eq(value), "true_vs_sale_ratio"
                        ].values
                        true_vs_sales_ratios = true_vs_sales_ratios[
                            ~pd.isna(true_vs_sales_ratios)
                        ]
                        df.loc[
                            df[location].eq(value), f"true_vs_sale_cod_{location}"
                        ] = calc_cod(true_vs_sales_ratios)

                        pred_vs_true_ratios = df.loc[
                            df[location].eq(value), "pred_vs_true_ratio"
                        ].values
                        pred_vs_true_ratios = pred_vs_true_ratios[
                            ~pd.isna(pred_vs_true_ratios)
                        ]
                        df.loc[
                            df[location].eq(value), f"pred_vs_true_cod_{location}"
                        ] = calc_cod(pred_vs_true_ratios)

    return dfs


def _write_model_results(results: SingleModelResults, outpath: str, settings: dict, location: str = None, verbose:bool = False):
    """
    Write model results to disk in parquet and CSV formats.
    """
    
    print(f"Write model results to {outpath}")
    
    dfs = _assemble_model_results(results, settings)
    path = f"{outpath}/{results.model_name}"
    if "*" in path:
        path = path.replace("*", "_star")
    os.makedirs(path, exist_ok=True)
    for key in dfs:
        df = dfs[key]
        
        if "geometry" in df.columns:
            df = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None))
            df = ensure_geometries(df)
        
        df.to_parquet(f"{path}/pred_{key}.parquet")
        if "geometry" in df:
            df = df.drop(columns=["geometry"])
        df.to_csv(f"{path}/pred_{key}.csv", index=False)

    results.df_sales.to_csv(f"{path}/sales.csv", index=False)
    results.df_universe.to_csv(f"{path}/universe.csv", index=False)

    with open(f"{path}/pred_test.pkl", "wb") as f:
        pickle.dump(results.pred_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{path}/pred_sales.pkl", "wb") as f:
        pickle.dump(results.pred_sales, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{path}/pred_universe.pkl", "wb") as f:
        pickle.dump(results.pred_univ, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    params_path = f"{path}"
    
    write_model_parameters(results.model, results, location, params_path, verbose=verbose)

    try:
        universe_parquet = gpd.read_parquet(f"{path}/pred_universe.parquet")
    except FileNotFoundError:
        universe_parquet = None
    df_contributions = None
    # "universe" is the canonical subset name; "univ" is read for back-compat
    # with outputs from older runs.
    for _fname in ["contributions_universe.csv", "contributions_univ.csv"]:
        try:
            df_contributions = pd.read_csv(f"{path}/{_fname}")
            break
        except FileNotFoundError:
            continue

    if universe_parquet is not None and df_contributions is not None:
        shap_contributions_map = get_shap_contributions_map(universe_parquet, results.df_universe, df_contributions)
        shap_contributions_map.to_parquet(f"{path}/contributions_map.parquet")
        print(f"Wrote contributions map to {path}/contributions_map.parquet")




def get_model_location(
    settings: dict,
    main_vacant: str,
    model_name: str,
    model_group: str
):
    mv = settings.get("modeling", {}).get("models", {}).get(main_vacant, {})
    mv = mv.get(model_group, mv)
    model_entry = mv.get(model_name, mv.get("default", {}))
    location = model_entry.get("location", None)
    if location is None:
        location = get_important_field(settings, "loc_market_area")
    return location


def _aggregate_ensemble(df: pd.DataFrame, ensemble_list: list[str], agg: str):
    """Reduce per-model prediction columns into a single ensemble prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing one column per model in ``ensemble_list``.
    ensemble_list : list[str]
        Columns to aggregate across (one per component model).
    agg : str
        Aggregation method, either ``"median"`` or ``"mean"``.

    Returns
    -------
    pandas.Series
        Row-wise aggregated prediction.
    """
    if agg == "mean":
        return df[ensemble_list].mean(axis=1)
    elif agg == "median":
        return df[ensemble_list].median(axis=1)
    raise ValueError(f"Unrecognized ensemble aggregation \"{agg}\"!")


def _write_ensemble_model_results(
    results: SingleModelResults,
    outpath: str,
    settings: dict,
    dfs: dict[str, pd.DataFrame],
    ensemble_list: list[str] | None,
):
    """
    Write ensemble model results to disk.
    """
    dfs_basic = _assemble_model_results(results, settings)
    path = f"{outpath}/{results.model_name}"
    os.makedirs(path, exist_ok=True)
    for key in dfs_basic:
        prim_keys = ["key"]
        merge_key = "key"
        if key in ["sales", "test"]:
            prim_keys.append("key_sale")
            merge_key = "key_sale"
        df_basic = dfs_basic[key]
        df_ensemble = dfs[key]
        if ensemble_list is not None:
            df_ensemble = df_ensemble[prim_keys + ensemble_list]
            if merge_key == "key_sale" and "key" in df_ensemble:
                df_ensemble = df_ensemble.drop(columns=["key"])
            df = df_basic.merge(df_ensemble, on=merge_key, how="left")
        else:
            df = df_basic
        df.to_parquet(f"{path}/pred_{key}.parquet")
        df.to_csv(f"{path}/pred_{key}.csv", index=False)


def _write_ensemble_meta(path: str, ensemble_type: str, members: list[str]):
    """Stamp the ensemble output directory with how it was produced.

    Only one ensemble type runs per model_group x (main/vacant), so the
    canonical ``{outpath}/ensemble`` directory is reused across types; this
    marker makes the output self-describing.
    """
    os.makedirs(path, exist_ok=True)
    meta = {"type": ensemble_type, "members": list(members)}
    with open(f"{path}/ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _find_member_contrib_file(outpath: str, m_key: str, candidate_files: list[str]):
    """Locate a member model's contributions CSV, tolerating the univ/universe
    filename inconsistency (tree writers use ``univ``, MRA/GWR use ``universe``)."""
    for fname in candidate_files:
        p = f"{outpath}/{m_key}/{fname}"
        if os.path.exists(p):
            return p
    return None


def _median_weights(P: pd.DataFrame) -> pd.DataFrame:
    """Per-(member, row) weights reproducing a row-wise median.

    A median over the members present in a row equals (odd count) the single
    central member's prediction, or (even count) the mean of the two central
    members. So the median is a convex combination of members: weight ``1`` on
    the central member, or ``0.5`` on each of the two central members. Member
    presence is per row (NaN predictions are skipped, matching
    ``pandas.median(axis=1)``).

    Parameters
    ----------
    P : pandas.DataFrame
        Per-member prediction matrix (rows x members), NaN where a member has no
        prediction for that row.

    Returns
    -------
    pandas.DataFrame
        Weights with the same shape/index/columns as ``P``; each row sums to 1
        (or 0 for rows with no present members).
    """
    members = list(P.columns)
    vals = P.to_numpy(dtype=float)
    R, M = vals.shape
    W = np.zeros((R, M), dtype=float)
    if M == 0:
        return pd.DataFrame(W, index=P.index, columns=members)

    finite = np.isfinite(vals)
    n = finite.sum(axis=1)
    # Sort each row ascending with absent members pushed to the end.
    filled = np.where(finite, vals, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")  # column indices, ascending
    rows = np.arange(R)

    odd = (n % 2 == 1) & (n > 0)
    even = (n % 2 == 0) & (n > 0)

    # Odd: the member at the central sorted rank carries the whole weight.
    k_mid = np.clip((n - 1) // 2, 0, M - 1)
    cols_mid = order[rows, k_mid]
    W[rows[odd], cols_mid[odd]] = 1.0

    # Even: the two central members split the weight 50/50.
    k_lo = np.clip(n // 2 - 1, 0, M - 1)
    k_hi = np.clip(n // 2, 0, M - 1)
    cols_lo = order[rows, k_lo]
    cols_hi = order[rows, k_hi]
    W[rows[even], cols_lo[even]] = 0.5
    W[rows[even], cols_hi[even]] = 0.5

    return pd.DataFrame(W, index=P.index, columns=members)


def _write_ensemble_contributions(
    results: SingleModelResults,
    outpath: str,
    settings: dict,
    ensemble_list: list[str],
    all_results: MultiModelResults,
    mode: str,
    local_selection: dict[str, pd.Series] | None = None,
    verbose: bool = False,
):
    """Assemble ensemble ``contributions_*`` and ``params_*`` files from members.

    Each supported ensemble is, per row, a convex combination of its members'
    predictions, so the ensemble's per-feature contribution is the matching
    per-row weighted sum of member contributions (weights summing to 1 per row):

    - ``mean``: equal weight across members present for the row.
    - ``local``: weight 1 on the row's selected model.
    - ``median``: weight 1 on the central member (odd present-count) or 0.5 on
      each of the two central members (even present-count).

    We accumulate the weighted feature contributions and set the base term as the
    residual ``prediction - sum(feature contributions)`` so reconstruction is
    exact and non-decomposable members (e.g. ``local_area``) fold cleanly into
    the base.

    Parameters
    ----------
    results : SingleModelResults
        The ensemble result; its ``df_test``/``df_sales``/``df_universe`` carry
        the final ensemble ``prediction`` column and the raw feature values.
    outpath : str
        Parent model output directory; the ensemble dir is ``{outpath}/ensemble``.
    ensemble_list : list[str]
        Member model keys that compose the ensemble.
    all_results : MultiModelResults
        Used to read each member's per-row predictions (mean/median weighting).
    mode : str
        ``"mean"``, ``"median"``, or ``"local"``.
    local_selection : dict[str, pandas.Series], optional
        For ``mode="local"``: per-subset Series mapping merge key -> selected
        model name (the painted ``local_model`` column).
    """
    path = f"{outpath}/{results.model_name}"
    os.makedirs(path, exist_ok=True)

    members = [m for m in ensemble_list if m in all_results.model_results]

    subset_specs = [
        ("test", results.df_test, "key_sale", ["contributions_test.csv"]),
        ("sales", results.df_sales, "key_sale", ["contributions_sales.csv"]),
        (
            # "universe" is canonical; "univ" is read for back-compat with older runs.
            "universe",
            results.df_universe,
            "key",
            ["contributions_universe.csv", "contributions_univ.csv"],
        ),
    ]

    warned_missing = set()

    for name, ref_df, merge_key, candidate_files in subset_specs:
        if ref_df is None or len(ref_df) == 0 or merge_key not in ref_df.columns:
            continue
        if "prediction" not in ref_df.columns:
            continue

        ref_index = pd.Index(ref_df[merge_key].astype(str), name=merge_key)
        # Drop any duplicate keys defensively (keys should be unique per subset).
        keep_mask = ~ref_index.duplicated()
        ref_df = ref_df.loc[keep_mask.tolist()]
        ref_index = ref_index[keep_mask]

        # Per-(member, row) weights, summing to 1 per row.
        weights = {}
        if mode in ("mean", "median"):
            # Per-member prediction matrix aligned to the ensemble rows.
            P = pd.DataFrame(index=ref_index)
            for m_key in members:
                mr = all_results.model_results[m_key]
                if name == "test":
                    kk, pv = mr.df_test[merge_key], mr.pred_test.y_pred
                elif name == "sales":
                    kk, pv = mr.df_sales[merge_key], mr.pred_sales.y_pred
                else:
                    kk, pv = mr.df_universe[merge_key], mr.pred_univ
                s = pd.Series(
                    np.asarray(pv, dtype=float),
                    index=pd.Index(kk.astype(str), name=merge_key),
                )
                s = s[~s.index.duplicated()]
                P[m_key] = s.reindex(ref_index)
            if mode == "mean":
                # Equal weight across the members present for the row.
                present = P.notna()
                n_row = present.sum(axis=1).replace(0, np.nan)
                for m_key in members:
                    weights[m_key] = present[m_key].astype(float) / n_row
            else:
                # Median == central member(s) per row.
                wmat = _median_weights(P)
                for m_key in members:
                    weights[m_key] = wmat[m_key]
        elif mode == "local":
            sel = local_selection.get(name) if local_selection else None
            if sel is None:
                continue
            sel = sel.reindex(ref_index)
            for m_key in members:
                weights[m_key] = (sel == m_key).astype(float)
        else:
            raise ValueError(f"Unrecognized ensemble contribution mode \"{mode}\"!")

        # Accumulate per-row weighted feature contributions across members.
        feat_total = pd.DataFrame(index=ref_index)
        for m_key in members:
            w = weights[m_key].reindex(ref_index).fillna(0.0)
            # Log-transformed members (mra/multi_mra with log=True) have contributions that are
            # additive in LOG space, not price space — they cannot be combined with the price-space
            # members. Exclude them from the feature attribution; their prediction still folds into
            # the ensemble base via the residual below (the ensemble *prediction* is unaffected).
            member_model = getattr(all_results.model_results.get(m_key), "model", None)
            if getattr(member_model, "log", False):
                if m_key not in warned_missing:
                    warnings.warn(
                        f"Ensemble member '{m_key}' is log-transformed (log=True); its "
                        f"contributions are in log space (written as log_contributions_*.csv) and "
                        f"cannot be meaningfully combined with price-space members, so its "
                        f"prediction folds into the ensemble base instead of being attributed to "
                        f"features."
                    )
                    warned_missing.add(m_key)
                continue
            cfile = _find_member_contrib_file(outpath, m_key, candidate_files)
            if cfile is None:
                # Non-decomposable (or unsaved) member -> folds into the base
                # via the residual below. Nothing to attribute to features.
                if m_key not in warned_missing:
                    warnings.warn(
                        f"Ensemble member '{m_key}' has no {candidate_files[0]}; "
                        f"folding its prediction into the ensemble base."
                    )
                    warned_missing.add(m_key)
                continue
            dfc = pd.read_csv(cfile)
            if merge_key not in dfc.columns:
                continue
            base_col = (
                "base_value"
                if "base_value" in dfc.columns
                else ("intercept" if "intercept" in dfc.columns else None)
            )
            drop = {merge_key, "key", "key_sale", "contribution_sum", "prediction", "check_delta"}
            if base_col is not None:
                drop.add(base_col)
            feat_cols = [c for c in dfc.columns if c not in drop]
            dfc[merge_key] = dfc[merge_key].astype(str)
            dfc = dfc[~dfc[merge_key].duplicated()].set_index(merge_key)
            for c in feat_cols:
                contrib_c = pd.to_numeric(
                    dfc[c].reindex(ref_index), errors="coerce"
                ).fillna(0.0)
                weighted = contrib_c * w
                if c in feat_total.columns:
                    feat_total[c] = feat_total[c] + weighted
                else:
                    feat_total[c] = weighted

        feat_cols = list(feat_total.columns)
        ens_pred = pd.to_numeric(
            pd.Series(ref_df["prediction"].values, index=ref_index), errors="coerce"
        )
        feat_sum = (
            feat_total.sum(axis=1) if feat_cols else pd.Series(0.0, index=ref_index)
        )
        # Residual base: guarantees contribution_sum == prediction (check_delta ~ 0)
        # and cleanly absorbs non-decomposable members and any missing-row slack.
        base_value = ens_pred - feat_sum

        out = pd.DataFrame(index=ref_index)
        out["base_value"] = base_value
        for c in feat_cols:
            out[c] = feat_total[c]
        out["contribution_sum"] = out[["base_value"] + feat_cols].sum(axis=1)
        out = out.reset_index()  # restores the merge_key column

        # Match the per-model contributions schema: key first, then key_sale.
        if merge_key == "key_sale" and "key" in ref_df.columns:
            key_map = dict(
                zip(ref_df[merge_key].astype(str), ref_df["key"].astype(str))
            )
            out.insert(0, "key", out["key_sale"].map(key_map))
        ordered = (
            [c for c in ["key", "key_sale"] if c in out.columns]
            + ["base_value"]
            + feat_cols
            + ["contribution_sum"]
        )
        out = out[ordered]

        df_final = _add_prediction_to_contribution(ref_df, out, split_name=name)

        max_delta = float(np.nanmax(np.abs(df_final["check_delta"].to_numpy()))) if len(df_final) else 0.0
        tol = 1e-6 * (1.0 + float(np.nanmean(np.abs(ens_pred.to_numpy()))) if len(ens_pred) else 1.0)
        if max_delta > tol:
            warnings.warn(
                f"Ensemble contributions for '{name}' don't reconstruct the "
                f"prediction (max|check_delta| = {max_delta:.4g})."
            )

        df_final.to_csv(f"{path}/contributions_{name}.csv", index=False)

        df_unit = _contrib_to_unit_values(out, ref_df, split_name=name)
        df_unit.to_csv(f"{path}/params_{name}.csv", index=False)

        if verbose:
            print(f"Wrote ensemble contributions/params for '{name}' (max|check_delta| = {max_delta:.4g})")

        # Universe-only contributions map, mirroring _write_model_results.
        if name == "universe":
            try:
                universe_parquet = gpd.read_parquet(f"{path}/pred_universe.parquet")
            except FileNotFoundError:
                universe_parquet = None
            if universe_parquet is not None:
                cmap = get_shap_contributions_map(
                    universe_parquet, results.df_universe, df_final
                )
                cmap.to_parquet(f"{path}/contributions_map.parquet")
                if verbose:
                    print(f"Wrote ensemble contributions map to {path}/contributions_map.parquet")


def _run_local_ensemble(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    dep_var: str,
    dep_var_test: str,
    all_results: MultiModelResults,
    settings: dict,
    outpath: str,
    verbose: bool = False,
    locations: list[str] = None,
):
    """
    Optimize the ensemble allocation over all iterations.
    """
    timing = TimingData()
    timing.start("total")
    timing.start("setup")

    first_key = list(all_results.model_results.keys())[0]
    test_keys = all_results.model_results[first_key].ds.test_keys
    train_keys = all_results.model_results[first_key].ds.train_keys

    if df_sales is None:
        df_universe = all_results.df_univ_orig
        df_sales = all_results.df_sales_orig

    ds = DataSplit(
        "ensemble",
        df_sales,
        df_universe,
        model_group,
        settings,
        dep_var,
        dep_var_test,
        [],
        [],
        {},
        test_keys,
        train_keys,
        vacant_only=vacant_only,
    )

    vacant_status = "vacant" if vacant_only else "main"
    df_test = ds.df_test
    df_train = ds.df_train
    df_sales = ds.df_sales
    df_univ = ds.df_universe
    
    if locations is None:
        locations = []
        warnings.warn("You didn't provide any locations! Local ensemble won't be very effective.")
    
    return _run_local_ensemble_test_and_paint(
        df_test=df_test,
        df_train=df_train,
        df_sales=df_sales,
        df_univ=df_univ,
        settings=settings,
        timing=timing,
        all_results=all_results,
        ds=ds,
        locations=locations,
        outpath=outpath,
        verbose=verbose
    )


def calc_df_mape(
    df: pd.DataFrame,
    field_prediction: str,
    settings: dict,
    dep_var: str,
    is_land_predictions: bool = False
):
    """
    Calculate MAPE 

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe you want to calculate MAPE for
    field_prediction : str
        The field name for predictions.
    settings : dict
        Settings dictionary
    dep_var: str
        The field you're trying to predict
    is_land_predictions: bool
        Are you predicting land values or not. If true, uses the `valid_for_land_ratio_study` validity flag.
    """
    
    # Clean arrays
    y = df[dep_var].to_numpy()
    df[dep_var] = pd.to_numeric(df[dep_var], errors="coerce")
    df[field_prediction] = pd.to_numeric(df[field_prediction], errors="coerce")
    
    # Get validity field
    valid_field = "valid_for_ratio_study"
    if is_land_predictions:
        valid_field = "valid_for_land_ratio_study"
        
    # select only values that are not NaN in either and are valid for ratio study:
    df_clean = df[
        df[valid_field] & 
        ~pd.isna(df[dep_var]) & 
        ~pd.isna(df[field_prediction])
    ]
    
    # Get y & y_pred
    y = df_clean[dep_var].to_numpy()
    y_pred = df_clean[field_prediction].to_numpy()
    
    # Calculate MAPE
    if len(y) > 0 and len(y_pred) > 0:
        return mean_absolute_percentage_error(y, y_pred)
    
    return float("nan")


def _test_mape_local_ensemble(
    df: pd.DataFrame,
    location_field: str,
    location_value: str,
    model_keys: list[str],
    settings: dict,
    dep_var: str,
    is_land_predictions: bool,
    best_mape: float = float('inf'),
    verbose: bool = False
):
    if location_field != "":
        df_slice = df[df[location_field].eq(location_value)].copy()
    else:
        df_slice = df
    best_model = None
    
    for key in model_keys:
        mape = calc_df_mape(df_slice, key, settings, dep_var, is_land_predictions)
        if mape < best_mape:
            best_model = key
            best_mape = mape
            if verbose:
                print(f"----> loc = {location_value:<10}, model = {key:<12}, mape = {best_mape:>6.3f}, samples = {len(df_slice):<12}")
    return best_model, best_mape


def _paint_best_local_ensemble(
    df_in: pd.DataFrame,
    locations: list[str],
    model_keys: list[str],
    settings: dict,
    dep_var: str,
    is_land_predictions: bool,
    best_map: dict
):
    df = df_in.copy()
    df["prediction"] = float('nan')
    df["local_model"] = None
    df["local_mape"] = None
    
    print(f"Painting location values...")
    
    for location in locations:
        if location == "":
            loc_values = [""]
        else:
            loc_values = df[location].unique()
        
        print(f"-->location = {location} loc_values = {len(loc_values)}")
        entry = best_map[location]
        
        for loc_value in loc_values:
            if loc_value in entry:
                loc_entry = entry[loc_value]
                model = loc_entry["model"]
                mape = loc_entry["mape"]
                if model is None:
                    continue
                if mape is None:
                    mape = float("nan")
                print(f"----> @ = {str(loc_value):<12}, model = {str(model):<12}, mape = {mape:>6.2f}")
                if location == "":
                    df["prediction"] = df[model]
                    df["local_model"] = model
                    df["local_mape"] = mape
                else:
                    df.loc[df[location].eq(loc_value), "prediction"] = df[model]
                    df.loc[df[location].eq(loc_value), "local_model"] = model
                    df.loc[df[location].eq(loc_value), "local_mape"] = mape
    return df


def _run_local_ensemble_test_and_paint(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    df_sales: pd.DataFrame,
    df_univ: pd.DataFrame,
    settings: dict,
    timing: TimingData,
    all_results: MultiModelResults,
    ds: DataSplit,
    locations: list[str],
    outpath: str,
    verbose: bool = False,
):
    # Local ensemble selects the best model per location value, so a collapsed
    # location silently merges unrelated zones — warn (or raise, if strict).
    warn_if_location_collapsed(
        settings, locations, context="local ensemble model selection"
    )
    # Get all the dataframes we need, ensure they have keys + location fields
    timing.start("setup")
    valid_field = "valid_for_ratio_study"
    if ds.is_land_predictions():
        valid_field = "valid_for_land_ratio_study"
    df_test_ensemble = df_test[["key_sale", "key", ds.dep_var_test, valid_field]+locations].copy()
    df_train_ensemble = df_train[["key_sale", "key", ds.dep_var_test, valid_field]+locations].copy()
    df_sales_ensemble = df_sales[["key_sale", "key", ds.dep_var_test, valid_field]+locations].copy()
    df_univ_ensemble = df_univ[["key"]+locations].copy()
    timing.stop("setup")

    timing.start("parameter_search")
    timing.stop("parameter_search")

    # Set up the dataframes so that we have one prediction column per input model
    timing.start("train")
    model_keys = [key for key in all_results.model_results.keys() if key not in ["assessor", "ground_truth"]]
    for m_key in model_keys:
        m_results: SingleModelResults = all_results.model_results[m_key]
        field_prediction = m_results.field_prediction
        df_pred_test = m_results.df_test[["key_sale", field_prediction]].copy()
        df_pred_test = df_pred_test.rename(columns={field_prediction: m_key})
        
        df_pred_train = m_results.df_train[["key_sale", field_prediction]].copy()
        df_pred_train = df_pred_train.rename(columns={field_prediction: m_key})
        
        df_pred_sales = m_results.df_sales[["key_sale", field_prediction]].copy()
        df_pred_sales = df_pred_sales.rename(columns={field_prediction: m_key})

        df_pred_univ = m_results.df_universe[["key", field_prediction]].copy()
        df_pred_univ = df_pred_univ.rename(columns={field_prediction: m_key})

        df_test_ensemble = df_test_ensemble.merge(df_pred_test, on="key_sale", how="left")
        df_train_ensemble = df_train_ensemble.merge(df_pred_train, on="key_sale", how="left")
        df_sales_ensemble = df_sales_ensemble.merge(df_pred_sales, on="key_sale", how="left")
        df_univ_ensemble = df_univ_ensemble.merge(df_pred_univ, on="key", how="left")
    timing.stop("train")
    
    # prepare some variables
    all_locations = [""] + locations
    dep_var = ds.dep_var
    is_land_predictions = ds.is_land_predictions()
    best_map = {}
    
    # FOr each location field, find the best performing model for each of its unique location values
    for location in all_locations:
        loc_entry = {}
        
        # If location is empty, it means the whole dataframe
        if location == "":
            loc_values = [""]
        else:
            loc_values = df_sales[location].unique()
        
        if verbose:
            print(f"--> location = {location} loc_values = {len(loc_values)}")
        
        for loc_value in loc_values:
            # Find the best model/best mape for this specific location value
            best_model, best_mape = _test_mape_local_ensemble(
                df=df_train_ensemble,
                location_field=location,
                location_value=loc_value,
                model_keys=model_keys,
                settings=settings,
                dep_var=dep_var,
                is_land_predictions=is_land_predictions,
                verbose=verbose
            )
            # Stash the results for this unique location value
            loc_entry[loc_value] = {
                "model": best_model,
                "mape": best_mape
            }
        # Stash the results for the entire location field
        best_map[location] = loc_entry
    
    # Now we generate whole-dataframe predictions by using the best model for each location
    # When multiple values are given, we favor the most specific results given
    timing.start("predict_test")
    df_test_ensemble = _paint_best_local_ensemble(df_test_ensemble, locations, model_keys, settings, dep_var, is_land_predictions, best_map)
    y_pred_test = df_test_ensemble["prediction"]
    timing.stop("predict_test")

    timing.start("predict_sales")
    df_sales_ensemble = _paint_best_local_ensemble(df_sales_ensemble, locations, model_keys, settings, dep_var, is_land_predictions, best_map)
    y_pred_sales = df_sales_ensemble["prediction"]
    timing.stop("predict_sales")

    timing.start("predict_univ")
    df_univ_ensemble = _paint_best_local_ensemble(df_univ_ensemble, locations, model_keys, settings, dep_var, is_land_predictions, best_map)
    y_pred_univ = df_univ_ensemble["prediction"]
    timing.stop("predict_univ")
    
    # Generate a SingleModelResults object for the whole ensemble
    results : SingleModelResults = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        model_name="ensemble",
        model_engine="ensemble",
        model="ensemble",
        y_pred_test=y_pred_test.to_numpy(),
        y_pred_sales=y_pred_sales.to_numpy(),
        y_pred_univ=y_pred_univ.to_numpy(),
        timing=timing,
        verbose=verbose
    )
    timing.stop("total")
    
    print(f"Results: score = {results.utility_sales_lookback}, r2 = {results.pred_sales_lookback.r2}, mape = {results.pred_sales_lookback.mape}, rmse = {results.pred_sales_lookback.rmse}")
    score = results.utility_sales_lookback
    
    dfs = {
        "sales": df_sales_ensemble,
        "universe": df_univ_ensemble,
        "test": df_test_ensemble,
    }

    _write_ensemble_model_results(results, outpath, settings, dfs, model_keys+["local_model","local_mape"])

    results.ensemble_type = "local"
    _write_ensemble_meta(f"{outpath}/{results.model_name}", "local", model_keys)

    # Local ensembles select exactly one member per row, so contributions are an
    # exact per-row pass-through of the painted model's contributions.
    local_selection = {
        "test": pd.Series(
            df_test_ensemble["local_model"].values,
            index=df_test_ensemble["key_sale"].astype(str),
        ),
        "sales": pd.Series(
            df_sales_ensemble["local_model"].values,
            index=df_sales_ensemble["key_sale"].astype(str),
        ),
        "universe": pd.Series(
            df_univ_ensemble["local_model"].values,
            index=df_univ_ensemble["key"].astype(str),
        ),
    }
    _write_ensemble_contributions(
        results,
        outpath,
        settings,
        model_keys,
        all_results,
        mode="local",
        local_selection=local_selection,
        verbose=verbose,
    )

    return results


def _optimize_ensemble(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    dep_var: str,
    dep_var_test: str,
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
    ensemble_list: list[str] = None,
    agg: str = "median",
):
    """
    Optimize the ensemble over all iterations.
    """
    timing = TimingData()
    timing.start("total")
    timing.start("setup")

    first_key = list(all_results.model_results.keys())[0]
    test_keys = all_results.model_results[first_key].ds.test_keys
    train_keys = all_results.model_results[first_key].ds.train_keys

    if df_sales is None:
        df_universe = all_results.df_univ_orig
        df_sales = all_results.df_sales_orig

    ds = DataSplit(
        "ensemble",
        df_sales,
        df_universe,
        model_group,
        settings,
        dep_var,
        dep_var_test,
        [],
        [],
        {},
        test_keys,
        train_keys,
        vacant_only=vacant_only,
    )

    vacant_status = "vacant" if vacant_only else "main"
    df_test = ds.df_test
    df_sales = ds.df_sales
    df_univ = ds.df_universe
    instructions = settings.get("modeling", {}).get("instructions", {})
    
    if ensemble_list is None:
        ensemble_inst = get_ensemble_instructions(settings, vacant_status)
        ensemble_list = list(ensemble_inst.get("models", []))

    if len(ensemble_list) == 0:
        ensemble_list = [key for key in all_results.model_results.keys()]

    if "assessor" in ensemble_list:
        ensemble_list.remove("assessor")

    if "ground_truth" in ensemble_list:
        ensemble_list.remove("ground_truth")

    best_list = []
    best_score = float("inf")

    while len(ensemble_list) > 1:
        if verbose:
            print(f"Ensembling with : {ensemble_list}")
        best_score, best_list = _optimize_ensemble_iteration(
            df_test,
            df_sales,
            df_univ,
            timing,
            all_results,
            ds,
            best_score,
            best_list,
            ensemble_list,
            verbose,
            agg=agg,
        )

    if verbose:
        print(f"-->Ensemble finished. Best score = {best_score:8.2f}, ensemble = {best_list}")
    return best_list


def _optimize_ensemble_iteration(
    df_test: pd.DataFrame,
    df_sales: pd.DataFrame,
    df_univ: pd.DataFrame,
    timing: TimingData,
    all_results: MultiModelResults,
    ds: DataSplit,
    best_score: float,
    best_list: list[str],
    ensemble_list: list[str],
    verbose: bool = False,
    agg: str = "median",
):
    df_test_ensemble = df_test[["key_sale", "key"]].copy()
    df_sales_ensemble = df_sales[["key_sale", "key"]].copy()
    df_univ_ensemble = df_univ[["key"]].copy()
    if len(ensemble_list) == 0:
        ensemble_list = [key for key in all_results.model_results.keys()]
    timing.stop("setup")

    timing.start("parameter_search")
    timing.stop("parameter_search")

    timing.start("train")
    for m_key in ensemble_list:
        m_results: SingleModelResults = all_results.model_results[m_key]
        field_prediction = m_results.field_prediction
        df_pred_test = m_results.df_test[["key_sale", field_prediction]].copy()
        df_pred_test = df_pred_test.rename(columns={field_prediction: m_key})

        df_pred_sales = m_results.df_sales[["key_sale", field_prediction]].copy()
        df_pred_sales = df_pred_sales.rename(columns={field_prediction: m_key})

        df_pred_univ = m_results.df_universe[["key", field_prediction]].copy()
        df_pred_univ = df_pred_univ.rename(columns={field_prediction: m_key})

        df_test_ensemble = df_test_ensemble.merge(
            df_pred_test, on="key_sale", how="left"
        )
        
        df_sales_ensemble = df_sales_ensemble.merge(
            df_pred_sales, on="key_sale", how="left"
        )
        df_univ_ensemble = df_univ_ensemble.merge(df_pred_univ, on="key", how="left")
    timing.stop("train")

    timing.start("predict_test")
    y_pred_test_ensemble = _aggregate_ensemble(df_test_ensemble, ensemble_list, agg)
    timing.stop("predict_test")

    timing.start("predict_sales")
    y_pred_sales_ensemble = _aggregate_ensemble(df_sales_ensemble, ensemble_list, agg)
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ_ensemble = _aggregate_ensemble(df_univ_ensemble, ensemble_list, agg)
    timing.stop("predict_univ")

    results : SingleModelResults = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        model_name="ensemble",
        model_engine="ensemble",
        model="ensemble",
        y_pred_test=y_pred_test_ensemble.to_numpy(),
        y_pred_sales=y_pred_sales_ensemble.to_numpy(),
        y_pred_univ=y_pred_univ_ensemble.to_numpy(),
        timing=timing,
        verbose=verbose
    )
    timing.stop("total")

    print(f"Results: score = {results.utility_sales_lookback}, r2 = {results.pred_sales_lookback.r2}, mape = {results.pred_sales_lookback.mape}, rmse = {results.pred_sales_lookback.rmse}")
    score = results.utility_sales_lookback

    # Add early exit if score is nan
    if pd.isna(score):
        print("Warning: Got NaN score, stopping ensemble optimization")
        ensemble_list.clear()  # Clear the list to force the loop to end
        return float("inf"), []

    if verbose:
        print(
            f"score = {score:5.2f}, best = {best_score:5.2f}, ensemble = {ensemble_list}..."
        )

    if score < best_score:  # and len(ensemble_list) >= 3:
        best_score = score
        best_list = ensemble_list.copy()

    # identify the WORST individual model:
    worst_model = None
    worst_score = float("-inf")
    for key in ensemble_list:
        if key in all_results.model_results:
            model_results = all_results.model_results[key]

            model_score = model_results.utility_sales_lookback

            if model_score > worst_score:
                worst_score = model_score
                worst_model = key

    if worst_model is not None and len(ensemble_list) > 1:
        ensemble_list.remove(worst_model)

    return best_score, best_list


def _run_ensemble(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    model_group: str,
    vacant_only: bool,
    dep_var: str,
    dep_var_test: str,
    outpath: str,
    ensemble_list: list[str],
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
    agg: str = "median",
):
    """Run the ensemble model based on the given ensemble list and write results.
    """
    timing = TimingData()
    timing.start("total")
    timing.start("setup")

    first_key = list(all_results.model_results.keys())[0]
    test_keys = all_results.model_results[first_key].ds.test_keys
    train_keys = all_results.model_results[first_key].ds.train_keys

    ds = DataSplit(
        "ensemble",
        df_sales,
        df_universe,
        model_group,
        settings,
        dep_var,
        dep_var_test,
        [],
        [],
        {},
        test_keys,
        train_keys,
        vacant_only=vacant_only,
    )
    ds.split()

    df_test = ds.df_test
    df_sales = ds.df_sales
    df_univ = ds.df_universe

    df_test_ensemble = df_test[["key_sale", "key"]].copy()
    df_sales_ensemble = df_sales[["key_sale", "key"]].copy()
    df_univ_ensemble = df_univ[["key"]].copy()

    if len(ensemble_list) == 0:
        ensemble_list = [key for key in all_results.model_results.keys()]
    timing.stop("setup")

    timing.start("parameter_search")
    timing.stop("parameter_search")
    timing.start("train")
    for m_key in ensemble_list:
        m_results = all_results.model_results[m_key]
        
        _df_test = m_results.df_test[["key_sale"]].copy()
        _df_test.loc[:, m_key] = m_results.pred_test.y_pred
        
        _df_sales = m_results.df_sales[["key_sale"]].copy()
        _df_sales.loc[:, m_key] = m_results.pred_sales.y_pred
        
        _df_univ = m_results.df_universe[["key"]].copy()
        _df_univ.loc[:, m_key] = m_results.pred_univ
        
        df_test_ensemble = df_test_ensemble.merge(_df_test, on="key_sale", how="left")
        df_sales_ensemble = df_sales_ensemble.merge(
            _df_sales, on="key_sale", how="left"
        )
        df_univ_ensemble = df_univ_ensemble.merge(_df_univ, on="key", how="left")

    timing.stop("train")

    timing.start("predict_test")
    y_pred_test_ensemble = _aggregate_ensemble(df_test_ensemble, ensemble_list, agg)
    timing.stop("predict_test")

    timing.start("predict_sales")
    y_pred_sales_ensemble = _aggregate_ensemble(df_sales_ensemble, ensemble_list, agg)
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ_ensemble = _aggregate_ensemble(df_univ_ensemble, ensemble_list, agg)
    timing.stop("predict_univ")

    results = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        model_name="ensemble",
        model_engine="ensemble",
        model="ensemble",
        y_pred_test=y_pred_test_ensemble.to_numpy(),
        y_pred_sales=y_pred_sales_ensemble.to_numpy(),
        y_pred_univ=y_pred_univ_ensemble.to_numpy(),
        timing=timing,
        verbose=verbose,
    )
    timing.stop("total")

    dfs = {
        "sales": df_sales_ensemble,
        "universe": df_univ_ensemble,
        "test": df_test_ensemble,
    }

    _write_ensemble_model_results(results, outpath, settings, dfs, ensemble_list)

    ensemble_type = "mean" if agg == "mean" else "median"
    results.ensemble_type = ensemble_type
    _write_ensemble_meta(f"{outpath}/{results.model_name}", ensemble_type, ensemble_list)

    # Both mean and median are per-row convex combinations of the members, so
    # contributions/params can be reassembled exactly (mean = equal weights;
    # median = central member, or the two central members averaged).
    if agg in ("mean", "median"):
        _write_ensemble_contributions(
            results,
            outpath,
            settings,
            ensemble_list,
            all_results,
            mode=agg,
            verbose=verbose,
        )

    return results


def _prepare_ds(
    name: str,
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    model_group: str,
    vacant_only: bool,
    settings: dict,
    ind_vars: list[str] | None = None,
):
    """Prepare a DataSplit object for modeling.
    
    """
    s = settings
    s_model = s.get("modeling", {})
    vacant_status = "vacant" if vacant_only else "main"
    model_entries = s_model.get("models", {}).get(vacant_status, {})
    model_entries = model_entries.get(model_group, model_entries)
    entry: dict | None = model_entries.get("model", model_entries.get("default", {}))

    if ind_vars is None:
        ind_vars: list | None = entry.get("ind_vars", None)
        if ind_vars is None:
            raise ValueError(f"ind_vars not found for model 'default'")

    # Check for duplicate variables in ind_vars
    if ind_vars is not None:
        seen_vars = set()
        duplicates = []
        deduped_vars = []

        for var in ind_vars:
            if var in seen_vars:
                duplicates.append(var)
            else:
                seen_vars.add(var)
                deduped_vars.append(var)

        if duplicates:
            print(f"\n⚠️ WARNING: Found duplicate variables in ind_vars: {duplicates}")
            print(f"Using only the first occurrence of each variable to avoid errors.")
            ind_vars = deduped_vars

    # Check for duplicate columns in DataFrame (e.g., from merges)
    duplicate_cols = df_sales.columns[df_sales.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"\n⚠️ WARNING: Found duplicate columns in DataFrame: {duplicate_cols}")
        print(f"This could cause errors. Keeping only first occurrence of each column.")
        df_sales = df_sales.loc[:, ~df_sales.columns.duplicated()]

    duplicate_cols_univ = df_universe.columns[df_universe.columns.duplicated()].tolist()
    if duplicate_cols_univ:
        print(
            f"\n⚠️ WARNING: Found duplicate columns in universe DataFrame: {duplicate_cols_univ}"
        )
        print(f"This could cause errors. Keeping only first occurrence of each column.")
        df_universe = df_universe.loc[:, ~df_universe.columns.duplicated()]

    fields_cat = get_fields_categorical(s, df_sales, include_boolean=True)
    interactions = get_variable_interactions(entry, s, df_sales)

    instructions = s.get("modeling", {}).get("instructions", {})
    dep_var = instructions.get("dep_var", "sale_price_time_adj")
    dep_var_test = instructions.get("dep_var_test", "sale_price_time_adj")

    test_keys, train_keys = _read_split_keys(model_group)

    ds = DataSplit(
        name=name,
        df_sales=df_sales,
        df_universe=df_universe,
        model_group=model_group,
        settings=settings,
        dep_var=dep_var,
        dep_var_test=dep_var_test,
        ind_vars=ind_vars,
        categorical_vars=fields_cat,
        interactions=interactions,
        test_keys=test_keys,
        train_keys=train_keys,
        vacant_only=vacant_only,
    )
    return ds


def _calc_variable_recommendations(
    ds: DataSplit,
    settings: dict,
    rep_results: dict,
    correlation_results: dict,
    enr_results: dict,
    r2_values_results: pd.DataFrame,
    p_values_results: dict,
    t_values_results: dict,
    vif_results: dict,
    report: MarkdownReport = None
):
    """Calculate variable recommendations based on various statistical metrics.
    """
    one_hot_descendents = ds.one_hot_descendants # Unpack this and delete it to get rid of some dataframe copies
    del ds
    feature_selection = (
        settings.get("modeling", {})
        .get("instructions", {})
        .get("feature_selection", {})
    )
    thresh = feature_selection.get("thresholds", {})
    weights = feature_selection.get("weights", {})
    
    stuff_to_merge = [
        correlation_results,
        {"final": r2_values_results},
        enr_results,
        p_values_results,
        t_values_results,
        vif_results,
    ]

    df: pd.DataFrame | None = None
    for thing in stuff_to_merge:
        if thing is None:
            continue
        if df is None:
            df = thing["final"]
        else:
            df = pd.merge(df, thing["final"], on="variable", how="outer")

    if df is None:
        raise ValueError("df is None, no data to merge")

    df["weighted_score"] = 0

    # remove "const" from df:
    df = df[df["variable"].ne("const")]

    adj_r2_thresh = thresh.get("adj_r2", 0.1)
    adj_r2_thresh_bonus = thresh.get("adj_r2_bonus", 0.25)

    # 1 point for being over the minimum amount
    df.loc[df["adj_r2"].gt(adj_r2_thresh), "weighted_score"] += 1

    if adj_r2_thresh_bonus > adj_r2_thresh:
        # 1 point for reaching a higher threshold
        df.loc[df["adj_r2"].gt(adj_r2_thresh_bonus), "weighted_score"] += 1

    weight_corr_score = weights.get("corr_score", 1)
    weight_enr_coef = weights.get("enr_coef", 1)
    weight_p_value = weights.get("p_value", 1)
    weight_t_value = weights.get("t_value", 1)
    weight_vif = weights.get("vif", 1)
    weight_coef_sign = weights.get("coef_sign", 1)

    if correlation_results is not None:
        df.loc[df["corr_score"].notna(), "weighted_score"] += weight_corr_score
    if enr_results is not None:
        df.loc[df["enr_coef"].notna(), "weighted_score"] += weight_enr_coef
    if p_values_results is not None:
        df.loc[df["p_value"].notna(), "weighted_score"] += weight_p_value
    if t_values_results is not None:
        df.loc[df["t_value"].notna(), "weighted_score"] += weight_t_value
    if vif_results is not None:
        df.loc[df["vif"].notna(), "weighted_score"] += weight_vif

    if t_values_results is not None and enr_results is not None:
        # check if "enr_coefficient", "t_value", and "coef_sign" are pointing in the same direction:
        df.loc[
            df["enr_coef_sign"].eq(df["t_value_sign"])
            & df["enr_coef_sign"].eq(df["coef_sign"]),
            "signs_match",
        ] = 1
        df.loc[df["signs_match"].eq(1), "weighted_score"] += weight_coef_sign

    bys = ["weighted_score"]
    ascs = [False]

    if "adj_r2" in df:
        bys.append("adj_r2")
        ascs.append(False)
    elif "r2" in df:
        bys.append("r2")
        ascs.append(False)

    df = df.sort_values(by=bys, ascending=ascs)

    if report is not None:
        dfr = df.copy()
        dfr = dfr.rename(
            columns={
                "variable": "Variable",
                "corr_score": "Correlation",
                "enr_coef": "ENR",
                "adj_r2": "R-squared",
                "p_value": "P Value",
                "t_value": "T Value",
                "vif": "VIF",
                "signs_match": "Coef. sign",
                "weighted_score": "Weighted Score",
            }
        )

        # Correlation:
        thresh_corr = thresh.get("correlation", 0.1)
        report.set_var("thresh_corr", thresh_corr, ".2f")
        corr_fields = ["variable", "corr_strength", "corr_clarity", "corr_score"]
        corr_renames = {
            "variable": "Variable",
            "corr_strength": "Strength",
            "corr_clarity": "Clarity",
            "corr_score": "Score",
        }
        
        # Representation:
        
        
        # VIF:
        thresh_vif = thresh.get("vif", 10)
        vif_renames = {"variable": "Variable", "vif": "VIF"}

        # P-value:
        thresh_p_value = thresh.get("p_value", 0.05)
        p_value_renames = {"variable": "Variable", "p_value": "P-value"}

        # T-value:
        thresh_t_value = thresh.get("t_value", 2)
        t_value_renames = {"variable": "Variable", "t_value": "T-value"}

        # ENR:
        thresh_enr = thresh.get("enr", 0.1)
        enr_renames = {"variable": "Variable", "enr_coef": "Coefficient"}

        # R-Squared:
        thresh_r2 = thresh.get("adj_r2", 0.1)
        r2_renames = {"variable": "Variable", "adj_r2": "R-squared"}

        # Coef signs:
        coef_sign_renames = {
            "variable": "Variable",
            "enr_coef_sign": "ENR sign",
            "t_value_sign": "T-value sign",
            "coef_sign": "Coef. sign",
        }

        for state in ["initial", "final"]:
            # Correlation:
            dfr_corr = correlation_results[state][corr_fields].copy()
            dfr_corr["Pass/Fail"] = dfr_corr["corr_score"].apply(
                lambda x: "✅" if x > thresh_corr else "❌"
            )
            for field in corr_fields:
                if field == "variable":
                    continue
                if field not in dfr_corr:
                    print("missing field", field)
                dfr_corr[field] = (
                    dfr_corr[field].apply(lambda x: f"{x:.2f}").astype("string")
                )

            dfr_corr = dfr_corr.rename(columns=corr_renames)
            dfr_corr["Rank"] = range(1, len(dfr_corr) + 1)
            dfr_corr = dfr_corr[
                ["Rank", "Variable", "Strength", "Clarity", "Score", "Pass/Fail"]
            ]
            dfr_corr.set_index("Rank", inplace=True)
            dfr_corr = _apply_dd_to_df_rows(
                dfr_corr, "Variable", settings, one_hot_descendents
            )
            report.set_var(f"table_corr_{state}", df_to_markdown(dfr_corr))

            # TODO: refactor this down to DRY it out a bit

            if vif_results is not None:
                # VIF:
                dfr_vif = vif_results[state][["variable", "vif"]].copy()
                dfr_vif = dfr_vif.sort_values(by="vif", ascending=True)
                dfr_vif["Pass/Fail"] = dfr_vif["vif"].apply(
                    lambda x: "✅" if x < thresh_vif else "❌"
                )
                dfr_vif["vif"] = (
                    dfr_vif["vif"]
                    .apply(
                        lambda x: (
                            f"{x:.2f}"
                            if x < 10
                            else f"{x:.1f}" if x < 100 else f"{x:,.0f}"
                        )
                    )
                    .astype("string")
                )
                dfr_vif = dfr_vif.rename(columns=vif_renames)
                dfr_vif["Rank"] = range(1, len(dfr_vif) + 1)
                dfr_vif = dfr_vif[["Rank", "Variable", "VIF", "Pass/Fail"]]
                dfr_vif.set_index("Rank", inplace=True)
                dfr_vif = _apply_dd_to_df_rows(
                    dfr_vif, "Variable", settings, one_hot_descendents
                )
                report.set_var(f"table_vif_{state}", df_to_markdown(dfr_vif))
            else:
                report.set_var(f"table_vif_{state}", "N/A")

            if p_values_results is not None:
                # P-value:
                dfr_p_value = p_values_results[state][["variable", "p_value"]].copy()
                dfr_p_value = dfr_p_value[dfr_p_value["variable"].ne("const")]
                dfr_p_value = dfr_p_value.sort_values(by="p_value", ascending=True)
                dfr_p_value["Pass/Fail"] = dfr_p_value["p_value"].apply(
                    lambda x: "✅" if x < thresh_p_value else "❌"
                )
                dfr_p_value["p_value"] = (
                    dfr_p_value["p_value"].apply(lambda x: f"{x:.3f}").astype("string")
                )
                dfr_p_value = dfr_p_value.rename(columns=p_value_renames)
                dfr_p_value["Rank"] = range(1, len(dfr_p_value) + 1)
                dfr_p_value = dfr_p_value[["Rank", "Variable", "P-value", "Pass/Fail"]]
                dfr_p_value.set_index("Rank", inplace=True)
                dfr_p_value = _apply_dd_to_df_rows(
                    dfr_p_value, "Variable", settings, one_hot_descendents
                )
                report.set_var(f"table_p_value_{state}", df_to_markdown(dfr_p_value))

            if t_values_results is not None:
                # T-value:
                dfr_t_value = t_values_results[state][["variable", "t_value"]].copy()
                dfr_t_value = dfr_t_value[dfr_t_value["variable"].ne("const")]
                dfr_t_value = dfr_t_value.sort_values(
                    by="t_value", ascending=False, key=abs
                )
                dfr_t_value["Pass/Fail"] = dfr_t_value["t_value"].apply(
                    lambda x: "✅" if abs(x) > thresh_t_value else "❌"
                )
                dfr_t_value["t_value"] = (
                    dfr_t_value["t_value"].apply(lambda x: f"{x:.2f}").astype("string")
                )
                dfr_t_value = dfr_t_value.rename(columns=t_value_renames)
                dfr_t_value["Rank"] = range(1, len(dfr_t_value) + 1)
                dfr_t_value = dfr_t_value[["Rank", "Variable", "T-value", "Pass/Fail"]]
                dfr_t_value.set_index("Rank", inplace=True)
                dfr_t_value = _apply_dd_to_df_rows(
                    dfr_t_value, "Variable", settings, one_hot_descendents
                )
                report.set_var(f"table_t_value_{state}", df_to_markdown(dfr_t_value))

            if enr_results is not None:
                # ENR:
                dfr_enr = enr_results[state][["variable", "enr_coef"]].copy()
                dfr_enr = dfr_enr.sort_values(by="enr_coef", ascending=False, key=abs)
                dfr_enr["Pass/Fail"] = dfr_enr["enr_coef"].apply(
                    lambda x: "✅" if abs(x) > thresh_enr else "❌"
                )
                dfr_enr["enr_coef"] = (
                    dfr_enr["enr_coef"]
                    .apply(lambda x: f"{x:.2f}" if abs(x) < 100 else f"{x:,.0f}")
                    .astype("string")
                )
                dfr_enr = dfr_enr.rename(columns=enr_renames)
                dfr_enr["Rank"] = range(1, len(dfr_enr) + 1)
                dfr_enr = dfr_enr[["Rank", "Variable", "Coefficient", "Pass/Fail"]]
                dfr_enr.set_index("Rank", inplace=True)
                dfr_enr = _apply_dd_to_df_rows(
                    dfr_enr, "Variable", settings, one_hot_descendents
                )
                report.set_var(f"table_enr_{state}", df_to_markdown(dfr_enr))

            if r2_values_results is not None:
                # R-squared
                dfr_r2 = r2_values_results.copy()
                dfr_r2 = dfr_r2.sort_values(by="adj_r2", ascending=False)
                dfr_r2["Pass/Fail"] = dfr_r2["adj_r2"].apply(
                    lambda x: "✅" if x > thresh_r2 else "❌"
                )
                dfr_r2["adj_r2"] = (
                    dfr_r2["adj_r2"].apply(lambda x: f"{x:.2f}").astype("string")
                )
                dfr_r2 = dfr_r2.rename(columns=r2_renames)
                dfr_r2["Rank"] = range(1, len(dfr_r2) + 1)
                dfr_r2 = dfr_r2[["Rank", "Variable", "R-squared", "Pass/Fail"]]
                dfr_r2.set_index("Rank", inplace=True)
                dfr_r2 = _apply_dd_to_df_rows(
                    dfr_r2, "Variable", settings, one_hot_descendents
                )
                if state == "final":
                    dfr_r2 = dfr_r2[dfr_r2["Pass/Fail"].eq("✅")]
                report.set_var(f"table_adj_r2_{state}", df_to_markdown(dfr_r2))

            if enr_results is not None and t_values_results is not None:
                # Coef sign:
                dfr_coef_sign = enr_results[state][["variable", "enr_coef_sign"]].copy()
                dfr_coef_sign = dfr_coef_sign.merge(
                    t_values_results[state][["variable", "t_value_sign"]],
                    on="variable",
                    how="outer",
                )
                dfr_coef_sign = dfr_coef_sign.merge(
                    r2_values_results[["variable", "coef_sign"]],
                    on="variable",
                    how="outer",
                )
                dfr_coef_sign["signs_match"] = False
                dfr_coef_sign.loc[
                    dfr_coef_sign["enr_coef_sign"].eq(dfr_coef_sign["t_value_sign"])
                    & dfr_coef_sign["enr_coef_sign"].eq(dfr_coef_sign["coef_sign"]),
                    "signs_match",
                ] = True
                dfr_coef_sign["Pass/Fail"] = dfr_coef_sign["signs_match"].apply(
                    lambda x: "✅" if x else "❌"
                )
                dfr_coef_sign = dfr_coef_sign.sort_values(
                    by="signs_match", ascending=False
                )
                dfr_coef_sign = dfr_coef_sign[dfr_coef_sign["variable"].ne("const")]
                dfr_coef_sign = dfr_coef_sign.rename(columns=coef_sign_renames)
                dfr_coef_sign = dfr_coef_sign[
                    ["Variable", "ENR sign", "T-value sign", "Coef. sign", "Pass/Fail"]
                ]
                for field in ["ENR sign", "T-value sign", "Coef. sign"]:
                    dfr_coef_sign[field] = (
                        dfr_coef_sign[field]
                        .apply(lambda x: f"{x:.0f}")
                        .astype("string")
                    )
                dfr_coef_sign = _apply_dd_to_df_rows(
                    dfr_coef_sign, "Variable", settings, one_hot_descendents
                )
                if state == "final":
                    dfr_coef_sign = dfr_coef_sign[dfr_coef_sign["Pass/Fail"].eq("✅")]
                report.set_var(
                    f"table_coef_sign_{state}", df_to_markdown(dfr_coef_sign)
                )

        dfr["Rank"] = range(1, len(dfr) + 1)
        dfr = _apply_dd_to_df_rows(dfr, "Variable", settings, one_hot_descendents)

        the_cols = [
            "Rank",
            "Weighted Score",
            "Variable",
            "VIF",
            "P Value",
            "T Value",
            "ENR",
            "Correlation",
            "Coef. sign",
            "R-squared",
        ]
        the_cols = [col for col in the_cols if col in dfr]

        dfr = dfr[the_cols]
        dfr.set_index("Rank", inplace=True)
        for col in dfr.columns:
            if col == "R-squared":
                dfr[col] = dfr[col].apply(lambda x: "✅" if x > adj_r2_thresh else "❌")
            elif col == "Coef. sign":
                dfr[col] = dfr[col].apply(lambda x: "✅" if x == 1 else "❌")
            elif col not in ["Rank", "Weighted Score", "Variable"]:
                dfr[col] = dfr[col].apply(lambda x: "✅" if not pd.isna(x) else "❌")
        report.set_var("pre_model_table", dfr.to_markdown())

    return df


def _perform_ensemble(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    outpath: str,
    dep_var: str,
    dep_var_test: str,
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
    t: TimingData = None
):
    mv = "vacant" if vacant_only else "main"
    ensemble_inst = get_ensemble_instructions(settings, mv)
    ensemble_type = ensemble_inst["type"]
    if ensemble_type in ("median", "mean"):
        ensemble_models = ensemble_inst.get("models", [])
        optimize = ensemble_inst.get("optimize", len(ensemble_models) == 0)
        # The ensemble type names the aggregation method directly ("median" or
        # "mean"); "default" was already normalized to "median" upstream.
        agg = ensemble_type
        return _perform_default_ensemble(
            df_sales=df_sales,
            df_universe=df_universe,
            model_group=model_group,
            vacant_only=vacant_only,
            outpath=outpath,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
            ensemble_list=ensemble_models,
            optimize=optimize,
            t=t,
            agg=agg,
        )
    elif ensemble_type == "local":
        ensemble_locations = ensemble_inst.get("locations", [])
        return _perform_local_ensemble(
            df_sales=df_sales,
            df_universe=df_universe,
            model_group=model_group,
            vacant_only=vacant_only,
            outpath=outpath,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
            locations=ensemble_locations,
            t=t
        )
    else:
        raise ValueError(f"Unrecognized ensemble type \"{ensemble_type}\"!")


def _perform_local_ensemble(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    outpath: str,
    dep_var: str,
    dep_var_test: str,
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
    locations: list[str] = None,
    t: TimingData = None
):
    if t is None:
        t = TimingData()
    t.start("run_ensemble")
    if verbose:
        print("Optimizing & running ensemble...")
    ensemble_results = _run_local_ensemble(
        df_sales=df_sales,
        df_universe=df_universe,
        model_group=model_group,
        vacant_only=vacant_only,
        dep_var=dep_var,
        dep_var_test=dep_var_test,
        all_results=all_results,
        settings=settings,
        outpath=outpath,
        locations=locations,
        verbose=verbose,
    )
    t.stop("run_ensemble")
    return ensemble_results


def _validate_ensemble_models(
    ensemble_list: list[str],
    all_results: MultiModelResults,
    verbose: bool = False,
) -> list[str]:
    """Filter a user-supplied ensemble model list down to models that actually ran.

    Any listed model with no results for this model group (a typo, or a model
    that was skipped here) is dropped with a warning, so a mistake visibly
    shrinks the ensemble rather than silently producing a KeyError downstream.

    Parameters
    ----------
    ensemble_list : list[str]
        Model keys requested for the ensemble (may be empty).
    all_results : MultiModelResults
        Results for every model that ran in this model group.
    verbose : bool
        Whether to print diagnostic output.

    Returns
    -------
    list[str]
        The subset of ``ensemble_list`` that has corresponding results, in the
        original order. An empty input list is returned unchanged.
    """
    if ensemble_list is None or len(ensemble_list) == 0:
        return []

    available = all_results.model_results
    kept = [m for m in ensemble_list if m in available]
    missing = [m for m in ensemble_list if m not in available]
    if missing:
        warnings.warn(
            f"Ensemble requested model(s) {missing} that produced no results for "
            f"this model group; ignoring them. Available models: "
            f"{list(available.keys())}"
        )
    if verbose and kept:
        print(f"Validated ensemble models: {kept}")
    return kept


def _perform_default_ensemble(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    outpath: str,
    dep_var: str,
    dep_var_test: str,
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
    ensemble_list: list[str] = None,
    optimize: bool = None,
    t: TimingData = None,
    agg: str = "median",
):
    if t is None:
        t = TimingData()

    if ensemble_list is None:
        ensemble_list = []

    # Drop any user-listed models that did not produce results for this model
    # group (e.g. a typo, or a model that was skipped here), warning loudly so a
    # mistake shrinks the ensemble visibly rather than silently.
    ensemble_list = _validate_ensemble_models(
        ensemble_list, all_results, verbose=verbose
    )

    # Default mirrors get_ensemble_instructions: optimize unless the caller gave
    # an explicit whitelist.
    if optimize is None:
        optimize = len(ensemble_list) == 0

    t.start("optimize_ensemble")
    if optimize:
        if verbose:
            print("Optimizing ensemble...")
        best_ensemble = _optimize_ensemble(
            df_sales=df_sales,
            df_universe=df_universe,
            model_group=model_group,
            vacant_only=vacant_only,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
            ensemble_list=ensemble_list if len(ensemble_list) > 0 else None,
            agg=agg,
        )
    else:
        # Manual selection: use exactly the models the user listed, no pruning.
        best_ensemble = list(ensemble_list)
        if verbose:
            print(f"Using manually-selected ensemble: {best_ensemble}")
    t.stop("optimize_ensemble")
    # Run the ensemble model
    t.start("run_ensemble")
    if verbose:
        print("Running ensemble...")
    ensemble_results = _run_ensemble(
        df_sales=df_sales,
        df_universe=df_universe,
        model_group=model_group,
        vacant_only=vacant_only,
        dep_var=dep_var,
        dep_var_test=dep_var_test,
        outpath=outpath,
        ensemble_list=best_ensemble,
        all_results=all_results,
        settings=settings,
        verbose=verbose,
        agg=agg,
    )
    t.stop("run_ensemble")
    return ensemble_results


def _fix_earliest_latest_dates(df: pd.DataFrame):
    sale_date = df["sale_date"]
    if sale_date is None:
        print("WARNING: sale_date is None, using index instead")
        earliest_date_test = "???"
        latest_date_test = "???"
    elif sale_date.dtype == "datetime64[ns]":
        earliest_date = sale_date.min()
        latest_date = sale_date.max()
        if not pd.isna(earliest_date):
            earliest_date = earliest_date.strftime("%Y-%m-%d")
        else:
            earliest_date = "???"

        if not pd.isna(latest_date):
            latest_date = latest_date.strftime("%Y-%m-%d")
        else:
            latest_date = "???"
    else:
        # Convert to datetime if not already
        df["sale_date"] = pd.to_datetime(
            df["sale_date"], errors="coerce"
        )
        if df["sale_date"].isna().any():
            print("WARNING: sale_date has NaN values after conversion")
        # Get min and max dates
        # using the converted column
        earliest_date = df["sale_date"].min()
        latest_date = df["sale_date"].max()
    return earliest_date, latest_date


def _model_performance_plots(
    model_group: str, all_results: MultiModelResults, title: str
):
    # Get first model_results from all_results:
    first_results: SingleModelResults = list(all_results.model_results.values())[0]
    test_count = len(first_results.df_test)
    sales_count = len(first_results.df_sales_lookback)
    
    earliest_date_test, latest_date_test = _fix_earliest_latest_dates(first_results.df_test)
    earliest_date_study, latest_date_study = _fix_earliest_latest_dates(first_results.df_sales_lookback)
    
    for model_name, model_result in all_results.model_results.items():

        dfs = {
            "test": model_result.df_test.copy(),
            "sales": model_result.df_sales_lookback.copy(),
        }

        for key in dfs:
            df = dfs[key]
            the_count = len(df)
            sales_count = len(model_result.pred_sales_lookback.y)

            label = key.upper()
            
            if key == "test":
                df["y_pred"] = model_result.pred_test.y_pred
                df["y_true"] = model_result.pred_test.y
                earliest_date = earliest_date_test
                latest_date = latest_date_test
            else:
                df["y_pred"] = model_result.pred_sales_lookback.y_pred
                df["y_true"] = model_result.pred_sales_lookback.y
                earliest_date = earliest_date_study
                latest_date = latest_date_study
                
            # Note any NA predictions:
            for field in ["y_pred", "y_true"]:
                if df[field].isna().any():
                    mask_na = df[field].isna()
                    count_na = mask_na.count()
                    print(f"WARNING: {field} has {count_na} NaN values!")
                    df = df[~mask_na]
            
            plot_title = f"{label}/{title}/{model_group}/{model_name}\n{the_count}/{sales_count} sales from {earliest_date} to {latest_date}"

            plot_scatterplot(
                df,
                "y_true",
                "y_pred",
                "Sale price",
                "Prediction",
                title=plot_title,
                best_fit_line=True,
                perfect_fit_line=True
                #metadata_field="metadata",
            )


def _model_shaps(
    model_group: str,
    all_results: MultiModelResults,
    title: str,
    outpath: str,
):

    for key in all_results.model_results:
        smr: SingleModelResults = all_results.model_results[key]
        _title = f"{title}/{model_group}/{key}"
        if smr.model_engine == "ensemble":
            # The ensemble has no explainable estimator; its SHAPs live on disk as
            # derived contributions. Rebuild a beeswarm from those instead.
            _ensemble_beeswarm(smr, outpath, _title)
        else:
            _quick_shap(smr, True, _title)


def _ensemble_beeswarm(
    smr: SingleModelResults,
    outpath: str,
    title: str,
    verbose: bool = False,
):
    """Draw an inline beeswarm for the ensemble from its derived contributions.

    The ensemble isn't fitted, so it carries no train contributions of its own.
    Other models plot their train subset, and train == sales rows whose
    ``key_sale`` is not in the test set (see ``DataSplit.split``). So we read the
    ensemble's ``contributions_sales.csv``, keep the train rows, and rebuild a
    ``shap.Explanation`` colored by the raw train feature values. Output mirrors
    ``_quick_shap`` exactly: an inline ``plt.show()`` with no file written.
    """
    path = f"{outpath}/{smr.model_name}"
    cfile = f"{path}/contributions_sales.csv"
    if not os.path.exists(cfile):
        if verbose:
            print(f"No ensemble contributions at {cfile}; skipping beeswarm.")
        return

    df_contrib = pd.read_csv(cfile)
    df_train = smr.ds.df_train
    if df_train is None or "key_sale" not in df_train.columns:
        return
    if "key_sale" not in df_contrib.columns:
        return

    train_keys = set(df_train["key_sale"].astype(str))
    df_contrib = df_contrib[df_contrib["key_sale"].astype(str).isin(train_keys)]
    if len(df_contrib) == 0:
        return

    expl = explanation_from_contributions(df_contrib, df_train, key_col="key_sale")
    plot_full_beeswarm(expl, title=title)


def _get_earliest_and_latest_date(df: pd.DataFrame):
    
    sale_date = df["sale_date"]
    
    if sale_date is None:
        print("WARNING: sale_date is None, using index instead")
        earliest_date = "???"
        latest_date = "???"
    elif sale_date.dtype == "datetime64[ns]":
        earliest_date = sale_date.min()
        latest_date = sale_date.max()
    else:
        # Convert to datetime if not already
        df["sale_date"] = pd.to_datetime(
            df["sale_date"], errors="coerce"
        )
        if df["sale_date"].isna().any():
            print("WARNING: sale_date has NaN values after conversion")
        # Get min and max dates
        # using the converted column
        earliest_date = df["sale_date"].min()
        latest_date = df["sale_date"].max()

        if not pd.isna(earliest_date):
            earliest_date = earliest_date.strftime("%Y-%m-%d")
        else:
            earliest_date = "N/A"

        if not pd.isna(latest_date):
            latest_date = latest_date.strftime("%Y-%m-%d")
        else:
            latest_date = "N/A"
    return earliest_date, latest_date


def _model_performance_metrics(
    model_group: str, 
    all_results: MultiModelResults, 
    title: str,
    max_trim: float
):
    # Get first model_results from all_results:
    first_results: SingleModelResults = list(all_results.model_results.values())[0]
    test_count = len(first_results.df_test)
    sales_count = len(first_results.df_sales)
    
    earliest_date, latest_date = _get_earliest_and_latest_date(first_results.df_test)    

    # Add performance metrics table
    text = f"\n************************************************************\n"
    text += f"{title} Benchmark ({model_group}) -- Academic Metrics\n"
    text += f"************************************************************\n"
    text += f"Testing {test_count}/{sales_count} sales from ({earliest_date} to {latest_date})\n"
    text += ("=" * 80) + "\n"
    metrics_data = {
        "Model": [],
        "count": [],
        "RMSE": [],
        "MSE": [],
        "MAPE": [],
        "m.ratio": [],
        "avg.ratio": [],
        "VEI": [],
        "VEI_sig": [],
        "Slope": []
    }
    trimmed_data = {
        "Model": [],
        "count": [],
        "RMSE": [],
        "MSE": [],
        "MAPE": [],
        "Slope": [],
        "m.ratio": [],
        "avg.ratio": [],
        "VEI": [],
        "VEI_sig": [],
    }

    for model_name, model_result in all_results.model_results.items():

        df_test = model_result.df_test.copy()

        df_test["y_pred"] = model_result.pred_test.y_pred
        df_test["y_true"] = model_result.pred_test.y

        # Note any NA predictions:
        if df_test["y_pred"].isna().any():
            mask_na = df_test["y_pred"].isna()
            count_na = mask_na.count()
            print(f"WARNING: y_pred has {count_na} NaN values!")
            df_test = df_test[~mask_na]

        # Get test set predictions and actual values
        y_pred = df_test["y_pred"].to_numpy()
        y_true = df_test["y_true"].to_numpy()

        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)

        y_ratio = y_pred / y_true
        mask = trim_outliers_mask(y_ratio, max_trim)
        
        if len(mask) == 0:
            y_true_trim = y_true
            y_pred_trim = y_pred
        else:
            y_true_trim = y_true[mask]
            y_pred_trim = y_pred[mask]

        if len(y_true) > 1 and len(y_pred) > 1:
            # MAPE calculation
            mape = mean_absolute_percentage_error(y_true, y_pred)

            # OLS R² calculation
            reg = _simple_ols(df_test, "y_true", "y_pred", intercept=False)
            slope, r2_0 = reg["slope"], reg["r2"]

            # MSE 
            mse = calc_mse(y_pred, y_true)
            rmse = np.sqrt(mse)
        else:
            slope = np.nan
            mse = np.nan
            rmse = np.nan
            mape = np.nan

        if len(y_true_trim) > 1 and len(y_pred_trim) > 1:
            # MAPE calculation
            mape_trim = mean_absolute_percentage_error(y_true_trim, y_pred_trim)
            
            # OLS R² calculation
            df_trim = pd.DataFrame(data={"y_true":y_true_trim,"y_pred":y_pred_trim})
            reg = _simple_ols(df_trim, "y_true", "y_pred", intercept=False)
            slope_trim, r2_trim = reg["slope"], reg["r2"]

            mse_trim = calc_mse(y_pred_trim, y_true_trim)
            rmse = np.sqrt(mse_trim)
        else:
            slope_trim = np.nan
            mape_trim = np.nan
            mse_trim = np.nan
            rmse_trim = np.nan
        
        count = len(y_true)
        count_trim = len(y_true_trim)
        
        metrics_data["Model"].append(model_name)
        metrics_data["count"].append(count)
        metrics_data["MAPE"].append(mape)
        metrics_data["MSE"].append(mse)
        metrics_data["RMSE"].append(rmse)
        metrics_data["m.ratio"].append(model_result.pred_test.ratio_study.median_ratio)
        metrics_data["avg.ratio"].append(model_result.pred_test.ratio_study.mean_ratio)
        metrics_data["VEI"].append(model_result.ve_test["vei"])
        metrics_data["VEI_sig"].append(model_result.ve_test["vei_significance"])
        metrics_data["Slope"].append(slope)

        trimmed_data["Model"].append(model_name)
        trimmed_data["count"].append(count_trim)
        trimmed_data["MAPE"].append(mape_trim)
        trimmed_data["MSE"].append(mse)
        trimmed_data["RMSE"].append(rmse)
        trimmed_data["m.ratio"].append(model_result.pred_test.ratio_study.median_ratio_trim)
        trimmed_data["avg.ratio"].append(model_result.pred_test.ratio_study.mean_ratio_trim)
        
        # Calculate VEI for trimmed data
        trimmed_data["VEI"].append(model_result.ve_test["vei"])
        trimmed_data["VEI_sig"].append(model_result.ve_test["vei_significance"])
        trimmed_data["Slope"].append(slope_trim)

    # Create and display metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Model", inplace=True)
    metrics_df["count"] = metrics_df["count"].apply(lambda x: f"{x:,}").astype(str)
    metrics_df["MSE"] = metrics_df["MSE"].apply(lambda x: fancy_format(x)).astype(str)
    metrics_df["RMSE"] = metrics_df["RMSE"].apply(lambda x: f"{x:,.0f}").astype(str)
    metrics_df["MAPE"] = metrics_df["MAPE"].apply(lambda x: f"{x:.2f}").astype(str)
    metrics_df["Slope"] = metrics_df["Slope"].apply(lambda x: f"{x:.2f}").astype(str)
    metrics_df["m.ratio"] = metrics_df["m.ratio"].apply(lambda x: f"{x:.2f}").astype(str)
    metrics_df["avg.ratio"] = metrics_df["avg.ratio"].apply(lambda x: f"{x:.2f}").astype(str)
    metrics_df["VEI"] = metrics_df["VEI"].apply(lambda x: f"{x:.2f}").astype(str)
    metrics_df["VEI_sig"] = metrics_df["VEI_sig"].apply(lambda x: f"{x:.2f}").astype(str)

    trimmed_df = pd.DataFrame(trimmed_data)
    trimmed_df.set_index("Model", inplace=True)
    trimmed_df["count"] = trimmed_df["count"].apply(lambda x: f"{x:,}").astype(str)
    trimmed_df["MSE"] = trimmed_df["MSE"].apply(lambda x: fancy_format(x)).astype(str)
    trimmed_df["RMSE"] = trimmed_df["RMSE"].apply(lambda x: f"{x:,.0f}").astype(str)
    trimmed_df["MAPE"] = trimmed_df["MAPE"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["Slope"] = trimmed_df["Slope"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["m.ratio"] = trimmed_df["m.ratio"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["avg.ratio"] = trimmed_df["avg.ratio"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["VEI"] = trimmed_df["VEI"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["VEI_sig"] = trimmed_df["VEI_sig"].apply(lambda x: f"{x:.2f}").astype(str)

    metrics_df = metrics_df[["count","MAPE","MSE","RMSE","m.ratio","avg.ratio","VEI","Slope"]]
    trimmed_df = trimmed_df[["count","MAPE","MSE","RMSE","m.ratio","avg.ratio","VEI","Slope"]]

    float_cols = metrics_df.select_dtypes(include=['float']).columns
    metrics_df[float_cols] = metrics_df[float_cols].map(lambda x: f"{x:.2f}")
    
    float_cols = trimmed_df.select_dtypes(include=['float']).columns
    trimmed_df[float_cols] = trimmed_df[float_cols].map(lambda x: f"{x:.2f}")
    
    text += "\nUNTRIMMED\n"
    text += metrics_df.to_markdown() + "\n"
    text += f"\nTRIMMED\n"
    text += trimmed_df.to_markdown() + "\n"
    text += ("=" * 80) + "\n"
    return text


_DETERMINISM_ANNOUNCED = False


def _announce_determinism(seed: int) -> None:
    """Loudly state the reproducibility contract once per process.

    Modeling is always deterministic. XGBoost/LightGBM stay parallel *and* reproducible
    via batched tuning; CatBoost/NGBoost tune serially. The seed is the one knob.
    """
    global _DETERMINISM_ANNOUNCED
    if _DETERMINISM_ANNOUNCED:
        return
    _DETERMINISM_ANNOUNCED = True
    print(
        "\n"
        "============================================================\n"
        f"  DETERMINISTIC MODELING  (seed = {seed})\n"
        "  Tree-model tuning is reproducible: same inputs -> same model.\n"
        "    - XGBoost / LightGBM : parallel batched search (fast + deterministic)\n"
        "    - CatBoost / NGBoost : serial search (deterministic)\n"
        "  Change the seed via  modeling.metadata.seed  in settings.json.\n"
        "============================================================\n"
    )


def _run_models(
    sup: SalesUniversePair,
    model_group: str,
    settings: dict,
    main_vacant: str = "main",
    save_params: bool = True,
    use_saved_params: bool = True,
    save_results: bool = False,
    verbose: bool = False,
    run_ensemble: bool = True,
    do_shaps: bool = False,
    do_plots: bool = False
):
    """
    Run models for a given model group and process ensemble results.
    """
    
    outdir = ""
    if main_vacant == "main":
        vacant_only = False
        outdir = "main"
        titleword = "MAIN"
    elif main_vacant == "vacant":
        vacant_only = True
        outdir = "vacant"
        titleword = "VACANT"
    else:
        raise ValueError(f"The only supported values are 'main' and 'vacant', got '{main_vacant}' instead!")
    
    t = TimingData()
    t.start("total")

    t.start("setup")
    df_univ = sup["universe"]
    df_sales = get_hydrated_sales_from_sup(sup)

    df_sales = df_sales[df_sales["model_group"].eq(model_group)].copy()
    df_univ = df_univ[df_univ["model_group"].eq(model_group)].copy()

    settings_model = settings.get("modeling", {})
    settings_model_instructions = settings_model.get("instructions", {})
    settings_mv = settings_model_instructions.get(main_vacant, {})

    default_value = get_sale_field(settings, df_sales)
    dep_var = settings_model_instructions.get("dep_var", default_value)
    dep_var_test = settings_model_instructions.get("dep_var_test", default_value)
    fields_cat = get_fields_categorical(settings, df_univ, include_boolean=True)
    models_to_run = settings_model_instructions.get(main_vacant, {}).get("run", None)
    models_to_skip = settings_model_instructions.get(main_vacant,{}).get("skip",{}).get(model_group,[])

    model_entries = settings_model.get("models").get(main_vacant, {})
    model_entries = model_entries.get(model_group, model_entries)

    if models_to_run is None:
        models_to_run = list(model_entries.keys())

    # Enforce that horizontal equity cluster ID's have already been calculated
    if "he_id" not in df_univ:
        warnings.warn("Could not find equity cluster ID's in the dataframe (he_id) -- no horizontal equity test will be performed!")

    model_results = {}
    outpath = f"out/models/{model_group}/{outdir}"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    df_sales_count = _get_sales(df_sales, settings, vacant_only, df_univ)

    if len(df_sales_count) == 0:
        print(
            f"No sales records found for model_group: {model_group}, vacant_only: {vacant_only}. Skipping..."
        )
        return None

    if len(df_sales_count) < 15:
        warnings.warn(
            f"For model_group: {model_group}, vacant_only: {vacant_only}, there are fewer than 15 sales records. Model might not be any good!"
        )
    t.stop("setup")
    
    # Check if we need to auto reduce variables globally
    auto_reduce_vars = False
    for model_name in models_to_run:
        if model_name in models_to_skip:
            print(f"Skipping model {model_name}.")
            continue
        model_entry = model_entries.get(model_name, model_entries.get("default", {}))
        model_engine = model_entry.get("engine", model_name)
        # For tree-based models, and multi-mra, we don't perform variable reduction
        if model_engine not in ["pass_through", "ground_truth", "xgboost", "lightgbm", "catboost", "multi_mra"]:
            auto_reduce_vars = True
            break
        
    if auto_reduce_vars:
        if verbose:
            print(f"Auto-reducing variables for model type \"{model_engine}\"")
        t.start("var_recs")
        # We do a "quick" variable optimization step here. It drops some of the more expensive tests for the sake of speed
        # If you want to do those more expensive tests, you should run them in try_variables instead
        var_recs = get_variable_recommendations(
            df_sales,
            df_univ,
            vacant_only,
            settings,
            model_group,
            tests_to_run = ["corr", "r2", "p_value", "t_value", "vif"], # Exclude ENR for speed
            do_report=False,
            do_cross=False, # Exclude cross-validation for speed
            verbose=True,
            t=t
        )
        
        t.stop("var_recs")
        
        best_variables = var_recs["variables"]
        del var_recs # Delete var_recs to drop the results dataframe it holds since we don't need it
    else:
        best_variables = None

    any_results = False

    # Announce the determinism contract once if any tunable tree model will run.
    _tunable = {"xgboost", "lightgbm", "catboost", "ngboost", "lcomp"}
    if any(
        model_entries.get(m, model_entries.get("default", {})).get("engine", m) in _tunable
        or m in _tunable
        for m in models_to_run
        if m not in models_to_skip
    ):
        _announce_determinism(get_model_seed(settings))

    # Run the models one by one and stash the results
    t.start("run_models")
    for model_name in models_to_run:
        if model_name in models_to_skip:
            print(f"Skipping model {model_name}.")
            continue
        model_entry = model_entries.get(model_name, model_entries.get("default", {}))
        model_engine = model_entry.get("engine", model_name)
        
        # Tree-based models don't auto-reduce variables ever
        if model_engine not in ["xgboost", "catboost", "lightgbm"]:
            model_variables = best_variables
        else:
            model_variables = None
        
        results = run_one_model(
            df_sales=df_sales,
            df_universe=df_univ,
            vacant_only=vacant_only,
            model_group=model_group,
            model_name=model_name,
            model_entries=model_entries,
            settings=settings,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            best_variables=model_variables,
            fields_cat=fields_cat,
            outpath=outpath,
            save_params=save_params,
            use_saved_params=use_saved_params,
            save_results=save_results,
            verbose=verbose,
        )
        if results is not None:
            model_results[model_name] = results
            any_results = True
        else:
            print(f"Could not generate results for model: {model_name}")

    if not any_results:
        print(
            f"No results generated for model_group: {model_group}, vacant_only: {vacant_only}. Skipping..."
        )
        return

    t.stop("run_models")

    t.start("calc benchmarks")
    # Calculate initial results (ensemble will use them). By default the assessor is left off
    # the pre-valuation random holdout (we can't know its holdout status); declaring
    # analysis.ratio_study.assessor_holdout: "shared" keeps it in.
    drop_assessor_from_test = get_assessor_holdout_mode(settings) != "shared"
    all_results = MultiModelResults(
        model_results=model_results,
        benchmark=_calc_benchmark(model_results, drop_assessor_from_test=drop_assessor_from_test),
        df_univ=df_univ,
        df_sales=df_sales,
        drop_assessor_from_test=drop_assessor_from_test,
    )
    t.stop("calc benchmarks")

    if run_ensemble:
        ensemble_results = _perform_ensemble(
            df_sales=df_sales,
            df_universe=df_univ,
            model_group=model_group,
            vacant_only=vacant_only,
            outpath=outpath,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
            t=t
        )
        
        if verbose:
            print(f"Writing ensemble pickle...")
        out_pickle = f"{outpath}/model_ensemble.pickle"
        with open(out_pickle, "wb") as file:
            pickle.dump(ensemble_results, file)

        if verbose:
            print(f"Adding ensemble to results...")
        # Calculate final results, including ensemble
        t.start("calc final results")
        all_results.add_model("ensemble", ensemble_results)
        t.stop("calc final results")

    
    if verbose:
        print("Generating results...")
    first_results: SingleModelResults = list(all_results.model_results.values())[0]
    test_count = len(first_results.df_test)
    study_count = len(first_results.df_sales_lookback)
    sales_count = len(first_results.df_sales)
    
    earliest_date, latest_date = _get_earliest_and_latest_date(first_results.df_test)
    earliest_date_study, latest_date_study = _get_earliest_and_latest_date(first_results.df_sales_lookback)
    earliest_date_full, latest_date_full = _get_earliest_and_latest_date(first_results.df_sales)
    
    print(f"\n************************************************************")
    print(f"{titleword} Benchmark ({model_group}) -- Assessor Metrics")
    print(f"************************************************************")
    print(f"Holdout set : {test_count}/{sales_count} sales from ({earliest_date} to {latest_date})")
    print(f"  Study set : {study_count}/{sales_count} sales from ({earliest_date_study} to {latest_date_study}")
    print(f"   Full set : {sales_count}/{sales_count} sales from ({earliest_date_full} to {latest_date_full})")
    print("=" * 80)
    print("\n")
    print(all_results.benchmark.print())

    title = titleword

    max_trim = _get_max_ratio_study_trim(settings, model_group)
    
    # Add performance metrics table
    perf_metrics = _model_performance_metrics(model_group, all_results, title, max_trim)
    print(perf_metrics)
    print("")

    if do_shaps:
        _model_shaps(model_group, all_results, title, outpath)

    if do_plots:
        _model_performance_plots(model_group, all_results, title)
    print("")

    # Post-valuation metrics
    if not all_results.benchmark.test_post_val_empty:
        post_val_results = _get_post_valuation_mmr(all_results)
        title = f"{title} (Post-valuation date)"
        perf_metrics = _model_performance_metrics(model_group, post_val_results, title, max_trim)
        if perf_metrics is not None:
            print(perf_metrics)
            print("")

            print("")

    t.stop("total")

    print("")
    print("****** TIMING FOR _RUN_MODELS ******")
    print(t.print())
    print("************************************")
    print("")

    return all_results


def _get_post_valuation_mmr(m: MultiModelResults):
    new_results = {}

    for model_name, smr in m.model_results.items():
        smr = _get_post_valuation_smr(smr)
        new_results[model_name] = smr

    benchmark = _calc_benchmark(new_results)

    return MultiModelResults(model_results=new_results, benchmark=benchmark, df_sales=m.df_sales_orig, df_univ=m.df_univ_orig)


def _get_post_valuation_smr(smr: SingleModelResults, verbose: bool = False):
    y_pred_test = smr.df_test[smr.field_prediction].copy()
    y_pred_sales = smr.df_sales[smr.field_prediction].copy()
    y_pred_univ = smr.df_universe[smr.field_prediction].copy()
    new_smr = SingleModelResults(
        smr.ds.copy(),
        smr.field_prediction,
        smr.field_horizontal_equity_id,
        smr.model_name,
        smr.model_engine,
        smr.model,
        y_pred_test,
        y_pred_sales,
        y_pred_univ,
        smr.timing,
        verbose,
        [
            "<",
            "sale_age_days",
            0,
        ],  # sale age days becomes negative PAST the valuation date
    )
    return new_smr


def _prepare_stacked_features(
    base_predictions: dict[str, np.ndarray],
    contextual_data: pd.DataFrame | None,
    models_to_use: list[str],
    feature_columns: list[str] | None,
    data_indices: np.ndarray | None = None,
    feature_set: str = "",
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Prepare features for stacked ensemble, using only interactions from training set
    contextual fields."""
    # Prepare base features
    base_features = []
    feature_names = []
    for model in models_to_use:
        if model in base_predictions:
            preds = base_predictions[model]
            if data_indices is not None:
                valid_indices = data_indices[data_indices < len(preds)]
                if len(valid_indices) < len(data_indices):
                    if verbose:
                        print(
                            f"Warning: Some indices were out of bounds for {feature_set} predictions"
                        )
                preds = preds[valid_indices]
            base_features.append(preds)
            feature_names.append(model)

    base_features = np.column_stack(base_features)

    if verbose:
        print(f"{feature_set} base features shape: {base_features.shape}")

    if contextual_data is None or feature_columns is None:
        return base_features, feature_names

    # Create interactions only for contextual fields from training
    interacted_features = []
    interaction_names = []

    for col in feature_columns:
        if col in contextual_data.columns:
            indicator = contextual_data[col].values.reshape(-1, 1)
            if data_indices is not None:
                valid_indices = data_indices[data_indices < len(indicator)]
                indicator = indicator[valid_indices]

            # Create interactions with each model's predictions
            for i, model in enumerate(models_to_use):
                if model in base_predictions:
                    model_preds = base_features[:, i].reshape(-1, 1)
                    # Ensure shapes match before multiplication
                    min_len = min(indicator.shape[0], model_preds.shape[0])
                    interaction = indicator[:min_len] * model_preds[:min_len]
                    interacted_features.append(interaction)
                    interaction_names.append(f"{col}_{model}")

    if interacted_features:
        interaction_matrix = np.hstack(interacted_features)
        if verbose:
            print(f"{feature_set} interaction terms shape: {interaction_matrix.shape}")
        # Use the base features up to the length of interaction matrix
        final_features = np.hstack(
            [base_features[: interaction_matrix.shape[0]], interaction_matrix]
        )
        if verbose:
            print(f"Final {feature_set} features shape: {final_features.shape}")
        return final_features, feature_names + interaction_names

    return base_features, feature_names


def _prepare_contextual_features(
    ds: DataSplit,
    contextual_feature_names: list[str],
    categorical_contextual_features: list[str],
    neighborhood_encoded_cols: list[str] | None,
    is_test: bool,
    settings: dict,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """Prepare contextual features for either training or test data.

    Args:
        ds: DataSplit object containing the data
        contextual_feature_names: List of feature names to include
        categorical_contextual_features: List of categorical features
        neighborhood_encoded_cols: List of encoded neighborhood columns (for test data)
        is_test: Whether preparing test or training data
        settings: Settings dictionary
        verbose: Whether to print verbose output
    """
    # Use appropriate DataFrame based on context
    if is_test:
        df = ds.df_test
    else:
        # For universe predictions, use df_universe
        df = ds.df_universe if hasattr(ds, "df_universe") else ds.df_sales

    if df is None or df.empty:
        if verbose:
            print(
                f"\n{'Test' if is_test else 'Universe/Training'} DataFrame is None or empty"
            )
        return None

    # Handle training data or test data without encoded columns
    available_context_cols = []
    for feature in contextual_feature_names:
        if feature in categorical_contextual_features:
            encoded_cols = [col for col in df.columns if col.startswith(f"{feature}_")]
            if encoded_cols:
                available_context_cols.extend(encoded_cols)
            else:
                field_name = get_important_field(settings, f"loc_{feature}", df)
                if field_name and field_name in df.columns:
                    if verbose:
                        print(f"Using raw field {field_name} for {feature}")
                    available_context_cols.append(field_name)
                elif verbose:
                    print(
                        f"Warning: No columns found for categorical feature {feature}"
                    )
        else:
            if feature in df.columns:
                available_context_cols.append(feature)
            elif verbose:
                print(f"Warning: Feature {feature} not found in data")

    if not available_context_cols:
        if verbose:
            print("No contextual features available")
        return None

    # Create contextual features DataFrame
    contextual_df = df[available_context_cols].copy()

    # For test/universe data, ensure all training columns exist (with zeros if needed)
    if is_test and neighborhood_encoded_cols:
        for col in neighborhood_encoded_cols:
            if col not in contextual_df.columns:
                contextual_df[col] = 0

    return contextual_df


def _collect_base_model_predictions(
    models_for_stacking: list[str],
    all_results: MultiModelResults,
    prediction_type: str,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, DataSplit | None]:
    """Collect predictions from base models.

    Args:
        models_for_stacking: List of models to include in stacking
        all_results: MultiModelResults containing model results
        prediction_type: Type of predictions to collect ('oof', 'test', or 'universe')
        verbose: Whether to print verbose output
    """
    predictions = {}
    true_values = None
    template_ds = None

    for model_name in models_for_stacking:
        if model_name not in all_results.model_results:
            if verbose:
                print(f"Model '{model_name}' not found in results")
            continue

        smr = all_results.model_results[model_name]

        if prediction_type == "oof":
            if smr.pred_sales is not None and smr.pred_sales.y_pred is not None:
                predictions[model_name] = smr.pred_sales.y_pred
                if true_values is None:
                    true_values = smr.pred_sales.y
                    template_ds = smr.ds
        elif prediction_type == "test":
            if smr.pred_test is not None and smr.pred_test.y_pred is not None:
                predictions[model_name] = smr.pred_test.y_pred
                if true_values is None:
                    true_values = smr.pred_test.y
                    template_ds = smr.ds
        elif prediction_type == "universe":
            if smr.pred_univ is not None:
                predictions[model_name] = smr.pred_univ
                template_ds = smr.ds

    return predictions, true_values, template_ds


def _quick_shap(
    smr: SingleModelResults, 
    plot: bool = False, 
    title: str = ""
):
    """
    Compute SHAP values for a given model and dataset and optionally plot it.

    Parameters
    ----------
    smr : SingleModelResults
        The SingleModelResults object containing the fitted model and data splits.
    plot : bool, optional
        If True, generate and display a SHAP summary plot. Defaults to False.
    title : str, optional
        Title to use for the SHAP plot if `plot` is True. Defaults to an empty string.

    Returns
    -------
    np.ndarray
        SHAP values array for the evaluation dataset.
    """

    if smr.model_engine not in ["xgboost", "catboost", "lightgbm", "ngboost", "lcomp"]:
        # SHAP is not supported for this model type
        return

    X_train = smr.ds.X_train

    shaps = _calc_shap(smr.model, X_train, X_train)

    # _calc_shap returns None if SHAP can't be computed (e.g. unexpected NGBoost internals)
    if shaps is None:
        return

    if plot:
        plot_full_beeswarm(shaps, title=title)
