import os
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
from sklearn.linear_model import LinearRegression

from openavmkit.data import (
    get_important_field,
    get_locations,
    _read_split_keys,
    SalesUniversePair,
    get_hydrated_sales_from_sup,
    get_report_locations,
    get_sale_field,
)
from openavmkit.modeling import (
    run_mra,
    run_gwr,
    run_xgboost,
    run_lightgbm,
    run_catboost,
    run_slice,
    SingleModelResults,
    run_garbage,
    run_average,
    run_naive_sqft,
    predict_garbage,
    run_kernel,
    run_local_sqft,
    run_pass_through,
    predict_average,
    predict_naive_sqft,
    predict_local_sqft,
    predict_pass_through,
    predict_kernel,
    predict_gwr,
    predict_xgboost,
    predict_catboost,
    predict_lightgbm,
    predict_slice,
    GarbageModel,
    AverageModel,
    DataSplit,
    run_ground_truth,
    predict_ground_truth,
    run_spatial_lag,
    predict_spatial_lag
)
from openavmkit.reports import MarkdownReport, _markdown_to_pdf
from openavmkit.shap_analysis import compute_shap
from openavmkit.time_adjustment import enrich_time_adjustment
from openavmkit.utilities.data import (
    div_df_z_safe,
    df_to_markdown,
    do_per_model_group,
    load_model_results,
)
from openavmkit.utilities.format import fancy_format, dig2_fancy_format
from openavmkit.utilities.modeling import (
    NaiveSqftModel,
    LocalSqftModel,
    PassThroughModel,
    LandSLICEModel,
    GWRModel,
    MRAModel,
    GroundTruthModel,
    SpatialLagModel,
)
from openavmkit.utilities.plotting import plot_scatterplot, _simple_ols
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_variable_interactions,
    get_valuation_date,
    get_model_group,
    _apply_dd_to_df_rows,
    get_model_group_ids,
    get_fields_boolean,
    _get_sales,
    _simulate_removed_buildings,
    _get_max_ratio_study_trim,
    get_look_back_dates
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
            result += _format_benchmark_df(self.df_stats_test_post_val)
            result += "\n\n"
        result += "Holdout set:\n"
        result += _format_benchmark_df(self.df_stats_test)
        result += "\n\n"
        result += "Study set:\n"
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

    def __init__(
        self, model_results: dict[str, SingleModelResults], benchmark: BenchmarkResults
    ):
        """Initialize a MultiModelResults instance.

        Parameters
        ----------
        model_results: dict[str, SingleModelResults]
            Dictionary of individual model results.
        benchmark: BenchmarkResults
            Benchmark results.
        """
        self.model_results = model_results
        self.benchmark = benchmark

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
        # Recalculate the benchmark based on updated model results.
        self.benchmark = _calc_benchmark(self.model_results)


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
    ):
        bests = {}

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

            var_recs = get_variable_recommendations(
                df_in,
                df_univ,
                vacant_only,
                settings,
                model_group,
                variables_to_use=variables_to_use,
                tests_to_run=["corr", "r2"],
                do_report=True,
                verbose=verbose,
            )

            best_variables = var_recs["variables"]
            df_results = var_recs["df_results"]

            if vacant_only:
                bests["vacant_only"] = df_results
            else:
                bests["main"] = df_results

        results[model_group] = bests

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
        },
        key="key_sale",
        skip=model_groups_to_skip
    )

    sale_field = get_sale_field(settings)

    print("")
    print("********** BEST VARIABLES ***********")
    for model_group in all_best_variables:
        entry = all_best_variables[model_group]
        for vacant_status in entry:
            print("")
            print(f"model group: {model_group} / {vacant_status}")
            results = entry[vacant_status]
            pd.set_option("display.max_rows", None)
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

            display(styled)
            file_out = f"out/try/{model_group}/{vacant_status}.csv"
            if not os.path.exists(os.path.dirname(file_out)):
                os.makedirs(os.path.dirname(file_out))
            results.to_csv(file_out, index=False)
            pd.set_option("display.max_rows", 15)

            for var in results["variable"].unique():
                if var in df_hydrated.columns:
                    # do a correlation scatter plot of the variable vs. the dependent variable (sale_field):
                    df_sub = df_hydrated[
                        df_hydrated["model_group"].eq(model_group)
                        & df_hydrated[var].notna()
                        & df_hydrated[sale_field].notna()
                    ]

                    for status in ["vacant", "improved"]:
                        # clear any previous plots with plt:
                        plt.clf()

                        if status == "vacant":
                            df_sub2 = df_sub[df_sub["vacant_sale"].eq(True)]
                        else:
                            df_sub2 = df_sub[df_sub["vacant_sale"].eq(False)]

                        if len(df_sub2) > 0 and plot:
                            # do a scatter plot of the variable vs. the dependent variable (sale_field):
                            df_sub2.plot.scatter(x=var, y=sale_field)
                            # labels
                            plt.xlabel(var)
                            plt.ylabel(sale_field)
                            plt.title(f"'{var}' vs '{sale_field}' ({status} only)")
                            plt.show()


def get_variable_recommendations(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    vacant_only: bool,
    settings: dict,
    model_group: str,
    variables_to_use: list[str] | None = None,
    tests_to_run: list[str] | None = None,
    do_report: bool = False,
    verbose: bool = False,
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
    verbose : bool, optional
        If True, prints additional debugging information.

    Returns
    -------
    dict
        A dictionary with keys "variables" (the best variables list) and "report"
        (the generated report).
    """

    report = MarkdownReport("variables")

    if tests_to_run is None:
        tests_to_run = ["corr", "r2", "p_value", "t_value", "enr", "vif"]

    if "sale_price_time_adj" not in df_sales:
        warnings.warn("Time adjustment was not found in sales data. Calculating now...")
        df_sales = enrich_time_adjustment(df_sales, settings, verbose=verbose)

    s = settings
    s_model = s.get("modeling", {})
    vacant_status = "vacant" if vacant_only else "main"
    model_entries = s_model.get("models", {}).get(vacant_status, {})
    entry: dict | None = model_entries.get("model", model_entries.get("default", {}))
    if variables_to_use is None:
        variables_to_use: list | None = entry.get("ind_vars", None)
    
    if len(variables_to_use) == 0 or variables_to_use is None:
        raise ValueError("No independent variables provided! Please define some!")
    
    cats = get_fields_categorical(settings, df_sales, include_boolean=False)
    flagged = []
    for variable in variables_to_use:
        if variable in cats:
            uniques = df_sales[variable].unique()
            if len(uniques) > 50:
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

    ds = _prepare_ds(
        df_sales, df_universe, model_group, vacant_only, settings, variables_to_use
    )
    ds = ds.encode_categoricals_with_one_hot()

    ds.split()

    feature_selection = (
        settings.get("modeling", {})
        .get("instructions", {})
        .get("feature_selection", {})
    )
    thresh = feature_selection.get("thresholds", {})

    X_sales = ds.X_sales[ds.ind_vars]
    y_sales = ds.y_sales

    if "corr" in tests_to_run:
        # Correlation
        X_corr = ds.df_sales[[ds.dep_var] + ds.ind_vars]
        corr_results = calc_correlations(X_corr, thresh.get("correlation", 0.1))
    else:
        corr_results = None

    if "enr" in tests_to_run:
        # Elastic net regularization
        try:
            enr_coefs = calc_elastic_net_regularization(
                X_sales, y_sales, thresh.get("enr", 0.01)
            )
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
        r2_values = calc_r2(ds.df_sales, ds.ind_vars, y_sales)
    else:
        r2_values = None

    if "p_value" in tests_to_run:
        # P Values
        p_values = calc_p_values_recursive_drop(
            X_sales, y_sales, thresh.get("p_value", 0.05)
        )
    else:
        p_values = None

    if "t_value" in tests_to_run:
        # T Values
        t_values = calc_t_values_recursive_drop(
            X_sales, y_sales, thresh.get("t_value", 2)
        )
    else:
        t_values = None

    if "vif" in tests_to_run:
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
    else:
        vif = None

    # Generate final results & recommendations
    df_results = _calc_variable_recommendations(
        ds=ds,
        settings=settings,
        correlation_results=corr_results,
        enr_results=enr_coefs,
        r2_values_results=r2_values,
        p_values_results=p_values,
        t_values_results=t_values,
        vif_results=vif,
        report=report,
    )

    curr_variables = df_results["variable"].tolist()
    best_variables = curr_variables.copy()
    best_score = float("inf")

    df_cross = df_results.copy()
    y = ds.y_sales
    while len(curr_variables) > 0:
        X = ds.df_sales[curr_variables]
        cv_score = calc_cross_validation_score(X, y)
        if cv_score < best_score:
            best_score = cv_score
            best_variables = curr_variables.copy()
        worst_idx = df_cross["weighted_score"].idxmin()
        worst_variable = df_cross.loc[worst_idx, "variable"]
        curr_variables.remove(worst_variable)
        # Remove the variable from the results dataframe.
        df_cross = df_cross[df_cross["variable"].ne(worst_variable)]

    # Create a table from the list of best variables.
    df_best = pd.DataFrame(best_variables, columns=["Variable"])
    df_best["Rank"] = range(1, len(df_best) + 1)
    df_best["Description"] = df_best["Variable"]
    df_best = _apply_dd_to_df_rows(
        df_best, "Variable", settings, ds.one_hot_descendants, "name"
    )
    df_best = _apply_dd_to_df_rows(
        df_best, "Description", settings, ds.one_hot_descendants, "description"
    )
    df_best = df_best[["Rank", "Variable", "Description"]]
    df_best.loc[df_best["Variable"].eq(df_best["Description"]), "Description"] = ""
    df_best.set_index("Rank", inplace=True)

    if do_report:
        report.set_var("summary_table", df_best.to_markdown())
        report = generate_variable_report(report, settings, model_group, best_variables)
    else:
        report = None

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
    run_hedonic: bool = True,
    run_ensemble: bool = True,
    do_shaps: bool = False,
    do_plots: bool = False
):
    """
    Runs predictive models on the given SalesUniversePair.

    This function takes detailed instructions from the provided settings dictionary and handles all the internal
    details like splitting the data, training the models, and saving the results. It performs basic statistic analysis
    on each model, and optionally combines results into an ensemble model.

    If "run_main" is true, it will run normal models as well as hedonic models (if the user so specifies),
    "hedonic" in this context meaning models that attempt to generate a land value and an improvement value separately.
    If "run_vacant" is true, it will run vacant models as well -- models that only use vacant models as evidence
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
    run_hedonic : bool, optional
        Whether to run hedonic models.
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
        for main_vacant_hedonic in ["main", "vacant", "hedonic"]:
            if main_vacant_hedonic == "main" and not run_main:
                continue
            if main_vacant_hedonic == "vacant" and not run_vacant:
                continue
            if main_vacant_hedonic == "hedonic" and not run_hedonic:
                continue

            models_to_skip = s_inst.get(main_vacant_hedonic, {}).get("skip", {}).get(model_group, [])

            if "all" in models_to_skip:
                if verbose:
                    print(
                        f"Skipping all models for model_group: {model_group}/{main_vacant_hedonic}"
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
                main_vacant_hedonic,
                save_params,
                use_saved_params,
                save_results,
                verbose,
                run_ensemble,
                do_shaps=do_shaps,
                do_plots=do_plots
            )
            if mg_results is not None:
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
        if "stacked_ensemble" in mm_results.model_results:
            output_models.append("stacked_ensemble")
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
                if model_type != "ensemble"
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
    name: str,
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
    hedonic: bool,
    hedonic_test_against_vacant_sales: bool = True,
):
    """
    Prepare a DataSplit object for a given model.

    Parameters
    ----------
    name : str
        Model name.
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
    hedonic : bool
        Whether to use hedonic pricing.
    hedonic_test_against_vacant_sales : bool, optional
        Whether to test hedonic models against vacant sales. Defaults to True.

    Returns
    -------
    DataSplit
        A DataSplit object.
    """
    if name == "local_sqft":
        _ind_vars = location_fields + ["bldg_area_finished_sqft", "land_area_sqft"]
    elif name == "slice":
        _ind_vars = ["land_area_sqft", "latitude", "longitude"]
    elif name == "assessor":
        _ind_vars = ["assr_land_value"] if hedonic else ["assr_market_value"]
    elif name == "ground_truth":
        _ind_vars = ["true_land_value"] if hedonic else ["true_market_value"]
    elif name == "spatial_lag":
        sale_field = get_sale_field(settings)
        field = f"spatial_lag_{sale_field}"
        if vacant_only or hedonic:
            field = f"{field}_vacant"
        _ind_vars = [field]
    elif name == "spatial_lag_sqft":
        sale_field = get_sale_field(settings)
        _ind_vars = [
            f"spatial_lag_{sale_field}_impr_sqft",
            f"spatial_lag_{sale_field}_land_sqft",
            "bldg_area_finished_sqft",
            "land_area_sqft",
        ]
    elif name == "catboost":
        df_sales = _clean_categoricals(df_sales, fields_cat, settings)
        df_universe = _clean_categoricals(df_universe, fields_cat, settings)
        _ind_vars = ind_vars
    else:
        _ind_vars = ind_vars
        if name == "gwr" or name == "kernel":
            exclude_vars = ["latitude", "longitude", "latitude_norm", "longitude_norm"]
            _ind_vars = [var for var in _ind_vars if var not in exclude_vars]

    return DataSplit(
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
        hedonic=hedonic,
        hedonic_test_against_vacant_sales=hedonic_test_against_vacant_sales,
    )


def run_one_model(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    vacant_only: bool,
    model_group: str,
    model: str,
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
    hedonic: bool = False,
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
    model : str
        Model name.
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
    hedonic : bool, optional
        Whether to use hedonic pricing.
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
    model_name = model

    entry: dict | None = model_entries.get(model, None)
    default_entry: dict | None = model_entries.get("default", {})
    if entry is None:
        entry = default_entry
        if entry is None:
            raise ValueError(
                f"Model entry for {model} not found, and there is no default entry!"
            )

    if "*" in model:
        sales_chase = 0.01
        model_name = model.replace("*", "")
    else:
        sales_chase = False

    if verbose:
        print(f"------------------------------------------------")
        print(f"Running model {model} on {len(df_sales)} rows...")

    are_ind_vars_default = entry.get("ind_vars", None) is None
    ind_vars: list | None = entry.get("ind_vars", default_entry.get("ind_vars", None))
    
    if vacant_only or hedonic:
        default_value = True
        if model_name == "assessor":
            default_value = False
        do_clamp = entry.get("do_clamp", default_value)
    
    # no duplicates!
    ind_vars = list(set(ind_vars))
    if ind_vars is None:
        raise ValueError(f"ind_vars not found for model {model}")

    if are_ind_vars_default:
        if (best_variables is not None) and (set(ind_vars) != set(best_variables)):
            if verbose:
                print(
                    f"--> using default variables, auto-optimized variable list: {best_variables}"
                )
            ind_vars = best_variables

    interactions = get_variable_interactions(entry, settings, df_sales)
    location_fields = get_locations(settings, df_sales)

    if test_keys is None or train_keys is None:
        test_keys, train_keys = _read_split_keys(model_group)
    t.stop("setup")

    t.start("data split")
    ds = get_data_split_for(
        name=model_name,
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
        hedonic=hedonic,
        hedonic_test_against_vacant_sales=True,
    )
    t.stop("data split")

    t.start("setup")
    if len(ds.y_sales) < 15:
        if verbose:
            print(f"--> model {model} has less than 15 sales. Skipping...")
        return None

    intercept = entry.get("intercept", True)
    n_trials = entry.get("n_trials", 50)
    t.stop("setup")

    t.start("run")
    if model_name == "garbage":
        results = run_garbage(
            ds, normal=False, sales_chase=sales_chase, verbose=verbose
        )
    elif model_name == "garbage_normal":
        results = run_garbage(ds, normal=True, sales_chase=sales_chase, verbose=verbose)
    elif model_name == "mean":
        results = run_average(
            ds, average_type="mean", sales_chase=sales_chase, verbose=verbose
        )
    elif model_name == "median":
        results = run_average(
            ds, average_type="median", sales_chase=sales_chase, verbose=verbose
        )
    elif model_name == "naive_sqft":
        results = run_naive_sqft(ds, sales_chase=sales_chase, verbose=verbose)
    elif model_name == "local_sqft":
        results = run_local_sqft(
            ds,
            location_fields=location_fields,
            sales_chase=sales_chase,
            verbose=verbose,
        )
    elif model_name == "assessor":
        results = run_pass_through(ds, verbose=verbose)
    elif model_name == "ground_truth":
        results = run_ground_truth(ds, verbose=verbose)
    elif model_name == "spatial_lag":
        results = run_spatial_lag(ds, per_sqft=False, verbose=verbose)
    elif model_name == "spatial_lag_sqft":
        results = run_spatial_lag(ds, per_sqft=True, verbose=verbose)
    elif model_name == "mra":
        results = run_mra(ds, intercept=intercept, verbose=verbose)
    elif model_name == "kernel":
        results = run_kernel(
            ds, outpath, save_params, use_saved_params, verbose=verbose
        )
    elif model_name == "gwr":
        results = run_gwr(ds, outpath, save_params, use_saved_params, verbose=verbose)
    elif model_name == "xgboost":
        results = run_xgboost(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose
        )
    elif model_name == "lightgbm":
        results = run_lightgbm(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose
        )
    elif model_name == "catboost":
        results = run_catboost(
            ds, outpath, save_params, use_saved_params, n_trials=n_trials, verbose=verbose
        )
    elif model_name == "slice":
        results = run_slice(ds, verbose=verbose)
    else:
        raise ValueError(f"Model {model_name} not found!")
    t.stop("run")

    if ds.vacant_only or ds.hedonic:
        # If this is a vacant or hedonic model, we attempt to load a corresponding "full value" model
        max_trim = _get_max_ratio_study_trim(settings, results.ds.model_group)
        if do_clamp:
            results = _clamp_land_predictions(results, results.ds.model_group, model_name, outpath, max_trim)

    if save_results:
        t.start("write")
        _write_model_results(results, outpath, settings)
        t.stop("write")

    return results


def run_one_hedonic_model(
    df_sales: pd.DataFrame,
    df_univ: pd.DataFrame,
    settings: dict,
    model: str,
    smr: SingleModelResults,
    model_group: str,
    dep_var: str,
    dep_var_test: str,
    fields_cat: list[str],
    outpath: str,
    hedonic_test_against_vacant_sales: bool = True,
    save_results: bool = False,
    verbose: bool = False,
):
    """Run a single hedonic model based on provided parameters and return its results.

    This function is similar to run_one_model but specifically tailored for hedonic models.

    Parameters
    ----------
    df_sales : pandas.DataFrame
        Sales DataFrame.
    df_univ : pandas.DataFrame
        Universe DataFrame.
    settings : dict
        Settings dictionary.
    model : str
        Model name.
    smr : SingleModelResults
        SingleModelResults object containing initial model results.
    model_group : str
        Model group identifier.
    dep_var : str
        Dependent variable for training.
    dep_var_test : str
        Dependent variable for testing.
    fields_cat : list[str]
        List of categorical fields.
    outpath : str
        Output path for saving results.
    hedonic_test_against_vacant_sales : bool, optional
        Whether to test hedonic models against vacant sales. Defaults to True.
    save_results : bool, optional
        Whether to save results. Defaults to False.
    verbose : bool, optional
        If True, prints additional information. Defaults to False.

    Returns
    -------
    SingleModelResults or None
        SingleModelResults if successful, else None.
    """
    location_field_neighborhood = get_important_field(
        settings, "loc_neighborhood", df_sales
    )
    location_field_market_area = get_important_field(
        settings, "loc_market_area", df_sales
    )
    location_fields = [location_field_neighborhood, location_field_market_area]

    ds = get_data_split_for(
        name=model,
        model_group=model_group,
        location_fields=location_fields,
        ind_vars=smr.ind_vars,
        df_sales=df_sales,
        df_universe=df_univ,
        settings=settings,
        dep_var=dep_var,
        dep_var_test=dep_var_test,
        fields_cat=fields_cat,
        interactions=smr.ds.interactions.copy(),
        test_keys=smr.ds.test_keys,
        train_keys=smr.ds.train_keys,
        vacant_only=False,
        hedonic=True,
        hedonic_test_against_vacant_sales=hedonic_test_against_vacant_sales,
    )
    # We call this here because we are re-running prediction without first calling run(), which would call this
    ds.split()
    if hedonic_test_against_vacant_sales and len(ds.y_sales) < 15:
        print(f"Skipping hedonic model because there are not enough sale records...")
        return None
    smr.ds = ds
    results = _predict_one_model(
        smr=smr,
        model=model,
        outpath=outpath,
        settings=settings,
        save_results=save_results,
        verbose=verbose,
    )
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
    hedonic: bool = False,
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
    hedonic : bool, optional
        Whether to use hedonic pricing. Defaults to False.

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
        hedonic=hedonic,
        ensemble_list=None,
    )
    if verbose:
        print("Running ensemble...")
    ensemble = _run_ensemble(
        df_sales,
        df_universe,
        model_group,
        vacant_only=vacant_only,
        hedonic=hedonic,
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


def _calc_benchmark(model_results: dict[str, SingleModelResults]):
    """
    Calculate benchmark statistics from individual model results.
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

    df_test.set_index("model", inplace=True)
    df_test_post_val.set_index("model", inplace=True)
    df_full.set_index("model", inplace=True)
    df_time.set_index("model", inplace=True)

    results = BenchmarkResults(df_time, df_test, df_test_post_val, df_full)
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


def _predict_one_model(
    smr: SingleModelResults,
    model: str,
    outpath: str,
    settings: dict,
    save_results: bool = False,
    verbose: bool = False,
) -> SingleModelResults:
    """
    Predict results for one model, using saved results if available.
    """
    model_name = model
    ds = smr.ds

    timing = TimingData()
    timing.start("total")

    results: SingleModelResults | None = None

    if model_name == "garbage":
        garbage_model: GarbageModel = smr.model
        results = predict_garbage(ds, garbage_model, timing, verbose)
    elif model_name == "garbage_normal":
        garbage_model: GarbageModel = smr.model
        results = predict_garbage(ds, garbage_model, timing, verbose)
    elif model_name == "mean":
        mean_model: AverageModel = smr.model
        results = predict_average(ds, mean_model, timing, verbose)
    elif model_name == "median":
        median_model: AverageModel = smr.model
        results = predict_average(ds, median_model, timing, verbose)
    elif model_name == "naive_sqft":
        sqft_model: NaiveSqftModel = smr.model
        results = predict_naive_sqft(ds, sqft_model, timing, verbose)
    elif model_name == "local_sqft":
        sqft_model: LocalSqftModel = smr.model
        results = predict_local_sqft(ds, sqft_model, timing, verbose)
    elif model_name == "assessor":
        assr_model: PassThroughModel = smr.model
        results = predict_pass_through(ds, assr_model, timing, verbose)
    elif model_name == "ground_truth":
        ground_truth_model: GroundTruthModel = smr.model
        results = predict_ground_truth(ds, ground_truth_model, timing, verbose)
    elif model_name == "spatial_lag" or model_name == "spatial_lag_sqft":
        lag_model: SpatialLagModel = smr.model
        results = predict_spatial_lag(ds, lag_model, timing, verbose)
    elif model_name == "mra":
        # MRA is a special case where we have to call run_ instead of predict_, because there's delicate state mangling.
        # We pass the pretrained `model` object to run_mra() to get it to skip training and move straight to prediction
        model: MRAModel = smr.model
        results = run_mra(ds, model.intercept, verbose, model)
    elif model_name == "kernel":
        kernel_reg: KernelReg = smr.model
        results = predict_kernel(ds, kernel_reg, timing, verbose)
    elif model_name == "gwr":
        gwr_model: GWRModel = smr.model
        results = predict_gwr(ds, gwr_model, timing, verbose)
    elif model_name == "xgboost":
        xgb_regressor: XGBRegressor = smr.model
        results = predict_xgboost(ds, xgb_regressor, timing, verbose)
    elif model_name == "lightgbm":
        lightgbm_regressor: Booster = smr.model
        results = predict_lightgbm(ds, lightgbm_regressor, timing, verbose)
    elif model_name == "catboost":
        catboost_regressor: CatBoostRegressor = smr.model
        results = predict_catboost(ds, catboost_regressor, timing, verbose)
    elif model_name == "slice":
        slice_model: LandSLICEModel = smr.model
        results = predict_slice(ds, slice_model, timing, verbose)
    
    if ds.vacant_only or ds.hedonic:
        # If this is a vacant or hedonic model, we attempt to load a corresponding "full value" model
        max_trim = _get_max_ratio_study_trim(settings, smr.ds.model_group)
        results = _clamp_land_predictions(results, smr.ds.model_group, model_name, outpath, max_trim)

    if save_results:
        _write_model_results(results, outpath, settings)

    return results


def _clamp_land_predictions(
    results: SingleModelResults, model_group: str, model_name: str, outpath: str, max_trim: float
):
    """
    Clamp land value predictions based on the full market value predictions.
    This function ensures that land value predictions are non-negative and do not exceed the full market value predictions.
    """

    lookpath = "main"
    if "vacant" in outpath:
        lookpath = "main"
    if "hedonic" in outpath:
        lookpath = "hedonic_full"

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
    locations = get_report_locations(settings)
    fields = [
        "key",
        "geometry",
        "prediction",
        "assr_market_value",
        "assr_land_value",
        "true_market_value",
        "true_land_value",
        "bldg_area_finished_sqft",
        "land_area_sqft",
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

        if "bldg_area_finished_sqft" in df:
            df["prediction_impr_sqft"] = div_df_z_safe(
                df, "prediction", "bldg_area_finished_sqft"
            )
        if "land_area_sqft" in df:
            df["prediction_land_sqft"] = div_df_z_safe(
                df, "prediction", "land_area_sqft"
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


def _write_model_results(results: SingleModelResults, outpath: str, settings: dict):
    """
    Write model results to disk in parquet and CSV formats.
    """
    dfs = _assemble_model_results(results, settings)
    path = f"{outpath}/{results.type}"
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
    path = f"{outpath}/{results.type}"
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


def _optimize_ensemble_allocation(
    df_sales: pd.DataFrame | None,
    df_universe: pd.DataFrame | None,
    model_group: str,
    vacant_only: bool,
    dep_var: str,
    dep_var_test: str,
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
    hedonic: bool = False,
    ensemble_list: list[str] = None,
):
    """
    Select the models that produce the best land allocation results for an ensemble
    model.
    """
    timing = TimingData()
    timing.start("total")
    timing.start("setup")

    if df_sales is None:
        first_key = list(all_results.model_results.keys())[0]
        df_universe = all_results.model_results[first_key].ds.df_universe_orig
        df_sales = all_results.model_results[first_key].ds.df_sales_orig

    test_keys, train_keys = _read_split_keys(model_group)

    ds = DataSplit(
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
        hedonic=hedonic,
    )

    vacant_status = "vacant" if vacant_only else "main"
    df_test = ds.df_test
    df_univ = ds.df_universe
    instructions = settings.get("modeling", {}).get("instructions", {})

    if ensemble_list is None:
        ensemble_list = instructions.get(vacant_status, {}).get("ensemble", [])

    if len(ensemble_list) == 0:
        ensemble_list = [key for key in all_results.model_results.keys()]

    if "assessor" in ensemble_list:
        ensemble_list.remove("assessor")

    if "ground_truth" in ensemble_list:
        ensemble_list.remove("ground_truth")

    best_list = []
    best_score = float("inf")

    while len(ensemble_list) > 1:
        best_score, best_list = _optimize_ensemble_allocation_iteration(
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
        )

    if verbose:
        if not np.isinf(best_score):
            print(f"Best score = {best_score:8.0f}, ensemble = {best_list}")
    return best_list


def _optimize_ensemble_allocation_iteration(
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
):
    """
    Perform one iteration of ensemble allocation optimization.
    """
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
        df_test_ensemble[m_key] = m_results.pred_test.y_pred
        df_sales_ensemble[m_key] = m_results.pred_sales.y_pred
        df_univ_ensemble[m_key] = m_results.pred_univ
    timing.stop("train")

    timing.start("predict_test")
    y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_test")

    timing.start("predict_sales")
    y_pred_sales_ensemble = df_sales_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_univ")

    results = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        "ensemble",
        model="ensemble",
        y_pred_test=y_pred_test_ensemble.to_numpy(),
        y_pred_sales=y_pred_sales_ensemble.to_numpy(),
        y_pred_univ=y_pred_univ_ensemble.to_numpy(),
        timing=timing,
        verbose=verbose
    )
    score = results.utility_train

    timing.stop("total")

    if verbose:
        print(
            f"score = {score:5.0f}, best = {best_score:5.0f}, ensemble = {ensemble_list}..."
        )

    if score < best_score and len(ensemble_list) >= 1:
        best_score = score
        best_list = ensemble_list.copy()

    # identify the WORST individual model:
    worst_model = None
    worst_score = float("-inf")
    for key in ensemble_list:
        if key in all_results.model_results:
            model_results = all_results.model_results[key]
            model_score = model_results.utility_train

            if model_score > worst_score:
                worst_score = model_score
                worst_model = key

    if worst_model is not None and len(ensemble_list) > 1:
        ensemble_list.remove(worst_model)

    return best_score, best_list


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
    hedonic: bool = False,
    ensemble_list: list[str] = None,
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
        df_universe = all_results.model_results[first_key].ds.df_universe_orig
        df_sales = all_results.model_results[first_key].ds.df_sales_orig

    ds = DataSplit(
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
        hedonic=hedonic,
    )

    vacant_status = "vacant" if vacant_only else "main"
    df_test = ds.df_test
    df_sales = ds.df_sales
    df_univ = ds.df_universe
    instructions = settings.get("modeling", {}).get("instructions", {})

    if ensemble_list is None:
        ensemble_list = instructions.get(vacant_status, {}).get("ensemble", [])

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
    y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_test")

    timing.start("predict_sales")
    y_pred_sales_ensemble = df_sales_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_univ")

    results : SingleModelResults = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        "ensemble",
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
    hedonic: bool,
    dep_var: str,
    dep_var_test: str,
    outpath: str,
    ensemble_list: list[str],
    all_results: MultiModelResults,
    settings: dict,
    verbose: bool = False,
):
    """Run the ensemble model based on the given ensemble list and write results.

    :param df_sales: Sales DataFrame.
    :type df_sales: pandas.DataFrame
    :param df_universe: Universe DataFrame.
    :type df_universe: pandas.DataFrame
    :param model_group: Model group identifier.
    :type model_group: str
    :param vacant_only: Whether it is a vacant-only model.
    :type vacant_only: bool
    :param hedonic: Whether it is a hedonic model.
    :type hedonic: bool
    :param dep_var: Dependent variable for training.
    :type dep_var: str
    :param dep_var_test: Dependent variable for testing.
    :type dep_var_test: str
    :param outpath: Output path for results.
    :type outpath: str
    :param ensemble_list: List of models to include in the ensemble.
    :type ensemble_list: list[str]
    :param all_results: MultiModelResults containing model results.
    :type all_results: MultiModelResults
    :param settings: Settings dictionary.
    :type settings: dict
    :param verbose: If True, prints additional information.
    :type verbose: bool, optional
    :returns: SingleModelResults for the ensemble.
    :rtype: SingleModelResults
    """
    timing = TimingData()
    timing.start("total")
    timing.start("setup")

    first_key = list(all_results.model_results.keys())[0]
    test_keys = all_results.model_results[first_key].ds.test_keys
    train_keys = all_results.model_results[first_key].ds.train_keys

    ds = DataSplit(
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
        hedonic=hedonic,
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
    y_pred_test_ensemble = df_test_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_test")

    timing.start("predict_sales")
    y_pred_sales_ensemble = df_sales_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_sales")

    timing.start("predict_univ")
    y_pred_univ_ensemble = df_univ_ensemble[ensemble_list].median(axis=1)
    timing.stop("predict_univ")
    
    results = SingleModelResults(
        ds,
        "prediction",
        "he_id",
        "ensemble",
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

    return results


def _prepare_ds(
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    model_group: str,
    vacant_only: bool,
    settings: dict,
    ind_vars: list[str] | None = None,
):
    """Prepare a DataSplit object for modeling.

    :param df_sales: Sales DataFrame. :type df_sales: pandas.DataFrame :param df_universe:
    Universe DataFrame. :type df_universe: pandas.DataFrame :param model_group: Model
    group identifier. :type model_group: str :param vacant_only: Whether to use only
    vacant sales. :type vacant_only: bool :param settings: Settings dictionary. :type
    settings: dict :param ind_vars: List of independent variables (optional) :type
    ind_vars: list[str] | None :returns: A DataSplit object. :rtype: DataSplit
    """
    s = settings
    s_model = s.get("modeling", {})
    vacant_status = "vacant" if vacant_only else "main"
    model_entries = s_model.get("models", {}).get(vacant_status, {})
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
    correlation_results: dict,
    enr_results: dict,
    r2_values_results: pd.DataFrame,
    p_values_results: dict,
    t_values_results: dict,
    vif_results: dict,
    report: MarkdownReport = None,
):
    """Calculate variable recommendations based on various statistical metrics.
    """
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
                dfr_corr, "Variable", settings, ds.one_hot_descendants
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
                    dfr_vif, "Variable", settings, ds.one_hot_descendants
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
                    dfr_p_value, "Variable", settings, ds.one_hot_descendants
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
                    dfr_t_value, "Variable", settings, ds.one_hot_descendants
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
                    dfr_enr, "Variable", settings, ds.one_hot_descendants
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
                    dfr_r2, "Variable", settings, ds.one_hot_descendants
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
                    dfr_coef_sign, "Variable", settings, ds.one_hot_descendants
                )
                if state == "final":
                    dfr_coef_sign = dfr_coef_sign[dfr_coef_sign["Pass/Fail"].eq("✅")]
                report.set_var(
                    f"table_coef_sign_{state}", df_to_markdown(dfr_coef_sign)
                )

        dfr["Rank"] = range(1, len(dfr) + 1)
        dfr = _apply_dd_to_df_rows(dfr, "Variable", settings, ds.one_hot_descendants)

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


def _run_hedonic_models(
    settings: dict,
    model_group: str,
    models_to_run: list[str],
    all_results: MultiModelResults,
    df_sales: pd.DataFrame,
    df_universe: pd.DataFrame,
    dep_var: str,
    dep_var_test: str,
    fields_cat: list[str],
    verbose: bool = False,
    save_results: bool = False,
    run_ensemble: bool = True,
    do_plots: bool = False
):
    """
    Run hedonic models and ensemble them, then update the benchmark.
    """
    hedonic_results = {}

    # Run hedonic models
    outpath = f"out/models/{model_group}/hedonic_land"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    location_field_neighborhood = get_important_field(
        settings, "loc_neighborhood", df_sales
    )
    location_field_market_area = get_important_field(
        settings, "loc_market_area", df_sales
    )
    location_fields = [location_field_neighborhood, location_field_market_area]

    # Re-run the models one by one and stash the results
    for model in models_to_run:
        if model not in all_results.model_results:
            continue
        smr = all_results.model_results[model]
        ds = get_data_split_for(
            name=model,
            model_group=model_group,
            location_fields=location_fields,
            ind_vars=smr.ind_vars,
            df_sales=df_sales,
            df_universe=df_universe,
            settings=settings,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            fields_cat=fields_cat,
            interactions=smr.ds.interactions.copy(),
            test_keys=smr.ds.test_keys,
            train_keys=smr.ds.train_keys,
            vacant_only=False,
            hedonic=True,
            hedonic_test_against_vacant_sales=True,
        )

        # if the other one is one-hot encoded, we need to reconcile the fields
        ds = ds.reconcile_fields_with_foreign(smr.ds)

        # We call this here because we are re-running prediction without first calling run(), which would call this
        ds.split()
        if len(ds.y_sales) < 15:
            print(
                f"Skipping hedonic model because there are not enough sale records...."
            )
            return
        smr.ds = ds
        results = _predict_one_model(
            smr=smr,
            model=model,
            outpath=outpath,
            settings=settings,
            save_results=save_results,
            verbose=verbose,
        )
        if results is not None:
            hedonic_results[model] = results

    all_hedonic_results = MultiModelResults(
        model_results=hedonic_results, benchmark=_calc_benchmark(hedonic_results)
    )

    if run_ensemble:
        best_ensemble = _optimize_ensemble(
            df_sales=df_sales,
            df_universe=df_universe,
            model_group=model_group,
            vacant_only=False,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            all_results=all_hedonic_results,
            settings=settings,
            verbose=verbose,
            hedonic=True,
        )
        # Run the ensemble model
        ensemble_results = _run_ensemble(
            df_sales=df_sales,
            df_universe=df_universe,
            model_group=model_group,
            vacant_only=False,
            hedonic=True,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            outpath=outpath,
            ensemble_list=best_ensemble,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
        )

        out_pickle = f"{outpath}/model_ensemble.pickle"
        with open(out_pickle, "wb") as file:
            pickle.dump(ensemble_results, file)

        # Calculate final results, including ensemble
        all_hedonic_results.add_model("ensemble", ensemble_results)

    # --- Run stacked ensemble for hedonic ---
    _run_stacked_ensemble_for_model_group(
        t=TimingData(),
        model_group=model_group,
        vacant_only=False,
        all_results=all_hedonic_results,
        outpath=outpath,
        settings=settings,
        save_results=save_results,
        verbose=verbose,
    )
    
    print(f"\n************************************************************")
    print(f"HEDONIC LAND BENCHMARK ({model_group}) -- Assessor Metrics")
    print(f"************************************************************\n")
    print(all_hedonic_results.benchmark.print())
    
    max_trim = _get_max_ratio_study_trim(settings, model_group)

    title = "HEDONIC LAND"
    perf_metrics = _model_performance_metrics(model_group, all_hedonic_results, title, max_trim)
    print(perf_metrics)
    print("")

    if do_plots:
        _model_performance_plots(model_group, all_hedonic_results, title)
        print("")

    # Post-valuation metrics
    title = f"{title} (POST-VALUATION DATE)"
    if not all_hedonic_results.benchmark.test_post_val_empty:
        post_val_results = _get_post_valuation_mmr(all_hedonic_results)
        perf_metrics = _model_performance_metrics(model_group, post_val_results, title, max_trim)
        print(perf_metrics)
        print("")

        print("")


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


def _model_shaps(model_group: str, all_results: MultiModelResults, title: str):

    for key in all_results.model_results:
        smr: SingleModelResults = all_results.model_results[key]
        _title = f"{title}/{model_group}/{key}"
        compute_shap(smr, True, _title)


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
        metrics_data["Slope"].append(slope)

        trimmed_data["Model"].append(model_name)
        trimmed_data["count"].append(count_trim)
        trimmed_data["MAPE"].append(mape_trim)
        trimmed_data["MSE"].append(mse)
        trimmed_data["RMSE"].append(rmse)
        trimmed_data["m.ratio"].append(model_result.pred_test.ratio_study.median_ratio_trim)
        trimmed_data["avg.ratio"].append(model_result.pred_test.ratio_study.mean_ratio_trim)
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

    trimmed_df = pd.DataFrame(trimmed_data)
    trimmed_df.set_index("Model", inplace=True)
    trimmed_df["count"] = trimmed_df["count"].apply(lambda x: f"{x:,}").astype(str)
    trimmed_df["MSE"] = trimmed_df["MSE"].apply(lambda x: fancy_format(x)).astype(str)
    trimmed_df["RMSE"] = trimmed_df["RMSE"].apply(lambda x: f"{x:,.0f}").astype(str)
    trimmed_df["MAPE"] = trimmed_df["MAPE"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["Slope"] = trimmed_df["Slope"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["m.ratio"] = trimmed_df["m.ratio"].apply(lambda x: f"{x:.2f}").astype(str)
    trimmed_df["avg.ratio"] = trimmed_df["avg.ratio"].apply(lambda x: f"{x:.2f}").astype(str)

    metrics_df = metrics_df[["count","MAPE","MSE","RMSE","m.ratio","avg.ratio","Slope"]]
    trimmed_df = trimmed_df[["count","MAPE","MSE","RMSE","m.ratio","avg.ratio","Slope"]]

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


def _trim_hedonic_sales(
    df_sales: pd.DataFrame,
    model_group: str,
    impr_to_vac_ratio: float,
    random_seed: int,
    verbose: bool = False
):
    test_keys, train_keys = _read_split_keys(model_group)
    rng = np.random.default_rng(random_seed)

    all_vac_keys = df_sales.loc[df_sales["vacant_sale"], "key_sale"]

    selected_keys: set[str] = set(all_vac_keys)

    if verbose:
        print(f"-->Trimming hedonic sales for model group '{model_group}'...")
        print(f"---->all sales   : {len(df_sales)}")
        print(f"---->vacant sales: {len(all_vac_keys)}")
    
    for is_test in [True, False]:
        subset_keys = test_keys if is_test else train_keys
        mask = df_sales["key_sale"].isin(subset_keys)

        vac_mask   = mask & df_sales["vacant_sale"].eq(True)
        impr_mask  = mask & df_sales["vacant_sale"].eq(False)
        
        n_vac   = vac_mask.sum()
        n_impr  = impr_mask.sum()

        target = min(n_impr, math.ceil(n_vac * impr_to_vac_ratio))
        if target == 0:
            continue
        
        improvs = df_sales.index[impr_mask]
        sampled_idx = rng.choice(improvs, size=target, replace=False)
        
        n_samples = len(df_sales.loc[sampled_idx, "key_sale"])
        
        if verbose:
            word = "test" if is_test else "train"
            print(f"------>{word} sales  : {mask.sum()}")
            print(f"------>vacant sales: {n_vac}")
            print(f"------>num sampled : {n_samples}")
        
        selected_keys.update(df_sales.loc[sampled_idx, "key_sale"])
    
    if verbose:
        print(f"-------->Final selection: {len(selected_keys)} keys")
    
    return df_sales[df_sales["key_sale"].isin(selected_keys)]


def _run_models(
    sup: SalesUniversePair,
    model_group: str,
    settings: dict,
    main_vacant_hedonic: str = "main",
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
    if main_vacant_hedonic == "main":
        is_hedonic = False
        vacant_only = False
        outdir = "main"
        titleword = "MAIN"
    elif main_vacant_hedonic == "vacant":
        is_hedonic = False
        vacant_only = True
        outdir = "vacant"
        titleword = "VACANT"
    elif main_vacant_hedonic == "hedonic":
        is_hedonic = True
        vacant_only = False
        outdir = "hedonic_full"
        titleword = "HEDONIC FULL"
    else:
        raise ValueError(f"The only supported values are 'main', 'vacant', and 'hedonic', got '{main_vacant_hedonic}' instead!")
    
    t = TimingData()
    t.start("total")

    t.start("setup")
    df_univ = sup["universe"]
    df_sales = get_hydrated_sales_from_sup(sup)

    df_sales = df_sales[df_sales["model_group"].eq(model_group)].copy()
    df_univ = df_univ[df_univ["model_group"].eq(model_group)].copy()

    s = settings
    s_model = s.get("modeling", {})
    s_inst = s_model.get("instructions", {})
    s_mvh = s_inst.get(main_vacant_hedonic, {})

    if is_hedonic:
        # For a hedonic model, we don't want to overload the vacant signal
        # so we grab only a modest amount of improved sales to supplement the vacants
        s_sel = s_mvh.get("select", {})
        impr_to_vac_ratio = s_sel.get("improved_to_vacant_ratio", 4.0)
        random_seed = s_inst.get("random_seed", 1337)
        
        df_sales = _trim_hedonic_sales(df_sales, model_group, impr_to_vac_ratio, random_seed, verbose)
        
    default_value = get_sale_field(settings, df_sales)
    dep_var = s_inst.get("dep_var", default_value)
    dep_var_test = s_inst.get("dep_var_test", default_value)
    fields_cat = get_fields_categorical(s, df_univ, include_boolean=True)
    models_to_run = s_inst.get(main_vacant_hedonic, {}).get("run", None)

    model_entries = s_model.get("models").get(main_vacant_hedonic, {})

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
    t.start("var_recs")

    var_recs = get_variable_recommendations(
        df_sales,
        df_univ,
        vacant_only,
        settings,
        model_group,
        do_report=True,
        verbose=True,
    )
    best_variables = var_recs["variables"]

    any_results = False

    # Run the models one by one and stash the results
    t.start("run_models")
    for model in models_to_run:

        model_variables = best_variables
        # For tree-based models, we don't perform variable reduction
        if model in ["xgboost", "lightgbm", "catboost"]:
            model_variables = None

        results = run_one_model(
            df_sales=df_sales,
            df_universe=df_univ,
            vacant_only=vacant_only,
            model_group=model_group,
            model=model,
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
            model_results[model] = results
            any_results = True
        else:
            print(f"Could not generate results for model: {model}")

    if not any_results:
        print(
            f"No results generated for model_group: {model_group}, vacant_only: {vacant_only}. Skipping..."
        )
        return

    t.stop("run_models")

    t.start("calc benchmarks")
    # Calculate initial results (ensemble will use them)
    all_results = MultiModelResults(
        model_results=model_results, benchmark=_calc_benchmark(model_results)
    )
    t.stop("calc benchmarks")

    if run_ensemble:
        if verbose:
            print(f"Optimizing ensemble...")
        t.start("optimize ensemble")
        best_ensemble = _optimize_ensemble(
            df_sales=df_sales,
            df_universe=df_univ,
            model_group=model_group,
            vacant_only=vacant_only,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
        )
        t.stop("optimize ensemble")

        # Run the ensemble model
        t.start("run ensemble")
        if verbose:
            print(f"Running ensemble...")
        ensemble_results = _run_ensemble(
            df_sales=df_sales,
            df_universe=df_univ,
            model_group=model_group,
            vacant_only=vacant_only,
            hedonic=False,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            outpath=outpath,
            ensemble_list=best_ensemble,
            all_results=all_results,
            settings=settings,
            verbose=verbose,
        )
        t.stop("run ensemble")

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

    # Run stacked ensemble if enabled
    _run_stacked_ensemble_for_model_group(
        t=t,
        model_group=model_group,
        vacant_only=vacant_only,
        all_results=all_results,
        outpath=outpath,
        settings=settings,
        save_results=save_results,
        verbose=verbose,
    )
    
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
        _model_shaps(model_group, all_results, title)

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

    if not vacant_only and is_hedonic:
        t.start("run hedonic models")
        _run_hedonic_models(
            settings=settings,
            model_group=model_group,
            models_to_run=models_to_run,
            all_results=all_results,
            df_sales=df_sales,
            df_universe=df_univ,
            dep_var=dep_var,
            dep_var_test=dep_var_test,
            fields_cat=fields_cat,
            verbose=verbose,
            save_results=save_results,
            run_ensemble=run_ensemble,
            do_plots=do_plots
        )
        t.stop("run hedonic models")

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

    return MultiModelResults(model_results=new_results, benchmark=benchmark)


def _get_post_valuation_smr(smr: SingleModelResults, verbose: bool = False):
    y_pred_test = smr.df_test[smr.field_prediction].copy()
    y_pred_sales = smr.df_sales[smr.field_prediction].copy()
    y_pred_univ = smr.df_universe[smr.field_prediction].copy()
    new_smr = SingleModelResults(
        smr.ds.copy(),
        smr.field_prediction,
        smr.field_horizontal_equity_id,
        smr.type,
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


def _run_stacked_ensemble(
    trained_meta_model: object,
    base_models_test_predictions: dict[str, np.ndarray],
    test_contextual_features: pd.DataFrame | None,
    stacked_ensemble_settings: dict,
    base_models_used_for_training: list[str],
    feature_columns: list[str],
    ds_template: DataSplit,
    settings: dict,
    all_results: MultiModelResults,
    verbose: bool = False,
) -> SingleModelResults:
    """Run stacked ensemble predictions using trained meta-model."""
    if verbose:
        print("Starting _run_stacked_ensemble...")

    # Prepare test features and make predictions
    test_features, _ = _prepare_stacked_features(
        base_models_test_predictions,
        test_contextual_features,
        base_models_used_for_training,
        feature_columns,
        ds_template.test_indices if hasattr(ds_template, "test_indices") else None,
        "test",
        verbose,
    )
    y_pred_test = trained_meta_model.predict(test_features)

    # Prepare sales predictions
    sales_predictions = {
        model: all_results.model_results[model].pred_sales.y_pred
        for model in base_models_used_for_training
        if model in all_results.model_results
    }

    sales_contextual = _prepare_contextual_features(
        ds_template,
        stacked_ensemble_settings.get("contextual_features", []),
        stacked_ensemble_settings.get("categorical_contextual_features", []),
        None,
        False,
        settings,
        verbose,
    )

    sales_features, _ = _prepare_stacked_features(
        sales_predictions,
        sales_contextual,
        base_models_used_for_training,
        feature_columns,
        ds_template.sales_indices if hasattr(ds_template, "sales_indices") else None,
        "sales",
        verbose,
    )
    y_pred_sales = trained_meta_model.predict(sales_features)

    # Prepare universe predictions
    universe_predictions = {
        model: all_results.model_results[model].pred_univ
        for model in base_models_used_for_training
        if model in all_results.model_results
    }

    # Get base predictions length
    univ_length = len(next(iter(universe_predictions.values())))

    # Create a DataFrame with just the base predictions first
    universe_features, base_feature_names = _prepare_stacked_features(
        universe_predictions,
        None,  # No contextual features yet
        base_models_used_for_training,
        None,  # No feature columns yet
        None,  # No indices needed
        "universe (base)",
        verbose,
    )

    # Now get contextual features
    universe_contextual = _prepare_contextual_features(
        ds_template,
        stacked_ensemble_settings.get("contextual_features", []),
        stacked_ensemble_settings.get("categorical_contextual_features", []),
        None,
        False,
        settings,
        verbose,
    )

    # If we have contextual features, create interaction features
    if universe_contextual is not None and len(universe_contextual) > 0:
        # Ensure universe_contextual has same length as predictions
        if len(universe_contextual) != univ_length:
            if verbose:
                print(
                    f"Adjusting contextual features from {len(universe_contextual)} to {univ_length} records"
                )
            # Create zero-filled DataFrame with correct length
            zero_filled = pd.DataFrame(
                0, index=range(univ_length), columns=universe_contextual.columns
            )
            # Fill in the values we have
            zero_filled.iloc[: len(universe_contextual)] = universe_contextual.values
            universe_contextual = zero_filled

        # Now create interaction features
        universe_features_with_interactions, _ = _prepare_stacked_features(
            universe_predictions,
            universe_contextual,
            base_models_used_for_training,
            feature_columns,
            None,
            "universe (with interactions)",
            verbose,
        )
        y_pred_univ = trained_meta_model.predict(universe_features_with_interactions)
    else:
        # Just use base predictions if no contextual features
        y_pred_univ = trained_meta_model.predict(universe_features)

    # Create SingleModelResults with all predictions
    results = SingleModelResults(
        ds=ds_template,
        field_prediction="prediction",
        field_horizontal_equity_id="he_id",
        type="stacked_ensemble",
        model="stacked_ensemble",
        y_pred_test=y_pred_test,
        y_pred_sales=y_pred_sales,
        y_pred_univ=y_pred_univ,
        timing=TimingData(),
        verbose=verbose,
    )

    return results


def _run_stacked_ensemble_for_model_group(
    t: TimingData,
    model_group: str,
    vacant_only: bool,
    all_results: MultiModelResults,
    outpath: str,
    settings: dict,
    save_results: bool = False,
    verbose: bool = False,
):
    """Run stacked ensemble for a model group if enabled in settings."""
    try:
        # Get settings
        vacant_status = "vacant" if vacant_only else "main"
        stacked_settings = (
            settings.get("modeling", {})
            .get("instructions", {})
            .get(vacant_status, {})
            .get("stacked_ensemble", {})
        )

        if not stacked_settings.get("enabled", False):
            return

        t.start("stacked_ensemble_train_and_predict")

        # Training phase
        trained_model, used_models, feature_columns = _train_stacked_ensemble_phase(
            all_results, stacked_settings, settings, verbose
        )

        # Prediction phase
        stacked_results = _run_stacked_ensemble(
            trained_meta_model=trained_model,
            base_models_test_predictions=_collect_base_model_predictions(
                used_models, all_results, "test", verbose
            )[0],
            test_contextual_features=_prepare_contextual_features(
                all_results.model_results[used_models[0]].ds,
                stacked_settings.get("contextual_features", []),
                stacked_settings.get("categorical_contextual_features", []),
                feature_columns,
                True,
                settings,
                verbose,
            ),
            stacked_ensemble_settings=stacked_settings,
            base_models_used_for_training=used_models,
            feature_columns=feature_columns,
            ds_template=all_results.model_results[used_models[0]].ds,
            settings=settings,
            all_results=all_results,
            verbose=verbose,
        )

        # Save results
        all_results.add_model("stacked_ensemble", stacked_results)
        if save_results:
            meta_model_path = f"{outpath}/model_stacked_ensemble_meta.pickle"
            with open(meta_model_path, "wb") as f_meta:
                pickle.dump(trained_model, f_meta)
            if verbose:
                print(f"Saved trained meta-model to {meta_model_path}")

            _write_ensemble_model_results(
                stacked_results,
                outpath,
                settings,
                dfs={
                    "test": stacked_results.ds.df_test,
                    "sales": stacked_results.ds.df_sales,
                    "universe": stacked_results.ds.df_universe,
                },
                ensemble_list=None,
            )
    except Exception as e:
        print(
            f"ERROR during stacked ensembling for model_group {model_group}, "
            f"vacant_only {vacant_only}: {e}"
        )
        warnings.warn(f"Stacked ensembling failed: {e}")
    finally:
        t.stop("stacked_ensemble_train_and_predict")


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


def _train_stacked_ensemble_phase(
    all_results: MultiModelResults,
    stacked_ensemble_settings: dict,
    settings: dict,
    verbose: bool = False,
) -> tuple[object, list[str], list[str] | None]:
    """Handle the training phase of stacked ensemble."""
    models_for_stacking = stacked_ensemble_settings.get("models_to_include", [])

    # Get base predictions and true values for training
    base_predictions = {}
    true_values = None
    template_ds = None

    # Collect OOF predictions for training
    for model_name in models_for_stacking:
        if model_name not in all_results.model_results:
            if verbose:
                print(f"Model '{model_name}' not found in results")
            continue

        smr = all_results.model_results[model_name]
        if smr.pred_sales is not None and smr.pred_sales.y_pred is not None:
            base_predictions[model_name] = smr.pred_sales.y_pred
            if true_values is None:
                true_values = smr.pred_sales.y
                template_ds = smr.ds

    if not base_predictions or true_values is None or template_ds is None:
        raise ValueError("Insufficient data for training stacked ensemble")

    # Get contextual features for training
    contextual_features = _prepare_contextual_features(
        template_ds,
        stacked_ensemble_settings.get("contextual_features", []),
        stacked_ensemble_settings.get("categorical_contextual_features", []),
        None,
        False,
        settings,
        verbose,
    )

    # Get encoded column names if they exist
    feature_columns = None
    if contextual_features is not None:
        feature_columns = contextual_features.columns.tolist()

    # Prepare training features
    train_features, feature_names = _prepare_stacked_features(
        base_predictions,
        contextual_features,
        list(base_predictions.keys()),
        feature_columns,
        None,
        "train",
        verbose,
    )

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(train_features, true_values)

    if verbose:
        print("Trained Ridge meta-model.")
        print(f"Using models: {list(base_predictions.keys())}")
        if feature_columns:
            print(f"Using contextual features: {feature_columns}")

    return meta_model, list(base_predictions.keys()), feature_columns


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
