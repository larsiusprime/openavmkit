import os
import warnings

import numpy as np
import pandas as pd
from pandas import read_pickle

import openavmkit.utilities.stats as stats
from openavmkit.data import get_vacant_sales, get_important_field
from openavmkit.reports import start_report, finish_report
from openavmkit.utilities.data import df_to_markdown, div_series_z_safe
from openavmkit.utilities.settings import (
    get_fields_categorical,
    get_data_dictionary,
    get_fields_impr_as_list,
    get_valuation_date,
    get_model_group_ids,
)


class RatioStudy:
    """
    Performs an IAAO-standard Ratio Study, generating all the relevant statistics.


    Attributes
    ----------
    predictions : np.ndarray
        Series representing predicted values
    ground_truth : np.ndarray
        Series representing ground truth values (typically observed sale prices)
    count : int
        The number of observations
    median_ratio : float
        The median value of all `prediction/ground_truth` ratios
    mean_ratio : float
        The mean value of all `prediction/ground_truth` ratios
    cod : float
        The coefficient of dispersion, a measure of variability (lower is better)
    cod_trim : float
        The coefficient of dispersion, after outlier ratios outside the interquartile range have been trimmed
    prd : float
        The price-related differential, a measure of vertical equity
    prb : float
        The price-related bias, a measure of vertical equity
    """

    predictions: np.ndarray
    ground_truth: np.ndarray
    count: int
    median_ratio: float
    median_ratio_trim: float
    cod: float
    cod_trim: float
    prd: float
    prb: float
    mean_ratio: float
    mean_ratio_trim: float

    def __init__(self, predictions: np.ndarray, ground_truth: np.ndarray, max_trim: float):
        """
        Initialize a ratio study object

        Parameters
        ----------
        predictions : np.ndarray
            Series representing predicted values
        ground_truth : np.ndarray
            Series representing ground truth values (typically observed sale prices)
        max_trim : float
            The maximum amount of records allowed to be trimmed in a ratio study
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("predictions and ground_truth must have the same length")

        if len(predictions) == 0:
            self.count = 0
            self.predictions = np.array([])
            self.ground_truth = np.array([])
            self.median_ratio = float("nan")
            self.cod = float("nan")
            self.cod_trim = float("nan")
            self.prd = float("nan")
            self.prb = float("nan")
            self.prd_trim = float("nan")
            self.prb_trim = float("nan")
            self.median_ratio_trim = float("nan")
            self.mean_ratio = float("nan")
            self.mean_ratio_trim = float("nan")
            return

        self.count = len(predictions)
        self.predictions = predictions
        self.ground_truth = ground_truth

        ratios = div_series_z_safe(predictions, ground_truth).astype(float)
        if len(ratios) > 0:
            median_ratio = float(np.median(ratios))
        else:
            median_ratio = float("nan")

        # trim the ratios to remove outliers -- trim to the interquartile range
        trim_mask = stats.trim_outliers_mask(ratios, max_trim)

        trim_ratios = ratios[trim_mask]
        trim_predictions = predictions[trim_mask]
        trim_ground_truth = ground_truth[trim_mask]
        
        self.count_trim = len(trim_ratios)

        cod = stats.calc_cod(ratios)
        cod_trim = stats.calc_cod(trim_ratios)

        prd = stats.calc_prd(predictions, ground_truth)
        prd_trim = stats.calc_prd(trim_predictions, trim_ground_truth)

        prb, _, _ = stats.calc_prb(predictions, ground_truth)
        prb_trim, _, _ = stats.calc_prb(trim_predictions, trim_ground_truth)

        self.median_ratio = median_ratio

        if len(ratios) == 0:
            self.mean_ratio = float("nan")
        else:
            self.mean_ratio = float(np.mean(ratios))

        if len(trim_ratios) == 0:
            self.mean_ratio_trim = float("nan")
            self.median_ratio_trim = float("nan")
        else:
            self.mean_ratio_trim = float(np.mean(trim_ratios))
            self.median_ratio_trim = float(np.median(trim_ratios))

        self.cod = cod
        self.cod_trim = cod_trim

        self.prd = prd
        self.prd_trim = prd_trim

        self.prb = prb
        self.prb_trim = prb_trim


class RatioStudyBootstrapped(RatioStudy):
    """
    Performs an IAAO-standard Ratio Study, generating all the relevant statistics.
    This extends the base RatioStudy class, adding confidence intervals.


    Attributes
    ----------
    iterations : float
        Number of bootstrap iterations
    cod_ci_low : float
        COD, bottom of the confidence interval
    cod_ci_high : float
        COD, top of the confidence interval
    cod_trim_ci_low : float
        Trimmed COD, bottom of the confidence interval
    cod_trim_ci_high : float
        Trimmed COD, top of the confidence interval
    prd_ci_low : float
        PRD, bottom of the confidence interval
    prd_ci_high : float
        PRD, top of the confidence interval
    """

    iterations: int
    cod_ci_low: float
    cod_ci_high: float
    cod_trim_ci_low: float
    cod_trim_ci_high: float
    prd_ci_low: float
    prd_ci_high: float

    def __init__(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        max_trim: float,
        confidence_interval: float = 0.95,
        iterations: int = 1000,
    ):
        """
        Initialize a Bootstrapped ratio study object

        Parameters
        ----------
        predictions : np.ndarray
            Series representing predicted values
        ground_truth : np.ndarray
            Series representing ground truth values (typically observed sale prices)
        max_trim : float
            The maximum amount of records allowed to be trimmed in a ratio study
        confidence_interval : float
            Desired confidence interval (default is 0.95, indicating 95% confidence)
        iterations : int
            How many bootstrap iterations to perform
        """
        super().__init__(predictions, ground_truth)

        if len(predictions) == 0:
            self.cod = float("nan")
            self.cod_ci_low = float("nan")
            self.cod_ci_high = float("nan")
            self.cod_trim = float("nan")
            self.cod_trim_ci_low = float("nan")
            self.cod_trim_ci_high = float("nan")
            self.prd = float("nan")
            self.prd_ci_low = float("nan")
            self.prd_ci_high = float("nan")
            self.prb = float("nan")
            self.prb_ci_low = float("nan")
            self.prb_ci_high = float("nan")

        self.iterations = iterations
        ratios = div_series_z_safe(predictions, ground_truth)
        med, low, high = stats.calc_cod_bootstrap(
            ratios, confidence_interval, iterations
        )

        self.cod = med
        self.cod_ci_low = low
        self.cod_ci_high = high

        med, low, high = stats.calc_cod_bootstrap(
            stats.trim_outliers(ratios, max_trim),
            confidence_interval,
            iterations,
        )

        self.cod_trim = med
        self.cod_trim_ci_low = low
        self.cod_trim_ci_high = high

        med, low, high = 0, 0, 0
        self.prd = med
        self.prd_ci_low = low
        self.prd_ci_high = high

        med, low, high = 0, 0, 0
        self.prb = med
        self.prb_ci_low = low
        self.prb_ci_high = high


def run_and_write_ratio_study_breakdowns(settings: dict):
    """Runs ratio studies, with breakdowns, and writes them to disk.

    Parameters
    ----------
    settings : dict
        Settings dictionary
    """
    model_groups = get_model_group_ids(settings)
    for model_group in model_groups:
        print(f"Generating report for {model_group}")
        path = f"out/models/{model_group}/main/model_ensemble.pickle"
        if os.path.exists(path):
            os.makedirs(f"out/models/{model_group}", exist_ok=True)
            ensemble_results = read_pickle(path)
            df_sales = ensemble_results.df_sales
            _run_and_write_ratio_study_breakdowns(
                settings, df_sales, model_group, f"out/models/{model_group}"
            )


#######################################
# PRIVATE
#######################################


def _run_and_write_ratio_study_breakdowns(
    settings: dict,
    df_sales: pd.DataFrame,
    model_group: str,
    path: str,
    confidence_interval=0.95,
    iterations=1000,
):
    breakdowns = _run_ratio_study_breakdowns(
        settings, model_group, df_sales, confidence_interval, iterations
    )
    _write_ratio_study_report(breakdowns, settings, model_group, path)


def _add_ratio_study(
    predictions,
    ground_truth,
    cluster,
    value,
    catch_all,
    confidence_interval,
    iterations,
    min_sales,
    max_trim
):
    # ignore na values in both predictions & ground truth
    idx_na = pd.isna(predictions) | pd.isna(ground_truth)
    if np.any(idx_na):
        predictions = predictions[~idx_na]
        ground_truth = ground_truth[~idx_na]

    if len(predictions) > min_sales:
        rs = RatioStudyBootstrapped(
            predictions, ground_truth, confidence_interval, iterations, max_trim
        )
        cluster[value] = rs
    else:
        catch_all["predictions"] = np.append(catch_all["predictions"], predictions)
        catch_all["ground_truth"] = np.append(catch_all["ground_truth"], ground_truth)
        catch_all["count"] += 1
    return cluster, catch_all


def _clean_label(label: str) -> str:
    label = label.replace("|", " ")
    label = label.replace("\r", "")
    label = label.replace("\n", "")
    if label.strip() in ["", "<NA>", "nan"]:
        label = "<BLANK>"
    return label


def _run_ratio_study_breakdowns(
    settings: dict, model_group: str, df_sales: pd.DataFrame, confidence_interval=0.95, iterations=10000
) -> dict:
    if "prediction" not in df_sales:
        raise ValueError("df_sales must have a 'prediction' column")

    rs = settings.get("analysis", {}).get("ratio_study", {})
    breakdowns = rs.get("breakdowns", [])

    look_back_years = rs.get("look_back_years", 1)
    val_date = get_valuation_date(settings)
    look_back_year = val_date.year - look_back_years

    df_sales = df_sales[df_sales["sale_year"].ge(look_back_year)]

    # insert "overall" breakdown into the first position:
    breakdowns.insert(0, {"by": "overall"})

    df_v = get_vacant_sales(df_sales, settings)
    df_i = get_vacant_sales(df_sales, settings, invert=True)

    cat_fields = get_fields_categorical(settings, df_sales, include_boolean=True)

    all = {"assessor": {}, "openavmkit": {}}

    min_sales = 15

    for is_assessor in [True, False]:
        if is_assessor:
            modeler = "assessor"
            prediction_field = "assr_market_value"
            if prediction_field not in df_sales:
                warnings.warn(
                    f"prediction_field '{prediction_field}' not found in df_sales, skipping assessor values"
                )
                continue
        else:
            modeler = "openavmkit"
            prediction_field = "prediction"

        results = {"vacant": [], "improved": []}

        predictions = df_sales[prediction_field].values
        ground_truth = df_sales["sale_price"].values

        idx_na = pd.isna(predictions) | pd.isna(ground_truth)
        predictions = predictions[~idx_na]
        ground_truth = ground_truth[~idx_na]
        
        max_trim = _get_max_ratio_study_trim(settings, model_group)

        results["overall"] = RatioStudyBootstrapped(
            predictions, ground_truth, confidence_interval, iterations, max_trim
        )

        for is_vacant in [False, True]:
            if is_vacant:
                df = df_v
                valid_field = "valid_for_land_ratio_study"
                results_key = "vacant"
            else:
                df = df_i
                valid_field = "valid_for_ratio_study"
                results_key = "improved"

            df = df[df[valid_field].eq(True)]

            for breakdown in breakdowns:
                by = breakdown.get("by")
                if by == "overall":
                    continue
                if "<" in by:
                    # TODO: when we have variable replacement in settings maybe we won't need this anymore
                    by = by.replace("<", "").replace(">", "")
                    by = get_important_field(settings, by, df)

                if by is None:
                    continue

                cluster = {}
                catch_all = {
                    "predictions": np.array([]),
                    "ground_truth": np.array([]),
                    "count": 0,
                }
                if by in cat_fields:
                    values = np.array(df[by].unique())
                    values.sort()
                    for value in values:
                        df_sub = df[df[by].eq(value)]
                        predictions = df_sub[prediction_field].values
                        ground_truth = df_sub["sale_price"].values
                        cluster, catch_all = _add_ratio_study(
                            predictions,
                            ground_truth,
                            cluster,
                            value,
                            catch_all,
                            confidence_interval,
                            iterations,
                            min_sales,
                            max_trim
                        )
                else:
                    quantiles = breakdown.get("quantiles", 0)
                    slice_size = breakdown.get("slice_size", 0)
                    df_sub = df.copy()
                    if quantiles > 0:
                        bins = [0]
                        labels = []
                        last_value = 0
                        for q in range(quantiles + 1):
                            try:
                                quantile_value = np.quantile(df_sub[by], q / quantiles)
                            except IndexError:
                                continue
                            percentile = f"{q / quantiles * 100:3.0f}th %ile<br>({last_value:,.0f} - {quantile_value:,.0f})"
                            if quantile_value not in bins:
                                bins.append(quantile_value)
                                labels.append(percentile)
                            last_value = quantile_value
                        df_sub["quantile"] = pd.cut(
                            df_sub[by],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,
                            duplicates="drop",
                        )

                        for q in labels:
                            q_clean = _clean_label(q)
                            df_slice = df_sub[df_sub["quantile"].eq(q)]
                            predictions = df_slice[prediction_field].values
                            ground_truth = df_slice["sale_price"].values
                            predictions = np.array(predictions)
                            ground_truth = np.array(ground_truth)
                            cluster, catch_all = _add_ratio_study(
                                predictions,
                                ground_truth,
                                cluster,
                                q_clean,
                                catch_all,
                                confidence_interval,
                                iterations,
                                min_sales,
                                max_trim
                            )

                    elif slice_size > 0:
                        # TODO: watch for nulls here in the future
                        df_sub["slice"] = (df_sub[by] // slice_size) * slice_size
                        values = np.array(df_sub["slice"].unique())
                        values.sort()
                        value_labels = {}
                        for value in values:
                            value_labels[value] = (
                                f"{value:,.0f} - {value + (slice_size-1):,.0f}"
                            )
                        df_sub["slice"] = df_sub["slice"].map(value_labels)
                        values = []
                        for key in value_labels:
                            values.append(value_labels[key])
                        for value in values:
                            df_slice = df_sub[df_sub["slice"].eq(value)]
                            predictions = df_slice[prediction_field].values
                            ground_truth = df_slice["sale_price"].values
                            cluster, catch_all = _add_ratio_study(
                                predictions,
                                ground_truth,
                                cluster,
                                value,
                                catch_all,
                                confidence_interval,
                                iterations,
                                min_sales,
                                max_trim
                            )
                    else:
                        if by not in df_sub:
                            raise ValueError(f"Field '{by}' not found in df_sub")
                        values = df_sub[by].unique()
                        for value in values:
                            df_slice = df_sub[df_sub[by].eq(value)]
                            predictions = df_slice[prediction_field].values
                            ground_truth = df_slice["sale_price"].values
                            cluster, catch_all = _add_ratio_study(
                                predictions,
                                ground_truth,
                                cluster,
                                value,
                                catch_all,
                                confidence_interval,
                                iterations,
                                min_sales,
                                max_trim
                            )

                if catch_all["count"] > 0:
                    rs = RatioStudyBootstrapped(
                        np.array(catch_all["predictions"]),
                        np.array(catch_all["ground_truth"]),
                        confidence_interval,
                        iterations,
                        max_trim
                    )
                    catch_all_count = catch_all["count"]
                    other_group = (
                        "other group" if catch_all_count <= 1 else "other groups"
                    )
                    cluster[
                        f"{catch_all_count} {other_group} with count < {min_sales}"
                    ] = rs

                results[results_key].append({"by": by, "cluster": cluster})

        all[modeler] = results

    return all


def _format_stat(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:6.1f}"


def _format_pair(value, value2):
    if pd.isna(value) or pd.isna(value2):
        return "N/A"
    return f"{value:6.1f} - {value2:6.1f}"


def _write_ratio_study_report(
    all_results: dict, settings: dict, model_group: str, path: str
):

    report = start_report("ratio_study", settings, model_group)

    locality_results = ""
    modeler_results = ""

    locality_name = settings.get("locality", {}).get("name", "Local Jurisdiction")

    dd = get_data_dictionary(settings)

    data_overall = {
        "Statistic": ["Count", "Median ratio", "COD", "COD 95% conf. range"],
        locality_name: [],
        "Our results": [],
    }

    data_overall_trim = {
        "Statistic": ["Count", "Median ratio", "COD", "COD 95% conf. range"],
        locality_name: [],
        "Our results": [],
    }

    impr_fields = get_fields_impr_as_list(settings)
    
    max_trim = _get_max_ratio_study_trim(settings, model_group)
    for modeler in all_results:
        modeler_entry = all_results[modeler]
        if modeler_entry == {}:
            overall_entry = RatioStudyBootstrapped(np.array([]), np.array([]), max_trim)
        else:
            overall_entry: RatioStudyBootstrapped = modeler_entry.get("overall")

        data_untrim = []
        data_trim = []

        data_untrim.append(f"{overall_entry.count:,.0f}")
        data_untrim.append(f"{overall_entry.median_ratio:5.2f}")
        data_untrim.append(f"{overall_entry.cod:5.1f}")
        data_untrim.append(
            f"{overall_entry.cod_ci_low:6.1f} - {overall_entry.cod_ci_high:6.1f}"
        )

        data_trim.append(f"{overall_entry.count_trim:,.0f}")
        data_trim.append(f"{overall_entry.median_ratio:5.2f}")
        data_trim.append(f"{overall_entry.cod_trim:5.1f}")
        data_trim.append(
            f"{overall_entry.cod_trim_ci_low:6.1f} - {overall_entry.cod_trim_ci_high:6.1f}"
        )

        if modeler == "assessor":
            data_overall[locality_name] = data_untrim
            data_overall_trim[locality_name] = data_trim
        else:
            data_overall["Our results"] = data_untrim
            data_overall_trim["Our results"] = data_trim

        for vacant_status in modeler_entry:
            if vacant_status == "overall":
                continue
            vacant_name = (
                "vacant land" if vacant_status == "vacant" else "improved property"
            )
            vacant_status_entry = modeler_entry[vacant_status]
            for breakdown_entry in vacant_status_entry:
                by: str = breakdown_entry.get("by")
                if by == "overall":
                    continue
                if vacant_status == "vacant" and by in impr_fields:
                    continue
                cluster: dict[str, RatioStudyBootstrapped] = breakdown_entry.get(
                    "cluster"
                )
                by_name = dd.get(by, {}).get("name", by)

                data = {
                    by_name: [],
                    "Count": [],
                    "Median ratio": [],
                    "COD": [],
                    "COD 95% conf. range": [],
                    "COD (trimmed)": [],
                    "COD (trimmed) 95% conf. range": [],
                }

                for cluster_key in cluster:
                    rs = cluster[cluster_key]
                    data[by_name].append(cluster_key)
                    data["Count"].append(f"{rs.count:,.0f}")

                    median_ratio = _format_stat(rs.median_ratio)
                    cod = _format_stat(rs.cod)
                    cod_ci = _format_pair(rs.cod_ci_low, rs.cod_ci_high)
                    cod_trim = _format_stat(rs.cod_trim)
                    cod_trim_ci = _format_pair(rs.cod_trim_ci_low, rs.cod_trim_ci_high)

                    data["Median ratio"].append(median_ratio)
                    data["COD"].append(cod)
                    data["COD 95% conf. range"].append(cod_ci)
                    data["COD (trimmed)"].append(cod_trim)
                    data["COD (trimmed) 95% conf. range"].append(cod_trim_ci)

                md_chunk = f"#### By {by_name}, {vacant_name} only\n\n"
                df = pd.DataFrame(data=data)
                if len(df) > 1:
                    md_chunk += df_to_markdown(df)
                    md_chunk += "\n\n"

                    if modeler == "assessor":
                        locality_results += md_chunk
                    else:
                        modeler_results += md_chunk

    df_untrim = pd.DataFrame(data=data_overall)
    df_trim = pd.DataFrame(data=data_overall_trim)

    overall_results = "#### Untrimmed\n\n"
    overall_results += df_to_markdown(df_untrim)
    overall_results += "\n\n"
    overall_results += "#### Trimmed\n\n"
    overall_results += df_to_markdown(df_trim)

    report.set_var("overall_results", overall_results)
    report.set_var("locality_results", locality_results)
    report.set_var("modeler_results", modeler_results)

    rs = settings.get("analysis", {}).get("ratio_study", {})
    look_back_years = rs.get("look_back_years", 1)

    val_date = get_valuation_date(settings)
    look_back_date = val_date.replace(year=val_date.year - look_back_years)
    look_back_date_str = look_back_date.strftime("%Y-%m-%d")

    report.set_var("sales_back_to_date", look_back_date_str)

    outpath = f"{path}/reports/ratio_study"

    finish_report(report, outpath, "ratio_study", settings)
