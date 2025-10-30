import warnings

import polars as pl
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import MissingDataError

from openavmkit.utilities.data import div_series_z_safe
from openavmkit.utilities.settings import get_fields_boolean, get_fields_categorical


def calc_chds(df_in: pd.DataFrame, field_cluster: str, field_value: str) -> pd.Series:
    """Calculate the Coefficient of Horizontal Dispersion (CHD) for each cluster in a
    DataFrame.

    CHD is the same statistic as COD, the Coefficient of Dispersion, but
    calculated for horizontal equity clusters and used to measure horizontal dispersion, on
    the theory that similar properties in similar locations should have similar valuations.
    The use of the name "CHD" is chosen to avoid confusion because assessors strongly
    associate "COD" with sales ratio studies.

    This function computes the CHD for each unique cluster in the input DataFrame based on
    the values in the specified field.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame
    field_cluster : str
        Name of the column representing cluster identifiers.
    field_value : str
        Name of the column containing the values for COD calculation

    Returns
    -------
    A Series of COD values for each row, aligned with df_in
    """
    # Create a dataframe matching the index of df_in with only the cluster id.
    df = df_in[[field_cluster]].copy()
    df["chd"] = 0.0

    clusters = df[field_cluster].unique()

    for cluster in clusters:
        df_cluster = df[df[field_cluster].eq(cluster)]
        # exclude negative and null values:
        df_cluster = df_cluster[
            ~pd.isna(df_cluster[field_value]) & df_cluster[field_value].gt(0)
        ]

        chd = calc_cod(df_cluster[field_value].values)
        df.loc[df[field_cluster].eq(cluster), "chd"] = chd

    return df["chd"]


def quick_median_chd_pl(
    df: pl.DataFrame, field_value: str, field_cluster: str
) -> float:
    """
    Calculate the median CHD for groups in a Polars DataFrame.

    This function filters out missing values for the given value field, groups the data
    by the specified cluster field, computes COD for each group, and returns the median
    COD value.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame.
    field_value : str
        Name of the field containing values for COD calculation.
    field_cluster : str
        Name of the field to group by for computing COD.

    Returns
    -------
    float
        The median COD value across all groups.
    """
    # Filter out rows with missing values for field_value.
    df = df.filter(~pd.isna(df[field_value]))
    df = df.filter(df[field_value].gt(0))

    chds = df.group_by(field_cluster).agg(pl.col(field_value).alias("values"))

    # Apply the calc_cod function to each group (the list of values)
    chd_values = np.array([calc_cod(group.to_numpy()) for group in chds["values"]])

    # Calculate the median of the CHD values
    median_chd = float(np.median(chd_values))
    return median_chd


def calc_cod(values: np.ndarray) -> float:
    """
    Calculate the Coefficient of Dispersion (COD) for an array of values.

    COD is defined as the average absolute deviation from the median, divided by the
    median, multiplied by 100. Special cases are handled if the median is zero.

    Parameters
    ----------
    values : numpy.ndarray
        Array of numeric values.

    Returns
    -------
    float
        The COD percentage.
    """
    if len(values) == 0:
        return float("nan")

    median_value = np.median(values)
    abs_delta_values = np.abs(values - median_value)
    avg_abs_deviation = np.sum(abs_delta_values) / len(values)
    if median_value == 0:
        # if every value is zero, the COD is zero:
        if np.all(values == 0):
            return 0.0
        else:
            # if the median is zero but not all values are zero, return infinity
            return float("inf")
    cod = avg_abs_deviation / median_value
    cod *= 100
    return cod


def calc_ratio_stats_bootstrap(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    confidence_interval: float = 0.95,
    iterations: int = 10000,
    seed: int = 777
) -> dict:
    """
    Calculate ratio study statistics (Median ratio, Mean ratio, COD, PRD) with
    bootstrap percentile confidence intervals, following IAAO definitions.
    
    Parameters
    ----------
    predictions: np.ndarray
        An array of predicted values
    ground_truth: np.ndarray
        An array of corresponding ground truth (e.g. sale price) values
    confidence_interval: float
        The size of the confidence interval (e.g. 0.95 = 95% confidence)
    iterations: int, optional
        The number of bootstrap iterations to perform. Defaults to 10,000.
    seed: int, optional
        Random seed, for reproducibility. Defaults to 777.
    
    Returns
    -------
    dict
        {
          "median_ratio": ConfidenceStat,
          "mean_ratio": ConfidenceStat,
          "cod": ConfidenceStat,   # COD = 100 * mean(|ri - median(r)|) / median(r)
          "prd": ConfidenceStat    # PRD = mean(r) / weighted_mean(r)
        }
    """
    # --- input hygiene ---
    p = np.asarray(predictions, dtype=float).ravel()
    s = np.asarray(ground_truth,  dtype=float).ravel()
    if p.shape != s.shape:
        raise ValueError("predictions and ground_truth must have the same shape")

    mask = np.isfinite(p) & np.isfinite(s) & (s > 0)  # sales must be > 0
    p, s = p[mask], s[mask]
    n = p.size
    if n == 0:
        return None

    r = p / s

    # Point estimates from the original sample
    med_ratio_point = np.median(r)
    mean_ratio_point = np.mean(r)
    cod_point = (np.mean(np.abs(r - med_ratio_point)) / med_ratio_point) * 100.0
    weighted_mean_ratio_point = p.sum() / s.sum()
    prd_point = mean_ratio_point / weighted_mean_ratio_point

    # Bootstrap on indices of the (p, s) pairs
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))

    r_samp = (p[idx] / s[idx])                     # (B, n) ratios per replicate
    med_samp = np.median(r_samp, axis=1)           # (B,)
    mean_samp = np.mean(r_samp, axis=1)            # (B,)
    
    # per-replicate weighted mean ratio: sum(p)/sum(s) for the bootstrap sample
    wmr_samp = (p[idx].sum(axis=1) / s[idx].sum(axis=1))
    
    # per-replicate COD uses the replicate's own median
    cod_samp = (np.mean(np.abs(r_samp - med_samp[:, None]), axis=1) / med_samp) * 100.0
    
    # per-replicate PRD
    prd_samp = mean_samp / wmr_samp

    # percentile confidence intervals
    alpha = (1.0 - confidence_interval) / 2.0
    def ci(a):
        lo, hi = np.quantile(a, [alpha, 1.0 - alpha])
        return float(lo), float(hi)

    med_lo, med_hi   = ci(med_samp)
    mean_lo, mean_hi = ci(mean_samp)
    cod_lo, cod_hi   = ci(cod_samp)
    prd_lo, prd_hi   = ci(prd_samp)
    
    return {
        "median_ratio": ConfidenceStat(med_ratio_point,  confidence_interval, med_lo,  med_hi),
        "mean_ratio":   ConfidenceStat(mean_ratio_point, confidence_interval, mean_lo, mean_hi),
        "cod":          ConfidenceStat(cod_point,        confidence_interval, cod_lo,  cod_hi),
        "prd":          ConfidenceStat(prd_point,        confidence_interval, prd_lo,  prd_hi),
    }


def calc_cod_bootstrap(
    values: np.ndarray,
    confidence_interval: float = 0.95,
    iterations: int = 10000,
    seed: int = 777
) -> tuple[float, float, float]:
    """
    Calculate COD using bootstrapping.

    This function bootstraps the input values (resampling with replacement) to
    generate a distribution of CODs, then returns the median COD along with the
    lower and upper bounds of the confidence interval.

    Parameters
    ----------
    values : numpy.ndarray
        Array of numeric values.
    confidence_interval : float, optional
        The desired confidence level. Defaults to 0.95.
    iterations : int, optional
        Number of bootstrap iterations. Defaults to 10000.
    seed : int, optional
        Random seed for reproducibility. Defaults to 777.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the median COD, lower bound, and upper bound of the confidence interval.
    """

    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    np.random.seed(seed)

    # Replace negative values with zero:
    values = np.where(values < 0, 0, values)

    median = np.median(values)
    samples = np.random.choice(values, size=(iterations, n), replace=True)
    abs_delta_values = np.abs(samples - median)
    bootstrap_cods = np.mean(abs_delta_values, axis=1) / median * 100
    alpha = (1.0 - confidence_interval) / 2
    lower_bound, upper_bound = np.quantile(bootstrap_cods, [alpha, 1.0 - alpha])
    median_cod = np.median(bootstrap_cods)
    return median_cod, lower_bound, upper_bound


def calc_prd(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate the Price Related Differential (PRD).

    PRD is computed as the ratio of the mean ratio to the weighted mean ratio of
    predictions to ground truth.

    Parameters
    ----------
    predictions : numpy.ndarray
        Array of predicted values.
    ground_truth : numpy.ndarray
        Array of ground truth values.

    Returns
    -------
    float
        The PRD value.
    """
    ratios = div_series_z_safe(predictions, ground_truth)
    if len(ratios) == 0:
        return float("nan")
    mean_ratio = np.mean(ratios)
    sum_ground_truth = np.sum(ground_truth)
    if sum_ground_truth == 0:
        return float("inf")
    weighted_mean_ratio = np.sum(predictions) / sum_ground_truth
    if weighted_mean_ratio == 0:
        return float("inf")
    prd = mean_ratio / weighted_mean_ratio
    return prd


def calc_prd_bootstrap(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    confidence_interval: float = 0.95,
    iterations: int = 10000,
    seed: int = 777,
) -> tuple[float, float, float]:
    """
    Calculate PRD with bootstrapping.

    This function bootstraps the prediction-to-ground_truth ratios to produce a
    distribution of PRD values and returns the lower bound, median, and upper bound of the
    confidence interval.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predicted values.
    ground_truth : np.ndarray
        Array of ground truth values.
    confidence_interval : float, optional
        The desired confidence level. Defaults to 0.95.
    iterations : int, optional
        Number of bootstrap iterations. Defaults to 10000.
    seed : int, optional
        Random seed for reproducibility. Defaults to 777.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing median PRD, the lower bound, and upper bound of the confidence interval.
    """

    np.random.seed(seed)
    n = len(predictions)
    ratios = predictions / ground_truth
    samples = np.random.choice(ratios, size=(iterations, n), replace=True)
    mean_ratios = np.mean(samples, axis=1)
    weighted_mean_ratios = np.sum(predictions) / np.sum(ground_truth)
    prds = mean_ratios / weighted_mean_ratios
    alpha = (1.0 - confidence_interval) / 2
    lower_bound, upper_bound = np.quantile(prds, [alpha, 1.0 - alpha])
    median_prd = np.median(prds)
    return median_prd, lower_bound, upper_bound


def trim_outlier_ratios(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    max_percent: float = 0.10,
    iqr_factor: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    
    ratios = div_series_z_safe(predictions, ground_truth).astype(float)
    trim_mask = trim_outliers_mask(ratios, max_percent, iqr_factor)
    trim_ratios = ratios[trim_mask]
    trim_predictions = predictions[trim_mask]
    trim_ground_truth = ground_truth[trim_mask]
    return trim_predictions, trim_ground_truth


def trim_outliers(
    values: np.ndarray,
    max_percent: float = 0.10,
    iqr_factor: float = 1.5
) -> np.ndarray:
    """
    Trim outliers using IQR fences per IAAO guidance, with a max trim cap.
    Fails immediately if NaNs are detected.

    1) Compute Q1, Q3, IQR = Q3 - Q1.
    2) Trim values outside [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR].
    3) If more than max_percent would be removed, instead trim by symmetric
       quantile cut so total trimmed <= max_percent.

    Parameters
    ----------
    values : np.ndarray
        1D numeric array with no NaNs allowed.
    max_percent : float, optional
        Maximum fraction to remove (e.g., 0.10 = 10%).
    iqr_factor : float, optional
        1.5 for standard outliers, 3.0 for extreme outliers.

    Returns
    -------
    np.ndarray
        Trimmed array according to the above rules.

    Raises
    ------
    ValueError
        If any NaN is detected in `values`.
    """
    if np.isnan(values).any():
        raise ValueError("NaN values detected — remove or impute them before trimming.")

    if values.size == 0:
        return values

    # IQR fences
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - iqr_factor * iqr
    upper_fence = q3 + iqr_factor * iqr

    within_fences = (values >= lower_fence) & (values <= upper_fence)
    trimmed_iqr = values[within_fences]

    n = values.size
    max_trim = int(np.floor(n * max_percent))

    # If IQR-based trimming exceeds cap, fall back to symmetric quantile trim
    trimmed_count = n - trimmed_iqr.size
    if trimmed_count > max_trim:
        tail_fraction = max_percent / 2.0
        low_q, high_q = tail_fraction, 1.0 - tail_fraction
        lo, hi = np.quantile(values, [low_q, high_q])
        keep = (values >= lo) & (values <= hi)
        trimmed = values[keep]
    else:
        trimmed = trimmed_iqr

    return trimmed

def trim_outliers_mask(
    values: np.ndarray,
    max_percent: float = 0.10,
    iqr_factor: float = 1.5,
) -> np.ndarray:
    """
    Trim outliers using IQR fences per IAAO guidance, with a max trim cap.
    Fails immediately if NaNs are detected.

    1) Compute Q1, Q3, IQR = Q3 - Q1.
    2) Trim values outside [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR].
    3) If more than max_percent would be removed, instead trim by symmetric
       quantile cut so total trimmed <= max_percent.

    Parameters
    ----------
    values : np.ndarray
        1D numeric array with no NaNs allowed.
    max_percent : float, optional
        Maximum fraction to remove (e.g., 0.10 = 10%).
    iqr_factor : float, optional
        1.5 for standard outliers, 3.0 for extreme outliers.
    
    Returns
    -------
    np.ndarray
        Boolean array where `True` indicates values within the quantile bounds.

    Raises
    ------
    ValueError
        If any NaN is detected in `values`.
    """
    if np.isnan(values).any():
        raise ValueError("NaN values detected — remove or impute them before trimming.")

    if values.size == 0:
        return values

    # IQR fences
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - iqr_factor * iqr
    upper_fence = q3 + iqr_factor * iqr

    within_fences = (values >= lower_fence) & (values <= upper_fence)
    trimmed_iqr = values[within_fences]

    n = values.size
    max_trim = int(np.floor(n * max_percent))

    # If IQR-based trimming exceeds cap, fall back to symmetric quantile trim
    trimmed_count = n - trimmed_iqr.size
    if trimmed_count > max_trim:
        tail_fraction = max_percent / 2.0
        low_q, high_q = tail_fraction, 1.0 - tail_fraction
        lo, hi = np.quantile(values, [low_q, high_q])
        return (values >= lo) & (values <= hi)
    else:
        return within_fences


def calc_prb(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    confidence_interval: float = 0.95
) -> tuple[float, float, float]:
    """
    Calculate the Price Related Bias (PRB) metric using a regression-based approach.

    This function fits an OLS model on the transformed ratios of predictions to ground
    truth, then returns the PRB value along with its lower and upper confidence bounds.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predicted values.
    ground_truth : np.ndarray
        Array of ground truth values.
    confidence_interval : float, optional
        Desired confidence interval. Defaults to 0.95.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:

        - PRB value
        - Lower bound of the confidence interval
        - Upper bound of the confidence interval

    Raises
    ------
    ValueError
        If `predictions` and `ground_truth` have different lengths.
    """
    # 1. Basic shape checks --------------------------------------------------
    predictions = np.asarray(predictions, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same length/shape")

    # 2. Clean rows that cannot be used --------------------------------------
    mask = (
        ~np.isnan(predictions)
        & ~np.isnan(ground_truth)
        & (predictions > 0)          # cannot take log2 of non‑positive numbers
        & (ground_truth > 0)
    )
    n_ok = int(mask.sum())
    if n_ok < 3:                     # OLS needs at least 3 rows
        warnings.warn(
            f"Only {n_ok} valid observation(s) after cleaning – PRB not computed."
        )
        return np.nan, np.nan, np.nan

    preds = predictions[mask]
    truth = ground_truth[mask]

    # 3. Build transformed variables -----------------------------------------
    ratios = preds / truth
    median_ratio = np.median(ratios)

    left = (ratios - median_ratio) / median_ratio
    right = np.log2(preds / median_ratio + truth)
    right = sm.add_constant(right, has_constant='add')   # adds intercept term

    # 4. Fit model + CI -------------------------------------------------------
    with np.errstate(all="ignore"):  # silence harmless internal numpy warnings
        model = sm.OLS(left, right).fit()

    # Guard against degenerate fit (rare but better to be explicit)
    if model.df_resid <= 0 or not np.isfinite(model.params[0]):
        return np.nan, np.nan, np.nan

    prb = float(model.params[0])
    prb_lower, prb_upper = (
        model.conf_int(alpha=1.0 - confidence_interval)[0].tolist()
    )

    return prb, prb_lower, prb_upper


def plot_correlation(corr: pd.DataFrame, title: str = "Correlation of Variables") -> None:
    """
    Plot a heatmap of a correlation matrix.

    Parameters
    ----------
    corr : pandas.DataFrame
        Correlation matrix as a DataFrame.
    title : str, optional
        Title of the plot. Defaults to "Correlation of Variables".
    """

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = cmap.reversed()

    plt.figure(figsize=(10, 8))

    # Create the heatmap with the correct labels
    sns.heatmap(
        corr,
        annot=True,
        fmt=".1f",
        cbar=True,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        xticklabels=corr.columns.tolist(),  # explicitly set the xticklabels
        yticklabels=corr.index.tolist(),  # explicitly set the yticklabels
        annot_kws={"size": 8},  # adjust font size if needed
    )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")  # rotate x labels if needed
    plt.yticks(rotation=0)  # keep y labels horizontal
    plt.tight_layout(pad=2)
    plt.show()


def calc_correlations(
    X: pd.DataFrame,
    threshold: float = 0.1,
    do_plots: bool = False
) -> dict:
    """
    Calculate correlations and iteratively drop variables with low combined scores.

    This function computes the correlation matrix of `X`, then calculates a combined score
    for each variable based on its correlation strength with the target variable and its
    average correlation with other variables. Variables whose scores fall below the
    specified `threshold` are removed.

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame containing the variables to evaluate.
    threshold : float, optional
        Minimum acceptable combined score for variables. Variables with a score below this
        value will be dropped. Defaults to 0.1.
    do_plots : bool, optional
        If True, plot the initial and final correlation heatmaps. Defaults to False.

    Returns
    -------
    dict
        Dictionary with two keys:

        - "initial": pandas.Series of combined scores from the first iteration.
        - "final": pandas.Series of combined scores after dropping low-scoring variables.
    """
    X = X.copy()
    first_run = None

    # Normalize all numerical values prior to computation:
    for col in X.columns:
        if X[col].dtype != "object":
            X[col] = (X[col] - X[col].mean()) / X[col].std()

    while True:
        # Compute the correlation matrix
        naive_corr = X.corr()

        # Identify variables with the highest correlation with the target variable (the first column)
        target_corr = naive_corr.iloc[:, 0].abs().sort_values(ascending=False)

        # Sort naive_corr by the correlation of the target variable
        naive_corr = naive_corr.loc[target_corr.index, target_corr.index]

        naive_corr_sans_target = naive_corr.iloc[1:, 1:]

        # Get the (absolute) strength of the correlation with the target variable
        strength = naive_corr.iloc[:, 0].abs()

        # drop the target variable from strength:
        strength = strength.iloc[1:]

        # Calculate the clarity of the correlation: how correlated it is with all other variables *except* the target variable
        clarity = 1 - (
            (naive_corr_sans_target.abs().sum(axis=1) - 1.0)
            / (len(naive_corr_sans_target.columns) - 1)
        )

        # Combine the strength and clarity into a single score -- bigger is better, and we want high strength and high clarity
        score = strength * clarity * clarity

        # Identify the variable with the lowest score
        min_score_idx = score.idxmin()

        if pd.isna(min_score_idx):
            min_score = score[0]
        else:
            min_score = score[min_score_idx]

        data = {"corr_strength": strength, "corr_clarity": clarity, "corr_score": score}
        df_score = pd.DataFrame(data)
        df_score = df_score.reset_index().rename(columns={"index": "variable"})

        if first_run is None:
            first_run = df_score
            first_run = first_run.sort_values("corr_score", ascending=False)

        if min_score < threshold:
            X = X.drop(min_score_idx, axis=1)
        else:
            break

    # sort by score:
    df_score = df_score.sort_values("corr_score", ascending=False)

    if do_plots:
        plot_correlation(naive_corr, "Correlation of Variables (initial)")

    # recompute the correlation matrix
    final_corr = X.corr()

    if do_plots:
        plot_correlation(final_corr, "Correlation of Variables (final)")

    return {"initial": first_run, "final": df_score}


def calc_elastic_net_regularization(
    X: pd.DataFrame,
    y: pd.Series,
    threshold_fraction: float = 0.05
) -> dict:
    """
    Calculate Elastic Net regularization coefficients while iteratively dropping variables with low coefficients.

    This function standardizes `X`, fits an Elastic Net model, and iteratively removes
    variables whose absolute coefficients fall below a fraction of the maximum coefficient.

    Parameters
    ----------
    X : pd.DataFrame
        Input features DataFrame.
    y : pd.Series
        Target variable series.
    threshold_fraction : float, optional
        Fraction of the maximum coefficient below which variables are dropped. Defaults to 0.05.

    Returns
    -------
    dict
        Dictionary with two keys:

        - "initial": pandas.Series of coefficients from the first Elastic Net fit.
        - "final": pandas.Series of coefficients after dropping low-magnitude variables.
    """

    X = X.copy()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    first_run = None

    while True:

        # Apply Elastic Net regularization
        elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
        elastic_net.fit(X_scaled, y)

        # Calculate the absolute values of the coefficients
        abs_coefficients = np.abs(elastic_net.coef_)

        # Determine the threshold as a fraction of the largest coefficient
        max_coef = np.max(abs_coefficients)
        threshold = max_coef * threshold_fraction

        coefficients = elastic_net.coef_
        # align coefficients into a dataframe with variable names:
        coefficients = pd.DataFrame(
            {
                "variable": X.columns,
                "enr_coef": coefficients,
                "enr_coef_sign": np.sign(coefficients),
            }
        )
        coefficients = coefficients.sort_values(
            "enr_coef", ascending=False, key=lambda x: x.abs()
        )

        # identify worst variable:
        min_coef_idx = np.argmin(abs_coefficients)
        min_coef = abs_coefficients[min_coef_idx]

        if first_run is None:
            first_run = coefficients

        if min_coef < threshold:
            # remove the worst variable from X_scaled:
            X_scaled = np.delete(X_scaled, min_coef_idx, axis=1)
            # remove corresponding column from X:
            X = X.drop(X.columns[min_coef_idx], axis=1)
        else:
            break

    return {"initial": first_run, "final": coefficients}


def calc_r2(
    df: pd.DataFrame,
    variables: list[str],
    y: pd.Series
) -> pd.DataFrame:
    """
    Calculate R² and adjusted R² values for each variable.

    For each variable in the provided list, an OLS model is fit and the R², adjusted R²,
    and the sign of the coefficient are recorded.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the variables.
    variables : list[str]
        List of variable names to evaluate.
    y : pandas.Series
        Target variable series.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for variable, R², adjusted R², and coefficient sign.
    """
    results = {"variable": [], "r2": [], "adj_r2": [], "coef_sign": []}
    for var in variables:
        data = pd.concat([df[var], y], axis=1).dropna()
        if len(data) < 3 or data[var].nunique() < 2:
            results["variable"].append(var)
            results["r2"].append(float("nan"))
            results["adj_r2"].append(float("nan"))
            results["coef_sign"].append(float("nan"))
            continue                          # skip ill-posed models

        X = sm.add_constant(data[var].astype(float), has_constant='add')
        try:
            model = sm.OLS(y, X).fit()
        except MissingDataError as e:
            print(f"Error fitting model for variable {var}: {e}")
            for var in variables:
                # check for missing/null/nan values:
                if df[var].isna().any():
                    n = df[var].isna().sum()
                    print(f'Variable "{var}" has {n} missing values.')
            raise e

        results["variable"].append(var)
        results["r2"].append(model.rsquared)
        results["adj_r2"].append(model.rsquared_adj)
        results["coef_sign"].append(1 if model.params[var] >= 0 else -1)
    df_results = pd.DataFrame(data=results)
    return df_results


def calc_p_values_recursive_drop(
    X: pd.DataFrame,
    y: pd.Series,
    sig_threshold: float = 0.05
) -> dict:
    """
    Recursively drop variables with p-values above a specified significance threshold.

    Fits an OLS model on `X` and iteratively removes the variable with the highest
    p-value until all remaining variables have p-values below the threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Input features DataFrame.
    y : pd.Series
        Target variable series.
    sig_threshold : float, optional
        Significance threshold for p-values. Variables with p-values above this
        threshold will be dropped. Defaults to 0.05.

    Returns
    -------
    dict
        Dictionary with two keys:

        - "initial": pd.Series of p-values from the first OLS fit.
        - "final": pd.Series of p-values after recursively dropping high-p-value variables.

    Raises
    ------
    ValueError
        If the OLS regression fails or no variables remain.
    """

    X = X.copy()
    X = sm.add_constant(X, has_constant='add')
    X = X.astype(np.float64)
    model = sm.OLS(y, X).fit()
    first_run = None
    while True:
        max_p_value = model.pvalues.max()
        p_values = model.pvalues
        if first_run is None:
            first_run = p_values
        if max_p_value > sig_threshold:
            var_to_drop = p_values.idxmax()
            if var_to_drop == "const":
                # don't pick const, pick the next variable to drop:
                try:
                    var_to_drop = p_values.iloc[1:].idxmax()
                except ValueError as e:
                    break
            if pd.isna(var_to_drop):
                break
            X = X.drop(var_to_drop, axis=1)
            try:
                new_model = sm.OLS(y, X).fit()
            except ValueError as e:
                print(f"Error fitting model after dropping {var_to_drop}: {e}")
                break
            model = new_model
        else:
            break

    # align p_values into a dataframe with variable names:
    p_values = (
        pd.DataFrame({"p_value": model.pvalues})
        .sort_values("p_value", ascending=True)
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    # do the same for "first_run":
    first_run = (
        pd.DataFrame({"p_value": first_run})
        .sort_values("p_value", ascending=True)
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    return {
        "initial": first_run,
        "final": p_values,
    }


def calc_t_values_recursive_drop(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 2
) -> dict:
    """
    Recursively drop variables with t-values below a given threshold.

    Fits an OLS model on `X` and iteratively removes the variable with the smallest
    absolute t-value until all remaining variables have |t-value| above the threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features DataFrame.
    y : pandas.Series
        Target variable series.
    threshold : float, optional
        Minimum acceptable absolute t-value. Variables with |t-value| below this threshold
        will be dropped. Defaults to 2.

    Returns
    -------
    dict
        Dictionary with two keys:

        - "initial": pandas.Series of t-values and their signs from the first OLS fit.
        - "final": pandas.Series of t-values and their signs after recursive dropping.
    """

    X = X.copy()
    X = sm.add_constant(X, has_constant='add')
    X = X.astype(np.float64)

    first_run = None
    i = 0
    while True:
        i += 1
        try:
            t_values = calc_t_values(X, y)
        except ValueError as e:
            t_values = pd.Series([float("nan")] * X.shape[1], index=X.columns)
        if first_run is None:
            first_run = t_values

        if t_values.isna().all():
            min_t_var = float("nan")
        else:
            min_t_var = t_values.abs().idxmin()

        if pd.isna(min_t_var):
            min_t_var = 0

        if len(t_values) > 0:
            min_t_val = float("nan")
        else:
            min_t_val = t_values[min_t_var]

        if min_t_val < threshold:
            X = X.drop(min_t_var, axis=1)
        else:
            break

    # align t_values into a dataframe with variable names:
    t_values = (
        pd.DataFrame({"t_value": t_values, "t_value_sign": np.sign(t_values)})
        .sort_values("t_value", ascending=False, key=lambda x: x.abs())
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    # do the same for "first_run":
    first_run = (
        pd.DataFrame({"t_value": first_run, "t_value_sign": np.sign(first_run)})
        .sort_values("t_value", ascending=False, key=lambda x: x.abs())
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    return {"initial": first_run, "final": t_values}


def calc_t_values(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Calculate t-values for an OLS model.

    Fits an ordinary least squares regression of `y` on `X` and returns the t-values
    of the estimated coefficients.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features DataFrame (should include a constant term column).
    y : pandas.Series
        Target variable series.

    Returns
    -------
    pandas.Series
        Series of t-values corresponding to each coefficient in `X`.
    """
    linear_model = sm.OLS(y, X)
    fitted_model = linear_model.fit()
    return fitted_model.tvalues


def calc_vif_recursive_drop(
    X: pd.DataFrame,
    threshold: float = 10.0,
    settings: dict = None
) -> dict:
    """
    Recursively drop variables with a Variance Inflation Factor (VIF) exceeding the threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features DataFrame.
    threshold : float, optional
        Maximum acceptable VIF. Variables with VIF above this threshold will be removed.
        Defaults to 10.0.
    settings : dict, optional
        Settings dictionary containing field classifications, if needed for VIF computation.
        Defaults to None.

    Returns
    -------
    dict
        Dictionary with two keys:

        - "initial": pandas.DataFrame of VIF values before dropping variables.
        - "final": pandas.DataFrame of VIF values after recursively dropping high-VIF variables.

    Raises
    ------
    ValueError
        If no columns remain for VIF calculation.
    """
    X = X.copy()
    X = X.astype(np.float64)

    # Get boolean and categorical variables from settings if provided
    bool_fields = get_fields_boolean(settings, X)
    cat_fields = get_fields_categorical(settings, X, include_boolean=False)
    exclude_vars = bool_fields + cat_fields

    # Remove boolean and categorical variables
    X = X.drop(
        columns=[col for col in X.columns if col in exclude_vars], errors="ignore"
    )

    # Drop constant columns (VIF cannot be calculated for constant columns)
    X = X.loc[:, X.nunique() > 1]  # Keep only columns with more than one unique value

    # If no columns are left after removing constant columns or dropping NaN values, raise an error
    if X.shape[1] == 0:
        raise ValueError(
            "All columns are constant or have missing values; VIF cannot be computed."
        )
    first_run = None
    while True:
        vif_data = calc_vif(X)
        if first_run is None:
            first_run = vif_data
        if vif_data["vif"].max() > threshold:
            max_vif_idx = vif_data["vif"].idxmax()
            X = X.drop(X.columns[max_vif_idx], axis=1)
        else:
            break
    return {"initial": first_run, "final": vif_data}


def calc_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for each variable in a DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        - "variable": Name of each feature in `X`.
        - "VIF": Variance Inflation Factor value for that feature.
    """
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns

    if len(X.values) < 5:
        warnings.warn("Can't calculate VIF for less than 5 samples")
        vif_data["vif"] = [float("nan")] * len(X.columns)
        return vif_data
    
    if len(X.columns) == 1:
        warnings.warn("Can't calculate VIF for one column")
        vif_data["vif"] = [float("nan")] * len(X.columns)
        return vif_data
    
    # Calculate VIF for each column
    vif_data["vif"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    return vif_data


def calc_mse(
    prediction: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calculate the Mean Squared Error (MSE) between predictions and ground truth.

    Parameters
    ----------
    prediction : numpy.ndarray
        Array of predicted values.
    ground_truth : numpy.ndarray
        Array of true values.

    Returns
    -------
    float
        The MSE value.
    """
    return float(np.mean((prediction - ground_truth) ** 2))


def calc_mse_r2_adj_r2(
    predictions: np.ndarray, ground_truth: np.ndarray, num_vars: int
):
    """
    Calculate the Mean Squared Error (MSE), r-squared, and adjusted r-squared

    Parameters
    ----------
    predictions : numpy.ndarray
        Array of predicted values.
    ground_truth : numpy.ndarray
        Array of true values.
    num_vars : int
        Number of independent variables used to produce the predictions

    Returns
    -------
    tuple[float, float, float]

        A tuple containing three values:

        - The MSE value
        - The r-squared value
        - The adjusted r-squared value
    """

    mse = np.mean((ground_truth - predictions) ** 2)
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("inf")

    n = len(predictions)
    k = num_vars
    divisor = n - k - 1
    if divisor == 0:
        adj_r2 = float("inf")
    else:
        adj_r2 = 1 - ((1 - r2) * (n - 1) / divisor)
    return mse, r2, adj_r2


def calc_cross_validation_score(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray
) -> float:
    """
    Calculate cross-validation score using negative mean squared error.

    This function fits a LinearRegression model using 5-fold cross validation and
    returns the mean cross-validated MSE (positive value).

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Input features for modeling.
    y : array-like or pandas.Series
        Target variable.

    Returns
    -------
    float
        The mean cross-validated mean squared error.
    """
    model = LinearRegression()
    # Use negative MSE and negate it to return positive MSE
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    except ValueError:
        return float("nan")

    return -scores.mean()  # Convert negative MSE to positive


class ConfidenceStat:
    """
    Any statistic along with it's confidence interval upper and lower bounds, and whether it is statistically significant
    
    Attributes
    ----------
    value : float
        The base value of the statistic
    confidence_interval : float
        The % value of the confidence interval (e.g. 0.95 for 95% confidence interval)
    low : float
        The lower bound of the confidence interval
    high : float
        The upper bound of the confidence interval
    """
    def __init__(
        self,
        value : float,
        confidence_interval : float,
        low : float,
        high : float
    ):
        self.value = value
        self.confidence_interval = confidence_interval
        self.low = low
        self.high = high
    