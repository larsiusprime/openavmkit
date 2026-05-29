"""
Vertical equity analysis.

Implements :class:`VerticalEquityStudy`, which measures whether high-value
parcels and low-value parcels are valued with the same accuracy — the
"vertical equity" question. Computes PRD and PRB (with bootstrap confidence
intervals) plus per-quantile median ratios.

Together with :mod:`openavmkit.horizontal_equity_study` and
:mod:`openavmkit.ratio_study`, this module forms OpenAVMKit's IAAO-aligned
equity analysis suite.
"""
from typing import Dict
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from openavmkit.utilities.stats import ConfidenceStat, calc_ratio_stats_bootstrap, calc_prb

def get_vertical_equity_scores(df, sale_field: str, valuation_field: str) -> Dict[str, float]:
    df = df.copy()
    df["ratio"] = df[valuation_field] / df[sale_field]
    median_ratio = df["ratio"].median()
    df["market_proxy"] = (df[sale_field] * 0.5) + (df[valuation_field]/median_ratio)
    observation_count = len(df)
    if 20 <= observation_count <= 50:
        percentile_group_count = 2
    elif 51 <= observation_count <= 500:
        percentile_group_count = 4
    elif observation_count > 501:
        percentile_group_count = 10
    else:
        # VEI cannot be calculated for less than 20 observations
        return {
            "vei": np.nan,
            "vei_significance": np.nan,
            "group_stats": None
        }

    def ci_90_lower(x):
        if len(x) <= 1: return np.nan
        # 90% CI uses the 95th percentile for a two-tailed test
        return np.mean(x) - stats.sem(x) * stats.t.ppf(0.95, df=len(x) - 1)

    def ci_90_upper(x):
        if len(x) <= 1: return np.nan
        return np.mean(x) + stats.sem(x) * stats.t.ppf(0.95, df=len(x) - 1)

    #split the df into percentile_group_count groups based on the market proxy
    df["percentile_group"] = pd.qcut(df["market_proxy"], q=percentile_group_count, labels=False)
    grouped = df.groupby("percentile_group")
    group_stats = grouped['ratio'].agg(
        ratio='median',
        lower=ci_90_lower,
        upper=ci_90_upper
    )
    #VEI is 100 * (median of last percentile grop - median of first percentile group)/median_ratio
    vei_score = 100 * (group_stats[group_stats.index == (percentile_group_count - 1)].iloc[0]["ratio"] - group_stats[group_stats.index == 0].iloc[0]["ratio"]) / median_ratio
    # VEI significance = 100 * (Lower CI for Highest PG - Upper CI for Lowest PG)/median
    vei_significance = 100 * (group_stats[group_stats.index == (percentile_group_count -1)].iloc[0]["lower"] - group_stats[group_stats.index == 0].iloc[0]["upper"]) / median_ratio
    return {
        "vei": vei_score,
        "vei_significance": vei_significance,
        "group_stats": group_stats
    }


class VerticalEquityStudy:
    """
    Perform vertical equity analysis and summarize the results.

    Attributes
    ----------
    rows : int
        Total number of rows in the input DataFrame.
    confidence_interval : float
        The confidence interval (e.g. 0.95 for 95% confidence)
    prd : ConfidenceStat
        The price-related differential, with confidence intervals
    prb : ConfidenceStat
        The price-related bias, with confidence intervals
    quantiles : pd.DataFrame
        A dataframe containing the median ratio, with confidence intervals, of all ten price quantile tiers
    """
    
    def __init__(
        self,
        df_sales_in: pd.DataFrame,
        field_sales: str,
        field_prediction: str,
        field_location: str,
        confidence_interval : float = 0.95,
        iterations: int = 10000,
        seed : int = 777
    ):
        df_sales = df_sales_in.copy()
        
        n = len(df_sales)
        self.rows = n
        self.confidence_interval = confidence_interval
        
        # Calculate PRD and PRB
        #----------------------
        
        predictions = df_sales[field_prediction].to_numpy()
        sales = df_sales[field_sales].to_numpy()
        
        results = calc_ratio_stats_bootstrap(predictions, sales, confidence_interval, iterations=iterations, seed=seed)
        
        prb_point, prb_low, prb_high = calc_prb(predictions, sales, confidence_interval)
        
        self.prb = ConfidenceStat(prb_point, confidence_interval, prb_low, prb_high)
        
        # Calculate quantiles (directly from price)
        #------------------------------------------
        
        df_sales["quantile"] = _calc_quantiles(df_sales_in, field_sales)
        df = _assemble_quantile_df(df_sales, field_sales, field_prediction, confidence_interval, iterations, seed)
        self.quantiles = df

        # Calculate quantiles (grouped price)
        #------------------------------------------
        
        if field_location is not None and field_location in df_sales_in.columns:
            df_sales["quantile"] = _calc_grouped_quantiles(df_sales_in, field_sales, field_location)
            df = _assemble_quantile_df(df_sales, field_sales,  field_prediction, confidence_interval, iterations, seed)
            self.grouped_quantiles = df
        else:
            warnings.warn(f"VerticalEquityStudy: could not find location field, \"{field_location}\" in sales dataframe")
            self.grouped_quantiles = None
 
    
    def summary(self):
        conf = f"{self.confidence_interval*100:0.0f}"
        upper = f"{conf}% CI, upper"
        lower = f"{conf}% CI, lower"
        stat_sig = "Statistically Significant"
        data = {
            "Statistic": [],
            "Point value": [],
            upper: [],
            lower: [],
            stat_sig: [],
            "IAAO recommended": [],
            "IAAO passing": []
        }
        
        prd = self.prd
        prb = self.prb
        
        prd_stat_sig = not (prd.low <= 1.00 and prd.high >= 1.00)
        prb_stat_sig = not (prb.low <= 0.00 and prb.high >= 0.00)
        
        prd_iaao_ok = (prd.low <= 0.98 and prd.high >= 1.03)
        prb_iaao_ok = (prb.low <= -0.05 and prb.high >= 0.05)
        
        prb_iaao_pass = (prb.low >= -0.1 and prb.high <= 0.1)
        
        data["Statistic"].append("PRD")
        data["Point value"].append(prd.value)
        data[upper].append(prd.high)
        data[lower].append(prd.low)
        data[stat_sig].append(prd_stat_sig)
        data["IAAO recommended"].append(prd_iaao_ok)
        data["IAAO passing"].append(prd_iaao_ok)
        
        data["Statistic"].append("PRB")
        data["Point value"].append(prb.value)
        data[upper].append(prb.high)
        data[lower].append(prb.low)
        data[stat_sig].append(prb_stat_sig)
        data["IAAO recommended"].append(prb_iaao_ok)
        data["IAAO passing"].append(prb_iaao_pass)

        return pd.DataFrame(data=data)
    
    
    def plot_quantiles(
        self,
        ci_bounds:bool = False,
        ylim=None,
        grouped=False
    ):
        df = self.grouped_quantiles if grouped else self.quantiles
        if df is None:
            warnings.warn(f"No quantile data available to plot.")
            return
        conf = f"{self.confidence_interval*100:0.0f}"
        
        max_y = df['ratio_high'].max()
        min_y = df['ratio_low'].min()
        
        # Plot a line chart
        plt.close()
        plt.plot(df['quantile'], df['ratio'], marker='o', label="Median ratio")  # marker='o' shows each point
        if ci_bounds:
            plt.plot(df['quantile'], df['ratio_low'], marker='v', label=f"{conf}% CI, lower bound")
            plt.plot(df['quantile'], df['ratio_high'], marker='^', label=f"{conf}% CI, upper bound")
        plt.title("Vertical equity")
        plt.xlabel("Price tier")
        plt.ylabel("Assessed vs. Sale ratio")
        
        if ylim == "min":
            plt.ylim(min_y, max_y)
        elif ylim is not None and len(ylim) == 2:
            plt.ylim(ylim[0], ylim[1])
        else:
            plt.ylim(0.0, 2.0)
        
        plt.grid(True)  # optional: adds grid lines
        plt.show()
        

def _calc_quantiles(df: pd.DataFrame, field: str):
    if df[field].isna().all():
        return pd.Series([np.nan] * len(df), index=df.index)
    bins = [0]
    labels = []
    last_value = 0
    quantiles = 10
    for i in range(1,quantiles+1):
        try:
            q = i/quantiles
            quantile_value = np.quantile(df[field], q)
        except IndexError:
            continue
        percentile = f"{q * 100:0.0f}"
        bins.append(quantile_value)
        labels.append(percentile)
        last_value = quantile_value
    
    return pd.cut(df[field], bins=bins, labels=labels, include_lowest=True, duplicates="drop")


def _calc_grouped_quantiles(df_in: pd.DataFrame, value_field: str, group_field: str):
    df = df_in.copy()
    df["quantile"] = _calc_quantiles(df, value_field)
    df2 = df_in.merge(df_group_to_quantile, on=group_field, how="left")
    return df2["quantile"]


def _assemble_quantile_df(df_in: pd.DataFrame, field_sales: str, field_prediction: str, confidence_interval: float, iterations: int, seed: int):
    data = {
        "quantile":[],
        "ratio":[],
        "ratio_low":[],
        "ratio_high":[]
    }
    labels = df_in["quantile"].unique()
    for label in labels:
        df_sub = df_in[df_in["quantile"].eq(label)]
        predictions = df_sub[field_prediction]
        sales = df_sub[field_sales]
        
        if len(predictions) > 0 and len(sales) > 0:
            results = calc_ratio_stats_bootstrap(predictions, sales, confidence_interval, iterations=iterations, seed=seed)
            med_ratio = results["median_ratio"]

            ratio = med_ratio.value
            low = med_ratio.low
            high = med_ratio.high

            data["ratio"].append(ratio)
            data["ratio_low"].append(low)
            data["ratio_high"].append(high)
            data["quantile"].append(label)
        # else:
            # data["ratio"].append(None)
            # data["ratio_low"].append(None)
            # data["ratio_high"].append(None)
            # data["quantile"].append(label)
    
    df = pd.DataFrame(data=data)
    df = df.sort_values(by="quantile", key=lambda col: col.astype(int))
    return df