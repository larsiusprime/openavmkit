import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm


def calc_chds(
		df_in: pd.DataFrame,
		field_cluster: str,
		field_value: str
):
	# create a dataframe matching the index of df_in to store the results but containing only the cluster_id
	df = df_in[[field_cluster]].copy()
	df["chd"] = 0.0

	clusters = df[field_cluster].unique()

	for cluster in clusters:
		df_cluster = df[df[field_cluster].eq(cluster)]
		chd = calc_cod(df_cluster[field_value].values)
		df.loc[df[field_cluster].eq(cluster), "chd"] = chd

	return df["chd"]


def quick_median_chd(df: pl.DataFrame, field_value: str, field_cluster: str) -> float:
	clusters = df.select(pl.col(field_cluster).unique()).to_numpy().flatten()
	chds = np.zeros(len(clusters))

	for i, cluster in enumerate(clusters):
		df_cluster = df.filter(pl.col(field_cluster) == cluster)
		chd = calc_cod(df_cluster[field_value].to_numpy())
		chds[i] = chd
		i += 1

	median_chd = float(np.median(chds))
	return median_chd


def calc_cod(values: np.ndarray) -> float:
	median_value = np.median(values)
	abs_delta_values = np.abs(values - median_value)
	sum_deltas = np.sum(abs_delta_values)
	avg_abs_deviation = sum_deltas / len(values)
	cod = avg_abs_deviation / median_value
	cod *= 100
	return cod


def calc_prd(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
	ratios = predictions / ground_truth
	mean_ratio = np.mean(ratios)
	weighted_mean_ratio = np.sum(predictions) / np.sum(ground_truth)
	prd = mean_ratio / weighted_mean_ratio
	return prd


def calc_prb(predictions: np.ndarray, ground_truth: np.ndarray) -> float:

	# TODO: this block is necessary because predictions is not guaranteed to have non-zero values
	predictions = predictions.copy()
	ground_truth = ground_truth.copy()

	zero_indices = np.where(predictions <= 0)
	predictions = np.delete(predictions, zero_indices)
	ground_truth = np.delete(ground_truth, zero_indices)

	predictions = predictions.astype(np.float64)

	ratios = predictions / ground_truth
	median_ratio = np.median(ratios)

	left_hand = (ratios - median_ratio) / median_ratio
	right_hand = np.log2(((predictions / median_ratio) + ground_truth))
	right_hand = sm.tools.tools.add_constant(right_hand)

	mra_model = sm.OLS(
		endog=left_hand,
		exog=right_hand
	).fit()
	prb = mra_model.params[0]

	return prb