import numpy as np
import statsmodels.api as sm


def calculate_cod(ratios: np.ndarray) -> float:
	median_ratio = np.median(ratios)
	abs_delta_ratios = np.abs(ratios - median_ratio)
	sum_deltas = np.sum(abs_delta_ratios)
	avg_abs_deviation = sum_deltas / len(ratios)
	cod = avg_abs_deviation / median_ratio
	cod *= 100
	return cod


def calculate_prd(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
	ratios = predictions / ground_truth
	mean_ratio = np.mean(ratios)
	weighted_mean_ratio = np.sum(predictions) / np.sum(ground_truth)
	prd = mean_ratio / weighted_mean_ratio
	return prd


def calculate_prb(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
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


class RatioStudy:
		predictions: np.ndarray
		ground_truth: np.ndarray
		count: int
		median_ratio: float
		cod: float
		prd: float
		prb: float

		def __init__(
				self,
				predictions: np.ndarray,
				ground_truth: np.ndarray
		):

			if len(predictions) != len(ground_truth):
				raise ValueError("predictions and ground_truth must have the same length")

			self.count = len(predictions)
			self.predictions = predictions
			self.ground_truth = ground_truth

			ratios = predictions / ground_truth
			median_ratio = float(np.median(ratios))
			cod = calculate_cod(ratios)
			prd = calculate_prd(predictions, ground_truth)
			prb = calculate_prb(predictions, ground_truth)

			self.median_ratio = median_ratio
			self.cod = cod
			self.prd = prd
			self.prb = prb



