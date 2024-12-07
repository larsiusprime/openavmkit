import numpy as np

import openavmkit.stats as stats


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
			cod = stats.calc_cod(ratios)
			prd = stats.calc_prd(predictions, ground_truth)
			prb = stats.calc_prb(predictions, ground_truth)

			self.median_ratio = median_ratio
			self.cod = cod
			self.prd = prd
			self.prb = prb



