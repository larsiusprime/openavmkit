import numpy as np

import openavmkit.utilities.stats as stats


class RatioStudy:
		predictions: np.ndarray
		ground_truth: np.ndarray
		count: int
		median_ratio: float
		cod: float
		cod_trim: float
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

			# trim the ratios to remove outliers -- trim to the interquartile range
			trim_ratios = stats.trim_outliers(ratios)

			cod = stats.calc_cod(ratios)
			cod_trim = stats.calc_cod(trim_ratios)
			prd = stats.calc_prd(predictions, ground_truth)
			prb = stats.calc_prb(predictions, ground_truth)

			self.median_ratio = median_ratio
			self.cod = cod
			self.cod_trim = cod_trim
			self.prd = prd
			self.prb = prb



