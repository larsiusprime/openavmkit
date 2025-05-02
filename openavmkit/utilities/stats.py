import warnings

import polars as pl
import statsmodels
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import MissingDataError


def calc_chds(df_in: pd.DataFrame, field_cluster: str, field_value: str):
	"""
  Calculate the Coefficient of Horizontal Dispersion (CHD) for each cluster in a DataFrame. CHD is the same statistic
  as COD, the Coefficient of Dispersion, but calculated for horizontal equity clusters and used to measure horizontal
  dispersion, on the theory that similar properties in similar locations should have similar valuations. The use of the
  name "CHD" is chosen to avoid confusion because assessors strongly associate "COD" with sales ratio studies.

  This function computes the CHD for each unique cluster in the input DataFrame based on the values in the specified
  field.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param field_cluster: Name of the column representing cluster identifiers.
  :type field_cluster: str
  :param field_value: Name of the column containing the values for COD calculation.
  :type field_value: str
  :returns: A pandas Series of COD values for each row, aligned with df_in.
  :rtype: pandas.Series
  """
	# Create a dataframe matching the index of df_in with only the cluster id.
	df = df_in[[field_cluster]].copy()
	df["chd"] = 0.0

	clusters = df[field_cluster].unique()

	for cluster in clusters:
		df_cluster = df[df[field_cluster].eq(cluster)]
		# exclude negative and null values:
		df_cluster = df_cluster[~pd.isna(df_cluster[field_value]) & df_cluster[field_value].gt(0)]

		chd = calc_cod(df_cluster[field_value].values)
		df.loc[df[field_cluster].eq(cluster), "chd"] = chd

	return df["chd"]


def quick_median_chd_pl(df: pl.DataFrame, field_value: str, field_cluster: str) -> float:
	"""
  Calculate the median CHD for groups in a Polars DataFrame.

  This function filters out missing values for the given field, groups the data by the
  specified cluster field, computes COD for each group, and returns the median COD value.

  :param df: Input Polars DataFrame.
  :type df: polars.DataFrame
  :param field_value: Name of the field containing values for COD calculation.
  :type field_value: str
  :param field_cluster: Name of the field to group by.
  :type field_cluster: str
  :returns: The median COD value across all groups.
  :rtype: float
  """
	# Filter out rows with missing values for field_value.
	df = df.filter(~pd.isna(df[field_value]))
	df = df.filter(df[field_value].gt(0))

	chds = (
		df
		.group_by(field_cluster)
		.agg(pl.col(field_value).alias("values"))
	)

	# Apply the calc_cod function to each group (the list of values)
	chd_values = np.array([calc_cod(group.to_numpy()) for group in chds["values"]])

	# Calculate the median of the CHD values
	median_chd = float(np.median(chd_values))
	return median_chd


def calc_cod(values: np.ndarray) -> float:
	"""
  Calculate the Coefficient of Dispersion (COD) for an array of values.

  COD is defined as the average absolute deviation from the median, divided by the median,
  multiplied by 100. Special cases are handled if the median is zero.

  :param values: Array of numeric values.
  :type values: numpy.ndarray
  :returns: The COD percentage.
  :rtype: float
  """
	if len(values) == 0:
		return float('nan')

	median_value = np.median(values)
	abs_delta_values = np.abs(values - median_value)
	avg_abs_deviation = np.sum(abs_delta_values) / len(values)
	if median_value == 0:
		# if every value is zero, the COD is zero:
		if np.all(values == 0):
			return 0.0
		else:
			# if the median is zero but not all values are zero, return infinity
			return float('inf')
	cod = avg_abs_deviation / median_value
	cod *= 100
	return cod


def calc_cod_bootstrap(values: np.ndarray, confidence_interval=0.95, iterations=10000, seed=777) -> (float, float, float):
	"""
  Calculate COD using bootstrapping.

  This function bootstraps the input values (which means resampling with replacement) to generate a distribution of
  CODs, then returns the median COD along with the lower and upper bounds of the confidence interval.

  :param values: Array of numeric values.
  :type values: numpy.ndarray
  :param confidence_interval: The desired confidence level (default is 0.95).
  :type confidence_interval: float, optional
  :param iterations: Number of bootstrap iterations (default is 10000).
  :type iterations: int, optional
  :param seed: Random seed for reproducibility (default is 777).
  :type seed: int, optional
  :returns: A tuple containing the median COD, lower bound, and upper bound.
  :rtype: tuple(float, float, float)
  """
	n = len(values)
	if n == 0:
		return float('nan'), float('nan'), float('nan')
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

  PRD is computed as the ratio of the mean ratio to the weighted mean ratio of predictions to ground truth.

  :param predictions: Array of predicted values.
  :type predictions: numpy.ndarray
  :param ground_truth: Array of ground truth values.
  :type ground_truth: numpy.ndarray
  :returns: The PRD value.
  :rtype: float
  """
	ratios = predictions / ground_truth
	mean_ratio = np.mean(ratios)
	sum_ground_truth = np.sum(ground_truth)
	if sum_ground_truth == 0:
		return float('inf')
	weighted_mean_ratio = np.sum(predictions) / sum_ground_truth
	if weighted_mean_ratio == 0:
		return float('inf')
	prd = mean_ratio / weighted_mean_ratio
	return prd


def calc_prd_bootstrap(predictions: np.ndarray, ground_truth: np.ndarray, confidence_interval=0.95, iterations=10000, seed=777) -> (float, float, float):
	"""
  Calculate PRD with bootstrapping.

  This function bootstraps the prediction-to-ground_truth ratios to produce a distribution
  of PRD values and returns the lower bound, median, and upper bound of the confidence interval.

  :param predictions: Array of predicted values.
  :type predictions: numpy.ndarray
  :param ground_truth: Array of ground truth values.
  :type ground_truth: numpy.ndarray
  :param confidence_interval: The desired confidence level (default is 0.95).
  :type confidence_interval: float, optional
  :param iterations: Number of bootstrap iterations (default is 10000).
  :type iterations: int, optional
  :param seed: Random seed for reproducibility (default is 777).
  :type seed: int, optional
  :returns: A tuple containing the lower bound, median PRD, and upper bound.
  :rtype: tuple(float, float, float)
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
	return lower_bound, median_prd, upper_bound


def trim_outliers(values: np.ndarray, lower_quantile: float = 0.25, upper_quantile: float = 0.75) -> np.ndarray:
	"""
  Trim outliers from an array of values based on quantile thresholds.

  :param values: Input array of numeric values.
  :type values: numpy.ndarray
  :param lower_quantile: Lower quantile bound (default is 0.25).
  :type lower_quantile: float, optional
  :param upper_quantile: Upper quantile bound (default is 0.75).
  :type upper_quantile: float, optional
  :returns: Array with values outside the quantile bounds removed.
  :rtype: numpy.ndarray
  """
	if len(values) == 0:
		return values
	lower_bound = np.quantile(values, lower_quantile)
	upper_bound = np.quantile(values, upper_quantile)
	return values[(values >= lower_bound) & (values <= upper_bound)]


def calc_prb(predictions: np.ndarray, ground_truth: np.ndarray, confidence_interval: float = 0.95) -> (float, float, float):
	"""
  Calculate the PRB (Price Related Bias) metric using a regression-based approach.

  This function fits an OLS model on the transformed ratios of predictions to ground truth,
  then returns the PRB value along with its lower and upper confidence bounds.

  :param predictions: Array of predicted values.
  :type predictions: numpy.ndarray
  :param ground_truth: Array of ground truth values.
  :type ground_truth: numpy.ndarray
  :param confidence_interval: Desired confidence interval (default is 0.95).
  :type confidence_interval: float, optional
  :returns: A tuple containing the PRB, its lower bound, and its upper bound.
  :rtype: tuple(float, float, float)
  :raises ValueError: If predictions and ground_truth lengths differ.
  """
	if len(predictions) != len(ground_truth):
		raise ValueError("predictions and ground_truth must have the same length")

	if predictions.size == 0 or ground_truth.size == 0:
		return float('nan'), float('nan'), float('nan')

	# TODO: this block is necessary because predictions is not guaranteed to have non-zero values
	predictions = predictions.copy()
	ground_truth = ground_truth.copy()

	na_indices = np.where(pd.isna(predictions))
	predictions = np.delete(predictions, na_indices)
	ground_truth = np.delete(ground_truth, na_indices)

	zero_indices = np.where(predictions <= 0)
	predictions = np.delete(predictions, zero_indices)
	ground_truth = np.delete(ground_truth, zero_indices)

	predictions = predictions.astype(np.float64)

	ratios = predictions / ground_truth
	median_ratio = np.median(ratios)

	try:
		left_hand = (ratios - median_ratio) / median_ratio
		right_hand = np.log2(((predictions / median_ratio) + ground_truth))
		right_hand = sm.tools.tools.add_constant(right_hand)
	except ValueError:
		return float('nan'), float('nan'), float('nan')

	mra_model = sm.OLS(
		endog=left_hand,
		exog=right_hand
	).fit()
	prb = mra_model.params[0]

	# get confidence interval from MRA model:
	conf_int = mra_model.conf_int(alpha=1.0-confidence_interval, cols=None)
	try:
		prb_lower = conf_int[0, 0]  # Lower bound for the first parameter
		prb_upper = conf_int[0, 1]  # Upper bound for the first parameter
	except IndexError:
		prb_lower = float('nan')
		prb_upper = float('nan')

	return prb, prb_lower, prb_upper




def plot_correlation(corr: pd.DataFrame, title: str = "Correlation of Variables"):
	"""
  Plot a heatmap of a correlation matrix.

  :param corr: Correlation matrix as a DataFrame.
  :type corr: pandas.DataFrame
  :param title: Title of the plot (default is "Correlation of Variables").
  :type title: str, optional
  :returns: None
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
		yticklabels=corr.index.tolist(),    # explicitly set the yticklabels
		annot_kws={"size": 8},      # adjust font size if needed
	)

	plt.title(title)
	plt.xticks(rotation=45, ha='right')  # rotate x labels if needed
	plt.yticks(rotation=0)  # keep y labels horizontal
	plt.tight_layout(pad=2)
	plt.show()


def calc_correlations(X: pd.DataFrame, threshold: float = 0.1, do_plots: bool = False):
	"""
  Calculate correlations and iteratively drop variables with low combined scores.

  The function computes the correlation matrix of X, then calculates a combined score
  based on the strength of the correlation with the target variable and the clarity (average
  correlation with other variables). Variables with a score below the threshold are dropped.

  :param X: Input DataFrame with variables.
  :type X: pandas.DataFrame
  :param threshold: Minimum acceptable combined score for variables (default is 0.1).
  :type threshold: float, optional
  :param do_plots: If True, plots initial and final correlation heatmaps.
  :type do_plots: bool, optional
  :returns: A dictionary with keys "initial" (first run scores) and "final" (final score DataFrame).
  :rtype: dict
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

		# Calculate the strength of the correlation with the target variable
		strength = naive_corr.iloc[:, 0].abs()

		# drop the target variable from strength:
		strength = strength.iloc[1:]

		# Calculate the clarity of the correlation: how correlated it is with all other variables *except* the target variable
		clarity = 1 - ((naive_corr_sans_target.abs().sum(axis=1) - 1.0) / (len(naive_corr_sans_target.columns)-1))

		# Combine the strength and clarity into a single score -- bigger is better, and we want high strength and high clarity
		score = strength * clarity * clarity

		# Identify the variable with the lowest score
		min_score_idx = score.idxmin()

		if pd.isna(min_score_idx):
			min_score = score[0]
		else:
			min_score = score[min_score_idx]

		data = {
			"corr_strength": strength,
			"corr_clarity": clarity,
			"corr_score": score
		}
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

	return {
		"initial": first_run,
		"final": df_score
	}


def calc_elastic_net_regularization(X: pd.DataFrame, y: pd.Series, threshold_fraction: float = 0.05):
	"""
  Calculate Elastic Net regularization coefficients while iteratively dropping variables with low coefficients.

  The function standardizes X, fits an Elastic Net model, and iteratively removes variables
  whose absolute coefficients are below a fraction of the maximum coefficient.

  :param X: Input features DataFrame.
  :type X: pandas.DataFrame
  :param y: Target variable series.
  :type y: pandas.Series
  :param threshold_fraction: Fraction of the maximum coefficient below which variables are dropped (default is 0.05).
  :type threshold_fraction: float, optional
  :returns: A dictionary with keys "initial" (first run coefficients) and "final" (final coefficients DataFrame).
  :rtype: dict
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
		coefficients = pd.DataFrame({
			"variable": X.columns,
			"enr_coef": coefficients,
			"enr_coef_sign": np.sign(coefficients)
		})
		coefficients = coefficients.sort_values("enr_coef", ascending=False, key=lambda x: x.abs())

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

	return {
		"initial": first_run,
		"final": coefficients
	}


def calc_r2(df: pd.DataFrame, variables: list[str], y: pd.Series):
	"""
  Calculate R² and adjusted R² values for each variable.

  For each variable in the provided list, an OLS model is fit and the R²,
  adjusted R², and the sign of the coefficient are recorded.

  :param df: DataFrame containing the variables.
  :type df: pandas.DataFrame
  :param variables: List of variable names to evaluate.
  :type variables: list[str]
  :param y: Target variable series.
  :type y: pandas.Series
  :returns: A DataFrame with columns for variable, R², adjusted R², and coefficient sign.
  :rtype: pandas.DataFrame
  """
	results = {
		"variable": [],
		"r2": [],
		"adj_r2": [],
		"coef_sign": []
	}
	for var in variables:
		X = df[var].copy()
		X = sm.add_constant(X)
		X = X.astype(np.float64)
		try:
			model = sm.OLS(y, X).fit()
		except MissingDataError as e:
			print(f"Error fitting model for variable {var}: {e}")
			for var in variables:
				# check for missing/null/nan values:
				if df[var].isna().any():
					n = df[var].isna().sum()
					print(f"Variable \"{var}\" has {n} missing values.")
			raise e

		results["variable"].append(var)
		results["r2"].append(model.rsquared)
		results["adj_r2"].append(model.rsquared_adj)
		results["coef_sign"].append(1 if model.params[var] >= 0 else -1)
	df_results = pd.DataFrame(data=results)
	return df_results


def calc_p_values_recursive_drop(X: pd.DataFrame, y: pd.Series, sig_threshold: float = 0.05):
	"""
  Recursively drop variables with p-values above a specified significance threshold.

  Fits an OLS model on X and drops the variable with the highest p-value iteratively
  until all p-values are below the threshold.

  :param X: Input features DataFrame.
  :type X: pandas.DataFrame
  :param y: Target variable series.
  :type y: pandas.Series
  :param sig_threshold: Significance threshold for p-values (default is 0.05).
  :type sig_threshold: float, optional
  :returns: A dictionary with keys "initial" and "final" containing DataFrames of p-values.
  :rtype: dict
  """
	X = X.copy()
	X = sm.add_constant(X)
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
	p_values = pd.DataFrame({
		"p_value": model.pvalues
	}).sort_values("p_value", ascending=True).reset_index().rename(columns={"index": "variable"})

	# do the same for "first_run":
	first_run = pd.DataFrame({
		"p_value": first_run
	}).sort_values("p_value", ascending=True).reset_index().rename(columns={"index": "variable"})

	return {
		"initial": first_run,
		"final": p_values,
	}


def calc_t_values_recursive_drop(X: pd.DataFrame, y: pd.Series, threshold: float = 2):
	"""
  Recursively drop variables with t-values below a given threshold.

  :param X: Input features DataFrame.
  :type X: pandas.DataFrame
  :param y: Target variable series.
  :type y: pandas.Series
  :param threshold: Minimum acceptable t-value (default is 2).
  :type threshold: float, optional
  :returns: A dictionary with keys "initial" and "final" containing DataFrames of t-values and their signs.
  :rtype: dict
  """
	X = X.copy()
	X = sm.add_constant(X)
	X = X.astype(np.float64)

	first_run = None
	i = 0
	while True:
		i += 1
		try:
			t_values = calc_t_values(X, y)
		except ValueError as e:
			t_values = pd.Series([float('nan')] * X.shape[1], index=X.columns)
		if first_run is None:
			first_run = t_values

		if t_values.isna().all():
			min_t_var = float('nan')
		else:
			min_t_var = t_values.abs().idxmin()

		if pd.isna(min_t_var):
			min_t_var = 0

		if len(t_values) > 0:
			min_t_val = float('nan')
		else:
			min_t_val = t_values[min_t_var]

		if min_t_val < threshold:
			X = X.drop(min_t_var, axis=1)
		else:
			break

	# align t_values into a dataframe with variable names:
	t_values = pd.DataFrame({
		"t_value": t_values,
		"t_value_sign": np.sign(t_values)
	}).sort_values("t_value", ascending=False, key=lambda x: x.abs()).reset_index().rename(columns={"index": "variable"})

	# do the same for "first_run":
	first_run = pd.DataFrame({
		"t_value": first_run,
		"t_value_sign": np.sign(first_run)
	}).sort_values("t_value", ascending=False, key=lambda x: x.abs()).reset_index().rename(columns={"index": "variable"})

	return {
		"initial": first_run,
		"final": t_values
	}


def calc_t_values(X: pd.DataFrame, y: pd.Series):
	"""
  Calculate t-values for an OLS model.

  :param X: Input features DataFrame (should include constant term).
  :type X: pandas.DataFrame
  :param y: Target variable series.
  :type y: pandas.Series
  :returns: A pandas Series of t-values.
  :rtype: pandas.Series
  """
	linear_model = sm.OLS(y, X)
	fitted_model = linear_model.fit()
	return fitted_model.tvalues


def calc_vif_recursive_drop(X: pd.DataFrame, threshold: float = 10, settings: dict = None):
	"""
  Recursively drop variables with a Variance Inflation Factor (VIF) exceeding the threshold.

  :param X: Input features DataFrame.
  :type X: pandas.DataFrame
  :param threshold: Maximum acceptable VIF (default is 10).
  :type threshold: float, optional
  :param settings: Settings dictionary containing field classifications.
  :type settings: dict, optional
  :returns: A dictionary with keys "initial" and "final" containing VIF DataFrames.
  :rtype: dict
  :raises ValueError: If no columns remain for VIF calculation.
  """
	X = X.copy()
	X = X.astype(np.float64)

	# Get boolean and categorical variables from settings if provided
	exclude_vars = []
	if settings is not None:
		field_classification = settings.get("field_classification", {})
		for category in ["land", "impr", "other"]:
			if category in field_classification:
				exclude_vars.extend(field_classification[category].get("boolean", []))
				exclude_vars.extend(field_classification[category].get("categorical", []))
				exclude_vars.extend(field_classification[category].get("boolean_na_false", []))
				exclude_vars.extend(field_classification[category].get("boolean_na_true", []))

	# Remove boolean and categorical variables
	X = X.drop(columns=[col for col in X.columns if col in exclude_vars], errors='ignore')

	# Drop constant columns (VIF cannot be calculated for constant columns)
	X = X.loc[:, X.nunique() > 1]  # Keep only columns with more than one unique value

	# If no columns are left after removing constant columns or dropping NaN values, raise an error
	if X.shape[1] == 0:
		raise ValueError("All columns are constant or have missing values; VIF cannot be computed.")
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
	return {
		"initial": first_run,
		"final": vif_data
	}


def calc_vif(X: pd.DataFrame):
	"""
  Calculate the Variance Inflation Factor (VIF) for each variable in a DataFrame.

  :param X: Input features DataFrame.
  :type X: pandas.DataFrame
  :returns: A DataFrame with variables and their VIF values.
  :rtype: pandas.DataFrame
  """
	vif_data = pd.DataFrame()
	vif_data["variable"] = X.columns

	if len(X.values) < 5:
		warnings.warn("Can't calculate VIF for less than 5 samples")
		vif_data["vif"] = [float('nan')] * len(X.columns)
		return vif_data

	# Calculate VIF for each column
	vif_data["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

	return vif_data


def calc_mse(prediction: np.ndarray, ground_truth: np.ndarray):
	"""
  Calculate the Mean Squared Error (MSE) between predictions and ground truth.

  :param prediction: Array of predicted values.
  :type prediction: numpy.ndarray
  :param ground_truth: Array of true values.
  :type ground_truth: numpy.ndarray
  :returns: The MSE value.
  :rtype: float
  """
	return np.mean((prediction - ground_truth) ** 2)


def calc_mse_r2_adj_r2(predictions: np.ndarray, ground_truth: np.ndarray, num_vars: int):
	#mse = calc_mse(predictions, ground_truth)

	mse = np.mean((ground_truth - predictions) ** 2)
	ss_res = np.sum((ground_truth - predictions) ** 2)
	ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)

	r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('inf')

	n = len(predictions)
	k = num_vars
	divisor = n - k - 1
	if divisor == 0:
		adj_r2 = float('inf')
	else:
		adj_r2 = 1 - ((1 - r2) * (n - 1) / divisor)
	return mse, r2, adj_r2


def calc_cross_validation_score(X, y):
	"""
  Calculate cross validation score using negative mean squared error.

  This function fits a LinearRegression model using 5-fold cross validation and returns the positive MSE.

  :param X: Input features.
  :type X: array-like or pandas.DataFrame
  :param y: Target variable.
  :type y: array-like or pandas.Series
  :returns: The mean cross validation MSE.
  :rtype: float
  """
	model = LinearRegression()
	# Use negative MSE and negate it to return positive MSE
	try:
		scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
	except ValueError:
		return float('nan')

	return -scores.mean()  # Convert negative MSE to positive