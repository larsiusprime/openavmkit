import polars as pl
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from IPython.core.display_functions import display
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')  # Set the interactive backend
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
	# limit df to rows where field_value is not null/nan/etc
	df = df.filter(df[field_value].is_not_null())

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
	if len(values) == 0:
		return float('nan')
	median_value = np.median(values)
	abs_delta_values = np.abs(values - median_value)
	avg_abs_deviation = np.sum(abs_delta_values) / len(values)
	cod = avg_abs_deviation / median_value
	cod *= 100
	return cod


def calc_cod_bootstrap(values: np.ndarray, confidence_interval=0.95, iterations=10000, seed=777) -> (float, float, float):
	n = len(values)
	if n == 0:
		return float('nan'), float('nan'), float('nan')
	np.random.seed(seed)
	median = np.median(values)
	samples = np.random.choice(values, size=(iterations, n), replace=True)
	abs_delta_values = np.abs(samples - median)
	bootstrap_cods = np.mean(abs_delta_values, axis=1) / median * 100
	alpha = (1.0 - confidence_interval) / 2
	lower_bound, upper_bound = np.quantile(bootstrap_cods, [alpha, 1.0 - alpha])
	median_cod = np.median(bootstrap_cods)
	return median_cod, lower_bound, upper_bound


def calc_prd(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
	ratios = predictions / ground_truth
	mean_ratio = np.mean(ratios)
	weighted_mean_ratio = np.sum(predictions) / np.sum(ground_truth)
	prd = mean_ratio / weighted_mean_ratio
	return prd


def calc_prd_bootstrap(predictions: np.ndarray, ground_truth: np.ndarray, confidence_interval=0.95, iterations=10000, seed=777) -> (float, float, float):
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
	if len(values) == 0:
		return values
	lower_bound = np.quantile(values, lower_quantile)
	upper_bound = np.quantile(values, upper_quantile)
	return values[(values >= lower_bound) & (values <= upper_bound)]


def calc_prb(predictions: np.ndarray, ground_truth: np.ndarray, confidence_interval: float = 0.95) -> (float, float, float):

	if len(predictions) != len(ground_truth):
		raise ValueError("predictions and ground_truth must have the same length")

	if predictions.size == 0 or ground_truth.size == 0:
		return float('nan')

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


def plot_correlation(corr: pd.DataFrame, title:str = "Correlation of Variables"):
	# Set a custom color map
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
	X = X.copy()

	first_run = None

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
		model = sm.OLS(y, X).fit()
		r2 = model.rsquared
		adj_r2 = model.rsquared_adj
		coef = model.params[var]
		coef_sign = 1 if coef >= 0 else -1
		results["variable"].append(var)
		results["r2"].append(r2)
		results["adj_r2"].append(adj_r2)
		results["coef_sign"].append(coef_sign)

	df_results = pd.DataFrame(data=results)
	return df_results


def calc_p_values_recursive_drop(X: pd.DataFrame, y: pd.Series, sig_threshold: float = 0.05):
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
			X = X.drop(var_to_drop, axis=1)
			model = sm.OLS(y, X).fit()
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
	X = X.copy()
	X = sm.add_constant(X)
	X = X.astype(np.float64)

	first_run = None
	i = 0
	while True:
		i += 1
		t_values = calc_t_values(X, y)
		if first_run is None:
			first_run = t_values
		min_t_var = t_values.abs().idxmin()
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
	linear_model = sm.OLS(y, X)
	fitted_model = linear_model.fit()
	return fitted_model.tvalues


def calc_vif_recursive_drop(X: pd.DataFrame, threshold: float = 10):
	X = X.copy()
	X = X.astype(np.float64)

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
		max_vif = vif_data["vif"].max()
		if max_vif > threshold:
			max_vif_idx = vif_data["vif"].idxmax()
			X = X.drop(X.columns[max_vif_idx], axis=1)
		else:
			break
	return {
		"initial": first_run,
		"final": vif_data
	}


def calc_vif(X: pd.DataFrame):
	# Create an empty dataframe for storing VIF values
	vif_data = pd.DataFrame()
	vif_data["variable"] = X.columns

	# Calculate VIF for each column
	vif_data["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

	return vif_data


def calc_cross_validation_score(X, y):
	model = LinearRegression()
	# Use negative MSE and negate it to return positive MSE
	scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
	return -scores.mean()  # Convert negative MSE to positive