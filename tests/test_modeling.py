import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from openavmkit.modeling import run_mra, run_xgboost, run_lightgbm, run_catboost, run_gwr
from openavmkit.synthetic_data import generate_basic


def test_gwr():

	# df = generate_basic(10)
	#
	# u = df['longitude']
	# v = df['latitude']
	# coords = list(zip(u,v))
	#
	# y = df['total_value'].values.reshape((-1, 1))
	# print(f"y.shape = {y.shape}")
	#
	# X = df[['distance_from_cbd']].values
	# print(f"X.shape = {X.shape}")
	#
	# gwr_selector = Sel_BW(coords, y, X)
	# gwr_bw = gwr_selector.search()
	# print(f"GWR Bandwidth: {gwr_bw}")
	# gwr = GWR(coords, y, X, gwr_bw)
	# gwr_results = gwr.fit()
	#
	# display(gwr_results.summary())
	#
	# np_coords = np.array(coords)
	#
	# print(f"coords.shape = {np_coords.shape}")
	#
	# gwr_results = gwr.predict(
	# 	np_coords, X
	# )
	#
	# y_pred = gwr_results.predictions
	#
	# print("")
	# print(f"y_pred.shape = {y_pred.shape}")
	# print(f"y_pred = {y_pred}")

	return True


def test_models():
	print("")
	df = generate_basic(10)
	ind_var = "total_value"
	dep_vars = [
		"bldg_area_finished_sqft",
		"land_area_sqft",
		"bldg_quality_num",
		"bldg_condition_num",
		"bldg_age_years",
		"distance_from_cbd"
	]

	models = ["mra", "gwr", "xgboost", "lightgbm", "catboost"]

	results = None
	for model in models:
		if model == "mra":
			results = run_mra(df, ind_var, dep_vars)
		elif model == "gwr":
			results = run_gwr(df, ind_var, dep_vars)
		elif model == "xgboost":
			results = run_xgboost(df, ind_var, dep_vars)
		elif model == "lightgbm":
			results = run_lightgbm(df, ind_var, dep_vars)
		elif model == "catboost":
			results = run_catboost(df, ind_var, dep_vars)
		if results is not None:
			display(results.summary())

	return True