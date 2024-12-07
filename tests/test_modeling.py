from IPython.core.display_functions import display
from openavmkit.modeling import run_mra, run_xgboost, run_lightgbm, run_catboost, run_gwr
from openavmkit.synthetic_data import generate_basic

def test_models():
	print("")
	df = generate_basic(100)
	ind_var = "sale_price"
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