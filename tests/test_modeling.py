import pandas as pd

from openavmkit.modeling import simple_ols

def test_simple_ols():

	data = {
		"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		"b": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
	}
	df = pd.DataFrame(data)

	results = simple_ols(df, "a", "b")

	assert results["slope"] - 2.0 < 1e-6
	assert results["intercept"] - 4.0 < 1e-6
	assert results["r2"] - 1.0 < 1e-6
	assert results["adj_r2"] - 1.0 < 1e-6