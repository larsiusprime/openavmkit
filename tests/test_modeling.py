import pandas as pd

from openavmkit.modeling import simple_ols, greedy_nn_limited, simple_mra
from openavmkit.utilities.assertions import lists_are_equal


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


def test_nearest_neighbor():

	def make_snake(lat_base=29.749907, lon_base=-95.358421,
			lat_size=0.001, lon_size=0.010,
			n_cols=100, n_rows=10):

		lats, lons, expected = [], [], []
		for col in range(n_cols):
			lon = lon_base + col * lon_size

			# even columns go bottom→top, odd go top→bottom
			if col % 2 == 0:
				rows = range(n_rows)
			else:
				rows = range(n_rows - 1, -1, -1)

			for row in rows:
				lat = lat_base + (row + 1) * lat_size
				lats.append(lat)
				lons.append(lon)
				expected.append(len(expected))

		return lats, lons, expected

	# in your test
	lats, lons, expected = make_snake()
	order = greedy_nn_limited(lats, lons, start_idx=0, k=16)
	assert list(order) == expected


def test_simple_mra():

	data = {
		"key": [0, 1, 2, 3, 4, 5],
		"bldg_area_finished_sqft": [1000, 1500, 2000, 2500, 3000, 3500],
		"land_area_sqft": [10000, 20000, 30000, 40000, 50000, 60000],
		"bldg_type_A": [1, 0, 0, 1, 0, 1],
		"bldg_type_B": [0, 1, 1, 0, 1, 0],
		"location_A": [1, 0, 0, 1, 0, 1],
		"location_B": [0, 1, 1, 0, 1, 0],
	}
	df = pd.DataFrame(data)
	df["location_A"] = df["location_A"] * df["land_area_sqft"]
	df["location_B"] = df["location_B"] * df["land_area_sqft"]
	df["bldg_type_A"] = df["bldg_type_A"] * df["bldg_area_finished_sqft"]
	df["bldg_type_B"] = df["bldg_type_B"] * df["bldg_area_finished_sqft"]

	true_coefs = {
		"bldg_type_A": 10.0,
		"bldg_type_B": 25.0,
		"location_A": 5,
		"location_B": 1
	}

	df["sale_price"] = 0.0

	for coef in true_coefs:
		df["sale_price"] += df[coef] * true_coefs[coef]

	results = simple_mra(df, ["bldg_type_A", "bldg_type_B", "location_A", "location_B"], "sale_price")
	coefs = results["coefs"]

	print(coefs)

	for coef in coefs:
		true_value = true_coefs[coef]
		mra_value = coefs[coef]
		assert abs(true_value - mra_value) < 1e-2, f"Coefficient for {coef} does not match: expected {true_value}, got {mra_value}"