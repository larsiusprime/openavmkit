import pandas as pd

from openavmkit.modeling import simple_ols, _greedy_nn_limited, simple_mra
from openavmkit.utilities.assertions import lists_are_equal
from openavmkit.utilities.stats import calc_vif
from openavmkit.modeling import DataSplit, run_mra

import warnings

def test_vif():
	
	data = {
		"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	}
	df = pd.DataFrame(data)
	
	data2 = {
		"a": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
		"b": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
		"c": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	}
	df2 = pd.DataFrame(data2)
	
	vif = calc_vif(df)
	vif2 = calc_vif(df2)


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
	order = _greedy_nn_limited(lats, lons, start_idx=0, k=16)
	assert list(order) == expected


def test_mra_constant():
    
    data = {
        "key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
        "key_sale": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
        "bldg_area_finished_sqft": [
            1000, 1000, 1000, 1000, 
            1500, 1500, 1500, 1500, 
            2000, 2000, 2000, 2000,
            2500, 2500, 2500, 2500,
            3000, 3000, 3000, 3000
        ],
        "land_area_sqft": [
            10000, 10000, 10000, 10000,
            20000, 20000, 20000, 20000,
            30000, 30000, 30000, 30000,
            40000, 40000, 40000, 40000,
            50000, 50000, 50000, 50000
        ],
        "bldg_type_A": [
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0
        ],
        "bldg_type_B": [
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1
        ],
        "location_A": [
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0
        ],
        "location_B": [
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1
        ],
        "flat": [
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 0  #putting a zero here will make this field a constant in the test set but not the train set, which is an edge case we need to guard against
        ],
        "model_group": [
            "a", "a", "a", "a",
            "a", "a", "a", "a",
            "a", "a", "a", "a",
            "a", "a", "a", "a",
            "a", "a", "a", "a"
        ]
    }
    
    df = pd.DataFrame(data)
    df["sale_price"] = df["bldg_area_finished_sqft"] * 20
    df["sale_price"] += df["bldg_type_A"] * df["bldg_area_finished_sqft"] * 2
    df["sale_price"] += df["bldg_type_B"] * df["bldg_area_finished_sqft"] * 5
    df["sale_price"] += df["location_A"] * df["land_area_sqft"] * 1
    df["sale_price"] += df["location_B"] * df["land_area_sqft"] * 2
    df["valid_sale"] = True
    df["vacant_sale"] = False
    df["sale_date"] = "2025-01-01"
    df["sale_date"] = pd.to_datetime(df["sale_date"], format="%Y-%m-%d")
    df["is_vacant"] = False
    df["valid_for_ratio_study"] = True
    df["sale_age_days"] = 0
    
    ind_vars = ["bldg_area_finished_sqft", "land_area_sqft", "bldg_type_A", "bldg_type_B", "location_A", "location_B", "flat"]
    df_sales = df.copy()
    df_universe = df[(["key","is_vacant"]+ind_vars)].copy()
    test_keys = ["0", "1", "2", "3", "4", "5"]
    train_keys = ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
    
    ds = DataSplit(
        df_sales,
        df_universe,
        "a",
        {},
        "sale_price",
        "sale_price",
        ind_vars,
        [],
        {},
        test_keys,
        train_keys
    )
    run_mra(ds, intercept=True)


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