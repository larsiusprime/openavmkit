import numpy as np
import pandas as pd


def generate_basic(
		size: int
):
	data = {
		"key": [],
		"bldg_area_finished_sqft": [],
		"land_area_sqft": [],
		"bldg_quality_num": [],
		"bldg_condition_num": [],
		"bldg_age_years": [],
		"land_value": [],
		"bldg_value": [],
		"total_value": [],
		"distance_from_cbd": [],
		"latitude": [],
		"longitude": []
	}

	latitude_center = 29.760762
	longitude_center = -95.361937

	height = 0.5
	width =  0.4

	nw_lat = latitude_center - width/2
	nw_lon = longitude_center - height/2

	base_land_value = 250
	base_bldg_value = 25
	quality_value = 5

	# set a random seed:
	np.random.seed(1337)

	for y in range(0, size):
		for x in range(0, size):

			_x = x/size
			_y = y/size

			latitude = nw_lat + (width * _x)
			longitude = nw_lon + (height * _y)

			dist_x = abs(_x - 0.5)
			dist_y = abs(_y - 0.5)
			dist_center = (dist_x**2 + dist_y**2)**0.5

			# base value with exponential falloff from center:
			_base_land_value = base_land_value - 1
			land_value_per_land_sqft = 1 + (_base_land_value * (1 - dist_center))

			key = f"{x}-{y}"
			land_area_sqft = np.random.randint(21780, 43560)
			bldg_area_finished_sqft = np.random.randint(1000, 2500)
			bldg_quality_num = np.clip(np.random.normal(3, 1), 0, 6)
			bldg_condition_num = np.clip(np.random.normal(3, 1), 0, 6)
			bldg_age_years = np.clip(np.random.normal(20, 10), 0, 100)
			land_value = land_area_sqft * land_value_per_land_sqft

			bldg_value_per_sqft = base_bldg_value + (quality_value * bldg_quality_num)

			depreciation_from_age = min(0.0, 1 - (bldg_age_years / 100))
			depreciation_from_condition = min(0.0, 1 - (bldg_condition_num / 6))

			total_depreciation = (depreciation_from_age + depreciation_from_condition)/2

			bldg_value_per_sqft = bldg_value_per_sqft * (1 - total_depreciation)
			bldg_value = bldg_area_finished_sqft * bldg_value_per_sqft

			total_value = land_value + bldg_value

			data["key"].append(str(key))
			data["bldg_area_finished_sqft"].append(bldg_area_finished_sqft)
			data["land_area_sqft"].append(land_area_sqft)
			data["bldg_quality_num"].append(bldg_quality_num)
			data["bldg_condition_num"].append(bldg_condition_num)
			data["bldg_age_years"].append(bldg_age_years)
			data["land_value"].append(land_value)
			data["bldg_value"].append(bldg_value)
			data["total_value"].append(total_value)
			data["distance_from_cbd"].append(dist_center)
			data["latitude"].append(latitude)
			data["longitude"].append(longitude)

	df = pd.DataFrame(data)
	return df