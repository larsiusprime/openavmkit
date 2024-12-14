import numpy as np
import pandas as pd


def generate_basic(
		size: int,
		percent_sales: float = 0.1,
		noise_sales: float = 0.05,
		seed: int = 1337
):
	data = {
		"key": [],
		"neighborhood": [],
		"bldg_area_finished_sqft": [],
		"land_area_sqft": [],
		"bldg_type": [],
		"bldg_quality_num": [],
		"bldg_condition_num": [],
		"bldg_age_years": [],
		"land_value": [],
		"bldg_value": [],
		"total_value": [],
		"dist_to_cbd": [],
		"valid_sale": [],
		"sale_price": [],
		"latitude": [],
		"longitude": [],
		"sale_age_days": []
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
	np.random.seed(seed)

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

			bldg_type = np.random.choice(["A", "B", "C"])

			bldg_type_mult = 1.0
			if bldg_type == "A":
				bldg_type_mult = 0.5
			elif bldg_type == "B":
				bldg_type_mult = 1.0
			elif bldg_type == "C":
				bldg_type_mult = 2.0

			bldg_value_per_sqft = (base_bldg_value + (quality_value * bldg_quality_num)) * bldg_type_mult

			depreciation_from_age = min(0.0, 1 - (bldg_age_years / 100))
			depreciation_from_condition = min(0.0, 1 - (bldg_condition_num / 6))

			total_depreciation = (depreciation_from_age + depreciation_from_condition)/2

			bldg_value_per_sqft = bldg_value_per_sqft * (1 - total_depreciation)
			bldg_value = bldg_area_finished_sqft * bldg_value_per_sqft

			total_value = land_value + bldg_value

			valid_sale = False
			sale_price = 0
			sale_age_days = 0

			# roll for a sale:
			if np.random.rand() < percent_sales:
				valid_sale = True
				sale_price = total_value * (1 + np.random.uniform(-noise_sales, noise_sales))
				sale_age_days = np.random.randint(0, 365)

			data["key"].append(str(key))
			data["neighborhood"].append("")
			data["bldg_area_finished_sqft"].append(bldg_area_finished_sqft)
			data["land_area_sqft"].append(land_area_sqft)
			data["bldg_quality_num"].append(bldg_quality_num)
			data["bldg_condition_num"].append(bldg_condition_num)
			data["bldg_age_years"].append(bldg_age_years)
			data["bldg_type"].append(bldg_type)
			data["land_value"].append(land_value)
			data["bldg_value"].append(bldg_value)
			data["total_value"].append(total_value)
			data["dist_to_cbd"].append(dist_center)
			data["latitude"].append(latitude)
			data["longitude"].append(longitude)
			data["valid_sale"].append(valid_sale)
			data["sale_price"].append(sale_price)
			data["sale_age_days"].append(sale_age_days)

	df = pd.DataFrame(data)

	# Derive neighborhood:
	distance_quantiles = [0.0, 0.25, 0.75, 1.0]
	distance_bins = [np.quantile(df["dist_to_cbd"], q) for q in distance_quantiles]
	distance_labels = ["urban", "suburban", "rural"]
	df["neighborhood"] = pd.cut(
		df["dist_to_cbd"],
		bins=distance_bins,
		labels=distance_labels,
		include_lowest=True
	)

	# Derive based on longitude/latitude what (nw,ne,sw,se) quadrant a parcel is in:
	df["quadrant"] = ""
	df.loc[df["latitude"].ge(latitude_center), "quadrant"] += "s"
	df.loc[df["latitude"].lt(latitude_center), "quadrant"] += "n"
	df.loc[df["longitude"].ge(longitude_center), "quadrant"] += "e"
	df.loc[df["longitude"].lt(longitude_center), "quadrant"] += "w"

	df["neighborhood"] = df["neighborhood"].astype(str) + "_" + df["quadrant"].astype(str)

	return df