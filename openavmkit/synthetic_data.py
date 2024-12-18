import math
from datetime import datetime as dt

import numpy as np
import pandas as pd

from openavmkit.time_adjustment import _generate_days


class SyntheticData:
	df: pd.DataFrame
	time_land_mult: pd.DataFrame
	time_bldg_mult: pd.DataFrame

	def __init__(
			self,
			df: pd.DataFrame,
			time_land_mult: pd.DataFrame,
			time_bldg_mult: pd.DataFrame
	):
		self.df = df
		self.time_land_mult = time_land_mult
		self.time_bldg_mult = time_bldg_mult


def generate_inflation_curve(
		start_year: int,
		end_year: int,
		annual_inflation_rate: float = 0.02,
		annual_inflation_rate_stdev: float = 0.01,
		seasonality_amplitude: float = 0.10,
		monthly_noise: float = 0.0,
		daily_noise: float = 0.0
):
	"""
	Generates a time series of inflation/deflation values over a given duration.
	"""

	start_date = dt(year=start_year, month=1, day=1)
	end_date = dt(year=end_year, month=12, day=31)

	duration_years = (end_year - start_year) + 1 # we add + 1 because we end in December of the end year
	duration_months = (duration_years * 12) + 1
	duration_days = (end_date - start_date).days + 1

	# First we generate a series of data points
	# +1 for the beginning value, then one for the end of each year:
	time_mult_years = np.array([1.0] * (duration_years+1))

	# We increase each point after the first by the annual inflation rate:
	for i in range(1, duration_years+1):
		curr_year_inflation_rate = np.random.normal(annual_inflation_rate, annual_inflation_rate_stdev)
		time_mult_years[i] = time_mult_years[i-1] * (1 + curr_year_inflation_rate)

	# We subdivide each year into months, interpolating between the yearly values:
	# +1 for the beginning value, then one for the end of each month:
	time_mult_months = np.array([1.0] * (duration_months))

	# We interpolate between the yearly values:
	# We start at 1.0, then each next value is for the end of that month
	month = 1
	year = 0
	for t in range(1, duration_months):
		curr_mult = time_mult_years[year]
		next_mult = time_mult_years[year+1]
		time_mult_months[t] = curr_mult + ((next_mult - curr_mult) * (month/12))
		month += 1
		if month > 12:
			month = 1
			year += 1

	# We prepare an array for seasonality:
	# +1 for the beginning value, then one for the end of each month:
	time_mult_season = np.array([1.0] * (duration_months))

	# We add seasonality amplitude:
	# - prices peak in May/June
	# - prices bottom out in December/January
	# - we use a sine wave to model this:
	t_m = 0
	for t in range(0, duration_months):
		# t_n is the normalized time, ranging from 0 to 1
		t_n = t_m / 12
		# 1.4 * pi is the phase shift to peak in May/June
		time_mult_season[t] = 1.0 + ((math.sin((1.4 * math.pi) - (2 * math.pi * t_n))) * seasonality_amplitude)
		t_m += 1
		if t_m > 12:
			t_m = 1

	# We overlay the seasonality amplitude onto time_mult_months:
	time_mult_months = time_mult_months * time_mult_season

	# We add monthly noise:
	monthly_noise_values = np.random.normal(1.0, monthly_noise, duration_months)
	monthly_noise_values[0] = 1.0

	# We overlay the monthly noise onto time_mult:
	time_mult_months = time_mult_months * monthly_noise_values

	# Then we subdivide each month into days, interpolating between the monthly values:
	time_mult_days = np.array([1.0] * duration_days)

	curr_date = start_date
	curr_month = curr_date.month - 1
	curr_month_len_in_days = (curr_date + pd.DateOffset(months=1) - curr_date).days
	t_month = 0

	day_of_month = 1

	# We iterate over the days, interpolating between the monthly values:
	for t in range(0, duration_days):
		# add a time delta to curr_date of one day:
		t_month_next = t_month + 1
		mult_curr = time_mult_months[t_month]
		mult_next = time_mult_months[t_month_next]
		frac = day_of_month / curr_month_len_in_days
		time_mult_days[t] = mult_curr + (mult_next - mult_curr) * frac

		# add daily noise
		time_mult_days[t] *= np.random.normal(1.0, daily_noise)

		curr_date = curr_date + pd.DateOffset(days=1)
		new_month = curr_date.month - 1
		day_of_month += 1
		if new_month != curr_month:
			day_of_month = 1
			t_month += 1
			curr_month = new_month
			curr_month_len_in_days = (curr_date + pd.DateOffset(months=1) - curr_date).days

	return time_mult_days


def generate_basic(
		size: int,
		percent_sales: float = 0.1,
		noise_sales: float = 0.05,
		seed: int = 1337,
		land_inflation: dict = None,
		bldg_inflation: dict = None
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
		"sale_price_per_impr_sqft": [],
		"sale_price_per_land_sqft": [],
		"latitude": [],
		"longitude": [],
		"sale_age_days": [],
		"sale_date": [],
		"sale_year": [],
		"sale_month": [],
		"sale_quarter": [],
		"sale_year_month": [],
		"sale_year_quarter": []
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

	start_date = dt(year=2020, month=1, day=1)
	end_date = dt(year=2024, month=12, day=31)

	days_duration = (end_date - start_date).days

	# default time/bldg inflation parameters:
	if land_inflation is None:
		land_inflation = {
			"start_year": start_date.year,
			"end_year": end_date.year,
			"annual_inflation_rate": 0.1,
			"annual_inflation_rate_stdev": 0.01,
			"seasonality_amplitude": 0.025,
			"monthly_noise": 0.0125,
			"daily_noise": 0.0025
		}
	if bldg_inflation is None:
		bldg_inflation = {
			"start_year": start_date.year,
			"end_year": end_date.year,
			"annual_inflation_rate": 0.02,
			"annual_inflation_rate_stdev": 0.005,
			"seasonality_amplitude": 0.00,
			"monthly_noise": 0.01,
			"daily_noise": 0.005
		}

	# generate the time adjustment if so desired, using `land_inflation` as parameters:
	time_land_mult = generate_inflation_curve(**land_inflation)
	time_bldg_mult = generate_inflation_curve(**bldg_inflation)

	df_time_land_mult = pd.DataFrame({
		"period": _generate_days(start_date, end_date),
		"value": time_land_mult
	})
	df_time_bldg_mult = pd.DataFrame({
		"period": _generate_days(start_date, end_date),
		"value": time_bldg_mult
	})
	df_time_land_mult["period"] = pd.to_datetime(df_time_land_mult["period"])
	df_time_bldg_mult["period"] = pd.to_datetime(df_time_bldg_mult["period"])

	for y in range(0, size):
		for x in range(0, size):

			_x = x/size
			_y = y/size

			latitude = nw_lat + (width * _x)
			longitude = nw_lon + (height * _y)

			dist_x = abs(_x - 0.5)
			dist_y = abs(_y - 0.5)
			dist_center = (dist_x**2 + dist_y**2)**0.5

			valid_sale = False
			# roll for a sale:
			if np.random.rand() < percent_sales:
				valid_sale = True

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

			# TODO: properly evolve the city over time with sales in "real time" so we don't wind up with weird situations
			# such as the one we're in, where the sale version of the price doesn't take into account that the building was
			# younger than at the valuation date

			sale_price = 0
			sale_price_per_land_sqft = 0
			sale_price_per_impr_sqft = 0
			sale_age_days = 0

			sale_date = None
			sale_year = None
			sale_month = None
			sale_quarter = None
			sale_year_month = None
			sale_year_quarter = None

			if valid_sale:
				# account for time inflation:
				sale_age_days = np.random.randint(0, days_duration)
				land_value_per_land_sqft_sale = land_value_per_land_sqft * time_land_mult[sale_age_days]
				#bldg_value_per_sqft_sale = bldg_value_per_sqft * time_bldg_mult[sale_age_days]
				bldg_value_per_sqft_sale = bldg_value_per_sqft

				# calculate total values:
				land_value_sale = land_area_sqft * land_value_per_land_sqft_sale
				bldg_value_sale = bldg_area_finished_sqft * bldg_value_per_sqft_sale
				total_value_sale = land_value_sale + bldg_value_sale

				# add some noise
				sale_price = total_value_sale * (1 + np.random.uniform(-noise_sales, noise_sales))

				sale_price_per_land_sqft = sale_price / land_area_sqft
				sale_price_per_impr_sqft = sale_price / bldg_area_finished_sqft

				sale_date = start_date + pd.DateOffset(days=sale_age_days)
				sale_year = sale_date.year
				sale_month = sale_date.month
				sale_quarter = (sale_month - 1) // 3 + 1
				sale_year_month = f"{sale_year:04}-{sale_month:02}"
				sale_year_quarter = f"{sale_year:04}-Q{sale_quarter}"

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
			data["sale_price_per_impr_sqft"].append(sale_price_per_impr_sqft)
			data["sale_price_per_land_sqft"].append(sale_price_per_land_sqft)
			data["sale_age_days"].append(sale_age_days)
			data["sale_date"].append(sale_date)
			data["sale_year"].append(sale_year)
			data["sale_month"].append(sale_month)
			data["sale_quarter"].append(sale_quarter)
			data["sale_year_month"].append(sale_year_month)
			data["sale_year_quarter"].append(sale_year_quarter)

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

	sd = SyntheticData(df, df_time_land_mult, df_time_bldg_mult)
	return sd