import os
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd
from pandas import Series
from typing import TypedDict, Literal
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import LineString

from openavmkit.calculations import crawl_calc_dict_for_fields, perform_calculations
from openavmkit.filters import resolve_filter, select_filter
from openavmkit.utilities.data import combine_dfs, div_field_z_safe
from openavmkit.utilities.geometry import get_crs, clean_geometry, identify_irregular_parcels, get_exterior_coords
from openavmkit.utilities.settings import get_fields_categorical, get_fields_impr, get_fields_boolean, \
	get_fields_numeric, get_model_group_ids, get_fields_date, get_long_distance_unit, get_valuation_date
from matplotlib import pyplot as plt

@dataclass
class SalesUniversePair:
	sales: pd.DataFrame
	universe: pd.DataFrame

	def __getitem__(self, key):
		return getattr(self, key)

	def set(self, key, value):
		if key == "sales":
			self.sales = value
		elif key == "universe":
			self.universe = value
		else:
			raise ValueError(f"Invalid key: {key}")

	def update_sales(self, new_sales: pd.DataFrame):

		# This function lets you push updates to "sales" while keeping it as an "overlay" that doesn't contain any redundant information

		# First we note what fields were in sales last time
		old_fields = self.sales.columns.values

		# We note what sales are in universe but were not in sales
		univ_fields = [field for field in self.universe.columns.values if field not in old_fields]

		# Note the new fields generated -- these are the fields that are in new_sales but not in old_sales nor in the universe
		new_fields = [field for field in new_sales.columns.values if field not in old_fields and field not in univ_fields]

		# Create a modified version of df_sales with only two changes:
		# - reduced to the correct selection of keys
		# - adds the newly generated fields

		# TODO: add support for "key_sale"
		return_keys = new_sales["key"].values
		reconciled = new_sales.copy()
		reconciled = reconciled[reconciled["key"].isin(return_keys)]
		reconciled = combine_dfs(reconciled, new_sales[["key"]+new_fields])
		self.sales = reconciled


SUPKey = Literal["sales", "universe"]


def get_hydrated_sales_from_sup(sup: SalesUniversePair):
	df_sales = sup["sales"]
	df_univ = sup["universe"]
	df_merged = combine_dfs(df_sales, df_univ, False, index="key")

	if "geometry" in df_merged and "geometry" not in df_sales:
		# convert df_merged to geodataframe:
		df_merged = gpd.GeoDataFrame(df_merged, geometry="geometry")

	return df_merged


def enrich_time(df: pd.DataFrame, time_formats: dict, settings: dict) -> pd.DataFrame:
	for key in time_formats:
		time_format = time_formats[key]
		if key in df:
			df[key] = pd.to_datetime(df[key], format=time_format, errors="coerce")

	for prefix in ["sale"]:
		do_enrich = False
		for col in df.columns.values:
			if f"{prefix}_" in col:
				do_enrich = True
				break
		if do_enrich:
			df = _enrich_time_field(df, prefix, add_year_month=True, add_year_quarter=True)
			if prefix == "sale":
				df = enrich_sale_age_days(df, settings)

	return df


def enrich_sale_age_days(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
	val_date = get_valuation_date(settings)
	# create a new field with dtype Int64
	df["sale_age_days"] = None
	df["sale_age_days"] = df["sale_age_days"].astype("Int64")
	sale_date_as_datetime = pd.to_datetime(df["sale_date"], format="%Y-%m-%d", errors="coerce")
	df.loc[~sale_date_as_datetime.isna(), "sale_age_days"] = (val_date - sale_date_as_datetime).dt.days
	return df


def enrich_year_built(df: pd.DataFrame, settings: dict, is_sales: bool = False):
	val_date = get_valuation_date(settings)
	for prefix in ["bldg", "bldg_effective"]:
		col = f"{prefix}_year_built"
		if col in df:
			new_col = f"{prefix}_age_years"
			df = _enrich_year_built(df, col, new_col, val_date, is_sales)
	return df


def _enrich_year_built(
		df: pd.DataFrame,
		col: str,
		new_col: str,
		val_date: datetime,
		is_sales: bool = False
) -> pd.DataFrame:

	if not is_sales:
		val_year = val_date.year
		df[new_col] = val_year - df[col]
	else:
		df.loc[df["sale_year"].notna(), new_col] = df["sale_year"] - df[col]

	return df



def _enrich_time_field(
		df: pd.DataFrame,
		prefix: str,
		add_year_month: bool = True,
		add_year_quarter: bool = True
) -> pd.DataFrame:

	if f"{prefix}_date" not in df:
		# Check if we have _year, _month, and _day:
		if f"{prefix}_year" in df and f"{prefix}_month" in df and f"{prefix}_day" in df:
			date_str_series = (
					df[f"{prefix}_year"].astype(str).str.pad(4,fillchar="0") + "-" +
					df[f"{prefix}_month"].astype(str).str.pad(2,fillchar="0") + "-" +
					df[f"{prefix}_day"].astype(str).str.pad(2,fillchar="0")
			)
			df[f"{prefix}_date"] = pd.to_datetime(date_str_series, format="%Y-%m-%d", errors="coerce")
		else:
			raise ValueError(f"The dataframe does not contain a '{prefix}_date' column.")

	# ensure f"{prefix}_date" is a datetime object:
	df[f"{prefix}_date"] = pd.to_datetime(df[f"{prefix}_date"], format="%Y-%m-%d", errors="coerce")

	# create a f"{prefix}_year" column if it does not exist:
	if f"{prefix}_year" not in df:
		df[f"{prefix}_year"] = df[f"{prefix}_date"].dt.year
	if f"{prefix}_month" not in df:
		df[f"{prefix}_month"] = df[f"{prefix}_date"].dt.month
	if f"{prefix}_quarter" not in df:
		df[f"{prefix}_quarter"] = df[f"{prefix}_date"].dt.quarter

	if add_year_month:
		if f"{prefix}_year_month" not in df:
			# format sale date in the form of "YYYY-MM"
			df[f"{prefix}_year_month"] = df[f"{prefix}_date"].dt.to_period("M").astype("str")

	if add_year_quarter:
		if f"{prefix}_year_quarter" not in df:
			# format sale date in the form of "YYYY-QX"
			df[f"{prefix}_year_quarter"] = df[f"{prefix}_date"].dt.to_period("Q").astype("str")

	checks = ["_year", "_month", "_day", "_year_month", "_year_quarter"]
	for check in checks:
		# Verify that the derived field a. exists and b. matches the value in the date field:
		if f"{prefix}{check}" in df:
			if f"{prefix}_date" in df:
				if check in ["_year", "_month", "_day"]:
					date_value = None
					if check == "_year":
						date_value = df[f"{prefix}_date"].dt.year.astype("Int64")
					elif check == "_month":
						date_value = df[f"{prefix}_date"].dt.month.astype("Int64")
					elif check == "_day":
						date_value = df[f"{prefix}_date"].dt.day.astype("Int64")
					if not df[f"{prefix}{check}"].astype("Int64").equals(date_value):
						# Count how many fields differ:
						n_diff = df[f"{prefix}{check}"].astype("Int64").ne(date_value).sum()
						raise ValueError(f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows.")
				elif check in ["_year_month", "_year_quarter"]:
					date_value = None
					if check == "_year_month":
						date_value = df[f"{prefix}_date"].dt.to_period("M").astype("str")
					elif check == "_year_quarter":
						date_value = df[f"{prefix}_date"].dt.to_period("Q").astype("str")
					if not df[f"{prefix}{check}"].equals(date_value):
						# Count how many fields differ:
						n_diff = df[f"{prefix}{check}"].ne(date_value).sum()
						raise ValueError(f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows.")

	return df


def old_enrich_time(df: pd.DataFrame) -> pd.DataFrame:
	if "sale_date" not in df:
		raise ValueError("The dataframe does not contain a 'sale_date' column.")
	# ensure "sale_date" is a datetime object:
	df["sale_date"] = pd.to_datetime(df["sale_date"], format="%Y-%m-%d", errors="coerce")
	# create a "sale_year" column if it does not exist:
	if "sale_year" not in df:
		df["sale_year"] = df["sale_date"].dt.year
	if "sale_month" not in df:
		df["sale_month"] = df["sale_date"].dt.month
	if "sale_quarter" not in df:
		df["sale_quarter"] = df["sale_date"].dt.quarter
	if "sale_year_month" not in df:
		# format sale date in the form of "YYYY-MM"
		df["sale_year_month"] = df["sale_date"].dt.to_period("M").astype("str")
	if "sale_year_quarter" not in df:
		# format sale date in the form of "YYYY-QX"
		df["sale_year_quarter"] = df["sale_date"].dt.to_period("Q").astype("str")
	return df


def simulate_removed_buildings(df: pd.DataFrame, settings: dict, idx_vacant: Series = None):

	if idx_vacant is None:
		# do the whole thing:
		idx_vacant = df.index

	fields_impr = get_fields_impr(settings, df)

	# Step 3: fill unknown values for categorical improvements:
	fields_impr_cat = fields_impr["categorical"]
	fields_impr_num = fields_impr["numeric"]
	fields_impr_bool = fields_impr["boolean"]

	for field in fields_impr_cat:
		df.loc[idx_vacant, field] = "UNKNOWN"

	for field in fields_impr_num:
		df.loc[idx_vacant, field] = 0

	for field in fields_impr_bool:
		df.loc[idx_vacant, field] = False

	return df


def get_sale_field(settings: dict) -> str:
	ta = settings.get("modeling", {}).get("instructions", {}).get("time_adjustment", {})
	use = ta.get("use", True)
	if use:
		return "sale_price_time_adj"
	return "sale_price"


def get_vacant_sales(df_in: pd.DataFrame, settings: dict, invert:bool = False) -> pd.DataFrame:
	df = df_in.copy()
	df = boolify_column_in_df(df, "vacant_sale")
	idx_vacant_sale = df["vacant_sale"].eq(True)
	if invert:
		idx_vacant_sale = ~idx_vacant_sale
	df_vacant_sales = df[idx_vacant_sale].copy()
	return df_vacant_sales


def  get_vacant(df_in: pd.DataFrame, settings: dict, invert:bool = False) -> pd.DataFrame:
	# TODO : support custom vacant filter from user settings
	df = df_in.copy()

	is_vacant_dtype = df["is_vacant"].dtype
	if is_vacant_dtype != bool:
		raise ValueError(f"The 'is_vacant' column must be a boolean type (found: {is_vacant_dtype})")

	idx_vacant = df["is_vacant"].eq(True)
	if invert:
		idx_vacant = ~idx_vacant
	df_vacant = df[idx_vacant].copy()
	return df_vacant


def get_sales(df_in: pd.DataFrame, settings: dict, vacant_only: bool = False) -> pd.DataFrame:
	df = df_in.copy()
	valid_sale_dtype = df["valid_sale"].dtype
	if valid_sale_dtype != bool:
		raise ValueError(f"The 'valid_sale' column must be a boolean type (found: {valid_sale_dtype})")

	if "vacant_sale" in df:
		vacant_sale_dtype = df["vacant_sale"].dtype
		if vacant_sale_dtype != bool:
			raise ValueError(f"The 'vacant_sale' column must be a boolean type (found: {vacant_sale_dtype})")
		# check for vacant sales:
		idx_vacant_sale = df["vacant_sale"].eq(True)
		df = simulate_removed_buildings(df, settings, idx_vacant_sale)

		# if a property was NOT vacant at time of sale, but is vacant now, then the sale is invalid, because we don't
		# know e.g. what the square footage was at the time of sale
		# TODO: deal with this better when we support frozen characteristics
		df.loc[
			~idx_vacant_sale &
			df["is_vacant"].eq(True),
			"valid_sale"
		] = False

	# get the sales
	df_sales : pd.DataFrame = df[
		df["sale_price"].gt(0) &
		df["valid_sale"].eq(True) &
		(df["vacant_sale"].eq(True) if vacant_only else True)
	].copy()

	return df_sales


def boolify_series(series: pd.Series):
	if series.dtype in ["object", "string", "str"]:
		series = series.astype(str).str.lower().str.strip()
		series = series.replace(["true", "t", "1"], 1)
		series = series.replace(["false", "f", "0"], 0)
	series = series.fillna(0)
	series = series.astype(bool)
	return series


def boolify_column_in_df(df: pd.DataFrame, field: str):
	series = df[field]
	series = boolify_series(series)
	df[field] = series
	return df


def get_report_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
	locations = settings.get("field_classification", {}).get("important", {}).get("report_locations", [])
	if df is not None:
		locations = [loc for loc in locations if loc in df]
	return locations


def get_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
	locations = settings.get("field_classification", {}).get("important", {}).get("locations", [])
	if df is not None:
		locations = [loc for loc in locations if loc in df]
	return locations


def get_important_fields(settings: dict, df: pd.DataFrame = None) -> list[str]:
	imp = settings.get("field_classification", {}).get("important", {})
	fields = imp.get("fields", {})
	list_fields = []
	if df is not None:
		for field in fields:
			other_name = fields[field]
			if other_name in df:
				list_fields.append(other_name)
	return list_fields


def get_important_field(settings: dict, field_name: str, df: pd.DataFrame = None) -> str | None:
	imp = settings.get("field_classification", {}).get("important", {})
	other_name = imp.get("fields", {}).get(field_name, None)
	if df is not None:
		if other_name is not None and other_name in df:
			return other_name
		else:
			return None
	return other_name


def get_field_classifications(settings: dict):
	field_map = {}
	for ftype in ["land", "impr", "other"]:
		nums = get_fields_numeric(settings, None, False, [ftype])
		cats = get_fields_categorical(settings, None, False, [ftype])
		bools = get_fields_boolean(settings, None, [ftype])
		for field in nums:
			field_map[field] = {"type": ftype, "class": "numeric"}
		for field in cats:
			field_map[field] = {"type": ftype, "class": "categorical"}
		for field in bools:
			field_map[field] = {"type": ftype, "class": "boolean"}
	return field_map


def get_dtypes_from_settings(settings: dict):
	cats = get_fields_categorical(settings, include_boolean=False)
	bools = get_fields_boolean(settings)
	nums = get_fields_numeric(settings, include_boolean=False)
	dtypes = {}
	for c in cats:
		dtypes[c] = "string"
	for b in bools:
		dtypes[b] = "bool"
	for n in nums:
		dtypes[n] = "Float64"
	return dtypes


def process_data(dataframes: dict[str : pd.DataFrame], settings: dict, verbose: bool = False) -> SalesUniversePair:
	"""
	Process the data from the settings.
	"""
	s_data = settings.get("data", {})
	s_process = s_data.get("process", {})
	s_merge = s_process.get("merge", {})

	merge_univ : list | None = s_merge.get("universe", None)
	merge_sales : list | None = s_merge.get("sales", None)

	if merge_univ is None:
		raise ValueError(f"No \"universe\" merge instructions found. data.process.merge must have exactly two keys: \"universe\", and \"sales\"")
	if merge_sales is None:
		raise ValueError(f"No \"sales\" merge instructions found. data.process.merge must have exactly two keys: \"universe\", and \"sales\"")

	df_univ = merge_dict_of_dfs(dataframes, merge_univ, settings)
	df_sales = merge_dict_of_dfs(dataframes, merge_sales, settings)

	if "valid_sale" not in df_sales:
		raise ValueError("The 'valid_sale' column is required in the sales data.")
	if "vacant_sale" not in df_sales:
		raise ValueError("The 'vacant_sale' column is required in the sales data.")

	df_sales = df_sales[df_sales["valid_sale"].eq(True)].copy().reset_index(drop=True)

	sup : SalesUniversePair = SalesUniversePair(universe=df_univ, sales=df_sales)

	enrich_data(sup, s_process.get("enrich", {}), dataframes, settings, verbose=verbose)

	return sup


def enrich_data(sup: SalesUniversePair, s_enrich: dict, dataframes: dict[str : pd.DataFrame], settings: dict, verbose: bool = False) -> SalesUniversePair:
	supkeys : list[SUPKey] = ["universe", "sales"]

	# Add the "both" entries to both "universe" and "sales" and delete the "both" entry afterward.
	if "both" in s_enrich:
		s_enrich2 = s_enrich.copy()
		s_both = s_enrich.get("both")
		for key in s_both:
			for supkey in supkeys:
				sup_entry = s_enrich.get(supkey, {})
				if key in sup_entry:
					# Check if the key already exists on "sales" or "universe"
					raise ValueError(f"Cannot enrich '{key}' twice -- found in both \"both\" and \"{supkey}\". Please remove one.")
				entry = s_both[key]

				# add the entry from "both" to both the "sales" & "universe" entry
				sup_entry2 = s_enrich2.get(supkey, {})
				sup_entry2[key] = entry
				s_enrich2[supkey] = sup_entry2

		del s_enrich2["both"] # remove the now-redundant "both" key
		s_enrich = s_enrich2

	for supkey in supkeys:
		if verbose:
			print(f"Enriching {supkey}...")
		df = sup[supkey]
		s_enrich_local : dict | None = s_enrich.get(supkey, None)
		if s_enrich_local is not None:

			df = _enrich_df_geometry(
				df,
				s_enrich_local,
				dataframes,
				settings,
				verbose=verbose
			)

			df = _enrich_df_basic(
				df,
				s_enrich_local,
				dataframes,
				settings,
				supkey == "sales",
				verbose=verbose
			)

			sup.set(supkey, df)

	return sup


def _enrich_df_basic(
		df_in: pd.DataFrame,
		s_enrich_this: dict,
		dataframes: dict[str: pd.DataFrame],
		settings: dict,
		is_sales: bool = False,
		verbose: bool = False
) -> pd.DataFrame:

	df = df_in.copy()

	s_ref = s_enrich_this.get("ref_tables", [])
	s_calc = s_enrich_this.get("calc", {})

	# reference tables:
	df = perform_ref_tables(df, s_ref, dataframes, verbose=verbose)

	# calculations:
	df = perform_calculations(df, s_calc)

	# enrich year built:
	df = enrich_year_built(df, settings, is_sales)

	# enrich vacant:
	df = _enrich_vacant(df)

	return df


def _finesse_columns(
		df_in: pd.DataFrame | gpd.GeoDataFrame,
		suffix_left: str,
		suffix_right: str
):
	df = df_in.copy()
	cols_to_finesse = []
	for col in df.columns.values:
		if col.endswith(suffix_left):
			base_col = col[:-len(suffix_left)]
			if base_col not in cols_to_finesse:
				cols_to_finesse.append(base_col)
	for col in cols_to_finesse:
		col_spatial = f"{col}{suffix_left}"
		col_data = f"{col}{suffix_right}"
		if col_spatial in df and col_data in df:
			df[col] = df[col_spatial].combine_first(df[col_data])
			df = df.drop(columns=[col_spatial, col_data], errors="ignore")
	return df


def _enrich_vacant(
		df_in: pd.DataFrame
) -> pd.DataFrame:
	df = df_in.copy()

	df["is_vacant"] = False

	df.loc[pd.isna(df["bldg_area_finished_sqft"]), "bldg_area_finished_sqft"] = 0
	df.loc[df["bldg_area_finished_sqft"].eq(0), "is_vacant"] = True

	# TODO: handle special case of sales, where "vacant_sale" and "is_vacant" don't line up. These should always be consistent.

	return df


def _enrich_df_geometry(
		df_in: pd.DataFrame,
		s_enrich_this: dict,
		dataframes: dict[str: pd.DataFrame],
		settings: dict,
		verbose: bool = False
) -> gpd.GeoDataFrame:

	df = df_in.copy()

	s_geom = s_enrich_this.get("geometry", [])
	s_dist = s_enrich_this.get("distances", {})

	gdf : gpd.GeoDataFrame

	# geometry
	gdf = perform_spatial_joins(s_geom, dataframes, verbose=verbose)

	# distances
	gdf = perform_distance_calculations(gdf, s_dist, dataframes, get_long_distance_unit(settings), verbose=verbose)

	# Merge everything together:
	try_keys = ["key", "key2", "key3"]
	success = False
	gdf_merged: gpd.GeoDataFrame | None = None
	for key in try_keys:
		if key in gdf and key in df:
			if verbose:
				print(f"Using \"{key}\" to merge shapefiles onto df")

			n_dupes_gdf = gdf.duplicated(subset=key).sum()
			n_dupes_df = df.duplicated(subset=key).sum()
			if n_dupes_gdf > 0 or n_dupes_df > 0:
				raise ValueError(f"Found {n_dupes_gdf} duplicate keys in the geo_parcels dataframe, and {n_dupes_df} duplicate keys in the base dataframe. Cannot perform spatial join. De-duplicate your dataframes and try again.")

			gdf_merged = gdf.merge(df, on=key, how="left", suffixes=("_spatial", "_data"))
			gdf_merged = _finesse_columns(gdf_merged, "_spatial", "_data")

			success = True
			break
	if not success:
		raise ValueError(f"Could not find a common key between geo_parcels and base dataframe. Tried keys: {try_keys}")

	# basic geometric enrichment
	gdf_merged = basic_geo_enrichment(gdf_merged, settings, verbose=verbose)

	return gdf_merged


def basic_geo_enrichment(gdf: gpd.GeoDataFrame, settings: dict, verbose: bool = False) -> gpd.GeoDataFrame:

	if verbose:
		print(f"Performing basic geometric enrichment...")

	if verbose:
		print(f"--> adding latitude/longitude...")
	# Temporarily convert gdf to lat/long coordinates:
	gdf_latlon = gdf.to_crs(get_crs(gdf, "latlon"))
	gdf_area = gdf.to_crs(get_crs(gdf, "equal_area"))

	# Get the lat/long coordinates of the centroid of each parcel:
	gdf["latitude"] = gdf_latlon.geometry.centroid.y
	gdf["longitude"] = gdf_latlon.geometry.centroid.x

	if verbose:
		print(f"--> calculate GIS area of each parcel...")
	# Calculate the GIS area of each parcel:
	gdf["land_area_gis_sqft"] = gdf_area.geometry.area

	# Fill missing land area with GIS area:
	gdf["land_area_given_sqft"] = gdf["land_area_sqft"]
	gdf["land_area_sqft"] = gdf["land_area_sqft"].combine_first(gdf["land_area_gis_sqft"])

	gdf["land_area_gis_delta_sqft"] = gdf["land_area_gis_sqft"] - gdf["land_area_sqft"]
	gdf["land_area_gis_delta_percent"] = div_field_z_safe(gdf["land_area_gis_delta_sqft"], gdf["land_area_sqft"])

	if verbose:
		print(f"--> counting vertices per parcel...")
	# Calculate the vertices of each parcel:
	gdf["geom_vertices"] = gdf.geometry.apply(get_exterior_coords)

	gdf = _calc_geom_stuff(gdf, verbose)

	return gdf


def _calc_geom_stuff(gdf: gpd.GeoDataFrame, verbose: bool = False) -> gpd.GeoDataFrame:
	"""Compute aspect ratios of geometries in a GeoDataFrame."""

	if verbose:
		print(f"--> calculating parcel rectangularity...")
	# Compute the minimum rotated rectangles for each geometry
	min_rotated_rects = gdf.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)

	min_rotated_rects_area_delta = np.abs(min_rotated_rects.area - gdf.geometry.area)
	min_rotated_rects_area_delta_percent = div_field_z_safe(min_rotated_rects_area_delta, gdf.geometry.area)

	gdf["geom_rectangularity_num"] = 1.0 - min_rotated_rects_area_delta_percent

	# Extract coordinates for each rectangle
	coords = min_rotated_rects.apply(lambda rect: np.array(rect.exterior.coords[:-1]))  # Drop duplicate last point

	if verbose:
		print(f"--> calculating parcel aspect ratios...")

	# Compute edge lengths efficiently using NumPy
	edge_lengths = coords.apply(lambda pts: np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))

	# Extract width and height (smallest two edges)
	dimensions = edge_lengths.apply(lambda lengths: np.sort(lengths)[:2])

	# Compute aspect ratio (width / height)
	aspect_ratios = dimensions.apply(lambda dims: dims[1] / dims[0] if dims[0] != 0 else float('inf'))

	gdf["geom_aspect_ratio"] = aspect_ratios

	gdf = identify_irregular_parcels(gdf, verbose)

	return gdf



def perform_spatial_joins(s_geom: list, dataframes: dict[str: pd.DataFrame], verbose: bool = False) -> gpd.GeoDataFrame:

	#  For geometry, provide a list of strings and/or objects, these represent spatial joins.
	#  Strings are interpreted as the IDs of loaded shapefiles. Objects must have these keys: 'id', and 'predicate',
	#  where 'predicate' is the name of the spatial join function to use.

	if not isinstance(s_geom, list):
		s_geom = [s_geom]

	# First, get our parcel geometry, look for a dataframe called "geo_parcels":
	if "geo_parcels" not in dataframes:
		raise ValueError("No 'geo_parcels' dataframe found in the dataframes. This layer is required, and it must contain parcel geometry.")

	gdf_parcels : gpd.GeoDataFrame = dataframes["geo_parcels"]

	gdf_merged = gdf_parcels.copy()

	if verbose:
		print(f"Performing spatial joins...")

	for geom in s_geom:
		if isinstance(geom, str):
			entry = {
				"id": str(geom),
				"predicate": "contains_centroid"
			}
		elif isinstance(geom, dict):
			entry = geom
		else:
			raise ValueError(f"Invalid geometry entry: {geom}")
		_id = entry.get("id")
		predicate = entry.get("predicate", "contains_centroid")

		if _id is None:
			raise ValueError("No 'id' found in geometry entry.")

		if verbose:
			if predicate != "contains_centroid":
				print(f"--> {_id} @ {predicate}")
			else:
				print(f"--> {_id}")

		gdf = dataframes[_id]

		fields_to_tag = entry.get("fields", None)
		if fields_to_tag is None:
			fields_to_tag = [field for field in gdf.columns if field != "geometry"]
		else:
			for field in fields_to_tag:
				if field not in gdf:
					raise ValueError(f"Field to tag '{field}' not found in geometry dataframe '{_id}'.")

		gdf_merged = _perform_spatial_join(gdf_merged, gdf, predicate, fields_to_tag)

	# identify parcels with no geometry
	gdf_no_geometry = gdf_merged[gdf_merged["geometry"].isna()]

	if len(gdf_no_geometry) > 0:
		warnings.warn(f"Found {len(gdf_no_geometry)} parcels with no geometry. These parcels will be excluded from the analysis. You can find them in out/errors/")
		os.makedirs("out/errors", exist_ok=True)
		gdf_no_geometry.to_parquet("out/errors/parcels_no_geometry.parquet")
		gdf_no_geometry.to_csv("out/errors/parcels_no_geometry.csv", index=False)
		gdf_no_geom_keys = gdf_no_geometry["key"].values
		# write out the keys:
		with open("out/errors/parcels_no_geometry_keys.txt", "w") as f:
			for key in gdf_no_geom_keys:
				f.write(f"{key}\n")
		# exclude no-geometry rows from gdf_merged:
		gdf_merged = gdf_merged.dropna(subset=["geometry"])

	return gdf_merged


def _perform_spatial_join_contains_centroid(gdf: gpd.GeoDataFrame, gdf_overlay: gpd.GeoDataFrame):
	# Compute centroids of each parcel
	gdf["geometry_centroid"] = gdf.geometry.centroid

	# Use within first
	gdf = gpd.sjoin(
		gdf.set_geometry("geometry_centroid"),
		gdf_overlay,
		how="left",
		predicate="within"
	)

	# remove extra columns like "index_right":
	gdf = gdf.drop(columns=["index_right"], errors="ignore")

	return gdf


def _perform_spatial_join(gdf_in: gpd.GeoDataFrame, gdf_overlay: gpd.GeoDataFrame, predicate: str, fields_to_tag: list[str]):
	gdf = gdf_in.copy()

	# Ensure both GeoDataFrames have the same CRS
	gdf_overlay = gdf_overlay.to_crs(gdf.crs)

	if "__overlay_id__" in gdf_overlay:
		raise ValueError("The overlay GeoDataFrame already contains a '__overlay_id__' column. This column is used internally by the spatial join function, and must not be present in the overlay GeoDataFrame.")

	# assign each overlay polygon a unique ID:
	gdf_overlay["__overlay_id__"] = range(len(gdf_overlay))

	# TODO: add more predicates as needed
	if predicate == "contains_centroid":
		gdf = _perform_spatial_join_contains_centroid(gdf, gdf_overlay)
	else:
		raise ValueError(f"Invalid spatial join predicate: {predicate}")

	# gdf is now properly tagged with "__overlay_id__"

	# Merge in the fields we want:
	gdf = gdf.drop(columns=fields_to_tag, errors="ignore")
	gdf = gdf.merge(gdf_overlay[["__overlay_id__"] + fields_to_tag], on="__overlay_id__", how="left")

	# clean up:
	gdf.set_geometry("geometry", inplace=True)
	gdf = gdf.drop(columns=["geometry_centroid", "__overlay_id__"], errors="ignore")
	return gdf


def _perform_distance_calculations(df_in: gpd.GeoDataFrame, gdf_in: gpd.GeoDataFrame, _id: str, unit: str = "km") -> pd.DataFrame:

	unit_factors = {"m": 1, "km": 0.001, "mile": 0.000621371, "ft": 3.28084}
	if unit not in unit_factors:
		raise ValueError(f"Unsupported unit '{unit}'")

	# Convert to equal-distance CRS for accurate distance calculations
	crs = get_crs(df_in, "equal_distance")
	df_projected = df_in.to_crs(crs).copy()
	gdf_projected = gdf_in.to_crs(crs).copy()

	# Perform nearest neighbor spatial join but keep only the key and distance column
	nearest = gpd.sjoin_nearest(df_projected, gdf_projected, how="left", distance_col=f"dist_to_{_id}")[["key", f"dist_to_{_id}"]]

	# Convert distance to the desired unit
	nearest[f"dist_to_{_id}"] *= unit_factors[unit]

	# count duplicated rows:
	n_duplicates_nearest = nearest.duplicated(subset="key").sum()
	n_duplicates_df = df_in.duplicated(subset="key").sum()

	if n_duplicates_df > 0:
		raise ValueError(f"Found {n_duplicates_nearest} duplicate keys in the base dataframe, cannot perform distance calculations. Please de-duplicate your dataframes and try again.")

	if n_duplicates_nearest > 0:
		# de-duplicate nearest:
		nearest = nearest.sort_values(by=["key", f"dist_to_{_id}"], ascending=[True, True])
		nearest = nearest.drop_duplicates(subset="key")

	# Merge results back into the original df_in (which is still in its original CRS)
	df_out = df_in.merge(nearest, on="key", how="left")

	return df_out


def perform_distance_calculations(df_in: gpd.GeoDataFrame, s_dist: dict, dataframes: dict[str: pd.DataFrame], unit: str = "km", verbose: bool = False) -> gpd.GeoDataFrame:
	# For distances, provide a list of strings and/or objects. Strings are interpreted as the IDs of loaded shapefiles.
	# Objects must have an 'id' key, and optionally a 'field' key. If a 'field' key is provided, distances will be
	# calculated for each row in the shapefile corresponding to a unique value for that field.

	df = df_in.copy()

	if verbose:
		print(f"Performing distance calculations...")

	for entry in s_dist:
		if isinstance(entry, str):
			entry = {
				"id": str(entry)
			}
		elif not isinstance(entry, dict):
			raise ValueError(f"Invalid distance entry: {entry}")
		_id = entry.get("id")
		if _id is None:
			raise ValueError("No 'id' found in distance entry.")
		if _id not in dataframes:
			raise ValueError(f"Distance table '{_id}' not found in dataframes.")
		gdf = dataframes[_id]
		field = entry.get("field", None)
		if verbose:
			print(f"--> {_id}")
		if field is None:
			df = _perform_distance_calculations(df, gdf, _id, unit)
		else:
			uniques = gdf[field].unique()
			for unique in uniques:
				if pd.isna(unique):
					continue
				gdf_subset = gdf[gdf[field].eq(unique)]
				df = _perform_distance_calculations(df, gdf_subset, f"{_id}_{unique}", unit)

	return df


def perform_ref_tables(df_in: pd.DataFrame | gpd.GeoDataFrame, s_ref: list | dict, dataframes: dict[str: pd.DataFrame], verbose: bool = False) -> pd.DataFrame | gpd.GeoDataFrame:
	df = df_in.copy()
	if not isinstance(s_ref, list):
		s_ref = [s_ref]

	if verbose:
		print(f"Performing reference table joins...")

	for ref in s_ref:
		_id = ref.get("id", None)
		key_ref_table = ref.get("key_ref_table", None)
		key_target = ref.get("key_target", None)
		add_fields = ref.get("add_fields", None)
		if verbose:
			print(f"--> {_id}")
		if _id is None:
			raise ValueError("No 'id' found in ref table.")
		if key_ref_table is None:
			raise ValueError("No 'key_ref_table' found in ref table.")
		if key_target is None:
			raise ValueError("No 'key_target' found in ref table.")
		if add_fields is None:
			raise ValueError("No 'add_fields' found in ref table.")
		if not isinstance(add_fields, list):
			raise ValueError("The 'add_fields' field must be a list of strings.")
		if len(add_fields) == 0:
			raise ValueError("The 'add_fields' field must contain at least one string.")

		if _id not in dataframes:
			raise ValueError(f"Ref table '{_id}' not found in dataframes.")

		df_ref = dataframes[_id]
		if key_ref_table not in df_ref:
			raise ValueError(f"Key field '{key_ref_table}' not found in ref table '{_id}'.")

		if key_target not in df:
			print(f"Target field '{key_target}' not found in base dataframe")
			print(f"base df columns = {df.columns.values}")
			raise ValueError(f"Target field '{key_target}' not found in base dataframe")

		for field in add_fields:
			if field not in df_ref:
				raise ValueError(f"Field '{field}' not found in ref table '{_id}'.")
			if field in df_in:
				raise ValueError(f"Field '{field}' already exists in base dataframe.")

		df_ref = df_ref[[key_ref_table] + add_fields]

		if key_ref_table == key_target:
			df = df.merge(df_ref, on=key_target, how="left")
		else:
			df = df.merge(df_ref, left_on=key_target, right_on=key_ref_table, how="left")
			df = df.drop(columns=[key_ref_table])

	return df


def get_calc_cols(settings: dict) -> list[str]:
	s_load = settings.get("data", {}).get("load", {})
	cols_to_load = []
	for key in s_load:
		entry = s_load[key]
		cols = _get_calc_cols(entry)
		cols_to_load += cols
	cols_to_load = list(set(cols_to_load))
	return cols_to_load


def _get_calc_cols(df_entry: dict) -> list[str]:
	e_calc = df_entry.get("calc", {})
	fields_in_calc = crawl_calc_dict_for_fields(e_calc)
	return fields_in_calc


def load_dataframe(entry: dict, settings: dict, verbose: bool = False, fields_cat: list = None, fields_bool: list = None, fields_num: list = None) -> pd.DataFrame | None:
	filename = entry.get("filename", "")
	filename = f"in/{filename}"
	if filename == "":
		return None
	ext = str(filename).split(".")[-1]

	column_names = snoop_column_names(filename)

	e_load = entry.get("load", {})
	e_calc = entry.get("calc", {})

	if verbose:
		print(f"Loading \"{filename}\"...")

	rename_map = {}
	dtype_map = {}
	extra_map = {}
	cols_to_load = []

	for rename_key in e_load:
		original = e_load[rename_key]
		original_key = None
		if isinstance(original, list):
			if len(original) > 0:
				original_key = original[0]
				cols_to_load += [original_key]
				rename_map[original_key] = rename_key
			if len(original) > 1:
				dtype_map[original_key] = original[1]
				if original[1] == "datetime":
					dtype_map[original_key] = "str"
			if len(original) > 2:
				extra_map[rename_key] = original[2]
		elif isinstance(original, str):
			cols_to_load += [original]
			rename_map[original] = rename_key

	# Get a list of every field that is either renamed or used in a calculation:
	fields_in_calc = crawl_calc_dict_for_fields(entry.get("calc", {}))

	cols_to_load += fields_in_calc

	# These are the columns we will actually load:
	cols_to_load = list(set(cols_to_load))

	is_geometry = False
	# Always load "geometry" column if it exists:
	if "geometry" in column_names and "geometry" not in cols_to_load:
		cols_to_load.append("geometry")
		is_geometry = True

	# Read the actual file:
	if ext == "parquet":
		if dtype_map:
			warnings.warn("dtypes are ignored when loading parquet files.")
		try:
			df = gpd.read_parquet(filename, columns=cols_to_load)
		except ValueError:
			df = pd.read_parquet(filename, columns=cols_to_load)
	elif ext == "csv":
		df = pd.read_csv(filename, usecols=cols_to_load, dtype=dtype_map)
	else:
		raise ValueError(f"Unsupported file extension: {ext}")

	# Perform calculations:
	df = perform_calculations(df, e_calc)

	# Perform renames:
	df = df.rename(columns=rename_map)

	# Fix up the types appropriately
	if fields_cat is None:
		fields_cat = get_fields_categorical(settings, include_boolean=False)
	if fields_bool is None:
		fields_bool = get_fields_boolean(settings)
	if fields_num is None:
		fields_num = get_fields_numeric(settings, include_boolean=False)

	for col in df.columns:
		if col in fields_cat:
			df[col] = df[col].astype("string")
		elif col in fields_bool:
			df[col] = boolify_series(df[col])
		elif col in fields_num:
			df[col] = df[col].astype("Float64")

	# Fix up all the dates
	date_fields = get_fields_date(settings, df)
	time_format_map = {}
	for xkey in extra_map:
		if xkey in date_fields:
			# The third parameter specifies a date, if it's a date field
			time_format_map[xkey] = extra_map[xkey]

	# Ensure that all date fields have a time format specified:
	for dkey in date_fields:
		if dkey not in time_format_map:
			raise ValueError(f"Date field '{dkey}' does not have a time format specified.")

	# Enrich the time fields (e.g. add year, month, quarter, etc., and ensure all sub-fields match the date field)
	df = enrich_time(df, time_format_map, settings)

	# Handle duplicated rows
	dupes = entry.get("dupes", None)
	dupes_was_none = dupes is None
	if dupes is None:
		if is_geometry:
			dupes = "auto"
		else:
			dupes = {}

	if dupes == "auto":
		if is_geometry:
			# For geometry columns, default to the first column, whatever it is
			cols = [col for col in df.columns.values if col != "geometry"]
			col = cols[0]
			dupes = {
				"subset": [col],
				"sort_by": [col, "asc"],
				"drop": True
			}
			if dupes_was_none:
				warnings.warn(f"'dupes' not found for geo df '{filename}', defaulting to \"{col}\" as de-dedupe key. Set 'dupes:\"auto\" to remove this warning.'")
		else:
			# For non-geometry columns, try to find the primary, secondary, or tertiary key
			keys = ["key", "key2", "key3"]
			for key in keys:
				if key in df:
					dupes = {
						"subset": [key],
						"sort_by": [key, "asc"],
						"drop": True
					}
					break

	df = handle_duplicated_rows(df, dupes)

	# Check if it's a geodataframe and if so clean it:
	if is_geometry:
		gdf : gpd.GeoDataFrame = gpd.GeoDataFrame(df, geometry="geometry")
		gdf = clean_geometry(gdf, ensure_polygon=True)
		df = gdf

	return df


def snoop_column_names(filename: str) -> list[str]:
	ext = str(filename).split(".")[-1]
	if ext == "parquet":
		parquet_file = pq.ParquetFile(filename)
		return parquet_file.schema.names
	elif ext == "csv":
		return pd.read_csv(filename, nrows=0).columns.tolist()
	raise ValueError(f"Unsupported file extension: \"{ext}\"")


def handle_duplicated_rows(df_in: pd.DataFrame, dupes: dict) -> pd.DataFrame:

	subset = dupes.get("subset", "key")

	if not isinstance(subset, list):
		subset = [subset]

	# if any of the specified keys are not in the dataframe, return the dataframe as is
	for key in subset:
		if key not in df_in:
			return df_in

	do_drop = dupes.get("drop", True)

	# Count duplicates:
	num_dupes = df_in.duplicated(subset=subset).sum()

	if num_dupes > 0:
		sort_by = dupes.get("sort_by", ["key", "asc"])
		if not isinstance(sort_by, list):
			raise ValueError("sort_by must be a list of string pairs of the form [<field_name>, <asc|desc>]")
		if len(sort_by) == 2:
			if isinstance(sort_by[0], str) and isinstance(sort_by[1], str):
				sort_by = [sort_by]
		else:
			for entry in sort_by:
				if not isinstance(entry, list):
					raise ValueError(f"sort_by must be a list of string pairs of the form [<field_name>, <asc|desc>], but found a non-list entry: {entry}")
				elif len(entry) != 2:
					raise ValueError(f"sort_by must be a list of string pairs of the form [<field_name>, <asc|desc], but found an entry with {len(entry)} members: {entry}")
				elif not isinstance(entry[0], str) or not isinstance(entry[1], str):
					raise ValueError(f"sort_by must be a list of string pairs of the form [<field_name>, <asc|desc], but found an entry with non-string members: {entry}")

		df = df_in.copy()
		bys = [x[0] for x in sort_by]
		ascendings = [x[1] == "asc" for x in sort_by]
		df = df.sort_values(by=bys, ascending=ascendings)
		if do_drop:
			df = df.drop_duplicates(subset=subset, keep="first")

		return df.reset_index(drop=True)

	return df_in


def merge_dict_of_dfs(dataframes: dict[str : pd.DataFrame], merge_list: list, settings: dict) -> pd.DataFrame:
	merges = []

	s_reconcile = settings.get("data", {}).get("process", {}).get("reconcile", {})

	for entry in merge_list:
		df_id = None
		how = "left"
		on = "key"
		if isinstance(entry, str):
			if entry not in dataframes:
				raise ValueError(f"Merge key '{entry}' not found in dataframes.")
			df_id = entry
		elif isinstance(entry, dict):
			df_id = entry.get("id", None)
			how = entry.get("how", how)
			on = entry.get("on", on)
		if df_id is None:
			raise ValueError("Merge entry must be either a string or a dictionary with an 'id' key.")
		if df_id not in dataframes:
			raise ValueError(f"Merge key '{df_id}' not found in dataframes.")
		merges.append({
			"id": df_id,
			"df": dataframes[df_id],
			"how": how,
			"on": on
		})

	df_merged: pd.DataFrame | None = None

	# Get a list of all columns that appear in any of the dataframes
	all_cols = []

	# Get a list of all columns that appear in more than one dataframe, indexed by base name -> list of suffixed names
	# This is to automatically detect and (later) resolve conflicts
	conflicts = {}
	for merge in merges:
		df = merge["df"]
		on = merge["on"]
		suffixes = {}

		for col in df.columns.values:
			if col == on:
				continue
			if col not in all_cols:
				all_cols.append(col)
			else:
				suffixed = f"{col}_{merge['id']}"
				suffixes = {col: suffixed}
				if col not in conflicts:
					conflicts[col] = []
				conflicts[col].append(suffixed)

		df = df.rename(columns=suffixes)
		merge["df"] = df


	# Merge everything together into one big fat dataframe
	for merge in merges:
		_id = merge["id"]
		df = merge.get("df", None)
		how = merge.get("how", "left")
		on = merge.get("on", "key")
		if df_merged is None:
			df_merged = df
		else:
			df_merged = pd.merge(df_merged, df, how=how, on=on, suffixes=("", f"_{_id}"))


	# If we've defined our own reconciliation rules:
	for base_field in s_reconcile:
		# Get the list of ids for each field, this specifies the priority order to load them from
		df_ids = s_reconcile[base_field]
		if base_field not in all_cols:
			raise ValueError(f"Reconciliation field '{base_field}' not found in any of the dataframes.")
		# Generate the child fields
		child_fields = [f"{base_field}_{df_id}" for df_id in df_ids]

		# If we already have an auto-generated conflict entry for this field, we will merge with it
		if base_field in conflicts:
			# Remove any values for ids that the user has specified
			old_child_fields = conflicts[base_field]
			old_child_fields = [field for field in old_child_fields if field not in child_fields]

			# Prioritize the user's named ids over the auto-generated ones
			child_fields = child_fields + old_child_fields

		# Update the entry and pass it on
		conflicts[base_field] = child_fields

	# Clean up the conflicts
	for base_field in conflicts:
		if base_field not in df_merged:
			warnings.warn(f"Warning: Reconciliation field '{base_field}' not found in merged dataframe.")
			continue
		# Get the child fields, representing the desired merge order for the suffixed columns
		child_fields = conflicts[base_field]
		if len(child_fields) > 1:
			# Start with the first child field, then fill in with the next one for whatever is missing, and so on
			df_merged[base_field] = df_merged[base_field].fillna(df_merged[child_fields[0]])
			for i in range(1, len(child_fields)):
				df_merged[base_field] = df_merged[base_field].fillna(df_merged[child_fields[i]])
			# Drop all child fields
			df_merged = df_merged.drop(columns=child_fields)

	calc_cols = get_calc_cols(settings)

	for col in df_merged.columns.values:
		if col in calc_cols:
			df_merged = df_merged.drop(columns=[col])

	if "key" not in df_merged:
		raise ValueError("No 'key' field found in merged dataframe. This field is required.")

	len_old = len(df_merged)

	# Drop any rows that lack primary keys:
	df_merged = df_merged.dropna(subset=["key"])

	len_new = len(df_merged)

	if len_new < len_old:
		warnings.warn(f"Dropped {len_old - len_new} rows due to missing primary key.")

	return df_merged


def write_canonical_splits(
		df_sales_in: pd.DataFrame,
		settings: dict
):
	df_sales = get_sales(df_sales_in, settings)
	model_groups = get_model_group_ids(settings, df_sales)
	instructions = settings.get("modeling", {}).get("instructions", {})
	test_train_frac = instructions.get("test_train_frac", 0.8)
	random_seed = instructions.get("random_seed", 1337)
	for model_group in model_groups:
		_write_canonical_split(model_group, df_sales, settings, test_train_frac, random_seed)


def _perform_canonical_split(
		model_group: str,
		df_sales_in: pd.DataFrame,
		settings: dict,
		test_train_fraction: float = 0.8,
		random_seed: int = 1337
):
	# Select only the relevant model group
	df = df_sales_in[df_sales_in["model_group"].eq(model_group)].copy()

	# Divide universe & sales into vacant & improved
	df_v = get_vacant_sales(df, settings)
	df_i = df.drop(df_v.index)

	# Do the split for vacant & improved property separately
	np.random.seed(random_seed)
	df_v_train = df_v.sample(frac=test_train_fraction)
	df_v_test = df_v.drop(df_v_train.index)

	df_i_train = df_i.sample(frac=test_train_fraction)
	df_i_test = df_i.drop(df_i_train.index)

	# Then piece them back together
	df_test = pd.concat([df_v_test, df_i_test]).reset_index(drop=True)
	df_train = pd.concat([df_v_train, df_i_train]).reset_index(drop=True)

	# Now the vacant test set is always guaranteed to be a perfect subset of the vacant + improved test set
	return df_test, df_train


def _write_canonical_split(
		model_group: str,
		df_sales_in: pd.DataFrame,
		settings: dict,
		test_train_fraction: float = 0.8,
		random_seed: int = 1337
):

	df_test, df_train = _perform_canonical_split(
		model_group,
		df_sales_in,
		settings,
		test_train_fraction,
		random_seed
	)

	outpath = f"out/models/{model_group}/_data"
	os.makedirs(outpath, exist_ok=True)

	df_train[["key"]].to_csv(f"{outpath}/train_keys.csv", index=False)
	df_test[["key"]].to_csv(f"{outpath}/test_keys.csv", index=False)


def read_split_keys(
		model_group: str
):
	path = f"out/models/{model_group}/_data"
	train_path = f"{path}/train_keys.csv"
	test_path = f"{path}/test_keys.csv"
	if not os.path.exists(train_path) or not os.path.exists(test_path):
		raise ValueError("No split keys found.")
	train_keys = pd.read_csv(train_path)["key"].astype(str).values
	test_keys = pd.read_csv(test_path)["key"].astype(str).values
	return test_keys, train_keys


def tag_model_groups_sup(
		sup: SalesUniversePair,
		settings: dict,
		verbose: bool = False
):
	df_sales = sup["sales"].copy()
	df_univ = sup["universe"].copy()

	# We "hydrate" the sales because we want to resolve modeling group separately;
	# It may be the case that at time of sale e.g. zoning or building information changed in such a way that would have a
	# meaningful consequence on the modeling group the parcel belongs to. E.g., a former ag parcel that later got rezoned
	# as a single-family parcel.
	df_sales_hydrated = get_hydrated_sales_from_sup(sup)

	mg = settings.get("modeling", {}).get("model_groups", {})

	print(f"Len univ before = {len(df_univ)}")
	print(f"Len sales before = {len(df_sales)} after = {len(df_sales_hydrated)}")

	print(f"Overall")
	print(f"--> {len(df_univ):,} parcels")
	print(f"--> {len(df_sales):,} sales")

	df_univ["model_group"] = None
	df_sales_hydrated["model_group"] = None
	for mg_id in mg:

		# only apply model groups to parcels that don't already have one
		idx_no_model_group = df_univ["model_group"].isnull()

		entry = mg[mg_id]
		_filter = entry.get("filter", [])

		univ_index = resolve_filter(df_univ, _filter)
		df_univ.loc[idx_no_model_group & univ_index, "model_group"] = mg_id

		idx_no_model_group = df_sales_hydrated["model_group"].isnull()
		sales_index = resolve_filter(df_sales_hydrated, _filter)
		df_sales_hydrated.loc[idx_no_model_group & sales_index, "model_group"] = mg_id

	df_univ.to_parquet("out/look/tag-univ-0.parquet")

	old_model_group = df_univ[["key", "model_group"]]

	for mg_id in mg:
		entry = mg[mg_id]
		print(f"Assigning model group {mg_id}...")
		common_area = entry.get("common_area", False)
		print("common_area --> ", common_area)
		if not common_area:
			continue
		print(f"Assigning common areas for model group {mg_id}...")

		common_area_filters: list | None = None
		if isinstance(common_area, list):
			common_area_filters = common_area

		print(f"common area filters = {common_area_filters}")

		df_univ = assign_modal_model_group_to_common_area(df_univ, mg_id, common_area_filters)

	df_univ.to_parquet("out/look/tag-univ-1.parquet")

	index_changed = ~old_model_group["model_group"].eq(df_univ["model_group"])
	rows_changed = df_univ[index_changed]

	print(f" --> {len(rows_changed)} parcels had their model group changed.")

	# TODO: fix this
	# Update sales for any rows that changed due to common area assignment
	# df_sales = combine_dfs(df_sales, rows_changed, df2_stomps=True, index="key")

	# Print stuff out
	for mg_id in mg:
		entry = mg[mg_id]
		name = entry.get("name", mg_id)
		_filter = entry.get("filter", [])
		univ_index = resolve_filter(df_univ, _filter)
		sales_index = resolve_filter(df_sales_hydrated, _filter)

		if verbose:
			valid_sales_index = sales_index & df_sales_hydrated["valid_sale"].eq(True)
			improved_sales_index = sales_index & valid_sales_index & ~df_sales_hydrated["vacant_sale"].eq(True)
			vacant_sales_index = sales_index & valid_sales_index & df_sales_hydrated["vacant_sale"].eq(True)
			print(f"{name}")
			print(f"--> {univ_index.sum():,} parcels")
			print(f"--> {valid_sales_index.sum():,} sales")
			print(f"----> {improved_sales_index.sum():,} improved sales")
			print(f"----> {vacant_sales_index.sum():,} vacant sales")

	df_univ.loc[df_univ["model_group"].isna(), "model_group"] = "UNKNOWN"
	#df_sales.loc[df_sales["model_group"].isna(), "model_group"] = "UNKNOWN"

	sup.set("universe", df_univ)
	sup.set("sales", df_sales)

	return sup



def identify_parcels_with_holes(df: gpd.GeoDataFrame) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
	"""
	Identify parcels with holes (interior rings)

	Parameters:
			df (GeoDataFrame): GeoDataFrame with parcel geometries.

	Returns:
			GeoDataFrame with parcels containing interior rings (holes).
	"""

	# Identify parcels with holes
	def has_holes(geom):
		if geom.is_valid:
			if geom.geom_type == "Polygon":
				return len(geom.interiors) > 0
			elif geom.geom_type == "MultiPolygon":
				return any(len(p.interiors) > 0 for p in geom.geoms)
		return False

	# Identify:
	parcels_with_holes = df[df.geometry.apply(has_holes)]

	# Remove duplicates:
	parcels_with_holes = parcels_with_holes.drop_duplicates(subset="key")

	return parcels_with_holes


def assign_modal_model_group_to_common_area(df_univ_in: gpd.GeoDataFrame, model_group_id: str, common_area_filters: list | None = None) -> gpd.GeoDataFrame:
	"""
	Assign the modal model_group of parcels inside an enveloping "COMMON AREA" parcel to the "COMMON AREA" parcel.

	Parameters:
			df_univ (GeoDataFrame): GeoDataFrame containing the entire set of parcels.

	Returns:
			GeoDataFrame: Modified GeoDataFrame (df) with updated model_groups for COMMON AREA parcels.
	"""

	df_univ = df_univ_in.copy()

	# Ensure geometry column is set
	if df_univ.geometry.name is None:
		raise ValueError("GeoDataFrame must have a geometry column.")

	# Reduce df_univ to ONLY those parcels that have holes in them:
	df = identify_parcels_with_holes(df_univ)

	print(f" {len(df)} parcels with holes found.")
	df.to_parquet("out/look/common_area-0-holes.parquet")
	df["has_holes"] = True

	if common_area_filters is not None:
		df_extra = select_filter(df_univ, common_area_filters).copy()
		df_extra["is_common_area"] = True
		print(f" {len(df_extra)} extra parcels found.")
		df = pd.concat([df, df_extra], ignore_index=True)
		# drop duplicate keys:
		df = df.drop_duplicates(subset="key")

	print(f" {len(df)} potential COMMON AREA parcels found.")
	df.to_parquet("out/look/common_area-1-common_area.parquet")

	print(f"Assigning modal model_group to {len(df)}/{len(df_univ_in)} potential parcels...")

	df["modal_tagged"] = None

	# Iterate over COMMON AREA parcels
	for idx, row in df.iterrows():
		# Get the envelope of the COMMON AREA parcel
		common_area_geom = row.geometry
		common_area_gs = gpd.GeoSeries([common_area_geom], crs=df.crs)
		common_area_envelope_geom = common_area_geom.envelope
		common_area_envelope_gs = gpd.GeoSeries([common_area_envelope_geom], crs=df.crs)

		geom = common_area_geom.buffer(0)
		if geom.geom_type == "Polygon":
			outer_polygon = Polygon(geom.exterior)
		elif geom.geom_type == "MultiPolygon":
			outer_polygons = [Polygon(poly.exterior) for poly in geom.geoms]
			outer_polygon = unary_union(outer_polygons)
		else:
			raise ValueError("Geometry must be a Polygon or MultiPolygon")
		#outer_polygon_gs = gpd.GeoSeries([outer_polygon], crs=df.crs)

		# Find parcels wholly inside the COMMON AREA envelope
		inside_parcels = df_univ_in[df_univ_in.geometry.within(common_area_envelope_geom)].copy()

		# buffer 0 on inside parcel geometry
		inside_parcels["geometry"] = inside_parcels["geometry"].apply(lambda g: g.buffer(0))

		count1 = len(inside_parcels)

		# Exclude the COMMON AREA parcel itself (if it is in df_univ)
		inside_parcels = inside_parcels[
			~inside_parcels.geometry.apply(lambda g: g.equals(common_area_geom))
		]
		count2 = len(inside_parcels)

		# Optionally use a tiny negative buffer to avoid boundary issues

		# Exclude parcels that are not wholly inside the COMMON AREA parcel (not just the envelope bounding box):

		if isinstance(outer_polygon, np.ndarray):
			if outer_polygon.size == 1:
				outer_polygon = outer_polygon[0]
			else:
				# If there are multiple elements, combine them into one geometry
				outer_polygon = unary_union(list(outer_polygon))

			print("outer_polygon type:", type(outer_polygon))

		inside_parcels = inside_parcels[
			inside_parcels.geometry.centroid.within(outer_polygon)
		]

		count3 = len(inside_parcels)

		print(f" {idx} --> {count1} parcels inside the envelope, {count2} after excluding the COMMON AREA, {count3} after excluding those not wholly inside the COMMON AREA")


		# If it's empty, continue:
		if inside_parcels.empty:
			continue

		# Check that at least one of the inside_parcels matches the target model_group_id, otherwise continue:
		if not inside_parcels["model_group"].eq(model_group_id).any():
			continue

		# Determine the modal model_group value
		modal_model_group = inside_parcels["model_group"].value_counts().index[0]
		if modal_model_group is not None and modal_model_group != "":
			print(f" {idx} --> modal model group = {modal_model_group} for {len(inside_parcels)} inside parcels")
			# Apply the modal model_group to the COMMON AREA parcel
			df.at[idx, "model_group"] = modal_model_group
			df.at[idx, "modal_tagged"] = True
		else:
			print(f" {idx} --> XXX modal model group is {modal_model_group} for {len(inside_parcels)} inside parcels")

	df.to_parquet("out/look/common_area-2-tagged.parquet")

	df_return = df_univ_in.copy()

	# Update and return df_univ

	df_return = combine_dfs(df_return, df[["key", "model_group"]], df2_stomps=True, index="key")

	df_return.to_parquet("out/look/common_area-3-return.parquet")

	return df_return