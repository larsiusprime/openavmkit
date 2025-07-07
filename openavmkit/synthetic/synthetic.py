import random
import warnings
from typing import Any

import openavmkit
import geopandas as gpd
from openavmkit.utilities.geometry import (
    create_geo_rect_shape_km,
    offset_coordinate_m,
    offset_coordinate_feet,
    create_geo_rect_shape_deg,
)


def generate_building(self, parcel, utilization, depreciation, year, is_new=True):
    density = parcel["density"]
    land_use = parcel["land_use"]
    max_floors = parcel["max_floors"]
    lot_coverage = parcel["lot_coverage"]
    apartments_allowed = parcel["apartments_allowed"]
    is_cbd = parcel["is_cbd"]

    # Building type
    building_type = ""

    if land_use == "R":
        if density == 0:
            building_type = "single_family"
        elif density == 1:
            building_type = "duplex"
        elif density == 2:
            building_type = "townhouse"
        elif density == 3:
            if apartments_allowed and random.random() < 0.5:
                building_type = "apartments"
            else:
                building_type = "townhouse"
        elif density == 4:
            if apartments_allowed:
                building_type = "apartments"
            else:
                building_type = "townhouse"
    elif land_use == "M":  # mixed-use
        if density == 0:
            building_type = "single_family"
        elif density == 1:
            if random.random() < 0.5:
                building_type = "townhouse"
            else:
                building_type = "mixed_retail"
        elif density == 2:
            if random.random() < 0.5:
                building_type = "mixed_retail"
            else:
                building_type = "commercial"
        elif density == 3:
            if apartments_allowed and random.random() < 0.5:
                building_type = "apartments"
            else:
                if random.random() < 0.5:
                    building_type = "mixed_retail"
                else:
                    building_type = "commercial"
        elif density == 4:
            if apartments_allowed:
                building_type = "apartments"
            else:
                building_type = "commercial"
    elif land_use == "C":  # commercial
        building_type = "commercial"
    elif land_use == "I":  # industrial
        building_type = "industrial"
    elif land_use == "CBD":
        building_type = "commercial"

    # Building size
    if self.units == "imperial":
        area_land_size = parcel["area_land_sqft"]
    else:
        area_land_size = parcel["area_land_sqm"]

    footprint_size = area_land_size * lot_coverage
    num_floors = max(1, int(max_floors * utilization))
    area_impr_finished_size = footprint_size * num_floors

    parcel["bldg_stories"] = num_floors

    if self.units == "imperial":
        parcel["area_impr_finished_sqft"] = area_impr_finished_size
        parcel["area_impr_footprint_sqft"] = footprint_size
    else:
        parcel["area_impr_finished_sqm"] = area_impr_finished_size
        parcel["area_impr_footprint_sqm"] = footprint_size

    # Building age & condition
    if is_new:
        parcel["bldg_year_built"] = year
        parcel["bldg_condition_num"] = 1.0
    else:
        # assume 1-20 years old
        age = random.randrange(1, 20)
        parcel["bldg_year_built"] = parcel["bldg_year_built"] - age
        parcel["bldg_condition_num"] = 1.0 - depreciation[land_use][age]

    # Building quality
    # needs to be semi-correlated with land value
    quality = 3.0

    if is_cbd:
        quality += 2.0

    parcel["bldg_quality_num"] = quality
    parcel["bldg_quality_txt"] = {
        0: "0 - none",
        1: "1 - poor",
        2: "2 - fair",
        3: "3 - average",
        4: "4 - good",
        5: "5 - very good",
        6: "6 - excellent",
    }[int(quality)]

    # Building quality


# Brazos county facts:
# - CBD block:
#   - 350x375 feet
#   - 5-10 parcels per block
# - Residential parcel:
#   - 150-160 x 50-80 feet
#   - 6-10 parcels per "strip"
#   - 2:3 aspect ratio


def make_geo_blocks(
    latitude, longitude, block_size_y, block_size_x, blocks: list, units: str, crs: Any
) -> gpd.GeoDataFrame:
    blocks = make_geo_blocks_raw(
        latitude, longitude, block_size_y, block_size_x, blocks, units
    )
    gdf = gpd.GeoDataFrame(data=blocks, geometry="geometry", crs="EPSG:4326")
    # We supress this warning because actually we DO want the centroid in latitude/longitude here!
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    gdf["latitude"] = gdf["geometry"].centroid.y
    gdf["longitude"] = gdf["geometry"].centroid.x
    warnings.filterwarnings("default", category=UserWarning, append=True)
    gdf = gdf.to_crs(crs)
    return gdf


def make_geo_blocks_raw(
    latitude, longitude, block_size_y, block_size_x, blocks: list, units: str
) -> list:
    o_y = latitude
    o_x = longitude

    y_size = block_size_y
    x_size = block_size_x

    if units == "ft":
        km_per_ft = 0.0003048
        offset_func = offset_coordinate_feet
        x_size_km = x_size * km_per_ft
        y_size_km = y_size * km_per_ft
    elif units == "m":
        offset_func = offset_coordinate_m
        km_per_m = 0.001
        x_size_km = x_size * km_per_m
        y_size_km = y_size * km_per_m
    else:
        raise ValueError(f"Unsupported units: {units}")

    for block in blocks:
        x = block["x"]
        y = block["y"]
        x_offset = x * x_size
        y_offset = y * y_size
        pt_y, pt_x = offset_func(o_y, o_x, y_offset, x_offset)
        geo = create_geo_rect_shape_km(pt_y, pt_x, x_size_km, y_size_km, "nw")
        block["geometry"] = geo

    return blocks


def make_geo_parcels(
    latitude, longitude, block_size_y, block_size_x, blocks: list, units: str, crs: Any
) -> gpd.GeoDataFrame:
    blocks = make_geo_blocks(
        latitude, longitude, block_size_y, block_size_x, blocks, units, crs
    )
    parcels = make_geo_parcels_raw(blocks, units, crs)
    return parcels


def make_geo_parcels_raw(
    blocks: gpd.GeoDataFrame, units: str, crs: Any
) -> gpd.GeoDataFrame | None:

    parcels = []

    # iterate through rows of blocks:
    for i, block in blocks.iterrows():
        # get the geometry of the block
        geo = block["geometry"]

        # get the upper left hand coordinate of the block's bounding box:
        origin_x, origin_y = geo.bounds[0], geo.bounds[3]
        max_width = geo.bounds[2] - geo.bounds[0]
        max_height = geo.bounds[3] - geo.bounds[1]

        rows_per_block = block["rows_per_block"]
        parcels_per_row = block["parcels_per_row"]

        parcel_width = max_width / rows_per_block
        parcel_height = max_height / parcels_per_row

        if i % 1000 == 0:
            perc = i / len(blocks)
            print(
                f"{perc:6.2%} Building parcels for block {i} ({block['x']}, {block['y']})"
            )

        print(
            f"Block: {i}, X: {block['x']}, Y: {block['y']}, rows_per_block = {rows_per_block}"
        )

        for row in range(0, rows_per_block):
            for parcel in range(0, parcels_per_row):

                # get the coordinates of the parcel
                x = origin_x + (row * parcel_width)
                y = origin_y - (parcel * parcel_height)

                print(f"--> Parcel: {parcel}, Row: {row}, X: {x}, Y: {y}")

                # create a rectangle for the parcel
                rect = create_geo_rect_shape_deg(
                    y, x, parcel_width, parcel_height, "nw"
                )

                minx = rect.bounds[0]
                miny = rect.bounds[1]
                maxx = rect.bounds[2]
                maxy = rect.bounds[3]

                longitude = (minx + maxx) / 2
                latitude = (miny + maxy) / 2

                block_x = block["x"]
                block_y = block["y"]
                block_name = f"{block_x}_x_{block_y}"

                key = f"{block_x}-{block_y}-{row}-{parcel}"

                road_type_w = block["road_type_w"]
                road_type_e = block["road_type_e"]
                road_type_n = block["road_type_n"]
                road_type_s = block["road_type_s"]

                road_type = 0
                is_corner_lot = False
                corner_lot_type = 0
                road_type_ew = 0
                road_type_ns = 0

                if rows_per_block == 1:
                    # both west & east side
                    road_type_ew = max(road_type_w, road_type_e)
                else:
                    if row == 0:
                        # west side
                        road_type_ew = max(road_type_w, road_type_ew)
                    if row == rows_per_block - 1:
                        # east side
                        road_type_ew = max(road_type_e, road_type_ew)

                if parcels_per_row == 1:
                    # both north & south side
                    road_type_ns = max(road_type_n, road_type_s)
                else:
                    if parcel == 0:
                        # north side
                        road_type_ns = max(road_type_n, road_type_ns)
                    if parcel == parcels_per_row - 1:
                        # south side
                        road_type_ns = max(road_type_s, road_type_ns)

                road_type = max(road_type_ew, road_type_ns)

                print(
                    f"--> road_type = {road_type}, road_type_ew = {road_type_ew}, road_type_ns = {road_type_ns}"
                )
                print(f"----> w = {road_type_w}")
                print(f"----> e = {road_type_e}")

                if road_type_ew > 0 and road_type_ns > 0:
                    is_corner_lot = True
                    corner_lot_type = max(road_type_ew, road_type_ns)

                land_use = block["land_use"]
                density = block["density"]
                max_floors = block["max_floors"]
                lot_coverage = block["lot_coverage"]
                apartments_allowed = land_use in ["C", "CBD"]
                zoning = block["zoning"]

                # if it's on a major road, then some additional density is allowed
                if road_type == 3:
                    density += 1
                    density = min(density, 4)
                    zoning = _get_zoning_name(land_use, density, block["cbd"])

                    # density bump along streets only applies to floor limit and lot coverage
                    _, _, max_floors, lot_coverage = _get_density_stats(density)

                    # apartments in mixed-use zones only allowed along major roads
                    if land_use in ["M"]:
                        apartments_allowed = True

                parcel = {
                    "key": key,
                    "block": block_name,
                    "zoning": zoning,
                    "land_use": block["land_use"],
                    "density": density,
                    "neighborhood": block["neighborhood"],
                    "district": block["district"],
                    "sector": block["sector"],
                    "max_floors": max_floors,
                    "lot_coverage": lot_coverage,
                    "apartments_allowed": apartments_allowed,
                    "road_type": road_type,
                    "is_corner_lot": is_corner_lot,
                    "corner_lot_type": corner_lot_type,
                    "latitude": latitude,
                    "longitude": longitude,
                    "geometry": rect,
                }
                parcels.append(parcel)

    gdf = gpd.GeoDataFrame(data=parcels, geometry="geometry", crs=crs)

    crs = openavmkit.utilities.geometry.get_crs(gdf, "equal_area")
    gdf[f"area_land_sqm"] = gdf.to_crs(crs).geometry.area

    if units == "ft":
        gdf[f"area_land_sqft"] = gdf[f"area_land_sqm"] * 10.7639
        gdf.drop(columns=["area_land_sqm"], inplace=True)

    return gdf


def _get_street_name(x: int, y: int, is_vertical: bool):
    if is_vertical:
        return f"{_ordinalize(x+1)} Avenue"
    else:
        return f"{_ordinalize(y+1)} Street"


def _ordinalize(n: int) -> str:
    if n % 10 == 1:
        return "1st"
    if n % 10 == 2:
        return "2nd"
    if n % 10 == 3:
        return "3rd"
    return f"{n}th"


def _get_zoning_name(land_use: str, density: int, in_cbd: bool):
    zoning = f"{land_use}{density}"
    if in_cbd:
        zoning = "CBD"
    return zoning


def _get_density_stats(density: int) -> tuple:

    if density == 0:
        rows_per_block = 1
        parcels_per_row = 2
        max_floors = 2
        lot_coverage = 0.25
    elif density == 1:
        rows_per_block = 1
        parcels_per_row = 4
        max_floors = 2
        lot_coverage = 0.50
    elif density == 2:
        rows_per_block = 2
        parcels_per_row = 5
        max_floors = 4
        lot_coverage = 0.50
    elif density == 3:
        rows_per_block = 2
        parcels_per_row = 8
        max_floors = 6
        lot_coverage = 0.75
    elif density == 4:
        rows_per_block = 2
        parcels_per_row = 8
        max_floors = 10
        lot_coverage = 0.90
    else:
        raise ValueError(f"Invalid density: {density}")

    return rows_per_block, parcels_per_row, max_floors, lot_coverage
