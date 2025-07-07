import random
import geopandas as gpd

from shapely.geometry import LineString

from openavmkit.synthetic.basic import generate_depreciation_curve
from openavmkit.synthetic.synthetic import (
    make_geo_parcels,
    make_geo_blocks,
    _get_zoning_name,
    _get_density_stats,
    _get_street_name,
    generate_building,
)
from openavmkit.utilities.geometry import offset_coordinate_feet, offset_coordinate_m


class SynCity:

    def __init__(self, params):
        self.latitude = params["latitude"]
        self.longitude = params["longitude"]
        self.crs = params["crs"]
        self.anchor = params["anchor"]
        self.units = params["units"]

        self.cbd_width_in_blocks = params["cbd_width_in_blocks"]
        self.cbd_height_in_blocks = params["cbd_height_in_blocks"]

        if self.units == "imperial":
            self.base_parcel_size = (150, 100)
        elif self.units == "metric":
            self.base_parcel_size = (45, 30)
        else:
            raise ValueError(f"Invalid units: {self.units}")

        self.width_in_blocks = params["width_in_blocks"]
        self.height_in_blocks = params["height_in_blocks"]

        self.max_width = self.base_parcel_size[0] * self.width_in_blocks
        self.max_height = self.base_parcel_size[1] * self.height_in_blocks

        self.gdf_roads: gpd.GeoDataFrame | None = None
        self.gdf_blocks: gpd.GeoDataFrame | None = None
        self.gdf_parcels: gpd.GeoDataFrame | None = None

        self.setup()

    def setup(self):
        # Set basic block size
        self.rows_per_block = 2
        self.parcels_per_row = 5

        self.block_size = (
            self.base_parcel_size[0] * self.rows_per_block,
            self.base_parcel_size[1] * self.parcels_per_row,
        )

        self.width_in_blocks = (
            self.width_in_blocks
        )  # int(self.max_width/self.block_size[0])
        self.height_in_blocks = (
            self.height_in_blocks
        )  # int(self.max_height/self.block_size[1])
        print(f"Width: {self.width_in_blocks}, Height: {self.height_in_blocks}")

        self.build_grid()

    def build_grid(self):

        self.medium_road_every_n_blocks = 8
        self.large_road_every_n_blocks = 32
        self.sector_every_n_blocks = 48

        blocks = []
        roads = []

        # Determine bounds of CBD
        cbd_center_x = int(self.width_in_blocks / 2)
        cbd_center_y = int(self.height_in_blocks / 2)
        cbd_start_x = cbd_center_x - int(self.cbd_width_in_blocks / 2)
        cbd_start_y = cbd_center_y - int(self.cbd_height_in_blocks / 2)
        cbd_end_x = cbd_center_x + int(self.cbd_width_in_blocks / 2)
        cbd_end_y = cbd_center_y + int(self.cbd_height_in_blocks / 2)

        # Determine locations of large roads:
        # - around the CBD
        # - at the edges of the city
        # - halfway between the CBD and the edges of the city

        mid_x_w = int(cbd_start_x / 2)
        mid_x_e = int((self.width_in_blocks - cbd_end_x) / 2)
        mid_x_n = int(cbd_start_x + (self.width_in_blocks - cbd_end_x) / 2)
        mid_x_s = int(cbd_start_x + (self.width_in_blocks - cbd_end_x) / 2)

        eighth_width = int(self.width_in_blocks / 8)
        eighth_height = int(self.height_in_blocks / 8)
        quarter_width = int(self.width_in_blocks / 4)
        quarter_height = int(self.height_in_blocks / 4)

        parcels_per_row = self.parcels_per_row
        rows_per_block = self.rows_per_block
        max_floors = 1

        # Append blocks
        for x in range(self.width_in_blocks):
            for y in range(self.height_in_blocks):
                in_cbd = False

                medium_roads = (
                    (x % self.medium_road_every_n_blocks == 0)
                    or ((x + 1) % self.medium_road_every_n_blocks == 0)
                ) + (
                    (y % self.medium_road_every_n_blocks == 0)
                    or ((y + 1) % self.medium_road_every_n_blocks == 0)
                )

                large_roads = (
                    (x % self.large_road_every_n_blocks == 0)
                    or ((x + 1) % self.large_road_every_n_blocks == 0)
                ) + (
                    (y % self.large_road_every_n_blocks == 0)
                    or ((y + 1) % self.large_road_every_n_blocks == 0)
                )

                road_type_n = 1
                road_type_s = 1
                road_type_w = 1
                road_type_e = 1

                on_medium_road = False
                on_large_road = False

                if x % self.medium_road_every_n_blocks == 0:
                    road_type_w = 2
                    on_medium_road = True
                if x % self.large_road_every_n_blocks == 0:
                    road_type_w = 3
                    on_large_road = True

                if (x + 1) % self.medium_road_every_n_blocks == 0:
                    road_type_e = 2
                    on_medium_road = True
                if (x + 1) % self.large_road_every_n_blocks == 0:
                    road_type_e = 3
                    on_large_road = True

                if y % self.medium_road_every_n_blocks == 0:
                    road_type_s = 2
                    on_medium_road = True
                if y % self.large_road_every_n_blocks == 0:
                    road_type_s = 3
                    on_large_road = True

                if (y + 1) % self.medium_road_every_n_blocks == 0:
                    road_type_n = 2
                    on_medium_road = True
                if (y + 1) % self.large_road_every_n_blocks == 0:
                    road_type_n = 3
                    on_large_road = True

                neighborhood_x = int(x / self.medium_road_every_n_blocks)
                neighborhood_y = int(y / self.medium_road_every_n_blocks)

                district_x = int(x / self.large_road_every_n_blocks)
                district_y = int(y / self.large_road_every_n_blocks)

                # get half the remainder between width in blocks and sector size
                sector_offset_x = int(
                    (self.width_in_blocks % self.sector_every_n_blocks)
                )
                sector_offset_y = int(
                    (self.height_in_blocks % self.sector_every_n_blocks)
                )

                sector_x = int((x + sector_offset_x) / (self.sector_every_n_blocks))
                sector_y = int((y + sector_offset_y) / (self.sector_every_n_blocks))

                neighborhood = f"{neighborhood_x}_x_{neighborhood_y}"
                district = f"{district_x}_x_{district_y}"
                sector = f"{sector_x}_x_{sector_y}"

                dist_to_cbd_x = abs(x - cbd_center_x)
                dist_to_cbd_y = abs(y - cbd_center_y)

                density = 1

                if dist_to_cbd_x < eighth_width and dist_to_cbd_y < eighth_height:
                    density = 3
                elif dist_to_cbd_x < quarter_width and dist_to_cbd_y < quarter_height:
                    density = 2

                if medium_roads < 1 and large_roads == 0:
                    land_use = "R"
                    density -= 1
                elif medium_roads < 2 and large_roads < 2:
                    land_use = "M"
                    density -= 1
                else:
                    if on_large_road:
                        land_use = "C"
                    else:
                        land_use = "M"

                if (x >= cbd_start_x and x < cbd_end_x) and (
                    y >= cbd_start_y and y < cbd_end_y
                ):
                    in_cbd = True
                    land_use = "CBD"
                    density = 4
                if (
                    in_cbd is False
                    and x >= cbd_end_x
                    and y >= cbd_start_y
                    and y < cbd_end_y
                ):
                    if land_use in ["R", "M", "C"]:
                        land_use = "I"
                        density += 1

                # adjacent to CBD
                if (x == cbd_start_x - 1 or x == cbd_end_x) and (
                    y == cbd_start_y - 1 or y == cbd_end_y
                ):
                    if land_use == "R":
                        land_use = "M"

                density = min(density, 4)

                zoning = _get_zoning_name(land_use, density, in_cbd)
                if in_cbd:
                    neighborhood = f"CBD_{neighborhood}"
                    district = f"CBD_{district}"
                    sector = f"CBD"

                rows_per_block, parcels_per_row, max_floors, lot_coverage = (
                    _get_density_stats(density)
                )

                block = {
                    "x": x,
                    "y": y,
                    "cbd": in_cbd,
                    "zoning": zoning,
                    "land_use": land_use,
                    "density": density,
                    "neighborhood": neighborhood,
                    "district": district,
                    "sector": sector,
                    "parcels_per_row": parcels_per_row,
                    "rows_per_block": rows_per_block,
                    "max_floors": max_floors,
                    "lot_coverage": lot_coverage,
                    "road_type_n": road_type_n,
                    "road_type_s": road_type_s,
                    "road_type_w": road_type_w,
                    "road_type_e": road_type_e,
                }
                blocks.append(block)

        # Append vertical rows
        for x in range(self.width_in_blocks + 1):
            x_type = 1
            if x % self.medium_road_every_n_blocks == 0:
                x_type = 2
            if x % self.large_road_every_n_blocks == 0:
                x_type = 3

            road = {
                "x": x,
                "y": -1,
                "road_type": x_type,
                "is_vertical": True,
                "name": _get_street_name(x, -1, True),
            }
            roads.append(road)

        # Append horizontal rows
        for y in range(self.height_in_blocks + 1):
            y_type = 1
            if y % self.medium_road_every_n_blocks == 0:
                y_type = 2
            if y % self.large_road_every_n_blocks == 0:
                y_type = 3

            road = {
                "x": -1,
                "y": y,
                "road_type": y_type,
                "is_vertical": False,
                "name": _get_street_name(-1, y, False),
            }
            roads.append(road)

        units = "m"
        if self.units == "imperial":
            units = "ft"

        self.gdf_blocks = make_geo_blocks_from_city(self, blocks, units)
        self.gdf_roads = make_geo_roads_from_city(self, roads, units)
        self.gdf_parcels = make_geo_parcels_from_city(self, blocks, units)

        # self.evolve_city()

    def evolve_city(self):

        self.quick_and_dirty_fill()

        # Fill out all buildings according to master plan as of 100 years ago
        # generate inflation curves for land, building, and city growth
        # generate transactions each year
        # depreciate buildings each year
        # tear down buildings each year according to building actuarial table
        # generate new construction each year according to building rate curve
        # favor new construction in highest value areas
        # build up to max density allowed, then sprawl out demand

    def quick_and_dirty_fill(self):

        # annual inflation rates
        inflation_impr_rate = 0.02  # construction costs inflate at 2% per year
        inflation_land_rate = 0.03  # land values inflate at 3% per year

        # annual depreciation rates
        depreciation = {
            "residential": generate_depreciation_curve(
                lifetime=60,
                weight_linear=0.2,
                weight_logistic=0.8,
                steepness=0.3,
                inflection_point=20,
            ),
            "apartments": generate_depreciation_curve(
                lifetime=60,
                weight_linear=0.3,
                weight_logistic=0.7,
                steepness=0.25,
                inflection_point=20,
            ),
            "commercial": generate_depreciation_curve(
                lifetime=50,
                weight_linear=0.5,
                weight_logistic=0.5,
                steepness=0.35,
                inflection_point=15,
            ),
            "industrial": generate_depreciation_curve(
                lifetime=60,
                weight_linear=0.4,
                weight_logistic=0.6,
                steepness=0.3,
                inflection_point=25,
            ),
        }
        sales = {
            "residential": 0.05,
            "apartments": 0.025,
            "commercial": 0.025,
            "industrial": 0.025,
        }

        parcels = self.gdf_parcels

        percent_vacant = 0.1

        random.seed(1337)

        parcels["is_vacant"] = False
        parcels["location_value"] = 0.0

        for i, parcel in parcels.iterrows():

            density = parcel["density"]
            land_use = parcel["land_use"]

            chance_vacant = percent_vacant * (0.9**density)

            # decide if it's vacant or not
            if random.random() < chance_vacant:
                is_vacant = True
            else:
                is_vacant = False
                parcel = generate_building(parcel)

            # calculate distance to amenities and add bonuses

            # calculate distance to nuisances and add penalties

            pass


def _get_draw_anchor_coords(city: SynCity):
    if city.units == "imperial":
        offset_func = offset_coordinate_feet
    elif city.units == "metric":
        offset_func = offset_coordinate_m

    if city.anchor == "center":
        # coordinate represents the center of the city
        lat, lon = offset_func(
            city.latitude, city.longitude, -city.max_height / 2, -city.max_width / 2
        )
    else:
        raise ValueError(f"Unsupported anchor: {city.anchor}")

    return lat, lon


def make_geo_roads_from_city(
    city: SynCity, roads: list, units: str
) -> gpd.GeoDataFrame:

    o_y, o_x = _get_draw_anchor_coords(city)

    x_length = city.block_size[0] * (city.width_in_blocks)
    y_length = city.block_size[1] * (city.height_in_blocks)

    offset_func = None
    if units == "ft":
        offset_func = offset_coordinate_feet
    elif units == "m":
        offset_func = offset_coordinate_m

    for road in roads:
        if road["is_vertical"]:
            x = road["x"]
            y = 0
        else:
            x = 0
            y = road["y"]

        x_offset = x * city.block_size[0]
        y_offset = (y - 1) * city.block_size[1]

        pt_y, pt_x = offset_func(o_y, o_x, y_offset, x_offset)

        lat_1 = pt_y
        lon_1 = pt_x

        if road["is_vertical"]:
            lat_2, lon_2 = offset_func(pt_y, pt_x, y_length, 0)
        else:
            lat_2, lon_2 = offset_func(pt_y, pt_x, 0, x_length)

        geo = LineString([(lon_1, lat_1), (lon_2, lat_2)])
        road["geometry"] = geo

    gdf = gpd.GeoDataFrame(data=roads, geometry="geometry", crs=city.crs)
    return gdf


def make_geo_parcels_from_city(
    city: SynCity, blocks: list, units: str
) -> gpd.GeoDataFrame:
    o_y, o_x = _get_draw_anchor_coords(city)
    return make_geo_parcels(
        o_y, o_x, city.block_size[1], city.block_size[0], blocks, units, city.crs
    )


def make_geo_blocks_from_city(
    city: SynCity, blocks: list, units: str
) -> gpd.GeoDataFrame:
    o_y, o_x = _get_draw_anchor_coords(city)
    return make_geo_blocks(
        o_y, o_x, city.block_size[1], city.block_size[0], blocks, units, city.crs
    )
