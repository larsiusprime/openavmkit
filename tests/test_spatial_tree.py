import os
import geopandas as gpd
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from PIL import ImageDraw, Image
from shapely import wkb

from openavmkit.checkpoint import read_checkpoint
from openavmkit.data import get_vacant_sales
from openavmkit.models.spatial_tree import KDTreeNode, build_kdtree, visualize_kdtree
from openavmkit.utilities.geometry import get_crs
from openavmkit.utilities.settings import load_settings


def test_spatial_tree():

  return True

  # print("")
  # # set working directory to the library's root/tests/data:
  # os.chdir("../notebooks/data/nc-guilford")
  #
  # # load the settings:
  # settings = load_settings()
  #
  # # df = pd.read_parquet("out/checkpoints/2-clean-03-out.parquet")
  # # df['geometry'] = df['geometry'].apply(wkb.loads)
  # # gdf = gpd.GeoDataFrame(df, geometry='geometry')
  # # gdf.set_crs(epsg=4326, inplace=True)
  #
  # df:gpd.GeoDataFrame = read_checkpoint("2-clean-03-out")
  #
  # crs_equal_distance = get_crs(df, "equal_distance")
  # print(f"CRS -->: {crs_equal_distance}")
  #
  # df = df.to_crs(crs_equal_distance)
  # #df = get_vacant_sales(df, settings)
  # ind_var = "sale_price_time_adj"
  # size_var = "land_area_sqft"
  #
  # df = df[
  #   df[ind_var].gt(0) &
  #   df[size_var].gt(0)
  # ].reset_index(drop=True)
  #
  # display(df)
  #
  #
  # print(f"CRS now: {df.crs}")
  #
  # # Build the Root Node (full bounding box)
  # min_y, max_y = df.bounds['miny'].min(), df.bounds['maxy'].max()
  # min_x, max_x = df.bounds['minx'].min(), df.bounds['maxx'].max()
  #
  # print(f"min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}")
  #
  # all_indices = np.arange(len(df))
  # root_node = KDTreeNode(min_x, max_x, min_y, max_y, all_indices, depth=0)
  #
  # print(f"Root Node: {root_node}")
  #
  # min_samples = 100
  # max_depth = 6
  #
  # build_kdtree(df, root_node, ind_var, size_var, min_samples, max_depth)
  #
  # visualize_kdtree(root_node, 1000, 1000)
  # print("DONE")
