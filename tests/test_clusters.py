import math

import pandas as pd

from openavmkit.utilities.clustering import make_clusters


def test_make_clusters():
  data = {}
  data["key"] = [i for i in range(0, 500)]
  data["hood"] = [i % 2 for i in range(0, 500)]
  data["size"] = [i for i in range(0, 500)]
  data["color"] = [i % 3 for i in range(0, 500)]

  locations = {
    "0": "North",
    "1": "South",
  }
  colors = {
    "0": "Red",
    "1": "Green",
    "2": "Blue",
  }
  df = pd.DataFrame(data=data)
  df["hood"] = df["hood"].astype(str).map(locations)
  df["color"] = df["color"].astype(str).map(colors)

  ids, fields_used, clusters = make_clusters(
    df,
    field_location="hood",
    fields_categorical=["color"],
    fields_numeric=["size"],
    min_cluster_size=5
  )

  ids_list = ids.tolist()