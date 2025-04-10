import math

import pandas as pd

from openavmkit.utilities.clustering import add_to_cluster_dict, make_clusters


def test_cluster_dict():

  data = {}
  data["key"] = [i for i in range(0, 100)]
  data["hood"] = [i % 2 for i in range(0, 100)]
  data["size"] = [i for i in range(0, 100)]
  data["size_label"] = [math.floor(i/50)*50 for i in range(0, 100)]
  data["color"] = [i % 3 for i in range(0, 100)]

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

  cluster_dict = {}

  cluster_dict = add_to_cluster_dict(cluster_dict, "location", "hood", 0, df)
  cluster_dict = add_to_cluster_dict(cluster_dict, "categorical", "color", 1, df)
  cluster_dict = add_to_cluster_dict(cluster_dict, "numeric", "size_label", 2, df, "size")

  expected = {'index':{},'iterations':{'0':{'North':[{'f':'hood','t':'l','v':'North'}],'South':[{'f':'hood','t':'l','v':'South'}]},'1':{'North_Blue':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Blue'}],'North_Green':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Green'}],'North_Red':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Red'}],'South_Blue':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Blue'}],'South_Green':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Green'}],'South_Red':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Red'}]},'2':{'North_Blue_0':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Blue'},{'f':'size','n':0,'t':'n','v':[0,49]}],'North_Blue_50':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Blue'},{'f':'size','n':50,'t':'n','v':[50,99]}],'North_Green_0':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Green'},{'f':'size','n':0,'t':'n','v':[0,49]}],'North_Green_50':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Green'},{'f':'size','n':50,'t':'n','v':[50,99]}],'North_Red_0':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Red'},{'f':'size','n':0,'t':'n','v':[0,49]}],'North_Red_50':[{'f':'hood','t':'l','v':'North'},{'f':'color','t':'c','v':'Red'},{'f':'size','n':50,'t':'n','v':[50,99]}],'South_Blue_0':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Blue'},{'f':'size','n':0,'t':'n','v':[0,49]}],'South_Blue_50':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Blue'},{'f':'size','n':50,'t':'n','v':[50,99]}],'South_Green_0':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Green'},{'f':'size','n':0,'t':'n','v':[0,49]}],'South_Green_50':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Green'},{'f':'size','n':50,'t':'n','v':[50,99]}],'South_Red_0':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Red'},{'f':'size','n':0,'t':'n','v':[0,49]}],'South_Red_50':[{'f':'hood','t':'l','v':'South'},{'f':'color','t':'c','v':'Red'},{'f':'size','n':50,'t':'n','v':[50,99]}]}}}

  assert cluster_dict == expected


def test_make_clusters():
  data = {}
  data["key"] = [i for i in range(0, 50)]
  data["hood"] = [i % 2 for i in range(0, 50)]
  data["size"] = [i for i in range(0, 50)]
  data["color"] = [i % 3 for i in range(0, 50)]

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

  ids, fields_used, cluster_dict = make_clusters(
    df,
    field_location="hood",
    fields_categorical=[],
    fields_numeric=["size"],
    min_cluster_size=5
  )

  print("")
  print(cluster_dict)