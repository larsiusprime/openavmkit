import math

import pandas as pd

from openavmkit.utilities.clustering import make_clusters, _crunch


def test_crunch_falls_back_to_coarser_bin_counts():
  """When the finest split leaves a bin under min_count, try the coarser levels.

  _crunch walks crunch_levels from 5 bins down to 2. Every branch of that loop used
  to either return or break, so levels 2 and 3 were unreachable under every input and
  the "dynamically adapts" promise in the docstring never held.
  """
  df = pd.DataFrame({"f": [float(i) for i in range(1000)]})

  # min_count small enough for the 5-bin split (200 per bin)
  assert _crunch(df, "f", min_count=50).nunique() == 5

  # too big for 5 bins (200 each) but fine for 2 bins (500 each): must fall back
  # rather than give up entirely
  coarser = _crunch(df, "f", min_count=300)
  assert coarser is not None, "should have fallen back to a coarser split"
  assert coarser.nunique() == 2

  # no level can satisfy this, so giving up is correct
  assert _crunch(df, "f", min_count=600) is None


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