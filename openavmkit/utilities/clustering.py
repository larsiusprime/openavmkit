import pandas as pd


def resolve_cluster_dict(
    cluster_dict: dict
) -> dict:
  final_dict = {}
  for key in cluster_dict["index"]:
    id = cluster_dict["index"][key]
    found = False
    for iteration in cluster_dict["iterations"]:
      entries = cluster_dict["iterations"][iteration]
      for entry_key in entries:
        if entry_key == key:
          found = True
          final_dict[id] = {
            "name": entry_key,
            "iteration": iteration,
            "clusters": entries[entry_key]
          }
          break
      if found:
        break
    if not found:
      final_dict[id] = {
        "name": "???",
        "iteration": -1,
        "clusters": []
      }

  return final_dict



def add_to_cluster_dict(
    cluster_dict: dict,
    type: str,
    field: str,
    iteration: int,
    df: pd.DataFrame,
    field_raw: str = ""
) -> dict:
  if type not in ["numeric", "categorical", "location", "boolean"]:
    raise ValueError(f"Invalid type: {type}")

  type_code = {
    "numeric": "n",
    "categorical": "c",
    "location": "l",
    "boolean": "b"
  }[type]

  # check if cluster_dict is empty or {}:
  if not cluster_dict:
    cluster_dict = {"iterations":{}, "index":{}}

  last_iteration = str(iteration-1)
  if last_iteration in cluster_dict["iterations"]:
    old_dict = cluster_dict["iterations"][last_iteration]
    old_keys = list(old_dict.keys())
  else:
    old_dict = {"":[]}
    old_keys = [""]

  new_dict = {}

  unique_values = df[field].unique()

  for old_key in old_keys:
    old_list = old_dict[old_key]
    for unique_value in unique_values:
      new_list = old_list.copy()
      entry = {
        "t": type_code,
        "f": field,
        "v": unique_value
      }
      if type == "numeric":
        df_cluster = df[df[field].eq(unique_value)]
        min_value = df_cluster[field_raw].min()
        max_value = df_cluster[field_raw].max()
        entry["f"] = field_raw
        entry["v"] = [min_value, max_value]
        entry["n"] = unique_value

      new_list.append(entry)
      if old_key == "":
        new_key = str(unique_value)
      else:
        new_key = str(old_key) + "_" + str(unique_value)
      new_dict[new_key] = new_list
  cluster_dict["iterations"][str(iteration)] = new_dict
  return cluster_dict


def make_clusters(
    df_in: pd.DataFrame,
    field_location: str|None,
    fields_categorical: list[str],
    fields_numeric: list[str | list[str]] = None,
    min_cluster_size: int = 15,
    verbose: bool = False,
    output_folder: str = ""
):
  df = df_in.copy()

  iteration = 0
  # We are assigning a unique id to each cluster

  cluster_dict = {}

  # Phase 1: split the data into clusters based on the location:
  if field_location is not None and field_location in df:
    df["cluster"] = df[field_location].astype(str)
    cluster_dict = add_to_cluster_dict(cluster_dict, "location", field_location, iteration, df)
    if verbose:
      print(f"--> crunching on location, {len(df['cluster'].unique())} clusters")
  else:
    df["cluster"] = ""

  fields_used = {}

  # Phase 2: split into vacant and improved:
  if "is_vacant" in df:
    df["cluster"] = df["cluster"] + "_" + df["is_vacant"].astype(str)
    cluster_dict = add_to_cluster_dict(cluster_dict, "boolean", "is_vacant", iteration, df)
    if verbose:
      print(f"--> crunching on is_vacant, {len(df['cluster'].unique())} clusters")

  # Phase 3: add to the cluster based on each categorical field:
  for field in fields_categorical:
    if field in df:
      df["cluster"] = df["cluster"] + "_" + df[field].astype(str)
      iteration+=1
      cluster_dict = add_to_cluster_dict(cluster_dict, "categorical", field, iteration, df)
      fields_used[field] = True

  if fields_numeric is None or len(fields_numeric) == 0:
    fields_numeric = [
      "land_area_sqft",
      "bldg_area_finished_sqft",
      "bldg_quality_num",
      ["bldg_effective_age_years", "bldg_age_years"], # Try effective age years first, then normal age
      "bldg_condition_num"
    ]

  # Phase 4: iterate over numeric fields, trying to crunch down whenever possible:
  for entry in fields_numeric:

    iteration+=1
    # get all unique clusters
    clusters = df["cluster"].unique()

    # store the base for the next iteration as the current cluster
    df["next_cluster"] = df["cluster"]

    if verbose:
      print(f"--> crunching on {entry}, {len(clusters)} clusters")

    i = 0
    # step through each unique cluster:
    for cluster in clusters:

      # get all the rows in this cluster
      mask = df["cluster"].eq(cluster)
      df_sub = df[mask]

      len_sub = mask.sum()

      # if the cluster is already too small, skip it
      if len_sub < min_cluster_size:
        continue

      # get the field to crunch
      field = _get_entry_field(entry, df_sub)
      if field == "" or field not in df_sub:
        continue

      # attempt to crunch into smaller clusters
      series = _crunch(df_sub, field, min_cluster_size)

      if series is not None and len(series) > 0:
        if verbose:
          if i % 100 == 0:
            print(f"----> {i}/{len(clusters)}, {i/len(clusters):0.0%} clustering on {cluster}, field = {field}, size = {len(series)}")
        # if we succeeded, update the cluster names with the new breakdowns
        df.loc[mask, "next_cluster"] = df.loc[mask, "next_cluster"] + "_" + series.astype(str)
        df.loc[mask, "__temp_series__"] = series.astype(str)
        cluster_dict = add_to_cluster_dict(cluster_dict, "numeric", "__temp_series__", iteration, df[mask], field)
        fields_used[field] = True

      i += 1

    # update the cluster column with the new cluster names, then iterate on those next
    df["cluster"] = df["next_cluster"]

  # assign a unique ID # to each cluster:
  i = 0
  df["cluster_id"] = "0"

  for cluster in df["cluster"].unique():
    cluster_dict["index"][cluster] = str(i)
    df.loc[df["cluster"].eq(cluster), "cluster_id"] = str(i)
    i += 1

  # print("")
  # print(cluster_dict)

  cluster_dict = resolve_cluster_dict(cluster_dict)

  list_fields_used = [field for field in fields_used]

  # return the new cluster ID's
  return df["cluster_id"], list_fields_used, cluster_dict


# PRIVATE:

def _get_entry_field(entry, df):
  field = ""
  if isinstance(entry, list):
    for _field in entry:
      if _field in df:
        field = _field
        break
  elif isinstance(entry, str):
    field = entry
  return field


def _crunch(_df, field, min_count):
  """
  Crunch a field into a smaller number of bins, each with at least min_count elements. Dynamically adapts to find the
  best number of bins to use.
  :param _df:
  :param field:
  :param min_count:
  :return:
  """
  crunch_levels = [
    (0.0, 0.5, 1.0),                # 2 clusters (high & low)
    (0.0, 0.25, 0.75, 1.0),         # 3 clusters (high, medium, low)
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # 5 clusters
  ]
  good_series = None
  too_small = False

  # if it's a boolean type:
  is_boolean = pd.api.types.is_bool_dtype(_df[field])
  if is_boolean:
    # convert to 0 and 1:
    bool_series = _df[field].astype(int)
    if bool_series.value_counts().min() < min_count:
      return None
    return bool_series

  # Try every configuration, starting with the most conservative
  for crunch_level in crunch_levels:
    test_bins = []
    for quantile in crunch_level:
      bin = _df[field].quantile(quantile)
      if bin not in test_bins and pd.isna(bin) == False:
        test_bins.append(bin)

    if len(test_bins) > 1:
      labels = test_bins[1:]
      series = pd.cut(_df[field], bins=test_bins, labels=labels, include_lowest=True)
    else:
      # if we only have one bin, this crunch is pointless
      too_small = True
      break

    if series.value_counts().min() < min_count:
      # if any of the bins are too small, give up on this level
      too_small = True
      break
    else:
      # if all bins are big enough, keep this series, and try the next level
      good_series = series

  # if we never found a good series, or if every series was too small, return None
  if too_small or good_series is None:
    return None

  # if we found a good series, return it
  return good_series