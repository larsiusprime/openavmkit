import pandas as pd


def resolve_cluster_dict(
    cluster_dict: dict
) -> dict:
  final_dict = {}
  
  # Create a mapping of iterations to their entries for faster lookups
  iteration_entries = {}
  for iteration, entries in cluster_dict["iterations"].items():
    for entry_key in entries:
      if entry_key not in iteration_entries:
        iteration_entries[entry_key] = {}
      iteration_entries[entry_key][iteration] = entries[entry_key]
  
  # Process each key in the index
  for key, id in cluster_dict["index"].items():
    # If we've pre-mapped this key to an iteration, use it directly
    if key in iteration_entries:
      # Find the highest iteration number for this key
      highest_iteration = max(iteration_entries[key].keys())
      final_dict[id] = {
        "name": key,
        "iteration": highest_iteration,
        "clusters": iteration_entries[key][highest_iteration]
      }
    else:
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

  # Initialize if empty
  if not cluster_dict:
    cluster_dict = {"iterations":{}, "index":{}}

  # Get previous iteration data if available
  last_iteration = str(iteration-1)
  if last_iteration in cluster_dict["iterations"]:
    old_dict = cluster_dict["iterations"][last_iteration]
    old_keys = list(old_dict.keys())
  else:
    old_dict = {"":[]}
    old_keys = [""]

  new_dict = {}
  
  # Get all unique values at once
  unique_values = df[field].unique()

  # For numeric fields with min/max values, precompute the needed data
  min_max_values = {}
  if type == "numeric" and field_raw:
    # Group by the field to calculate min/max values for each unique value
    grouped = df.groupby(field, observed=False)
    min_max_values = {
      v: (grouped.get_group(v)[field_raw].min(), grouped.get_group(v)[field_raw].max()) 
      for v in unique_values if v in grouped.groups
    }

  # Process each old key and unique value
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
        if unique_value in min_max_values:
          min_value, max_value = min_max_values[unique_value]
          entry["f"] = field_raw
          entry["v"] = [min_value, max_value]
          entry["n"] = unique_value

      new_list.append(entry)
      
      # Create the new key
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
      ["bldg_effective_age_years", "bldg_age_years"],
      "bldg_condition_num"
    ]

  # Phase 4: iterate over numeric fields, trying to crunch down whenever possible:
  for entry in fields_numeric:
    iteration+=1
    
    # Pre-calculate clusters once per numeric field
    clusters = df["cluster"].unique()
    
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
  list_fields_used = list(fields_used.keys())

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
  # Avoid unnecessary copy if possible
  df = _df
  
  # Check if boolean type first - fastest path
  if pd.api.types.is_bool_dtype(df[field]):
    bool_series = df[field].astype(int)
    if bool_series.value_counts().min() < min_count:
      return None
    return bool_series

  # Pre-calculate field values once - avoid recalculating for each level
  field_values = df[field]
  
  crunch_levels = [
    (0.0, 0.5, 1.0),                # 2 clusters (high & low)
    (0.0, 0.25, 0.75, 1.0),         # 3 clusters (high, medium, low)
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # 5 clusters
  ]
  
  # Calculate all quantiles at once instead of repeatedly
  all_quantiles = field_values.quantile([q for level in crunch_levels for q in level]).unique()

  # Try every configuration, starting with the most conservative
  for crunch_level in crunch_levels:
    test_bins = [field_values.quantile(q) for q in crunch_level]
    test_bins = [bin for bin in test_bins if not pd.isna(bin)]
    test_bins = sorted(set(test_bins))  # Remove duplicates and ensure sorted

    if len(test_bins) <= 1:
      continue  # Skip if we can't make meaningful bins

    labels = test_bins[1:]
    series = pd.cut(field_values, bins=test_bins, labels=labels, include_lowest=True)
    
    # Check if any bin is too small
    if series.value_counts().min() >= min_count:
      return series  # Return first good series

  # No good binning found
  return None