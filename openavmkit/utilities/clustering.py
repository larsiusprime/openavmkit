import pandas as pd


def make_clusters(
    df_in: pd.DataFrame,
    field_location: str,
    fields_categorical: list[str],
    fields_numeric: list[str | list[str]] = None,
    min_cluster_size: int = 15,
    verbose: bool = False
):
  df = df_in.copy()

  # We are assigning a unique id to each cluster

  # Phase 1: split the data into clusters based on the location:
  if field_location in df:
    df["cluster"] = df[field_location].astype(str)

  fields_used = {}

  # Phase 2: add to the cluster based on each categorical field:
  for field in fields_categorical:
    if field in df:
      df["cluster"] = df["cluster"] + "_" + df[field].astype(str)
      fields_used[field] = True

  if fields_numeric is None or len(fields_numeric) == 0:
    fields_numeric = [
      "land_area_sqft",
      "bldg_area_finished_sqft",
      "bldg_quality_num",
      ["bldg_effective_age_years", "bldg_age_years"], # Try effective age years first, then normal age
      "bldg_condition_num"
    ]

  if verbose:
    print(f"Clustering by location and big five:")

  # iterate over numeric fields, trying to crunch down whenever possible:
  for entry in fields_numeric:

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
      df_sub = df[df["cluster"].eq(cluster)]

      # if the cluster is already too small, skip it
      if len(df_sub) < min_cluster_size:
        continue

      # get the field to crunch
      field = _get_entry_field(entry, df_sub)
      if field == "":
        continue

      # attempt to crunch into smaller clusters
      series = _crunch(df_sub, field, min_cluster_size)

      if series is not None and len(series) > 0:
        if verbose:
          if i % 100 == 0:
            print(f"----> {i}/{len(clusters)}, {i/len(clusters):0.0%} clustering on {cluster}, field = {field}, size = {len(series)}")
        # if we succeeded, update the cluster names with the new breakdowns
        df_sub.loc[:, "next_cluster"] = df_sub["next_cluster"] + "_" + series.astype(str)
        df.loc[df["cluster"].eq(cluster), "next_cluster"] = df_sub["next_cluster"].values
        fields_used[field] = True

      i += 1

    # update the cluster column with the new cluster names, then iterate on those next
    df["cluster"] = df["next_cluster"]

  # assign a unique ID # to each cluster:
  i = 0
  df["cluster_id"] = "0"
  for cluster in df["cluster"].unique():
    df.loc[df["cluster"].eq(cluster), "cluster_id"] = str(i)
    i += 1

  list_fields_used = [field for field in fields_used]

  # return the new cluster ID's
  return df["cluster_id"], list_fields_used


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
  crunch_levels = [
    (0.0, 0.5, 1.0),                # 2 clusters (high & low)
    (0.0, 0.25, 0.75, 1.0),         # 3 clusters (high, medium, low)
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # 5 clusters
  ]
  good_series = None
  too_small = False

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
      too_small = True
      break

    if series.value_counts().min() < min_count:
      too_small = True
      break
    else:
      good_series = series

  if too_small or good_series is None:
    return None

  return good_series