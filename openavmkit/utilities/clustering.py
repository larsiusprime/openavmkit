import pandas as pd
import duckdb as db
from typing import Optional, Sequence

from openavmkit.utilities.data import assure_arrow
from openavmkit.utilities.timing import TimingData


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


def make_clusters_duckdb(
    df_in: pd.DataFrame,
    field_location: str | None,
    fields_categorical: Sequence[str],
    fields_numeric: Sequence[str | list[str]] | None = None,
    min_cluster_size: int = 15,
    verbose: bool = False,
    output_folder: str = "",
    conn: Optional[db.DuckDBPyConnection] = None
):
  """
  DuckDB-powered variant of `make_clusters`.
  """
  t = TimingData()
  t.start("all")
  # -----------------------------------------------------------------------
  # 0.  In-memory connection & Arrow guarantee
  # -----------------------------------------------------------------------
  t.start("init")
  local_conn = conn is None
  if local_conn:
    conn = db.connect(database=":memory:")
  t.stop("init")

  t.start("in copy")
  df = df_in.copy()
  t.stop("in copy")

  t.start("assure arrow")
  if "geometry" in df:
    df = df.drop(columns=["geometry"])
  df = assure_arrow(df)
  t.stop("assure arrow")

  df["rowid"] = df.index

  t.start("register")
  # initial read-only view
  conn.register("t", df)
  t.stop("register")

  iteration = 0
  cluster_dict: dict = {}

  # will be filled below
  df["cluster"] = ""
  fields_used = {}

  # 1. Phase 1 – location
  # This is fast enough to do in pandas

  t.start("location")
  if field_location and field_location in df:
    df["cluster"] = df[field_location].astype(str)
    cluster_dict = add_to_cluster_dict(
      cluster_dict, "location", field_location, iteration, df
    )
    if verbose:
      print(f"--> crunching on location, {df[field_location].nunique()} clusters")
  t.stop("location")

  # 2. Phase 2 – boolean is_vacant
  # Also fast enough to do in pandas

  t.start("is_vacant")
  if "is_vacant" in df:
    df["cluster"] = df["cluster"] + "_" + df["is_vacant"].astype(str)
    cluster_dict = add_to_cluster_dict(
      cluster_dict, "boolean", "is_vacant", iteration, df
    )
    if verbose:
      print(f"--> crunching on is_vacant, {df['cluster'].nunique()} clusters")
  t.stop("is_vacant")

  # 3. Phase 3 – categoricals
  # Just more pandas string concatenation

  t.start("cats")
  for field in fields_categorical:
    if field not in df:
      continue
    iteration += 1
    df["cluster"] = df["cluster"] + "_" + df[field].astype(str)
    cluster_dict = add_to_cluster_dict(
      cluster_dict, "categorical", field, iteration, df
    )
    fields_used[field] = True
  t.stop("cats")

  # Refresh DuckDB’s view so the numeric phase sees the updated clusters
  t.start("unreg_reg_1")
  conn.unregister("t")
  conn.register("t", df)
  t.stop("unreg_reg_1")

  # 4.  Phase 4 – numeric fields with adaptive binning
  # This is where we drop down to DuckDB for speed

  if not fields_numeric:
    fields_numeric = [
      "land_area_sqft",
      "bldg_area_finished_sqft",
      "bldg_quality_num",
      ["bldg_effective_age_years", "bldg_age_years"],
      "bldg_condition_num",
    ]

  t.start("numeric")
  for entry in fields_numeric:
    iteration += 1
    entry_name = _get_entry_field(entry, df) or "(missing)"
    if verbose:
      print(f"--> crunching on {entry_name}, {df['cluster'].nunique()} clusters")

    # -------- refresh the DuckDB view for *this* numeric field ---------
    t.start("unreg_reg_2")
    conn.unregister("t")
    conn.register("t", df)
    t.stop("unreg_reg_2")

    # Snapshot cluster -> next_cluster in pandas
    df["next_cluster"] = df["cluster"]

    field = _get_entry_field(entry, df)
    if not field or field not in df:
      continue

    t.start("large clusters")
    large_clusters = conn.execute(f"""
                SELECT cluster
                FROM t
                GROUP BY cluster
                HAVING COUNT(*) >= {min_cluster_size}
            """).fetchdf().cluster.to_list()
    t.stop("large clusters")

    t.start("cluster_rows")
    cluster_rows = (
      df[["cluster"]]
      .reset_index(names="rowid")          # rowid = original index
      .rename(columns={"cluster": "the_cluster"})
    )
    t.stop("cluster_rows")

    # ---------- loop only over the large clusters ------------
    t.start("enumerate")
    for i, cl in enumerate(large_clusters, 1):
      # Boolean mask on the small cluster_rows table
      mask = cluster_rows["the_cluster"].eq(cl)

      # Extract the real df labels from the 'rowid' column
      row_ids = cluster_rows.loc[mask, "rowid"].to_numpy()

      # Build the slice to crunch
      sub = df.loc[row_ids, [field]].reset_index(drop=True)

      t.start("crunch")
      t.start(f"crunch_{field}")
      series = crunch_duckdb(sub, field, min_cluster_size, conn)
      t.stop(f"crunch_{field}")
      t.stop("crunch")
      if series is None:
        t.stop("enumerate")
        continue

      new_suffix = series.astype(str).to_numpy()

      t.start("__temp_series__")
      df.loc[row_ids, "__temp_series__"] = new_suffix
      df.loc[row_ids, "next_cluster"] = (
          df.loc[row_ids, "next_cluster"].astype(str)
          + "_" + new_suffix
      )
      t.stop("__temp_series__")

      t.start("add_to_cluster_dict")
      cluster_dict = add_to_cluster_dict(
        cluster_dict, "numeric", "__temp_series__", iteration,
        df.loc[row_ids], field
      )
      t.stop("add_to_cluster_dict")
      fields_used[field] = True
    t.stop("enumerate")

    # promote next_cluster → cluster
    df["cluster"] = df["next_cluster"]
  t.stop("numeric")

  # 5.  Final – cluster_id assignment
  # Again just do this in pandas

  t.start("unique")
  unique_clusters = df["cluster"].unique()
  t.stop("unique")
  t.start("cluster_id_map")
  cluster_id_map = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
  t.stop("cluster_id_map")

  cluster_dict.setdefault("index", {})
  t.start("update index")
  cluster_dict["index"].update(
    {cluster: str(idx) for cluster, idx in cluster_id_map.items()}
  )
  t.stop("update index")

  # stringify to match the original signature
  t.start("cluster_id")
  df["cluster_id"] = df["cluster"].map(cluster_id_map).astype(str)
  t.stop("cluster_id")

  t.start("final_dict")
  final_dict = resolve_cluster_dict(cluster_dict)
  t.stop("final_dict")

  # 6. Wrap up (close connection, return)
  t.start("close")
  if local_conn:
    conn.close()
  t.stop("close")
  t.stop("all")

  print("")
  print("TIMINGS FOR MAKE CLUSTERS:")
  print(t.print())
  print("")

  return df["cluster_id"], list(fields_used), final_dict, df["cluster"]


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
  return df["cluster_id"], list_fields_used, cluster_dict, df["cluster"]


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


# Pre-define the quantile break-point schemes you want to try
CRUNCH_LEVELS: Sequence[Sequence[float]] = (
  (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),   # 5 bins
  (0.0, 0.25, 0.75, 1.0),           # 3 bins
  (0.0, 0.5, 1.0)                   # 2 bins
)

def crunch_duckdb(
    df: pd.DataFrame,
    field: str,
    min_count: int,
    conn: Optional[db.DuckDBPyConnection] = None
) -> Optional[pd.Series]:
  """
  Re-implementation of `_crunch` that:
    • Uses DuckDB for quantiles (fast C++)
    • Uses pandas.cut for bin assignment (also C++)
  Returns a pandas Series of labels or None if no valid bins.
  """
  # 0.  connection bookkeeping
  local_conn = conn is None
  if local_conn:
    conn = db.connect(database=":memory:")
  # register the view
  conn.register("t", df)

  # 1.  fast-path for booleans
  if pd.api.types.is_bool_dtype(df[field]):
    counts = conn.execute(
      f"SELECT COUNT(*) cnt FROM t GROUP BY {field}"
    ).fetchdf().cnt
    if counts.min() < min_count:
      if local_conn:
        conn.close()
      return None
    return df[field].astype(int)

  # 2.  try each quantile scheme
  for qs in CRUNCH_LEVELS:
    # 2a. pull all quantiles in one go
    pct_sql = ", ".join(str(q) for q in qs)
    res = conn.execute(
      f"""
            SELECT quantile_cont({field}, ARRAY[{pct_sql}]::DOUBLE[])
            FROM t
            """
    ).fetchone()

    # if we got nothing or a SQL NULL, move on
    if not res or res[0] is None:
      continue

    qlist = res[0]

    # dedupe & sort
    bins = sorted({q for q in qlist if q is not None})
    if len(bins) <= 1:
      continue

    # 2b. assign bins via pandas.cut
    labels = bins[1:]
    series = pd.cut(
      df[field],
      bins=bins,
      labels=labels,
      include_lowest=True
    )

    # 2c. check minimum size
    if series.value_counts().min() < min_count:
      continue

    # success!
    if local_conn:
      conn.close()
    return series

  # 3.  no scheme worked
  if local_conn:
    conn.close()
  return None


def _crunch(_df, field, min_count):
  """
  Crunch a field into a smaller number of bins, each with at least min_count elements.
  Dynamically adapts to find the best number of bins to use.
  :param _df: DataFrame containing the field
  :param field: Name of the field to crunch
  :param min_count: Minimum count required per bin
  :return: A series with binned values or None if no valid configuration is found
  """
  crunch_levels = [
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), # 5 clusters
    (0.0, 0.25, 0.75, 1.0),         # 3 clusters (high, medium, low)
    (0.0, 0.5, 1.0)                # 2 clusters (high & low)
  ]
  good_series = None
  too_small = False

  # Cache the column to avoid repeated attribute lookups
  field_values = _df[field]

  # Boolean fast path
  if pd.api.types.is_bool_dtype(field_values):
    bool_series = field_values.astype(int)
    if bool_series.value_counts().min() < min_count:
      return None
    return bool_series

  # Precompute all unique quantiles required by all crunch levels
  unique_qs = {q for level in crunch_levels for q in level}
  quantile_values = {q: field_values.quantile(q) for q in unique_qs}

  # Iterate over each crunch level
  for crunch_level in crunch_levels:
    test_bins = []
    for q in crunch_level:
      bin_val = quantile_values[q]
      # Only add non-NaN and new bin values to test_bins
      if not pd.isna(bin_val) and bin_val not in test_bins:
        test_bins.append(bin_val)

    if len(test_bins) > 1:
      labels = test_bins[1:]
      series = pd.cut(field_values, bins=test_bins, labels=labels, include_lowest=True)
    else:
      # if we only have one bin, this crunch is pointless
      too_small = True
      break

    if series.value_counts().min() < min_count:
      # if any of the bins are too small, give up on this level
      too_small = True
      break
    else:
      # if all bins are big enough, return this series
      return series

  return None