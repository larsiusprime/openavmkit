import pandas as pd
import numpy as np
from openavmkit.utilities.timing import TimingData


def make_clusters(
    df_in: pd.DataFrame,
    field_location: str | None,
    fields_categorical: list[str],
    fields_numeric: list[str | list[str]] = None,
    split_on_vacant: bool = True,
    min_cluster_size: int = 15,
    verbose: bool = False,
    output_folder: str = "",
    t: TimingData = None
):
    """
    Partition a DataFrame into hierarchical clusters based on location, vacancy,
    categorical, and numeric fields.

    Clustering proceeds in phases:

    1. **Location split**: if `field_location` is given and present in `df_in`,
       rows are initially grouped by unique values of that column.
    2. **Vacancy split**: if the column `is_vacant` exists, clusters are further
       subdivided by vacancy status (`True`/`False`).
    3. **Categorical split**: for each column in `fields_categorical`, clusters
       are refined by appending the stringified category value.
    4. **Numeric split**: for each entry in `fields_numeric`, attempt to subdivide
       each cluster on a numeric field (or first available from a list) by calling
       `_crunch()`.  Clusters smaller than `min_cluster_size` are skipped, ensuring
       no cluster falls below this threshold.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input data to cluster.  Each row will be assigned a final cluster ID.
    field_location : str or None
        Column name to use for an initial split.  If None or not found, all rows
        start in one cluster.
    fields_categorical : list of str
        Categorical column names for successive splits.  Each unique value in these
        fields refines cluster labels.
    fields_numeric : list of str or list of str, default None
        Numeric fields (or lists of fallbacks) for recursive clustering.  If None,
        a default set is used.  Each entry represents a variable to attempt
        splitting upon, in order.
    split_on_vacant: bool
        whether to split on vacant status or not, default True
    min_cluster_size : int, default 15
        Minimum number of rows required to split a cluster on a numeric field.
    verbose : bool, default False
        If True, print progress messages at each phase and sub-cluster iteration.
    output_folder : str, default ""
        Path to save any intermediate outputs (currently unused).
    t : TimingData, optional
        TimingData object to record performance metrics.

    Returns
    -------
    cluster_ids : pandas.Series
        Zero-based string IDs for each row’s final cluster.
    fields_used : list of str
        Names of fields (categorical or numeric) that resulted in at least one split.
    cluster_labels : pandas.Series
        Hierarchical cluster labels encoding the sequence of splits applied to each row.
    """
    if t is None:
    t = TimingData()
    t.start("make clusters")
    df = df_in.copy()

    iteration = 0
    # We are assigning a unique id to each cluster

    t.start("categoricals")
    # Phase 1: split the data into clusters based on the location:
    if field_location is not None and field_location in df:
        df["cluster"] = df[field_location].astype(str)
        if verbose:
            print(f"--> crunching on location, {len(df['cluster'].unique())} clusters")
    else:
        df["cluster"] = ""

    fields_used = {}

    # Phase 2: split into vacant and improved:
    if split_on_vacant and "is_vacant" in df:
        df["cluster"] = df["cluster"] + "_" + df["is_vacant"].astype(str)
        if verbose:
            print(f"--> crunching on is_vacant, {len(df['cluster'].unique())} clusters")

    # Phase 3: add to the cluster based on each categorical field:
    for field in fields_categorical:
        if field in df:
            df["cluster"] = df["cluster"] + "_" + df[field].astype(str)
            iteration += 1
            fields_used[field] = True
    t.stop("categoricals")

    t.start("numerics")
    if fields_numeric is None or len(fields_numeric) == 0:
        fields_numeric = [
            "land_area_sqft",
            "bldg_area_finished_sqft",
            "bldg_quality_num",
            [
                "bldg_effective_age_years",
                "bldg_age_years",
            ],  # Try effective age years first, then normal age
            "bldg_condition_num",
        ]

    # Phase 4: iterate over numeric fields, trying to crunch down whenever possible:
    for entry in fields_numeric:

        iteration += 1
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
                        print(
                            f"----> {i}/{len(clusters)}, {i/len(clusters):0.0%} clustering on {cluster}, field = {field}, size = {len(series)}"
                        )
                # if we succeeded, update the cluster names with the new breakdowns
                df.loc[mask, "next_cluster"] = (
                    df.loc[mask, "next_cluster"] + "_" + series.astype(str)
                )
                df.loc[mask, "__temp_series__"] = series.astype(str)
                fields_used[field] = True

            i += 1

        # update the cluster column with the new cluster names, then iterate on those next
        df["cluster"] = df["next_cluster"]
    t.stop("numerics")

    # assign a unique ID # to each cluster:
    i = 0
    df["cluster_id"] = "0"

    for cluster in df["cluster"].unique():
        df.loc[df["cluster"].eq(cluster), "cluster_id"] = str(i)
        i += 1

    list_fields_used = [field for field in fields_used]
    t.stop("make clusters")

    # return the new cluster ID's
    return df["cluster_id"], list_fields_used, df["cluster"]


#######################################
# PRIVATE
#######################################


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
    """Crunch a field into a smaller number of bins, each with at least min_count
    elements.

    Dynamically adapts to find the best number of bins to use.
    """
    crunch_levels = [
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # 5 clusters
        (0.0, 0.25, 0.75, 1.0),  # 3 clusters (high, medium, low)
        (0.0, 0.5, 1.0),  # 2 clusters (high & low)
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

    def _value_in_list(value, lst):
        for v in lst:
            delta = abs(value - v)
            if delta < 1e-6:
                return True
        return False

    # Iterate over each crunch level
    for crunch_level in crunch_levels:
        test_bins = []
        for q in crunch_level:
            bin_val = quantile_values[q]
            # Only add non-NaN and new bin values to test_bins
            if not pd.isna(bin_val) and not _value_in_list(bin_val, test_bins):
                test_bins.append(bin_val)

        if len(test_bins) > 1:
            labels = test_bins[1:]
            series = pd.cut(
                field_values, bins=test_bins, labels=labels, include_lowest=True
            )
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
