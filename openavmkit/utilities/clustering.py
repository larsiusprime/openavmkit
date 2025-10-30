import numpy as np
import pandas as pd

from typing import Tuple

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
) -> Tuple[pd.Series, list[str], pd.Series]:
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
    t.start("make_clusters")
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
        
        t.start("factorize")
        # integer codes in first-seen order (same as unique())
        codes, uniques = pd.factorize(df["cluster"].to_numpy(copy=False), sort=False)
        t.stop("factorize")

        t.start("sort")
        # stable sort to get contiguous blocks per cluster
        order = codes.argsort(kind="mergesort")           # stable
        sorted_codes = codes[order]
        t.stop("sort")

        t.start("block boundaries")
        # find block boundaries
        boundaries = 1 + (sorted_codes[1:] != sorted_codes[:-1]).nonzero()[0]
        starts = np.r_[0, boundaries]
        ends   = np.r_[boundaries, len(order)]
        t.stop("block boundaries")
        
        t.start("next_cluster copy")
        next_cluster = df["cluster"].copy()
        t.stop("next_cluster copy")

        for i in range(len(starts)):
            t.start("cluster")
            block_idx = df.index[order[starts[i]:ends[i]]]   # index labels for this cluster
            
            # if the cluster is already too small, skip it
            if (ends[i] - starts[i]) < min_cluster_size:
                t.stop("cluster")
                continue
            
            t.start("df_sub loc")
            df_sub = df.loc[block_idx]
            t.stop("df_sub loc")
            
            # get the field to crunch
            field = _get_entry_field(entry, df_sub)
            if not field or field not in df_sub:
                t.stop("cluster")
                continue
            
            t.start("crunch")
            series = _crunch(df_sub, field, min_cluster_size)
            t.stop("crunch")
            if series is not None and len(series) > 0:
                if verbose:
                    if i % 100 == 0:
                        ls = len(starts)
                        print(
                            f"----> {i}/{ls}, {i/ls:0.0%}, field = {field}, size = {len(series)}"
                        )
                
                # if we succeeded, update the cluster names with the new breakdowns
                t.start("series string")
                s = series.astype("string")
                t.stop("series string")
                t.start("next_cluster loc")
                next_cluster.loc[block_idx] = next_cluster.loc[block_idx].str.cat(s, sep="_")
                t.stop("next_cluster loc")
                fields_used[field] = True
            t.stop("cluster")
        
        t.start("next_cluster")
        df["cluster"] = next_cluster
        t.stop("next_cluster")
        
    if verbose:
        print("Done clustering")
    
    t.stop("numerics")

    t.start("cluster id 0")
    # assign a unique ID # to each cluster:
    i = 0
    df["cluster_id"] = "0"
    t.stop("cluster id 0")
    
    if verbose:
        print("Assigning cluster id names")
    
    t.start("assign_cluster_id_name")
    df["cluster_id"] = df.groupby("cluster", sort=False).ngroup().astype(str)
    t.stop("assign_cluster_id_name")
    
    if verbose:
        print("Finished assigning cluster id names")

    list_fields_used = [field for field in fields_used]
    
    t.stop("make_clusters")
    
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
