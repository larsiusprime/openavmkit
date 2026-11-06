"""
Hierarchical clustering for equity studies.

Partitions parcels into nested clusters by location, vacancy status, and
configurable categorical / numeric similarity fields. The resulting cluster
IDs are used as the unit of analysis in horizontal equity studies (parcels
in the same cluster should have similar predicted-to-sale-price ratios).
"""
import numpy as np
import pandas as pd

from typing import Tuple

from openavmkit.utilities.timing import TimingData
from openavmkit.utilities.settings import area_unit


def make_clusters(
    df_in: pd.DataFrame,
    field_location: str | None,
    fields_categorical: list[str],
    fields_numeric: list[str | list[str]] = None,
    split_on_vacant: bool = True,
    min_cluster_size: int = 15,
    unit: str = "sqft",
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
    unit : str, default "sqft"
        What unit you are using for area. "sqft" or "sqm"
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
    if field_location is not None and field_location in df.columns:
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
            f"land_area_{unit}",
            f"bldg_area_finished_{unit}",
            "bldg_quality_num",
            [
                "bldg_effective_age_years",
                "bldg_age_years",
            ],  # Try effective age years first, then normal age
            "bldg_condition_num",
        ]

    # Phase 4: iterate over numeric fields, trying to crunch down whenever possible:
    for entry in fields_numeric:
        # The field to split on. Every cluster's rows have the table's columns, so it's the same field for every
        # cluster: looked up once, and a field the table doesn't have splits nothing.
        field = _get_entry_field(entry, df)
        if not field or field not in df:
            continue
        column = df[field]

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
            # This cluster's rows of the one column it's split on (not a copy of every column)
            df_sub = column.loc[block_idx].to_frame()
            t.stop("df_sub loc")

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


# The quantiles every crunch level needs: 5 bins, then 3 (high, medium, low), then 2 (high & low).
_CRUNCH_LEVELS = (
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    (0.0, 0.25, 0.75, 1.0),
    (0.0, 0.5, 1.0),
)
_CRUNCH_QS = np.array(sorted({q for level in _CRUNCH_LEVELS for q in level}))


def _crunch(_df, field, min_count):
    """Crunch a field into a smaller number of bins, each with at least min_count
    elements.

    Dynamically adapts to find the best number of bins to use: 5 equal-count bins,
    else 3, else 2, the first level whose every bin holds at least min_count rows.
    Returns the bins as a Categorical Series labelled by each bin's upper edge, or
    None when no level works.

    The work is done with NumPy on the column (_crunch_numpy): one percentile call
    for all the edges, and the binning by searchsorted, as pandas does them. It's
    called once per cluster per numeric field, often thousands of times on small
    clusters, where each pandas call's own overhead was most of the time. Columns
    NumPy can't reproduce pandas on exactly (dates, float32, objects) take the
    pandas path (_crunch_pandas); the result is the same either way.
    """
    field_values = _df[field]

    # Boolean fast path
    if pd.api.types.is_bool_dtype(field_values):
        bool_series = field_values.astype(int)
        if bool_series.value_counts().min() < min_count:
            return None
        return bool_series

    values = _exact_float64(field_values)
    if values is None:
        return _crunch_pandas(field_values, min_count)
    return _crunch_numpy(field_values, values, min_count)


def _quantile_type(s: pd.Series, values: np.ndarray):
    """The type pandas' Series.quantile gives a whole-number quantile of this column: its own integer type for a
    nullable integer column with blanks (pandas casts each quantile back when it's whole), else float64."""
    dtype = s.dtype
    if dtype.kind in "iu" and isinstance(dtype, pd.api.extensions.ExtensionDtype) and np.isnan(values).any():
        return dtype.numpy_dtype.type
    return None


def _exact_float64(s: pd.Series):
    """The column as float64 with NaN for blanks, when that's exactly what pandas'
    quantile and cut work on (integers, float64, and their nullable kinds); else
    None."""
    dtype = s.dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        if not isinstance(dtype, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                                  pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
                                  pd.Float64Dtype)):
            return None
        values = s.to_numpy(dtype=np.float64, na_value=np.nan)
    elif dtype.kind in "iu" or dtype == np.float64:
        values = s.to_numpy(dtype=np.float64)
    else:
        return None
    if dtype.kind in "iu" and np.abs(values[~np.isnan(values)]).max(initial=0) >= 2**53:
        return None  # integers float64 can't hold exactly
    return values


def _distinct_edges(quantile_values, crunch_level):
    """A level's bin edges: its quantiles, each one within 1e-6 of an earlier one
    left out, and blanks too."""
    test_bins = []
    for q in crunch_level:
        bin_val = quantile_values[q]
        if not pd.isna(bin_val) and not any(abs(bin_val - v) < 1e-6 for v in test_bins):
            test_bins.append(bin_val)
    return test_bins


def _crunch_numpy(field_values: pd.Series, values: np.ndarray, min_count):
    present = ~np.isnan(values)
    if not present.any():
        return None
    # As pandas' Series.quantile: np.percentile on the values that aren't blank, at q * 100, linear.
    edges_all = np.percentile(values[present], _CRUNCH_QS * 100.0, method="linear")
    as_int = _quantile_type(field_values, values)
    quantile_values = {
        q: as_int(v) if as_int is not None and float(v).is_integer() else np.float64(v)
        for q, v in zip(_CRUNCH_QS.tolist(), edges_all.tolist())
    }

    for crunch_level in _CRUNCH_LEVELS:
        test_bins = _distinct_edges(quantile_values, crunch_level)
        if len(test_bins) <= 1:
            # if we only have one bin, this crunch is pointless
            return None
        edges = np.asarray(test_bins, dtype=np.float64)
        if not np.all(np.diff(edges) > 0):
            # pd.cut would raise on edges that don't increase (infinite values): let it
            return _crunch_pandas(field_values, min_count)
        # As pd.cut(right=True, include_lowest=True): (e0, e1], (e1, e2], ..., e0 itself in the first bin; blanks
        # and values outside the edges in none.
        ids = np.searchsorted(edges, values, side="left")
        ids[values == edges[0]] = 1
        outside = ~present | (ids == len(edges)) | (ids == 0)
        codes = ids - 1
        codes[outside] = -1
        # Every bin counted, empty ones too (as value_counts on a Categorical does).
        counts = np.bincount(codes[codes >= 0], minlength=len(edges) - 1)
        if counts.min() < min_count:
            # If any bin at this granularity is too small, fall through to the next
            # (coarser) crunch level rather than abandoning the search.
            continue
        # The labels as pd.cut makes them: a Categorical of each bin's upper edge, in order.
        labels = test_bins[1:]
        dtype = pd.Categorical(labels, categories=labels, ordered=True).dtype
        cat = pd.Categorical.from_codes(codes, dtype=dtype)
        return pd.Series(cat, index=field_values.index, name=field_values.name)

    return None


def _crunch_pandas(field_values: pd.Series, min_count):
    """_crunch on any column pandas can take quantiles of and cut, with pandas."""
    quantile_values = {q: field_values.quantile(q) for q in _CRUNCH_QS.tolist()}

    for crunch_level in _CRUNCH_LEVELS:
        test_bins = _distinct_edges(quantile_values, crunch_level)
        if len(test_bins) > 1:
            labels = test_bins[1:]
            series = pd.cut(
                field_values, bins=test_bins, labels=labels, include_lowest=True
            )
        else:
            # if we only have one bin, this crunch is pointless
            break

        if series.value_counts().min() < min_count:
            # If any bin at this granularity is too small, fall through to the next
            # (coarser) crunch level rather than abandoning the search: `break` here
            # made the 3-bin and 2-bin levels unreachable under every input, so the
            # "dynamically adapts" promise in the docstring never held.
            continue
        else:
            # if all bins are big enough, return this series
            return series

    return None
