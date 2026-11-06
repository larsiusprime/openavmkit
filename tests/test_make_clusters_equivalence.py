"""make_clusters gives each cluster only the column it's split on (not a copy of every column), looking the field up
once: these hold it to exactly what it gave before, on tables of every shape. _make_clusters_reference below is that
version, verbatim (openavmkit master, 2026-11-06, after #423)."""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities.clustering import _crunch, _get_entry_field, make_clusters
from openavmkit.utilities.timing import TimingData


def _make_clusters_reference(
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


def _sales(seed: int, unit: str = "sqft", index: str = "shuffled") -> pd.DataFrame:
  rng = np.random.default_rng(seed)
  n = int(rng.choice([40, 300, 2_000]))
  df = pd.DataFrame(
    {
      "neighborhood": rng.choice([f"N{i}" for i in range(int(rng.choice([1, 5, 30])))], n),
      "is_vacant": rng.random(n) < 0.15,
      "bldg_type": rng.choice(["ranch", "colonial", "cape"], n),
      f"land_area_{unit}": rng.lognormal(9, 0.6, n),
      f"bldg_area_finished_{unit}": np.where(rng.random(n) < 0.1, np.nan, rng.lognormal(7.5, 0.4, n)),
      "bldg_quality_num": pd.array(np.where(rng.random(n) < 0.1, None, rng.integers(1, 6, n)), dtype="Int64"),
      "bldg_age_years": rng.integers(0, 120, n).astype(float),
      "bldg_condition_num": rng.integers(1, 6, n).astype(float),
      "price": rng.lognormal(12.5, 0.5, n),
    }
  )
  if rng.random() < 0.5:
    df["bldg_effective_age_years"] = df["bldg_age_years"] * 0.8  # the list entry's first choice, when present
  if index == "shuffled":
    df.index = pd.Index(rng.permutation(n * 2)[:n])
  elif index == "repeated":
    df.index = pd.Index(rng.integers(0, n // 2, n))  # repeated labels: picked by label, as before
  return df


def _same(got, want):
  pd.testing.assert_series_equal(got[0], want[0])
  assert got[1] == want[1]
  pd.testing.assert_series_equal(got[2], want[2])


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("min_size", [5, 15, 60])
def test_make_clusters_matches_the_version_that_copied_every_column(seed, min_size):
  df = _sales(seed)
  for args in (
    dict(field_location="neighborhood", fields_categorical=["bldg_type"], fields_numeric=None),
    dict(field_location=None, fields_categorical=[], fields_numeric=["price", "not_a_column", ["nope", "bldg_age_years"]]),
    dict(field_location="neighborhood", fields_categorical=["bldg_type", "missing_cat"], fields_numeric=None, split_on_vacant=False),
  ):
    _same(
      make_clusters(df, min_cluster_size=min_size, **args),
      _make_clusters_reference(df, min_cluster_size=min_size, **args),
    )


@pytest.mark.parametrize("index", ["range", "repeated"])
def test_metric_units_and_any_index(index):
  df = _sales(3, unit="sqm", index=index)
  args = dict(field_location="neighborhood", fields_categorical=["bldg_type"], fields_numeric=None, unit="sqm")
  _same(make_clusters(df, **args), _make_clusters_reference(df, **args))
