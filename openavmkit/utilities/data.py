import os
import pickle

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from collections import defaultdict

from scipy.spatial._ckdtree import cKDTree

from openavmkit.utilities.settings import get_model_group_ids


def is_column_of_type(df: pd.DataFrame, col: str, type_name: str) -> bool:
    series = df[col]
    if type_name == "str" or type_name == "string":
        return (
            pd.api.types.is_string_dtype(series) |
            pd.api.types.is_object_dtype(series)
        )
    if type_name == "num" or type_name == "number":
        return pd.api.types.is_numeric_dtype(series)
    if type_name == "int" or type_name == "integer":
        return (
            pd.api.types.is_integer_dtype(series) | 
            pd.api.types.is_int64_dtype(series)
        )
    if type_name == "float":
        return (
            pd.api.types.is_float_dtype(series)
        )
    if type_name == "date" or type_name == "datetime":
        return (
            pd.api.types.is_datetime64_any_dtype(series) |
            pd.api.types.is_datetime64_dtype(series) |
            pd.api.types.is_datetime64_ns_dtype(series) |
            isinstance(series.dtype, pd.DatetimeTZDtype)
        )
    else:
        raise ValueError("Unknown type name: {type_name}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the column names in a DataFrame by replacing forbidden characters with legal
    representations. For one-hot encoded columns (containing '='), ensures clean formatting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names
    """
    # Find column names that contain forbidden characters and replace them with legal representations.
    replace_map = {
        "[": "_",
        "]": "_",
        "<NA>": "_NA_",
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",  # Replace spaces with underscores
        "-": "_",  # Replace hyphens with underscores
        ",": "_",  # Replace commas with underscores
        ";": "_",  # Replace semicolons with underscores
        ".": "_",  # Replace periods with underscores
        "(": "_",  # Replace parentheses with underscores
        ")": "_",
    }

    # First pass - replace special characters
    for key in replace_map:
        df.columns = df.columns.str.replace(key, replace_map[key], regex=False)

    # Second pass - clean up one-hot encoded column names
    new_columns = []
    for col in df.columns:
        if "=" in col:
            # Handle one-hot encoded columns
            base, value = col.split("=", 1)
            # Clean up the base and value
            base = base.strip()
            value = value.strip()
            # Replace multiple underscores with single underscore
            base = "_".join(filter(None, base.split("_")))
            value = "_".join(filter(None, value.split("_")))
            new_col = f"{base}__{value}"  # Use double underscore as separator
        else:
            # For non-one-hot columns, just clean up multiple underscores
            new_col = "_".join(filter(None, col.split("_")))

        new_columns.append(new_col)

    df.columns = new_columns
    return df


def clean_series(series: pd.Series) -> pd.Series:
    """Clean the values in a Series by replacing forbidden characters with legal representations.

    Parameters
    ----------
    series : pd.Series
        The series to be cleaned

    Returns
    -------
    pd.Series
        The cleaned series
    """
    replace_map = {
        "[": "_LBRKT_",
        "]": "_RBRKT_",
        "<NA>": "_NA_",
        "/": "_SLASH_",
        "\\": "_BSLASH_",
        ":": "_COLON_",
        "*": "_STAR_",
        "?": "_QMARK_",
        '"': "_DQUOT_",
        "<": "_LT_",
        ">": "_GT_",
        "|": "_PIPE_",
    }

    for key in replace_map:
        series = series.str.replace(key, replace_map[key], regex=False)

    return series


def div_series_z_safe(
    numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray
) -> pd.Series | np.ndarray:
    """Perform a divide-by-zero-safe division of two series or arrays, replacing division
    by zero with None.

    Parameters
    ----------
    numerator : pd.Series | np.ndarray
        The series/array that serves as the numerator
    denominator : pd.Series | np.ndarray
        The series/array that serves as the denominator/divisor

    Returns
    -------
    pd.Series | np.ndarray
        The result of the division with divide-by-zero cases replaced by ``None``
    """

    # Handle both pandas and bodo.pandas array types
    if hasattr(numerator, 'to_numpy') and not hasattr(numerator, 'index'):
        numerator = pd.Series(numerator)
    if hasattr(denominator, 'to_numpy') and not hasattr(denominator, 'index'):
        denominator = pd.Series(denominator)

    # fast path for ndarray
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        num = np.asarray(numerator, dtype=np.float64, order='K')
        den = np.asarray(denominator, dtype=np.float64, order='K')

        # pre-allocate the output filled with NaN
        out = np.full_like(num, np.nan, dtype=np.float64)

        # element-wise division only where the denominator is non‑zero
        # np.divide writes directly into `out`
        np.divide(num, den, out=out, where=den != 0)

        return out
    # ---------- pandas path (preferred for DF math) ----------
    if hasattr(numerator, 'to_numpy') and hasattr(denominator, 'to_numpy'):
        num = numerator
        den = denominator.reindex(num.index)  # preserve num's order; no sorting

        a = num.to_numpy(dtype=np.float64, copy=False)
        b = den.to_numpy(dtype=np.float64, copy=False)

        out = np.full_like(a, np.nan, dtype=np.float64)
        mask = (b != 0) & ~np.isnan(b)
        
        np.divide(a, b, out=out, where=mask)
        return pd.Series(out, index=num.index, dtype="Float64")
    raise ValueError(f"Can only operate on Series-like objects or np.ndarray, found: {type(numerator), type(denominator)}")


def div_df_z_safe(df: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Perform a divide-by-zero-safe division of two columns in a DataFrame, replacing
    division by zero with None.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numerator : str
        Name of the column to use as the numerator
    denominator : str
        Name of the column to use as the numerator/divisor

    Returns
    -------
    pd.Series
        The result of the division with divide-by-zero cases replaced by ``None``
    """
    # Get the index of all rows where the denominator is zero.
    idx_denominator_zero = df[denominator].eq(0)

    # Get the numerator and denominator for rows where the denominator is not zero.
    series_numerator = df.loc[~idx_denominator_zero, numerator]
    series_denominator = df.loc[~idx_denominator_zero, denominator]

    # Make a copy of the denominator.
    result = df[denominator].copy()

    # Replace values where denominator is zero with None.
    result[idx_denominator_zero] = None

    # Replace other values with the result of the division.

    result = result.astype("Float64")  # ensure it can accept the result

    result[~idx_denominator_zero] = series_numerator / series_denominator
    return result


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown-formatted string.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    str
        Markdown representation of the DataFrame
    """
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = "\n".join("| " + " | ".join(row) + " |" for row in df.astype(str).values)
    return f"{header}\n{separator}\n{rows}"


def rename_dict(original_dict: dict, renames:dict) -> dict:
    """Rename the keys of a dictionary according to a given rename map.

    Parameters
    ----------
    original_dict : Dictionary
        Original dictionary.

    renames : Dictionary
        Diciontary mapping old keys to new keys.

    Returns
    -------
    New dictionary with keys renamed
    """
    new_dict = {}
    for key in original_dict:
        new_key = renames.get(key, key)
        new_dict[new_key] = original_dict[key]
    return new_dict


def do_per_model_group(
    df_in: pd.DataFrame,
    settings: dict,
    func: callable,
    params: dict,
    key: str = "key",
    verbose: bool = False,
    instructions=None,
    skip:list|None=None
) -> pd.DataFrame:
    """Apply a function to each subset of the DataFrame grouped by ``model_group``, updating
    rows based on matching indices.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame
    settings : dict
        Settings dictionary
    func : callable
        Function to apply to each subset
    params : dict
        Additional parameters for the function
    key : str, optional
        Column name to use as the index for alignment (default is "key")
    verbose : bool, optional
        Whether to print verbose output. Default is False.
    instructions : Any, optional
        Special instructions for the function
    skip : list, optional
        List of model group names to skip

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with updates from the function.
    """
    df = df_in.copy()

    if instructions is None:
        instructions = {}

    model_groups = get_model_group_ids(settings, df_in)
    verbose = params.get("verbose", verbose)

    for model_group in model_groups:
        if pd.isna(model_group):
            continue
        if skip is not None and model_group in skip:
            if verbose:
                print(f"Skipping model group: {model_group}")
            continue

        if verbose:
            print(f"Processing model group: {model_group}")

        # Copy params locally to avoid side effects.
        params_local = params.copy()
        params_local["model_group"] = model_group

        # Filter the subset using .loc to avoid SettingWithCopyWarning
        mask = df["model_group"].eq(model_group)
        df_sub = df.loc[mask].copy()

        # Apply the function.
        df_sub_updated = func(df_sub, **params_local)

        if df_sub_updated is not None:
            # Ensure consistent data types between df and the updated subset.
            just_stomp_columns = instructions.get("just_stomp_columns", [])
            if len(just_stomp_columns) > 0:
                for col in just_stomp_columns:
                    if col in df_sub_updated.columns:
                        df.loc[mask, col] = df_sub_updated[col]
            else:
                for col in df_sub_updated.columns:
                    if col == key:
                        continue
                    df = combine_dfs(
                        df, df_sub_updated[[key, col]], df2_stomps=True, index=key
                    )

    return df


def fill_from_df(
    df_a: pd.DataFrame, 
    df_b: pd.DataFrame,
    key: str,
    field: str
) -> pd.DataFrame:
    """
    Copies values from field `field` in `df_b`, to the corresponding field in `df_a`, but only
    where the values in `df_a` are empty. Aligns on `key` as the index.
    
    Parameters
    ----------
    df_a : pd.DataFrame
        Base DataFrame you want to copy values TO
    df_b : pd.DataFrame
        Other DataFrame you want copy values FROM
    key : str
        Key field that you want to use to align the two Dataframes
    field : str
        Name of the field you want to copy from one DataFrame to the other
    
    Returns
    -------
    The modified DataFrame with copied values
    """
    df_a = df_a.merge(
        df_b[[key, field]],
        on=key,
        how="left",
        suffixes=("","___b___")
    )
    if f"{field}___b___" in df_a:
        df_a[field] = df_a[field].fillna(df_a[f"{field}___b___"])
        df_a = df_a.drop(columns=f"{field}___b___")
    return df_a


def merge_and_stomp_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df2_stomps=False,
    on: str | list = "key",
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge two DataFrames and resolve overlapping columns by 'stomping'.

    Performs a pandas merge of `df1` and `df2` on key(s) `on`, using suffixes
    '_1' and '_2' for overlapping column names.  After merging, for each
    common column (excluding join keys) the function selects values from
    `df2` wherever non-null if `df2_stomps=True`, otherwise prefers `df1`'s
    non-null values.  Intermediate suffixed columns are dropped before
    returning the final DataFrame.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Base DataFrame whose values are used when `df2_stomps=False` or when
        `df2` has nulls in overlapping columns.
    df2 : pandas.DataFrame
        Secondary DataFrame whose values may overwrite those in `df1`
        when `df2_stomps=True` and non-null.
    df2_stomps : bool, default False
        If True, prefer non-null values from `df2` over `df1` in overlapping
        columns; if False, prefer non-null values from `df1`.
    on : str or list of str, default 'key'
        Column name or list of column names to join on.
    how : str, default 'left'
        Type of join to perform: 'left', 'right', 'inner', or 'outer'.

    Returns
    -------
    pandas.DataFrame
        The merged DataFrame with overlapping columns resolved according to the
        `df2_stomps` policy.  All original columns and merged non-overlapping
        columns are retained; intermediate '_1' and '_2' suffix columns are
        removed.
    """
    common_columns = [col for col in df1.columns if col in df2.columns]
    df_merge = pd.merge(df1, df2, on=on, how=how, suffixes=("_1", "_2"))
    suffixed_columns = [col + "_1" for col in common_columns] + [
        col + "_2" for col in common_columns
    ]
    suffixed_columns = [col for col in suffixed_columns if col in df_merge.columns]

    for col in common_columns:
        if col == on or (isinstance(on, list) and col in on):
            continue
        if df2_stomps:
            # prefer df2's column value everywhere df2 has a non-null value
            # Filter out empty entries before combining
            df2_col = df_merge[col + "_2"].dropna()
            df1_col = df_merge[col + "_1"].dropna()
            if df2_col.size > 0 and df1_col.size > 0:
                df_merge[col] = df2_col.combine_first(df1_col)
            elif df2_col.size > 0:
                df_merge[col] = df2_col
            else:
                df_merge[col] = df1_col
        else:
            # prefer df1's column value everywhere df1 has a non-null value
            s1 = df_merge[f"{col}_1"]
            s2 = df_merge[f"{col}_2"]
            df_merge[col] = _left_wins(s1, s2)

    df_merge.drop(columns=suffixed_columns, inplace=True)
    return df_merge


def combine_dfs(
    df1: pd.DataFrame, df2: pd.DataFrame, df2_stomps=False, index: str = "key"
) -> pd.DataFrame:
    """Combine two DataFrames on a given index column.

    If ``df2_stomps`` is False, NA values in df1 are filled with values from df2. If
    ``df2_stomps`` is True, values in df1 are overwritten by those in df2 for matching keys.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame
    df2 : pd.DataFrame
        Second DataFrame
    df2_stomps : bool, optional
        Flag indicating if df2 values should overwrite df1 values (default is False).
    index : str, optional
        Column name to use as the index for alignment (default is "key").

    Returns
    -------
    pd.DataFrame
        Combined DataFrame
    """
    df = df1.copy()
    # Save the original index for restoration
    original_index = df.index.copy()

    # Work on a copy so we don't modify df2 outside this function.
    df2 = df2.copy()

    # Set the index to the key column for alignment.
    df.index = df[index]
    df2.index = df2[index]

    # Iterate over columns in df2 (skip the key column).
    for col in df2.columns:
        if col == index:
            continue
        if col in df.columns:
            # Find the common keys to avoid KeyErrors if df2 has extra keys.
            common_idx = df.index.intersection(df2.index)
            if df2_stomps:
                # Overwrite all values in df for common keys.
                df.loc[common_idx, col] = df2.loc[common_idx, col]
            else:
                # For common keys, fill only NA values.
                na_mask = pd.isna(df.loc[common_idx, col])
                # Only assign where df2 has a value and df is NA.
                df.loc[common_idx[na_mask], col] = df2.loc[common_idx[na_mask], col]
        else:
            # Add the new column, aligning by index.
            # (Rows in df without a corresponding key in df2 will get NaN.)
            df[col] = df2[col]

    # Restore the original index.
    df.index = original_index
    return df


def add_sqft_fields(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add per-square-foot fields to the DataFrame for land and improvement values.

    This function creates new columns based on existing value fields and area fields.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with additional per-square-foot-fields
    """
    df = df_in.copy()
    land_sqft = [
        "model_market_value",
        "model_land_value",
        "assr_market_value",
        "assr_land_value",
    ]
    impr_sqft = [
        "model_market_value",
        "model_impr_value",
        "assr_market_value",
        "assr_impr_value",
    ]
    for field in land_sqft:
        if field in df:
            df[field + "_land_sqft"] = div_series_z_safe(
                df[field], df["land_area_sqft"]
            )
    for field in impr_sqft:
        if field in df:
            df[field + "_impr_sqft"] = div_series_z_safe(
                df[field], df["bldg_area_finished_sqft"]
            )
    return df


def count_values_in_common(
    a: pd.DataFrame, b: pd.DataFrame, a_field: str, b_field: str = None
) -> tuple[int, int]:
    """Count all the unique values that two columns of two dataframes have in common

    Parameters
    ----------
    a : pd.DataFrame
        The first DataFrame
    b : pd.DataFrame
        The second DataFrame
    a_field : str
        The column from the first DataFrame
    b_field : str, optional
        The column from the second DataFrame

    Returns
    -------
    Tuple[int, int]

        - a in b: The number of a's unique values that are also found in b
        - b in a: The number of b's unique values that are also found in a
    """
    if b_field is None:
        b_field = a_field
    a_values = set(a[a_field].dropna().unique())
    b_values = set(b[b_field].dropna().unique())
    a_in_b = a_values.intersection(b_values)
    b_in_a = b_values.intersection(a_values)
    return len(a_in_b), len(b_in_a)


def ensure_categories(
    df: pd.DataFrame, df_other: pd.DataFrame, field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonize categorical levels between two DataFrames for a specified column.

    If both `df[field]` and `df_other[field]` are of pandas Categorical dtype,
    this routine computes the union of their categories (preserving the order
    from `df[field]` first, then any additional categories from
    `df_other[field]`) and sets both Series to use the combined category list.
    If either column is not categorical, the DataFrames are returned unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Primary DataFrame containing the categorical column to standardize.
    df_other : pandas.DataFrame
        Secondary DataFrame whose categorical column will be aligned to the
        same category set.
    field : str
        Name of the column in both DataFrames to synchronize categories on.

    Returns
    -------
    tuple of pandas.DataFrame
        A 2-tuple `(df_out, df_other_out)` where both DataFrames have their
        `field` column set to the same Categorical categories.  If the column
        dtype in either DataFrame is not Categorical, both DataFrames are
        returned without modification.
    """
    if hasattr(df[field].dtype, 'categories') and hasattr(
        df_other[field].dtype, 'categories'
    ):

        # union keeps order of appearance in the first operands
        cats = df[field].cat.categories.union(df_other[field].cat.categories)

        # give *both* Series the identical category list
        df[field] = df[field].cat.set_categories(cats)
        df_other[field] = df_other[field].cat.set_categories(cats)

    return df, df_other


def align_categories(
    df_left: pd.DataFrame, df_right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure matching categorical dtypes and unified category sets across two DataFrames.

    For each column present in either DataFrame, if *either* side has a
    pandas Categorical dtype, this function will:

    1. Convert the other side's column to Categorical (if not already), using
       the first side's existing categories.
    2. Compute the union of both categorical sets (preserving order: first
       df_left's then any new from df_right) and assign this combined set
       to both DataFrames.

    Parameters
    ----------
    df_left : pandas.DataFrame
        First DataFrame whose categorical columns will be aligned.
    df_right : pandas.DataFrame
        Second DataFrame whose categorical columns will be aligned.

    Returns
    -------
    left_aligned : pandas.DataFrame
        Copy of `df_left` where any column that was categorical in either input
        is now Categorical with the union of both category sets.
    right_aligned : pandas.DataFrame
        Copy of `df_right` similarly adjusted to share the same categories.

    Notes
    -----
    * Columns not of Categorical dtype on either side remain unchanged.
    * Missing values are preserved and treated as NaN in the Categorical dtype.
    * The original column order and non-categorical columns are unaffected.
    """

    for col in df_left.columns.union(df_right.columns):

        left_is_cat = hasattr(
            df_left.get(col, pd.Series(dtype="object")).dtype, 'categories'
        )
        right_is_cat = hasattr(
            df_right.get(col, pd.Series(dtype="object")).dtype, 'categories'
        )

        # If exactly one side is categorical, convert the other side first
        if left_is_cat and not right_is_cat:
            df_right[col] = pd.Categorical(
                df_right[col], categories=df_left[col].cat.categories
            )
            right_is_cat = True
        elif right_is_cat and not left_is_cat:
            df_left[col] = pd.Categorical(
                df_left[col], categories=df_right[col].cat.categories
            )
            left_is_cat = True

        # Now, if both are categorical, give them the same (union) category list
        if left_is_cat and right_is_cat:
            cats = df_left[col].cat.categories.union(df_right[col].cat.categories)
            df_left[col] = df_left[col].cat.set_categories(cats)
            df_right[col] = df_right[col].cat.set_categories(cats)

    return df_left, df_right


def calc_spatial_lag(
    df_sample: pd.DataFrame,
    df_univ: pd.DataFrame,
    value_fields: list[str],
    neighbors: int = 5,
    exclude_self_in_sample: bool = False,
) -> pd.DataFrame:
    """Compute spatial lag features via Gaussian-weighted averages of nearest neighbors.

    Builds a cKDTree on the coordinates in `df_sample` and, for each location in
    `df_univ`, finds its `neighbors` nearest points in `df_sample`.  A spatial lag
    is calculated for each field in `value_fields` as the weighted mean of the
    neighbor values using a Gaussian kernel with bandwidth equal to the mean
    neighbor distance (σ) for each prediction point.  Missing or zero distances
    are handled to avoid division by zero.  Optionally excludes the point itself
    when computing its own lag.

    Parameters
    ----------
    df_sample : pandas.DataFrame
        DataFrame of sample points containing at least columns 'latitude',
        'longitude', and each field in `value_fields`.  Used to train the
        nearest-neighbor tree and source values for lag computation.
    df_univ : pandas.DataFrame
        DataFrame of prediction points containing 'latitude' and 'longitude'.
        May include additional columns; output will append lag columns to this.
    value_fields : list of str
        List of column names in `df_sample` whose spatial lags will be computed.
    neighbors : int, default 5
        Number of nearest neighbors to query for each prediction point.  Must be
        at least 2 to allow exclusion of self when `exclude_self_in_sample=True`.
    exclude_self_in_sample : bool, default False
        If True, the nearest neighbor at distance zero (self) is excluded from
        the lag calculation by dropping the first neighbor in the query results.

    Returns
    -------
    pandas.DataFrame
        A copy of `df_univ` with new columns named 'spatial_lag_<field>' for each
        requested field.  Missing lag values are filled with the median of the
        corresponding field in `df_sample`.

    Raises
    ------
    ValueError
        If `neighbors < 2`, since at least two neighbors are required to compute
        a spatial lag (especially when excluding the self-distance).

    Notes
    -----
    - Uses SciPy’s cKDTree for efficient nearest-neighbor lookup.
    - Gaussian kernel weights are computed as:
      ```
      exp(–(d_ij²) / (2 · σ_i²))
      ```
      , where ``d_ij`` is the distance from point ``i`` to neighbor ``j``,
      and ``σ_i`` is the mean of its ``k`` neighbor distances.

    - Weights are then normalized so that they sum to 1 for each prediction point.


    """
    df = df_univ.copy()

    # Build a cKDTree from df_sales coordinates

    # we TRAIN on these coordinates -- coordinates that are NOT in the test set
    coords_train = df_sample[["latitude", "longitude"]].values
    tree = cKDTree(coords_train)

    # we PREDICT on these coordinates -- all the coordinates in the universe
    coords_all = df[["latitude", "longitude"]].values

    for value_field in value_fields:
        if value_field not in df_sample:
            print("Value field not in df_sample, skipping")
            continue

        # Choose the number of nearest neighbors to use
        k = neighbors  # You can adjust this number as needed

        # Query the tree: for each parcel in df_universe, find the k nearest parcels
        # distances: shape (n_universe, k); indices: corresponding indices in df_sales
        distances, indices = tree.query(coords_all, k=k)

        if exclude_self_in_sample:
            distances = distances[:, 1:]  # Exclude self-distance
            indices = indices[:, 1:]  # Exclude self-index

        # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
        if k < 2:
            raise ValueError("k must be at least 2 to compute spatial lag.")

        # For each universe parcel, compute sigma as the mean distance to its k neighbors.
        sigma = distances.mean(axis=1, keepdims=True)

        # Handle zeros in sigma
        sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

        # Compute Gaussian kernel weights for all neighbors
        weights = np.exp(-(distances**2) / (2 * sigma**2))

        # Normalize the weights so that they sum to 1 for each parcel
        weights_norm = weights / weights.sum(axis=1, keepdims=True)

        # Get the values corresponding to the neighbor indices
        parcel_values = df_sample[value_field].values
        neighbor_values = parcel_values[indices]  # shape (n_universe, k)

        # Compute the weighted average (spatial lag) for each parcel in the universe
        spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_values)).sum(
            axis=1
        )

        # Add the spatial lag as a new column
        df[f"spatial_lag_{value_field}"] = spatial_lag

        median_value = df_sample[value_field].median()
        df[f"spatial_lag_{value_field}"] = df[f"spatial_lag_{value_field}"].fillna(
            median_value
        )

    return df


def load_model_results(
    model_group: str,
    model_name: str,
    subset: str = "universe",
    model_type: str = "main",
) -> pd.DataFrame | None:
    """
    Load model prediction results for a specified subset from disk, if available.

    The function searches for prediction files under
    ``out/models/{model_group}/{model_type}/{model_name}`` in two formats:

    1. **Parquet**: Looks for either
       ``pred_{subset}.parquet`` or
       ``pred_{model_name}_{subset}.parquet``.  If found, reads the file,
       renames column ``key_x`` to ``key`` (if present), and returns a
       DataFrame with columns ``['key', 'prediction']``.

    2. **Pickle**: If no parquet is found, checks for
       ``pred_{subset}.pkl``.  Loads the pickled object (expected to have
       attributes ``df_universe``, ``df_sales``, and ``df_test``), selects
       the DataFrame matching ``subset``, and returns its ``['key',
       'prediction']`` columns.

    Parameters
    ----------
    model_group : str
        Top-level folder grouping for the model outputs (e.g., experiment
        or category name).
    model_name : str
        Subfolder name identifying the specific model within the group.
    subset : str, default "universe"
        Which dataset predictions to load.  Must be one of:
        - ``'universe'``: all parcels
        - ``'sales'``: parcels with sales
        - ``'test'``: test split
    model_type : str, default "main"
        Subdirectory under ``model_group`` for model variations
        (e.g., "main", "vacant", "hedonic").

    Returns
    -------
    pandas.DataFrame or None
        - A DataFrame with exactly two columns: ``'key'`` and
          ``'prediction'`` for the requested subset, if a prediction file
          was successfully found and read.
        - ``None`` if no matching prediction file exists on disk.
    """
    outpath = f"out/models/{model_group}/{model_type}"

    filepath = f"{outpath}/{model_name}"
    if os.path.exists(filepath):
        fpred = f"{filepath}/pred_{subset}.parquet"
        if not os.path.exists(fpred):
            fpred = f"{filepath}/pred_{model_name}_{subset}.parquet"

        if os.path.exists(fpred):
            df = pd.read_parquet(fpred)
            # if "key_x" in df:
                # # If the DataFrame has a 'key_x' column, rename it to 'key'
                # df.rename(columns={"key_x": "key"}, inplace=True)
            fields = [f for f in ["key", "key_sale", "prediction"] if f in df]
            df = df[fields].copy()
            return df

    fpred_results = f"{filepath}/pred_{subset}.pkl"
    if os.path.exists(fpred_results):
        if model_type != "main":
            with open(fpred_results, "rb") as file:
                results = pickle.load(file)
                if subset == "universe":
                    df = results.df_universe[["key", "prediction"]].copy()
                elif subset == "sales":
                    df = results.df_sales[["key", "key_sale", "prediction"]].copy()
                elif subset == "test":
                    df = results.df_test[["key", "key_sale", "prediction"]].copy()
                return df

    return None


def _left_wins(s1, s2):
    """
    Return a Series that keeps s1’s values wherever they’re non-NA,
    otherwise falls back to s2 – even when both are Categoricals.
    """
    if hasattr(s1.dtype, 'categories') and hasattr(s2.dtype, 'categories'):
        # make both columns share the **union** of their categories
        cats = s1.cat.categories.union(s2.cat.categories)
        s1 = s1.cat.set_categories(cats)
        s2 = s2.cat.set_categories(cats)

    # element-wise choose left over right
    return s1.where(s1.notna(), s2)


# TODO: WIP

def _encode_city_blocks(place: str):

    # 1. Download & simplify the OSM network
    highway_types = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
    ]
    custom_filter = f'["highway"~"{"|".join(highway_types)}"]'
    G = ox.graph_from_place(
        place, network_type="drive", simplify=True, custom_filter=custom_filter
    )

    # 2. Extract edges GeoDataFrame with u/v node IDs
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()[
        ["u", "v", "geometry", "name", "highway", "osmid"]
    ]

    # 3. Explode multilines, drop empties, reproject to metric CRS
    #    (choose a suitable projected CRS for distance-based ops)
    crs_eq = "EPSG:3857"
    edges = (
        edges.explode(index_parts=False)
        .dropna(subset=["geometry"])
        .to_crs(crs_eq)
        .reset_index(drop=True)
    )

    # 4. Unwrap any list‐values & fill missing names
    edges["road_name"] = edges["name"].apply(
        lambda v: v[0] if isinstance(v, (list, tuple)) else v
    )
    edges["road_type"] = edges["highway"].apply(
        lambda v: v[0] if isinstance(v, (list, tuple)) else v
    )
    # fallback: use osmid as a string if name was null
    edges["road_name"] = edges["road_name"].fillna(edges["osmid"].astype(str))

    # 5. Build node-->roads mapping, skipping service‐type roads
    node_to_names = defaultdict(set)
    for _, row in edges[["u", "v", "road_name", "road_type"]].iterrows():
        if row["road_type"] == "service":
            # never include service roads as cross‐streets
            continue
        node_to_names[row.u].add(row.road_name)
        node_to_names[row.v].add(row.road_name)

    # 6. Helper: pick the first “other” road at a junction
    def first_other(names_set, self_name):
        for nm in names_set:
            if nm != self_name:
                return nm
        return "?"

    # 7. Compute cross‐street names at each end
    edges["cross_w"] = [
        first_other(node_to_names[u], rn)
        for u, rn in zip(edges["u"], edges["road_name"])
    ]
    edges["cross_e"] = [
        first_other(node_to_names[v], rn)
        for v, rn in zip(edges["v"], edges["road_name"])
    ]

    # 8. Build the final name_loc field
    edges["name_loc"] = (
        edges["road_name"] + " between " + edges["cross_w"] + " and " + edges["cross_e"]
    )

    # 9. (Optional) drop service‐road segments entirely
    # edges = edges[edges.road_type != "service"]

    # Done!
    print(edges[["u", "v", "road_name", "cross_w", "cross_e", "name_loc"]].head())
