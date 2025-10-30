import os
import json
import pickle

import pandas as pd
import geopandas as gpd

from openavmkit.utilities.assertions import (
    objects_are_equal,
    dicts_are_equal,
    dfs_are_equal,
)
from openavmkit.utilities.geometry import ensure_geometries


def write_cache(
    filename: str,
    payload: dict | str | pd.DataFrame | gpd.GeoDataFrame | bytes,
    signature: dict | str,
    filetype: str,
) -> None:
    """Caches the data to disk

    Parameters
    ----------
    filename : str
        The filename to associate with this data
    payload : dict | str | pd.DataFrame | gpd.GeoDataFrame | bytes
        The data to cache to disk
    signature : dict | str
        A "signature" value that, if changed in the future, indicates the cache has been broken.
        That is, when you go to load cached data, if the signature has changed from the one written to disk, the system
        knows not to trust the cached data and will generate the data from scratch instead. If the signatures match,
        the system will trust the cached data and skip the potentially expensive generation step.
    filetype : str
        The type of file ("dict", "str", "pickle", or "df")
    """
    extension = _get_extension(filetype)
    path = f"cache/{filename}.{extension}"
    base_path = os.path.dirname(path)
    os.makedirs(base_path, exist_ok=True)
    if filetype == "dict":
        with open(path, "w") as file:
            json.dump(payload, file)
    elif filetype == "str":
        with open(path, "w") as file:
            file.write(payload)
    elif filetype == "pickle":
        with open(path, "wb") as file:
            pickle.dump(payload, file)
    elif filetype == "df":
        if hasattr(payload, 'to_numpy'):
            if hasattr(payload, 'geometry'):  # GeoDataFrame-like
                payload.to_parquet(path, engine="pyarrow")
            else:  # DataFrame-like
                payload.to_parquet(path)
        else:
            raise TypeError("Payload must be a DataFrame for df type.")

    if type(signature) is dict:
        sig_ext = "json"
    elif type(signature) is str:
        sig_ext = "txt"
    else:
        raise TypeError(
            f"Unsupported type for signature value: {type(signature)} sig = {signature}"
        )

    signature_path = f"cache/{filename}.signature.{sig_ext}"
    with open(signature_path, "w") as file:
        if sig_ext == "json":
            json.dump(signature, file)
        else:
            file.write(signature)


def read_cache(filename: str, filetype: str) -> dict | str | object | pd.DataFrame | gpd.GeoDataFrame | None:
    """Reads cached data from disk

    Parameters
    ----------
    filename : str
        The filename of the data to load (relative to the cache/ directory)
    filetype : str
        The type of file ("dict", "str", "pickle", or "df")

    Returns
    -------
    dict | str | object | pd.DataFrame | gpd.GeoDataFrame | None
        The cached data
    """
    extension = _get_extension(filetype)
    path = f"cache/{filename}.{extension}"
    if os.path.exists(path):
        if filetype == "dict":
            with open(path, "r") as file:
                return json.load(file)
        elif filetype == "str":
            with open(path, "r") as file:
                return file.read()
        elif filetype == "pickle":
            with open(path, "rb") as file:
                return pickle.load(file)
        elif filetype == "df":
            try:
                df = gpd.read_parquet(path)
                if "geometry" in df:
                    df = gpd.GeoDataFrame(df, geometry="geometry")
                    ensure_geometries(df, "geometry", df.crs)
            except ValueError:
                df = pd.read_parquet(path)
            return df
    return None


def check_cache(filename: str, signature: dict | str, filetype: str) -> bool:
    """Check if the cached data exists and if the signatures match

    Parameters
    ----------
    filename : str
        The filename of the cached data
    signature : str
        The signature of the cached data
    filetype : str
        The type of file ("dict", "str", "pickle", or "df")

    Returns
    -------
    bool
        True if the file exists AND the signatures match, False otherwise.
    """
    ext = _get_extension(filetype)
    path = f"cache/{filename}"
    match = _match_signature(path, signature)
    if match:
        path_exists = os.path.exists(f"{path}.{ext}")
        return path_exists
    return False


def clear_cache(filename: str, filetype: str) -> None:
    """Clear the specified cache data

    Parameters
    ----------
    filename : str
        The filename of the cached data to clear
    filetype : str
        The type of file ("dict", "str", "pickle", or "df")
    """
    ext = _get_extension(filetype)
    path = f"cache/{filename}"
    if os.path.exists(f"{path}.{ext}"):
        os.remove(f"{path}.{ext}")
    if os.path.exists(f"{path}.cols{ext}"):
        os.remove(f"{path}..cols.{ext}")
    if os.path.exists(f"{path}.rows{ext}"):
        os.remove(f"{path}.rows{ext}")
    if os.path.exists(f"{path}.signature.json"):
        os.remove(f"{path}.signature.json")
    if os.path.exists(f"{path}.cols.signature.json"):
        os.remove(f"{path}.cols.signature.json")
    if os.path.exists(f"{path}.rows.signature.json"):
        os.remove(f"{path}.rows.signature.json")


def write_cached_df(
    df_orig: pd.DataFrame,
    df_new: pd.DataFrame,
    filename: str,
    key: str = "key",
    extra_signature: dict | str = None,
    changed_cols: list[str] = None,
    check_equal: bool = True
) -> pd.DataFrame | None:
    """Update an on-disk cache with row- or column-level differences between two
    ``pandas`` DataFrames and return the fully reconstructed, cached DataFrame.

    The function compares *df_new* against *df_orig* using the primary key
    column *key*.  Any newly added rows, deleted rows, or modified columns are
    written to cache files ``<filename>.rows`` and/or ``<filename>.cols`` via
    :pyfunc:`write_cache`.  A deterministic signature (optionally augmented by
    *extra_signature*) is stored alongside each cache fragment to protect
    against cache poisoning.  The routine then calls
    :pyfunc:`get_cached_df` to rebuild the complete DataFrame and verifies that
    the round-trip result is equal to *df_new* (allowing for NaN equality and
    primary-key re-ordering).

    Parameters
    ----------
    df_orig : pandas.DataFrame
        The **baseline** DataFrame previously loaded from cache (or computed
        earlier in the session).  Must contain the *key* column.
    df_new : pandas.DataFrame
        The **candidate** DataFrame whose content should be cached.
    filename : str
        Base file name (no extension) used when writing cache fragments.
        Two files may be created:

        * ``<filename>.cols`` – modified columns for existing rows
        * ``<filename>.rows`` – entirely new rows
    key : str, default ``"key"``
        Name of the primary-key column that uniquely identifies each row.
    extra_signature : dict or str, optional
        Additional entropy to include in the cache signature.  Use this when
        the same data structure can vary by external configuration (e.g.,
        feature flags or environment).
    changed_cols : list[str]
        (optional) A list of modified columns, if known in advance. Default is None. Supplying this skips potentially costly modification checks.
    check_equal : bool
        (optional) Check if the cached result yields the same as the intended result. Default is True.

    Returns
    -------
    pandas.DataFrame or None
        * If **no** columns/rows differ between *df_new* and *df_orig*, the
          original DataFrame *df_orig* is returned immediately and **no** cache
          files are written.
        * Otherwise, the function returns the DataFrame reconstructed from the
          cache (identical in content to *df_new*).  The return type is never
          *None* unless the underlying helpers are altered.

    Raises
    ------
    ValueError
        If the DataFrame reconstructed from cache does **not** match *df_new*
        after a diff has been written.  This guards against cache corruption or
        mismatched signatures.

    Notes
    -----
    * Column comparison treats *NaN* as equal to *NaN*.
    * Only the minimum required data (changed columns and/or new rows) is
      written, which keeps cache artifacts small even for wide tables.
    * Deleted rows are **not** written to disk; instead they are omitted when
      *df_new* is reconstructed.

    Examples
    --------
    >>> df_baseline = pd.DataFrame({"key": [1, 2], "a": [10, 20]})
    >>> df_update   = pd.DataFrame({"key": [1, 2, 3], "a": [10, 99, 30]})
    >>> df_cached   = write_cached_df(df_baseline, df_update, "mycache")
    >>> df_cached.equals(df_update)
    True
    """

    df_new = df_new.copy()

    orig_cols = set(df_orig.columns)
    new_cols = [c for c in df_new.columns if c not in orig_cols]
    common = [c for c in df_new.columns if c in orig_cols]

    orig_rows_by_key = df_orig[key].values
    new_rows_by_key = df_new[key].values
    if len(orig_rows_by_key) > len(new_rows_by_key):
        added_rows = [key for key in new_rows_by_key if key not in orig_rows_by_key]
        orig_set = set(orig_rows_by_key)
        new_set = set(new_rows_by_key)
        added_rows = []
        deleted_rows = list(orig_set - new_set)
    elif len(orig_rows_by_key) < len(new_rows_by_key):
        orig_set = set(orig_rows_by_key)
        new_set = set(new_rows_by_key)
        added_rows = list(new_set - orig_set)
        deleted_rows = []
    else:
        added_rows = []
        deleted_rows = []
    
    if not changed_cols:
        modified = []
        for c in common:
            col_new = df_new[c].reset_index(drop=True)
            col_orig = df_orig[c].reset_index(drop=True)

            is_different = False
            if len(col_new) == len(col_orig):
                values_equal = col_new.values == col_orig.values
                na_equal = col_new.isna() & col_orig.isna()

                count_na_equal = na_equal.sum()
                count_values_equal = values_equal.sum()

                count_to_match = len(col_new)

                all_equal = (
                    count_na_equal == count_to_match
                    and count_values_equal == count_to_match
                )
                if not all_equal:
                    is_different = True
            else:
                is_different = True

            if is_different:
                modified.append(c)
                continue

        changed_cols = new_cols + modified
    
    if not changed_cols:
        # nothing new or modified → no cache update needed
        return df_orig

    the_cols = changed_cols
    if key not in the_cols:
        the_cols = [key] + changed_cols
    
    df_diff_cols = df_new[the_cols].copy()
    df_diff_cols = df_diff_cols[~df_diff_cols[key].isin(added_rows)]
    signature = _get_df_signature(df_orig, extra_signature)
    df_type = "df"
    write_cache(f"{filename}.cols", df_diff_cols, signature, df_type)
    if len(deleted_rows) > 0:
        df_new = df_new[~df_new[key].isin(deleted_rows)].copy()

    if len(added_rows) > 0:
        df_diff_rows = df_new[df_new[key].isin(added_rows)].copy()
        if not df_diff_rows.empty:
            write_cache(f"{filename}.rows", df_diff_rows, signature, df_type)

    df_cached = get_cached_df(df_orig, filename, key, extra_signature)
    
    if check_equal:
        are_equal = dfs_are_equal(df_new, df_cached, allow_weak=True, primary_key=key)
        if not are_equal:
            raise ValueError(f"Cached DataFrame does not match the original DataFrame.")

    return df_cached


def get_cached_df(
    df: pd.DataFrame,
    filename: str,
    key: str = "key",
    extra_signature: dict | str = None,
    only_signature: dict | str = None,
) -> pd.DataFrame | gpd.GeoDataFrame | None:
    """
    Reconstruct a DataFrame from cached row and column diffs on disk.

    This function looks for cache fragments named
    ``<filename>.cols`` (column-level diffs) and
    ``<filename>.rows`` (new row fragments), validates them against a
    signature derived from the base DataFrame and optional ``extra_signature``,
    and merges them to produce an updated DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The base DataFrame to which column diffs will be applied.  This
        should correspond to the state of the data when the cache
        fragments were generated.
    filename : str
        Base file name (no extension) for the cache files.  The function
        will check for ``<filename>.cols`` and ``<filename>.rows``.
    key : str, default 'key'
        Name of the primary-key column to align rows between the base DataFrame
        and cached fragments.
    extra_signature : dict or str, optional
        Additional signature to include when computing the cache signature via
        ``_get_df_signature``.
    only_signature : dict or str, optional
        If provided, this signature is used directly (instead of recomputing
        from ``df`` + ``extra_signature``) when validating cache fragments.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame or None

        - A merged DataFrame incorporating all column updates and appended rows
          from the cache.  The returned type matches ``df`` (GeoDataFrame if
          geometry diffs were cached).
        - Returns ``None`` if neither ``<filename>.cols`` nor
          ``<filename>.rows`` exist or pass the signature check.

    Notes
    -----
    - Column diffs are read first: existing rows in ``df`` are filtered to
      those present in the columns fragment, old columns dropped, and new
      columns merged in.
    - Row fragments are then appended to the merged DataFrame.
    - The function preserves the data type of the primary key column.
    - If both column and row fragments are absent or invalid, the function
      returns ``None``.
    """

    if only_signature is not None:
        signature = only_signature
    else:
        signature = _get_df_signature(df, extra_signature)

    filename_rows = f"{filename}.rows"
    filename_cols = f"{filename}.cols"

    df_merged = None

    if check_cache(filename_cols, signature, "df"):
        # Merge new columns
        df_diff = read_cache(filename_cols, "df")
        if not df_diff is None and not df_diff.empty:
            df_diff[key] = df_diff[key].astype(df[key].dtype)

            cols_to_replace = [c for c in df_diff.columns if c != key]

            # Drop the columns that are going to be replaced
            df_base = df.drop(columns=cols_to_replace, errors="ignore")

            # Drop the keys that are not in the diff
            df_base = df_base[df_base["key"].isin(df_diff[key])].copy()

            df_merged = df_base.merge(df_diff, how="left", on=key)

            if hasattr(df_diff, 'geometry'):  # GeoDataFrame-like
                df_merged = gpd.GeoDataFrame(df_merged, geometry="geometry")
                df_merged = ensure_geometries(df_merged, "geometry", df_diff.crs)

    if check_cache(filename_rows, signature, "df"):
        # Add new rows
        df_diff = read_cache(filename_rows, "df")
        if not df_diff is None and not df_diff.empty:
            df_diff[key] = df_diff[key].astype(df[key].dtype)

            if df_merged is None:
                df_merged = df.copy()

            # add the new rows onto the end of the DataFrame
            df_merged = pd.concat([df_merged, df_diff], ignore_index=True)

    return df_merged


#######################################
# PRIVATE
#######################################


def _get_model_group_signature(df: pd.DataFrame)->dict:
    if "model_group" in df:
        vcs = df["model_group"].value_counts()
        sig = {}
        for key in vcs.index:
            sig[key] = vcs[key].astype("str")
        return sig
    return {}


def _get_df_signature(df: pd.DataFrame, extra: dict | str = None) -> dict:
    sorted_columns = sorted(df.columns)
    signature = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": sorted_columns,
        "model_groups": _get_model_group_signature(df)
    }
    if extra is not None:
        signature["extra"] = extra
    return signature


def _match_signature(filename: str, signature: dict | str) -> bool:
    if type(signature) is dict:
        sig_ext = "json"
    elif type(signature) is str:
        sig_ext = "txt"
    else:
        raise TypeError(f"Unsupported type for signature value: {type(signature)}")
    sig_file = f"{filename}.signature.{sig_ext}"
    match = False
    if os.path.exists(sig_file):
        if sig_ext == "json":
            with open(sig_file, "r") as file:
                cache_signature = json.load(file)
            match = dicts_are_equal(signature, cache_signature)
        else:
            with open(sig_file, "r") as file:
                cache_signature = file.read()
            match = signature == cache_signature
    return match


def _get_extension(filetype: str) -> str:
    if filetype == "dict":
        return "json"
    elif filetype == "str":
        return "txt"
    elif filetype == "df":
        return "parquet"
    elif filetype == "pickle":
        return "pickle"
    elif filetype == "json":
        raise ValueError(f"Filetype 'json' is unsupported, did you mean 'dict'?")
    elif filetype == "txt" or filetype == "text":
        raise ValueError(f"Filetype '{filetype}' is unsupported, did you mean 'str'?")
    elif filetype == "parquet":
        raise ValueError(f"Filetype 'parquet' is ambiguous: please use 'df' instead")
    raise ValueError(f"Unsupported filetype: '{filetype}'")
