import json
import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_integer_dtype, is_float_dtype


def to_parquet_safe(df : pd.DataFrame, path: str, geometry_col : str | None = None):
    """
    Writes the dataframe to parquet, performing a bunch of safety checks so you don't have to
    
    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe or geopandas geodataframe
    path : str
        The path you want to write the parquet to
    geometry_col : str
        The name of the geometry column. Default is None
    """
    df_san = sanitize_df(df, geometry_col = geometry_col)
    if hasattr(df_san, 'to_numpy'):
        if hasattr(df_san, 'geometry'):  # GeoDataFrame-like
            df_san.to_parquet(path, engine="pyarrow")
        else:  # DataFrame-like
            df_san.to_parquet(path)
    else:
        raise TypeError("df must be a DataFrame.")


def sanitize_df(df : pd.DataFrame, geometry_col: str | None = None):
    df = df.copy()
    
    if geometry_col is None and "geometry" in df:
        geometry_col = "geometry"
    
    for col in df.columns:
        if geometry_col and col == geometry_col:
            continue
        s = df[col]

        # Already good: numeric, bool, datetime, categorical, nullable ints
        if (is_integer_dtype(s) or is_float_dtype(s) or
            pd.api.types.is_bool_dtype(s) or
            pd.api.types.is_datetime64_any_dtype(s) or
            pd.api.types.is_categorical_dtype(s) or
            pd.api.types.is_string_dtype(s) or
            pd.api.types.is_extension_array_dtype(s)):  # covers 'Int64', 'string', etc.
            continue

        if s.dtype == object:
            kind = infer_dtype(s, skipna=True)
            if kind in {"integer", "mixed-integer", "mixed-integer-float", "floating"}:
                df[col] = pd.to_numeric(s, errors="coerce").astype("Float64")
            elif kind in {"string", "unicode", "mixed"}:
                # If there are dicts/lists lurking, stringify them
                if s.map(lambda x: isinstance(x, (dict, list, set, tuple))).any():
                    df[col] = s.map(lambda x: json.dumps(x) if isinstance(x, (dict, list, set, tuple)) else x).astype("string")
                else:
                    df[col] = s.astype("string")
            else:
                # Fallback: stringify unknown/mixed objects
                df[col] = s.astype("string")
    return df