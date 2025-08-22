import os
import pickle
from typing import Any

import pandas as pd
import geopandas as gpd
from shapely import wkb

from openavmkit.utilities.geometry import is_likely_epsg4326


def from_checkpoint(
    path: str, func: callable, params: dict, use_checkpoint: bool = True
) -> pd.DataFrame:
    """Run a function with parameters, using a checkpoint if available.

    If a checkpoint exists at the specified path, it will read from it,
    return the results, and not execute the function.

    If a checkpoint does not exist, it will execute the function with
    the provided parameters, save the result to a checkpoint, and return
    the result.

    Parameters
    ----------
    path : str
        The path to the checkpoint file (without extension).
    func : callable
        The function to execute if the checkpoint does not exist.
    params : dict
        The parameters to pass to the function.
    use_checkpoint : bool, optional
        Whether to use the checkpoint if it exists. Defaults to True.

    Returns
    -------
    pd.DataFrame
        The result of the function execution or the checkpoint data.
    """
    if use_checkpoint and exists_checkpoint(path):
        return read_checkpoint(path)
    else:
        result = func(**params)
        write_checkpoint(result, path)
        return result


def exists_checkpoint(path: str):
    """Check if a checkpoint exists at the specified path.

    Parameters
    ----------
    path : str
        The path to the checkpoint file (without extension).

    Returns
    -------
    bool
        True if a checkpoint exists, False otherwise.
    """
    extensions = ["parquet", "pickle"]
    for ext in extensions:
        if os.path.exists(f"out/checkpoints/{path}.{ext}"):
            return True
    return False


def read_checkpoint(path: str) -> Any:
    """Read a checkpoint from the specified path.

    Parameters
    ----------
    path : str
        The path to the checkpoint file (without extension).

    Returns
    -------
    Any
        The data read from the checkpoint, which can be a DataFrame or GeoDataFrame.
    """
    full_path = f"out/checkpoints/{path}.parquet"
    if os.path.exists(full_path):
        try:
            # Attempt to load as a GeoDataFrame
            return gpd.read_parquet(full_path)
        except ValueError:
            # Fallback to loading as a regular DataFrame
            df = pd.read_parquet(full_path)

            # Check if 'geometry' column exists and try to convert
            if "geometry" in df.columns:
                df["geometry"] = df["geometry"].apply(wkb.loads)
                gdf = gpd.GeoDataFrame(df, geometry="geometry")

                # Try to infer if CRS is EPSG:4326
                if is_likely_epsg4326(gdf):
                    gdf.set_crs(epsg=4326, inplace=True)
                    return gdf
                else:
                    raise ValueError(
                        "Parquet found with geometry, but CRS is ambiguous. Failed to load."
                    )
        # Return as a regular DataFrame if no geometry column
        return df
    else:
        # If we don't find a parquet file, try to load a pickle
        full_path = f"out/checkpoints/{path}.pickle"
        with open(full_path, "rb") as file:
            return pickle.load(file)


def write_checkpoint(data: Any, path: str):
    """Write data to a checkpoint file.

    Parameters
    ----------
    data : Any
        The data to write to the checkpoint, which can be a DataFrame or GeoDataFrame.
    path : str
        The path to the checkpoint file (without extension).
    """
    os.makedirs("out/checkpoints", exist_ok=True)
    if isinstance(data, gpd.GeoDataFrame):
        data.to_parquet(f"out/checkpoints/{path}.parquet", engine="pyarrow")
    elif hasattr(data, 'to_numpy'):
        data.to_parquet(f"out/checkpoints/{path}.parquet")
    else:
        with open(f"out/checkpoints/{path}.pickle", "wb") as file:
            pickle.dump(data, file)


def delete_checkpoints(prefix: str):
    """Delete all checkpoint files that start with the given prefix.

    Parameters
    ----------
    prefix : str
        The prefix to match checkpoint files against.
    """
    os.makedirs("out/checkpoints", exist_ok=True)
    for file in os.listdir("out/checkpoints"):
        if file.startswith(prefix):
            os.remove(f"out/checkpoints/{file}")


def read_pickle(path: str) -> Any:
    """Read a pickle file from the specified path.

    Parameters
    ----------
    path : str
        The path to the pickle file (without extension).

    Returns
    -------
    Any
        The data read from the pickle file.
    """
    full_path = f"{path}.pickle"
    with open(full_path, "rb") as file:
        return pickle.load(file)
