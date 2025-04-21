import os
import pickle
from typing import Any

import pandas as pd
import geopandas as gpd
from shapely import wkb

from openavmkit.utilities.geometry import is_likely_epsg4326


def from_checkpoint(path: str, func: callable, params: dict, use_checkpoint: bool = True)->pd.DataFrame:
  if use_checkpoint and exists_checkpoint(path):
    return read_checkpoint(path)
  else:
    result = func(**params)
    write_checkpoint(result, path)
    return result


def exists_checkpoint(path: str):
  extensions = ["parquet", "pickle"]
  for ext in extensions:
   if os.path.exists(f"out/checkpoints/{path}.{ext}"):
     return True
  return False


def read_checkpoint(path: str) -> Any:
  full_path = f"out/checkpoints/{path}.parquet"
  if os.path.exists(full_path):
    try:
      # Attempt to load as a GeoDataFrame
      return gpd.read_parquet(full_path)
    except ValueError:
      # Fallback to loading as a regular DataFrame
      df = pd.read_parquet(full_path)

      # Check if 'geometry' column exists and try to convert
      if 'geometry' in df.columns:
        df['geometry'] = df['geometry'].apply(wkb.loads)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

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
  os.makedirs("out/checkpoints", exist_ok=True)
  if isinstance(data, gpd.GeoDataFrame):
    data.to_parquet(f"out/checkpoints/{path}.parquet", engine="pyarrow")
  elif isinstance(data, pd.DataFrame):
    data.to_parquet(f"out/checkpoints/{path}.parquet")
  else:
    with open(f"out/checkpoints/{path}.pickle", "wb") as file:
      pickle.dump(data, file)


def delete_checkpoints(prefix: str):
  os.makedirs("out/checkpoints", exist_ok=True)
  for file in os.listdir("out/checkpoints"):
    if file.startswith(prefix):
      os.remove(f"out/checkpoints/{file}")


def read_pickle(path: str) -> Any:
  full_path = f"{path}.pickle"
  with open(full_path, "rb") as file:
    return pickle.load(file)
