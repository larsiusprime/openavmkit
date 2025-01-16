import os

import pandas as pd
import geopandas as gpd
from shapely import wkb

from openavmkit.utilities.geometry import is_likely_epsg4326


def from_checkpoint(path: str, func: callable, params: dict)->pd.DataFrame:
  if exists_checkpoint(path):
    return read_checkpoint(path)
  else:
    result = func(**params)
    write_checkpoint(result, path)
    return result


def exists_checkpoint(path: str):
  return os.path.exists(f"out/checkpoints/{path}.parquet")


def read_checkpoint(path: str) -> pd.DataFrame | gpd.GeoDataFrame:
  full_path = f"out/checkpoints/{path}.parquet"

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


def write_checkpoint(df: pd.DataFrame | gpd.GeoDataFrame, path: str):
  os.makedirs("out/checkpoints", exist_ok=True)
  if isinstance(df, gpd.GeoDataFrame):
    df.to_parquet(f"out/checkpoints/{path}.parquet", index=False, engine="pyarrow")
  else:
    df.to_parquet(f"out/checkpoints/{path}.parquet", index=False)


def delete_checkpoints(prefix: str):
  for file in os.listdir("out/checkpoints"):
    if file.startswith(prefix):
      os.remove(f"out/checkpoints/{file}")