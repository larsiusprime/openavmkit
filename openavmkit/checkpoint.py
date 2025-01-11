import os

import pandas as pd


def read_checkpoint(path: str):
  parquet = f"out/checkpoints/{path}.parquet"
  if os.path.exists(parquet):
    return pd.read_parquet(parquet)
  return None


def write_checkpoint(df: pd.DataFrame, path: str):
  os.makedirs("out/checkpoints", exist_ok=True)
  df.to_parquet(f"out/checkpoints/{path}.parquet", index=False)