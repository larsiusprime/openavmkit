import os

import pandas as pd


def from_checkpoint(path: str, func: callable, params: dict)->pd.DataFrame:
  if exists_checkpoint(path):
    return read_checkpoint(path)
  else:
    result = func(**params)
    write_checkpoint(result, path)
    return result


def exists_checkpoint(path: str):
  return os.path.exists(f"out/checkpoints/{path}.parquet")


def read_checkpoint(path: str):
  parquet = f"out/checkpoints/{path}.parquet"
  if os.path.exists(parquet):
    return pd.read_parquet(parquet)
  return None


def write_checkpoint(df: pd.DataFrame, path: str):
  os.makedirs("out/checkpoints", exist_ok=True)
  df.to_parquet(f"out/checkpoints/{path}.parquet", index=False)


def delete_checkpoints(prefix: str):
  for file in os.listdir("out/checkpoints"):
    if file.startswith(prefix):
      os.remove(f"out/checkpoints/{file}")