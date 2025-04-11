import os
import json
import pandas as pd
import geopandas as gpd

from openavmkit.utilities.assertions import objects_are_equal


def write_cache(
    filename: str,
    payload: dict | str | pd.DataFrame | gpd.GeoDataFrame,
    signature: dict | str,
    filetype: str
):
  extension = _get_extension(filetype)
  path = f"cache/{filename}.{extension}"
  base_path = os.path.dirname(path)
  os.makedirs(base_path, exist_ok=True)
  with open(path, "w") as file:
    if filetype == "dict":
      json.dump(payload, file)
    elif filetype == "str":
      file.write(payload)
    elif filetype == "gdf":
      if isinstance(payload, gpd.GeoDataFrame):
        payload.to_parquet(path)
      else:
        raise TypeError("Payload must be a GeoDataFrame for gdf type.")
    elif filetype == "df":
      if isinstance(payload, pd.DataFrame):
        payload.to_parquet(path)
      else:
        raise TypeError("Payload must be a DataFrame for df type.")

  if type(signature) is dict:
    sig_ext = "json"
  elif type(signature) is str:
    sig_ext = "txt"
  else:
    raise TypeError(f"Unsupported type for signature value: {type(signature)} sig = {signature}")

  signature_path = f"cache/{filename}.signature.{sig_ext}"
  with open(signature_path, "w") as file:
    if sig_ext == "json":
      json.dump(signature, file)
    else:
      file.write(signature)


def read_cache(
    filename: str,
    filetype: str
):
  extension = _get_extension(filetype)
  path = f"cache/{filename}.{extension}"
  if os.path.exists(path):
    with open(path, "r") as file:
      if filetype == "dict":
        return json.load(file)
      elif filetype == "str":
        return file.read()
      elif filetype == "gdf":
        import geopandas as gpd
        return gpd.read_parquet(path)
      elif filetype == "df":
        import pandas as pd
        return pd.read_parquet(path)
  return None


def check_cache(
    filename: str,
    signature: dict | str
):
  path = f"cache/{filename}"
  match = _match_signature(path, signature)
  print(f"Check cache match for '{filename}' = ? {match}")
  if match:
    return os.path.exists(path)
  return False


def _match_signature(
    filename: str,
    signature: dict | str
)->bool:
  if type(signature) is dict:
    sig_ext = "json"
  elif type(signature) is str:
    sig_ext = "txt"
  else:
    raise TypeError(f"Unsupported type for signature value: {type(signature)}")
  sig_file = f"{filename}.signature.{sig_ext}"
  match = False
  if os.path.exists(sig_file):
    with open(sig_file, "r") as file:
      signature = file.read()
    if sig_ext == ".json":
      signature = json.loads(signature)
      match = objects_are_equal(signature, signature)
    else:
      match = signature == signature
  return match


def _get_extension(filetype:str):
  if filetype == "dict":
    return "json"
  elif filetype == "str":
    return "txt"
  elif filetype == "gdf":
    return "parquet"
  elif filetype == "df":
    return "parquet"
  raise ValueError(f"Unsupported filetype: {filetype}")
