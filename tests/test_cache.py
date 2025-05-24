import os

import geopandas
import pandas as pd

from openavmkit.synthetic.basic import generate_basic
from openavmkit.utilities.assertions import dicts_are_equal, dfs_are_equal
from openavmkit.utilities.cache import check_cache, write_cache, read_cache, get_cached_df, write_cached_df, clear_cache
from openavmkit.utilities.geometry import ensure_geometries


def test_cache():
  signature = {
    'id': '12345'
  }

  synthetic = generate_basic(100)
  gdf = synthetic.df_universe
  df = gdf.drop(columns=["geometry"])
  df = pd.DataFrame(df)

  trials = [
    {
      "extension": "txt",
      "filetype": "str",
      "payload": "I am the very model of a modern Major General, I've information vegetable, animal, and mineral. I know the kings of England, and I quote the fights historical, from Marathon to Waterloo, in order categorical. I'm very well acquainted too with matters mathematical, I understand equations, both the simple and quadratical. About binomial theorem I'm teeming with a lot o' news, with many cheerful facts about the square of the hypotenuse."
    },
    {
      "extension": "json",
      "filetype": "dict",
      "payload":{
        'a': [i for i in range(0, 3)],
        'b': [i*2 for i in range(0, 3)],
        'c': [i*3 for i in range(0, 3)],
      }
    },
    {
      "extension": "parquet",
      "filetype": "df",
      "payload": df
    },
    {
      "extension": "parquet",
      "filetype": "df",
      "payload": gdf
    }
  ]

  for trial in trials:

    ext = trial["extension"]
    filetype = trial["filetype"]
    payload: dict | str | pd.DataFrame | geopandas.GeoDataFrame = trial["payload"]

    os.makedirs("cache", exist_ok=True)
    # clear every FILE in the cache:
    for file in os.listdir("cache"):
      file_path = os.path.join("cache", file)
      if os.path.isfile(file_path):
        os.remove(file_path)

    is_cached = check_cache("test_cache", signature, filetype)

    assert is_cached == False

    write_cache("test_cache", payload, signature, filetype)

    # check if the cache file exists:
    assert os.path.exists(f"cache/test_cache.{ext}")

    # check if the cache file is not empty:
    assert os.path.getsize(f"cache/test_cache.{ext}") > 0

    # check if the cache file is equal to the payload:
    if filetype == "dict":
      with open("cache/test_cache.json", "r") as f:
        cache = f.read()
        assert str(cache).replace("\"", "'") == str(payload).replace("\"", "'")
    elif filetype == "str":
      with open("cache/test_cache.txt", "r") as f:
        cache = f.read()
        assert str(cache) == str(payload)
    elif filetype == "df" or filetype == "gdf":
      try:
        cache = geopandas.read_parquet("cache/test_cache.parquet")
        if "geometry" in cache:
          cache = ensure_geometries(cache, "geometry", cache.crs)
      except ValueError:
        cache = pd.read_parquet("cache/test_cache.parquet")

      assert dfs_are_equal(cache, payload, "key")

    is_cached = check_cache("test_cache", signature, filetype)

    assert is_cached == True

    # read the cache file:
    cached_file = read_cache("test_cache", filetype)

    # check if the cached file is equal to the payload:
    if filetype == "dict":
      assert dicts_are_equal(cached_file, payload)
    elif filetype == "str":
      assert str(cached_file) == str(payload)
    elif filetype == "df" or filetype == "gdf":
      assert dfs_are_equal(cached_file, payload, "key")

    # change the signature:
    dirty_signature = {
      "id": "54321"
    }

    # check the cache with the wrong signature:
    is_cached = check_cache("test_cache", dirty_signature, filetype)

    assert is_cached == False


def test_cache_df():

  # clear every FILE in the cache:
  for file in os.listdir("cache"):
    file_path = os.path.join("cache", file)
    if os.path.isfile(file_path):
      os.remove(file_path)

  data = {
    "key": [1, 2, 3],
    "fruit": ["apple", "banana", "cherry"],
    "quantity": [10, 20, 30],
    "price": [0.5, 0.25, 0.75]
  }
  df = pd.DataFrame(data)

  expected = {
    "key": [1, 2, 3],
    "fruit": ["snapple", "banana", "cherry"],
    "quantity": [10, 20, 36],
    "price": [0.5, 0.25, 0.75],
    "calories": [52, 89, 50],
    "score": [0.8, 0.9, 0.7],
    "awesomeness": [0.9, 0.95, 0.85]
  }
  df_expected = pd.DataFrame(expected)

  def enrich_fruit(df_in: pd.DataFrame) -> (pd.DataFrame, str):
    _df = get_cached_df(df_in, "fruit", "key")
    if _df is not None:
      return _df, "cached"

    _df = df_in.copy()
    _df["calories"] = [52, 89, 50]
    _df["score"] = [0.8, 0.9, 0.7]
    _df["awesomeness"] = [0.9, 0.95, 0.85]

    _df.loc[_df["key"].eq(1), "fruit"] = "snapple"
    _df.loc[_df["key"].eq(3), "quantity"] = 36

    write_cached_df(df_in, _df, "fruit", "key")
    return _df, "uncached"

  df_enriched, was_cached = enrich_fruit(df)

  assert was_cached == "uncached"
  assert dfs_are_equal(df_enriched, df_expected, "key")

  df_enriched, was_cached = enrich_fruit(df)

  assert was_cached == "cached"
  assert dfs_are_equal(df_enriched, df_expected, "key")
  clear_cache("fruit", "df")


def test_cache_df2():

  clear_cache("synthetic", "df")

  synthetic = generate_basic(100)
  gdf = synthetic.df_universe

  gdf_skinny = gdf[["key", "geometry"]].copy()
  gdf_stuff = gdf.drop(columns=["geometry"]).copy()

  def enrich_gdf(gdf_in: geopandas.GeoDataFrame):
    gdf_out = get_cached_df(gdf_in, "synthetic", "key")
    if gdf_out is not None:
      return gdf_out, "cached"

    gdf_out = gdf_in.copy()
    gdf_out = gdf_out.merge(gdf_stuff, on="key", how="left")

    write_cached_df(gdf_in, gdf_out, "synthetic", "key")
    return gdf_out, "uncached"

  gdf_enriched, was_cached = enrich_gdf(gdf_skinny)
  assert was_cached == "uncached"
  assert dfs_are_equal(gdf_enriched, gdf, "key")

  gdf_enriched, was_cached = enrich_gdf(gdf_skinny)
  assert was_cached == "cached"
  assert dfs_are_equal(gdf_enriched, gdf, "key")
  clear_cache("synthetic", "df")


def test_cache_df3():

  clear_cache("synthetic", "df")

  synthetic = generate_basic(100)
  gdf = synthetic.df_universe

  gdf_extra_rows = gdf.copy()
  gdf_extra_rows["key"] = gdf_extra_rows["key"].astype(str) + "_extra"

  gdf_skinny = gdf[["key", "geometry"]].copy()
  gdf_stuff = gdf.drop(columns=["geometry"]).copy()

  # Also stick a bunch of extra rows on
  gdf = pd.concat([gdf, gdf_extra_rows], ignore_index=False)

  def enrich_gdf(gdf_in: geopandas.GeoDataFrame):
    gdf_out = get_cached_df(gdf_in, "synthetic", "key")
    if gdf_out is not None:
      return gdf_out, "cached"

    # Enrichment merges the skinny gdf with the stuff gdf, and adds some extra rows
    gdf_out = gdf_in.copy()
    gdf_out = gdf_out.merge(gdf_stuff, on="key", how="left")
    gdf_out = pd.concat([gdf_out, gdf_extra_rows], ignore_index=False)

    write_cached_df(gdf_in, gdf_out, "synthetic", "key")
    return gdf_out, "uncached"

  gdf_enriched, was_cached = enrich_gdf(gdf_skinny)

  assert was_cached == "uncached"
  assert dfs_are_equal(gdf_enriched, gdf, "key")

  gdf_enriched, was_cached = enrich_gdf(gdf_skinny)
  assert was_cached == "cached"
  assert dfs_are_equal(gdf_enriched, gdf, "key")
  clear_cache("synthetic", "df")