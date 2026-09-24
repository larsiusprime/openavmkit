"""clear_cache must remove the .cols / .rows diff fragments, not just the base file.

``write_cached_df`` stores column-level and row-level diffs under ``<filename>.cols``
and ``<filename>.rows``, which ``write_cache`` lands at ``cache/<filename>.cols.<ext>``.
``clear_cache`` looked for ``cache/<filename>.cols<ext>`` -- no dot before the
extension -- so the existence check never matched and the fragments were silently left
behind. (The removal path it would have used was wrong too: a doubled dot.)

These caches are the large ones. On a real locality the column-diff parquet is the bulk
of the cache directory, so "clearing" it freed almost nothing and a stale fragment could
outlive the base file it belonged to.
"""
import os

import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities.cache import write_cached_df, clear_cache


@pytest.fixture
def cached(tmp_path, monkeypatch):
    """Seed a cache that has BOTH a .cols and a .rows fragment, plus signatures."""
    monkeypatch.chdir(tmp_path)

    base = pd.DataFrame({
        "key": [f"k{i}" for i in range(50)],
        "a": np.arange(50, dtype=float),
        "b": ["x"] * 50,
    })
    write_cached_df(pd.DataFrame(columns=base.columns), base.copy(), "sweep", key="key")

    df_new = base.copy()
    df_new.loc[0, "a"] = 999.0  # a modified column -> .cols fragment
    df_new = pd.concat(
        [df_new, pd.DataFrame({"key": ["k50"], "a": [50.0], "b": ["x"]})],
        ignore_index=True,
    )  # a new row -> .rows fragment
    write_cached_df(base.copy(), df_new, "sweep", key="key")

    files = sorted(os.listdir("cache"))
    # Guard the fixture: without fragments these tests would pass vacuously.
    assert any(".cols." in f for f in files), f"no .cols fragment written: {files}"
    assert any(".rows." in f for f in files), f"no .rows fragment written: {files}"
    return tmp_path, files


def test_fragments_are_named_with_a_dot_before_the_extension(cached):
    """Pin the on-disk contract clear_cache has to match."""
    _, files = cached
    assert "sweep.cols.parquet" in files
    assert "sweep.rows.parquet" in files


def test_clear_cache_removes_everything_it_wrote(cached):
    _, _ = cached
    clear_cache("sweep", "df")
    leftover = [f for f in os.listdir("cache") if f.startswith("sweep")]
    assert leftover == [], f"clear_cache left artifacts behind: {leftover}"


def test_clear_cache_removes_the_signature_sidecars(cached):
    _, _ = cached
    clear_cache("sweep", "df")
    remaining = os.listdir("cache")
    assert not [f for f in remaining if f.endswith(".signature.json")]


def test_clear_cache_is_safe_when_nothing_is_there(tmp_path, monkeypatch):
    """Clearing a cache that was never written must not raise."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("cache", exist_ok=True)
    clear_cache("never_written", "df")  # must not raise
    assert os.listdir("cache") == []


def test_clear_cache_leaves_other_entries_alone(cached):
    """Only the named cache is swept."""
    _, _ = cached
    other = pd.DataFrame({"key": ["z1"], "a": [1.0]})
    write_cached_df(pd.DataFrame(columns=other.columns), other, "keepme", key="key")

    clear_cache("sweep", "df")
    remaining = os.listdir("cache")
    assert not [f for f in remaining if f.startswith("sweep")]
    assert [f for f in remaining if f.startswith("keepme")], (
        "clear_cache removed an unrelated cache entry"
    )
