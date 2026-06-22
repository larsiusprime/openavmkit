import os

from openavmkit.utilities.cache import clear_cache


def test_clear_cache_removes_cols_and_rows(tmp_path, monkeypatch):
    """clear_cache must remove the .cols / .rows fragments and their signatures.

    Regression test: the path templates for the .cols/.rows data fragments were
    malformed (missing dot in the existence check, double dot in the removal),
    so clear_cache silently left the large .cols.parquet column-diff caches on
    disk. Recreate every artifact a "df" cache entry produces and assert a clean
    sweep.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("cache", exist_ok=True)

    artifacts = [
        "foo.parquet",
        "foo.cols.parquet",
        "foo.rows.parquet",
        "foo.signature.json",
        "foo.cols.signature.json",
        "foo.rows.signature.json",
    ]
    for name in artifacts:
        open(os.path.join("cache", name), "w").close()

    clear_cache("foo", "df")

    for name in artifacts:
        assert not os.path.exists(os.path.join("cache", name)), f"{name} was not removed"
