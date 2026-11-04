"""quick_median_chd: the median CHD across clusters, vectorised in pandas (it replaced a polars version,
quick_median_chd_pl). It must give what the old one did: each cluster's COD (calc_cod) of its positive,
present values, then the median of those."""

import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities.stats import calc_cod, quick_median_chd


def reference(df: pd.DataFrame, value: str, cluster: str) -> float:
    """The old algorithm, written out: filter, group (a missing ID is a group too), calc_cod each, median."""
    sub = df[df[value].notna() & df[value].gt(0)]
    if sub.empty:
        return float("nan")
    keys = sub[cluster].astype(object).where(sub[cluster].notna(), "__missing__")
    chds = [calc_cod(sub.loc[keys.eq(k), value].to_numpy()) for k in pd.unique(keys)]
    return float(np.median(chds))


def frame(seed: int, n: int = 2_000, k: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = rng.lognormal(12, 0.6, n)
    v[rng.random(n) < 0.05] = np.nan  # missing values
    v[rng.random(n) < 0.03] = 0.0  # left out (not positive)
    v[rng.random(n) < 0.02] = -5.0  # left out (not positive)
    ids = rng.integers(0, k, n).astype(object)
    ids[rng.random(n) < 0.04] = None  # rows with no cluster: a cluster of their own
    return pd.DataFrame({"value": v, "he_id": ids})


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_old_algorithm(seed):
    df = frame(seed)
    assert quick_median_chd(df, "value", "he_id") == pytest.approx(reference(df, "value", "he_id"), rel=1e-12)


def test_cluster_ids_as_strings_numbers_and_categories():
    df = frame(42)
    want = reference(df, "value", "he_id")
    as_str = df.assign(he_id=df["he_id"].map(lambda x: None if x is None else f"c{x}"))
    as_num = df.assign(he_id=pd.to_numeric(df["he_id"]))
    # A categorical with categories no row has: they must not count as (empty) clusters.
    as_cat = df.assign(he_id=pd.Categorical(df["he_id"], categories=list(range(100))))
    for d in (as_str, as_num, as_cat):
        assert quick_median_chd(d, "value", "he_id") == pytest.approx(want, rel=1e-12)


def test_one_row_clusters_and_known_values():
    df = pd.DataFrame({"v": [100.0, 100.0, 50.0, 150.0, 7.0], "c": ["a", "a", "b", "b", "solo"]})
    # a: COD 0; b: median 100, mean abs dev 50 -> 50; solo: 0. Median of (0, 50, 0) = 0.
    assert quick_median_chd(df, "v", "c") == 0.0
    df2 = pd.DataFrame({"v": [50.0, 150.0, 80.0, 120.0], "c": ["b", "b", "d", "d"]})
    # b: 50; d: median 100, mean abs dev 20 -> 20. Median of (50, 20) = 35.
    assert quick_median_chd(df2, "v", "c") == pytest.approx(35.0)


def test_nothing_left_is_nan():
    df = pd.DataFrame({"v": [np.nan, 0.0, -1.0], "c": ["a", "b", "c"]})
    assert np.isnan(quick_median_chd(df, "v", "c"))
    assert np.isnan(quick_median_chd(df.iloc[0:0], "v", "c"))


def test_reads_only_its_two_columns():
    df = frame(3).assign(geometry=object(), other=pd.Categorical(["x"] * 2_000))
    assert quick_median_chd(df, "value", "he_id") == pytest.approx(reference(df, "value", "he_id"), rel=1e-12)


def test_land_prediction_results_chds():
    """LandPredictionResults' three CHDs (total, land, improvement), straight from the universe: no copy or
    string conversion of the whole table, which existed only for polars. Nothing constructs it yet, so this is
    the only test that reaches that code."""
    from shapely.geometry import Point

    from openavmkit.data import SalesUniversePair
    from openavmkit.modeling import LandPredictionResults

    rng = np.random.default_rng(11)
    n = 600
    land = rng.lognormal(10, 0.5, n)
    impr = rng.lognormal(11, 0.7, n)
    impr[rng.random(n) < 0.05] = 0.0  # vacant: left out of the improvement CHD
    he = rng.integers(0, 25, n)
    land_he = rng.integers(0, 15, n).astype(object)
    land_he[rng.random(n) < 0.05] = None  # missing IDs: a cluster of their own
    universe = pd.DataFrame(
        {
            "key": [f"p{i}" for i in range(n)],
            "pred_land": land,
            "pred_impr": impr,
            "pred_total": land + impr,
            "true_land_value": land * rng.uniform(0.9, 1.1, n),
            # A categorical with unused categories, a string column, an object column and the shapes:
            "he_id": pd.Categorical(he, categories=list(range(40))),
            "land_he_id": land_he,
            "impr_he_id": pd.array([f"i{x}" for x in rng.integers(0, 20, n)], dtype="string"),
            "is_vacant": impr == 0,
            "land_area_sqft": rng.uniform(4_000, 20_000, n),
            "bldg_area_finished_sqft": rng.uniform(800, 3_000, n),
            "geometry": [Point(float(i), 0.0) for i in range(n)],
        }
    )
    sold = universe.sample(150, random_state=1)
    sales = pd.DataFrame(
        {
            "key": sold["key"].to_numpy(),
            "key_sale": [f"s{i}" for i in range(150)],
            "sale_price": (sold["pred_total"] * rng.uniform(0.85, 1.15, 150)).to_numpy(),
            "valid_sale": True,
            "vacant_sale": False,
            "valid_for_land_ratio_study": True,
            "valid_for_ratio_study": True,
        }
    )
    r = LandPredictionResults(
        "pred_land", "pred_impr", "pred_total", "true_land_value", [], SalesUniversePair(sales, universe), 0.0, "sqft"
    )
    assert r.total_chd == pytest.approx(reference(universe, "pred_total", "he_id"), rel=1e-12)
    assert r.land_chd == pytest.approx(reference(universe, "pred_land", "land_he_id"), rel=1e-12)
    assert r.impr_chd == pytest.approx(reference(universe, "pred_impr", "impr_he_id"), rel=1e-12)
