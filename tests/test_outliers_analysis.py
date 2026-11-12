"""outliers: a sale's comps by layered comps, how it's unlike the others, what a model's misses have in common."""

import numpy as np
import pandas as pd

from openavmkit import outliers


def _sales(n=300, seed=3):
    rng = np.random.default_rng(seed)
    nbhd = rng.choice(["A", "B", "C"], n)
    area = rng.uniform(1000, 3000, n)
    per = np.select([nbhd == "A", nbhd == "B"], [150.0, 250.0], 350.0)
    price = area * per * rng.normal(1, 0.05, n)
    vacant = rng.random(n) < 0.1
    pool = rng.random(n) < 0.25
    value = price * np.where(pool, 0.7, 1.0) * rng.normal(1, 0.04, n)
    return pd.DataFrame(
        {
            "key_sale": [f"s{i}" for i in range(n)],
            "neighborhood": nbhd,
            "bldg_area_finished_sqft": np.where(vacant, 0.0, area),
            "land_area_sqft": rng.uniform(5000, 20000, n),
            "vacant_sale": vacant,
            "sale_price_time_adj": price,
            "ratio": value / price,
            "pool": np.where(pool, "yes", "no"),
            "pool_too": np.where(pool, 1, 0).astype(str),
            "noise": rng.normal(0, 1, n),
            "latitude": 38 + rng.random(n) / 10,
            "longitude": -77 + rng.random(n) / 10,
        }
    )


def test_a_sales_comps_leave_it_out_and_share_its_path():
    df = _sales()
    m = outliers.fit_comp_models(
        df, ["neighborhood", "bldg_area_finished_sqft", "land_area_sqft"], "sqft"
    )
    assert m.price is not None and m.impr is not None
    me = df[~df["vacant_sale"]].iloc[0]
    c = outliers.find_comps(m, df, me, min_comps=5)
    assert c is not None and c.basis == "impr"
    assert len(c.keys) >= 5 and me["key_sale"] not in c.keys
    comps = df[df["key_sale"].isin(c.keys)]
    assert not comps["vacant_sale"].any()
    # Every comp meets every condition on the path, as the sale does.
    for cond in c.conditions:
        col = comps[cond["field"]]
        if cond["op"] in ("<=", ">"):
            v = float(cond["value"])
            assert ((col <= v) if cond["op"] == "<=" else (col > v)).all()
            assert (me[cond["field"]] <= v) == (cond["op"] == "<=")
        else:
            assert (
                (col.astype(str) == str(cond["value"])) == (cond["op"] == "=")
            ).all()
    assert 0 < c.depth <= c.full_depth
    # In its neighbourhood, at about its price per area; what they say is that times its size.
    assert (
        c.per_area is not None and c.says == c.per_area * me["bldg_area_finished_sqft"]
    )
    # The trees round-trip.
    again = outliers.find_comps(
        outliers.CompModels.from_dict(m.to_dict()), df, me, min_comps=5
    )
    assert again is not None and again.keys == c.keys


def test_a_vacant_sale_is_compared_on_land():
    df = _sales()
    m = outliers.fit_comp_models(df, ["neighborhood", "land_area_sqft"], "sqft")
    me = df[df["vacant_sale"]].iloc[0]
    c = outliers.find_comps(m, df, me, min_comps=3)
    assert (
        c is not None
        and c.basis == "land"
        and df[df["key_sale"].isin(c.keys)]["vacant_sale"].all()
    )


def test_how_a_sale_is_unlike_the_others():
    df = _sales()
    me = df.iloc[0].copy()
    me["bldg_area_finished_sqft"] = 99_999.0
    rows = outliers.compare_fields(
        me,
        df,
        df["key_sale"].iloc[1:6].tolist(),
        ["noise", "bldg_area_finished_sqft", "pool"],
    )
    assert (
        rows[0]["field"] == "bldg_area_finished_sqft"
        and rows[0]["flag"]
        and rows[0]["pct"] > 95
    )
    pool = next(r for r in rows if r["field"] == "pool")
    assert pool["share"] is not None and not pool["numeric"]


def test_what_the_misses_have_in_common_and_whether_they_cluster():
    df = _sales()
    out = ((df["ratio"] < 0.75) | (df["ratio"] > 1.25)).to_numpy()
    p = outliers.error_patterns(
        df, ["noise", "pool", "pool_too", "neighborhood"], outliers=out
    )
    assert p[0]["field"] in ("pool", "pool_too") and all(
        x["field"] != "noise" for x in p
    )
    top = p[0]
    assert abs(top["off"] - (-0.3)) < 0.05 and abs(top["rest_off"]) < 0.05
    second = next(x for x in p if x["field"] in ("pool", "pool_too") and x is not top)
    assert second["goes_with"] == top["field"]
    assert outliers.error_clustering(df)[0] is False
    # Errors that follow the map: clustered.
    df2 = df.copy()
    df2["ratio"] = np.exp((df2["latitude"] - 38.05) * 10)
    clustered, moran = outliers.error_clustering(df2)
    assert clustered and moran > 0.5


def test_a_comp_path_merged_per_field():
    path = [
        {"field": "nbhd", "op": "!=", "value": "A"},
        {"field": "age", "op": ">", "value": 36.0},
        {"field": "nbhd", "op": "!=", "value": "B"},
        {"field": "age", "op": ">", "value": 40.0},
        {"field": "age", "op": "<=", "value": 80.0},
        {"field": "type", "op": "!=", "value": "X"},
        {"field": "type", "op": "=", "value": "Y"},
    ]
    assert outliers.summarize_conditions(path) == [
        {"field": "nbhd", "at_most": None, "over": None, "is": None, "is_not": ["A", "B"]},
        {"field": "age", "at_most": 80.0, "over": 40.0, "is": None, "is_not": []},
        {"field": "type", "at_most": None, "over": None, "is": "Y", "is_not": []},
    ]
