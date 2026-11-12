"""Explaining outliers: the sales a model valued furthest from their price, and why.

The analysis behind ``pipeline.identify_outliers``, as functions that return data rather than print it, so a notebook
or an app can show it:

- :func:`fit_comp_models` and :func:`find_comps`: a sale's comparable sales, by layered comps. A tree of sales split on
  the variables (price, and price per building or land area), the sale traced down it to the smallest group of sales
  like it. Each comp tree is fitted once per model group, on every sale; a sale's comps leave the sale itself out, and
  the trace stops where at least ``min_comps`` other sales are left.
- :func:`compare_fields`: how a sale is unlike the others, field by field: against its comps, and where it sits among
  the group's sales.
- :func:`error_patterns`: what a model's misses have in common: for each field, the share of the model's error that
  lines up with it, kept only when it beats chance (shuffled errors).
- :func:`error_clustering`: whether the errors are more alike between neighbours than chance (Moran's I).

The error of a sale is ``log(value / price)``: 0 when valued at its price, below 0 when valued low.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CompModels",
    "Comps",
    "fit_comp_models",
    "find_comps",
    "summarize_conditions",
    "compare_fields",
    "error_patterns",
    "error_clustering",
]

MIN_FIT = 5  # sales a comp tree needs to be fitted


def _as_object(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as plain objects (as layered comps reads them)."""
    X = X.copy()
    for c in X.select_dtypes(["category"]).columns:
        X[c] = X[c].astype("object")
    return X


# ---------- Comparable sales, by layered comps ----------


@dataclass
class CompModels:
    """A model group's comp trees: on price, on price per building area (improved sales) and on price per land area
    (vacant sales); any can be None where there are too few sales to fit it."""

    features: list[str]
    unit: str
    price_field: str
    price: Any = None
    impr: Any = None
    land: Any = None

    def to_dict(self) -> dict:
        import layeredcompmodel

        return {
            "features": self.features,
            "unit": self.unit,
            "price_field": self.price_field,
            "lib_version": getattr(layeredcompmodel, "__version__", None),
            **{
                k: (_tree_state(m) if m is not None else None)
                for k, m in self._models().items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompModels:
        from layeredcompmodel import LayeredCompModel

        def load(x):
            return LayeredCompModel.from_dict(x) if x else None

        return cls(
            features=list(d["features"]),
            unit=d["unit"],
            price_field=d["price_field"],
            price=load(d.get("price")),
            impr=load(d.get("impr")),
            land=load(d.get("land")),
        )

    def _models(self) -> dict:
        return {"price": self.price, "impr": self.impr, "land": self.land}


def _tree_state(t) -> dict:
    """A fitted comp tree with what predicting needs (as LayeredCompBaggingModel.to_dict saves each of its trees)."""
    return {
        "weight_falloff": float(t.weight_falloff),
        "split_metric": t._split_metric_name,
        "columns": list(t.columns_),
        "n_features_in": int(t.n_features_in_),
        "tree": t.to_dict(),
    }


def _sizes(unit: str) -> tuple[str, str]:
    return f"bldg_area_finished_{unit}", f"land_area_{unit}"


def _per_area(df: pd.DataFrame, price_field: str, size: str) -> pd.Series:
    price = pd.to_numeric(df[price_field], errors="coerce").astype(float)
    s = (
        pd.to_numeric(df[size], errors="coerce").astype(float)
        if size in df
        else pd.Series(np.nan, index=df.index)
    )
    return price / s.where(s > 0)


def fit_comp_models(
    df_sales: pd.DataFrame,
    features: list[str],
    unit: str,
    price_field: str = "sale_price_time_adj",
) -> CompModels:
    """Fit a model group's comp trees on its sales (each with ``vacant_sale``, the price and the features): price on
    every sale, price per finished area on the improved ones, price per land area on the vacant ones."""
    from layeredcompmodel import LayeredCompModel

    feats = [c for c in features if c in df_sales.columns]
    out = CompModels(features=feats, unit=unit, price_field=price_field)
    if not feats or price_field not in df_sales:
        return out
    df = df_sales[pd.to_numeric(df_sales[price_field], errors="coerce").fillna(0) > 0]
    if len(df) < MIN_FIT:
        return out
    impr_size, land_size = _sizes(unit)
    vacant = (
        df["vacant_sale"].astype("boolean").fillna(False)
        if "vacant_sale" in df
        else pd.Series(False, df.index)
    )

    def fit(rows: pd.DataFrame, y: pd.Series):
        ok = y.notna() & np.isfinite(y) & (y > 0)
        if ok.sum() < MIN_FIT:
            return None
        return LayeredCompModel(split_metric="mae", n_jobs=1).fit(
            _as_object(rows.loc[ok, feats]), y[ok]
        )

    out.price = fit(df, pd.to_numeric(df[price_field], errors="coerce").astype(float))
    improved = df[~vacant]
    out.impr = (
        fit(improved, _per_area(improved, price_field, impr_size))
        if len(improved)
        else None
    )
    vacants = df[vacant]
    out.land = (
        fit(vacants, _per_area(vacants, price_field, land_size))
        if len(vacants)
        else None
    )
    return out


@dataclass
class Comps:
    """A sale's comparable sales: the sales of the deepest part of its comp tree with at least ``min_comps`` others."""

    basis: str  # "impr": price per building area; "land": price per land area; "price": price
    keys: list[str]  # the comps' key_sale (the sale itself left out)
    conditions: list[dict] = field(
        default_factory=list
    )  # [{field, op, value}]: what makes them alike, in order
    depth: int = 0  # the conditions kept
    full_depth: int = 0  # how deep the sale's own path goes
    per_area: float | None = (
        None  # the comps' (Wilson) mean price per area (basis impr/land)
    )
    says: float | None = (
        None  # what the comps say the sale is worth: per area times its size, or their mean price
    )


def _wilson(y: np.ndarray) -> float | None:
    from layeredcompmodel.model import calculate_wilson_mean

    y = y[np.isfinite(y)]
    return float(calculate_wilson_mean(y)) if len(y) else None


def _branch(node, value) -> Any:
    """The child of a split node the value goes to (None: it can't be told: a blank number)."""
    if node.is_numeric:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(v):
            return None
        want = "<=" if float(v) <= float(node.filter_val) else ">"
    else:
        v = (
            "NaN"
            if value is None or (isinstance(value, float) and math.isnan(value))
            else value
        )
        want = "=" if v == node.filter_val or str(v) == str(node.filter_val) else "!="
    return next((c for c in node.children if c.variant == want), None)


def _keep(X: pd.DataFrame, node, child) -> np.ndarray:
    """Which rows follow a split node to the given child."""
    col = X[node.filter_col]
    if node.is_numeric:
        x = pd.to_numeric(col, errors="coerce").astype(float)
        hit = (
            x <= float(node.filter_val)
            if child.variant == "<="
            else x > float(node.filter_val)
        )
        return (x.notna() & hit).to_numpy()
    c = col.astype(object).where(col.notna(), "NaN")
    same = (c == node.filter_val) | (c.astype(str) == str(node.filter_val))
    return (same if child.variant == "=" else ~same).to_numpy()


def find_comps(
    models: CompModels,
    df_sales: pd.DataFrame,
    subject: pd.Series,
    min_comps: int = 5,
) -> Comps | None:
    """A sale's comps: traced down its comp tree (per building area if improved, per land area if vacant, else on
    price), the sale itself left out, to the deepest node with at least ``min_comps`` other sales of ``df_sales`` (the
    group's, as fitted on). None when there's no tree for it."""
    impr_size, land_size = _sizes(models.unit)
    vacant = bool(
        pd.Series([subject.get("vacant_sale")]).astype("boolean").fillna(False).iloc[0]
    )
    basis, model, size = (
        ("land", models.land, land_size) if vacant else ("impr", models.impr, impr_size)
    )
    if (
        model is None
        or not (
            pd.to_numeric(pd.Series([subject.get(size)]), errors="coerce").fillna(0) > 0
        ).iloc[0]
    ):
        basis, model, size = "price", models.price, None
    if model is None:
        return None
    X = _as_object(df_sales[models.features])
    others = (df_sales["key_sale"].astype(str) != str(subject["key_sale"])).to_numpy()
    if basis != "price":
        v = (
            df_sales["vacant_sale"].astype("boolean").fillna(False).to_numpy()
            if "vacant_sale" in df_sales
            else None
        )
        if v is not None:
            others &= v if vacant else ~v
    y = (
        _per_area(df_sales, models.price_field, size).to_numpy()
        if size
        else pd.to_numeric(df_sales[models.price_field], errors="coerce")
        .astype(float)
        .to_numpy()
    )
    others &= np.isfinite(y) & (y > 0)
    row = _as_object(subject.to_frame().T.reindex(columns=models.features)).iloc[0]

    mask = others.copy()
    best_mask, best_depth, conditions, kept = mask.copy(), 0, [], []
    node, depth = model.tree_, 0
    while node is not None and node.filter_col is not None and node.children:
        child = _branch(node, row.get(node.filter_col))
        if child is None:
            break
        mask = mask & _keep(X, node, child)
        depth += 1
        conditions.append(
            {"field": node.filter_col, "op": child.variant, "value": node.filter_val}
        )
        if mask.sum() >= min_comps:
            best_mask, best_depth, kept = mask.copy(), depth, list(conditions)
        node = child
    keys = df_sales["key_sale"].astype(str).to_numpy()[best_mask].tolist()
    mean = _wilson(y[best_mask])
    if size:
        my = pd.to_numeric(pd.Series([subject.get(size)]), errors="coerce").iloc[0]
        says = mean * float(my) if mean is not None and pd.notna(my) else None
        per_area = mean
    else:
        says, per_area = mean, None
    return Comps(
        basis=basis,
        keys=keys,
        conditions=kept,
        depth=best_depth,
        full_depth=depth,
        per_area=per_area,
        says=says,
    )


def summarize_conditions(conditions: list[dict]) -> list[dict]:
    """A comp path's conditions merged per field, in the order fields first come up: the tightest bounds of a number
    (``at_most``, ``over``), a category's value (``is``) or the values it isn't (``is_not``). What the comps share, in
    as few words as the path allows."""
    out: dict[str, dict] = {}
    for c in conditions:
        d = out.setdefault(c["field"], {"field": c["field"], "at_most": None, "over": None, "is": None, "is_not": []})
        v = c["value"]
        if c["op"] == "<=":
            d["at_most"] = float(v) if d["at_most"] is None else min(d["at_most"], float(v))
        elif c["op"] == ">":
            d["over"] = float(v) if d["over"] is None else max(d["over"], float(v))
        elif c["op"] == "=":
            d["is"], d["is_not"] = v, []
        elif d["is"] is None and v not in d["is_not"]:
            d["is_not"].append(v)
    return list(out.values())


# ---------- How a sale is unlike the others ----------


def compare_fields(
    subject: pd.Series,
    df_group: pd.DataFrame,
    comp_keys: list[str],
    fields: list[str],
    numeric: set[str] | None = None,
) -> list[dict]:
    """Each field: the sale's value, its comps' (median of a number, most common category), where it sits among the
    group's sales (a number: its percentile; a category: the share with its value), and whether it stands out: among
    the group's extremes (under the 5th or over the 95th percentile, a category under 5%), or unlike its comps (more
    than the group's middle half apart; a category its comps mostly don't share). Standing out first."""
    numeric = (
        numeric
        if numeric is not None
        else {f for f in fields if pd.api.types.is_numeric_dtype(df_group[f])}
    )
    comps = df_group[df_group["key_sale"].astype(str).isin([str(k) for k in comp_keys])]
    out = []
    for f in fields:
        if f not in df_group:
            continue
        col, x = df_group[f], subject.get(f)
        if f in numeric:
            vals = pd.to_numeric(col, errors="coerce").astype(float).dropna()
            cv = pd.to_numeric(comps[f], errors="coerce").astype(float).dropna()
            xn = pd.to_numeric(pd.Series([x]), errors="coerce").astype(float).iloc[0]
            cm = float(cv.median()) if len(cv) else None
            pct = None
            flag = False
            if pd.notna(xn) and len(vals):
                pct = float(((vals < xn).mean() + 0.5 * (vals == xn).mean()) * 100)
                iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
                flag = (
                    pct < 5
                    or pct > 95
                    or (cm is not None and iqr > 0 and abs(xn - cm) > iqr)
                )
            elif pd.isna(xn):
                flag = bool(len(vals)) and float(col.isna().mean()) < 0.2
            out.append(
                {
                    "field": f,
                    "numeric": True,
                    "value": None if pd.isna(xn) else float(xn),
                    "comps": cm,
                    "pct": pct,
                    "share": None,
                    "flag": bool(flag),
                }
            )
        else:
            vals = col.dropna().astype(str)
            xs = (
                None
                if x is None or (isinstance(x, float) and math.isnan(x))
                else str(x)
            )
            cs = comps[f].dropna().astype(str)
            mode = str(cs.mode().iloc[0]) if len(cs) else None
            agree = float((cs == mode).mean()) if mode is not None else 0.0
            share = float((vals == xs).mean()) if xs is not None and len(vals) else None
            flag = (share is not None and share < 0.05) or (
                mode is not None and xs is not None and xs != mode and agree >= 0.6
            )
            out.append(
                {
                    "field": f,
                    "numeric": False,
                    "value": xs,
                    "comps": mode,
                    "pct": None,
                    "share": share,
                    "flag": flag,
                }
            )
    out.sort(key=lambda r: not r["flag"])
    return out


# ---------- What the misses have in common ----------


def _buckets(
    col: pd.Series, numeric: bool, min_bucket: int
) -> tuple[np.ndarray, list[Any]]:
    """Each sale's bucket (-1: none read), and each bucket: ("under", a), ("between", a, b), ("over", a), ("is", v),
    ("blank",). A number in fifths (one with 6 values or fewer as categories); a category by value, `min_bucket` sales
    at least."""
    codes = np.full(len(col), -1, dtype=int)
    labels: list[Any] = []
    blank = col.isna().to_numpy()
    x = pd.to_numeric(col, errors="coerce").astype(float) if numeric else None
    if x is not None and x.nunique() > 6:
        xs = x.to_numpy()
        edges = sorted(set(np.nanquantile(xs, [0.2, 0.4, 0.6, 0.8]).tolist()))
        pos = np.searchsorted(edges, xs, side="right")
        for i in range(len(edges) + 1):
            if i == 0:
                labels.append(("under", edges[0]))
            elif i == len(edges):
                labels.append(("over", edges[-1]))
            else:
                labels.append(("between", edges[i - 1], edges[i]))
        codes = np.where(blank, -1, pos)
    else:
        s = (x if x is not None else col.astype(object)).where(~col.isna())
        counts = s.value_counts()
        for v in counts.index:
            if counts[v] >= min_bucket:
                codes[(s == v).to_numpy()] = len(labels)
                labels.append(("is", v.item() if hasattr(v, "item") else v))
    if blank.sum() >= min_bucket:
        codes[blank] = len(labels)
        labels.append(("blank",))
    return codes, labels


def _eta(
    codes: np.ndarray, n_b: int, r: np.ndarray, perms: np.ndarray, min_bucket: int
):
    """The share of the error's spread between the buckets (a one-way ANOVA's R²), for r and for each shuffle."""
    ok = codes >= 0
    c = codes[ok]
    if n_b < 2 or len(c) < 2 * min_bucket:
        return None, None
    counts = np.bincount(c, minlength=n_b).astype(float)
    use = counts > 0

    def one(vals):
        sums = np.stack([np.bincount(c, weights=row, minlength=n_b) for row in vals])
        mean = vals.mean(axis=1, keepdims=True)
        between = ((sums[:, use] / counts[use] - mean) ** 2 * counts[use]).sum(axis=1)
        total = ((vals - mean) ** 2).sum(axis=1)
        return between / np.where(total > 0, total, 1)

    return float(one(r[ok][None, :])[0]), one(perms[:, ok])


def error_patterns(
    df_group: pd.DataFrame,
    fields: list[str],
    ratio_field: str = "ratio",
    numeric: set[str] | None = None,
    outliers: np.ndarray | None = None,
    min_bucket: int = 10,
    shuffles: int = 300,
    min_share: float = 0.02,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict]:
    """What a model's misses have in common. Over every sale of the group (not only the outliers: chosen for being
    wrong, they'd mislead), each field's buckets and the share of the error's spread between them (0 when the model's
    mistakes are random with respect to the field). Kept when it's ``min_share`` or more and beats ``shuffles``
    shuffles of the errors (fewer than ``alpha`` as high); ranked. Each with the bucket that pulls hardest, its sales,
    the outliers among them (``outliers``: a mask), the model's average error on them and on the rest (-0.18: valued
    18% low), and ``goes_with``: a field ranked above whose bucket is mostly the same sales (Jaccard 0.5 or more)."""
    g = df_group.reset_index(drop=True)
    r = np.log(pd.to_numeric(g[ratio_field], errors="coerce").astype(float).to_numpy())
    ok_r = np.isfinite(r)
    g, r = g[ok_r].reset_index(drop=True), r[ok_r]
    out_mask = (
        np.zeros(len(g), dtype=bool)
        if outliers is None
        else np.asarray(outliers, dtype=bool)[ok_r]
    )
    if len(g) < 2 * min_bucket:
        return []
    numeric = (
        numeric
        if numeric is not None
        else {f for f in fields if f in g and pd.api.types.is_numeric_dtype(g[f])}
    )
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(r) for _ in range(shuffles)])
    found: list[tuple[dict, set[int]]] = []
    for f in fields:
        if f not in g:
            continue
        codes, labels = _buckets(g[f], f in numeric, min_bucket)
        eta, shuffled = _eta(codes, len(labels), r, perms, min_bucket)
        if eta is None or shuffled is None:
            continue
        chance = float((shuffled >= eta).mean())
        if chance >= alpha or eta < min_share:
            continue
        mean = r[codes >= 0].mean()
        best, pull = -1, -1.0
        for b in range(len(labels)):
            m = codes == b
            if (
                m.sum() >= min_bucket
                and (p := m.sum() * (r[m].mean() - mean) ** 2) > pull
            ):
                best, pull = b, p
        if best < 0:
            continue
        m = codes == best
        found.append(
            (
                {
                    "field": f,
                    "bucket": labels[best],
                    "n": int(m.sum()),
                    "outliers": int((m & out_mask).sum()),
                    "off": float(math.expm1(r[m].mean())),
                    "rest_off": float(math.expm1(r[~m].mean())) if (~m).any() else 0.0,
                    "share": round(eta, 4),
                    "chance": round(chance, 4),
                    "goes_with": None,
                },
                set(np.flatnonzero(m).tolist()),
            )
        )
    found.sort(key=lambda x: -x[0]["share"])
    for i, (p, members) in enumerate(found):
        for q, theirs in found[:i]:
            if len(members & theirs) / max(len(members | theirs), 1) >= 0.5:
                p["goes_with"] = q["field"]
                break
    return [p for p, _ in found]


def error_clustering(
    df_group: pd.DataFrame,
    ratio_field: str = "ratio",
    k: int = 8,
    shuffles: int = 199,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[bool | None, float | None]:
    """Whether the errors are more alike between neighbours than chance: Moran's I over each sale's ``k`` nearest
    (latitude, longitude), and whether it beats ``shuffles`` shuffles. (None, None) without places for 30 sales."""
    if "latitude" not in df_group or "longitude" not in df_group:
        return None, None
    lat = pd.to_numeric(df_group["latitude"], errors="coerce").astype(float).to_numpy()
    lon = pd.to_numeric(df_group["longitude"], errors="coerce").astype(float).to_numpy()
    r = np.log(
        pd.to_numeric(df_group[ratio_field], errors="coerce").astype(float).to_numpy()
    )
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(r)
    if ok.sum() < 30:
        return None, None
    from sklearn.neighbors import NearestNeighbors

    xy = np.column_stack([lat[ok], lon[ok] * np.cos(np.radians(lat[ok].mean()))])
    kk = min(k, len(xy) - 1)
    nb = (
        NearestNeighbors(n_neighbors=kk + 1)
        .fit(xy)
        .kneighbors(xy, return_distance=False)[:, 1:]
    )
    z = r[ok] - r[ok].mean()
    denom = float((z**2).sum()) or 1.0

    def moran(zz: np.ndarray) -> float:
        return float((zz * zz[nb].mean(axis=1)).sum() / denom)

    rng = np.random.default_rng(seed)
    i = moran(z)
    beats = float(np.mean([moran(rng.permutation(z)) >= i for _ in range(shuffles)]))
    return bool(beats < alpha), round(i, 4)
