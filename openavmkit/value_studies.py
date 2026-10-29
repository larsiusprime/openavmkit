"""Assessment-quality studies on any value field, written to disk.

The ``assessment_quality`` notebook runs ratio, vertical-equity and
horizontal-equity studies on a value field (typically the assessor's current
market value) one model group at a time and displays the results. This module
runs the same studies for every model group plus the whole jurisdiction, for one
or more value fields, and writes the results as JSON so that reports and other
front ends can use them without re-running anything.

No modeling is required: the studies only need sales, model groups and the value
field(s). Horizontal equity additionally needs the equity clusters marked in the
clean stage (``he_id``); without them it is skipped.

Sales are scored on raw ``sale_price`` within the ratio-study window
(``analysis.ratio_study.look_back_years`` before the valuation date), matching
the formal ratio study.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

from openavmkit.data import SalesUniversePair, get_hydrated_sales_from_sup
from openavmkit.horizontal_equity_study import HorizontalEquityStudy
from openavmkit.ratio_study import RatioStudyBootstrapped
from openavmkit.utilities.settings import get_look_back_dates, get_model_group_ids
from openavmkit.utilities.stats import ConfidenceStat, calc_median_confidence_interval
from openavmkit.vertical_equity_study import VerticalEquityStudy, get_vertical_equity_scores

OVERALL = "__overall__"

# The IAAO's targets (Standard on Ratio Studies, 2013; VEI from the May 2026 exposure draft): the level and price
# bias ranges, the most a VEI may stray, the fewest sales judged, and the COD ceiling for each class of property.
IAAO_STANDARDS = {
    "level": [0.90, 1.10],
    "prd": [0.98, 1.03],
    "prb": [-0.05, 0.05],
    "vei": 10.0,
    "min_sales": 5,
    "cod": {"residential": 15.0, "income": 20.0, "vacant": 25.0},
}


def run_value_field_studies(
    sup: SalesUniversePair,
    settings: dict,
    fields: list[str],
    *,
    location: Optional[str] = None,
    start_date=None,
    end_date=None,
    confidence_interval: float = 0.90,
    max_trim: float = 0.05,
    outpath: Optional[str] = "out/studies",
    sale_field: str = "sale_price",
    trim_outliers: bool = False,
    iqr_factor: float = 1.5,
    median_ci: str = "bootstrap",
    standards: Optional[dict] = None,
    classes: Optional[dict] = None,
    region: Optional[str] = None,
) -> dict:
    """Run ratio and equity studies on value fields for every model group.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe, after the clean stage (or at least after model groups
        are tagged).
    settings : dict
        Settings dictionary.
    fields : list[str]
        Universe value fields to study, e.g. ``["assr_market_value"]``.
    location : str, optional
        Location field for the grouped vertical-equity quantiles. Defaults to the
        first entry of ``field_classification.important.locations`` present in the data.
    start_date, end_date : date-like, optional
        Sales window. Defaults to the ratio-study look-back window ending at the
        valuation date.
    confidence_interval : float, optional
        Confidence level for bootstrap intervals. Defaults to 0.90.
    max_trim : float, optional
        Maximum share of ratios trimmed for the trimmed statistics. Defaults to 0.05.
    outpath : str or None, optional
        Folder to write ``<field>.json`` into (and, with ``trim_outliers``, ``<field>.sales.parquet``: each sale
        studied, its ratio and whether it was trimmed). ``None`` writes nothing.
    sale_field : str, optional
        The price the values are compared with: ``sale_price`` (as sold, the formal ratio study) or e.g.
        ``sale_price_time_adj``.
    trim_outliers : bool, optional
        Leave ratios outside ``iqr_factor`` × the interquartile range out of every statistic, per model group (at
        most 10% of a group's sales, the furthest out first; none in a group of fewer than 10), and list them under
        ``trimmed``; the overall study leaves out the groups'.
        Default False: the untrimmed statistics, with the trimmed ones beside them (``*_trim``).
    iqr_factor : float, optional
        1.5 (outliers, the IAAO's usual) or 3.0 (extreme outliers only).
    median_ci : str, optional
        ``"bootstrap"`` (default) or ``"iaao"``: the order-statistic interval of the Standard's Appendix D.2, as
        the VEI uses.
    standards : dict, optional
        Targets to judge each study against (``IAAO_STANDARDS``'s keys); adds ``verdict``.
    classes : dict, optional
        Model group -> class (``residential``, ``income``, ``vacant``, or None: not judged), for the COD target.
    region : str, optional
        A location field (a neighborhood, say): adds ``regions``, the median ratio by the price level of the region
        each sale is in (openratiostudy.com's "neighborhood price level" statistics; not an IAAO test). It tells
        real vertical inequity (low-priced areas over-assessed) from invalid sales at low prices in dear areas.

    Returns
    -------
    dict
        ``{field: {"window": ..., "groups": {model_group: {...}}, "overall": {...}}}``.
    """
    if start_date is None or end_date is None:
        start_date, end_date = get_look_back_dates(settings)
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)

    df_sales = get_hydrated_sales_from_sup(sup)
    if sale_field != "sale_price":
        df_sales = df_sales.drop(columns=["sale_price"], errors="ignore").rename(columns={sale_field: "sale_price"})
    sale_dates = pd.to_datetime(df_sales["sale_date"], errors="coerce")
    df_sales = df_sales[sale_dates.ge(start_date) & sale_dates.le(end_date)]
    valid_field = "valid_for_ratio_study" if "valid_for_ratio_study" in df_sales else "valid_sale"
    if valid_field in df_sales:
        df_sales = df_sales[df_sales[valid_field].eq(True)]
    df_univ = sup.universe

    if location is None:
        locations = settings.get("field_classification", {}).get("important", {}).get("locations", [])
        location = next((loc for loc in locations if loc in df_sales), None)

    model_groups = [mg for mg in get_model_group_ids(settings, df_univ) if mg in set(df_univ["model_group"])]

    results = {}
    for field in fields:
        if field not in df_univ:
            raise ValueError(f"Value field '{field}' not found in the universe.")
        groups = {}
        opts = dict(confidence_interval=confidence_interval, max_trim=max_trim, median_ci=median_ci, region=region)
        usable = _usable(df_sales, field)
        trimmed = pd.Series(False, index=usable.index)
        if trim_outliers:
            for mg in model_groups:
                mine = usable["model_group"].eq(mg)
                trimmed[mine] = _trimmed(usable[mine], field, iqr_factor)
        for mg in model_groups:
            mine = usable["model_group"].eq(mg)
            groups[mg] = _study_one(
                usable[mine & ~trimmed], df_univ[df_univ["model_group"].eq(mg)], field, location, **opts
            )
            if trim_outliers:
                groups[mg]["trimmed"] = _sale_rows(usable[mine & trimmed], field)
        overall = _study_one(usable[~trimmed], df_univ, field, location, **opts)
        if trim_outliers:
            overall["count_trimmed"] = int(trimmed.sum())
        if standards is not None:
            for mg, g in groups.items():
                g["verdict"] = _verdict(g, standards, (classes or {}).get(mg))
        results[field] = {
            "field": field,
            "window": {"start": start_date.date().isoformat(), "end": end_date.date().isoformat()},
            "sale_field": sale_field,
            "confidence_interval": confidence_interval,
            "median_ci": median_ci,
            "trim_outliers": trim_outliers,
            "location": location,
            "groups": groups,
            "overall": overall,
        }
        if outpath is not None:
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, f"{field}.json"), "w") as f:
                json.dump(results[field], f, indent=2, default=_json_default)
            if trim_outliers:
                table = _sale_table(usable, field, region)
                table["trimmed"] = trimmed.to_numpy()
                table.to_parquet(os.path.join(outpath, f"{field}.sales.parquet"), index=False)
    return results


def _usable(df_sales: pd.DataFrame, field: str) -> pd.DataFrame:
    values = pd.to_numeric(df_sales[field], errors="coerce")
    return df_sales[values.notna() & np.isfinite(values) & df_sales["sale_price"].gt(0)]


def _trimmed(df: pd.DataFrame, field: str, iqr_factor: float, max_share: float = 0.10) -> np.ndarray:
    """Which sales' ratios are outside the IQR fences: at most ``max_share`` of the group, the furthest out first
    (a group of fewer than 10 sales has none trimmed)."""
    ratios = (df[field] / df["sale_price"]).to_numpy(dtype=float)
    out = np.zeros(len(ratios), dtype=bool)
    cap = int(max_share * len(ratios))
    if cap == 0:
        return out
    q1, q3 = np.quantile(ratios, [0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    beyond = np.where(ratios < low, (low - ratios), np.where(ratios > high, (ratios - high), 0.0)) / (iqr or 1.0)
    candidates = np.flatnonzero(beyond > 0)
    out[candidates[np.argsort(-beyond[candidates], kind="stable")][:cap]] = True
    return out


def _sale_table(df: pd.DataFrame, field: str, region: Optional[str] = None) -> pd.DataFrame:
    cols = [c for c in ("key_sale", "key", "model_group", "sale_date", "latitude", "longitude") if c in df]
    if region and region in df and region not in cols:
        cols.append(region)
    out = df[cols].copy()
    out["sale_price"] = df["sale_price"].astype(float)
    out["value"] = df[field].astype(float)
    out["ratio"] = out["value"] / out["sale_price"]
    return out.reset_index(drop=True)


def _sale_rows(df: pd.DataFrame, field: str) -> list[dict]:
    t = _sale_table(df, field)
    if "sale_date" in t:
        t["sale_date"] = pd.to_datetime(t["sale_date"], errors="coerce").dt.date.astype(str)
    return json.loads(t.sort_values("ratio").to_json(orient="records"))


def _verdict(g: dict, standards: dict, cls: Optional[str]) -> dict:
    """Each statistic against its target: True met, False missed, None not judged (too few sales, no class, or no
    statistic). The VEI is missed only when it's significant (its groups' intervals apart, and over the limit)."""
    n = g["ratio"]["count"]
    cod_max = (standards.get("cod") or {}).get(cls) if cls else None
    judged = cls is not None and n >= standards.get("min_sales", 5)

    def within(value, bounds):
        if not judged or value is None or bounds is None:
            return None
        return bool(bounds[0] <= value <= bounds[1])

    v = g.get("vertical") or {}
    vei = v.get("vei_verdict")
    return {
        "judged": judged,
        "class": cls,
        "level": within(g["ratio"]["median_ratio"]["value"], standards.get("level")),
        "cod": None if not judged or cod_max is None or g["ratio"]["cod"]["value"] is None
        else bool(g["ratio"]["cod"]["value"] <= cod_max),
        "cod_target": cod_max,
        "prd": within((v.get("prd") or {}).get("value"), standards.get("prd")),
        "prb": within((v.get("prb") or {}).get("value"), standards.get("prb")),
        "vei": None if not judged or vei in (None, "insufficient") else vei in ("compliant", "inconclusive"),
    }


def _group_by_sort_key(key: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Rows in ``n_groups`` near-equal groups by ascending ``key`` (leftovers to the first groups), a group's end
    moved past ties so equal keys stay together (openratiostudy.com's grouping)."""
    order = np.argsort(key, kind="stable")
    n = len(order)
    base, extra = divmod(n, n_groups)
    groups, start = [], 0
    for g in range(n_groups):
        end = start + base + (1 if g < extra else 0)
        while 0 < end < n and key[order[end - 1]] == key[order[end]]:
            end += 1
        if start < end:
            groups.append(order[start:end])
        start = end
        if start >= n:
            break
    return groups


def _region_stats(df: pd.DataFrame, field: str, region: str, limit: float = 10.0) -> dict:
    """The median ratio by the price level of each sale's region: the region's median market value proxy (half the
    price, half the value at the median ratio), regions grouped by level as the VEI groups sales."""
    region_text = df[region].astype("string").str.strip()
    has = region_text.notna() & region_text.ne("")
    d = df[has]
    out = {"field": region, "regions": int(region_text[has].nunique()), "sales": int(len(d)), "strata": []}
    out |= {"vei": None, "vei_significance": None, "vei_verdict": "insufficient"}
    if len(d) < 20 or out["regions"] < 2:
        return out
    ratios = (d[field] / d["sale_price"]).to_numpy(dtype=float)
    median = float(np.median((df[field] / df["sale_price"]).to_numpy(dtype=float)))
    proxy = 0.5 * d["sale_price"].to_numpy(dtype=float) + 0.5 * d[field].to_numpy(dtype=float) / median
    level = pd.Series(proxy).groupby(region_text[has].to_numpy()).median()
    key = region_text[has].map(level).to_numpy(dtype=float)
    n_groups = 2 if len(d) <= 50 else 4 if len(d) <= 500 else 10
    strata = []
    for rows in _group_by_sort_key(key, n_groups):
        r = ratios[rows]
        low, high = calc_median_confidence_interval(r, 0.90) if len(r) >= 2 else (np.nan, np.nan)
        strata.append(
            {
                "n": int(len(r)),
                "regions": int(pd.Series(region_text[has].to_numpy()[rows]).nunique()),
                "level": _num(np.median(key[rows])),
                "median": _num(np.median(r)),
                "low": _num(low),
                "high": _num(high),
            }
        )
    out["strata"] = strata
    if len(strata) >= 2:
        first, last = strata[0], strata[-1]
        vei = 100 * (last["median"] - first["median"]) / median
        sig = None
        if last["low"] is not None and first["high"] is not None:
            sig = 100 * (last["low"] - first["high"]) / median
        out |= {"vei": _num(vei), "vei_significance": _num(sig), "vei_verdict": _vei_verdict(vei, sig, limit)}
    return out


def _vei_verdict(vei: Optional[float], significance: Optional[float], limit: float = 10.0) -> str:
    """The VEI's reading: compliant within the limit; beyond it, regressive or progressive only when significant
    (as openratiostudy reads it), else inconclusive."""
    if vei is None:
        return "insufficient"
    if abs(vei) <= limit:
        return "compliant"
    if significance is not None and significance > limit:
        return "progressive" if vei > 0 else "regressive"
    return "inconclusive"


def _study_one(
    usable, df_univ, field, location, confidence_interval, max_trim, median_ci="bootstrap", region=None
) -> dict:
    out = {"count_sales": int(len(usable)), "count_parcels": int(len(df_univ))}
    if region is not None and region in usable:
        out["regions"] = _region_stats(usable, field, region)

    rs = RatioStudyBootstrapped(
        usable[field].to_numpy(), usable["sale_price"].to_numpy(), max_trim, confidence_interval
    )
    out["ratio"] = {
        "count": int(rs.count),
        "count_trim": int(rs.count_trim),
        "median_ratio": _median(rs.median_ratio, usable, field, confidence_interval, median_ci),
        "cod": _stat(rs.cod),
        "prd": _stat(rs.prd),
        "median_ratio_trim": _stat(rs.median_ratio_trim),
        "cod_trim": _stat(rs.cod_trim),
        "prd_trim": _stat(rs.prd_trim),
    }

    out["vertical"] = None
    if len(usable) >= 3 and usable["sale_price"].nunique() > 1:
        ves = VerticalEquityStudy(usable, "sale_price", field, location, confidence_interval)
        vei = get_vertical_equity_scores(usable, "sale_price", field)
        gs = vei.get("group_stats")
        out["vertical"] = {
            "prd": _stat(ves.prd),
            "prb": _stat(ves.prb),
            "vei": _num(vei["vei"]),
            "vei_significance": _num(vei["vei_significance"]),
            "vei_verdict": _vei_verdict(_num(vei["vei"]), _num(vei["vei_significance"])),
            "vei_strata": [] if gs is None else [
                {"n": int(r["n"]), "median": _num(r["ratio"]), "low": _num(r["lower"]), "high": _num(r["upper"])}
                for _, r in gs.iterrows()
            ],
        }

    out["horizontal"] = None
    if "he_id" in df_univ and len(df_univ) > 0:
        he = HorizontalEquityStudy(df_univ, "he_id", field).summary
        out["horizontal"] = {
            "parcels": int(he.rows),
            "clusters": int(he.clusters),
            "median_chd": _num(he.median_chd),
            "min_chd": _num(he.min_chd),
            "max_chd": _num(he.max_chd),
        }
    return out


def _median(stat: ConfidenceStat, usable, field, confidence_interval, median_ci) -> dict:
    out = _stat(stat)
    if median_ci == "iaao" and len(usable):
        ratios = (usable[field] / usable["sale_price"]).to_numpy(dtype=float)
        low, high = calc_median_confidence_interval(ratios, confidence_interval)
        out = {"value": _num(np.median(ratios)), "low": _num(low), "high": _num(high)}
    return out


def _num(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def _stat(s: ConfidenceStat) -> dict:
    return {"value": _num(s.value), "low": _num(s.low), "high": _num(s.high)}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _num(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
