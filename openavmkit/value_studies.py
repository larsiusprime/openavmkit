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
from openavmkit.utilities.stats import ConfidenceStat
from openavmkit.vertical_equity_study import VerticalEquityStudy, get_vertical_equity_scores

OVERALL = "__overall__"


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
        Folder to write ``<field>.json`` into. ``None`` writes nothing.

    Returns
    -------
    dict
        ``{field: {"window": ..., "groups": {model_group: {...}}, "overall": {...}}}``.
    """
    if start_date is None or end_date is None:
        start_date, end_date = get_look_back_dates(settings)
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)

    df_sales = get_hydrated_sales_from_sup(sup)
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
        for mg in model_groups:
            groups[mg] = _study_one(
                df_sales[df_sales["model_group"].eq(mg)],
                df_univ[df_univ["model_group"].eq(mg)],
                field, location, confidence_interval, max_trim,
            )
        overall = _study_one(df_sales, df_univ, field, location, confidence_interval, max_trim)
        results[field] = {
            "field": field,
            "window": {"start": start_date.date().isoformat(), "end": end_date.date().isoformat()},
            "confidence_interval": confidence_interval,
            "location": location,
            "groups": groups,
            "overall": overall,
        }
        if outpath is not None:
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, f"{field}.json"), "w") as f:
                json.dump(results[field], f, indent=2, default=_json_default)
    return results


def _study_one(df_sales, df_univ, field, location, confidence_interval, max_trim) -> dict:
    usable = df_sales[
        df_sales[field].notna() & np.isfinite(df_sales[field]) & df_sales["sale_price"].gt(0)
    ]
    out = {"count_sales": int(len(usable)), "count_parcels": int(len(df_univ))}

    rs = RatioStudyBootstrapped(
        usable[field].to_numpy(), usable["sale_price"].to_numpy(), max_trim, confidence_interval
    )
    out["ratio"] = {
        "count": int(rs.count),
        "count_trim": int(rs.count_trim),
        "median_ratio": _stat(rs.median_ratio),
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
        out["vertical"] = {
            "prd": _stat(ves.prd),
            "prb": _stat(ves.prb),
            "vei": _num(vei["vei"]),
            "vei_significance": _num(vei["vei_significance"]),
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
