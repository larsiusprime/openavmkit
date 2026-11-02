"""Multi-parcel ("bulk") deed detection helpers for sales scrutiny.

A bulk deed conveys several parcels for one consideration, and that one price usually gets
stamped onto every parcel's sales record. None of those per-parcel prices is a usable
arm's-length price, so the heuristics in ``run_heuristics``
(:mod:`openavmkit.sales_scrutiny_study`) try to find them:

1. ``flag_dupe_deed_date`` -- rows sharing a deed (``analysis.sales_scrutiny.deed_id``) and
   sale date.
2. ``flag_dupe_date_price`` -- rows sharing a sale date and sale price. This is the fallback
   for jurisdictions without usable deed IDs: identical date + price across parcels is the
   footprint a bulk deed leaves behind. When deed IDs *are* available it is narrowed so that
   unrelated sales on separate deeds that merely coincide on date and price are not flagged.
3. ``flag_bulk_deed`` -- deeds that covered more than one parcel at ingestion
   (``sale_parcel_count``, computed in :func:`openavmkit.data.process_data` before the
   valid-sale filter), which catches bulk deeds whose sibling rows were filtered out and so
   left nothing for (1) or (2) to match against.

This module only depends on pandas so that :mod:`openavmkit.data` can use it without a
circular import.
"""
from __future__ import annotations

import pandas as pd


SALE_PARCEL_COUNT = "sale_parcel_count"


def _missing_id(s: pd.Series) -> pd.Series:
    """True where an identifier is null or blank."""
    return s.isna() | s.astype(str).str.strip().eq("")


def _signature(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Join fields into one string key, the way the original heuristics built them."""
    out = df[fields[0]].astype(str)
    for f in fields[1:]:
        out = out + "---" + df[f].astype(str)
    return out


def deed_group_ids(
    df: pd.DataFrame, deed_id: str | None = None, jurisdiction: str | None = None
) -> pd.Series | None:
    """Identify the deed (transaction) each sales row belongs to.

    With a configured ``deed_id`` column: ``[jurisdiction] + deed + sale_date``, and NA for
    rows whose deed is missing (unknown deeds must not group together). The jurisdiction is
    only used when that column is present in ``df``. Without a usable ``deed_id``: the row's
    ``key_sale``. Returns None when neither is available.
    """
    if deed_id is not None and deed_id in df.columns:
        fields = ([jurisdiction] if jurisdiction is not None and jurisdiction in df.columns else [])
        fields += [deed_id, "sale_date"]
        return _signature(df, fields).mask(_missing_id(df[deed_id]))
    if "key_sale" in df.columns:
        return df["key_sale"].astype(str).mask(_missing_id(df["key_sale"]))
    return None


def count_parcels_per_deed(
    df: pd.DataFrame, deed_id: str | None = None, jurisdiction: str | None = None
) -> pd.Series | None:
    """Number of distinct parcels (``key``) on each row's deed, as nullable ``Int64``.

    NA where the deed is unknown. Returns None when deeds or parcel keys can't be identified.
    """
    deeds = deed_group_ids(df, deed_id, jurisdiction)
    if deeds is None or "key" not in df.columns:
        return None
    counts = df["key"].groupby(deeds).transform("nunique")  # NA deed -> NaN
    return counts.astype("Int64")


def flag_dupe_deed_date(
    df: pd.DataFrame, deed_id: str, jurisdiction: str | None = None
) -> pd.Series:
    """Rows that share their deed and sale date with another row.

    Rows with a missing deed are never flagged here; they're left to ``flag_dupe_date_price``.
    """
    deeds = deed_group_ids(df, deed_id, jurisdiction)
    sizes = deeds.groupby(deeds).transform("size")
    return sizes.gt(1).fillna(False).astype(bool)


def flag_dupe_date_price(
    df: pd.DataFrame, jurisdiction: str | None = None, deed_id: str | None = None
) -> pd.Series:
    """Rows that share their sale date and sale price with another row.

    Without deed IDs, every row in a repeated (date, price) group is flagged: that pattern is
    the only trace a bulk deed leaves when you can't see the deed itself, and dropping a few
    coincidental matches is the price of catching them.

    With deed IDs, a row in a repeated (date, price) group is flagged only if something
    suggests it's part of a bundle or a duplicate record:

    - its deed is missing, or some other row in the group has a missing deed (which could be
      its sibling), or
    - it shares its deed with another row in the group, or
    - the same parcel appears more than once in the group.

    So unrelated sales on separate, known deeds that happen to match on date and price (a
    round price, or a builder selling identical lots on separate deeds) are kept.
    """
    fields = ([jurisdiction] if jurisdiction is not None and jurisdiction in df.columns else [])
    sig = _signature(df, fields + ["sale_date", "sale_price"])
    in_group = sig.groupby(sig).transform("size").gt(1)

    if deed_id is None or deed_id not in df.columns:
        return in_group.astype(bool)

    deed_missing = _missing_id(df[deed_id])
    group_has_missing = deed_missing.groupby(sig).transform("any")
    frame = pd.DataFrame({"sig": sig, "deed": df[deed_id].astype(str)})
    shares_deed = frame.duplicated(subset=["sig", "deed"], keep=False) & ~deed_missing
    suspicious = deed_missing | group_has_missing | shares_deed
    if "key" in df.columns:
        frame["key"] = df["key"].astype(str)
        suspicious |= frame.duplicated(subset=["sig", "key"], keep=False)
    return (in_group & suspicious).astype(bool)


def flag_bulk_deeds(df: pd.DataFrame) -> pd.Series:
    """Rows whose deed covered more than one parcel when the sales were loaded.

    Uses ``sale_parcel_count`` from :func:`openavmkit.data.process_data`, which is counted
    before invalid sales are filtered out. That catches an "orphan" bulk deed: one whose other
    parcels' rows were dropped, so the duplicate-based heuristics have nothing left to match
    against, but whose surviving row still carries the whole bundle's price. Without that
    column, nothing is flagged.
    """
    if SALE_PARCEL_COUNT not in df.columns:
        return pd.Series(False, index=df.index)
    return df[SALE_PARCEL_COUNT].fillna(1).astype("int64").gt(1)
