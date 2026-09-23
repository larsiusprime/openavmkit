"""
Settings.json loader, preprocessor, and typed accessors.

This module is the single source of truth for reading ``settings.json``.
It performs four transformations on the user's file before any other
module sees it:

1. **Comment stripping** — keys prefixed with ``__`` are removed.
2. **Variable resolution** — string values prefixed with ``$$`` are
   replaced by the value at the dotted path inside the same settings
   tree (recursive until stable).
3. **Template merging** — the user's settings are merged with the built-in
   ``settings.template.json``, so users only need to specify overrides.
4. **Flag handling** — ``!key`` overwrites the template instead of merging,
   ``+key`` extends template lists instead of replacing them.

After loading, a large collection of typed accessors (``get_valuation_date``,
``get_model_group_ids``, ``get_fields_categorical``, ``area_unit``, etc.)
provides a stable, well-typed interface to the resulting dict — prefer
these over reaching into the dict directly.

See :doc:`/advanced_settings` for a user-facing reference of the
preprocessor and high-impact settings.
"""
import importlib
import json
import os
import warnings
import geopandas as gpd

import pandas as pd
from datetime import datetime


def load_settings(
    settings_file: str = "in/settings.json", settings_object: dict = None, error:bool=True, warning:bool=True
) -> dict | None:
    """
    Load settings file from disk

    Parameters
    ----------
    settings_file : str
        Path to the settings file
    settings_object : dict, optional
        Already loaded settings object
    error : bool, optional
        Whether to raise errors or simply emit warnings if something is wrong
    warning : bool, optional
        Whether to emit warnings if something is wrong

    Returns
    -------
    dict
        The settings object
    """
    settings : dict | None = None

    if settings_object is None:
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
        except FileNotFoundError:
            cwd = os.getcwd()
            full_path = os.path.join(cwd, settings_file)
            exists = os.path.exists(full_path)
            msg = f"Could not find settings file: {settings_file}. Go to '{cwd}' and create a settings.json file there! {full_path} exists? {exists}"
            if error:
                raise FileNotFoundError(msg)
            else:
                if warning:
                    warnings.warn(msg)

    else:
        settings = settings_object

    if settings is None:
        return None

    template = _load_settings_template()
    # merge settings with template; settings will overwrite template values
    settings = _merge_settings(template, settings)
    base_dd = {"data_dictionary": _load_data_dictionary_template()}
    settings = _merge_settings(base_dd, settings)
    settings = _remove_comments_from_settings(settings)
    settings = _replace_variables(settings)
    return settings


def get_model_group(s: dict, key: str) -> dict:
    """
    Get a model group definition object from the settings dictionary

    Parameters
    ----------
    s : dict
        Settings object
    key : str
        The name of the model group

    Returns
    -------
    dict
        Model group definition
    """
    return s.get("modeling", {}).get("model_groups", {}).get(key, {})


def get_valuation_date(s: dict) -> datetime:
    """
    Get the valuation date from the settings dictionary

    Parameters
    ----------
    s : dict
        Settings dictionary

    Returns
    -------
    datetime
        The valuation date
    """
    val_date_str: str | None = (
        s.get("modeling", {}).get("metadata", {}).get("valuation_date", None)
    )

    if val_date_str is None:
        # return January 1 of this year:
        return datetime(datetime.now().year, 1, 1)

    # process the date from string to datetime using format YYYY-MM-DD:
    val_date = datetime.strptime(val_date_str, "%Y-%m-%d")
    return val_date


def get_model_seed(s: dict) -> int:
    """Return the random seed used for model tuning and fitting.

    Read from ``modeling.metadata.seed`` (default ``42``). This is the single source of
    truth for reproducibility of the (otherwise nondeterministic) tree-based models: it
    seeds the Optuna hyperparameter sampler, the cross-validation folds, and the final
    model fits.

    Modeling is **always deterministic** — there is no nondeterministic mode. The
    XGBoost/LightGBM tuners stay parallel *and* reproducible via batched ask-and-tell
    (see ``_run_batched`` in :mod:`openavmkit.tuning`), so determinism costs no
    parallelism. Provide your own integer to vary the seed; an absent or ``null`` value
    falls back to ``42``.

    Parameters
    ----------
    s : dict
        Settings dictionary.

    Returns
    -------
    int
        The model seed (always an integer).
    """
    seed = s.get("modeling", {}).get("metadata", {}).get("seed", 42)
    return 42 if seed is None else int(seed)


def get_assessor_holdout_mode(s: dict) -> str:
    """Return how the assessor's values relate to the test holdout.

    openavmkit cannot know whether a third party's values respect its randomly-drawn
    holdout, so by default it does not show the assessor head-to-head on that holdout. If
    *you* are the assessor (or otherwise know the holdout status), set
    ``analysis.ratio_study.assessor_holdout`` to declare it:

    - ``"unknown"`` (default): holdout status of the assessor's values is unknown, so the
      assessor is not shown on the pre-valuation random holdout.
    - ``"shared"``: the assessor's values were produced honoring this same test holdout
      (either openavmkit's generated keys, or your own keys supplied via
      ``modeling.instructions.test_keys_file``), so the assessor *is* shown head-to-head on
      the holdout.

    Parameters
    ----------
    s : dict
        Settings dictionary.

    Returns
    -------
    str
        ``"unknown"`` or ``"shared"``.
    """
    mode = (
        s.get("analysis", {})
        .get("ratio_study", {})
        .get("assessor_holdout", "unknown")
    )
    return str(mode).lower()


def get_look_back_dates(s: dict):
    rs = s.get("analysis", {}).get("ratio_study", {})
    look_back_years = rs.get("look_back_years", 1)
    val_date = get_valuation_date(s)

    # Look back N years BEFORE the valuation date
    look_back_date = val_date - pd.DateOffset(years=look_back_years)

    return look_back_date, val_date


def _parse_use_sales_from_entry(entry, val_year: int) -> tuple[int | None, int | None]:
    """Resolve one ``use_sales_from`` entry into ``(improved_year, vacant_year)``.

    An entry is either an ``int`` (same cutoff for both sale types) or a dict
    ``{"improved": YYYY, "vacant": YYYY}`` (missing keys fall back to ``val_year - 5``).
    ``None`` -> ``(None, None)`` (no threshold).
    """
    if entry is None:
        return None, None
    if isinstance(entry, int):
        return entry, entry
    if isinstance(entry, dict):
        return entry.get("improved", val_year - 5), entry.get("vacant", val_year - 5)
    return None, None


def _is_per_group_use_sales_from(usf) -> bool:
    """True if ``use_sales_from`` uses the per-model-group schema (``default`` /
    ``by_model_group``) rather than the legacy scalar / ``{improved, vacant}`` forms."""
    return isinstance(usf, dict) and ("by_model_group" in usf or "default" in usf)


def resolve_use_sales_from(
    s: dict, model_group: str | None = None
) -> tuple[int | None, int | None]:
    """Resolve ``modeling.metadata.use_sales_from`` into per-type year thresholds.

    The setting can take four forms:

      * ``None`` (missing) — returns ``(None, None)``; callers should treat as
        "no threshold".
      * ``int`` — single cutoff applied to both improved and vacant sales.
      * ``dict`` ``{"improved": YYYY, "vacant": YYYY}`` — per-type cutoffs (missing
        keys fall back to ``val_year - 5``).
      * ``dict`` ``{"default": <entry>, "by_model_group": {<group>: <entry>}}`` —
        **per-model-group** cutoffs, where each ``<entry>`` is itself an ``int`` or a
        ``{improved, vacant}`` dict. A group listed in ``by_model_group`` uses its own
        window; any other group (or ``model_group=None``) uses ``default``.

    Returns ``(improved_year, vacant_year)`` for the requested ``model_group`` (or the
    default when no group is given). Always use this helper instead of parsing
    ``use_sales_from`` inline — the dict forms need careful branching, and naïve scalar
    comparisons against a ``Series`` crash with "TypeError: len() of unsized object".

    See also :func:`use_sales_from_floor`, which the cleaning/clipping stages use to keep
    the widest window any group needs (they hard-drop, and run before the per-group split).
    """
    md = s.get("modeling", {}).get("metadata", {})
    if "use_sales_from" not in md:
        return None, None
    usf = md["use_sales_from"]
    if usf is None:
        return None, None
    if isinstance(usf, int):
        return usf, usf
    if isinstance(usf, dict):
        val_year = get_valuation_date(s).year
        if _is_per_group_use_sales_from(usf):
            by_group = usf.get("by_model_group", {}) or {}
            if model_group is not None and model_group in by_group:
                return _parse_use_sales_from_entry(by_group[model_group], val_year)
            return _parse_use_sales_from_entry(usf.get("default"), val_year)
        # legacy per-type {improved, vacant}
        return usf.get("improved", val_year - 5), usf.get("vacant", val_year - 5)
    # Fall through: malformed value — return None/None and let callers no-op.
    return None, None

def use_sales_from_floor(s: dict) -> tuple[int | None, int | None]:
    """The most-permissive (oldest) ``use_sales_from`` across all groups.

    The cleaning / clipping stages permanently drop too-old sales *before* the per-group
    train/test split runs, so they must keep down to the widest window any model group
    needs — otherwise a group with a longer reach (e.g. data-starved commercial) would
    have its older sales deleted before it is ever modeled. This returns that floor; the
    per-group narrowing then happens in ``get_data_split_for`` via
    :func:`resolve_use_sales_from` with a ``model_group``.

    Floor semantics, per sale type: ``None`` (unbounded) if *any* relevant window is
    ``None``; otherwise the minimum year. For scalar / legacy ``{improved, vacant}``
    configs this is identical to :func:`resolve_use_sales_from`.
    """
    md = s.get("modeling", {}).get("metadata", {})
    if "use_sales_from" not in md or md["use_sales_from"] is None:
        return None, None
    usf = md["use_sales_from"]
    if isinstance(usf, int):
        return usf, usf
    if isinstance(usf, dict):
        val_year = get_valuation_date(s).year
        if _is_per_group_use_sales_from(usf):
            entries = [usf.get("default")] + list((usf.get("by_model_group", {}) or {}).values())
            pairs = [_parse_use_sales_from_entry(e, val_year) for e in entries]
            imprs = [p[0] for p in pairs]
            vacs = [p[1] for p in pairs]
            floor_impr = None if any(x is None for x in imprs) else min(imprs)
            floor_vac = None if any(x is None for x in vacs) else min(vacs)
            return floor_impr, floor_vac
        return usf.get("improved", val_year - 5), usf.get("vacant", val_year - 5)
    return None, None


def get_center(s: dict, gdf: gpd.GeoDataFrame = None) -> tuple[float, float]:
    """
    Get the centroid of all the provided parcel geometry

    Parameters
    ----------
    s : dict
        Settings dictionary
    gdf : gpd.GeoDataFrame
        Parcel geometry

    Return
    ------
    tuple[float, float]
        Centroid of all the parcel geometry
    """
    center: dict | None = s.get("locality", {}).get("center", None)
    if center is not None:
        if "longitude" not in center or "latitude" not in center:
            raise ValueError(
                "Could not find both 'longitude' and 'latitude' in 'settings.locality.center'!"
            )
        latitude = center["latitude"]
        longitude = center["longitude"]
        return longitude, latitude
    elif gdf is not None:
        # calculate the center of the gdf
        centroid = gdf.geometry.unary_union.centroid
        return centroid.x, centroid.y
    else:
        raise ValueError("Could not find locality.center in settings!")


# ---------------------------------------------------------------------------
# Area-statistic ("neighborhood enrichment") field naming and classification.
#
# Area stats are per-location summary statistics stamped onto every parcel as
# ``area_stat_<location>_<field>_<stat>`` features (see ``openavmkit.area_stats``).
# The naming + classification helpers live here so they stay the single source of
# truth shared by the enrichment code (which generates the columns) and the field
# getters (which must auto-discover them for modeling).
# ---------------------------------------------------------------------------

AREA_STAT_PREFIX = "area_stat_"

# Numeric base fields: these stats are emitted by default, the rest are opt-in.
AREA_STATS_NUMERIC_DEFAULT = ["mean", "median", "std"]
AREA_STATS_NUMERIC_OPTIONAL = ["cv", "p25", "p75", "iqr", "min", "max", "sum"]

# Categorical base fields: ``mode``/``mode_frac`` by default, the rest opt-in.
AREA_STATS_CATEGORICAL_DEFAULT = ["mode", "mode_frac"]
AREA_STATS_CATEGORICAL_OPTIONAL = ["nunique", "entropy"]

# Stats whose output is itself categorical rather than numeric.
AREA_STATS_CATEGORICAL_OUTPUT = {"mode"}


def get_area_stats_config(s: dict) -> dict:
    """Return the ``data.process.enrich.area_stats`` config block (or ``{}``).

    Parameters
    ----------
    s : dict
        Settings dictionary.

    Returns
    -------
    dict
        The area-stats configuration, or an empty dict if the feature is not
        configured for this locality.
    """
    return (
        s.get("data", {})
        .get("process", {})
        .get("enrich", {})
        .get("area_stats", {})
    ) or {}


def make_area_stat_field_name(location: str, field: str, stat: str) -> str:
    """Build the derived column name for one (location, field, stat) combination.

    Examples
    --------
    ``make_area_stat_field_name("neighborhood", "bldg_area_finished_sqft", "mean")``
    returns ``"area_stat_neighborhood_bldg_area_finished_sqft_mean"``.
    """
    return f"{AREA_STAT_PREFIX}{location}_{field}_{stat}"


# Per-location count columns always emitted by area-stats enrichment:
#   count                 -> universe parcels in the location
#   sales_count           -> training valid sales in the location
#   sales_count_improved  -> of those, improved sales
#   sales_count_vacant    -> of those, vacant sales
AREA_STAT_COUNT_KINDS = ["count", "sales_count", "sales_count_improved", "sales_count_vacant"]


def make_area_stat_count_field_name(location: str, kind: str = "count") -> str:
    """Build a per-location count column name (e.g. ``area_stat_neighborhood_sales_count``).

    ``kind`` is one of :data:`AREA_STAT_COUNT_KINDS`; defaults to the universe parcel count.
    """
    return f"{AREA_STAT_PREFIX}{location}_{kind}"


def is_sale_derived_field(s: dict, field: str) -> bool:
    """Whether ``field`` is the sale price (or a sale-price variant).

    Sale-derived fields must be aggregated over training valid sales only to
    avoid leaking the target; everything else aggregates over the universe. The
    sale field is always ``sale_price`` or ``sale_price_time_adj`` (see
    ``openavmkit.data.get_sale_field``), so a prefix check covers both without a
    circular import back into ``data``.
    """
    return str(field).startswith("sale_price")


# Bare sale-price entries in an area_stats `fields` list trigger auto-expansion into the
# full per-area family. The colloquial "_time_adjusted" spelling is accepted too.
AREA_STATS_SALE_PRICE_TRIGGERS = {
    "sale_price",
    "sale_price_time_adj",
    "sale_price_time_adjusted",
}


def expand_area_stats_fields(s: dict, fields: list) -> list:
    """Expand a bare sale-price entry into the full per-area sale-rate family.

    Listing ``sale_price`` or ``sale_price_time_adj`` (treated as aliases) auto-generates
    the price level plus the three area-normalized rates (``_impr_<unit>``,
    ``_vacant_land_<unit>``, ``_impr_land_<unit>``) — no suffixes needed. It uses the
    canonical sale field, matching :func:`openavmkit.data.get_sale_field`: the
    time-adjusted price when time adjustment is enabled, otherwise the raw sale price (one
    base, not both). All other fields pass through unchanged; the result is de-duplicated
    and order-preserving.

    Parameters
    ----------
    s : dict
        Settings dictionary.
    fields : list
        The configured ``area_stats.fields`` list.

    Returns
    -------
    list
        The expanded field list.
    """
    unit = area_unit(s)
    ta_on = bool(get_time_adjustment_instructions(s).get("use", True))
    base = "sale_price_time_adj" if ta_on else "sale_price"

    expanded: list = []
    for field in fields:
        if field in AREA_STATS_SALE_PRICE_TRIGGERS:
            expanded.append(base)  # price level
            if unit:
                expanded.append(f"{base}_impr_{unit}")
                expanded.append(f"{base}_vacant_land_{unit}")
                expanded.append(f"{base}_impr_land_{unit}")
        else:
            expanded.append(field)

    seen, out = set(), []
    for field in expanded:
        if field not in seen:
            seen.add(field)
            out.append(field)
    return out


def _classify_base_field_raw(s: dict, field: str) -> tuple[str | None, str | None]:
    """Resolve a base field's ``(bucket, kind)`` from the *raw* classification lists.

    Reads ``field_classification`` directly (not via :func:`get_fields_numeric` /
    :func:`get_fields_categorical`) so it can be called from inside those getters
    without recursion. Returns ``(None, None)`` if the field is unclassified.
    """
    if is_sale_derived_field(s, field):
        return "other", "numeric"
    fc = s.get("field_classification", {})
    for bucket in ("land", "impr", "other"):
        b = fc.get(bucket, {})
        if field in b.get("numeric", []) or field in b.get("boolean", []):
            return bucket, "numeric"
        if field in b.get("categorical", []):
            return bucket, "categorical"
    return None, None


def get_area_stats_fields(s: dict, df: pd.DataFrame = None) -> dict:
    """Enumerate the area-stat derived fields and their classifications.

    For each configured ``location × field × stat`` combination this returns the
    generated column name mapped to metadata describing how it should be treated:

    - ``bucket``: ``"land"`` / ``"impr"`` / ``"other"`` inherited from the base field
    - ``kind``: ``"numeric"`` or ``"categorical"`` (the *output* kind of the stat)
    - ``location`` / ``base_field`` / ``stat``: the components it was built from

    A per-location ``count`` column (group size) is always included. Base fields
    that are unclassified in settings are skipped (so they don't pollute the
    model's feature lists with unknown buckets).

    Parameters
    ----------
    s : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        If given, only columns actually present in ``df`` are returned.

    Returns
    -------
    dict
        Mapping of derived column name to its metadata dict.
    """
    cfg = get_area_stats_config(s)
    out: dict = {}
    if not cfg:
        return out

    locations = cfg.get("locations", []) or []
    fields = expand_area_stats_fields(s, cfg.get("fields", []) or [])
    num_stats = cfg.get("stats", AREA_STATS_NUMERIC_DEFAULT) or []
    cat_stats = cfg.get("categorical_stats", AREA_STATS_CATEGORICAL_DEFAULT) or []

    for location in locations:
        for count_kind in AREA_STAT_COUNT_KINDS:
            out[make_area_stat_count_field_name(location, count_kind)] = {
                "bucket": "other",
                "kind": "numeric",
                "location": location,
                "base_field": None,
                "stat": count_kind,
            }
        for field in fields:
            bucket, kind = _classify_base_field_raw(s, field)
            if kind is None:
                continue
            stats_list = num_stats if kind == "numeric" else cat_stats
            for stat in stats_list:
                name = make_area_stat_field_name(location, field, stat)
                out_kind = (
                    "categorical" if stat in AREA_STATS_CATEGORICAL_OUTPUT else "numeric"
                )
                out[name] = {
                    "bucket": bucket,
                    "kind": out_kind,
                    "location": location,
                    "base_field": field,
                    "stat": stat,
                }

    if df is not None:
        out = {k: v for k, v in out.items() if k in df.columns}
    return out


def get_fields_land(s: dict, df: pd.DataFrame = None) -> dict:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to land.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        All fields pertaining to land, organized as a dictionary containing three keys:

          - "categorical": list of categorical fields
          - "numeric": list of numerical fields
          - "boolean": list of boolean fields
    """
    fields_land = _get_fields(s, "land", df)
    fields_unclassified = _get_unclassified_fields(s, df)

    for field in fields_unclassified:
        if field.startswith("dist_to_") or field.startswith("within_") or field.startswith("proximity_to_") or field.startswith("spatial_lag_"):
            fields_land["numeric"].append(field)
        # Defensive net for area-stat columns that reach the getter without their
        # config context (e.g. loaded from cache). Numeric stats only -- the
        # categorical ``_mode`` variant is handled by get_fields_categorical.
        elif field.startswith(AREA_STAT_PREFIX) and not field.endswith("_mode"):
            fields_land["numeric"].append(field)

    for key in fields_land:
        # remove duplicates:
        fields_land[key] = list(set(fields_land[key]))

    return fields_land


def get_fields_land_as_list(s: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to land.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list
        A list of all field names pertaining to land
    """
    fields = get_fields_land(s, df)
    return (
        fields.get("categorical", [])
        + fields.get("numeric", [])
        + fields.get("boolean", [])
    )


def get_fields_impr(s: dict, df: pd.DataFrame = None) -> dict:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        All fields pertaining to buildings/improvements, organized as a dictionary containing three keys:

          - "categorical": list of categorical fields
          - "numeric": list of numerical fields
          - "boolean": list of boolean fields
    """
    return _get_fields(s, "impr", df)


def get_fields_impr_as_list(s: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list
        A list of all field names pertaining to buildings/improvements
    """
    fields = get_fields_impr(s, df)
    return (
        fields.get("categorical", [])
        + fields.get("numeric", [])
        + fields.get("boolean", [])
    )


def get_fields_other(s: dict, df: pd.DataFrame = None) -> dict:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to neither land nor
    buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        All fields pertaining neither to land nor to buildings/improvements,
        organized as a dictionary containing three keys:

          - "categorical": list of categorical fields
          - "numeric": list of numerical fields
          - "boolean": list of boolean fields
    """
    return _get_fields(s, "other", df)


def get_fields_other_as_list(s: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all fields in the given dataframe that are classified in settings as pertaining to neither land nor to
    buildings/improvements.

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list
        A list of all field names pertaining neither to land nor to buildings/improvements
    """
    fields = get_fields_other(s, df)
    return (
        fields.get("categorical", [])
        + fields.get("numeric", [])
        + fields.get("boolean", [])
    )


def get_fields_date(s: dict, df: pd.DataFrame):
    """
    Get all fields pertaining to dates

    Parameters
    ----------
    s : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list[str]
        List of field names pertaining to dates
    """

    # TODO: add to this as necessary
    all_date_fields = ["sale_date", "date"]
    date_fields = [field for field in all_date_fields if field in df]
    for field in df:
        if "_date" in field and field not in date_fields:
            date_fields.append(field)

    return date_fields


def get_fields_boolean(
    s: dict,
    df: pd.DataFrame = None,
    types: list[str] = None
) -> list[str]:
    """
    Retrieve boolean field names based on settings and optional filters.

    Parameters
    ----------
    s : dict
        Settings dictionary containing field configurations.
    df : pandas.DataFrame, optional
        DataFrame to filter fields by presence. Defaults to None.
    types : list[str], optional
        List of field classification types to include (e.g., ["land", "impr", "other"]).
        Defaults to None, which includes all types.

    Returns
    -------
    list[str]
        List of boolean field names matching the specified criteria.
    """
    if types is None:
        types = ["land", "impr", "other"]
    bools = []

    # Determine which boolean field to get based on na_handling
    field_type = "boolean"

    if "land" in types:
        bools += s.get("field_classification", {}).get("land", {}).get(field_type, [])
    if "impr" in types:
        bools += s.get("field_classification", {}).get("impr", {}).get(field_type, [])
    if "other" in types:
        bools += s.get("field_classification", {}).get("other", {}).get(field_type, [])

    if df is not None:
        bools = [bool for bool in bools if bool in df]
    return bools


def get_fields_categorical(
    s: dict,
    df: pd.DataFrame = None,
    include_boolean: bool = False,
    types: list[str] = None,
) -> list[str]:
    """
    Retrieve categorical field names based on settings and optional filters.

    Parameters
    ----------
    s : dict
        Settings dictionary containing field configurations.
    df : pandas.DataFrame, optional
        DataFrame to filter fields by presence. Defaults to None.
    include_boolean : bool, optional
        Whether to include boolean fields in the results or not. Defaults to False.
    types : list[str], optional
        List of field classification types to include (e.g., ["land", "impr", "other"]).
        Defaults to None, which includes all types.

    Returns
    -------
    list[str]
        List of categorical field names matching the specified criteria.
    """
    if types is None:
        types = ["land", "impr", "other"]
    cats = []
    if "land" in types:
        cats += s.get("field_classification", {}).get("land", {}).get("categorical", [])
    if "impr" in types:
        cats += s.get("field_classification", {}).get("impr", {}).get("categorical", [])
    if "other" in types:
        cats += (
            s.get("field_classification", {}).get("other", {}).get("categorical", [])
        )
    if include_boolean:
        if "land" in types:
            cats += s.get("field_classification", {}).get("land", {}).get("boolean", [])
        if "impr" in types:
            cats += s.get("field_classification", {}).get("impr", {}).get("boolean", [])
        if "other" in types:
            cats += (
                s.get("field_classification", {}).get("other", {}).get("boolean", [])
            )

    # collapse_sparse_categories `output_field` variants (e.g. "subdivision_collapsed")
    # are categorical by construction -- they inherit their source field's classification.
    # Derived from settings on every call so the inheritance persists across notebooks
    # (settings reload fresh each notebook). Without this, a collapsed variant reaches the
    # tree models as a raw string column and LightGBM/XGBoost reject it.
    collapse = (
        s.get("data", {}).get("process", {}).get("collapse_sparse_categories", {})
    )
    if isinstance(collapse, dict):
        for src_field, cfg in collapse.items():
            if not isinstance(cfg, dict):
                continue
            out_field = cfg.get("output_field")
            if out_field and src_field in cats and out_field not in cats:
                cats.append(out_field)

    # area-stat derived fields whose output is categorical (e.g. ``..._mode``)
    # inherit their base field's bucket; include those in the requested types.
    for name, meta in get_area_stats_fields(s, df).items():
        if meta["kind"] == "categorical" and meta["bucket"] in types and name not in cats:
            cats.append(name)

    if df is not None:
        cats = [cat for cat in cats if cat in df]
    return cats


def get_fields_numeric(
    s: dict,
    df: pd.DataFrame = None,
    include_boolean: bool = False,
    types: list[str] = None,
) -> list[str]:
    """
     Retrieve numeric field names based on settings and optional filters.

     Parameters
     ----------
     s : dict
         Settings dictionary containing field configurations.
     df : pandas.DataFrame, optional
         DataFrame to filter fields by presence. Defaults to None.
     include_boolean : bool, optional
         Whether to include boolean fields in the results or not. Defaults to False.
     types : list[str], optional
         List of field classification types to include (e.g., ["land", "impr", "other"]).
         Defaults to None, which includes all types.

     Returns
     -------
     list[str]
         List of numeric field names matching the specified criteria.
     """
    if types is None:
        types = ["land", "impr", "other"]
    nums = []
    if "land" in types:
        nums += s.get("field_classification", {}).get("land", {}).get("numeric", [])
    if "impr" in types:
        nums += s.get("field_classification", {}).get("impr", {}).get("numeric", [])
    if "other" in types:
        nums += s.get("field_classification", {}).get("other", {}).get("numeric", [])
    if include_boolean:
        if "land" in types:
            nums += s.get("field_classification", {}).get("land", {}).get("boolean", [])
        if "impr" in types:
            nums += s.get("field_classification", {}).get("impr", {}).get("boolean", [])
        if "other" in types:
            nums += (
                s.get("field_classification", {}).get("other", {}).get("boolean", [])
            )

    # area-stat derived fields with numeric output inherit their base field's
    # bucket; include those in the requested types so models auto-discover them.
    for name, meta in get_area_stats_fields(s, df).items():
        if meta["kind"] == "numeric" and meta["bucket"] in types and name not in nums:
            nums.append(name)

    if df is not None:
        nums = [num for num in nums if num in df]
    return nums


def get_variable_interactions(entry: dict, settings: dict, df: pd.DataFrame = None) -> dict:
    """
    Get variable interaction information from a dictionary object

    Parameters
    ----------
    entry : dict
        The dictionary object that may contain variable interactions
    settings : dict
        Global settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    dict
        Interactions dictionary which maps field names to other field names, indicating variable interactions.

        Example:
        Interacting a categorical field like "neighborhood" with a numeric field like "land_area_{unit}" means that
        every one-hot-encoded descendant like "neighborhood=River Heights" will be multiplied against the numeric
        value of "land_area_{unit}", so this is a way to interact neighborhood dummies with land size.
    """
    unit = area_unit(settings)
    interactions: dict | None = entry.get("interactions", None)
    if interactions is None:
        return {}
    is_default = interactions.get("default", False)
    if is_default:
        result = {}
        fields_land = get_fields_categorical(
            settings, df, include_boolean=True, types=["land"]
        )
        fields_impr = get_fields_categorical(
            settings, df, include_boolean=True, types=["impr"]
        )
        for field in fields_land:
            result[field] = f"land_area_{unit}"
        for field in fields_impr:
            result[field] = f"bldg_area_finished_{unit}"
        return result
    else:
        return interactions.get("fields", {})


def get_data_dictionary(settings: dict) -> dict:
    """
    Get the data dictionary object

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    dict
        The data dictionary for this locality
    """
    return settings.get("data_dictionary", {})


def get_grouped_fields_from_data_dictionary(
    dd: dict, group: str, types: list[str] = None
) -> list[str]:
    """
    Get all field names from the data dictionary of the named group and, optionally, of the designated types.

    Parameters
    ----------
    dd : dict
        The data dictionary
    group : str
        Name of a particular group in the data dictionary
    types : list, optional
        If None, returns all field names in the group. If not, targets only those fields that match the
        listed types. Legal values are: "boolean", "str", "number", "percent", "date"

    Returns
    -------
    list[str]
        A list of field names belonging to the specified group
    """
    result = []
    for key in dd:
        entry = dd[key]
        if group in entry.get("groups", []):
            if types is None or entry.get("type") in types:
                result.append(key)
    return result


def get_model_group_ids(settings: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Get all model group ids specified in settings, in the preferred order specified by the user

    Parameters
    ----------
    settings : dict
        Settings dictionary
    df : pd.DataFrame
        Your dataset

    Returns
    -------
    list[str]
        Ordered list of model group ids
    """
    modeling = settings.get("modeling", {})

    # Get the model groups defined in the settings
    model_groups = modeling.get("model_groups", {})

    # Get the preferred order, if any
    order = modeling.get("instructions", {}).get("model_group_order", [])

    if df is not None:
        # If a dataframe is provided, filter out model groups that are not present in the DataFrame
        model_groups_in_df = df["model_group"].unique()
        model_group_ids = [key for key in model_groups if key in model_groups_in_df]
    else:
        model_group_ids = [key for key in model_groups]

    # Order the model groups according to the preferred order
    ordered_ids = [key for key in order if key in model_group_ids]
    unordered_ids = [key for key in model_group_ids if key not in ordered_ids]
    ordered_ids += unordered_ids

    return ordered_ids


def length_unit(settings: dict)-> str|None:
    """
    Get the designated "small" length unit (feet or meters)
    
    Parameters
    ----------
    settings : dict
        Settings dictionary
        
    Returns
    -------
    str
        "ft" if units are imperial and "m" if units are metric
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "ft"
    elif base_units == "metric":
        return "m"


def big_length_unit(settings: dict):
    """
    Get the designated "big" length unit (miles or kilometers)
    
    Parameters
    ----------
    settings : dict
        Settings dictionary
        
    Returns
    -------
    str
        "mi" if units are imperial and "km" if units are metric
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "mi"
    elif base_units == "metric":
        return "km"


def area_unit(settings: dict):
    """
    Get the designated "small" area unit (square feet or square meters)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "sqft" if units are imperial and "sqm" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "sqft"
    elif base_units == "metric":
        return "sqm"


def big_area_unit(settings: dict)-> str|None:
    """
    Get the designated "large" area unit (acre or hectare)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "acre" if units are imperial and "ha" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "acre"
    elif base_units == "metric":
        return "ha"  # hectare


def get_short_distance_unit(settings: dict) -> str|None:
    """
    Get the designated "short" distance unit (foot or meter)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "ft" if units are imperial and "m" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "ft"
    elif base_units == "metric":
        return "m"


def get_long_distance_unit(settings: dict) -> str|None:
    """
    Get the designated "long" distance unit (mile or kilometer)

    Parameters
    ----------
    settings : dict
        Settings dictionary

    Returns
    -------
    str|None
        "mile" if units are imperial and "km" if units are metric
        None otherwise
    """
    base_units = settings.get("locality", {}).get("units", "imperial")
    if base_units == "imperial":
        return "mile"
    elif base_units == "metric":
        return "km"


def get_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
    """
    Retrieve location fields from settings. These are all the fields that are considered locations.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        Optional DataFrame to filter available locations.

    Returns
    -------
    list[str]
        List of location field names.
    """

    locations = (
        settings.get("field_classification", {})
        .get("important", {})
        .get("locations", [])
    )
    if df is not None:
        locations = [loc for loc in locations if loc in df]
    return locations
    

def get_ensemble_instructions(settings: dict, mv: str) -> dict:
    """
    Retrieves ensemble instructions for a particular modeling section
    
    Parameters
    ----------
    settings : dict
        Settings dictionary.
    mv : string
        Which section -- "main" or "vacant"
        
    Returns
    -------
    dict
        Dictionary object containing ensemble settings
    """
    
    instructions = settings.get("modeling", {}).get("instructions", {}).get(mv, {})
    
    ensemble = instructions.get("ensemble", {})
    type = ensemble.get("type", "default")
    # "default" is just an alias for "median" (the historical default
    # aggregation); normalize it so downstream only ever sees "median".
    if type == "default":
        type = "median"
    if type in ("median", "mean"):
        models = ensemble.get("models", [])
        # "optimize" controls whether the greedy backward-elimination optimizer
        # runs. Its default depends on whether the user supplied an explicit
        # model list:
        #   - models given, optimize unspecified  -> False (use the list as-is;
        #     it is a whitelist of exactly which models to ensemble)
        #   - models given, optimize=True          -> optimize *from* the whitelist
        #   - models omitted, optimize unspecified -> True (optimize over all
        #     models -- the historical default)
        #   - models omitted, optimize=True        -> optimize over all models
        optimize = ensemble.get("optimize", len(models) == 0)
        return {
            "type": type,
            "models": models,
            "optimize": optimize,
        }
    elif type == "local":
        locations = ensemble.get("locations", None)
        if locations is None:
            locations = get_locations(settings)
        if locations is None:
            locations = []
        return {
            "type": "local",
            "locations": locations
        }

def get_time_adjustment_instructions(settings: dict):
    return settings.get("data", {}).get("process", {}).get("time_adjustment", {})


def get_collapse_sparse_categories_config(settings: dict) -> dict:
    """Get the ``data.process.collapse_sparse_categories`` config block.

    Parameters
    ----------
    settings : dict
        Settings dictionary.

    Returns
    -------
    dict
        Mapping from field name to its per-field collapse config (keys:
        ``sales_min``, ``univ_min``, optional ``replacement_value``). Empty
        dict if the section is absent.
    """
    return (
        settings.get("data", {}).get("process", {}).get("collapse_sparse_categories", {})
    )


def is_collapse_strict(settings: dict) -> bool:
    """Whether cardinality-collapse location guards should raise instead of warn.

    Opt in by setting the reserved boolean key
    ``data.process.collapse_sparse_categories.strict`` to ``true``. (Note: a
    ``__strict`` key would be stripped as a comment, so the flag is a single
    reserved ``strict`` key alongside the per-field entries.)

    Parameters
    ----------
    settings : dict
        Settings dictionary.

    Returns
    -------
    bool
        True if guards should raise ``ValueError``; False (default) to warn.
    """
    return bool(get_collapse_sparse_categories_config(settings).get("strict", False))


def get_collapsed_fields(settings: dict) -> set[str]:
    """Return the set of columns that carry a collapsed (``"Other"``) bucket.

    For each entry in ``data.process.collapse_sparse_categories``, the collapsed
    column is the entry's ``output_field`` if set, otherwise the source field
    (the dict key). This is the column downstream code should treat as
    cardinality-collapsed — when ``output_field`` is used the raw source field is
    left intact and is *not* considered collapsed. Reserved non-dict keys (e.g.
    ``strict``) are skipped.

    Parameters
    ----------
    settings : dict
        Settings dictionary.

    Returns
    -------
    set[str]
        Column names that have been (or will be) cardinality-collapsed.
    """
    config = get_collapse_sparse_categories_config(settings)
    collapsed = set()
    for field, field_cfg in config.items():
        if not isinstance(field_cfg, dict):
            continue
        collapsed.add(field_cfg.get("output_field", field))
    return collapsed


def is_field_collapsed(settings: dict, field: str) -> bool:
    """Return True if ``field`` is a cardinality-collapsed output column.

    See :func:`get_collapsed_fields`.
    """
    return field in get_collapsed_fields(settings)


def get_location_fields(settings: dict, df: pd.DataFrame = None) -> set[str]:
    """Return every field the settings treat as a coherent geographic location.

    Collapsing any of these *in place* via ``collapse_sparse_categories`` is a
    footgun: downstream grouping / clustering / breakdowns assume each location
    value is a geographically coherent zone, and a collapsed location merges
    unrelated zones into one ``"Other"`` bucket. This aggregates, from
    ``settings``:

    - ``field_classification.important.locations``
    - ``field_classification.important.fields.loc_*`` (the mapped field names)
    - ``analysis.{sales_scrutiny,horizontal_equity,land_equity,impr_equity}.location``
    - ``analysis.ratio_study.breakdowns[].by`` entries written as ``<loc_*>``
      (resolved through ``important.fields``)
    - ``modeling.instructions.{main,vacant}.ensemble.locations`` and any
      per-model ``modeling.models.*.*.locations``
    - ``land.lycd.*.location``

    ``field_classification.important.report_locations`` is intentionally excluded
    — those are used only as report output columns (benign if collapsed).

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    df : pandas.DataFrame, optional
        If given, the result is filtered to columns present in ``df``.

    Returns
    -------
    set[str]
        Field names used as coherent geographic locations.
    """
    fc = settings.get("field_classification", {})
    important = fc.get("important", {})
    fields_map = important.get("fields", {}) or {}

    out: set[str] = set()
    out.update(important.get("locations", []) or [])
    for alias, actual in fields_map.items():
        if isinstance(alias, str) and alias.startswith("loc_") and actual:
            out.add(actual)

    analysis = settings.get("analysis", {})
    for sect in ("sales_scrutiny", "horizontal_equity", "land_equity", "impr_equity"):
        loc = analysis.get(sect, {}).get("location")
        if loc:
            out.add(loc)

    for bd in analysis.get("ratio_study", {}).get("breakdowns", []) or []:
        by = bd.get("by") if isinstance(bd, dict) else None
        if isinstance(by, str) and by.startswith("<") and by.endswith(">"):
            actual = fields_map.get(by[1:-1])
            if actual:
                out.add(actual)

    modeling = settings.get("modeling", {})
    instr = modeling.get("instructions", {})
    for mv in ("main", "vacant"):
        ens = instr.get(mv, {}).get("ensemble", {})
        out.update(ens.get("locations", []) or [])
    for mv_block in modeling.get("models", {}).values():
        if not isinstance(mv_block, dict):
            continue
        for model_cfg in mv_block.values():
            if isinstance(model_cfg, dict):
                out.update(model_cfg.get("locations", []) or [])

    for cfg in settings.get("land", {}).get("lycd", {}).values():
        if isinstance(cfg, dict) and cfg.get("location"):
            out.add(cfg["location"])

    out.discard(None)
    if df is not None:
        out = {f for f in out if f in df.columns}
    return out


# Tracks (field, context) pairs already warned about, so per-model-group loops
# don't emit the same location-collapse warning dozens of times in one run.
_LOCATION_COLLAPSE_WARNED: set = set()


def warn_if_location_collapsed(
    settings: dict, fields, context: str, df: pd.DataFrame = None
) -> None:
    """Warn (or raise, if strict) when a location field used here was collapsed.

    Call this at sites that consume a field as a coherent geographic grouping key
    (equity clustering, ratio-study breakdowns, local-ensemble selection, etc.).
    If the field is in :func:`get_collapsed_fields`, collapsing has merged
    unrelated zones into the replacement bucket, so grouping by it is wrong.

    Deduplicates on ``(field, context)`` so repeated per-model-group calls warn
    only once per run.

    Parameters
    ----------
    settings : dict
        Settings dictionary.
    fields : str or Iterable[str]
        The location field(s) about to be used at this site.
    context : str
        Short description of the consuming site, e.g. ``"horizontal equity
        clustering"`` — used in the message.
    df : pandas.DataFrame, optional
        Unused; accepted so callers can pass it uniformly.
    """
    if fields is None:
        return
    if isinstance(fields, str):
        fields = [fields]
    collapsed = get_collapsed_fields(settings)
    strict = is_collapse_strict(settings)
    for field in fields:
        if not field or field not in collapsed:
            continue
        key = (field, context)
        if key in _LOCATION_COLLAPSE_WARNED:
            continue
        _LOCATION_COLLAPSE_WARNED.add(key)
        msg = (
            f"Location field '{field}' is being used as a geographic grouping key "
            f"for {context}, but it was cardinality-collapsed via "
            f"data.process.collapse_sparse_categories. Collapsing a location merges "
            f"unrelated zones into one replacement bucket (e.g. 'Other'), which "
            f"corrupts {context}. Best practice: collapse into a separate "
            f"'{field}_collapsed' modeling variant (set 'output_field' on the collapse "
            f"config) and use that ONLY as a model feature, leaving '{field}' intact "
            f"as the location."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)


#######################################
# PRIVATE
#######################################



def _apply_dd_to_df_cols(
    df: pd.DataFrame,
    settings: dict,
    one_hot_descendants: dict = None,
    dd_field: str = "name",
) -> pd.DataFrame:
    dd = settings.get("data_dictionary", {})

    rename_map = {}
    for column in df.columns:
        rename_map[column] = dd.get(column, {}).get(dd_field, column)

    if one_hot_descendants is not None:
        for ancestor in one_hot_descendants:
            descendants = one_hot_descendants[ancestor]
            for descendant in descendants:
                rename_map[descendant] = (
                    dd.get(ancestor, {}).get(dd_field, ancestor)
                    + " = "
                    + descendant[len(ancestor) + 1 :]
                )

    df = df.rename(columns=rename_map)
    return df


def _apply_dd_to_df_rows(
    df: pd.DataFrame,
    column: str,
    settings: dict,
    one_hot_descendants: dict = None,
    dd_field: str = "name",
) -> pd.DataFrame:
    dd = settings.get("data_dictionary", {})

    df[column] = df[column].map(lambda x: dd.get(x, {}).get(dd_field, x))
    if one_hot_descendants is not None:
        one_hot_rename_map = {}
        for ancestor in one_hot_descendants:
            descendants = one_hot_descendants[ancestor]
            for descendant in descendants:
                one_hot_rename_map[descendant] = (
                    dd.get(ancestor, {}).get(dd_field, ancestor)
                    + " = "
                    + descendant[len(ancestor) + 1 :]
                )
        df[column] = df[column].map(lambda x: one_hot_rename_map.get(x, x))
    return df


def _get_unclassified_fields(s: dict, df: pd.DataFrame = None):
    # Get all fields that are not classified as categorical, numeric, or boolean
    all = []
    for t in ["land", "impr", "other"]:
        cats = s.get("field_classification", {}).get(t, {}).get("categorical", [])
        nums = s.get("field_classification", {}).get(t, {}).get("numeric", [])
        bools = s.get("field_classification", {}).get(t, {}).get("boolean", [])
        all += cats + nums + bools

    if df is not None:
        all = [f for f in all if f in df]
        for col in df:
            if col not in all:
                all.append(col)

    return all


def _get_fields(s: dict, type: str, df: pd.DataFrame = None) -> dict:
    cats = s.get("field_classification", {}).get(type, {}).get("categorical", [])
    nums = s.get("field_classification", {}).get(type, {}).get("numeric", [])
    bools = s.get("field_classification", {}).get(type, {}).get("boolean", [])

    if df is not None:
        cats = [c for c in cats if c in df]
        nums = [n for n in nums if n in df]
        bools = [b for b in bools if b in df]

    return {"categorical": cats, "numeric": nums, "boolean": bools}


def _get_base_dir(s: dict) -> str:
    slug: str|None = s.get("locality", {}).get("slug", None)
    if slug is None:
        raise ValueError("Could not find settings.locality.slug!")
    return slug


def _process_settings(settings: dict):
    s = settings.copy()

    # Step 1: remove any and all keys that are prefixed with the string "__":
    s = _remove_comments_from_settings(s)

    # Step 2: do variable replacement:
    s = _replace_variables(s)

    return s


def _remove_comments_from_settings(s: dict) -> dict:
    comment_token = "__"
    keys_to_remove = []
    for key in s:
        entry = s[key]
        if key.startswith(comment_token):
            keys_to_remove.append(key)
        elif isinstance(entry, dict):
            s[key] = _remove_comments_from_settings(entry)
    for k in keys_to_remove:
        del s[k]
    return s


def _replace_variables(settings: dict) -> dict:

    result = settings.copy()
    failsafe = 999
    changes = 1

    while changes > 0 and failsafe > 0:
        result, changes = _do_replace_variables(result, settings)
        failsafe -= 1

    return result


def _do_replace_variables(
    node: dict | list | str, settings: dict, var_token: str = "$$"
) -> tuple[dict|list|str, int]:
    # For each key-value pair, search for values that are strings prefixed with $$, and replace them accordingly

    changes = 0
    replacement = node

    if isinstance(node, str):
        # Case 1 -- node is string
        str_value = str(node)
        if str_value.startswith(var_token):
            var_name = str_value[len(var_token) :]
            var_value = _lookup_variable_in_settings(settings, var_name)
            replacement = var_value
            if replacement is None:
                raise ValueError(f"Variable {var_name} not found in settings!")
            changes += 1

    elif isinstance(node, dict):
        # Case 2 -- node is a dict
        _replacements = {}
        for key in node:
            entry = node[key]
            replacement, _changes = _do_replace_variables(entry, settings, var_token)
            if _changes > 0:
                _replacements[key] = replacement
                changes += _changes
        if changes > 0:
            for key in _replacements:
                node[key] = _replacements[key]
        replacement = node

    elif isinstance(node, list):
        # Case 3 -- node is a list. Go through each entry in the list.
        _replacements = {}
        for i, entry in enumerate(node):
            replacement, _changes = _do_replace_variables(entry, settings, var_token)
            if _changes > 0:
                _replacements[i] = replacement
                changes += _changes
        if changes > 0:
            for i in _replacements:
                node[i] = _replacements[i]
        replacement = node

    return replacement, changes


def _lookup_variable_in_settings(s: dict, var_name: str, path: list[str] = None):
    if path is None:
        # no path is provided, but the variable name exists:
        # split it by periods, if it has any
        path = var_name.split(".")

    if path is not None and len(path) > 0:
        first_bit = path[0]
        if first_bit in s:
            if len(path) == 1:
                # this is the last bit of the path
                return s[first_bit]
            else:
                return _lookup_variable_in_settings(s[first_bit], "", path[1:])

    return None


def _load_data_dictionary_template():
    with importlib.resources.open_text(
        "openavmkit.resources.settings", f"data_dictionary.json", encoding="utf-8"
    ) as file:
        data_dictionary = json.load(file)
    return data_dictionary


def _load_settings_template():
    with importlib.resources.open_text(
        "openavmkit.resources.settings", f"settings.template.json", encoding="utf-8"
    ) as file:
        settings = json.load(file)
    return settings


def _is_key_in(object: dict, key: str) -> tuple[bool, str]:
    flags = ["+", "!"]
    for flag in ["+", "!", ""]:
        if f"{flag}{key}" in object:
            return True, flag
    return False, ""


def _strip_flags(settings: dict | list) -> dict | list:
    flags = ["+", "!"]

    if isinstance(settings, list):
        for i, item in enumerate(settings):
            if isinstance(item, list) or isinstance(item, dict):
                settings[i] = _strip_flags(item)
        return settings

    if isinstance(settings, dict):
        keys_in_settings = [key for key in settings]

        for key_ in keys_in_settings:
            if key_ not in settings:
                continue
            entry = settings[key_]
            key = key_
            for flag in flags:
                if key_.startswith(flag):
                    key = key_[1:]
                    settings[key] = settings[key_]
                    del settings[key_]
            if isinstance(entry, dict):
                entry = _strip_flags(entry)
                settings[key] = entry
            elif isinstance(entry, list):
                for i, item in enumerate(entry):
                    if isinstance(item, list) or isinstance(item, dict):
                        entry[i] = _strip_flags(item)
            settings[key] = entry
    return settings


def _merge_settings(template: dict, local: dict, indent: str = ""):
    # Start by copying the template
    merged = template.copy()

    # Iterate over keys of local:
    for key_ in local:

        key = key_
        local_stomps = False
        if key_.startswith("!"):
            local_stomps = True
            key = key_[1:]

        entry_l = local[key_]

        key_exists, flag = _is_key_in(template, key)

        # If the key is in both template and local, reconcile them:
        if key_exists:
            local_key = f"{flag}{key}"
            add_template = False
            if not local_stomps and flag == "+":
                add_template = True

            if local_stomps:
                merged[key] = entry_l
            else:
                entry_t = template[local_key]
                if isinstance(entry_t, dict) and isinstance(entry_l, dict):
                    # If both are dictionaries, merge them recursively:
                    merged[key] = _merge_settings(entry_t, entry_l, indent + "  ")
                elif isinstance(entry_t, list) and isinstance(entry_l, list):
                    if add_template:
                        # If both are lists, add any new local items that aren't already in template:
                        for item in entry_l:
                            if item not in entry_t:
                                entry_t.append(item)
                        merged[key] = entry_t
                    else:
                        merged[key] = entry_l
                else:
                    merged[key] = entry_l

            if flag != "" and local_key in merged:
                del merged[local_key]

        else:
            merged[key] = entry_l

    merged = _strip_flags(merged)

    return merged


def _get_sales(
    df_in: pd.DataFrame,
    settings: dict,
    vacant_only: bool = False,
    df_univ: pd.DataFrame = None,
) -> pd.DataFrame:
    """Retrieve valid sales from the input DataFrame. Also simulates removed buildings if
    applicable.

    Filters for sales with a positive sale price, valid_sale marked True. If vacant_only
    is True, only includes rows where vacant_sale is True.
    """
    df = df_in.copy().reset_index(drop=True)

    if "vacant_sale" in df.columns:
        # check for vacant sales:
        idx_vacant_sale = df["vacant_sale"].eq(True)

        # simulate removed buildings for vacant sales
        # (if we KNOW it was a vacant sale, then the building characteristics have to go)
        df = _simulate_removed_buildings(df, settings, idx_vacant_sale)

        # TODO: smell
        if "is_vacant" not in df.columns and df_univ is not None:
            df = df.merge(df_univ[["key", "is_vacant"]], on="key", how="left")

        if "model_group" not in df.columns and df_univ is not None:
            df = df.merge(df_univ[["key", "model_group"]], on="key", how="left")

        # if a property was NOT vacant at time of sale, but is vacant now, then the sale is invalid:
        idx_is_vacant = df["is_vacant"].eq(True)
        df.loc[~idx_vacant_sale & idx_is_vacant, "valid_sale"] = False
        

    # Use sale_price_time_adj if it exists, otherwise use sale_price
    sale_field = "sale_price_time_adj" if "sale_price_time_adj" in df.columns and len(df["sale_price_time_adj"].dropna()) > 0 else "sale_price"
    idx_positive_sale_price = df[sale_field].gt(0)
    
    
    idx_valid_sale = df["valid_sale"].eq(True)
    idx_vacant_sale = df["vacant_sale"].eq(True)
    
    
    if vacant_only:
        idx_all = idx_positive_sale_price & idx_valid_sale & idx_vacant_sale
    else:
        idx_all = idx_positive_sale_price & idx_valid_sale
    

    df_sales: pd.DataFrame = df[idx_all].copy()

    return df_sales


def _is_series_all_bools(series: pd.Series) -> bool:
    dtype = series.dtype
    if dtype == bool:
        return True
    # Also accept pandas' nullable BooleanDtype and any other bool dtype
    # variants. Earlier this function compared ``type(unique)`` to the built-in
    # ``bool``, which rejected ``np.bool_`` and ``pandas.BooleanDtype`` arrays
    # even though their values are unambiguous booleans — which broke
    # cleaning whenever a source merged in vacant_sale as a nullable boolean.
    if pd.api.types.is_bool_dtype(dtype):
        return True
    import numpy as _np
    for unique in series.unique():
        if pd.isna(unique):
            continue
        if isinstance(unique, (bool, _np.bool_)):
            continue
        return False
    return True


def _get_max_ratio_study_trim(settings: dict, model_group: str)->float:
    trim = settings.get("analysis",{}).get("ratio_study",{}).get("trim",{})
    entry = trim.get(model_group, trim.get("default", {}))
    return entry.get("max_percent", 0.1)


def _simulate_removed_buildings(
    df: pd.DataFrame, settings: dict, idx_vacant: pd.Series = None
) -> pd.DataFrame:
    """Simulate removed buildings by changing improvement fields to values that reflect
    the absence of a building.

    For all improvement fields, fills categorical fields with "UNKNOWN", numeric fields
    with 0, and boolean fields with False for the rows specified by idx_vacant (or all
    rows if idx_vacant is None).
    """
    if idx_vacant is None:
        # do the whole thing:
        idx_vacant = df.index

    fields_impr = get_fields_impr(settings, df)
    
    # fill unknown values for categorical improvements:
    fields_impr_cat = fields_impr["categorical"]
    fields_impr_num = fields_impr["numeric"]
    fields_impr_bool = fields_impr["boolean"]

    for field in fields_impr_cat:
        if not hasattr(df[field].dtype, 'categories'):
            df[field] = df[field].astype("category")
        # add UNKNOWN if needed
        if "UNKNOWN" not in df[field].cat.categories:
            df[field] = df[field].cat.add_categories(["UNKNOWN"])

    for field in fields_impr_cat:
        df.loc[idx_vacant, field] = "UNKNOWN"

    for field in fields_impr_num:
        df.loc[idx_vacant, field] = 0.0

    for field in fields_impr_bool:
        # Convert to boolean type first if needed
        if df[field].dtype != bool:
            df[field] = df[field].astype(bool)
        df.loc[idx_vacant, field] = False

    unit = area_unit(settings)
    # just to be safe, ensure that the "bldg_area_finished_{unit}" field is set to 0 for vacant sales
    # and update "is_vacant" to perfectly match
    # TODO: if we add support for a custom vacancy filter, we will need to adjust this
    if f"bldg_area_finished_{unit}" in df:
        df.loc[idx_vacant, f"bldg_area_finished_{unit}"] = 0
        # Convert is_vacant to boolean first
        if "is_vacant" not in df or df["is_vacant"].dtype != bool:
            df["is_vacant"] = False
        df.loc[idx_vacant, "is_vacant"] = True

    return df


def get_dupes(entry: dict, df: pd.DataFrame = None, is_geometry: bool = False):
    dupes = entry.get("dupes", None)
    dupes_was_none = dupes is None
    if dupes is None:
        if is_geometry:
            dupes = "auto"
        else:
            dupes = {}
    if dupes == "auto":
        if df is not None:
            if is_geometry:
                cols = [col for col in df.columns.values if col != "geometry"]
                col = cols[0]
                dupes = {"subset": [col], "sort_by": [col, "asc"], "drop": True}
                if dupes_was_none:
                    warnings.warn(
                        f"'dupes' not found, defaulting to \"{col}\" as de-dedupe key. Set 'dupes:\"auto\" to remove this warning.'"
                    )
            else:
                keys = ["key_sale", "key", "key2", "key3"]
                matched = False
                for key in keys:
                    if key in df:
                        dupes = {"subset": [key], "sort_by": [key, "asc"], "drop": True}
                        matched = True
                        break
                if not matched:
                    # Reference tables and other auxiliary loads don't have a canonical
                    # key column. Fall back to the first column.
                    cols = list(df.columns.values)
                    if cols:
                        col = cols[0]
                        dupes = {"subset": [col], "sort_by": [col, "asc"], "drop": True}
                        if dupes_was_none:
                            warnings.warn(
                                f"'dupes' not found and no canonical key column "
                                f"({', '.join(keys)}) present; defaulting to "
                                f"\"{col}\" as de-dupe key. Set dupes explicitly to "
                                f"silence this warning."
                            )
                    else:
                        dupes = {"subset": ["key"], "sort_by": ["key", "asc"], "drop": True}
        else:
            dupes = {"subset": ["key"], "sort_by": ["key", "asc"], "drop": True}
    elif dupes == "allow":
        # Explicit "keep all rows" signal. Must be distinct from {} (the no-dupes-specified
        # default), which means "de-dupe on key". Without this flag both collapse to {} and a
        # keyed source declared dupes:"allow" would be silently de-duplicated on key.
        dupes = {"allow": True}
    return dupes