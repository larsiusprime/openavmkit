"""
Data cleaning and missing-value handling.

Operates on a :class:`openavmkit.data.SalesUniversePair` to produce a
modeling-ready dataset. Responsibilities include:

- Filling missing values per the rules under ``data.process.fill.*`` in
  ``settings.json`` (see :doc:`/advanced_settings` for the full method
  reference: ``zero``, ``unknown``, ``none``, ``false``, ``mode``, ``median``,
  ``mean``, ``max``, ``min``, ``custom``, plus ``_impr`` / ``_vacant`` suffixes).
- Reconciling year-built / age-years pairs against the valuation date.
- Auto-filling residual categorical and boolean fields.
- Validating sales arms-length-ness when ``data.validation.enabled = true``.
- Cleaning and filtering invalid sales.

Public entry points are surfaced through :mod:`openavmkit.pipeline`.
"""
from warnings import warn

import pandas as pd

from openavmkit.data import SalesUniversePair, get_hydrated_sales_from_sup
from openavmkit.utilities.data import ensure_categories, align_categories
from openavmkit.utilities.settings import (
  get_valuation_date,
  get_fields_categorical,
  get_fields_boolean,
  get_grouped_fields_from_data_dictionary,
  get_data_dictionary,
  get_model_group_ids,
  get_collapse_sparse_categories_config,
  get_collapsed_fields,
  get_location_fields,
  is_collapse_strict,
  _is_series_all_bools,
)
from openavmkit.utilities.cache import write_cache
from openavmkit.calculations import resolve_filter, perform_calculations


def clean_valid_sales(sup: SalesUniversePair, settings: dict) -> SalesUniversePair:
    """Clean and validate sales data in the SalesUniversePair.

    This function processes the sales data to ensure that only valid sales are retained.
    It also ensures that the sales data is consistent with the universe data, particularly regarding
    the vacancy status of parcels. Invalid sales are scrubbed of their metadata, and valid sales are
    properly classified for ratio studies.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    settings : dict
        The settings dictionary containing configuration for the cleaning process.

    Returns
    -------
    SalesUniversePair
        The updated SalesUniversePair with cleaned and validated sales data.
    """
    # load metadata
    val_date = get_valuation_date(settings)
    val_year = val_date.year
    # Use the FLOOR (widest window any model group needs), not a per-group window: this
    # stage permanently drops too-old sales and runs before the per-group train/test
    # split, so dropping to a group's narrower window here would starve a longer-reach
    # group (e.g. commercial). Per-group narrowing happens later in get_data_split_for.
    from openavmkit.utilities.settings import use_sales_from_floor
    use_sales_from_impr, use_sales_from_vacant = use_sales_from_floor(settings)
    if use_sales_from_impr is None:
        use_sales_from_impr = val_year - 5
    if use_sales_from_vacant is None:
        use_sales_from_vacant = val_year - 5

    df_sales = sup["sales"].copy()
    df_univ = sup["universe"]

    # temporarily merge in universe's vacancy status (how the parcel is now)
    df_univ_vacant = (
        df_univ[["key", "is_vacant"]]
        .copy()
        .rename(columns={"is_vacant": "univ_is_vacant"})
    )

    # check df_univ for duplicate keys:
    if len(df_univ["key"].unique()) != len(df_univ):
        print("WARNING: df_univ has duplicate keys, this will cause problems")
        # print how many:
        dupe_key_count = len(df_univ) - len(df_univ["key"].unique())
        print(f"--> {dupe_key_count} rows with duplicate keys found")

    print(f"Before univ merge len = {len(df_sales)}")

    df_sales = df_sales.merge(df_univ_vacant, on="key", how="left")

    print(f"After univ merge len = {len(df_sales)}")
    
    # Apply per-type sale-age thresholds. ``use_sales_from`` can be either an
    # int (same cutoff for both) or a dict ``{improved: ..., vacant: ...}`` —
    # the latter lets a jurisdiction keep its improved-sale ratio-study window
    # tight while still allowing older vacant/teardown sales into the land flow.
    df_sales.loc[
        df_sales["sale_year"].lt(use_sales_from_impr)
        & df_sales["vacant_sale"].eq(False),
        "valid_sale",
    ] = False
    df_sales.loc[
        df_sales["sale_year"].lt(use_sales_from_vacant)
        & df_sales["vacant_sale"].eq(True),
        "valid_sale",
    ] = False

    # sale prices of 0 and negative and null are invalid
    df_sales.loc[
        df_sales["sale_price"].isna() | df_sales["sale_price"].le(0), "valid_sale"
    ] = False

    # scrub sales info from invalid sales
    idx_invalid = df_sales["valid_sale"].eq(False)
    fields_to_scrub = [
        "sale_date",
        "sale_price",
        "sale_year",
        "sale_month",
        "sale_day",
        "sale_quarter",
        "sale_year_quarter",
        "sale_year_month",
        "sale_age_days",
        "sale_price_per_land_sqft",
        "sale_price_per_land_sqm",
        "sale_price_per_impr_sqft",
        "sale_price_per_impr_sqm",
        "sale_price_time_adj",
        "sale_price_time_adj_per_land_sqft",
        "sale_price_time_adj_per_land_sqm",
        "sale_price_time_adj_per_impr_sqft",
        "sale_price_time_adj_per_impr_sqm",
    ]

    for field in fields_to_scrub:
        if field in df_sales:
            df_sales.loc[idx_invalid, field] = None

    # drop all invalid sales:
    df_sales = df_sales[df_sales["valid_sale"].eq(True)].copy()

    # initialize these -- we want to further determine which valid sales are valid for ratio studies
    df_sales["valid_for_ratio_study"] = False
    df_sales["valid_for_land_ratio_study"] = False

    # NORMAL RATIO STUDIES:
    # If it's a valid sale, and its vacancy status matches its status at time of sale, it's valid for a ratio study
    # This is because how it looked at time of sale matches how it looks now, so the prediction is comparable to the sale
    # If the vacancy status has changed since it sold, we can't meaningfully compare sale price to current valuation
    df_sales.loc[
        df_sales["valid_sale"] & df_sales["vacant_sale"].eq(df_sales["univ_is_vacant"]),
        "valid_for_ratio_study",
    ] = True

    # LAND RATIO STUDIES:
    # If it's a valid sale, and it was vacant at time of sale, it's valid for a LAND ratio study regardless of whether it
    # is valid for a normal ratio study. That's because we will come up with a land value prediction no matter what, and
    # we can always compare that to what it sold for, as long as it was vacant at time of sale
    # we can always compare that to what it sold for, as long as it was vacant at time of sale
    df_sales.loc[
        df_sales["valid_sale"] & df_sales["vacant_sale"].eq(True),
        "valid_for_land_ratio_study",
    ] = True

    print(f"Using {len(df_sales[df_sales['valid_sale'].eq(True)])} sales...")
    print(f"--> {len(df_sales[df_sales['vacant_sale'].eq(True)])} vacant sales")
    print(f"--> {len(df_sales[df_sales['vacant_sale'].eq(False)])} improved sales")
    print(
        f"--> {len(df_sales[df_sales['valid_for_ratio_study'].eq(True)])} valid for ratio study"
    )
    print(
        f"--> {len(df_sales[df_sales['valid_for_land_ratio_study'].eq(True)])} valid for land ratio study"
    )

    # We need to ensure that the flag "is_vacant" is valid to train on
    # So in sales it needs to reflect the sale's vacant status
    # When hydrating, this will stomp the universe's vacant status, which is exactly what we want in a training set
    # Meanwhile, during prediction, it will infer based on the universe's vacant status
    df_sales["is_vacant"] = df_sales["vacant_sale"]

    df_sales = df_sales.drop(columns=["univ_is_vacant"])

    # enforce some booleans:
    bool_fields = [
        "valid_sale",
        "vacant_sale",
        "valid_for_ratio_study",
        "valid_for_land_ratio_study",
    ]
    for b in bool_fields:
        if b in df_sales:
            dtype = df_sales[b].dtype
            if dtype != bool:
                if _is_series_all_bools(df_sales[b]):
                    df_sales[b] = df_sales[b].astype(bool)
                else:
                    raise ValueError(
                        f"Field '{b}' contains non-boolean values that cannot be coerced to boolean. Unique values = {df_sales[b].unique()}"
                    )

    sup.update_sales(df_sales, allow_remove_rows=True)

    return sup


def collapse_sparse_categories_sup(
    sup: SalesUniversePair, settings: dict
) -> SalesUniversePair:
    """Collapse rare categorical values into a per-field replacement bucket.

    For each field configured under ``data.process.collapse_sparse_categories``,
    any category whose row count falls below ``sales_min`` in the hydrated sales
    set OR below ``univ_min`` in the universe set is replaced by the configured
    ``replacement_value`` (default ``"Other"``). The same mapping is applied
    to both the sales and universe DataFrames so the model and downstream
    artifacts see a single consistent vocabulary.

    If fewer than two categories would be collapsed for a field, the field is
    left untouched — renaming a single category to ``"Other"`` would not buy
    any generalization benefit and would only mask the original label.

    Per-field config keys: ``sales_min`` and ``univ_min`` (required),
    ``replacement_value`` (optional, default ``"Other"``), and ``output_field``
    (optional). When ``output_field`` is set, the collapsed result is written to
    that **new** column and the source field is left untouched — this is the
    recommended way to make a cardinality-reduced *modeling variant* of a
    **location** field (e.g. ``neighborhood`` -> ``neighborhood_collapsed``)
    without corrupting the raw location used for breakdowns/equity/spatial work.
    The variant column is always created (a copy of the source) even when nothing
    collapses, so downstream references to it never break.

    A reserved boolean key ``strict`` (not a field) turns the location-collapse
    guard from a warning into a ``ValueError``.

    **Location footgun:** collapsing a field *in place* that is used as a coherent
    geographic location (see :func:`get_location_fields`) merges unrelated zones
    into the replacement bucket. This function warns loudly (or raises when
    ``strict``) in that case. Use ``output_field`` instead.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair to modify.
    settings : dict
        The settings dictionary. Reads ``data.process.collapse_sparse_categories``.

    Returns
    -------
    SalesUniversePair
        The updated SalesUniversePair with sparse categories collapsed.

    Raises
    ------
    ValueError
        If a configured field is missing ``sales_min`` or ``univ_min``, if the
        field is not listed in ``field_classification.*.categorical``, or if a
        location field is collapsed in place while ``strict`` is set.
    """
    config = get_collapse_sparse_categories_config(settings)
    if not config:
        return sup

    known_categoricals = set(get_fields_categorical(settings, include_boolean=False))

    # Upfront location-collapse guard: a field collapsed *in place* (no
    # output_field) whose result column is used as a coherent location is almost
    # always a mistake. Warn once per offender here (or raise when strict).
    location_fields = get_location_fields(settings)
    strict = is_collapse_strict(settings)
    for field, field_cfg in config.items():
        if not isinstance(field_cfg, dict):
            continue
        result_col = field_cfg.get("output_field", field)
        if result_col in location_fields:
            msg = (
                f"collapse_sparse_categories is collapsing '{result_col}', which is "
                f"used as a coherent geographic location (in field_classification."
                f"important.locations, analysis.*.location, ratio-study breakdowns, "
                f"and/or model/ensemble locations). Collapsing a location merges "
                f"unrelated zones into '{field_cfg.get('replacement_value', 'Other')}', "
                f"corrupting equity clustering, ratio-study breakdowns, local-ensemble "
                f"selection, and sales-scrutiny clusters. Best practice: set "
                f"'output_field' (e.g. '{field}_collapsed') so the collapse writes a "
                f"separate modeling variant and leaves '{field}' intact as the location, "
                f"then use the variant ONLY as a model feature."
            )
            if strict:
                raise ValueError(msg)
            warn(msg)

    df_sales = sup["sales"].copy()
    df_univ = sup["universe"].copy()
    df_sales_hydrated = get_hydrated_sales_from_sup(sup)

    for field, field_cfg in config.items():
        if not isinstance(field_cfg, dict):
            # Reserved scalar keys (e.g. "strict") are not field configs.
            continue
        if "sales_min" not in field_cfg or "univ_min" not in field_cfg:
            raise ValueError(
                f"collapse_sparse_categories['{field}'] requires both "
                f"'sales_min' and 'univ_min' to be set."
            )
        if field not in known_categoricals:
            raise ValueError(
                f"collapse_sparse_categories['{field}']: field is not "
                f"declared in field_classification.*.categorical. Add it "
                f"there or remove it from collapse_sparse_categories."
            )

        sales_min = field_cfg["sales_min"]
        univ_min = field_cfg["univ_min"]
        replacement_value = field_cfg.get("replacement_value", "Other")
        output_field = field_cfg.get("output_field")
        target = output_field or field

        if field not in df_univ.columns:
            warn(
                f"collapse_sparse_categories: field '{field}' is not in the "
                f"universe DataFrame, skipping."
            )
            continue

        # When writing to a separate variant, seed it as a copy of the source up
        # front so the column always exists for downstream references — even if
        # too few categories collapse below.
        if output_field:
            df_univ[output_field] = df_univ[field]
            if field in df_sales.columns:
                df_sales[output_field] = df_sales[field]

        sales_counts = (
            df_sales_hydrated[field].value_counts(dropna=False)
            if field in df_sales_hydrated.columns
            else pd.Series(dtype="int64")
        )
        univ_counts = df_univ[field].value_counts(dropna=False)

        all_values = set(sales_counts.index).union(set(univ_counts.index))
        # Skip NA/NaN. Missing values are a fill concern, not a rare category to
        # bucket into the replacement value, and including pd.NA here makes the
        # sorted() comparison raise "boolean value of NA is ambiguous".
        sparse_values = sorted(
            v
            for v in all_values
            if not pd.isna(v)
            and (sales_counts.get(v, 0) < sales_min or univ_counts.get(v, 0) < univ_min)
        )

        if len(sparse_values) < 2:
            if len(sparse_values) == 1:
                print(
                    f"collapse_sparse_categories: {field} — only 1 sparse "
                    f"category ('{sparse_values[0]}'), leaving as-is"
                )
            continue

        sales_rows_affected = int(
            sum(sales_counts.get(v, 0) for v in sparse_values)
        )
        univ_rows_affected = int(
            sum(univ_counts.get(v, 0) for v in sparse_values)
        )

        into = f" into '{output_field}'" if output_field else ""
        print(f"collapse_sparse_categories: {field}{into}")
        print(f"  thresholds: sales_min={sales_min}, univ_min={univ_min}")
        print(
            f"  {len(sparse_values)} categories collapsed into "
            f"'{replacement_value}' ({sales_rows_affected} sales rows, "
            f"{univ_rows_affected} universe rows affected)"
        )
        print(f"  collapsed: {sparse_values}")

        df_univ = _apply_collapse_mapping(
            df_univ, target, sparse_values, replacement_value
        )
        if target in df_sales.columns:
            df_sales = _apply_collapse_mapping(
                df_sales, target, sparse_values, replacement_value
            )

    sup.set("sales", df_sales)
    sup.set("universe", df_univ)
    return sup


def _apply_collapse_mapping(
    df: pd.DataFrame, field: str, sparse_values: list, replacement_value: str
) -> pd.DataFrame:
    """Replace ``sparse_values`` in ``df[field]`` with ``replacement_value``.

    Handles both object and pandas Categorical dtypes. For Categorical
    columns, the new category is added before the rename and unused
    categories are pruned afterward.
    """
    series = df[field]
    if isinstance(series.dtype, pd.CategoricalDtype):
        if replacement_value not in series.cat.categories:
            series = series.cat.add_categories([replacement_value])
        mask = series.isin(sparse_values)
        series = series.where(~mask, replacement_value)
        series = series.cat.remove_unused_categories()
    else:
        series = series.replace({v: replacement_value for v in sparse_values})
    df[field] = series
    return df


def fill_unknown_values_sup(
    sup: SalesUniversePair, settings: dict
) -> SalesUniversePair:
    """Fill unknown values with default values as specified in settings.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    settings : dict
        The settings dictionary containing configuration for filling unknown values.

    Returns
    -------
    SalesUniversePair
        The updated SalesUniversePair with filled unknown values.
    """
    df_sales = sup["sales"].copy()
    df_univ = sup["universe"].copy()

    # Fill ALL unknown values for the universe
    df_univ = _fill_unknown_values_per_model_group(df_univ, settings)

    # For sales, fill ONLY the unknown values that pertain to sales metadata
    # df_sales can contain characteristics, but we want to preserve the blanks in those fields because they function
    # as overlays on top of the universe data
    dd = get_data_dictionary(settings)
    sale_fields = get_grouped_fields_from_data_dictionary(dd, "sale")
    sale_fields = [field for field in sale_fields if field in df_sales]

    df_sales_subset = df_sales[sale_fields].copy()
    df_sales_subset = _fill_unknown_values(df_sales_subset, settings)
    for col in df_sales_subset:
        df_sales[col] = df_sales_subset[col]

    sup.set("sales", df_sales)
    sup.set("universe", df_univ)

    return sup


def filter_invalid_sales(
    sup: SalesUniversePair, settings: dict, verbose: bool = False
) -> SalesUniversePair:
    """Validate arms-length sales based on configurable filter conditions.

    Parameters
    ----------
    sup : SalesUniversePair
        The SalesUniversePair containing sales and universe data.
    settings : dict
        The settings dictionary containing configuration for arms-length validation.
    verbose : bool, optional
        If True, prints detailed information about the validation process. Default is False.

    Returns
    -------
    SalesUniversePair
        The updated SalesUniversePair with arms-length validation applied to sales data.
    """
    s_data = settings.get("data", {})
    s_process = s_data.get("process", {})
    s_validation = s_process.get("invalid_sales", {})
    
    if not s_validation.get("enabled", False):
        if verbose:
            print("Invalid sales validation filter disabled, skipping...")
        return sup
        
    if verbose:
        print("Filtering out invalid sales...")

    # Get sales data (hydrated: universe fields such as assr_market_value are merged on,
    # so the filter / calc below can reference them alongside the raw sale fields).
    df_sales = get_hydrated_sales_from_sup(sup)
    total_sales = len(df_sales)
    excluded_sales = []
    total_excluded = 0

    # Optional derived fields for the filter. The filter DSL compares a field to a scalar
    # or another field but cannot do inline arithmetic, so relative rules
    # (e.g. "sale_price < 0.5 * assr_market_value") need a precomputed ratio column. Compute
    # any `invalid_sales.calc` entries on the hydrated frame before resolving the filter.
    s_calc = s_validation.get("calc", {})
    if s_calc:
        df_sales = perform_calculations(df_sales, s_calc)

    # Identify sales by filter
    filter_conditions = s_validation.get("filter", [])
    if s_validation.get("enabled", False):
        if verbose:
            print("\nApplying filter method...")

        # Get filter conditions from settings
        if not filter_conditions:
            raise ValueError("No filter conditions defined in settings")

        # Resolve filter using standard filter resolution
        filter_mask = resolve_filter(df_sales, filter_conditions)

        # Get keys of filtered sales
        filtered_keys = df_sales[filter_mask]["key_sale"].tolist()
        if filtered_keys:
            excluded_info = {
                "method": "filter",
                "key_sales": filtered_keys,
                "total_sales": total_sales,
                "excluded": len(filtered_keys),
                "conditions": filter_conditions,
            }
            excluded_sales.append(excluded_info)
            total_excluded += len(filtered_keys)

            # Mark these sales as invalid
            df_sales.loc[df_sales["key_sale"].isin(filtered_keys), "valid_sale"] = False

            if verbose:
                print(f"--> Found {len(filtered_keys)} sales excluded by filter method")

    if verbose:
        print(f"\nOverall summary:")
        print(f"--> Total sales processed: {total_sales}")
        print(
            f"--> Total sales excluded: {total_excluded} ({total_excluded/total_sales*100:.1f}%)"
        )

    # Cache the excluded sales info
    if excluded_sales:
        cache_data = {
            "excluded_sales": excluded_sales,
            "total_sales": total_sales,
            "total_excluded": total_excluded,
            "settings": s_validation,
        }
        write_cache("arms_length_validation", cache_data, cache_data, "dict")

    # Filter out invalid sales
    df_sales = df_sales[df_sales["valid_sale"].eq(True)].copy()
    
    # Update the SalesUniversePair to match
    sup.limit_sales_to_keys(df_sales["key_sale"].values)
    
    return sup


#######################################
# PRIVATE
#######################################


def _fill_with(df: pd.DataFrame, field: str, value):
    if field not in df:
        return df

    if hasattr(df[field].dtype, 'categories'):  # Categorical-like dtype
        if value not in df[field].cat.categories:
            df[field] = df[field].cat.add_categories(value)

    df.loc[df[field].isna(), field] = value
    return df


def _fill_custom(df: pd.DataFrame, entry: dict):
    field = entry.get("field")
    value = entry.get("value")
    return _fill_with(df, field, value)


def _fill_thing(df: pd.DataFrame, field: str | dict, fill_method: str):
    if fill_method == "custom":
        if isinstance(field, dict):
            df = _fill_custom(df, field)
        else:
            raise ValueError("Entry must be a dictionary when fill_method is 'custom'")
    if fill_method == "zero":
        df = _fill_with(df, field, 0)
    elif fill_method == "unknown":
        df = _fill_with(df, field, "UNKNOWN")
    elif fill_method == "none":
        df = _fill_with(df, field, "NONE")
    elif fill_method == "false":
        if "str" in str(df[field].dtype).lower():
            df = _fill_with(df, field, "False")
        else:
            df = _fill_with(df, field, False)
    elif fill_method == "mode":
        modal_values = df[~df[field].isna()][field].mode()
        if len(modal_values) > 0:
            modal_value = modal_values.iloc[0]
        else:
            # Rare edge case -- there's NO non-null modal value. Default to 0/unknown depending on dtype.
            dtype_str = str(df[field].dtype).lower()
            if "int" in dtype_str or "float" in dtype_str:
                modal_value = 0
            else:
                modal_value = "UNKNOWN"
        df = _fill_with(df, field, modal_value)
    elif fill_method == "median":
        df = _fill_with(df, field, df[~df[field].isna()][field].median())
    elif fill_method == "mean":
        df = _fill_with(df, field, df[~df[field].isna()][field].mean())
    elif fill_method == "max":
        df = _fill_with(df, field, df[~df[field].isna()][field].max())
    elif fill_method == "min":
        df = _fill_with(df, field, df[~df[field].isna()][field].min())
    return df


def _fill_unknown_values_per_model_group(df_in: pd.DataFrame, settings: dict):
    df = df_in.copy()
    model_groups = get_model_group_ids(settings, df)

    # TODO: this is a hacky one off, probably need a more systemic way to handle cases where we need to hit all rows no matter what
    model_groups.append(None)
    model_groups.append("UNKNOWN")
    model_groups = list(set(model_groups))

    for model_group in model_groups:
        if model_group is None:
            df_mg = df[pd.isna(df["model_group"])].copy()
        else:
            df_mg = df[df["model_group"].eq(model_group)].copy()
        df_mg = _fill_unknown_values(df_mg, settings)
        df, df_mg = align_categories(df, df_mg)
        try:
            df.loc[df_mg.index, :] = df_mg
        except:
            warn(f"model_group: {model_group}, len: {len(df_mg)}")
            warn("Column type mismatch. You may be lacking a field type definition. The problem is likely in one of the following fields.")
            for col in df_mg.columns:
                if df_mg[col].dtype != df[col].dtype:
                    warn(f"{col}: {df_mg[col].dtype}, {df[col].dtype}")
            raise

    return df


def _fill_unknown_values(df, settings: dict):
    fills = settings.get("data", {}).get("process", {}).get("fill", {})

    for key in fills:
        fill_list = fills[key]
        for field in fill_list:
            field_name = field
            if type(field) is dict:
                field_name = field.get("field")
            if field_name not in df:
                continue
            fill_method = key
            try:
                if key.endswith("_impr"):
                    fill_method = key[:-5]
                    df_impr = df[df["is_vacant"].eq(False)].copy()
                    df_impr = _fill_thing(df_impr, field, fill_method)
                    df, df_impr = ensure_categories(df, df_impr, field_name)
                    df.loc[df_impr.index, field_name] = df_impr[field_name]
                elif key.endswith("_vacant"):
                    fill_method = key[:-7]
                    df_vacant = df[df["is_vacant"].eq(True)].copy()
                    df_vacant = _fill_thing(df_vacant, field, fill_method)
                    df, df_vacant = ensure_categories(df, df_vacant, field_name)
                    df.loc[df_vacant.index, field_name] = df_vacant[field_name]
                else:
                    df = _fill_thing(df, field, fill_method)
            except Exception as e:
                dtype = df[field_name].dtype if field_name in df else "<field absent>"
                raise ValueError(
                    f"Fill failed for field '{field_name}' using the "
                    f"'data.process.fill.{key}' list (fill method '{fill_method}', "
                    f"current column dtype: {dtype}). This almost always means the "
                    f"field's dtype doesn't match the fill -- e.g. a numeric fill "
                    f"('zero'/'median'/'mean'/'min'/'max') applied to a string/object "
                    f"column, or a string fill ('unknown'/'none') applied to a numeric "
                    f"column. Fix by either classifying '{field_name}' so it loads as the "
                    f"right type (field_classification.*.numeric vs .categorical, or an "
                    f"explicit dtype in data.load), or moving it to a fill method that "
                    f"matches its type. Original error: {e}"
                ) from e

    # After all fills, clean up

    # Partial fills on true/false are converted to a string in processing to avoid errors. Convert them back to booleans
    false_fills = settings.get("data", {}).get("process", {}).get("fill", {}).get("false", [])
    for fill_col in false_fills:
        if fill_col in df.columns:
            df[fill_col] = df[fill_col].astype(str).str.lower().eq("true")


    valuation_date = get_valuation_date(settings)
    valuation_year = valuation_date.year

    # Ensure year built and age in years are consistent
    # If year built exists, derive age in years from that
    # If year built doesn't exist, but age in years does, derive year built from that

    if "bldg_year_built" in df:
        df.loc[df["bldg_year_built"].gt(0), "bldg_age_years"] = (
            valuation_year - df["bldg_year_built"]
        )
        df.loc[df["bldg_year_built"].le(0), "bldg_age_years"] = 0
    elif "bldg_age_years" in df:
        df["bldg_year_built"] = valuation_year - df["bldg_age_years"]

    if "bldg_effective_year_built" in df:
        df.loc[df["bldg_effective_year_built"].gt(0), "bldg_effective_age_years"] = (
            valuation_year - df["bldg_effective_year_built"]
        )
        df.loc[df["bldg_effective_year_built"].le(0), "bldg_effective_age_years"] = 0
    elif "bldg_effective_age_years" in df:
        df["bldg_effective_year_built"] = (
            valuation_year - df["bldg_effective_age_years"]
        )

    # fill year/age with zero after they've been normalized
    year_age = [
        "bldg_year_built",
        "bldg_effective_year_built",
        "bldg_age_years",
        "bldg_effective_age_years",
    ]
    for field in year_age:
        if field in df:
            df = _fill_thing(df, field, "zero")

    # remaining fields get auto-filled

    cat_fields = get_fields_categorical(settings, df, include_boolean=False)
    bool_fields = get_fields_boolean(settings, df)

    if cat_fields is not None:
        for field in cat_fields:
            if field in df:
                df[field] = df[field].astype("str")
                df[field] = df[field].fillna("UNKNOWN")

    if bool_fields is not None:
        for field in bool_fields:
            if field in df.columns:
                df[field] = df[field].fillna(False).astype(str).str.lower().eq("true")
    return df
