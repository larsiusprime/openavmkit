"""Condo resolution: bring condo units into the "one row per parcel" model.

Condo units are assessment parcels with no geometry of their own -- they physically
sit inside a shared building/land parcel. openavmkit drops geometry-less rows, so condos
normally vanish from the universe. This module gives each condo unit a *borrowed* building
polygon (so every centroid-based enrichment -- DEM, census, OSM distances, basic-geo,
Overture -- works and yields identical "shared parcel" values per building), assigns a
``condo_group`` building identifier (a location, like neighborhood), and computes a per-unit
allocated land size.

It is opt-in via ``data.process.condos`` in settings and runs once at the top of
``process_data`` -- BEFORE the universe merge / geometry attach -- mutating the loaded
``dataframes`` dict. After it runs, the existing pipeline does everything else unchanged.

Settings schema (see resolve_condos docstring)::

    "condos": {
      "enabled": true,
      "select": ["isin", "bldg_type", ["CONDOMINIUM", ...]],
      "link":   { "method": "id_prefix", "id_field": "parcel_num", "prefix_len": 9, "from": "geo_parcels" },
      "group_field": "condo_group",
      "borrow_geometry": true,
      "land_share": { "method": "field", "field": "land_area_sqft" }
    }
"""
import warnings

import geopandas as gpd
import pandas as pd

from openavmkit.filters import resolve_filter
from openavmkit.utilities.geometry import get_crs

LAND_ALLOC_FIELD = "land_area_alloc_sqft"
BORROWED_FLAG = "geometry_borrowed"


def _settings_condos(settings: dict) -> dict:
    return settings.get("data", {}).get("process", {}).get("condos", {})


def _polygon_areas_sqft(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Area in square feet for each polygon, via a local equal-area projection."""
    crs_ea = get_crs(gdf, "equal_area")
    return gdf.to_crs(crs_ea).geometry.area * 10.7639


def _resolve_link(condos: pd.DataFrame, dataframes: dict, link: dict, verbose: bool):
    """Return (link_ids, geom_map): a per-condo-row Series of building ids (NaN where
    unresolved) and a dict {building_id -> representative shapely geometry}.
    """
    method = link.get("method", "id_prefix")
    from_id = link.get("from", "geo_parcels")
    from_df = dataframes.get(from_id)
    if from_df is None or "geometry" not in from_df:
        raise ValueError(f"condos.link.from='{from_id}' has no geometry to borrow.")
    gdf = from_df[from_df.geometry.notna() & ~from_df.geometry.is_empty].copy()

    if method == "id_prefix":
        id_field = link.get("id_field", link.get("key"))
        n = int(link["prefix_len"])
        for nm, frame in (("from", gdf), ("condos", condos)):
            if id_field not in frame.columns:
                raise ValueError(
                    f"condos.link.id_field='{id_field}' not found in {nm} dataframe."
                )
        gdf["_prefix"] = gdf[id_field].astype(str).str.slice(0, n)
        gdf["_area"] = _polygon_areas_sqft(gdf).values
        # Representative polygon per prefix = the largest (the building footprint/pad).
        rep = gdf.sort_values("_area", ascending=False).drop_duplicates("_prefix", keep="first")
        geom_map = dict(zip(rep["_prefix"], rep.geometry))
        link_ids = condos[id_field].astype(str).str.slice(0, n)
        link_ids = link_ids.where(link_ids.isin(geom_map.keys()))
        return link_ids, geom_map

    if method == "parent_id":
        parent_field = link["parent_field"]
        from_key = link.get("from_key", "key")
        if parent_field not in condos.columns:
            raise ValueError(f"condos.link.parent_field='{parent_field}' not in condo rows.")
        geom_map = dict(zip(gdf[from_key].astype(str), gdf.geometry))
        link_ids = condos[parent_field].astype(str)
        link_ids = link_ids.where(link_ids.isin(geom_map.keys()))
        return link_ids, geom_map

    if method == "spatial":
        raise NotImplementedError(
            "condos.link.method='spatial' is not implemented yet; use 'id_prefix' or 'parent_id'."
        )
    raise ValueError(f"Unknown condos.link.method: {method!r}")


def _apply_land_share(
    df: pd.DataFrame, condo_idx, link_ids, geom_map, land_share: dict, verbose: bool
) -> pd.DataFrame:
    """Write per-unit allocated land size to LAND_ALLOC_FIELD and overwrite condos'
    land_area_sqft with it (so the standard land feature is per-unit for condos).
    """
    method = land_share.get("method", "field")
    if LAND_ALLOC_FIELD not in df.columns:
        df[LAND_ALLOC_FIELD] = pd.NA

    if method == "field":
        field = land_share["field"]
        if field not in df.columns:
            raise ValueError(f"condos.land_share.field='{field}' not found.")
        alloc = pd.to_numeric(df.loc[condo_idx, field], errors="coerce")

    elif method == "floor_area":
        floor_field = land_share.get("floor_field", "bldg_area_finished_sqft")
        if floor_field not in df.columns:
            raise ValueError(f"condos.land_share.floor_field='{floor_field}' not found.")
        floor = pd.to_numeric(df.loc[condo_idx, floor_field], errors="coerce").fillna(0.0)
        bid = pd.Series(link_ids.values, index=condo_idx)
        total_floor = floor.groupby(bid).transform("sum")
        # Parcel land = area of the borrowed building polygon, in sqft.
        parcel_land = bid.map(
            lambda b: _GEOM_AREA_CACHE.get(b) if b in _GEOM_AREA_CACHE else None
        )
        share = (floor / total_floor).where(total_floor > 0, 0.0)
        alloc = share * pd.to_numeric(parcel_land, errors="coerce")
    else:
        raise ValueError(f"Unknown condos.land_share.method: {method!r}")

    df.loc[condo_idx, LAND_ALLOC_FIELD] = alloc.values
    if "land_area_sqft" in df.columns:
        # condos' land feature = their allocated share (not the full footprint)
        valid = alloc.notna()
        df.loc[alloc.index[valid], "land_area_sqft"] = alloc[valid].values
    if verbose:
        print(f"resolve_condos: land_share '{method}' set {int(alloc.notna().sum())} units")
    return df


_GEOM_AREA_CACHE: dict = {}


def _auto_register_fields(settings: dict, s: dict) -> None:
    """Make condo_group a categorical location and land_area_alloc_sqft a numeric land
    field, unless already declared. Mutates settings IN MEMORY only.

    The default field names (``condo_group``, ``land_area_alloc_sqft``, ``geometry_borrowed``)
    ship in the settings template's field_classification, so for default-named outputs this is
    a no-op and they are recognized across all notebooks. This in-memory registration is only a
    convenience for a CUSTOM ``group_field``; because each notebook reloads settings.json fresh,
    a custom name must also be declared in settings.json to persist -- we warn when that's the case.
    """
    group_field = s.get("group_field", "condo_group")
    fc = settings.setdefault("field_classification", {})
    land = fc.setdefault("land", {})
    cat = land.setdefault("categorical", [])
    num = land.setdefault("numeric", [])
    important = fc.setdefault("important", {})
    locs = important.setdefault("locations", [])

    if group_field not in cat:
        cat.append(group_field)
        warnings.warn(
            f"resolve_condos: condo group_field '{group_field}' is not declared in "
            f"field_classification.land.categorical. Auto-registering it for THIS run only -- "
            f"settings reload fresh each notebook, so add '{group_field}' to "
            f"field_classification.land.categorical (and important.locations) in your settings.json "
            f"to persist it. The default 'condo_group' already ships in the settings template."
        )
    if group_field not in locs:
        locs.append(group_field)
    if LAND_ALLOC_FIELD not in num:
        num.append(LAND_ALLOC_FIELD)


def resolve_condos(dataframes: dict, settings: dict, verbose: bool = False) -> dict:
    """Resolve condo units into the universe by borrowing building geometry.

    Opt-in via ``data.process.condos.enabled``. For each universe-merge source frame
    containing condo rows (matched by ``select``), this:
      1. links each unit to a building polygon (``link.method``: id_prefix | parent_id | spatial);
      2. borrows that polygon as the unit's geometry (appends rows to ``geo_parcels``);
      3. writes ``group_field`` (the building id) onto the source;
      4. writes a per-unit allocated land size (``land_share.method``: field | floor_area)
         to ``land_area_alloc_sqft`` and into ``land_area_sqft`` for condos;
      5. auto-registers the new fields in field_classification.

    Returns the (mutated) dataframes dict. A no-op when disabled.
    """
    s = _settings_condos(settings)
    if not s.get("enabled", False):
        return dataframes

    geo = dataframes.get("geo_parcels")
    if geo is None or "geometry" not in geo:
        warnings.warn("resolve_condos: no 'geo_parcels' with geometry; skipping.")
        return dataframes

    select = s.get("select")
    link = s.get("link", {})
    group_field = s.get("group_field", "condo_group")
    borrow = s.get("borrow_geometry", True)
    land_share = s.get("land_share", {})

    merge_univ = (
        settings.get("data", {}).get("process", {}).get("merge", {}).get("universe", [])
    )
    univ_ids = [e if isinstance(e, str) else e.get("id") for e in merge_univ]

    geo_keys = set(geo["key"].astype(str))
    if BORROWED_FLAG not in geo.columns:
        geo[BORROWED_FLAG] = False

    total_borrowed = 0
    for uid in univ_ids:
        df = dataframes.get(uid)
        if df is None or "key" not in df.columns:
            continue
        mask = resolve_filter(df, select) if select else pd.Series(True, index=df.index)
        if mask.sum() == 0:
            continue
        condos = df[mask]
        condo_idx = condos.index

        link_ids, geom_map = _resolve_link(condos, dataframes, link, verbose)

        # cache building polygon areas (sqft) for floor_area land-share
        if geom_map:
            rep_gdf = gpd.GeoDataFrame(
                {"_b": list(geom_map.keys())},
                geometry=list(geom_map.values()),
                crs=geo.crs,
            )
            _GEOM_AREA_CACHE.update(dict(zip(rep_gdf["_b"], _polygon_areas_sqft(rep_gdf).values)))

        # 3) condo_group
        if group_field not in df.columns:
            df[group_field] = pd.NA
        df.loc[condo_idx, group_field] = link_ids.values

        # 2) borrow geometry for units lacking their own polygon
        if borrow:
            keys = condos["key"].astype(str)
            need = condos[(~keys.isin(geo_keys)).values & link_ids.notna().values]
            need_ids = link_ids.loc[need.index]
            rows = []
            id_field = link.get("id_field")
            for i, (_, row) in enumerate(need.iterrows()):
                bid = need_ids.iloc[i]
                geom = geom_map.get(bid)
                if geom is None:
                    continue
                rec = {"key": str(row["key"]), "geometry": geom, BORROWED_FLAG: True}
                if id_field and id_field in need.columns:
                    rec[id_field] = row[id_field]
                rows.append(rec)
            if rows:
                add = gpd.GeoDataFrame(rows, geometry="geometry", crs=geo.crs)
                geo = gpd.GeoDataFrame(
                    pd.concat([geo, add], ignore_index=True), geometry="geometry", crs=geo.crs
                )
                geo_keys.update(add["key"].astype(str))
                total_borrowed += len(add)

        # 4) land share
        df = _apply_land_share(df, condo_idx, link_ids, geom_map, land_share, verbose)
        dataframes[uid] = df

        if verbose:
            print(
                f"resolve_condos['{uid}']: {int(mask.sum())} condo rows, "
                f"{int(link_ids.notna().sum())} linked"
            )

    dataframes["geo_parcels"] = geo
    if verbose:
        print(f"resolve_condos: borrowed geometry for {total_borrowed} condo units total")
    _auto_register_fields(settings, s)
    return dataframes
