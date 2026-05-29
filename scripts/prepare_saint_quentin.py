#!/usr/bin/env python3
"""Prepare a local Saint-Quentin OpenAVMKit smoke-test dataset.

The generated data lives under notebooks/pipeline/data/fr-02-saint_quentin,
which is ignored by git. The script writes normalized parquet inputs and a
minimal settings.json, then can run the OpenAVMKit load/process path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCALITY = "fr-02-saint_quentin"
DATA_DIR = ROOT / "notebooks" / "pipeline" / "data" / LOCALITY
RAW_DIR = DATA_DIR / "raw"
IN_DIR = DATA_DIR / "in"

CADASTRE_URL = (
    "https://services1.arcgis.com/5nIW6mZeb2YNJ7np/ArcGIS/rest/services/"
    "SIG_CADASTRE/FeatureServer"
)
DVF_API_URL = "https://www.data.gouv.fr/api/1/datasets/demandes-de-valeurs-foncieres/"
SQM_TO_SQFT = 10.763910416709722
ARCGIS_PAGE_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024],
        help="DVF years to download and filter. Default: 2024.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download/rebuild raw files.")
    parser.add_argument("--skip-smoke", action="store_true", help="Only prepare files.")
    return parser.parse_args()


def get_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def arcgis_count(layer_id: int) -> int:
    payload = get_json(
        f"{CADASTRE_URL}/{layer_id}/query",
        {"where": "1=1", "returnCountOnly": "true", "f": "json"},
    )
    return int(payload["count"])


def fetch_arcgis_layer(layer_id: int, name: str, force: bool) -> gpd.GeoDataFrame:
    out_path = RAW_DIR / f"{name}.geojson"
    if out_path.exists() and not force:
        return gpd.read_file(out_path)

    count = arcgis_count(layer_id)
    features: list[dict] = []
    for offset in range(0, count, ARCGIS_PAGE_SIZE):
        payload = get_json(
            f"{CADASTRE_URL}/{layer_id}/query",
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": ARCGIS_PAGE_SIZE,
                "orderByFields": "objectid",
            },
        )
        batch = payload.get("features", [])
        features.extend(batch)
        print(f"{name}: fetched {len(features):,}/{count:,}")

    collection = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(collection), encoding="utf-8")
    return gpd.GeoDataFrame.from_features(collection, crs="EPSG:4326")


def clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm_key(value: object) -> str:
    return re.sub(r"\s+", "", clean_cell(value)).upper()


def parcel_key_parts(codeident: object) -> tuple[str, str, str]:
    raw = clean_cell(codeident)
    if len(raw) < 6:
        return "", "", ""
    dep = raw[:2]
    commune = raw[3:6]
    return dep, commune, dep + commune


def normalize_parcels(parcels: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parcels = parcels.copy()
    parcels["key"] = parcels["codeident"].map(norm_key)
    parts = parcels["codeident"].map(parcel_key_parts)
    parcels["dvf_departement"] = parts.map(lambda x: x[0])
    parcels["dvf_commune"] = parts.map(lambda x: x[1])
    parcels["commune_code"] = parts.map(lambda x: x[2])
    parcels["codcomm_arcgis"] = parcels["codeident"].astype(str).str[:6]
    parcels["cadastral_section"] = parcels["sect_cad"].astype(str).str.strip()
    parcels["parcel_number"] = pd.to_numeric(parcels["parcelle"], errors="coerce").astype("Int64")
    parcels["neighborhood"] = parcels["cadastral_section"].replace("", pd.NA).fillna("unknown")

    parcels_m = parcels.to_crs("EPSG:3949")
    parcels_m["land_area_sqm"] = parcels_m.geometry.area
    parcels_m["land_area_sqft"] = parcels_m["land_area_sqm"] * SQM_TO_SQFT

    buildings = buildings.copy()
    buildings["building_area_footprint_sqm"] = pd.to_numeric(
        buildings.get("surface"), errors="coerce"
    )
    buildings_m = buildings.to_crs("EPSG:3949")
    missing_area = buildings_m["building_area_footprint_sqm"].isna()
    buildings_m.loc[missing_area, "building_area_footprint_sqm"] = buildings_m.loc[
        missing_area, "geometry"
    ].area
    buildings_m["building_area_shon_sqm"] = pd.to_numeric(buildings_m.get("shon"), errors="coerce")
    buildings_pts = buildings_m.copy()
    buildings_pts["geometry"] = buildings_pts.geometry.representative_point()

    joined = gpd.sjoin(
        buildings_pts[["type", "building_area_footprint_sqm", "building_area_shon_sqm", "geometry"]],
        parcels_m[["key", "geometry"]],
        how="left",
        predicate="within",
    ).dropna(subset=["key"])

    def first_mode(values: pd.Series) -> object:
        modes = values.dropna().mode()
        return modes.iloc[0] if len(modes) else pd.NA

    bldg_agg = joined.groupby("key").agg(
        bldg_count=("key", "size"),
        bldg_area_footprint_sqm=("building_area_footprint_sqm", "sum"),
        bldg_area_shon_sqm=("building_area_shon_sqm", "sum"),
        bldg_type=("type", first_mode),
    )
    parcels_m = parcels_m.merge(bldg_agg, on="key", how="left")
    parcels_m["bldg_count"] = parcels_m["bldg_count"].fillna(0).astype("Int64")
    for col in ["bldg_area_footprint_sqm", "bldg_area_shon_sqm"]:
        parcels_m[col] = parcels_m[col].fillna(0.0)
    parcels_m["bldg_area_finished_sqm"] = parcels_m["bldg_area_shon_sqm"]
    use_footprint = parcels_m["bldg_area_finished_sqm"].le(0)
    parcels_m.loc[use_footprint, "bldg_area_finished_sqm"] = parcels_m.loc[
        use_footprint, "bldg_area_footprint_sqm"
    ]
    parcels_m["bldg_area_finished_sqft"] = parcels_m["bldg_area_finished_sqm"] * SQM_TO_SQFT
    parcels_m["bldg_area_footprint_sqft"] = parcels_m["bldg_area_footprint_sqm"] * SQM_TO_SQFT
    parcels_m["is_vacant"] = parcels_m["bldg_area_finished_sqm"].le(0)

    cols = [
        "key",
        "codeident",
        "codcomm_arcgis",
        "dvf_departement",
        "dvf_commune",
        "commune_code",
        "cadastral_section",
        "parcel_number",
        "neighborhood",
        "land_area_sqm",
        "land_area_sqft",
        "bldg_count",
        "bldg_area_footprint_sqm",
        "bldg_area_footprint_sqft",
        "bldg_area_finished_sqm",
        "bldg_area_finished_sqft",
        "bldg_type",
        "is_vacant",
        "geometry",
    ]
    return parcels_m[cols].to_crs("EPSG:4326")


def dvf_resources() -> dict[int, str]:
    dataset = get_json(DVF_API_URL)
    resources = {}
    for resource in dataset.get("resources", []):
        title = resource.get("title", "")
        match = re.search(r"(\d{4})", title)
        if resource.get("type") == "main" and match:
            resources[int(match.group(1))] = resource["url"]
    return resources


def download(url: str, path: Path, force: bool) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        return
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def dvf_codeident(row: pd.Series) -> str:
    dep = clean_cell(row.get("Code departement", "")).zfill(2)
    commune = clean_cell(row.get("Code commune", "")).zfill(3)
    prefix = clean_cell(row.get("Prefixe de section", ""))
    prefix = "   " if prefix in {"", "0", "00", "000", "nan", "<NA>"} else prefix.zfill(3)
    section = clean_cell(row.get("Section", "")).upper().rjust(2)
    plan_raw = clean_cell(row.get("No plan", ""))
    try:
        plan = str(int(float(plan_raw))).zfill(4)
    except ValueError:
        plan = plan_raw.zfill(4)
    return norm_key(f"{dep}0{commune}{prefix}{section}{plan}")


def parse_fr_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def stable_sale_suffix(row: pd.Series) -> str:
    payload = "|".join(str(row.get(col, "")) for col in row.index)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def prepare_sales(parcels: gpd.GeoDataFrame, years: list[int], force: bool) -> pd.DataFrame:
    out_path = IN_DIR / "sales.parquet"
    required_cols = {
        "valid_for_ratio_study",
        "valid_for_land_ratio_study",
    }
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        existing_years = set(pd.to_datetime(existing["sale_date"]).dt.year.dropna().astype(int))
        if existing_years == set(years) and required_cols.issubset(existing.columns):
            return existing

    resources: dict[int, str] = {}
    parcel_keys = set(parcels["key"])
    departments = set(parcels["dvf_departement"])
    communes = set(parcels["dvf_commune"])
    chunks: list[pd.DataFrame] = []

    for year in years:
        zip_path = RAW_DIR / f"valeursfoncieres-{year}.txt.zip"
        print(f"DVF {year}: downloading/filtering")
        if not zip_path.exists() or force:
            if not resources:
                resources = dvf_resources()
            if year not in resources:
                raise ValueError(f"No DVF resource found for {year}")
            download(resources[year], zip_path, force)
        with zipfile.ZipFile(zip_path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if not names:
                raise ValueError(f"No .txt file found in {zip_path}")
            with archive.open(names[0]) as handle:
                reader = pd.read_csv(handle, sep="|", dtype="string", chunksize=200_000)
                for chunk in reader:
                    chunk["Code departement"] = chunk["Code departement"].str.strip().str.zfill(2)
                    chunk["Code commune"] = chunk["Code commune"].str.strip().str.zfill(3)
                    chunk = chunk[
                        chunk["Code departement"].isin(departments)
                        & chunk["Code commune"].isin(communes)
                    ].copy()
                    if chunk.empty:
                        continue
                    chunk["key"] = chunk.apply(dvf_codeident, axis=1)
                    chunk = chunk[chunk["key"].isin(parcel_keys)].copy()
                    if chunk.empty:
                        continue
                    chunks.append(chunk)

    if not chunks:
        raise ValueError("No DVF sales matched the Saint-Quentin parcel layer")

    raw = pd.concat(chunks, ignore_index=True)
    raw["sale_date"] = pd.to_datetime(raw["Date mutation"], format="%d/%m/%Y", errors="coerce")
    raw["sale_price"] = parse_fr_number(raw["Valeur fonciere"])
    raw["surface_reelle_bati_sqm"] = parse_fr_number(raw["Surface reelle bati"])
    raw["sale_land_area_sqm"] = parse_fr_number(raw["Surface terrain"])
    raw["rooms"] = parse_fr_number(raw["Nombre pieces principales"])
    raw["sale_nature"] = raw["Nature mutation"].astype("string")
    raw["property_type"] = raw["Type local"].astype("string")
    raw = raw[raw["sale_date"].notna() & raw["sale_price"].gt(0)].copy()
    raw = raw[raw["sale_nature"].str.contains("Vente", case=False, na=False)].copy()

    grouped = raw.groupby(
        ["key", "sale_date", "sale_price", "sale_nature"],
        dropna=False,
        as_index=False,
    ).agg(
        property_type=("property_type", lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA),
        bldg_area_finished_sqm=("surface_reelle_bati_sqm", "max"),
        sale_land_area_sqm=("sale_land_area_sqm", "max"),
        rooms=("rooms", "max"),
    )
    grouped["sale_date"] = grouped["sale_date"].dt.strftime("%Y-%m-%d")
    grouped["bldg_area_finished_sqm"] = grouped["bldg_area_finished_sqm"].fillna(0.0)
    grouped["bldg_area_finished_sqft"] = grouped["bldg_area_finished_sqm"] * SQM_TO_SQFT
    grouped["valid_sale"] = True
    grouped = grouped.merge(parcels[["key", "is_vacant"]], on="key", how="left")
    current_is_vacant = grouped["is_vacant"].fillna(False)
    sale_looks_vacant = grouped["bldg_area_finished_sqm"].le(0)
    grouped["vacant_sale"] = sale_looks_vacant
    grouped["valid_for_ratio_study"] = grouped["valid_sale"] & sale_looks_vacant.eq(
        current_is_vacant
    )
    grouped["valid_for_land_ratio_study"] = grouped["valid_sale"] & grouped["vacant_sale"]
    grouped = grouped.drop(columns=["is_vacant"])
    grouped["sale_hash"] = grouped.apply(stable_sale_suffix, axis=1)
    grouped["key_sale"] = (
        grouped["key"]
        + "---"
        + grouped["sale_date"]
        + "---"
        + grouped["sale_hash"]
    )

    cols = [
        "key_sale",
        "key",
        "sale_date",
        "sale_price",
        "sale_nature",
        "property_type",
        "bldg_area_finished_sqm",
        "bldg_area_finished_sqft",
        "sale_land_area_sqm",
        "rooms",
        "valid_sale",
        "vacant_sale",
        "valid_for_ratio_study",
        "valid_for_land_ratio_study",
    ]
    sales = grouped[cols].sort_values(["sale_date", "key_sale"]).reset_index(drop=True)
    sales.to_parquet(out_path, index=False)
    return sales


def write_settings(years: list[int]) -> None:
    main_model_features = [
        "land_area_sqm",
        "bldg_area_finished_sqm",
        "bldg_area_footprint_sqm",
        "bldg_count",
        "neighborhood",
        "commune_code",
        "cadastral_section",
    ]
    vacant_model_features = [
        "land_area_sqm",
        "neighborhood",
        "commune_code",
        "cadastral_section",
    ]
    naive_area_features = [
        "land_area_sqm",
        "bldg_area_finished_sqm",
    ]
    main_area_models = {
        "default": {
            "ind_vars": main_model_features,
        },
        "naive_area": {
            "model": "naive_area",
            "ind_vars": naive_area_features,
        },
        "local_area": {
            "model": "local_area",
            "ind_vars": main_model_features,
            "locations": ["neighborhood", "commune_code", "cadastral_section"],
        },
        "lightgbm": {
            "engine": "lightgbm",
            "model": "lightgbm",
            "ind_vars": main_model_features,
            "n_trials": 10,
        },
    }
    vacant_area_models = {
        "default": {
            "ind_vars": vacant_model_features,
        },
        "naive_area": {
            "model": "naive_area",
            "ind_vars": naive_area_features,
        },
        "local_area": {
            "model": "local_area",
            "ind_vars": vacant_model_features,
            "locations": ["neighborhood", "commune_code", "cadastral_section"],
        },
        "lightgbm": {
            "engine": "lightgbm",
            "model": "lightgbm",
            "ind_vars": vacant_model_features,
            "n_trials": 10,
        },
    }
    settings = {
        "locality": {
            "name": "Saint-Quentin",
            "country": "FR",
            "state": "02",
            "slug": LOCALITY,
            "units": "metric",
        },
        "data": {
            "load": {
                "geo_parcels": {
                    "key": "geo_parcels",
                    "filename": "parcels.parquet",
                    "dupes": {
                        "subset": ["key"],
                        "sort_by": ["key", "asc"],
                        "drop": True,
                    },
                    "load": {
                        "key": ["key", "string"],
                        "codeident": ["codeident", "string"],
                        "commune_code": ["commune_code", "string"],
                        "cadastral_section": ["cadastral_section", "string"],
                        "neighborhood": ["neighborhood", "string"],
                        "land_area_sqm": ["land_area_sqm", "float"],
                        "land_area_sqft": ["land_area_sqft", "float"],
                        "bldg_count": ["bldg_count", "float"],
                        "bldg_area_footprint_sqm": ["bldg_area_footprint_sqm", "float"],
                        "bldg_area_footprint_sqft": ["bldg_area_footprint_sqft", "float"],
                        "bldg_area_finished_sqm": ["bldg_area_finished_sqm", "float"],
                        "bldg_area_finished_sqft": ["bldg_area_finished_sqft", "float"],
                        "bldg_type": ["bldg_type", "string"],
                        "is_vacant": ["is_vacant", "boolean", "na_false"],
                    },
                },
                "sales": {
                    "key": "sales",
                    "filename": "sales.parquet",
                    "geometry": False,
                    "dupes": {
                        "subset": ["key_sale"],
                        "sort_by": ["key_sale", "asc"],
                        "drop": True,
                    },
                    "load": {
                        "key_sale": ["key_sale", "string"],
                        "key": ["key", "string"],
                        "sale_date": ["sale_date", "datetime", "%Y-%m-%d"],
                        "sale_price": ["sale_price", "float"],
                        "sale_nature": ["sale_nature", "string"],
                        "property_type": ["property_type", "string"],
                        "bldg_area_finished_sqm": ["bldg_area_finished_sqm", "float"],
                        "bldg_area_finished_sqft": ["bldg_area_finished_sqft", "float"],
                        "sale_land_area_sqm": ["sale_land_area_sqm", "float"],
                        "rooms": ["rooms", "float"],
                        "valid_sale": ["valid_sale", "boolean", "na_false"],
                        "vacant_sale": ["vacant_sale", "boolean", "na_false"],
                        "valid_for_ratio_study": [
                            "valid_for_ratio_study",
                            "boolean",
                            "na_false",
                        ],
                        "valid_for_land_ratio_study": [
                            "valid_for_land_ratio_study",
                            "boolean",
                            "na_false",
                        ],
                    },
                },
            },
            "process": {
                "merge": {
                    "universe": ["geo_parcels"],
                    "sales": ["sales"],
                },
                "enrich": {},
            },
        },
        "modeling": {
            "metadata": {
                "valuation_date": f"{max(years) + 1}-01-01",
                "use_sales_from": min(years),
                "test_sales_from": max(years),
                "modeler": "Saint-Quentin metric test",
            },
            "model_groups": {
                "all": {
                    "name": "All parcels",
                    "filter": [">=", "land_area_sqm", 0],
                },
            },
            "instructions": {
                "dep_var": "sale_price_time_adj",
                "dep_var_test": "sale_price_time_adj",
                "time_adjustment": {
                    "period": "M",
                },
                "main": {
                    "run": ["naive_area", "local_area", "lightgbm"],
                },
                "vacant": {
                    "run": ["naive_area", "local_area", "lightgbm"],
                },
                "hedonic": {
                    "skip": {"all": ["all"]},
                },
            },
            "models": {
                "main": main_area_models,
                "vacant": vacant_area_models,
                "default": {
                    "ind_vars": main_model_features,
                }
            },
        },
        "field_classification": {
            "important": {
                "fields": {
                    "loc_neighborhood": "neighborhood",
                    "land_category": "cadastral_section",
                    "impr_category": "property_type",
                },
                "locations": ["neighborhood", "commune_code", "cadastral_section"],
                "report_locations": ["neighborhood", "commune_code"],
            },
            "land": {
                "+numeric": ["land_area_sqm", "land_area_sqft"],
                "+categorical": ["commune_code", "cadastral_section", "neighborhood"],
            },
            "impr": {
                "+numeric": [
                    "bldg_count",
                    "bldg_area_footprint_sqm",
                    "bldg_area_footprint_sqft",
                    "bldg_area_finished_sqm",
                    "bldg_area_finished_sqft",
                    "rooms",
                ],
                "+categorical": ["bldg_type", "property_type"],
            },
            "other": {
                "+categorical": ["codeident", "sale_nature"],
                "+numeric": ["sale_land_area_sqm"],
            },
        },
    }
    settings_path = IN_DIR / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def install_optional_import_stubs() -> None:
    if "census" not in sys.modules:
        census_mod = types.ModuleType("census")

        class Census:  # pragma: no cover - smoke-test import shim
            pass

        census_mod.Census = Census
        sys.modules["census"] = census_mod
    for name in ["lightgbm", "xgboost"]:
        sys.modules.setdefault(name, types.ModuleType(name))


def smoke_test() -> None:
    install_optional_import_stubs()
    from openavmkit.data import load_dataframe, process_data
    from openavmkit.utilities.settings import (
        get_fields_boolean,
        get_fields_categorical,
        get_fields_numeric,
        load_settings,
    )

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(DATA_DIR)
        settings = load_settings("in/settings.json")
        fields_cat = get_fields_categorical(settings, include_boolean=False)
        fields_bool = get_fields_boolean(settings)
        fields_num = get_fields_numeric(settings, include_boolean=False)
        dataframes = {
            key: load_dataframe(
                entry,
                settings,
                verbose=True,
                fields_cat=fields_cat,
                fields_bool=fields_bool,
                fields_num=fields_num,
            )
            for key, entry in settings["data"]["load"].items()
        }
        sup = process_data(dataframes, settings, verbose=True)
        print(
            "SMOKE OK: "
            f"universe={len(sup.universe):,}, "
            f"sales={len(sup.sales):,}, "
            f"vacant_universe={int(sup.universe['is_vacant'].sum()):,}, "
            f"vacant_sales={int(sup.sales['vacant_sale'].sum()):,}"
        )
    finally:
        import os

        os.chdir(old_cwd)


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IN_DIR.mkdir(parents=True, exist_ok=True)

    parcels_raw = fetch_arcgis_layer(1, "cadastre_parcels", args.force)
    buildings_raw = fetch_arcgis_layer(0, "cadastre_buildings", args.force)
    parcels = normalize_parcels(parcels_raw, buildings_raw)
    parcels.to_parquet(IN_DIR / "parcels.parquet", index=False)
    sales = prepare_sales(parcels, args.years, args.force)
    write_settings(args.years)

    print(f"Prepared {len(parcels):,} parcels and {len(sales):,} sales for {LOCALITY}")
    if not args.skip_smoke:
        smoke_test()


if __name__ == "__main__":
    main()
