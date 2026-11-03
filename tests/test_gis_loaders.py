"""data.load reads GeoPackage and shapefile parcel layers, like Parquet and CSV."""

import os

import geopandas as gpd
import pytest
from shapely.geometry import box

from openavmkit.data import load_dataframe


def _parcels(n=4):
	return gpd.GeoDataFrame(
		{
			"PIN": [f"00{i}" for i in range(1, n + 1)],
			"ACRES": [0.5 * i for i in range(1, n + 1)],
			"OWNER": ["A", "B", "C", "D"][:n],
		},
		geometry=[box(i, 0, i + 1, 1) for i in range(n)],
		crs="EPSG:4326",
	)


@pytest.fixture
def in_dir(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	os.makedirs("in", exist_ok=True)
	return tmp_path / "in"


ENTRY = {
	"load": {"key": ["PIN", "string"], "acres": "ACRES"},
	"calc": {"land_area_sqft": ["*", "ACRES", 43560]},
}


def _write(in_dir, filename):
	if filename.isupper():  # e.g. a Windows export: PARCELS.SHP, PARCELS.DBF, ...
		_parcels().to_file(in_dir / filename.lower())
		for f in in_dir.iterdir():
			f.rename(in_dir / f.name.upper())
	else:
		_parcels().to_file(in_dir / filename)


@pytest.mark.parametrize("filename", ["parcels.gpkg", "parcels.shp", "PARCELS.SHP"])
def test_gis_file_loads_like_parquet(in_dir, filename):
	_write(in_dir, filename)
	df = load_dataframe({**ENTRY, "filename": filename}, {}, verbose=False)

	assert isinstance(df, gpd.GeoDataFrame)
	assert len(df) == 4
	assert df["key"].tolist() == ["001", "002", "003", "004"]  # read as text, zeros kept
	assert df["land_area_sqft"].tolist() == pytest.approx([21780, 43560, 65340, 87120])
	assert "OWNER" not in df.columns  # only mapped columns are loaded, as for Parquet/CSV
	assert df.crs is not None and df.crs.to_epsg() == 4326
	assert df.geometry.iloc[0].equals(box(0, 0, 1, 1))


def test_geopackage_layer_can_be_chosen(in_dir):
	_parcels().to_file(in_dir / "county.gpkg", layer="parcels")
	gpd.GeoDataFrame({"ZONE": ["R1"]}, geometry=[box(0, 0, 9, 9)], crs="EPSG:4326").to_file(
		in_dir / "county.gpkg", layer="zoning"
	)
	parcels = load_dataframe({**ENTRY, "filename": "county.gpkg", "layer": "parcels"}, {})
	zoning = load_dataframe({"filename": "county.gpkg", "layer": "zoning", "load": {"zone": "ZONE"}}, {})

	assert len(parcels) == 4 and "acres" in parcels.columns
	assert zoning["zone"].tolist() == ["R1"]
