import os
import warnings
import geopandas as gpd
import pandas as pd
import traceback
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs
from tqdm import tqdm
import shapely.wkb

from openavmkit.utilities.geometry import get_crs
from openavmkit.utilities.timing import TimingData


class OvertureService:
    """Service for fetching and processing Overture building data.

    Attributes
    ----------
    settings : dict
        Overture settings dictionary
    fs : S3FileSystem
    bucket : str
    prefix : str
    """
    
    DEFAULT_COLUMNS = [
        "id", "geometry", "bbox",
        "height", "est_height", "num_floors", "num_floors_underground",
        "subtype", "class", "sources"   # sources = per-property confidence
    ]

    def __init__(self, settings: dict):
        """Initialize the Overture service with settings.

        Parameters
        ----------
        settings : dict
            Settings dictionary
        """
        self.settings = settings.get("overture", {})
        if not self.settings:
            warnings.warn("No Overture settings found in settings dictionary")
        self.cache_dir = "cache/overture"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize S3 filesystem
        self.fs = fs.S3FileSystem(anonymous=True, region="us-west-2")
        self.bucket = "overturemaps-us-west-2"
        self.prefix = self._resolve_latest_buildings_prefix()

    def _get_cache_path(self, cache_type: str, bbox: tuple) -> str:
        """Get the cache path for a given type and bounding box."""
        cache_key = f"{cache_type}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")

    def _get_dataset(self):
        """Get the PyArrow dataset for buildings."""
        path = f"{self.bucket}/{self.prefix}"
        return ds.dataset(path, filesystem=self.fs, format="parquet")

    def _geoarrow_schema_adapter(self, schema: pa.Schema) -> pa.Schema:
        """Convert a geoarrow-compatible schema to a proper geoarrow schema."""
        geometry_field_index = schema.get_field_index("geometry")
        geometry_field = schema.field(geometry_field_index)
        geoarrow_geometry_field = geometry_field.with_metadata(
            {b"ARROW:extension:name": b"geoarrow.wkb"}
        )
        return schema.set(geometry_field_index, geoarrow_geometry_field)

    def _batch_to_geodataframe(self, batch: pa.RecordBatch) -> gpd.GeoDataFrame:
        """Convert a PyArrow batch to a GeoDataFrame with proper geometry handling."""
        # Convert to pandas DataFrame first
        df = batch.to_pandas()

        # Convert WKB geometry to shapely geometry
        df["geometry"] = df["geometry"].apply(lambda wkb: shapely.wkb.loads(wkb) if pd.notnull(wkb) else None)

        # Create GeoDataFrame with WGS84 CRS (EPSG:4326)
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def get_buildings(
        self, 
        bbox, 
        columns: list[str] | None = None,
        use_cache=True, 
        verbose=False
    ):
        """Fetch building data from Overture within the specified bounding box.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Tuple of (minx, miny, maxx, maxy) in WGS84 coordinates
        columns : list[str]
            Desired columns to load
        use_cache : bool, optional
            Whether to use cached data. Default is True.
        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with building footprints
        """
        
        typical_floor_height_m: float = 3.2
        
        t = TimingData()
        try:
            if verbose:
                print(f"--> Current settings: {self.settings}")

            if not self.settings:
                if verbose:
                    print("--> No Overture settings found")
                return gpd.GeoDataFrame()

            if not self.settings.get("enabled", False):
                if verbose:
                    print("--> Overture service disabled in settings")
                return gpd.GeoDataFrame()

            if verbose:
                print(f"--> Bounding box: {bbox}")

            # Get cache path for buildings
            cache_path = self._get_cache_path("buildings", bbox)

            # Check cache
            if use_cache and os.path.exists(cache_path):
                if verbose:
                    print(f"--> Loading buildings from cache: {cache_path}")
                return gpd.read_parquet(cache_path)

            if verbose:
                print("--> Fetching data from Overture...")

            try:
                # Create bounding box filter
                xmin, ymin, xmax, ymax = bbox
                filter = (
                    (pc.field("bbox", "xmin") < xmax)
                    & (pc.field("bbox", "xmax") > xmin)
                    & (pc.field("bbox", "ymin") < ymax)
                    & (pc.field("bbox", "ymax") > ymin)
                )
                
                # Decide which columns to fetch
                proj_cols = columns if columns is not None else self.DEFAULT_COLUMNS.copy()
                # Ensure required columns are present
                for req in ("geometry", "bbox"):
                    if req not in proj_cols:
                        proj_cols.append(req)
                
                # Get dataset and apply filter+projection
                dataset = self._get_dataset()
                if verbose:
                    print("--> Dataset columns:", dataset.schema.names)
                available = set(dataset.schema.names)
                missing = [c for c in proj_cols if c not in available]
                proj_cols = [c for c in proj_cols if c in available]
                if verbose and missing:
                    print(f"--> Skipping unavailable columns: {missing}")
                batches = dataset.to_batches(filter=filter, columns=proj_cols)

                # Count total batches for progress bar
                if verbose:
                    print("--> Counting batches...")
                    total_batches = sum(1 for _ in batches)
                    print(f"--> Found {total_batches} batches")
                    batches = dataset.to_batches(filter=filter, columns=proj_cols)  # Reset iterator

                # Process batches with progress bar
                dfs = []
                buildings_found = 0

                with tqdm(
                    total=total_batches if verbose else None,
                    desc="Processing batches",
                    disable=not verbose,
                ) as pbar:
                    for batch in batches:
                        if batch.num_rows > 0:
                            try:
                                # Convert batch to GeoDataFrame with proper geometry handling
                                df = self._batch_to_geodataframe(batch)
                                if not df.empty:
                                    df = self._derive_height_and_floors(df, typical_floor_height_m)
                                    dfs.append(df)
                                    buildings_found += len(df)
                            except Exception as e:
                                if verbose:
                                    print(f"--> Error processing batch: {str(e)}")
                        pbar.update(1)

                if verbose:
                    print(f"--> Found {buildings_found} buildings")

                if not dfs:
                    if verbose:
                        print("--> No buildings found in the area")
                    return gpd.GeoDataFrame()

                # Combine all dataframes
                gdf = pd.concat(dfs, ignore_index=True)

                if verbose:
                    print(f"--> Available columns: {gdf.columns.tolist()}")

                if not gdf.empty:
                    if verbose:
                        print("--> Calculating building footprint areas...")
                    
                    # Get UTM CRS for the area
                    utm_crs = gdf.estimate_utm_crs()
                    if verbose:
                        print(f"--> Using UTM CRS: {utm_crs}")
                    
                    # Calculate footprint areas
                    t.start("area")
                    gdf["bldg_area_footprint_sqm"] = (
                        gdf.to_crs(utm_crs).area
                    )
                    gdf["bldg_area_footprint_sqft"] = gdf["bldg_area_footprint_sqm"] * 10.764  # Convert m² to ft²
                    t.stop("area")
                    
                    if use_cache:
                        t.start("save")
                        gdf.to_parquet(cache_path)
                        t.stop("save")
                        if verbose:
                            print(f"--> Saving buildings to cache: {cache_path}")
                            print(f"--> Building columns = {gdf.columns.values}")
                        gdf.to_parquet(cache_path)

                return gdf

            except Exception as e:
                if verbose:
                    print(f"--> Failed to fetch Overture data: {str(e)}")
                raise

        except Exception as e:
            if verbose:
                print(f"--> Error in get_buildings: {str(e)}")
                print(f"--> Traceback: {traceback.format_exc()}")
            warnings.warn(
                f"Failed to fetch Overture building data: {str(e)}\n{traceback.format_exc()}"
            )
            return gpd.GeoDataFrame()
    
    
    def calculate_building_stats(
        self,
        gdf: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        footprint_units: str,
        footprint_field: str,
        height_units: str,
        height_field: str,
        verbose: bool = False,
    ) -> gpd.GeoDataFrame:
        """Calculate relevant stats for each parcel by intersecting with building geometries.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing parcels
        buildings : gpd.GeoDataFrame
            GeoDataFrame containing building footprints
        footprint_units : str
            Units for area calculation (supported: "sqft", "sqm")
        footprint_field : str
            Field name to write the calculated footprint sizes to
        height_units : str
            Units for height calculation (supported: "ft", "m")
        height_field : str
            Field name to write the calculated height sizes to
        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added building footprint areas
        """
        
        gdf = self.calculate_building_footprints(gdf, buildings, footprint_units, footprint_field, verbose)
        gdf = self.calculate_building_heights(gdf, buildings, height_units, height_field, verbose)
        return gdf
    
    
    def calculate_building_footprints(
        self,
        gdf: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        desired_units: str,
        field_name: str,
        verbose: bool = False,
    ) -> gpd.GeoDataFrame:
        """Calculate building footprint areas for each parcel by intersecting with
        building geometries.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing parcels
        buildings : gpd.GeoDataFrame
            GeoDataFrame containing building footprints
        desired_units : str
            Units for area calculation (supported: "sqft", "sqm")
        field_name : str
            Field name to write the calculated footprint sizes to
        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added building footprint areas
        """
        
        if verbose:
            print("Calculating building footprint areas!")
        
        t = TimingData()
        if buildings.empty:
            if verbose:
                print("--> No buildings found, returning original GeoDataFrame")
            gdf["bldg_area_footprint_sqft"] = 0
            return gdf

        # Get appropriate unit conversion
        unit_mult = 1.0
        if desired_units == "sqft":
            unit_mult = 10.764  # Convert m² to sqft
        elif desired_units == "sqm":
            unit_mult = 1.0
        else:
            raise ValueError(
                f"Unsupported units: {desired_units}. Supported units are 'sqft' and 'sqm'."
            )

        t.start("crs")
        # Get cache path for intersection areas
        cache_path = self._get_cache_path("intersections_area", gdf.total_bounds)

        # Check cache
        if os.path.exists(cache_path):
            if verbose:
                print(f"--> Loading intersection areas from cache: {cache_path}")
            return gpd.read_parquet(cache_path)

        # Convert both to same CRS for spatial operations
        buildings = buildings.to_crs(gdf.crs)

        # Get appropriate CRS for area calculations
        area_crs = get_crs(gdf, "equal_area")

        # Project both datasets to equal area CRS for accurate area calculations
        buildings_projected = buildings.to_crs(area_crs)
        gdf_projected = gdf.to_crs(area_crs)
        t.stop("crs")

        if verbose:
            _t = t.get("crs")
            print(f"--> Projected to equal area CRS...({_t:.2f}s)")

        t.start("join")
        # Perform spatial join to find all building-parcel intersections
        joined = gpd.sjoin(
            gdf_projected, buildings_projected, how="left", predicate="intersects"
        )
        t.stop("join")

        if verbose:
            _t = t.get("join")
            print(
                f"--> Calculated building footprint intersections with parcels...({_t:.2f}s)"
            )

        if verbose:
            print(f"--> Found {len(joined)} potential building-parcel intersections")

        def calculate_intersection_area(row):
            try:
                parcel_geom = gdf_projected.loc[row.name, "geometry"]
                building_idx = row["index_right"]
                if pd.isna(building_idx):
                    return 0.0
                building_geom = buildings_projected.loc[building_idx, "geometry"]
                if parcel_geom.intersects(building_geom):
                    intersection = parcel_geom.intersection(building_geom)
                    return intersection.area * unit_mult  # Convert to desired units
                return 0.0
            except Exception as e:
                if verbose:
                    print(f"Warning: Error calculating intersection area: {e}")
                return 0.0

        t.start("calc_area")
        # TODO: Optimize this step using vectorized operations if possible
        # Calculate intersection areas
        joined[field_name] = joined.apply(calculate_intersection_area, axis=1)
        t.stop("calc_area")

        if verbose:
            _t = t.get("calc_area")
            print(f"--> Calculated precise intersection areas...({_t:.2f}s)")

        # Aggregate total building footprint area per parcel
        t.start("agg")
        agg = joined.groupby("key")[field_name].sum().reset_index()
        t.stop("agg")

        if verbose:
            _t = t.get("agg")
            print(f"--> Aggregated building footprint areas...({_t:.2f}s)")

        t.start("finish")
        # Merge back to original dataframe
        gdf = gdf.merge(agg, on="key", how="left", suffixes=("", "_agg"))

        if f"{field_name}_agg" in gdf.columns:
            # If the original field name existed, then we will stomp with non-null values from the calculated field
            gdf.loc[~gdf[f"{field_name}_agg"].isna(), field_name] = gdf[
                f"{field_name}_agg"
            ]
            gdf.drop(columns=[f"{field_name}_agg"], inplace=True)

        # Fill NaN values with 0 (parcels with no buildings)
        gdf[field_name] = gdf[field_name].fillna(0)
        t.stop("finish")

        if verbose:
            _t = t.get("finish")
            print(f"--> Finished up...({_t:.2f}s)")
            print(f"--> Added building footprint areas to {len(agg)} parcels")
            print(
                f"--> Total building footprint area: {gdf[field_name].sum():,.0f} sqft"
            )
            print(
                f"--> Average building footprint area: {gdf[field_name].mean():,.0f} sqft"
            )
            print(
                f"--> Number of parcels with buildings: {(gdf[field_name] > 0).sum():,}"
            )

        # Save to cache
        if verbose:
            print(f"--> Saving intersection areas to cache: {cache_path}")
        gdf.to_parquet(cache_path)

        return gdf
    
    
    def calculate_building_heights(
        self,
        gdf: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        desired_units: str,
        field_name: str,
        verbose: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Write, per parcel, the max building height and max floors among intersecting buildings.
        - `field_name` will store the height.
        - Also writes 'bldg_stories' for floors.
        """

        if verbose:
            print("Calculating building heights & floors!")

        t = TimingData()

        # Early exits / checks
        if buildings.empty or "height_m_best" not in buildings.columns:
            if verbose:
                print("--> No buildings or missing height_m_best; returning original GeoDataFrame with zeros")
            gdf[field_name] = 0
            gdf["bldg_stories"] = 0
            return gdf

        # Units
        if desired_units == "ft":
            unit_mult = 3.2808399
        elif desired_units == "m":
            unit_mult = 1.0
        else:
            raise ValueError("Unsupported units: {desired_units}. Use 'ft' or 'm'.")

        # Cache
        cache_path = self._get_cache_path("intersections_height", gdf.total_bounds)
        if os.path.exists(cache_path):
            if verbose:
                print(f"--> Loading parcel heights from cache: {cache_path}")
            return gpd.read_parquet(cache_path)

        # Align CRS for spatial ops
        t.start("crs")
        buildings = buildings.to_crs(gdf.crs)
        # Equal-area CRS isn’t strictly required if we’re just picking max values,
        # so we can skip projecting to an equal-area CRS in this fast path.
        t.stop("crs")
        if verbose:
            print(f"--> CRS aligned...({t.get('crs'):.2f}s)")

        # Spatial join: parcels (left) × buildings (right), predicate=intersects
        t.start("join")
        joined = gpd.sjoin(gdf, buildings, how="left", predicate="intersects")
        t.stop("join")
        if verbose:
            print(f"--> Spatial join done...({t.get('join'):.2f}s), rows={len(joined):,}")

        # If nothing matched, return zeros
        if joined["index_right"].isna().all():
            if verbose:
                print("--> No parcel-building intersections; returning zeros")
            gdf[field_name] = 0
            gdf["bldg_stories"] = 0
            return gdf

        # Compute height in requested units directly from attributes
        # (height_m_best is meters)
        joined["_height_out"] = pd.to_numeric(joined["height_m_best"], errors="coerce") * unit_mult
        # Keep floors_best if present; else coerce to numeric
        if "floors_best" in joined.columns:
            joined["_floors_out"] = pd.to_numeric(joined["floors_best"], errors="coerce")
        else:
            joined["_floors_out"] = pd.NA

        # Aggregate per parcel key: choose max height and max floors
        t.start("agg")
        height_agg = joined.groupby("key")["_height_out"].max(min_count=1)
        floors_agg = joined.groupby("key")["_floors_out"].max(min_count=1)
        t.stop("agg")
        if verbose:
            print(f"--> Aggregated heights/floors...({t.get('agg'):.2f}s)")

        # Merge back
        t.start("finish")
        out = gdf.copy()
        out[field_name] = out["key"].map(height_agg).fillna(0)
        out["bldg_stories"] = out["key"].map(floors_agg).fillna(0)
        t.stop("finish")
        if verbose:
            print(f"--> Finished...({t.get('finish'):.2f}s)")

        # Cache result
        if verbose:
            print(f"--> Saving to cache: {cache_path}")
        out.to_parquet(cache_path)
        return out

    
    
    def _derive_height_and_floors(
        self, df: pd.DataFrame, typical_floor_height_m: float = 3.2
    ) -> pd.DataFrame:
        # helper to return a numeric Series or NA series if the column is absent
        def _num_series(name: str):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
            # all-NA float series aligned to df
            return pd.Series(pd.NA, index=df.index, dtype="Float64")

        h  = _num_series("height")
        eh = _num_series("est_height")            # may be all-NA if column is missing
        nf = _num_series("num_floors")

        # Best height (meters): height -> est_height -> floors * typical
        df["height_m_best"] = h                   # safe even if all-NA
        use_eh = df["height_m_best"].isna() & eh.notna()
        df.loc[use_eh, "height_m_best"] = eh

        use_nf = df["height_m_best"].isna() & nf.notna()
        df.loc[use_nf, "height_m_best"] = nf[use_nf] * float(typical_floor_height_m)

        # Convenience feet
        df["height_ft_best"] = df["height_m_best"] * 3.28084

        # Best floors: prefer num_floors; else infer from height
        df["floors_best"] = nf
        infer_mask = df["floors_best"].isna() & df["height_m_best"].notna()
        df.loc[infer_mask, "floors_best"] = (
            (df.loc[infer_mask, "height_m_best"] / float(typical_floor_height_m))
            .round()
            .clip(lower=1)
        )

        # Optional confidences from sources[]
        if "sources" in df.columns:
            def max_conf_for(prop_path: str, src):
                try:
                    items = src or []
                    vals = [
                        s.get("confidence")
                        for s in items
                        if s.get("property") == prop_path and s.get("confidence") is not None
                    ]
                    return max(vals) if vals else None
                except Exception:
                    return None

            df["height_confidence"] = df["sources"].apply(lambda s: max_conf_for("/properties/height", s))
            df["num_floors_confidence"] = df["sources"].apply(lambda s: max_conf_for("/properties/num_floors", s))

        return df
    
    
    def _resolve_latest_buildings_prefix(self) -> str:
        # list under 'release/' and choose the newest folder that has buildings/type=building
        selector = fs.FileSelector("overturemaps-us-west-2/release", recursive=False)
        # list "release/" at bucket root
        releases = self.fs.get_file_info(selector)
        # keep only directories like release/<date>.<n>/
        folders = sorted(
            [r.base_name for r in releases if r.type == fs.FileType.Directory],
            reverse=True
        )
        for rel in folders:
            candidate = f"release/{rel}/theme=buildings/type=building/"
            if self.fs.get_file_info(f"{self.bucket}/{candidate}").type == fs.FileType.Directory:
                return candidate
        raise FileNotFoundError("No buildings release found on S3.")

def init_service_overture(settings: dict) -> OvertureService:
    """Initialize the Overture service.

    Parameters
    ----------
    settings : dict
        Settings Dictionary

    Returns
    -------
    OvertureService
        An initialized OvertureService object
    """
    return OvertureService(settings)
