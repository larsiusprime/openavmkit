from typing import Dict, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import osmnx as ox

from openavmkit.utilities.cache import check_cache, read_cache, write_cache
from openavmkit.utilities.data import clean_series


class OpenStreetMapService:
    """Service for retrieving and processing data from OpenStreetMap.

    Attributes
    ----------
    settings : dict
        Settings dictionary
    features : dict
        Dictionary containing internal features that have been loaded

    """

    def __init__(self, settings: dict = None):
        """Initialize the OpenStreetMap service.

        Parameters
        ----------
        settings : dict
            Configuration settings for the service
        """
        self.settings = settings or {}
        self.features = {}

    def _get_utm_crs(self, bbox: Tuple[float, float, float, float]) -> str:
        """Helper method to get the appropriate UTM CRS for a given bounding box.
        """
        if not all(isinstance(x, (int, float)) for x in bbox):
            raise ValueError(
                f"Invalid bbox coordinates. All values must be numeric. Got: {bbox}"
            )

        # Validate coordinate ranges
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError(
                f"Invalid longitude values. Must be between -180 and 180. Got: min_lon={min_lon}, max_lon={max_lon}"
            )
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError(
                f"Invalid latitude values. Must be between -90 and 90. Got: min_lat={min_lat}, max_lat={max_lat}"
            )

        # Find the appropriate UTM zone based on the centroid of the bbox
        centroid_lon = (min_lon + max_lon) / 2
        centroid_lat = (min_lat + max_lat) / 2

        # Calculate UTM zone
        utm_zone = int((centroid_lon + 180) / 6) + 1
        hemisphere = "north" if centroid_lat >= 0 else "south"
        return (
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        )

    def _get_tags(self, thing: str, config: dict = None):
        if thing == "water_bodies":
            return {
                "natural": ["water", "bay", "strait"],
                "water": ["river", "lake", "reservoir", "canal", "stream"],
            }
        elif thing == "rivers":
            return {
                "water": ["river", "stream"]
            }
        elif thing == "lake":
            return {
                "water": ["lake"]
            }
        elif thing == "water":
            return {
                "natural": ["water", "bay", "strait"],
                "water": ["lake", "reservoir", "canal"]
            }
        elif thing == "transportation":
            return {"railway": ["rail", "subway", "light_rail", "monorail", "tram"]}
        elif thing == "airport":
            return {
               "airport": ["aerodrome"],
               "site": ["airport"]
           }
        elif thing == "educational":
            return {"amenity": ["university", "college"]}
        elif thing == "parks":
            return {
                "leisure": ["park", "garden", "playground"],
                "landuse": ["recreation_ground"],
            }
        elif thing == "golf_courses":
            return {"leisure": ["golf_course"]}
        else:
            if config is not None:
                osm_tags = config.get("osm_tags", None)
                if osm_tags is not None:
                    return osm_tags
                if config.get("osm", False) == True:
                    raise ValueError(f"'{thing}' isn't a built-in type, and you didn't provide an 'osm_tags' entry, so I don't know how to load that from OSM. Please provide custom tags.")
            raise ValueError(f"'{thing}' isn't a built-in type and I wasn't able to load it.")

    def get_features(
        self,
        thing: str,
        bbox: Tuple[float, float, float, float],
        settings: dict,
        use_cache: bool = True,
        gdf: gpd.GeoDataFrame = None
    ) -> gpd.GeoDataFrame:

        if not settings.get("enabled", False):
            return gpd.GeoDataFrame()
        
        is_osm = settings.get("osm", False)
        osm_dir = "geom/osm" if is_osm else "geom/source"

        # if it's from OSM, check if we have already cached this data, AND the settings are the same
        if use_cache and check_cache(
            f"{osm_dir}/{thing}", signature=settings, filetype="df"
        ):
            print(f"----> using cached {thing}")
            # if so return the cached version
            return read_cache(f"{osm_dir}/{thing}", "df")
        
            
        min_area = settings.get("min_area", 10000)
        top_n = settings.get("top_n", 5)

        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])

        try:

            if gdf is None:
                # Get from OSM
                print(f"Getting {thing} from OSM...")
                tags = self._get_tags(thing, settings)
                osm_features = ox.features.features_from_polygon(polygon, tags=tags)

                if osm_features.empty:
                    return gpd.GeoDataFrame()
            else:
                print(f"Getting {thing} from source file...")
                osm_features = gdf
                if "name" not in osm_features:
                    raise ValueError(f"Geodataframe source for \"{thing}\" distances is missing required field \"name\"!")

            # Project to UTM for accurate area calculation
            utm_crs = self._get_utm_crs(bbox)
            osm_features_proj = osm_features.to_crs(utm_crs)

            # Calculate areas and filter by minimum area
            osm_features_proj["area"] = osm_features_proj.geometry.area
            osm_features_filtered = osm_features_proj[
                osm_features_proj["area"] >= min_area
            ]

            if osm_features_filtered.empty:
                return gpd.GeoDataFrame()

            # Project back to WGS84
            osm_features_filtered = osm_features_filtered.to_crs("EPSG:4326")

            # Clean up names
            osm_features_filtered["name"] = osm_features_filtered["name"].fillna(
                f"unnamed_{thing}"
            )
            osm_features_filtered["name"] = (
                osm_features_filtered["name"].astype(str).str.lower().str.replace(" ", "_")
            )
            osm_features_filtered["name"] = clean_series(osm_features_filtered["name"])

            # Create a copy for top N features
            osm_features_top = osm_features_filtered.nlargest(top_n, "area").copy()

            # Store both dataframes
            self.features[f"{thing}"] = osm_features_filtered
            self.features[f"{thing}_top"] = osm_features_top

            # write to cache so we can skip on next run
            write_cache(f"osm/{thing}", osm_features_filtered, settings, "df")

            return osm_features_filtered

        except Exception as e:
            print(f"ERROR in _get_thing: {str(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return gpd.GeoDataFrame()


    def get_elevation_data(
        self, bbox: Tuple[float, float, float, float], resolution: int = 30
    ) -> np.ndarray:
        """Get digital elevation model (DEM) data from USGS.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]):
            Bounding box (min_lon, min_lat, max_lon, max_lat)
        resolution : int
            Resolution in meters (default: 30m)

        Returns
        -------
        np.ndarray
            Elevation data as a 2D array
        """
        # This is a placeholder. In a real implementation, you would use the USGS API
        # or a library like elevation to download DEM data
        # For now, we'll return a dummy array
        print("DEM data retrieval not implemented yet. Using dummy data.")

        # Create a dummy elevation array
        # In a real implementation, this would be replaced with actual DEM data
        lat_range = np.linspace(bbox[1], bbox[3], 100)
        lon_range = np.linspace(bbox[0], bbox[2], 100)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

        # Create a simple elevation model (for demonstration)
        elevation = 100 + 50 * np.sin(lon_grid * 10) + 50 * np.cos(lat_grid * 10)

        return elevation, (lon_range, lat_range)


    def calculate_elevation_stats(
        self,
        gdf: gpd.GeoDataFrame,
        elevation_data: np.ndarray,
        lon_lat_ranges: Tuple[np.ndarray, np.ndarray],
    ) -> pd.DataFrame:
        """Calculate elevation statistics for each parcel.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Parcel GeoDataFrame
        elevation_data : np.ndarray
            Elevation data as a 2D array
        lon_lat_ranges : Tuple[np.ndarray, np.ndarray])
            Longitude and latitude ranges

        Returns
        -------
        pd.DataFrame
            DataFrame containing elevation statistics
        """

        lon_range, lat_range = lon_lat_ranges

        # Initialize arrays for elevation statistics
        avg_elevation = np.full(len(gdf), np.nan)
        avg_slope = np.full(len(gdf), np.nan)

        # For each parcel, calculate elevation statistics
        for i, geom in enumerate(gdf.geometry):
            # Get the bounds of the parcel
            minx, miny, maxx, maxy = geom.bounds

            # Find the indices in the elevation grid that correspond to the parcel bounds
            lon_indices = np.where((lon_range >= minx) & (lon_range <= maxx))[0]
            lat_indices = np.where((lat_range >= miny) & (lat_range <= maxy))[0]

            if len(lon_indices) == 0 or len(lat_indices) == 0:
                continue

            # Extract the elevation data for the parcel
            parcel_elevation = elevation_data[
                lat_indices[0] : lat_indices[-1] + 1,
                lon_indices[0] : lon_indices[-1] + 1,
            ]

            # Calculate average elevation
            avg_elevation[i] = np.mean(parcel_elevation)

            # Calculate slope (simplified)
            # In a real implementation, you would use a more sophisticated method
            if parcel_elevation.shape[0] > 1 and parcel_elevation.shape[1] > 1:
                # Calculate slope in x and y directions
                slope_x = np.gradient(parcel_elevation, axis=1)
                slope_y = np.gradient(parcel_elevation, axis=0)

                # Calculate average slope
                avg_slope[i] = np.mean(np.sqrt(slope_x**2 + slope_y**2))

        # Create a DataFrame with the elevation statistics
        elevation_stats = pd.DataFrame(
            {"avg_elevation": avg_elevation, "avg_slope": avg_slope}, index=gdf.index
        )

        return elevation_stats


    def calculate_distances(
        self, gdf: gpd.GeoDataFrame, features: gpd.GeoDataFrame, feature_type: str
    ) -> pd.DataFrame:
        """Calculate distances to features, both aggregate and specific top N features.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Parcel GeoDataFrame
        features : gpd.GeoDataFrame
            Features GeoDataFrame
        feature_type : str
            Type of feature (e.g., 'water', 'park')

        Returns
        -------
        pd.DataFrame
            DataFrame with distances
        """

        # check if we have already cached this data, AND the settings are the same
        # construct a unique signature:
        signature = {"feature_type": feature_type, "features": hash(features.to_json())}
        if check_cache(
            f"osm/{feature_type}_distances", signature=signature, filetype="df"
        ):
            print("----> using cached distances")
            # if so return the cached version
            return read_cache(f"osm/{feature_type}_distances", "df")

        # Project to UTM for accurate distance calculation
        utm_crs = self._get_utm_crs(gdf.total_bounds)
        gdf_proj = gdf.to_crs(utm_crs)
        features_proj = features.to_crs(utm_crs)

        # Initialize dictionary to store all distance calculations
        distance_data = {}

        # Calculate aggregate distance (distance to nearest feature of any type)
        distance_data[f"dist_to_{feature_type}_any"] = gdf_proj.geometry.apply(
            lambda g: features_proj.geometry.distance(g).min()
        )

        # Calculate distances to top N features if available
        if f"{feature_type}_top" in self.features:
            top_features = self.features[f"{feature_type}_top"]
            for _, feature in top_features.iterrows():
                feature_name = feature["name"]
                feature_geom = feature.geometry
                feature_proj = gpd.GeoSeries([feature_geom]).to_crs(utm_crs)[0]

                distance_data[f"dist_to_{feature_type}_{feature_name}"] = (
                    gdf_proj.geometry.apply(lambda g: feature_proj.distance(g))
                )

        # write to cache so we can skip on next run
        write_cache(f"osm/{feature_type}_distances", signature, distance_data, "df")

        # Create DataFrame from all collected distances at once
        return pd.DataFrame(distance_data, index=gdf.index)


    def enrich_parcels(
        self,
        gdf: gpd.GeoDataFrame,
        settings: Dict
    ) -> Dict[str, gpd.GeoDataFrame]:
        """Get OpenStreetMap features and prepare them for spatial joins. Returns a
        dictionary of feature dataframes for use by data.py's spatial join logic.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Parcel GeoDataFrame (used for bbox)
        settings : dict
            Settings for enrichment

        Returns
        -------
        dict[str, gpd.GeoDataFrame]
            Dictionary of feature dataframes
        """
        # Get the bounding box of the GeoDataFrame
        bbox = gdf.total_bounds

        # Dictionary to store all dataframes
        dataframes = {}

        # Process each feature type based on settings
        if settings.get("water_bodies", {}).get("enabled", False):
            water_bodies = self.get_features(bbox, "water_bodies", settings["water_bodies"])
            if not water_bodies.empty:
                # Store both the main and top features in dataframes
                dataframes["water_bodies"] = self.features["water_bodies"]
                dataframes["water_bodies_top"] = self.features["water_bodies_top"]

        if settings.get("transportation", {}).get("enabled", False):
            transportation = self.get_features(bbox, "transportation", settings["transportation"])
            if not transportation.empty:
                dataframes["transportation"] = self.features["transportation"]
                dataframes["transportation_top"] = self.features["transportation_top"]

        if settings.get("educational", {}).get("enabled", False):
            institutions = self.get_features(bbox, "educational", settings["educational"])
            if not institutions.empty:
                dataframes["educational"] = self.features["educational"]
                dataframes["educational_top"] = self.features["educational_top"]

        if settings.get("parks", {}).get("enabled", False):
            parks = self.get_features(bbox, "parks", settings["parks"])
            if not parks.empty:
                dataframes["parks"] = self.features["parks"]
                dataframes["parks_top"] = self.features["parks_top"]

        if settings.get("golf_courses", {}).get("enabled", False):
            golf_courses = self.get_features(bbox, "golf_courses", settings["golf_courses"])
            if not golf_courses.empty:
                dataframes["golf_courses"] = self.features["golf_courses"]
                dataframes["golf_courses_top"] = self.features["golf_courses_top"]

        return dataframes


def init_service_openstreetmap(settings: Dict = None) -> OpenStreetMapService:
    """Initialize an OpenStreetMap service with the provided settings.

    Parameters
    ----------
    settings : dict
        Configuration settings for the service

    Returns
    -------
    OpenStreetMapService
        Initialized OpenStreetMap service
    """
    return OpenStreetMapService(settings)
