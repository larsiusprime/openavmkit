"""
OpenStreetMap data fetching service.

Wraps ``osmnx`` to download tagged OSM features (parks, water bodies, schools,
streets, transportation networks, etc.) for a given bounding box, with
caching to avoid re-downloading on subsequent runs. Used by the distance
enrichment (``data.process.enrich.distances``) and the streets enrichment
(``data.process.enrich.streets``) when ``osm: true`` is configured for a
feature.
"""
from typing import Dict, Tuple
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
                "water": ["river", "reservoir", "canal", "stream"],
            }
        elif thing == "rivers":
            return {
                "water": ["river", "stream"]
            }
        elif thing == "water":
            return {
                "natural": ["water", "bay", "strait"],
                "water": ["reservoir", "canal"]
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
        elif thing == "coastline":
            return {"natural": ["coastline"]}
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
        min_length = settings.get("min_length", 0)
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

            # Identify geometry types
            geom = osm_features_proj.geometry
            is_poly = geom.geom_type.isin(["Polygon", "MultiPolygon"])
            is_line = geom.geom_type.isin(["LineString", "MultiLineString"])
            
            # Compute metrics (only where they make sense)
            osm_features_proj["area"] = 0.0
            osm_features_proj.loc[is_poly, "area"] = geom[is_poly].area
            
            osm_features_proj["length"] = 0.0
            osm_features_proj.loc[is_line, "length"] = geom[is_line].length
            
            # Keep anything that meets either threshold
            osm_features_filtered = osm_features_proj[
                (osm_features_proj["area"] >= min_area) |
                (osm_features_proj["length"] >= min_length)
            ].copy()

            if osm_features_filtered.empty:
                return gpd.GeoDataFrame()

            # Project back to WGS84
            osm_features_filtered = osm_features_filtered.to_crs("EPSG:4326")

            # Clean up names. osmnx only includes a "name" column when at least one
            # returned feature carries a name tag; for feature classes that are commonly
            # unnamed (e.g. riverbank polygons from water=river/stream), the column can be
            # absent entirely, which previously raised KeyError here. Backfill it.
            if "name" not in osm_features_filtered.columns:
                osm_features_filtered["name"] = np.nan
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

            # write to cache so we can skip on next run -- must use the same osm_dir
            # the cache check above reads from, or the cache can never hit
            write_cache(f"{osm_dir}/{thing}", osm_features_filtered, settings, "df")

            return osm_features_filtered

        except Exception as e:
            print(f"ERROR in _get_thing: {str(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return gpd.GeoDataFrame()


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
