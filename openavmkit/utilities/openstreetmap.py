import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import geopandas as gpd
import requests
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
import json
import osmnx as ox
from openavmkit.utilities.geometry import distance_km, get_crs, stamp_geo_field_onto_df

class OpenStreetMapService:
    """
    Service for retrieving and processing data from OpenStreetMap.
    """
    
    def __init__(self, settings: Dict = None):
        """
        Initialize the OpenStreetMap service.
        
        Args:
            settings (Dict): Configuration settings for the service
        """
        self.settings = settings or {}
        self.features = {}
        self.dem_data = None
    
    def _get_utm_crs(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        Helper method to get the appropriate UTM CRS for a given bounding box.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            str: UTM CRS string
        """
        # Find the appropriate UTM zone based on the centroid of the bbox
        centroid_lon = (bbox[0] + bbox[2]) / 2
        centroid_lat = (bbox[1] + bbox[3]) / 2
        
        # Calculate UTM zone
        utm_zone = int((centroid_lon + 180) / 6) + 1
        hemisphere = 'north' if centroid_lat >= 0 else 'south'
        return f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"

    def get_water_bodies(self, bbox: Tuple[float, float, float, float], min_area: float = 10000) -> gpd.GeoDataFrame:
        """
        Get water bodies (rivers, lakes, etc.) from OpenStreetMap.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            min_area (float): Minimum area in square meters for water bodies to include
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing water bodies
        """
        print(f"[DEBUG] Getting water bodies with bbox: {bbox}, min_area: {min_area}")
        
        # Define tags for water bodies
        tags = {
            'natural': ['water', 'bay', 'strait'],
            'water': ['river', 'lake', 'reservoir', 'canal', 'stream']
        }
        print(f"[DEBUG] Using tags: {tags}")
        
        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        print(f"[DEBUG] Created polygon from bbox: {polygon}")
        
        # Get water bodies from OSM
        print("[DEBUG] Fetching water bodies from OSM...")
        water_bodies = ox.features.features_from_polygon(
            polygon,
            tags=tags
        )
        
        if water_bodies.empty:
            print("[DEBUG] No water bodies found, returning empty GeoDataFrame")
            return gpd.GeoDataFrame()
        
        print(f"[DEBUG] Found {len(water_bodies)} water bodies before filtering")
        
        # Project to UTM for accurate area calculation
        utm_crs = self._get_utm_crs(bbox)
        print(f"[DEBUG] Using UTM CRS: {utm_crs}")
        water_bodies_proj = water_bodies.to_crs(utm_crs)
        
        # Filter by minimum area (now in square meters)
        water_bodies_proj['area'] = water_bodies_proj.geometry.area
        print(f"[DEBUG] Area range: min={water_bodies_proj['area'].min()}, max={water_bodies_proj['area'].max()}")
        water_bodies_filtered = water_bodies_proj[water_bodies_proj['area'] >= min_area]
        print(f"[DEBUG] After area filtering: {len(water_bodies_filtered)} water bodies remain")
        
        # Project back to WGS84 for consistency
        if not water_bodies_filtered.empty:
            print("[DEBUG] Projecting back to WGS84")
            water_bodies_filtered = water_bodies_filtered.to_crs('EPSG:4326')
        
        # Simplify geometries for better performance
        print("[DEBUG] Simplifying geometries")
        water_bodies_filtered.geometry = water_bodies_filtered.geometry.simplify(0.0001)
        
        print(f"[DEBUG] Returning {len(water_bodies_filtered)} water bodies")
        return water_bodies_filtered
    def get_transportation(self, bbox: Tuple[float, float, float, float], min_length: float = 100) -> gpd.GeoDataFrame:
        """
        Get transportation networks (roads, railways) from OpenStreetMap.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            min_length (float): Minimum length in meters for transportation features to include
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing transportation networks
        """
        # Define tags for transportation
        tags = {
            'highway': True,
            'railway': True
        }
        
        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get transportation from OSM
        transportation = ox.features.features_from_polygon(
            polygon,
            tags=tags
        )
        
        if transportation.empty:
            return gpd.GeoDataFrame()
        
        # Project to UTM for accurate length calculation
        utm_crs = self._get_utm_crs(bbox)
        transportation_proj = transportation.to_crs(utm_crs)
        
        # Filter by minimum length
        transportation_proj['length'] = transportation_proj.geometry.length
        transportation_filtered = transportation_proj[transportation_proj['length'] >= min_length]
        
        # Project back to WGS84 for consistency
        if not transportation_filtered.empty:
            transportation_filtered = transportation_filtered.to_crs('EPSG:4326')
        
        # Simplify geometries for better performance
        transportation_filtered.geometry = transportation_filtered.geometry.simplify(0.0001)
        
        return transportation_filtered
    
    def get_elevation_data(self, bbox: Tuple[float, float, float, float], resolution: int = 30) -> np.ndarray:
        """
        Get digital elevation model (DEM) data from USGS.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            resolution (int): Resolution in meters (default: 30m)
            
        Returns:
            np.ndarray: Elevation data as a 2D array
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
    
    def get_educational_institutions(self, bbox: Tuple[float, float, float, float], min_area: float = 1000) -> gpd.GeoDataFrame:
        """
        Get colleges and universities from OpenStreetMap.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            min_area (float): Minimum area in square meters for institutions to include
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing educational institutions
        """
        # Define tags for educational institutions
        tags = {
            'amenity': ['university', 'college', 'school'],
            'building': ['university', 'college']
        }
        
        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get educational institutions from OSM
        institutions = ox.features.features_from_polygon(
            polygon,
            tags=tags
        )
        
        if institutions.empty:
            return gpd.GeoDataFrame()
        
        # Filter to keep only higher education institutions
        institutions = institutions[
            (institutions['amenity'].isin(['university', 'college'])) |
            (institutions['building'].isin(['university', 'college']))
        ]
        
        if not institutions.empty:
            # Project to UTM for accurate area calculation
            utm_crs = self._get_utm_crs(bbox)
            institutions_proj = institutions.to_crs(utm_crs)
            
            # Filter by minimum area for polygon geometries
            institutions_proj['area'] = institutions_proj.geometry.area
            institutions_filtered = institutions_proj[
                (institutions_proj.geometry.geom_type == 'Point') |
                ((institutions_proj.geometry.geom_type != 'Point') & (institutions_proj['area'] >= min_area))
            ]
            
            # Project back to WGS84 for consistency
            institutions_filtered = institutions_filtered.to_crs('EPSG:4326')
            
            # Simplify geometries for better performance (only for non-point geometries)
            mask = institutions_filtered.geometry.geom_type != 'Point'
            institutions_filtered.loc[mask, 'geometry'] = institutions_filtered.loc[mask, 'geometry'].simplify(0.0001)
            
            return institutions_filtered
        
        return institutions
    
    def get_parks(self, bbox: Tuple[float, float, float, float], min_area: float = 1000) -> gpd.GeoDataFrame:
        """
        Get parks from OpenStreetMap.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            min_area (float): Minimum area in square meters for parks to include
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing parks
        """
        # Define tags for parks
        tags = {
            'leisure': ['park', 'garden', 'playground'],
            'landuse': ['recreation_ground']
        }
        
        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get parks from OSM
        parks = ox.features.features_from_polygon(
            polygon,
            tags=tags
        )
        
        if parks.empty:
            return gpd.GeoDataFrame()
        
        # Project to UTM for accurate area calculation
        utm_crs = self._get_utm_crs(bbox)
        parks_proj = parks.to_crs(utm_crs)
        
        # Filter by minimum area
        parks_proj['area'] = parks_proj.geometry.area
        parks_filtered = parks_proj[parks_proj['area'] >= min_area]
        
        # Project back to WGS84 for consistency
        if not parks_filtered.empty:
            parks_filtered = parks_filtered.to_crs('EPSG:4326')
            
            # Simplify geometries for better performance
            parks_filtered.geometry = parks_filtered.geometry.simplify(0.0001)
        
        return parks_filtered
    
    def get_golf_courses(self, bbox: Tuple[float, float, float, float], min_area: float = 10000) -> gpd.GeoDataFrame:
        """
        Get golf courses from OpenStreetMap.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            min_area (float): Minimum area in square meters for golf courses to include
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing golf courses
        """
        # Define tags for golf courses
        tags = {
            'leisure': ['golf_course']
        }
        
        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get golf courses from OSM
        golf_courses = ox.features.features_from_polygon(
            polygon,
            tags=tags
        )
        
        if golf_courses.empty:
            return gpd.GeoDataFrame()
        
        # Project to UTM for accurate area calculation
        utm_crs = self._get_utm_crs(bbox)
        golf_courses_proj = golf_courses.to_crs(utm_crs)
        
        # Filter by minimum area
        golf_courses_proj['area'] = golf_courses_proj.geometry.area
        golf_courses_filtered = golf_courses_proj[golf_courses_proj['area'] >= min_area]
        
        # Project back to WGS84 for consistency
        if not golf_courses_filtered.empty:
            golf_courses_filtered = golf_courses_filtered.to_crs('EPSG:4326')
            
            # Simplify geometries for better performance
            golf_courses_filtered.geometry = golf_courses_filtered.geometry.simplify(0.0001)
        
        return golf_courses_filtered
    
    def calculate_elevation_stats(self, gdf: gpd.GeoDataFrame, elevation_data: np.ndarray, 
                                 lon_lat_ranges: Tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
        """
        Calculate elevation statistics for each parcel.
        
        Args:
            gdf (gpd.GeoDataFrame): Parcel GeoDataFrame
            elevation_data (np.ndarray): Elevation data as a 2D array
            lon_lat_ranges (Tuple[np.ndarray, np.ndarray]): Longitude and latitude ranges
            
        Returns:
            pd.DataFrame: DataFrame containing elevation statistics
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
            parcel_elevation = elevation_data[lat_indices[0]:lat_indices[-1]+1, 
                                             lon_indices[0]:lon_indices[-1]+1]
            
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
        elevation_stats = pd.DataFrame({
            'avg_elevation': avg_elevation,
            'avg_slope': avg_slope
        }, index=gdf.index)
        
        return elevation_stats
    
    def enrich_parcels(self, gdf: gpd.GeoDataFrame, settings: Dict) -> gpd.GeoDataFrame:
        """
        Enrich parcels with OpenStreetMap data based on settings.
        
        Args:
            gdf (gpd.GeoDataFrame): Parcel GeoDataFrame
            settings (Dict): Settings for enrichment
            
        Returns:
            gpd.GeoDataFrame: Enriched GeoDataFrame
        """
        # Make a copy of the input GeoDataFrame
        enriched_gdf = gdf.copy()
        
        # Get the bounding box of the GeoDataFrame
        bbox = enriched_gdf.total_bounds
        
        # Process each feature type based on settings
        if settings.get('water_bodies', False):
            water_bodies = self.get_water_bodies(bbox, min_area=settings.get('water_bodies_min_area', 10000))
            if not water_bodies.empty:
                # Add unique identifier if not present
                if 'id' not in water_bodies.columns:
                    water_bodies['id'] = range(len(water_bodies))
                
                # Buffer the water bodies
                utm_crs = self._get_utm_crs(bbox)
                water_bodies_proj = water_bodies.to_crs(utm_crs)
                water_bodies_proj['geometry'] = water_bodies_proj.geometry.buffer(1000)  # 1km buffer
                water_bodies = water_bodies_proj.to_crs('EPSG:4326')
                
                # Spatial join to find parcels near water
                near_water = gpd.sjoin(enriched_gdf, water_bodies, how='inner', predicate='intersects')
                near_water_ids = near_water.index.unique()
                
                # Calculate distances only for parcels near water
                if len(near_water_ids) > 0:
                    near_water_gdf = enriched_gdf.loc[near_water_ids]
                    water_distances = self.calculate_distances(near_water_gdf, water_bodies, 'water')
                    enriched_gdf.loc[near_water_ids, 'distance_to_water'] = water_distances
                    
                    # Add waterfront flags
                    enriched_gdf['is_waterfront'] = False
                    enriched_gdf.loc[near_water_ids, 'is_waterfront'] = enriched_gdf.loc[near_water_ids, 'distance_to_water'] <= 0.1  # 100m threshold
                    
                    # Stamp water body characteristics
                    water_fields = ['name', 'water', 'natural']  # Add more fields as needed
                    enriched_gdf = stamp_geo_field_onto_df(enriched_gdf, water_bodies, water_fields, 'water_')
        
        if settings.get('transportation', False):
            transportation = self.get_transportation(bbox, min_length=settings.get('transportation_min_length', 100))
            if not transportation.empty:
                # Buffer the transportation features
                utm_crs = self._get_utm_crs(bbox)
                transportation_proj = transportation.to_crs(utm_crs)
                transportation_proj['geometry'] = transportation_proj.geometry.buffer(500)  # 500m buffer
                transportation = transportation_proj.to_crs('EPSG:4326')
                
                # Spatial join to find parcels near transportation
                near_transport = gpd.sjoin(enriched_gdf, transportation, how='inner', predicate='intersects')
                near_transport_ids = near_transport.index.unique()
                
                # Calculate distances only for parcels near transportation
                if len(near_transport_ids) > 0:
                    near_transport_gdf = enriched_gdf.loc[near_transport_ids]
                    transport_distances = self.calculate_distances(near_transport_gdf, transportation, 'transportation')
                    enriched_gdf.loc[near_transport_ids, 'distance_to_transportation'] = transport_distances
        
        if settings.get('educational', False):
            institutions = self.get_educational_institutions(bbox, min_area=settings.get('educational_min_area', 1000))
            if not institutions.empty:
                # Add unique identifier if not present
                if 'id' not in institutions.columns:
                    institutions['id'] = range(len(institutions))
                
                # Buffer the institutions
                utm_crs = self._get_utm_crs(bbox)
                institutions_proj = institutions.to_crs(utm_crs)
                institutions_proj['geometry'] = institutions_proj.geometry.buffer(1000)  # 1km buffer
                institutions = institutions_proj.to_crs('EPSG:4326')
                
                # Spatial join to find parcels near institutions
                near_institutions = gpd.sjoin(enriched_gdf, institutions, how='inner', predicate='intersects')
                near_institutions_ids = near_institutions.index.unique()
                
                # Calculate distances only for parcels near institutions
                if len(near_institutions_ids) > 0:
                    near_institutions_gdf = enriched_gdf.loc[near_institutions_ids]
                    institution_distances = self.calculate_distances(near_institutions_gdf, institutions, 'educational')
                    enriched_gdf.loc[near_institutions_ids, 'distance_to_educational'] = institution_distances
                    
                    # Stamp institution characteristics
                    institution_fields = ['name', 'amenity', 'building']  # Add more fields as needed
                    enriched_gdf = stamp_geo_field_onto_df(enriched_gdf, institutions, institution_fields, 'educational_')
        
        if settings.get('parks', False):
            parks = self.get_parks(bbox, min_area=settings.get('park_min_area', 1000))
            if not parks.empty:
                # Add unique identifier if not present
                if 'id' not in parks.columns:
                    parks['id'] = range(len(parks))
                
                # Buffer the parks
                utm_crs = self._get_utm_crs(bbox)
                parks_proj = parks.to_crs(utm_crs)
                parks_proj['geometry'] = parks_proj.geometry.buffer(1000)  # 1km buffer
                parks = parks_proj.to_crs('EPSG:4326')
                
                # Spatial join to find parcels near parks
                near_parks = gpd.sjoin(enriched_gdf, parks, how='inner', predicate='intersects')
                near_parks_ids = near_parks.index.unique()
                
                # Calculate distances only for parcels near parks
                if len(near_parks_ids) > 0:
                    near_parks_gdf = enriched_gdf.loc[near_parks_ids]
                    park_distances = self.calculate_distances(near_parks_gdf, parks, 'park')
                    enriched_gdf.loc[near_parks_ids, 'distance_to_park'] = park_distances
                    
                    # Stamp park characteristics
                    park_fields = ['name', 'leisure', 'landuse']  # Add more fields as needed
                    enriched_gdf = stamp_geo_field_onto_df(enriched_gdf, parks, park_fields, 'park_')
        
        if settings.get('golf_courses', False):
            golf_courses = self.get_golf_courses(bbox, min_area=settings.get('golf_course_min_area', 10000))
            if not golf_courses.empty:
                # Add unique identifier if not present
                if 'id' not in golf_courses.columns:
                    golf_courses['id'] = range(len(golf_courses))
                
                # Buffer the golf courses
                utm_crs = self._get_utm_crs(bbox)
                golf_courses_proj = golf_courses.to_crs(utm_crs)
                golf_courses_proj['geometry'] = golf_courses_proj.geometry.buffer(1000)  # 1km buffer
                golf_courses = golf_courses_proj.to_crs('EPSG:4326')
                
                # Spatial join to find parcels near golf courses
                near_golf = gpd.sjoin(enriched_gdf, golf_courses, how='inner', predicate='intersects')
                near_golf_ids = near_golf.index.unique()
                
                # Calculate distances only for parcels near golf courses
                if len(near_golf_ids) > 0:
                    near_golf_gdf = enriched_gdf.loc[near_golf_ids]
                    golf_distances = self.calculate_distances(near_golf_gdf, golf_courses, 'golf_course')
                    enriched_gdf.loc[near_golf_ids, 'distance_to_golf_course'] = golf_distances
                    
                    # Stamp golf course characteristics
                    golf_fields = ['name', 'leisure']  # Add more fields as needed
                    enriched_gdf = stamp_geo_field_onto_df(enriched_gdf, golf_courses, golf_fields, 'golf_')
        
        return enriched_gdf

def init_service_openstreetmap(settings: Dict = None) -> OpenStreetMapService:
    """
    Initialize an OpenStreetMap service with the provided settings.
    
    Args:
        settings (Dict): Configuration settings for the service
        
    Returns:
        OpenStreetMapService: Initialized OpenStreetMap service
    """
    return OpenStreetMapService(settings) 