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

    def get_water_bodies(self, bbox: Tuple[float, float, float, float], settings: dict) -> gpd.GeoDataFrame:
        """
        Get water bodies (rivers, lakes, etc.) from OpenStreetMap.
        Stores both all water bodies and top N largest ones for distance calculations.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            settings (dict): Settings for water bodies including min_area and top_n
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing all water bodies
        """
        if not settings.get('enabled', False):
            return gpd.GeoDataFrame()
        
        min_area = settings.get('min_area', 10000)
        top_n = settings.get('top_n', 5)
        
        # Define tags for water bodies
        tags = {
            'natural': ['water', 'bay', 'strait'],
            'water': ['river', 'lake', 'reservoir', 'canal', 'stream']
        }
        
        # Create polygon from bbox
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get water bodies from OSM
        water_bodies = ox.features.features_from_polygon(
            polygon,
            tags=tags
        )
        
        if water_bodies.empty:
            return gpd.GeoDataFrame()
        
        # Project to UTM for accurate area calculation
        utm_crs = self._get_utm_crs(bbox)
        water_bodies_proj = water_bodies.to_crs(utm_crs)
        
        # Calculate areas and filter by minimum area
        water_bodies_proj['area'] = water_bodies_proj.geometry.area
        water_bodies_filtered = water_bodies_proj[water_bodies_proj['area'] >= min_area]
        
        if water_bodies_filtered.empty:
            return gpd.GeoDataFrame()
        
        # Project back to WGS84
        water_bodies_filtered = water_bodies_filtered.to_crs('EPSG:4326')
        
        # Clean up names
        water_bodies_filtered['name'] = water_bodies_filtered['name'].fillna('unnamed_water_body')
        water_bodies_filtered['name'] = water_bodies_filtered['name'].str.lower().str.replace(' ', '_')
        
        # Create a copy for top N features
        water_bodies_top = water_bodies_filtered.nlargest(top_n, 'area').copy()
        
        # Store both dataframes
        self.features['water_bodies'] = water_bodies_filtered
        self.features['water_bodies_top'] = water_bodies_top
        
        return water_bodies_filtered

    def get_transportation(self, bbox: Tuple[float, float, float, float], settings: dict) -> gpd.GeoDataFrame:
        """
        Get major transportation networks (roads, railways) from OpenStreetMap.
        Stores both all routes and top N longest ones for distance calculations.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            settings (dict): Settings for transportation including min_length and top_n
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing all transportation routes
        """
        if not settings.get('enabled', False):
            return gpd.GeoDataFrame()
            
        min_length = settings.get('min_length', 1000)
        top_n = settings.get('top_n', 5)
        
        # Define tags for major transportation routes
        tags = {
            'highway': ['motorway', 'trunk', 'primary', 'secondary', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link'],
            'railway': ['rail', 'subway', 'light_rail', 'monorail', 'tram']
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
        
        # Calculate lengths and filter by minimum length
        transportation_proj['length'] = transportation_proj.geometry.length
        transportation_filtered = transportation_proj[transportation_proj['length'] >= min_length]
        
        if transportation_filtered.empty:
            return gpd.GeoDataFrame()
        
        # Project back to WGS84
        transportation_filtered = transportation_filtered.to_crs('EPSG:4326')
        
        # Clean up names
        transportation_filtered['name'] = transportation_filtered['name'].fillna('unnamed_route')
        transportation_filtered['name'] = transportation_filtered['name'].str.lower().str.replace(' ', '_')
        
        # Create a copy for top N features
        transportation_top = transportation_filtered.nlargest(top_n, 'length').copy()
        
        # Store both dataframes
        self.features['transportation'] = transportation_filtered
        self.features['transportation_top'] = transportation_top
        
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
    
    def get_educational_institutions(self, bbox: Tuple[float, float, float, float], settings: dict) -> gpd.GeoDataFrame:
        """
        Get educational institutions from OpenStreetMap.
        Stores both all institutions and top N largest ones for distance calculations.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            settings (dict): Settings for educational institutions including min_area and top_n
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing all educational institutions
        """
        if not settings.get('enabled', False):
            return gpd.GeoDataFrame()
            
        min_area = settings.get('min_area', 1000)
        top_n = settings.get('top_n', 5)
        
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
            
        # Project to UTM for accurate area calculation
        utm_crs = self._get_utm_crs(bbox)
        institutions_proj = institutions.to_crs(utm_crs)
        
        # Calculate areas and filter by minimum area
        institutions_proj['area'] = institutions_proj.geometry.area
        institutions_filtered = institutions_proj[institutions_proj['area'] >= min_area]
        
        if institutions_filtered.empty:
            return gpd.GeoDataFrame()
            
        # Project back to WGS84
        institutions_filtered = institutions_filtered.to_crs('EPSG:4326')
        
        # Clean up names
        institutions_filtered['name'] = institutions_filtered['name'].fillna('unnamed_institution')
        institutions_filtered['name'] = institutions_filtered['name'].str.lower().str.replace(' ', '_')
        
        # Create a copy for top N features
        institutions_top = institutions_filtered.nlargest(top_n, 'area').copy()
        
        # Store both dataframes
        self.features['educational'] = institutions_filtered
        self.features['educational_top'] = institutions_top
        
        return institutions_filtered
    
    def get_parks(self, bbox: Tuple[float, float, float, float], settings: dict) -> gpd.GeoDataFrame:
        """
        Get parks from OpenStreetMap.
        Stores both all parks and top N largest ones for distance calculations.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            settings (dict): Settings for parks including min_area and top_n
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing all parks
        """
        if not settings.get('enabled', False):
            return gpd.GeoDataFrame()
            
        min_area = settings.get('min_area', 1000)
        top_n = settings.get('top_n', 5)
        
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
        
        # Calculate areas and filter by minimum area
        parks_proj['area'] = parks_proj.geometry.area
        parks_filtered = parks_proj[parks_proj['area'] >= min_area]
        
        if parks_filtered.empty:
            return gpd.GeoDataFrame()
            
        # Project back to WGS84
        parks_filtered = parks_filtered.to_crs('EPSG:4326')
        
        # Clean up names
        parks_filtered['name'] = parks_filtered['name'].fillna('unnamed_park')
        parks_filtered['name'] = parks_filtered['name'].str.lower().str.replace(' ', '_')
        
        # Create a copy for top N features
        parks_top = parks_filtered.nlargest(top_n, 'area').copy()
        
        # Store both dataframes
        self.features['parks'] = parks_filtered
        self.features['parks_top'] = parks_top
        
        return parks_filtered
    
    def get_golf_courses(self, bbox: Tuple[float, float, float, float], settings: dict) -> gpd.GeoDataFrame:
        """
        Get golf courses from OpenStreetMap.
        Stores both all golf courses and top N largest ones for distance calculations.
        
        Args:
            bbox (Tuple[float, float, float, float]): Bounding box (min_lon, min_lat, max_lon, max_lat)
            settings (dict): Settings for golf courses including min_area and top_n
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing all golf courses
        """
        if not settings.get('enabled', False):
            return gpd.GeoDataFrame()
            
        min_area = settings.get('min_area', 10000)
        top_n = settings.get('top_n', 3)
        
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
        
        # Calculate areas and filter by minimum area
        golf_courses_proj['area'] = golf_courses_proj.geometry.area
        golf_courses_filtered = golf_courses_proj[golf_courses_proj['area'] >= min_area]
        
        if golf_courses_filtered.empty:
            return gpd.GeoDataFrame()
            
        # Project back to WGS84
        golf_courses_filtered = golf_courses_filtered.to_crs('EPSG:4326')
        
        # Clean up names
        golf_courses_filtered['name'] = golf_courses_filtered['name'].fillna('unnamed_golf_course')
        golf_courses_filtered['name'] = golf_courses_filtered['name'].str.lower().str.replace(' ', '_')
        
        # Create a copy for top N features
        golf_courses_top = golf_courses_filtered.nlargest(top_n, 'area').copy()
        
        # Store both dataframes
        self.features['golf_courses'] = golf_courses_filtered
        self.features['golf_courses_top'] = golf_courses_top
        
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
    
    def calculate_distances(self, gdf: gpd.GeoDataFrame, features: gpd.GeoDataFrame, feature_type: str) -> pd.DataFrame:
        """
        Calculate distances to features, both aggregate and specific top N features.
        
        Args:
            gdf (gpd.GeoDataFrame): Parcel GeoDataFrame
            features (gpd.GeoDataFrame): Features GeoDataFrame
            feature_type (str): Type of feature (e.g., 'water', 'park')
            
        Returns:
            pd.DataFrame: DataFrame with distances
        """
        # Project to UTM for accurate distance calculation
        utm_crs = self._get_utm_crs(gdf.total_bounds)
        gdf_proj = gdf.to_crs(utm_crs)
        features_proj = features.to_crs(utm_crs)
        
        # Calculate aggregate distance (distance to nearest feature of any type)
        distances = pd.DataFrame(index=gdf.index)
        distances[f'dist_to_{feature_type}_any'] = gdf_proj.geometry.apply(
            lambda g: features_proj.geometry.distance(g).min()
        )
        
        # Calculate distances to top N features if available
        if f'{feature_type}_top' in self.features:
            top_features = self.features[f'{feature_type}_top']
            for _, feature in top_features.iterrows():
                feature_name = feature['name']
                feature_geom = feature.geometry
                feature_proj = gpd.GeoSeries([feature_geom]).to_crs(utm_crs)[0]
                
                distances[f'dist_to_{feature_type}_{feature_name}'] = gdf_proj.geometry.apply(
                    lambda g: feature_proj.distance(g)
                )
        
        return distances

    def enrich_parcels(self, gdf: gpd.GeoDataFrame, settings: Dict) -> Dict[str, gpd.GeoDataFrame]:
        """
        Get OpenStreetMap features and prepare them for spatial joins.
        Returns a dictionary of feature dataframes for use by data.py's spatial join logic.
        
        Args:
            gdf (gpd.GeoDataFrame): Parcel GeoDataFrame (used for bbox)
            settings (Dict): Settings for enrichment
            
        Returns:
            Dict[str, gpd.GeoDataFrame]: Dictionary of feature dataframes
        """
        # Get the bounding box of the GeoDataFrame
        bbox = gdf.total_bounds
        
        # Dictionary to store all dataframes
        dataframes = {}
        
        # Process each feature type based on settings
        if settings.get('water_bodies', {}).get('enabled', False):
            water_bodies = self.get_water_bodies(bbox, settings['water_bodies'])
            if not water_bodies.empty:
                # Store both the main and top features in dataframes
                dataframes['water_bodies'] = self.features['water_bodies']
                dataframes['water_bodies_top'] = self.features['water_bodies_top']
        
        if settings.get('transportation', {}).get('enabled', False):
            transportation = self.get_transportation(bbox, settings['transportation'])
            if not transportation.empty:
                dataframes['transportation'] = self.features['transportation']
                dataframes['transportation_top'] = self.features['transportation_top']
        
        if settings.get('educational', {}).get('enabled', False):
            institutions = self.get_educational_institutions(bbox, settings['educational'])
            if not institutions.empty:
                dataframes['educational'] = self.features['educational']
                dataframes['educational_top'] = self.features['educational_top']
        
        if settings.get('parks', {}).get('enabled', False):
            parks = self.get_parks(bbox, settings['parks'])
            if not parks.empty:
                dataframes['parks'] = self.features['parks']
                dataframes['parks_top'] = self.features['parks_top']
        
        if settings.get('golf_courses', {}).get('enabled', False):
            golf_courses = self.get_golf_courses(bbox, settings['golf_courses'])
            if not golf_courses.empty:
                dataframes['golf_courses'] = self.features['golf_courses']
                dataframes['golf_courses_top'] = self.features['golf_courses_top']
        
        return dataframes

def init_service_openstreetmap(settings: Dict = None) -> OpenStreetMapService:
    """
    Initialize an OpenStreetMap service with the provided settings.
    
    Args:
        settings (Dict): Configuration settings for the service
        
    Returns:
        OpenStreetMapService: Initialized OpenStreetMap service
    """
    return OpenStreetMapService(settings) 