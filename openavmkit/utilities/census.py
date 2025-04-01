import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import geopandas as gpd
from openavmkit.cloud.census import Census
import requests
from shapely.geometry import Point, Polygon
import json

class CensusCredentials:
    def __init__(self, api_key: str):
        self.api_key = api_key

class CensusService:
    def __init__(self, credentials: CensusCredentials):
        self.credentials = credentials
        self.census_client = Census(credentials.api_key)

    def get_census_data(self, fips_code: str, year: int = 2022) -> pd.DataFrame:
        """
        Get Census demographic data for block groups in a given FIPS code.
        
        Args:
            fips_code (str): 5-digit FIPS code (state + county)
            year (int): Census year to query (default: 2022)
        
        Returns:
            pd.DataFrame: DataFrame containing Census demographic data
            
        Raises:
            TypeError: If fips_code is not a string or year is not an int
            ValueError: If fips_code is not 5 digits
        """
        if not isinstance(fips_code, str):
            raise TypeError("fips_code must be a string")
        if not isinstance(year, int):
            raise TypeError("year must be an integer")
        if len(fips_code) != 5:
            raise ValueError("fips_code must be 5 digits (state + county)")

        # Split FIPS code into state and county
        state_fips = fips_code[:2]
        county_fips = fips_code[2:]
        
        # Get block group data
        data = self.census_client.acs5.state_county_blockgroup(
            fields=['NAME',
                    'B19013_001E',  # Median income
                    'B01003_001E',  # Total population
                    'B03002_003E',  # White alone
                    'B03002_004E',  # Black alone
                    'B03002_012E'],  # Hispanic/Latino
            state_fips=state_fips,
            county_fips=county_fips,
            blockgroup='*',  # All block groups
            year=year
        )

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Rename columns
        df = df.rename(columns={
            'B19013_001E': 'median_income',
            'B01003_001E': 'total_pop',
            'B03002_003E': 'white_pop',
            'B03002_004E': 'black_pop',
            'B03002_012E': 'hispanic_pop'
        })

        # Create GEOID for block groups (state+county+tract+block group)
        df['state_fips'] = df['state']
        df['county_fips'] = df['county']
        df['tract_fips'] = df['tract']
        df['bg_fips'] = df['block group']

        # Create standardized GEOID
        df['std_geoid'] = df['state_fips'] + df['county_fips'] + df['tract_fips'] + df['bg_fips']

        return df

    def get_census_blockgroups_shapefile(self, fips_code: str) -> gpd.GeoDataFrame:
        """
        Get Census Block Group shapefiles for a given FIPS code from the Census TIGERweb service.
        
        Args:
            fips_code (str): 5-digit FIPS code (state + county)
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing Census Block Group boundaries
            
        Raises:
            TypeError: If fips_code is not a string
            ValueError: If fips_code is not 5 digits
            requests.RequestException: If API request fails
        """
        if not isinstance(fips_code, str):
            raise TypeError("fips_code must be a string")
        if len(fips_code) != 5:
            raise ValueError("fips_code must be 5 digits (state + county)")

        # TIGERweb REST API endpoint
        base_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query"
        
        # Query parameters
        params = {
            'where': f"STATE='{fips_code[:2]}' AND COUNTY='{fips_code[2:]}'",
            'outFields': '*',
            'returnGeometry': 'true',
            'f': 'geojson',
            'outSR': '4326'  # WGS84
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            geojson_data = response.json()
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            
            # Create standardized GEOID components
            gdf['state_fips'] = gdf['STATE']
            gdf['county_fips'] = gdf['COUNTY']
            gdf['tract_fips'] = gdf['TRACT']
            gdf['bg_fips'] = gdf['BLKGRP']

            # Create standardized GEOID
            gdf['std_geoid'] = gdf['state_fips'] + gdf['county_fips'] + gdf['tract_fips'] + gdf['bg_fips']
            
            return gdf
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch Census Block Group data: {str(e)}")

    def get_census_data_with_boundaries(
        self,
        fips_code: str,
        year: int = 2022
    ) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Get both Census demographic data and boundary files for block groups in a FIPS code.
        
        Args:
            fips_code (str): 5-digit FIPS code (state + county)
            year (int): Census year to query (default: 2022)
        
        Returns:
            Tuple[pd.DataFrame, gpd.GeoDataFrame]: 
                - Census demographic data DataFrame
                - Census Block Group boundaries GeoDataFrame
                
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If inputs have invalid values
            requests.RequestException: If API requests fail
        """
        # Get demographic data
        census_data = self.get_census_data(fips_code, year)
        
        # Get boundary files
        census_boundaries = self.get_census_blockgroups_shapefile(fips_code)
        
        # Merge demographic data with boundaries
        census_boundaries = census_boundaries.merge(
            census_data,
            on='std_geoid',
            how='left'
        )
        
        return census_data, census_boundaries

def init_service_census(credentials: CensusCredentials) -> CensusService:
    """
    Initialize a Census service with the provided credentials.
    
    Args:
        credentials (CensusCredentials): Census API credentials
        
    Returns:
        CensusService: Initialized Census service
        
    Raises:
        ValueError: If credentials are invalid
    """
    if not isinstance(credentials, CensusCredentials):
        raise ValueError("Invalid credentials for Census service.")
    return CensusService(credentials)

def get_creds_from_env_census() -> CensusCredentials:
    """
    Get Census credentials from environment variables.
    
    Returns:
        CensusCredentials: Census API credentials
        
    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("Missing Census API key in environment.")
    return CensusCredentials(api_key) 