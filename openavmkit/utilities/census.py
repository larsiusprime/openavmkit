import os
from typing import Tuple
import pandas as pd
import geopandas as gpd
import requests
import warnings
from census import Census
from openavmkit.utilities.geometry import get_crs


class CensusCredentials:
    """Object for storing US Census API credentials

    Attributes
    ----------
    api_key : str
        API Key for the US Census
    """

    def __init__(self, api_key: str):
        """Initialize a CensusCredentials object

        api_key : str
            API Key for the US Census
        """
        self.api_key = api_key


class CensusService:
    """Provides functions for downloading data from the US Census

    Attributes
    ----------
    credentials: CensusCredentials
        Credentials for the US Census
    census_client : census.core.Census
        US Census API Client object

    """

    def __init__(self, credentials: CensusCredentials):
        """Initialize the CensusService object

        Parameters
        ----------
        credentials : CensusCredentials
            Credentials for the US Census
        """
        self.credentials = credentials
        self.census_client = Census(credentials.api_key)
    
    def get_census_map(self, census_settings: dict) -> dict:
        
        if census_settings is None:
            census_settings = {}
        
        return {
            "B19013_001E": "median_income",
            "B01003_001E": "total_population",
            "B25064_001E": "median_g_rent",
            "B25058_001E": "median_c_rent"
        }
    
    
    def get_census_data(
        self, 
        fips_code: str, 
        year: int = 2022,
        census_settings: dict = None
    ) -> pd.DataFrame:
        """Get Census demographic data for block groups in a given FIPS code.

        Parameters
        ----------
        fips_code : str
            5-digit FIPS code (state + county)
        year : int
            Census year to query (default: 2022)
        census_settings : dict
            Census settings

        Returns
        -------
        pd.DataFrame
            DataFrame containing Census demographic data

        Raises
        ------
        TypeError
            If fips_code is not a string or year is not an int
        ValueError
            If fips_code is not 5 digits
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

        map = self.get_census_map(census_settings)
        fields = ["NAME"] + [key for key in map]
        
        # Get block group data
        data = self.census_client.acs5.state_county_blockgroup(
            fields,
            state_fips=state_fips,
            county_fips=county_fips,
            blockgroup="*",  # All block groups
            year=year,
        )

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Rename columns
        df = df.rename(
            columns=map
        )

        # Create GEOID for block groups (state+county+tract+block group)
        df["state_fips"] = df["state"]
        df["county_fips"] = df["county"]
        df["tract_fips"] = df["tract"]
        df["bg_fips"] = df["block group"]

        # Create standardized GEOID
        df["std_geoid"] = (
            df["state_fips"] + df["county_fips"] + df["tract_fips"] + df["bg_fips"]
        )

        return df

    def get_census_blockgroups_shapefile(self, fips_code: str) -> gpd.GeoDataFrame:
        """Get Census Block Group shapefiles for a given FIPS code from the Census TIGERweb service.

        Parameters
        ----------
        fips_code : str
            5-digit FIPS code (state + county)

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing Census Block Group boundaries

        Raises
        ------
        TypeError
            If fips_code is not a string
        ValueError
            If fips_code is not 5 digits
        requests.RequestException
            If API request fails
        """
        if not isinstance(fips_code, str):
            raise TypeError("fips_code must be a string")
        if len(fips_code) != 5:
            raise ValueError("fips_code must be 5 digits (state + county)")

        # TIGERweb REST API endpoint for block groups
        base_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/1/query"

        # Query parameters
        params = {
            "where": f"STATE='{fips_code[:2]}' AND COUNTY='{fips_code[2:]}'",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326",  # WGS84
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            geojson_data = response.json()

            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

            # Create standardized GEOID components
            gdf["state_fips"] = gdf["STATE"]
            gdf["county_fips"] = gdf["COUNTY"]
            gdf["tract_fips"] = gdf["TRACT"]
            gdf["bg_fips"] = gdf["BLKGRP"]

            # Create standardized GEOID
            gdf["std_geoid"] = (
                gdf["state_fips"]
                + gdf["county_fips"]
                + gdf["tract_fips"]
                + gdf["bg_fips"]
            )

            # Explicitly set the CRS to EPSG:4326 (WGS84)
            gdf.crs = "EPSG:4326"

            return gdf

        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to fetch Census Block Group data: {str(e)}"
            )

    def get_census_data_with_boundaries(
        self, fips_code: str, year: int = 2022, census_settings : dict = None
    ) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Get both Census demographic data and boundary files for block groups in a
        FIPS code.

        Parameters
        ----------
        fips_code :str
            5-digit FIPS code (state + county)
        year : int
            Census year to query (default: 2022)
        census_settings : dict
            Census settings object
        Returns
        -------
        Tuple[pd.DataFrame, gpd.GeoDataFrame]:

            - Census demographic data DataFrame
            - Census Block Group boundaries GeoDataFrame

        Raises
        ------
        TypeError
            If inputs have wrong types
        ValueError
            If inputs have invalid values
        requests.RequestException
            If API requests fail
        """
        # Get demographic data first
        census_data = self.get_census_data(fips_code, year, census_settings)
        # Get the list of block groups we have data for
        valid_block_groups = census_data["std_geoid"].unique()

        # Get boundary files
        census_boundaries = self.get_census_blockgroups_shapefile(fips_code)
        # Filter boundaries to only include block groups we have data for
        census_boundaries = census_boundaries[
            census_boundaries["std_geoid"].isin(valid_block_groups)
        ]

        # Merge demographic data with boundaries
        census_boundaries = census_boundaries.merge(
            census_data, on="std_geoid", how="left"
        )
        # Verify the merge
        missing_geoids = census_boundaries[census_boundaries.isna().any(axis=1)][
            "std_geoid"
        ].unique()
        if len(missing_geoids) > 0:
            print(
                f"\nWarning: Found {len(missing_geoids)} block groups with missing data"
            )
            print("First few missing GEOIDs:", missing_geoids[:5])

        return census_data, census_boundaries


def match_to_census_blockgroups(
    gdf: gpd.GeoDataFrame, census_gdf: gpd.GeoDataFrame, join_type: str = "left"
) -> gpd.GeoDataFrame:
    """Match each row in a GeoDataFrame to its corresponding Census Block Group using
    spatial join.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame to match
    census_gdf : gpd.GeoDataFrame
        Census Block Group boundaries GeoDataFrame
    join_type : str
        Type of join to perform ('left', 'right', 'inner', 'outer')

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with Census Block Group data appended
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")
    if not isinstance(census_gdf, gpd.GeoDataFrame):
        raise TypeError("census_gdf must be a GeoDataFrame")
    if join_type not in ["left", "right", "inner", "outer"]:
        raise ValueError("join_type must be one of: 'left', 'right', 'inner', 'outer'")

    # Create a copy of the input GeoDataFrame to avoid modifying the original
    gdf_for_join = gdf.copy()
    
    if "geometry" not in gdf_for_join:
        raise ValueError("Input dataframe must have a 'geometry' column!")

    # If the data is in a geographic CRS (like WGS84/EPSG:4326),
    # reproject to a projected CRS before calculating centroids
    if gdf_for_join.crs.is_geographic:
        # Use the geometry utility to get an appropriate equal-area CRS based on the data
        projected_crs = get_crs(gdf_for_join, "equal_area")
        gdf_for_join = gdf_for_join.to_crs(projected_crs)
        census_gdf = census_gdf.to_crs(projected_crs)

    # Calculate centroids in the projected CRS
    gdf_for_join["centroid"] = gdf_for_join.geometry.centroid

    # Create a temporary GeoDataFrame with centroids for the spatial join
    centroid_gdf = gpd.GeoDataFrame(
        gdf_for_join.drop(columns=["geometry"]),
        geometry="centroid",
        crs=gdf_for_join.crs,
    )

    # Perform the spatial join
    census_gdf = census_gdf.to_crs(centroid_gdf.crs)
    
    joined = centroid_gdf.sjoin(census_gdf, predicate="intersects", how=join_type)
    if "index_right" in joined:
        joined = joined.drop(columns="index_right")
    
    # If we have matches, process them
    if not joined.empty:
        # Calculate areas for each match
        joined["area"] = joined.geometry.area

        # Group by the index and find the smallest area for each
        smallest_areas = joined.groupby(level=0)["area"].idxmin()
        joined = joined.loc[smallest_areas]

        # Calculate and print percentage of records with valid census geoid
        valid_geoid_count = joined["std_geoid"].notna().sum()
        valid_percentage = (valid_geoid_count / len(gdf)) * 100
        print(
            f"Census block group matching: {valid_geoid_count} of {len(gdf)} records have valid census geoid ({valid_percentage:.2f}%)"
        )
        
        # restore the original geometry column
        joined = joined.merge(gdf[["key","geometry"]], on="key", how="left")
        joined = joined.set_geometry("geometry")
        joined = joined.drop(columns=["centroid","area"])

        return joined
    else:
        # No matches, so just return the original GeoDataFrame with census columns added (all NaN)
        census_columns = ["std_geoid", "median_income", "total_pop"]
        for col in census_columns:
            if col in census_gdf.columns:
                gdf[col] = None

        print(f"Census block group matching: 0 of {len(gdf)} records matched (0.00%)")
        return gdf


def init_service_census(credentials: CensusCredentials) -> CensusService:
    """Initialize a Census service with the provided credentials.

    Parameters
    ----------
    credentials : CensusCredentials
        Census API credentials

    Returns
    -------
    CensusService
        Initialized Census service

    Raises
    ------
    ValueError
        If credentials are invalid
    """
    if not isinstance(credentials, CensusCredentials):
        raise ValueError("Invalid credentials for Census service.")
    return CensusService(credentials)


def get_creds_from_env_census() -> CensusCredentials:
    """Get Census credentials from environment variables.

    Returns
    -------
    CensusCredentials
        Census API credentials

    Raises
    ------
    ValueError
        If required environment variables are missing
    """
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        warnings.warn("Missing Census API key in environment.")
        return None
    return CensusCredentials(api_key)
