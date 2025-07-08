# Configuring Census API Access

OpenAVMKit can enrich your data with Census information using the Census API. To use this feature, you'll need to:

1. Get a Census API key from [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
2. Add your Census API key to the `.env` file in the `notebooks/` directory

## Getting a Census API Key

1. Visit [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
2. Fill out the form with your information
3. Agree to the Census terms of service
4. You will receive your API key via email

## Configuring the Census API Key

Add your Census API key to the `.env` file in the `notebooks/` directory:

```
CENSUS_API_KEY=your_api_key_here
```

## Using Census Enrichment

To enable Census enrichment in your locality settings, add the following to your `settings.json`:

```json
{
  "process": {
    "enrich": {
      "universe": {
        "census": {
          "enabled": true,
          "year": 2022,
          "fips": "24510",
          "fields": [
            "median_income",
            "total_pop"
          ]
        }
      }
    }
  }
}
```

Key settings:

- `enabled`: Set to `true` to enable Census enrichment  
- `year`: The Census year to query (default: 2022)  
- `fips`: The 5-digit FIPS code for your locality (state + county)  
- `fields`: List of Census fields to include  

The Census enrichment will automatically join Census block group data to your parcels using spatial joins, adding demographic information to your dataset.