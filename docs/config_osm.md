# Configuring OpenStreetMap Enrichment

OpenAVMKit can enrich your data with geographic features from OpenStreetMap, such as water bodies, parks, educational institutions, transportation networks, and golf courses. This enrichment adds distance-based features to your dataset, which can be valuable for property valuation.

## Using OpenStreetMap Enrichment

To enable OpenStreetMap enrichment in your locality settings, add the following to your `settings.json`:

```json
{
  "process": {
    "enrich": {
      "universe": {
        "openstreetmap": {
          "enabled": true,
          "water_bodies": {
            "enabled": true,
            "min_area": 10000,
            "top_n": 5,
            "sort_by": "area"
          },
          "transportation": {
            "enabled": true,
            "min_length": 1000,
            "top_n": 5,
            "sort_by": "length"
          },
          "educational": {
            "enabled": true,
            "min_area": 1000,
            "top_n": 5,
            "sort_by": "area"
          },
          "parks": {
            "enabled": true,
            "min_area": 2000,
            "top_n": 5,
            "sort_by": "area"
          },
          "golf_courses": {
            "enabled": true,
            "min_area": 10000,
            "top_n": 3,
            "sort_by": "area"
          }
        },
        "distances": [
          {
            "id": "water_bodies",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "water_bodies_top",
            "field": "name",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "parks",
            "max_distance": 800,
            "unit": "m"
          },
          {
            "id": "parks_top",
            "field": "name",
            "max_distance": 800,
            "unit": "m"
          },
          {
            "id": "golf_courses",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "golf_courses_top",
            "field": "name",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "educational",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "educational_top",
            "field": "name",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "transportation",
            "max_distance": 1500,
            "unit": "m"
          },
          {
            "id": "transportation_top",
            "field": "name",
            "max_distance": 1500,
            "unit": "m"
          }
        ]
      }
    }
  }
}
```

## Feature Types and Settings

The OpenStreetMap enrichment supports the following feature types:

1. **Water Bodies**: Rivers, lakes, reservoirs, etc.
    - `min_area`: Minimum area in square meters (default: 10000)
    - `top_n`: Number of largest water bodies to track individually (default: 5)

2. **Transportation**: Major roads, railways, etc.
    - `min_length`: Minimum length in meters (default: 1000)
    - `top_n`: Number of longest transportation routes to track individually (default: 5)

3. **Educational Institutions**: Universities, colleges, etc.
    - `min_area`: Minimum area in square meters (default: 1000)
    - `top_n`: Number of largest institutions to track individually (default: 5)

4. **Parks**: Public parks, gardens, playgrounds, etc.
    - `min_area`: Minimum area in square meters (default: 2000)
    - `top_n`: Number of largest parks to track individually (default: 5)

5. **Golf Courses**: Golf courses and related facilities
    - `min_area`: Minimum area in square meters (default: 10000)
    - `top_n`: Number of largest golf courses to track individually (default: 3)

## Distance Calculations

For each feature type, the enrichment calculates:

1. **Aggregate distances**: Distance to the nearest feature of that type
    - Output variable: `dist_to_[feature_type]_any` (in meters)

2. **Individual distances**: Distance to each of the top N largest features
    - Output variable: `dist_to_[feature_type]_[feature_name]` (in meters)
    - Example: `dist_to_parks_central_park`

## Configuration Options

- `enabled`: Set to `true` to enable OpenStreetMap enrichment
- `min_area`/`min_length`: Filter out features smaller than this threshold
- `top_n`: Number of largest features to track individually
- `sort_by`: Property to use for sorting features (area or length)
- `max_distance`: Maximum distance to calculate (in meters)
- `unit`: Unit of measurement for distances (m for meters)

The OpenStreetMap enrichment will automatically join geographic feature data to your parcels using spatial joins, adding distance-based features to your dataset.