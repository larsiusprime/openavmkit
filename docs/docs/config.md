# Configuration

## Configuring cloud storage

`openavmkit` includes a module for working with remote storage services. At this time the library supports three cloud storage methods:

- Microsoft Azure
- Hugging Face
- SFTP

To configure cloud storage, you will need to create a file that stores your connection credentials (such as API keys or passwords). This file should be named `.env` and should be placed in the `notebooks/` directory.

This file is already ignored by git, but do make sure you don't accidentally commit this file to the repository or share it with others, as it contains your sensitive login information!

This file should be a plain text file formatted like this:
```
SOME_VARIABLE=some_value
ANOTHER_VARIABLE=another_value
YET_ANOTHER_VARIABLE=123
```

That's just an example of the format; here are the actual variables that it recognizes:

| Variable Name                     | Description                                                                                                                                                             |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AZURE_ACCESS`                    | The type of access your azure account has.<br>Legal values are: `read_only`, `read_write`.                                                                              |
| `AZURE_STORAGE_CONTAINER_NAME`    | The name of the Azure storage container                                                                                                                                 |
| `AZURE_STORAGE_CONNECTION_STRING` | The connection string for the Azure storage account                                                                                                                     |
| `HF_ACCESS`                       | The type of access your huggingface account has.<br>Legal values are: `read_only`, `read_write`.                                                                        |
| `HF_TOKEN`                        | The Hugging Face API token                                                                                                                                              |
| `HF_REPO_ID`                      | The Hugging Face repository ID                                                                                                                                          |
| `SFTP_ACCESS`                     | The type of access your SFTP account has.<br>Legal values are: `read_only`, `read_write`.                                                                               |
| `SFTP_HOST`                       | The hostname of the SFTP server                                                                                                                                         |
| `SFTP_USERNAME`                   | The username for the SFTP server                                                                                                                                        |
| `SFTP_PASSWORD`                   | The password for the SFTP server                                                                                                                                        |
| `SFTP_PORT`                       | The port number for the SFTP server                                                                                                                                     |

You only need to provide values for the service that you're actually using. For instance, here's what the file might look like if you are using Hugging Face:

```
HF_ACCESS=read_write
HF_REPO_ID=landeconomics/localities-public
HF_TOKEN=<YOUR_HUGGING_FACE_API_TOKEN>
```

If you're just getting started, you can just use read-only access to an existing public repository. Here's an example of how to access the public datasets provided by the [The Center for Land Economics](https://landeconomics.org):

```
HF_ACCESS=read_only
HF_REPO_ID=landeconomics/localities
```

This will let you download the inputs for any of the Center for Land Economics' public datasets. Note that you will be unable to upload your changes and outputs to repositories that you have read-only access to.

If you want to sync with your own cloud storage, you will need to set up your own hosting account and then provide the appropriate credentials in the `.env` file.

If you have multiple projects stored on different cloud services, you can set the `CLOUD_TYPE` and `CLOUD_ACCESS` variables in your settings.json. This will allow you to switch between cloud services on a per-project basis. **Do not ever store credentials in your settings.json, however, as this file is uploaded to the cloud!**

## Configuring PDF report generation

`openavmkit` includes a module for generating PDF reports. This module uses the `pdfkit` library, which is a Python wrapper for the `wkhtmltopdf` command line tool. Although `pdfkit` will be installed automatically along with the rest of the dependencies, you will need to install `wkhtmltopdf` manually on your system to use this feature. If you skip this step, don't worry, you'll still be able to use the rest of the library, it just won't generate PDF reports.

### Installing wkhtmltopdf:

#### Manual installation

Visit the [wkhtmltopdf download page](https://wkhtmltopdf.org/downloads.html) and download the appropriate installer for your operating system.

#### Windows

1. Download the installer linked above
2. Run the installer and follow the instructions
3. Add the `wkhtmltopdf` installation directory to your system's PATH environment variable

If you don't know what 3. means:

The idea is that you want the `wkhtmltopdf` executable to be available from any command prompt, so you can run it from anywhere. For that to work, you need to make sure that the folder that the `wkhtmltopdf` executable is in is listed in your system's PATH environment variable.

Here's how to do that:

1. Find the folder where `wkhtmltopdf` was installed. It's probably in `C:\Program Files\wkhtmltopdf\bin`, but it could be somewhere else. Pay attention when you install it.
2. Follow this [tutorial](https://web.archive.org/web/20250131024033/https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/) to edit your PATH environment variable. You want to add the folder from step 1 to the PATH variable.
3. Open a new command prompt and type `wkhtmltopdf --version`. If you see a version number, you're all set!

#### Linux

On Debian/Ubuntu, run:

```bash
sudo apt-get update
sudo apt-get install wkhtmltopdf
```

#### macOS

Ensure you have [Homebrew](https://brew.sh) installed. Then run:

```bash
brew install wkhtmltopdf
```

## Configuring Census API Access

OpenAVMKit can enrich your data with Census information using the Census API. To use this feature, you'll need to:

1. Get a Census API key from [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
2. Add your Census API key to the `.env` file in the `notebooks/` directory

### Getting a Census API Key

1. Visit [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
2. Fill out the form with your information
3. Agree to the Census terms of service
4. You will receive your API key via email

### Configuring the Census API Key

Add your Census API key to the `.env` file in the `notebooks/` directory:

```
CENSUS_API_KEY=your_api_key_here
```

### Using Census Enrichment

To enable Census enrichment in your locality settings, add the following to your `settings.json`:

```json
{
  "process": {
    "enrich": {
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
```

Key settings:

- `enabled`: Set to `true` to enable Census enrichment
- `year`: The Census year to query (default: 2022)
- `fips`: The 5-digit FIPS code for your locality (state + county)
- `fields`: List of Census fields to include

The Census enrichment will automatically join Census block group data to your parcels using spatial joins, adding demographic information to your dataset.

## Configuring OpenStreetMap Enrichment

OpenAVMKit can enrich your data with geographic features from OpenStreetMap, such as water bodies, parks, educational institutions, transportation networks, and golf courses. This enrichment adds distance-based features to your dataset, which can be valuable for property valuation.

### Using OpenStreetMap Enrichment

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

### Feature Types and Settings

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

### Distance Calculations

For each feature type, the enrichment calculates:

1. **Aggregate distances**: Distance to the nearest feature of that type
    - Output variable: `dist_to_[feature_type]_any` (in meters)

2. **Individual distances**: Distance to each of the top N largest features
    - Output variable: `dist_to_[feature_type]_[feature_name]` (in meters)
    - Example: `dist_to_parks_central_park`

### Configuration Options

- `enabled`: Set to `true` to enable OpenStreetMap enrichment
- `min_area`/`min_length`: Filter out features smaller than this threshold
- `top_n`: Number of largest features to track individually
- `sort_by`: Property to use for sorting features (area or length)
- `max_distance`: Maximum distance to calculate (in meters)
- `unit`: Unit of measurement for distances (m for meters)

The OpenStreetMap enrichment will automatically join geographic feature data to your parcels using spatial joins, adding distance-based features to your dataset.