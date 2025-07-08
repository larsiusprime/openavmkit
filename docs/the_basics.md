# The Basics

## Creating a new locality

OpenAVMKit operates on the concept of a "locality", which is a geographic area that contains a set of properties. This can represent a city, a county, a neighborhood, or any other region or jurisdiction you want to analyze. To set one up, create a folder like this within openavmkit's `notebooks/` directory:

```
data/<locality_slug>/
```

Where `<locality_slug>` is a unique identifying name for your locality in a particularly opinionated format. That format is:

```
<country_code>-<state_or_province_code>-<locality_name>
```

- **Country code**: The 2-letter country code according to the [ISO 3166-1 standard](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2). For example, the country code for the United States is `us`, and the country code for Norway is `no`.

- **State/province code**: The 2-letter state or province code according to the [ISO 3166-2 standard](https://en.wikipedia.org/wiki/ISO_3166-2). For example, the state code for Texas is `tx`, and the state code for California is `ca`.

- **Locality name**: A human-readable name for the locality itself. This follows no particular standard and is entirely up to you.

The slug itself should be all lowercase and contain no spaces or special characters other than underscores.

Some examples:

```
us-nc-guilford    # Guilford County, North Carolina, USA
us-tx-austin      # City of Austin, Texas, USA
no-03-oslo        # City of Oslo, Norway
no-50-orkdal      # Orkdal kommune (county), Norway
```

Once you have your locality set up, you will want to set it up like this (using `us-nc-guilford` as an example):

```
data/
├──us-nc-guilford/
    ├── in/
    ├── out/
    ├── settings.json
```

The `in/` directory is where you will put your raw data files.   
The `out/` directory is where the output files will be saved.  
The `settings.json` file will drive all your modeling and analysis decisions for the library. For now you can just put a blank `{}` in there so that it will load, but you will want to consult the documentation / tutorials for how to construct this file.


## Terminology


**Parcel**  
A single unit of real estate. In this context, each row in a modeling dataframe represents one parcel.

**Building**  
A freestanding structure or dwelling on a parcel. A parcel can have multiple buildings.

**Improvement**  
Any non-moveable physical structure that improves the value of the parcel. This includes buildings, but also other structures like fences, pools, or sheds. It also includes things like landscaping, paved driveways, and in agricultural contexts, irrigation, crops, orchards, timber, etc.

**Model group**  
A named list of parcels that share similar characteristics and, most importantly, prospective buyers and sellers, and are therefore valued using the same model. For example, a model group might be "Single Family Residential" or "Commercial".

### Characteristics

**Characteristic**  
A feature of a parcel that affects its value. This can be a physical characteristic like square footage, or a locational characteristic like proximity to a park. Characteristics come in three flavors -- categorical, numeric, and boolean.

**Categorical**  
A characteristic that can take on one of a fixed set of values. For example, "zoning" might be a categorical characteristic with values like "residential", "commercial", "industrial", etc.

**Numeric**  
A characteristic that can take on any numeric value. For example, "finished square footage" might be a numeric characteristic.

**Boolean**  
A characteristic that can take on one of two values, "true" or "false. For example, "has swimming pool" might be a boolean characteristic.

### Value

**Prediction**  
An opinion of the value of a parcel, based on a model. There are many different kinds of predictions.

**Valuation date**  
The date for which the value is being predicted. This is typically January 1st of the upcoming year, but not always.

**Full market value**  
The price a parcel would sell for in an open market, with a willing buyer and a willing seller, neither under duress. In a modeling context, this is the value we are trying to predict: the full value of the property.

**Improvement value**  
The portion of the full market value due solely to the improvement(s) on a parcel. This is the value of the building(s) and other improvements, but not the land itself.

**Land value**  
The portion of the full market value due solely to the land itself, without any improvements.

### Data sets

**Data set**  
This refers to any collection of parcel records grouped together by some criteria. Except for the omni set, these data sets are always within the context of a specific model group, such as "Single Family Residential", or "Commercial."

**Sales set**  
This refers to the subset of parcels that have a valid sale within the study period. We will use these to train our models as well as to evaluate them.

**Training set**  
The portion of the sales set (typically 80%) that we use to train our models from.

**Testing set**  
The portion of the sale set (typically 20%) that we set aside to evaluate our models. These are sales that the predictive models have never seen before.

**Universe set**  
The full set of parcels in the jurisdiction, regardless of whether a particular parcel has sold or not. This is the data set we will generate predictions for.

**Multiverse set**  
The full set of parcels in the jurisdiction, *regardless* of the current modeling group. The difference between this and the universe set is that the universe set is limited to the current model group, as are all the other above data sets.

### Modeling

**Main**  
The main model is the primary model. It operates on the full data set, and predicts *full market value*.

**Hedonic**

**Vacant**


### Avoid these terms

These terms are ambiguous and can refer to different things in different contexts, so we avoid them in our documentation.

**Property**  
In casual conversation this can mean a parcel, a building, or a piece of land. But in the context of coding, it can also refer to a characteristic or variable.


## Code modules

Here's how you can import and use the core modules directly in your own Python code.

For instance, here's a simple example that demonstrates how to calculate the Coefficient of Dispersion (COD) for a list of ratios:

```python
import openavmkit

ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
cod = openavmkit.utilities.stats.calc_cod(ratios)
print(cod)
```

You can also specify the specific module you want to import:

```python
from openavmkit.utilities import stats

ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
cod = stats.calc_cod(ratios)
```

Or even import specific functions directly:

```python
from openavmkit.utilities.stats import calc_cod

ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
cod = calc_cod(ratios)
```

## Using the Jupyter Notebooks

Make sure that you've already installed the `jupyter` library. If not, see [Quick Start](getting_started.md#quick-start) for instructions.

The `notebooks/` directory contains several pre-written Jupyter notebooks that demonstrate how to use the library interactively. These notebooks are especially useful for new users, as they contain step-by-step explanations and examples.

1. Launch the Jupyter notebook server:
```bash
jupyter notebook
```

This should open a new tab in your web browser with the Jupyter interface.

![Jupyter interface](assets/images/jupyter_01.png)

2. Navigate to the `notebooks/` directory in the Jupyter interface and open the notebook you want to run.

![Open notebook](assets/images/jupyter_02.png)

3. Double-click on your chosen notebook to open it.

![Running notebook](assets/images/jupyter_03.png)

For information on how to use Jupyter notebooks in general, refer to the [official Jupyter notebook documentation](https://jupyter-notebook.readthedocs.io/en/stable/).