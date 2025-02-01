# Terminology



**Parcel**  
A single unit of real estate. In this context, each row in a modeling dataframe represents one parcel.

**Building**  
A freestanding structure or dwelling on a parcel. A parcel can have multiple buildings.

**Improvement**  
Any non-moveable physical structure that improves the value of the parcel. This includes buildings, but also other structures like fences, pools, or sheds. It also includes things like landscaping, paved driveways, and in agricultural contexts, irrigation, crops, orchards, timber, etc.

**Model group**  
A named list of parcels that share similar characteristics and, most importantly, prospective buyers and sellers, and are therefore valued using the same model. For example, a model group might be "Single Family Residential" or "Commercial".

## Characteristics

**Characteristic**  
A feature of a parcel that affects its value. This can be a physical characteristic like square footage, or a locational characteristic like proximity to a park. Characteristics come in three flavors -- categorical, numeric, and boolean.

**Categorical**  
A characteristic that can take on one of a fixed set of values. For example, "zoning" might be a categorical characteristic with values like "residential", "commercial", "industrial", etc.

**Numeric**  
A characteristic that can take on any numeric value. For example, "finished square footage" might be a numeric characteristic.

**Boolean**  
A characteristic that can take on one of two values, "true" or "false. For example, "has swimming pool" might be a boolean characteristic.

## Value

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

## Data sets

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

## Modeling

**Main**  
The main model is the primary model. It operates on the full data set, and predicts *full market value*.

**Hedonic**

**Vacant**


## Avoid these terms

These terms are ambiguous and can refer to different things in different contexts, so we avoid them in our documentation.

**Property**  
In casual conversation this can mean a parcel, a building, or a piece of land. But in the context of coding, it can also refer to a characteristic or variable.

