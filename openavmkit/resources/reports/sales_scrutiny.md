# Anomalous Sales Analysis

Locality: {{locality}}  
Valuation date: {{val_date}}  
Model group: {{model_group}}

## Executive Summary

Invalid sales wreck valuations in two ways:  

- Invalid sales poison the training set, causing bad predictions  
- Predictions get tested against invalid sales, causing artificially poor ratio studies  

Traditionally, sales are validated from transaction details such as deed type and the relationship between buyer and seller, with the aim of identifying non-arms-length transfers. However, sales can also be statistically validated based on the following principle: 

> Similar properties sold in similar locations should have similar prices, especially on a price/sqft basis.

Similar properties sold in similar locations that do *not* have similar prices may reveal anomalies. (All prices in this analysis are time-adjusted, taking temporal variation out of the equation.)

### Method

We group sales into similar clusters, then look for sales that stick out within their groups. These anomalies could indicate invalid sale, incorrect property characteristics, or missing information. Not every anomaly is necessarily invalid--some are simply natural outliers--but they should all be manually reviewed by {{locality}}.

### Results

We flagged **{{num_sales_flagged}}** anomalous sales out of **{{num_sales_total}}** sales total for the **{{model_group}}** model group.

Here is a breakdown of anomalies by type:

| Anomaly type                                           |                Number of sales |               % of Total sales |
|--------------------------------------------------------|-------------------------------:|-------------------------------:|
| 1 - High OR low price/sqft, high OR low square footage | {{num_sales_flagged_type_1}} | {{pct_sales_flagged_type_1}} |
| 2 - Low price, low price/sqft                          | {{num_sales_flagged_type_2}} | {{pct_sales_flagged_type_2}} |  
| 3 - High price, high price/sqft                        | {{num_sales_flagged_type_3}} | {{pct_sales_flagged_type_3}} |
| 4 - Price in range, high price/sqft                    | {{num_sales_flagged_type_4}} | {{pct_sales_flagged_type_4}} |
| 5 - Price in range, low price/sqft                     | {{num_sales_flagged_type_5}} | {{pct_sales_flagged_type_5}} |
| **Flagged sales**                                      |    **{{num_sales_flagged}}** |    **{{pct_sales_flagged}}** |
| **Total sales**                                        |      **{{num_sales_total}}** |                       **100%** |

Note that a sale can be flagged as having multiple anomalies, so the total number flagged will not necessarily be the total of each of the sub categories.


For an explanation of these different types of anomalies and what to do about them, read below.

## Sales Anomaly Overview

This table provides a quick diagnostic overview of the most common causes of common sales anomalies, as well as what to do about them.

| # | Price    | Price/sqft | Square footage | Likely causes                                                                  | Recommended Action                                      |
|---|----------|------------|----------------|--------------------------------------------------------------------------------|---------------------------------------------------------|
| 1 | Any      | High/Low   | High/Low       | Incorrect square footage                                                       | Check square footage / characteristics at time of sale. | 
| 2 | Low      | Low        | In range       | Misclassified vacant status/characteristics at time of sale.                   | Check vacant status / characteristics at time of sale.  |
| 3 | High     | High       | In range       | Non-arm's length sale, or valid with misclassified characteristics.            | Check sale status / characteristics at time of sale.    |
| 4 | In range | High       | In range       | Possibly missing/misclassified characteristics                                 | Check characteristics at time of sale.                  |
| 5 | In range | Low        | In range       | Distressed/non-arm's length sale, or valid with misclassified characteristics. | Check sale status / characteristics at time of sale.    |

Detailed explanations for each type, along with practical examples and recommendations, follow from here. 

## 1. Anomalous price/sqft, anomalous square footage

This is typically caused by under- or over-reported square footage.

- If the building area is **under-reported**, the total sale price might look normal while the calculated price/sqft is **too high**.  
- If the building area is **over-reported**, then price/sqft can appear **too low**.  

Here's a (fictional) example:

| Finished Sqft | Price | Price/sqft | Median $/sqft | Relative Ratio | Std Dev from Median |  Flagged |
|--------------:|------:|-----------:|--------------:|---------------:|--------------------:|---------:|
|        1,800	 | 360,000 |        200 | 200 |           1.00 |                0.00 |    FALSE | 
|        2,000	 | 400,000 |        200 | 200 |           1.00 |                0.00 |    FALSE | 
|        2,200	 | 440,000 |        200 | 200 |           1.00 |                0.00 |    FALSE | 
|        1,900	 | 380,000 |        200 | 200 |           1.00 |                0.00 |    FALSE | 
|    **1,000**	 | 400,000 |    **400** | 200 |       **2.00** |            **2.24** | **TRUE** | 

- All “normal” sales: ~$200/sqft.  
- One anomaly: 1,000 sqft at $400k → $400/sqft is twice the median.  
- The finished area was likely under-reported. (Price is fine; square footage is too small.)  

**Recommended action:** 

- Verify square footage at time of sale. If the area is wrong, correct it; otherwise consider excluding the sale as non-representative.  

## 2. Low price, in-range square footage

In this case, the sale price itself is anomalously low, but there's nothing remarkable about the square footage. 

This can be caused by any number of common issues, including:

1. **Distressed/forced sale** (e.g., foreclosure, REO, short sale) at below-market price.
2. **Gift/family sale** well below fair market value.
3. **Partial-interest sale** mistakenly being treated as a 100% sale.
4. **Severe condition issues** (but mislabeled as e.g. "average")

...or any other hidden issues that could cause a sale to be below market value.

Here's a (fictional) example:

| Finished Sqft | Price | Price/sqft | Median $/sqft | Relative Ratio | Std Dev from Median |  Flagged |
|--------------:|------:|-----------:|--------------:|---------------:|--------------------:|---------:|
| 1,800 | 360,000 | 200 | 200 |           1.00 |               0.00	 |    FALSE |
| 2,000 | 400,000 | 200 | 200 |           1.00 |               0.00	 |    FALSE |
| 1,900 | 380,000 | 200 | 200 |           1.00 |               0.00	 |    FALSE |
| 2,000 | **200,000** | **100** | 200 |           **0.50** |           **-2.23** | **TRUE** |
| 2,100 | 420,000 | 200 | 200 |           1.00 |               0.00	 |    FALSE |

- Normal cluster sales: $360k-$420K at ~$200/sqft.  
- One flagged anomaly: 2,000 sqft at $200K → $100/sqft, half the median.  
- This was likely a distressed sale, partial interest sale, or etc.  

**Recommended action:**

- Check the characteristics, particularly age and condition.  
- Check the transaction details (foreclosure, related-party, partial-interest). If there's nothing physically wrong with the property, and you can't confirm that it's arm's length with valid conditions, consider excluding it as non-representative.  

## 3. High price, in-range square footage

In this case, the sale price itself is anomalously high, but there's nothing remarkable about the square footage.

This can be caused by any number of common issues, including:

1. **Multi-parcel sale classified as single-parcel** and with only one parcel's square footage.
2. **Non-arm's length sale** where the price is artificially inflated.
2. **Bundled intangible assets or personal property** that were not properly accounted for.
3. **Incorrectly calculated partial-interest extrapolation** (e.g., not accounting for an included control premium, then extrapolating the value out to 100%)

Here's a (fictional) example:

| Finished Sqft | Price | Price/sqft | Median $/sqft | Relative Ratio | Std Dev from Median |  Flagged |
|--------------:|------:|-----------:|--------------:|---------------:|--------------------:|---------:|
| 1,800 | 360,000 | 200 | 200 | 1.00 |                0.00 | FALSE |
| 2,000 | 400,000 | 200 | 200 | 1.00 |                0.00 | FALSE |
| 1,900 | 380,000 | 200 | 200 | 1.00 |                0.00 | FALSE |
| 2,100 | 420,000 | 200 | 200 | 1.00 |                0.00 | FALSE |
| 2,000 | **700,000** | **350** | 200 | **1.75** |            **2.24** | **TRUE** |

- Normal cluster sales: $360k-$420K at ~$200/sqft.
- One flagged anomaly: 2,000 sqft at $700K → $350/sqft, 75% higher than the median.
- This was likely a non-arm's length sale, or included intangible assets.

**Recommended action:**

- Check the deed for extra business assets or personal property. Verify if the buyer/seller had any related-party arrangement.   
- Check to make sure this wasn't actually a multi-parcel sale, and that the square footage only pertains to one of the parcels.  
- If this was a partial-interest sale, check for any control premium or other factor that might artificially inflate the amount paid for the portion of the sale. Extrapolating the higher rate to a full 100% interest could have lead to an inflated overall price.  
- If there's nothing physically wrong with the property, and you can't confirm that it's arm's length with valid conditions and isn't multi-parcel, consider excluding it as non-representative.  

## 4. In-range price, high price/sqft, in-range square footage

In this case, the sale price and square footage are in the normal range, but the price per square foot is anomalously high.

Typical causes include:

1. **Under-reported finished area** (the building is actually bigger than stated).
2. **Unrecorded quality upgrades** (e.g., a recent renovation or addition).

In the former case, the price per square foot will be artificially inflated. Even though the square footage is in line with the cluster, the price/sqft signal might indicate that the recorded size is not correct. In the later case, unrecorded characteristics might be enough to push the price per square foot up noticeably, even if the overall price is still within the normal range of variation.

Here's a (fictional) example:

| Finished Sqft | Price | Price/sqft | Median $/sqft | Relative Ratio | Std Dev from Median |  Flagged |
|--------------:|------:|-----------:|--------------:|---------------:|--------------------:|---------:|
| 2,000 | 400,000 | 200 | 200 | 1.00 |                0.00 |    FALSE |
| 1,900 | 380,000 | 200 | 200 | 1.00 |                0.00 |    FALSE |
| 2,100 | 420,000 | 200 | 200 | 1.00 |                0.00 |    FALSE |
| 2,000 | 410,000 | 205 | 200 | 1.03 |                0.23 |    FALSE |
| **1,600** | **400,000** | **250** | 200 | **1.25** |            **2.28** | **TRUE** |

- Normal cluster sales: $380k-$420K at ~$200/sqft.  
- One flagged anomaly: 1,600 sqft at $400K → $250/sqft, 25% higher than the median.  

**Recommended Action**:

- Verify finished area  
- Re-check quality and condition  
- If all the physical characteristics are correct, this could be a perfectly valid sale, just a bit of a natural outlier.  

## 5. In-range price, low price/sqft, in-range square footage

In this case, the sale price and square footage are in the normal range, but the price per square foot is anomalously low.

Typical causes include:

1. **Over-reported finished area** (the real building area is smaller, so the official $/sqft appears artificially low)
2. **Misclassified multi-parcel sale, but in reverse** (the total area includes extra square footage that isn't truly part of the recorded sale)

Here's a (fictional) example:

| Finished Sqft |       Price | Price/sqft | Median $/sqft | Relative Ratio | Std Dev from Median |  Flagged |
|--------------:|------------:|-----------:|--------------:|---------------:|--------------------:|---------:|
|         1,800 |     360,000 |        200 | 200 |           1.00 |                0.00 |    FALSE |
|         2,000 |     400,000 |        200 | 200 |           1.00 |                0.00 |    FALSE |
|         2,100 |     420,000 |        200 | 200 |           1.00 |                0.00 |    FALSE |
|         1,900 |     380,000 |        200 | 200 |           1.00 |                0.00 |    FALSE |
|     **2,500** | **400,000** |    **160** | 200 |       **0.80** |           **-2.24** | **TRUE** |
 