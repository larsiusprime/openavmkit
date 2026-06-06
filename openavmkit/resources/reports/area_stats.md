# Area statistics that drive value

|  |  |
|--|--|
| Locality | {{locality}} |
| Valuation date | {{val_date}} |
| Sale price field | {{sale_field}} |
| Features evaluated | {{num_features}} |

## Executive summary

*Area statistics* (also called neighborhood enrichment) summarize each parcel's surroundings.
For every location you care about — neighborhood, census tract, city, and so on — we compute
summary statistics (mean, median, dispersion, dominant category, ...) of the characteristics you
select, then stamp those summaries back onto every parcel as new variables named
`area_stat_<location>_<field>_<stat>`.

These behave like a quantized, uniform form of spatial lag: rather than a smooth surface drawn from
nearby sales, they describe the discrete neighborhood a parcel sits in. Coarser locations act as a
natural fallback for finer ones where data is thin.

## Correlation with sale price

The table below ranks every generated area-stat feature by how strongly it correlates with sale
price. **Correlation strength** is the absolute correlation with sale price (higher is better).
**Correlation clarity** is how *un*-correlated the feature is with the other features (higher means
less redundant). **Correlation score** combines the two as `strength × clarity²`.

Sale-derived features are computed from the training set of valid sales only, so this ranking does
not leak the target.

{{correlation_table}}
