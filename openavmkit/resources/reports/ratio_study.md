# Ratio Studies

|| |
|-|--|
| Locality | {{locality}} |
| Valuation date | {{val_date}} |
| Model group | {{model_group}} |  
| Study period | {{sales_back_to_date}} - {{val_date}} |

## Executive Summary

Ratio studies are the primary tool for evaluating the *accuracy* of mass appraisal predictions. The IAAO defines the standards for these in its [Standard on Ratio Studies](https://www.iaao.org/wp-content/uploads/Standard_on_Ratio_Studies.pdf).

We perform IAAO-standard ratio studies for all model groups, but we  go beyond that with additional relevant statistics, as well as detailed breakdowns by property type, location, price tier, and other relevant factors.

### Overall results

The following table provides overall results for the entire locality for the {{model_group}} model group.

{{overall_results}}

> **Reading the {{locality}} column.** This is an audit of the finished assessment roll
> against *all* sales in the study period — the standard IAAO frame, which has no holdout
> requirement. Our figures here are computed on the same sales. Two things to keep in mind
> when comparing the columns: (1) the comparison is only meaningful if the **valuation date
> above is aligned with the roll-close date** of the values being compared — otherwise the
> two columns describe values as of different dates; and (2) our models are scored on sales
> they never trained on, whereas we cannot know the holdout status of values we did not
> generate. If those values were informed by these same sales, their figures reflect an
> in-sample fit rather than an out-of-sample test — not better or worse work, just a
> different test. The sales-chasing check below helps gauge whether that is in play.

## Sales-chasing check

*Sales chasing* — moving an appraised value toward its observed sale price, whether
deliberately or as a side effect of a methodology — makes a roll look very strong on sold
parcels without improving uniformity among comparable unsold parcels. Because a ratio study
only sees sold parcels, it cannot tell genuine accuracy from sale-conditioning on its own.
The checks below look for that pattern; they are a prompt to interpret a very tight result
with context, not a judgment. See the [tutorial](../../docs/tutorial.md) for background.

{{sales_chasing}}

## Breakdowns

The following tables provide detailed breakdowns of the results by various factors, comparing results for {{locality}} with our own.

### {{locality}} results

{{locality_results}}

### Our results

{{modeler_results}}


