# Variables that drive value

Valuation date: {{{val_date}}}

## Executive Summary

**Modeling group: {{{modeling_group}}}**

The most significant variables that drive real estate value in {{{jurisdiction}}} are:

| Rank | Variable | Description |
|------|----------|-------------|
| 1    | Blah     | Whatever    |
| 2    | Blah     | Whatever    |
| 3    | Blah     | Whatever    |

{{{summary_table}}}

There are countless variables that can be included in any particular model, but only certain ones will wind up being significant. If the most significant variables that drive real estate value can be identified, then {{{jurisdiction}}} can focus their data-gathering efforts on just those.

We perform several kinds of analysis to determine which variables are the most important, an analysis we perform separately for each modeling group.


### Why these variables?

Let's put complicated statistics aside for a moment and just think about what an ideal variable looks like. The ideal variable:

- Drives value
- Is not redundant
- Is intuitive

What do we mean by that?

A variable that **drives value** is one whose presence is strongly associated with what we're trying to predict -- `selling price`. `Total finished square footage` is a typical example -- bigger houses tend to sell for more money, and the bigger they are, the more expensive they are. There can be variables that drive value *down* as well, building age being a typical example -- as a house's age increases (gets older), all else held equal, its price tends to decrease.

A variable is **not redundant** if it is not correlated too much with other variables. Good examples of highly correlated variables include `number of bedrooms`, `number of bathrooms`, and `number of stories`. Most statistical tests will show these all have a positive relationship with price. However, they are all strongly correlated with finished square footage -- bigger homes tend to have more bedrooms, bathrooms, and stories. Just because a variable is correlated with another variable isn't a reason to rule it out, however -- what matters is how *strongly* it is correlated with other variables. Redundant variables can destroy a model's performance and/or make it difficult to interpret.

A variable is **intuitive** if its relationship with price matches what you would naturally expect. You would naturally expect the price to go up when total square footage goes up, for instance, and would expect the value your model puts on total square footage to reflect that. If total square footage causes the marginal price to go *down*, that would be surprising and might suggest the model is **over-fit**. This is the least important characteristic by itself, but can serve as a good gut check on a model's explainability.

### What is an "over-fit" model?

An over-fit model is one that has learned to cheat without actually understanding the material. For instance, you can get a model that seems predictive just by feeding it each parcel's primary key. However, this model hasn't actually learned anything -- it's simply memorizing selling prices for particular keys and echoing them back. We want to force the model to learn the real relationship between `selling price` and the various characteristics like `location`, `built square footage`, `building age`, `building quality`, etc., are, and apply those consistently. The problem with including variables with a weak relationship with `selling price`, or which are redundant with other variables, is they give the model too much irrelevant noise which can distract it from learning the true relationship. We have many methods for detecting and avoiding over-fitting throughout the Open AVM Kit pipeline, starting with variable selection right at the start.


## Pre-modeling analysis

Before modeling, we run various statistical tests to determine which variables are the most likely to be significant. These tests include:

- Correlation testing
- Elastic net regularization
- R-squared analysis
- P-value analysis
- T-value analysis
- Variance inflation factor (VIF) analysis
- Coefficient sign consistency

Each variable is tested against each of these metrics, and given a pass/fail score for each one. Those pass/fail scores are then weighted against the overall importance of each test, then summed together. The variables are then ranked by their total weighted score.

Here are the weights for each statistic:

{{{pre_model_weights}}}

And here are the overall test results for each variable:

{{{pre_model_table}}}

You can find an explanation of each of these statistical tests in the Appendix.

## Post-modeling analysis

After modeling, we run a different series of tests to determine which variables actually wound up being the most significant. These tests include:

- Coefficient analysis
- Principal component analysis
- SHAP value analysis

This lets us know not just which variables wound up being the most important, but also how much value each variable  contributed to the prediction, as measured in dollars.

Here are the overall test results for each variable, expressed as % of overall value for the modeling group.

| Rank | Variable | Contribution | Contribution % |
|------|----------|--------------|----------------|
| 1    | Blah     | 40000        | 40%            |
| 2    | Blah     | 30000        | 30%            |
| 3    | Blah     | 20000        | 20%            |
| 4    | Blah     | 10000        | 10%            |
|      | **Total**| **100000**   | **100%**       |

{{{post_model_table}}}

Statistics nerds can find an explanation of each of statistical tests, along with detailed results for each variable, in the Appendix.

## Appendix

### Correlation analysis

This test looks for variables that are closely associated with `selling price`, but are not too closely associated with one another. We do this by calculating two scores for each variable -- `correlation strength`, which is how closely the variable is associated with `selling price`, and `correlation clarity`, which is how closely the variable is associated with other variables. We normalize these both to 0.0-1.0 scales. An ideal variable will have high `correlation strength`, indicating strong association with `selling price`, and a high `correlation clarity`, indicating little association with other variables.

We then combine these into an overall `correlation score`, which is `(correlation score) * (correlation strength)^2`. This gives us a good single metric for important variables that are not too redundant with others.

Our test considers a correlation score of {{{thresh_correlation}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

| Variable | Correlation Strength | Correlation Clarity | Correlation Score | Pass/Fail |
|----------|----------------------|---------------------|-------------------|-----------|
| Blah     | 0.9                  | 0.1                 | 0.009             | ❌        |
| Blah     | 0.8                  | 0.2                 | 0.032             | ✅        |
| Blah     | 0.7                  | 0.3                 | 0.063             | ✅        |

{{{table_corr_initial}}}

And here are the results after we removed variables with a correlation score lower than {{{thresh_corr_score}}}:

| Variable | Correlation Strength | Correlation Clarity | Correlation Score | Pass/Fail |
|----------|----------------------|---------------------|-------------------|-----------|
| Blah     | 0.8                  | 0.2                 | 0.032             | ✅        |
| Blah     | 0.7                  | 0.3                 | 0.063             | ✅        |

{{{table_corr_final}}}


### VIF -- Variance Inflation Factor

[Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) measures a property called "multi-collinearity." Multi-collinearity is when two or more variables in a regression model are highly correlated. This can cause problems because it can be difficult to determine the effect of each variable on the dependent variable.

| VIF  | Interpretation                                      |
|------|-----------------------------------------------------|
| 1    | No correlation between this variable and the others |
| 1-5  | Moderate correlation                                |
| 5-10 | High correlation                                    |
| 10+  | Very high correlation                               |

Our test considers a VIF lower than {{{thresh_vif}}} as a *pass* (✅).

Here are the initial results from the variables in this model:

| Variable | VIF | Pass/Fail |
|----------|-----|-----------|
| Blah     | 1.2 | ✅        |
| Blah     | 5   | ✅        |
| Blah     | 12  | ❌        |

{{{table_vif_initial}}}

And here are the results after we removed variables with a VIF higher than {{{thresh_vif}}}:

| Variable | VIF | Pass/Fail |
|----------|-----|-----------|
| Blah     | 1.2 | ✅        |
| Blah     | 5   | ✅        |

{{{table_vif_final}}}

### P-value -- Probability value

[P-value](https://en.wikipedia.org/wiki/P-value) is a common measure of statistical significance. It is expressed as a percentage, and pertains to a concept called the "Null hypothesis" -- which in this case, would be the hypothesis that this particular variable being studies is *not* significant and has no relationship with `selling price`.

Contrary to popular belief, p-value is *not* a measure of the probability that the "Null hypothesis" is *true*, instead it is the probability of getting the same statistical result as extreme (or more extreme) than the one we actually got, *in a hypothetical world where the "Null hypothesis" is true.* In other words, p-value represents the chance of us getting the same answer if we stepped in a magic portal and were transported to a world where we were sure that this variable had no relationship with `selling price`.

If a p-value is sufficiently low, that's strong evidence *against* the "null hypothesis" -- ie, it's strong evidence *for* the variable being significant. However, the p-value should never be taken as evidence all by itself, just one indicator among many.

Our test considers a p-value of {{{thresh_p_value}}} or lower as a **pass** (✅).

Here are the initial results from the variables in this model:

| Variable | P-value | Pass/Fail |
|----------|---------|-----------|
| Blah     | 0.01    | ✅        |
| Blah     | 0.1     | ❌        |
| Blah     | 0.5     | ❌        |

{{{table_p_value_initial}}}

And here are the results after we removed variables with a p-value higher than {{{thresh_p_value}}}:

| Variable | P-value | Pass/Fail |
|----------|---------|-----------|
| Blah     | 0.01    | ✅        |

{{{table_p_value_final}}}

### T-value

[T-value](https://en.wikipedia.org/wiki/Student%27s_t-test) is a measure from the Student's T-test. It measures the strength of the relationship between the independent variable and the dependent variable. The higher the T-value, the more significant the relationship.

Our test considers a T-value of {{{thresh_t_value}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

| Variable | T-value | Pass/Fail |
|----------|---------|-----------|
| Blah     | 1       | ❌        |
| Blah     | 2       | ✅        |
| Blah     | 3       | ✅        |

{{{table_t_value_initial}}}

Here are the results after we removed variables with a T-value lower than {{{thresh_t_value}}}:

| Variable | T-value | Pass/Fail |
|----------|---------|-----------|
| Blah     | 2       | ✅        |
| Blah     | 3       | ✅        |

{{{table_t_value_final}}}

### ENR -- Elastic Net Regularization

[Elastic Net Regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization) is a method of "regularization". [Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) is a way to avoid over-fitting; it works by encouraging the model not to focus too much on irrelevant details. Elastic Net Regularization specifically combines the effects of two methods -- "Ridge Regularization", and "Lasso Regularization." Taken together, these two methods help the model avoid relying too heavily on any one variable, and also encourages it to eliminate useless variables entirely.

The bottom line is this method spits out a score for each variable that tells you how important a variable is to the model, with useless variables being shrunk very close to zero.

Our test considers a score of {{{thresh_enr_coef}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

| Variable | ENR   | Pass/Fail |
|----------|-------|-----------|
| Blah     | 0.005 | ❌        |
| Blah     | 0.015 | ✅        |
| Blah     | 0.250 | ✅        |

{{{table_enr_initial}}}

Here are the results after we removed variables with an ENR lower than {{{thresh_enr}}}:

| Variable | ENR   | Pass/Fail |
|----------|-------|-----------|
| Blah     | 0.015 | ✅        |
| Blah     | 0.250 | ✅        |

{{{table_enr_final}}}

### R-squared

[R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination) is a measure of how well the model fits the data. For a single-variable model, it represents the percent of the variation in that variable (say, `finished square footage`) that can be explained by the `selling price`.

For our test, we run a single-variable linear model for every variable in the dataset, and calculate the R-squared and the [Adjusted R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2) (generally considered a more robust statistic than R-squared alone) for each one. A higher value means the variable is more important to the model.

Our test considers a threshold of {{{thresh_r_squared}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

| Variable | Adjusted R-squared | Pass/Fail |
|----------|--------------------|-----------|
| Blah     | 0.05               | ❌        |
| Blah     | 0.5                | ✅        |
| Blah     | 0.9                | ✅        |

{{{table_r_squared_initial}}}

Here are the results after we removed variables with an Adjusted R-squared lower than {{{thresh_r_squared}}}:

| Variable | Adjusted R-squared | Pass/Fail |
|----------|--------------------|-----------|
| Blah     | 0.5                | ✅        |
| Blah     | 0.9                | ✅        |

{{{table_r_squared_final}}}

### Coefficient sign

Three of these tests produce "coefficients" for each variable. These represent the value (in dollars) that the model puts on that variable. I.e, how many dollars one `finished square foot`, is worth, or how many dollars one `building age year` is worth. These coefficients can be either positive (adding to value) or negative (subtracting from value); for instance, `building age year` is a good example of a common negative coefficient -- as a house gets older, it tends to be worth less money (generally speaking).

In models where the variables are weak or the overall model is over-fit, it's not uncommon for the coefficients to *change signs* under different model configurations. That's a warning sign that the variable might be worth dropping as it's not stable and could be confusing the model.

When we run the R-squared test, the Elastic Net Regularization test, and the T-value test, we save the coefficients for each variable. Then we check if the three signs for any given variable are all positive, or all negative. If they are, we consider that a **pass** (✅).

Here are the initial results from the variables in this model:

| Variable | R-Squared Coef | ENR Coef | T-value Coef | Pass/Fail |
|----------|----------------|----------|--------------|-----------|
| Blah     | 1000           | 1000     | 1000         | ✅        |
| Blah     | -1000          | -1000    | -1000        | ✅        |
| Blah     | 1000           | -1000    | 1000         | ❌        |

{{{table_coef_initial}}}

Here are the results after we removed variables with a coefficient sign mismatch:

| Variable | R-Squared Coef | ENR Coef | T-value Coef | Pass/Fail |
|----------|----------------|----------|--------------|-----------|
| Blah     | 1000           | 1000     | 1000         | ✅        |
| Blah     | -1000          | -1000    | -1000        | ✅        |

{{{table_coef_final}}}
