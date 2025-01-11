# Variables that drive value

|  |  |
|--|--|
| Locality | {{{locality}}} |  
| Valuation date | {{{val_date}}} |  
| Modeling group | {{{modeling_group}}} |

## Executive Summary

The most significant variables that drive real estate value in {{{locality}}} are:

{{{summary_table}}}

There are countless variables that can be included in any particular model, but only certain ones will wind up being significant. If the most significant variables that drive real estate value can be identified, then {{{locality}}} can focus their data-gathering efforts on just those.

We perform several kinds of analysis to determine which variables are the most important, an analysis we perform separately for each modeling group.


### Why these variables?

Let's put complicated statistics aside for a moment and just think about what an ideal variable looks like. The ideal variable:

- Drives value
- Is not redundant
- Is intuitive

What do we mean by that?

A variable that **drives value** is one whose presence is strongly associated with what we're trying to predict -- `selling price`. A variable like `finished square footage` is a typical example -- bigger houses tend to sell for more money, and the bigger they are, the more expensive they are. There can be variables that drive value *down* as well, `building age in years` being a typical example -- as a house's age increases (gets older), all else held equal, its price tends to decrease.

A variable is **not redundant** if it is not correlated too much with other variables. What are some variables that are highly correlated with others? Good examples include `number of bedrooms`, `number of bathrooms`, and `number of stories`. In most localities, these will all have a positive relationship with price, but also a strong relationship with one another. That's because they're all strongly correlated with `finished square footage` -- bigger homes tend to have more bedrooms, bathrooms, and stories. Some correlation with other variables is okay, but truly redundant variables can not only destroy a model's predictive power, they can also make it impossible to interpret.

Finally, a variable is **intuitive** if its relationship with price matches what common sense would predict. We naturally expect `selling price` to go up when `finished square footage` goes up, for instance. If `finished square footage` causes the marginal price to go *down*, that would be surprising and might suggest the model is **over-fit**. Noticing when variables defy intuition can serve as a good gut-check.

### What is an "over-fit" model?

An over-fit model is one that has learned to cheat at its predictions without understanding the underlying material. For instance, one can create a highly predictive model just by feeding it `selling price` paired to each parcel's `primary key`. However, this model hasn't actually learned anything -- it's simply memorizing individual prices and echoing them back without understanding anything about the houses themselves. We want to force the model to learn the real relationship between `selling price` and fundamental characteristics like `location`, `built square footage`, `building age`, `building quality`, etc. The problem with variables with a weak relationship with `selling price`, or which are highly correlated with other variables, is that they distract and overwhelm the model with noise. We have many methods for detecting and avoiding over-fitting throughout the Open AVM Kit pipeline, starting with variable analysis.


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

{{{table_corr_initial}}}

And here are the results after we removed variables with a correlation score lower than {{{thresh_correlation}}}:

{{{table_corr_final}}}


### VIF -- Variance Inflation Factor

[Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor), or VIF, is another measure of a variable's correlation with other variables ("multi-collinearity"). Variables with high VIF can cause problems because it can be difficult to determine the effect of each variable on the dependent variable.

| VIF  | Interpretation                                      |
|------|-----------------------------------------------------|
| 1    | No correlation between this variable and the others |
| 1-5  | Moderate correlation                                |
| 5-10 | High correlation                                    |
| 10+  | Very high correlation                               |

Our test considers a VIF lower than {{{thresh_vif}}} as a *pass* (✅).

Here are the initial results from the variables in this model:

{{{table_vif_initial}}}

And here are the results after we removed variables with a VIF higher than {{{thresh_vif}}}:

{{{table_vif_final}}}

### P-value -- Probability value

[P-value](https://en.wikipedia.org/wiki/P-value) is a common measure of statistical significance. It is expressed as a percentage, and pertains to a concept called the "null hypothesis." What's that? In this context, it's the hypothesis that whatever variable we're looking at basically does nothing; it's *not* significant and has no relationship with `selling price`. Imagine that we stepped into a magic portal to another world, a world in which we are 100% sure that whatever variable we are studying is totally meaningless. The p-value tells us our likelihood of getting the same statistical signal (or an even stronger one) in that world, purely by chance, that we actually get here in the real world, where we're not sure whether the variable is significant or not.

Here's a made up example -- is `vinyl siding` a significant variable or not? Imagine we have access to two worlds -- Earth 1 and Earth 2, and we can travel between them any time. Over on Earth 2, God comes down from the heavens and informs us that `vinyl siding` has absolutely no statistical relationship whatsoever with `selling price`. But over here on Earth 1, God won't tell us what the relationship is. We run some naive statistics, and it sure _looks_ like there might be a relationship -- `vinyl siding` is positively correlated with sale price, and it has a decent r-squared statistic. We run a p-value test on `vinyl siding`, and it comes out to 15% -- meaning that we have a 15% chance of getting the same good-looking statistics on Earth 2 as we did here on Earth 1, even though we know that `vinyl siding` isn't significant over there.

A common misconception is that the p-value directly tells you the % chance that your variable is significant. That's not actually true. It just tells you how likely it is for you to get the same good-looking results (or even better ones)--even in a world where your variable is actually meaningless--purely by chance.

The bottom line is that the lower the p-value is, the better. A sufficiently low p-value is strong evidence *against* the "null hypothesis." That is, it's strong evidence *for* the variable being significant. 

However, the p-value should never be taken as evidence all by itself, just one indicator among many.

Our test considers a p-value of {{{thresh_p_value}}} or lower as a **pass** (✅).

Here are the initial results from the variables in this model:

{{{table_p_value_initial}}}

And here are the results after we removed variables with a p-value higher than {{{thresh_p_value}}}:

{{{table_p_value_final}}}

### T-value

[T-value](https://en.wikipedia.org/wiki/Student%27s_t-test) is a measure from the Student's T-test. It measures the strength of the relationship between the independent variable and the dependent variable. The higher the T-value, the more significant the relationship.

Our test considers a T-value with an absolute value of {{{thresh_t_value}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

{{{table_t_value_initial}}}

Here are the results after we removed variables with a T-value whose absolute value is lower than {{{thresh_t_value}}}:

{{{table_t_value_final}}}

### ENR -- Elastic Net Regularization

[Elastic Net Regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization) uses a method called [Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) to avoid over-fitting; it works by encouraging the model not to focus too much on irrelevant details. Elastic Net Regularization specifically combines the effects of two methods -- "Ridge Regularization", and "Lasso Regularization." Taken together, these two methods help the model avoid relying too heavily on any one variable, and also encourages it to eliminate useless variables entirely.

The bottom line is this method spits out a score for each variable that tells you how important that variable is to the model, with useless variables being shrunk very close to zero.

Our test considers a score of {{{thresh_enr_coef}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

{{{table_enr_initial}}}

Here are the results after we removed variables with an ENR lower than {{{thresh_enr}}}:

{{{table_enr_final}}}

### R-squared

[R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination) is a measure of how well the model fits the data. For a single-variable model, it represents the percent of the variation in that variable (say, `finished square footage`) that can be explained by the `selling price`.

For our test, we run a single-variable linear model for every variable in the dataset, and calculate the R-squared and the [Adjusted R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2) (generally considered a more robust statistic than R-squared alone) for each one. A higher value means the variable is more important to the model.

Our test considers a threshold of {{{thresh_adj_r2}}} or higher as a **pass** (✅).

Here are the initial results from the variables in this model:

{{{table_adj_r2_initial}}}

Here are the results after we removed variables with an Adjusted R-squared lower than {{{thresh_adj_r2}}}:

{{{table_adj_r2_final}}}

### Coefficient sign

Three of these tests produce "coefficients" for each variable. These represent the value (in dollars) that the model puts on that variable. I.e, how many dollars one `finished square foot`, is worth, or how many dollars one `building age year` is worth. These coefficients can be either positive (adding to value) or negative (subtracting from value); for instance, `building age year` is a good example of a common negative coefficient -- as a house gets older, it tends to be worth less money (generally speaking).

In models where the variables are weak or the overall model is over-fit, it's not uncommon for the coefficients to *change signs* under different model configurations. That's a warning sign that the variable might be worth dropping as it's not stable and could be confusing the model.

When we run the R-squared test, the Elastic Net Regularization test, and the T-value test, we save the coefficients for each variable. Then we check if the three signs for any given variable are all positive, or all negative. If they are, we consider that a **pass** (✅).

Here are the initial results from the variables in this model:

{{{table_coef_sign_initial}}}

Here are the results after we removed variables with a coefficient sign mismatch:

{{{table_coef_sign_final}}}
