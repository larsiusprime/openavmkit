import numpy as np
import pandas as pd
import statsmodels.api as sm


def project_trend(time_series: np.ndarray, time_index: int):
    """Fits a linear trend of your observed time series values,
    then gives you the projected value at the specified time index.

    Parameters
    ----------
    time_series : np.ndarray
        Time series values

    time_index : int
        The array index representing the time for which you want to predict a value for

    Returns
    -------
    float
        The predicted value at `time_index`
    """

    if len(time_series) < 2:
        # Not enough data for a measurable trend
        return time_series[0]

    y = [i for i in range(0, len(time_series))]
    const = [1.0 for i in range(0, len(time_series))]
    x = pd.DataFrame(data={"slope": time_series, "intercept": const})

    model = sm.OLS(y, x, hasconst=False).fit()

    # given:
    # y = mx + b
    # y - b = mx
    # (y - b)/m = x

    # solve for x:
    return (time_index - model.params["intercept"]) / model.params["slope"]
