import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_histogram_df(df: pd.DataFrame, fields: list[str], xlabel: str = "", ylabel: str = "", title: str = "", bins = 500, x_lim=None, out_file: str = None):
  entries = []
  for field in fields:
    data = df[field]
    entries.append({
      "data": data,
      "label": field,
      "alpha": 0.25
    })
  plot_histogram_mult(entries, xlabel, ylabel, title, bins, x_lim, out_file)

def plot_histogram_mult(entries: list[dict],  xlabel:str = "", ylabel: str = "", title: str = "", bins=500, x_lim=None, out_file: str = None):
  plt.close('all')
  ylim_min = 0
  ylim_max = 0
  for entry in entries:
    data = entry["data"].copy()
    if x_lim is not None:
      data[data.lt(x_lim[0])] = x_lim[0]
      data[data.gt(x_lim[1])] = x_lim[1]
    if bins is not None:
      _bins = bins
    else:
      _bins = data.get("bins", None)
    label = entry["label"]
    alpha = entry.get("alpha", 0.25)
    data = data[~data.isna()]
    counts, _, _ = plt.hist(data, bins=_bins, label=label, alpha=alpha)
    _ylim_max = np.percentile(counts, 95)
    if(_ylim_max > ylim_max):
      ylim_max = _ylim_max
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend()
  if x_lim is not None:
    plt.xlim(x_lim[0], x_lim[1])
  plt.ylim(ylim_min, ylim_max)
  if out_file is not None:
    plt.savefig(out_file)
  plt.show()

def highest_middle_quantile_count(series: pd.Series, min_value: float, max_value: float, num_quantiles:int):
  series = series[~series.isna()]
  series = series[series.ge(min_value) & series.le(max_value)]
  if num_quantiles < 3:
    raise ValueError("Number of quantiles must be at least 3")
  quantiles = pd.qcut(series, q=num_quantiles, duplicates="drop")
  quantile_counts = quantiles.value_counts()
  return quantile_counts.max()