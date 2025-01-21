import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram_df(df: pd.DataFrame, fields: str, xlabel: str = "", ylabel: str = "", title: str = "", bins = 500, xlim=None, ylim=None, out_file: str = None):
  entries = []
  for field in fields:
    data = df[field]
    entries.append({
      "data": data,
      "label": field,
      "alpha": 0.25
    })
  plot_histogram_mult(entries, xlabel, ylabel, title, bins, xlim, ylim, out_file)


def plot_histogram_mult(entries: list[dict],  xlabel:str = "", ylabel: str = "", title: str = "", bins=500, xlim=None, ylim=None, out_file: str = None):
  plt.close('all')
  for entry in entries:
    data = entry["data"]
    if bins is not None:
      _bins = bins
    else:
      _bins = data.get("bins", None)
    label = entry["label"]
    alpha = entry.get("alpha", 0.25)
    data = data[~data.isna()]
    plt.hist(data, bins=_bins, label=label, alpha=alpha)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend()
  if xlim is not None:
    plt.xlim(xlim[0], xlim[1])
  if ylim is not None:
    plt.ylim(ylim[0], ylim[1])
  if out_file is not None:
    plt.savefig(out_file)
  plt.show()


def highest_middle_quantile_count(series: pd.Series, num_quantiles):
  series = series[~series.isna()]
  if num_quantiles < 3:
    raise ValueError("Number of quantiles must be at least 3")
  quantiles = pd.qcut(series, q=num_quantiles, duplicates="drop")
  quantile_counts = quantiles.value_counts().sort_index()
  i_low = 1
  i_high = 1
  return quantile_counts.quantile(0.95) * 2