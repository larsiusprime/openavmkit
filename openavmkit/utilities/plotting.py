import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorsys

def get_nice_random_colors(n: int):
  """
  Generate a list of n aesthetically pleasing random colors for plotting
  """
  colors = []
  for i in range(n):
    h = i / n

    # pick a nice saturation value, don't use the full range, just an aesthetic choice:
    s = 0.75 + 0.25 * np.sin(i / n * 2 * np.pi)

    # pick a nice value, don't use the full range, just an aesthetic choice:
    v = 0.75 + 0.25 * np.cos(i / n * 2 * np.pi)

    # convert from hsv to rgb:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    hex_code = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    colors.append(hex_code)
  return colors


def plot_scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    x_lim: tuple = None,
    y_lim: tuple = None,
    out_file: str = None,
    style: dict = None
):
  plt.close('all')

  if xlabel == "":
    xlabel = x
  if ylabel == "":
    ylabel = y
  if title == "":
    title = f"{x} vs {y}"

  color = None
  if style is not None:
    if "random_color_by" in style:
      random_color_by = style.get("random_color_by", "")
      if random_color_by in df:
        unique_values = df[random_color_by].unique()
        colors = get_nice_random_colors(len(unique_values))
        color_map = dict(zip(unique_values, colors))
        color = df[random_color_by].map(color_map).values

  plt.scatter(df[x], df[y], alpha=0.25, c=color)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  if x_lim is not None:
    plt.xlim(x_lim[0], x_lim[1])
  if y_lim is not None:
    plt.ylim(y_lim[0], y_lim[1])
  if out_file is not None:
    plt.savefig(out_file)
  plt.show()


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

