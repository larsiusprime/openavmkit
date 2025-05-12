import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorsys

def get_nice_random_colors(n: int, shuffle=False, seed=1337):
  """
  Generate a list of n aesthetically pleasing and perceptually distinct colors for plotting.

  Parameters:
  - n: Number of colors
  - shuffle: Whether to shuffle the color order to make the sequence appear more visually distinct

  Returns:
  - List of hex color codes
  """
  colors = []
  golden_ratio_conjugate = 0.61803398875  # For perceptually even hue spacing
  random.seed(seed)
  h = random.random()  # Start with a random hue base

  for i in range(n):
    # Evenly spaced hue using golden ratio increment
    h = (h + golden_ratio_conjugate) % 1

    # Fix saturation and value in a pleasing range
    s = 0.65  # More muted than full saturation
    v = 0.85  # High value, good for line plots on white background

    # Convert to RGB and then to hex
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    hex_code = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    colors.append(hex_code)

  if shuffle:
    random.shuffle(colors)

  return colors

def _get_color_by(df: pd.DataFrame, style: dict):
  color = None
  if style is not None:
    if "random_color_by" in style:
      random_color_by = style.get("random_color_by", "")
      if random_color_by in df:
        unique_values = df[random_color_by].unique()
        colors = get_nice_random_colors(len(unique_values))
        color_map = dict(zip(unique_values, colors))
        color = df[random_color_by].map(color_map).values
  return color

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

  color = _get_color_by(df, style)

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


def plot_bar(
    df: pd.DataFrame,
    data_field: str,
    height = 1.0,
    width = 1.0,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    out_file: str = None,
    style: dict = None
):
  color = _get_color_by(df, style)

  df = df.sort_values(by=data_field, ascending=True)

  data = df[data_field]
  data = data[~data.isna()]

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.bar(data, height=height, width=width, color=color)
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

