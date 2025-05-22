import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorsys
import mpld3
import statsmodels.api as sm
from mpld3 import plugins

from IPython.display import display, HTML


def _simple_ols(
    df: pd.DataFrame,
    ind_var: str,
    dep_var: str,
    intercept: bool = True
):
  y = df[dep_var].copy()
  X = df[ind_var].copy()
  if intercept:
    X = sm.add_constant(X)
  X = X.astype(np.float64)
  model = sm.OLS(y, X).fit()

  return {
    "slope": model.params[ind_var],
    "intercept": model.params["const"] if "const" in model.params else 0,
    "r2": model.rsquared,
    "adj_r2": model.rsquared_adj,
    "pval": model.pvalues[ind_var],
    "mse": model.mse_resid,
    "rmse": np.sqrt(model.mse_resid),
    "std_err": model.bse[ind_var]
  }


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
    df,
    x: str,
    y: str,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    out_file: str = None,
    style: dict = None,
    best_fit_line: bool = False,
    perfect_fit_line: bool = False,
    metadata_field: str = None
):
  """
  Scatterplot with inline mpld3 tooltips showing df[metadata_field].
  """
  # 1) Defaults
  xlabel = xlabel or x
  ylabel = ylabel or y
  title   = title   or f"{x} vs {y}"

  # 2) New figure & axis
  fig, ax = plt.subplots()

  # 3) Color/style helper (your existing function)
  color = _get_color_by(df, style)

  # 4) Scatter
  sc = ax.scatter(df[x], df[y], s=4, c=color)

  # 5) Optional best‐fit line
  if best_fit_line:
    results = _simple_ols(df, x, y, intercept=False)
    slope, intercept, r2 = results["slope"], results["intercept"], results["r2"]
    ax.plot(df[x], slope * df[x],
      color="red", alpha=0.5, label=f"Best fit line (r²={r2:.2f})")

  if perfect_fit_line:
    # Add a perfect line (y=x)
    ax.plot(df[x], df[x], color="blue", alpha=0.5, label="Perfect Line (y=x)")

  # 6) Labels & title
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  # 7) Save if requested
  if out_file:
    fig.savefig(out_file)

  # 8) Build tooltip labels from your metadata field
  if metadata_field is not None:
    labels = df[metadata_field].astype(str).tolist()
    tooltip = plugins.PointLabelTooltip(sc, labels=labels)
    plugins.connect(fig, tooltip)

  # 9) Display the interactive HTML
  html = mpld3.fig_to_html(fig)
  display(HTML(html))

  # Close the plot without showing it:
  plt.close(fig)

  return fig


def plot_bar(
    df: pd.DataFrame,
    data_field: str,
    height=1.0,
    width=1.0,
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


def plot_histogram_df(df: pd.DataFrame, fields: list[str], xlabel: str = "", ylabel: str = "", title: str = "",
    bins=500, x_lim=None, out_file: str = None):
  entries = []
  for field in fields:
    data = df[field]
    entries.append({
      "data": data,
      "label": field,
      "alpha": 0.25
    })
  plot_histogram_mult(entries, xlabel, ylabel, title, bins, x_lim, out_file)


def plot_histogram_mult(entries: list[dict], xlabel: str = "", ylabel: str = "", title: str = "", bins=500, x_lim=None,
    out_file: str = None):
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
    if (_ylim_max > ylim_max):
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
