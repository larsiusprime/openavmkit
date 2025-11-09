import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorsys
import mpld3
import statsmodels.api as sm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpld3 import plugins

from IPython.display import display, HTML


def get_nice_random_colors(n: int, shuffle=False, seed=1337):
    """Generate a list of n aesthetically pleasing and perceptually distinct colors for
    plotting.

    Parameters
    ----------
    n : int
        Number of colors
    shuffle : bool, optional
        Whether to shuffle the color order to make the sequence appear more visually distinct. Default is False.
    seed : int
        Random seed for determinicity.

    Returns
    -------
    list[str]
        List of hex color codes
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
    metadata_field: str = None,
) -> Figure:
    """Scatterplot with inline mpld3 tooltips showing df[metadata_field].

    Parameters
    ----------
    df : pd.DataFrame
        Your dataset
    x : str
        Variable for the x-axis
    y : str
        Variable for the y-axis
    xlabel : str, optional
        Label for the x-axis
    ylabel : str, optional
        Label for the y-axis
    title : str, optional
        Title for the plot
    out_file : str, optional
        If provided, writes the image to this file.
    style : dict, optional
        Style dictionary
    best_fit_line : bool, optional
        If True, draws a best fit line.
    perfect_fit_line : bool, optional
        If True, draws a x=y line.
    metadata_field : str, optional
        If provided, shows tooltips for this field on hover.

    Returns
    -------
    Figure
        The plot figure

    """
    # 1) Defaults
    xlabel = xlabel or x
    ylabel = ylabel or y
    title = title or f"{x} vs {y}"

    # 2) New figure & axis
    fig, ax = plt.subplots()

    ax.ticklabel_format(axis='x', style='plain')
    ax.xaxis.set_major_formatter(_human_fmt(digits=3))
    ax.ticklabel_format(axis='y', style='plain')
    ax.yaxis.set_major_formatter(_human_fmt(digits=3))

    # 3) Color/style helper (your existing function)
    color = _get_color_by(df, style)

    # 4) Scatter
    sc = ax.scatter(df[x], df[y], s=4, c=color)

    legend_arr = [None]

    # 5) Optional best-fit line
    if best_fit_line:
        results = _simple_ols(df, x, y, intercept=False)
        slope, intercept, r2 = results["slope"], results["intercept"], results["r2"]
        best_fit_label = f"Best fit line (r²={r2:.2f})"
        ax.plot(
            df[x],
            slope * df[x],
            color="red",
            alpha=0.5,
            label=best_fit_label,
        )
        legend_arr.append(best_fit_label)

    if perfect_fit_line:
        perfect_fit_label = "Perfect fit line (y=x)"
        # Add a perfect line (y=x)
        ax.plot(df[x], df[x], color="blue", alpha=0.5, label=perfect_fit_label)
        legend_arr.append(perfect_fit_label)

    ax.legend(legend_arr)

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
    else:
        plt.show()

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
    style: dict = None,
) -> None:
    """
    Plots a simple bar graph

    Parameters
    ----------
    df : pd.DataFrame
        Your dataset
    data_field : str
        The field you want to graph
    height : float
        The height of the bars
    width : float
        The width of the bars
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis
    title : str
        Title for the plot
    out_file : str
        If provided, writes the image to this file.
    style : dict
        Style dictionary
    """
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


def plot_histogram_df(
    df: pd.DataFrame,
    fields: list[str],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    bins=500,
    x_lim=None,
    out_file: str = None,
) -> None:
    """
    Plots an overlaid histogram of one or more sets of values

    Parameters
    ----------
    df : pd.Dataframe
        Your dataset
    fields : list[str]
        Field or fields you want to plot
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis
    title : str
        Title for the plot
    bins : int, optional
        How many bins
    x_lim : Any, optional
        x-axis limiter
    out_file : str, optional
        If provided, writes the image to this file.
    """
    entries = []
    for field in fields:
        data = df[field]
        entries.append({"data": data, "label": field, "alpha": 0.25})
    _plot_histogram_mult(entries, xlabel, ylabel, title, bins, x_lim, out_file)


#######################################
# PRIVATE
#######################################


def _human_fmt(digits=0):
    """Return a FuncFormatter that appends K, M, B … suffixes."""
    units = [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')]

    def _fmt(x, pos):
        abs_x = abs(x)
        # pick the largest suffix whose value fits the data point
        for value, suffix in units:
            if abs_x >= value:
                scaled, end = x / value, suffix
                break
        else:                                   # no suffix (< 1 000)
            scaled, end = x, ''

        # format with the requested significant-digit budget
        s = f'{scaled:.{digits}g}'             # general format → trims zeros
        return f'{s}{end}'

    return FuncFormatter(_fmt)


def _plot_histogram_mult(
    entries: list[dict],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    bins: int | None = 500,
    x_lim=None,
    out_file: str | None = None,
) -> None:
    plt.close("all")

    plt.ticklabel_format(axis="x", style="plain")          # no sci‑notation
    plt.gca().xaxis.set_major_formatter(ScalarFormatter()) # (optional) turn off offset

    ylim_min, ylim_max = 0, 0

    for entry in entries:
        data = entry["data"].copy()
        if x_lim is not None:
            data = data.clip(*x_lim)
        _bins = bins if bins is not None else entry.get("bins")
        label = entry["label"]
        alpha = entry.get("alpha", 0.25)

        data = data.dropna()
        counts, _, _ = plt.hist(data, bins=_bins, label=label, alpha=alpha)
        ylim_max = max(ylim_max, np.percentile(counts, 95))
    
    plt.gca().xaxis.set_major_formatter(_human_fmt(digits=3))

    # capture y-limits
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if x_lim is not None:
        plt.xlim(*x_lim)
    plt.ylim(ylim_min, ylim_max)

    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
    plt.show()


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


def _simple_ols(
    df_in: pd.DataFrame, ind_var: str, dep_var: str, intercept: bool = True
) -> dict:

    # check for nans/nulls in y:
    df = df_in[~df_in[dep_var].isna() & ~df_in[ind_var].isna()].copy()

    y = df[dep_var]
    X = df[ind_var]
    if intercept:
        X = sm.add_constant(X, has_constant='add')
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    model = sm.OLS(y, X).fit()

    return {
        "slope": model.params[ind_var],
        "intercept": model.params["const"] if "const" in model.params else 0,
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "pval": model.pvalues[ind_var],
        "mse": model.mse_resid,
        "rmse": np.sqrt(model.mse_resid),
        "std_err": model.bse[ind_var],
    }

