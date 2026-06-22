import numpy as np
import pandas as pd

from openavmkit.ratio_study import _compute_breakdown_edges


def _strictly_increasing(edges):
    return all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))


def test_quantiles_handle_zero_inflated_column():
    # 80% of rows share value 0, then a spread. Equal-count quantile edges
    # collapse onto 0; the previous np.quantile + "not in bins" logic produced
    # non-monotonic edges and pd.cut raised "bins must increase monotonically".
    values = pd.Series([0.0] * 80 + list(np.linspace(0.05, 1.0, 20)))
    edges, labels = _compute_breakdown_edges(values, quantiles=5)
    assert edges is not None
    assert _strictly_increasing(edges)
    assert len(labels) == len(edges) - 1


def test_quantiles_ignore_nan():
    values = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0])
    edges, labels = _compute_breakdown_edges(values, quantiles=4)
    assert edges is not None
    assert all(np.isfinite(e) for e in edges)
    assert _strictly_increasing(edges)


def test_degenerate_columns_return_none():
    # constant column -> no distinct edges -> caller skips the breakdown
    assert _compute_breakdown_edges(pd.Series([0.0] * 50), quantiles=5) == (None, None)
    # all-NaN column -> same
    assert _compute_breakdown_edges(pd.Series([np.nan, np.nan, np.nan]), quantiles=5) == (None, None)


def test_explicit_bins_with_labels():
    edges, labels = _compute_breakdown_edges(
        pd.Series([0.05, 0.3, 0.8]),
        bins_cfg=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
        bin_labels=["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"],
    )
    assert edges == [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    assert labels == ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"]


def test_explicit_bins_autogenerate_labels():
    edges, labels = _compute_breakdown_edges(pd.Series([1, 2, 3]), bins_cfg=[0, 10, 20])
    assert edges == [0, 10, 20]
    assert len(labels) == 2
