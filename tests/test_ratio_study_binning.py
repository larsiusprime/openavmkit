import warnings

import numpy as np
import pandas as pd
import pytest

from openavmkit.ratio_study import _compute_breakdown_edges, _run_ratio_study_breakdowns


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


def test_degenerate_columns():
    # all-NaN column -> nothing to bin -> caller skips the breakdown
    assert _compute_breakdown_edges(pd.Series([np.nan, np.nan, np.nan]), quantiles=5) == (None, None)
    assert _compute_breakdown_edges(pd.Series([], dtype=float), quantiles=5) == (None, None)
    # constant column -> one bin holding every row (as before for a positive constant)
    for v in (0.0, 1.0, -3.0):
        s = pd.Series([v] * 50)
        edges, labels = _compute_breakdown_edges(s, quantiles=5)
        assert len(labels) == 1
        assert _bin_counts(s, edges, labels) == [50]


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


# --- Bins start at the minimum, never at 0 -----------------------------------------------


def _bin_counts(values, edges, labels):
    cut = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
    return [int((cut == label).sum()) for label in labels]


def test_quantiles_give_equal_count_bins_with_no_sliver_bin():
    # The old builder always started at edge 0, so the first "bin" ran from 0 to the
    # minimum and held a single row: quantiles=5 gave 6 slices, the first of size 1.
    values = pd.Series(np.random.default_rng(0).lognormal(7.5, 0.4, 1000))
    edges, labels = _compute_breakdown_edges(values, quantiles=5)
    assert _bin_counts(values, edges, labels) == [200] * 5
    assert labels[0].startswith(" 20th %ile")


def test_negative_values_are_binned_not_dropped():
    # Old code crashed; starting the bins at 0 would silently drop every negative row.
    values = pd.Series(np.random.default_rng(1).normal(0, 10, 500))
    edges, labels = _compute_breakdown_edges(values, quantiles=5)
    counts = _bin_counts(values, edges, labels)
    assert counts == [100] * 5
    assert sum(counts) == len(values)


def test_tied_minimum_keeps_its_own_bin():
    # e.g. stories: 70% are 1-story. The 1s must not be lumped in with the 2s.
    values = pd.Series([1.0] * 70 + [2.0] * 25 + [3.0] * 5)
    edges, labels = _compute_breakdown_edges(values, quantiles=5)
    assert _bin_counts(values, edges, labels) == [70, 25, 5]
    assert labels[0] == " 60th %ile<br>(1 - 1)"  # highest percentile the tied value reaches


@pytest.mark.parametrize("seed", range(20))
def test_every_row_lands_in_exactly_one_non_empty_bin(seed):
    rng = np.random.default_rng(seed)
    kind = seed % 4
    if kind == 0:
        values = rng.integers(-3, 4, 200).astype(float)          # few distinct values
    elif kind == 1:
        values = np.concatenate([np.zeros(150), rng.uniform(0, 1, 50)])  # zero-inflated
    elif kind == 2:
        values = rng.normal(0, 1, 200)
        values[rng.choice(200, 30, replace=False)] = np.nan       # partial coverage
    else:
        values = rng.lognormal(12, 1, 200).round(-4)              # rounded prices, many ties
    s = pd.Series(values)
    edges, labels = _compute_breakdown_edges(s, quantiles=10)
    assert _strictly_increasing(edges)
    counts = _bin_counts(s, edges, labels)
    assert min(counts) > 0
    assert sum(counts) == int(s.notna().sum())
    assert len(set(labels)) == len(labels)


def test_quantile_labels_keep_their_format():
    values = pd.Series([100.0, 200.0, 300.0, 400.0])
    _, labels = _compute_breakdown_edges(values, quantiles=4)
    assert labels == [
        " 25th %ile<br>(100 - 100)",
        " 50th %ile<br>(100 - 200)",
        " 75th %ile<br>(200 - 300)",
        "100th %ile<br>(300 - 400)",
    ]


# --- Explicit bins are validated ----------------------------------------------------------

@pytest.mark.parametrize(
    "bins_cfg, bin_labels, match",
    [
        ([0, 0.5, 0.5, 1], None, "strictly increasing"),
        ([1, 0], None, "strictly increasing"),
        ([0], None, "strictly increasing"),
        ([0, "half", 1], None, "list of numbers"),
        ([0, 0.5, 1], ["only one"], "one label per bin"),
        ([0, 0.5, 1], ["same", "same"], "unique"),
    ],
)
def test_explicit_bins_reject_bad_config(bins_cfg, bin_labels, match):
    with pytest.raises(ValueError, match=match):
        _compute_breakdown_edges(pd.Series([0.1]), bins_cfg=bins_cfg, bin_labels=bin_labels, field="pct_x")


# --- End to end through the ratio study ---------------------------------------------------

def _sales(n=300):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "key": [f"k{i}" for i in range(n)],
        "key_sale": [f"s{i}" for i in range(n)],
        "sale_price": rng.lognormal(12, 0.4, n),
        "sale_year": 2024,
        "vacant_sale": False,
        "valid_for_ratio_study": True,
        "valid_for_land_ratio_study": True,
        "elev_delta": rng.normal(0, 10, n),
        "all_nan": np.nan,
        "pct_x": rng.uniform(0, 1, n),
    })
    df["prediction"] = df["sale_price"] * rng.normal(1, 0.1, n)
    df.loc[:20, "elev_delta"] = np.nan
    return df


def _breakdowns(breakdowns):
    settings = {
        "modeling": {"metadata": {"valuation_date": "2025-01-01"}},
        "analysis": {"ratio_study": {"look_back_years": 1, "breakdowns": breakdowns}},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run_ratio_study_breakdowns(settings, "all", _sales(), iterations=10)
    by = {r["by"]: r["cluster"] for r in out["openavmkit"]["improved"]}
    messages = [str(w.message) for w in caught if "Ratio study breakdown" in str(w.message)]
    return by, messages


def test_ratio_study_survives_nan_and_negative_values():
    # Previously this crashed the whole ratio study.
    by, messages = _breakdowns([{"by": "elev_delta", "quantiles": 4}])
    assert len(by["elev_delta"]) == 4
    assert messages == []


def test_ratio_study_skips_all_nan_column_with_a_warning():
    by, messages = _breakdowns([{"by": "all_nan", "quantiles": 4}])
    assert by["all_nan"] == {}
    assert len(messages) == 1 and "all_nan" in messages[0] and "Skipping" in messages[0]


def test_ratio_study_explicit_bins_warn_about_rows_outside_edges():
    by, messages = _breakdowns([{"by": "pct_x", "bins": [0, 0.5, 0.9], "bin_labels": ["low", "high"]}])
    assert set(by["pct_x"]) == {"low", "high"}
    assert len(messages) == 1 and "fall outside the `bins` edges" in messages[0]
