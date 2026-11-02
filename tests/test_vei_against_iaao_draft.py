"""VEI must reproduce the IAAO exposure draft's own worked example.

Source: IAAO *Standard on Ratio Studies*, Exposure Draft May 2026 --
Appendix E ("Vertical Equity Indicator"), with the 54-observation sample from
Section 6.1 Table 1 and the published results in Appendix E.4 Table 25.

Published answers (R7 percentile method, which is numpy/pandas' default):

    sample median ratio      0.8678
    Quartile 1   14 sales    median 0.7280   CI [0.6429, 0.8476]
    Quartile 4   14 sales    median 1.1491   CI [1.0000, 1.2476]
    VEI                      48.53%
    VEI Significance         17.57%
"""
import numpy as np
import pandas as pd
import pytest

from openavmkit.utilities.stats import calc_median_confidence_interval
from openavmkit.vertical_equity_study import get_vertical_equity_scores


# Section 6.1, Table 1 -- assessed value and sale price for 54 observations.
_AV = [240000, 144000, 392000, 199200, 340800, 472000, 336000, 284800, 436000,
       191200, 225000, 194000, 481600, 338400, 164800, 252800, 945000, 280000,
       407000, 423000, 306800, 236800, 361000, 290400, 235200, 415000, 381000,
       380000, 875000, 680000, 329000, 457000, 516000, 295200, 580000, 840000,
       640000, 660000, 354000, 800000, 390000, 800000, 469000, 1000000, 920000,
       527000, 800000, 786000, 496000, 1000000, 320000, 960000, 952000, 327200]
_SP = [690000, 296250, 787500, 372000, 574500, 795000, 559500, 465000, 693600,
       298500, 350000, 295000, 732000, 495000, 237000, 352500, 1289000, 379000,
       540000, 560000, 390000, 300000, 450000, 345000, 277500, 485000, 442000,
       435000, 998000, 772500, 365000, 487000, 547500, 300000, 580000, 840000,
       622500, 637500, 340000, 750000, 352500, 705000, 410000, 859500, 787500,
       440000, 648000, 630000, 388500, 765000, 243750, 720000, 705000, 240000]


@pytest.fixture
def draft_sample():
    assert len(_AV) == len(_SP) == 54
    return pd.DataFrame({"sale_price": _SP, "prediction": _AV})


@pytest.fixture
def scores(draft_sample):
    return get_vertical_equity_scores(draft_sample, "sale_price", "prediction")


def test_vei_point_estimate_matches_the_draft(scores):
    """E.1 Step 5: VEI = 100 * ((MEDIAN Last PG - MEDIAN First PG) / Sample MEDIAN)."""
    assert scores["vei"] == pytest.approx(48.53, abs=0.02)


def test_vei_significance_matches_the_draft(scores):
    """E.1 Step 7, using the highest- and lowest-median groups."""
    assert scores["vei_significance"] == pytest.approx(17.57, abs=0.02)


def test_percentile_group_medians_match_the_draft(scores):
    gs = scores["group_stats"]
    assert len(gs) == 4, "54 observations must split into quartiles (E.1 Step 3)"
    assert gs.loc[0, "ratio"] == pytest.approx(0.7280, abs=5e-4)
    assert gs.loc[3, "ratio"] == pytest.approx(1.1491, abs=5e-4)


def test_percentile_group_confidence_intervals_match_the_draft(scores):
    """The CIs are order statistics per Appendix D.2, not a t-interval on the mean.

    A symmetric mean-based interval does not reproduce these bounds.
    """
    gs = scores["group_stats"]
    assert gs.loc[0, "lower"] == pytest.approx(0.6429, abs=5e-4)
    assert gs.loc[0, "upper"] == pytest.approx(0.8476, abs=5e-4)
    assert gs.loc[3, "lower"] == pytest.approx(1.0000, abs=5e-4)
    assert gs.loc[3, "upper"] == pytest.approx(1.2476, abs=5e-4)


# ---------------------------------------------------------------------------
# Appendix D.2 median confidence interval
# ---------------------------------------------------------------------------

def test_median_ci_is_made_of_order_statistics():
    """Both bounds must be actual observations, never interpolated values."""
    rng = np.random.default_rng(2)
    for n in (13, 14, 31, 40, 101, 500):
        v = rng.lognormal(0, 0.3, n)
        lo, hi = calc_median_confidence_interval(v, 0.90)
        assert lo in set(v.tolist()), f"n={n}: lower bound is not an observation"
        assert hi in set(v.tolist()), f"n={n}: upper bound is not an observation"
        assert lo <= np.median(v) <= hi, f"n={n}: interval does not bracket the median"


def test_median_ci_ranks_match_the_draft_formula():
    """n=14 with z=1.645 selects ranks 4 and 11, as Appendix E.4 shows."""
    v = np.arange(1.0, 15.0)  # rank i has value i, so bounds reveal the ranks
    lo, hi = calc_median_confidence_interval(v, 0.90)
    assert (lo, hi) == (4.0, 11.0)


def test_median_ci_widens_with_confidence():
    rng = np.random.default_rng(5)
    v = rng.normal(100, 10, 200)
    lo90, hi90 = calc_median_confidence_interval(v, 0.90)
    lo95, hi95 = calc_median_confidence_interval(v, 0.95)
    assert lo95 <= lo90 and hi95 >= hi90


def test_median_ci_rejects_unsupported_confidence():
    with pytest.raises(ValueError):
        calc_median_confidence_interval(np.arange(50.0), 0.99)


def test_median_ci_handles_tiny_and_empty_samples():
    assert all(np.isnan(x) for x in calc_median_confidence_interval(np.array([]), 0.90))
    # a 3-observation sample saturates at the extremes rather than raising
    lo, hi = calc_median_confidence_interval(np.array([1.0, 2.0, 3.0]), 0.90)
    assert (lo, hi) == (1.0, 3.0)


def test_vei_significance_uses_highest_and_lowest_median_groups():
    """Not simply the last and first groups -- draft p.79 is explicit about this.

    Built so the LAST group is not the highest-median group: ratios rise across the
    value proxy and then fall back in the top group.
    """
    rng = np.random.default_rng(11)
    n = 600
    sp = np.sort(rng.uniform(100_000, 900_000, n))
    ratio = np.interp(np.arange(n), [0, n * 0.8, n - 1], [0.80, 1.30, 0.95])
    ratio = ratio + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"sale_price": sp, "prediction": sp * ratio})

    res = get_vertical_equity_scores(df, "sale_price", "prediction")
    gs = res["group_stats"]
    last, first = gs.index.max(), gs.index.min()
    assert gs["ratio"].idxmax() != last, (
        "fixture failed to make a non-monotonic shape; the test would prove nothing"
    )

    expected = 100 * (
        gs.loc[gs["ratio"].idxmax(), "lower"] - gs.loc[gs["ratio"].idxmin(), "upper"]
    ) / df["prediction"].div(df["sale_price"]).median()
    assert res["vei_significance"] == pytest.approx(expected, rel=1e-9)

    # and it genuinely differs from the first/last reading the code used to take
    naive = 100 * (gs.loc[last, "lower"] - gs.loc[first, "upper"]) / \
        df["prediction"].div(df["sale_price"]).median()
    assert not np.isclose(res["vei_significance"], naive)
