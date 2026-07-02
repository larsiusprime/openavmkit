"""Tests for openavmkit.vertical_equity_study.get_vertical_equity_scores.

Regression tests for a crash when the VEI "market proxy" is degenerate: with
>= 20 observations (so the small-sample guard doesn't apply) but a market proxy
that is all-NaN or has fewer distinct values than requested percentile groups,
``pd.qcut`` raised ``ValueError: Bin edges must be unique``, aborting the whole
modeling run. The function should instead degrade gracefully to a NaN result,
exactly like it already does for < 20 observations.
"""
import numpy as np
import pandas as pd

from openavmkit.vertical_equity_study import get_vertical_equity_scores


def test_all_zero_valuation_returns_nan_not_crash():
	# Reproduces the real trigger: a tax-exempt model group whose assessed
	# (valuation) values are all $0. Then median_ratio == 0, so
	# market_proxy = sale*0.5 + valuation/0 = NaN for every row, and qcut on an
	# all-NaN column raises. With >= 20 rows the small-sample guard doesn't fire.
	rng = np.random.default_rng(0)
	df = pd.DataFrame({
		"sale": rng.uniform(50_000, 500_000, size=40),
		"valuation": np.zeros(40),
	})

	result = get_vertical_equity_scores(df, "sale", "valuation")

	assert np.isnan(result["vei"])
	assert np.isnan(result["vei_significance"])


def test_constant_market_proxy_returns_nan_not_crash():
	# Fewer distinct market-proxy values than requested percentile groups also
	# makes qcut raise "Bin edges must be unique". Constant sale + valuation is
	# the extreme case (one distinct value).
	df = pd.DataFrame({
		"sale": np.full(30, 100_000.0),
		"valuation": np.full(30, 90_000.0),
	})

	result = get_vertical_equity_scores(df, "sale", "valuation")

	assert np.isnan(result["vei"])


def test_concentrated_values_do_not_crash():
	# Even with many distinct values, a heavy concentration at one value makes
	# several quantile boundaries land on the same edge, so plain qcut(q=N) still
	# raises "Bin edges must be unique". This is common in assessment data (capped
	# or round-number values). Should degrade gracefully, not crash.
	sale = np.concatenate([np.full(70, 100_000.0),
						   np.linspace(120_000, 900_000, 30)])
	df = pd.DataFrame({"sale": sale, "valuation": sale * 0.9})

	result = get_vertical_equity_scores(df, "sale", "valuation")

	# Either a finite VEI (from the tiers that could be formed) or NaN, but no raise.
	assert np.isnan(result["vei"]) or np.isfinite(result["vei"])


def test_healthy_data_still_returns_finite_vei():
	# Guard must not disturb the happy path: well-spread values with a real
	# regressive tilt should yield a finite VEI and per-group stats.
	rng = np.random.default_rng(42)
	true_value = rng.uniform(50_000, 500_000, size=200)
	sale = true_value * rng.uniform(0.95, 1.05, size=200)
	# regressive tilt: low-value assessed high, high-value assessed low
	tilt = (true_value - true_value.min()) / (true_value.max() - true_value.min())
	valuation = true_value * (1.15 - 0.3 * tilt)
	df = pd.DataFrame({"sale": sale, "valuation": valuation})

	result = get_vertical_equity_scores(df, "sale", "valuation")

	assert np.isfinite(result["vei"])
	assert result["group_stats"] is not None
