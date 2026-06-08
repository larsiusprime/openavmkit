"""Sales-chasing detection.

*Sales chasing* is the practice -- sometimes deliberate, sometimes an unintended side
effect of a valuation methodology -- of moving a parcel's appraised value toward its
observed sale price. It makes a valuation look very strong on *sold* parcels (ratios
collapse onto 1.0 and COD drops) while not improving (and sometimes worsening) uniformity
among comparable *unsold* parcels. Because a ratio study can only be scored against parcels
that actually sold, a roll affected by sales chasing can post numbers that look better than
the values would out-of-sample. openavmkit's own models are evaluated out-of-sample, so a
naive head-to-head can understate openavmkit relative to such a roll.

This is an information-gap problem, not an accusation: the point is simply that scoring a
roll on the same sales that may have informed it is not the same test our held-out models
face. This module turns that asymmetry into measurable signals so a very tight assessor
ratio study can be interpreted with the right context rather than taken at face value.
Three signals are computed; each degrades gracefully when its inputs are unavailable:

1. **Ratio spike at 1.0** -- the share of sold parcels whose ``value / sale_price`` lands
   within ``eps`` of 1.0. A large mass of ratios sitting exactly on 1.0 is a strong
   indicator that values were taken directly from sale prices.
2. **COD-CHD divergence** -- mirrors the model utility score's ``sales_chase_score``
   (``(1 / COD) * CHD``; see :func:`openavmkit.modeling.compute_utility_score`): a
   suspiciously low ratio-COD on *sold* parcels paired with high within-cluster dispersion
   (CHD) of the values themselves. A genuinely accurate roll has both low COD and low CHD;
   a chased roll buys low COD on sold parcels without the matching uniformity. Requires a
   horizontal-equity cluster column.
3. **In/out-of-sample COD gap** -- COD on pre-valuation sold parcels vs. post-valuation
   sold parcels. A large jump once we include sales the assessor could not have seen at
   roll-close measures the in-sample advantage directly. Requires ``sale_age_days`` and an
   aligned ``valuation_date`` (see the ratio study docs).
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd
import polars as pl

from openavmkit.utilities.data import div_series_z_safe
from openavmkit.utilities.stats import calc_cod, quick_median_chd_pl

# Horizontal-equity cluster columns to try, in order, when none is supplied explicitly.
_CLUSTER_FIELD_CANDIDATES = ("he_id", "impr_he_id", "land_he_id")


@dataclass
class SalesChasingSignal:
    """One sales-chasing signal and its verdict.

    Attributes
    ----------
    name : str
        Human-readable signal name.
    value : float or None
        The signal value for the field under suspicion (None if it could not be computed).
    reference : float or None
        The same signal computed for the reference field (e.g. our own model), for context.
        None when no reference field was supplied or it could not be computed.
    flagged : bool
        Whether this signal indicates sales chasing.
    detail : str
        Short explanation of the verdict.
    """

    name: str
    value: float | None
    reference: float | None
    flagged: bool
    detail: str


@dataclass
class SalesChasingResult:
    """Combined sales-chasing verdict for one valuation field."""

    field: str
    signals: list[SalesChasingSignal] = dataclass_field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return sum(1 for s in self.signals if s.flagged)

    @property
    def flagged(self) -> bool:
        return self.n_flagged > 0

    @property
    def verdict(self) -> str:
        """``"likely"`` (>=2 signals), ``"possible"`` (1 signal), or ``"no signal"``."""
        n = self.n_flagged
        if n >= 2:
            return "likely"
        if n == 1:
            return "possible"
        return "no signal"

    def to_markdown(self) -> str:
        """Render the result as a small Markdown table plus a verdict line."""
        header = "| Signal | Value | Reference | Flag |\n| --- | --- | --- | --- |"
        rows = []
        for s in self.signals:
            val = "n/a" if s.value is None or not np.isfinite(s.value) else f"{s.value:.2f}"
            ref = (
                "n/a"
                if s.reference is None or not np.isfinite(s.reference)
                else f"{s.reference:.2f}"
            )
            flag = "**yes**" if s.flagged else "no"
            rows.append(f"| {s.name} | {val} | {ref} | {flag} |")
        table = header + "\n" + "\n".join(rows)
        verdict = {
            "likely": (
                "**Multiple sales-chasing signals fired.** Interpret the assessor's results "
                "on sold parcels with this context; it is a prompt to look closer, not a verdict."
            ),
            "possible": (
                "**One sales-chasing signal fired.** Worth a closer look, but not conclusive."
            ),
            "no signal": "No sales-chasing signals detected.",
        }[self.verdict]
        details = "\n".join(f"- *{s.name}*: {s.detail}" for s in self.signals)
        return f"{verdict}\n\n{table}\n\n{details}\n"


def _ratios(df: pd.DataFrame, value_field: str, sale_price_field: str) -> np.ndarray:
    """Finite, positive value/sale_price ratios for the given field."""
    ratios = div_series_z_safe(df[value_field], df[sale_price_field])
    ratios = pd.to_numeric(pd.Series(ratios), errors="coerce").to_numpy(dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    return ratios[ratios > 0]


def _cod(df: pd.DataFrame, value_field: str, sale_price_field: str) -> float:
    ratios = _ratios(df, value_field, sale_price_field)
    return calc_cod(ratios) if len(ratios) else float("nan")


def _spike_share(
    df: pd.DataFrame, value_field: str, sale_price_field: str, eps: float
) -> float:
    """Share of sold parcels whose ratio is within ``eps`` of 1.0."""
    ratios = _ratios(df, value_field, sale_price_field)
    if not len(ratios):
        return float("nan")
    return float(np.mean(np.abs(ratios - 1.0) <= eps))


def _median_chd(df: pd.DataFrame, value_field: str, cluster_field: str) -> float:
    """Median within-cluster COD (CHD) of the raw values, via the canonical helper."""
    sub = df[[value_field, cluster_field]].dropna()
    if sub.empty:
        return float("nan")
    return quick_median_chd_pl(pl.from_pandas(sub), value_field, cluster_field)


def _resolve_cluster_field(df: pd.DataFrame, cluster_field: str | None) -> str | None:
    if cluster_field is not None:
        return cluster_field if cluster_field in df.columns else None
    for candidate in _CLUSTER_FIELD_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def detect_sales_chasing(
    df: pd.DataFrame,
    suspect_field: str,
    sale_price_field: str = "sale_price",
    reference_field: str | None = None,
    cluster_field: str | None = None,
    sale_age_field: str = "sale_age_days",
    spike_eps: float = 0.02,
    spike_min_share: float = 0.10,
    spike_ratio_vs_ref: float = 1.5,
    cod_ratio_max: float = 0.7,
    chd_ratio_min: float = 0.9,
    oos_cod_jump: float = 1.5,
) -> SalesChasingResult:
    """Run the sales-chasing signals on ``suspect_field``.

    Parameters
    ----------
    df : pandas.DataFrame
        Sold parcels. Must contain ``suspect_field`` and ``sale_price_field``. May contain a
        horizontal-equity cluster column and ``sale_age_field`` for the optional signals.
    suspect_field : str
        Valuation column under examination (e.g. ``"assr_market_value"``).
    sale_price_field : str, optional
        Sale-price column used as ground truth. Defaults to ``"sale_price"``.
    reference_field : str, optional
        A second valuation column (e.g. our own ``"prediction"``) used as a sanity baseline
        for the relative thresholds. If omitted, the relative comparisons are skipped and the
        spike signal falls back to its absolute threshold only.
    cluster_field : str, optional
        Horizontal-equity cluster column. If omitted, the first of ``he_id``/``impr_he_id``/
        ``land_he_id`` present is used; if none are present, the COD-CHD signal is skipped.
    sale_age_field : str, optional
        Column of days between sale and valuation date (positive = before valuation). Used by
        the in/out-of-sample signal; skipped if absent.
    spike_eps, spike_min_share, spike_ratio_vs_ref
        Spike-at-1.0 thresholds: bucket half-width, minimum suspect share, and the factor by
        which the suspect must exceed the reference share (see module docstring).
    cod_ratio_max, chd_ratio_min
        COD-CHD divergence thresholds. Sales chasing is flagged when the suspect's ratio-COD on
        sold parcels is at most ``cod_ratio_max`` times the reference's (suspiciously *better*
        on sold parcels) while its within-cluster dispersion (CHD) is still at least
        ``chd_ratio_min`` times the reference's (no matching gain in uniformity). Requires a
        reference field.
    oos_cod_jump : float
        In/out-of-sample threshold: flag when post-valuation COD is at least this multiple of
        pre-valuation COD.

    Returns
    -------
    SalesChasingResult
        The per-signal verdicts and a combined verdict.
    """
    result = SalesChasingResult(field=suspect_field)

    cluster = _resolve_cluster_field(df, cluster_field)
    have_ref = reference_field is not None and reference_field in df.columns

    # --- Signal 1: ratio spike at 1.0 ---------------------------------------------------
    spike = _spike_share(df, suspect_field, sale_price_field, spike_eps)
    ref_spike = (
        _spike_share(df, reference_field, sale_price_field, spike_eps) if have_ref else None
    )
    spike_flag = bool(np.isfinite(spike) and spike >= spike_min_share)
    if spike_flag and ref_spike is not None and np.isfinite(ref_spike) and ref_spike > 0:
        # If we have a baseline, require the suspect to spike notably more than it.
        spike_flag = spike >= ref_spike * spike_ratio_vs_ref
    result.signals.append(
        SalesChasingSignal(
            name=f"Ratio spike at 1.0 (±{spike_eps:g})",
            value=None if not np.isfinite(spike) else spike * 100.0,
            reference=None if ref_spike is None or not np.isfinite(ref_spike) else ref_spike * 100.0,
            flagged=spike_flag,
            detail=(
                f"{spike * 100:.1f}% of sold ratios sit within ±{spike_eps:g} of 1.0"
                if np.isfinite(spike)
                else "could not compute ratios"
            ),
        )
    )

    # --- Signal 2: COD-CHD divergence ---------------------------------------------------
    # Sales chasing buys a low ratio-COD on *sold* parcels without the matching uniformity
    # (CHD) among *similar* parcels. An honestly-better valuation lowers both COD and CHD;
    # a chaser lowers COD while CHD stays put (or worsens). We compare against the reference
    # rather than the absolute (1/COD)*CHD score (the model utility scorer's `sales_chase_score`,
    # modeling.py:1712), which avoids dividing by a near-zero reference. Needs a reference.
    if cluster is not None and have_ref:
        cod_s = _cod(df, suspect_field, sale_price_field)
        chd_s = _median_chd(df, suspect_field, cluster)
        cod_r = _cod(df, reference_field, sale_price_field)
        chd_r = _median_chd(df, reference_field, cluster)
        computable = all(np.isfinite(x) for x in (cod_s, chd_s, cod_r, chd_r)) and cod_r > 0
        # Suspiciously better COD on sold parcels...
        better_cod = computable and cod_s <= cod_r * cod_ratio_max
        # ...with no matching improvement in within-cluster uniformity.
        not_more_uniform = computable and chd_s >= chd_r * chd_ratio_min
        div_flag = bool(better_cod and not_more_uniform)
        # Display the (1/COD)*CHD score for suspect and reference for context.
        score = (1.0 / cod_s) * chd_s if computable else float("nan")
        ref_score = (1.0 / cod_r) * chd_r if computable else float("nan")
        result.signals.append(
            SalesChasingSignal(
                name="COD-CHD divergence (1/COD × CHD)",
                value=None if not np.isfinite(score) else score,
                reference=None if not np.isfinite(ref_score) else ref_score,
                flagged=div_flag,
                detail=(
                    f"sold-COD {cod_s:.1f} vs ours {cod_r:.1f} (much tighter), but CHD "
                    f"{chd_s:.1f} vs ours {chd_r:.1f} (no better) — clustered on '{cluster}'"
                    if div_flag
                    else f"sold-COD {cod_s:.1f} vs ours {cod_r:.1f}, CHD {chd_s:.1f} vs "
                    f"ours {chd_r:.1f} (clustered on '{cluster}')"
                    if computable
                    else "could not compute COD/CHD"
                ),
            )
        )

    # --- Signal 3: in/out-of-sample COD gap ---------------------------------------------
    if sale_age_field in df.columns:
        seen = df[df[sale_age_field] >= 0]  # at/before valuation: assessor could have seen
        unseen = df[df[sale_age_field] < 0]  # after valuation: out-of-sample for the roll
        cod_seen = _cod(seen, suspect_field, sale_price_field)
        cod_unseen = _cod(unseen, suspect_field, sale_price_field)
        gap_flag = bool(
            np.isfinite(cod_seen)
            and cod_seen > 0
            and np.isfinite(cod_unseen)
            and cod_unseen >= cod_seen * oos_cod_jump
        )
        result.signals.append(
            SalesChasingSignal(
                name="In/out-of-sample COD gap",
                value=None if not np.isfinite(cod_unseen) else cod_unseen,
                reference=None if not np.isfinite(cod_seen) else cod_seen,
                flagged=gap_flag,
                detail=(
                    f"COD jumps from {cod_seen:.1f} (pre-valuation sold) to "
                    f"{cod_unseen:.1f} (post-valuation sold)"
                    if np.isfinite(cod_seen) and np.isfinite(cod_unseen)
                    else "not enough pre/post-valuation sold parcels"
                ),
            )
        )

    return result
