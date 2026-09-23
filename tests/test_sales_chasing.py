"""Tests for the sales-chasing detector (openavmkit.sales_chasing).

The synthetic frames here mirror what the modeling module's ``sales_chase`` flag does
(prediction = sale_price * jitter; see ``_run_garbage`` / ``run_average`` in modeling.py),
plus richer variants the flag can't produce: heterogeneous (normal) chasing and partial
chasing layered on an honest local-area model. A genuine market-scatter term is baked into
the sale prices so an honest baseline has realistic, non-zero COD.
"""

import numpy as np
import pandas as pd

from openavmkit.sales_chasing import detect_sales_chasing


def _build(rng, n_clusters=60, per=12, market_scatter=0.12, within_cluster=0.05):
    """Return (sale, he_id, honest_area_prediction) for a clustered synthetic universe.

    Clusters group *similar* properties, so the honest local-area model (cluster-median
    sale price) has low within-cluster dispersion (CHD) -- the realistic baseline against
    which sales chasing stands out.
    """
    he, base = [], []
    for c in range(n_clusters):
        cv = rng.uniform(1.5e5, 7e5)
        for _ in range(per):
            he.append(c)
            base.append(cv)
    he = np.array(he)
    base = np.array(base, dtype=float)
    n = len(he)
    true_val = base * rng.normal(1.0, within_cluster, n)
    sale = true_val * np.exp(rng.normal(0.0, market_scatter, n))
    area_pred = (
        pd.DataFrame({"sale": sale, "he_id": he})
        .groupby("he_id")["sale"]
        .transform("median")
        .to_numpy()
    )
    return sale, he, area_pred


def _frame(sale, he, suspect, reference, sale_age_days=None):
    data = {
        "sale_price": sale,
        "he_id": he,
        "assr_market_value": suspect,
        "prediction": reference,
    }
    if sale_age_days is not None:
        data["sale_age_days"] = sale_age_days
    return pd.DataFrame(data)


def test_honest_model_not_flagged():
    # An honest local-area model compared against itself must not trip any signal.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    res = detect_sales_chasing(
        _frame(sale, he, area, area), "assr_market_value", reference_field="prediction"
    )
    assert not res.flagged
    assert res.verdict == "no signal"


def test_tight_uniform_chase_flagged_likely():
    # prediction = sale_price * U(0.95, 1.05): tight uniform chase -> spike + COD-CHD.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    suspect = sale * rng.uniform(0.95, 1.05, len(sale))
    res = detect_sales_chasing(
        _frame(sale, he, suspect, area), "assr_market_value", reference_field="prediction"
    )
    assert res.flagged
    assert res.verdict == "likely"


def test_heterogeneous_normal_chase_flagged():
    # prediction = sale_price * N(1, 0.03): some tightly chased, some loose -> still flags.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    suspect = sale * rng.normal(1.0, 0.03, len(sale))
    res = detect_sales_chasing(
        _frame(sale, he, suspect, area), "assr_market_value", reference_field="prediction"
    )
    assert res.flagged


def test_partial_chase_half_flagged():
    # Honest area model with 50% of parcels snapped to their sale price.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    suspect = area.copy()
    idx = rng.choice(len(sale), len(sale) // 2, replace=False)
    suspect[idx] = sale[idx]
    res = detect_sales_chasing(
        _frame(sale, he, suspect, area), "assr_market_value", reference_field="prediction"
    )
    assert res.flagged


def test_cod_chd_divergence_is_the_robust_signal():
    # Moderate chase (U +/- 0.10) the spike misses, but COD-CHD divergence should catch.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    suspect = sale * rng.uniform(0.90, 1.10, len(sale))
    res = detect_sales_chasing(
        _frame(sale, he, suspect, area), "assr_market_value", reference_field="prediction"
    )
    div = next(s for s in res.signals if s.name.startswith("COD-CHD"))
    assert div.flagged


def test_in_out_of_sample_gap_fires_for_date_restricted_chase():
    # A real assessor can only chase sales known at roll close. Chase pre-valuation sales
    # tightly, leave post-valuation honest -> COD jumps out-of-sample.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    age = rng.integers(-200, 400, len(sale))  # negative == post-valuation (unseen)
    pre = age >= 0
    suspect = area.copy()
    suspect[pre] = sale[pre] * rng.uniform(0.99, 1.01, pre.sum())
    res = detect_sales_chasing(
        _frame(sale, he, suspect, area, sale_age_days=age),
        "assr_market_value",
        reference_field="prediction",
    )
    gap = next(s for s in res.signals if s.name.startswith("In/out"))
    assert gap.flagged


def test_degrades_without_cluster_or_reference():
    # No cluster column -> COD-CHD signal is skipped; detector still runs the others.
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    suspect = sale * rng.uniform(0.95, 1.05, len(sale))
    df = _frame(sale, he, suspect, area).drop(columns=["he_id"])
    res = detect_sales_chasing(df, "assr_market_value", reference_field="prediction")
    assert not any(s.name.startswith("COD-CHD") for s in res.signals)
    # Spike signal still present and fires for a tight chase.
    spike = next(s for s in res.signals if s.name.startswith("Ratio spike"))
    assert spike.flagged


def test_to_markdown_renders():
    rng = np.random.default_rng(7)
    sale, he, area = _build(rng)
    suspect = sale * rng.uniform(0.95, 1.05, len(sale))
    res = detect_sales_chasing(
        _frame(sale, he, suspect, area), "assr_market_value", reference_field="prediction"
    )
    md = res.to_markdown()
    assert "| Signal |" in md
    assert "sales-chasing signals fired" in md.lower()


def test_ratio_study_template_resolves_all_vars_including_sales_chasing():
    # Guards the template<->code wiring: every variable the ratio_study template uses
    # (including the new {{sales_chasing}}) must be settable, with none left unresolved.
    from openavmkit.reports import MarkdownReport

    r = MarkdownReport("ratio_study")
    for k in [
        "locality",
        "val_date",
        "model_group",
        "sales_back_to_date",
        "overall_results",
        "locality_results",
        "modeler_results",
        "sales_chasing",
    ]:
        r.set_var(k, "X")
    out = r.render()
    assert "## Sales-chasing check" in out
    assert "{{" not in out  # no unresolved template variables
