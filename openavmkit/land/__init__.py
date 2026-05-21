"""
Land valuation: witness curation, painters, and tests.

This package consolidates the land-side of the AVM:

* :mod:`~openavmkit.land.evidence` — witness curation. Six W-streams
  (W1 vacant, W2 teardown, W3 extraction, W4 low-FAR, W5 prior-xfer,
  W6 pred-residual) plus per-stream filters and cross-stream anomaly
  flagging. Output is the witness pool consumed by the painters.
* :mod:`~openavmkit.land.tables` — production-grade per-cell painter.
  Per-cell tables (``base_lot`` / ``size_curve`` / ``puv``) with
  zoning-anchored size-curve breakpoints fit empirically from the
  witness pool.
* :mod:`~openavmkit.land.lycd` — simpler "Least You Can Do" uniform-rate
  painter. One ``$/sqft`` per cell from a global or local allocation %.
  Used at Rung 1.0/1.1.
* :mod:`~openavmkit.land.tests` — Lars-Tests harness (L1 improvement-
  neutrality, L2 within-cluster uniformity, L3 vacant-burden flip,
  L4 desirability Spearman, L5 density-FAR ordering, L6 per-cell size
  decay, L7 improvement-cost-table COD).

The package re-exports the principal public symbols so callers can write
``from openavmkit.land import curate_witnesses, build_all_tables, ...``
without depending on the internal submodule layout.

Upstream prerequisites:

* :mod:`openavmkit.neighborhoods` — VCS / cascade hierarchy
* :mod:`openavmkit.zoning` — empirical (jurisdiction, zoning) reference table
"""
from __future__ import annotations

from openavmkit.land.evidence import (
    WitnessConfig,
    curate_witnesses,
    curate_w1_clean_vacant,
    curate_w2_teardown,
    curate_w3_extraction,
    curate_w4_low_far,
    curate_w5_prior_xfer,
    curate_w6_pred_residual,
    evaluate_sale_anomaly_flags,
)
from openavmkit.land.tables import (
    Tier,
    LandTable,
    AdjustmentSpec,
    DEFAULT_TIER_DECAYS,
    DEFAULT_TIER_PERCENTILES,
    DEFAULT_ZONING_BP1_MULT,
    DEFAULT_ZONING_BP2_MULT,
    BASE_LOT_CV_THRESHOLD,
    apply_adjustments,
    fit_pooled_zoning_breakpoints,
    build_neighborhood_table,
    build_all_tables,
    paint_from_tables,
    build_explanations_for_subset,
    write_land_tables_artifact,
    write_evidence_packets,
)
from openavmkit.land.lycd import (
    LycdConfig,
    derive_allocation_pct,
    paint_lycd,
)
from openavmkit.land.tests import (
    LarsTestsResult,
    run_lars_tests,
    run_holdout_vacant_test,
    write_lars_tests_report,
)

__all__ = [
    # evidence
    "WitnessConfig",
    "curate_witnesses",
    "curate_w1_clean_vacant",
    "curate_w2_teardown",
    "curate_w3_extraction",
    "curate_w4_low_far",
    "curate_w5_prior_xfer",
    "curate_w6_pred_residual",
    "evaluate_sale_anomaly_flags",
    # tables (production painter)
    "Tier",
    "LandTable",
    "AdjustmentSpec",
    "DEFAULT_TIER_DECAYS",
    "DEFAULT_TIER_PERCENTILES",
    "DEFAULT_ZONING_BP1_MULT",
    "DEFAULT_ZONING_BP2_MULT",
    "BASE_LOT_CV_THRESHOLD",
    "apply_adjustments",
    "fit_pooled_zoning_breakpoints",
    "build_neighborhood_table",
    "build_all_tables",
    "paint_from_tables",
    "build_explanations_for_subset",
    "write_land_tables_artifact",
    "write_evidence_packets",
    # lycd (uniform painter)
    "LycdConfig",
    "derive_allocation_pct",
    "paint_lycd",
    # tests (Lars-Tests)
    "LarsTestsResult",
    "run_lars_tests",
    "run_holdout_vacant_test",
    "write_lars_tests_report",
]
