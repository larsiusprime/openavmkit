"""Land AVM — an evidence-driven land model that mimics hand valuation.

The *creation* counterpart to ``openavmkit.lvi`` (which only validates). It shares the
``lvi.evidence`` layer (the observed-land training data) and is graded by the ``lvi.battery``
(the objective function). Design: ``research/land_avm_spec.md``.

Phase 0 (current): the held-out harness + crude baselines, so the scoreboard exists before any
real modeling. ``run_land_avm`` paints a candidate land series, scores it out-of-fold against the
assessor, and returns both the painted universe and the LVI results.

    from openavmkit.land import run_land_avm
    res = run_land_avm(u, sales, cfg, model="neighborhood_rate")
    print(res.scorecard)
"""
from __future__ import annotations

from types import SimpleNamespace

from openavmkit.lvi.evidence import build_land_observations
from openavmkit.land import baselines, harness

__all__ = ["run_land_avm", "baselines", "harness", "build_land_observations"]


def run_land_avm(u, sales, cfg, model="neighborhood_rate", test_frac=0.25, seed=0,
                 include_assessor=True):
    """Paint a land series with ``model``, score it on a held-out fold, return results.

    Parameters mirror ``lvi.run_land_value_integrity``: ``u`` is the universe filtered to this model
    group (carrying ``assr_*`` + feature columns), ``sales`` the hydrated valid sales, ``cfg`` a
    ``lvi.config.GroupConfig``. Returns a SimpleNamespace with ``painted`` (universe), ``results``,
    ``runs``, and ``scorecard``."""
    obs = build_land_observations(sales, u, cfg)
    train_obs, test_obs = harness.split_observations(obs, test_frac=test_frac, seed=seed)

    paint = baselines.PAINTERS[model]
    painted = paint(u, train_obs, cfg)
    results, flags = harness.score_series(model, painted, sales, test_obs, cfg)

    runs = [(model, results)]
    if include_assessor:
        from openavmkit.lvi.run import assessor_series
        a_results, _ = harness.score_series("assessor", assessor_series(u, cfg), sales, test_obs, cfg)
        runs.append(("assessor", a_results))

    return SimpleNamespace(painted=painted, results=results, flags=flags, obs=obs,
                           train_obs=train_obs, test_obs=test_obs, runs=runs,
                           scorecard=harness.compare(runs))
