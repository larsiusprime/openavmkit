"""Land Value Integrity (LVI) — a config-driven battery that scores any existing per-parcel land
series ("are these land values good?") and produces a comparative scorecard plus per-parcel
evidence packets.

Separate by design from ``openavmkit.land`` (the land *creation* / painter system); LVI only
*validates*. Configuration is keyed by model group in ``settings.json`` under
``land_value_integrity`` (see ``openavmkit.lvi.config``); unconfigured groups are skipped. LVI does
no classification of its own — *which* sales are evidence is decided by filters the user owns.

Typical use (per model group)::

    from openavmkit.lvi import run_land_value_integrity
    from openavmkit.lvi.config import load_lvi_configs
    cfg = load_lvi_configs(settings)["single_family"]
    res = run_land_value_integrity({"assessor": assessor_universe}, sales, cfg)
    print(res.scorecard()); res.packets["assessor"].to_csv("evidence.csv", index=False)

The driver ``openavmkit.lvi.run`` loops every configured group for a jurisdiction.

Methodology: ``research/land_value_integrity_spec.md`` (§10-21) and ``..._RESULTS.md``.
"""
from __future__ import annotations

from types import SimpleNamespace

from openavmkit.lvi import evidence, battery, support as _support, report
from openavmkit.lvi.config import GroupConfig, load_lvi_configs
from openavmkit.lvi.evidence import build_land_observations, reconstruct_rcn, summarize
from openavmkit.lvi.report import scorecard as _scorecard, evidence_packet

__all__ = [
    "run_land_value_integrity", "load_lvi_configs", "GroupConfig",
    "evidence", "battery", "support", "report", "build_land_observations", "reconstruct_rcn",
]


class LVIResult(SimpleNamespace):
    """One model group's run: results, flags, diagnostics, support, obs, packets."""
    def scorecard(self, title=None):
        t = title or f"LAND VALUE INTEGRITY — {self.model_group}"
        return _scorecard(list(self.results.items()), title=t)


def run_land_value_integrity(series_universes, sales, cfg, run_support=True):
    """Run the LVI battery for ONE model group.

    Parameters
    ----------
    series_universes : dict[str, DataFrame]
        Name -> universe carrying ``land_value``/``impr_value``/``total_value`` (the series under
        test) plus the feature columns named in ``cfg.fields``. Already filtered to this group.
        The first entry is "primary" (used for the shared evidence pool, diagnostics, propagation).
    sales : DataFrame
        Hydrated valid sales for this model group.
    cfg : GroupConfig
        From ``load_lvi_configs(settings)[model_group]``.
    """
    primary = next(iter(series_universes))
    u0 = series_universes[primary]

    obs = build_land_observations(sales, u0, cfg)

    results, flags = {}, {}
    for name, u in series_universes.items():
        results[name], flags[name] = battery.run_battery(name, u, sales, obs, cfg)

    diagnostics = {"A0": battery.diag_a0_unit(obs, cfg),
                   "depreciation": battery.diag_depreciation(sales, u0, cfg)}

    support, packets, prop = {}, {}, None
    if run_support:
        prop = _support.propagate_land_levels(u0, sales, obs, cfg)
        cov, _ = _support.coverage_map(u0, obs, cfg)
        _diffs, diff_sum = _support.location_differentials(u0, obs, cfg)
        support = dict(propagation=prop, coverage=cov, differentials=diff_sum)
    for name, u in series_universes.items():
        packets[name] = report.evidence_packet(u, obs, flags[name], cfg, prop=prop)

    return LVIResult(model_group=cfg.model_group, results=results, flags=flags,
                     diagnostics=diagnostics, support=support, obs=obs, packets=packets)
