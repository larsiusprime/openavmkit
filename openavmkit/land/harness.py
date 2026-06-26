"""Held-out evaluation harness for the land AVM.

The LVI battery is the objective function; this module is the closed loop around it. It splits the
land-observation pool into train / test folds **by parcel**, lets a candidate land model calibrate on
the train fold, then scores the painted universe against the *held-out* fold — so a model is graded
on evidence it never saw (the L8-holdout discipline). ``compare`` prints the same head-to-head
scorecard we use for jurisdictions, with one column per candidate series + the assessor baseline.

A "land model" is any callable ``paint(u, train_obs, cfg) -> universe`` that returns a copy of the
universe with the three series columns (``cfg.fields.land_value`` / ``impr_value`` / ``total_value``)
filled in. Baselines live in ``openavmkit.land.baselines``; the ported painter will conform to the
same signature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from openavmkit.lvi import battery


def split_observations(obs, test_frac=0.25, seed=0):
    """Split the land-observation pool into (train, test) by PARCEL key (no parcel in both).

    Parcels with direct evidence are split independently from cost-residual-only parcels so both
    folds keep gold direct anchors in proportion."""
    rng = np.random.default_rng(seed)
    keys = obs["key"].unique()
    has_direct = set(obs.loc[obs["kind"] == "direct", "key"])
    direct_keys = np.array([k for k in keys if k in has_direct])
    other_keys = np.array([k for k in keys if k not in has_direct])
    test = set()
    for arr in (direct_keys, other_keys):
        if len(arr):
            pick = rng.choice(arr, size=int(round(len(arr) * test_frac)), replace=False)
            test.update(pick.tolist())
    is_test = obs["key"].isin(test)
    return obs[~is_test].copy(), obs[is_test].copy()


def score_series(name, u_painted, sales, test_obs, cfg):
    """Run the full LVI battery on one painted series, scoring A3/A5/VE on the HELD-OUT obs."""
    return battery.run_battery(name, u_painted, sales, test_obs, cfg)


# headline metrics for the head-to-head table: (label, test, key, fmt)
_ROWS = [
    ("Total median ratio", "total", "median_ratio", "{:.3f}"),
    ("Total COD",          "total", "cod",          "{:.1f}"),
    ("A1 impr partial-R^2","A1",    "partial_r2",   "{:.3f}"),
    ("A2 land CHD",        "A2",    "land_chd",     "{:.1f}"),
    ("A3 direct held-out n","A3",   "direct_n",     "{:.0f}"),
    ("A3 direct median",   "A3",    "direct_median","{:.3f}"),
    ("A3 direct COD",      "A3",    "direct_cod",   "{:.1f}"),
    ("A3 gold median",     "A3",    "median_ratio", "{:.3f}"),
    ("A3 gold COD",        "A3",    "cod",          "{:.1f}"),
    ("A5 desirability rho","A5",    "rho_market",   "{:.3f}"),
    ("A7 sanity viol %",   "A7",    "pct_violations","{:.2f}"),
    ("B3 hot-pixel %",     "B3",    "hot_pixel_pct","{:.1f}"),
    ("VE direct VEI",      "VE",    "vei",          "{:.1f}"),
]


def compare(runs, title="LAND AVM head-to-head (held-out)"):
    """runs: list of (name, results_dict). Returns a printable scorecard string."""
    def cell(results, test, key, fmt):
        v = results.get(test, {}).get(key, np.nan)
        try:
            return fmt.format(v) if (v is not None and np.isfinite(v)) else "n/a"
        except (TypeError, ValueError):
            return "n/a"

    names = [n for n, _ in runs]
    w = max(14, max((len(n) for n in names), default=14) + 2)
    lines = ["", f"################  {title}  ################"]
    hdr = f"{'metric':<22}" + "".join(n.rjust(w) for n in names)
    lines += [hdr, "-" * len(hdr)]
    for label, test, key, fmt in _ROWS:
        lines.append(f"{label:<22}" + "".join(cell(r, test, key, fmt).rjust(w) for _, r in runs))
    return "\n".join(lines)
