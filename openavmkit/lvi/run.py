"""Jurisdiction driver / CLI for the LVI battery.

Loads the cleaned SalesUniversePair and the `land_value_integrity` config block, then runs the
battery for EVERY configured model group (groups not in the config are skipped), assembling the
assessor land series (and an AVM series where a genuine land/improvement split is present).

    python -m openavmkit.lvi.run [DATA_DIR]      # default: notebooks/pipeline/data/us-nc-wake
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from openavmkit.lvi import run_land_value_integrity
from openavmkit.lvi.config import load_lvi_configs
from openavmkit.lvi.evidence import summarize

DEFAULT_DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")


def _load(data_dir):
    from openavmkit.pipeline import load_settings, read_pickle
    from openavmkit.data import get_hydrated_sales_from_sup
    cwd = os.getcwd(); os.chdir(data_dir)
    try:
        settings = load_settings("in/settings.json")
        sup = read_pickle("out/2-clean-sup")
    finally:
        os.chdir(cwd)
    u = sup.universe.copy(); u["key"] = u["key"].astype(str)
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["valid_sale"] == True) & (s["sale_price"] > 0)].copy(); s["key"] = s["key"].astype(str)
    return settings, u, s


def assessor_series(u, cfg):
    F = cfg.fields
    m = u.copy()
    m[F.land_value], m[F.impr_value], m[F.total_value] = m["assr_land_value"], m["assr_impr_value"], m["assr_market_value"]
    return m


def avm_series_if_present(u, cfg):
    """AVM series only if a GENUINE land/improvement split exists (not the total relabeled)."""
    F = cfg.fields
    if "prediction_land_sqft" not in u.columns or "prediction" not in u.columns:
        return None
    land = pd.to_numeric(u["prediction_land_sqft"], errors="coerce") * pd.to_numeric(u[F.land_area], errors="coerce")
    if np.isclose(land, pd.to_numeric(u["prediction"], errors="coerce"), rtol=1e-3).mean() >= 0.5:
        return None
    m = u.copy()
    m[F.land_value] = land
    m[F.impr_value] = pd.to_numeric(u["prediction_impr_sqft"], errors="coerce") * pd.to_numeric(u[F.bldg_area], errors="coerce")
    m[F.total_value] = pd.to_numeric(u["prediction"], errors="coerce")
    return m


def main(data_dir=DEFAULT_DATA):
    warnings.filterwarnings("ignore")
    settings, u, s = _load(data_dir)
    configs = load_lvi_configs(settings)
    if not configs:
        print("No `land_value_integrity` config block found — nothing to score.")
        return
    print(f"loaded: universe={u.shape}  sales={s.shape}  configured groups: {list(configs)}")

    out_dir = os.path.join(data_dir, "out", "lvi"); os.makedirs(out_dir, exist_ok=True)
    for mg, cfg in configs.items():
        F = cfg.fields
        ug = u[u[F.model_group] == mg].copy()
        sg = s[s[F.model_group] == mg].copy()
        if not len(ug):
            print(f"\n[{mg}] no parcels in universe — skipping."); continue
        series = {"assessor": assessor_series(ug, cfg)}
        avm = avm_series_if_present(ug, cfg)
        if avm is not None:
            series["avm"] = avm
        res = run_land_value_integrity(series, sg, cfg)
        print(f"\n##### {mg} #####")
        print(summarize(res.obs))
        print(res.scorecard())
        print("diagnostics: " + res.diagnostics["A0"]["detail"])
        print("             " + res.diagnostics["depreciation"]["detail"])
        with open(os.path.join(out_dir, f"scorecard_{mg}.txt"), "w") as f:
            f.write(res.scorecard() + "\n")
        for name, pkt in res.packets.items():
            pkt.to_csv(os.path.join(out_dir, f"evidence_{mg}_{name}.csv"), index=False)
    print(f"\nwrote per-group scorecards + evidence packets to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
