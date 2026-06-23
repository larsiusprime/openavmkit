"""A0 — land unit-of-comparison selection (pro forma for single-family).

IAAO rule: the best unit is the one whose RATE varies least across comparable land sales (lowest
COV/COD), measured WITHIN a homogeneous area so location doesn't swamp it. Candidates we can
compute for SF: $/sqft and $/lot (site value). ($/acre ≡ $/sqft rescaled — identical COV;
$/front-foot needs frontage, absent in Wake; $/buildable-unit ≈ $/lot for SF, density ~1.)

Decisive diagnostic: within-neighborhood size elasticity  beta = dlog(land)/dlog(area):
  beta ~ 1  -> price scales with area -> $/sqft constant  -> use $/sqft
  beta ~ 0  -> price flat vs area     -> $/lot constant    -> use $/lot (site value)
Plus median within-neighborhood COD of each rate (lower wins).

Run from repo root:  python research/lvi_a0_unit.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"


def within_nb_elasticity(d):
    """FWL within-neighborhood slope of log(land) on log(area)."""
    dd = d[(d["observed_land"] > 0) & (d["land_sqft"] > 0)].copy()
    dd["ly"] = np.log(dd["observed_land"]); dd["lx"] = np.log(dd["land_sqft"])
    yw = dd["ly"] - dd.groupby("nbhd")["ly"].transform("mean")
    xw = dd["lx"] - dd.groupby("nbhd")["lx"].transform("mean")
    denom = float(np.sum(xw * xw))
    return float(np.sum(xw * yw) / denom) if denom > 0 else np.nan


def within_nb_cod(d, rate_col, k=4):
    cods = []
    for _, g in d.groupby("nbhd"):
        gg = g[g[rate_col] > 0]
        if len(gg) >= k:
            cods.append(calc_cod(gg[rate_col].values))
    return (float(np.median(cods)) if cods else np.nan), len(cods)


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    obs = A.build_land_observations(s, ui, NB)
    prime = A.prime_lot_mask(u, NB, MG, residential_zoning=True)
    obs["prime_lot"] = obs["key"].map(prime).fillna(False)
    q = obs["qualified"].fillna(False)

    vac = obs[(obs["kind"] == "vacant") & obs["prime_lot"] & q]
    rcn = obs[(obs["kind"] == "rcn_resid") & q]
    pooled = pd.concat([vac, rcn]).drop_duplicates("key")

    print("=== A0 unit-of-comparison: $/sqft vs $/lot (site value), single-family ===")
    print(f"{'evidence':<26}{'n':>6}{'beta(size elast.)':>18}{'COD $/sqft':>12}{'COD $/lot':>11}{'winner':>9}")
    for name, d in [("vacant prime+qual", vac), ("rcn_resid (frozen RCN)", rcn), ("pooled", pooled)]:
        d = d[(d["observed_land"] > 0) & (d["land_sqft"] > 0)].copy()
        d["psf"] = d["observed_land"] / d["land_sqft"]
        d["perlot"] = d["observed_land"]
        beta = within_nb_elasticity(d)
        cod_psf, n_psf = within_nb_cod(d, "psf")
        cod_lot, n_lot = within_nb_cod(d, "perlot")
        win = "$/sqft" if (np.isfinite(cod_psf) and np.isfinite(cod_lot) and cod_psf < cod_lot) else \
              ("$/lot" if np.isfinite(cod_lot) else "n/a")
        bs = f"{beta:.2f}" if np.isfinite(beta) else "n/a"
        cps = f"{cod_psf:.1f}({n_psf})" if np.isfinite(cod_psf) else "n/a"
        cls = f"{cod_lot:.1f}({n_lot})" if np.isfinite(cod_lot) else "n/a"
        print(f"{name:<26}{len(d):>6}{bs:>18}{cps:>12}{cls:>11}{win:>9}")

    print("\n  beta near 1 => price scales with lot size => $/sqft is the unit")
    print("  beta near 0 => price flat vs lot size  => $/lot (site value) is the unit")
    print("  (median within-neighborhood COD shown with #neighborhoods having >=4 land obs)")


if __name__ == "__main__":
    main()
