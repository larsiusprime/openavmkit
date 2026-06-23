"""Land vertical equity: is expensive land under-assessed more than cheap land (regressive)?

Runs VEI + decile charts (median assr_land/observed_land ratio by land-value group) on each
evidence stream, so we can tell whether any regressivity is REAL (shows up in the clean vacant
evidence, which subtracts no RCN) or a RESIDUAL ARTIFACT (only in price−RCN, where high-value
homes have large RCN and the residual is a noisy difference of big numbers).

Two value axes (spec §A3): GLOBAL land-value level, and NEIGHBORHOOD-RELATIVE level —
agreement = genuine within-neighborhood inequity; divergence = between-neighborhood artifact.

Run from repo root:  python research/lvi_land_vertical.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.vertical_equity_study import get_vertical_equity_scores
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"


def vei_and_chart(d, label):
    """d has columns observed_land (sale-side land), assr_land (assessment). Print VEI + deciles."""
    df = pd.DataFrame({"obs": pd.to_numeric(d["observed_land"], errors="coerce"),
                       "assr": pd.to_numeric(d["assr_land"], errors="coerce")}).dropna()
    df = df[(df["obs"] > 0) & (df["assr"] > 0)]
    res = get_vertical_equity_scores(df, sale_field="obs", valuation_field="assr")
    vei, sig, gs = res["vei"], res["vei_significance"], res["group_stats"]
    veis = f"{vei:.1f}" if vei is not None and np.isfinite(vei) else "n/a"
    sigs = f"{sig:.1f}" if sig is not None and np.isfinite(sig) else "n/a"
    chart = ""
    if gs is not None:
        rr = gs["ratio"].tolist()
        chart = "  deciles(low->high value): " + " ".join(f"{x:.2f}" for x in rr)
    print(f"  {label:<34} n={len(df):>5}  VEI={veis:>6} (sig {sigs})" + chart)
    return res


def nbhd_relative_chart(d, label, q=10):
    """Decile chart on NEIGHBORHOOD-RELATIVE land value (obs $/sqft vs nbhd median)."""
    df = pd.DataFrame({"obs": pd.to_numeric(d["observed_land"], errors="coerce"),
                       "assr": pd.to_numeric(d["assr_land"], errors="coerce"),
                       "la": pd.to_numeric(d["land_sqft"], errors="coerce"),
                       "nb": d["nbhd"].values}).dropna()
    df = df[(df["obs"] > 0) & (df["assr"] > 0) & (df["la"] > 0)]
    df["ratio"] = df["assr"] / df["obs"]
    df["obs_psf"] = df["obs"] / df["la"]
    df["rel"] = df["obs_psf"] / df.groupby("nb")["obs_psf"].transform("median")
    if df["rel"].nunique() < q or len(df) < 2 * q:
        print(f"  {label:<34} (too thin for {q} bins)"); return
    df["bin"] = pd.qcut(df["rel"], q=q, labels=False, duplicates="drop")
    ch = df.groupby("bin")["ratio"].median()
    rho = np.corrcoef(ch.index.values, ch.values)[0, 1]
    print(f"  {label:<34} rel-deciles(low->high): " + " ".join(f"{x:.2f}" for x in ch.tolist())
          + f"   trend rho={rho:+.2f}")


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    obs = A.build_land_observations(s, ui, NB)
    prime = A.prime_lot_mask(u, NB, MG, residential_zoning=True)
    obs["prime_lot"] = obs["key"].map(prime).fillna(False)
    obs["assr_land"] = pd.to_numeric(obs["key"].map(ui["assr_land_value"]), errors="coerce")
    q = obs["qualified"].fillna(False)

    sets = {
        "all-vacant": obs[obs["kind"] == "vacant"],
        "qualified vacant": obs[(obs["kind"] == "vacant") & q],
        "prime+qual vacant (purest)": obs[(obs["kind"] == "vacant") & obs["prime_lot"] & q],
        "rcn_resid dep~0 (frozen RCN)": obs[(obs["kind"] == "rcn_resid") & q],
        "rcnld_resid (+ deprec.)": obs[(obs["kind"] == "rcnld_resid") & q],
    }

    print("=== LAND vertical equity — VEI + GLOBAL value deciles (median ratio low->high land value) ===")
    print("    (VEI<0 = expensive land under-assessed MORE = regressive; flat deciles = equitable)")
    for label, d in sets.items():
        vei_and_chart(d, label)

    print("\n=== NEIGHBORHOOD-RELATIVE value deciles (within-nbhd) ===")
    print("    (agreement with global => genuine within-nbhd inequity; divergence => between-nbhd artifact)")
    for label, d in sets.items():
        nbhd_relative_chart(d, label)

    print("\nclean read = vacant streams (no RCN subtraction); residual streams amplify high-value noise.")


if __name__ == "__main__":
    main()
