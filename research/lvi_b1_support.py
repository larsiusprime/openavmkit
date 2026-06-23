"""B1 — market support / coverage (the protest-risk map), by evidence type.

How much of the single-family land base sits NEAR market land evidence (on-manifold) vs FAR
(extrapolated)? Run separately for each evidence stream so we can see how much the coverage
depends on the frozen-SOV RCN residuals vs the pure direct vacant sales.

Run from repo root:  python research/lvi_b1_support.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"
MI_PER_DEG_LAT = 69.0


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    u["assr_land_value"] = pd.to_numeric(u["assr_land_value"], errors="coerce")
    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    obs = A.build_land_observations(s, ui, NB)
    prime = A.prime_lot_mask(u, NB, MG, residential_zoning=True)
    obs["prime_lot"] = obs["key"].map(prime).fillna(False)
    q = obs["qualified"].fillna(False)

    evsets = {
        "vacant (all valid)":        obs[obs["kind"] == "vacant"]["key"].unique(),
        "vacant (prime+qualified)":  obs[(obs["kind"] == "vacant") & obs["prime_lot"] & q]["key"].unique(),
        "rcn_resid (qualified)":     obs[(obs["kind"] == "rcn_resid") & q]["key"].unique(),
        "pooled (vacant+rcn)": np.unique(np.concatenate([
            obs[(obs["kind"] == "vacant") & obs["prime_lot"] & q]["key"].unique(),
            obs[(obs["kind"] == "rcn_resid") & q]["key"].unique()])),
    }

    u = u[u["latitude"].notna() & u["longitude"].notna() & (u["assr_land_value"] > 0)].copy()
    lat0 = np.radians(u["latitude"].mean())
    def to_xy(df):
        return np.column_stack([df["longitude"].to_numpy() * MI_PER_DEG_LAT * np.cos(lat0),
                                df["latitude"].to_numpy() * MI_PER_DEG_LAT])
    uxy = to_xy(u); tot = u["assr_land_value"].sum(); nnb = u[NB].nunique()

    print(f"single-family parcels: {len(u):,}   (neighborhoods: {nnb:,})\n")
    hdr = (f"{'evidence set':<26}{'pts':>6}{'med mi':>8}{'p90 mi':>8}"
           f"{'%val<1mi':>10}{'%val>2mi':>10}{'%val in-zone':>14}{'nbhds':>7}")
    print(hdr); print("-" * len(hdr))
    for label, keys in evsets.items():
        ev = u[u["key"].isin(keys)]
        if len(ev) < 1:
            print(f"{label:<26}{'0':>6}  (no points)"); continue
        tree = cKDTree(to_xy(ev))
        d, _ = tree.query(uxy, k=1)
        in_zone = u[NB].isin(ev[NB].unique())
        med, p90 = np.median(d), np.quantile(d, 0.9)
        val_lt1 = 100 * u.loc[d < 1.0, "assr_land_value"].sum() / tot
        val_gt2 = 100 * u.loc[d > 2.0, "assr_land_value"].sum() / tot
        val_inzone = 100 * u.loc[in_zone, "assr_land_value"].sum() / tot
        ncov = ev[NB].nunique()
        print(f"{label:<26}{len(ev):>6}{med:>8.2f}{p90:>8.2f}{val_lt1:>9.1f}%{val_gt2:>9.1f}%"
              f"{val_inzone:>13.1f}%{ncov:>7}")

    print("\n  med/p90 mi = distance to nearest evidence point;  %val = share of assessed land VALUE")
    print("  %val in-zone = land value in neighborhoods (VCS) that contain >=1 evidence point")


if __name__ == "__main__":
    main()
