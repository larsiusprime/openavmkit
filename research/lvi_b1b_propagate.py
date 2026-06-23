"""B1b — analogical support / evidence propagation (the "sudoku" coverage tool).

75% of land value sits in VCS zones with no DIRECT land evidence (B1). But a zone can be tied to
a directly-supported zone through MATCHED BUILDINGS: given building-value location-invariance
(A8, validated CHD 6.3), matched buildings (same impr_he_id) selling in zones A and B satisfy
land_B - land_A = price_B - price_A — the building cancels. So a supported zone's validated land
level propagates to unsupported zones along matched-building bridges (transfer validated by A5,
rho=0.911). This measures how much land value that brings under support.

Cascade: DIRECT (in-zone evidence) -> +1 hop (shares matched buildings with a direct zone)
-> +full propagation (connected component) -> unreachable (no improved sales to bridge on).

Run from repo root:  python research/lvi_b1b_propagate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
import lvi_anchors as A

NB = "neighborhood_filled"; MG = "single_family"


def main():
    os.chdir(os.path.join("notebooks", "pipeline", "data", "us-nc-wake"))
    sup = read_pickle("out/2-clean-sup"); u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str); ui = u.drop_duplicates("key").set_index("key")
    u["assr_land_value"] = pd.to_numeric(u["assr_land_value"], errors="coerce")
    nb_val = u.groupby(NB)["assr_land_value"].sum()
    tot = nb_val.sum()
    def vshare(nbset): return 100 * nb_val[nb_val.index.isin(nbset)].sum() / tot

    s = get_hydrated_sales_from_sup(sup); s = s[(s["model_group"] == MG) & (s["valid_sale"] == True)].copy()
    s["key"] = s["key"].astype(str)
    obs = A.build_land_observations(s, ui, NB)
    prime = A.prime_lot_mask(u, NB, MG, residential_zoning=True)
    obs["prime_lot"] = obs["key"].map(prime).fillna(False); q = obs["qualified"].fillna(False)
    direct = set(pd.concat([
        obs[(obs["kind"] == "vacant") & obs["prime_lot"] & q]["nbhd"],
        obs[(obs["kind"] == "rcn_resid") & q]["nbhd"]]).dropna().unique())

    # matched-building bridges: improved sales carry impr_he_id; a cluster present in two zones
    # bridges them. Require >=1 sale per zone in the shared cluster.
    imp = s[(s.get("vacant_sale", False) != True) & s["impr_he_id"].notna()].copy()
    imp["nbhd"] = imp["key"].map(ui[NB])
    imp = imp.dropna(subset=["nbhd"])
    nb_clusters = imp.groupby("nbhd")["impr_he_id"].agg(lambda x: set(x))
    cluster_nbhds = imp.groupby("impr_he_id")["nbhd"].agg(lambda x: set(x))
    nbhds_with_sales = set(nb_clusters.index)

    print(f"single-family: {u[NB].nunique():,} neighborhoods, ${tot/1e9:.1f}B assessed land")
    print(f"neighborhoods with improved sales (bridge-able): {len(nbhds_with_sales):,}")
    print(f"directly-supported (in-zone land evidence): {len(direct):,}\n")

    # 1-hop: a zone shares >=min_shared matched-building clusters with a DIRECT zone
    def n_shared_with(target_set, nbhd):
        cl = nb_clusters.get(nbhd, set())
        return sum(1 for c in cl if len(cluster_nbhds[c] & target_set) > 0)
    for min_shared in (1, 3):
        onehop = set(direct)
        for nb in nbhds_with_sales - direct:
            if n_shared_with(direct, nb) >= min_shared:
                onehop.add(nb)
        print(f"+1 hop (>= {min_shared} matched-building bridge(s) to a direct zone): "
              f"{len(onehop):,} zones, {vshare(onehop):.1f}% of land value")

    # full propagation: connected component(s) of the matched-building graph containing direct zones
    supported = set(direct); frontier = set(direct)
    while frontier:
        touched = set()
        for nb in frontier:
            for c in nb_clusters.get(nb, set()):
                touched |= cluster_nbhds[c]
        new = touched - supported
        supported |= new; frontier = new
    print(f"+full propagation (connected to direct evidence via any bridge chain): "
          f"{len(supported):,} zones, {vshare(supported):.1f}% of land value")

    unreachable_with_sales = nbhds_with_sales - supported
    no_sales = set(nb_val.index) - nbhds_with_sales
    print(f"\nstill unreachable: {len(unreachable_with_sales):,} zones with sales but no bridge chain "
          f"({vshare(unreachable_with_sales):.1f}% value)")
    print(f"no improved sales at all (can't bridge — need a fallback): {len(no_sales):,} zones "
          f"({vshare(no_sales):.1f}% value)")
    print("\n  per-link evidence = matched-building (impr_he_id) price difference = land difference")
    print("  (building cancels by A8); anchored to direct land evidence; gradient validated by A5.")


if __name__ == "__main__":
    main()
