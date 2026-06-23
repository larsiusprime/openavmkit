"""Assessor LAND ratio study, broken down BY MODEL GROUP and by anchor-cleaning stage.

For each model group with vacant-land sales, computes the assessor land ratio
(assr_land_value / vacant sale_price_time_adj) at three stages:
  all vacant        -> every vacant_sale
  genuinely vacant  -> drop teardowns / mislabeled (a structure present)
  prime             -> + comparable size vs that group's neighborhood built peers, sane shape,
                       and (for residential groups) residential zoning

The point: see where the assessor's published land split is level/uniform and where it isn't,
per market segment, on a defensible "prime buildable" anchor set.

Run from repo root:  python research/lvi_land_ratio_by_group.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from openavmkit.pipeline import read_pickle
from openavmkit.data import get_hydrated_sales_from_sup
from openavmkit.utilities.stats import calc_cod, trim_outlier_ratios
import lvi_anchors as A

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
NB = "neighborhood_filled"
DEP = "sale_price_time_adj"


def _rstats(land, price):
    land = pd.to_numeric(land, errors="coerce").values
    price = pd.to_numeric(price, errors="coerce").values
    m = np.isfinite(land) & np.isfinite(price) & (price > 0) & (land > 0)
    land, price = land[m], price[m]
    if len(land) < 5:
        return len(land), np.nan, np.nan, np.nan
    ratios = land / price
    lt, pt = trim_outlier_ratios(land, price, max_percent=0.10)
    return len(land), float(np.median(ratios)), float(calc_cod(ratios)), float(calc_cod(lt / pt))


def main():
    os.chdir(DATA)
    sup = read_pickle("out/2-clean-sup")
    u = sup.universe
    assr_land = u.drop_duplicates("key").set_index("key")["assr_land_value"]
    s = get_hydrated_sales_from_sup(sup)
    v = s[(s["valid_sale"] == True) & (s.get("vacant_sale", False) == True)].copy()
    v["key"] = v["key"].astype(str)
    v["assr_land_value"] = v["key"].map(assr_land)
    # use time-adjusted price; fall back to raw price where time-adj is absent (e.g. the
    # unmodeled UNKNOWN group has no sale_price_time_adj). Land price = the vacant sale price.
    ta = pd.to_numeric(v[DEP], errors="coerce")
    v[DEP] = ta.where(ta > 0, pd.to_numeric(v["sale_price"], errors="coerce"))

    # ---- validate the zoning classifier on the high-count codes ----
    print("=== zoning classifier check (top codes by parcel count) ===")
    zc = u["zoning"].astype(str).value_counts().head(30)
    by_class = {}
    for code, n in zc.items():
        cls = A.classify_zoning(code)
        by_class.setdefault(cls, []).append(f"{code}({n})")
    for cls in ("residential", "rural", "nonres", "other"):
        print(f"  {cls:<12}: {', '.join(by_class.get(cls, []))}")

    # ---- per-model-group land ratio study, three stages ----
    print("\n=== assessor land ratio study by model group "
          "(median | COD | COD_trim) ===")
    hdr = f"{'model_group':<20}{'stage':<18}{'n':>6}{'median':>9}{'COD':>8}{'COD_tr':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for mg, g in v.groupby("model_group"):
        if len(g) < 5:
            continue
        residential = mg in A.RESIDENTIAL_GROUPS
        bands = A.neighborhood_size_bands(u, NB, mg)
        gp = A.add_prime_flags(g.copy(), bands, NB, residential_zoning=residential)
        gp["qualified"] = A.qualified_sale_mask(gp)   # explicit A/C deed code (non-circular)
        stages = [
            ("all vacant", gp),
            ("genuinely vacant", gp[~gp["dq_teardown"]]),
            ("prime", gp[gp["prime"]]),
            ("qualified (A/C)", gp[gp["qualified"]]),
            ("prime+qualified", gp[gp["prime"] & gp["qualified"]]),
        ]
        for label, df in stages:
            n, med, cod, codt = _rstats(df["assr_land_value"], df[DEP])
            rows.append((mg, label, n, med, cod, codt))
            ms = f"{med:.3f}" if np.isfinite(med) else "  n/a"
            cs = f"{cod:.1f}" if np.isfinite(cod) else " n/a"
            ct = f"{codt:.1f}" if np.isfinite(codt) else " n/a"
            print(f"{mg:<20}{label:<18}{n:>6}{ms:>9}{cs:>8}{ct:>8}")
        print()

    pd.DataFrame(rows, columns=["model_group", "stage", "n", "median", "cod", "cod_trim"]) \
        .to_csv("out/lvi/land_ratio_by_group.csv", index=False)
    print("wrote out/lvi/land_ratio_by_group.csv")


if __name__ == "__main__":
    main()
