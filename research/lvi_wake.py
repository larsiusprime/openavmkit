"""Land-value integrity battery — Wake County flagship runner.

Scores the assessor's EXISTING land series (assr_land_value / assr_impr_value /
assr_market_value). Reads from the cleaned sup (out/2-clean-sup) — it carries every signal the
battery + anchor scrutiny need on one internally-consistent parcel set: he_ids, cost-book fields
(bldg_condition_pct), zoning, geom_rectangularity_num, and the deed qualification code
(disq_flag) used for the NON-CIRCULAR land gold standard. (The multi_mra model emits no genuine
land split — prediction_land_sqft == prediction/land_area — so there is no AVM land series to
compare; an AVM column auto-enables only if a real land_value/impr_value export appears.)

Run from repo root:  python research/lvi_wake.py
Outputs (gitignored):  out/lvi/scorecard.txt , out/lvi/evidence_assessor.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import lvi_anchors
import lvi_battery as B

DATA = os.path.join("notebooks", "pipeline", "data", "us-nc-wake")
MG = "single_family"
NB = "neighborhood_filled"
DEP = "sale_price_time_adj"


def _load():
    os.chdir(DATA)
    from openavmkit.pipeline import read_pickle
    from openavmkit.data import get_hydrated_sales_from_sup
    sup = read_pickle("out/2-clean-sup")
    u = sup.universe[sup.universe["model_group"] == MG].copy()
    u["key"] = u["key"].astype(str)
    s = get_hydrated_sales_from_sup(sup)
    s = s[(s["model_group"] == MG) & (s["valid_sale"] == True) & (s["sale_price"] > 0)].copy()
    s["key"] = s["key"].astype(str)
    return u, s


def _assessor_universe(u):
    m = u.copy()
    m["land_value"] = m["assr_land_value"]
    m["impr_value"] = m["assr_impr_value"]
    m["total_value"] = m["assr_market_value"]
    return m


def _avm_universe(u):
    m = u.copy()
    m["land_value"] = m["prediction_land_sqft"] * m["land_area_sqft"]
    m["impr_value"] = m["prediction_impr_sqft"] * m["bldg_area_finished_sqft"]
    m["total_value"] = m["prediction"]
    return m


def _has_real_land_split(u):
    """True only if a genuine land/improvement decomposition is present to test. The sup carries
    no model prediction split, and Wake's multi_mra `prediction_land_sqft` is just
    prediction/land_area (the total relabeled), so this is False today — the AVM column stays
    suppressed until a real land_value/impr_value export appears."""
    if "prediction_land_sqft" not in u.columns or "prediction" not in u.columns:
        return False
    land = u["prediction_land_sqft"] * u["land_area_sqft"]
    return np.isclose(land, u["prediction"], rtol=1e-3).mean() < 0.5


def _fmt(results, key, field, fmt="{:.2f}"):
    v = results[key].get(field)
    return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"


def scorecard(res_a, res_b=None, name_a="assessor", name_b="avm"):
    rows = [
        ("Step1 total: median ratio", "total", "median_ratio", "{:.3f}"),
        ("Step1 total: COD_trim",      "total", "cod",          "{:.1f}"),
        ("Step1 total: VEI",           "total", "vei",          "{:.1f}"),
        ("A1 improvement partial-R^2", "A1",    "partial_r2",   "{:.3f}"),
        ("A2 land $/sqft CHD",         "A2",    "land_chd",     "{:.1f}"),
        ("A2 impr $/sqft CHD",         "A2",    "impr_chd",     "{:.1f}"),
        ("A3 land vs anchors: median", "A3",    "median_ratio", "{:.3f}"),
        ("A3 land vs anchors: COD",    "A3",    "cod",          "{:.1f}"),
        ("A3 land vs anchors: VEI",    "A3",    "vei",          "{:.1f}"),
        ("A7 sanity violations %",     "A7",    "pct_violations","{:.2f}"),
        ("A8 impr loc-invariance CHD", "A8",    "impr_chd",     "{:.1f}"),
    ]
    has_b = res_b is not None
    out = []
    out.append("=" * 80)
    out.append(f"  LAND VALUE INTEGRITY SCORECARD - Wake County / {MG}")
    if has_b:
        out.append(f"  {'metric':<32}{name_a:>14}{name_b:>14}   verdict(a/b)")
    else:
        out.append(f"  {'metric':<32}{name_a:>14}   verdict")
    out.append("-" * 80)
    for label, tkey, field, fmt in rows:
        a = _fmt(res_a, tkey, field, fmt)
        va = res_a[tkey].get("verdict", "?")
        if has_b:
            b = _fmt(res_b, tkey, field, fmt)
            vb = res_b[tkey].get("verdict", "?")
            out.append(f"  {label:<32}{a:>14}{b:>14}   {va}/{vb}")
        else:
            out.append(f"  {label:<32}{a:>14}   {va}")
    out.append("=" * 80)
    for tkey in ("total", "A1", "A2", "A3", "A7", "A8"):
        out.append(f"  [{tkey}] {name_a}: {res_a[tkey]['detail']}")
        if has_b:
            out.append(f"  [{tkey}] {name_b}: {res_b[tkey]['detail']}")
    return "\n".join(out)


def evidence_packet(name, u, obs, flags):
    """Per-parcel defensibility table for one land series."""
    d = u[["key", "land_value", "impr_value", "total_value", "land_area_sqft",
           NB, "land_he_id"]].copy()
    d["land_psf"] = d["land_value"] / d["land_area_sqft"]
    d["cluster_median_psf"] = d.groupby("land_he_id")["land_psf"].transform("median")
    d["cluster_ratio"] = d["land_psf"] / d["cluster_median_psf"]
    # cost-based residual cross-check: land ~ total - RCNLD(=assr_impr_value).
    # NOTE degenerate (==1.0) for the assessor series since total-assr_impr==assr_land by
    # construction; it is a meaningful silver cross-check only for the AVM series.
    ui = u.set_index("key")
    d["cost_rcn"] = lvi_anchors.reconstruct_rcn(ui).reindex(d["key"]).values
    d["residual_land"] = d["total_value"].values - ui.loc[d["key"], "assr_impr_value"].values
    d["residual_ratio"] = d["land_value"] / d["residual_land"].where(d["residual_land"] > 0)
    # direct evidence: neighborhood-median observed land $/sqft from anchors
    nb_psf = (obs["observed_land"] / obs["land_sqft"]).groupby(obs["nbhd"]).median()
    nb_n = obs.groupby("nbhd")["observed_land"].count()
    d["nbhd_anchor_psf"] = d[NB].map(nb_psf)
    d["nbhd_anchor_n"] = d[NB].map(nb_n).fillna(0).astype(int)
    d["anchor_ratio"] = d["land_psf"] / d["nbhd_anchor_psf"]
    d = d.merge(flags, on="key", how="left")
    flag_cols = [c for c in d.columns if c.startswith("flag_")]
    d["n_flags"] = d[flag_cols].fillna(False).astype(bool).sum(axis=1)
    rr = pd.to_numeric(d["residual_ratio"], errors="coerce")
    ar = pd.to_numeric(d["anchor_ratio"], errors="coerce")
    cr = pd.to_numeric(d["cluster_ratio"], errors="coerce")
    ok_resid = rr.isna() | rr.between(0.6, 1.6)
    ok_anchor = (d["nbhd_anchor_n"] >= 3) & ar.between(0.6, 1.6)
    ok_cluster = cr.between(0.5, 2.0)
    d["land_integrity_confidence"] = np.where(
        d["n_flags"] > 0, "low",
        np.where(ok_cluster & (ok_anchor | ok_resid), "high", "med"))
    return d


def main():
    u, s = _load()
    print(f"loaded (from sup): universe={u.shape}  sales={s.shape}")

    ucols = u.drop_duplicates("key").set_index("key")
    obs = lvi_anchors.build_land_observations(s, ucols, NB, DEP)
    # GOLD standard = PRIME (genuinely buildable, locally-comparable) AND explicitly QUALIFIED
    # (disq_flag whitelist A/C — non-circular deed code). Both gates required for vacant anchors.
    prime = lvi_anchors.prime_lot_mask(u, NB, MG, residential_zoning=True)
    obs["prime_lot"] = obs["key"].map(prime).fillna(False)
    obs["prime"] = (obs["kind"] == "vacant") & obs["prime_lot"] & obs["qualified"].fillna(False)
    n_vac = int((obs["kind"] == "vacant").sum())
    n_primelot = int(((obs["kind"] == "vacant") & obs["prime_lot"]).sum())
    print(lvi_anchors.summarize(obs)
          + f"  | vacant {n_vac} -> prime-lot {n_primelot} -> +qualified(A/C) {int(obs['prime'].sum())}")

    # Step-1 total ratio study: assessor total vs sale on the current valid ratio-study sales
    s_assr = s[s.get("valid_for_ratio_study", True) == True].copy()
    s_assr["total_value"] = s_assr["assr_market_value"]

    res_a, flags_a = B.run_battery("assessor", _assessor_universe(u), s_assr, obs, NB)
    series = [("assessor", _assessor_universe(u), flags_a)]
    res_b = None
    if _has_real_land_split(u):
        s_avm = s.copy()  # would carry a real AVM total here
        res_b, flags_b = B.run_battery("avm", _avm_universe(u), s_avm, obs, NB)
        series.append(("avm", _avm_universe(u), flags_b))
    else:
        print("NOTE: no genuine land/improvement split available (multi_mra exports "
              "prediction_land_sqft == prediction/land_area). Scoring the assessor roll only; "
              "AVM column auto-enables when a real land_value/impr_value export exists.")

    card = scorecard(res_a, res_b)
    print("\n" + card + "\n")

    os.makedirs("out/lvi", exist_ok=True)
    with open("out/lvi/scorecard.txt", "w") as f:
        f.write(card + "\n")
    for nm, ures, fl in series:
        ev = evidence_packet(nm, ures, obs, fl)
        ev.to_csv(f"out/lvi/evidence_{nm}.csv", index=False)
        print(f"{nm} confidence: " + ", ".join(
            f"{k}={v}" for k, v in ev['land_integrity_confidence'].value_counts().items()))
    print(f"\nwrote out/lvi/scorecard.txt and evidence for: {[s[0] for s in series]}")


if __name__ == "__main__":
    main()
