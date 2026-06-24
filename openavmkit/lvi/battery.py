"""Land-value integrity test battery — runs on one land series, driven by a GroupConfig.

Per-series tests: Step 1 total ratio study, A1 improvement-independence, A2 uniformity, A3
land-vs-anchors (gold), A5 desirability gradient, A6 sales-chasing, A7 summation/sanity, A8
building location-invariance, B3 local spatial uniformity, vertical equity. Data diagnostics
(once, not per series): A0 unit selection, depreciation calibration.

All column names, verdict thresholds, and the A1 feature set come from the config — nothing
jurisdiction-specific is hard-coded. Evidence selection is upstream (the config's filters).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from openavmkit.utilities.stats import calc_cod, calc_ratio_stats_bootstrap, trim_outlier_ratios
from openavmkit.vertical_equity_study import get_vertical_equity_scores
from openavmkit.filters import resolve_filter


# ----------------------------------------------------------------------------- helpers

def _verdict(value, good, warn, higher_is_worse=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if higher_is_worse:
        return "pass" if value <= good else ("warn" if value <= warn else "fail")
    return "pass" if value >= good else ("warn" if value >= warn else "fail")


def _ratio_block(pred, truth, max_trim=0.10):
    pred = np.asarray(pred, float); truth = np.asarray(truth, float)
    m = np.isfinite(pred) & np.isfinite(truth) & (truth > 0) & (pred > 0)
    pred, truth = pred[m], truth[m]
    n = len(pred)
    if n < 5:
        return dict(n=n, n_trim=n, median_ratio=np.nan, median_lo=np.nan, median_hi=np.nan,
                    cod=np.nan, cod_trim=np.nan, vei=np.nan, vei_sig=np.nan, group_stats=None)
    p_t, t_t = trim_outlier_ratios(pred, truth, max_percent=max_trim)
    boot = calc_ratio_stats_bootstrap(p_t, t_t, iterations=2000)
    ve = get_vertical_equity_scores(pd.DataFrame({"truth": truth, "pred": pred}),
                                    sale_field="truth", valuation_field="pred")
    return dict(n=n, n_trim=len(p_t),
                median_ratio=float(boot["median_ratio"].value),
                median_lo=float(boot["median_ratio"].low), median_hi=float(boot["median_ratio"].high),
                cod=float(calc_cod(pred / truth)), cod_trim=float(boot["cod"].value),
                vei=float(ve["vei"]) if ve["vei"] is not None else np.nan,
                vei_sig=float(ve["vei_significance"]) if ve["vei_significance"] is not None else np.nan,
                group_stats=ve["group_stats"])


def _median_chd(df, cluster_field, value_field, min_n=5):
    d = df[[cluster_field, value_field]].copy()
    d = d[d[value_field].notna() & (d[value_field] > 0)]
    chds = [calc_cod(g[value_field].values) for _, g in d.groupby(cluster_field) if len(g) >= min_n]
    return (float(np.median(chds)) if chds else np.nan), len(chds)


def _partial_r2_within(df, y_field, x_fields, group_field):
    xs = [c for c in x_fields if c in df.columns]
    d = df[[y_field, group_field] + xs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 50 or not xs:
        return np.nan, len(d), xs
    gm = d.groupby(group_field)
    yw = (d[y_field] - gm[y_field].transform("mean")).values
    Xw = np.column_stack([(d[c] - gm[c].transform("mean")).values for c in xs])
    ss_tot = float(np.sum(yw ** 2))
    if ss_tot <= 0:
        return np.nan, len(d), xs
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return max(0.0, 1.0 - float(np.sum((yw - Xw @ beta) ** 2)) / ss_tot), len(d), xs


# ----------------------------------------------------------------------------- per-series tests

def test_total(name, s_ratio, total_field, cfg):
    rb = _ratio_block(s_ratio[total_field], s_ratio[cfg.dep])
    g, w = cfg.verdicts.total_cod
    return dict(test="Step1 total", series=name, n=rb["n"], median_ratio=rb["median_ratio"],
                cod=rb["cod_trim"], vei=rb["vei"], verdict=_verdict(rb["cod_trim"], g, w),
                detail=f"median {rb['median_ratio']:.3f} [{rb['median_lo']:.3f},{rb['median_hi']:.3f}] "
                       f"COD_trim {rb['cod_trim']:.1f} VEI {rb['vei']:.1f}")


def test_A1(name, u, cfg):
    F = cfg.fields
    r2, n, xs = _partial_r2_within(u, "_land_psf", list(F.impr_feats), F.neighborhood)
    g, w = cfg.verdicts.a1_partial_r2
    return dict(test="A1 impr-independence", series=name, n=n, partial_r2=r2,
                verdict=_verdict(r2, g, w),
                detail=(f"partial R^2(improvements | {F.neighborhood} FE) = {r2:.3f} on {xs}"
                        if np.isfinite(r2) else "n/a"))


def test_A2(name, u, cfg):
    F = cfg.fields
    land_chd, nl = _median_chd(u, F.land_he_id, "_land_psf")
    impr_chd, ni = _median_chd(u, F.impr_he_id, "_impr_psf")
    g, w = cfg.verdicts.a2_chd
    return dict(test="A2 uniformity", series=name, land_chd=land_chd, impr_chd=impr_chd,
                verdict=_verdict(land_chd, g, w),
                detail=f"land $/sqft CHD {land_chd:.1f} ({nl} clusters); impr $/sqft CHD {impr_chd:.1f} ({ni} clusters)")


def test_A3(name, u, obs, cfg):
    F = cfg.fields
    j = obs.merge(u[["key", F.land_value]].rename(columns={F.land_value: "_lv"}), on="key", how="inner")
    j = j[j["_lv"].notna() & (j["_lv"] > 0) & (j["observed_land"] > 0)].copy()
    blocks = {k: _ratio_block(g["_lv"], g["observed_land"]) for k, g in j.groupby("kind") if len(g) >= 5}
    direct = j[j["kind"] == "direct"]
    gold = pd.concat([direct, j[j["kind"] == "cost_residual"]]).drop_duplicates("key") \
        if cfg.evidence.frozen_sov else direct
    gold_label = "direct + cost_residual" if cfg.evidence.frozen_sov else "direct"
    gb = _ratio_block(gold["_lv"], gold["observed_land"])
    do = _ratio_block(direct["_lv"], direct["observed_land"])
    gd, wd = cfg.verdicts.a3_cod
    kind_str = "  ".join(f"{k}: med {blocks[k]['median_ratio']:.3f} COD {blocks[k]['cod_trim']:.1f} "
                         f"(n={blocks[k]['n']})" for k in ("direct", "cost_residual") if k in blocks)
    return dict(test="A3 land-vs-anchors", series=name, n=gb["n"], gold_label=gold_label,
                median_ratio=gb["median_ratio"], cod=gb["cod_trim"], vei=gb["vei"],
                direct_median=do["median_ratio"], direct_cod=do["cod_trim"], direct_n=do["n"],
                verdict=_verdict(gb["cod_trim"], gd, wd),
                detail=(f"GOLD({gold_label}) n={gb['n']} median {gb['median_ratio']:.3f} "
                        f"[{gb['median_lo']:.3f},{gb['median_hi']:.3f}] COD {gb['cod_trim']:.1f} "
                        f"VEI {gb['vei']:.1f} | direct-only median {do['median_ratio']:.3f} "
                        f"COD {do['cod_trim']:.1f} (n={do['n']}) | by-kind: {kind_str}"))


def test_A5(name, u, obs, cfg):
    """Desirability gradient: does the series' neighborhood land $/sqft track the MARKET land
    gradient (median direct-evidence $/sqft by neighborhood)?"""
    from scipy.stats import spearmanr
    F = cfg.fields
    d = obs[obs["kind"] == "direct"].copy()
    d["psf"] = d["observed_land"] / d["land_sqft"]
    mkt = d.groupby("nbhd")["psf"].agg(["median", "count"])
    u = u.copy(); u["_lp"] = pd.to_numeric(u[F.land_value], errors="coerce") / pd.to_numeric(u[F.land_area], errors="coerce")
    ser = u[u["_lp"] > 0].groupby(F.neighborhood)["_lp"].median()
    g = pd.DataFrame({"mkt": mkt["median"], "n": mkt["count"], "ser": ser}).dropna()
    g = g[g["n"] >= 3]
    rho = spearmanr(g["mkt"], g["ser"])[0] if len(g) >= 5 else np.nan
    gp, wp = cfg.verdicts.a5_rho
    return dict(test="A5 desirability-gradient", series=name, n=len(g), rho_market=rho,
                verdict=_verdict(rho, gp, wp, higher_is_worse=False),
                detail=(f"land vs market-land gradient: Spearman rho={rho:.3f} (n={len(g)} nbhds)"
                        if np.isfinite(rho) else "n/a (too few direct-evidence nbhds)"))


def test_A6(name, u, obs, s, cfg):
    """Sales chasing: is the series silently set to the sale price on direct-evidence parcels?"""
    from openavmkit.sales_chasing import detect_sales_chasing
    F = cfg.fields
    ui = u.drop_duplicates("key").set_index("key")
    si = s.copy(); si["key"] = si["key"].astype(str)
    age = si.drop_duplicates("key").set_index("key")[F.sale_age_days] if F.sale_age_days in si.columns else None
    v = obs[obs["kind"] == "direct"].copy()
    v["suspect"] = pd.to_numeric(v["key"].map(ui[F.land_value]), errors="coerce")
    v["price"] = v["observed_land"]
    v["land_he_id"] = v["key"].map(ui[F.land_he_id])
    v["sale_age_days"] = v["key"].map(age) if age is not None else np.nan
    v = v[v["suspect"].notna() & (v["suspect"] > 0)]
    r = v["suspect"] / v["price"]
    spike = float(np.mean((r - 1.0).abs() <= 0.02)) if len(r) else np.nan
    verdict, detail = "n/a", "n/a"
    if len(v) >= 20:
        try:
            res = detect_sales_chasing(v, suspect_field="suspect", sale_price_field="price",
                                       cluster_field="land_he_id", sale_age_field="sale_age_days")
            vd = getattr(res, "verdict", None) or getattr(res, "summary", str(res))
            verdict = "fail" if "likely" in str(vd).lower() else ("warn" if "possible" in str(vd).lower() else "pass")
            detail = f"detector: {vd}; ratio-at-1.0 spike {100*spike:.1f}% (median ratio {r.median():.3f})"
        except Exception as e:
            detail = f"detector errored ({e}); spike {100*spike:.1f}%, median {r.median():.3f}"
    return dict(test="A6 sales-chasing", series=name, n=len(v), spike=spike, verdict=verdict, detail=detail)


def test_A7(name, u, cfg):
    F = cfg.fields
    d = u.copy()
    land = pd.to_numeric(d[F.land_value], errors="coerce")
    impr = pd.to_numeric(d[F.impr_value], errors="coerce")
    total = pd.to_numeric(d[F.total_value], errors="coerce")

    def _b(series):
        return series.fillna(False).to_numpy(dtype=bool)

    resid = (land + impr) - total
    flags = pd.DataFrame({
        "key": d["key"].to_numpy(),
        "flag_nan": _b(land.isna()), "flag_negative": _b(land < 0),
        "flag_land_exceeds_total": _b((land > total * 1.001) & total.notna()),
        "flag_impr_exceeds_total": _b((impr > total * 1.001) & total.notna()),
        "flag_summation": _b(resid.abs() > (0.01 * total.abs() + 1)),
    })
    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    pct = 100.0 * float(flags[flag_cols].any(axis=1).mean())
    g, w = cfg.verdicts.a7_pct
    return dict(test="A7 sanity", series=name, n=len(d), pct_violations=pct, verdict=_verdict(pct, g, w),
                detail=(f"violations {pct:.2f}% (land>total {100*flags['flag_land_exceeds_total'].mean():.2f}%, "
                        f"neg {100*flags['flag_negative'].mean():.2f}%, nan {100*flags['flag_nan'].mean():.2f}%)")), flags


def test_A8(name, u, cfg):
    F = cfg.fields
    impr_chd, _ = _median_chd(u, F.impr_he_id, "_impr_psf")
    g_ = u.groupby(F.impr_he_id)[F.neighborhood].nunique()
    avg_nbhds = float(g_[g_.index.notna()].mean())
    g, w = cfg.verdicts.a8_chd
    return dict(test="A8 impr loc-invariance", series=name, impr_chd=impr_chd, avg_nbhds_per_cluster=avg_nbhds,
                verdict=_verdict(impr_chd, g, w),
                detail=f"impr $/sqft CHD within {F.impr_he_id} = {impr_chd:.1f} (clusters span avg {avg_nbhds:.1f} neighborhoods)")


def test_B3(name, u, cfg, k=8):
    from scipy.spatial import cKDTree
    F = cfg.fields
    b = u.copy()
    b["_lp"] = pd.to_numeric(b[F.land_value], errors="coerce") / pd.to_numeric(b[F.land_area], errors="coerce")
    b = b[(b["_lp"] > 0) & b[F.latitude].notna() & b[F.longitude].notna()].copy()
    if len(b) < k + 5:
        return dict(test="B3 local-uniformity", series=name, n=len(b), moran=np.nan, verdict="n/a", detail="n/a")
    b["z"] = np.log(b["_lp"]); b["z"] = b["z"] - b.groupby(F.neighborhood)["z"].transform("mean")
    xy = b[[F.latitude, F.longitude]].to_numpy()
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    z = b["z"].to_numpy(); neigh = z[idx[:, 1:]].mean(axis=1)
    moran = float(np.sum(z * neigh) / np.sum(z * z))
    hot = float(np.mean(np.abs(z - neigh) > np.log(2)))
    g, w = cfg.verdicts.b3_hot
    return dict(test="B3 local-uniformity", series=name, n=len(b), moran=moran, hot_pixel_pct=100 * hot,
                verdict=_verdict(hot, g, w), detail=f"Moran's I {moran:.3f} (>0 = smooth); hot-pixel share {100*hot:.1f}%")


def test_vertical_equity(name, u, obs, cfg):
    F = cfg.fields
    j = obs[obs["kind"] == "direct"].merge(
        u[["key", F.land_value]].rename(columns={F.land_value: "_lv"}), on="key", how="inner")
    df = pd.DataFrame({"obs": pd.to_numeric(j["observed_land"], errors="coerce"),
                       "ser": pd.to_numeric(j["_lv"], errors="coerce")}).dropna()
    df = df[(df["obs"] > 0) & (df["ser"] > 0)]
    if len(df) < 20:
        return dict(test="vertical-equity", series=name, n=len(df), vei=np.nan, verdict="n/a", detail="n/a (n<20)")
    vei = get_vertical_equity_scores(df, sale_field="obs", valuation_field="ser")["vei"]
    g, w = cfg.verdicts.ve_vei_abs
    return dict(test="vertical-equity", series=name, n=len(df), vei=vei,
                verdict=_verdict(abs(vei) if vei is not None and np.isfinite(vei) else np.nan, g, w),
                detail=(f"VEI {vei:.1f} on direct evidence (n={len(df)}); "
                        f"{'regressive' if (vei or 0) < 0 else 'progressive'} tilt"
                        if vei is not None and np.isfinite(vei) else "n/a"))


# ----------------------------------------------------------------------------- data diagnostics

def diag_a0_unit(obs, cfg):
    """A0: within-neighborhood size elasticity (beta) + COD of $/sqft vs $/lot on the evidence."""
    d = obs[(obs["observed_land"] > 0) & (obs["land_sqft"] > 0)].drop_duplicates("key").copy()
    if len(d) < 50:
        return dict(test="A0 unit", n=len(d), beta=np.nan, detail="n/a")
    d["ly"] = np.log(d["observed_land"]); d["lx"] = np.log(d["land_sqft"])
    yw = d["ly"] - d.groupby("nbhd")["ly"].transform("mean")
    xw = d["lx"] - d.groupby("nbhd")["lx"].transform("mean")
    beta = float(np.sum(xw * yw) / np.sum(xw * xw)) if float(np.sum(xw * xw)) > 0 else np.nan
    d["psf"] = d["observed_land"] / d["land_sqft"]
    cod_sqft = np.median([calc_cod(g["psf"].values) for _, g in d.groupby("nbhd") if len(g) >= 4] or [np.nan])
    cod_lot = np.median([calc_cod(g["observed_land"].values) for _, g in d.groupby("nbhd") if len(g) >= 4] or [np.nan])
    winner = "$/lot" if (np.isfinite(cod_lot) and cod_lot < cod_sqft) else "$/sqft"
    return dict(test="A0 unit", n=len(d), beta=beta, cod_sqft=float(cod_sqft), cod_lot=float(cod_lot), winner=winner,
                detail=f"size elasticity beta={beta:.2f}; within-nbhd COD $/sqft {cod_sqft:.1f} vs $/lot {cod_lot:.1f} -> {winner}")


def diag_depreciation(s, u, cfg):
    """Depreciation check: within a neighborhood, residual land (sale - cost_bldg_value)/sqft should
    be flat with building age. A non-trivial negative slope => under-depreciation."""
    F = cfg.fields
    ui = u.drop_duplicates("key").set_index("key")
    s = s.copy(); s["key"] = s["key"].astype(str)
    valid = resolve_filter(s, cfg.total_sale_filter)
    s = s[valid].copy()
    for c in [F.cost_bldg_value, F.land_area, F.bldg_area, F.bldg_year_built]:
        s[c] = pd.to_numeric(s["key"].map(ui[c]), errors="coerce")
    pr = pd.to_numeric(s[cfg.dep], errors="coerce")
    s["price"] = pr.where(pr > 0, pd.to_numeric(s[F.sale_price], errors="coerce"))
    s["age"] = pd.to_datetime(s[F.sale_date], errors="coerce").dt.year - s[F.bldg_year_built]
    s[F.neighborhood] = s["key"].map(ui[F.neighborhood])
    imp = s[(s[F.bldg_area] > 0) & (s[F.cost_bldg_value] > 0) & (s["price"] > 0) &
            (s[F.land_area] > 0) & (s["age"].between(0, 200))].copy()
    if len(imp) < 50:
        return dict(test="depreciation", n=len(imp), slope=np.nan, verdict="n/a", detail="n/a")
    imp["resid_psf"] = (imp["price"] - imp[F.cost_bldg_value]) / imp[F.land_area]
    dd = imp.dropna(subset=["resid_psf", "age", F.neighborhood])
    yw = dd["resid_psf"] - dd.groupby(F.neighborhood)["resid_psf"].transform("mean")
    aw = dd["age"] - dd.groupby(F.neighborhood)["age"].transform("mean")
    slope = float(np.sum(aw * yw) / np.sum(aw * aw)) if float(np.sum(aw * aw)) > 0 else np.nan
    g, w = cfg.verdicts.depreciation_drift50
    return dict(test="depreciation", n=len(imp), slope=slope, verdict=_verdict(abs(slope * 50), g, w),
                detail=f"within-nbhd residual-land slope {slope:.3f} $/sqft/yr ({50*slope:+.0f} $/sqft over 50 yrs)")


# ----------------------------------------------------------------------------- orchestrator

def run_battery(name, u, s, obs, cfg):
    """Run the per-series battery on one named land series under ``cfg``. Returns (results, flags)."""
    F = cfg.fields
    u = u.copy()
    u["_land_psf"] = pd.to_numeric(u[F.land_value], errors="coerce") / pd.to_numeric(u[F.land_area], errors="coerce")
    u["_impr_psf"] = pd.to_numeric(u[F.impr_value], errors="coerce") / pd.to_numeric(u[F.bldg_area], errors="coerce")
    ui = u.drop_duplicates("key").set_index("key")
    s = s.copy(); s["key"] = s["key"].astype(str)
    s_ratio = s[resolve_filter(s, cfg.total_sale_filter)].copy()
    s_ratio["__total"] = s_ratio["key"].map(ui[F.total_value])
    a7, flags = test_A7(name, u, cfg)
    results = {
        "total": test_total(name, s_ratio, "__total", cfg),
        "A1": test_A1(name, u, cfg),
        "A2": test_A2(name, u, cfg),
        "A3": test_A3(name, u, obs, cfg),
        "A5": test_A5(name, u, obs, cfg),
        "A6": test_A6(name, u, obs, s, cfg),
        "A7": a7,
        "A8": test_A8(name, u, cfg),
        "B3": test_B3(name, u, cfg),
        "VE": test_vertical_equity(name, u, obs, cfg),
    }
    return results, flags
