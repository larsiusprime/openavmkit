"""Land-value integrity battery (MVP) — method-agnostic tests on an existing land series.

Implements a calibrated subset of research/land_value_integrity_spec.md:

  Step 1  total ratio study (precondition)         -> level/COD/VEI of total value vs sales
  A1      improvement independence (HEADLINE)       -> partial R^2 of improvement block on land $/sqft
  A2      uniformity (land + building)              -> median CHD of land $/sqft / impr $/sqft in HE clusters
  A3      land ratio study vs anchors (gold)        -> ratio of est land vs observed land obs
  A7      summation & sanity                        -> identity + bound checks
  A8      improvement location-invariance (A1 mirror)-> median CHD of impr $/sqft within impr_he_id

Each test returns a dict: {metric(s), verdict in {pass,warn,fail}, detail, flags?}.
run_battery() runs them all on one named land series and returns the results + a per-parcel
flag table. The comparative scorecard (assessor vs AVM) is produced by the caller running this
twice and printing side by side.

Reused openavmkit utilities (verified): utilities/stats.{calc_cod, calc_ratio_stats_bootstrap,
trim_outlier_ratios}, vertical_equity_study.get_vertical_equity_scores.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from openavmkit.utilities.stats import calc_cod, calc_ratio_stats_bootstrap, trim_outlier_ratios
from openavmkit.vertical_equity_study import get_vertical_equity_scores

IMPR_FEATS = ["bldg_area_finished_sqft", "bldg_age_years", "bldg_quality_num", "bldg_condition_num"]


# ----------------------------------------------------------------------------- helpers

def _verdict(value, good, warn, higher_is_worse=True):
    """Map a metric to pass/warn/fail given two thresholds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if higher_is_worse:
        return "pass" if value <= good else ("warn" if value <= warn else "fail")
    return "pass" if value >= good else ("warn" if value >= warn else "fail")


def _ratio_block(pred, truth, max_trim=0.10):
    """IAAO level/uniformity on a (pred, truth) pair, trimmed; + VEI. Returns a flat dict."""
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, float)
    m = np.isfinite(pred) & np.isfinite(truth) & (truth > 0) & (pred > 0)
    pred, truth = pred[m], truth[m]
    n = len(pred)
    if n < 5:
        return dict(n=n, n_trim=n, median_ratio=np.nan, median_lo=np.nan, median_hi=np.nan,
                    cod=np.nan, cod_trim=np.nan, vei=np.nan, vei_sig=np.nan, group_stats=None)
    p_t, t_t = trim_outlier_ratios(pred, truth, max_percent=max_trim)
    boot = calc_ratio_stats_bootstrap(p_t, t_t, iterations=2000)
    df = pd.DataFrame({"truth": truth, "pred": pred})
    ve = get_vertical_equity_scores(df, sale_field="truth", valuation_field="pred")
    return dict(
        n=n, n_trim=len(p_t),
        median_ratio=float(boot["median_ratio"].value),
        median_lo=float(boot["median_ratio"].low), median_hi=float(boot["median_ratio"].high),
        cod=float(calc_cod(pred / truth)),          # COD of untrimmed ratios
        cod_trim=float(boot["cod"].value),          # bootstrap COD on trimmed ratios
        vei=float(ve["vei"]) if ve["vei"] is not None else np.nan,
        vei_sig=float(ve["vei_significance"]) if ve["vei_significance"] is not None else np.nan,
        group_stats=ve["group_stats"],
    )


def _median_chd(df, cluster_field, value_field, min_n=5):
    """Median across clusters of the within-cluster COD of `value_field`. Clusters with
    fewer than `min_n` valid rows are dropped (a singleton's COD is a meaningless 0)."""
    d = df[[cluster_field, value_field]].copy()
    d = d[d[value_field].notna() & (d[value_field] > 0)]
    chds = []
    for _, g in d.groupby(cluster_field):
        if len(g) >= min_n:
            chds.append(calc_cod(g[value_field].values))
    return float(np.median(chds)) if chds else np.nan, len(chds)


def _partial_r2_within(df, y_field, x_fields, group_field):
    """Partial R^2 of the x-block on y, *after absorbing group fixed effects* (Frisch-Waugh-
    Lovell within transform): demean y and X by group, regress, report R^2. This is the share
    of within-neighborhood land-rate variation explained by what's built on the lot -> the A1
    cornerstone. ~0 is good; large means land tracks the improvement (allocation signature)."""
    xs = [c for c in x_fields if c in df.columns]
    cols = [y_field, group_field] + xs
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 50 or not xs:
        return np.nan, len(d), xs
    # within transform: subtract group means
    gm = d.groupby(group_field)
    yw = (d[y_field] - gm[y_field].transform("mean")).values
    Xw = np.column_stack([(d[c] - gm[c].transform("mean")).values for c in xs])
    # drop all-zero (singleton-group) rows? FWL keeps them; they contribute 0 variance.
    ss_tot = float(np.sum(yw ** 2))
    if ss_tot <= 0:
        return np.nan, len(d), xs
    # least squares (no intercept; data is already demeaned)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
    return max(0.0, r2), len(d), xs


def _decile_chart(df, ratio_col, level_col, q=10):
    """Median ratio per decile of `level_col`; returns (DataFrame, is_flat). Used for the
    A3 vertical-equity decile charts (global vs neighborhood-relative)."""
    d = df[[ratio_col, level_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d[level_col].nunique() < q or len(d) < 2 * q:
        return None, None
    d["bin"] = pd.qcut(d[level_col], q=q, labels=False, duplicates="drop")
    chart = d.groupby("bin")[ratio_col].median()
    # crude monotone-trend test: Spearman of bin vs median ratio
    rho = np.corrcoef(chart.index.values, chart.values)[0, 1]
    return chart, bool(abs(rho) < 0.5)


# ----------------------------------------------------------------------------- tests

def test_total(name, s, total_field, dep="sale_price_time_adj"):
    rb = _ratio_block(s[total_field], s[dep])
    level_off = abs(rb["median_ratio"] - 1.0) if np.isfinite(rb["median_ratio"]) else np.nan
    return dict(
        test="Step1 total", series=name, n=rb["n"],
        median_ratio=rb["median_ratio"], cod=rb["cod_trim"], vei=rb["vei"],
        verdict=_verdict(rb["cod_trim"], 15, 25),  # IAAO improved-residential COD bands
        detail=f"median {rb['median_ratio']:.3f} [{rb['median_lo']:.3f},{rb['median_hi']:.3f}] "
               f"COD_trim {rb['cod_trim']:.1f} VEI {rb['vei']:.1f}",
    )


def test_A1(name, u, land_psf_field, nb):
    r2, n, xs = _partial_r2_within(u, land_psf_field, IMPR_FEATS, nb)
    return dict(
        test="A1 impr-independence", series=name, n=n,
        partial_r2=r2, verdict=_verdict(r2, 0.05, 0.15),
        detail=(f"partial R^2(improvements | {nb} FE) = {r2:.3f}  on {xs}"
                if np.isfinite(r2) else "n/a"),
    )


def test_A2(name, u, land_psf_field, impr_psf_field):
    land_chd, n_land = _median_chd(u, "land_he_id", land_psf_field)
    impr_chd, n_impr = _median_chd(u, "impr_he_id", impr_psf_field)
    return dict(
        test="A2 uniformity", series=name,
        land_chd=land_chd, impr_chd=impr_chd,
        verdict=_verdict(land_chd, 15, 25),
        detail=f"land $/sqft CHD {land_chd:.1f} ({n_land} clusters); "
               f"impr $/sqft CHD {impr_chd:.1f} ({n_impr} clusters)",
    )


def test_A3(name, u, obs, land_field="land_value"):
    """Land ratio study vs observed land. The GOLD standard is PRIME vacant lots — genuinely
    buildable, locally-comparable lots (obs['prime']); whole sale price is land, fully
    independent of the estimate. Reported alongside:
      - all vacant / genuinely-vacant (context: does prime-cleaning change the verdict?)
      - NEW_CONSTR = a SILVER cost-residual cross-check (land = price - RCNLD); semi-circular
        for the *assessor* series, so reported, not used for its verdict.
    """
    j = obs.merge(u[["key", land_field]], on="key", how="inner")
    j = j[j[land_field].notna() & (j[land_field] > 0) & (j["observed_land"] > 0)].copy()
    j["ratio"] = j[land_field] / j["observed_land"]
    has_prime = "prime" in j.columns

    blocks = {}
    for kind in ("vacant", "teardown", "rcn_resid", "rcnld_resid"):
        sub = j[j["kind"] == kind]
        if len(sub) >= 5:
            blocks[kind] = _ratio_block(sub[land_field], sub["observed_land"])

    vacant = j[j["kind"] == "vacant"]
    prime = vacant[vacant["prime"] == True] if has_prime else vacant
    # GOLD STANDARD (predicated on the frozen, publicly-adopted Schedule-of-Values cost table):
    #   prime+qualified VACANT  UNION  qualified RCN-residuals at depreciation~0.
    # With RCN exogenous (frozen SOV), observed_land = price - RCN is non-circular, so the dep~0
    # residual is bona-fide direct land evidence (~60x the vacant sample). rcnld_resid stays OUT
    # (its depreciation factor is a separate, un-vouched assessor judgment).
    qual = j["qualified"].fillna(False) if "qualified" in j.columns else True
    rcn_gold = j[(j["kind"] == "rcn_resid") & qual]
    gold = pd.concat([prime, rcn_gold]).drop_duplicates("key")
    gold_label = "prime-vacant + rcn_resid(dep~0)"
    gb = _ratio_block(gold[land_field], gold["observed_land"])
    allvac = _ratio_block(vacant[land_field], vacant["observed_land"])
    prime_only = _ratio_block(prime[land_field], prime["observed_land"])  # purest, for contrast
    pooled = _ratio_block(j[land_field], j["observed_land"])

    # decile charts on the gold sample: global level + neighborhood-relative level
    g2 = gold.copy()
    g2["obs_psf"] = g2["observed_land"] / g2["land_sqft"]
    g2["rel_level"] = g2["obs_psf"] / g2.groupby("nbhd")["obs_psf"].transform("median")
    global_chart, global_flat = _decile_chart(g2, "ratio", "observed_land")
    rel_chart, rel_flat = _decile_chart(g2, "ratio", "rel_level")

    by_kind = j.groupby("kind").size().to_dict()
    n_prime = int((vacant["prime"] == True).sum()) if has_prime else None
    kind_str = "  ".join(
        f"{k}: med {blocks[k]['median_ratio']:.3f} COD {blocks[k]['cod_trim']:.1f} (n={blocks[k]['n']})"
        for k in ("vacant", "teardown", "rcn_resid", "rcnld_resid") if k in blocks)
    return dict(
        test="A3 land-vs-anchors", series=name, n=gb["n"], by_kind=by_kind, gold_label=gold_label,
        median_ratio=gb["median_ratio"], cod=gb["cod_trim"], vei=gb["vei"],
        allvac_median=allvac["median_ratio"], allvac_cod=allvac["cod_trim"],
        pooled_median=pooled["median_ratio"], pooled_cod=pooled["cod_trim"],
        n_prime=n_prime, blocks=blocks,
        verdict=_verdict(gb["cod_trim"], 15, 25),  # IAAO vacant-land COD band (very-large/active)
        global_chart=global_chart, global_flat=global_flat,
        rel_chart=rel_chart, rel_flat=rel_flat,
        prime_only_median=prime_only["median_ratio"], prime_only_cod=prime_only["cod_trim"],
        prime_only_n=prime_only["n"],
        detail=(f"GOLD({gold_label}) n={gb['n']} median {gb['median_ratio']:.3f} "
                f"[{gb['median_lo']:.3f},{gb['median_hi']:.3f}] COD {gb['cod_trim']:.1f} "
                f"VEI {gb['vei']:.1f} | purest vacant-prime-qual median {prime_only['median_ratio']:.3f} "
                f"COD {prime_only['cod_trim']:.1f} (n={prime_only['n']}) | by-kind: {kind_str}"),
    )


def test_A7(name, u, land_field="land_value", impr_field="impr_value", total_field="total_value"):
    d = u.copy()
    land = pd.to_numeric(d[land_field], errors="coerce")
    impr = pd.to_numeric(d[impr_field], errors="coerce")
    total = pd.to_numeric(d[total_field], errors="coerce")
    n = len(d)

    def _b(series):  # NA-safe boolean column -> plain numpy bool
        return series.fillna(False).to_numpy(dtype=bool)

    resid = (land + impr) - total
    floored = pd.to_numeric(d.get("land_value_raw", land), errors="coerce") < 0
    flags = pd.DataFrame({
        "key": d["key"].to_numpy(),
        "flag_nan": _b(land.isna()),
        "flag_negative": _b(land < 0),
        "flag_land_exceeds_total": _b((land > total * 1.001) & total.notna()),
        "flag_impr_exceeds_total": _b((impr > total * 1.001) & total.notna()),
        "flag_summation": _b(resid.abs() > (0.01 * total.abs() + 1)),
        "flag_floored": _b(floored),
    })
    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    viol = flags[flag_cols].any(axis=1)
    pct_viol = 100.0 * float(viol.mean())
    return dict(
        test="A7 sanity", series=name, n=n, pct_violations=pct_viol,
        verdict=_verdict(pct_viol, 1.0, 5.0),
        detail=(f"violations {pct_viol:.2f}%  "
                f"(land>total {100*flags['flag_land_exceeds_total'].mean():.2f}%, "
                f"neg {100*flags['flag_negative'].mean():.2f}%, "
                f"nan {100*flags['flag_nan'].mean():.2f}%, "
                f"floored {100*flags['flag_floored'].mean():.2f}%)"),
    ), flags


def test_A8(name, u, impr_psf_field):
    impr_chd, n = _median_chd(u, "impr_he_id", impr_psf_field)
    # how cross-neighborhood are these clusters? avg distinct nbhds per impr_he_id
    g = u.groupby("impr_he_id")["neighborhood_filled"].nunique()
    avg_nbhds = float(g[g.index.notna()].mean())
    return dict(
        test="A8 impr loc-invariance", series=name,
        impr_chd=impr_chd, avg_nbhds_per_cluster=avg_nbhds,
        verdict=_verdict(impr_chd, 15, 25),
        detail=f"impr $/sqft CHD within impr_he_id = {impr_chd:.1f} "
               f"(clusters span avg {avg_nbhds:.1f} neighborhoods)",
    )


# ----------------------------------------------------------------------------- orchestrator

def run_battery(name, u, s, obs, nb, total_field="total_value", land_field="land_value",
                impr_field="impr_value"):
    """Run the MVP battery on one named land series. `u` carries the per-parcel land series
    plus features; `s` is the (hydrated, valid) sales for the total ratio study; `obs` is the
    land-observation table from lvi_anchors. Returns (results dict, per-parcel flag table)."""
    u = u.copy()
    u["_land_psf"] = pd.to_numeric(u[land_field], errors="coerce") / u["land_area_sqft"]
    u["_impr_psf"] = pd.to_numeric(u[impr_field], errors="coerce") / u["bldg_area_finished_sqft"]

    results = {}
    results["total"] = test_total(name, s, total_field)
    results["A1"] = test_A1(name, u, "_land_psf", nb)
    results["A2"] = test_A2(name, u, "_land_psf", "_impr_psf")
    results["A3"] = test_A3(name, u, obs, land_field)
    a7, flags = test_A7(name, u, land_field, impr_field, total_field)
    results["A7"] = a7
    results["A8"] = test_A8(name, u, "_impr_psf")
    return results, flags
