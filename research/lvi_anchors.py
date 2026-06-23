"""Ground-truth land observations for the land-value integrity battery (A3).

This module does NOT value land. It extracts *observed* land values from the market —
the concrete evidence the battery tests an estimate against:

  - vacant land sales      : the whole (time-adjusted) sale price IS land.
  - teardown sales         : a building sold to be demolished -> price is land.
  - new-construction sales : a fresh build has ~no depreciation, so
                             land = sale_price - replacement_cost(building).

For Wake the assessor publishes a cost-book improvement value (`assr_impr_value`,
i.e. RCN-less-depreciation) and a percent-good (`bldg_condition_pct`), so we can
reconstruct replacement cost new (RCN) directly:  RCN = assr_impr_value / pct_good.
For brand-new builds depreciation ~ 0, so assr_impr_value ~ RCN and the
new-construction land is simply price - assr_impr_value.

Modeled on the anchor logic in research/mb_config.py:prepare, but self-contained
(no market-basket dependency) and parameterised by column names.
"""
from types import SimpleNamespace
import re
import numpy as np
import pandas as pd

DEP = "sale_price_time_adj"

# prime-lot thresholds (a "prime" lot is a genuinely buildable, locally-comparable reflection
# of land value — see research/land_value_integrity_spec.md). Tunable.
RECT_MIN = 0.40           # geom_rectangularity_num below this = weird/irregular shape
MIN_PEERS = 10            # a neighborhood needs this many BUILT peers to define a size band
SIZE_LO, SIZE_HI = 0.05, 0.95   # lot must fall within built-peer size percentiles

RESIDENTIAL_GROUPS = ("single_family", "multifamily", "apartment", "UNKNOWN")

# Wake Disq_and_Qual deed codes: A/C = qualified arm's-length; E(family) F(fractional)
# D(non-warranty) G(life-estate) T/L(other) = disqualified. The land GOLD standard uses a
# WHITELIST (must be explicitly A/C) — NON-CIRCULAR: a deed code, independent of price/assessment.
QUAL_CODES = ("A", "C")


def qualified_sale_mask(s: pd.DataFrame, field: str = "disq_flag",
                        codes=QUAL_CODES) -> pd.Series:
    """Boolean: sale carries an explicit qualified deed code (whitelist). Unstamped (NaN) and
    disqualified codes both fail — for the small, high-stakes land gold standard we require
    positive confirmation, not absence of a disqualifier."""
    if field not in s.columns:
        return pd.Series(False, index=s.index)
    return s[field].astype(str).str.strip().str.upper().isin(codes)


def classify_zoning(code) -> str:
    """Bucket a Wake/Raleigh zoning code into residential / rural / nonres / other.

    Wake mixes old county codes (R-40W, RA, RR) with Raleigh UDO codes (R-4, RX, UR, SR, GR,
    NMX, CX, IX...). Checks are ordered: rural and non-residential first (so e.g. 'NMX' and
    'NC' don't get caught by a loose residential rule), then the residential families. The
    size/shape/peer filters are the real workhorses, so an approximate code map is fine.
    """
    z = str(code).upper().strip()
    # rural / agricultural — restrictive, large-lot, not "prime" comparable urban land
    if re.match(r"^(RA|RR)(\b|[\d-])", z):
        return "rural"
    # non-residential: industrial / office / commercial / mixed-use / downtown / neighborhood-commercial
    if re.match(r"^(IX|LI|HI|IND|OX|OI|ORD|CX|CC|CM|CMX|GC|GCP|GCM|HB|HC|NMX|NMU|NX|NC|NAC|UMX|DT|TCR)", z):
        return "nonres"
    # residential families (incl. watershed R-*W, planned units PUD/PD, conditional-use CU-R)
    if re.match(r"^(R[\d-]|R&|R/|RL|RM|RX|RHD|RT|GR\d|SR|UR|MDR?|LDR?|HDR?|NR|TR|TND|MF|MH|PUD|PD|CU-R)", z):
        return "residential"
    return "other"


def neighborhood_size_bands(u: pd.DataFrame, nb: str, model_group: str) -> pd.DataFrame:
    """Per-neighborhood built-lot size band for one model group (the comparability yardstick).

    Built = parcels of this model group that are not vacant and carry finished area. Returns
    columns [nb, peer_n, peer_lo, peer_hi, peer_med]."""
    b = u[(u["model_group"] == model_group) & (u["is_vacant"] == False) &
          (pd.to_numeric(u["bldg_area_finished_sqft"], errors="coerce") > 0)].copy()
    b["la"] = pd.to_numeric(b["land_area_sqft"], errors="coerce")
    return b.groupby(nb)["la"].agg(
        peer_n="count",
        peer_lo=lambda x: x.quantile(SIZE_LO),
        peer_hi=lambda x: x.quantile(SIZE_HI),
        peer_med="median").reset_index()


def add_prime_flags(v: pd.DataFrame, bands: pd.DataFrame, nb: str,
                    residential_zoning: bool = True) -> pd.DataFrame:
    """Attach disqualification flags + a single `prime` boolean to a vacant-sale frame `v`.

    `bands` are this model group's neighborhood size bands (from neighborhood_size_bands).
    `residential_zoning=True` requires residential zoning (for residential model groups);
    pass False for commercial/agricultural groups where that allowlist doesn't apply."""
    v = v.merge(bands, on=nb, how="left")
    la = pd.to_numeric(v["land_area_sqft"], errors="coerce")
    rect = pd.to_numeric(v["geom_rectangularity_num"], errors="coerce")
    bldg = pd.to_numeric(v.get("bldg_area_finished_sqft", 0), errors="coerce").fillna(0)
    zclass = v["zoning"].map(classify_zoning) if "zoning" in v.columns else pd.Series("other", index=v.index)

    v["dq_teardown"] = (v.get("is_vacant", True) == False) | (bldg > 0)     # a structure is/was present
    v["dq_nonres"] = (zclass != "residential") if residential_zoning else False
    v["dq_no_peers"] = v["peer_n"].fillna(0) < MIN_PEERS
    v["dq_size"] = ~la.between(v["peer_lo"], v["peer_hi"]) & v["peer_n"].notna()
    v["dq_shape"] = rect < RECT_MIN
    flags = ["dq_teardown", "dq_nonres", "dq_no_peers", "dq_size", "dq_shape"]
    v["prime"] = ~v[flags].any(axis=1)
    return v


def prime_lot_mask(u: pd.DataFrame, nb: str, model_group: str,
                   residential_zoning: bool = True) -> pd.Series:
    """Per-parcel `prime` boolean over the universe for one model group: genuinely buildable,
    locally-comparable lots (see add_prime_flags). Returned as a Series indexed by `key`, for
    mapping onto the A3 anchor observations."""
    bands = neighborhood_size_bands(u, nb, model_group)
    uu = u.drop_duplicates("key").copy()
    flagged = add_prime_flags(uu, bands, nb, residential_zoning=residential_zoning)
    return flagged.set_index("key")["prime"]


def reconstruct_rcn(df: pd.DataFrame,
                    impr_field: str = "assr_impr_value",
                    pctgood_field: str = "bldg_condition_pct") -> pd.Series:
    """Replacement-cost-new per parcel, straight from the assessor cost book.

    RCN = RCNLD / percent_good = assr_impr_value / (bldg_condition_pct / 100).
    Returns a float Series aligned to `df`; NaN where percent_good or impr is non-positive
    (vacant land, missing cost data). Percent-good is clipped to [0.10, 1.20] before dividing
    so a near-zero or wild over-100 factor can't manufacture an absurd RCN.
    """
    pct = pd.to_numeric(df[pctgood_field], errors="coerce") / 100.0
    impr = pd.to_numeric(df[impr_field], errors="coerce")
    pct = pct.where((pct >= 0.10) & (pct <= 1.20))
    rcn = impr.where(impr > 0) / pct
    return rcn


def build_land_observations(s: pd.DataFrame, ucols: pd.DataFrame, nb: str,
                            dep: str = DEP,
                            impr_field: str = "assr_impr_value",
                            teardown_frac: float = 0.50) -> pd.DataFrame:
    """Return one tidy table of observed land values, across validated evidence streams.

    Columns: key, kind {vacant, teardown, rcn_resid, rcnld_resid}, observed_land, land_sqft,
             observed_land_sqft, nbhd, qualified. One row per parcel per kind (deduped on key).

    Streams (see research/land_value_integrity_spec.md §11-13):
      vacant       direct land (whole price); purest, fully non-circular.
      teardown     sold-with-building, new build shortly after, price << neighborhood developed
                   level (priced like land). ~0 in Wake — the naive age<0 rule caught new homes.
      rcn_resid    improved, depreciation~0 (percent_good>=0.95): price - assr_impr (= price-RCN).
      rcnld_resid  improved, low/moderate dep: price - assr_impr (= price-RCNLD); SEMI-CIRCULAR
                   for the assessor's own series (shares assr_impr) — reserve for external series.

    `ucols` is the universe indexed by `key` (land_area_sqft, nbhd, assr_impr_value,
    bldg_condition_pct, bldg_area_finished_sqft). `teardown_frac` = price/land-sqft ceiling
    (as a fraction of the neighborhood developed $/land-sqft) to count as a teardown.
    """
    s = s.copy()
    s["key"] = s["key"].astype(str)
    s["sale_yr"] = pd.to_datetime(s["sale_date"], errors="coerce").dt.year
    s["age_at_sale"] = s["sale_yr"] - pd.to_numeric(s["bldg_year_built"], errors="coerce")

    land_sqft = s["key"].map(ucols["land_area_sqft"])
    nbhd = s["key"].map(ucols[nb])
    impr = pd.to_numeric(s["key"].map(ucols[impr_field]), errors="coerce")
    pg = pd.to_numeric(s["key"].map(ucols["bldg_condition_pct"]), errors="coerce") / 100.0  # percent-good
    bldg_area = pd.to_numeric(s["key"].map(ucols["bldg_area_finished_sqft"]), errors="coerce").fillna(0)
    price = pd.to_numeric(s[dep], errors="coerce")
    price = price.where(price > 0, pd.to_numeric(s["sale_price"], errors="coerce"))  # coalesce raw
    price_psf = price / land_sqft
    qualified = qualified_sale_mask(s)   # explicit A/C deed code (non-circular)
    vacant = s.get("vacant_sale", pd.Series(False, index=s.index)) == True

    # neighborhood DEVELOPED benchmark: median price/land-sqft of improved sales (finished level),
    # used to identify genuine teardowns (price priced like land, far below the developed level).
    imp_mask = (~vacant) & (bldg_area > 0) & (price > 0) & (land_sqft > 0)
    dev_psf = pd.DataFrame({"nb": nbhd, "psf": price_psf})[imp_mask].groupby("nb")["psf"].median()
    nb_dev = nbhd.map(dev_psf)

    frames = []

    def _pack(mask, observed_land, kind):
        d = pd.DataFrame({
            "key": s["key"], "kind": kind,
            "observed_land": observed_land, "land_sqft": land_sqft, "nbhd": nbhd,
            "qualified": qualified.values,
        })[mask]
        d = d[d["observed_land"].notna() & (d["observed_land"] > 0) & (d["land_sqft"] > 0)]
        return d.drop_duplicates("key")

    valid_land = s.get("valid_for_land_ratio_study", pd.Series(True, index=s.index)) == True
    # ---- VACANT: whole price is land (direct, purest evidence) ----
    frames.append(_pack((vacant & valid_land & (price > 2000)), price, "vacant"))

    # ---- TEARDOWN (proper def): sold WITH a building, a new building shortly after
    # (age_at_sale<0), and price/land-sqft SUBSTANTIALLY below the neighborhood developed level
    # (priced like land, old structure ~worthless). NOTE the naive age<0 test alone catches
    # finished new-home sales — the price screen is what makes this a real teardown. ----
    teardown = ((~vacant) & (s["age_at_sale"] < 0) & nb_dev.notna()
                & (price_psf < teardown_frac * nb_dev))
    frames.append(_pack(teardown, price, "teardown"))

    # ---- RCN residual (depreciation ~ 0, percent_good >= 0.95): RCNLD == RCN, so
    # observed_land = price - assr_impr ( = price - RCN ); the cleanest residual. Captures recent
    # new builds, incl. the finished new homes the old teardown rule mislabeled. ----
    rcn = (~vacant) & (bldg_area > 0) & (impr > 0) & (pg >= 0.95) & ~teardown
    frames.append(_pack(rcn, price - impr, "rcn_resid"))

    # ---- RCNLD residual (low/moderate depreciation, 0.75 <= percent_good < 0.95):
    # observed_land = price - assr_impr ( = price - RCNLD ). Semi-circular for the assessor's
    # OWN series (shares assr_impr) -> use for an EXTERNAL land series, not the assessor verdict. ----
    rcnld = (~vacant) & (bldg_area > 0) & (impr > 0) & (pg >= 0.75) & (pg < 0.95) & ~teardown
    frames.append(_pack(rcnld, price - impr, "rcnld_resid"))

    obs = pd.concat(frames, ignore_index=True)
    obs["observed_land_sqft"] = obs["observed_land"] / obs["land_sqft"]
    return obs


def summarize(obs: pd.DataFrame) -> str:
    counts = obs.groupby("kind").size().to_dict()
    total = len(obs)
    return (f"land observations: total={total}  " +
            "  ".join(f"{k}={counts.get(k, 0)}"
                      for k in ("vacant", "teardown", "rcn_resid", "rcnld_resid")))
