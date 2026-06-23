"""Phase-1 verification for land_value_integrity_spec.md (read-only).

Maps each test's required columns to the cleaned Guilford(_subset) data and reports
anchor counts. No models are trained; nothing is written.

Run: python research/lvi_verify_guilford.py [guilford_subset|guilford]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_config

JURIS = sys.argv[1] if len(sys.argv) > 1 else "guilford_subset"

# spec-required columns grouped by test (universe + sales pooled for the check)
REQUIRED = {
    "A0 unit/frontage/depth": ["land_area_sqft", "land_area_somers_ft", "frontage_ft_1",
                                "depth_ft_1", "zoning"],
    "A1 improvement-indep":   ["bldg_area_finished_sqft", "bldg_age_years", "bldg_quality_num",
                                "bldg_condition_num", "land_he_id", "spatial_lag_sale_price",
                                "spatial_lag_sale_price_time_adj"],
    "A2 horiz land equity":   ["land_he_id"],
    "A3 ratio study / VE":    ["vacant_sale", "valid_sale", "valid_for_ratio_study",
                                "sale_price", "sale_price_time_adj", "bldg_value_replacement"],
    "A4 tax incidence":       ["assr_market_value", "assr_land_value", "assr_impr_value",
                                "market_value", "is_vacant"],
    "A5 desirability grad":   ["spatial_lag_sale_price", "neighborhood_filled", "impr_he_id"],
    "A6 sales chasing":       ["he_id", "land_he_id", "sale_age_days"],
    "B2 developability":      ["slope_mean_deg", "elevation_mean_ft", "elevation_mean_m",
                                "rectangularity", "floodplain"],
    "B3 local uniformity":    ["latitude", "longitude", "latitude_norm", "longitude_norm"],
    "B4 expected spikes":     ["dist_to_water", "is_corner", "corner"],
}


def main():
    p = mb_config.prepare(JURIS)
    u, s = p.u, p.s
    cols = set(u.columns) | set(s.columns)
    print(f"\n=== {JURIS}: universe={u.shape}, sales={s.shape} (model_group={p.cfg['mg_list']}) ===")
    print(f"anchors -> vacant={len(p.vac)}  teardown={len(p.td)}  "
          f"new_constr={0 if p.nc is None else len(p.nc)}\n")

    for test, need in REQUIRED.items():
        present = [c for c in need if c in cols]
        missing = [c for c in need if c not in cols]
        print(f"[{test}]")
        print(f"   present: {present}")
        print(f"   MISSING: {missing}\n")

    # what assessor value columns DO exist (for A4 comparison series + total_value)?
    val_like = sorted(c for c in cols if any(k in c.lower() for k in
                      ("assr", "market_value", "land_value", "impr_value", "_value")))
    print("value-like columns present:", val_like)
    # HE cluster + spatial lag families actually present
    print("he_id family:", sorted(c for c in cols if "he_id" in c))
    print("spatial_lag family:", sorted(c for c in cols if c.startswith("spatial_lag"))[:12])
    print("frontage/depth/somers:", sorted(c for c in cols if any(
        k in c for k in ("frontage", "depth", "somers"))))
    print("DEM family:", sorted(c for c in cols if any(
        k in c for k in ("slope", "elevation"))))


if __name__ == "__main__":
    main()
