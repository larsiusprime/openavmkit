from openavmkit.synthetic.basic import generate_inflation_curve, generate_basic


def test_inflation_curve():
	print("")
	time_mult = generate_inflation_curve(
		start_year=2020,
		end_year=2024,
		annual_inflation_rate=0.05,
		annual_inflation_rate_stdev=0.01,
		seasonality_amplitude=0.20,
		monthly_noise=0.05,
		daily_noise=0.01
	)


def test_unit_argument_switches_to_metric():
	"""`unit="sqm"` must actually produce metric areas.

	The conversion was guarded by `if "unit" == "sqm"`, which compares two string
	literals and is therefore always False, so the generator only ever emitted
	imperial areas whatever the caller asked for.
	"""
	sd_ft = generate_basic(30, seed=7, unit="sqft")
	sd_m = generate_basic(30, seed=7, unit="sqm")

	assert "land_area_sqft" in sd_ft.df_universe.columns
	assert "land_area_sqm" in sd_m.df_universe.columns

	ratio = sd_ft.df_universe["land_area_sqft"].mean() / sd_m.df_universe["land_area_sqm"].mean()
	assert abs(ratio - 10.7639) < 0.1, f"sqft/sqm ratio was {ratio:.4f}, expected ~10.7639"

	bldg_ft = sd_ft.df_universe["bldg_area_finished_sqft"]
	bldg_m = sd_m.df_universe["bldg_area_finished_sqm"]
	bratio = bldg_ft[bldg_ft > 0].mean() / bldg_m[bldg_m > 0].mean()
	assert abs(bratio - 10.7639) < 0.1, f"bldg sqft/sqm ratio was {bratio:.4f}"


def test_buildings_actually_depreciate():
	"""Synthetic buildings must lose value with age and gain it with condition.

	Both depreciation terms were wrapped in `min(0.0, 1 - x)` where the inner
	expression always sits in [0, 1], so the clamp forced them to 0.0 and nothing
	ever depreciated.
	"""
	sd = generate_basic(60, seed=11, percent_vacant=0.0)
	u = sd.df_universe
	impr = u[u["bldg_area_finished_sqft"] > 0].copy()
	assert len(impr) > 100, "need a decent improved sample to correlate"

	impr["value_per_sqft"] = impr["bldg_value"] / impr["bldg_area_finished_sqft"]

	r_age = impr["value_per_sqft"].corr(impr["bldg_age_years"])
	r_cond = impr["value_per_sqft"].corr(impr["bldg_condition_num"])

	assert r_age < -0.05, f"older buildings should be worth less per sqft (r={r_age:+.4f})"
	assert r_cond > 0.05, f"better condition should be worth more per sqft (r={r_cond:+.4f})"
	assert impr["bldg_value"].gt(0).all(), "depreciation must not zero out building values"

