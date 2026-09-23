import pandas as pd
import pytest

from openavmkit.model_runner import _validate_ind_vars_across_frames as validate


def _frames():
	sales = pd.DataFrame({
		"bldg_area_finished_sqft": [1.0],
		"bldg_style_desc": pd.Categorical(["RANCH"]),
		"bldg_style_code": pd.Categorical(["R"]),
	})
	universe = pd.DataFrame({
		"bldg_area_finished_sqft": [1.0],
		"bldg_style_code": pd.Categorical(["R"]),
	})
	return sales, universe


def test_sales_only_field_is_rejected_and_named():
	# The real-world trap: a field loaded from a sales-only source trains fine and
	# then blows up at predict time inside LightGBM with no column named.
	sales, universe = _frames()
	with pytest.raises(ValueError) as exc:
		validate(
			["bldg_area_finished_sqft", "bldg_style_desc"],
			sales, universe, ["bldg_style_code"], "lgbm_x", "single_family",
		)
	msg = str(exc.value)
	assert "bldg_style_desc" in msg
	assert "present in sales, absent from universe" in msg
	assert "lgbm_x" in msg and "single_family" in msg
	# and it should point at the universe-side equivalent
	assert "bldg_style_code" in msg


def test_field_absent_from_both_frames_warns_instead_of_raising():
	# Symmetric absence is NOT a correctness hazard: DataSplit filters ind_vars down to the columns
	# each frame actually has, so the field is dropped from train, test, sales and universe alike and
	# the model stays coherent. Raising here would turn any unavailable optional enrichment into a
	# dead pipeline (census fields are absent whenever CENSUS_API_KEY is unset). Warn, and still
	# offer the typo suggestion, since a typo is the other common cause.
	sales, universe = _frames()
	with pytest.warns(UserWarning) as rec:
		validate(["bldg_aera_sqft"], sales, universe, [], "m", "g")
	msg = str(rec[0].message)
	assert "bldg_aera_sqft" in msg
	assert "NEITHER" in msg
	assert "bldg_area_finished_sqft" in msg  # the "did you mean" suggestion survives


def test_unavailable_optional_enrichment_does_not_break_the_run():
	# The Petersburg CI case: catboost lists census-derived ind_vars, census enrichment does not run
	# without a CENSUS_API_KEY, so the columns never exist. This must degrade, not crash.
	census = ["median_income", "total_population", "median_g_rent", "median_c_rent"]
	sales = pd.DataFrame({"bldg_area_finished_sqft": [1.0]})
	universe = pd.DataFrame({"bldg_area_finished_sqft": [1.0]})
	with pytest.warns(UserWarning, match="NEITHER"):
		validate(["bldg_area_finished_sqft"] + census, sales, universe, [], "catboost", "sfu")


def test_sales_only_field_still_raises_even_alongside_absent_ones():
	# Downgrading the symmetric case must not soften the asymmetric one: training on a feature you
	# cannot predict with is still fatal, and is still named.
	sales = pd.DataFrame({"x": [1.0], "seller_attorney": ["smith"]})
	universe = pd.DataFrame({"x": [1.0]})
	with pytest.warns(UserWarning):
		with pytest.raises(ValueError) as exc:
			validate(["x", "seller_attorney", "median_income"], sales, universe, [], "m", "g")
	msg = str(exc.value)
	assert "seller_attorney" in msg
	assert "median_income" not in msg  # that one warned, it is not part of the fatal list


def test_unclassified_categorical_dtype_mismatch_is_rejected():
	# category in one frame, plain object in the other, and not classified: this is
	# what makes LightGBM's categorical_feature counts disagree.
	sales = pd.DataFrame({"x": [1.0], "foo": pd.Categorical(["A"])})
	universe = pd.DataFrame({"x": [1.0], "foo": ["A"]})
	with pytest.raises(ValueError) as exc:
		validate(["x", "foo"], sales, universe, [], "lgbm_x", "single_family")
	msg = str(exc.value)
	assert "foo" in msg
	assert "field_classification" in msg


def test_classified_categorical_dtype_mismatch_is_allowed():
	# If the field IS classified, encode_categoricals_as_categories syncs the vocab
	# across splits, so a dtype difference here is not a problem.
	sales = pd.DataFrame({"c": pd.Categorical(["A"])})
	universe = pd.DataFrame({"c": ["A"]})
	validate(["c"], sales, universe, ["c"], "m", "g")


def test_consistent_frames_pass():
	sales = pd.DataFrame({"x": [1.0], "c": pd.Categorical(["A"])})
	universe = pd.DataFrame({"x": [1.0], "c": pd.Categorical(["A"])})
	validate(["x", "c"], sales, universe, ["c"], "m", "g")


def test_empty_ind_vars_pass():
	sales, universe = _frames()
	validate([], sales, universe, [], "m", "g")


def test_datasplit_synthesized_sales_fields_are_not_flagged_missing():
	# Regression: the Petersburg CI locality failed here. `sale_age_days` / `sale_year` are absent
	# from the RAW universe frame, but DataSplit synthesizes them (universe scored as "every parcel
	# sold on the valuation date"), so they are legitimate ind_vars. This validation runs BEFORE
	# DataSplit builds them, so it must consult the contract rather than the frame in front of it.
	sales = pd.DataFrame({
		"bldg_area_finished_sqft": [1.0],
		"sale_age_days": [365.0],
		"sale_year": [2024],
	})
	universe = pd.DataFrame({"bldg_area_finished_sqft": [1.0]})
	validate(
		["bldg_area_finished_sqft", "sale_age_days", "sale_year"],
		sales, universe, [], "mra", "single_family_urban",
	)


def test_genuinely_missing_field_still_rejected_alongside_synthesized_ones():
	# The exemption must not become a blanket amnesty: a real typo/sales-only field is still caught
	# even when synthesized fields are present in the same list.
	sales = pd.DataFrame({"x": [1.0], "sale_year": [2024], "seller_attorney": ["smith"]})
	universe = pd.DataFrame({"x": [1.0]})
	with pytest.raises(ValueError) as exc:
		validate(["x", "sale_year", "seller_attorney"], sales, universe, [], "mra", "g")
	msg = str(exc.value)
	assert "seller_attorney" in msg
	assert "sale_year" not in msg


def test_synthesized_field_set_matches_what_datasplit_actually_builds():
	# Anti-drift: if someone adds a field to DataSplit's universe synthesis (or removes one) without
	# updating the exported set, the validator silently starts rejecting or accepting the wrong
	# things. Assert the contract against a real DataSplit rather than trusting the constant.
	import numpy as np
	from openavmkit.modeling import DataSplit, UNIVERSE_SYNTHESIZED_FIELDS

	nn = 20
	keys = [str(i) for i in range(nn)]
	df = pd.DataFrame({
		"key": keys, "key_sale": keys,
		"bldg_area_finished_sqft": np.linspace(800, 4000, nn),
		"land_area_sqft": np.linspace(3000, 20000, nn),
		"model_group": ["a"] * nn,
	})
	df["sale_price"] = 100000.0
	df["valid_sale"] = True; df["vacant_sale"] = False; df["is_vacant"] = False
	df["valid_for_ratio_study"] = True
	df["sale_date"] = pd.to_datetime("2025-01-01"); df["sale_age_days"] = 0
	ind_vars = ["bldg_area_finished_sqft", "land_area_sqft"]
	# The raw universe deliberately carries NONE of the synthesized fields.
	df_universe = df[["key", "is_vacant"] + ind_vars].copy()

	ds = DataSplit("", df, df_universe, "a", {}, "sale_price", "sale_price",
	               ind_vars, [], {}, keys[:5], keys[5:])

	# Equality, not containment, so drift is caught in BOTH directions: a field the constant claims
	# but DataSplit no longer builds (validator wrongly accepts it), and a field DataSplit starts
	# building that the constant does not list (validator wrongly rejects it — the original bug).
	added = set(ds.df_universe.columns) - set(df_universe.columns)
	assert added == set(UNIVERSE_SYNTHESIZED_FIELDS), (
		"DataSplit's universe synthesis and UNIVERSE_SYNTHESIZED_FIELDS have drifted apart.\n"
		f"  built but not declared: {sorted(added - UNIVERSE_SYNTHESIZED_FIELDS)}\n"
		f"  declared but not built: {sorted(UNIVERSE_SYNTHESIZED_FIELDS - added)}"
	)
