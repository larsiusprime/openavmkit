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


def test_field_absent_from_both_frames_is_reported_differently():
	sales, universe = _frames()
	with pytest.raises(ValueError) as exc:
		validate(["bldg_aera_sqft"], sales, universe, [], "m", "g")
	assert "absent from BOTH sales and universe" in str(exc.value)


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
