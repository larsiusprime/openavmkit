import warnings

import pandas as pd
import pytest

from openavmkit.cleaning import collapse_sparse_categories_sup
from openavmkit.data import SalesUniversePair
import openavmkit.utilities.settings as oavk_settings
from openavmkit.utilities.settings import (
	get_collapsed_fields,
	is_field_collapsed,
	get_location_fields,
	warn_if_location_collapsed,
)

_LOCATION_MSG = "used as a coherent geographic location"


def _make_sup(
	universe_roof_material: list,
	sales_keys: list,
	sales_roof_material: list = None,
) -> SalesUniversePair:
	"""Build a minimal SalesUniversePair with one categorical field.

	The universe holds one row per key. Sales reference a subset of those keys
	and may have their own ``roof_material`` overlay; when ``None`` the
	hydrated sales inherit the universe value.
	"""
	keys = [f"p{i}" for i in range(len(universe_roof_material))]
	df_univ = pd.DataFrame({
		"key": keys,
		"roof_material": universe_roof_material,
	})
	sales_data = {
		"key": sales_keys,
		"key_sale": [f"s{i}" for i in range(len(sales_keys))],
	}
	if sales_roof_material is not None:
		sales_data["roof_material"] = sales_roof_material
	df_sales = pd.DataFrame(sales_data)
	return SalesUniversePair(sales=df_sales, universe=df_univ)


def _settings(collapse_cfg: dict) -> dict:
	return {
		"field_classification": {
			"land": {"categorical": []},
			"impr": {"categorical": []},
			"other": {"categorical": ["roof_material"]},
		},
		"data": {
			"process": {
				"collapse_sparse_categories": collapse_cfg,
			}
		},
	}


def _settings_with_locations(collapse_cfg: dict, locations: list) -> dict:
	"""Same as ``_settings`` but registers ``locations`` as coherent locations."""
	s = _settings(collapse_cfg)
	s["field_classification"]["important"] = {"locations": locations}
	return s


def test_collapses_two_or_more_sparse_categories():
	# universe: 10 Chocolate, 8 Vanilla, 1 Giraffe, 1 Cinnamon (2 sparse)
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8 + ["Giraffe Sauce"] + ["Cinnamon Butt"]
	)
	# sales: 5 Chocolate, 3 Vanilla, 1 Giraffe (Cinnamon Butt only in universe)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 3, 4, 10, 11, 12, 18]]
	sales_overlay = (
		["Chocolate"] * 5 + ["Vanilla"] * 3 + ["Giraffe Sauce"]
	)
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {"sales_min": 2, "univ_min": 5}
	})

	out = collapse_sparse_categories_sup(sup, settings)

	assert set(out["universe"]["roof_material"].unique()) == {
		"Chocolate", "Vanilla", "Other"
	}
	assert (out["universe"]["roof_material"] == "Other").sum() == 2
	assert set(out["sales"]["roof_material"].unique()) == {
		"Chocolate", "Vanilla", "Other"
	}
	assert (out["sales"]["roof_material"] == "Other").sum() == 1


def test_skips_when_only_one_sparse_category():
	# Only Giraffe is sparse (1 row); others all pass the threshold.
	universe = ["Chocolate"] * 10 + ["Vanilla"] * 8 + ["Giraffe Sauce"]
	sales_keys = [f"p{i}" for i in [0, 1, 2, 10, 11, 18]]
	sales_overlay = ["Chocolate"] * 3 + ["Vanilla"] * 2 + ["Giraffe Sauce"]
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {"sales_min": 2, "univ_min": 5}
	})

	out = collapse_sparse_categories_sup(sup, settings)

	# Untouched: Giraffe still present, no "Other" introduced.
	assert "Giraffe Sauce" in set(out["universe"]["roof_material"].unique())
	assert "Other" not in set(out["universe"]["roof_material"].unique())


def test_missing_sales_min_raises():
	sup = _make_sup(["Chocolate"] * 5, ["p0"], ["Chocolate"])
	settings = _settings({
		"roof_material": {"univ_min": 5}
	})
	with pytest.raises(ValueError, match="sales_min"):
		collapse_sparse_categories_sup(sup, settings)


def test_field_not_in_field_classification_raises():
	sup = _make_sup(["Chocolate"] * 5, ["p0"], ["Chocolate"])
	settings = _settings({
		"some_unknown_field": {"sales_min": 2, "univ_min": 5}
	})
	with pytest.raises(ValueError, match="field_classification"):
		collapse_sparse_categories_sup(sup, settings)


def test_empty_config_is_noop():
	universe = ["Chocolate"] * 5 + ["Giraffe Sauce"]
	sup = _make_sup(universe, ["p0", "p5"], ["Chocolate", "Giraffe Sauce"])
	settings = _settings({})

	before_univ = sup["universe"]["roof_material"].tolist()
	before_sales = sup["sales"]["roof_material"].tolist()
	out = collapse_sparse_categories_sup(sup, settings)
	assert out["universe"]["roof_material"].tolist() == before_univ
	assert out["sales"]["roof_material"].tolist() == before_sales


def test_custom_replacement_value():
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8 + ["Giraffe Sauce"] + ["Cinnamon Butt"]
	)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 10, 11, 18]]
	sales_overlay = ["Chocolate"] * 3 + ["Vanilla"] * 2 + ["Giraffe Sauce"]
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {
			"sales_min": 2,
			"univ_min": 5,
			"replacement_value": "Catch-All",
		}
	})

	out = collapse_sparse_categories_sup(sup, settings)
	assert "Catch-All" in set(out["universe"]["roof_material"].unique())
	assert "Other" not in set(out["universe"]["roof_material"].unique())


def test_categorical_dtype_handled():
	# Same data but stored as pandas Categorical dtype.
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8 + ["Giraffe Sauce"] + ["Cinnamon Butt"]
	)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 10, 11, 18, 19]]
	sales_overlay = ["Chocolate"] * 3 + ["Vanilla"] * 2 + ["Giraffe Sauce", "Cinnamon Butt"]
	sup = _make_sup(universe, sales_keys, sales_overlay)
	sup.universe["roof_material"] = sup.universe["roof_material"].astype("category")
	sup.sales["roof_material"] = sup.sales["roof_material"].astype("category")

	settings = _settings({
		"roof_material": {"sales_min": 2, "univ_min": 5}
	})

	out = collapse_sparse_categories_sup(sup, settings)

	univ_cats = set(out["universe"]["roof_material"].cat.categories)
	assert univ_cats == {"Chocolate", "Vanilla", "Other"}
	assert (out["universe"]["roof_material"] == "Other").sum() == 2


def test_na_values_are_skipped_not_collapsed():
	# Regression: fields filled with None on a subset of rows (e.g. the
	# `none_vacant` fill for improvement characteristics) carry pd.NA. NA must
	# be skipped, not bucketed into "Other", and must not raise
	# "boolean value of NA is ambiguous" from the sorted() comparison.
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8
		+ ["Giraffe Sauce"] + ["Cinnamon Butt"]
		+ [None] * 6
	)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 3, 4, 10, 11, 12, 18]]
	sales_overlay = ["Chocolate"] * 5 + ["Vanilla"] * 3 + ["Giraffe Sauce"]
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {"sales_min": 2, "univ_min": 5}
	})

	out = collapse_sparse_categories_sup(sup, settings)

	# The two genuine sparse categories collapse; NA is left as-is (still NA),
	# never turned into "Other".
	uniq = set(out["universe"]["roof_material"].dropna().unique())
	assert uniq == {"Chocolate", "Vanilla", "Other"}
	assert (out["universe"]["roof_material"] == "Other").sum() == 2
	assert out["universe"]["roof_material"].isna().sum() == 6


def test_na_values_skipped_categorical_dtype():
	# Same regression for pandas Categorical dtype (NA stored as a missing code).
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8
		+ ["Giraffe Sauce"] + ["Cinnamon Butt"]
		+ [None] * 6
	)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 3, 4, 10, 11, 12, 18]]
	sales_overlay = ["Chocolate"] * 5 + ["Vanilla"] * 3 + ["Giraffe Sauce"]
	sup = _make_sup(universe, sales_keys, sales_overlay)
	sup.universe["roof_material"] = sup.universe["roof_material"].astype("category")
	sup.sales["roof_material"] = sup.sales["roof_material"].astype("category")

	settings = _settings({
		"roof_material": {"sales_min": 2, "univ_min": 5}
	})

	out = collapse_sparse_categories_sup(sup, settings)

	univ_cats = set(out["universe"]["roof_material"].cat.categories)
	assert univ_cats == {"Chocolate", "Vanilla", "Other"}
	assert out["universe"]["roof_material"].isna().sum() == 6


def test_sales_and_universe_share_mapping():
	# A category that meets univ_min but fails sales_min must still be collapsed
	# in BOTH dataframes (shared mapping). Plus a second sparse category so the
	# skip-if-single rule doesn't fire.
	#   - Chocolate: 10 univ, 5 sales (passes both)
	#   - Vanilla:   10 univ, 1 sales (passes univ_min=5, fails sales_min=3)
	#   - Giraffe:   2 univ, 0 sales (fails both)
	universe = ["Chocolate"] * 10 + ["Vanilla"] * 10 + ["Giraffe Sauce"] * 2
	sales_keys = [f"p{i}" for i in [0, 1, 2, 3, 4, 10]]
	sales_overlay = ["Chocolate"] * 5 + ["Vanilla"]
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {"sales_min": 3, "univ_min": 5}
	})

	out = collapse_sparse_categories_sup(sup, settings)

	# Both Vanilla and Giraffe should be folded into Other in universe,
	# and the same mapping must apply to sales.
	assert (out["universe"]["roof_material"] == "Other").sum() == 12
	assert "Vanilla" not in set(out["sales"]["roof_material"].unique())
	assert "Giraffe Sauce" not in set(out["sales"]["roof_material"].unique())
	assert (out["sales"]["roof_material"] == "Other").sum() == 1


# ---------------------------------------------------------------------------
# output_field: collapse into a modeling variant, leave the source intact
# ---------------------------------------------------------------------------

def test_output_field_creates_variant_and_leaves_source_intact():
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8 + ["Giraffe Sauce"] + ["Cinnamon Butt"]
	)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 3, 4, 10, 11, 12, 18]]
	sales_overlay = ["Chocolate"] * 5 + ["Vanilla"] * 3 + ["Giraffe Sauce"]
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {
			"sales_min": 2, "univ_min": 5, "output_field": "roof_material_collapsed"
		}
	})
	out = collapse_sparse_categories_sup(sup, settings)

	# Source field is untouched (still has the rare labels).
	assert set(out["universe"]["roof_material"].unique()) == {
		"Chocolate", "Vanilla", "Giraffe Sauce", "Cinnamon Butt"
	}
	# Variant carries the collapsed vocabulary.
	assert set(out["universe"]["roof_material_collapsed"].unique()) == {
		"Chocolate", "Vanilla", "Other"
	}
	assert (out["universe"]["roof_material_collapsed"] == "Other").sum() == 2
	# Variant exists on the sales frame too.
	assert "roof_material_collapsed" in out["sales"].columns


def test_output_field_variant_created_even_when_nothing_collapses():
	# No category is sparse (both clear univ_min=5) -> variant is an exact copy.
	universe = ["Chocolate"] * 10 + ["Vanilla"] * 8
	sales_keys = [f"p{i}" for i in [0, 1, 2, 10, 11]]
	sales_overlay = ["Chocolate"] * 3 + ["Vanilla"] * 2
	sup = _make_sup(universe, sales_keys, sales_overlay)

	settings = _settings({
		"roof_material": {
			"sales_min": 2, "univ_min": 5, "output_field": "roof_material_collapsed"
		}
	})
	out = collapse_sparse_categories_sup(sup, settings)

	assert "roof_material_collapsed" in out["universe"].columns
	assert (out["universe"]["roof_material_collapsed"] == "Other").sum() == 0
	assert (
		out["universe"]["roof_material_collapsed"].tolist()
		== out["universe"]["roof_material"].tolist()
	)


# ---------------------------------------------------------------------------
# Location footgun guard (upfront, at collapse time)
# ---------------------------------------------------------------------------

def _location_sup():
	universe = (
		["Chocolate"] * 10 + ["Vanilla"] * 8 + ["Giraffe Sauce"] + ["Cinnamon Butt"]
	)
	sales_keys = [f"p{i}" for i in [0, 1, 2, 3, 4, 10, 11, 12, 18]]
	sales_overlay = ["Chocolate"] * 5 + ["Vanilla"] * 3 + ["Giraffe Sauce"]
	return _make_sup(universe, sales_keys, sales_overlay)


def test_collapse_location_in_place_warns():
	sup = _location_sup()
	settings = _settings_with_locations(
		{"roof_material": {"sales_min": 2, "univ_min": 5}}, ["roof_material"]
	)
	with pytest.warns(UserWarning, match=_LOCATION_MSG):
		collapse_sparse_categories_sup(sup, settings)


def test_collapse_location_into_output_field_does_not_warn():
	sup = _location_sup()
	settings = _settings_with_locations(
		{"roof_material": {
			"sales_min": 2, "univ_min": 5, "output_field": "roof_material_collapsed"
		}},
		["roof_material"],
	)
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		out = collapse_sparse_categories_sup(sup, settings)
	assert not any(_LOCATION_MSG in str(w.message) for w in caught)
	# Raw location preserved; variant collapsed.
	assert set(out["universe"]["roof_material"].unique()) == {
		"Chocolate", "Vanilla", "Giraffe Sauce", "Cinnamon Butt"
	}
	assert "Other" in set(out["universe"]["roof_material_collapsed"].unique())


def test_collapse_location_in_place_strict_raises():
	sup = _location_sup()
	settings = _settings_with_locations(
		{"strict": True, "roof_material": {"sales_min": 2, "univ_min": 5}},
		["roof_material"],
	)
	with pytest.raises(ValueError, match=_LOCATION_MSG):
		collapse_sparse_categories_sup(sup, settings)


# ---------------------------------------------------------------------------
# Detection / registry helpers
# ---------------------------------------------------------------------------

def test_get_collapsed_fields_is_output_field_aware():
	s = {"data": {"process": {"collapse_sparse_categories": {
		"strict": True,
		"a": {"sales_min": 1, "univ_min": 1},
		"b": {"sales_min": 1, "univ_min": 1, "output_field": "b_collapsed"},
	}}}}
	assert get_collapsed_fields(s) == {"a", "b_collapsed"}
	assert is_field_collapsed(s, "a")
	assert is_field_collapsed(s, "b_collapsed")
	assert not is_field_collapsed(s, "b")  # source preserved, not collapsed
	assert not is_field_collapsed(s, "strict")  # reserved key, not a field


def test_get_location_fields_aggregates_registry_and_excludes_report():
	s = {
		"field_classification": {"important": {
			"locations": ["neighborhood"],
			"report_locations": ["ward"],  # benign -> excluded
			"fields": {"loc_market_area": "census_tract", "impr_category": "property_use"},
		}},
		"analysis": {
			"horizontal_equity": {"location": "he_loc"},
			"ratio_study": {"breakdowns": [
				{"by": "<loc_market_area>"},
				{"by": "sale_price", "quantiles": 10},  # not a location
			]},
		},
		"modeling": {
			"instructions": {"main": {"ensemble": {"locations": ["ens_loc"]}}},
			"models": {"main": {"local_area": {"locations": ["model_loc"]}}},
		},
		"land": {"lycd": {"default": {"location": "lycd_loc"}}},
	}
	got = get_location_fields(s)
	assert got == {
		"neighborhood", "census_tract", "he_loc", "ens_loc", "model_loc", "lycd_loc"
	}
	assert "ward" not in got          # report_locations excluded
	assert "sale_price" not in got    # numeric breakdown, not a location
	assert "property_use" not in got  # non-loc_* important field


def test_warn_if_location_collapsed_dedups_and_respects_strict():
	oavk_settings._LOCATION_COLLAPSE_WARNED.clear()
	s = {"data": {"process": {"collapse_sparse_categories": {
		"neighborhood": {"sales_min": 5, "univ_min": 25}
	}}}}
	# First call warns...
	with pytest.warns(UserWarning, match="geographic grouping key"):
		warn_if_location_collapsed(s, "neighborhood", context="ctx")
	# ...identical (field, context) call is deduped (no warning).
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		warn_if_location_collapsed(s, "neighborhood", context="ctx")
	assert not any("neighborhood" in str(w.message) for w in caught)
	# A non-collapsed field never warns.
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		warn_if_location_collapsed(s, "subdivision", context="ctx")
	assert len(caught) == 0
	# None (e.g. an undefined analysis.*.location) and lists with None are safe.
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		warn_if_location_collapsed(s, None, context="ctx")
		warn_if_location_collapsed(s, [None, "subdivision"], context="ctx")
	assert len(caught) == 0
	# Strict escalates to ValueError.
	oavk_settings._LOCATION_COLLAPSE_WARNED.clear()
	s_strict = {"data": {"process": {"collapse_sparse_categories": {
		"strict": True,
		"neighborhood": {"sales_min": 5, "univ_min": 25},
	}}}}
	with pytest.raises(ValueError, match="geographic grouping key"):
		warn_if_location_collapsed(s_strict, "neighborhood", context="ctx2")
