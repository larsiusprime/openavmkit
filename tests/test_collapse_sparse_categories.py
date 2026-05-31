import pandas as pd
import pytest

from openavmkit.cleaning import collapse_sparse_categories_sup
from openavmkit.data import SalesUniversePair


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
