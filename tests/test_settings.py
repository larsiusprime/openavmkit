from openavmkit.utilities.settings import _merge_settings, _remove_comments_from_settings, _lookup_variable_in_settings, \
	_replace_variables
from openavmkit.utilities.assertions import dicts_are_equal, objects_are_equal


def test_basic():
	print("")
	# test the following:
	# 1. merging lists
	# 2. merging dictionaries
	# 3. merging new keys
	#    a. lists
	#    b. dictionaries
	#    c. strings
	# 4. stomp rules:
	#    a. (blank) (local stomps template entry)
	#    b. + (template adds to local entry)
	#    c. ! (local stomps template entry even if template is set to +)



	template = {
		"version": "abc",
		"oranges": ["Navel", "Mandarin"],
		"apples": ["Macintosh", "Granny Smith", "Red Delicious"],
		"limes": ["Key", "Persian"],
		"pantry": {
			"wood": "pine",
			"spices": ["cinnamon", "nutmeg", "allspice"],
			"other": {
				"+baking": ["baking powder", "baking soda"],
			}
		},
		"+marbles": ["red", "blue", "green"],
	}
	local = {
		"version": "def",
		"apples": ["Fuji", "Honeycrisp", "Gala", "Cosmic Crisp"],
		"bananas": ["Gros Michel", "Cavendish", "Red", "Burro"],
		"!limes": ["Mexican"],
		"pantry": {
			"wood": "oak",
			"other": {
				"baking": ["flour", "sugar", "baking soda"],
				"cooking": ["salt", "pepper"]
			}
		}
	}

	merged = _merge_settings(template, local)

	expected = {
		"version": "def",
		"oranges": ["Navel", "Mandarin"],
		"apples": ["Fuji", "Honeycrisp", "Gala", "Cosmic Crisp"],
		"bananas": ["Gros Michel", "Cavendish", "Red", "Burro"],
		"limes": ["Mexican"],
		"pantry": {
			"wood": "oak",
			"spices": ["cinnamon", "nutmeg", "allspice"],
			"other": {
				"baking": ["baking powder", "baking soda", "flour", "sugar"],
				"cooking": ["salt", "pepper"]
			}
		},
		"marbles": ["red", "blue", "green"]
	}

	assert dicts_are_equal(merged, expected), f"Expected VS Result:\n{expected}\n{merged}"


def test_comments():

	provided = {
		"version": "def",
		"__schversion": "def",
		"apples": ["Macintosh", "Granny Smith", "Red Delicious", "Fuji", "Honeycrisp", "Gala", "Cosmic Crisp"],
		"__schnapples": ["Macintosh", "Granny Smith", "Red Delicious", "Fuji", "Honeycrisp", "Gala", "Cosmic Crisp"],
		"bananas": ["Gros Michel", "Cavendish", "Red", "Burro"],
		"__schbananas": ["Gros Michel", "Cavendish", "Red", "Burro"],
		"pantry": {
			"wood": "oak",
			"__schwood": "oak",
			"spices": ["cinnamon", "nutmeg", "allspice", "cardamon", "clove", "ginger"],
			"__schspices": ["cinnamon", "nutmeg", "allspice", "cardamon", "clove", "ginger"],
			"other": {
				"baking": ["baking powder", "baking soda", "flour", "sugar"],
				"__schbaking": ["baking powder", "baking soda", "flour", "sugar"],
				"cooking": ["salt", "pepper"],
				"__schcooking": ["salt", "pepper"]
			},
			"__schother": {
				"baking": ["baking powder", "baking soda", "flour", "sugar"],
				"__schbaking": ["baking powder", "baking soda", "flour", "sugar"],
				"cooking": ["salt", "pepper"],
				"__schcooking": ["salt", "pepper"]
			}
		},
		"__schpantry": {
			"wood": "oak",
			"__schwood": "oak",
			"spices": ["cinnamon", "nutmeg", "allspice", "cardamon", "clove", "ginger"],
			"__schspices": ["cinnamon", "nutmeg", "allspice", "cardamon", "clove", "ginger"],
			"other": {
				"baking": ["baking powder", "baking soda", "flour", "sugar"],
				"__schbaking": ["baking powder", "baking soda", "flour", "sugar"],
				"cooking": ["salt", "pepper"],
				"__schcooking": ["salt", "pepper"]
			},
			"__schother": {
				"baking": ["baking powder", "baking soda", "flour", "sugar"],
				"__schbaking": ["baking powder", "baking soda", "flour", "sugar"],
				"cooking": ["salt", "pepper"],
				"__schcooking": ["salt", "pepper"]
			}
		}
	}

	provided = _remove_comments_from_settings(provided)

	expected = {
		"version": "def",
		"apples": ["Macintosh", "Granny Smith", "Red Delicious", "Fuji", "Honeycrisp", "Gala", "Cosmic Crisp"],
		"bananas": ["Gros Michel", "Cavendish", "Red", "Burro"],
		"pantry": {
			"wood": "oak",
			"spices": ["cinnamon", "nutmeg", "allspice", "cardamon", "clove", "ginger"],
			"other": {
				"baking": ["baking powder", "baking soda", "flour", "sugar"],
				"cooking": ["salt", "pepper"]
			}
		}
	}

	assert dicts_are_equal(provided, expected), f"Expected VS Results:\n{expected}\n{provided}"


def test_lookup_variable_in_settings():
	data = {
		"earth": {
			"north_america": {
				"usa": {
					"texas": {
						"houston": {
							"greenspoint": {
								"haystack": "needle",
								"truck": {
									"bed": {
										"lunchbox": {
											"drink": "coke",
											"food": "sandwich",
											"dessert": "cookie"
										}
									}
								}
							}
						}
					}
				},
				"mexico": {
					"chihuahua": {
						"juarez": {
							"almiar": "aguja"
						}
					}
				}
			}
		},
		"mars": {
			"rover": {
				"cargo": ["rock", "dirt", "sand"]
			}
		}
	}

	a = _lookup_variable_in_settings(data, "earth.north_america.usa.texas.houston.greenspoint.haystack")
	b = _lookup_variable_in_settings(data, "earth.north_america.mexico.chihuahua.juarez.almiar")
	c = _lookup_variable_in_settings(data, "mars.rover.cargo")
	d = _lookup_variable_in_settings(data, "earth.north_america.usa.texas.houston.greenspoint.truck.bed.lunchbox")

	a_expected = "needle"
	a_unexpected = "haystack"
	b_expected = "aguja"
	b_unexpected = "needle"
	c_expected = ["rock", "dirt", "sand"]
	c_unexpected = ["rock", "dirt"]
	d_expected = {
		"drink": "coke",
		"food": "sandwich",
		"dessert": "cookie"
	}
	d_unexpected = {
		"drink": "coke",
		"food": "sandwich",
		"dessert": "cake"
	}

	assert a == a_expected, f"Expected VS Result:\n{a_expected}\n{a}"
	assert b == b_expected, f"Expected VS Result:\n{b_expected}\n{b}"
	assert c == c_expected, f"Expected VS Result:\n{c_expected}\n{c}"
	assert dicts_are_equal(d, d_expected), f"Expected VS Result:\n{d_expected}\n{d}"

	assert a != a_unexpected, f"Unexpected VS Result:\n{a_unexpected}\n{a}"
	assert b != b_unexpected, f"Unexpected VS Result:\n{b_unexpected}\n{b}"
	assert c != c_unexpected, f"Unexpected VS Result:\n{c_unexpected}\n{c}"
	assert False == dicts_are_equal(d, d_unexpected), f"Unexpected VS Result:\n{d_unexpected}\n{d}"


def test_replace_variables_in_settings():

	data = {
		"lunchboxes": [
			["$$def.coke", "$$def.sandwich", "$$def.cookie"],
			["$$def.water", "$$def.sandwich", "$$def.cake"]
		],
		"calorie_counts": {
			"coke": "$$def.coke.calories",
			"water": "$$def.water.calories",
			"sandwich": "$$def.sandwich.calories",
			"cookie": "$$def.cookie.calories",
			"cake": "$$def.cake.calories"
		},
		"types": {
			"coke": "$$def.coke.type",
			"water": "$$def.water.type",
			"sandwich": "$$def.sandwich.type",
			"cookie": "$$def.cookie.type",
			"cake": "$$def.cake.type"
		},
		"menu": "$$def.menu",
		"def": {
			"menu": ["coke", "water", "sandwich", "cookie", "cake"],
			"coke": {
				"name": "coke",
				"type": "drink",
				"calories": 140
			},
			"water": {
				"name": "water",
				"type": "drink",
				"calories": 0
			},
			"sandwich": {
				"name": "sandwich",
				"type": "food",
				"calories": 400
			},
			"cookie": {
				"name": "cookie",
				"type": "dessert",
				"calories": 200
			},
			"cake": {
				"name": "cake",
				"type": "dessert",
				"calories": 500
			}
		}
	}

	expected = {
		"lunchboxes": [
			[
				{
					"name": "coke",
					"type": "drink",
					"calories": 140
				},
				{
					"name": "sandwich",
					"type": "food",
					"calories": 400
				},
				{
					"name": "cookie",
					"type": "dessert",
					"calories": 200
				}
			],
			[
				{
					"name": "water",
					"type": "drink",
					"calories": 0
				},
				{
					"name": "sandwich",
					"type": "food",
					"calories": 400
				},
				{
					"name": "cake",
					"type": "dessert",
					"calories": 500
				}
			]
		],
		"calorie_counts": {
			"coke": 140,
			"water": 0,
			"sandwich": 400,
			"cookie": 200,
			"cake": 500
		},
		"types": {
			"coke": "drink",
			"water": "drink",
			"sandwich": "food",
			"cookie": "dessert",
			"cake": "dessert"
		},
		"menu": ["coke", "water", "sandwich", "cookie", "cake"],
		"def": {
			"menu": ["coke", "water", "sandwich", "cookie", "cake"],
			"coke": {
				"name": "coke",
				"type": "drink",
				"calories": 140
			},
			"water": {
				"name": "water",
				"type": "drink",
				"calories": 0
			},
			"sandwich": {
				"name": "sandwich",
				"type": "food",
				"calories": 400
			},
			"cookie": {
				"name": "cookie",
				"type": "dessert",
				"calories": 200
			},
			"cake": {
				"name": "cake",
				"type": "dessert",
				"calories": 500
			}
		}
	}

	replaced = _replace_variables(data)

	assert objects_are_equal(replaced, expected), f"Expected VS Result:\n{expected}\n{replaced}"
	assert not objects_are_equal(data, replaced), f"Unexpected VS Result:\n{data}\n{replaced}"


def test_recursive_replace_variables_in_settings():

	data = {
		"a": "$$b",
		"b": "$$c",
		"c": "$$d",
		"d": "$$e",
		"e": "hello",
		"a_deep": "$$b_deep.c.d.e",
		"b_deep": {
			"c": {
				"d": {
					"e": "$$a"
				}
			}
		}
	}

	expected = {
		"a": "hello",
		"b": "hello",
		"c": "hello",
		"d": "hello",
		"e": "hello",
		"a_deep": "hello",
		"b_deep": {
			"c": {
				"d": {
					"e": "hello"
				}
			}
		}
	}

	replaced = _replace_variables(data)

	assert objects_are_equal(replaced, expected), f"Expected VS Result:\n{expected}\n{replaced}"
	assert not objects_are_equal(data, replaced), f"Unexpected VS Result:\n{data}\n{replaced}"

def _md(use_sales_from):
	return {"modeling": {"metadata": {"use_sales_from": use_sales_from, "valuation_date": "2026-01-01"}}}


def test_use_sales_from_scalar_and_legacy():
	from openavmkit.utilities.settings import resolve_use_sales_from, use_sales_from_floor
	# scalar -> same cutoff both types; model_group ignored
	assert resolve_use_sales_from(_md(2020)) == (2020, 2020)
	assert resolve_use_sales_from(_md(2020), model_group="anything") == (2020, 2020)
	assert use_sales_from_floor(_md(2020)) == (2020, 2020)
	# legacy per-type dict
	leg = _md({"improved": 2023, "vacant": 2018})
	assert resolve_use_sales_from(leg) == (2023, 2018)
	assert use_sales_from_floor(leg) == (2023, 2018)
	# missing entirely
	assert resolve_use_sales_from({"modeling": {"metadata": {"valuation_date": "2026-01-01"}}}) == (None, None)


def test_use_sales_from_per_model_group():
	from openavmkit.utilities.settings import resolve_use_sales_from, use_sales_from_floor
	s = _md({"default": 2022, "by_model_group": {"sf_suburban": 2023, "commercial": 2016}})
	# listed groups use their own window; everything else (and None) uses default
	assert resolve_use_sales_from(s, model_group="sf_suburban") == (2023, 2023)
	assert resolve_use_sales_from(s, model_group="commercial") == (2016, 2016)
	assert resolve_use_sales_from(s, model_group="sf_urban") == (2022, 2022)
	assert resolve_use_sales_from(s) == (2022, 2022)
	# floor = widest (oldest) across all windows -> driven by commercial 2016
	assert use_sales_from_floor(s) == (2016, 2016)


def test_use_sales_from_per_group_entries_can_be_per_type():
	from openavmkit.utilities.settings import resolve_use_sales_from, use_sales_from_floor
	# a per-group entry may itself be an {improved, vacant} dict
	s = _md({"default": {"improved": 2022, "vacant": 2019},
	         "by_model_group": {"commercial": {"improved": 2016, "vacant": 2014}}})
	assert resolve_use_sales_from(s, model_group="commercial") == (2016, 2014)
	assert resolve_use_sales_from(s) == (2022, 2019)
	# floor takes the min per type independently
	assert use_sales_from_floor(s) == (2016, 2014)


def test_use_sales_from_floor_unbounded_when_default_missing():
	from openavmkit.utilities.settings import use_sales_from_floor, resolve_use_sales_from
	# only per-group overrides, no default -> unlisted groups are unbounded, so the
	# cleaning floor must be None (don't drop) to preserve their sales
	s = _md({"by_model_group": {"commercial": 2016}})
	assert use_sales_from_floor(s) == (None, None)
	assert resolve_use_sales_from(s, model_group="commercial") == (2016, 2016)
	assert resolve_use_sales_from(s, model_group="sf_urban") == (None, None)
