from openavmkit.utilities.settings import merge_settings, remove_comments_from_settings, lookup_variable_in_settings, \
	replace_variables
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

	template = {
		"version": "abc",
		"apples": ["Macintosh", "Granny Smith", "Red Delicious"],
		"pantry": {
			"wood": "pine",
			"spices": ["cinnamon", "nutmeg", "allspice"],
			"other": {
				"baking": ["baking powder", "baking soda"],
			}
		}
	}
	local = {
		"version": "def",
		"apples": ["Fuji", "Honeycrisp", "Gala", "Cosmic Crisp"],
		"bananas": ["Gros Michel", "Cavendish", "Red", "Burro"],
		"pantry": {
			"wood": "oak",
			"spices": ["cinnamon", "cardamon", "clove", "ginger"],
			"other": {
				"baking": ["flour", "sugar", "baking soda"],
				"cooking": ["salt", "pepper"]
			}
		}
	}

	merged = merge_settings(template, local)

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

	provided = remove_comments_from_settings(provided)

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

	a = lookup_variable_in_settings(data, "earth.north_america.usa.texas.houston.greenspoint.haystack")
	b = lookup_variable_in_settings(data, "earth.north_america.mexico.chihuahua.juarez.almiar")
	c = lookup_variable_in_settings(data, "mars.rover.cargo")
	d = lookup_variable_in_settings(data, "earth.north_america.usa.texas.houston.greenspoint.truck.bed.lunchbox")

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

	replaced = replace_variables(data)

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

	replaced = replace_variables(data)

	assert objects_are_equal(replaced, expected), f"Expected VS Result:\n{expected}\n{replaced}"
	assert not objects_are_equal(data, replaced), f"Unexpected VS Result:\n{data}\n{replaced}"