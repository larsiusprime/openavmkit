from openavmkit.utilities.settings import merge_settings
from openavmkit.utilities.assertions import dicts_are_equal

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