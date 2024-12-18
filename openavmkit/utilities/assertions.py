def objects_are_equal(a, b):
	# ensure that the two objects contain the same information:
	if isinstance(a, dict) and isinstance(b, dict):
		return dicts_are_equal(a, b)
	elif isinstance(a, list) and isinstance(b, list):
		return lists_are_equal(a, b)
	else:
		# ensure types are the same:
		if type(a) != type(b):
			return False
		return a == b


def lists_are_equal(a: list, b: list):
	# ensure that the two lists contain the same information:
	result = True
	if len(a) != len(b):
		result = False
	else:
		for i in range(len(a)):
			entry_a = a[i]
			entry_b = b[i]
			result = objects_are_equal(entry_a, entry_b)
	if not result:
		# print both lists for debugging:
		print(a)
		print(b)
		return False
	return True


def dicts_are_equal(a: dict, b: dict):
	# ensure that the two dictionaries contain the same information:
	if len(a) != len(b):
		return False
	for key in a:
		if key not in b:
			return False
		entry_a = a[key]
		entry_b = b[key]
		return objects_are_equal(entry_a, entry_b)