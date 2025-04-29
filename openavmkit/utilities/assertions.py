import numpy as np
import pandas as pd


def objects_are_equal(a, b, epsilon:float = 1e-6):
	a_str = isinstance(a, str)
	b_str = isinstance(b, str)

	if a_str and b_str:
		return a == b

	a_dict = isinstance(a, dict)
	b_dict = isinstance(b, dict)

	if a_dict and b_dict:
		return dicts_are_equal(a, b)

	a_list = isinstance(a, list)
	b_list = isinstance(b, list)

	if a_list and b_list:
		return lists_are_equal(a, b)
	else:
		a_other = a_str or a_dict or a_list
		b_other = b_str or b_dict or b_list

		a_is_num = (not a_other) and (isinstance(a, (int, float)) or np.isreal(a))
		b_is_num = (not b_other) and (isinstance(b, (int, float)) or np.isreal(b))

		if a is None and b is None:
			return True
		elif a is None or b is None:
				return False

		if a_is_num and b_is_num:
			a_is_float = isinstance(a, float)
			b_is_float = isinstance(b, float)

			if a_is_float and b_is_float:
				a_is_nan = np.isnan(a)
				b_is_nan = np.isnan(b)

				if a_is_nan and b_is_nan:
					return True
				if a_is_nan or b_is_nan:
					return False

			# compare floats with epsilon:
			return abs(a - b) < epsilon

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
		if not objects_are_equal(entry_a, entry_b):
			return False
	return True


def dfs_are_equal(a: pd.DataFrame, b: pd.DataFrame, primary_key=None, allow_weak=False):
	if primary_key is not None:
		a = a.sort_values(by=primary_key)
		b = b.sort_values(by=primary_key)

	# sort column names so they're in the same order:
	a = a.reindex(sorted(a.columns), axis=1)
	b = b.reindex(sorted(b.columns), axis=1)

	# ensure that the two dataframes contain the same information:
	if not a.columns.equals(b.columns):
		print(f"Columns do not match:\nA={a.columns}\nB={b.columns}")
		return False
	if not a.index.equals(b.index):
		print("Indices do not match")
		print(a.index)
		print("VS")
		print(b.index)
		return False
	for col in a.columns:
		if not series_are_equal(a[col], b[col]):
			old_val = pd.get_option('display.max_columns')
			pd.set_option('display.max_columns', None)

			bad_rows_a = a[~a[col].eq(b[col])]
			bad_rows_b = b[~a[col].eq(b[col])]

			weak_fail = False

			if len(bad_rows_a) == 0 and len(bad_rows_b) == 0:
				weak_fail = True
				if not allow_weak:
					print(f"Column '{col}' does not match even though rows are naively equal, look:")
					print(a[col])
					print(b[col])
			else:
				print(f"Column '{col}' does not match, look:")
				# print rows that are not equal:
				print(bad_rows_a)
				print(bad_rows_b)

			pd.set_option('display.max_columns', old_val)

			if weak_fail and allow_weak:
				continue

			return False
	return True


def series_are_equal(a: pd.Series, b: pd.Series):

	# deal with 32-bit vs 64-bit type nonsense:

	a_type = a.dtype
	b_type = b.dtype

	a_is_float = "float" in str(a_type).lower()
	b_is_float = "float" in str(b_type).lower()

	a_is_int = "int" in str(a_type).lower()
	b_is_int = "int" in str(b_type).lower()

	if a_is_float and b_is_float:
		# compare floats with epsilon:
		result = a.subtract(b).abs().max() < 1e-6
		return result

	if a_is_int and b_is_int:
		# compare integers directly:
		result = a.subtract(b).abs().max() == 0
		return result

	# ensure that the two series contain the same information:
	if not a.equals(b):
		return False
	return True