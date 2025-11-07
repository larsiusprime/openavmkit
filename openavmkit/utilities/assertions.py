import numpy as np
import pandas as pd

from openavmkit.utilities.timing import TimingData

def objects_are_equal(a, b, epsilon: float = 1e-6) -> bool:
    """Test whether two objects are equal

    Checks strings, dicts, lists, ints, floats, and objects

    Parameters
    ----------
    a : Any
        A value of any type
    b : Any
        Another value of any type
    epsilon : float
        If the values are both floats, the maximum allowed tolerance

    Returns
    -------
    bool
        Whether the two objects are equal or not
    """
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


def lists_are_equal(a: list, b: list) -> bool:
    """Test whether two lists are equal

    Parameters
    ----------
    a : list
        A list
    b : list
        Another list

    Returns
    -------
    bool
        Whether the two lists are equal or not
    """
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


def dicts_are_equal(a: dict, b: dict) -> bool:
    """Test whether two dictionaries are equal

    Parameters
    ----------
    a : dict
        A dictionary
    b : dict
        Another dictionary

    Returns
    -------
    bool
        Whether the two dictionaries are equal or not
    """
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


def dfs_are_equal(a: pd.DataFrame, b: pd.DataFrame, primary_key=None, allow_weak=False) -> bool:
    """Test whether two DataFrames are equal

    Parameters
    ----------
    a : pd.DataFrame
        A DataFrame
    b : pd.DataFrame
        Another DataFrame
    primary_key : str
        The primary key for the first DataFrame
    allow_weak : bool
        Whether to ignore trivial differences (such as nominally different types for columns with otherwise identical values)

    Returns
    -------
    bool
        Whether the two DataFrames are equal or not
    """
    a = a.copy()
    b = b.copy()

    # If a primary key is provided, preserve original behavior: sort by PK
    if primary_key is not None:
        a = a.sort_values(by=primary_key)
        b = b.sort_values(by=primary_key)

    # Match original: sort column names so they're in the same order
    a = a.reindex(sorted(a.columns), axis=1)
    b = b.reindex(sorted(b.columns), axis=1)

    # Columns must match exactly
    if not a.columns.equals(b.columns):
        print(f"Columns do not match:\nA={a.columns}\nB={b.columns}")
        a_not_in_b = [col for col in a if col not in b]
        b_not_in_a = [col for col in b if col not in a]
        print(f"--> Cols in A not in B = {a_not_in_b}")
        print(f"--> Cols in B not in A = {b_not_in_a}")
        return False

    a_sorted_index = a.index.sort_values()
    b_sorted_index = b.index.sort_values()

    # Precompute primary-key–indexed views only once (may or may not be used later)
    a_indexed = None
    b_indexed = None

    if not a_sorted_index.equals(b_sorted_index):
        if primary_key is not None:
            # Same logic: report symmetric key diffs if any; otherwise prepare PK indexing
            a_not_in_b = a[~a[primary_key].isin(b[primary_key])][primary_key].values
            b_not_in_a = b[~b[primary_key].isin(a[primary_key])][primary_key].values
            if len(a_not_in_b) > 0:
                print(f"{len(a_not_in_b)} keys in A not in B: {a_not_in_b}")
                return False
            if len(b_not_in_a) > 0:
                print(f"{len(b_not_in_a)} keys in B not in A: {b_not_in_a}")
                print(f"len(a) = {len(a)} VS len(b) = {len(b)}")
                return False
            else:
                # Prepare PK-indexed views once
                a_indexed = a.set_index(primary_key)
                b_indexed = b.set_index(primary_key)
        else:
            print("Indices do not match")
            print(a_sorted_index)
            print("VS")
            print(b_sorted_index)
            return False
    else:
        # Indices already match; still prepare PK-indexed views up-front if we may need them later
        if primary_key is not None and primary_key in a.columns and primary_key in b.columns:
            a_indexed = a.set_index(primary_key)
            b_indexed = b.set_index(primary_key)

    # ===== Hot path: column-wise checks =====
    cols = a.columns

    # Iterate columns once; reuse precomputed PK-indexed views if needed.
    for col in cols:
        if col == primary_key:
            # Same logic: skip the primary key column
            continue

        # First attempt: direct comparison
        if not series_are_equal(a[col], b[col]):
            no_match = False

            # Second attempt (only if PK provided): use prebuilt PK-indexed Series
            if primary_key is not None:
                # a_indexed / b_indexed are guaranteed prepared above when PK is available
                # Compare the same column after aligning on PK
                if not series_are_equal(a_indexed[col], b_indexed[col]):
                    no_match = True
            else:
                no_match = True

            if no_match:
                # Keep the original debug/print behavior (including the "weak" case)
                with pd.option_context('display.max_columns', None):
                    # (compare a[col] to b[col] directly, using a's mask twice).
                    bad_rows_a = a[~a[col].eq(b[col])]
                    bad_rows_b = b[~a[col].eq(b[col])]

                    weak_fail = False
                    if len(bad_rows_a) == 0 and len(bad_rows_b) == 0:
                        weak_fail = True
                        if not allow_weak:
                            print(
                                f"Column '{col}' does not match even though rows are naively equal, look:"
                            )
                            print(a[col])
                            print(b[col])
                    else:
                        print(f"Column '{col}' does not match, look:")
                        # print rows that are not equal:
                        print(bad_rows_a[col])
                        print(bad_rows_b[col])

                if weak_fail and allow_weak:
                    # Preserve original behavior: tolerate weak mismatch when allowed
                    continue

                print(f"Column '{col}' does not match for some reason.")
                return False

    return True



def series_are_equal(a: pd.Series, b: pd.Series) -> bool:
    """Test whether two series are equal

    Parameters
    ----------
    a : pd.Series
        A series
    b : pd.Series
        Another series

    Returns
    -------
    bool
        Whether the two series are equal or not
    """
    # deal with type differences (pandas vs bodo.pandas, pyarrow vs numpy)
    
    # Convert both to numpy arrays for comparison to avoid dtype issues
    # Handle nullable integer types (Int64) more carefully
    try:
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        
        # Check if they're exactly equal first (only if no NA values)
        if not (pd.isna(a).any() or pd.isna(b).any()):
            if np.array_equal(a_np, b_np):
                return True
    except (TypeError, ValueError):
        # If numpy conversion fails (e.g., with nullable types), fall back to pandas comparison
        pass
    
    # If not exactly equal, check for float precision differences
    a_type = a.dtype
    b_type = b.dtype

    # Use proper pandas API instead of string parsing
    # Handle both float and double types (pandas vs bodo.pandas)
    a_is_float = pd.api.types.is_float_dtype(a_type) or "double" in str(a_type).lower()
    b_is_float = pd.api.types.is_float_dtype(b_type) or "double" in str(b_type).lower()

    a_is_int = pd.api.types.is_integer_dtype(a_type)
    b_is_int = pd.api.types.is_integer_dtype(b_type)

    if a_is_float and b_is_float:

        if a.isna().sum() != b.isna().sum():
            print(
                f"Number of NaN values do not match: a={a.isna().sum()} b={b.isna().sum()}"
            )
            return False

        a_fill_na = a.fillna(0)
        b_fill_na = b.fillna(0)

        # compare floats with epsilon:
        FLOAT_COMPARISON_TOLERANCE = 1e-6  # Relative tolerance for float comparisons
        result = a_fill_na.subtract(b_fill_na).abs().max() < FLOAT_COMPARISON_TOLERANCE
        if result == False:
            print(
                f"Comparing floats with epsilon:\n{a_fill_na.subtract(b_fill_na).abs().max()}"
            )
        return result

    if a_is_int and b_is_int:
        # Handle nullable integer types (Int64) more carefully
        try:
            # compare integers directly:
            result = a.subtract(b).abs().max() == 0
            if result == False:
                print(f"Comparing integers directly:\n{a.subtract(b).abs().max()}")
            return result
        except (TypeError, ValueError):
            # If subtraction fails (e.g., with nullable types), use pandas comparison
            return a.equals(b)

    # ensure that the two series contain the same information:
    if not a.equals(b):

        # Check for "NONE" values in either one and replace with NaN:
        a.loc[a.isna()] = np.nan
        b.loc[b.isna()] = np.nan

        # check which values are NaN:
        a_is_nan = pd.isna(a)  # Single, reliable method
        b_is_nan = pd.isna(b)  # Single, reliable method

        # mask out the NaN values and see if those sections are equal:
        a_masked = a[~a_is_nan]
        b_masked = b[~b_is_nan]

        if not a_masked.equals(b_masked):

            # if both are datetimes:
            if pd.api.types.is_datetime64_any_dtype(
                a_masked
            ) and pd.api.types.is_datetime64_any_dtype(b_masked):
                # compare datetimes directly:
                result = a_masked.subtract(b_masked).abs().max() == pd.Timedelta(0)
                if result == False:
                    print(
                        f"Comparing datetimes directly:\n{a_masked.subtract(b_masked).abs().max()}"
                    )
                return result
            else:
                # attempt to cast both as floats and compare:
                try:
                    a_masked = a_masked.astype(float)
                    b_masked = b_masked.astype(float)
                    delta = a_masked.subtract(b_masked).abs().max()
                    result = delta < 1e-6
                    if not result:
                        print(f"Masked values are not equal, max delta = {delta}")
                        return False
                    else:
                        return True
                except ValueError:
                    print("Masked values are not equal and cannot be cast to float")

                return False
        else:
            # if the masked values are equal, then the two series are equal except for the NaN values
            # check we have an equal number of NaN values:
            if a_is_nan.sum() != b_is_nan.sum():
                print(
                    f"Number of NaN values do not match: a={a_is_nan.sum()} b={b_is_nan.sum()}"
                )
                return False
            else:
                # now check if the NaN values are in the same places:
                if a_is_nan.equals(b_is_nan):
                    return True
                else:
                    print("NaN values are in different places")
                    return False

    return True
