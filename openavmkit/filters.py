import pandas as pd
from openavmkit.utilities.data import is_column_of_type

def select_filter(df: pd.DataFrame, f: list) -> pd.DataFrame:
    """
    Select a subset of the DataFrame based on a list of filters.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    f : list
        Filter expressed as a list.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    resolved_index = resolve_filter(df, f)
    return df.loc[resolved_index]


def resolve_not_filter(df: pd.DataFrame, f: list) -> pd.Series:
    """
    Resolve a NOT filter.

    The first element of the filter list must be "not", followed by a filter list.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    f : list
        Filter list.

    Returns
    -------
    pandas.Series
        Boolean Series resulting from applying the NOT operator.
    """
    if len(f) < 2:
        raise ValueError("NOT operator requires at least one argument")

    values = f[1:]
    if len(values) > 1:
        raise ValueError(f"NOT operator only accepts one argument")

    selected_index = resolve_filter(df, values[0])
    return ~selected_index


def resolve_bool_filter(df: pd.DataFrame, f: list) -> pd.Series:
    """
    Resolve a list of filters using a boolean operator.

    Iterates through each filter in the list (after the operator) and combines their
    boolean indices using the specified boolean operator ("and", "or", "nand", "nor",
    "xor", "xnor").

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    f : list
        List where the first element is the boolean operator and the remaining elements
        are filter objects.

    Returns
    -------
    pandas.Series
        Boolean Series resulting from applying the boolean operator.
    """

    operator = f[0]
    values = f[1:]

    final_index = None

    for v in values:
        selected_index = resolve_filter(df, v)

        if final_index is None:
            final_index = selected_index
            continue

        if operator == "and":
            final_index = final_index & selected_index
        elif operator == "nand":
            final_index = ~(final_index & selected_index)
        elif operator == "or":
            final_index = final_index | selected_index
        elif operator == "nor":
            final_index = ~(final_index | selected_index)
        elif operator == "xor":
            final_index = final_index ^ selected_index
        elif operator == "xnor":
            final_index = ~(final_index ^ selected_index)

    return final_index


def resolve_filter(df: pd.DataFrame, f: list, rename_map: dict = None) -> pd.Series:
    """
    Resolve a filter list into a boolean Series for the DataFrame (which can be used for selection).

    For basic operators, the filter list must contain an operator, a field, and an
    optional value. For boolean operators, the filter list must contain a boolean
    operator, followed by a list of filters.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    f : list
        Filter list.
    rename_map : dict, optional
        Optional mapping of original to renamed columns.

    Returns
    -------
    pandas.Series
        Boolean Series corresponding to the filter.

    Raises
    ------
    ValueError
        If the operator is unknown.
    """

    if len(f) == 0:
        return pd.Series(False, index=df.index)

    operator = f[0]

    # check if operator is a boolean operator:
    if operator == "not":
        return resolve_not_filter(df, f)
    elif _is_bool_operator(operator):
        return resolve_bool_filter(df, f)
    else:
        field = f[1]
        # Handle field name resolution with rename_map
        if rename_map:
            # Create reverse map for looking up original names
            reverse_map = {v: k for k, v in rename_map.items()}
            if field in reverse_map and reverse_map[field] in df:
                field = reverse_map[field]
            elif field in rename_map and rename_map[field] in df:
                field = rename_map[field]

        if len(f) == 3:
            value = f[2]
        else:
            value = None

        if isinstance(value, str):
            if value.startswith("str:"):
                value = value[4:]
            else:
                if value in df:
                    value = df[value]
                else:
                    raise ValueError(f"Could not find field named \"{value}\" in dataframe during filtering operation. If you meant to use it as a string literal, please prepend it with \"str:\"")

        if operator == ">":
            if is_column_of_type(df, field, "number"):
                return df[field].fillna(0).gt(value)
            else:
                return df[field].gt(value)
        if operator == "<":
            if is_column_of_type(df, field, "number"):
                return df[field].fillna(0).lt(value)
            else:
                return df[field].lt(value)
        if operator == ">=":
            if is_column_of_type(df, field, "number"):
                return df[field].fillna(0).ge(value)
            else:
                return df[field].le(value)
        if operator == "<=":
            if is_column_of_type(df, field, "number"):
                return df[field].fillna(0).le(value)
            else:
                return df[field].ge(value)
        if operator == "==":
            return df[field].eq(value)
        if operator == "!=":
            return df[field].ne(value)
        if operator == "isin":
            return df[field].isin(value)
        if operator == "notin":
            return ~df[field].isin(value)
        if operator == "isempty":
            return pd.isna(df[field]) | df[field].astype(str).str.strip().eq("")
        if operator == "iszero":
            return df[field].eq(0)
        if operator == "iszeroempty":
            return (
                df[field].eq(0)
                | pd.isna(df[field])
                | df[field].astype(str).str.strip().eq("")
            )
        if operator == "contains":
            if isinstance(value, str):
                selection = df[field].str.contains(value)
            elif isinstance(value, list):
                selection = df[field].str.contains(value[0])
                for v in value[1:]:
                    selection |= df[field].str.contains(v)
            else:
                raise ValueError(
                    f"Value must be a string or list for operator {operator}, found: {type(value)}"
                )
            return selection
        if operator == "contains_case_insensitive":
            if isinstance(value, str):
                selection = df[field].str.contains(value, case=False)
            elif isinstance(value, list):
                selection = df[field].str.contains(value[0], case=False)
                for v in value[1:]:
                    selection |= df[field].str.contains(v, case=False)
            else:
                raise ValueError(
                    f"Value must be a string or list for operator {operator}, found: {type(value)}"
                )
            return selection

    raise ValueError(f"Unknown operator {operator}")


def validate_filter_list(filters: list[list]):
    """
    Validate a list of filter lists.

    Parameters
    ----------
    filters : list[list]
        List of filters (each filter is a list).

    Returns
    -------
    bool
        True if all filters are valid.
    """
    for f in filters:
        validate_filter(f)
    return True


def validate_filter(f: list):
    """
    Validate a single filter list.

    Checks that the filter's operator is appropriate for the value type.

    Parameters
    ----------
    f : list
        Filter expressed as a list.

    Returns
    -------
    bool
        True if the filter is valid.

    Raises
    ------
    ValueError
        If the value type does not match the operator requirements.
    """
    operator = f[0]
    if operator in ["and", "or"]:
        pass
    else:
        value = f[2]

        if operator in [">", "<", ">=", "<="]:
            if not isinstance(value, (int, float, bool)):
                raise ValueError(f"Value must be a number for operator {operator}")
        if operator in ["isin", "notin"]:
            if not isinstance(value, list):
                raise ValueError(f"Value must be a list for operator {operator}")
        if operator == "contains":
            if not isinstance(value, str):
                raise ValueError(f"Value must be a string for operator {operator}")
    return True


#######################################
# PRIVATE
#######################################


def _resolve_field_name(
    df: pd.DataFrame, field: str, rename_map: dict = None
) -> str | None:
    """Helper function to resolve a field name using the rename map. Returns the resolved
    field name if found, None otherwise.
    """
    if field in df:
        return field
    if rename_map:
        # Create reverse map for looking up original names
        reverse_map = {v: k for k, v in rename_map.items()}
        if field in reverse_map and reverse_map[field] in df:
            return reverse_map[field]
        elif field in rename_map and rename_map[field] in df:
            return rename_map[field]
    return None


def _is_basic_operator(s: str) -> bool:
    """Check if the operator is a basic comparison operator.

    :param s: Operator as a string. :type s: str :returns: True if it is a basic operator.
    :rtype: bool
    """
    return s in ["<", ">", "<=", ">=", "==", "!=", "isin", "notin", "contains"]


def _is_bool_operator(s: str) -> bool:
    """Check if the operator is a boolean operator.

    :param s: Operator as a string. :type s: str :returns: True if it is a boolean
    operator. :rtype: bool
    """
    return s in ["and", "or", "nand", "nor", "xor", "xnor"]
