import re

import numpy as np
import pandas as pd


def dig2_fancy_format(num: float | int) -> str:
    """Fancy-formats a number, and if the absolute value of the number is less than 100, shows it with two digits

    Parameters
    ----------
    num : float | int
        The number to format

    Returns
    -------
    str
        The formatted number as a string
    """
    if abs(num) < 100:
        return "{:.2f}".format(num)
    else:
        return fancy_format(num)


def fancy_format(num: float | int) -> str:
    """Formats a number in a pleasing and efficient way

    Parameters
    ----------
    num : float | int
        The number to format

    Returns
    -------
    str
        The formatted number as a string

    Notes
    -----
    - Renders infinity as ∞
    - Renders any null, nan, or similar values as "N/A"
    - Shows decimals for small numbers
    - Shows large numbers with suffixes (K for thousand, M for million, etc)
    """
    if not isinstance(num, (int, float, np.number)):
        # if NoneType:
        if num is None:
            return "N/A"
        return str(num) + "-->?(type=" + str(type(num)) + ")"

    if np.isinf(num):
        return "∞" if num > 0 else "-∞"

    if np.isinf(num):
        if num > 0:
            return " ∞"
        else:
            return "-∞"
    if pd.isna(num):
        return "N/A"
    if num == 0:
        return "0.00"
    if 1 > abs(num) > 0:
        return "{:.2f}".format(num)
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000 and abs(num) > 1e-6:
        magnitude += 1
        num /= 1000.0
    if magnitude <= 11:
        magletter = ["", "K", "M", "B", "T", "Q", "Qi", "S", "Sp", "O", "N", "D"][
            magnitude
        ]
        return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), magletter)
    else:
        # format num in scientific notation with 2 decimal places
        return "{:e}".format(num)


def round_decimals_in_dict(obj: dict | list, places: int = 2) -> dict:
    """Recursively walk dicts/lists, and for every string:

    - Finds all substrings that look like stringified floating point numbers
    - Replaces each with its ``float`` value rounded to `places` places
    - Returns a new structure.

    Parameters
    ----------
    obj : dict | list
        The object to traverse
    places : int, optional
        The number of decimal places to show. Defaults to 2.

    Returns
    -------
    dict | list
        The newly formatted object
    """
    DEC_RE = re.compile(r"(-?\d+\.\d+)")

    def _recurse(x):
        if isinstance(x, dict):
            return {
                _recurse(k) if isinstance(k, str) else k: _recurse(v)
                for k, v in x.items()
            }
        elif isinstance(x, list):
            return [_recurse(v) for v in x]
        elif isinstance(x, str):
            # substitute each decimal substring
            return DEC_RE.sub(lambda m: f"{float(m.group(1)):.{places}f}", x)
        else:
            return x

    return _recurse(obj)
