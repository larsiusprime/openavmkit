import json
import re

import numpy as np
import pandas as pd


def dig2_fancy_format(num):

  if abs(num) < 100:
    return '{:.2f}'.format(num)
  else:
    return fancy_format(num)


def fancy_format(num):
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
    return '0.00'
  if 1 > abs(num) > 0:
    return '{:.2f}'.format(num)
  num = float('{:.3g}'.format(num))
  magnitude = 0
  while abs(num) >= 1000 and abs(num) > 1e-6:
    magnitude += 1
    num /= 1000.0
  if magnitude <= 11:
    magletter = ['', 'K', 'M', 'B', 'T', 'Q', 'Qi', 'S', 'Sp', 'O', 'N', 'D'][magnitude]
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), magletter)
  else:
    # format num in scientific notation with 2 decimal places
    return '{:e}'.format(num)


def round_decimals_in_dict(obj: dict, places: int = 2) -> dict:
  """
    Recursively walk dicts/lists, and for every string:
      • find all substrings that look like stringified floating point numbers
      • replace each with its float rounded to `places` places
    Returns a new structure.
    """
  DEC_RE = re.compile(r"(-?\d+\.\d+)")

  def _recurse(x):
    if isinstance(x, dict):
      return {
        _recurse(k) if isinstance(k, str) else k:
          _recurse(v)
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
