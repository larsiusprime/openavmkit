import numpy as np
import pandas as pd


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
  if 1 > num > 0:
    return '{:.2f}'.format(num)
  num = float('{:.3g}'.format(num))
  magnitude = 0
  while abs(num) >= 1000 and abs(num) > 1e-6:
    magnitude += 1
    num /= 1000.0
  return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])