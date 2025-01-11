import numpy as np
import pandas as pd


def fancy_format(num):
  if np.isinf(num):
    if num > 0:
      return " ∞"
    else:
      return "-∞"
  if pd.isna(num):
    return "N/A"
  if num == 0:
    return '0.00'
  if num < 1:
    return '{:.2f}'.format(num)
  num = float('{:.3g}'.format(num))
  magnitude = 0
  while abs(num) >= 1000 and abs(num) > 1e-6:
    magnitude += 1
    num /= 1000.0
  return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])