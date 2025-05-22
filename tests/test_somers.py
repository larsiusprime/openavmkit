import numpy as np
from openavmkit.utilities.somers import get_depth_percent_ft, get_unit_ft, get_lot_value_ft


def test_get_depth():
  expected = {
    0: 0.0,
    10: 0.255,
    20: 0.415,
    50: 0.726,
    100: 1.000,
    250: 1.263
  }
  actual = {}
  for d in (0, 10, 20, 50, 100, 250):
    actual[d] = get_depth_percent_ft(d)
    print(f"{d} = {actual[d]}")
  for d in expected:
    assert abs(actual[d] - expected[d]) < 1e-1, f"Depth curve at {d} ft: expected {expected[d]}, got {actual[d]}"


def test_get_depth_numpy():
  expected = [0.0, 0.255, 0.415, 0.726, 1.000, 1.08, 1.263]
  actual = get_depth_percent_ft(np.array([0, 10, 20, 50, 100, 125, 250]))
  print(actual)
  deltas = abs(actual - expected)
  for i, d in enumerate(deltas):
    assert d < 1e-2, f"Depth curve at {i} ft: expected {expected[i]}, got {actual[i]}"


def test_get_unit():
  params = [
    [50, 50, 100, 1.00],
    [50, 50,  50, 1.37],
    [50, 50, 125, 0.92],
    [100, 100, 100, 1.00],
    [100, 100,  50, 1.37],
    [100, 100, 125, 0.92],
    [200, 100, 100, 2.00],
    [200, 100,  50, 2.75],
    [200, 100, 125, 1.85]
  ]
  for param in params:
    lot_value = param[0]
    frontage_ft = param[1]
    depth_ft = param[2]
    expected = param[3]
    unit_ft = get_unit_ft(lot_value, frontage_ft, depth_ft)
    delta = abs(unit_ft - expected)
    assert delta < 1e-2, f"Unit ft: expected {expected}, got {unit_ft}"


def test_get_unit_numpy():
  params = [
    [50, 50, 100, 1.00],
    [50, 50,  50, 1.37],
    [50, 50, 125, 0.92],
    [100, 100, 100, 1.00],
    [100, 100,  50, 1.37],
    [100, 100, 125, 0.92],
    [200, 100, 100, 2.00],
    [200, 100,  50, 2.75],
    [200, 100, 125, 1.85]
  ]

  lot_values = np.array([param[0] for param in params])
  frontages = np.array([param[1] for param in params])
  depths = np.array([param[2] for param in params])
  expected = np.array([param[3] for param in params])

  unit_fts = get_unit_ft(lot_values, frontages, depths)
  deltas = abs(unit_fts - expected)
  for i in range(len(deltas)):
    assert deltas[i] < 1e-2, f"Unit ft: expected {expected[i]}, got {unit_fts[i]}"


def test_get_lot_value():
  params = [
    [50, 50, 100, 1.00],
    [50, 50,  50, 1.37],
    [50, 50, 125, 0.92],
    [100, 100, 100, 1.00],
    [100, 100,  50, 1.37],
    [100, 100, 125, 0.92],
    [200, 100, 100, 2.00],
    [200, 100,  50, 2.75],
    [200, 100, 125, 1.85]
  ]
  for param in params:
    expected = param[0]
    frontage_ft = param[1]
    depth_ft = param[2]
    unit_ft = param[3]
    lot_value = get_lot_value_ft(unit_ft, frontage_ft, depth_ft)
    delta = abs(lot_value - expected)
    assert delta < 1.0, f"Lot value: expected {expected}, got {lot_value}"


def test_get_lot_value_numpy():
  params = [
    [50, 50, 100, 1.00],
    [50, 50,  50, 1.37],
    [50, 50, 125, 0.92],
    [100, 100, 100, 1.00],
    [100, 100,  50, 1.37],
    [100, 100, 125, 0.92],
    [200, 100, 100, 2.00],
    [200, 100,  50, 2.75],
    [200, 100, 125, 1.85]
  ]

  lot_values = np.array([param[0] for param in params])
  frontages = np.array([param[1] for param in params])
  depths = np.array([param[2] for param in params])
  expected = np.array([param[3] for param in params])

  lot_values_out = get_lot_value_ft(expected, frontages, depths)
  deltas = abs(lot_values_out - lot_values)
  for i in range(len(deltas)):
    assert deltas[i] < 1.0, f"Lot value: expected {lot_values[i]}, got {lot_values_out[i]}"