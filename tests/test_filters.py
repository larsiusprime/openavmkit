import pandas as pd

from openavmkit.filters import resolve_filter, validate_filter_list, validate_filter, select_filter
from openavmkit.utilities.assertions import lists_are_equal


def test_filter_logic():
  data = {
    "num": [0, 1, 2, 3],
    "str": ["a", "b", "c", "abc"]
  }

  df = pd.DataFrame(data=data)

  filters = [
    ([">", "num", 1],[False, False, True, True]),
    (["<", "num", 1],[True, False, False, False]),
    ([">=", "num", 1],[False, True, True, True]),
    (["<=", "num", 1],[True, True, False, False]),
    (["==", "num", 1],[False, True, False, False]),
    (["!=", "num", 1],[True, False, True, True]),
    (["isin", "str", ["a", "b"]], [True, True, False, False]),
    (["notin", "str", ["a", "b"]], [False, False, True, True]),
    (["contains", "str", "a"], [True, False, False, True])
  ]

  list_filters = []
  for f, expected in filters:
    results = resolve_filter(df, f).tolist()
    assert(lists_are_equal(expected, results))
    list_filters.append(f)

  validate_filter_list(list_filters)

  bad_filters = [
    [">", "num", "a"],
    ["<", "num", "a"],
    [">=", "num", "a"],
    ["<=", "num", "a"],
    ["isin", "str", "a"],
    ["notin", "str", "a"],
    ["contains", "str", ["a"]]
  ]

  for b in bad_filters:
    error = False
    try:
      validate_filter(b)
    except ValueError as e:
      error = True
    assert error == True


def test_filter_resolve():
  data = {
    "num": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "text": ["a", "b", "c", "abc", "a", "b", "c", "abc", "a", "b"],
    "bool": [True, False, True, False, True, True, False, True, False, True]
  }

  df = pd.DataFrame(data=data)

  _filter = [
    "and",
    [">", "num", 2],
    ["<=", "num", 8],
    ["contains", "text", "str:a"],
    ["!=", "bool", False]
  ]

  expected_individual = [
    [False, False, False, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, False],
    [True, False, False, True, True, False, False, True, True, False],
    [True, False, True, False, True, True, False, True, False, True]
  ]

  expected_result = [
    False, False, False, False, True, False, False, True, False, False
  ]

  _filters = _filter[1:]
  for i, f in enumerate(_filters):
    results = resolve_filter(df, f)
    assert(lists_are_equal(expected_individual[i], results.tolist()))

  final_results = resolve_filter(df, _filter)
  assert(lists_are_equal(expected_result, final_results.tolist()))


def test_filter_select():
  data = {
    "num": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "str": ["a", "b", "c", "abc", "a", "b", "c", "abc", "a", "b"],
    "bool": [True, False, True, False, True, True, False, True, False, True]
  }

  df = pd.DataFrame(data=data)

  filter = [
    "and",
    [">", "num", 2],
    ["<=", "num", 8],
    ["contains", "str", "a"],
    ["!=", "bool", False]
  ]

  expected_individual = [
    [3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 3, 4, 7, 8],
    [0, 2, 4, 5, 7, 9]
  ]

  expected_result = [4, 7]

  _filters = filter[1:]
  for i, f in enumerate(_filters):
    results = select_filter(df, f)
    assert(lists_are_equal(expected_individual[i], results.index.tolist()))

  final_results = select_filter(df, filter)
  assert(lists_are_equal(expected_result, final_results.index.tolist()))


def test_boolean_filters():

  def bool_to_int(values):
    return [int(v) for v in values]

  data = {
    "a": [0, 0, 1, 1],
    "b": [0, 1, 0, 1],
    "and": [0, 0, 0, 1],
    "nand": [1, 1, 1, 0],
    "or": [0, 1, 1, 1],
    "nor": [1, 0, 0, 0],
    "xor": [0, 1, 1, 0],
    "xnor": [1, 0, 0, 1]
  }

  df = pd.DataFrame(data=data)

  a = resolve_filter(df, ["==", "a", 1])
  b = resolve_filter(df, ["==", "b", 1])
  assert (lists_are_equal([0, 0, 1, 1], bool_to_int(a)))

  a_and_b = a & b
  a_and_b_r = resolve_filter(df,
  ["and",
      ["==", "a", 1],
      ["==", "b", 1]
    ]
  )

  a_or_b = a | b
  a_or_b_r = resolve_filter(df,
    ["or",
     ["==", "a", 1],
     ["==", "b", 1]
   ]
  )

  a_nand_b = ~(a & b)
  a_nand_b_r = resolve_filter(df,
    ["nand",
     ["==", "a", 1],
     ["==", "b", 1]
   ]
  )

  a_nor_b = ~(a | b)
  a_nor_b_r = resolve_filter(df,
    ["nor",
     ["==", "a", 1],
     ["==", "b", 1]
   ]
  )

  a_xand_b = a ^ b
  a_xand_b_r = resolve_filter(df,
    ["xor",
     ["==", "a", 1],
     ["==", "b", 1]
   ]
  )

  a_xnor_b = ~(a ^ b)
  a_xnor_b_r = resolve_filter(df,
    ["xnor",
     ["==", "a", 1],
     ["==", "b", 1]
   ]
  )

  assert(lists_are_equal([0, 0, 0, 1], bool_to_int(a_and_b)))
  assert(lists_are_equal([0, 1, 1, 1], bool_to_int(a_or_b)))
  assert(lists_are_equal([1, 1, 1, 0], bool_to_int(a_nand_b)))
  assert(lists_are_equal([1, 0, 0, 0], bool_to_int(a_nor_b)))
  assert(lists_are_equal([0, 1, 1, 0], bool_to_int(a_xand_b)))
  assert(lists_are_equal([1, 0, 0, 1], bool_to_int(a_xnor_b)))

  assert(lists_are_equal(bool_to_int(a_and_b), bool_to_int(a_and_b_r)))
  assert(lists_are_equal(bool_to_int(a_or_b), bool_to_int(a_or_b_r)))
  assert(lists_are_equal(bool_to_int(a_nand_b), bool_to_int(a_nand_b_r)))
  assert(lists_are_equal(bool_to_int(a_nor_b), bool_to_int(a_nor_b_r)))
  assert(lists_are_equal(bool_to_int(a_xand_b), bool_to_int(a_xand_b_r)))
  assert(lists_are_equal(bool_to_int(a_xnor_b), bool_to_int(a_xnor_b_r)))

  bool_filters = {
    "and":["and",
      ["==", "a", 1],
      ["==", "b", 1]
    ],
    "or":["or",
      ["==", "a", 1],
      ["==", "b", 1]
    ],
    "nand":["nand",
      ["==", "a", 1],
      ["==", "b", 1]
    ],
    "nor":["nor",
      ["==", "a", 1],
      ["==", "b", 1]
    ],
    "xor":["xor",
      ["==", "a", 1],
      ["==", "b", 1]
    ],
    "xnor":["xnor",
      ["==", "a", 1],
      ["==", "b", 1]
    ]
  }

  for op in bool_filters:
    f = bool_filters[op]
    results = bool_to_int(resolve_filter(df, f))
    expected = data[op]
    assert(lists_are_equal(expected, results))


def test_filter_complex():
  data = {
    "bldg_area_finished_sqft": [
      0, 0, 0, 0, 0,
      1000, 1500, 2000, 2000,
      2000, 3000, 4000, 5000, 5000
    ],
    "bldg_type": [
      None, None, None, None, None,
      "TOWNHOUSE", "TOWNHOME", "TOWNHOME", "HOUSE",
      "SINGLEFAMILY", "SINGLEFAMILY", "SF", "SF", "SF"
    ],
    "neighborhood": [
      "RIVER OAKS TOWNHOUSES", "GREEN HILLS -- TOWNHOUSES", "RIVER OAKS [SINGLEFAMILY]", "SF: GREEN HILLS", "GREEN HILLS SF",
      "RIVER OAKS", "RIVER OAKS (TOWNHOMES)", "RIVER OAKS", "GREEN HILLS (TOWNHOUSES)",
      "GREEN HILLS", "RIVER OAKS (SINGLEFAMILY)", "GREEN HILLS (SINGLEFAMILY)", "RIVER OAKS", "GREEN HILLS",
    ],
    "land_class": [
      "VACANT", "VACANT", "VACANT", "VACANT", "VACANT",
      "TOWNHOUSE", "TOWNHOUSE", "TOWNHOUSE", "TOWNHOUSE",
      "SINGLEFAMILY", "SINGLEFAMILY", "SINGLEFAMILY", "SINGLEFAMILY", "SINGLEFAMILY"
    ],
    "ground_truth": [
      "0_vacant_th", "1_vacant_th", "2_vacant_sf", "3_vacant_sf", "4_vacant_sf",
      "5_improved_th", "6_improved_th", "7_improved_th", "8_improved_th",
      "9_improved_sf", "10_improved_sf", "11_improved_sf", "12_improved_sf", "13_improved_sf"
    ]
  }
  df = pd.DataFrame(data=data)

  filter_is_improved = [">", "bldg_area_finished_sqft", 0]
  filter_is_vacant = ["==", "bldg_area_finished_sqft", 0]
  filter_th_building = ["isin", "bldg_type", ["TOWNHOUSE", "TOWNHOME"]]
  filter_th_neighborhood = ["contains", "neighborhood", ["TOWNHOMES", "TOWNHOUSES"]]
  filter_th_land_class = ["isin", "land_class", ["TOWNHOUSE"]]

  filter_sf_building = ["==", "bldg_type", "SINGLEFAMILY"]
  filter_sf_neighborhood = ["contains", "neighborhood", ["SINGLEFAMILY", "SF"]]
  filter_sf_land_class = ["isin", "land_class", ["SINGLEFAMILY"]]

  filter_improved_th = ["and",
    filter_is_improved,
    ["or",
      filter_th_building,
      filter_th_neighborhood,
      filter_th_land_class
    ]
  ]

  results = select_filter(df, filter_improved_th)["ground_truth"]
  expected = ["5_improved_th", "6_improved_th", "7_improved_th", "8_improved_th"]
  assert(lists_are_equal(expected, results.tolist()))

  filter_vacant_th = ["and",
    filter_is_vacant,
    ["or",
      filter_th_neighborhood,
      filter_th_land_class
    ]
  ]

  results = select_filter(df, filter_vacant_th)["ground_truth"]
  expected = ["0_vacant_th", "1_vacant_th"]
  assert(lists_are_equal(expected, results.tolist()))

  filter_improved_sf = ["and",
    filter_is_improved,
    ["or",
      filter_sf_building,
      filter_sf_neighborhood,
      filter_sf_land_class
    ]
  ]

  results = select_filter(df, filter_improved_sf)["ground_truth"]
  expected = ["9_improved_sf", "10_improved_sf", "11_improved_sf", "12_improved_sf", "13_improved_sf"]
  assert(lists_are_equal(expected, results.tolist()))

  filter_vacant_sf = ["and",
    filter_is_vacant,
    ["or",
      filter_sf_neighborhood,
      filter_sf_land_class
    ]
  ]

  results = select_filter(df, filter_vacant_sf)["ground_truth"]
  expected = ["2_vacant_sf", "3_vacant_sf", "4_vacant_sf"]
  assert(lists_are_equal(expected, results.tolist()))


