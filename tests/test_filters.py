import numpy as np
import pandas as pd

from openavmkit.filters import resolve_filter, validate_filter_list, validate_filter, select_filter
from openavmkit.utilities.assertions import lists_are_equal
from openavmkit.utilities.settings import _replace_variables
from openavmkit.calculations import _do_calc

def test_filter_calc():
    data = {
        "num": [0,1,2,3],
        "str": ["a","b","c","d"]
    }
    df = pd.DataFrame(data=data)
    
    filter = ["isin", "str", ["a","b"]]
    expected = [True, True, False, False]
    results = resolve_filter(df, filter).tolist()
    
    assert lists_are_equal(expected, results)
    
    entry = [
        "?",
        [
            "isin",
            "str",
            ["a","b"]
        ]
    ]
    
    results = _do_calc(df, entry).tolist()
    
    assert lists_are_equal(expected, results)


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
    (["contains", "str", "str:a"], [True, False, False, True])
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
    ["contains", "str", "str:a"],
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

  filter_sf_building = ["==", "bldg_type", "str:SINGLEFAMILY"]
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


def test_filter_debug():
  data = {'key': {188926: '71399'},
   'census_tract': {188926: '37081016007'},
   'census_block_group': {188926: '370810160071'},
   'city': {188926: None},
   'zoning_class': {188926: None},
   'zoning_desc': {188926: None},
   'zoning_class_desc': {188926: None},
   'school_district': {188926: '5'},
   'dist_to_cbd': {188926: 5.506255435217201},
   'dist_to_airport': {188926: 1.3559316067912803},
   'dist_to_universities_Elon University School of Law': {188926: 5.923459582503213},
   'dist_to_universities_Guilford Technical Community College': {188926: 8.768701467870912},
   'dist_to_universities_North Carolina A&T State University Farm': {188926: 8.361196069239428},
   'dist_to_universities_North Carolina Agricultural and Technical State University': {188926: 6.533271843007366},
   'dist_to_universities_UNC Greensboro South Campus': {188926: 5.456244541194027},
   'dist_to_universities_University of North Carolina - Greensboro': {188926: 4.945217952263609},
   'dist_to_universities_None': {188926: np.nan},
   'dist_to_colleges_Bennett College': {188926: 6.613962953411121},
   'dist_to_colleges_Greensboro College': {188926: 5.460695141884095},
   'dist_to_colleges_Guilford College': {188926: 0.9995568666696698},
   'dist_to_greenspace': {188926: 0.5752759367499076},
   'address': {188926: '4 MEADOW CROSSING CT'},
   'land_class': {188926: 'TOWNHOUSE'},
   'property_owner': {188926: 'HINKLE, MELINDA D;HINKLE, MARK LAWING;HINKLE, THOMAS L'},
   'assr_land_value': {188926: 45000.0},
   'assr_impr_value': {188926: 100600.0},
   'assr_market_value': {188926: 145600.0},
   'neighborhood': {188926: '7836A04-KESWICK PLACE TOWNHOMES'},
   'bldg_count': {188926: 1.0},
   'bldg_plumbing': {188926: '2.5'},
   'bldg_class': {188926: '041-TOWNHOME'},
   'land_area_sqft': {188926: 1148.1841008},
   'bldg_desc': {188926: '041-TOWNHOME'},
   'bldg_type': {188926: '01-SFR-CONST'},
   'bldg_units': {188926: 1.0},
   'bldg_area_finished_sqft': {188926: 1024.0},
   'bldg_stories': {188926: 1.0},
   'bldg_style': {188926: 'UNKNOWN'},
   'bldg_exterior': {188926: 'ALUMINUM OR VINYL'},
   'bldg_heating': {188926: 'FORCED AIR-DUCTED'},
   'bldg_ac': {188926: 'CENTRAL'},
   'bldg_fixtures': {188926: '8'},
   'bldg_year_built': {188926: 2001.0},
   'bldg_effective_year_built': {188926: 2009.0},
   'bldg_additions': {188926: 4.0},
   'bldg_year_remodeled': {188926: None},
   'bldg_quality_num': {188926: 1.4},
   'bldg_quality_txt': {188926: 'B'},
   'bldg_condition_txt': {188926: '0.0'},
   'bldg_condition_num': {188926: 0.12},
   'assr_impr_value_building': {188926: 100600.0},
   'bldg_rooms_bed': {188926: 2.0},
   'bldg_foundation': {188926: 'CONTFOOT'},
   'bldg_area_footprint_sqft': {188926: 512.0},
   'total_replacement_value': {188926: 114298.0},
   'total_depreciated_value': {188926: 100582.0},
   'outbldg_value': {188926: 0.0},
   'bldg_rooms_bath_full': {188926: 2.0},
   'bldg_rooms_bath_half': {188926: 1.0},
   'percent_depreciation': {188926: 0.12000209977427427},
   'total_depreciation': {188926: 13716.0},
   'elevation_mean': {188926: 244.236198425293},
   'elevation_stdev': {188926: 0.141960144042969},
   'slope_mean': {188926: 89.9947776794434},
   'slope_stdev': {188926: 0.0002403259277343},
   'noise_mean': {188926: None},
   'noise_max': {188926: None},
   'key2': {188926: '7836920864-000'},
   'zoning': {188926: 'RM-8-RESIDENTIAL, MULTI-FAMILY, 8 UNITS PER ACRE'},
   'zoning_category': {188926: None},
   'model_group': {188926: 'single_family'}
  }
  df = pd.DataFrame(data=data)

  settings = {
    "ref":{
      "filters": {
        "is_vacant": ["<=", "bldg_area_finished_sqft", 0],
        "is_improved": [">", "bldg_area_finished_sqft", 0],
        "not_too_big": ["<=", "land_area_sqft", 43560],
        "th": {
          "select_improved":
            ["and",
              "$$ref.filters.is_improved",
              ["or",
                "$$ref.filters.th.building",
                "$$ref.filters.th.neighborhood",
                "$$ref.filters.th.land_class"
              ]
            ],
          "select_vacant":
            ["and",
              "$$ref.filters.is_vacant",
              ["or",
                "$$ref.filters.th.neighborhood",
                "$$ref.filters.th.land_class"
              ]
            ],
          "building": ["or",
            ["isin", "bldg_desc", ["041-TOWNHOME", "05-PATIOHM", "042-DETACHEDTOWNHOME"]]
          ],
          "land_class": ["isin", "land_class", ["TOWNHOUSE", "TWINHOME"]],
          "neighborhood":  ["contains", "neighborhood",
            ["TOWN HOME", "TOWN HOUSE", "TOWNHOM","TOWNHOME", "TOWNHOUSE", "TOWNHME", "TWIN HOME"]
          ]
        },
        "sf": {
          "select_improved": ["and",
                              "$$ref.filters.is_improved",
                              "$$ref.filters.not_too_big",
                              "$$ref.filters.sf.building"
                              ],
          "select_vacant": ["and",
                            "$$ref.filters.is_vacant",
                            "$$ref.filters.not_too_big",
                            ["or",
                             "$$ref.filters.sf.zoning1",
                             "$$ref.filters.sf.zoning2",
                             "$$ref.filters.pud.land",
                             "$$ref.filters.sf.land_class"
                             ]
                            ],
          "building": ["or",
                       ["isin", "bldg_type", ["01-SFR-CONST"]],
                       ["isin", "bldg_desc", ["01-SFR", "013-TINY HOUSE"]]
                       ],
          "zoning1": ["or",
                      ["isin", "zoning_category", ["RESIDENTIAL SINGLE FAMILY"]],
                      ["isin", "zoning", ["SFR"]],
                      ["contains", "zoning", ["Single-Family"]]
                      ],
          "zoning2": ["and",
                      ["or",
                       ["isin", "zoning", ["SFR"]],
                       ["contains", "zoning", ["Single-Family"]],
                       ["isempty", "zoning_class"],
                       ["isin", "zoning_class", ["MIXED USE"]]
                       ],
                      ["isin", "land_class", ["VACANT", "COMMON AREA"]]
                      ],
          "land_class": ["and",
                         ["isempty", "zoning_category"],
                         ["isin", "land_class", ["RESIDENTIAL"]]
                         ]
        },
        "pud": {
          "zoning": ["or",
                     ["isin", "zoning", ["PUD"]],
                     ["isin", "zoning_class", ["PLANNED UNIT DEVELOPMENT DISTRICT"]],
                     ["isin", "zoning_category", ["PLANNED DEVELOPMENT", "PLANNED UNIT DEVELOPMENT"]]
                     ],
          "land": ["and",
                   "$$ref.filters.pud.zoning",
                   ["or",
                    ["isin", "land_class", ["VACANT", "DEVELOPMT RESTRICTED", "RESIDENTIAL", "COMMON AREA"]],
                    ["isempty", "land_class", "str:"]
                    ]
                   ]
        }
      }
    }
  }

  settings = _replace_variables(settings)
  th = settings["ref"]["filters"]["th"]

  th_building = th["building"]
  th_land_class = th["land_class"]
  th_neighborhood = th["neighborhood"]
  th_select_improved = th["select_improved"]
  th_select_vacant = th["select_vacant"]

  results_th_building = select_filter(df, th_building)["key"].values
  results_th_land_class = select_filter(df, th_land_class)["key"].values
  results_th_neighborhood = select_filter(df, th_neighborhood)["key"].values
  results_th_select_improved = select_filter(df, th_select_improved)["key"].values
  results_th_select_vacant = select_filter(df, th_select_vacant)["key"].values

  assert lists_are_equal(results_th_building, ['71399'])
  assert lists_are_equal(results_th_land_class, ['71399'])
  assert lists_are_equal(results_th_neighborhood, ['71399'])
  assert lists_are_equal(results_th_select_improved, ['71399'])
  assert lists_are_equal(results_th_select_vacant, [])

  data = {'key': {64315: '165177'},
    'census_tract': {64315: '37081016201'},
    'census_block_group': {64315: '370810162012'},
    'city': {64315: None},
    'zoning_class': {64315: None},
    'zoning_desc': {64315: None},
    'zoning_class_desc': {64315: None},
    'school_district': {64315: '5'},
    'dist_to_cbd': {64315: 12.334659422365801},
    'dist_to_airport': {64315: 3.054274088106628},
    'dist_to_universities_Elon University School of Law': {64315: 12.742663677292681},
    'dist_to_universities_Guilford Technical Community College': {64315: 15.508875542792254},
    'dist_to_universities_North Carolina A&T State University Farm': {64315: 15.194037094490872},
    'dist_to_universities_North Carolina Agricultural and Technical State University': {64315: 13.366405015869573},
    'dist_to_universities_UNC Greensboro South Campus': {64315: 12.121014557860626},
    'dist_to_universities_University of North Carolina - Greensboro': {64315: 11.690678514343912},
    'dist_to_universities_None': {64315: np.nan},
    'dist_to_colleges_Bennett College': {64315: 13.430077445869019},
    'dist_to_colleges_Greensboro College': {64315: 12.254151036030194},
    'dist_to_colleges_Guilford College': {64315: 7.084873212548866},
    'dist_to_greenspace': {64315: 4.1387322347710365},
    'address': {64315: '5804 BILLET RD'},
    'land_class': {64315: 'RESIDENTIAL'},
    'property_owner': {64315: 'CHANEY, RACHEL A'},
    'assr_land_value': {64315: 78000.0},
    'assr_impr_value': {64315: 211700.0},
    'assr_market_value': {64315: 291800.0},
    'neighborhood': {64315: '7807A03-TWELVE OAKS/OAK RIDGE PLANTATION'},
    'bldg_count': {64315: 1.0},
    'bldg_plumbing': {64315: '2.5'},
    'bldg_class': {64315: '01-SFR'},
    'land_area_sqft': {64315: 20000.0},
    'bldg_desc': {64315: '01-SFR'},
    'bldg_type': {64315: '01-SFR-CONST'},
    'bldg_units': {64315: 1.0},
    'bldg_area_finished_sqft': {64315: 2366.0},
    'bldg_stories': {64315: 1.0},
    'bldg_style': {64315: '1.0 STORY'},
    'bldg_exterior': {64315: 'FACE BRK'},
    'bldg_heating': {64315: 'FORCED AIR-DUCTED'},
    'bldg_ac': {64315: 'CENTRAL'},
    'bldg_fixtures': {64315: '8'},
    'bldg_year_built': {64315: 1998.0},
    'bldg_effective_year_built': {64315: 2006.0},
    'bldg_additions': {64315: 3.0},
    'bldg_year_remodeled': {64315: None},
    'bldg_quality_num': {64315: 1.15},
    'bldg_quality_txt': {64315: 'B'},
    'bldg_condition_txt': {64315: '0.0'},
    'bldg_condition_num': {64315: 0.15},
    'assr_impr_value_building': {64315: 213800.0},
    'bldg_rooms_bed': {64315: 3.0},
    'bldg_foundation': {64315: 'CONTFOOT'},
    'bldg_area_footprint_sqft': {64315: 2366.0},
    'total_replacement_value': {64315: 249037.0},
    'total_depreciated_value': {64315: 211681.0},
    'outbldg_value': {64315: 2100.0},
    'bldg_rooms_bath_full': {64315: 2.0},
    'bldg_rooms_bath_half': {64315: 1.0},
    'percent_depreciation': {64315: 0.15000180696041152},
    'total_depreciation': {64315: 37356.0},
    'elevation_mean': {64315: 281.722068911693},
    'elevation_stdev': {64315: 0.84372043530057},
    'slope_mean': {64315: 89.9841312345911},
    'slope_stdev': {64315: 0.0079087717601446},
    'noise_mean': {64315: None},
    'noise_max': {64315: None},
    'key2': {64315: '7807683483-000'},
    'zoning': {64315: 'RS-40-Residential, Single-Family, 1 unit per acre'},
    'zoning_category': {64315: None},
    'model_group': {64315: None}
  }
  df = pd.DataFrame(data=data)

  sf = settings["ref"]["filters"]["sf"]

  sf_building = sf["building"]
  sf_land_class = sf["land_class"]
  sf_zoning1 = sf["zoning1"]
  sf_zoning2 = sf["zoning2"]
  sf_select_improved = sf["select_improved"]
  sf_select_vacant = sf["select_vacant"]

  results_sf_building = select_filter(df, sf_building)["key"].values
  results_sf_land_class = select_filter(df, sf_land_class)["key"].values
  results_sf_select_zoning1 = select_filter(df, sf_zoning1)["key"].values
  results_sf_select_zoning2 = select_filter(df, sf_zoning2)["key"].values
  results_sf_select_improved = select_filter(df, sf_select_improved)["key"].values
  results_sf_select_vacant = select_filter(df, sf_select_vacant)["key"].values

  assert lists_are_equal(results_sf_building, ['165177'])
  assert lists_are_equal(results_sf_land_class, ['165177'])
  assert lists_are_equal(results_sf_select_zoning1, ['165177'])
  assert lists_are_equal(results_sf_select_zoning2, [])
  assert lists_are_equal(results_sf_select_improved, ['165177'])
  assert lists_are_equal(results_sf_select_vacant, [])

