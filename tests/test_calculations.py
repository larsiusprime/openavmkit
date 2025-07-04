import pandas as pd

from openavmkit.calculations import perform_calculations, _crawl_calc_dict_for_fields, perform_tweaks
from openavmkit.utilities.assertions import dfs_are_equal


def test_calculations_str():
  data = {
    "neighborhood": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "", " ", None],
    "census_tract": ["1", "1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "4"]
  }
  df = pd.DataFrame(data=data)
  calc = {
    "fillna": ["fillna", "neighborhood", "census_tract"],
    "fillempty": ["fillempty", "neighborhood", "census_tract"]
  }
  expected = {
    "neighborhood": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "", " ", None],
    "census_tract": ["1", "1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "4"],
    "fillna": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "", " ", "4"],
    "fillempty": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "4", "4", "4"]
  }
  df_expected = pd.DataFrame(data=expected)
  df_results = perform_calculations(df, calc)

  assert dfs_are_equal(df_results, df_expected, allow_weak = True)


def test_calculations_math():
  data = {
    "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "b": [2, 2, 2, 2, 2, 2, 3, 3, 3, 3,  3,  0]
  }
  df = pd.DataFrame(data=data)
  calc = {
    "a+b" : ["+", "a", "b"],
    "a-b" : ["-", "a", "b"],
    "a*b" : ["*", "a", "b"],
    "a/b" : ["/", "a", "b"],
    "a/0b": ["/0", "a", "b"]
  }
  expected = {
    "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "b": [2, 2, 2, 2, 2, 2, 3, 3, 3, 3,  3, 0],
    "a+b": [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 11],
    "a-b": [-2, -1, 0, 1, 2, 3, 3, 4, 5, 6, 7, 11],
    "a*b": [0, 2, 4, 6, 8, 10, 18, 21, 24, 27, 30, 0],
    "a/b": [0, .5, 1, 1.5, 2, 2.5, 2, 7/3, 8/3, 3, 10/3, float('inf')],
    "a/0b": [0, .5, 1, 1.5, 2, 2.5, 2, 7/3, 8/3, 3, 10/3, None]
  }
  df_expected = pd.DataFrame(data=expected)
  df_results = perform_calculations(df, calc)

  assert dfs_are_equal(df_results, df_expected)


def test_calculations_math_2():
  data = {
    "a": [3.14, 1.5, 1.49, 1.51, 10.1, 10.25, 10.51, 10.5, -43.8, 99.9]
  }
  df = pd.DataFrame(data=data)
  calc = {
    "asint(a)": ["asint", "a"],
    "asfloat(a)": ["asfloat", "a"],
    "asstr(a)": ["asstr", "a"],
    "floor(a)": ["floor", "a"],
    "ceil(a)": ["ceil", "a"],
    "round(a)": ["round", "a"],
    "round_nearest(a)": ["round_nearest", "a", 5],
    "abs": ["abs", "a"]
  }
  expected = {
    "a": [3.14, 1.5, 1.49, 1.51, 10.1, 10.25, 10.51, 10.5, -43.8, 99.9],
    "asint(a)": [3, 1, 1, 1, 10, 10, 10, 10, -43, 99],
    "asfloat(a)": [3.14, 1.5, 1.49, 1.51, 10.1, 10.25, 10.51, 10.5, -43.8, 99.9],
    "asstr(a)": ["3.14", "1.5", "1.49", "1.51", "10.1", "10.25", "10.51", "10.5", "-43.8", "99.9"],
    "floor(a)": [3.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, -44.0, 99.0],
    "ceil(a)": [4.0, 2.0, 2.0, 2.0, 11.0, 11.0, 11.0, 11.0, -43.0, 100.0],
    "round(a)": [3.0, 2.0, 1.0, 2.0, 10.0, 10.0, 11.0, 10.0, -44.0, 100.0],
    "round_nearest(a)": [5.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -45.0, 100.0],
    "abs": [3.14, 1.5, 1.49, 1.51, 10.1, 10.25, 10.51, 10.5, 43.8, 99.9]
  }
  df_expected = pd.DataFrame(data=expected)
  df_results = perform_calculations(df, calc)
  assert dfs_are_equal(df_results, df_expected)


def test_calculations_txt():
  data = {
    "quality_num": [  1,   2,   3,   4,   5,   2,   3,   4,   4,   4,   4,   5,   4,   3,   2,   1],
    "quality_txt": ["f", "d", "c", "b", "a", "d", "c", "b", "b", "b", "b", "a", "b", "c", "d", "f"],
    "condition_num": [3, 7.4, 11, 28, 31, 34, 42, 47.5, 50.12314, 59, 61, 66, 79, 84.56, 89.999, 95.875]
  }
  df = pd.DataFrame(data=data)
  calc = {
    "num->txt": ["map", "quality_num", {"1": "f", "2": "d", "3": "c", "4": "b", "5": "a"}],
    "txt->num": ["map", "quality_txt", {"f":   1,   "d": 2,   "c": 3,   "b": 4,   "a": 5}],
    "quality_desc": [
      "join", ["values", "quality_num", "quality_txt"], "str: - "
    ],
    "condition_round": ["//", ["round_nearest", "condition_num", 20], 20],
    "condition_map": ["map",
      ["//", ["round_nearest", "condition_num", 20], 20],
      {
        "0": "f",
        "1": "d",
        "2": "c",
        "3": "b",
        "4": "a",
        "5": "a"
      }
    ],
    "condition_join": [
      "join",
      [
        "values",
        ["//", ["round_nearest", "condition_num", 20], 20],
        ["map",
          ["//", ["round_nearest", "condition_num", 20], 20],
          {
            "0": "f",
            "1": "d",
            "2": "c",
            "3": "b",
            "4": "a",
            "5": "a"
          }
        ]
      ],
      "str: - "
     ],
  }
  expected = {
    "quality_num": [  1,   2,   3,   4,   5,   2,   3,   4,   4,   4,   4,   5,   4,   3,   2,   1],
    "quality_txt": ["f", "d", "c", "b", "a", "d", "c", "b", "b", "b", "b", "a", "b", "c", "d", "f"],
    "condition_num": [3, 7.4, 11, 28, 31, 34, 42, 47.5, 50.12314, 59, 61, 66, 79, 84.56, 89.999, 95.875],
    "num->txt":    ["f", "d", "c", "b", "a", "d", "c", "b", "b", "b", "b", "a", "b", "c", "d", "f"],
    "txt->num":    [  1,   2,   3,   4,   5,   2,   3,   4,   4,   4,   4,   5,   4,   3,   2,   1],
    "quality_desc":["1 - f", "2 - d", "3 - c", "4 - b", "5 - a", "2 - d", "3 - c", "4 - b", "4 - b", "4 - b", "4 - b", "5 - a", "4 - b", "3 - c", "2 - d", "1 - f"],
    "condition_round": [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5],
    "condition_map": ["f", "f", "d", "d", "c", "c", "c", "c", "b", "b", "b", "b", "a", "a", "a", "a"],
    "condition_join": ["0 - f", "0 - f", "1 - d", "1 - d", "2 - c", "2 - c", "2 - c", "2 - c", "3 - b", "3 - b", "3 - b", "3 - b", "4 - a", "4 - a", "4 - a", "5 - a"]
  }
  df_expected = pd.DataFrame(data=expected)
  df_results = perform_calculations(df, calc)

  assert dfs_are_equal(df_results, df_expected)


def test_calculations_filter2():
  data = {
    "bldg_area_finished_sqft": [100, 100, 100, 100, 100, 100, 100, 100],
    "osm_bldg_area_footprint_sqft": [0, 0, None, float('nan'), 0, 10, 10, 10]
  }
  df = pd.DataFrame(data=data)
  calc = {
    "bldg_area_finished_sqft": [
      "*", "bldg_area_finished_sqft", [
        "?", ["not", ["iszeroempty", "osm_bldg_area_footprint_sqft"]]
      ]
    ]
  }
  expected = {
    "bldg_area_finished_sqft": [0, 0, 0, 0, 0, 100, 100, 100],
    "osm_bldg_area_footprint_sqft": [0, 0, None, float('nan'), 0, 10, 10, 10],
  }
  df_expected = pd.DataFrame(data=expected)
  df_results = perform_calculations(df, calc)
  assert dfs_are_equal(df_results, df_expected)


def test_calculations_filter():
  data = {
    "quality_num": [  1,   2,   3,   4,   5,   2,   3,   4,   4,   4,   4,   5,   4,   3,   2,   1],
    "quality_txt": ["f", "d", "c", "b", "a", "d", "c", "b", "b", "b", "b", "a", "b", "c", "d", "f"],
    "condition_num": [3, 7.4, 11, 28, 31, 34, 42, 47.5, 50.12314, 59, 61, 66, 79, 84.56, 89.999, 95.875]
  }
  df = pd.DataFrame(data=data)
  calc = {
    "txt=a": ["?", ["==", "quality_txt", "str:a"]],
    "num>3": ["?", [">", "quality_num", 3]],
    "txt=abc": ["?", ["isin", "quality_txt", ["a","b","c"]]],
    "con<50": ["?", ["<", "condition_num", 50]],
    "txt=abc&con<50": ["?",
      ["and",
        ["isin", "quality_txt", ["a","b","c"]],
        ["<", "condition_num", 50]
      ]
    ]
  }
  expected = {
    "quality_num": [  1,   2,   3,   4,   5,   2,   3,   4,   4,   4,   4,   5,   4,   3,   2,   1],
    "quality_txt": ["f", "d", "c", "b", "a", "d", "c", "b", "b", "b", "b", "a", "b", "c", "d", "f"],
    "condition_num": [3, 7.4, 11, 28, 31, 34, 42, 47.5, 50.12314, 59, 61, 66, 79, 84.56, 89.999, 95.875],
    "txt=a": [False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False],
    "num>3": [False, False, False, True, True, False, False, True, True, True, True, True, True, False, False, False],
    "txt=abc": [False, False, True, True, True, False, True, True, True, True, True, True, True, True, False, False],
    "con<50": [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False],
    "txt=abc&con<50": [False, False, True, True, True, False, True, True, False, False, False, False, False, False, False, False]
  }
  df_expected = pd.DataFrame(data=expected)
  df_results = perform_calculations(df, calc)

  assert dfs_are_equal(df_results, df_expected)


def test_crawl_calc_list_for_fields():
  #crawl_calc_dict_for_fields
  calc = {
    "a+b" : ["+", "a", "b"],
    "a-b" : ["-", "a", "b"],
    "a*b" : ["*", "a", "b"],
    "a/b" : ["/", "a", "b"],
    "a/0b": ["/0", "a", "b"],
    "asint(a)": ["asint", "a"],
    "asfloat(a)": ["asfloat", "a"],
    "asstr(a)": ["asstr", "a"],
    "floor(a)": ["floor", "a"],
    "ceil(a)": ["ceil", "a"],
    "round(a)": ["round", "a"],
    "round_nearest(a)": ["round_nearest", "a", 5],
    "abs": ["abs", "a"],
    "num->txt": ["map", "quality_num", {"1": "f", "2": "d", "3": "c", "4": "b", "5": "a"}],
    "txt->num": ["map", "quality_txt", {"f":   1,   "d": 2,   "c": 3,   "b": 4,   "a": 5}],
    "quality_desc": [
      "join", ["values", "quality_num", "quality_txt"], "str: - "
    ],
    "condition_round": ["//", ["round_nearest", "condition_num", 20], 20],
    "condition_map": ["map",
                      ["//", ["round_nearest", "condition_num", 20], 20],
                      {
                        "0": "f",
                        "1": "d",
                        "2": "c",
                        "3": "b",
                        "4": "a",
                        "5": "a"
                      }
                      ],
    "condition_join": [
      "join",
      [
        "values",
        ["//", ["round_nearest", "condition_num", 20], 20],
        ["map",
         ["//", ["round_nearest", "condition_num", 20], 20],
         {
           "0": "f",
           "1": "d",
           "2": "c",
           "3": "b",
           "4": "a",
           "5": "a"
         }
         ]
      ],
      "str: - "
    ],
  }
  results = _crawl_calc_dict_for_fields(calc)
  results.sort()
  expected = ['a', 'b', 'condition_num', 'quality_num', 'quality_txt']
  assert results == expected


def test_split_quality():
  data = {
    "BLDG_DESC": [
      "A  180%",
      "A+10  190%",
      "A+15  195%",
      "A+20  200%",
      "A+25  205%",
      "A+5  185%",
      "A-10  170%",
      "A-15  165%",
      "A-20  160%",
      "A-25  155%",
      "A-5  175%",
      "AA  275%",
      "AA+1  325%",
      "AA+1  350%",
      "AA+2  375%",
      "AA+2  400%",
      "AA+3  425%",
      "AA+3  450%",
      "AA+4  475%",
      "AA+4  500%",
      "AA+5  300%",
      "AA+5  550%",
      "AA+6  600%",
      "AA+7  625%",
      "AA-1  230%",
      "AA-1  245%",
      "AA-2  210%",
      "AA-2  220%",
      "AA-5  260%",
      "Above Average",
      "Average",
      "B  125%",
      "B+10  135%",
      "B+15  140%",
      "B+20  145%",
      "B+25  150%",
      "B+5  130%",
      "B-10  115%",
      "B-5  120%",
      "C  100%",
      "C  105%",
      "C+10  110%",
      "C+15  114%",
      "C+20  119%",
      "C+25  124%",
      "C+5  105%",
      "C-10  90%",
      "C-5  95%",
      "D  75%",
      "D+10  85%",
      "D+5  80%",
      "D-10  60%",
      "D-5  70%",
      "E  60%",
      "E+5  65%",
      "E-10  50%",
      "E-15  45%",
      "E-20  40%",
      "E-25  35%",
      "E-30  30%",
      "E-35  25%",
      "E-40  20%",
      "E-45  15%",
      "E-5  55%",
      "Excellent",
      "Fair",
      "Good",
      "GRADE",
      "High",
      "Highest",
      "Low",
      "Lowest",
      "Very Good"
    ],
  }

  data_expected = {
    "BLDG_DESC": data["BLDG_DESC"],
    "bldg_quality_txt_score": [
        "A  180%",
        "A+10  190%",
        "A+15  195%",
        "A+20  200%",
        "A+25  205%",
        "A+5  185%",
        "A-10  170%",
        "A-15  165%",
        "A-20  160%",
        "A-25  155%",
        "A-5  175%",
        "AA  275%",
        "AA+1  325%",
        "AA+1  350%",
        "AA+2  375%",
        "AA+2  400%",
        "AA+3  425%",
        "AA+3  450%",
        "AA+4  475%",
        "AA+4  500%",
        "AA+5  300%",
        "AA+5  550%",
        "AA+6  600%",
        "AA+7  625%",
        "AA-1  230%",
        "AA-1  245%",
        "AA-2  210%",
        "AA-2  220%",
        "AA-5  260%",
        "B  125%",
        "C  100%",
        "B  125%",
        "B+10  135%",
        "B+15  140%",
        "B+20  145%",
        "B+25  150%",
        "B+5  130%",
        "B-10  115%",
        "B-5  120%",
        "C  100%",
        "C  105%",
        "C+10  110%",
        "C+15  114%",
        "C+20  119%",
        "C+25  124%",
        "C+5  105%",
        "C-10  90%",
        "C-5  95%",
        "D  75%",
        "D+10  85%",
        "D+5  80%",
        "D-10  60%",
        "D-5  70%",
        "E  60%",
        "E+5  65%",
        "E-10  50%",
        "E-15  45%",
        "E-20  40%",
        "E-25  35%",
        "E-30  30%",
        "E-35  25%",
        "E-40  20%",
        "E-45  15%",
        "E-5  55%",
        "AA  275%",
        "D  75%",
        "B  125%",
        "C  100%",
        "A  180%",
        "AA  275%",
        "E  60%",
        "E  60%",
        "A  180%"
      ],
    "bldg_quality_txt_score_split_before": [
      "A",
      "A+10",
      "A+15",
      "A+20",
      "A+25",
      "A+5",
      "A-10",
      "A-15",
      "A-20",
      "A-25",
      "A-5",
      "AA",
      "AA+1",
      "AA+1",
      "AA+2",
      "AA+2",
      "AA+3",
      "AA+3",
      "AA+4",
      "AA+4",
      "AA+5",
      "AA+5",
      "AA+6",
      "AA+7",
      "AA-1",
      "AA-1",
      "AA-2",
      "AA-2",
      "AA-5",
      "B",
      "C",
      "B",
      "B+10",
      "B+15",
      "B+20",
      "B+25",
      "B+5",
      "B-10",
      "B-5",
      "C",
      "C",
      "C+10",
      "C+15",
      "C+20",
      "C+25",
      "C+5",
      "C-10",
      "C-5",
      "D",
      "D+10",
      "D+5",
      "D-10",
      "D-5",
      "E",
      "E+5",
      "E-10",
      "E-15",
      "E-20",
      "E-25",
      "E-30",
      "E-35",
      "E-40",
      "E-45",
      "E-5",
      "AA",
      "D",
      "B",
      "C",
      "A",
      "AA",
      "E",
      "E",
      "A"
    ],
    "bldg_quality_txt_score_split_after": [
      "180%",
      "190%",
      "195%",
      "200%",
      "205%",
      "185%",
      "170%",
      "165%",
      "160%",
      "155%",
      "175%",
      "275%",
      "325%",
      "350%",
      "375%",
      "400%",
      "425%",
      "450%",
      "475%",
      "500%",
      "300%",
      "550%",
      "600%",
      "625%",
      "230%",
      "245%",
      "210%",
      "220%",
      "260%",
      "125%",
      "100%",
      "125%",
      "135%",
      "140%",
      "145%",
      "150%",
      "130%",
      "115%",
      "120%",
      "100%",
      "105%",
      "110%",
      "114%",
      "119%",
      "124%",
      "105%",
      "90%",
      "95%",
      "75%",
      "85%",
      "80%",
      "60%",
      "70%",
      "60%",
      "65%",
      "50%",
      "45%",
      "40%",
      "35%",
      "30%",
      "25%",
      "20%",
      "15%",
      "55%",
      "275%",
      "75%",
      "125%",
      "100%",
      "180%",
      "275%",
      "60%",
      "60%",
      "180%"
    ],
    "bldg_quality_txt_plain": [
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "AA",
      "B",
      "C",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "C",
      "C",
      "C",
      "C",
      "C",
      "C",
      "C",
      "C",
      "C",
      "D",
      "D",
      "D",
      "D",
      "D",
      "E",
      "E",
      "E",
      "E",
      "E",
      "E",
      "E",
      "E",
      "E",
      "E",
      "E",
      "AA",
      "D",
      "B",
      "C",
      "A",
      "AA",
      "E",
      "E",
      "A"
    ],
    "bldg_quality_num_plain": [
      180,
      190,
      195,
      200,
      205,
      185,
      170,
      165,
      160,
      155,
      175,
      275,
      325,
      350,
      375,
      400,
      425,
      450,
      475,
      500,
      300,
      550,
      600,
      625,
      230,
      245,
      210,
      220,
      260,
      125,
      100,
      125,
      135,
      140,
      145,
      150,
      130,
      115,
      120,
      100,
      105,
      110,
      114,
      119,
      124,
      105,
      90,
      95,
      75,
      85,
      80,
      60,
      70,
      60,
      65,
      50,
      45,
      40,
      35,
      30,
      25,
      20,
      15,
      55,
      275,
      75,
      125,
      100,
      180,
      275,
      60,
      60,
      180
    ],
    "bldg_quality_num": [
      1.8,
      1.9,
      1.95,
      2,
      2.05,
      1.85,
      1.7,
      1.65,
      1.6,
      1.55,
      1.75,
      2.75,
      3.25,
      3.5,
      3.75,
      4,
      4.25,
      4.5,
      4.75,
      5,
      3,
      5.5,
      6,
      6.25,
      2.3,
      2.45,
      2.1,
      2.2,
      2.6,
      1.25,
      1,
      1.25,
      1.35,
      1.4,
      1.45,
      1.5,
      1.3,
      1.15,
      1.2,
      1,
      1.05,
      1.1,
      1.14,
      1.19,
      1.24,
      1.05,
      0.9,
      0.95,
      0.75,
      0.85,
      0.8,
      0.6,
      0.7,
      0.6,
      0.65,
      0.5,
      0.45,
      0.4,
      0.35,
      0.3,
      0.25,
      0.2,
      0.15,
      0.55,
      2.75,
      0.75,
      1.25,
      1,
      1.8,
      2.75,
      0.6,
      0.6,
      1.8
    ]
  }

  df = pd.DataFrame(data=data)
  df_expected = pd.DataFrame(data=data_expected)

  qual_to_perc_map = {
    "Excellent": "AA  275%",
    "Highest": "AA  275%",

    "High": "A  180%",
    "Very Good": "A  180%",

    "Good": "B  125%",
    "Above Average": "B  125%",

    "Average": "C  100%",
    "GRADE": "C  100%",

    "Fair": "D  75%",

    "Low": "E  60%",
    "Lowest": "E  60%"
  }

  calc = {
    "bldg_quality_txt_score": ["map", "BLDG_DESC", qual_to_perc_map],
    "bldg_quality_txt_score_split_before": ["split_before", ["map", "BLDG_DESC", qual_to_perc_map], "str:  "],
    "bldg_quality_txt_score_split_after": ["split_after", ["map", "BLDG_DESC", qual_to_perc_map], "str:  "],
    "bldg_quality_txt_plain": [
      "split_before", ["split_before", ["split_before", ["map", "BLDG_DESC", qual_to_perc_map], "str:  "], "str:-"], "str:+"
    ],
    "bldg_quality_num_plain":[
      "asint", ["replace", ["split_after", ["map", "BLDG_DESC", qual_to_perc_map], "str:  "], {"%": ""}]
    ],
    "bldg_quality_num":[
      "/", ["asfloat", ["replace", ["split_after", ["map", "BLDG_DESC", qual_to_perc_map], "str:  "], {"%": ""}]], 100
    ]
  }

  pd.set_option("display.max_columns", None)
  df = perform_calculations(df, calc)

  assert dfs_are_equal(df, df_expected)


def test_split_condition():
  data = {
    "CONDITION": [
      "A  0%",
      "A  1%",
      "A  10%",
      "A  11%",
      "A  12%",
      "A  13%",
      "A  14%",
      "A  15%",
      "A  16%",
      "A  17%",
      "A  18%",
      "A  19%",
      "A  2%",
      "A  20%",
      "A  21%",
      "A  22%",
      "A  23%",
      "A  24%",
      "A  25%",
      "A  26%",
      "A  27%",
      "A  28%",
      "A  29%",
      "A  3%",
      "A  30%",
      "A  31%",
      "A  32%",
      "A  33%",
      "A  34%",
      "A  35%",
      "A  36%",
      "A  37%",
      "A  38%",
      "A  39%",
      "A  4%",
      "A  40%",
      "A  41%",
      "A  42%",
      "A  43%",
      "A  44%",
      "A  45%",
      "A  46%",
      "A  47%",
      "A  48%",
      "A  49%",
      "A  5%",
      "A  50%",
      "A  51%",
      "A  52%",
      "A  53%",
      "A  54%",
      "A  55%",
      "A  56%",
      "A  57%",
      "A  58%",
      "A  59%",
      "A  6%",
      "A  60%",
      "A  61%",
      "A  62%",
      "A  63%",
      "A  64%",
      "A  65%",
      "A  66%",
      "A  67%",
      "A  68%",
      "A  69%",
      "A  7%",
      "A  70%",
      "A  8%",
      "A  9%",
      "A 1.00",
      "A 10.00",
      "A 100.00",
      "A 11.00",
      "A 12.00",
      "A 13.00",
      "A 14.00",
      "A 15.00",
      "A 16.00",
      "A 17.00",
      "A 18.00",
      "A 19.00",
      "A 2.00",
      "A 20.00",
      "A 21.00",
      "A 22.00",
      "A 23.00",
      "A 24.00",
      "A 25.00",
      "A 26.00",
      "A 27.00",
      "A 28.00",
      "A 29.00",
      "A 3.00",
      "A 30.00",
      "A 31.00",
      "A 32.00",
      "A 33.00",
      "A 34.00",
      "A 35.00",
      "A 36.00",
      "A 37.00",
      "A 38.00",
      "A 39.00",
      "A 4.00",
      "A 40.00",
      "A 41.00",
      "A 42.00",
      "A 43.00",
      "A 44.00",
      "A 45.00",
      "A 46.00",
      "A 47.00",
      "A 48.00",
      "A 49.00",
      "A 5.00",
      "A 50.00",
      "A 51.00",
      "A 52.00",
      "A 53.00",
      "A 54.00",
      "A 55.00",
      "A 56.00",
      "A 57.00",
      "A 58.00",
      "A 59.00",
      "A 6.00",
      "A 60.00",
      "A 61.00",
      "A 62.00",
      "A 63.00",
      "A 64.00",
      "A 65.00",
      "A 66.00",
      "A 67.00",
      "A 68.00",
      "A 69.00",
      "A 7.00",
      "A 70.00",
      "A 8.00",
      "A 9.00",
      "B  0%",
      "B  1%",
      "B  10%",
      "B  11%",
      "B  16%",
      "B  18%",
      "B  2%",
      "B  22%",
      "B  24%",
      "B  28%",
      "B  30%",
      "B  31%",
      "B  36%",
      "B  38%",
      "B  4%",
      "B  52%",
      "B  54%",
      "B  58%",
      "B  6%",
      "B  62%",
      "B  8%",
      "MSVPO - COMM ONLY 61.00"
    ],
  }

  data_expected = {
    "CONDITION": data["CONDITION"],
    "bldg_condition_txt_split_before": [
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "A",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "B",
      "MSVPO_COMM_ONLY"
    ],
    "bldg_condition_txt_split_after": [
      "0%",
      "1%",
      "10%",
      "11%",
      "12%",
      "13%",
      "14%",
      "15%",
      "16%",
      "17%",
      "18%",
      "19%",
      "2%",
      "20%",
      "21%",
      "22%",
      "23%",
      "24%",
      "25%",
      "26%",
      "27%",
      "28%",
      "29%",
      "3%",
      "30%",
      "31%",
      "32%",
      "33%",
      "34%",
      "35%",
      "36%",
      "37%",
      "38%",
      "39%",
      "4%",
      "40%",
      "41%",
      "42%",
      "43%",
      "44%",
      "45%",
      "46%",
      "47%",
      "48%",
      "49%",
      "5%",
      "50%",
      "51%",
      "52%",
      "53%",
      "54%",
      "55%",
      "56%",
      "57%",
      "58%",
      "59%",
      "6%",
      "60%",
      "61%",
      "62%",
      "63%",
      "64%",
      "65%",
      "66%",
      "67%",
      "68%",
      "69%",
      "7%",
      "70%",
      "8%",
      "9%",
      "1.00",
      "10.00",
      "100.00",
      "11.00",
      "12.00",
      "13.00",
      "14.00",
      "15.00",
      "16.00",
      "17.00",
      "18.00",
      "19.00",
      "2.00",
      "20.00",
      "21.00",
      "22.00",
      "23.00",
      "24.00",
      "25.00",
      "26.00",
      "27.00",
      "28.00",
      "29.00",
      "3.00",
      "30.00",
      "31.00",
      "32.00",
      "33.00",
      "34.00",
      "35.00",
      "36.00",
      "37.00",
      "38.00",
      "39.00",
      "4.00",
      "40.00",
      "41.00",
      "42.00",
      "43.00",
      "44.00",
      "45.00",
      "46.00",
      "47.00",
      "48.00",
      "49.00",
      "5.00",
      "50.00",
      "51.00",
      "52.00",
      "53.00",
      "54.00",
      "55.00",
      "56.00",
      "57.00",
      "58.00",
      "59.00",
      "6.00",
      "60.00",
      "61.00",
      "62.00",
      "63.00",
      "64.00",
      "65.00",
      "66.00",
      "67.00",
      "68.00",
      "69.00",
      "7.00",
      "70.00",
      "8.00",
      "9.00",
      "0%",
      "1%",
      "10%",
      "11%",
      "16%",
      "18%",
      "2%",
      "22%",
      "24%",
      "28%",
      "30%",
      "31%",
      "36%",
      "38%",
      "4%",
      "52%",
      "54%",
      "58%",
      "6%",
      "62%",
      "8%",
      "61.00",
    ],
    "bldg_condition_num_plain": [
      0,
      1,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      2,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      3,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      4,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      5,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      6,
      60,
      61,
      62,
      63,
      64,
      65,
      66,
      67,
      68,
      69,
      7,
      70,
      8,
      9,
      1,
      10,
      100,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      2,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      3,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      4,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      5,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      6,
      60,
      61,
      62,
      63,
      64,
      65,
      66,
      67,
      68,
      69,
      7,
      70,
      8,
      9,
      0,
      1,
      10,
      11,
      16,
      18,
      2,
      22,
      24,
      28,
      30,
      31,
      36,
      38,
      4,
      52,
      54,
      58,
      6,
      62,
      8,
      61,
    ],
    "bldg_condition_num": [
      0,
      0.01,
      0.1,
      0.11,
      0.12,
      0.13,
      0.14,
      0.15,
      0.16,
      0.17,
      0.18,
      0.19,
      0.02,
      0.2,
      0.21,
      0.22,
      0.23,
      0.24,
      0.25,
      0.26,
      0.27,
      0.28,
      0.29,
      0.03,
      0.3,
      0.31,
      0.32,
      0.33,
      0.34,
      0.35,
      0.36,
      0.37,
      0.38,
      0.39,
      0.04,
      0.4,
      0.41,
      0.42,
      0.43,
      0.44,
      0.45,
      0.46,
      0.47,
      0.48,
      0.49,
      0.05,
      0.5,
      0.51,
      0.52,
      0.53,
      0.54,
      0.55,
      0.56,
      0.57,
      0.58,
      0.59,
      0.06,
      0.6,
      0.61,
      0.62,
      0.63,
      0.64,
      0.65,
      0.66,
      0.67,
      0.68,
      0.69,
      0.07,
      0.7,
      0.08,
      0.09,
      0.01,
      0.1,
      1,
      0.11,
      0.12,
      0.13,
      0.14,
      0.15,
      0.16,
      0.17,
      0.18,
      0.19,
      0.02,
      0.2,
      0.21,
      0.22,
      0.23,
      0.24,
      0.25,
      0.26,
      0.27,
      0.28,
      0.29,
      0.03,
      0.3,
      0.31,
      0.32,
      0.33,
      0.34,
      0.35,
      0.36,
      0.37,
      0.38,
      0.39,
      0.04,
      0.4,
      0.41,
      0.42,
      0.43,
      0.44,
      0.45,
      0.46,
      0.47,
      0.48,
      0.49,
      0.05,
      0.5,
      0.51,
      0.52,
      0.53,
      0.54,
      0.55,
      0.56,
      0.57,
      0.58,
      0.59,
      0.06,
      0.6,
      0.61,
      0.62,
      0.63,
      0.64,
      0.65,
      0.66,
      0.67,
      0.68,
      0.69,
      0.07,
      0.7,
      0.08,
      0.09,
      0,
      0.01,
      0.1,
      0.11,
      0.16,
      0.18,
      0.02,
      0.22,
      0.24,
      0.28,
      0.3,
      0.31,
      0.36,
      0.38,
      0.04,
      0.52,
      0.54,
      0.58,
      0.06,
      0.62,
      0.08,
      0.61
    ]
  }

  df = pd.DataFrame(data=data)
  df_expected = pd.DataFrame(data=data_expected)

  cond_to_perc_map = {
    "MSVPO - COMM ONLY 61.00": "MSVPO_COMM_ONLY 61.00"
  }

  calc = {
    "bldg_condition_txt_split_before": ["split_before", ["replace", "CONDITION", cond_to_perc_map], "str: "],
    "bldg_condition_txt_split_after": ["strip", ["split_after", ["replace", "CONDITION", cond_to_perc_map], "str: "]],
    "bldg_condition_num_plain":[
      "asint", ["replace", ["strip", ["split_after", ["replace", "CONDITION", cond_to_perc_map], "str: "]], {"%":""}],
    ],
    "bldg_condition_num":[
      "/", ["asint", ["replace", ["strip", ["split_after", ["replace", "CONDITION", cond_to_perc_map], "str: "]], {"%":""}]], 100
    ]
  }

  pd.set_option("display.max_columns", None)
  df = perform_calculations(df, calc)

  assert dfs_are_equal(df, df_expected)


def test_tweaks():
  data = {
    "id": [0, 1, 2, 3, 4, 5],
    "fruit": ["apple", "banana", "cherry", "date", "elderberry", "crapple"],
    "fruit_score": ["A  100%", "B  80%", "C  60%", "D  40%", "E  20%", "F  0%"],
  }
  df = pd.DataFrame(data=data)
  expected = {
    "id": [0, 1, 2, 3, 4, 5],
    "fruit": ["crapple", "banana", "cherry", "date", "apple", "strawberry"],
    "fruit_score": ["F  0%", "B  80%", "C  60%", "D  40%", "A  100%", "AA  120%"],
  }
  df_expected = pd.DataFrame(data=expected)

  tweaks = [
    {
      "field": "fruit",
      "key": "id",
      "values": {
        0: "crapple",
        4: "apple",
        5: "strawberry"
      }
    },
    {
      "field": "fruit_score",
      "key": "id",
      "values": {
        0: "F  0%",
        4: "A  100%",
        5: "AA  120%"
      }
    }
  ]

  df_result = perform_tweaks(df, tweaks)
  assert dfs_are_equal(df_result, df_expected)