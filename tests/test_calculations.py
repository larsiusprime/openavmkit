import pandas as pd
from IPython.core.display_functions import display

from openavmkit.calculations import perform_calculations, crawl_calc_dict_for_fields
from openavmkit.filters import resolve_filter
from openavmkit.utilities.assertions import dfs_are_equal


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
  results = crawl_calc_dict_for_fields(calc)
  results.sort()
  expected = ['a', 'b', 'condition_num', 'quality_num', 'quality_txt']
  assert results == expected