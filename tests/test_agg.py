import pandas as pd
import warnings
from openavmkit.data import _handle_duplicated_rows
from openavmkit.utilities.assertions import dfs_are_equal
from io import StringIO

def test_agg():
    df = get_agg_csv()
    
    dupes = {
        "subset": ["key"],
        "sort_by": ["bldg_num", "asc"],
        "agg": {
            "stories": {
                "field": "stories",
                "op": "max"
            },
            "bldg_type": {
                "field": "bldg_type",
                "op": "first",
                "sort_by": ["gross_living_area", "desc"]
            },
            "bldg_rooms_bed": {
                "field": "bedrooms",
                "op": "sum"
            },
            "bldg_rooms_bath": {
                "field": "bathrooms",
                "op": "sum"
            },
            "bldg_quality_num": {
                "field": "quality",
                "op": "first",
                "sort_by": ["bldg_num", "asc"]
            },
            "bldg_year_built": {
                "field": "bldg_year_built",
                "op": "mean"
            },
            "bldg_effective_year_built": {
                "field": "bldg_effective_year_built",
                "op": "mean"
            }
        }
    }
    
    df_expected = get_agg_deduped_csv()
    df_results = _handle_duplicated_rows(df, dupes, True)
    
    assert dfs_are_equal(df_results, df_expected, primary_key="key")


def get_agg_deduped_csv():
    txt = """key,bldg_num,bldg_type,bldg_class,stories,bldg_rooms_bed,bldg_rooms_bath,bedrooms,bathrooms,bldg_quality_num,quality,actual_area,effective_area,heated_effective_area,gross_living_area,bldg_year_built,bldg_effective_year_built,depreciation
1,1,ranch,single family residential,1,10,10,5,7,3,3,21386,9168,5882,5882,1988.5,1996,0.77
2,1,colonial,single family residential,2,6,8,3,3.5,3,3,14363,5825,4061,4061,2010,2012,0.88
3,1,cape_cod,single family residential,1.5,3,4,3,4,5,5,9216,5255,4486,4239,1987,1987,0.554
4,1,ranch,single family residential,2,7,7,4,5,5,5,19503,9910,6239,5729,2003,2005,0.826
5,1,unknown,single family residential,3,4,4.5,4,4.5,3,3,0,0,4432,4432,1985,1985,0.746
6,1,key_west,single family residential,1.8,8,9.5,5,5,5,5,17965,8257,6751,6751,1995,1996,0.564
7,1,colonial,single family residential,2,7,11,3,5.5,5,5,11511,7192,5339,5302,2006.5,2007,0.88
8,1,cape_cod,single family residential,1.5,2,2,2,2,5,5,7737,3591,2752,2752,1984,1991,0.706
9,1,key_west,single family residential,1.5,4,3,4,3,5,5,9454,5552,4364,4364,1986,1998,0.762
10,1,cape_cod,single family residential,1.8,5,5.5,3,3.5,5,5,8195,4423,3438,3426,1985,1990.5,0.67
11,1,colonial,single family residential,2,6,7,3,4,4,4,11734,6850,4773,4773,1988.5,1991.5,0.7
12,1,colonial,single family residential,2,4,4.5,4,4.5,3,3,9832,5366,3877,3719,1990,1997,0.714
"""
    df = pd.read_csv(StringIO(txt))
    return df

    
def get_agg_csv():
    txt = """key,bldg_num,bldg_type,bldg_class,stories,bedrooms,bathrooms,quality,actual_area,effective_area,heated_effective_area,gross_living_area,bldg_year_built,bldg_effective_year_built,depreciation
1,1,ranch,single family residential,1,5,7,3,21386,9168,5882,5882,2001,2001,0.77
1,2,key_west,single family residential,1,5,3,3,4468,3172,2939,2538,1976,1991,0.67
2,1,ranch,single family residential,1,3,3.5,3,14363,5825,4061,4061,2012,2012,0.88
2,2,colonial,single family residential,2,3,4.5,4,12597,6524,5025,5025,2008,2012,0.88
3,1,cape_cod,single family residential,1.5,3,4,5,9216,5255,4486,4239,1987,1987,0.554
4,1,ranch,single family residential,1,4,5,5,19503,9910,6239,5729,2003,2006,0.826
4,2,colonial,single family residential,2,3,2,5,2672,2199,2099,2099,2003,2004,0.8
5,1,unknown,single family residential,3,4,4.5,3,0,0,4432,4432,1985,1985,0.746
6,1,key_west,single family residential,1.5,5,5,5,17965,8257,6751,6751,1991,1992,0.564
6,2,cape_cod,single family residential,1.8,3,4.5,4,12489,5965,4817,4817,1999,2000,0.786
7,1,colonial,single family residential,2,3,5.5,5,11511,7192,5339,5302,2004,2004,0.88
7,2,colonial,single family residential,2,4,5.5,5,20020,11100,8660,8660,2009,2010,0.86
8,1,cape_cod,single family residential,1.5,2,2,5,7737,3591,2752,2752,1984,1991,0.706
9,1,key_west,single family residential,1.5,4,3,5,9454,5552,4364,4364,1986,1998,0.762
10,1,cape_cod,single family residential,1.5,3,3.5,5,8195,4423,3438,3426,1980,1991,0.67
10,2,cape_cod,single family residential,1.8,2,2,4,7307,1749,1118,1118,1990,1990,0.66
11,1,colonial,single family residential,2,3,4,4,11734,6850,4773,4773,1992,1994,0.7
11,2,contemporary,single family residential,1,3,3,4,6819,3057,2689,2667,1985,1989,0.69
12,1,colonial,single family residential,2,4,4.5,3,9832,5366,3877,3719,1990,1997,0.714
"""
    df = pd.read_csv(StringIO(txt))
    return df
