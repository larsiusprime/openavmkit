import pandas as pd

from openavmkit.model_runner import BenchmarkResults


def _stats() -> pd.DataFrame:
    return pd.DataFrame(
        {"count_sales": [1234.0], "cod": [12.3456], "vei": [-130.4]}, index=pd.Index(["mra"], name="model")
    )


def test_print_leaves_the_benchmark_numbers_alone():
    time = pd.DataFrame({"total": [1.0]}, index=pd.Index(["mra"], name="model"))
    b = BenchmarkResults(time, _stats(), None, _stats())
    first = b.print()
    assert b.df_stats_test.loc["mra", "count_sales"] == 1234.0
    assert b.df_stats_test.loc["mra", "cod"] == 12.3456
    assert b.df_time.loc["mra", "total"] == 1.0
    # Printing twice gives the same text, rather than failing on numbers already turned into text.
    assert b.print() == first
