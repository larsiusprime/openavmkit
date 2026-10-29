"""Time adjustment: a user's own file (a price index by day, month, quarter or year), and borrowing another model
group's time adjustment."""

import numpy as np
import pandas as pd

from openavmkit.time_adjustment import _cover, apply_time_adjustment, read_schedule


def _write(tmp_path, text):
    path = tmp_path / "ta.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_quarterly_index_becomes_a_daily_schedule(tmp_path):
    df, problems = read_schedule(_write(tmp_path, "period,index\n2022-Q1,100\n2022-Q2,105\n2022-Q3,110\n"))
    assert problems == [] and df is not None
    assert df["period"].iloc[0] == "2022-01-01" and df["period"].iloc[-1] == "2022-07-01"
    assert len(df) == 182  # every day, Jan 1 to Jul 1
    # Prices are brought to the last period: a sale on the first day is worth 10% more by then.
    assert np.isclose(df["correction_factor"].iloc[0], 1.10)
    assert np.isclose(df["correction_factor"].iloc[-1], 1.0)
    # Interpolated between periods: halfway through Q1-Q2, halfway between 100 and 105.
    mid = df.set_index("period").loc["2022-02-15", "start_indexed"]
    assert 1.02 < mid < 1.03


def test_months_years_and_openavmkits_own_schedule_are_read(tmp_path):
    df, problems = read_schedule(_write(tmp_path, "Period,Index\n2022-01,1.0\n2022-02,1.01\n"))
    assert problems == [] and df is not None and len(df) == 32
    df, problems = read_schedule(_write(tmp_path, "period,index\n2021,100\n2022,107\n"))
    assert problems == [] and df is not None and np.isclose(df["correction_factor"].iloc[0], 1.07)
    own = "period,start_indexed,end_indexed,correction_factor\n2022-01-01,1.0,0.5,2.0\n2022-01-03,2.0,1.0,1.0\n"
    df, problems = read_schedule(_write(tmp_path, own))
    assert problems == [] and df is not None and len(df) == 3


def test_every_problem_in_a_file_is_said(tmp_path):
    df, problems = read_schedule(_write(tmp_path, "date,price\n2022-01,1\n"))
    assert df is None and len(problems) == 2  # no period, no index
    text = "period,index\n2022-Q1,100\nlast spring,101\n2022-Q1,-3\n2022-Q3,abc\n"
    df, problems = read_schedule(_write(tmp_path, text))
    assert df is None
    joined = " ".join(problems)
    assert "last spring" in joined and "-3" in joined and "abc" in joined and "more than once" in joined


def test_held_flat_outside_its_dates():
    df = pd.DataFrame(
        {
            "period": ["2022-01-01", "2022-01-02"],
            "start_indexed": [1.0, 1.1],
            "end_indexed": [1 / 1.1, 1.0],
            "correction_factor": [1.1, 1.0],
        }
    )
    out = _cover(df, pd.Series(pd.to_datetime(["2021-12-30", "2022-01-05"])))
    assert out["period"].iloc[0] == "2021-12-30" and out["period"].iloc[-1] == "2022-01-05"
    assert np.isclose(out["correction_factor"].iloc[0], 1.1) and np.isclose(out["correction_factor"].iloc[-1], 1.0)


def _sales():
    days = pd.to_datetime(["2022-01-10", "2022-03-10", "2022-05-10", "2022-06-20"])
    return pd.DataFrame(
        {
            "key_sale": ["a1", "a2", "b1", "b2"],
            "model_group": ["a", "a", "b", "b"],
            "sale_date": days,
            "sale_price": [100_000.0] * 4,
        }
    )


def test_a_group_borrows_another_groups_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # each group's schedule is written under out/
    path = _write(tmp_path, "period,index\n2022-01,100\n2022-07,120\n")
    settings = {
        "modeling": {"model_groups": {"a": {}, "b": {}}},
        "data": {"process": {"time_adjustment": {"from_file": {"a": path}, "borrow": {"b": "a"}}}},
    }
    out = apply_time_adjustment(_sales(), settings).set_index("key_sale")
    # a's own file and b's borrowed one are the same schedule: same date, same adjustment.
    assert out.loc["a1", "sale_price_time_adj"] > 100_000
    a2, b1 = out.loc["a2", "sale_price_time_adj"], out.loc["b1", "sale_price_time_adj"]
    assert a2 > b1 > 100_000  # earlier sales are brought forward more
    assert np.isclose(out.loc["b2", "sale_price_time_adj"], 100_000 * 120 / (100 + 20 * 170 / 181), rtol=1e-3)
    assert (tmp_path / "out" / "time_adjustment" / "b" / "time_adjustment_schedule.csv").is_file()
