"""data.load casts CSV columns to bool the same way as other formats."""

import pandas as pd

from openavmkit.data import load_dataframe


def test_csv_yes_no_column_loads_as_bool(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	(tmp_path / "in").mkdir()
	pd.DataFrame({"PIN": ["1", "2", "3", "4"], "VAC": ["Y", "N", "", "yes"]}).to_csv(tmp_path / "in" / "cama.csv", index=False)
	for na_handling, blank in (("na_true", True), ("na_false", False)):
		entry = {"filename": "cama.csv", "load": {"key": ["PIN", "string"], "is_vacant": ["VAC", "bool", na_handling]}}
		df = load_dataframe(entry, {})
		assert df["is_vacant"].tolist() == [True, False, blank, True]  # the blank follows na_handling
