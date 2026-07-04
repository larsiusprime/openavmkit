import numpy as np
import pandas as pd

from openavmkit.cleaning import _fill_unknown_values


def test_residual_categorical_fill_uses_unknown_not_nan():
  df = pd.DataFrame(
    {
      "key": ["a", "b", "c"],
      "bldg_style": ["RAMBLER", np.nan, "SPLIT"],
    }
  )
  settings = {
    "field_classification": {
      "impr": {"categorical": ["bldg_style"]}
    }
  }
  out = _fill_unknown_values(df, settings)
  values = set(out["bldg_style"].astype(str))
  assert "UNKNOWN" in values
  assert "nan" not in values
  assert "<NA>" not in values
