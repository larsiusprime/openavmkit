import numpy as np
import pandas as pd

from openavmkit.utilities.data import div_z_safe


def test_div_z_safe():
	print("")
	df = pd.DataFrame({
		"numerator": [1, 2, 3, 4, 5],
		"denominator": [0, 1, 2, 0, 4]
	})
	result = div_z_safe(df, "numerator", "denominator")
	assert result.isna().sum() == 2
	assert result.astype(str).eq(["nan","2.0","1.5","nan","1.25"]).all()