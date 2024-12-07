from IPython.core.display_functions import display

from openavmkit.horizontal_equity_study import cluster_by_location_and_big_five
from openavmkit.synthetic_data import generate_basic


def test_clusters():
	print("")
	df = generate_basic(100)
	df["he_id"] = cluster_by_location_and_big_five(
		df,
		"neighborhood",
		[]
	)
	display(df["he_id"].value_counts())