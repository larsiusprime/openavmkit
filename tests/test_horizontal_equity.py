from IPython.core.display_functions import display

from openavmkit.horizontal_equity_study import make_clusters, mark_horizontal_equity_clusters
from openavmkit.synthetic_data import generate_basic


def test_clusters():
	print("")
	sd = generate_basic(100)
	df = sd.df
	df = mark_horizontal_equity_clusters(df, {})
	display(df["he_id"].value_counts())