from IPython.display import display

from openavmkit.data import SalesUniversePair
from openavmkit.horizontal_equity_study import mark_horizontal_equity_clusters_per_model_group_sup
from openavmkit.pipeline import load_settings
from openavmkit.synthetic.basic import generate_basic


def test_clusters():
	print("")
	sd = generate_basic(100)

	sup = SalesUniversePair(sd.df_sales, sd.df_universe)
	sup.universe["model_group"] = "test"

	settings = {
		"modeling":{
			"model_groups":{
				"test":{}
			}
		}
	}

	settings = load_settings("", settings)

	verbose=True

	sup = mark_horizontal_equity_clusters_per_model_group_sup(sup, settings, verbose=verbose)
	df = sup.universe

	display(df["he_id"].value_counts())