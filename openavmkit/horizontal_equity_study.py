import numpy as np
import pandas as pd
import openavmkit.utilities.stats as stats
from openavmkit.data import SalesUniversePair
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import do_per_model_group


class HorizontalEquitySummary:
	rows: int
	clusters: int
	min_chd: float
	max_chd: float
	median_chd: float

	def __init__(
			self,
			rows: int,
			clusters: int,
			min_chd: float,
			max_chd: float,
			median_chd: float
	):
		self.rows = rows
		self.clusters = clusters
		self.min_chd = min_chd
		self.max_chd = max_chd
		self.median_chd = median_chd


class HorizontalEquityClusterSummary:
	id: str
	count: int
	chd: float
	min: float
	max: float
	median: float

	def __init__(
			self,
			id: str,
			count: int,
			chd: float,
			min: float,
			max: float,
			median: float
	):
		self.id = id
		self.count = count
		self.chd = chd
		self.min = min
		self.max = max
		self.median = median


class HorizontalEquityStudy:
	summary: HorizontalEquitySummary
	cluster_summaries: dict[str, HorizontalEquityClusterSummary]

	def __init__(
			self,
			df: pd.DataFrame,
			field_cluster: str,
			field_value: str
	):
		clusters = df[field_cluster].unique()
		self.cluster_summaries = {}

		chds = np.array([])
		for cluster in clusters:
			df_cluster = df[df[field_cluster].eq(cluster)]
			count = len(df_cluster)
			if count > 0:
				chd = stats.calc_cod(df_cluster[field_value].values)
				min_value = df_cluster[field_value].min()
				max_value = df_cluster[field_value].max()
				median_value = df_cluster[field_value].median()
			else:
				chd = float('nan')
				min_value = float('nan')
				max_value = float('nan')
				median_value = float('nan')
			summary = HorizontalEquityClusterSummary(cluster, count, chd, min_value, max_value, median_value)
			self.cluster_summaries[cluster] = summary
			chds = np.append(chds, chd)

		if len(chds) > 0:
			min_chd = np.min(chds)
			max_chd = np.max(chds)
			med_chd = float(np.median(chds))
		else:
			min_chd = float('nan')
			max_chd = float('nan')
			med_chd = float('nan')

		self.summary = HorizontalEquitySummary(
			len(df),
			len(clusters),
			min_chd,
			max_chd,
			med_chd
		)


def _mark_he_ids(df_in: pd.DataFrame, model_group: str, settings: dict, verbose: bool):
	df = mark_horizontal_equity_clusters(df_in, settings, verbose)
	df["he_id"] = model_group + "_" + df["he_id"]


def mark_horizontal_equity_clusters_per_model_group_sup(
		sup: SalesUniversePair,
		settings: dict,
		verbose: bool = False
):
	df_universe = sup["universe"]
	df_universe = mark_horizontal_equity_clusters(df_universe, settings, verbose)
	sup.set("universe", df_universe)
	return sup


def mark_horizontal_equity_clusters_per_model_group(df_in: pd.DataFrame, settings: dict, verbose: bool = False):
	return do_per_model_group(df_in, _mark_he_ids, {"settings": settings, "verbose": verbose})


def mark_horizontal_equity_clusters(df: pd.DataFrame, settings: dict, verbose: bool = False):
	he = settings.get("analysis", {}).get("horizontal_equity", {})
	location = he.get("location", "neighborhood")
	fields_categorical = he.get("fields_categorical", [])
	fields_numeric = he.get("fields_numeric", None)
	df["he_id"], _ = make_clusters(df, location, fields_categorical, fields_numeric, verbose=verbose)
	return df
