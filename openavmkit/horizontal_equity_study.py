import numpy as np
import pandas as pd
import openavmkit.utilities.stats as stats
from openavmkit.utilities.clustering import make_clusters


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
			chd = stats.calc_cod(df_cluster[field_value].values)
			min_value = df_cluster[field_value].min()
			max_value = df_cluster[field_value].max()
			median_value = df_cluster[field_value].median()
			summary = HorizontalEquityClusterSummary(cluster, count, chd, min_value, max_value, median_value)
			self.cluster_summaries[cluster] = summary
			chds = np.append(chds, chd)

		self.summary = HorizontalEquitySummary(
			len(df),
			len(clusters),
			np.min(chds),
			np.max(chds),
			float(np.median(chds))
		)


def mark_horizontal_equity_clusters(df: pd.DataFrame, settings: dict, verbose: bool = False):
	he = settings.get("analysis", {}).get("horizontal_equity", {})
	location = he.get("location", "neighborhood")
	fields_categorical = he.get("fields_categorical", [])
	fields_numeric = he.get("fields_numeric", None)
	df["he_id"], _ = make_clusters(df, location, fields_categorical, fields_numeric, verbose=verbose)
	return df
