import numpy as np
import pandas as pd
import openavmkit.utilities.stats as stats
from openavmkit.data import SalesUniversePair
from openavmkit.utilities.assertions import dfs_are_equal
from openavmkit.utilities.cache import get_cached_df, write_cached_df
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import do_per_model_group


class HorizontalEquitySummary:
	"""
  Summary statistics for horizontal equity analysis.

  Attributes:
      rows (int): Total number of rows in the input DataFrame.
      clusters (int): Total number of clusters identified.
      min_chd (float): Minimum CHD (Coefficient of Horizontal Dispersion) value of any cluster.
      max_chd (float): Maximum CHD value of any cluster.
      median_chd (float): Median CHD value of all clusters.
  """

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
		"""
    Initialize a HorizontalEquitySummary instance.

    :param rows: Total number of rows in the DataFrame.
    :type rows: int
    :param clusters: Total number of clusters.
    :type clusters: int
    :param min_chd: Minimum COD value.
    :type min_chd: float
    :param max_chd: Maximum COD value.
    :type max_chd: float
    :param median_chd: Median COD value.
    :type median_chd: float
    """
		self.rows = rows
		self.clusters = clusters
		self.min_chd = min_chd
		self.max_chd = max_chd
		self.median_chd = median_chd


class HorizontalEquityClusterSummary:
	"""
  Summary for an individual horizontal equity cluster.

  Attributes:
      id (str): Identifier of the cluster.
      count (int): Number of records in the cluster.
      chd (float): CHD value for the cluster.
      min (float): Minimum value in the cluster.
      max (float): Maximum value in the cluster.
      median (float): Median value in the cluster.
  """

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
		"""
    Initialize a HorizontalEquityClusterSummary instance.

    :param id: Cluster identifier.
    :type id: str
    :param count: Number of records in the cluster.
    :type count: int
    :param chd: COD value for the cluster.
    :type chd: float
    :param min: Minimum value in the cluster.
    :type min: float
    :param max: Maximum value in the cluster.
    :type max: float
    :param median: Median value in the cluster.
    :type median: float
    """
		self.id = id
		self.count = count
		self.chd = chd
		self.min = min
		self.max = max
		self.median = median


class HorizontalEquityStudy:
	"""
  Perform horizontal equity analysis and summarize the results.

  Attributes:
      summary (HorizontalEquitySummary): Overall summary statistics.
      cluster_summaries (dict[str, HorizontalEquityClusterSummary]): Dictionary mapping cluster IDs to their summaries.
  """

	summary: HorizontalEquitySummary
	cluster_summaries: dict[str, HorizontalEquityClusterSummary]

	def __init__(
			self,
			df: pd.DataFrame,
			field_cluster: str,
			field_value: str
	):
		"""
    Initialize a HorizontalEquityStudy instance by computing cluster summaries.

    :param df: Input DataFrame containing data for horizontal equity analysis.
    :type df: pandas.DataFrame
    :param field_cluster: Column name indicating cluster membership.
    :type field_cluster: str
    :param field_value: Column name of the values to analyze.
    :type field_value: str
    """
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


def mark_horizontal_equity_clusters_per_model_group_sup(
		sup: SalesUniversePair,
		settings: dict,
		verbose: bool = False,
		use_cache: bool = True
):
	"""
  Mark horizontal equity clusters on the 'universe' DataFrame of a SalesUniversePair.

  Updates the 'universe' DataFrame with horizontal equity clusters by calling
  :func:`mark_horizontal_equity_clusters` and then sets the updated DataFrame in sup.

  :param sup: SalesUniversePair containing sales and universe data.
  :type sup: SalesUniversePair
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :returns: Updated SalesUniversePair with marked horizontal equity clusters.
  :rtype: SalesUniversePair
  """
	df_universe = sup["universe"]
	if verbose:
		print("")
		print("Marking horizontal equity clusters...")
	df_universe = mark_horizontal_equity_clusters_per_model_group(df_universe, settings, verbose, output_folder="horizontal_equity/general", use_cache=use_cache)
	if verbose:
		print("")
		print("Marking LAND horizontal equity clusters...")
	df_universe = mark_horizontal_equity_clusters_per_model_group(df_universe, settings, verbose, settings_object="land_equity", id_name="land_he_id", output_folder="horizontal_equity/land", use_cache=use_cache)
	if verbose:
		print("")
		print("Marking IMPROVEMENT horizontal equity clusters...")
	df_universe = mark_horizontal_equity_clusters_per_model_group(df_universe, settings, verbose, settings_object="impr_equity", id_name="impr_he_id", output_folder="horizontal_equity/improvement", use_cache=use_cache)
	sup.set("universe", df_universe)
	return sup


def mark_horizontal_equity_clusters_per_model_group(df_in: pd.DataFrame, settings: dict, verbose: bool = False, settings_object="horizontal_equity", id_name="he_id", output_folder="", use_cache=True):
	"""
  Mark horizontal equity clusters for each model group within the DataFrame.

  Applies the :func:`_mark_he_ids` function on each model group subset using :func:`do_per_model_group`.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :param settings_object: The settings object to use for horizontal equity analysis.
  :type settings_object: str, optional
  :param id_name: Name of the column to store the horizontal equity cluster ID.
  :type id_name: str, optional
  :param output_folder: Output folder path (stores information about the clusters for later use).
  :type output_folder: str, optional
  :param use_cache: If True, uses cached DataFrame if available.
  :type use_cache: bool, optional
  :returns: DataFrame with horizontal equity cluster IDs marked.
  :rtype: pandas.DataFrame
  """

	he = settings.get("analysis", {}).get(settings_object, {})
	if use_cache:
		df_out = get_cached_df(df_in, id_name, "key", he)
		if df_out is not None:
			return df_out

	df_out = do_per_model_group(df_in, settings, _mark_he_ids, params={
		"settings": settings, "verbose": verbose, "settings_object": settings_object, "id_name": id_name, "output_folder": output_folder
	}, key="key", verbose=verbose)

	if use_cache:
		df_result = write_cached_df(df_in, df_out, id_name, "key", he)
		assert dfs_are_equal(df_out, df_result)

	return df_out


def mark_horizontal_equity_clusters(df: pd.DataFrame, settings: dict, verbose: bool = False, settings_object="horizontal_equity", id_name: str = "he_id", output_folder: str = ""):
	"""
  Compute and mark horizontal equity clusters in the DataFrame.

  Uses clustering (via :func:`make_clusters`) based on a location field and categorical/numeric fields specified
  in settings to generate a horizontal equity cluster ID which is stored in the "he_id" column.

  :param df: Input DataFrame
  :type df: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :param settings_object: The settings object to use for horizontal equity analysis.
  :type settings_object: str, optional
  :param id_name: Name of the column to store the horizontal equity cluster ID.
  :type id_name: str, optional
  :param output_folder: Output folder path (stores information about the clusters for later use).
  :type output_folder: str, optional
  :returns: DataFrame with a new "he_id" column.
  :rtype: pandas.DataFrame
  """
	he = settings.get("analysis", {}).get(settings_object, {})
	location = he.get("location", None)
	fields_categorical = he.get("fields_categorical", [])
	fields_numeric = he.get("fields_numeric", None)
	df[id_name], _, _ = make_clusters(df, location, fields_categorical, fields_numeric, verbose=verbose, output_folder=output_folder)
	return df


def _mark_he_ids(df_in: pd.DataFrame, model_group: str, settings: dict, verbose: bool, settings_object="horizontal_equity", id_name: str = "he_id", output_folder: str = ""):
	"""
  Append the model group identifier to the horizontal equity cluster IDs.

  :param df_in: Input DataFrame with horizontal equity clusters already marked.
  :type df_in: pandas.DataFrame
  :param model_group: The model group identifier.
  :type model_group: str
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool
  :param settings_object: The settings object to use for horizontal equity analysis.
  :type settings_object: str, optional
  :param id_name: Name of the column to store the horizontal equity cluster ID.
  :type id_name: str, optional
  :param output_folder: Output folder path (stores information about the clusters for later use).
  :type output_folder: str, optional
  :returns: DataFrame with updated `id_name` column that includes the model group.
  :rtype: pandas.DataFrame
  """
	df = mark_horizontal_equity_clusters(df_in, settings, verbose, settings_object, id_name, output_folder)
	df.loc[:, id_name] = model_group + "_" + df[id_name].astype(str)
	return df
