import numpy as np
import pandas as pd
import warnings
import openavmkit.utilities.stats as stats
from openavmkit.data import SalesUniversePair
from openavmkit.utilities.assertions import dfs_are_equal
from openavmkit.utilities.cache import get_cached_df, write_cached_df
from openavmkit.utilities.clustering import make_clusters
from openavmkit.utilities.data import do_per_model_group
from openavmkit.utilities.timing import TimingData


class HorizontalEquitySummary:
    """
    Summary statistics for horizontal equity analysis.

    Attributes
    ----------
    rows : int
        Total number of rows in the input DataFrame.
    clusters : int
        Total number of clusters identified.
    min_chd : float
        Minimum CHD (Coefficient of Horizontal Dispersion) value of any cluster.
    max_chd : float
        Maximum CHD value of any cluster.
    median_chd : float
        Median CHD value of all clusters.
    p05_chd : float
        5th percentile CHD value
    p25_chd : float
        25th percentile CHD value    
    p75_chd : float
        75th percentile CHD value
    p95_chd : float
        95th percentile CHD value
    """

    def __init__(
        self,
        rows: int,
        clusters: int,
        min_chd: float,
        max_chd: float,
        median_chd: float,
        p05_chd: float,
        p25_chd: float,
        p75_chd: float,
        p95_chd: float
    ):
        """
        Initialize a HorizontalEquitySummary instance.

        Parameters
        ----------
        rows : int
            Total number of rows in the DataFrame.
        clusters : int
            Total number of clusters.
        min_chd : float
            Minimum CHD value.
        max_chd : float
            Maximum CHD value.
        median_chd : float
            Median CHD value.
        p05_chd : float
            5th percentile CHD value
        p25_chd : float
            25th percentile CHD value
        p75_chd : float
            75th percentile CHD value
        p95_chd : float
            95th percentile CHD value
        """
        self.rows = rows
        self.clusters = clusters
        self.min_chd = min_chd
        self.max_chd = max_chd
        self.median_chd = median_chd
        self.p05_chd = p05_chd
        self.p25_chd = p25_chd
        self.p75_chd = p75_chd
        self.p95_chd = p95_chd
    
    def print(self):
        data = {
            "Rows": [self.rows],
            "Clusters": [self.clusters],
            "Min CHD": [self.min_chd],
            "5th %ile CHD": [self.p05_chd],
            "25th %ile CHD": [self.p25_chd],
            "Median CHD": [self.median_chd],
            "75th %ile CHD": [self.p75_chd],
            "95th %ile CHD": [self.p95_chd],
            "Max CHD": [self.max_chd],
        }
        df = pd.DataFrame(data=data)
        chd_fields = [field for field in df.columns.values if "CHD" in field]
        for field in chd_fields:
            df[field] = df[field].astype(float).apply(lambda x: f"{x:0.2f}").astype("string")
        for field in ["Rows","Clusters"]:
            df[field] = df[field].astype(int).apply(lambda x: f"{x:,d}").astype("string")
        df.index = ["Statistic"]
        return df.transpose()


class HorizontalEquityClusterSummary:
    """
    Summary for an individual horizontal equity cluster.

    Attributes
    ----------
    id : str
        Identifier of the cluster.
    count : int
        Number of records in the cluster.
    chd : float
        CHD value for the cluster.
    min : float
        Minimum value in the cluster.
    max : float
        Maximum value in the cluster.
    median : float
        Median value in the cluster.
    """

    def __init__(
        self, id: str, count: int, chd: float, min: float, max: float, median: float
    ):
        """
        Initialize a HorizontalEquityClusterSummary instance.

        Parameters
        ----------
        id : str
            Cluster identifier.
        count : int
            Number of records in the cluster.
        chd : float
            COD value for the cluster.
        min : float
            Minimum value in the cluster.
        max : float
            Maximum value in the cluster.
        median : float
            Median value in the cluster.
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

    Attributes
    ----------
    summary : HorizontalEquitySummary
        Overall summary statistics.
    cluster_summaries : dict[str, HorizontalEquityClusterSummary]
        Dictionary mapping cluster IDs to their summaries.
    """

    def __init__(self, df: pd.DataFrame, field_cluster: str, field_value: str):
        """
        Initialize a HorizontalEquityStudy instance by computing cluster summaries.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing data for horizontal equity analysis.
        field_cluster : str
            Column name indicating cluster membership.
        field_value : str
            Column name of the values to analyze.
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
                chd = float("nan")
                min_value = float("nan")
                max_value = float("nan")
                median_value = float("nan")
            summary = HorizontalEquityClusterSummary(
                cluster, count, chd, min_value, max_value, median_value
            )
            self.cluster_summaries[cluster] = summary
            chds = np.append(chds, chd)

        if len(chds) > 0:
            min_chd = np.min(chds)
            max_chd = np.max(chds)
            med_chd = float(np.median(chds))
            p05_chd = np.quantile(chds, 0.05)
            p25_chd = np.quantile(chds, 0.25)
            p75_chd = np.quantile(chds, 0.75)
            p95_chd = np.quantile(chds, 0.95)
        else:
            min_chd = float("nan")
            max_chd = float("nan")
            med_chd = float("nan")
            p05_chd = float("nan")
            p25_chd = float("nan")
            p75_chd = float("nan")
            p95_chd = float("nan")

        self.summary = HorizontalEquitySummary(
            len(df), len(clusters), min_chd, max_chd, med_chd, p05_chd, p25_chd, p75_chd, p95_chd
        )


def mark_horizontal_equity_clusters_per_model_group_sup(
    sup: SalesUniversePair,
    settings: dict,
    verbose: bool = False,
    use_cache: bool = True,
    do_land_clusters: bool = True,
    do_impr_clusters: bool = True,
) -> SalesUniversePair:
    """
    Mark horizontal equity clusters on the 'universe' DataFrame of a SalesUniversePair.

    Updates the 'universe' DataFrame with horizontal equity clusters by calling
    `mark_horizontal_equity_clusters` and then sets the updated DataFrame in `sup`.

    Parameters
    ----------
    sup : SalesUniversePair
        SalesUniversePair containing sales and universe data.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.
    use_cache : bool, optional
        If True, uses cached DataFrame if available.
    do_land_clusters : bool, optional
        If True, marks land horizontal equity clusters.
    do_impr_clusters : bool, optional
        If True, marks improvement horizontal equity clusters.

    Returns
    -------
    SalesUniversePair
        Updated SalesUniversePair with marked horizontal equity clusters.
    """
    
    he = settings.get("analysis", {}).get("horizontal_equity", {})
    enabled = he.get("enabled", True)
    
    if enabled == False:
        if verbose:
            print(f"Skipping horizontal equity clustering...")
        return sup
        
    df_universe = sup["universe"]
    if verbose:
        print("")
        print("Marking horizontal equity clusters...")
    df_universe = _mark_horizontal_equity_clusters_per_model_group(
        df_universe,
        settings,
        verbose,
        output_folder="horizontal_equity/general",
        use_cache=use_cache,
    )
    if do_land_clusters:
        if verbose:
            print("")
            print("Marking LAND horizontal equity clusters...")
        le = settings.get("analysis", {}).get("land_equity", {})
        location = le.get("location", None)
        if location is None:
            warnings.warn("You are creating land equity clusters, but you haven't defined `data.analysis.land_equity.location`. You should at least provide a location field if you want to use this feature.")
        df_universe = _mark_horizontal_equity_clusters_per_model_group(
            df_universe,
            settings,
            verbose,
            settings_object="land_equity",
            id_name="land_he_id",
            output_folder="horizontal_equity/land",
            use_cache=use_cache,
        )
    if do_impr_clusters:
        if verbose:
            print("")
            print("Marking IMPROVEMENT horizontal equity clusters...")
        df_universe = _mark_horizontal_equity_clusters_per_model_group(
            df_universe,
            settings,
            verbose,
            settings_object="impr_equity",
            id_name="impr_he_id",
            output_folder="horizontal_equity/improvement",
            use_cache=use_cache,
        )
        sup.set("universe", df_universe)
    return sup


def mark_horizontal_equity_clusters(
    df: pd.DataFrame,
    settings: dict,
    verbose: bool = False,
    settings_object: str = "horizontal_equity",
    id_name: str = "he_id",
    output_folder: str = "",
    t: TimingData = None,
) -> pd.DataFrame:
    """
    Compute and mark horizontal equity clusters in the DataFrame.

    Uses clustering (via `make_clusters`) based on a location field and categorical/numeric
    fields specified in settings to generate a horizontal equity cluster ID which is stored
    in the specified `id_name` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.
    settings_object : str, optional
        The settings object to use for horizontal equity analysis.
    id_name : str, optional
        Name of the column to store the horizontal equity cluster ID.
    output_folder : str, optional
        Output folder path (stores information about the clusters for later use).
    t : TimingData, optional
        TimingData object to record performance metrics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new cluster ID column (`id_name`).
    """

    he = settings.get("analysis", {}).get(settings_object, {})
    location = he.get("location", None)
    fields_categorical = he.get("fields_categorical", [])
    fields_numeric = he.get("fields_numeric", None)
    
    split_on_vacant = True
    if "land" in id_name:
        split_on_vacant = False
    df[id_name], _, _ = make_clusters(
        df,
        location,
        fields_categorical,
        fields_numeric,
        split_on_vacant=split_on_vacant,
        verbose=verbose,
        output_folder=output_folder,
        t=t
    )
    return df


#######################################
# PRIVATE
#######################################


def _mark_horizontal_equity_clusters_per_model_group(
    df_in: pd.DataFrame,
    settings: dict,
    verbose: bool = False,
    settings_object: str = "horizontal_equity",
    id_name: str = "he_id",
    output_folder: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Mark horizontal equity clusters for each model group within the DataFrame.

    Applies the `_mark_he_ids` function on each model group subset using `do_per_model_group`.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Input DataFrame.
    settings : dict
        Settings dictionary.
    verbose : bool, optional
        If True, prints progress information.
    settings_object : str, optional
        The settings object to use for horizontal equity analysis.
    id_name : str, optional
        Name of the column to store the horizontal equity cluster ID.
    output_folder : str, optional
        Output folder path (stores information about the clusters for later use).
    use_cache : bool, optional
        If True, uses cached DataFrame if available.

    Returns
    -------
    pandas.DataFrame
        DataFrame with horizontal equity cluster IDs marked.
    """

    t = TimingData()
    t.start("all")

    he = settings.get("analysis", {}).get(settings_object, {})
    if use_cache:
        df_out = get_cached_df(df_in, id_name, "key", he)
        if df_out is not None:
            return df_out

    df_out = do_per_model_group(
        df_in,
        settings,
        _mark_he_ids,
        params={
            "settings": settings,
            "verbose": verbose,
            "settings_object": settings_object,
            "id_name": id_name,
            "output_folder": output_folder,
            "t": t,
        },
        key="key",
        instructions={"just_stomp_columns": [id_name]},
        verbose=verbose,
    )

    if use_cache:
        df_result = write_cached_df(df_in, df_out, id_name, "key", he)
        df_result = write_cached_df(
            df_in, 
            df_out, 
            id_name, 
            "key", 
            he,
            changed_cols=[id_name],  # we're just writing key + id, no need to guess (speedup)
            check_equal=False        # skip this check for speed purposes, it's safe
        )

    print("")
    print("TIMING: Mark Horizontal Equity Clusters")
    t.stop("all")
    print(t.print())
    print("")

    return df_out


def _mark_he_ids(
    df_in: pd.DataFrame,
    model_group: str,
    settings: dict,
    verbose: bool,
    settings_object="horizontal_equity",
    id_name: str = "he_id",
    output_folder: str = "",
    t: TimingData = None,
):
    """Append the model group identifier to the horizontal equity cluster IDs.

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
    df = mark_horizontal_equity_clusters(
        df_in, settings, verbose, settings_object, id_name, output_folder, t
    )
    df[id_name] = df[id_name].astype("string")
    df.loc[:, id_name] = model_group + "_" + df[id_name]
    return df
