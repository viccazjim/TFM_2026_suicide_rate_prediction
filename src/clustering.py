"""
Unsupervised clustering — validates (or refutes) the a priori EU_REGIONS
grouping used for descriptive purposes in 02_eda.py, and provides a
standalone unsupervised analysis of the 27 EU countries.

Kept separate from models.py: that module is about supervised
prediction of the target; the descriptive half of this one never
touches "Suicide rate" as something to predict — only as one more
property of a country to describe it by, alongside socioeconomic
indicators.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from scipy.cluster.hierarchy import linkage, fcluster


def aggregate_country_features(
    df: pd.DataFrame,
    predictor_features: list[str],
    target: str = None,
    year_col: str = "Year",
    id_col: str = "Code",
) -> pd.DataFrame:
    """
    Collapses a country-year panel into one row per country — clustering
    country-year rows directly would let the same country land in
    different clusters across years, which doesn't make sense for a
    label (Region, or a supervised Cluster feature) that shouldn't
    change year to year.

    For each country: the mean of every predictor (the country's
    typical level over the rows in `df`). If `target` is given, also
    the target's mean and its linear trend over time (slope from an
    OLS fit of target ~ year) — level alone would treat a country with
    a flat high suicide rate the same as one climbing rapidly toward it.

    Parameters
    ----------
    df : pd.DataFrame
        Country-year panel, e.g. df_development, or a train-only subset.
    predictor_features : list[str]
        Predictor columns to average (does not include target or year).
    target : str, optional
        Column to additionally compute mean and trend for. Pass a
        column name (e.g. "Suicide rate") for descriptive/validation
        use. Leave as None when the result will
        be used as a supervised-model feature — including the target
        here would leak target information into its own predictor.
    year_col : str, default "Year"
    id_col : str, default "Code"
        Column identifying the country (ISO code).

    Returns
    -------
    pd.DataFrame
        One row per country. Columns: id_col, "{feature}_mean" for each
        predictor, and — only if `target` is given — "{target}_mean"
        and "{target}_trend" (slope, target units per year — positive
        means rising).
    """
    agg_cols = list(predictor_features) + ([target] if target else [])
    level = df.groupby(id_col)[agg_cols].mean()
    level.columns = [f"{c}_mean" for c in level.columns]

    if target is None:
        return level.reset_index()

    trend = {}
    for code, group in df.groupby(id_col):
        years = group[year_col].to_numpy(dtype=float)
        values = group[target].to_numpy(dtype=float)
        if len(years) >= 2:
            slope = np.polyfit(years, values, 1)[0]
        else:
            slope = np.nan
        trend[code] = slope
    trend_series = pd.Series(trend, name=f"{target}_trend")
    trend_series.index.name = id_col

    result = level.join(trend_series).reset_index()
    return result


def sweep_kmeans(
    X_scaled: pd.DataFrame, k_range=range(2, 9), random_state: int = 42
) -> pd.DataFrame:
    """
    Fits K-Means for each k in k_range and records inertia (for the
    elbow method) and silhouette score (higher is better, range -1 to 1)
    — run before committing to a specific k.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        Scaled country-level features (e.g. output of
        aggregate_country_features(), scaled with RobustScaler).
    k_range : iterable of int, default range(2, 9)
        Candidate values of k. Must not include 1 (silhouette is
        undefined for a single cluster).
    random_state : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: "k", "inertia", "silhouette".
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def run_kmeans(X_scaled: pd.DataFrame, k: int, random_state: int = 42):
    """
    Fits a final K-Means model with a chosen k.

    Parameters
    ----------
    X_scaled : pd.DataFrame
    k : int
    random_state : int, default 42

    Returns
    -------
    tuple[np.ndarray, sklearn.cluster.KMeans]
        (cluster labels, fitted model).
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def run_hierarchical(X_scaled: pd.DataFrame, k: int, method: str = "ward"):
    """
    Runs agglomerative hierarchical clustering and cuts the resulting
    dendrogram at k clusters.

    Parameters
    ----------
    X_scaled : pd.DataFrame
    k : int
        Number of clusters to cut the dendrogram into.
    method : str, default "ward"
        Linkage method passed to scipy.cluster.hierarchy.linkage.
        "ward" minimizes within-cluster variance — the usual default
        for Euclidean feature spaces like this one.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (linkage_matrix, cluster_labels). linkage_matrix is what
        plot_dendrogram() expects; cluster_labels are 1-indexed (scipy
        convention), same length/order as X_scaled.
    """
    linkage_matrix = linkage(X_scaled, method=method)
    labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
    return linkage_matrix, labels


def cluster_region_agreement(cluster_labels, region_labels) -> dict:
    """
    Quantifies how much a clustering result agrees with the a priori
    EU_REGIONS labels, using two complementary metrics that don't
    require the cluster numbering to match the region naming (a
    cluster labeled "2" can be a perfect match for "Baltics" without
    ever being told that name).

    Parameters
    ----------
    cluster_labels : array-like
        Output of run_kmeans() or run_hierarchical(), one label per country.
    region_labels : array-like
        The corresponding EU_REGIONS value per country, same order.

    Returns
    -------
    dict
        {"ARI": float, "NMI": float}.
        Both range 0 (no agreement beyond chance) to 1 (perfect
        agreement). ARI penalizes disagreement more strictly; NMI is
        more forgiving of clusters that split a region into two
        sub-groups without mixing regions together.
    """
    return {
        "ARI": adjusted_rand_score(region_labels, cluster_labels),
        "NMI": normalized_mutual_info_score(region_labels, cluster_labels),
    }


# --------------------------------------------------------------------------
# Leakage-safe clustering as a supervised feature
# --------------------------------------------------------------------------
def fit_country_clusters(
    df_train: pd.DataFrame,
    predictor_features: list[str],
    id_col: str = "Code",
    k: int = 4,
    random_state: int = 42,
):
    """
    Fits a country-clustering model using ONLY `df_train` and ONLY the
    socioeconomic/mental-health predictors — never the target. This is
    the fitting half of the leakage-safe cluster-as-feature pipeline:
    call this once per split (Option A: on the training countries;
    Option B: on the training years), then use assign_country_clusters()
    to label every row of every split from this same fitted model.

    Parameters
    ----------
    df_train : pd.DataFrame
        The training subset only (e.g. df_train_A or df_train_B) — never
        pass test/val data here, or the cluster boundaries themselves
        would be informed by data the model shouldn't have access to.
    predictor_features : list[str]
        Same predictor list used for the supervised models. The target
        column must not be in this list.
    id_col : str, default "Code"
    k : int, default 4
        Number of clusters — 4 by default to mirror EU_REGIONS, but
        this is an independent choice; there is no requirement that the
        supervised-feature clustering use the same k as the descriptive
        validation in 03_clustering.py.
    random_state : int, default 42

    Returns
    -------
    dict
        {"kmeans": fitted KMeans, "scaler": fitted RobustScaler,
         "feature_cols": list of the "_mean" column names the scaler
         and KMeans expect, in order}. Pass this whole dict straight
         into assign_country_clusters().
    """
    train_agg = aggregate_country_features(
        df_train, predictor_features, target=None, id_col=id_col
    )
    feature_cols = [c for c in train_agg.columns if c != id_col]

    scaler = RobustScaler().fit(train_agg[feature_cols])
    X_scaled = scaler.transform(train_agg[feature_cols])

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X_scaled)

    return {"kmeans": kmeans, "scaler": scaler, "feature_cols": feature_cols}


def assign_country_clusters(
    df: pd.DataFrame,
    cluster_model: dict,
    predictor_features: list[str],
    id_col: str = "Code",
) -> pd.DataFrame:
    """
    Assigns a cluster label to every country in `df`, using a model
    already fitted by fit_country_clusters() — never refits anything.

    `df` may contain countries the cluster model has never seen (Option
    A's test/val countries): their own predictor history is aggregated
    the same way as in training and passed through .predict() against
    the existing cluster centroids, which is legitimate — it only uses
    that country's own (non-target) features, nothing the model
    wouldn't have at inference time.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to derive country profiles from. For Option B, pass the
        *training* rows only, even when you intend to label test/val
        rows too — see add_cluster_feature()'s docstring for why.
    cluster_model : dict
        Output of fit_country_clusters().
    predictor_features : list[str]
    id_col : str, default "Code"

    Returns
    -------
    pd.DataFrame
        Columns: [id_col, "Cluster"] — one row per country in `df`.
    """
    agg = aggregate_country_features(df, predictor_features, target=None, id_col=id_col)
    X_scaled = cluster_model["scaler"].transform(agg[cluster_model["feature_cols"]])
    agg["Cluster"] = cluster_model["kmeans"].predict(X_scaled)
    return agg[[id_col, "Cluster"]]


def add_cluster_feature(
    df_panel: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    id_col: str = "Code",
    all_clusters=None,
) -> pd.DataFrame:
    """
    Merges a per-country "Cluster" label onto every row of a country-year
    panel, as one-hot dummy columns ready to use as model features.

    Usage differs by split, and this function is intentionally dumb
    (just a merge) so the caller controls that difference explicitly:

    - **Option A** (different countries per split): call
      assign_country_clusters() separately for df_train_A, df_test_A,
      and df_val_A (each aggregating that split's own countries' own
      history), then add_cluster_feature() each split with its own
      assignments — but pass the SAME `all_clusters` (e.g.
      `range(cluster_model["kmeans"].n_clusters)`) to all three calls,
      since a given split's countries may not happen to cover every
      cluster, and train/test/val must still end up with identical
      columns for the downstream models to accept them.
    - **Option B** (same countries, different years per split): call
      assign_country_clusters() ONCE on df_train_B only, then
      add_cluster_feature() all three of df_train_B/df_test_B/df_val_B
      with that SAME assignment table (all_clusters can be left as
      None here, since one shared assignment table naturally has
      consistent columns across all three calls). A country's cluster
      must stay the fixed, training-period-derived value across its
      own train, test, and val rows — recomputing it from test-period
      rows alone would both leak future information and could
      disagree with the label its own training rows got, which breaks
      the "cluster is a static country attribute" premise the model
      relies on.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Country-year rows to enrich (e.g. df_train_A, df_test_B, ...).
    cluster_assignments : pd.DataFrame
        Output of assign_country_clusters() — [id_col, "Cluster"].
    id_col : str, default "Code"
    all_clusters : iterable[int], optional
        The full set of cluster ids to create dummy columns for.
        Defaults to the unique values in `cluster_assignments`, which
        is only safe when every split shares one assignment table
        (Option B). For Option A, pass this explicitly and identically
        across the train/test/val calls.

    Returns
    -------
    pd.DataFrame
        `df_panel` with one new "Cluster_{k}" dummy column per cluster
        in `all_clusters`, consistent across calls that pass the same
        `all_clusters`.
    """
    if all_clusters is None:
        all_clusters = sorted(cluster_assignments["Cluster"].unique())
    merged = df_panel.merge(cluster_assignments, on=id_col, how="left")
    dummies = pd.DataFrame(
        {f"Cluster_{k}": (merged["Cluster"] == k).astype(int) for k in all_clusters},
        index=merged.index,
    )
    return pd.concat([merged.drop(columns=["Cluster"]), dummies], axis=1)
