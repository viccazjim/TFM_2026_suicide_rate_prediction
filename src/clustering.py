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
from typing import cast
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
    target: str | None = None,
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
    agg_cols: list[str] = list(predictor_features) + ([target] if target else [])
    level = cast(pd.DataFrame, df.groupby(id_col)[agg_cols].mean())
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
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)  # type: ignore[arg-type]  -- sklearn accepts int at runtime; stub only declares str
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
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)  # type: ignore[arg-type]  -- sklearn accepts int at runtime; stub only declares str
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

