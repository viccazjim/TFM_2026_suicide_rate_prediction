"""
04 — Unsupervised clustering: validating the EU_REGIONS grouping.

Refactor of notebooks/04_clustering.ipynb into a runnable script.
Aggregates each country's socioeconomic/mental-health profile (level +
suicide rate trend), clusters with K-Means and hierarchical clustering,
and checks how well the result agrees with the a priori EU_REGIONS
grouping used descriptively in 02_eda.py.

Usage:
    python prod/04_clustering.py

Requires data/processed/df_development.parquet to already exist and be
cleaned (run prod/01_data_pipeline.py and prod/02_eda.py first).
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import (
    ID_COLS,
    TARGET,
    EU_REGIONS,
    build_predictor_list,
    save_figure,
)
from src.clustering import (
    aggregate_country_features,
    sweep_kmeans,
    run_kmeans,
    run_hierarchical,
    cluster_region_agreement,
)
from src.diagnostics import (
    plot_kmeans_elbow_silhouette,
    plot_dendrogram,
    plot_cluster_vs_region_pca,
    plot_suicide_trend_by_group,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
FIG_PREFIX = "04_"
K = 4  # matches the 4 EU_REGIONS, for direct comparability


def run():
    """
    Runs the full descriptive clustering validation and saves figures
    plus an agreement-metrics table to disk.

    Returns
    -------
    dict
        {"agreement_kmeans", "agreement_hierarchical"} — each an
        {"ARI": float, "NMI": float} dict.
    """
    df_development = pd.read_parquet(DEVELOPMENT_PATH)
    predictor_features = build_predictor_list(df_development, ID_COLS, TARGET)
    logger.info(
        "df_development: %d rows | %d predictors",
        df_development.shape[0],
        len(predictor_features),
    )

    country_df = aggregate_country_features(
        df_development, predictor_features, target=TARGET
    )
    feature_cols = [c for c in country_df.columns if c != "Code"]
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(country_df[feature_cols]),
        columns=feature_cols,
        index=country_df.index,
    )
    region_labels = country_df["Code"].map(EU_REGIONS)  # type: ignore[arg-type]
    logger.info(
        "Aggregated to %d countries, %d features",
        country_df.shape[0],
        len(feature_cols),
    )

    # --- How many clusters does the data support? ---
    sweep = sweep_kmeans(X_scaled, k_range=range(2, 9))
    logger.info("K-Means sweep (k, inertia, silhouette):\n%s", sweep.to_string())
    fig = plot_kmeans_elbow_silhouette(sweep)
    save_figure(
        fig,
        name="kmeans_elbow_silhouette",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- K-Means and hierarchical at k=4 ---
    labels_kmeans, _ = run_kmeans(X_scaled, k=K)
    linkage_matrix, labels_hier = run_hierarchical(X_scaled, k=K)

    fig = plot_dendrogram(linkage_matrix, labels=country_df["Code"].tolist(), k=K)
    save_figure(
        fig, name="dendrogram_k4", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR)
    )

    fig = plot_cluster_vs_region_pca(
        X_scaled, labels_kmeans, region_labels, country_df["Code"].tolist()
    )
    save_figure(
        fig,
        name="cluster_vs_region_pca",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- Agreement with EU_REGIONS ---
    agreement_kmeans = cluster_region_agreement(labels_kmeans, region_labels)
    agreement_hier = cluster_region_agreement(labels_hier, region_labels)
    logger.info(
        "K-Means vs EU_REGIONS      — ARI: %.3f | NMI: %.3f",
        agreement_kmeans["ARI"],
        agreement_kmeans["NMI"],
    )
    logger.info(
        "Hierarchical vs EU_REGIONS — ARI: %.3f | NMI: %.3f",
        agreement_hier["ARI"],
        agreement_hier["NMI"],
    )

    # --- Temporal evolution: a priori regions vs algorithm clusters ---
    df_development["Region"] = df_development["Code"].map(EU_REGIONS)  # type: ignore[arg-type]
    cluster_map = dict(zip(country_df["Code"], labels_kmeans))
    df_development["Cluster"] = (
        df_development["Code"].map(cluster_map).apply(lambda c: f"Cluster {c}")  # type: ignore[arg-type]
    )

    fig = plot_suicide_trend_by_group(
        df_development,
        "Region",
        legend_title="EU region (a priori)",
        title="Suicide rate evolution by a priori EU region (2000-2021)",
    )
    save_figure(
        fig,
        name="suicide_trend_by_region_a_priori",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    fig = plot_suicide_trend_by_group(
        df_development,
        "Cluster",
        legend_title="K-Means cluster",
        title="Suicide rate evolution by data-driven cluster (2000-2021)",
    )
    save_figure(
        fig,
        name="suicide_trend_by_cluster",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- Save agreement table ---
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    agreement_table = pd.DataFrame(
        [
            {"Method": "K-Means", **agreement_kmeans},
            {"Method": "Hierarchical", **agreement_hier},
        ]
    )
    agreement_table.to_parquet(
        TABLES_DIR / "cluster_region_agreement.parquet", index=False
    )
    logger.info("Saved: %s", TABLES_DIR / "cluster_region_agreement.parquet")

    return {
        "agreement_kmeans": agreement_kmeans,
        "agreement_hierarchical": agreement_hier,
    }


if __name__ == "__main__":
    run()
