"""
02 — Exploratory Data Analysis (EDA).

Refactor of notebooks/02_eda.ipynb into a runnable script. Generates
all the exploratory-phase figures and tables (time evolution,
region-level trends, feature distributions, IQR outliers, VIF
multicollinearity check), saves figures to outputs/figures/ with the
"02_" prefix, and overwrites data/processed/df_development.csv with
the "Region" column added and "Eating disorders" dropped (high VIF).

Usage:
    python prod/02_eda.py

Requires prod/01_data_pipeline.py (or the equivalent notebook) to have
run first — reads data/processed/df_development.csv.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import (
    ID_COLS,
    TARGET,
    EU_REGIONS,
    SOCIAL_ECONOMIC_FEATURES,
    compute_vif,
    flag_outliers_iqr,
    build_predictor_list,
    suicide_evolution_graph,
    plot_suicide_trend_by_region,
    plot_suicide_boxplot_by_country,
    plot_feature_distributions,
    plot_suicide_dispersion_stripplot,
    plot_vif_bar,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DEVELOPMENT_CSV_PATH = REPO_ROOT / "data" / "processed" / "df_development.csv"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
FIG_PREFIX = "02_"


def run() -> pd.DataFrame:
    """
    Runs the full EDA and overwrites df_development.csv with the
    resulting changes (Region added, Eating disorders dropped).

    Returns
    -------
    pd.DataFrame
        The cleaned df_development (same object saved to disk).
    """
    df_development = pd.read_csv(DEVELOPMENT_CSV_PATH)
    logger.info(
        "df_development: %d rows, %d columns",
        df_development.shape[0],
        df_development.shape[1],
    )

    # --- Suicide rate evolution: country spotlights ---
    logger.info("Generating country-evolution plots (LTU, GRC, DEU)")
    fig = suicide_evolution_graph(df_development, "LTU", "Lithuania")
    save_figure(
        fig,
        name="suicide_evolution_ltu",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )
    fig = suicide_evolution_graph(df_development, "GRC", "Greece")
    save_figure(
        fig,
        name="suicide_evolution_grc",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )
    fig = suicide_evolution_graph(df_development, "DEU", "Germany")
    save_figure(
        fig,
        name="suicide_evolution_deu",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- Trend by region ---
    logger.info("Generating region-level trend plot")
    df_development["Region"] = df_development["Code"].map(EU_REGIONS)
    fig = plot_suicide_trend_by_region(df_development)
    save_figure(
        fig,
        name="suicide_trend_by_region",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- Cross-country distribution ---
    logger.info("Generating cross-country distribution boxplot")
    fig = plot_suicide_boxplot_by_country(df_development)
    save_figure(
        fig,
        name="suicide_boxplot_by_country",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- Feature distributions ---
    logger.info("Generating feature distributions")
    predictor_features = build_predictor_list(df_development, ID_COLS, TARGET)
    fig = plot_feature_distributions(df_development, predictor_features)
    save_figure(
        fig,
        name="feature_distributions",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )
    skew_summary = (
        df_development[predictor_features].skew().sort_values(ascending=False).round(2)
    )
    logger.info("Skewness summary:\n%s", skew_summary)

    # --- Outliers (IQR) ---
    logger.info("Detecting outliers (IQR method)")
    socioeconomic_cols_for_outliers = [
        c for c in SOCIAL_ECONOMIC_FEATURES if c in df_development.columns
    ]
    outlier_summary = flag_outliers_iqr(df_development, socioeconomic_cols_for_outliers)
    logger.info("Outlier summary (IQR, threshold=1.5):\n%s", outlier_summary)
    fig = plot_suicide_dispersion_stripplot(df_development)
    save_figure(
        fig,
        name="suicide_dispersion_stripplot",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    # --- Multicollinearity (VIF) — before dropping Eating disorders ---
    logger.info("Computing VIF (full predictor set)")
    vif_results = compute_vif(df_development, predictor_features)
    logger.info("VIF (full set):\n%s", vif_results)
    fig = plot_vif_bar(vif_results)
    save_figure(
        fig, name="vif_bar_before_drop", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR)
    )

    # --- Drop Eating disorders (high VIF) and recompute ---
    logger.info("Dropping 'Eating disorders' (high VIF) and recomputing VIF")
    df_development = df_development.drop(columns=["Eating disorders"], errors="ignore")
    predictor_features = build_predictor_list(df_development, ID_COLS, TARGET)
    vif_results = compute_vif(df_development, predictor_features)
    logger.info("VIF (after dropping Eating disorders):\n%s", vif_results)
    fig = plot_vif_bar(vif_results)
    save_figure(
        fig, name="vif_bar_after_drop", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR)
    )

    # --- Save ---
    df_development.to_csv(DEVELOPMENT_CSV_PATH, index=False)
    logger.info("Saved (updated): %s", DEVELOPMENT_CSV_PATH)

    return df_development


if __name__ == "__main__":
    run()
