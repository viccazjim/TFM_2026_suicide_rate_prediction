"""
07 — Export the Power BI dashboard's data workbook.

Assembles the flat, denormalised tables the dashboard needs — country
panel, predictions, SHAP importance, model comparison, cluster/PCA
lookup — purely from artifacts earlier stages already produce
(df_development.parquet, predict.py's and
06_visualize_predictions.py's saved predictions, the persisted
production CatBoost model + scaler, and src/clustering.py's
descriptive clustering functions), and writes them to a single
formatted .xlsx. No new model is fit and no new result is computed
here beyond that assembly — see src/export.py's module docstring.

Output: outputs/powerbi/TFM_PowerBI_data.xlsx

Usage:
    python prod/07_export_powerbi.py

Requires prod/predict.py and prod/06_visualize_predictions.py to have
run first (for the CatBoost and SARIMAX predictions this reads).
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import (
    ID_COLS,
    TARGET,
    EU_REGIONS,
    build_predictor_list,
    aggregate_country_features,
    run_kmeans,
    compute_pca_coords,
    load_artifact,
    build_suicide_rate_panel_table,
    build_predictions_table,
    build_shap_importance_table,
    build_model_comparison_table,
    build_cluster_lookup_table,
    write_powerbi_workbook,
)
from src.explainability import compute_shap_values

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
POWERBI_DIR = REPO_ROOT / "outputs" / "powerbi"
OUTPUT_PATH = POWERBI_DIR / "PowerBI_data.xlsx"

N_CLUSTERS = 4  # mirrors EU_REGIONS' 4 groups — see 04_clustering.py


def run():
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading historical data and predictor list")
    df = pd.read_parquet(DEVELOPMENT_PATH)
    predictor_features = build_predictor_list(df, ID_COLS, TARGET)

    # --- Descriptive clustering + PCA, matching 04_clustering.py exactly ---
    logger.info(
        "Running descriptive clustering (k=%d) for the cluster/region lookup",
        N_CLUSTERS,
    )
    agg = aggregate_country_features(df, predictor_features, target=TARGET)
    feature_cols = [c for c in agg.columns if c != "Code"]
    X_scaled = pd.DataFrame(
        RobustScaler().fit_transform(agg[feature_cols]), columns=feature_cols
    )
    cluster_labels, _ = run_kmeans(X_scaled, k=N_CLUSTERS)
    pca_coords, var_explained = compute_pca_coords(X_scaled)
    logger.info(
        "PCA explained variance (2 components): %.1f%%", 100 * var_explained.sum()
    )
    cluster_lookup = build_cluster_lookup_table(
        agg["Code"], cluster_labels, pca_coords, EU_REGIONS
    )

    # --- SHAP importance for the production CatBoost model ---
    logger.info("Computing SHAP importance for the production model")
    model = load_artifact(str(MODELS_DIR / "catboost_option_b.joblib"))
    scaler = load_artifact(str(MODELS_DIR / "scaler_option_b.joblib"))
    X_all = pd.DataFrame(
        scaler.transform(df[predictor_features]), columns=predictor_features
    )
    _, shap_values, _ = compute_shap_values(
        model, X_all, sample_size=500, random_state=42
    )
    shap_table = build_shap_importance_table(shap_values, predictor_features)

    # --- Predictions (CatBoost + SARIMAX), already saved by earlier stages ---
    logger.info("Loading CatBoost and SARIMAX predictions")
    catboost_predictions = pd.read_parquet(TABLES_DIR / "predictions.parquet")
    sarimax_predictions = pd.read_parquet(TABLES_DIR / "predictions_temporal.parquet")
    predictions_table = build_predictions_table(
        catboost_predictions, sarimax_predictions, cluster_lookup
    )

    # --- Model comparison, already saved by 03_train.py ---
    logger.info("Loading panel-model result tables")
    result_tables = {
        ("Option A", "Test"): pd.read_parquet(TABLES_DIR / "test_geographical.parquet"),
        ("Option A", "Validation"): pd.read_parquet(
            TABLES_DIR / "val_geographical.parquet"
        ),
        ("Option B", "Test"): pd.read_parquet(TABLES_DIR / "test_temporal.parquet"),
        ("Option B", "Validation"): pd.read_parquet(
            TABLES_DIR / "val_temporal.parquet"
        ),
    }
    model_comparison_table = build_model_comparison_table(result_tables)

    # --- Suicide rate panel (wide: target + determinants + cluster/region) ---
    panel_table = build_suicide_rate_panel_table(
        df, predictor_features, cluster_lookup, target=TARGET
    )

    # --- Already-saved supporting tables, passed through unchanged ---
    temporal_persistence_table = pd.read_parquet(
        TABLES_DIR / "temporal_persistence_check.parquet"
    )
    cluster_agreement_table = pd.read_parquet(
        TABLES_DIR / "cluster_region_agreement.parquet"
    )

    tables = {
        "Suicide_Rate_Panel": panel_table,
        "Predictions_2022_2023": predictions_table,
        "SHAP_Importance": shap_table,
        "Model_Comparison": model_comparison_table,
        "Cluster_PCA": cluster_lookup,
        "Temporal_Persistence_Check": temporal_persistence_table,
        "Cluster_Region_Agreement": cluster_agreement_table,
    }

    logger.info("Writing Power BI workbook: %s", OUTPUT_PATH)
    write_powerbi_workbook(tables, str(OUTPUT_PATH))
    logger.info("Done. %d sheets written.", len(tables))


if __name__ == "__main__":
    run()
