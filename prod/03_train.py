"""
03 — Model training and evaluation.

Refactor of notebooks/03_models.ipynb into a runnable script. Trains
the 6 panel regression models (Ridge, Lasso, SVR, Random Forest,
XGBoost, CatBoost) under both split strategies (Option A —
geographical, Option B — time-based), evaluates, saves result tables
to outputs/tables/ and figures to outputs/figures/ (prefix "03_").

Usage:
    python prod/03_train.py

Requires data/processed/df_development.parquet to already exist and be
cleaned (run prod/01_data_pipeline.py and prod/02_eda.py first, or
their notebook equivalents).
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import (
    ID_COLS,
    TARGET,
    EU_COUNTRIES_ISO,
    SOCIAL_ECONOMIC_FEATURES,
    HEALTH_RELATED_FEATURES,
    build_predictor_list,
    geographical_split,
    temporal_split,
    param_grids,
    make_models,
    train_model,
    evaluate_model,
    build_results_table,
    get_eval_entry,
    plot_correlation_heatmaps,
    plot_rmse_comparison,
    plot_r2_comparison,
    plot_actual_vs_predicted,
    plot_residual_histogram,
    plot_residuals_vs_predicted,
    plot_error_by_year,
    mean_absolute_error_by_country,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
    save_figure,
    save_artifact,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
FIG_PREFIX = "03_"

# Chosen production model — CatBoost under Option B (time split). Not the
# best Test R² in isolation, but the one that generalizes to Validation in a
# stable way across both metrics (see docs/ for the full reasoning).
PRODUCTION_MODEL_NAME = "CatBoost"


def _run_option(df_development, predictor_features, split_fn, split_args, cv, label):
    """
    Runs split + scaling + training + evaluation for one of the two
    options (A or B) — avoids duplicating this sequence twice.

    Parameters
    ----------
    df_development : pd.DataFrame
    predictor_features : list[str]
    split_fn : callable
        geographical_split or temporal_split.
    split_args : tuple
        Extra positional arguments for split_fn (beyond df_development).
    cv : int or sklearn cross-validator
    label : str
        "A" or "B" — used in logs and figure names.

    Returns
    -------
    dict
        {"df_train", "df_test", "df_val", "trained", "eval", "X_val_scaled", "scaler"}
    """
    df_train, df_test, df_val, *_ = split_fn(df_development, *split_args)

    if label == "B":
        df_train = df_train.sort_values(["Year", "Code"]).reset_index(drop=True)

    logger.info(
        "Option %s — Train: %d rows | Test: %d rows | Val: %d rows",
        label,
        len(df_train),
        len(df_test),
        len(df_val),
    )

    fig = plot_correlation_heatmaps(
        df_train, SOCIAL_ECONOMIC_FEATURES, HEALTH_RELATED_FEATURES
    )
    save_figure(
        fig,
        name=f"correlation_heatmaps_option_{label}",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    scaler = RobustScaler()
    X_train = df_train[predictor_features].copy()
    X_test = df_test[predictor_features].copy()
    X_val = df_val[predictor_features].copy()
    y_train, y_test, y_val = (
        df_train[TARGET].copy(),
        df_test[TARGET].copy(),
        df_val[TARGET].copy(),
    )

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=predictor_features, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=predictor_features, index=X_test.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=predictor_features, index=X_val.index
    )

    models = make_models(random_state=42)
    trained = {}
    logger.info("Option %s — training %d models", label, len(models))
    for name, model in models.items():
        trained[name] = train_model(
            name=name,
            model=model,
            param_grid=param_grids[name],
            X_train=X_train_scaled,
            y_train=y_train,
            cv=cv,
        )

    eval_results = []
    for name in trained:
        eval_results.append(
            evaluate_model(trained[name], X_test_scaled, y_test, "Test")
        )
        eval_results.append(evaluate_model(trained[name], X_val_scaled, y_val, "Val"))

    return {
        "df_train": df_train,
        "df_test": df_test,
        "df_val": df_val,
        "trained": trained,
        "eval": eval_results,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val,
        "scaler": scaler,
    }


def run():
    """
    Runs full training and evaluation (Option A + B, 6 models each),
    saves comparison tables/figures, and persists the production model
    (CatBoost, Option B) + its scaler.

    Returns
    -------
    dict
        {"table_A_test", "table_A_val", "table_B_test", "table_B_val",
         "production_model_path", "production_scaler_path"}
    """
    np.random.seed(42)
    df_development = pd.read_parquet(DEVELOPMENT_PATH)
    predictor_features = build_predictor_list(df_development, ID_COLS, TARGET)
    logger.info(
        "df_development: %d rows | %d predictors",
        df_development.shape[0],
        len(predictor_features),
    )

    result_A = _run_option(
        df_development,
        predictor_features,
        geographical_split,
        (EU_COUNTRIES_ISO,),
        cv=5,
        label="A",
    )
    result_B = _run_option(
        df_development,
        predictor_features,
        temporal_split,
        (),
        cv=TimeSeriesSplit(n_splits=5),
        label="B",
    )

    table_A_test = build_results_table(
        result_A["eval"], result_A["trained"], "Test", "Option A"
    )
    logger.info("\n%s", table_A_test.to_string())
    table_A_val = build_results_table(
        result_A["eval"], result_A["trained"], "Val", "Option A"
    )
    logger.info("\n%s", table_A_val.to_string())
    table_B_test = build_results_table(
        result_B["eval"], result_B["trained"], "Test", "Option B"
    )
    logger.info("\n%s", table_B_test.to_string())
    table_B_val = build_results_table(
        result_B["eval"], result_B["trained"], "Val", "Option B"
    )
    logger.info("\n%s", table_B_val.to_string())

    fig = plot_rmse_comparison(
        table_A_test, table_A_val, "Option A — Geographical split"
    )
    save_figure(
        fig,
        name="rmse_comparison_option_A",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )
    fig = plot_rmse_comparison(table_B_test, table_B_val, "Option B — Time split")
    save_figure(
        fig,
        name="rmse_comparison_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )
    fig = plot_r2_comparison(table_A_test, table_A_val, "Option A — Geographical split")
    save_figure(
        fig,
        name="r2_comparison_option_A",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )
    fig = plot_r2_comparison(table_B_test, table_B_val, "Option B — Time split")
    save_figure(
        fig,
        name="r2_comparison_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_A_test.to_parquet(TABLES_DIR / "test_geographical.parquet", index=False)
    table_A_val.to_parquet(TABLES_DIR / "val_geographical.parquet", index=False)
    table_B_test.to_parquet(TABLES_DIR / "test_temporal.parquet", index=False)
    table_B_val.to_parquet(TABLES_DIR / "val_temporal.parquet", index=False)
    logger.info("Result tables saved to %s", TABLES_DIR)

    # --- Result diagnostics and interpretability ---
    logger.info(
        "Generating diagnostic and SHAP figures for %s (Option B, Validation)",
        PRODUCTION_MODEL_NAME,
    )
    test_entry = get_eval_entry(result_B["eval"], PRODUCTION_MODEL_NAME, "Test")
    val_entry = get_eval_entry(result_B["eval"], PRODUCTION_MODEL_NAME, "Val")
    logger.info(
        "Diagnosing: %s | Option B — Test RMSE: %s | R²: %s | Val RMSE: %s | R²: %s",
        PRODUCTION_MODEL_NAME,
        test_entry["rmse"],
        test_entry["r2"],
        val_entry["rmse"],
        val_entry["r2"],
    )

    fig = plot_actual_vs_predicted(
        val_entry["actuals"],
        val_entry["predictions"],
        title=f"{PRODUCTION_MODEL_NAME} — Actual vs Predicted (Option B, Validation)",
    )
    save_figure(
        fig,
        name="actual_vs_predicted_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    fig = plot_residual_histogram(
        val_entry["actuals"],
        val_entry["predictions"],
        title=f"{PRODUCTION_MODEL_NAME} — Residual Distribution (Option B, Validation)",
    )
    save_figure(
        fig,
        name="residual_histogram_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    fig = plot_residuals_vs_predicted(
        val_entry["actuals"],
        val_entry["predictions"],
        title=f"{PRODUCTION_MODEL_NAME} — Residuals vs Predicted (Option B, Validation)",
    )
    save_figure(
        fig,
        name="residuals_vs_predicted_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    fig = plot_error_by_year(
        result_B["df_val"][["Year"]].reset_index(drop=True),
        val_entry["actuals"],
        val_entry["predictions"],
        title=f"{PRODUCTION_MODEL_NAME} — Mean Absolute Error by Year (Option B, Validation)",
    )
    save_figure(
        fig,
        name="error_by_year_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    error_by_country = mean_absolute_error_by_country(
        result_B["df_val"][["Country"]].reset_index(drop=True),
        val_entry["actuals"],
        val_entry["predictions"],
        top_n=10,
    )
    logger.info(
        "Highest mean absolute error by country (Validation):\n%s",
        error_by_country.to_string(),
    )

    explainer, shap_values, X_shap_sample = compute_shap_values(
        result_B["trained"][PRODUCTION_MODEL_NAME]["best_estimator"],
        result_B["X_val_scaled"],
        sample_size=500,
        random_state=42,
    )
    fig = plot_shap_summary(
        shap_values,
        X_shap_sample,
        title=f"{PRODUCTION_MODEL_NAME} — SHAP Summary (Option B, Validation sample)",
    )
    save_figure(
        fig,
        name="shap_summary_option_B",
        prefix=FIG_PREFIX,
        figures_dir=str(FIGURES_DIR),
    )

    fig = plot_shap_waterfall(
        shap_values,
        index=0,
        title=f"{PRODUCTION_MODEL_NAME} — SHAP Waterfall (single Validation prediction)",
    )
    save_figure(
        fig, name="shap_waterfall", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR)
    )

    # --- Persist the production model (CatBoost, Option B) ---
    production_model = result_B["trained"][PRODUCTION_MODEL_NAME]["best_estimator"]
    production_scaler = result_B["scaler"]
    model_path = save_artifact(
        production_model, str(MODELS_DIR / "catboost_option_b.joblib")
    )
    scaler_path = save_artifact(
        production_scaler, str(MODELS_DIR / "scaler_option_b.joblib")
    )
    logger.info("Production model saved: %s", model_path)
    logger.info("Production scaler saved: %s", scaler_path)

    return {
        "table_A_test": table_A_test,
        "table_A_val": table_A_val,
        "table_B_test": table_B_test,
        "table_B_val": table_B_val,
        "production_model_path": model_path,
        "production_scaler_path": scaler_path,
    }


if __name__ == "__main__":
    run()
