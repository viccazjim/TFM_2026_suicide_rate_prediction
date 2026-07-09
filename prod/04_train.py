"""
04 — Model training and evaluation.

Refactor of notebooks/03_models.ipynb into a runnable script, extended
per thesis advisor feedback with two additions:

1. **Country cluster as a feature.** Instead of only validating
   EU_REGIONS descriptively after modelling (03_clustering.py), a
   leakage-safe country cluster is fit and injected as a feature
   directly into the 6 panel models below — see
   src/clustering.py::fit_country_clusters for exactly how it avoids
   leaking target information (no "Suicide rate" in the clustering
   inputs, fit on training data only).
2. **Genuine time-series models for Option B.** SARIMAX and Prophet are
   fit one-per-country on Option B's temporal split, since Option B is
   literally a time series problem that the original 6 panel models
   never modelled as one (see src/timeseries_models.py). These are
   Option-B-only — a per-country time-series model cannot forecast a
   country with zero training history, which rules out Option A by
   construction.

Also persists the production model — one Prophet model per country
(Option B, chosen by Validation RMSE/R² over both SARIMAX and the 6
panel models) — under outputs/models/, so prod/predict.py can forecast
new years for a known country without retraining. The CatBoost/scaler/
cluster-model artifacts are still saved too (for reference, and because
retiring them would silently break anyone still pointing at them), but
predict.py no longer uses them by default.

Usage:
    python prod/04_train.py

Requires data/processed/df_development.parquet to already exist and be
cleaned (run prod/01_data_pipeline.py and prod/02_eda.py first, or
their notebook equivalents).
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this script only saves figures to disk, never displays them

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
    plot_correlation_heatmaps,
    plot_rmse_comparison,
    plot_r2_comparison,
    save_figure,
    save_artifact,
)
from src.clustering import fit_country_clusters, assign_country_clusters, add_cluster_feature
from src.timeseries_models import train_evaluate_sarimax, train_evaluate_prophet, fit_prophet_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
FIG_PREFIX = "04_"

CLUSTER_K = 4  # independent of 03_clustering.py's k — see fit_country_clusters()'s docstring

# Production model — Prophet, one per country, Option B. Chosen over the 6
# panel models AND over SARIMAX by Validation performance specifically
# (RMSE 2.49 vs SARIMAX's 2.98, R² 0.72 vs 0.60) — Test is close between the
# two (SARIMAX marginally ahead), but Val is this project's established
# generalization check, so it decides. This changes predict.py's whole
# contract: instead of scoring an arbitrary feature vector, it forecasts a
# specific (known) country forward from its own history — see predict.py's
# module docstring for what that means in practice (new/unseen countries
# cannot be scored this way, unlike the CatBoost-based approach it replaces).
PRODUCTION_MODEL_NAME = "Prophet"


def _timeseries_trained_stub(elapsed_s: float) -> dict:
    """
    build_results_table() expects trained_dict[model]["cv_rmse"] and
    ["time_s"] to exist for every model in eval_list. SARIMAX/Prophet
    aren't cross-validated the way the panel models are (no CV loop —
    each is fit once per country), so cv_rmse is genuinely not
    applicable here; None is the honest value, not a placeholder to
    fill in later.
    """
    return {"cv_rmse": None, "time_s": round(elapsed_s, 2)}


def _run_option(df_development, predictor_features, split_fn, split_args, cv, label):
    """
    Runs split + cluster feature engineering + scaling + training +
    evaluation for one of the two options (A or B).

    Parameters
    ----------
    df_development : pd.DataFrame
    predictor_features : list[str]
        Original (pre-cluster) predictor list — the cluster dummy
        columns are added on top of this, not in place of it.
    split_fn : callable
        geographical_split or temporal_split.
    split_args : tuple
    cv : int or sklearn cross-validator
    label : str
        "A" or "B" — used in logs and figure names.

    Returns
    -------
    dict
        {"df_train", "df_test", "df_val", "trained", "eval",
         "X_train_scaled", "X_test_scaled", "X_val_scaled", "y_val",
         "scaler", "cluster_model", "full_predictor_features"}
    """
    df_train, df_test, df_val, *_ = split_fn(df_development, *split_args)
    logger.info("Option %s — Train: %d rows | Test: %d rows | Val: %d rows", label, len(df_train), len(df_test), len(df_val))

    fig = plot_correlation_heatmaps(df_train, SOCIAL_ECONOMIC_FEATURES, HEALTH_RELATED_FEATURES)
    save_figure(fig, name=f"correlation_heatmaps_option_{label}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    # --- Leakage-safe cluster feature: fit on train only, never on the target ---
    cluster_model = fit_country_clusters(df_train, predictor_features, k=CLUSTER_K, random_state=42)

    if label == "A":
        # Different countries per split: each split's own countries, own history.
        # all_clusters is passed explicitly and identically to all three calls,
        # since a given split's countries may not happen to cover every cluster
        # (e.g. no test country falls in cluster 3) — without this, train/test/val
        # would end up with different numbers of Cluster_* columns.
        all_clusters = list(range(cluster_model["kmeans"].n_clusters))
        assign_train = assign_country_clusters(df_train, cluster_model, predictor_features)
        assign_test = assign_country_clusters(df_test, cluster_model, predictor_features)
        assign_val = assign_country_clusters(df_val, cluster_model, predictor_features)
        df_train = add_cluster_feature(df_train, assign_train, all_clusters=all_clusters)
        df_test = add_cluster_feature(df_test, assign_test, all_clusters=all_clusters)
        df_val = add_cluster_feature(df_val, assign_val, all_clusters=all_clusters)
    else:
        # Same countries, different years: fit + assign ONCE on train years,
        # reuse that same assignment for test/val — see add_cluster_feature()'s
        # docstring for why recomputing per-split here would be wrong.
        assign_train = assign_country_clusters(df_train, cluster_model, predictor_features)
        df_train = add_cluster_feature(df_train, assign_train)
        df_test = add_cluster_feature(df_test, assign_train)
        df_val = add_cluster_feature(df_val, assign_train)

    cluster_cols = [c for c in df_train.columns if c.startswith("Cluster_")]
    full_predictor_features = predictor_features + cluster_cols
    logger.info("Option %s — added %d cluster dummy columns: %s", label, len(cluster_cols), cluster_cols)

    # --- Scale only the original continuous predictors; cluster dummies pass through unscaled ---
    scaler = RobustScaler()
    X_train_cont = pd.DataFrame(
        scaler.fit_transform(df_train[predictor_features]), columns=predictor_features, index=df_train.index
    )
    X_test_cont = pd.DataFrame(
        scaler.transform(df_test[predictor_features]), columns=predictor_features, index=df_test.index
    )
    X_val_cont = pd.DataFrame(
        scaler.transform(df_val[predictor_features]), columns=predictor_features, index=df_val.index
    )
    X_train_scaled = pd.concat([X_train_cont, df_train[cluster_cols].reset_index(drop=True).set_axis(X_train_cont.index)], axis=1)
    X_test_scaled = pd.concat([X_test_cont, df_test[cluster_cols].reset_index(drop=True).set_axis(X_test_cont.index)], axis=1)
    X_val_scaled = pd.concat([X_val_cont, df_val[cluster_cols].reset_index(drop=True).set_axis(X_val_cont.index)], axis=1)

    y_train, y_test, y_val = df_train[TARGET].copy(), df_test[TARGET].copy(), df_val[TARGET].copy()

    models = make_models(random_state=42)
    trained = {}
    logger.info("Option %s — training %d models", label, len(models))
    for name, model in models.items():
        trained[name] = train_model(
            name=name, model=model, param_grid=param_grids[name],
            X_train=X_train_scaled, y_train=y_train, cv=cv,
        )

    eval_results = []
    for name in trained:
        eval_results.append(evaluate_model(trained[name], X_test_scaled, y_test, "Test"))
        eval_results.append(evaluate_model(trained[name], X_val_scaled, y_val, "Val"))

    return {
        "df_train": df_train, "df_test": df_test, "df_val": df_val,
        "trained": trained, "eval": eval_results,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled, "X_val_scaled": X_val_scaled,
        "y_val": y_val, "scaler": scaler, "cluster_model": cluster_model,
        "full_predictor_features": full_predictor_features,
    }


def run():
    """
    Runs full training and evaluation (Option A + B, 6 panel models
    each, plus SARIMAX/Prophet for Option B only), saves comparison
    tables/figures, and persists the production model (CatBoost,
    Option B, with cluster feature) + its scaler + cluster model.

    Returns
    -------
    dict
        {"table_A_test", "table_A_val", "table_B_test", "table_B_val",
         "production_model_path", "production_scaler_path",
         "production_cluster_model_path"}
    """
    np.random.seed(42)
    df_development = pd.read_parquet(DEVELOPMENT_PATH)
    predictor_features = build_predictor_list(df_development, ID_COLS, TARGET)
    logger.info("df_development: %d rows | %d predictors", df_development.shape[0], len(predictor_features))

    result_A = _run_option(df_development, predictor_features, geographical_split, (EU_COUNTRIES_ISO,), cv=5, label="A")
    result_B = _run_option(df_development, predictor_features, temporal_split, (), cv=TimeSeriesSplit(n_splits=5), label="B")

    # --- Time-series models, Option B only (see module docstring for why) ---
    logger.info("Training per-country SARIMAX (Option B only — see module docstring)")
    t0 = time.time()
    eval_sarimax, per_country_sarimax = train_evaluate_sarimax(
        result_B["df_train"], result_B["df_test"], result_B["df_val"], TARGET
    )
    sarimax_time = time.time() - t0
    n_converged = per_country_sarimax.loc[per_country_sarimax["Split"] == "Test", "converged"].sum()
    logger.info("SARIMAX: %d/%d countries converged (%.1fs)", n_converged, result_B["df_train"]["Code"].nunique(), sarimax_time)

    logger.info("Training per-country Prophet (Option B only)")
    t0 = time.time()
    eval_prophet, per_country_prophet = train_evaluate_prophet(
        result_B["df_train"], result_B["df_test"], result_B["df_val"], TARGET
    )
    prophet_time = time.time() - t0
    n_converged = per_country_prophet.loc[per_country_prophet["Split"] == "Test", "converged"].sum()
    logger.info("Prophet: %d/%d countries converged (%.1fs)", n_converged, result_B["df_train"]["Code"].nunique(), prophet_time)

    result_B["eval"].extend(eval_sarimax)
    result_B["eval"].extend(eval_prophet)
    result_B["trained"]["SARIMAX"] = _timeseries_trained_stub(sarimax_time)
    result_B["trained"]["Prophet"] = _timeseries_trained_stub(prophet_time)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_country_sarimax.to_csv(TABLES_DIR / "sarimax_per_country.csv", index=False)
    per_country_prophet.to_csv(TABLES_DIR / "prophet_per_country.csv", index=False)

    # --- Results tables (now including SARIMAX/Prophet in Option B) ---
    table_A_test = build_results_table(result_A["eval"], result_A["trained"], "Test", "Option A")
    logger.info("\n%s", table_A_test.to_string())
    table_A_val = build_results_table(result_A["eval"], result_A["trained"], "Val", "Option A")
    logger.info("\n%s", table_A_val.to_string())
    table_B_test = build_results_table(result_B["eval"], result_B["trained"], "Test", "Option B")
    logger.info("\n%s", table_B_test.to_string())
    table_B_val = build_results_table(result_B["eval"], result_B["trained"], "Val", "Option B")
    logger.info("\n%s", table_B_val.to_string())

    fig = plot_rmse_comparison(table_A_test, table_A_val, "Option A — Geographical split")
    save_figure(fig, name="rmse_comparison_option_A", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))
    fig = plot_rmse_comparison(table_B_test, table_B_val, "Option B — Time split")
    save_figure(fig, name="rmse_comparison_option_B", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))
    fig = plot_r2_comparison(table_A_test, table_A_val, "Option A — Geographical split")
    save_figure(fig, name="r2_comparison_option_A", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))
    fig = plot_r2_comparison(table_B_test, table_B_val, "Option B — Time split")
    save_figure(fig, name="r2_comparison_option_B", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    table_A_test.to_csv(TABLES_DIR / "test_geographical.csv", index=False)
    table_A_val.to_csv(TABLES_DIR / "val_geographical.csv", index=False)
    table_B_test.to_csv(TABLES_DIR / "test_temporal.csv", index=False)
    table_B_val.to_csv(TABLES_DIR / "val_temporal.csv", index=False)
    logger.info("Result tables saved to %s", TABLES_DIR)

    # --- Persist the production model: one Prophet model per country (Option B) ---
    production_prophet_models = fit_prophet_models(result_B["df_train"], TARGET)
    n_fitted = len(production_prophet_models)
    n_countries = result_B["df_train"]["Code"].nunique()
    if n_fitted < n_countries:
        logger.warning(
            "Only %d/%d country Prophet models fitted successfully — predict.py will not "
            "be able to score the missing countries.", n_fitted, n_countries
        )
    prophet_models_path = save_artifact(production_prophet_models, str(MODELS_DIR / "prophet_models_option_b.joblib"))
    logger.info("Production model saved: %s (%d country models)", prophet_models_path, n_fitted)

    # --- Also persist the CatBoost/scaler/cluster artifacts, for reference —
    # not the default predict.py path anymore, but not deleted either, since
    # some other script or a future comparison might still want them.
    reference_model = result_B["trained"]["CatBoost"]["best_estimator"]
    reference_scaler = result_B["scaler"]
    reference_cluster_model = result_B["cluster_model"]
    reference_cluster_assignments = assign_country_clusters(
        result_B["df_train"], reference_cluster_model, predictor_features
    )
    save_artifact(reference_model, str(MODELS_DIR / "catboost_option_b.joblib"))
    save_artifact(reference_scaler, str(MODELS_DIR / "scaler_option_b.joblib"))
    save_artifact(reference_cluster_model, str(MODELS_DIR / "cluster_model_option_b.joblib"))
    save_artifact(reference_cluster_assignments, str(MODELS_DIR / "cluster_assignments_option_b.joblib"))
    logger.info("Reference CatBoost artifacts also saved (not used by predict.py by default)")

    return {
        "table_A_test": table_A_test, "table_A_val": table_A_val,
        "table_B_test": table_B_test, "table_B_val": table_B_val,
        "production_model_path": prophet_models_path,
    }


if __name__ == "__main__":
    run()
