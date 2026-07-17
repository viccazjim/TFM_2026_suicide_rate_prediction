"""
05 — Temporal persistence check: a second, different question.

**Once the original thesis question has been answered, this script aims
to answer how much of suicide rate is simply explained by itself
— by its own persistence from one year to the next — independent of
any determinant?**.

Usage:
    python prod/05_temporal_persistence_check.py

Requires data/processed/df_development.parquet to already exist and be
cleaned (run prod/01_data_pipeline.py and prod/02_eda.py first).
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import TARGET, temporal_split
from src.timeseries_models import train_evaluate_sarimax, train_evaluate_prophet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"

# The single exogenous feature used for the "+1 exog" runs.
EXOG_FEATURE = "Alcohol use disorders"


def _naive_persistence_baseline(
    df_train, df_val, target, id_col="Code", year_col="Year"
):
    """
    No model at all: for each country, predicts every validation-period
    row as that country's last known training-period value. This is
    the number every model below needs to beat to demonstrate it has
    learned something beyond "suicide rate doesn't change much year to
    year".

    Returns
    -------
    dict
        {"rmse": float, "r2": float}
    """
    preds, actuals = [], []
    for code, group in df_val.groupby(id_col):
        train_c = df_train[df_train[id_col] == code].sort_values(year_col)
        if train_c.empty:
            continue
        last_known_value = train_c[target].iloc[-1]
        preds.extend([last_known_value] * len(group))
        actuals.extend(group[target].tolist())

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    r2 = float(r2_score(actuals, preds))
    return {"rmse": round(rmse, 4), "r2": round(r2, 4)}


def run():
    """
    Runs the naive baseline, SARIMAX, and Prophet (each univariate and
    with the single curated exogenous feature) on Option B, and saves a
    single comparison table.

    Returns
    -------
    pd.DataFrame
        One row per variant, columns: Model, Variant, Test RMSE, Test
        R², Val RMSE, Val R².
    """
    df_development = pd.read_parquet(DEVELOPMENT_PATH)
    df_train_B, df_test_B, df_val_B, *_ = temporal_split(df_development)
    logger.info(
        "Option B — Train: %d rows | Test: %d rows | Val: %d rows",
        len(df_train_B),
        len(df_test_B),
        len(df_val_B),
    )

    rows = []

    # --- Naive persistence baseline ---
    naive_val = _naive_persistence_baseline(df_train_B, df_val_B, TARGET)
    logger.info(
        "Naive persistence baseline — Val RMSE: %s | Val R²: %s",
        naive_val["rmse"],
        naive_val["r2"],
    )
    rows.append(
        {
            "Model": "Naive persistence",
            "Variant": "no model",
            "Test RMSE": None,
            "Test R²": None,
            "Val RMSE": naive_val["rmse"],
            "Val R²": naive_val["r2"],
        }
    )

    # --- SARIMAX / Prophet, univariate and +1 exogenous feature ---
    for model_name, train_fn in [
        ("SARIMAX", train_evaluate_sarimax),
        ("Prophet", train_evaluate_prophet),
    ]:
        for variant, exog in [
            ("univariate", None),
            (f"+1 exog ({EXOG_FEATURE})", [EXOG_FEATURE]),
        ]:
            t0 = time.time()
            eval_results, per_country = train_fn(
                df_train_B, df_test_B, df_val_B, TARGET, exog_features=exog
            )
            elapsed = time.time() - t0
            test_e = [e for e in eval_results if e["split"] == "Test"][0]
            val_e = [e for e in eval_results if e["split"] == "Val"][0]
            n_converged = per_country.loc[
                per_country["Split"] == "Test", "converged"
            ].sum()
            logger.info(
                "%s (%s) — %d/%d converged (%.1fs) — Test R²: %s | Val R²: %s",
                model_name,
                variant,
                n_converged,
                df_train_B["Code"].nunique(),
                elapsed,
                test_e["r2"],
                val_e["r2"],
            )
            rows.append(
                {
                    "Model": model_name,
                    "Variant": variant,
                    "Test RMSE": test_e["rmse"],
                    "Test R²": test_e["r2"],
                    "Val RMSE": val_e["rmse"],
                    "Val R²": val_e["r2"],
                }
            )
            per_country.to_parquet(
                TABLES_DIR
                / f"{model_name.lower()}_{'univariate' if exog is None else 'exog'}_per_country.parquet",
                index=False,
            )

    comparison = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(
        TABLES_DIR / "temporal_persistence_check.parquet", index=False
    )
    logger.info("\n%s", comparison.to_string())
    logger.info("Saved: %s", TABLES_DIR / "temporal_persistence_check.parquet")

    return comparison


if __name__ == "__main__":
    run()
