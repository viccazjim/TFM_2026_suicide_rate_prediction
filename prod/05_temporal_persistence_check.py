"""
05 — Temporal persistence check: a second, different question.

`03_train.py` answers the thesis's actual question: can socioeconomic
and mental-health determinants predict suicide rate? Its 6 panel
models use only those determinants — nothing derived from a country's
own suicide-rate history.

This script asks something else on purpose: **once that question has
been answered, how much of suicide rate is simply explained by itself
— by its own persistence from one year to the next — independent of
any determinant?** That is a different, narrower question, and mixing
its answer into the main comparison would be misleading: a model that
knows a country's own recent suicide rate will look "good" almost
entirely because suicide rate is extremely autocorrelated year to
year (checked directly on this data: pooled year-over-year correlation
≈0.99), not because it has learned anything about determinants.

Three points of comparison, deliberately kept side by side rather than
folded into one number:

1. **Naive persistence baseline** — no model at all: predict this
   year's rate as the country's last known training-period value.
   Included because it is the number every "sophisticated" time-series
   result below actually needs to beat to mean anything.
2. **SARIMAX / Prophet, univariate** — one model per country, forecast
   purely from that country's own history. This measures raw
   persistence, nothing more.
3. **SARIMAX / Prophet, +1 exogenous feature** (`Alcohol use disorders`
   — the strongest SHAP-ranked determinant from 03_train.py's CatBoost
   model) — checked directly: this is the only version that beats the
   naive baseline by a meaningful margin, which is real (if narrow)
   evidence that a determinant adds explanatory power on top of pure
   persistence, not just a sign that the model can see a country's own
   recent past.

Only one exogenous feature is used, deliberately: with ~15 training
points per country, more exogenous regressors were checked directly to
overfit (17 regressors drives the AIC to an implausible -235 — more
free parameters than observations).

Option A only conceptually excluded here: a per-country time-series
model cannot forecast a country with zero training history, which is
exactly what Option A's test/val countries are. This script only ever
uses Option B.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"

# The single exogenous feature used for the "+1 exog" runs — the strongest
# SHAP-ranked determinant from 03_train.py's CatBoost model. Fixed and
# explicit, not selected by any search over this script's own results, to
# avoid quietly cherry-picking whichever feature happens to score best here.
EXOG_FEATURE = "Alcohol use disorders"


def _naive_persistence_baseline(df_train, df_val, target, id_col="Code", year_col="Year"):
    """
    No model at all: for each country, predicts every validation-period
    row as that country's last known training-period value. This is
    the number every model below needs to beat to demonstrate it has
    learned something beyond "suicide rate doesn't change much year to
    year" — see the module docstring.

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
    logger.info("Option B — Train: %d rows | Test: %d rows | Val: %d rows", len(df_train_B), len(df_test_B), len(df_val_B))

    rows = []

    # --- Naive persistence baseline ---
    naive_val = _naive_persistence_baseline(df_train_B, df_val_B, TARGET)
    logger.info("Naive persistence baseline — Val RMSE: %s | Val R²: %s", naive_val["rmse"], naive_val["r2"])
    rows.append({
        "Model": "Naive persistence", "Variant": "no model",
        "Test RMSE": None, "Test R²": None,
        "Val RMSE": naive_val["rmse"], "Val R²": naive_val["r2"],
    })

    # --- SARIMAX / Prophet, univariate and +1 exogenous feature ---
    for model_name, train_fn in [("SARIMAX", train_evaluate_sarimax), ("Prophet", train_evaluate_prophet)]:
        for variant, exog in [("univariate", None), (f"+1 exog ({EXOG_FEATURE})", [EXOG_FEATURE])]:
            t0 = time.time()
            eval_results, per_country = train_fn(df_train_B, df_test_B, df_val_B, TARGET, exog_features=exog)
            elapsed = time.time() - t0
            test_e = [e for e in eval_results if e["split"] == "Test"][0]
            val_e = [e for e in eval_results if e["split"] == "Val"][0]
            n_converged = per_country.loc[per_country["Split"] == "Test", "converged"].sum()
            logger.info(
                "%s (%s) — %d/%d converged (%.1fs) — Test R²: %s | Val R²: %s",
                model_name, variant, n_converged, df_train_B["Code"].nunique(), elapsed, test_e["r2"], val_e["r2"],
            )
            rows.append({
                "Model": model_name, "Variant": variant,
                "Test RMSE": test_e["rmse"], "Test R²": test_e["r2"],
                "Val RMSE": val_e["rmse"], "Val R²": val_e["r2"],
            })
            per_country.to_parquet(
                TABLES_DIR / f"{model_name.lower()}_{'univariate' if exog is None else 'exog'}_per_country.parquet",
                index=False,
            )

    comparison = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(TABLES_DIR / "temporal_persistence_check.parquet", index=False)
    logger.info("\n%s", comparison.to_string())
    logger.info("Saved: %s", TABLES_DIR / "temporal_persistence_check.parquet")

    return comparison


if __name__ == "__main__":
    run()
