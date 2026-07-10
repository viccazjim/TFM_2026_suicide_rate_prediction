"""
06 — Visualize predict.py output, alongside the best temporal model.

Loads the predictions saved by prod/predict.py (CatBoost — the
production model that actually answers the thesis question, see
predict.py's module docstring) together with the historical
df_development.parquet, and generates:

  - Trend continuation plots (actual vs predicted) for the same
    spotlight countries used in 02_eda.py (Lithuania, Greece, Germany),
    as a sanity check that predictions look like a plausible
    continuation rather than an implausible jump — CatBoost alone,
    and CatBoost overlaid with the best temporal-persistence model
    (SARIMAX + 1 exogenous feature, from 05_temporal_persistence_check.py)
    for the same years, so the two can be read side by side.
  - Bar charts ranking countries by predicted suicide rate, one per
    predicted year — again both single-model and side-by-side.
  - The same comparison aggregated up to the a priori EU_REGIONS
    grouping (used descriptively in 02_eda.py and 04_clustering.py) —
    trend and bar-chart versions, so it's visible whether each model's
    predictions line up with that regional grouping once averaged, not
    just at the individual-country level above.

**Why compare against a second model that 05_temporal_persistence_check.py
already showed answers a different question.** The point isn't to pick
a "better" forecast — it's to see, country by country, whether a model
built purely from socioeconomic/mental-health determinants (CatBoost)
and one built purely from a country's own persistence plus one
determinant (SARIMAX + exog) end up telling a similar story or
diverging. Agreement is a mild corroborating signal; disagreement is
not a bug in either model, just a reminder that they were built to
answer different questions and aren't expected to coincide exactly.

Figures are saved to outputs/figures/ with the "06_" prefix.

Usage:
    python prod/06_visualize_predictions.py
    python prod/06_visualize_predictions.py --predictions path/to/other_predictions.parquet

Requires prod/predict.py to have run first (for the CatBoost
predictions). The temporal-model forecast is computed inline here —
it's a comparison plot, not a second production inference path the
way predict.py is — but the fitted per-country SARIMAX (+exog) models
are still saved to outputs/models/sarimax_exog_models.joblib on every
run, purely so they're available for later inspection or reuse without
re-fitting; this script always fits fresh rather than loading an
existing artifact, the same way 03_train.py always retrains.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this script only saves figures to disk, never displays them

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import (
    TARGET,
    EU_REGIONS,
    temporal_split,
    plot_predictions_trend,
    plot_predictions_by_country,
    plot_predictions_model_comparison,
    plot_predictions_by_country_comparison,
    plot_predictions_by_region_comparison,
    plot_predictions_trend_by_region,
    save_figure,
    save_artifact,
)
from src.timeseries_models import fit_sarimax_models, forecast_sarimax

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
REAL_WORLD_PATH = REPO_ROOT / "data" / "processed" / "df_real_world.parquet"
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "outputs" / "tables" / "predictions.parquet"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
FIG_PREFIX = "06_"

# Same spotlight countries used in 02_eda.py, for a consistent narrative
# across the EDA and the prediction sanity-check plots.
SPOTLIGHT_COUNTRIES = [("LTU", "Lithuania"), ("GRC", "Greece"), ("DEU", "Germany")]

MODEL_A_NAME = "CatBoost"
MODEL_B_NAME = "SARIMAX +1 exog"
TEMPORAL_EXOG_FEATURE = "Alcohol use disorders"  # matches 05_temporal_persistence_check.py's choice
TEMPORAL_ORDER = (1, 1, 0)


def _forecast_temporal_model(df_history: pd.DataFrame, df_real_world: pd.DataFrame) -> pd.DataFrame:
    """
    Fits SARIMAX + the single curated exogenous feature on Option B's
    training years, then forecasts forward through every intervening
    year up to df_real_world's years (2022-2023) — SARIMAX walks
    forward step by step from the end of training, so the test/val
    years have to be forecast too even though only 2022-2023 is kept
    (see src/timeseries_models.py::forecast_sarimax's docstring).

    Returns
    -------
    pd.DataFrame
        Same shape/columns as predict.py's output: input df_real_world
        rows plus a "Predicted suicide rate" column.
    """
    df_train_B, df_test_B, df_val_B, *_ = temporal_split(df_history)
    logger.info("Fitting SARIMAX (+%s) per country on Option B training years", TEMPORAL_EXOG_FEATURE)
    models = fit_sarimax_models(
        df_train_B, TARGET, exog_features=[TEMPORAL_EXOG_FEATURE], order=TEMPORAL_ORDER
    )
    n_converged = len(models)
    n_countries = df_train_B["Code"].nunique()
    if n_converged < n_countries:
        missing = set(df_train_B["Code"].unique()) - set(models.keys())
        logger.warning("Only %d/%d countries converged — missing: %s", n_converged, n_countries, sorted(missing))

    # Persisted mainly so the fitted per-country models are available for
    # later inspection/reuse without re-fitting — not because this script
    # itself needs to reload them; it always fits fresh each run, the same
    # way 03_train.py always retrains rather than checking for an existing
    # artifact first.
    models_path = save_artifact(models, str(MODELS_DIR / "sarimax_exog_models.joblib"))
    logger.info("SARIMAX (+%s) models saved: %s", TEMPORAL_EXOG_FEATURE, models_path)

    future_block = pd.concat([df_test_B, df_val_B, df_real_world]).sort_values(["Code", "Year"])

    rows = []
    for code, group in df_real_world.groupby("Code"):
        if code not in models:
            continue
        country_future = future_block[future_block["Code"] == code]
        forecast = forecast_sarimax(models, code, country_future, exog_features=[TEMPORAL_EXOG_FEATURE])
        for _, row in group.iterrows():
            rows.append({
                "Country": row["Country"], "Code": code, "Year": row["Year"],
                "Predicted suicide rate": forecast.loc[row["Year"]],
            })

    return pd.DataFrame(rows)


def run(predictions_path: Path):
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"No predictions file found at {predictions_path}. Run first: python prod/predict.py"
        )

    logger.info("Loading historical data: %s", DEVELOPMENT_PATH)
    df_history = pd.read_parquet(DEVELOPMENT_PATH)
    df_real_world = pd.read_parquet(REAL_WORLD_PATH)

    logger.info("Loading CatBoost predictions: %s", predictions_path)
    df_predictions_catboost = pd.read_parquet(predictions_path) if predictions_path.suffix == ".parquet" else pd.read_csv(predictions_path)
    logger.info("  %d rows, years: %s", len(df_predictions_catboost), sorted(df_predictions_catboost["Year"].unique().tolist()))

    df_predictions_temporal = _forecast_temporal_model(df_history, df_real_world)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_predictions_temporal.to_parquet(TABLES_DIR / "predictions_temporal.parquet", index=False)
    logger.info("Saved: %s", TABLES_DIR / "predictions_temporal.parquet")

    # --- Single-model plots (CatBoost only, as before) ---
    for code, name in SPOTLIGHT_COUNTRIES:
        fig = plot_predictions_trend(df_history, df_predictions_catboost, code, name)
        save_figure(fig, name=f"predictions_trend_{code.lower()}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))
    for year in sorted(df_predictions_catboost["Year"].unique()):
        fig = plot_predictions_by_country(df_predictions_catboost, year)
        save_figure(fig, name=f"predictions_by_country_{year}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    # --- Comparison plots: CatBoost vs SARIMAX +1 exog ---
    for code, name in SPOTLIGHT_COUNTRIES:
        logger.info("Generating model-comparison trend for %s", name)
        fig = plot_predictions_model_comparison(
            df_history, df_predictions_catboost, df_predictions_temporal,
            MODEL_A_NAME, MODEL_B_NAME, code, name,
        )
        save_figure(fig, name=f"predictions_comparison_trend_{code.lower()}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    for year in sorted(df_predictions_catboost["Year"].unique()):
        logger.info("Generating model-comparison bar chart for %d", year)
        fig = plot_predictions_by_country_comparison(
            df_predictions_catboost, df_predictions_temporal, MODEL_A_NAME, MODEL_B_NAME, year,
        )
        save_figure(fig, name=f"predictions_comparison_by_country_{year}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    # --- Region-level comparison: how do the predictions look for the a
    # priori EU_REGIONS grouping used descriptively throughout this project
    # (02_eda.py, 04_clustering.py)? Countries are averaged up into their
    # region for both the trend and the bar-chart comparisons. ---
    df_history["Region"] = df_history["Code"].map(EU_REGIONS)
    df_predictions_catboost["Region"] = df_predictions_catboost["Code"].map(EU_REGIONS)
    df_predictions_temporal["Region"] = df_predictions_temporal["Code"].map(EU_REGIONS)

    for region_name in sorted(set(EU_REGIONS.values())):
        logger.info("Generating region-level trend comparison for %s", region_name)
        fig = plot_predictions_trend_by_region(
            df_history, df_predictions_catboost, df_predictions_temporal,
            MODEL_A_NAME, MODEL_B_NAME, region_name,
        )
        save_figure(fig, name=f"predictions_comparison_trend_region_{region_name.lower().replace('/', '_').replace(' ', '_')}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    for year in sorted(df_predictions_catboost["Year"].unique()):
        logger.info("Generating region-level bar chart for %d", year)
        fig = plot_predictions_by_region_comparison(
            df_predictions_catboost, df_predictions_temporal, MODEL_A_NAME, MODEL_B_NAME, year,
        )
        save_figure(fig, name=f"predictions_comparison_by_region_{year}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    logger.info("Done. Figures saved under %s with prefix '%s'.", FIGURES_DIR, FIG_PREFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH,
                         help=f"CatBoost predictions file to visualize (default: {DEFAULT_PREDICTIONS_PATH})")
    args = parser.parse_args()
    run(args.predictions)
