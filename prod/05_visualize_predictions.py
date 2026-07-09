"""
05 — Visualize predict.py output.

Loads the predictions saved by prod/predict.py (default:
outputs/tables/predictions.csv) together with the historical
df_development.csv, and generates:

  - A trend continuation plot (actual vs predicted) for the same
    spotlight countries used in 02_eda.py (Lithuania, Greece, Germany),
    as a sanity check that predictions look like a plausible
    continuation rather than an implausible jump.
  - A bar chart ranking all countries by predicted suicide rate, one
    per predicted year.

Figures are saved to outputs/figures/ with the "04_" prefix.

Usage:
    python prod/05_visualize_predictions.py
    python prod/05_visualize_predictions.py --predictions path/to/other_predictions.csv

Requires prod/predict.py to have run first.
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

from src import plot_predictions_trend, plot_predictions_by_country, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVELOPMENT_PATH = REPO_ROOT / "data" / "processed" / "df_development.parquet"
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "outputs" / "tables" / "predictions.csv"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
FIG_PREFIX = "05_"

# Same spotlight countries used in 02_eda.py, for a consistent narrative
# across the EDA and the prediction sanity-check plots.
SPOTLIGHT_COUNTRIES = [("LTU", "Lithuania"), ("GRC", "Greece"), ("DEU", "Germany")]


def run(predictions_path: Path):
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"No predictions file found at {predictions_path}. Run first: python prod/predict.py"
        )

    logger.info("Loading historical data: %s", DEVELOPMENT_PATH)
    df_history = pd.read_parquet(DEVELOPMENT_PATH)

    logger.info("Loading predictions: %s", predictions_path)
    df_predictions = pd.read_csv(predictions_path)
    logger.info("  %d rows, years: %s", len(df_predictions), sorted(df_predictions["Year"].unique().tolist()))

    # --- Trend continuation, per spotlight country ---
    for code, name in SPOTLIGHT_COUNTRIES:
        logger.info("Generating actual-vs-predicted trend for %s", name)
        fig = plot_predictions_trend(df_history, df_predictions, code, name)
        save_figure(fig, name=f"predictions_trend_{code.lower()}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    # --- Bar chart ranking, per predicted year ---
    for year in sorted(df_predictions["Year"].unique()):
        logger.info("Generating predicted-by-country bar chart for %d", year)
        fig = plot_predictions_by_country(df_predictions, year)
        save_figure(fig, name=f"predictions_by_country_{year}", prefix=FIG_PREFIX, figures_dir=str(FIGURES_DIR))

    logger.info("Done. Figures saved under %s with prefix '%s'.", FIGURES_DIR, FIG_PREFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH,
                         help=f"Predictions CSV to visualize (default: {DEFAULT_PREDICTIONS_PATH})")
    args = parser.parse_args()
    run(args.predictions)
