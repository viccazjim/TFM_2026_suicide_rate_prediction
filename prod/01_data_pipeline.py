"""
01 — Data pipeline: ingestion and cleaning.

Refactor of notebooks/01_data_loading_cleaning.ipynb into a runnable
script. Builds df_development (2000-2021, labeled, used for modelling)
and df_real_world (2022-2023, unlabeled, reference only) from three
sources:

  1. IHME — mental health disorder prevalence (local CSV)
  2. World Bank API — socioeconomic indicators
  3. WHO API — suicide rate (target variable)

Usage:
    python prod/01_data_pipeline.py

Requires internet access (World Bank + WHO APIs). No arguments needed:
paths are resolved relative to the repo root, regardless of the
directory the script is run from.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import (
    EU_COUNTRIES_ISO,
    WORLD_BANK_INDICATORS,
    WHO_SUICIDE_INDICATOR,
    load_ihme_data,
    fetch_worldbank_indicators,
    fetch_who_suicide_rates,
    build_master_dataset,
    interpolate_with_trend_extrapolation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IHME_CSV_PATH = REPO_ROOT / "data" / "raw" / "IHME-GBD_2023_DATA-b62eec84-1.csv"
DEVELOPMENT_CSV_PATH = REPO_ROOT / "data" / "processed" / "df_development.csv"
REAL_WORLD_CSV_PATH = REPO_ROOT / "data" / "processed" / "df_real_world.csv"


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full ingestion and cleaning pipeline.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_development, df_real_world) — the same pair produced by
        01_data_loading_cleaning.ipynb, already saved to
        data/processed/ by the time this returns.
    """
    np.random.seed(42)

    # --- Step 1: IHME (mental health prevalence) ---
    logger.info("Step 1/4 — Loading IHME base dataset from %s", IHME_CSV_PATH)
    df_base = load_ihme_data(str(IHME_CSV_PATH), min_year=2000)
    logger.info("  IHME: %d rows, %d columns", df_base.shape[0], df_base.shape[1])

    # --- Step 2: World Bank (socioeconomic indicators, API) ---
    logger.info("Step 2/4 — Fetching World Bank indicators (API)")
    df_features = fetch_worldbank_indicators(
        df_base,
        eu_countries_iso=EU_COUNTRIES_ISO,
        indicators=WORLD_BANK_INDICATORS,
        region="ALL",
        date_range="2000:2026",
    )
    logger.info(
        "  After World Bank merge: %d rows, %d columns",
        df_features.shape[0], df_features.shape[1],
    )

    # --- Step 3: WHO (suicide rate, target variable, API) ---
    logger.info("Step 3/4 — Fetching WHO suicide rate (API)")
    df_who = fetch_who_suicide_rates(indicator=WHO_SUICIDE_INDICATOR)
    df_development, df_real_world = build_master_dataset(
        df_features, df_who, development_cutoff_year=2021
    )
    logger.info(
        "  df_development: %d rows (2000-2021, labeled) | df_real_world: %d rows (2022-2023, unlabeled)",
        len(df_development), len(df_real_world),
    )

    # --- Step 4: missing value imputation ---
    features_with_nan = df_development.columns[df_development.isnull().any()].tolist()
    if features_with_nan:
        logger.info("Step 4/4 — Imputing missing values in: %s", features_with_nan)
        df_development = interpolate_with_trend_extrapolation(df_development, features_with_nan)
        remaining = df_development.isnull().sum()
        if remaining.any():
            logger.warning("  Missing values remain after imputation:\n%s", remaining[remaining > 0])
        else:
            logger.info("  No missing values after imputation.")
    else:
        logger.info("Step 4/4 — No missing values, skipping imputation.")

    # --- Save ---
    DEVELOPMENT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_development.to_csv(DEVELOPMENT_CSV_PATH, index=False)
    df_real_world.to_csv(REAL_WORLD_CSV_PATH, index=False)
    logger.info("Saved: %s", DEVELOPMENT_CSV_PATH)
    logger.info("Saved: %s", REAL_WORLD_CSV_PATH)

    return df_development, df_real_world


if __name__ == "__main__":
    run()
