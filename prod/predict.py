"""
Inference: forecasts the suicide rate for known countries using the
production model persisted by prod/04_train.py — one Prophet model per
country (Option B), chosen over both SARIMAX and the 6 panel models by
Validation RMSE/R² (see 04_train.py's module docstring for the exact
numbers and why Val decided it over Test).

**This changes what "scoring new data" means compared to the CatBoost
version this replaces.** Prophet forecasts a country forward from its
own history — it cannot score a country it has never seen, no matter
what predictor values you give it, because there is no history to
extend. In practice this means:

- Works: scoring df_real_world.parquet (2022-2023) — same 27 EU
  countries the model was trained on, just later years.
- Does NOT work: a country absent from training, or a hypothetical
  "what if this country had these socioeconomic values" query — there
  is no country-level history to forecast from in either case. Use the
  CatBoost/scaler/cluster-model artifacts saved alongside the Prophet
  models (see 04_train.py) if you need that kind of arbitrary-row
  scoring; predict.py does not fall back to them automatically.

Usage:
    # Forecast df_real_world.parquet's rows (2022-2023) — default case
    python prod/predict.py

    # Forecast your own file — needs at minimum "Code" and "Year"
    # columns identifying which country/year to forecast; any other
    # columns are carried through untouched, not used by the model
    python prod/predict.py --input path/to/your_data.parquet --output path/to/output.csv

Requires prod/04_train.py to have run first (so that
outputs/models/prophet_models_option_b.joblib exists).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import load_artifact, forecast_prophet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROPHET_MODELS_PATH = REPO_ROOT / "outputs" / "models" / "prophet_models_option_b.joblib"
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "df_real_world.parquet"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "tables" / "predictions.csv"


def predict(input_df: pd.DataFrame, id_col: str = "Code", year_col: str = "Year") -> pd.DataFrame:
    """
    Forecasts "Predicted suicide rate" for every (country, year) row in
    `input_df`, one country's rows at a time, using that country's
    persisted Prophet model.

    Parameters
    ----------
    input_df : pd.DataFrame
        Must contain `id_col` and `year_col`. Any other columns (e.g.
        socioeconomic predictors) are carried through in the output
        unchanged, but are NOT used by the model — Prophet forecasts
        from the country's own history alone (no exogenous regressors
        by default; see src/timeseries_models.py).
    id_col : str, default "Code"
    year_col : str, default "Year"

    Returns
    -------
    pd.DataFrame
        `input_df` with an extra "Predicted suicide rate" column. Rows
        for countries with no fitted model get NaN in that column
        (logged as a warning, not silently dropped — the row-count
        must stay predictable for the caller).

    Raises
    ------
    FileNotFoundError
        If the Prophet models artifact doesn't exist — run
        prod/04_train.py first.
    """
    if not PROPHET_MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at {PROPHET_MODELS_PATH}. Run first: python prod/04_train.py"
        )

    models = load_artifact(str(PROPHET_MODELS_PATH))

    output_df = input_df.copy()
    output_df["Predicted suicide rate"] = float("nan")

    unknown_codes = set(input_df[id_col].unique()) - set(models.keys())
    if unknown_codes:
        logger.warning(
            "No fitted Prophet model for: %s — those rows will get NaN predictions. "
            "Prophet cannot forecast a country it has no training history for.",
            sorted(unknown_codes),
        )

    for code, group in input_df.groupby(id_col):
        if code not in models:
            continue
        years = group[year_col].tolist()
        predictions = forecast_prophet(models, code, years)
        output_df.loc[group.index, "Predicted suicide rate"] = predictions

    return output_df


def run(input_path: Path, output_path: Path):
    logger.info("Loading input data: %s", input_path)
    input_df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    logger.info("  %d rows, %d columns", input_df.shape[0], input_df.shape[1])

    output_df = predict(input_df)

    n_missing = output_df["Predicted suicide rate"].isna().sum()
    if n_missing:
        logger.warning("%d/%d rows have no prediction (unknown country)", n_missing, len(output_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Predictions saved: %s", output_path)
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH,
                         help=f"Input file, .parquet or .csv, needs Code+Year columns (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                         help=f"Output CSV (default: {DEFAULT_OUTPUT_PATH})")
    args = parser.parse_args()
    run(args.input, args.output)
