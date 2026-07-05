"""
Inference: scores new rows with the production model persisted by
prod/03_train.py (CatBoost, Option B — the one that generalizes in a
stable way, see docs/ for the reasoning).

Usage:
    # Score df_real_world.csv (2022-2023, no WHO label yet) — default case
    python prod/predict.py

    # Score your own CSV (must have the same predictor columns as
    # data/processed/df_development.csv, without the "Suicide rate" column)
    python prod/predict.py --input path/to/your_data.csv --output path/to/output.csv

Requires prod/03_train.py to have run first (so that
outputs/models/catboost_option_b.joblib and scaler_option_b.joblib exist).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src import ID_COLS, TARGET, build_predictor_list, load_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = REPO_ROOT / "outputs" / "models" / "catboost_option_b.joblib"
SCALER_PATH = REPO_ROOT / "outputs" / "models" / "scaler_option_b.joblib"
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "df_real_world.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "tables" / "predictions.csv"


def predict(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores a DataFrame with the production model.

    Parameters
    ----------
    input_df : pd.DataFrame
        Must contain the same predictor columns used at training time
        (see src/features.py:build_predictor_list). May or may not
        include ID columns (Country, Code, Year, Region) and the target
        column — both are ignored if present.

    Returns
    -------
    pd.DataFrame
        `input_df` with an extra "Predicted suicide rate" column.

    Raises
    ------
    FileNotFoundError
        If the model/scaler artifacts don't exist — run
        prod/03_train.py first.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at {MODEL_PATH} or scaler not found at "
            f"{SCALER_PATH}. Run first: python prod/03_train.py"
        )

    model = load_artifact(str(MODEL_PATH))
    scaler = load_artifact(str(SCALER_PATH))

    predictor_features = build_predictor_list(input_df, ID_COLS, TARGET)
    missing = [f for f in scaler.feature_names_in_ if f not in predictor_features]
    if missing:
        raise ValueError(
            f"Missing predictor columns required by the model: {missing}"
        )

    X = input_df[list(scaler.feature_names_in_)].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    output_df = input_df.copy()
    output_df["Predicted suicide rate"] = model.predict(X_scaled)
    return output_df


def run(input_path: Path, output_path: Path):
    logger.info("Loading input data: %s", input_path)
    input_df = pd.read_csv(input_path)
    logger.info("  %d rows, %d columns", input_df.shape[0], input_df.shape[1])

    output_df = predict(input_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Predictions saved: %s", output_path)
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH,
                         help=f"Input CSV (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                         help=f"Output CSV (default: {DEFAULT_OUTPUT_PATH})")
    args = parser.parse_args()
    run(args.input, args.output)
