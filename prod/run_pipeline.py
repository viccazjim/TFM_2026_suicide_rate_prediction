"""
Orchestrator: runs the full pipeline in order —
Ingestion/Cleaning (01) -> EDA (02) -> Training/Evaluation (03) ->
Clustering validation (04) -> Temporal persistence check (05) ->
Inference (Pred) -> Prediction plots (06) -> Power BI export (07).

Each stage runs as an independent process (not an import), on purpose:
that way a failure in one stage doesn't leave the interpreter in a
half-built state for the next one, and each stage's log stays separate
with its own exit code.

Usage:
    python prod/run_pipeline.py            # every stage, in order
    python prod/run_pipeline.py --skip-01  # reuse existing data/processed/
    python prod/run_pipeline.py --only 03  # training only (requires 01-02 already run)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_DIR = REPO_ROOT / "prod"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

STAGES = {
    "01": PROD_DIR / "01_data_pipeline.py",
    "02": PROD_DIR / "02_eda.py",
    "03": PROD_DIR / "03_train.py",
    "04": PROD_DIR / "04_clustering.py",
    "05": PROD_DIR / "05_temporal_persistence_check.py",
    "Pred": PROD_DIR / "predict.py",
    "06": PROD_DIR / "06_visualize_predictions.py",
    "07": PROD_DIR / "07_export_powerbi.py",
}

DEFAULT_ORDER = ["01", "02", "03", "04", "05", "Pred", "06", "07"]


def run_stage(stage_key: str):
    """
    Runs one pipeline stage as a fresh subprocess (`python <script>`,
    inheriting stdout/stderr so its own logging shows inline) and exits
    the whole orchestrator immediately if that stage fails — nothing
    downstream is attempted on top of a stage that didn't succeed.

    Parameters
    ----------
    stage_key : str
        A key into STAGES, e.g. "01", "Pred", "07".

    Raises
    ------
    SystemExit
        If the stage's subprocess returns a non-zero exit code — this
        function calls sys.exit() with that same code rather than
        raising, so run_pipeline.py's own exit code reflects it too.
    """
    script_path = STAGES[stage_key]
    logger.info("=" * 60)
    logger.info("Stage %s — %s", stage_key, script_path.name)
    logger.info("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT))
    if result.returncode != 0:
        logger.error(
            "Stage %s failed (exit code %d) — stopping the pipeline.",
            stage_key,
            result.returncode,
        )
        sys.exit(result.returncode)
    logger.info("Stage %s completed successfully.\n", stage_key)


def run(stage_keys):
    """
    Runs every stage in `stage_keys`, in order, via `run_stage()` —
    the orchestrator's own entry point, called either with the full
    DEFAULT_ORDER or a single stage from `--only`.

    Parameters
    ----------
    stage_keys : list[str]
        Stage keys to run in sequence, e.g. DEFAULT_ORDER or a
        single-element list for `--only`.
    """
    for key in stage_keys:
        run_stage(key)
    logger.info("Pipeline complete. Production model available in outputs/models/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-01", action="store_true", help="Skip stage 01 (ingestion/cleaning)"
    )
    parser.add_argument("--skip-02", action="store_true", help="Skip stage 02 (EDA)")
    parser.add_argument(
        "--only", choices=list(STAGES.keys()), help="Run only one specific stage"
    )
    args = parser.parse_args()

    if args.only:
        run([args.only])
    else:
        keys = [
            k
            for k in DEFAULT_ORDER
            if not ((k == "01" and args.skip_01) or (k == "02" and args.skip_02))
        ]
        run(keys)
