"""
Result-comparison tables and metrics.

Origin (EDA_models_VC.ipynb + thesis conversation):
- build_results_table    <- cell 55
- persistence_baseline   <- added when checking whether lag-based models beat
                            a naive "y(t) = y(t-1)" predictor
- normalized_rmse_table  <- added to compare RMSE across splits whose target
                            variance (std) differs — R² is not directly
                            comparable across splits with different
                            SS_total, RMSE/std(y) is.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_results_table(eval_list, trained_dict, split_filter, approach_label):
    """
    Builds a comparison DataFrame from evaluate_model() outputs, sorted by
    RMSE ascending.
    """
    rows = []
    for r in eval_list:
        if r["split"] != split_filter:
            continue
        rows.append(
            {
                "Model": r["model"],
                "CV RMSE": trained_dict[r["model"]]["cv_rmse"],
                "RMSE": r["rmse"],
                "MAE": r["mae"],
                "R²": r["r2"],
                "Time (s)": trained_dict[r["model"]]["time_s"],
            }
        )
    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print(f"\n{approach_label} — {split_filter} set results:")
    return df


def persistence_baseline(df_split, target: str, lag_col: str, split_label: str) -> dict:
    """
    Naive baseline: predicts y(t) = y(t-1) directly, no model at all.
    Used to check whether autoregressive/lag-based models are adding any
    real value over simply copying the previous year.
    """
    y_true = df_split[target]
    y_pred = df_split[lag_col]
    return {
        "model": "Persistence (y_t = y_t-1)",
        "split": split_label,
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def normalized_rmse_table(eval_list, y_test_std: float, y_val_std: float) -> pd.DataFrame:
    """
    Adds NRMSE = RMSE / std(y) to each evaluation result. RMSE is in the
    original units and is comparable across splits with different target
    variance; R² is not, because SS_total (the variance of y in that split)
    is part of its denominator.
    """
    rows = []
    for ev in eval_list:
        std_ref = y_test_std if ev["split"] == "Test" else y_val_std
        nrmse = round(ev["rmse"] / std_ref, 4)
        rows.append(
            {
                "model": ev["model"],
                "split": ev["split"],
                "rmse": ev["rmse"],
                "nrmse": nrmse,
                "r2": ev["r2"],
            }
        )
    return pd.DataFrame(rows)
