"""
Result-comparison tables and metrics.

"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_results_table(eval_list, trained_dict, split_filter, approach_label):
    """
    Builds a single comparison table from a list of evaluate_model()
    outputs, for one split at a time.

    Parameters
    ----------
    eval_list : list[dict]
        Concatenated outputs of evaluate_model() calls, possibly mixing
        multiple models and multiple splits — this function filters down
        to split_filter internally.
    trained_dict : dict[str, dict]
        Mapping {model_name: train_model() output} — used to pull in
        "cv_rmse" and "time_s", which are not part of evaluate_model()'s
        output.
    split_filter : str
        Which split to build the table for, e.g. "Test" or "Val". Rows in
        eval_list with a different "split" value are ignored.
    approach_label : str
        Label printed above the table (e.g. "Option B", "No Year") —
        cosmetic only, not stored in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: "Model", "CV RMSE", "RMSE", "MAE", "R²", "Time (s)".
        One row per model present in eval_list for split_filter, sorted
        by "RMSE" ascending (best model first).
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


def get_eval_entry(eval_list, model_name, split_filter):
    """
    Retrieves a single evaluate_model() output from an eval_list, by
    model name and split — used to pull the exact predictions/actuals
    needed for result diagnostics and SHAP (e.g. "give me CatBoost's
    Validation predictions") without re-running evaluate_model().

    Parameters
    ----------
    eval_list : list[dict]
        Concatenated outputs of evaluate_model() calls (e.g. eval_A, eval_B).
    model_name : str
        E.g. "CatBoost" — must match the "model" key set by evaluate_model().
    split_filter : str
        E.g. "Test" or "Val".

    Returns
    -------
    dict
        The matching evaluate_model() output, containing at least
        "model", "split", "rmse", "mae", "r2", "predictions", "actuals".

    Raises
    ------
    ValueError
        If no entry matches both model_name and split_filter.
    """
    for r in eval_list:
        if r["model"] == model_name and r["split"] == split_filter:
            return r
    raise ValueError(
        f"No entry found for model={model_name!r}, split={split_filter!r}. "
        f"Available: {[(r['model'], r['split']) for r in eval_list]}"
    )
