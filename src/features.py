"""
Feature engineering utilities.

"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from typing import Optional


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Computes the Variance Inflation Factor (VIF) for each predictor in
    feature_cols, using all other columns in feature_cols as regressors.
    VIF > 5: moderate concern; VIF > 10: high concern.

    Rows with any NaN in feature_cols are dropped before fitting —
    this means the row count used
    here may be lower than len(df) if any predictor has missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the columns listed in feature_cols.
    feature_cols : list[str]
        Predictor columns to check for multicollinearity against each
        other. Should not include the target or ID columns.

    Returns
    -------
    pd.DataFrame
        Columns "Feature" and "VIF", one row per feature_cols entry,
        sorted by VIF descending.
    """
    X = df[feature_cols].dropna()
    X_const = add_constant(X)
    X_const_array = np.asarray(X_const)
    vif_data = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "VIF": [
                    variance_inflation_factor(X_const_array, i + 1)
                    for i in range(len(X.columns))
                ],
            }
        )
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
    return vif_data


def build_predictor_list(
    df: pd.DataFrame,
    id_cols: list[str],
    target: str,
    extra_exclude: Optional[list[str]] = None,
) -> list[str]:
    """
    Returns every column in df that is neither an ID column, the target,
    nor in extra_exclude.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.
    id_cols : list[str]
        Columns that have no predictive value (e.g. ["Country", "Code", "Year", "Region"]).
    target : str
        Name of the target column ("Suicide rate") — excluded to
        avoid leaking it into its own feature set.
    extra_exclude : list[str], optional
        Additional columns to drop, e.g. low-importance features
        identified from a SHAP/feature-importance ranking that you want
        to test removing without hardcoding the change into id_cols.

    Returns
    -------
    list[str]
        Column names in df, in their original order, excluding id_cols,
        target, and extra_exclude.
    """
    exclude = set(id_cols) | {target} | set(extra_exclude or [])
    return [c for c in df.columns if c not in exclude]


def flag_outliers_iqr(
    df: pd.DataFrame, columns: list[str], threshold: float = 1.5
) -> pd.DataFrame:
    """
    Flags rows as outliers using the IQR method for each of the given
    columns, independently per column (a row can be an outlier on one
    column and not another — returns per-feature counts).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check.
    columns : list[str]
        Columns to test for outliers.
    threshold : float, default 1.5
        IQR multiplier defining the outlier bounds
        (lower = Q1 - threshold*IQR, upper = Q3 + threshold*IQR).
        1.5 = standard convention, 3.0 = extreme values only.

    Returns
    -------
    pd.DataFrame
        One row per column in `columns` (that exists in df), with:
        "Feature", "Outlier count", "Outlier %", "Lower bound", "Upper bound".
        Sorted by "Outlier count" descending.
    """
    results = []
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        results.append(
            {
                "Feature": col,
                "Outlier count": mask.sum(),
                "Outlier %": round(mask.sum() / len(df) * 100, 2),
                "Lower bound": round(lower, 2),
                "Upper bound": round(upper, 2),
            }
        )
    return pd.DataFrame(results).sort_values("Outlier count", ascending=False)
