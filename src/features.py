"""
Feature engineering utilities.

Origin (EDA_models_VC.ipynb):
- compute_vif           <- cell 26
- build_predictor_list  <- logic reused from cells 22 / 28 / 63
- add_lag_features      <- cell 61 (cleaned: NO deltas, see conversation history —
                           suicide_rate_delta was dropped because it leaked the
                           current-year target: delta = y(t) - y(t-1), and
                           lag1 + delta = y(t) exactly)
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Computes the Variance Inflation Factor (VIF) for each predictor.
    VIF > 5: moderate concern; VIF > 10: high concern.
    """
    X = df[feature_cols].dropna()
    X_const = add_constant(X)
    vif_data = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "VIF": [
                    variance_inflation_factor(X_const.values, i + 1)
                    for i in range(len(X.columns))
                ],
            }
        )
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
    return vif_data


def build_predictor_list(df: pd.DataFrame, id_cols: list[str], target: str) -> list[str]:
    """Every column that is not an ID column and not the target."""
    return [c for c in df.columns if c not in id_cols + [target]]


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    predictor_features: list[str],
    lag_all_predictors: bool = True,
) -> pd.DataFrame:
    """
    Adds t-1 lag features, sorted by Country/Year. No delta features are
    created — see module docstring for why.

    Parameters
    ----------
    lag_all_predictors : if True, also lags every column in predictor_features
        (needed for the ablation study in notebook 04). If False, only the
        target lag is added.

    Returns a new DataFrame with rows containing NaN lags (first year per
    country) already dropped.
    """
    out = df.copy().sort_values(["Country", "Year"]).reset_index(drop=True)

    out[f"{target.lower().replace(' ', '_')}_lag1"] = (
        out.groupby("Country")[target].shift(1)
    )
    lag_cols = [f"{target.lower().replace(' ', '_')}_lag1"]

    if lag_all_predictors:
        for col in predictor_features:
            lag_col = f"{col}_lag1"
            out[lag_col] = out.groupby("Country")[col].shift(1)
            lag_cols.append(lag_col)

    out = out.dropna(subset=lag_cols).reset_index(drop=True)
    return out
