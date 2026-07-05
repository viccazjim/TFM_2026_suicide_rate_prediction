"""
Model registry, hyperparameter grids, and train/evaluate functions.

"""

import time
from math import prod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# --------------------------------------------------------------------------
# Linear / kernel / tree baseline models
# --------------------------------------------------------------------------
param_grids = {
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100, 1000]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    "SVR": {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.1, 0.5, 1.0],
        "kernel": ["linear", "rbf"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "CatBoost": {
        "iterations": [200, 400],
        "depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [3, 7, 15],
    },
}


def make_models(random_state: int = 42) -> dict:
    """
    Builds fresh, unfitted instances of all models.

    Created because GridSearchCV mutates the estimator it is given,
    so a new instance for each run needs to be created.

    Parameters
    ----------
    random_state : int, default 42
        Passed to Random Forest for reproducibility. Ridge/Lasso/SVR have
        no stochastic component here and ignore this.

    Returns
    -------
    dict[str, sklearn estimator]
        {"Ridge": Ridge(), "Lasso": Lasso(...), ...}, unfitted.
    """
    return {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "SVR": SVR(),
        "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(
            random_state=random_state, n_jobs=-1, objective="reg:squarederror"
        ),
        "CatBoost": CatBoostRegressor(random_state=random_state, verbose=0),
    }


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def train_model(name, model, param_grid, X_train, y_train, cv):
    """
    Runs GridSearchCV for a single model, refits on the full training set
    with the best found parameters, and returns the fitted estimator plus
    training metadata.

    Parameters
    ----------
    name : str
        Display name for the model (used in print statements and as the
        key when storing results elsewhere, e.g. build_results_table()).
    model : sklearn-compatible estimator
        Unfitted estimator.
    param_grid : dict
        Hyperparameter grid for GridSearchCV, e.g. param_grids["Ridge"].
    X_train : pd.DataFrame
        Scaled training features.
    y_train : pd.Series
        Training target.
    cv : int or sklearn cross-validator
        Pass an int for standard k-fold (geographical_split — countries
        are independent observations, order does not matter). Pass a
        TimeSeriesSplit instance for temporal_split — years are ordered
        and must not be shuffled, or the model would train on future data
        to predict the past within a fold.

    Returns
    -------
    dict
        {
          "name": str,
          "best_estimator": fitted sklearn estimator (refit on full X_train/y_train
                             with the best params found),
          "best_params": dict,
          "cv_rmse": float (mean RMSE across CV folds, at the best params),
          "time_s": float (wall-clock seconds for the GridSearchCV.fit call),
        }
    """
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(
        f"  Grid size: {len(param_grid)} params | "
        f"Combinations: {prod(len(v) for v in param_grid.values())}"
    )
    print(f"{'=' * 60}")

    start = time.time()
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)
    cv_rmse = round(-search.best_score_, 4)

    print(f"  Best params : {search.best_params_}")
    print(f"  CV RMSE     : {cv_rmse}")
    print(f"  Training time: {elapsed}s")

    return {
        "name": name,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_rmse": cv_rmse,
        "time_s": elapsed,
    }


def evaluate_model(trained, X, y, split_label):
    """
    Evaluates a trained model (from train_model()) on a given data split
    and returns its metrics plus the raw predictions/actuals.

    Parameters
    ----------
    trained : dict
        Output of train_model() — must contain "best_estimator" and "name".
    X : pd.DataFrame
        Scaled features for the split being evaluated (e.g. test or val).
    y : pd.Series
        True target values for that split.
    split_label : str
        Label identifying the split (e.g. "Test", "Val") — stored in the
        output and printed, used downstream to filter results
        (e.g. build_results_table(..., split_filter="Test", ...)).

    Returns
    -------
    dict
        {
          "model": str, "split": str,
          "rmse": float, "mae": float, "r2": float,
          "predictions": np.ndarray, "actuals": np.ndarray,
        }
        predictions/actuals are included so downstream diagnostics (e.g.
        plot_residuals_by_year) can recompute residuals without
        re-predicting.
    """
    model = trained["best_estimator"]
    name = trained["name"]
    y_pred = model.predict(X)

    rmse = round(np.sqrt(mean_squared_error(y, y_pred)), 4)
    mae = round(mean_absolute_error(y, y_pred), 4)
    r2 = round(r2_score(y, y_pred), 4)

    print(
        f"  {name:20s} | {split_label:10s} — "
        f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}"
    )

    return {
        "model": name,
        "split": split_label,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": y_pred,
        "actuals": y.values,
    }
