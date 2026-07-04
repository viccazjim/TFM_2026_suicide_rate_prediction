"""
Model registry, hyperparameter grids, and train/evaluate functions.

Origin (EDA_models_VC.ipynb):
- param_grids / models      <- cell 48
- param_grids_boost / models_boost <- added later in the thesis conversation
  (XGBoost/CatBoost), grids kept conservative on purpose: this dataset has
  a small sample size after the train/test/val split (27 EU countries x
  ~20 years), so deep trees overfit easily.
- train_model / evaluate_model <- cell 49
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
}


def make_baseline_models(random_state: int = 42) -> dict:
    """Fresh (unfitted) instances — call this each time you start a new
    GridSearchCV run, never reuse a fitted estimator across configs."""
    return {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "SVR": SVR(),
        "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
    }


# --------------------------------------------------------------------------
# Boosted models
# --------------------------------------------------------------------------
param_grids_boost = {
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
        "l2_leaf_reg": [3, 7],
    },
}


def make_boosted_models(random_state: int = 42) -> dict:
    return {
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
    Runs GridSearchCV for a single model and returns the best estimator
    along with training metadata. Evaluation is handled separately by
    evaluate_model(), keeping the two concerns cleanly separated.

    cv: int for standard k-fold (Option A, countries are independent
        observations), or a TimeSeriesSplit object (Option B, years are
        ordered and must not be shuffled).
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
    """Evaluates a trained model on a given split and returns a metrics dict."""
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
