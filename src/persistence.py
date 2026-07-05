"""
Model/scaler persistence (joblib).

Separated from models.py: that module builds and evaluates estimators
in-memory for comparison purposes; this one is what makes a trained
estimator usable outside the process that trained it.
"""

import os
import joblib


def save_artifact(obj, path: str) -> str:
    """
    Saves any picklable object (a fitted estimator, a fitted scaler,
    etc.) to `path` via joblib. Creates parent directories if needed.

    Parameters
    ----------
    obj : Any
        Typically a fitted sklearn/CatBoost/XGBoost estimator or a
        fitted scaler (e.g. RobustScaler).
    path : str
        Destination file path, e.g. "../outputs/models/catboost_option_b.joblib".

    Returns
    -------
    str
        The path the object was saved to (same as input, returned for
        convenience/chaining).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifact(path: str):
    """
    Loads an object previously saved with save_artifact().

    Parameters
    ----------
    path : str
        Path to a .joblib file.

    Returns
    -------
    Any
        The deserialized object (estimator, scaler, ...).

    Raises
    ------
    FileNotFoundError
        If `path` does not exist — raised by joblib/pickle directly,
        not caught here, since a missing model artifact should stop
        inference rather than silently returning None.
    """
    return joblib.load(path)
