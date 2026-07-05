"""
Model interpretability via SHAP.

Kept separate from diagnostics.py: diagnostics describes prediction
*error* (residuals, actual vs predicted), this module explains *why* the
model predicts what it predicts, per feature.

"""

import numpy as np
import pandas as pd
import shap
from typing import Optional

TREE_MODEL_MODULES = ("sklearn.ensemble", "xgboost", "catboost")


def _is_tree_model(model) -> bool:
    """
    Check for whether `model` is a tree.
    """
    module = type(model).__module__
    return any(module.startswith(prefix) for prefix in TREE_MODEL_MODULES)


def make_shap_explainer(
    model,
    background: Optional[pd.DataFrame] = None,
    background_size: int = 100,
    random_state: int = 42,
):
    """
    Builds a SHAP explainer appropriate for the given model.

    Tree ensembles (Random Forest, XGBoost, CatBoost) use `TreeExplainer`,
    which is exact and fast. Everything else (Ridge, Lasso, SVR) falls
    back to the generic `shap.Explainer`, which needs a background
    (reference) sample and is considerably slower — it approximates
    Shapley values by perturbing features against that background.

    Parameters
    ----------
    model : fitted estimator
        E.g. `trained_B["CatBoost"]["best_estimator"]`.
    background : pd.DataFrame, optional
        Reference sample for the non-tree fallback. Required if `model`
        is not a tree ensemble. Ignored for tree models.
    background_size : int, default 100
        If `background` has more rows than this, it is subsampled —
        the generic Explainer's cost scales with background size.
    random_state : int, default 42

    Returns
    -------
    shap.Explainer or shap.TreeExplainer
    """
    if _is_tree_model(model):
        return shap.TreeExplainer(model)

    if background is None:
        raise ValueError(
            f"{type(model).__name__} is not a tree ensemble — SHAP needs a "
            "`background` sample (e.g. X_train_B_scaled) to explain it."
        )
    if len(background) > background_size:
        background = shap.sample(background, background_size, random_state=random_state)
    return shap.Explainer(model.predict, background)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 500,
    background_size: int = 100,
    random_state: int = 42,
):
    """
    Computes SHAP values for a random sample of `X`.

    Parameters
    ----------
    model : fitted estimator
        E.g. `trained_B["CatBoost"]["best_estimator"]`.
    X : pd.DataFrame
        Scaled feature set to explain (e.g. X_val_B_scaled). Sampled down
        to `sample_size` rows — SHAP cost grows with sample size, and a
        few hundred rows is enough to read the overall feature ranking.
    sample_size : int, default 500
        Rows to explain. If `X` has fewer rows, all of them are used.
    background_size : int, default 100
        Passed to `make_shap_explainer` for non-tree models. Ignored for
        tree ensembles.
    random_state : int, default 42

    Returns
    -------
    tuple[shap.Explainer, shap.Explanation, pd.DataFrame]
        (explainer, shap_values, X_sample) — X_sample is returned because
        downstream plotting functions need the exact rows that were
        explained, in the same order as shap_values.
    """
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    explainer = make_shap_explainer(
        model, background=X, background_size=background_size, random_state=random_state
    )
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, title: Optional[str] = None):
    """
    SHAP summary (beeswarm) plot: one row per feature, ranked by average
    impact on the prediction. Dot color encodes the feature's own value
    (red = high, blue = low), position on the x-axis encodes the SHAP
    value (impact on the prediction, in target units).

    Wraps `shap.summary_plot`, which draws directly onto the current
    matplotlib figure rather than returning one — `plt.gcf()` is
    returned here for consistency with the rest of `diagnostics.py`, but
    the plot itself is already rendered by the time this function returns.

    Parameters
    ----------
    shap_values : shap.Explanation
        Output of `compute_shap_values()`.
    X_sample : pd.DataFrame
        The exact rows explained — second element returned by
        `compute_shap_values()`.
    title : str, optional
        If given, set as the plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_sample, show=False)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, index: int = 0, title: Optional[str] = None):
    """
    SHAP waterfall plot for a single prediction: starts at the model's
    baseline (average prediction over the background/training data) and
    shows how each feature pushes that single prediction up or down to
    reach the final predicted value.

    Parameters
    ----------
    shap_values : shap.Explanation
        Output of `compute_shap_values()`.
    index : int, default 0
        Row (within the sample passed to `compute_shap_values()`) to
        explain.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    shap.plots.waterfall(shap_values[index], show=False)
    fig = plt.gcf()
    if title:
        fig.axes[-1].set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig
