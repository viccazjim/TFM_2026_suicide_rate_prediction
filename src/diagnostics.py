"""
Diagnostic plots.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA


def save_figure(
    fig, name: str, prefix: str = "", figures_dir: str = "../outputs/figures"
) -> str:
    """
    Saves a matplotlib figure as PNG under `figures_dir`, named
    "{prefix}{name}.png".

    The prefix is meant to encode which notebook/pipeline step produced
    the figure (e.g. "02_" for 02_eda.ipynb, "03_" for 03_models.ipynb),
    so outputs/figures/ stays organized by pipeline stage as more
    notebooks contribute plots, without every notebook needing to know
    about the others' naming.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Returned by any plotting function in diagnostics.py / explainability.py.
    name : str
        Descriptive, filesystem-safe name, without extension or prefix
        (e.g. "vif_before_drop").
    prefix : str, default ""
        Prepended directly to `name` — no separator is added
        automatically, so pass e.g. "02_" (with the trailing
        underscore) if you want "02_vif_before_drop.png".
    figures_dir : str, default "../outputs/figures"
        Path relative to the notebook's own working directory
        (typically notebooks/), so the default resolves to
        outputs/figures/ at the repo root. Created if it doesn't exist.

    Returns
    -------
    str
        Full path the figure was saved to.
    """
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{prefix}{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def suicide_evolution_graph(dataframe, country_code, country_name):
    """
    Generates and displays a line plot showing the chronological evolution
    of the suicide rate for a specific country.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The consolidated master dataframe containing 'Code', 'Year', and 'Suicide rate'.
    country_code : str
        The 3-letter ISO code of the target country (e.g., "GRC").
    country_name : str
        The common name of the country to display in the plot title (e.g., "Greece").

    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Filtering dataframe for a specific country code
    df_country = dataframe[dataframe["Code"] == country_code].sort_values("Year")

    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_country,
        x="Year",
        y="Suicide rate",
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2.5,
    )

    plt.title(
        f"Suicide rate evolution in {country_name}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate per 100,000 inhabitants", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    min_year, max_year = int(df_country["Year"].min()), int(df_country["Year"].max())
    plt.xticks(range(min_year, max_year + 1, 2))

    plt.tight_layout()
    return fig


def plot_suicide_trend_by_region(df: pd.DataFrame):
    """
    Line plot of average suicide rate over time, one line per EU region.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "Year", "Suicide rate", and "Region" (add it first
        via df["Region"] = df["Code"].map(EU_REGIONS) if not present).

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig = plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df,
        x="Year",
        y="Suicide rate",
        hue="Region",
        marker="o",
        linewidth=2.5,
        errorbar=None,
    )
    plt.title(
        "Suicide rate evolution in the EU by Economic-Geographic Regions (2000-2021)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Year")
    plt.ylabel("Average rate per 100,000 inhabitants")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="EU regions")
    plt.tight_layout()
    return fig


def plot_suicide_trend_by_group(df: pd.DataFrame, group_col: str, legend_title: str = None, title: str = None):
    """
    Line plot of average suicide rate over time, one line per group —
    generalizes plot_suicide_trend_by_region() to any grouping column,
    e.g. a K-Means cluster label. Built specifically so the a priori
    EU_REGIONS trend and the data-driven cluster trend can be produced
    with the same function and compared visually side by side, rather
    than eyeballing two differently-styled plots.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "Year", "Suicide rate", and `group_col`.
    group_col : str
        Column to group by — e.g. "Region" or "Cluster".
    legend_title : str, optional
        Defaults to `group_col`.
    title : str, optional
        Defaults to a generic title mentioning `group_col`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df,
        x="Year",
        y="Suicide rate",
        hue=group_col,
        marker="o",
        linewidth=2.5,
        errorbar=None,
        palette="tab10",
    )
    plt.title(
        title or f"Suicide rate evolution in the EU by {group_col} (2000-2021)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Year")
    plt.ylabel("Average rate per 100,000 inhabitants")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title=legend_title or group_col)
    plt.tight_layout()
    return fig


def plot_suicide_boxplot_by_country(df: pd.DataFrame):
    """
    Boxplot of suicide rate distribution per country, ordered by median
    descending (highest-median country first).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "Code" and "Suicide rate".

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig = plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df,
        x="Code",
        y="Suicide rate",
        hue="Code",
        legend=False,
        palette="vlag",
        order=df.groupby("Code")["Suicide rate"]
        .median()
        .sort_values(ascending=False)
        .index,
    )
    plt.title(
        "Suicide Rate Distribution and Dispersion across EU Nations (2000-2021)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Country ISO Code")
    plt.ylabel("Suicide Rate per 100,000 inhabitants")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_feature_distributions(df: pd.DataFrame, predictor_features: list[str]):
    """
    Grid of histograms + KDE, one subplot per predictor, each annotated
    with its skewness.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the columns in predictor_features.
    predictor_features : list[str]
        Columns to plot (e.g. output of build_predictor_list()). Each
        column's NaNs are dropped independently before plotting.

    Returns
    -------
    matplotlib.figure.Figure
        A 3-column grid, with unused subplot slots hidden if
        len(predictor_features) is not a multiple of 3.
    """

    ncols = 3
    nrows = (len(predictor_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 25))
    axes = axes.flatten()

    for i, col in enumerate(predictor_features):
        ax = axes[i]
        data_col = df[col].dropna()
        sns.histplot(x=data_col, kde=True, ax=ax, color="#4C72B0", bins=20)
        skew_val = data_col.skew()
        ax.set_title(
            f"{col}\n(skewness: {skew_val:.2f})", fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=8)

    for j in range(len(predictor_features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Feature Distributions — EU Development Dataset (2000-2021)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig


def plot_suicide_dispersion_stripplot(df: pd.DataFrame):
    """
    Stripplot (jittered scatter) of suicide rate per country, ordered by
    median descending — used to visually contextualise outliers flagged
    by flag_outliers_iqr() against the actual per-country spread.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "Code" and "Suicide rate".

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.stripplot(
        data=df,
        x="Code",
        y="Suicide rate",
        hue="Code",
        legend=False,
        order=df.groupby("Code")["Suicide rate"]
        .median()
        .sort_values(ascending=False)
        .index,
        jitter=True,
        alpha=0.6,
        palette="Set2",
        ax=ax,
    )
    ax.set_title(
        "Suicide Rate Dispersion by Country — Outlier Context",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Country ISO Code")
    ax.set_ylabel("Suicide Rate per 100,000")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_correlation_heatmaps(
    df_train: pd.DataFrame,
    social_economic_features: list[str],
    health_related_features: list[str],
):
    """
    Three correlation heatmaps in a mosaic layout: socioeconomic features,
    mental-health/resource features, and all numeric features combined.

    To be computed on a TRAINING split only.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split only (e.g. df_train_A or df_train_B).
    social_economic_features : list[str]
        Columns for the first heatmap (from src.config.SOCIAL_ECONOMIC_FEATURES).
    health_related_features : list[str]
        Columns for the second heatmap (from src.config.HEALTH_RELATED_FEATURES).

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, axs = plt.subplot_mosaic(
        [["top_left", "top_right"], ["bottom", "bottom"]], figsize=(25, 20)
    )
    sns.heatmap(
        df_train[social_economic_features].corr(),
        ax=axs["top_left"],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axs["top_left"].set_title(
        "Correlation: Socio-economic factors vs Suicide rate",
        fontsize=10,
        fontweight="bold",
    )

    sns.heatmap(
        df_train[health_related_features].corr(),
        ax=axs["top_right"],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axs["top_right"].set_title(
        "Correlation: Mental health and resources vs Suicide rate",
        fontsize=10,
        fontweight="bold",
    )

    corr_matrix = df_train.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr_matrix,
        ax=axs["bottom"],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axs["bottom"].set_title("Correlation: all features", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_vif_bar(vif_results: pd.DataFrame):
    """
    Horizontal bar chart of Variance Inflation Factor (VIF) per feature,
    color-coded by severity: green (VIF <= 5), orange (5 < VIF <= 10),
    red (VIF > 10).

    Parameters
    ----------
    vif_results : pd.DataFrame
        Must contain columns "Feature" and "VIF" — the output of
        compute_vif().

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#d62728" if v > 10 else "#ff7f0e" if v > 5 else "#2ca02c"
        for v in vif_results["VIF"]
    ]
    ax.barh(vif_results["Feature"], vif_results["VIF"], color=colors)
    ax.axvline(
        x=5, color="#ff7f0e", linestyle="--", linewidth=1.2, label="VIF=5 (moderate)"
    )
    ax.axvline(
        x=10, color="#d62728", linestyle="--", linewidth=1.2, label="VIF=10 (high)"
    )
    ax.set_xlabel("VIF Score")
    ax.set_title(
        "Multicollinearity Check — Variance Inflation Factor", fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    return fig


def plot_rmse_comparison(table_test: pd.DataFrame, table_val: pd.DataFrame, label: str):
    """
    Grouped bar chart comparing CV RMSE, Test RMSE and Val RMSE side by
    side for each model.

    Parameters
    ----------
    table_test : pd.DataFrame
        Test-set results, must contain columns "Model", "CV RMSE", "RMSE"
        (the output of build_results_table(..., split_filter="Test", ...)).
    table_val : pd.DataFrame
        Val-set results, must contain columns "Model", "RMSE" (the output
        of build_results_table(..., split_filter="Val", ...)). Must have
        the same models in the same row order as table_test.
    label : str
        Chart title (e.g. "Option B — Time split").

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(table_test))
    width = 0.25

    ax.bar(
        x - width,
        table_test["CV RMSE"],
        width,
        label="CV RMSE",
        color="#55A868",
        alpha=0.85,
    )
    ax.bar(x, table_test["RMSE"], width, label="Test RMSE", color="#4C72B0", alpha=0.85)
    ax.bar(
        x + width,
        table_val["RMSE"],
        width,
        label="Val RMSE",
        color="#DD8452",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(table_test["Model"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("RMSE (suicide rate per 100,000 inhabitants)")
    ax.set_title(label, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_r2_comparison(table_test: pd.DataFrame, table_val: pd.DataFrame, label: str):
    """
    Grouped bar chart comparing Test R² and Val R² side by side for each
    model.

    Parameters
    ----------
    table_test : pd.DataFrame
        Test-set results, must contain columns "Model", "R²".
    table_val : pd.DataFrame
        Val-set results, must contain columns "Model", "R²". Must have the
        same models in the same row order as table_test.
    label : str
        Chart title (e.g. "Option B — Time split").

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(table_test))
    width = 0.3

    ax.bar(
        x - width / 2,
        table_test["R²"],
        width,
        label="Test R²",
        color="#4C72B0",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        table_val["R²"],
        width,
        label="Val R²",
        color="#DD8452",
        alpha=0.85,
    )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(table_test["Model"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("R² score")
    ax.set_title(label, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Per-prediction result diagnostics (actual vs predicted, residuals, error breakdowns)
# --------------------------------------------------------------------------
def plot_actual_vs_predicted(actual, predicted, title="Actual vs Predicted"):
    """
    Scatter of predicted vs actual target values, with the y=x reference
    line — points on the line are perfect predictions, points below/above
    are under/over-estimates.

    Parameters
    ----------
    actual : array-like
        True target values (e.g. evaluate_model()[...]["actuals"]).
    predicted : array-like
        Model predictions (e.g. evaluate_model()[...]["predictions"]).
    title : str, default "Actual vs Predicted"

    Returns
    -------
    matplotlib.figure.Figure
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.6, s=50, edgecolor="white", linewidth=0.3)

    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1.5, label="y = x")

    ax.set_xlabel("Actual suicide rate")
    ax.set_ylabel("Predicted suicide rate")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residual_histogram(actual, predicted, title="Residual Distribution"):
    """
    Histogram of residuals (actual - predicted). A distribution centered
    on 0 with no strong skew indicates the model is not systematically
    over- or under-predicting.

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    title : str, default "Residual Distribution"

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = np.asarray(actual) - np.asarray(predicted)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=20, edgecolor="black", alpha=0.8, color="#4C72B0")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_residuals_vs_predicted(actual, predicted, title="Residuals vs Predicted"):
    """
    Scatter of residuals against predicted values. A random scatter around
    0 with no visible funnel/trend supports the constant-variance
    assumption; a funnel shape indicates heteroscedasticity (the model's
    error grows or shrinks with the predicted value).

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    title : str, default "Residuals vs Predicted"

    Returns
    -------
    matplotlib.figure.Figure
    """
    predicted = np.asarray(predicted)
    residuals = np.asarray(actual) - predicted

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predicted, residuals, alpha=0.6, s=50, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted suicide rate")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_error_by_year(
    df_years, actual, predicted, title="Mean Absolute Error by Year"
):
    """
    Line plot of mean absolute error per year. Useful for spotting whether
    error is concentrated in specific periods (e.g. a structural shift or
    an external shock) rather than spread evenly across the evaluation set.

    Parameters
    ----------
    df_years : pd.DataFrame or array-like
        Must align positionally with `actual`/`predicted` (same row order,
        same length) and contain a "Year" column — e.g.
        df_val_B[["Year"]].reset_index(drop=True).
    actual : array-like
    predicted : array-like
    title : str, default "Mean Absolute Error by Year"

    Returns
    -------
    matplotlib.figure.Figure
    """
    years = pd.DataFrame(df_years).reset_index(drop=True)["Year"].to_numpy()
    abs_error = np.abs(np.asarray(actual) - np.asarray(predicted))

    error_by_year = (
        pd.DataFrame({"Year": years, "Absolute Error": abs_error})
        .groupby("Year")["Absolute Error"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        error_by_year.index.to_numpy(),
        error_by_year.to_numpy(),
        marker="o",
        color="#4C72B0",
    )
    ax.set_xticks(error_by_year.index.astype(int))
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def mean_absolute_error_by_country(df_countries, actual, predicted, top_n=10):
    """
    Mean absolute error per country, sorted descending — the countries the
    model struggles with most appear first.

    Parameters
    ----------
    df_countries : pd.DataFrame or array-like
        Must align positionally with `actual`/`predicted` (same row order,
        same length) and contain a "Country" column — e.g.
        df_val_B[["Country"]].reset_index(drop=True).
    actual : array-like
    predicted : array-like
    top_n : int, default 10
        Number of highest-error countries to return.

    Returns
    -------
    pd.Series
        Index: country name. Values: mean absolute error, descending,
        truncated to top_n.
    """
    countries = pd.DataFrame(df_countries).reset_index(drop=True)["Country"].to_numpy()
    abs_error = np.abs(np.asarray(actual) - np.asarray(predicted))

    error_by_country = (
        pd.DataFrame({"Country": countries, "Absolute Error": abs_error})
        .groupby("Country")["Absolute Error"]
        .mean()
        .sort_values(ascending=False)
    )
    return error_by_country.head(top_n)


# --------------------------------------------------------------------------
# Visualizing predict.py output (df_real_world scoring)
# --------------------------------------------------------------------------
def plot_predictions_trend(df_history, df_predictions, country_code, country_name):
    """
    Plots a country's historical suicide rate (solid line) together
    with its predicted rate for years outside the training data
    (dashed line, connected to the last historical point) — a visual
    sanity check for whether predict.py's output looks like a
    plausible continuation of the country's own trend, rather than an
    implausible jump.

    Parameters
    ----------
    df_history : pd.DataFrame
        Historical data with "Code", "Year", "Suicide rate" — typically
        df_development (2000-2021, actual values).
    df_predictions : pd.DataFrame
        Output of predict.py, with "Code", "Year", "Predicted suicide rate".
    country_code : str
        3-letter ISO code (e.g. "GRC").
    country_name : str
        Display name for the title (e.g. "Greece").

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_hist = df_history[df_history["Code"] == country_code].sort_values("Year")
    df_pred = df_predictions[df_predictions["Code"] == country_code].sort_values("Year")

    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        df_hist["Year"], df_hist["Suicide rate"],
        color="#1f77b4", linestyle="-", marker="o", linewidth=2.5, label="Actual",
    )

    if not df_hist.empty and not df_pred.empty:
        # Prepend the last actual point so the predicted segment visually
        # connects to history instead of floating as a disjointed line.
        connector_years = [df_hist["Year"].iloc[-1]] + df_pred["Year"].tolist()
        connector_values = [df_hist["Suicide rate"].iloc[-1]] + df_pred["Predicted suicide rate"].tolist()
        plt.plot(
            connector_years, connector_values,
            color="#d62728", linestyle="--", marker="s", linewidth=2.5, label="Predicted",
        )

    plt.title(f"Suicide rate: actual vs predicted — {country_name}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate per 100,000 inhabitants", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_predictions_by_country(df_predictions, year, top_n=None):
    """
    Bar chart of predicted suicide rate per country for a given year,
    sorted descending.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Output of predict.py, with "Country", "Year", "Predicted suicide rate".
    year : int
        Which predicted year to plot (e.g. 2022).
    top_n : int, optional
        If given, only the top_n highest-predicted countries are shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_year = df_predictions[df_predictions["Year"] == year].sort_values(
        "Predicted suicide rate", ascending=False
    )
    if top_n:
        df_year = df_year.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_year["Country"], df_year["Predicted suicide rate"], color="#4C72B0")
    ax.set_title(f"Predicted suicide rate by country — {year}", fontweight="bold")
    ax.set_xlabel("Country")
    ax.set_ylabel("Predicted rate per 100,000 inhabitants")
    ax.tick_params(axis="x", rotation=75)
    plt.tight_layout()
    return fig


def plot_predictions_model_comparison(df_history, df_predictions_a, df_predictions_b, model_a_name, model_b_name, country_code, country_name):
    """
    Like plot_predictions_trend(), but overlays TWO predicted series
    instead of one — e.g. the production CatBoost model (which answers
    the thesis's actual question, using only socioeconomic/mental-health
    determinants) against the best temporal-persistence model from
    05_temporal_persistence_check.ipynb (which answers a different
    question — see that notebook's introduction). Plotting them
    together shows directly whether the two tell a similar story for a
    given country or diverge, rather than comparing single numbers in
    a table.

    Parameters
    ----------
    df_history : pd.DataFrame
        Historical data with "Code", "Year", "Suicide rate".
    df_predictions_a, df_predictions_b : pd.DataFrame
        Each with "Code", "Year", "Predicted suicide rate" — e.g.
        predict.py's output for A, a SARIMAX/Prophet forecast for B.
    model_a_name, model_b_name : str
        Labels for the legend (e.g. "CatBoost", "SARIMAX +1 exog").
    country_code : str
    country_name : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_hist = df_history[df_history["Code"] == country_code].sort_values("Year")
    df_pred_a = df_predictions_a[df_predictions_a["Code"] == country_code].sort_values("Year")
    df_pred_b = df_predictions_b[df_predictions_b["Code"] == country_code].sort_values("Year")

    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        df_hist["Year"], df_hist["Suicide rate"],
        color="#1f77b4", linestyle="-", marker="o", linewidth=2.5, label="Actual",
    )

    for df_pred, color, label in [
        (df_pred_a, "#d62728", model_a_name),
        (df_pred_b, "#2ca02c", model_b_name),
    ]:
        if df_hist.empty or df_pred.empty:
            continue
        connector_years = [df_hist["Year"].iloc[-1]] + df_pred["Year"].tolist()
        connector_values = [df_hist["Suicide rate"].iloc[-1]] + df_pred["Predicted suicide rate"].tolist()
        plt.plot(
            connector_years, connector_values,
            color=color, linestyle="--", marker="s", linewidth=2.5, label=label,
        )

    plt.title(f"Suicide rate: actual vs predicted — {country_name}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate per 100,000 inhabitants", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_predictions_by_country_comparison(df_predictions_a, df_predictions_b, model_a_name, model_b_name, year, top_n=None):
    """
    Grouped bar chart comparing two models' predicted suicide rate per
    country, for a given year — side-by-side bars per country instead
    of plot_predictions_by_country()'s single series.

    Parameters
    ----------
    df_predictions_a, df_predictions_b : pd.DataFrame
        Each with "Country", "Year", "Predicted suicide rate".
    model_a_name, model_b_name : str
        Labels for the legend.
    year : int
    top_n : int, optional
        If given, keeps only the top_n countries by model_a's prediction.

    Returns
    -------
    matplotlib.figure.Figure
    """
    a = df_predictions_a[df_predictions_a["Year"] == year][["Country", "Predicted suicide rate"]]
    b = df_predictions_b[df_predictions_b["Year"] == year][["Country", "Predicted suicide rate"]]
    merged = a.merge(b, on="Country", suffixes=(f"_{model_a_name}", f"_{model_b_name}"))
    merged = merged.sort_values(f"Predicted suicide rate_{model_a_name}", ascending=False)
    if top_n:
        merged = merged.head(top_n)

    x = np.arange(len(merged))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, merged[f"Predicted suicide rate_{model_a_name}"], width, label=model_a_name, color="#d62728")
    ax.bar(x + width / 2, merged[f"Predicted suicide rate_{model_b_name}"], width, label=model_b_name, color="#2ca02c")
    ax.set_title(f"Predicted suicide rate by country — {year} ({model_a_name} vs {model_b_name})", fontweight="bold")
    ax.set_xlabel("Country")
    ax.set_ylabel("Predicted rate per 100,000 inhabitants")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["Country"], rotation=75, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_predictions_by_region_comparison(df_predictions_a, df_predictions_b, model_a_name, model_b_name, year):
    """
    Grouped bar chart comparing two models' predicted suicide rate,
    averaged up to the a priori EU_REGIONS level, for a given year —
    the region-level counterpart to plot_predictions_by_country_comparison().
    Averaging countries into regions trades away the country-level
    detail in exchange for a view of whether each model's predictions
    line up with the a priori regional grouping used descriptively
    throughout this project (02_eda.py, 04_clustering.py).

    Parameters
    ----------
    df_predictions_a, df_predictions_b : pd.DataFrame
        Each with "Code", "Year", "Predicted suicide rate", and
        "Region" (add it first via
        df["Region"] = df["Code"].map(EU_REGIONS) if not present).
    model_a_name, model_b_name : str
        Labels for the legend.
    year : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    a = df_predictions_a[df_predictions_a["Year"] == year].groupby("Region")["Predicted suicide rate"].mean()
    b = df_predictions_b[df_predictions_b["Year"] == year].groupby("Region")["Predicted suicide rate"].mean()
    merged = pd.DataFrame({model_a_name: a, model_b_name: b}).dropna().sort_values(model_a_name, ascending=False)

    x = np.arange(len(merged))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, merged[model_a_name], width, label=model_a_name, color="#d62728")
    ax.bar(x + width / 2, merged[model_b_name], width, label=model_b_name, color="#2ca02c")
    ax.set_title(f"Predicted suicide rate by EU region — {year} ({model_a_name} vs {model_b_name})", fontweight="bold")
    ax.set_xlabel("EU region (a priori grouping)")
    ax.set_ylabel("Average predicted rate per 100,000 inhabitants")
    ax.set_xticks(x)
    ax.set_xticklabels(merged.index, rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_predictions_trend_by_region(df_history, df_predictions_a, df_predictions_b, model_a_name, model_b_name, region_name):
    """
    Region-level counterpart to plot_predictions_model_comparison():
    historical average suicide rate for every country in `region_name`
    (solid line), plus both models' region-average predicted rate for
    the predicted years (two dashed lines) — same visual language,
    aggregated up from country to region.

    Parameters
    ----------
    df_history : pd.DataFrame
        Historical data with "Region", "Year", "Suicide rate" (add
        "Region" first via df["Region"] = df["Code"].map(EU_REGIONS)
        if not present).
    df_predictions_a, df_predictions_b : pd.DataFrame
        Each with "Region", "Year", "Predicted suicide rate".
    model_a_name, model_b_name : str
    region_name : str
        e.g. "Baltics".

    Returns
    -------
    matplotlib.figure.Figure
    """
    hist = df_history[df_history["Region"] == region_name].groupby("Year")["Suicide rate"].mean().reset_index()
    pred_a = df_predictions_a[df_predictions_a["Region"] == region_name].groupby("Year")["Predicted suicide rate"].mean().reset_index()
    pred_b = df_predictions_b[df_predictions_b["Region"] == region_name].groupby("Year")["Predicted suicide rate"].mean().reset_index()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(hist["Year"], hist["Suicide rate"], color="#1f77b4", linestyle="-", marker="o", linewidth=2.5, label="Actual")

    for pred, color, label in [(pred_a, "#d62728", model_a_name), (pred_b, "#2ca02c", model_b_name)]:
        if hist.empty or pred.empty:
            continue
        connector_years = [hist["Year"].iloc[-1]] + pred["Year"].tolist()
        connector_values = [hist["Suicide rate"].iloc[-1]] + pred["Predicted suicide rate"].tolist()
        plt.plot(connector_years, connector_values, color=color, linestyle="--", marker="s", linewidth=2.5, label=label)

    plt.title(f"Suicide rate: actual vs predicted — {region_name} (region average)", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average rate per 100,000 inhabitants", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Clustering visualization (validating EU_REGIONS + general unsupervised analysis)
# --------------------------------------------------------------------------
def plot_kmeans_elbow_silhouette(sweep_df: pd.DataFrame):
    """
    Side-by-side elbow (inertia) and silhouette plots across the k
    values tested by sweep_kmeans() — the elbow suggests where adding
    another cluster stops meaningfully reducing within-cluster
    variance, the silhouette score directly measures cluster
    separation (higher is better).

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of clustering.sweep_kmeans() — columns "k", "inertia", "silhouette".

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(sweep_df["k"], sweep_df["inertia"], marker="o", color="#4C72B0")
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Inertia (within-cluster variance)")
    axes[0].set_title("Elbow method", fontweight="bold")
    axes[0].grid(linestyle="--", alpha=0.5)

    axes[1].plot(sweep_df["k"], sweep_df["silhouette"], marker="o", color="#DD8452")
    best_k = sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"]
    axes[1].axvline(best_k, color="gray", linestyle="--", alpha=0.7, label=f"Best: k={int(best_k)}")
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score", fontweight="bold")
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_dendrogram(linkage_matrix, labels, k: int = None, title: str = "Hierarchical clustering — dendrogram"):
    """
    Dendrogram for agglomerative hierarchical clustering, with country
    codes as leaf labels. Lets you see visually where a k-cluster cut
    would fall, rather than only getting the final labels.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        First element returned by clustering.run_hierarchical().
    labels : list[str]
        Leaf labels, one per row of the original data, same order —
        typically country ISO codes.
    k : int, optional
        If given, draws a horizontal line at the height that would cut
        the tree into k clusters, and colors branches accordingly.
    title : str, default "Hierarchical clustering — dendrogram"

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    color_threshold = None
    if k is not None and k > 1:
        heights = np.sort(linkage_matrix[:, 2])
        n_merges_to_keep = len(heights) - (k - 1)
        color_threshold = heights[n_merges_to_keep - 1] if n_merges_to_keep > 0 else 0

    dendrogram(linkage_matrix, labels=labels, ax=ax, color_threshold=color_threshold, leaf_font_size=10)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Country")
    ax.set_ylabel("Distance (Ward linkage)")
    plt.tight_layout()
    return fig


def plot_cluster_vs_region_pca(X_scaled: pd.DataFrame, cluster_labels, region_labels, country_codes):
    """
    Two side-by-side PCA scatter plots (2 components) of the same
    country-level feature space: one colored by cluster assignment, one
    colored by the a priori EU_REGIONS label — a direct visual check of
    whether the two partitions carve up the countries the same way.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        Scaled country-level features.
    cluster_labels : array-like
        One cluster id per row of X_scaled, same order.
    region_labels : array-like
        One EU_REGIONS value per row of X_scaled, same order.
    country_codes : array-like
        One ISO code per row, same order — used to annotate points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, hue, title in [
        (axes[0], cluster_labels, "Colored by cluster"),
        (axes[1], region_labels, "Colored by EU_REGIONS (a priori)"),
    ]:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=hue, palette="tab10", s=120, ax=ax, legend="full")
        for i, code in enumerate(country_codes):
            ax.annotate(code, (coords[i, 0], coords[i, 1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%} var.)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%} var.)")
        ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig
