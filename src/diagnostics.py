"""
Diagnostic plots.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


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
