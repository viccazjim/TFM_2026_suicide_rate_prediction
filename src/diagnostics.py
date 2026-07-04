"""
Diagnostic plots.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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
