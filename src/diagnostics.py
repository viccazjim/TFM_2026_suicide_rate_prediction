"""
Diagnostic plots.

"""

import matplotlib.pyplot as plt
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
    None
        Renders a matplotlib plot.
    """
    # Filtering dataframe for a specific country code
    df_country = dataframe[dataframe["Code"] == country_code].sort_values("Year")

    # Creating the figure and sizing it
    plt.figure(figsize=(12, 6))

    # Plotting the line with the values for the specified country
    plt.plot(
        df_country["Year"],
        df_country["Suicide rate"],
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2.5,
    )

    # Beauty personalization
    plt.title(
        f"Suicide rate evolution in {country_name}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate per 100,000 inhabitants", fontsize=12)

    # Showing every two years values in X axis
    min_year, max_year = int(df_country["Year"].min()), int(df_country["Year"].max())
    plt.xticks(range(min_year, max_year + 1, 2))

    # Adjusting margins and showing the graph
    plt.tight_layout()
    plt.show()


def plot_residuals_by_year(
    trained_dict, df_split, X_split_scaled, y_split, split_label
):
    fig, axes = plt.subplots(
        1, len(trained_dict), figsize=(6 * len(trained_dict), 5), sharey=True
    )
    if len(trained_dict) == 1:
        axes = [axes]

    for ax, (name, trained) in zip(axes, trained_dict.items()):
        model = trained["best_estimator"]
        y_pred = model.predict(X_split_scaled)
        residuals = y_split.values - y_pred

        resid_df = pd.DataFrame(
            {"Year": df_split["Year"].values, "Residual": residuals}
        )
        by_year = resid_df.groupby("Year")["Residual"]
        means = by_year.mean()
        stds = by_year.std()

        ax.errorbar(
            means.index,
            means.values,
            yerr=stds.values,
            fmt="o-",
            capsize=4,
            color="#4C72B0",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name} — Residuals by year ({split_label})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Residual (actual − predicted)")

    plt.tight_layout()
    return fig


def plot_vif_bar(vif_results: pd.DataFrame):
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
