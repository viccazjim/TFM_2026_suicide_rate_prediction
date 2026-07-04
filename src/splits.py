"""
Train / test / validation splitting strategies.

"""

import numpy as np
import pandas as pd


def geographical_split(
    df: pd.DataFrame, eu_countries_iso: list[str], random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Geographical split: countries are divided 70/15/15 into train/test/val.
    Tests whether the model generalizes to *unseen countries*.
    """
    rng = np.random.RandomState(random_state)
    countries = eu_countries_iso.copy()
    rng.shuffle(countries)

    total = len(countries)
    train_split = int(total * 0.70)
    test_split = int(total * 0.85)

    train_countries = countries[:train_split]
    test_countries = countries[train_split:test_split]
    val_countries = countries[test_split:]

    df_train = df[df["Code"].isin(train_countries)].copy()
    df_test = df[df["Code"].isin(test_countries)].copy()
    df_val = df[df["Code"].isin(val_countries)].copy()
    return df_train, df_test, df_val


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[int], list[int]]:
    """
    Temporal split: years divided 70/15/15 into train/test/val (in
    chronological order, no shuffling). Tests whether the model generalizes
    to *future years*.

    Returns the three DataFrames plus the year lists actually used, so
    downstream notebooks can print/log them instead of assuming fixed
    year ranges.
    """
    year_range = sorted(df["Year"].unique())
    total_years = len(year_range)
    train_split = int(total_years * 0.70)
    test_split = int(total_years * 0.85)

    train_years = year_range[:train_split]
    test_years = year_range[train_split:test_split]
    val_years = year_range[test_split:]

    df_train = df[df["Year"].isin(train_years)].copy()
    df_test = df[df["Year"].isin(test_years)].copy()
    df_val = df[df["Year"].isin(val_years)].copy()
    return df_train, df_test, df_val, train_years, test_years, val_years
