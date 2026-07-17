"""
Train / test / validation splitting strategies for the dataset.

"""

import numpy as np
import pandas as pd


def geographical_split(
    df: pd.DataFrame, eu_countries_iso: list[str], random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """
    Splits the dataset by country: 70% of countries go to train, 15% to
    test, 15% to validation. Every row for a given country goes to the
    same split.

    Used to test whether the model generalizes to countries it has
    never seen.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split, must contain "Code" (ISO alpha-3).
    eu_countries_iso : list[str]
        Full list of country codes to split.
    random_state : int, default 42
        Seed for the shuffle, for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]
        (df_train, df_test, df_val, train_countries, test_countries, val_countries):
        the three DataFrame splits, followed by the list of country codes
        assigned to each.
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
    return df_train, df_test, df_val, train_countries, test_countries, val_countries


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[int], list[int]]:
    """
    Splits the dataset by year, in chronological order (no shuffling):
    the earliest 70% of distinct years go to train, the next 15% to test,
    the final 15% to validation. All countries appear in every split.

    Used to test whether the model generalizes to future years.

    The exact year boundaries are not hardcoded, and the function returns the
    actual year lists used so callers can log/display them rather than
    assuming a fixed range (e.g. do not assume test is always "2016-2018").

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split, must contain "Year".

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[int], list[int]]
        (df_train, df_test, df_val, train_years, test_years, val_years):
        the three DataFrame splits, followed by the sorted list of years
        assigned to each.
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
