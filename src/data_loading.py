"""
Data loading and cleaning: IHME (local CSV), World Bank (API), WHO (API).

This module produces df_development / df_real_world.
"""

import numpy as np
import pandas as pd
import pycountry
import requests
import urllib3
from typing import cast
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REQUEST_TIMEOUT = 30  # seconds — a hung connection with no timeout blocks forever


def _build_session() -> requests.Session:
    """
    A requests.Session that retries transient failures — connection drops
    (e.g. RemoteDisconnected), timeouts, and 5xx server errors — with
    exponential backoff, instead of giving up after a single attempt.
    World Bank's and WHO's public APIs are usable but not perfectly
    reliable moment to moment, particularly from behind a corporate
    network or proxy; a bare `requests.get()` with no retry turns any
    one-off blip into a full pipeline failure. A custom User-Agent is set
    too, since some hosts are pickier about the default one `requests`
    sends.

    Returns
    -------
    requests.Session
    """
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2,  # waits 2s, 4s, 8s, 16s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; TFM-suicide-rate-pipeline/1.0)"}
    )
    return session


def country_code(geo_name: str) -> str | None:
    """
    Looks up the ISO-Alpha 3 country code based on a given country text name.

    Parameters:
    -----------
    geo_name : str
        The textual name of the country (e.g., "Austria", "Greece").

    Returns:
    --------
    str or None
        The 3-letter ISO code if found (e.g., "AUT"), otherwise None.
    """
    try:
        country = pycountry.countries.lookup(geo_name)
        return country.alpha_3
    except LookupError:
        return None


def load_ihme_data(csv_path: str, min_year: int = 2000) -> pd.DataFrame:
    """
    Loads the IHME mental-health-prevalence base dataset (local CSV,
    already filtered to EU countries at extraction time), pivots it wide
    (one column per cause_name), and drops the 'All causes' row.

    Parameters
    ----------
    csv_path : str
        Path to the IHME export CSV.
    min_year : int, default 2000
        Rows with "year" below this value are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: "Country", "Code", "Year", plus one column per distinct
        cause_name in the source file (e.g. "Depressive disorders",
        "Anxiety disorders", ...), each holding the "val" for that cause.
        "Code" is derived by looking up "location_name" via country_code().
    """
    base_data = pd.read_csv(csv_path)
    base_data["Code"] = base_data["location_name"].apply(country_code)

    base_data_short = base_data[["location_name", "Code", "year", "cause_name", "val"]]
    base_data_pivot = base_data_short.pivot(
        index=["location_name", "Code", "year"], columns="cause_name", values="val"
    )
    base_data_pivot.columns.name = None
    base_data_pivot = base_data_pivot.reset_index()

    df_base = base_data_pivot.rename(
        columns={"location_name": "Country", "year": "Year"}
    )
    df_base = df_base.drop(columns=["All causes"])
    df_base = df_base.loc[df_base["Year"] >= min_year].copy()
    return df_base


def fetch_worldbank_indicators(
    df_base: pd.DataFrame,
    eu_countries_iso: list[str],
    indicators: dict[str, str],
    region: str = "ALL",
    date_range: str = "2000:2026",
) -> pd.DataFrame:
    """
    Merges World Bank API indicators onto df_base, one indicator at a time
    (the World Bank API does not support multi-indicator requests).

    Failure handling: each indicator's request retries transient failures
    (connection drops, timeouts, 5xx errors) automatically — see
    _build_session(). If a single indicator's request still fails after
    retries, or returns a non-200 status, or returns no data, that
    indicator is skipped with a printed warning — the function does not
    raise, and continues with the remaining indicators. This means the
    returned DataFrame can silently be missing a column if that
    indicator's API call failed even after retrying; check the printed
    output or the resulting columns if you need to be sure all indicators
    came through.

    Parameters
    ----------
    df_base : pd.DataFrame
        Dataset to merge onto.
        Must contain "Code" and "Year".
    eu_countries_iso : list[str]
        ISO alpha-3 codes to keep from each indicator's response — the
        World Bank API returns all countries/aggregates, this filters to
        the EU set before merging.
    indicators : dict[str, str]
        Mapping of {World Bank indicator code: output column name}, e.g.
        {"NY.GDP.PCAP.CD": "GDP per capita"}.
    region : str, default "ALL"
        World Bank API region/country filter in the URL path.
    date_range : str, default "2000:2026"
        World Bank API date range filter, "start:end" format.

    Returns
    -------
    pd.DataFrame
        df_base with one additional column per successfully-fetched
        indicator, left-joined on ["Code", "Year"]. Features rate units
        adapted to the ones used elsewhere (indicator per 100,000 inhabitants - WHO/IHME).
    """
    df_features = df_base.copy()
    session = _build_session()

    for code, column_name in indicators.items():
        url_wb = (
            f"https://api.worldbank.org/v2/country/{region}/indicator/{code}"
            f"?format=json&per_page=10000&date={date_range}"
        )
        # SSL verification disabled for local network compatibility.
        # Remove verify=False if running in a production or public environment.
        try:
            response = session.get(url_wb, verify=False, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"World Bank request failed for '{column_name}' after retries: {e}")
            continue

        if response.status_code != 200:
            print(
                f"World Bank request failed for '{column_name}' (status {response.status_code})."
            )
            continue

        raw_data = response.json()[1]  # [0] is paging metadata, [1] is the actual data
        if not raw_data:
            print(f"World Bank returned no data for '{column_name}'.")
            continue

        df_wb = pd.DataFrame(raw_data)
        df_wb["Country"] = df_wb["country"].apply(lambda x: x["value"])
        df_wb["Code"] = df_wb["countryiso3code"].str.upper()
        df_wb["Year"] = df_wb["date"].astype(int)
        df_wb[column_name] = df_wb["value"]

        df_final_wb = cast(
            pd.DataFrame, df_wb[["Country", "Code", "Year", column_name]]
        ).dropna()
        df_final_eu = df_final_wb.loc[df_final_wb["Code"].isin(eu_countries_iso)].copy()

        df_features = pd.merge(
            df_features,
            cast(pd.DataFrame, df_final_eu[["Code", "Year", column_name]]),
            on=["Code", "Year"],
            how="left",
        )
        print(
            f"World Bank indicator '{column_name}' merged. Rows retrieved: {len(df_final_eu)}."
        )

    # Physicians per 1000 -> per 100000, matching WHO/IHME rate units
    if "Physicians per 1000" in df_features.columns:
        df_features["Physicians per 100000"] = df_features["Physicians per 1000"] * 100
        df_features = df_features.drop(columns=["Physicians per 1000"])

    return df_features


def fetch_who_suicide_rates(indicator: str = "SDGSUICIDE") -> pd.DataFrame:
    """
    Fetches suicide mortality rate (both sexes, all age groups) from the
    WHO Global Health Observatory API.

    Raises on failure rather than skipping silently since it is the target variable.

    Parameters
    ----------
    indicator : str, default "SDGSUICIDE"
        WHO GHO OData indicator code.

    Returns
    -------
    pd.DataFrame
        Columns "Code", "Year", "Suicide rate", filtered to Dim1 ==
        "SEX_BTSX" (both sexes) and Dim2 == "AGEGROUP_YEARSALL" (all age
        groups), sorted by ["Code", "Year"].

    Raises
    ------
    RuntimeError
        If the HTTP request does not return status 200 (even after
        retrying transient failures — see _build_session()), or if it
        returns 200 but with an empty "value" list.
    """
    url = f"https://ghoapi.azureedge.net/api/{indicator}"
    print("Downloading suicide rate dataset from the WHO API.")
    session = _build_session()
    try:
        response = session.get(url, verify=False, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"WHO API request failed after retries: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(
            f"WHO API request failed with status {response.status_code}."
        )

    data = response.json()["value"]
    if len(data) == 0:
        raise RuntimeError("WHO API responded but returned no data for this indicator.")

    df_raw = pd.DataFrame(data)
    df_filtered = df_raw.loc[
        (df_raw["Dim1"] == "SEX_BTSX") & (df_raw["Dim2"] == "AGEGROUP_YEARSALL")
    ].copy()

    df_clean = cast(
        pd.DataFrame, df_filtered[["SpatialDim", "TimeDim", "NumericValue"]]
    ).copy()
    df_clean.columns = ["Code", "Year", "Suicide rate"]
    df_clean = df_clean.sort_values(by=["Code", "Year"])
    return df_clean


def build_master_dataset(
    df_features: pd.DataFrame, df_who: pd.DataFrame, development_cutoff_year: int = 2021
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges WHO suicide rates onto the feature set and splits the result
    into a labeled modelling set and an unlabeled reference set by year.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of fetch_worldbank_indicators() (or equivalent), must
        contain "Country", "Code", "Year".
    df_who : pd.DataFrame
        Output of fetch_who_suicide_rates(), must contain "Code", "Year",
        "Suicide rate".
    development_cutoff_year : int, default 2021
        Last year considered "labeled". Chosen because WHO suicide rate
        data is not yet available after this year at the time of writing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_development, df_real_world):
        - df_development: rows with Year <= development_cutoff_year, has
          "Suicide rate" populated, used for modelling.
        - df_real_world: rows with Year > development_cutoff_year, kept
          for reference only — "Suicide rate" will be NaN for these rows
          since WHO has no data yet, so this set is not usable for
          training or evaluation as-is.
        Both are sorted by ["Country", "Year"].
    """
    df_complete = pd.merge(
        df_features,
        df_who[["Code", "Year", "Suicide rate"]],
        on=["Code", "Year"],
        how="left",
    )
    df_complete = df_complete.sort_values(by=["Country", "Year"]).reset_index(drop=True)

    df_development = df_complete.loc[
        df_complete["Year"] <= development_cutoff_year
    ].copy()
    df_real_world = df_complete.loc[
        df_complete["Year"] > development_cutoff_year
    ].copy()
    return df_development, df_real_world


def interpolate_with_trend_extrapolation(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """
    Interior gaps: standard linear interpolation (unchanged).
    Boundary gaps: linear extrapolation using an OLS fit on the country's
    own available years, instead of flat-filling with the nearest known
    value. Requires at least 2 known points per country to fit a trend;
    falls back to flat-fill if a country has only 1 known point (a trend
    can't be estimated from a single point).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing "Country" and the columns to impute.
    columns : list[str]
        Column names to interpolate. Columns not in this list are
        returned unchanged.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the given columns interpolated per country.
    """
    out = df.copy()
    for col in columns:

        def _fill_group(group):
            """
            Interpolates one country's series for `col` (closed over
            from the enclosing loop): interior gaps via linear
            interpolation, boundary gaps via linear extrapolation from
            an OLS trend fit on the country's known points, falling
            back to flat-fill if fewer than 2 points are known.
            """
            s = group[col]
            s_interp = s.interpolate(
                method="linear", limit_direction=None
            )  # interior only
            if s_interp.isna().sum() == 0:
                return s_interp
            known = s_interp.dropna()
            if len(known) < 2:
                return s_interp.interpolate(
                    method="linear", limit_direction="both"
                )  # fallback: flat-fill
            coeffs = np.polyfit(
                group.loc[known.index, "Year"],
                known.values,
                deg=1,
            )
            trend = np.poly1d(coeffs)
            missing_idx = s_interp[s_interp.isna()].index
            s_interp.loc[missing_idx] = trend(group.loc[missing_idx, "Year"])
            return s_interp

        out[col] = out.groupby("Country", group_keys=False).apply(_fill_group)
    return out
