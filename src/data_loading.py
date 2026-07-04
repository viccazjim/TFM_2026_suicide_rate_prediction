"""
Data loading and cleaning: IHME (local CSV), World Bank (API), WHO (API).

Origin (EDA_models_VC.ipynb):
- country_code            <- cell 4
- load_ihme_data          <- cell 5
- fetch_worldbank_indicators <- cell 7
- fetch_who_suicide_rates <- cell 9 (WHO part only)
- build_master_dataset    <- cell 9 (merge + development/real_world split)
- impute_missing_values   <- cell 14

No modelling decisions live here — this module only produces
df_development / df_real_world exactly as the original notebook did.
"""

import pandas as pd
import pycountry
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    (one column per cause_name), and drops the aggregate 'All causes' row.
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
    df_base = df_base[df_base["Year"] >= min_year].copy()
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
    (World Bank API does not support multi-indicator requests).
    """
    df_features = df_base.copy()

    for code, column_name in indicators.items():
        url_wb = (
            f"http://api.worldbank.org/v2/country/{region}/indicator/{code}"
            f"?format=json&per_page=10000&date={date_range}"
        )
        # SSL verification disabled for local network compatibility.
        # Remove verify=False if running in a production or public environment.
        response = requests.get(url_wb, verify=False)

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

        df_final_wb = df_wb[["Country", "Code", "Year", column_name]].dropna()
        df_final_eu = df_final_wb[df_final_wb["Code"].isin(eu_countries_iso)].copy()

        df_features = pd.merge(
            df_features,
            df_final_eu[["Code", "Year", column_name]],
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
    """
    url = f"https://ghoapi.azureedge.net/api/{indicator}"
    print("Downloading suicide rate dataset from the WHO API.")
    response = requests.get(url, verify=False)

    if response.status_code != 200:
        raise RuntimeError(
            f"WHO API request failed with status {response.status_code}."
        )

    data = response.json()["value"]
    if len(data) == 0:
        raise RuntimeError("WHO API responded but returned no data for this indicator.")

    df_raw = pd.DataFrame(data)
    df_filtered = df_raw[
        (df_raw["Dim1"] == "SEX_BTSX") & (df_raw["Dim2"] == "AGEGROUP_YEARSALL")
    ].copy()

    df_clean = df_filtered[["SpatialDim", "TimeDim", "NumericValue"]].copy()
    df_clean.columns = ["Code", "Year", "Suicide rate"]
    df_clean = df_clean.sort_values(by=["Code", "Year"])
    return df_clean


def build_master_dataset(
    df_features: pd.DataFrame, df_who: pd.DataFrame, development_cutoff_year: int = 2021
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges WHO suicide rates onto the feature set and splits into:
    - df_development: years <= development_cutoff_year, has suicide rate labels,
      used for modelling.
    - df_real_world: years > development_cutoff_year, kept for reference only —
      WHO has no suicide rate data for these years, so it is unlabeled and
      not used in modelling.
    """
    df_complete = pd.merge(
        df_features,
        df_who[["Code", "Year", "Suicide rate"]],
        on=["Code", "Year"],
        how="left",
    )
    df_complete = df_complete.sort_values(by=["Country", "Year"]).reset_index(drop=True)

    df_development = df_complete[df_complete["Year"] <= development_cutoff_year].copy()
    df_real_world = df_complete[df_complete["Year"] > development_cutoff_year].copy()
    return df_development, df_real_world


def impute_missing_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Linear interpolation within each country for the given columns. Chosen
    (over other imputation strategies) because the missing values occur in
    consecutive year runs within a country's series.
    """
    out = df.copy()
    for col in columns:
        out[col] = out.groupby("Country")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
    return out
