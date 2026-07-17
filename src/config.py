"""
Shared constants. Centralized here so every notebook/module references the
same lists instead of redefining them.
"""

# EU countries list since they are the ones of interest in this analysis
EU_COUNTRIES_ISO = [
    "DEU",
    "AUT",
    "BEL",
    "BGR",
    "CYP",
    "HRV",
    "DNK",
    "SVK",
    "SVN",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "CZE",
    "ROU",
    "SWE",
]

# Mapping dictionary based on the ISO alpha-3 codes for the EU
EU_REGIONS = {
    "DEU": "Western Europe/Nordics",
    "FRA": "Western Europe/Nordics",
    "NLD": "Western Europe/Nordics",
    "BEL": "Western Europe/Nordics",
    "LUX": "Western Europe/Nordics",
    "AUT": "Western Europe/Nordics",
    "DNK": "Western Europe/Nordics",
    "FIN": "Western Europe/Nordics",
    "SWE": "Western Europe/Nordics",
    "IRL": "Western Europe/Nordics",
    "ESP": "Mediterranean",
    "ITA": "Mediterranean",
    "PRT": "Mediterranean",
    "GRC": "Mediterranean",
    "CYP": "Mediterranean",
    "MLT": "Mediterranean",
    "POL": "Eastern Europe",
    "CZE": "Eastern Europe",
    "SVK": "Eastern Europe",
    "HUN": "Eastern Europe",
    "ROU": "Eastern Europe",
    "BGR": "Eastern Europe",
    "SVN": "Eastern Europe",
    "HRV": "Eastern Europe",
    "EST": "Baltics",
    "LVA": "Baltics",
    "LTU": "Baltics",
}

SOCIAL_ECONOMIC_FEATURES = [
    "Suicide rate",  # Target
    "GDP per capita",
    "Unemployment rate (%)",
    "Health expenditure (% GDP)",
    "Population",
    "Urban population (%)",
    "Physicians per 100000",
    "Gini index",
    "Internet users (% of population)",
]

HEALTH_RELATED_FEATURES = [
    "Suicide rate",  # Target
    "Alcohol use disorders",
    "Alzheimer's disease and other dementias",
    "Anxiety disorders",
    "Attention-deficit/hyperactivity disorder",
    "Autism spectrum disorders",
    "Bipolar disorder",
    "Conduct disorder",
    "Depressive disorders",
    "Schizophrenia",
    "Drug use disorders",
    # Eating disorders excluded — high VIF (12.098), dropped in the multicollinearity step. Kept out of the list once the decision was made in the EDA notebook.
]

ID_COLS = ["Country", "Code", "Year", "Region"]
TARGET = "Suicide rate"

# World Bank indicator codes: readable column names
WORLD_BANK_INDICATORS = {
    "NY.GDP.PCAP.CD": "GDP per capita",
    "SL.UEM.TOTL.ZS": "Unemployment rate (%)",
    "SH.XPD.CHEX.GD.ZS": "Health expenditure (% GDP)",
    "SP.POP.TOTL": "Population",
    "SI.POV.GINI": "Gini index",
    "SP.URB.TOTL.IN.ZS": "Urban population (%)",
    "SH.MED.PHYS.ZS": "Physicians per 1000",
    "IT.NET.USER.ZS": "Internet users (% of population)",
}

WHO_SUICIDE_INDICATOR = "SDGSUICIDE"
