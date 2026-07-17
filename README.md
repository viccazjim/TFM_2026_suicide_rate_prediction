# TFM 2026 — Predictive Analysis of Suicide Rates in the EU

Master's thesis (TFM, MSc Data Science and Artificial Intelligence) project: predicting suicide rates across EU countries from socioeconomic determinants and mental health prevalence. The project exists in two parallel, functionally equivalent forms — exploratory notebooks (`notebooks/`) and a modular production pipeline (`prod/` + `src/`).

## Project structure

```
.
├── data/
│   ├── raw/                    # IHME source CSV (external input, not generated)
│   └── processed/               # df_development.parquet, df_real_world.parquet
├── docs/
│   ├── orientaciones_y_pautas.pdf   # Program guidelines for the TFM
│   └── predictive_analysis_of_suicide_rates_2026_CV_CM_CL.docx   # Thesis write-up (current draft)
├── notebooks/                   # Exploratory track — mirrors prod/ 1:1
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_models.ipynb                       # The 6 panel models — answers the thesis's actual question
│   ├── 03b_model_improvements.ipynb          # Follow-up hypotheses on 03's SHAP findings
│   ├── 04_clustering.ipynb                   # Descriptive validation of EU_REGIONS (not fed into 03)
│   ├── 05_temporal_persistence_check.ipynb   # A second, different question — see below
│   ├── 05b_temporal_persistence_improvements.ipynb  # Follow-up hypotheses on 05's SARIMAX/Prophet results
│   └── 06_visualize_predictions.ipynb        # CatBoost vs best temporal model, country + region level
│   └── 07_export_powerbi.ipynb               # Assembles the Power BI dashboard's data workbook
├── outputs/
│   ├── figures/                 # All saved plots, prefixed by pipeline stage (02_/03_/04_/06_)
│   ├── models/                  # Persisted production model, scaler, and SARIMAX+exog models (joblib)
│   ├── tables/                  # Result tables and predictions (Parquet)
│   └── powerbi/                 # TFM_PowerBI_data.xlsx — the dashboard's data workbook
├── prod/                        # Production track — modular scripts, no notebooks
│   ├── 01_data_pipeline.py      # Ingestion + cleaning
│   ├── 02_eda.py                # EDA (figures + VIF), cleans df_development and df_real_world
│   ├── 03_train.py              # Trains + evaluates the 6 panel models, persists the production model
│   ├── 04_clustering.py         # Descriptive clustering validation (standalone, not a model feature)
│   ├── 05_temporal_persistence_check.py     # SARIMAX/Prophet vs a naive persistence baseline
│   ├── 06_visualize_predictions.py          # Predictions comparison, country + EU-region level
│   ├── 07_export_powerbi.py                 # Assembles the Power BI dashboard's data workbook (outputs/powerbi/)
│   ├── predict.py               # Inference — scores new data with the persisted CatBoost model
│   ├── run_pipeline.py          # Orchestrates 01 → 02 → 03 → 04 → 05 → predict → 06 → 07
│   └── clean_outputs.py         # Deletes every pipeline-generated artifact if needed (data/processed, outputs/, catboost_info/)
├── src/                          # Shared logic — imported by both notebooks/ and prod/
│   ├── config.py                 # Constants (country lists, feature lists, indicator codes)
│   ├── data_loading.py           # IHME / World Bank / WHO fetch + cleaning + imputation
│   ├── features.py               # VIF, predictor list builder, IQR outlier flagging
│   ├── splits.py                 # geographical_split(), temporal_split()
│   ├── models.py                 # Panel model registry, hyperparameter grids, train/evaluate
│   ├── clustering.py              # Descriptive clustering (K-Means, hierarchical, ARI/NMI, PCA coords)
│   ├── timeseries_models.py       # Per-country SARIMAX / Prophet fit, forecast, evaluate
│   ├── metrics.py                 # Result tables, get_eval_entry(), metrics_by_period()
│   ├── diagnostics.py             # All plotting functions (EDA, results, predictions) + save_figure()
│   ├── explainability.py          # SHAP-based model interpretation
│   ├── persistence.py             # save_artifact() / load_artifact() (joblib)
│   └── export.py                  # Builds the Power BI dashboard's tables + writes the .xlsx

├── requirements.txt
└── TODO.md
```

Notebooks and `prod/` scripts are both thin orchestration layers over `src/` — the actual logic (fetching, feature engineering, splitting, modelling, plotting, persistence) lives in `src/` and is imported, not redefined, in either track. Every `.py` script in `prod/` has a directly corresponding notebook with the same numbering and the same underlying `src/` calls, so results are reproducible in either environment, and a fix made in `src/` applies to both without touching either.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a full environment freeze, not a curated minimal list. The packages the pipeline actually imports: `numpy`, `pandas`, `pyarrow` (Parquet I/O), `scikit-learn`, `xgboost`, `catboost`, `statsmodels` (SARIMAX), `prophet`, `shap`, `joblib`, `scipy`, `pycountry`, `requests`, `matplotlib`, `seaborn`.

## Data sources

| Source | What it provides | Access |
|---|---|---|
| [IHME Global Burden of Disease](https://www.healthdata.org/) | Mental health / neurological disorder prevalence, per country/year/cause | Local CSV export (`data/raw/`) |
| World Bank API | GDP per capita, unemployment, health expenditure, population, Gini index, urban population %, physicians per 1,000, internet users % | Live API call, no key required |
| WHO Global Health Observatory API | Suicide rate per 100,000 inhabitants (target variable) | Live API call, no key required |

Scope: 27 EU member states, years 2000–2021 (`df_development`, labeled) plus 2022–2023 (`df_real_world`, unlabeled — WHO has not published suicide rates for these years yet). `data/processed/` is stored as Parquet rather than CSV: columnar storage, and exact dtype preservation across save/load, a habit worth having even though this dataset's size doesn't strictly require it yet.

Both API-fetching functions (`fetch_worldbank_indicators`, `fetch_who_suicide_rates` in `src/data_loading.py`) use a `requests.Session` with automatic retry-with-backoff on connection drops, timeouts, and 5xx errors (`REQUEST_TIMEOUT = 30`), and call the World Bank API over `https://` directly rather than `http://` — the plain-http endpoint redirects, and that extra hop is where a flaky connection is most likely to fail before a retry gets a chance to run.

## Running the pipeline

### Option 1 — Notebooks (exploratory)

Run `01` → `02` → `03` in order, each reading what the previous one wrote. `03b`, `04`, `05`, `05b`, `06` can run in any order after `03` (none of them feed back into it) — `06` additionally needs `predict.py`'s output, so run that first if working outside `run_pipeline.py`. `07` needs both `predict.py`'s and `06`'s saved predictions, so run it last.

### Option 2 — Production scripts

```bash
python prod/run_pipeline.py                # full pipeline: 01 → 02 → 03 → 04 → 05 → predict → 06 → 07
python prod/run_pipeline.py --skip-01       # reuse existing data/processed/
python prod/run_pipeline.py --only 03       # training only

python prod/predict.py                      # score df_real_world.parquet with CatBoost
python prod/predict.py --input custom.parquet --output custom_predictions.csv

python prod/07_export_powerbi.py            # rebuild outputs/powerbi/TFM_PowerBI_data.xlsx alone

python prod/clean_outputs.py                # wipe data/processed/, outputs/, catboost_info/ before a clean re-run
python prod/clean_outputs.py --dry-run      # preview what would be deleted, without deleting
```

`02_eda.py`, `03_train.py`, `04_clustering.py`, and `06_visualize_predictions.py` force matplotlib's `Agg` backend internally, since they only save figures to disk and never display them.

## Two questions, kept deliberately separate

This project answers two related but different questions, in two different places, on purpose — conflating them was tried during development and reverted (see `03_train.py`'s and `05_temporal_persistence_check.py`'s module docstrings for the full reasoning):

1. **Can socioeconomic and mental-health determinants predict suicide rate?** — `03_models.ipynb` / `03_train.py`. The 6 panel models (Ridge, Lasso, SVR, Random Forest, XGBoost, CatBoost) use only those determinants as predictors — nothing derived from a country's own suicide-rate history. This is the thesis's actual hypothesis, and the only place it's tested.
2. **How much of suicide rate is just its own year-to-year persistence, independent of any determinant?** — `05_temporal_persistence_check.ipynb` / `05_temporal_persistence_check.py`. SARIMAX and Prophet, fit one model per country, forecast each country from its own history alone (or with one deliberately curated exogenous determinant). A naive "repeat last year's value" baseline is included explicitly, because it turned out to score close to — sometimes better than — the univariate time-series models, which would have been a misleading result to present without that comparison point.

`04_clustering.ipynb` / `04_clustering.py` is a third, independent, descriptive analysis — whether the a priori `EU_REGIONS` grouping used throughout the EDA holds up empirically — and does not feed into either of the two questions above.

## Results summary

From the current result tables in `outputs/tables/`:

### 1. Determinants (Option A vs Option B, 6 panel models)

- **Option A (geographical split):** all six models reach a positive Test R² (0.08–0.44), but five of six drop to a **negative** R² on Validation (as low as −0.62 for XGBoost) — the positive Test score does not transfer to genuinely unseen countries. Only SVR holds up (0.44 → 0.13).
- **Option B (time split):** stronger and more stable for the top four models — Test R² 0.33–0.91, Validation R² −0.08–0.65. CatBoost (Test 0.88, Val 0.64) and SVR (Test 0.91, Val 0.62) are the strongest and most consistent performers, and CatBoost is the persisted production model (`outputs/models/catboost_option_b.joblib`, depth=5, iterations=400, l2_leaf_reg=7, learning_rate=0.1).

SHAP on the production CatBoost model shows `Alcohol use disorders` as by far the most influential predictor (mean |SHAP| ≈ 2.9, roughly 3× the next-ranked feature). Validation-set error is not flat across years: it rises from ≈1.8 (2018) to ≈2.6 (2021), a nearly linear progression with no sharp break at the 2020 boundary — consistent with growing distance from the training period (2000–2014) rather than a COVID-specific shock alone.

### 2. Temporal persistence (Option B only)

| Model | Variant | Test R² | Val R² |
|---|---|---|---|
| Naive persistence | no model | — | 0.62 |
| SARIMAX | univariate | 0.91 | 0.60 |
| SARIMAX | +1 exog (`Alcohol use disorders`) | **0.94** | **0.77** |
| Prophet | univariate | 0.90 | 0.72 |
| Prophet | +1 exog (`Alcohol use disorders`) | 0.91 | 0.69 |

Suicide rate's pooled year-over-year autocorrelation is ≈0.99 — a model with access to a country's own recent value looks strong almost for free, which is exactly why the naive baseline is reported alongside the "real" models rather than left out. Univariate SARIMAX does not even beat the naive baseline; SARIMAX + the single curated exogenous feature is the only result that constitutes real evidence that a determinant adds explanatory power on top of pure persistence, and it is not used as the production model regardless (see below).

### 3. Clustering (descriptive, not a model feature)

K-Means and hierarchical clustering agree with the a priori `EU_REGIONS` grouping at ARI ≈ 0.573, NMI ≈ 0.721 — moderate-to-good, not perfect. Baltics and Eastern Europe are recovered perfectly; all the disagreement concentrates at the Mediterranean / Western-Europe-Nordics boundary. The data's best-separated partition (by silhouette score) is k=2, not k=4, and tracks economic/population scale (Germany, France, Spain, Italy vs. the rest) rather than geography.

## Why CatBoost, not the best-scoring temporal model, is in production

SARIMAX + exog scores higher than CatBoost on both Test and Validation. It is not the model behind `predict.py` anyway: it forecasts a *known* country forward from its own history, and cannot score a country it has no training history for, or an arbitrary new row of predictor values. CatBoost answers the question `predict.py` is actually asked — "what would this row's suicide rate be, given these determinants" — which SARIMAX/Prophet structurally cannot. `06_visualize_predictions.py` still fits SARIMAX + exog inline for comparison purposes and persists it to `outputs/models/sarimax_exog_models.joblib`, but only as a secondary reference point, not as an alternative production path.

## Cluster-as-feature was tried and reverted

An earlier version of the pipeline fed the country cluster from `04_clustering.py` back into `03_train.py` as a supervised-model feature (built in a leakage-safe way — no target information, fit on training data only). It was reverted after discussion: even done safely, it blurs together two results that are cleaner read separately (see "Two questions" above). The leakage-safe implementation was removed from `src/clustering.py` in a later cleanup pass rather than left in place unused — see git history if a future iteration wants to revisit the decision.

## Power BI dashboard

`prod/07_export_powerbi.py` (and its notebook counterpart) assembles `outputs/powerbi/TFM_PowerBI_data.xlsx` — seven flat, denormalised tables (country panel, predictions, SHAP importance, model comparison, cluster/PCA lookup, temporal persistence check, cluster/region agreement) meant to be connected directly via Power BI's "Get Data > Excel", one visual per table with no further transformation needed. `src/export.py` builds every table purely from artifacts the rest of the pipeline already produces — no new model is fit and no new result is computed there; see that module's docstring. `Code` is the join key between `Suicide_Rate_Panel`, `Predictions_2022_2023`, and `Cluster_PCA`.
