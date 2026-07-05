# TFM 2026 — Predictive Analysis of Suicide Rates in the EU

Master's thesis (TFM, MDS) project: predicting suicide rates across EU countries from socioeconomic determinants and mental health prevalence. The project exists in two parallel forms — exploratory notebooks (`notebooks/`) and a modular production pipeline (`prod/` + `src/`).

## Project structure

```
.
├── data/
│   ├── raw/                  # IHME source CSV (input, not generated)
│   └── processed/            # df_development.csv, df_real_world.csv
├── deprecated/                # Superseded files (old monolithic EDA.ipynb, initial thesis proposal)
├── docs/
│   ├── orientaciones_y_pautas.pdf       # Program guidelines for the TFM
│   └── TFM_Redaccion_Metodologia.docx   # Thesis methodology write-up (current draft)
├── notebooks/                 # Exploratory track
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_models.ipynb
│   └── 03b_model_improvements.ipynb     # Follow-up experiments on the SHAP findings from 03
├── outputs/
│   ├── figures/                # All saved plots, prefixed by pipeline stage (01_/02_/03_/04_)
│   ├── models/                 # Persisted production model + scaler (joblib)
│   └── tables/                 # Result tables (RMSE/MAE/R²) and predictions.csv
├── prod/                       # Production track — modular scripts, no notebooks
│   ├── 01_data_pipeline.py     # Ingestion + cleaning
│   ├── 02_eda.py               # EDA (figures + VIF), cleans df_development and df_real_world
│   ├── 03_train.py             # Trains + evaluates all 6 models, persists the production model
│   ├── predict.py              # Inference — scores new data with the persisted model
│   ├── 04_visualize_predictions.py  # Plots predict.py's output
│   └── run_pipeline.py         # Orchestrates 01 -> 02 -> 03 as separate processes
├── src/                        # Shared logic — imported by both notebooks/ and prod/
│   ├── config.py               # Constants (country lists, feature lists, indicator codes)
│   ├── data_loading.py         # IHME / World Bank / WHO fetch + cleaning + imputation
│   ├── features.py             # VIF, predictor list builder, IQR outlier flagging
│   ├── splits.py                # geographical_split(), temporal_split()
│   ├── models.py                # Model registry, hyperparameter grids, train/evaluate
│   ├── metrics.py               # Result tables, get_eval_entry(), metrics_by_period()
│   ├── diagnostics.py           # All plotting functions (EDA, results, predictions) + save_figure()
│   ├── explainability.py        # SHAP-based model interpretation
│   └── persistence.py           # save_artifact() / load_artifact() (joblib)
├── requirements.txt
└── TODO.md                      # Open TODOs / future work
```

Notebooks and `prod/` scripts are both thin orchestration layers over `src/` — the actual logic (fetching, feature engineering, splitting, modelling, plotting, persistence) lives in `src/` and is imported in either track.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a full environment freeze (includes the Jupyter/JupyterLab stack), not a curated minimal dependency list. The packages the pipeline actually imports: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `catboost`, `shap`, `joblib`, `statsmodels`, `pycountry`, `requests`, `matplotlib`, `seaborn`.

## Data sources

| Source | What it provides | Access |
|---|---|---|
| [IHME Global Burden of Disease](https://www.healthdata.org/) | Mental health / neurological disorder prevalence, per country/year/cause | Local CSV export (`data/raw/`) |
| World Bank API | GDP per capita, unemployment, health expenditure, population, Gini index, urban population %, physicians per 1,000, internet users % | Live API call, no key required |
| WHO Global Health Observatory API | Suicide rate per 100,000 inhabitants (target variable) | Live API call, no key required |

Scope: 27 EU member states, years 2000–2021 (`df_development`, labeled) plus 2022–2023 (`df_real_world`, unlabeled — WHO has not published suicide rates for these years yet).

## Running the pipeline

### Option 1 — Notebooks (exploratory)

Run in order, each reading what the previous one wrote to `data/processed/`:

1. **`01_data_loading_cleaning.ipynb`** — fetches and merges the three sources, imputes missing values, saves `df_development.csv` / `df_real_world.csv`.
2. **`02_eda.ipynb`** — trends, distributions, outliers, VIF-based multicollinearity check (drops `Eating disorders`).
3. **`03_models.ipynb`** — trains and compares all 6 models under both splits, result diagnostics, SHAP interpretability.
4. **`03b_model_improvements.ipynb`** — follow-up experiments testing hypotheses raised by 03's SHAP results (see "Model improvement experiments" below).

### Option 2 — Production scripts

```bash
python prod/run_pipeline.py                # full pipeline: 01 -> 02 -> 03
python prod/run_pipeline.py --skip-01       # reuse existing data/processed/
python prod/run_pipeline.py --only 03       # training only

python prod/predict.py                      # score df_real_world.csv
python prod/predict.py --input custom.csv --output custom_predictions.csv

python prod/04_visualize_predictions.py     # plot predict.py's output
```

`03_train.py` runs the full 6-model comparison (same as the notebook) **and** persists the chosen production model — CatBoost, Option B (time split) — plus its `RobustScaler`, to `outputs/models/`. `predict.py` loads those two artifacts directly; it never retrains. [Seguro] `01_data_pipeline.py` requires live internet access to the World Bank and WHO APIs — the other four scripts only need `data/processed/` and, for `predict.py`/`04_visualize_predictions.py`, an already-trained model.

`02_eda.py`, `03_train.py`, and `04_visualize_predictions.py` force matplotlib's `Agg` backend internally, since they only save figures to disk and never display them — this also avoids GUI-backend errors (e.g. broken Tcl/Tk installs) on machines without a working interactive backend.

## Results summary

From the current result tables in `outputs/tables/`:

- **Option A (geographical split):** all six models reach a positive Test R² (0.08–0.44, best: SVR 0.44, XGBoost 0.44), but five of six drop to a **negative** R² on Validation (as low as −0.62 for XGBoost) — the positive Test score does not transfer to genuinely unseen countries. Only SVR holds up (0.44 → 0.13).
- **Option B (time split):** stronger and more stable — Test R² 0.18–0.88, Validation R² 0.08–0.63. CatBoost (Test 0.88, Val 0.61) and SVR (Test 0.86, Val 0.63) are the most consistent performers.
This suggests the available indicators carry real predictive signal for near-future rates within known countries, but do not capture the country-specific structural factors (healthcare system, culture, history) that drive cross-country differences.

### SHAP interpretability (CatBoost, Option B)

`Alcohol use disorders` is by far the most influential predictor (mean |SHAP| ≈ 2.9, roughly 3x the next-ranked feature). Classic socioeconomic determinants (`Unemployment rate`, `Health expenditure`) rank lowest. This likely reflects that Option B only asks the model to explain variation *within* a fixed set of countries over time — features that vary mostly *across* countries rather than *within* them over time (like unemployment) contribute less to that specific task, not necessarily less to suicide risk in general.

Validation-set error is not flat across years: mean absolute error rises from ≈2.0 (2018) to ≈2.9 (2021) — degradation is concentrated in the two COVID-19 years (2020–2021), which the training period (2000–2014) could not have anticipated. Part of the Option A/B Test-Val gap reported above is attributable to this specific shock, not only to generic time-distance decay.

### Model improvement experiments (`03b_model_improvements.ipynb`)

Three hypotheses raised by the SHAP results were tested directly against the data. Only one held up:

| Hypothesis | Outcome |
|---|---|
| Hidden multicollinearity within the mental-health feature block | **Not confirmed** — max VIF 3.64 within the block |
| Test/Val gap is partly a COVID-period effect | **Confirmed** — Val R² 0.72 (pre-COVID) vs 0.47 (COVID years) |
| Dropping lowest-importance features improves generalization | **Not confirmed** — Val and Test R² both dropped slightly |

Negative results are documented rather than discarded.

## Status / open items

See `TODO.md` for the current list.
