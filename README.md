# TFM — Predictive analysis of suicide rate in EU countries

## 1. Entorno virtual

Desde la raíz del proyecto (`tfm-suicide-rate/`):

# Windows (PowerShell)
```bash
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Registrar el entorno como kernel de Jupyter (para poder seleccionarlo dentro de VS Code / JupyterLab):

```bash
python -m ipykernel install --user --name=tfm-suicide-rate --display-name "TFM Suicide Rate"
```

Arrancar JupyterLab:

```bash
jupyter lab
```

Al abrir cualquier notebook de `notebooks/`, selecciona el kernel **"TFM Suicide Rate"** (no el kernel Python global) para asegurarte de que usa el entorno con las versiones fijadas en `requirements.txt`.

### Congelar versiones exactas (reproducibilidad)

Una vez que el proyecto esté funcionando y estable, sustituye `requirements.txt` por el listado exacto de lo que tienes instalado, así cualquiera que clone el repo (o tú misma en otro ordenador) reproduce exactamente el mismo entorno:

```bash
pip freeze > requirements.txt
```

## 2. Estructura del proyecto

```
tfm-suicide-rate/
├── data/
│   ├── raw/                          # datos originales (WHO, World Bank) sin modificar
│   └── processed/                    # df_development.parquet, df_enhanced.parquet
├── src/
│   ├── features.py                   # lags, VIF, lista de predictores
│   ├── splits.py                     # Option A (geográfico) / Option B (temporal)
│   ├── models.py                     # registro de modelos, grids, train_model, evaluate_model
│   ├── metrics.py                    # tablas de resultados, NRMSE, baseline de persistencia
│   └── diagnostics.py                # gráficos de residuos, VIF, comparativas RMSE/R²
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_ablation_lags.ipynb
│   ├── 05_boosting_and_trend.ipynb
│   └── 06_results_summary.ipynb
├── outputs/
│   ├── figures/                      # PNGs para pegar en la memoria de la tesis
│   └── tables/                       # CSVs de resultados finales
├── requirements.txt
└── README.md
```

## 3. Mapeo de celdas del notebook original (`EDA_models_VC.ipynb`) a la nueva estructura

| Celdas originales | Contenido | Destino |
|---|---|---|
| 0–20 | Carga de datos WHO/World Bank, imputación de missing values, EDA descriptivo (tendencias por país/región) | `notebooks/01_eda.ipynb` (guarda `df_development` procesado en `data/processed/`) |
| 21–28 | Distribución de features, outliers, VIF, drop de "Eating disorders" | `notebooks/01_eda.ipynb` (llama a `src/features.compute_vif`) |
| 29–41 | Option A / Option B: splits, correlaciones, escalado | `notebooks/03_baseline_models.ipynb` (llama a `src/splits.option_a_split` / `option_b_split`) |
| 42–58 | Definición de modelos, entrenamiento Ridge/Lasso/SVR/RF, tablas y gráficos de resultados | `notebooks/03_baseline_models.ipynb` (llama a `src/models.*`, `src/metrics.build_results_table`, `src/diagnostics.plot_rmse_comparison`) |
| 59–63 | Feature engineering con lags (sin deltas — ver nota en `src/features.py`) | `notebooks/02_feature_engineering.ipynb` (llama a `src/features.add_lag_features`) |
| Ablation A/B/C + persistence baseline (de la conversación) | Comparación de configs de lag | `notebooks/04_ablation_lags.ipynb` (llama a `src/metrics.persistence_baseline`) |
| Residuos por año, XGBoost/CatBoost, NRMSE, tendencia temporal (`Year`) | Diagnóstico + modelos de boosting | `notebooks/05_boosting_and_trend.ipynb` (llama a `src/diagnostics.plot_residuals_by_year`, `src/models.make_boosted_models`, `src/metrics.normalized_rmse_table`) |
| 64–72 | Comparación final baseline vs improved | `notebooks/06_results_summary.ipynb` |

**Regla práctica al trocear:** cualquier celda que define una función reutilizable (`compute_vif`, `train_model`, `build_results_table`, etc.) va a `src/`. Cualquier celda que ejecuta esa función sobre tus datos concretos y genera un gráfico/tabla para la memoria se queda en el notebook correspondiente.

## 4. Cómo importar `src/` desde un notebook

Al principio de cada notebook en `notebooks/`:

```python
import sys
sys.path.append("..")

from src.features import compute_vif, build_predictor_list, add_lag_features
from src.splits import option_a_split, option_b_split
from src.models import (
    param_grids, make_baseline_models,
    param_grids_boost, make_boosted_models,
    train_model, evaluate_model,
)
from src.metrics import build_results_table, persistence_baseline, normalized_rmse_table
from src.diagnostics import (
    plot_residuals_by_year, plot_vif_bar,
    plot_rmse_comparison, plot_r2_comparison,
)
```

## 5. Guardar/cargar datos procesados entre notebooks

En `02_feature_engineering.ipynb`, al final:

```python
df_development.to_parquet("../data/processed/df_development.parquet")
df_enhanced.to_parquet("../data/processed/df_enhanced.parquet")
```

En cualquier notebook posterior:

```python
import pandas as pd
df_development = pd.read_parquet("../data/processed/df_development.parquet")
```

Esto evita depender de ejecutar los notebooks en un orden exacto cada vez que abres el proyecto.
