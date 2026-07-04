from src.config import ID_COLS, TARGET, EU_COUNTRIES_ISO, EU_REGIONS

from src.data_loading import (
    load_ihme_data,
    fetch_worldbank_indicators,
    fetch_who_suicide_rates,
    build_master_dataset,
    impute_missing_values,
)
from src.features import compute_vif, build_predictor_list, flag_outliers_iqr

from src.splits import geographical_split, temporal_split

from src.models import param_grids, make_baseline_models, train_model, evaluate_model

from src.metrics import build_results_table, normalized_rmse_table, persistence_baseline

from src.diagnostics import (
    suicide_evolution_graph,
    plot_vif_bar,
    plot_rmse_comparison,
    plot_r2_comparison,
)
