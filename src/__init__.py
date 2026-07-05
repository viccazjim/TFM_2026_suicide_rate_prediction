from src.config import (
    ID_COLS,
    TARGET,
    EU_COUNTRIES_ISO,
    EU_REGIONS,
    SOCIAL_ECONOMIC_FEATURES,
    HEALTH_RELATED_FEATURES,
    WORLD_BANK_INDICATORS,
    WHO_SUICIDE_INDICATOR,
)
from src.data_loading import (
    country_code,
    load_ihme_data,
    fetch_worldbank_indicators,
    fetch_who_suicide_rates,
    build_master_dataset,
    interpolate_with_trend_extrapolation,
)
from src.features import (
    compute_vif,
    build_predictor_list,
    flag_outliers_iqr,
)
from src.splits import geographical_split, temporal_split
from src.models import (
    param_grids,
    make_models,
    train_model,
    evaluate_model,
)
from src.metrics import (
    build_results_table,
    get_eval_entry,
    metrics_by_period,
)
from src.diagnostics import (
    suicide_evolution_graph,
    plot_vif_bar,
    plot_suicide_trend_by_region,
    plot_suicide_boxplot_by_country,
    plot_feature_distributions,
    plot_correlation_heatmaps,
    plot_suicide_dispersion_stripplot,
    plot_rmse_comparison,
    plot_r2_comparison,
    plot_actual_vs_predicted,
    plot_residual_histogram,
    plot_residuals_vs_predicted,
    plot_error_by_year,
    mean_absolute_error_by_country,
)

from src.explainability import (
    make_shap_explainer,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
)
