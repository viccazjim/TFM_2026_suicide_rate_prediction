"""
Per-country time series models (SARIMAX, Prophet) — Option B only.

Unlike models.py's panel regressors (one model fit on all countries
pooled together), these are genuine time-series models: each one is
fit on a single country's own history and forecasts that same
country's future. This means they fundamentally cannot be used for
Option A (geographical split) — a country with zero training rows has
no history to forecast from, no matter which time-series model is
chosen. Only call these with Option B (temporal split) data.

With ~15 training points per country (70% of 2000-2021), the
parameter budget is tight: adding many exogenous regressors easily
exceeds the number of observations and produces a model that "fits"
without generalizing (verified empirically — 17 regressors on 15 points
drives the AIC to an implausible -235, a clear overfitting signature).
Both functions therefore default to a purely univariate fit (the
target's own history only) unless a short, deliberately chosen
`exog_features` list is passed in.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def _pooled_eval_dict(model_name: str, split: str, actuals, predictions) -> dict:
    """
    Builds an evaluate_model()-style dict from pooled (all countries
    concatenated) predictions, so per-country time series results plug
    into the same build_results_table() pipeline as the panel models
    without either one needing to know about the other.
    """
    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    mae = float(mean_absolute_error(actuals, predictions))
    r2 = float(r2_score(actuals, predictions)) if len(actuals) >= 2 else float("nan")
    return {
        "model": model_name,
        "split": split,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "predictions": predictions,
        "actuals": actuals,
    }


def train_evaluate_sarimax(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    target: str,
    exog_features: list[str] = None,
    id_col: str = "Code",
    year_col: str = "Year",
    order: tuple = (1, 1, 1),
):
    """
    Fits one SARIMAX per country on df_train's years only, forecasts
    the df_test and df_val years, and pools every country's forecasts
    into Test/Val evaluations comparable to the panel models' output.

    Countries where SARIMAX fails to converge are skipped (logged as a
    warning) and excluded from the pooled metrics — a silently-omitted
    country would understate error; an included NaN would break the
    metric calculation entirely. `per_country_results` records exactly
    which countries succeeded or failed, so the omission is visible
    rather than silent.

    Parameters
    ----------
    df_train, df_test, df_val : pd.DataFrame
        Option B splits — same countries, different years. Do not pass
        Option A splits (see module docstring).
    target : str
    exog_features : list[str], optional
        Exogenous regressor columns (SARIMAX natively supports this).
        Default None (univariate) — see module docstring for why a
        long list here is a real overfitting risk at this sample size.
    id_col : str, default "Code"
    year_col : str, default "Year"
    order : tuple, default (1, 1, 1)
        Non-seasonal (p, d, q) — no seasonal term, since annual data
        has no sub-annual seasonality to capture, and kept low-order
        given how few points each country provides. (1,1,1) was
        selected over the originally-used (1,1,0) after
        notebooks/05b_temporal_persistence_improvements.ipynb tested
        it directly against four alternative orders and found it beat
        (1,1,0) on both Test and Validation R² for the +1-exogenous-
        feature configuration. Mechanically, AR(1) alone models each
        country's year as depending only on its own previous level;
        MA(1) additionally lets the model account for last period's
        forecast error, capturing the short-lived echo a one-off
        shock tends to leave in the following year — one extra
        parameter, not a materially larger search space, given the
        small per-country sample. See that notebook for the comparison.

    Returns
    -------
    tuple[list[dict], pd.DataFrame]
        (eval_results, per_country_results). eval_results is
        [Test dict, Val dict], each in evaluate_model()'s format.
        per_country_results has one row per country per split, with
        that country's own RMSE and a "converged" flag.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    test_preds, test_actuals = [], []
    val_preds, val_actuals = [], []
    per_country_rows = []

    for code in sorted(df_train[id_col].unique()):
        train_c = df_train[df_train[id_col] == code].sort_values(year_col)
        test_c = df_test[df_test[id_col] == code].sort_values(year_col)
        val_c = df_val[df_val[id_col] == code].sort_values(year_col)

        y_train = train_c[target].to_numpy()
        exog_train = train_c[exog_features].to_numpy() if exog_features else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train, exog=exog_train, order=order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)

            exog_forecast = None
            if exog_features:
                exog_forecast = pd.concat([test_c, val_c])[exog_features].to_numpy()
            forecast = np.asarray(model.forecast(steps=len(test_c) + len(val_c), exog=exog_forecast))

            test_forecast, val_forecast = forecast[:len(test_c)], forecast[len(test_c):]

        except Exception as e:
            logger.warning("SARIMAX failed to converge for %s: %s", code, e)
            for split_name in ("Test", "Val"):
                per_country_rows.append({"Code": code, "Split": split_name, "RMSE": float("nan"), "converged": False})
            continue

        test_preds.extend(test_forecast); test_actuals.extend(test_c[target].to_numpy())
        val_preds.extend(val_forecast); val_actuals.extend(val_c[target].to_numpy())

        for split_name, actual, pred in [("Test", test_c[target].to_numpy(), test_forecast),
                                          ("Val", val_c[target].to_numpy(), val_forecast)]:
            rmse = float(np.sqrt(np.mean((actual - pred) ** 2))) if len(actual) else float("nan")
            per_country_rows.append({"Code": code, "Split": split_name, "RMSE": round(rmse, 4), "converged": True})

    eval_results = [
        _pooled_eval_dict("SARIMAX", "Test", test_actuals, test_preds),
        _pooled_eval_dict("SARIMAX", "Val", val_actuals, val_preds),
    ]
    return eval_results, pd.DataFrame(per_country_rows)


def fit_sarimax_models(
    df_train: pd.DataFrame,
    target: str,
    exog_features: list[str] = None,
    id_col: str = "Code",
    year_col: str = "Year",
    order: tuple = (1, 1, 1),
) -> dict:
    """
    Fits one SARIMAX per country and returns the fitted results objects,
    without forecasting or evaluating — used to extend a forecast past
    the years train_evaluate_sarimax() checked (e.g. into
    df_real_world's years), separate from that function's fit +
    forecast + pooled-evaluation flow used to compare against the
    naive baseline and the panel models.

    Parameters
    ----------
    df_train : pd.DataFrame
    target : str
    exog_features : list[str], optional
        Same overfitting caveat as train_evaluate_sarimax() — default
        None (univariate).
    id_col : str, default "Code"
    year_col : str, default "Year"
    order : tuple, default (1, 1, 1) — see train_evaluate_sarimax()'s
        docstring for why this isn't (1,1,0) anymore.

    Returns
    -------
    dict[str, SARIMAXResultsWrapper]
        {country_code: fitted results}. Countries where fitting failed
        are simply absent.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    models = {}
    for code in sorted(df_train[id_col].unique()):
        train_c = df_train[df_train[id_col] == code].sort_values(year_col)
        y_train = train_c[target].to_numpy()
        exog_train = train_c[exog_features].to_numpy() if exog_features else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models[code] = SARIMAX(
                    y_train, exog=exog_train, order=order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
        except Exception as e:
            logger.warning("SARIMAX failed to fit for %s: %s", code, e)

    return models


def forecast_sarimax(models: dict, code: str, years_exog_df: pd.DataFrame, year_col: str = "Year",
                      exog_features: list[str] = None) -> pd.Series:
    """
    Forecasts a contiguous block of years for one country using an
    already-fitted SARIMAX from fit_sarimax_models().

    Unlike Prophet, SARIMAX's `.forecast()` walks forward step by step
    from the end of its training data — to reach a year far past
    training (e.g. df_real_world's 2022-2023, when the model was fit
    through ~2014), you must forecast *every* intervening year in
    order, exogenous values included, and take only the years you
    actually want from the result. `years_exog_df` must therefore
    contain every year from immediately after training through the
    furthest year you want — not just the years you're ultimately
    interested in — sorted ascending.

    Parameters
    ----------
    models : dict[str, SARIMAXResultsWrapper]
        Output of fit_sarimax_models().
    code : str
        Country ISO code — must be a key in `models`.
    years_exog_df : pd.DataFrame
        One row per year, `year_col` plus `exog_features` (if used),
        covering the *entire* contiguous span from just after training
        through the last year needed. Sorted ascending by year.
    year_col : str, default "Year"
    exog_features : list[str], optional
        Must match what the model in `models` was fit with.

    Returns
    -------
    pd.Series
        Forecasted values, indexed by year, covering every year in
        `years_exog_df` — slice out the specific years you need from
        the result (e.g. `result.loc[[2022, 2023]]`).

    Raises
    ------
    KeyError
        If `code` has no fitted model.
    """
    if code not in models:
        raise KeyError(f"No fitted SARIMAX model for country '{code}'")

    years_exog_df = years_exog_df.sort_values(year_col)
    exog_forecast = years_exog_df[exog_features].to_numpy() if exog_features else None
    forecast = models[code].forecast(steps=len(years_exog_df), exog=exog_forecast)
    return pd.Series(np.asarray(forecast), index=years_exog_df[year_col].to_numpy())


def fit_prophet_models(
    df_train: pd.DataFrame,
    target: str,
    exog_features: list[str] = None,

    id_col: str = "Code",
    year_col: str = "Year",
) -> dict:
    """
    Fits one Prophet model per country and returns them, without
    forecasting or evaluating — used to persist production models for
    predict.py, separate from train_evaluate_prophet()'s fit + forecast
    + pooled-evaluation flow used to compare against the panel models.

    Parameters
    ----------
    df_train : pd.DataFrame
    target : str
    exog_features : list[str], optional
        Same overfitting caveat as train_evaluate_prophet() — default
        None (univariate).
    id_col : str, default "Code"
    year_col : str, default "Year"

    Returns
    -------
    dict[str, Prophet]
        {country_code: fitted Prophet model}. Countries where fitting
        failed are simply absent — check `set(df_train[id_col].unique())
        - set(result.keys())` to see which, if any.
    """
    from prophet import Prophet

    models = {}
    for code in sorted(df_train[id_col].unique()):
        train_c = df_train[df_train[id_col] == code].sort_values(year_col)
        prophet_train = pd.DataFrame({
            "ds": pd.to_datetime(train_c[year_col], format="%Y"),
            "y": train_c[target].to_numpy(),
        })
        if exog_features:
            for feat in exog_features:
                prophet_train[feat] = train_c[feat].to_numpy()

        try:
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            if exog_features:
                for feat in exog_features:
                    model.add_regressor(feat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_train)
            models[code] = model
        except Exception as e:
            logger.warning("Prophet failed to fit for %s: %s", code, e)

    return models


def forecast_prophet(models: dict, code: str, years, exog_df: pd.DataFrame = None) -> np.ndarray:
    """
    Forecasts specific years for one country using an already-fitted
    Prophet model from fit_prophet_models(). Works for years beyond
    what the model was evaluated on — Prophet simply extends its
    fitted trend — but forecast uncertainty grows the further out you
    go, and nothing here warns you about that; treat forecasts far
    past the training period with proportionally more caution.

    Parameters
    ----------
    models : dict[str, Prophet]
        Output of fit_prophet_models().
    code : str
        Country ISO code — must be a key in `models`.
    years : array-like of int
    exog_df : pd.DataFrame, optional
        Required only if the model was fit with exog_features — must
        have one row per year in `years`, with those exact columns.

    Returns
    -------
    np.ndarray
        Forecasted values, same order as `years`.

    Raises
    ------
    KeyError
        If `code` has no fitted model (e.g. it failed during fitting,
        or was never a training country at all — see predict.py for
        how the caller should handle this rather than letting it raise).
    """
    if code not in models:
        raise KeyError(f"No fitted Prophet model for country '{code}'")

    future_df = pd.DataFrame({"ds": pd.to_datetime(pd.Series(years), format="%Y")})
    if exog_df is not None:
        future_df = pd.concat([future_df.reset_index(drop=True), exog_df.reset_index(drop=True)], axis=1)

    return models[code].predict(future_df)["yhat"].to_numpy()


def train_evaluate_prophet(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    target: str,
    exog_features: list[str] = None,
    id_col: str = "Code",
    year_col: str = "Year",
):
    """
    Fits one Prophet model per country on df_train's years only,
    forecasts the df_test and df_val years, and pools every country's
    forecasts into Test/Val evaluations. Same Option-B-only constraint,
    same overfitting caveat about `exog_features`, as
    train_evaluate_sarimax() — see that function's and the module's
    docstrings.

    Parameters
    ----------
    df_train, df_test, df_val : pd.DataFrame
    target : str
    exog_features : list[str], optional
        Passed to Prophet via add_regressor(). Requires the same
        columns to exist in df_test/df_val — true here, since they are
        held-out historical rows with known predictor values, not
        genuinely unknown future data.
    id_col : str, default "Code"
    year_col : str, default "Year"

    Returns
    -------
    tuple[list[dict], pd.DataFrame]
        Same shape as train_evaluate_sarimax().
    """
    from prophet import Prophet

    test_preds, test_actuals = [], []
    val_preds, val_actuals = [], []
    per_country_rows = []

    for code in sorted(df_train[id_col].unique()):
        train_c = df_train[df_train[id_col] == code].sort_values(year_col)
        test_c = df_test[df_test[id_col] == code].sort_values(year_col)
        val_c = df_val[df_val[id_col] == code].sort_values(year_col)

        prophet_train = pd.DataFrame({
            "ds": pd.to_datetime(train_c[year_col], format="%Y"),
            "y": train_c[target].to_numpy(),
        })
        if exog_features:
            for feat in exog_features:
                prophet_train[feat] = train_c[feat].to_numpy()

        try:
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            if exog_features:
                for feat in exog_features:
                    model.add_regressor(feat)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_train)

            future_c = pd.concat([test_c, val_c]).sort_values(year_col)
            future_df = pd.DataFrame({"ds": pd.to_datetime(future_c[year_col], format="%Y")})
            if exog_features:
                for feat in exog_features:
                    future_df[feat] = future_c[feat].to_numpy()

            forecast = model.predict(future_df)["yhat"].to_numpy()
            test_forecast, val_forecast = forecast[:len(test_c)], forecast[len(test_c):]

        except Exception as e:
            logger.warning("Prophet failed to converge for %s: %s", code, e)
            for split_name in ("Test", "Val"):
                per_country_rows.append({"Code": code, "Split": split_name, "RMSE": float("nan"), "converged": False})
            continue

        test_preds.extend(test_forecast); test_actuals.extend(test_c[target].to_numpy())
        val_preds.extend(val_forecast); val_actuals.extend(val_c[target].to_numpy())

        for split_name, actual, pred in [("Test", test_c[target].to_numpy(), test_forecast),
                                          ("Val", val_c[target].to_numpy(), val_forecast)]:
            rmse = float(np.sqrt(np.mean((actual - pred) ** 2))) if len(actual) else float("nan")
            per_country_rows.append({"Code": code, "Split": split_name, "RMSE": round(rmse, 4), "converged": True})

    eval_results = [
        _pooled_eval_dict("Prophet", "Test", test_actuals, test_preds),
        _pooled_eval_dict("Prophet", "Val", val_actuals, val_preds),
    ]
    return eval_results, pd.DataFrame(per_country_rows)
