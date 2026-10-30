"""Five-year city HPI forecasting candidates.

This module is compatible with scripts/train.py's interface:
    FORECAST_METHODS, HORIZON_QUARTERS, MIN_HISTORY_QUARTERS, forecast_index

All statistical models fit only the historical HPI values passed to forecast_index.
No future observations or property listing target fields are used by the forecast.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

HORIZON_QUARTERS = 20
# Three years of quarterly history gives the seasonal models enough observations
# to start while preserving more rolling-origin validation dates. The five-year target still requires a longer complete series overall.
MIN_HISTORY_QUARTERS = 12

# Simple baselines remain in the comparison; complex models must beat them.
FORECAST_METHODS = [
    "last_value",
    "seasonal_naive",
    "drift",
    "trailing_1y_growth",
    "trailing_3y_growth",
    "trailing_5y_growth",
    "damped_recent_growth",
    "linear_log_trend",
    "ets_damped",
    "ets_damped_seasonal",
    "theta",
    "arima_011",
    "arima_110",
    "arima_111",
    "sarima_011_011_4",
    "structural_local_trend",
    "ensemble_trend_average",
    "global_ridge_growth",
    "global_random_forest_growth",
    "global_extra_trees_growth",
    "global_hist_gradient_boosting_growth",
]
GLOBAL_FORECAST_METHODS = {
    "global_ridge_growth",
    "global_random_forest_growth",
    "global_extra_trees_growth",
    "global_hist_gradient_boosting_growth",
}


def _clean_history(history: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(history), dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("HPI history cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("HPI history contains missing or non-finite values")
    if (values <= 0).any():
        raise ValueError("HPI values must be positive")
    return values


class ForecastInstabilityError(ValueError):
    """Raised when a model produces a non-finite or implausible HPI forecast."""


def _log_growth_limits(history: Iterable[float], horizon: int) -> tuple[float, float]:
    """Derive permissive stability limits from history available at this origin.

    These are numerical safeguards, not statistical confidence intervals. The
    quarterly bound catches exploding forecast paths; the cumulative bound
    catches otherwise finite but extreme long-horizon endpoints.
    """
    y = _clean_history(history)
    returns = np.diff(np.log(y))
    if returns.size == 0:
        return 0.12, 0.90
    median = float(np.median(returns))
    mad = float(np.median(np.abs(returns - median)))
    robust_scale = max(1.4826 * mad, 0.25 * float(np.std(returns)), 0.003)
    abs_q99 = float(np.quantile(np.abs(returns), 0.99))
    quarterly_limit = max(0.10, abs_q99 * 2.0, abs(median) + 8.0 * robust_scale)
    cumulative_limit = max(0.90, float(horizon) * (abs(median) + 3.0 * robust_scale))
    return quarterly_limit, cumulative_limit


def validate_forecast_endpoint(
    history: Iterable[float], forecast_value: float, horizon: int, method: str = "forecast"
) -> float:
    """Validate an HPI point forecast against positivity and history-based bounds."""
    y = _clean_history(history)
    try:
        value = float(forecast_value)
    except (TypeError, ValueError) as exc:
        raise ForecastInstabilityError(f"{method}: forecast is not numeric") from exc
    if not np.isfinite(value) or value <= 0:
        raise ForecastInstabilityError(f"{method}: forecast must be finite and positive")
    log_change = float(np.log(value) - np.log(y[-1]))
    _, cumulative_limit = _log_growth_limits(y, horizon)
    if abs(log_change) > cumulative_limit:
        raise ForecastInstabilityError(
            f"{method}: five-year log change {log_change:.3f} exceeds history-based limit "
            f"±{cumulative_limit:.3f}; candidate rejected as unstable"
        )
    return value


def forecast_index_safe(
    history: Iterable[float], horizon: int, method: str
) -> tuple[float, bool, str | None]:
    """Forecast safely; if a candidate fails, return last-value baseline and reason.

    The fallback is explicitly recorded by the training script, so an unstable
    candidate cannot silently emit an extreme price projection.
    """
    y = _clean_history(history)
    try:
        value = forecast_index(y, horizon, method)
        return validate_forecast_endpoint(y, value, horizon, method), False, None
    except Exception as exc:
        return float(y[-1]), True, f"{type(exc).__name__}: {str(exc)[:240]}"


def _growth_forecast(y: np.ndarray, horizon: int, quarters: int) -> np.ndarray:
    """Compound recent average log growth; use available history if shorter."""
    log_y = np.log(y)
    growth = np.diff(log_y)
    window = min(quarters, len(growth))
    if window < 1:
        return np.repeat(y[-1], horizon)
    rate = float(np.mean(growth[-window:]))
    return np.exp(log_y[-1] + rate * np.arange(1, horizon + 1))


def _damped_recent_growth(y: np.ndarray, horizon: int) -> np.ndarray:
    """Blend recent and long-run log growth, then damp it toward a flat trend."""
    log_y = np.log(y)
    growth = np.diff(log_y)
    if len(growth) == 0:
        return np.repeat(y[-1], horizon)
    recent = float(np.mean(growth[-min(8, len(growth)):]))
    long_run = float(np.mean(growth))
    rate = 0.65 * recent + 0.35 * long_run
    # Near-term momentum fades gradually rather than compounding unchanged forever.
    damping = 0.96
    cumulative = np.cumsum(rate * damping ** np.arange(horizon))
    return np.exp(log_y[-1] + cumulative)


def _linear_log_trend(y: np.ndarray, horizon: int) -> np.ndarray:
    log_y = np.log(y)
    x = np.arange(len(log_y), dtype=float)
    slope, intercept = np.polyfit(x, log_y, deg=1)
    future_x = np.arange(len(log_y), len(log_y) + horizon, dtype=float)
    return np.exp(intercept + slope * future_x)


def _fit_statsmodels(y: np.ndarray, horizon: int, method: str) -> np.ndarray:
    """Forecast log HPI with stability checks; reject divergent fits instead of clipping.

    Important: a huge ARIMA forecast must not be hidden by exponentiation clipping.
    Non-converged fits or forecasts outside broad history-derived limits are rejected
    and the training workflow records a last-value fallback for that origin/city.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    log_y = np.log(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "ets_damped":
            fit = ExponentialSmoothing(
                log_y, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
            prediction = fit.forecast(horizon)
        elif method == "ets_damped_seasonal":
            fit = ExponentialSmoothing(
                log_y, trend="add", damped_trend=True, seasonal="add",
                seasonal_periods=4, initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
            prediction = fit.forecast(horizon)
        elif method == "theta":
            fit = ThetaModel(log_y, period=4, deseasonalize=True).fit()
            prediction = fit.forecast(horizon)
        elif method == "sarima_011_011_4":
            fit = SARIMAX(
                log_y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 4),
                enforce_stationarity=True, enforce_invertibility=True,
            ).fit(disp=False, maxiter=150)
            prediction = fit.forecast(steps=horizon)
            retvals = getattr(fit, "mle_retvals", {}) or {}
            if retvals.get("converged") is False:
                raise ForecastInstabilityError(f"{method}: SARIMAX optimizer did not converge")
        elif method == "structural_local_trend":
            fit = UnobservedComponents(log_y, level="local linear trend", seasonal=4).fit(disp=False)
            prediction = fit.forecast(horizon)
            retvals = getattr(fit, "mle_retvals", {}) or {}
            if retvals.get("converged") is False:
                raise ForecastInstabilityError(f"{method}: structural model optimizer did not converge")
        else:
            orders = {
                "arima_011": (0, 1, 1),
                "arima_110": (1, 1, 0),
                "arima_111": (1, 1, 1),
            }
            fit = ARIMA(
                log_y, order=orders[method],
                enforce_stationarity=True, enforce_invertibility=True,
            ).fit(method_kwargs={"maxiter": 150})
            retvals = getattr(fit, "mle_retvals", {}) or {}
            if retvals.get("converged") is False:
                raise ForecastInstabilityError(f"{method}: ARIMA optimizer did not converge")
            prediction = fit.forecast(steps=horizon)

    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if prediction.size != horizon or not np.isfinite(prediction).all():
        raise ForecastInstabilityError(f"{method}: non-finite or wrong-length log forecast")

    # Check each quarterly forecast step and the total endpoint change before exp().
    log_path = np.concatenate(([float(log_y[-1])], prediction))
    future_changes = np.diff(log_path)
    quarterly_limit, cumulative_limit = _log_growth_limits(y, horizon)
    if float(np.max(np.abs(future_changes))) > quarterly_limit:
        raise ForecastInstabilityError(
            f"{method}: forecast step change {float(np.max(np.abs(future_changes))):.3f} "
            f"exceeds history-based limit ±{quarterly_limit:.3f}"
        )
    total_change = float(prediction[-1] - log_y[-1])
    if abs(total_change) > cumulative_limit:
        raise ForecastInstabilityError(
            f"{method}: five-year log change {total_change:.3f} exceeds history-based limit "
            f"±{cumulative_limit:.3f}"
        )
    if (prediction > 20).any() or (prediction < -20).any():
        raise ForecastInstabilityError(f"{method}: forecast log-level outside numerical safety range")
    forecast = np.exp(prediction)
    if not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ForecastInstabilityError(f"{method}: exponentiated forecast is non-finite or non-positive")
    return forecast


def _ensemble(y: np.ndarray, horizon: int) -> np.ndarray:
    """Fixed geometric blend of complementary trend candidates."""
    components = [
        _damped_recent_growth(y, horizon),
        _growth_forecast(y, horizon, 12),
        _linear_log_trend(y, horizon),
    ]
    # Include ETS only when it fits; fail-soft here because this is an ensemble
    # candidate, while standalone statistical candidates still report failures.
    try:
        components.append(_fit_statsmodels(y, horizon, "ets_damped"))
    except Exception:
        pass
    log_stack = np.vstack([np.log(np.maximum(c, 1e-9)) for c in components])
    return np.exp(np.mean(log_stack, axis=0))


def _global_feature_row(city: str, log_history: np.ndarray, next_quarter_index: int) -> dict:
    """Build strictly lagged features for one city at one forecast origin."""
    returns = np.diff(log_history)
    def lag_return(q: int) -> float:
        return float(log_history[-1] - log_history[-1 - q]) if len(log_history) > q else 0.0
    tail4 = returns[-min(4, len(returns)):]
    tail8 = returns[-min(8, len(returns)):]
    return {
        "city": city,
        "log_level": float(log_history[-1]),
        "return_1q": lag_return(1),
        "return_2q": lag_return(2),
        "return_4q": lag_return(4),
        "return_8q": lag_return(8),
        "mean_return_4q": float(np.mean(tail4)) if len(tail4) else 0.0,
        "mean_return_8q": float(np.mean(tail8)) if len(tail8) else 0.0,
        "std_return_8q": float(np.std(tail8, ddof=1)) if len(tail8) > 1 else 0.0,
        "quarter_sin": float(np.sin(2 * np.pi * (next_quarter_index % 4) / 4)),
        "quarter_cos": float(np.cos(2 * np.pi * (next_quarter_index % 4) / 4)),
    }


def _global_regressor(method: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor

    numeric = [
        "log_level", "return_1q", "return_2q", "return_4q", "return_8q",
        "mean_return_4q", "mean_return_8q", "std_return_8q", "quarter_sin", "quarter_cos",
    ]
    prep = ColumnTransformer([
        ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ], remainder="drop")
    estimators = {
        "global_ridge_growth": Ridge(alpha=10.0),
        "global_random_forest_growth": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=5, max_features=0.9,
            random_state=42, n_jobs=1,
        ),
        "global_extra_trees_growth": ExtraTreesRegressor(
            n_estimators=120, min_samples_leaf=5, max_features=0.9,
            random_state=42, n_jobs=1,
        ),
        "global_hist_gradient_boosting_growth": HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        ),
    }
    if method not in estimators:
        raise ValueError(f"Unknown global HPI method: {method!r}")
    return Pipeline([("preprocess", prep), ("model", estimators[method])])


def forecast_global_panel(
    histories: dict[str, Iterable[float]], horizon: int, method: str
) -> dict[str, float]:
    """Fit one pooled lag-based ML model on all cities, then recursively forecast.

    Historical training examples use only transitions fully observed within each
    supplied history. `histories` must contain data available at one common origin.
    Returns the point forecast at the requested horizon for each city.
    """
    if method not in GLOBAL_FORECAST_METHODS:
        raise ValueError(f"Not a global panel forecasting method: {method!r}")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least one quarter")
    clean = {str(city): _clean_history(vals) for city, vals in histories.items()}
    if not clean:
        raise ValueError("At least one city history is required")
    lengths = {len(v) for v in clean.values()}
    if len(lengths) != 1:
        raise ValueError("All city histories must share the same forecast origin")

    # Build pooled one-quarter-ahead log-growth samples using information up to t.
    x_rows, y_rows = [], []
    for city, values in clean.items():
        log_y = np.log(values)
        # Eight quarters are required to produce the longest lag feature.
        for t in range(8, len(log_y) - 1):
            x_rows.append(_global_feature_row(city, log_y[:t + 1], t + 1))
            y_rows.append(float(log_y[t + 1] - log_y[t]))
    if len(x_rows) < 20:
        raise ValueError(
            f"Too few pooled training examples ({len(x_rows)}) for {method}; "
            "need at least 20 transitions across cities."
        )
    X = pd.DataFrame(x_rows)
    y = np.asarray(y_rows, dtype=float)
    model = _global_regressor(method)
    model.fit(X, y)

    logs = {city: np.log(values).astype(float).tolist() for city, values in clean.items()}
    # Recursively generate each future quarter using only already-observed or
    # previously predicted values. Forecast growth is clipped to a robust range
    # based on observed pooled quarter-to-quarter log returns.
    all_returns = np.concatenate([np.diff(np.log(v)) for v in clean.values() if len(v) > 1])
    low, high = np.quantile(all_returns, [0.02, 0.98])
    padding = max(0.015, float(high - low) * 0.25)
    low, high = float(low - padding), float(high + padding)
    for step in range(horizon):
        rows = [
            _global_feature_row(city, np.asarray(log_hist, dtype=float), len(log_hist))
            for city, log_hist in logs.items()
        ]
        features = pd.DataFrame(rows)
        predicted_growth = np.asarray(model.predict(features), dtype=float)
        for idx, city in enumerate(logs):
            growth = float(np.clip(predicted_growth[idx], low, high))
            logs[city].append(logs[city][-1] + growth)
    return {city: float(np.exp(log_hist[-1])) for city, log_hist in logs.items()}


def forecast_index(history: Iterable[float], horizon: int, method: str) -> float:
    """Forecast a positive quarterly city HPI series for ``horizon`` quarters.

    Returns the forecasted HPI point at the end of the requested horizon. The
    caller evaluates candidates chronologically and retains a final temporal
    holdout. Global pooled ML methods use `forecast_global_panel` instead.
    """
    y = _clean_history(history)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least one quarter")
    if len(y) < 2 and method not in {"last_value", "seasonal_naive"}:
        raise ValueError(f"{method} needs at least two historical HPI observations")

    if method == "last_value":
        forecast = np.repeat(y[-1], horizon)
    elif method == "seasonal_naive":
        season = y[-min(4, len(y)):]
        forecast = np.resize(season, horizon)
    elif method == "drift":
        slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
        forecast = y[-1] + slope * np.arange(1, horizon + 1)
    elif method == "trailing_1y_growth":
        forecast = _growth_forecast(y, horizon, 4)
    elif method == "trailing_3y_growth":
        forecast = _growth_forecast(y, horizon, 12)
    elif method == "trailing_5y_growth":
        forecast = _growth_forecast(y, horizon, 20)
    elif method == "damped_recent_growth":
        forecast = _damped_recent_growth(y, horizon)
    elif method == "linear_log_trend":
        forecast = _linear_log_trend(y, horizon)
    elif method in {
        "ets_damped", "ets_damped_seasonal", "theta", "arima_011", "arima_110",
        "arima_111", "sarima_011_011_4", "structural_local_trend"
    }:
        forecast = _fit_statsmodels(y, horizon, method)
    elif method == "ensemble_trend_average":
        forecast = _ensemble(y, horizon)
    elif method in GLOBAL_FORECAST_METHODS:
        raise ValueError(f"{method} needs cross-city histories; use forecast_global_panel()")
    else:
        raise ValueError(f"Unknown HPI forecasting method: {method!r}")

    forecast = np.asarray(forecast, dtype=float).reshape(-1)
    if len(forecast) != horizon:
        raise ForecastInstabilityError(f"{method} returned {len(forecast)} values; expected {horizon}")
    if not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ForecastInstabilityError(f"{method} returned non-finite or non-positive forecast values")
    # Catch explosive trends in non-ARIMA candidates too.
    validate_forecast_endpoint(y, float(forecast[-1]), horizon, method)
    return float(forecast[-1])
