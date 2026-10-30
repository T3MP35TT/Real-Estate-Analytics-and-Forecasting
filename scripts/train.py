"""Cross-validated model comparison and held-out evaluation for proxy targets.

Important: these targets are generated from rules, not observed investment or
resale outcomes. Metrics measure reconstruction of those rules only.
"""
import json
import hashlib
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
    cross_validate,
    train_test_split,
)

# Running `python scripts/train.py` puts scripts/ on sys.path, not the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNUAL_APPRECIATION_ASSUMPTION,
    AGE_DEPRECIATION_PER_YEAR,
    ARTIFACT_DIR,
    CLASSIFICATION_FEATURES,
    DATA_PATH,
    HPI_PATH,
    HPI_SOURCE_URL,
    LISTING_SOURCE_URL,
    FORECAST_YEARS,
    GOOD_INVESTMENT_THRESHOLD,
    REPORT_DIR,
    REFERENCE_YEAR,
    RANDOM_STATE,
    REGRESSION_FEATURES,
)
from src.data import load_data, model_frame
from src.hpi_forecast import (
    FORECAST_METHODS,
    GLOBAL_FORECAST_METHODS,
    HORIZON_QUARTERS,
    MIN_HISTORY_QUARTERS,
    forecast_global_panel,
    forecast_index,
    forecast_index_safe,
    validate_forecast_endpoint,
)
from src.modeling import classifier_candidates, regressor_candidates

CV_FOLDS = 5
CV_SAMPLE_LIMIT = 10_000
TUNING_SAMPLE_LIMIT = 4_000
TEST_SIZE = 0.20


def _tuning_grid(task, name):
    """Small, declared search spaces; the untouched city holdout stays final."""
    if task == "classification":
        return {
            "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
            "decision_tree": {
                "model__max_depth": [6, 12, None],
                "model__min_samples_leaf": [10, 30],
            },
            "random_forest": {
                "model__n_estimators": [80, 120],
                "model__max_depth": [None, 16],
                "model__min_samples_leaf": [5],
            },
            "extra_trees": {
                "model__n_estimators": [80, 120],
                "model__max_depth": [None, 16],
                "model__min_samples_leaf": [5],
            },
            "gradient_boosting": {
                "model__n_estimators": [80, 120],
                "model__learning_rate": [0.05],
                "model__max_depth": [1, 2],
            },
        }.get(name, {})
    return {
        "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "decision_tree": {
            "model__max_depth": [6, 12, None],
            "model__min_samples_leaf": [10, 30],
        },
        "random_forest": {
            "model__n_estimators": [80, 120],
            "model__max_depth": [None, 16],
            "model__min_samples_leaf": [5],
        },
        "extra_trees": {
            "model__n_estimators": [80, 120],
            "model__max_depth": [None, 16],
            "model__min_samples_leaf": [5],
        },
        "gradient_boosting": {
            "model__n_estimators": [80, 120],
            "model__learning_rate": [0.05],
            "model__max_depth": [1, 2],
        },
    }.get(name, {})


def _tune_selected(task, name, estimator, X, y):
    """Tune the CV-selected family; grouped holdout is reserved for final scoring."""
    X_tune, y_tune = _cv_sample(X, y, TUNING_SAMPLE_LIMIT,
                                classification=(task == "classification"))
    print(f"  Tuning selected {task} model: {name} ({len(X_tune):,} rows)", flush=True)
    folds = GroupKFold(n_splits=CV_FOLDS)
    search = GridSearchCV(
        estimator, _tuning_grid(task, name),
        scoring="f1" if task == "classification" else "neg_mean_absolute_error",
        cv=folds, n_jobs=1, refit=True, return_train_score=True,
        error_score="raise",
    )
    search.fit(X_tune, y_tune, groups=X_tune["City"])
    print(f"  Refitting selected {task} model on all training rows", flush=True)
    final_model = search.best_estimator_.fit(X, y)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.insert(0, "task", task)
    cv_results.insert(1, "model", name)
    return final_model, search, cv_results


def _segment_metrics(task, X, y, predictions):
    """Summarize final holdout errors by supported business segments."""
    frame = pd.DataFrame({
        "City": X["City"].to_numpy(),
        "Property_Type": X["Property_Type"].to_numpy(),
        "actual": np.asarray(y),
        "predicted": np.asarray(predictions),
    })
    rows = []
    for segment in ("City", "Property_Type"):
        for label, part in frame.groupby(segment, dropna=False):
            row = {"task": task, "segment": segment, "value": str(label), "rows": len(part)}
            if task == "classification":
                row.update({
                    "f1": f1_score(part.actual, part.predicted, zero_division=0),
                    "precision": precision_score(part.actual, part.predicted, zero_division=0),
                    "recall": recall_score(part.actual, part.predicted, zero_division=0),
                    "mae": np.nan,
                })
            else:
                row.update({
                    "f1": np.nan, "precision": np.nan, "recall": np.nan,
                    "mae": mean_absolute_error(part.actual, part.predicted),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hpi_backtest_metrics(rows):
    errors = rows["predicted_hpi"] - rows["actual_hpi"]
    mae = float(errors.abs().mean())
    naive_errors = rows["actual_hpi"] - rows["current_hpi"]
    naive_mae = float(naive_errors.abs().mean())
    fallback = rows.get("fallback_used", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
    return {
        "forecast_origins": int(rows["origin"].nunique()),
        "city_origin_forecasts": int(len(rows)),
        "mae_hpi_points": mae,
        "rmse_hpi_points": float(np.sqrt(np.mean(errors**2))),
        "mape_percent": float((errors.abs() / rows["actual_hpi"]).mean() * 100),
        "naive_last_value_mae_hpi_points": naive_mae,
        "mae_skill_vs_last_value": float(1 - mae / naive_mae) if naive_mae > 0 else None,
        "directional_accuracy": float(
            ((rows["predicted_hpi"] >= rows["current_hpi"])
             == (rows["actual_hpi"] >= rows["current_hpi"])).mean()
        ),
        "fallback_count": int(fallback.sum()),
        "fallback_rate_percent": float(fallback.mean() * 100),
    }


def _safe_global_forecast(histories, method):
    """Forecast pooled model with per-city endpoint validation and explicit fallback."""
    try:
        raw = forecast_global_panel(histories, HORIZON_QUARTERS, method)
        results = {}
        for city, history in histories.items():
            try:
                value = validate_forecast_endpoint(history, raw[city], HORIZON_QUARTERS, method)
                results[city] = {"value": value, "fallback_used": False, "fallback_method": None, "fallback_reason": None}
            except Exception as exc:
                results[city] = {"value": float(np.asarray(list(history), dtype=float)[-1]),
                                 "fallback_used": True, "fallback_method": "last_value",
                                 "fallback_reason": f"{type(exc).__name__}: {str(exc)[:240]}"}
        return results
    except Exception as exc:
        reason = f"{type(exc).__name__}: {str(exc)[:240]}"
        return {city: {"value": float(np.asarray(list(history), dtype=float)[-1]),
                       "fallback_used": True, "fallback_method": "last_value",
                       "fallback_reason": reason}
                for city, history in histories.items()}


def _resale_scenario_backtest(selected_rows, anchor_prices=(50.0, 100.0, 200.0)):
    """Scale historical HPI forecasts to hypothetical property prices.

    This evaluates the HPI-to-price conversion and monetary implications of HPI
    forecast errors. It is NOT a validation against observed property resales.
    """
    if selected_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = []
    for row in selected_rows.itertuples(index=False):
        current_hpi = float(row.current_hpi)
        predicted_multiplier = float(row.predicted_hpi) / current_hpi
        reference_multiplier = float(row.actual_hpi) / current_hpi
        fallback_used = bool(getattr(row, "fallback_used", False))
        for anchor in anchor_prices:
            predicted_value = float(anchor) * predicted_multiplier
            reference_value = float(anchor) * reference_multiplier
            baseline_value = float(anchor)  # last-value HPI baseline implies no index change
            out.append({
                "city": row.city,
                "method": row.method,
                "split": row.split,
                "forecast_origin": row.origin,
                "target_quarter": row.target_quarter,
                "illustrative_starting_price_lakhs": float(anchor),
                "current_hpi": current_hpi,
                "predicted_hpi": float(row.predicted_hpi),
                "observed_future_hpi": float(row.actual_hpi),
                "predicted_growth_multiplier": predicted_multiplier,
                "observed_index_growth_multiplier": reference_multiplier,
                "predicted_scenario_value_lakhs": predicted_value,
                "index_implied_reference_value_lakhs": reference_value,
                "signed_error_lakhs": predicted_value - reference_value,
                "absolute_error_lakhs": abs(predicted_value - reference_value),
                "absolute_percentage_error": abs(predicted_value - reference_value) / reference_value * 100,
                "last_value_baseline_value_lakhs": baseline_value,
                "baseline_absolute_error_lakhs": abs(baseline_value - reference_value),
                "fallback_used": fallback_used,
                "interpretation": "Illustrative price-scaled HPI backtest; reference value is index-implied, not an observed property resale price.",
            })
    detail = pd.DataFrame(out)
    metrics = []
    for (split, anchor), part in detail.groupby(["split", "illustrative_starting_price_lakhs"], sort=False):
        err = part["signed_error_lakhs"].to_numpy(float)
        base_err = part["last_value_baseline_value_lakhs"].to_numpy(float) - part["index_implied_reference_value_lakhs"].to_numpy(float)
        mae = float(np.mean(np.abs(err)))
        baseline_mae = float(np.mean(np.abs(base_err)))
        metrics.append({
            "split": split,
            "illustrative_starting_price_lakhs": float(anchor),
            "city_origin_forecasts": int(len(part)),
            "scenario_mae_lakhs": mae,
            "scenario_rmse_lakhs": float(np.sqrt(np.mean(err**2))),
            "scenario_mape_percent": float(part["absolute_percentage_error"].mean()),
            "scenario_bias_lakhs": float(np.mean(err)),
            "scenario_abs_error_p90_lakhs": float(np.quantile(np.abs(err), 0.90)),
            "last_value_baseline_mae_lakhs": baseline_mae,
            "mae_skill_vs_last_value_baseline": float(1 - mae / baseline_mae) if baseline_mae > 0 else None,
            "fallback_rate_percent": float(part["fallback_used"].mean() * 100),
            "interpretation": "Scaled HPI validation only; does not measure error against real property resale transactions.",
        })
    return detail, pd.DataFrame(metrics)


def _train_hpi_forecast():
    """Cross-validate city-index methods, score the temporal holdout, and refit."""
    if not HPI_PATH.exists():
        raise FileNotFoundError(f"Missing HPI data at {HPI_PATH}; run scripts/merge_nhb_hpi.py first.")
    frame = pd.read_csv(HPI_PATH)
    if "City" not in frame.columns:
        raise ValueError("HPI input must contain City and quarter month-year columns")
    period_columns = []
    for column in frame.columns[1:]:
        try:
            period_columns.append((pd.to_datetime(column, format="%b %Y"), column))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid HPI quarter heading {column!r}; expected 'Mon YYYY'") from exc
    period_columns.sort(key=lambda pair: pair[0])
    if len({date for date, _ in period_columns}) != len(period_columns):
        raise ValueError("Duplicate HPI quarter columns found")
    frame = frame.set_index("City")
    if frame.index.has_duplicates:
        raise ValueError("HPI input must have one row per city")
    dates = [date for date, _ in period_columns]
    values_by_city = {}
    for city, row in frame.iterrows():
        values = pd.to_numeric(row[[column for _, column in period_columns]], errors="coerce").to_numpy(float)
        if len(values) < MIN_HISTORY_QUARTERS + HORIZON_QUARTERS:
            continue
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{city}: HPI must be positive and complete for every quarter")
        values_by_city[str(city)] = values
    if not values_by_city:
        raise ValueError("No cities have enough complete history for a five-year HPI backtest")

    last_origin = len(dates) - HORIZON_QUARTERS - 1
    # Use eight consecutive origins for the final temporal holdout.
    # With a 20-quarter horizon, this tests forecasts targeting 2024-2025
    # rather than basing final assessment on only four origin dates.
    holdout_count = 8
    holdout_start = last_origin - holdout_count + 1
    cv_origins = range(MIN_HISTORY_QUARTERS - 1, holdout_start)
    holdout_origins = range(holdout_start, last_origin + 1)
    if not len(cv_origins) or not len(holdout_origins):
        raise ValueError("Not enough HPI quarters for rolling-origin CV and temporal holdout")

    predictions = []
    # Forecast each historical origin using only values available at that date.
    # Unstable candidates use an explicit last-value fallback, which is reported.
    evaluation_origins = list(cv_origins) + list(holdout_origins)
    for origin in evaluation_origins:
        split = "cross_validation" if origin < holdout_start else "final_holdout"
        histories = {city: values[:origin + 1] for city, values in values_by_city.items()}
        forecasts_by_method = {}
        for method in FORECAST_METHODS:
            if method in GLOBAL_FORECAST_METHODS:
                forecasts_by_method[method] = _safe_global_forecast(histories, method)
            else:
                method_results = {}
                for city, history in histories.items():
                    value, fallback_used, fallback_reason = forecast_index_safe(
                        history, HORIZON_QUARTERS, method
                    )
                    method_results[city] = {
                        "value": value,
                        "fallback_used": fallback_used,
                        "fallback_method": "last_value" if fallback_used else None,
                        "fallback_reason": fallback_reason,
                    }
                forecasts_by_method[method] = method_results

        for city, values in sorted(values_by_city.items()):
            history = histories[city]
            for method in FORECAST_METHODS:
                result = forecasts_by_method[method][city]
                predictions.append({
                    "city": city,
                    "method": method,
                    "split": split,
                    "origin": dates[origin].strftime("%b %Y"),
                    "target_quarter": dates[origin + HORIZON_QUARTERS].strftime("%b %Y"),
                    "history_quarters": len(history),
                    "current_hpi": float(history[-1]),
                    "predicted_hpi": float(result["value"]),
                    "actual_hpi": float(values[origin + HORIZON_QUARTERS]),
                    "fallback_used": bool(result["fallback_used"]),
                    "fallback_method": result["fallback_method"],
                    "fallback_reason": result["fallback_reason"],
                })
    prediction_table = pd.DataFrame(predictions)
    cv = prediction_table[prediction_table["split"] == "cross_validation"]
    holdout = prediction_table[prediction_table["split"] == "final_holdout"]
    cv_table = pd.DataFrame([
        {"method": method, **_hpi_backtest_metrics(rows)}
        for method, rows in cv.groupby("method")
    ])
    # A method that falls back too frequently is considered operationally unreliable.
    # It remains in the report, but is excluded from selection when any method
    # with <=5% fallback use has valid CV results.
    reliability_threshold = 5.0
    eligible = cv_table[cv_table["fallback_rate_percent"] <= reliability_threshold]
    selection_pool = eligible if not eligible.empty else cv_table
    cv_table["selection_eligible"] = cv_table["fallback_rate_percent"] <= reliability_threshold
    cv_table = cv_table.sort_values(["mape_percent", "mae_hpi_points"]).reset_index(drop=True)
    selected = str(selection_pool.sort_values(["mape_percent", "mae_hpi_points"]).iloc[0]["method"])
    holdout_selected = holdout[holdout["method"] == selected]
    holdout_metrics = _hpi_backtest_metrics(holdout_selected)

    # Empirical pooled relative-error band from CV forecasts. This is a useful
    # uncertainty indicator, not a formal guarantee for an individual city.
    selected_cv = cv[cv["method"] == selected].copy()
    relative_actual = selected_cv["actual_hpi"] / selected_cv["predicted_hpi"]
    ratio_low, ratio_high = (float(relative_actual.quantile(0.10)),
                             float(relative_actual.quantile(0.90)))

    if selected in GLOBAL_FORECAST_METHODS:
        projected_results = _safe_global_forecast(values_by_city, selected)
    else:
        projected_results = {}
        for city, history in values_by_city.items():
            value, fallback_used, fallback_reason = forecast_index_safe(
                history, HORIZON_QUARTERS, selected
            )
            projected_results[city] = {
                "value": value, "fallback_used": fallback_used,
                "fallback_method": "last_value" if fallback_used else None,
                "fallback_reason": fallback_reason,
            }

    forecasts = []
    for city, values in sorted(values_by_city.items()):
        current = float(values[-1])
        result = projected_results[city]
        projected = float(result["value"])
        lower_hpi = max(1e-6, projected * ratio_low)
        upper_hpi = max(lower_hpi, projected * ratio_high)
        forecasts.append({
            "City": city,
            "Selected_Method": selected,
            "Latest_Quarter": dates[-1].strftime("%b %Y"),
            "Forecast_Quarter": (dates[-1] + pd.DateOffset(months=3 * HORIZON_QUARTERS)).strftime("%b %Y"),
            "Latest_HPI": current,
            "Forecast_HPI_5Y": projected,
            "Forecast_HPI_80_Lower_Empirical": lower_hpi,
            "Forecast_HPI_80_Upper_Empirical": upper_hpi,
            "HPI_Growth_Multiplier_5Y": projected / current,
            "HPI_Growth_Multiplier_80_Lower_Empirical": lower_hpi / current,
            "HPI_Growth_Multiplier_80_Upper_Empirical": upper_hpi / current,
            "Implied_CAGR_Percent": ((projected / current) ** (1 / 5) - 1) * 100,
            "Historical_Quarters_Used": len(values),
            "Forecast_Fallback_Used": bool(result["fallback_used"]),
            "Forecast_Fallback_Method": result["fallback_method"],
            "Forecast_Fallback_Reason": result["fallback_reason"],
        })
    forecast_table = pd.DataFrame(forecasts)

    # Translate index forecast errors into price-scale scenarios at several
    # illustrative starting prices. These are index-implied values, not sale data.
    selected_predictions = prediction_table[prediction_table["method"] == selected].copy()
    scenario_detail, scenario_metrics = _resale_scenario_backtest(selected_predictions)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(ARTIFACT_DIR / "hpi_forecast_by_city.csv", index=False)
    cv_table.to_csv(REPORT_DIR / "hpi_forecast_cv_results.csv", index=False)
    holdout_selected.to_csv(REPORT_DIR / "hpi_forecast_holdout_predictions.csv", index=False)
    prediction_table.to_csv(REPORT_DIR / "hpi_forecast_backtest_predictions.csv", index=False)
    failures = prediction_table[prediction_table["fallback_used"]].copy()
    failures.to_csv(REPORT_DIR / "hpi_forecast_fallbacks.csv", index=False)
    scenario_detail.to_csv(REPORT_DIR / "resale_value_scenario_backtest.csv", index=False)
    scenario_metrics.to_csv(REPORT_DIR / "resale_value_scenario_metrics.csv", index=False)

    summary = {
        "target": "NHB city HPI 20 quarters after forecast origin",
        "selected_method": selected,
        "selection_metric": "lowest mean absolute percentage error across rolling-origin CV",
        "candidates_compared": FORECAST_METHODS,
        "holdout_origins": holdout_count,
        "candidate_reliability": {
            "fallback_policy": "If a candidate fails convergence, positivity, finiteness, or history-based stability checks, replace that city/origin forecast with the last observed HPI and record the reason.",
            "selection_fallback_threshold_percent": reliability_threshold,
            "eligible_candidates": cv_table.loc[cv_table["selection_eligible"], "method"].tolist(),
            "fallbacks_in_final_holdout_for_selected_method": int(holdout_selected.get("fallback_used", pd.Series(False, index=holdout_selected.index)).sum()),
            "fallbacks_in_final_current_forecast": int(forecast_table["Forecast_Fallback_Used"].sum()),
            "fallback_report": "reports/hpi_forecast_fallbacks.csv",
        },
        "resale_value_scenario_validation": {
            "detail_report": "reports/resale_value_scenario_backtest.csv",
            "metrics_report": "reports/resale_value_scenario_metrics.csv",
            "starting_prices_lakhs": [50.0, 100.0, 200.0],
            "method": "Apply predicted and observed five-year city-HPI multipliers to illustrative starting values.",
            "important_limitation": "This validates the monetary scaling of HPI forecast errors. The reference values are index-implied and are not observed property resale prices; it does not establish property-level resale accuracy.",
            "cv_and_holdout_metrics": scenario_metrics.to_dict(orient="records"),
        },
        "empirical_relative_error_band": {
            "nominal_level": "80%",
            "lower_actual_to_prediction_ratio": ratio_low,
            "upper_actual_to_prediction_ratio": ratio_high,
            "derived_from": "selected method's pooled rolling-origin CV forecasts",
            "caveat": "Pooled empirical 10th–90th percentile actual/predicted HPI ratio. This is an indicative uncertainty band, not a formally calibrated prediction interval; city-specific uncertainty may differ.",
        },
        "cross_validation": {
            "strategy": "expanding-window rolling-origin; each forecast uses only HPI values available at its origin",
            "origin_start": dates[MIN_HISTORY_QUARTERS - 1].strftime("%b %Y"),
            "origin_end": dates[holdout_start - 1].strftime("%b %Y"),
            "horizon_quarters": HORIZON_QUARTERS,
            "metrics_by_method": cv_table.to_dict(orient="records"),
            "note": "Five-year windows overlap, so forecast errors are correlated; CV is historical evidence, not a guarantee.",
        },
        "final_holdout": {
            "origin_start": dates[holdout_start].strftime("%b %Y"),
            "origin_end": dates[last_origin].strftime("%b %Y"),
            "metrics": holdout_metrics,
        },
        "deployment": {
            "use": "Apply each city's projected HPI growth multiplier to a listing's asking price.",
            "limitation": "City-level estimate only; no matched property resale outcomes or neighborhood/property effects. Scenario backtests use index-implied reference values, not observed individual resales.",
            "latest_hpi_quarter": dates[-1].strftime("%b %Y"),
            "forecast_quarter": (dates[-1] + pd.DateOffset(months=3 * HORIZON_QUARTERS)).strftime("%b %Y"),
        },
        "source": {
            "file": HPI_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "url": HPI_SOURCE_URL,
            "sha256": _sha256(HPI_PATH),
            "cities": len(values_by_city),
            "quarters": len(dates),
        },
    }
    (REPORT_DIR / "hpi_forecast_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "hpi_forecast_model.json").write_text(json.dumps({
        "method": selected,
        "horizon_quarters": HORIZON_QUARTERS,
        "selection_metric": summary["selection_metric"],
        "cv_mape_percent": float(cv_table.iloc[0]["mape_percent"]),
        "holdout_metrics": holdout_metrics,
        "source_sha256": summary["source"]["sha256"],
    }, indent=2), encoding="utf-8")
    return summary


def _cv_sample(X, y, limit, classification=False):
    """Use a reproducible, stratified sample for affordable model selection."""
    if len(X) <= limit:
        return X, y
    _, X_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=limit,
        random_state=RANDOM_STATE,
        stratify=y if classification else None,
    )
    return X_sample, y_sample


def _group_holdout_split(X, y, groups, test_size=TEST_SIZE):
    """Hold out whole cities so test metrics measure transfer to unseen cities."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _feature_importance(pipeline):
    """Extract comparable absolute importance from tree or linear estimators."""
    estimator = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
    )


def _classification_benchmark(
    X_train, y_train, X_test, y_test, rule_year_train, rule_year_test
):
    X_cv, y_cv = _cv_sample(X_train, y_train, CV_SAMPLE_LIMIT, classification=True)
    folds = GroupKFold(n_splits=CV_FOLDS)
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}
    cv_rows, reference_rows = [], []

    # The label is generated from property age, so record the rule itself as an
    # oracle reference. It is not a fitted candidate and cannot validate returns.
    cv_age = (REFERENCE_YEAR - rule_year_train.loc[X_cv.index]).clip(lower=0).fillna(0)
    cv_rule = (
        ANNUAL_APPRECIATION_ASSUMPTION
        - cv_age * AGE_DEPRECIATION_PER_YEAR
        > GOOD_INVESTMENT_THRESHOLD
    ).astype(int)
    test_age = (REFERENCE_YEAR - rule_year_test).clip(lower=0).fillna(0)
    test_rule = (
        ANNUAL_APPRECIATION_ASSUMPTION
        - test_age * AGE_DEPRECIATION_PER_YEAR
        > GOOD_INVESTMENT_THRESHOLD
    ).astype(int)
    reference_rows.append({
        "reference": "target_rule_oracle",
        "evaluation_split": "selection_sample; not cross-validated",
        "rows": len(X_cv),
        "f1": f1_score(y_cv, cv_rule, zero_division=0),
        "roc_auc": 1.0,
        "accuracy": accuracy_score(y_cv, cv_rule),
    })
    reference_rows.append({
        "reference": "target_rule_oracle",
        "evaluation_split": "held-out cities",
        "rows": len(X_test),
        "f1": f1_score(y_test, test_rule, zero_division=0),
        "roc_auc": 1.0,
        "accuracy": accuracy_score(y_test, test_rule),
    })

    for name, candidate in classifier_candidates().items():
        print(f"  CV classifier: {name}", flush=True)
        result = cross_validate(
            candidate, X_cv, y_cv,
            cv=folds.split(X_cv, y_cv, groups=X_cv["City"]), scoring=scoring,
            n_jobs=1, return_train_score=True, error_score="raise",
        )
        cv_row = {
            "model": name,
            "cv_folds": CV_FOLDS,
            "cv_rows": len(X_cv),
            "cv_city_groups": X_cv["City"].nunique(),
            "cv_strategy": "5-fold GroupKFold by City",
            "cv_f1_mean": result["test_f1"].mean(),
            "cv_f1_std": result["test_f1"].std(ddof=1),
            "cv_train_f1_mean": result["train_f1"].mean(),
            "cv_train_test_f1_gap": result["train_f1"].mean() - result["test_f1"].mean(),
            "cv_roc_auc_mean": result["test_roc_auc"].mean(),
            "cv_roc_auc_std": result["test_roc_auc"].std(ddof=1),
            "cv_accuracy_mean": result["test_accuracy"].mean(),
        }
        cv_rows.append(cv_row)

    cv_table = pd.DataFrame(cv_rows).sort_values(
        ["cv_f1_mean", "cv_roc_auc_mean"], ascending=False
    ).reset_index(drop=True)
    best_name = cv_table.iloc[0]["model"]
    best_model, tuning_search, tuning_results = _tune_selected(
        "classification", best_name, classifier_candidates()[best_name],
        X_train, y_train,
    )
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]
    holdout_table = pd.DataFrame([{
        "model": f"tuned_selected_{best_name}",
        "train_city_groups": X_train["City"].nunique(),
        "test_city_groups": X_test["City"].nunique(),
        "test_accuracy": accuracy_score(y_test, best_pred),
        "test_precision": precision_score(y_test, best_pred, zero_division=0),
        "test_recall": recall_score(y_test, best_pred, zero_division=0),
        "test_f1": f1_score(y_test, best_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, best_prob) if y_test.nunique() > 1 else np.nan,
    }])
    best_metrics = {
        "selected_model": best_name,
        "selection_metric": "mean 5-fold CV F1; ROC AUC tie-break",
        "tuning_metric": "5-fold city-group CV F1 within selected model family",
        "tuned_parameters": tuning_search.best_params_,
        "tuning_best_cv_f1": tuning_search.best_score_,
        "cv_strategy": "5-fold GroupKFold by City",
        "test_strategy": "20% city-group holdout; test cities are excluded from training",
        "cv_rows": len(X_cv),
        "cv_city_groups": X_cv["City"].nunique(),
        "cv_folds": CV_FOLDS,
        "train_city_groups": X_train["City"].nunique(),
        "test_city_groups": X_test["City"].nunique(),
        "holdout": {
            "accuracy": accuracy_score(y_test, best_pred),
            "precision": precision_score(y_test, best_pred, zero_division=0),
            "recall": recall_score(y_test, best_pred, zero_division=0),
            "f1": f1_score(y_test, best_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, best_prob) if y_test.nunique() > 1 else None,
        },
        "classification_report": classification_report(
            y_test, best_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
    }
    cv_table.to_csv(ARTIFACT_DIR / "classification_cv_results.csv", index=False)
    tuning_results.to_csv(ARTIFACT_DIR / "classification_tuning_results.csv", index=False)
    holdout_table.to_csv(ARTIFACT_DIR / "classification_holdout_results.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(
        ARTIFACT_DIR / "classification_reference_results.csv", index=False
    )
    joblib.dump(best_model, ARTIFACT_DIR / "classifier.joblib")
    _feature_importance(best_model).to_csv(
        ARTIFACT_DIR / "classifier_importance.csv", index=False
    )
    _segment_metrics("classification", X_test, y_test, best_pred).to_csv(
        ARTIFACT_DIR / "classification_segment_metrics.csv", index=False
    )
    with mlflow.start_run(run_name=f"selected_classifier_{best_name}"):
        mlflow.log_params({"task": "classification", "selected_model": best_name,
                           "selection_metric": "mean 5-fold CV F1",
                           **{f"best_{key}": str(value) for key, value in tuning_search.best_params_.items()}})
        mlflow.log_metrics({f"holdout_{k}": float(v) for k, v in best_metrics["holdout"].items()})
        # The fitted pipeline is already saved with joblib above. Keep MLflow
        # focused on parameters and metrics; tree models may require explicit
        # skops trust configuration to serialize as MLflow model artifacts.
    return best_metrics


def _regression_benchmark(X_train, y_train, X_test, y_test):
    X_cv, y_cv = _cv_sample(X_train, y_train, CV_SAMPLE_LIMIT)
    folds = GroupKFold(n_splits=CV_FOLDS)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_rows, reference_rows = [], []

    # This deterministic formula is the true baseline because the target itself
    # was generated by compounding today's price at the configured rate.
    formula_pred = X_cv["Price_in_Lakhs"] * (
        1 + ANNUAL_APPRECIATION_ASSUMPTION
    ) ** FORECAST_YEARS
    formula_mae = mean_absolute_error(y_cv, formula_pred)
    formula_rmse = float(np.sqrt(mean_squared_error(y_cv, formula_pred)))
    formula_r2 = r2_score(y_cv, formula_pred)
    reference_rows.append({
        "reference": "known_formula_baseline",
        "evaluation_split": "selection_sample; not cross-validated",
        "rows": len(X_cv),
        "mae_lakhs": formula_mae,
        "rmse_lakhs": formula_rmse,
        "r2": formula_r2,
    })
    formula_test = X_test["Price_in_Lakhs"] * (
        1 + ANNUAL_APPRECIATION_ASSUMPTION
    ) ** FORECAST_YEARS
    reference_rows.append({
        "reference": "known_formula_baseline",
        "evaluation_split": "held-out cities",
        "rows": len(X_test),
        "mae_lakhs": mean_absolute_error(y_test, formula_test),
        "rmse_lakhs": float(np.sqrt(mean_squared_error(y_test, formula_test))),
        "r2": r2_score(y_test, formula_test),
    })

    for name, candidate in regressor_candidates().items():
        print(f"  CV regressor: {name}", flush=True)
        result = cross_validate(
            candidate, X_cv, y_cv,
            cv=folds.split(X_cv, y_cv, groups=X_cv["City"]), scoring=scoring,
            n_jobs=1, return_train_score=True, error_score="raise",
        )
        cv_row = {
            "model": name,
            "cv_folds": CV_FOLDS,
            "cv_rows": len(X_cv),
            "cv_city_groups": X_cv["City"].nunique(),
            "cv_strategy": "5-fold GroupKFold by City",
            "cv_mae_mean": -result["test_mae"].mean(),
            "cv_mae_std": result["test_mae"].std(ddof=1),
            "cv_train_mae_mean": -result["train_mae"].mean(),
            "cv_train_test_mae_gap": -result["test_mae"].mean() + result["train_mae"].mean(),
            "cv_rmse_mean": -result["test_rmse"].mean(),
            "cv_rmse_std": result["test_rmse"].std(ddof=1),
            "cv_r2_mean": result["test_r2"].mean(),
            "cv_r2_std": result["test_r2"].std(ddof=1),
        }
        cv_rows.append(cv_row)

    cv_table = pd.DataFrame(cv_rows).sort_values("cv_mae_mean").reset_index(drop=True)
    best_name = cv_table.iloc[0]["model"]
    best_model, tuning_search, tuning_results = _tune_selected(
        "regression", best_name, regressor_candidates()[best_name],
        X_train, y_train,
    )
    best_pred = best_model.predict(X_test)
    holdout_table = pd.DataFrame([{
        "model": f"tuned_selected_{best_name}",
        "train_city_groups": X_train["City"].nunique(),
        "test_city_groups": X_test["City"].nunique(),
        "test_mae_lakhs": mean_absolute_error(y_test, best_pred),
        "test_rmse_lakhs": float(np.sqrt(mean_squared_error(y_test, best_pred))),
        "test_r2": r2_score(y_test, best_pred),
    }])
    best_metrics = {
        "selected_model": best_name,
        "selection_metric": "lowest mean 5-fold CV MAE",
        "tuning_metric": "5-fold city-group CV MAE within selected model family",
        "tuned_parameters": tuning_search.best_params_,
        "tuning_best_cv_mae_lakhs": -tuning_search.best_score_,
        "cv_strategy": "5-fold GroupKFold by City",
        "test_strategy": "20% city-group holdout; test cities are excluded from training",
        "cv_rows": len(X_cv),
        "cv_city_groups": X_cv["City"].nunique(),
        "cv_folds": CV_FOLDS,
        "train_city_groups": X_train["City"].nunique(),
        "test_city_groups": X_test["City"].nunique(),
        "holdout": {
            "mae_lakhs": mean_absolute_error(y_test, best_pred),
            "rmse_lakhs": float(np.sqrt(mean_squared_error(y_test, best_pred))),
            "r2": r2_score(y_test, best_pred),
        },
        "deployment_predictor": "known_formula_baseline",
        "deployment_note": "The app uses the configured compounding formula because it exactly defines this proxy target; the selected ML candidate is retained as a benchmark only.",
        "formula_baseline_holdout": reference_rows[-1],
    }
    cv_table.to_csv(ARTIFACT_DIR / "regression_cv_results.csv", index=False)
    tuning_results.to_csv(ARTIFACT_DIR / "regression_tuning_results.csv", index=False)
    holdout_table.to_csv(ARTIFACT_DIR / "regression_holdout_results.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(
        ARTIFACT_DIR / "regression_reference_results.csv", index=False
    )
    joblib.dump(best_model, ARTIFACT_DIR / "regression_benchmark.joblib")
    _feature_importance(best_model).to_csv(
        ARTIFACT_DIR / "regressor_importance.csv", index=False
    )
    _segment_metrics("regression", X_test, y_test, best_pred).to_csv(
        ARTIFACT_DIR / "regression_segment_metrics.csv", index=False
    )
    with mlflow.start_run(run_name=f"selected_regressor_{best_name}"):
        mlflow.log_params({"task": "regression", "selected_model": best_name,
                           "selection_metric": "lowest mean 5-fold CV MAE",
                           **{f"best_{key}": str(value) for key, value in tuning_search.best_params_.items()}})
        mlflow.log_metrics({f"holdout_{k}": float(v)
                            for k, v in best_metrics["holdout"].items()})
    return best_metrics


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Training and validating the five-year NHB city HPI forecast...", flush=True)
    hpi_forecast = _train_hpi_forecast()
    print(
        f"  Selected HPI method: {hpi_forecast['selected_method']} "
        f"(CV MAPE {hpi_forecast['cross_validation']['metrics_by_method'][0]['mape_percent']:.1f}%; "
        f"holdout MAPE {hpi_forecast['final_holdout']['metrics']['mape_percent']:.1f}%)",
        flush=True,
    )
    source_df = load_data()
    quality = source_df.attrs.get("quality", {})
    df = model_frame(source_df)
    mlflow.set_experiment("real-estate-investment-advisor")

    cls_df = df.loc[df["Good_Investment"].notna()].copy()
    reg_df = df.loc[df["Price_After_5Y_Proxy"].notna()].copy()
    X_cls, y_cls = cls_df[CLASSIFICATION_FEATURES], cls_df["Good_Investment"].astype(int)
    rule_year = cls_df["Year_Built"]
    X_train, X_test, y_train, y_test = _group_holdout_split(
        X_cls, y_cls, groups=X_cls["City"]
    )
    rule_year_train = rule_year.loc[X_train.index]
    rule_year_test = rule_year.loc[X_test.index]
    print("Comparing classifiers with 5-fold City-grouped CV and a City-group holdout...")
    classification = _classification_benchmark(
        X_train, y_train, X_test, y_test, rule_year_train, rule_year_test
    )

    X_reg, y_reg = reg_df[REGRESSION_FEATURES], reg_df["Price_After_5Y_Proxy"]
    Xr_train, Xr_test, yr_train, yr_test = _group_holdout_split(
        X_reg, y_reg, groups=X_reg["City"]
    )
    print("Comparing regressors with 5-fold City-grouped CV and a City-group holdout...")
    regression = _regression_benchmark(Xr_train, yr_train, Xr_test, yr_test)

    results = {
        "target_warning": (
            "Targets are formula-generated proxies; CV and test scores do not measure "
            "real-world investment or future resale accuracy."
        ),
        "split": {
            "test_size": TEST_SIZE,
            "strategy": "GroupShuffleSplit by City",
            "random_state": RANDOM_STATE,
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "strategy": "GroupKFold by City",
            "max_sample_rows_per_training_task": CV_SAMPLE_LIMIT,
            "note": "CV uses reproducible row samples when required. City groups remain separate across folds; preprocessing is fitted inside each fold. Hyperparameters for the CV-selected model family are tuned using grouped CV on training data; the city holdout remains untouched until final scoring.",
        },
        "data_quality": quality,
        "classification": classification,
        "regression": regression,
        "hpi_forecast": hpi_forecast,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    manifest = {
        "source_file": DATA_PATH.name,
        "source_url": LISTING_SOURCE_URL,
        "source_sha256": _sha256(DATA_PATH),
        "additional_source": {
            "file": HPI_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "url": HPI_SOURCE_URL,
            "sha256": _sha256(HPI_PATH) if HPI_PATH.exists() else None,
            "used_for_training": True,
            "note": "Used for rolling-origin city HPI forecasting; listing-level classifier and proxy-regression targets remain assumption-generated.",
        },
        "cleaned_rows": len(source_df),
        "classification_rows_with_known_year_built": len(cls_df),
        "regression_rows": len(reg_df),
        "cleaning_summary": quality,
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "scipy", "mlflow", "joblib")
        },
        "random_state": RANDOM_STATE,
        "assumptions": {
            "reference_year": REFERENCE_YEAR,
            "annual_appreciation": ANNUAL_APPRECIATION_ASSUMPTION,
            "age_depreciation_per_year": AGE_DEPRECIATION_PER_YEAR,
            "good_investment_threshold": GOOD_INVESTMENT_THRESHOLD,
            "forecast_years": FORECAST_YEARS,
        },
    }
    (ARTIFACT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
