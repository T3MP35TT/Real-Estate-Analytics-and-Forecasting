"""Smoke-test all univariate and pooled HPI forecasting candidates."""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hpi_forecast import (
    FORECAST_METHODS,
    GLOBAL_FORECAST_METHODS,
    HORIZON_QUARTERS,
    MIN_HISTORY_QUARTERS,
    forecast_global_panel,
    forecast_index,
    forecast_index_safe,
    validate_forecast_endpoint,
    ForecastInstabilityError,
)

histories = {}
source_label = "synthetic fallback"
candidates = [
    ROOT / "data" / "processed" / "nhb_hpi_combined.csv",
    Path(__file__).resolve().parents[2] / "delhi_hpi_analysis" / "delhi_hpi_quarterly_clean.csv",
]
for candidate in candidates:
    if not candidate.exists():
        continue
    frame = pd.read_csv(candidate)
    if "City" in frame.columns:
        period_columns = []
        for col in frame.columns[1:]:
            try:
                pd.to_datetime(col, format="%b %Y")
                period_columns.append(col)
            except (TypeError, ValueError):
                pass
        if period_columns:
            for _, row in frame.iterrows():
                values = pd.to_numeric(row[period_columns], errors="coerce").to_numpy(float)
                if len(values) >= MIN_HISTORY_QUARTERS and np.isfinite(values).all() and (values > 0).all():
                    histories[str(row["City"])] = values
            if histories:
                source_label = str(candidate)
                break
    if "composite_hpi" in frame.columns:
        values = pd.to_numeric(frame["composite_hpi"], errors="coerce").dropna().to_numpy(float)
        if len(values) >= MIN_HISTORY_QUARTERS and (values > 0).all():
            histories["Delhi"] = values
            source_label = str(candidate)
            break

if not histories:
    x = np.arange(51)
    histories = {f"Illustrative City {i}": 100 * np.exp(0.012 * x + 0.02 * np.sin((x + i) * 2 * np.pi / 4)) for i in range(1, 6)}

reference_city = next(iter(histories))
print(f"Source: {source_label}")
print(f"Testing {len(FORECAST_METHODS)} methods across {len(histories)} city histories; representative univariate city: {reference_city}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for method in FORECAST_METHODS:
        if method in GLOBAL_FORECAST_METHODS:
            predictions = forecast_global_panel(histories, HORIZON_QUARTERS, method)
            assert set(predictions) == set(histories), method
            assert all(np.isfinite(value) and value > 0 for value in predictions.values()), method
            print(f"PASS {method:37s} cities={len(predictions)}")
        else:
            prediction, fallback_used, reason = forecast_index_safe(
                histories[reference_city], HORIZON_QUARTERS, method
            )
            assert np.isscalar(prediction) and np.isfinite(prediction) and prediction > 0, method
            if fallback_used:
                print(f"SAFE FALLBACK {method:29s} -> last_value | {reason}")
            else:
                print(f"PASS {method:37s} endpoint={prediction:.3f}")
# The stability guard must reject an explosive endpoint instead of returning a
# huge but finite number that can contaminate model selection.
try:
    validate_forecast_endpoint(histories[reference_city], 1e12, HORIZON_QUARTERS, "guard-test")
except ForecastInstabilityError:
    print("PASS explosive-forecast guard rejects unrealistic endpoint")
else:
    raise AssertionError("Explosive endpoint unexpectedly passed the stability guard")

print("All HPI candidate smoke tests passed (unstable candidates must report a safe fallback).")
