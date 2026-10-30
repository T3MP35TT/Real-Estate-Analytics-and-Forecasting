"""Reusable helpers for the Streamlit real-estate dashboard.

This module contains dashboard data-access, formatting, and lookup helpers.
Visual styling itself belongs in ``utils/theme.py``; this module only calls
that shared theme API when a page needs to apply it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR, HPI_PATH, REPORT_DIR
from src.data import load_data
from utils.theme import (
    format_inr_lakh,
    page_header as _page_header,
    section_header as _section_header,
)

ROOT = Path(__file__).resolve().parents[1]

# Known label differences between listing records and NHB RESIDEX city names.
# Unknown names are returned unchanged: this helper deliberately does not
# guess a proxy city for a location without an HPI forecast.
CITY_ALIASES: dict[str, str] = {
    "bangalore": "Bengaluru",
    "gurgaon": "Gurugram",
    "trivandrum": "Thiruvananthapuram",
    "vishakhapatnam": "Vizag",
    "visakhapatnam": "Vizag",
    "new delhi": "Delhi",
    "dwarka": "Delhi",
    "mangalore": "Mangaluru",
    "mysore": "Mysuru",
}


def _empty_hpi_history() -> pd.DataFrame:
    """Return the standard empty shape used by HPI-history charts."""
    return pd.DataFrame(columns=["City", "Quarter", "HPI"])


@st.cache_data(show_spinner="Loading property listings…")
def get_listings() -> pd.DataFrame:
    """Load the canonical listing frame provided by ``src.data``.

    Raises a clear error when the data loader returns an empty or invalid
    object, so the calling page can show a useful warning rather than
    presenting misleading zero-valued KPIs.
    """
    frame = load_data()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("The listing data loader did not return a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The listing dataset is empty or could not be loaded.")
    return frame


@st.cache_data(show_spinner=False)
def get_hpi_forecasts() -> pd.DataFrame:
    """Load the generated city-level HPI forecast artifact, if it exists."""
    path = ARTIFACT_DIR / "hpi_forecast_by_city.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_hpi_history() -> pd.DataFrame:
    """Convert the wide, one-row-per-city HPI CSV into tidy quarterly rows.

    Expected input shape: a ``City`` column plus date-like columns such as
    ``Jun 2013`` or ``Dec 2025``. Non-date columns are ignored.
    """
    if not HPI_PATH.exists():
        return _empty_hpi_history()

    try:
        wide = pd.read_csv(HPI_PATH)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return _empty_hpi_history()

    if wide.empty or "City" not in wide.columns:
        return _empty_hpi_history()

    period_cols: list[tuple[str, pd.Timestamp]] = []
    for column in wide.columns:
        if column == "City":
            continue
        date = pd.to_datetime(str(column), format="%b %Y", errors="coerce")
        if pd.isna(date):
            # Allow other unambiguous date labels without treating arbitrary
            # metadata columns as time periods.
            date = pd.to_datetime(str(column), errors="coerce")
        if pd.notna(date):
            period_cols.append((column, pd.Timestamp(date)))

    if not period_cols:
        return _empty_hpi_history()

    selected_columns = [column for column, _ in period_cols]
    tidy = wide[["City", *selected_columns]].melt(
        id_vars="City",
        value_vars=selected_columns,
        var_name="Quarter_Label",
        value_name="HPI",
    )
    date_map = {column: date for column, date in period_cols}
    tidy["Quarter"] = tidy["Quarter_Label"].map(date_map)
    tidy["HPI"] = pd.to_numeric(tidy["HPI"], errors="coerce")
    tidy["City"] = tidy["City"].astype("string").str.strip()
    tidy = tidy.dropna(subset=["City", "Quarter", "HPI"])
    tidy = tidy[tidy["City"] != ""]
    return tidy[["City", "Quarter", "HPI"]].sort_values(
        ["City", "Quarter"]
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_report_csv(name: str) -> pd.DataFrame:
    """Load a CSV from ``reports/``; return an empty frame if unavailable."""
    path = REPORT_DIR / Path(name).name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_artifact_csv(name: str) -> pd.DataFrame:
    """Load a CSV from ``artifacts/``; return an empty frame if unavailable."""
    path = ARTIFACT_DIR / Path(name).name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_report_json(name: str = "hpi_forecast_metrics.json") -> dict[str, Any]:
    """Load a JSON report from ``reports/``; return an empty dict if missing."""
    path = REPORT_DIR / Path(name).name
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}




def page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    *,
    icon: str | None = None,
) -> None:
    """Render the shared page banner from ``utils/theme.py``."""
    _page_header(eyebrow, title, subtitle, icon=icon)


def section_header(title: str, description: str | None = None) -> None:
    """Render the shared section heading from ``utils/theme.py``."""
    _section_header(title, description)


def money_lakh(value: Any, decimals: int = 1) -> str:
    """Format an amount in Indian rupee lakhs; missing values become an em dash."""
    return format_inr_lakh(value, decimals=decimals)


def city_to_hpi(city: str) -> str:
    """Map common listing labels to NHB city labels without proxy-city guessing."""
    normalized = str(city).strip()
    return CITY_ALIASES.get(normalized.casefold(), normalized)


def find_forecast(city: str, forecasts: pd.DataFrame) -> pd.Series | None:
    """Return the forecast row for a matched city, or ``None`` if unsupported."""
    if forecasts.empty or "City" not in forecasts.columns:
        return None

    target = city_to_hpi(city).casefold()
    city_values = forecasts["City"].astype("string").str.strip().str.casefold()
    hits = forecasts.loc[city_values == target]
    return None if hits.empty else hits.iloc[0]


def finite_number(value: Any) -> float | None:
    """Convert a value to a finite float; return ``None`` for invalid values."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def hpi_summary(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Extract stable headline metrics from the HPI forecast report JSON."""
    empty_summary: dict[str, Any] = {
        "method": "—",
        "cv_mape": None,
        "holdout_mape": None,
        "directional_accuracy": None,
        "latest_quarter": "—",
        "forecast_quarter": "—",
        "cities": None,
        "quarters": None,
    }
    if not isinstance(metrics, dict) or not metrics:
        return empty_summary

    forecast = metrics.get("hpi_forecast", metrics)
    if not isinstance(forecast, dict):
        return empty_summary

    holdout_block = forecast.get("final_holdout", {})
    holdout = holdout_block.get("metrics", {}) if isinstance(holdout_block, dict) else {}
    if not isinstance(holdout, dict):
        holdout = {}

    cv_block = forecast.get("cross_validation", {})
    cv_methods = cv_block.get("metrics_by_method", []) if isinstance(cv_block, dict) else []
    if not isinstance(cv_methods, list):
        cv_methods = []

    selected_method = forecast.get("selected_method", "—")
    selected_cv = next(
        (
            row for row in cv_methods
            if isinstance(row, dict)
            and str(row.get("method", "")).casefold() == str(selected_method).casefold()
        ),
        None,
    )
    if selected_cv is None:
        selected_cv = next((row for row in cv_methods if isinstance(row, dict)), {})

    deployment = forecast.get("deployment", {})
    source = forecast.get("source", {})
    if not isinstance(deployment, dict):
        deployment = {}
    if not isinstance(source, dict):
        source = {}

    return {
        "method": selected_method,
        "cv_mape": selected_cv.get("mape_percent"),
        "holdout_mape": holdout.get("mape_percent"),
        "directional_accuracy": holdout.get("directional_accuracy"),
        "latest_quarter": deployment.get("latest_hpi_quarter", "—"),
        "forecast_quarter": deployment.get("forecast_quarter", "—"),
        "cities": source.get("cities"),
        "quarters": source.get("quarters"),
    }


def safe_export_csv(frame: pd.DataFrame) -> bytes:
    """Return CSV bytes with a UTF-8 BOM for reliable spreadsheet opening."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("safe_export_csv expects a pandas DataFrame.")
    return frame.to_csv(index=False).encode("utf-8-sig")


def pretty_metric_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of a metrics table, optionally restricted to existing columns."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if columns is None:
        return frame.copy()
    selected = [column for column in columns if column in frame.columns]
    return frame.loc[:, selected].copy()
