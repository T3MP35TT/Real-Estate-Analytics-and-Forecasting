"""Shared data access, page styling, chart helpers, and formatting for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.theme import apply_theme as _apply_theme, page_header as _page_header, section_header as _section_header

from src.config import ARTIFACT_DIR, DATA_PATH, FORECAST_YEARS, HPI_PATH, REPORT_DIR
from src.data import load_data

ROOT = Path(__file__).resolve().parents[1]

CITY_ALIASES = {
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


@st.cache_data(show_spinner="Loading property listings…")
def get_listings() -> pd.DataFrame:
    """Load the project's canonical cleaned/engineered listing frame."""
    frame = load_data()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The listing dataset is empty or could not be loaded.")
    return frame


@st.cache_data(show_spinner=False)
def get_hpi_forecasts() -> pd.DataFrame:
    path = ARTIFACT_DIR / "hpi_forecast_by_city.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_hpi_history() -> pd.DataFrame:
    """Convert the one-row-per-city wide HPI file to a tidy quarterly table."""
    if not HPI_PATH.exists():
        return pd.DataFrame(columns=["City", "Quarter", "HPI"])
    wide = pd.read_csv(HPI_PATH)
    if "City" not in wide.columns:
        return pd.DataFrame(columns=["City", "Quarter", "HPI"])
    period_cols: list[tuple[str, pd.Timestamp]] = []
    for column in wide.columns:
        if column == "City":
            continue
        date = pd.to_datetime(str(column), format="%b %Y", errors="coerce")
        if pd.notna(date):
            period_cols.append((column, date))
    if not period_cols:
        return pd.DataFrame(columns=["City", "Quarter", "HPI"])
    selected = [column for column, _ in period_cols]
    tidy = wide[["City", *selected]].melt(
        id_vars="City", value_vars=selected, var_name="Quarter_Label", value_name="HPI"
    )
    date_map = {column: date for column, date in period_cols}
    tidy["Quarter"] = tidy["Quarter_Label"].map(date_map)
    tidy["HPI"] = pd.to_numeric(tidy["HPI"], errors="coerce")
    tidy = tidy.dropna(subset=["City", "Quarter", "HPI"])
    return tidy.sort_values(["City", "Quarter"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_report_csv(name: str) -> pd.DataFrame:
    path = REPORT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_artifact_csv(name: str) -> pd.DataFrame:
    path = ARTIFACT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_report_json(name: str = "hpi_forecast_metrics.json") -> dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def apply_page_style(page_name: str | None = None) -> None:
    """Backward-compatible entry point; applies the shared utils.theme styling."""
    _apply_theme(page_name)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a shared page banner using the central theme module."""
    _page_header(eyebrow, title, subtitle)


def section_header(title: str, description: str | None = None) -> None:
    """Render a shared section label using the central theme module."""
    _section_header(title, description)


def money_lakh(value: Any, decimals: int = 1) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "—"
        return f"₹{number:,.{decimals}f} lakh"
    except (TypeError, ValueError):
        return "—"


def city_to_hpi(city: str) -> str:
    """Resolve common listing-city labels to NHB city labels, without geographic guessing."""
    normalized = str(city).strip()
    return CITY_ALIASES.get(normalized.casefold(), normalized)


def find_forecast(city: str, forecasts: pd.DataFrame) -> pd.Series | None:
    if forecasts.empty or "City" not in forecasts.columns:
        return None
    target = city_to_hpi(city).casefold()
    hits = forecasts.loc[forecasts["City"].astype(str).str.strip().str.casefold() == target]
    return None if hits.empty else hits.iloc[0]


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def hpi_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    forecast = metrics.get("hpi_forecast", metrics)
    holdout = forecast.get("final_holdout", {}).get("metrics", {})
    cv_methods = forecast.get("cross_validation", {}).get("metrics_by_method", [])
    return {
        "method": forecast.get("selected_method", "—"),
        "cv_mape": (cv_methods[0].get("mape_percent") if cv_methods else None),
        "holdout_mape": holdout.get("mape_percent"),
        "directional_accuracy": holdout.get("directional_accuracy"),
        "latest_quarter": forecast.get("deployment", {}).get("latest_hpi_quarter", "—"),
        "forecast_quarter": forecast.get("deployment", {}).get("forecast_quarter", "—"),
        "cities": forecast.get("source", {}).get("cities"),
        "quarters": forecast.get("source", {}).get("quarters"),
    }


def safe_export_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def pretty_metric_table(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if columns:
        out = out[[c for c in columns if c in out.columns]]
    return out
