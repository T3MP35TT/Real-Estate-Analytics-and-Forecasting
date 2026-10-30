from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import streamlit as st

from utils.theme import apply_theme


def _format_lakh(value: Any) -> str:
    try:
        return f"₹{float(value):,.1f} L"
    except (TypeError, ValueError, OverflowError):
        return "—"


def _format_filter(value: Any) -> str:
    if value is None or value == "" or value == [] or value == ():
        return "All"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value)) if value else "All"
    return str(value)


def _compact_row(label: str, value: Any) -> None:
    """Render one stable-height row without Streamlit's nested columns."""
    st.markdown(
        f"""
        <div style="
            display:grid;
            grid-template-columns:minmax(0, 1fr) minmax(0, 1.15fr);
            align-items:center;
            column-gap:10px;
            min-height:27px;
            padding:4px 0;
            border-bottom:1px solid rgba(160,180,165,.13);
            width:100%;
            box-sizing:border-box;">
          <span style="color:#AFC6BA;font-size:.76rem;line-height:1.25;">
            {label}
          </span>
          <span style="color:#E8F0EA;font-size:.78rem;font-weight:700;
                       line-height:1.25;text-align:right;overflow-wrap:anywhere;">
            {value}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_label(label: str) -> None:
    st.markdown(
        f"""
        <div style="color:#D5AD68;font-size:.66rem;font-weight:800;
                    letter-spacing:.11em;text-transform:uppercase;
                    margin:9px 0 3px;line-height:1.3;">
          {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    page_name: str,
    *,
    listings: pd.DataFrame | None = None,
    filtered_listings: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
    model_metrics: Mapping[str, Any] | None = None,
    active_filters: Mapping[str, Any] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Render the shared, page-aware Estate Intelligence sidebar."""
    apply_theme(page_name, sidebar_branding=False)

    # Do not globally compress Streamlit vertical blocks: doing so makes
    # labels and metric rows collide in narrow sidebars.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            margin-top: 0 !important;
            margin-bottom: .25rem !important;
            line-height: 1.35 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: .35rem !important;
            margin-bottom: .35rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            min-height: 38px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if isinstance(filtered_listings, pd.DataFrame):
        view = filtered_listings
    elif isinstance(listings, pd.DataFrame):
        view = listings
    else:
        view = pd.DataFrame()

    forecast_df = forecasts if isinstance(forecasts, pd.DataFrame) else pd.DataFrame()
    model = dict(model_metrics or {})
    filters = dict(active_filters or {})
    extras = dict(extra_metrics or {})

    # One outer card contains the workspace and page-intelligence cards.
    with st.sidebar.container(border=True):
        st.markdown(
            """
            <div style="font-size:.94rem;font-weight:800;line-height:1.4;
                        color:#E8F0EA;margin:2px 0 3px;">
              🏡 Estate Intelligence
            </div>
            <div style="font-size:.72rem;line-height:1.4;color:#AFC6BA;
                        margin-bottom:10px;">
              Property analytics · Market outlook
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="font-size:.66rem;font-weight:700;letter-spacing:.10em;
                            color:#AFC6BA;margin-bottom:7px;">
                  CURRENT WORKSPACE
                </div>
                <div style="font-size:.88rem;font-weight:750;color:#E8F0EA;
                            line-height:1.4;">
                  {page_name}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("Decision-support prototype · Not financial advice")

        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size:.66rem;font-weight:700;letter-spacing:.10em;
                            color:#AFC6BA;margin-bottom:8px;">
                  PAGE INTELLIGENCE
                </div>
                """,
                unsafe_allow_html=True,
            )

            _section_label("Market snapshot")
            _compact_row("Listings in view", f"{len(view):,}")
            city_count = view["City"].nunique() if not view.empty and "City" in view.columns else 0
            _compact_row("Cities in view", f"{city_count:,}")

            if not view.empty and "Price_in_Lakhs" in view.columns:
                prices = pd.to_numeric(view["Price_in_Lakhs"], errors="coerce").dropna()
                median_price = prices.median() if not prices.empty else None
                _compact_row("Median asking price", _format_lakh(median_price))
            else:
                _compact_row("Median asking price", "—")

            if not view.empty and {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(view.columns):
                price = pd.to_numeric(view["Price_in_Lakhs"], errors="coerce")
                area = pd.to_numeric(view["Size_in_SqFt"], errors="coerce")
                rates = (price * 100000 / area.where(area > 0)).dropna()
                rate_text = f"₹{rates.median():,.0f}/sq ft" if not rates.empty else "—"
                _compact_row("Median asking rate", rate_text)
            else:
                _compact_row("Median asking rate", "—")

            if filters:
                with st.expander(
                    f"ACTIVE FILTERS  ·  {len(filters)} groups",
                    expanded=False,
                ):
                    for label, value in filters.items():
                        st.markdown(
                            f"""
                            <div style="display:grid;
                                        grid-template-columns:minmax(0, .8fr) minmax(0, 1.5fr);
                                        gap:10px;padding:6px 0;
                                        border-bottom:1px solid rgba(160,180,165,.14);">
                              <span style="color:#AFC6BA;font-size:.76rem;">{label}</span>
                              <span style="color:#E8F0EA;font-size:.76rem;font-weight:650;
                                           text-align:right;overflow-wrap:anywhere;">
                                {_format_filter(value)}
                              </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if not forecast_df.empty or model:
                _section_label("Forecast monitor")
                if not forecast_df.empty and "City" in forecast_df.columns:
                    _compact_row("Cities with forecasts", f"{forecast_df['City'].nunique():,}")

                method = model.get("method")
                if method and str(method) not in {"—", "None"}:
                    _compact_row("Selected model", str(method).replace("_", " ").title())

                for key, label in (
                    ("holdout_mape", "Holdout MAPE"),
                    ("cv_mape", "CV MAPE"),
                ):
                    metric_value = model.get(key)
                    if isinstance(metric_value, (int, float)) and pd.notna(metric_value):
                        _compact_row(label, f"{metric_value:.1f}%")

                directional = model.get("directional_accuracy")
                if isinstance(directional, (int, float)) and pd.notna(directional):
                    _compact_row("Directional accuracy", f"{directional:.1%}")

                latest = model.get("latest_quarter")
                target = model.get("forecast_quarter")
                if latest or target:
                    _compact_row("HPI horizon", f"{latest or '—'} → {target or '—'}")

            if extras:
                _section_label("Page-specific insights")
                for label, metric_value in extras.items():
                    _compact_row(str(label), str(metric_value))

        st.caption(
            "Listing metrics follow the current page filters. HPI projections "
            "are city-level scenarios, not property-specific resale guarantees."
        )
