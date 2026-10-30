from __future__ import annotations

import math
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.helper import (
    get_hpi_forecasts,
    get_listings,
    get_report_json,
    hpi_summary,
    page_header,
    section_header,
)
from utils.sidebar import render_sidebar

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Overview | Real Estate Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Push the hero/banner down slightly without changing the sidebar layout.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.65rem !important;
            padding-bottom: 3rem !important;
        }
        .executive-takeaway {
            margin-top: -0.35rem;
            margin-bottom: 1.15rem;
            padding: 0.85rem 1rem;
            border-left: 3px solid #D6AD5C;
            border-radius: 0 9px 9px 0;
            background: rgba(214, 173, 92, 0.09);
            color: #D9E5DF;
            font-size: 0.91rem;
            line-height: 1.55;
        }
        .executive-takeaway b {
            color: #F0C979;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.75rem;
        }
        .insight-card {
            border: 1px solid rgba(214, 173, 92, 0.28);
            border-radius: 12px;
            padding: 0.95rem 1rem;
            background: rgba(17, 35, 27, 0.72);
            min-height: 112px;
        }
        .insight-label {
            color: #AFC6BA;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .insight-value {
            color: #F4F4E9;
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.25;
        }
        .insight-detail {
            color: #AFC6BA;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(18, 35, 28, 0.72);
            border: 1px solid rgba(214, 173, 92, 0.23);
            border-top: 3px solid #D6AD5C;
            border-radius: 12px;
            padding: 0.85rem 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

page_header(
    "EXECUTIVE BRIEF  /  REAL ESTATE ANALYTICS",
    "Real Estate Investment Advisor",
    "A decision-support workspace for exploring listing data, comparing city markets, and stress-testing five-year housing-market scenarios.",
)

# ---------------------------------------------------------------------
# Data loading and reusable helpers
# ---------------------------------------------------------------------
try:
    listings = get_listings()
except Exception as exc:
    st.error(f"Could not load the property dataset: {exc}")
    st.info("Check the DATA_PATH setting in src/config.py and confirm that the source CSV is present.")
    st.stop()

try:
    forecasts = get_hpi_forecasts()
    if forecasts is None:
        forecasts = pd.DataFrame()
except Exception as exc:
    forecasts = pd.DataFrame()
    st.warning(f"City forecast data could not be loaded: {exc}")

try:
    metrics = get_report_json()
    hpi = hpi_summary(metrics)
except Exception:
    hpi = {
        "method": None,
        "cv_mape": None,
        "holdout_mape": None,
        "directional_accuracy": None,
        "latest_quarter": "—",
        "forecast_quarter": "—",
    }

# Keep the dashboard resilient to small differences in source-column naming.
PRICE_COL = "Price_in_Lakhs" if "Price_in_Lakhs" in listings.columns else None
CITY_COL = "City" if "City" in listings.columns else None
STATE_COL = "State" if "State" in listings.columns else None

work = listings.copy()
if PRICE_COL:
    work[PRICE_COL] = pd.to_numeric(work[PRICE_COL], errors="coerce")
    work = work[work[PRICE_COL].gt(0)].copy()

# ---------------------------------------------------------------------
# Global executive filters — every listing-based KPI and chart uses this
# filtered dataset. Forecast validation metrics remain model-wide.
# ---------------------------------------------------------------------
PROPERTY_TYPE_COL = next(
    (c for c in ["Property_Type", "PropertyType", "Property_Category"] if c in work.columns),
    None,
)
BEDROOM_COL = next(
    (c for c in ["BHK", "Bedrooms", "Bedroom", "No_of_Bedrooms"] if c in work.columns),
    None,
)

with st.expander("🔎 Filter executive overview", expanded=True):
    f1, f2, f3 = st.columns(3, gap="medium")

    if STATE_COL:
        state_options = sorted(work[STATE_COL].dropna().astype(str).unique().tolist())
        with f1:
            selected_states = st.multiselect(
                "State / territory",
                options=state_options,
                placeholder="All states",
                key="exec_filter_states",
            )
    else:
        selected_states = []
        with f1:
            st.caption("State filter unavailable — State field not found.")

    state_scope = work.copy()
    if selected_states and STATE_COL:
        state_scope = state_scope[state_scope[STATE_COL].astype(str).isin(selected_states)]

    if CITY_COL:
        city_options = sorted(state_scope[CITY_COL].dropna().astype(str).unique().tolist())
        with f2:
            selected_cities = st.multiselect(
                "City",
                options=city_options,
                placeholder="All cities",
                key="exec_filter_cities",
            )
    else:
        selected_cities = []
        with f2:
            st.caption("City filter unavailable — City field not found.")

    with f3:
        if PRICE_COL and not state_scope.empty:
            valid_prices = state_scope[PRICE_COL].dropna()
            if not valid_prices.empty:
                price_min = float(valid_prices.min())
                price_max = float(valid_prices.max())
                if price_min < price_max:
                    selected_price_range = st.slider(
                        "Asking price range (₹ lakh)",
                        min_value=price_min,
                        max_value=price_max,
                        value=(price_min, price_max),
                        step=max((price_max - price_min) / 200, 0.1),
                        key="exec_filter_price",
                    )
                else:
                    selected_price_range = (price_min, price_max)
                    st.caption(f"Asking price: ₹{price_min:,.1f} lakh")
            else:
                selected_price_range = None
                st.caption("No valid asking prices for the selected geography.")
        else:
            selected_price_range = None
            st.caption("Price filter unavailable — Price_in_Lakhs field not found.")

    f4, f5 = st.columns(2, gap="medium")
    if PROPERTY_TYPE_COL:
        property_options = sorted(work[PROPERTY_TYPE_COL].dropna().astype(str).unique().tolist())
        with f4:
            selected_property_types = st.multiselect(
                "Property type",
                options=property_options,
                placeholder="All property types",
                key="exec_filter_property_type",
            )
    else:
        selected_property_types = []
        with f4:
            st.caption("Property-type filter unavailable in this dataset.")

    if BEDROOM_COL:
        bedroom_options = sorted(work[BEDROOM_COL].dropna().astype(str).unique().tolist())
        with f5:
            selected_bedrooms = st.multiselect(
                "Bedrooms / BHK",
                options=bedroom_options,
                placeholder="All configurations",
                key="exec_filter_bedrooms",
            )
    else:
        selected_bedrooms = []
        with f5:
            st.caption("Bedroom filter unavailable in this dataset.")

# Apply selections once, then use this same frame throughout the page.
if selected_states and STATE_COL:
    work = work[work[STATE_COL].astype(str).isin(selected_states)]
if selected_cities and CITY_COL:
    work = work[work[CITY_COL].astype(str).isin(selected_cities)]
if selected_price_range and PRICE_COL:
    work = work[work[PRICE_COL].between(*selected_price_range)]
if selected_property_types and PROPERTY_TYPE_COL:
    work = work[work[PROPERTY_TYPE_COL].astype(str).isin(selected_property_types)]
if selected_bedrooms and BEDROOM_COL:
    work = work[work[BEDROOM_COL].astype(str).isin(selected_bedrooms)]

render_sidebar(
    "Executive Overview",
    listings=listings,
    filtered_listings=work,
    forecasts=forecasts,
    model_metrics=hpi,
    active_filters={
        "State": selected_states or "All",
        "City": selected_cities or "All",
        "Asking price": selected_price_range or "All",
        "Property type": selected_property_types or "All",
        "Bedrooms / BHK": selected_bedrooms or "All",
    },
)

st.caption(f"Showing {len(work):,} of {len(listings):,} property records after filters.")


def takeaway(message: str) -> None:
    st.markdown(
        f'<div class="executive-takeaway"><b>Executive takeaway</b><br>{message}</div>',
        unsafe_allow_html=True,
    )

def insight_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def rupees_lakh(value: float) -> str:
    return f"₹{value:,.1f} L"

def plotly_theme(fig, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=16, t=18, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D8E4DE", family="Arial"),
        title_font=dict(size=15, color="#F1F4EB"),
        legend_title_text="",
        xaxis=dict(gridcolor="rgba(180,205,190,0.12)", zerolinecolor="rgba(180,205,190,0.18)"),
        yaxis=dict(gridcolor="rgba(180,205,190,0.12)", zerolinecolor="rgba(180,205,190,0.18)"),
    )
    return fig

# ---------------------------------------------------------------------
# Portfolio and market snapshot
# ---------------------------------------------------------------------
n_listings = len(work)
n_cities = work[CITY_COL].nunique() if CITY_COL else 0
n_states = work[STATE_COL].nunique() if STATE_COL else 0
median_price = work[PRICE_COL].median() if PRICE_COL and not work.empty else float("nan")
hpi_cities = (
    forecasts["City"].nunique()
    if not forecasts.empty and "City" in forecasts.columns
    else 0
)

st.markdown("### Portfolio & market snapshot")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Property records", f"{n_listings:,}", help="Records in the supplied listing dataset. Prices represent asking prices.")
k2.metric("Cities covered", f"{n_cities:,}")
k3.metric("States / territories", f"{n_states:,}")
k4.metric("Median asking price", rupees_lakh(median_price) if pd.notna(median_price) else "—")
k5.metric("Cities with HPI forecasts", f"{hpi_cities:,}", help="Cities with usable historical NHB HPI series.")

st.caption(
    "Data lens: listing metrics describe advertised supply and asking prices. "
    "They should not be interpreted as completed transaction prices or realised investment returns."
)

# Derived executive signals from the supplied listing and forecast data.
city_supply = (
    work[CITY_COL].value_counts()
    if CITY_COL and not work.empty
    else pd.Series(dtype="int64")
)
top_supply_city = str(city_supply.index[0]) if not city_supply.empty else "—"
top_supply_share = (
    city_supply.iloc[0] / city_supply.sum() * 100
    if not city_supply.empty and city_supply.sum()
    else float("nan")
)

city_medians = (
    work.dropna(subset=[CITY_COL, PRICE_COL])
    .groupby(CITY_COL)[PRICE_COL].median()
    .sort_values(ascending=False)
    if CITY_COL and PRICE_COL and not work.empty
    else pd.Series(dtype=float)
)
highest_median_city = str(city_medians.index[0]) if not city_medians.empty else "—"
highest_median_value = city_medians.iloc[0] if not city_medians.empty else float("nan")

forecast_view = pd.DataFrame()
if not forecasts.empty and {"City", "HPI_Growth_Multiplier_5Y"}.issubset(forecasts.columns):
    forecast_view = forecasts.copy()
    forecast_view["Projected change (%)"] = (
        pd.to_numeric(forecast_view["HPI_Growth_Multiplier_5Y"], errors="coerce") - 1
    ) * 100
    forecast_view = forecast_view.dropna(subset=["Projected change (%)"])
    forecast_city_scope = selected_cities if selected_cities else (city_options if selected_states and CITY_COL else [])
    if forecast_city_scope:
        forecast_view = forecast_view[forecast_view["City"].astype(str).isin(forecast_city_scope)]

best_forecast_city = (
    str(forecast_view.loc[forecast_view["Projected change (%)"].idxmax(), "City"])
    if not forecast_view.empty else "—"
)
best_forecast_growth = (
    forecast_view["Projected change (%)"].max()
    if not forecast_view.empty else float("nan")
)

s1, s2, s3 = st.columns(3, gap="medium")
with s1:
    insight_card(
        "Largest listing concentration",
        top_supply_city,
        f"{top_supply_share:.1f}% of records in this dataset" if pd.notna(top_supply_share) else "Listing concentration unavailable",
    )
with s2:
    insight_card(
        "Highest city median asking price",
        highest_median_city,
        rupees_lakh(highest_median_value) if pd.notna(highest_median_value) else "Price comparison unavailable",
    )
with s3:
    insight_card(
        "Highest projected HPI growth",
        best_forecast_city,
        f"{best_forecast_growth:+.1f}% over five years · city-level scenario" if pd.notna(best_forecast_growth) else "Forecast comparison unavailable",
    )

st.divider()

# ---------------------------------------------------------------------
# Market intelligence: supply, pricing and geographic mix
# ---------------------------------------------------------------------
section_header(
    "Market intelligence",
    "Understand where the listing supply is concentrated and how advertised prices differ across markets",
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### Listing supply by city")
    if not city_supply.empty:
        supply_chart = city_supply.head(12).rename_axis("City").reset_index(name="Property records")
        fig = px.bar(
            supply_chart.sort_values("Property records"),
            x="Property records",
            y="City",
            orientation="h",
            text="Property records",
            color="Property records",
            color_continuous_scale=["#B9D7D8", "#147D83", "#D6AD5C"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Number of listings")
        st.plotly_chart(plotly_theme(fig, max(340, len(supply_chart) * 28)), use_container_width=True)
        share = top_supply_share
        takeaway(
            f"{top_supply_city} contributes {share:.1f}% of the listings in this dataset. "
            "A high listing count signals greater observed supply in the sample; it does not, by itself, prove stronger demand or better investment returns."
            if pd.notna(share) else
            "The chart ranks cities by the number of supplied listing records. Treat the distribution as dataset coverage, not a measure of transaction volume."
        )
    else:
        st.info("City-level listing counts are unavailable because the City field is missing or empty.")

with right:
    st.markdown("#### Asking-price distribution")
    if PRICE_COL and not work.empty:
        prices = work[PRICE_COL].dropna()
        if not prices.empty:
            upper = prices.quantile(0.99)
            display_prices = prices[prices <= upper]
            fig = px.histogram(
                display_prices.to_frame(name="Asking price (₹ lakh)"),
                x="Asking price (₹ lakh)",
                nbins=35,
                color_discrete_sequence=["#D6AD5C"],
                labels={"count": "Listings"},
            )
            fig.update_layout(xaxis_title="Asking price (₹ lakh)", yaxis_title="Number of listings")
            st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)
            med = prices.median()
            p75 = prices.quantile(0.75)
            takeaway(
                f"The median asking price is {rupees_lakh(med)}, while the 75th percentile is {rupees_lakh(p75)}. "
                "The chart trims only the visual display at the 99th percentile to keep the main distribution readable; KPI calculations use all valid prices."
            )
        else:
            st.info("No valid asking-price values are available.")
    else:
        st.info("Asking-price distribution requires the Price_in_Lakhs field.")

# Median asking price comparison
st.markdown("#### City-level asking-price comparison")
if not city_medians.empty:
    city_price_df = city_medians.rename("Median asking price (₹ lakh)").reset_index()
    city_price_df = city_price_df.sort_values("Median asking price (₹ lakh)", ascending=False).head(12)
    fig = px.bar(
        city_price_df.sort_values("Median asking price (₹ lakh)"),
        x="Median asking price (₹ lakh)",
        y=CITY_COL,
        orientation="h",
        text="Median asking price (₹ lakh)",
        color="Median asking price (₹ lakh)",
        color_continuous_scale=["#B9D7D8", "#147D83", "#D6AD5C"],
    )
    fig.update_traces(texttemplate="₹%{x:.1f}L", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Median asking price (₹ lakh)", yaxis_title="")
    st.plotly_chart(plotly_theme(fig, max(390, len(city_price_df) * 29)), use_container_width=True)
    takeaway(
        f"{highest_median_city} has the highest median asking price among the cities represented ({rupees_lakh(highest_median_value)}). "
        "Use this as a relative pricing benchmark, then compare like-for-like property type, size, configuration and locality before drawing an affordability or value conclusion."
    )
else:
    st.info("City-level price comparison requires both City and Price_in_Lakhs fields.")

# Geographic mix and optional property mix
mix_left, mix_right = st.columns(2, gap="large")
with mix_left:
    st.markdown("#### Listing distribution by state")
    if STATE_COL and not work.empty:
        state_counts = work[STATE_COL].dropna().value_counts().head(10).rename_axis("State").reset_index(name="Listings")
        if not state_counts.empty:
            fig = px.bar(
                state_counts.sort_values("Listings"),
                x="Listings", y="State", orientation="h",
                color_discrete_sequence=["#147D83"],
                text="Listings",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title="Number of listings", yaxis_title="")
            st.plotly_chart(plotly_theme(fig, max(320, len(state_counts) * 27)), use_container_width=True)
            top_state = state_counts.iloc[state_counts["Listings"].argmax()]
            state_share = top_state["Listings"] / max(1, len(work)) * 100
            takeaway(
                f"{top_state['State']} is the largest state-level contributor in the supplied sample ({state_share:.1f}% of valid listing records). "
                "This highlights geographic coverage and concentration, not state-wide market share."
            )
        else:
            st.info("No state values are available.")
    else:
        st.info("State-level distribution is unavailable because the State field is missing.")

with mix_right:
    st.markdown("#### Property profile mix")
    profile_col = next(
        (c for c in ["Property_Type", "PropertyType", "Property_Category", "BHK", "Bedrooms", "Furnishing", "Furnishing_Status"]
         if c in work.columns),
        None,
    )
    if profile_col:
        profile = work[profile_col].fillna("Not specified").astype(str).value_counts().head(10)
        profile_df = profile.rename_axis("Profile").reset_index(name="Listings")
        fig = px.pie(
            profile_df, names="Profile", values="Listings", hole=0.58,
            color_discrete_sequence=["#D6AD5C", "#147D83", "#B9D7D8", "#477B64", "#8FA99A", "#A67C52"],
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)
        leading_profile = profile.index[0]
        leading_share = profile.iloc[0] / max(1, profile.sum()) * 100
        takeaway(
            f"The most frequently represented profile is {leading_profile} ({leading_share:.1f}% of the displayed categories). "
            "This describes the composition of available listings and may reflect dataset collection patterns."
        )
    else:
        st.info(
            "No recognised property-type, bedroom or furnishing field was found in the current listing data. "
            "Add a relevant field to the source dataset to enable this breakdown."
        )

st.divider()

# ---------------------------------------------------------------------
# Five-year outlook
# ---------------------------------------------------------------------
section_header(
    "Five-year city outlook",
    "City-level NHB HPI projections translated into indicative percentage change",
)

if not forecast_view.empty:
    chart = forecast_view.sort_values("Projected change (%)").copy()
    low = pd.to_numeric(
        chart.get("HPI_Growth_Multiplier_80_Lower_Empirical", pd.Series(index=chart.index, dtype=float)),
        errors="coerce",
    )
    high = pd.to_numeric(
        chart.get("HPI_Growth_Multiplier_80_Upper_Empirical", pd.Series(index=chart.index, dtype=float)),
        errors="coerce",
    )
    has_intervals = low.notna().any() and high.notna().any()
    if has_intervals:
        low_change = (low.fillna(chart["HPI_Growth_Multiplier_5Y"]) - 1) * 100
        high_change = (high.fillna(chart["HPI_Growth_Multiplier_5Y"]) - 1) * 100
        chart["Lower error"] = (chart["Projected change (%)"] - low_change).clip(lower=0)
        chart["Upper error"] = (high_change - chart["Projected change (%)"]).clip(lower=0)

    hover_cols = [
        c for c in ["Selected_Method", "Latest_Quarter", "Forecast_Quarter", "Forecast_Fallback_Used"]
        if c in chart.columns
    ]
    fig = px.bar(
        chart,
        x="Projected change (%)",
        y="City",
        orientation="h",
        color="Projected change (%)",
        color_continuous_scale=["#B9D7D8", "#147D83", "#D6AD5C"],
        hover_data=hover_cols,
        labels={"City": "", "Projected change (%)": "Projected city HPI change (%)"},
    )
    if has_intervals:
        fig.update_traces(
            error_x=dict(
                type="data",
                symmetric=False,
                array=chart["Upper error"].tolist(),
                arrayminus=chart["Lower error"].tolist(),
                color="#AFC6BA",
                thickness=1.2,
                width=3,
            )
        )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Projected city HPI change over five years (%)",
        yaxis_title="",
    )
    st.plotly_chart(plotly_theme(fig, max(440, len(chart) * 23)), use_container_width=True)

    low_city_row = chart.iloc[0]
    high_city_row = chart.iloc[-1]
    takeaway(
        f"{high_city_row['City']} has the highest projected five-year HPI change ({high_city_row['Projected change (%)']:+.1f}%), "
        f"while {low_city_row['City']} has the lowest ({low_city_row['Projected change (%)']:+.1f}%). "
        "These are city-index scenarios, not property-specific price targets. Validate locality, property condition, asking-price competitiveness and transaction costs before investment decisions."
    )
else:
    st.info("No usable city forecast table is available. Run the forecasting pipeline to populate this section.")

st.info(
    "**Scenario boundary:** the five-year value applies a city-level NHB HPI growth multiplier to an entered asking price. "
    "It is not trained on matched individual-property resale transactions and excludes locality-level variation, property condition, negotiation, taxes and transaction costs."
)

st.divider()

# ---------------------------------------------------------------------
# Forecast governance and model health
# ---------------------------------------------------------------------
section_header(
    "Forecast model health",
    "Validation metrics and operational checks for the latest generated forecast",
)

method = hpi.get("method") or "Not available"
cv_mape = hpi.get("cv_mape")
holdout_mape = hpi.get("holdout_mape")
directional = hpi.get("directional_accuracy")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected method", str(method).replace("_", " ").title())
m2.metric("Holdout MAPE", f"{holdout_mape:.1f}%" if isinstance(holdout_mape, (int, float)) and math.isfinite(holdout_mape) else "—")
m3.metric("Cross-validation MAPE", f"{cv_mape:.1f}%" if isinstance(cv_mape, (int, float)) and math.isfinite(cv_mape) else "—")
m4.metric("Directional accuracy", f"{directional:.1%}" if isinstance(directional, (int, float)) and math.isfinite(directional) else "—")

if hpi.get("latest_quarter") and hpi.get("latest_quarter") != "—":
    st.caption(
        f"HPI source period: {hpi.get('latest_quarter', '—')}  ·  "
        f"Forecast horizon: {hpi.get('forecast_quarter', '—')}"
    )

if forecasts.empty:
    st.warning("Forecast artifacts are missing. Run `python scripts/train.py` to generate them.")
else:
    fallback_flags = (
        forecasts.get("Forecast_Fallback_Used", pd.Series(False, index=forecasts.index))
        .astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    )
    fallback_count = int(fallback_flags.sum())
    if fallback_count:
        st.warning(
            f"{fallback_count} of {len(forecasts)} city forecast(s) use a fallback method. "
            "Review Forecast Performance and the affected city-level outputs before relying on those scenarios."
        )
    else:
        st.success("No fallback methods were flagged in the current city forecast file.")

takeaway(
    "Use holdout and cross-validation errors to assess historical forecasting performance, and directional accuracy to assess whether the model generally captured the direction of movement. "
    "None of these metrics establishes the accuracy of an individual property's future resale value."
)

st.caption(
    "Decision-support prototype · Not financial advice · Listing prices are asking prices · "
    "Forecasts are modelled scenarios and should be independently assessed."
)
