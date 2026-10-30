from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.helper import (
    get_hpi_forecasts,
    get_hpi_history,
    get_listings,
    page_header,
    section_header,
)
from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="Market Insights | Real Estate Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add breathing room above the title tile without changing the shared theme.
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2.65rem !important;
        padding-bottom: 2rem;
      }
      div[data-testid="stMetric"] {
        background: rgba(20, 38, 31, 0.72);
        border: 1px solid rgba(228, 166, 75, 0.30);
        border-top: 3px solid #e4a64b;
        border-radius: 13px;
        padding: 15px 17px;
      }
      .insight-note {
        background: rgba(20, 45, 37, 0.78);
        border-left: 3px solid #e4a64b;
        border-radius: 8px;
        padding: 11px 14px;
        margin: 7px 0 20px 0;
        color: #e7efeb;
        font-size: 0.92rem;
      }
      .scope-note {
        background: rgba(30, 61, 73, 0.72);
        border: 1px solid rgba(110, 160, 174, 0.24);
        border-radius: 10px;
        padding: 12px 15px;
        margin: 8px 0 18px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

page_header(
    "MARKET MONITORING  /  LISTING & PRICE INTELLIGENCE",
    "Market insights",
    "Understand where listings are concentrated, how asking prices vary across markets, and how historical NHB housing indices inform the five-year outlook.",
)

try:
    listings = get_listings()
except Exception as exc:
    st.error(f"Could not load the listing dataset: {exc}")
    st.info("Check the DATA_PATH setting in src/config.py and confirm that the source CSV is present.")
    st.stop()

forecasts = get_hpi_forecasts()
history = get_hpi_history()

required = {"City", "Price_in_Lakhs"}
missing = required.difference(listings.columns)
if missing:
    st.error(f"The listing dataset is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df = listings.copy()
df["City"] = df["City"].astype(str)
df["Price_in_Lakhs"] = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")
if "State" in df:
    df["State"] = df["State"].fillna("Unknown").astype(str)
if "Property_Type" in df:
    df["Property_Type"] = df["Property_Type"].fillna("Unknown").astype(str)
if "Size_in_SqFt" in df:
    df["Size_in_SqFt"] = pd.to_numeric(df["Size_in_SqFt"], errors="coerce")

# ---------------------------------------------------------------------
# Global filters
# ---------------------------------------------------------------------
section_header("Market filters", "Selections below update the listing KPIs and all listing-based charts.")
with st.expander("Refine the market view", expanded=True):
    f1, f2, f3 = st.columns(3)
    state_values = sorted(df["State"].dropna().unique().tolist()) if "State" in df else []
    selected_states = f1.multiselect(
        "State / territory",
        state_values,
        default=state_values,
        key="market_states",
    )

    state_scope = df[df["State"].isin(selected_states)] if "State" in df and selected_states else df.copy()
    city_values = sorted(state_scope["City"].dropna().unique().tolist())
    selected_cities = f2.multiselect(
        "City",
        city_values,
        default=city_values,
        key="market_cities",
    )

    type_values = sorted(state_scope["Property_Type"].dropna().unique().tolist()) if "Property_Type" in state_scope else []
    selected_types = f3.multiselect(
        "Property type",
        type_values,
        default=type_values,
        key="market_property_types",
    )

    f4, f5, f6 = st.columns(3)
    bhk_values = sorted(state_scope["BHK"].dropna().unique().tolist()) if "BHK" in state_scope else []
    selected_bhk = f4.multiselect(
        "Bedrooms / BHK",
        bhk_values,
        default=bhk_values,
        key="market_bhk",
    )

    valid_prices = state_scope["Price_in_Lakhs"].dropna()
    if not valid_prices.empty:
        price_min, price_max = float(valid_prices.min()), float(valid_prices.max())
        if price_min == price_max:
            price_range = (price_min, price_max)
            f5.caption(f"Asking-price range: ₹{price_min:,.1f} L")
        else:
            price_range = f5.slider(
                "Asking price (₹ lakh)",
                min_value=float(price_min),
                max_value=float(price_max),
                value=(float(price_min), float(price_max)),
                key="market_price_range",
            )
    else:
        price_range = (0.0, 0.0)
        f5.caption("No valid asking prices in this state selection.")

    f6.caption("Clear a multi-select to include all values in that dimension.")

filtered = state_scope.copy()
if selected_cities:
    filtered = filtered[filtered["City"].isin(selected_cities)]
if selected_types and "Property_Type" in filtered:
    filtered = filtered[filtered["Property_Type"].isin(selected_types)]
if selected_bhk and "BHK" in filtered:
    filtered = filtered[filtered["BHK"].isin(selected_bhk)]
if not valid_prices.empty:
    filtered = filtered[
        filtered["Price_in_Lakhs"].between(price_range[0], price_range[1], inclusive="both")
    ]

# Empty multiselects are interpreted as all values within the selected state scope.
if not selected_cities:
    filtered = state_scope.copy()
if not selected_types and "Property_Type" in filtered:
    filtered = filtered[filtered["Property_Type"].isin(type_values)]
if not selected_bhk and "BHK" in filtered:
    filtered = filtered[filtered["BHK"].isin(bhk_values)]

if filtered.empty:
    st.warning("No listings match the selected filters. Widen the price range or adjust the state, city, property-type or BHK selections.")
    st.stop()

render_sidebar(
    "Market Insights",
    listings=df,
    filtered_listings=filtered,
    forecasts=forecasts,
    active_filters={
        "State": selected_states or "All",
        "City": selected_cities or "All",
        "Property type": selected_types or "All",
        "BHK": selected_bhk or "All",
        "Asking price": f"₹{price_range[0]:,.1f}–₹{price_range[1]:,.1f} L",
    },
)

price = filtered["Price_in_Lakhs"].dropna()
area = (
    pd.to_numeric(filtered["Size_in_SqFt"], errors="coerce")
    if "Size_in_SqFt" in filtered
    else pd.Series(dtype=float)
)
valid_area = area[area > 0]
price_per_sqft = (
    filtered["Price_in_Lakhs"] * 100000 / area.where(area > 0)
    if "Size_in_SqFt" in filtered
    else pd.Series(dtype=float)
)

# ---------------------------------------------------------------------
# Executive market snapshot
# ---------------------------------------------------------------------
section_header("Market snapshot", "A filtered view of the supplied property listings.")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Listings in view", f"{len(filtered):,}", help="Number of listings matching the current filters.")
m2.metric("Cities covered", f"{filtered['City'].nunique():,}")
m3.metric("Median asking price", f"₹{price.median():,.1f} L" if not price.empty else "—")
m4.metric("Median area", f"{valid_area.median():,.0f} sq ft" if not valid_area.empty else "—")
m5.metric(
    "Median asking rate",
    f"₹{price_per_sqft.median():,.0f}/sq ft" if price_per_sqft.notna().any() else "—",
)

st.markdown(
    '<div class="scope-note"><b>Interpretation:</b> asking prices are seller/listing expectations, not verified transaction prices. '
    'Price-per-square-foot comparisons are descriptive and can be affected by location, property age, amenities and listing mix.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# City-level market positioning
# ---------------------------------------------------------------------
st.divider()
section_header(
    "City market positioning",
    "Compare listing supply and median asking prices to identify markets that merit deeper investigation.",
)
city_stats = (
    filtered.dropna(subset=["Price_in_Lakhs"])
    .groupby("City", as_index=False)
    .agg(
        listings=("Price_in_Lakhs", "size"),
        median_price_lakhs=("Price_in_Lakhs", "median"),
        q25_price_lakhs=("Price_in_Lakhs", lambda s: s.quantile(0.25)),
        q75_price_lakhs=("Price_in_Lakhs", lambda s: s.quantile(0.75)),
    )
    .sort_values("median_price_lakhs", ascending=True)
)

if not city_stats.empty:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Median asking price by city**")
        fig = px.bar(
            city_stats.tail(20),
            x="median_price_lakhs",
            y="City",
            orientation="h",
            color="median_price_lakhs",
            color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"],
            hover_data={
                "listings": ":,",
                "q25_price_lakhs": ":.1f",
                "q75_price_lakhs": ":.1f",
                "median_price_lakhs": ":.1f",
            },
            labels={"City": "", "median_price_lakhs": "Median asking price (₹ lakh)"},
        )
        fig.update_layout(height=max(390, 25 * min(20, len(city_stats))), coloraxis_showscale=False,
                          margin=dict(l=5, r=10, t=12, b=10))
        st.plotly_chart(fig, use_container_width=True)
        highest = city_stats.iloc[-1]
        lowest = city_stats.iloc[0]
        st.markdown(
            f'<div class="insight-note"><b>Key takeaway:</b> {highest["City"]} has the highest median asking price '
            f'(₹{highest["median_price_lakhs"]:,.1f} L) in this filtered view, while {lowest["City"]} has the lowest '
            f'(₹{lowest["median_price_lakhs"]:,.1f} L). This is a price-positioning comparison, not a value-for-money ranking.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Listing supply by city**")
        supply = city_stats.sort_values("listings", ascending=False).head(20).sort_values("listings")
        fig = px.bar(
            supply, x="listings", y="City", orientation="h",
            color="listings", color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"],
            labels={"listings": "Number of listings", "City": ""},
        )
        fig.update_layout(height=max(390, 25 * min(20, len(supply))), coloraxis_showscale=False,
                          margin=dict(l=5, r=10, t=12, b=10))
        st.plotly_chart(fig, use_container_width=True)
        leader = city_stats.sort_values("listings", ascending=False).iloc[0]
        total = int(city_stats["listings"].sum())
        share = leader["listings"] / total * 100 if total else 0
        st.markdown(
            f'<div class="insight-note"><b>Key takeaway:</b> {leader["City"]} contributes {share:.1f}% of listings '
            f'in the selected market view. A high listing count indicates greater observed supply in this dataset, '
            f'not necessarily stronger demand.</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No valid asking-price observations are available for the selected filters.")

# ---------------------------------------------------------------------
# Price distribution and property characteristics
# ---------------------------------------------------------------------
st.divider()
section_header(
    "Pricing and property characteristics",
    "Understand the spread of asking prices and how listed property size relates to price.",
)
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Asking-price distribution**")
    price_dist = filtered.dropna(subset=["Price_in_Lakhs"]).copy()
    if not price_dist.empty:
        color_col = "Property_Type" if "Property_Type" in price_dist else None
        fig = px.histogram(
            price_dist, x="Price_in_Lakhs", color=color_col, marginal="box",
            nbins=45, barmode="overlay", opacity=0.72,
            color_discrete_sequence=["#147d83", "#e4a64b", "#446c9b", "#8b7ab8", "#d97565"],
            labels={"Price_in_Lakhs": "Asking price (₹ lakh)", "count": "Listings"},
        )
        fig.update_layout(height=420, legend_title_text="Property type", margin=dict(l=5, r=5, t=12, b=10))
        st.plotly_chart(fig, use_container_width=True)
        median_price = price_dist["Price_in_Lakhs"].median()
        q75 = price_dist["Price_in_Lakhs"].quantile(0.75)
        st.markdown(
            f'<div class="insight-note"><b>Key takeaway:</b> the median asking price is ₹{median_price:,.1f} L; '
            f'75% of observed listings are priced at or below ₹{q75:,.1f} L. The distribution helps reveal the dominant '
            f'price bands and whether a small number of premium listings stretch the upper tail.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No valid asking-price values are available.")
with right:
    st.markdown("**Asking price versus property area**")
    if "Size_in_SqFt" in filtered:
        scatter = filtered[["Size_in_SqFt", "Price_in_Lakhs", "City"]].copy()
        if "Property_Type" in filtered:
            scatter["Property_Type"] = filtered["Property_Type"]
        scatter = scatter.dropna(subset=["Size_in_SqFt", "Price_in_Lakhs"])
        scatter = scatter[(scatter["Size_in_SqFt"] > 0) & (scatter["Price_in_Lakhs"] > 0)]
        if len(scatter) > 5000:
            scatter = scatter.sample(5000, random_state=42)
        if not scatter.empty:
            fig = px.scatter(
                scatter, x="Size_in_SqFt", y="Price_in_Lakhs",
                color="Property_Type" if "Property_Type" in scatter else None,
                hover_data=["City"], opacity=0.55,
                color_discrete_sequence=["#147d83", "#e4a64b", "#446c9b", "#8b7ab8", "#d97565"],
                labels={"Size_in_SqFt": "Area (sq ft)", "Price_in_Lakhs": "Asking price (₹ lakh)"},
            )
            fig.update_layout(height=420, margin=dict(l=5, r=5, t=12, b=10))
            st.plotly_chart(fig, use_container_width=True)
            corr = scatter[["Size_in_SqFt", "Price_in_Lakhs"]].corr().iloc[0, 1] if len(scatter) > 1 else np.nan
            corr_text = f"{corr:.2f}" if pd.notna(corr) else "not available"
            st.markdown(
                f'<div class="insight-note"><b>Key takeaway:</b> the observed size–asking-price correlation is {corr_text}. '
                f'This describes association in the selected listings only; it does not isolate the effect of area from '
                f'city, property type, age or other attributes.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Not enough valid area and price observations for the scatterplot.")
    else:
        st.info("The dataset does not contain Size_in_SqFt, so the area comparison is unavailable.")

# ---------------------------------------------------------------------
# Rate and segment comparison
# ---------------------------------------------------------------------
st.divider()
section_header(
    "Property segment comparison",
    "Compare asking rates per square foot across property types where area data is available.",
)
if "Property_Type" in filtered and "Size_in_SqFt" in filtered:
    ppsf = filtered[["Property_Type", "Price_in_Lakhs", "Size_in_SqFt"]].copy()
    ppsf["Asking price per sq ft (₹)"] = (
        pd.to_numeric(ppsf["Price_in_Lakhs"], errors="coerce") * 100000
        / pd.to_numeric(ppsf["Size_in_SqFt"], errors="coerce").where(
            pd.to_numeric(ppsf["Size_in_SqFt"], errors="coerce") > 0
        )
    )
    ppsf = ppsf.dropna(subset=["Asking price per sq ft (₹)"])
    if not ppsf.empty:
        fig = px.box(
            ppsf, x="Property_Type", y="Asking price per sq ft (₹)",
            color="Property_Type", points=False,
            color_discrete_sequence=["#147d83", "#e4a64b", "#446c9b", "#8b7ab8", "#d97565"],
        )
        fig.update_layout(showlegend=False, height=430, xaxis_title="Property type",
                          margin=dict(l=5, r=5, t=12, b=10))
        st.plotly_chart(fig, use_container_width=True)
        segment = ppsf.groupby("Property_Type")["Asking price per sq ft (₹)"].median().sort_values(ascending=False)
        top_segment = segment.index[0]
        bottom_segment = segment.index[-1]
        st.markdown(
            f'<div class="insight-note"><b>Key takeaway:</b> {top_segment} has the highest median asking rate '
            f'(₹{segment.iloc[0]:,.0f}/sq ft), while {bottom_segment} has the lowest '
            f'(₹{segment.iloc[-1]:,.0f}/sq ft) in this filtered sample. Compare like-for-like locations and property '
            f'attributes before interpreting this as a segment premium.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No valid price-per-square-foot observations are available.")
else:
    st.info("Property type and size fields are required for this segment comparison.")

# ---------------------------------------------------------------------
# Historical NHB HPI and forecast endpoint
# ---------------------------------------------------------------------
st.divider()
section_header(
    "Historical NHB housing price index",
    "Quarterly city index observations used by the forecasting workflow, with the five-year forecast endpoint where available.",
)
if history.empty:
    st.info("Historical HPI series not available. Confirm the configured HPI_PATH and run the merge script.")
elif "City" not in history or "Quarter" not in history or "HPI" not in history:
    st.warning("The HPI history file does not contain the expected City, Quarter and HPI fields.")
else:
    history_cities = sorted(history["City"].dropna().astype(str).unique())
    forecast_cities = set(forecasts["City"].dropna().astype(str)) if not forecasts.empty and "City" in forecasts else set()
    selected_history_cities = [c for c in history_cities if c in set(filtered["City"].astype(str))]
    options = selected_history_cities or history_cities
    default_city = options[0]
    hpi_city = st.selectbox("HPI city", options, index=0, key="market_hpi_city")
    series = history.loc[history["City"].astype(str) == hpi_city].copy()
    series["Quarter"] = pd.to_datetime(series["Quarter"], errors="coerce")
    series["HPI"] = pd.to_numeric(series["HPI"], errors="coerce")
    series = series.dropna(subset=["Quarter", "HPI"]).sort_values("Quarter")

    forecast_row = None
    if not forecasts.empty and "City" in forecasts:
        matched = forecasts.loc[forecasts["City"].astype(str).str.casefold() == hpi_city.casefold()]
        if not matched.empty:
            forecast_row = matched.iloc[0]

    if not series.empty:
        fig = px.line(
            series, x="Quarter", y="HPI", markers=True,
            labels={"Quarter": "Quarter", "HPI": "Housing Price Index"},
        )
        fig.update_traces(line=dict(color="#147d83", width=3), marker=dict(size=5), name="Historical HPI")
        if forecast_row is not None and {"Forecast_Quarter", "Latest_HPI", "Forecast_HPI_5Y"}.issubset(forecast_row.index):
            latest_date = series["Quarter"].max()
            target_date = pd.to_datetime(str(forecast_row["Forecast_Quarter"]), format="%b %Y", errors="coerce")
            latest_hpi = pd.to_numeric(pd.Series([forecast_row["Latest_HPI"]]), errors="coerce").iloc[0]
            target_hpi = pd.to_numeric(pd.Series([forecast_row["Forecast_HPI_5Y"]]), errors="coerce").iloc[0]
            if pd.notna(target_date) and pd.notna(latest_hpi) and pd.notna(target_hpi):
                fig.add_scatter(
                    x=[latest_date, target_date], y=[latest_hpi, target_hpi],
                    mode="lines+markers", name="Latest-to-forecast endpoint",
                    line=dict(color="#e4a64b", width=3, dash="dash"),
                )
        fig.update_layout(height=430, margin=dict(l=5, r=5, t=12, b=10), legend_title_text="Series")
        st.plotly_chart(fig, use_container_width=True)
        first_hpi, last_hpi = series.iloc[0]["HPI"], series.iloc[-1]["HPI"]
        hist_change = (last_hpi / first_hpi - 1) * 100 if first_hpi else np.nan
        forecast_change = None
        if forecast_row is not None:
            multiplier = pd.to_numeric(
                pd.Series([forecast_row.get("HPI_Growth_Multiplier_5Y", np.nan)]), errors="coerce"
            ).iloc[0]
            if pd.notna(multiplier):
                forecast_change = (multiplier - 1) * 100
        forecast_text = (
            f"The model's five-year city HPI scenario indicates {forecast_change:+.1f}% projected change."
            if forecast_change is not None else
            "A five-year forecast endpoint is not available for this city."
        )
        st.markdown(
            f'<div class="insight-note"><b>Key takeaway:</b> {hpi_city} historical HPI changed {hist_change:+.1f}% '
            f'between the first and latest available observations. {forecast_text} The dashed segment joins endpoints only; '
            f'it is not a quarterly forecast path or an individual-property return estimate.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"No valid quarterly HPI observations are available for {hpi_city}.")

# ---------------------------------------------------------------------
# Decision-use notes
# ---------------------------------------------------------------------
st.divider()
section_header("How to use these insights", "A practical interpretation guide for investment screening.")
guide1, guide2, guide3 = st.columns(3)
with guide1:
    st.markdown(
        "**1. Screen the market**\n\nUse supply concentration and median asking prices to shortlist cities for further research."
    )
with guide2:
    st.markdown(
        "**2. Compare like with like**\n\nUse property type, BHK, area and asking-rate filters to avoid misleading comparisons across different listing mixes."
    )
with guide3:
    st.markdown(
        "**3. Treat forecasts as scenarios**\n\nCity HPI projections are broad market indicators. They do not establish the resale value or profitability of a specific property."
    )
