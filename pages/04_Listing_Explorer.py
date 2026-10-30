from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.helper import get_listings, page_header, safe_export_csv, section_header
from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="Listing Explorer | Real Estate Intelligence",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """<style>
    .block-container {
        padding-top: 2.65rem !important;
        padding-bottom: 2.4rem !important;
    }
    .listing-takeaway {
        border-left: 3px solid #D6AD5C;
        background: rgba(214,173,92,.08);
        border-radius: 0 8px 8px 0;
        padding: .65rem .85rem;
        margin: .25rem 0 1rem 0;
        color: #D9E5DF;
        line-height: 1.5;
        font-size: .88rem;
    }
    </style>""",
    unsafe_allow_html=True,
)

page_header(
    "LISTING ANALYTICS",
    "Listing explorer",
    "Explore the housing-listing inventory by geography, property configuration, asking price, size, amenities and availability. Compare market segments and export the filtered records.",
)
st.caption(
    "All price measures are source-file asking prices, not registered transaction values. "
    "Listing counts describe this dataset's coverage and should not be interpreted as market-wide supply."
)

try:
    df = get_listings().copy()
except Exception as exc:
    st.error(f"Could not load the listing dataset: {exc}")
    st.stop()

required = ["State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs"]
missing = [column for column in required if column not in df.columns]
if missing:
    st.error(f"The listing dataset is missing required columns: {', '.join(missing)}")
    st.stop()

# Normalize the fields used in filters and calculations.
for column in ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Age_of_Property"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

for column in ["State", "City", "Property_Type"]:
    df[column] = df[column].astype("string").str.strip()

df["Price_per_SqFt_Calculated"] = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"].where(df["Size_in_SqFt"] > 0)
)

# Build dependent filter options from the selected geography.
section_header(
    "1. Search and filter listings",
    "Filters are applied to the listing table, summary metrics and every chart below. Clear a multi-select to include all values in that dimension."
)

with st.container(border=True):
    f1, f2, f3 = st.columns(3)
    state_options = sorted(df["State"].dropna().unique().tolist())
    selected_states = f1.multiselect(
        "State / territory",
        state_options,
        default=state_options,
        key="listing_states_v2",
    )
    state_scope = df[df["State"].isin(selected_states)] if selected_states else df
    city_options = sorted(state_scope["City"].dropna().unique().tolist())
    selected_cities = f2.multiselect(
        "City",
        city_options,
        default=city_options,
        key="listing_cities_v2",
    )
    geo_scope = state_scope[state_scope["City"].isin(selected_cities)] if selected_cities else state_scope
    type_options = sorted(geo_scope["Property_Type"].dropna().unique().tolist())
    selected_types = f3.multiselect(
        "Property type",
        type_options,
        default=type_options,
        key="listing_types_v2",
    )

    f4, f5, f6 = st.columns(3)
    bhk_options = sorted(geo_scope["BHK"].dropna().unique().tolist())
    selected_bhk = f4.multiselect(
        "Bedrooms / BHK",
        bhk_options,
        default=bhk_options,
        format_func=lambda value: f"{int(value)} BHK" if float(value).is_integer() else str(value),
        key="listing_bhk_v2",
    )

    locality_options = (
        sorted(geo_scope["Locality"].dropna().astype(str).unique().tolist())
        if "Locality" in geo_scope.columns else []
    )
    selected_localities = f5.multiselect(
        "Locality",
        locality_options,
        default=locality_options,
        key="listing_localities_v2",
        disabled=not locality_options,
    )

    furnishing_options = (
        sorted(geo_scope["Furnished_Status"].dropna().astype(str).unique().tolist())
        if "Furnished_Status" in geo_scope.columns else []
    )
    selected_furnishing = f6.multiselect(
        "Furnishing",
        furnishing_options,
        default=furnishing_options,
        key="listing_furnishing_v2",
        disabled=not furnishing_options,
    )

    f7, f8 = st.columns(2)
    availability_options = (
        sorted(geo_scope["Availability_Status"].dropna().astype(str).unique().tolist())
        if "Availability_Status" in geo_scope.columns else []
    )
    selected_availability = f7.multiselect(
        "Availability status",
        availability_options,
        default=availability_options,
        key="listing_availability_v2",
        disabled=not availability_options,
    )
    if "Owner_Type" in geo_scope.columns:
        owner_options = sorted(geo_scope["Owner_Type"].dropna().astype(str).unique().tolist())
        selected_owner = f8.multiselect(
            "Owner type",
            owner_options,
            default=owner_options,
            key="listing_owner_v2",
        )
    else:
        selected_owner = []

    # Numeric range filters use the current geographic scope, then apply together.
    n1, n2 = st.columns(2)
    area_values = geo_scope["Size_in_SqFt"].dropna()
    price_values = geo_scope["Price_in_Lakhs"].dropna()

    if not area_values.empty:
        area_min, area_max = float(area_values.min()), float(area_values.max())
        area_range = n1.slider(
            "Built-up area (sq ft)",
            min_value=float(np.floor(area_min)),
            max_value=float(np.ceil(area_max)),
            value=(float(np.floor(area_min)), float(np.ceil(area_max))),
            step=max(1.0, round((area_max - area_min) / 500, 1)),
            key="listing_area_range_v2",
        )
    else:
        area_range = (0.0, 0.0)
        n1.caption("Area range unavailable")

    if not price_values.empty:
        price_min, price_max = float(price_values.min()), float(price_values.max())
        price_range = n2.slider(
            "Asking price (₹ lakh)",
            min_value=float(np.floor(price_min * 10) / 10),
            max_value=float(np.ceil(price_max * 10) / 10),
            value=(float(np.floor(price_min * 10) / 10), float(np.ceil(price_max * 10) / 10)),
            step=max(0.1, round((price_max - price_min) / 500, 1)),
            key="listing_price_range_v2",
        )
    else:
        price_range = (0.0, 0.0)
        n2.caption("Asking-price range unavailable")

# Apply all selected filters. Empty multi-selects mean "All", not "no records".
filtered = geo_scope.copy()
if selected_types:
    filtered = filtered[filtered["Property_Type"].isin(selected_types)]
if selected_bhk:
    filtered = filtered[filtered["BHK"].isin(selected_bhk)]
if selected_localities and "Locality" in filtered.columns:
    filtered = filtered[filtered["Locality"].astype(str).isin(selected_localities)]
if selected_furnishing and "Furnished_Status" in filtered.columns:
    filtered = filtered[filtered["Furnished_Status"].astype(str).isin(selected_furnishing)]
if selected_availability and "Availability_Status" in filtered.columns:
    filtered = filtered[filtered["Availability_Status"].astype(str).isin(selected_availability)]
if selected_owner and "Owner_Type" in filtered.columns:
    filtered = filtered[filtered["Owner_Type"].astype(str).isin(selected_owner)]
filtered = filtered[
    filtered["Size_in_SqFt"].between(*area_range)
    & filtered["Price_in_Lakhs"].between(*price_range)
].copy()

render_sidebar(
    "Listing Explorer",
    listings=df,
    filtered_listings=filtered,
    active_filters={
        "State": selected_states or "All",
        "City": selected_cities or "All",
        "Property type": selected_types or "All",
        "BHK": selected_bhk or "All",
        "Locality": selected_localities or "All",
        "Furnishing": selected_furnishing or "All",
        "Availability": selected_availability or "All",
        "Owner type": selected_owner or "All",
        "Area": f"{area_range[0]:,.0f}–{area_range[1]:,.0f} sq ft",
        "Asking price": f"₹{price_range[0]:,.1f}–₹{price_range[1]:,.1f} L",
    },
)

if filtered.empty:
    st.warning("No listings match the current filters. Widen one or more selections.")
    st.stop()

# Derived analytical measures.
price = filtered["Price_in_Lakhs"]
area = filtered["Size_in_SqFt"]
rate = filtered["Price_per_SqFt_Calculated"]
median_price = float(price.median())
mean_price = float(price.mean())
median_area = float(area.median())
median_rate = float(rate.median()) if rate.notna().any() else np.nan
city_count = filtered["City"].nunique()
locality_count = filtered["Locality"].nunique() if "Locality" in filtered.columns else np.nan

section_header(
    "2. Filtered portfolio snapshot",
    "The figures below recalculate from the currently selected listings."
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Matching listings", f"{len(filtered):,}")
k2.metric("Median asking price", f"₹{median_price:,.1f} L")
k3.metric("Median asking rate", f"₹{median_rate:,.0f}/sq ft" if np.isfinite(median_rate) else "—")
k4.metric("Median built-up area", f"{median_area:,.0f} sq ft")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Cities represented", f"{city_count:,}")
k6.metric("Localities represented", f"{int(locality_count):,}" if pd.notna(locality_count) else "—")
k7.metric("Average asking price", f"₹{mean_price:,.1f} L")
k8.metric("Price 25th–75th percentile", f"₹{price.quantile(.25):,.1f}–{price.quantile(.75):,.1f} L")

# Segment comparisons.
st.divider()
section_header(
    "3. Market and segment comparisons",
    "Compare the filtered market across geography and property configuration. Median asking price is used to reduce sensitivity to unusually high-priced listings."
)

left, right = st.columns(2, gap="large")
with left:
    city_summary = (
        filtered.groupby("City", as_index=False)
        .agg(
            Listings=("Price_in_Lakhs", "count"),
            Median_Price_Lakh=("Price_in_Lakhs", "median"),
            Median_Rate=("Price_per_SqFt_Calculated", "median"),
            Median_Area=("Size_in_SqFt", "median"),
        )
        .sort_values("Median_Price_Lakh")
    )
    city_chart = (
        city_summary.nlargest(12, "Listings")
        .sort_values("Median_Price_Lakh")
        .copy()
    )
    fig = px.bar(
        city_chart,
        x="Median_Price_Lakh",
        y="City",
        orientation="h",
        color="Median_Price_Lakh",
        color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"],
        hover_data={
            "Listings": ":,",
            "Median_Price_Lakh": ":.1f",
            "Median_Rate": ":,.0f",
            "Median_Area": ":,.0f",
        },
        labels={
            "Median_Price_Lakh": "Median asking price (₹ lakh)",
            "Median_Rate": "Median rate (₹/sq ft)",
            "Median_Area": "Median area (sq ft)",
            "City": "",
        },
        title="Median asking price · 12 cities with most listings",
    )
    fig.update_layout(
        height=340,
        coloraxis_showscale=False,
        margin=dict(l=5, r=8, t=42, b=8),
        yaxis=dict(title="", tickfont=dict(size=10)),
        xaxis=dict(title="Median asking price (₹ lakh)", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)
    highest = city_summary.sort_values("Median_Price_Lakh", ascending=False).iloc[0]
    lowest = city_summary.sort_values("Median_Price_Lakh").iloc[0]
    st.markdown(
        f'<div class="listing-takeaway"><b>Key takeaway:</b> {highest["City"]} has the highest '
        f'median asking price (₹{highest["Median_Price_Lakh"]:,.1f} L) and {lowest["City"]} '
        f'the lowest (₹{lowest["Median_Price_Lakh"]:,.1f} L) in this filtered sample. '
        f'Compare listing counts before interpreting a city with a small sample.</div>',
        unsafe_allow_html=True,
    )

with right:
    type_summary = (
        filtered.groupby("Property_Type", as_index=False)
        .agg(
            Listings=("Price_in_Lakhs", "count"),
            Median_Price_Lakh=("Price_in_Lakhs", "median"),
            Median_Rate=("Price_per_SqFt_Calculated", "median"),
        )
        .sort_values("Median_Price_Lakh", ascending=False)
    )
    fig = px.bar(
        type_summary,
        x="Property_Type",
        y="Median_Price_Lakh",
        color="Property_Type",
        text="Listings",
        hover_data={"Median_Rate": ":,.0f", "Median_Price_Lakh": ":.1f"},
        color_discrete_sequence=["#147d83", "#e4a64b", "#446c9b", "#8b7ab8", "#d97565"],
        labels={
            "Property_Type": "Property type",
            "Median_Price_Lakh": "Median asking price (₹ lakh)",
            "Median_Rate": "Median rate (₹/sq ft)",
        },
        title="Asking price by property type",
    )
    fig.update_layout(height=330, showlegend=False, margin=dict(l=5, r=5, t=42, b=8))
    st.plotly_chart(fig, use_container_width=True)
    leader = type_summary.iloc[0]
    st.markdown(
        f'<div class="listing-takeaway"><b>Key takeaway:</b> {leader["Property_Type"]} has the '
        f'highest median asking price (₹{leader["Median_Price_Lakh"]:,.1f} L) among the selected '
        f'property types. The labels show the number of listings in each segment.</div>',
        unsafe_allow_html=True,
    )

# Price and rate distributions.
st.divider()
section_header(
    "4. Asking-price and rate distribution",
    "Use the distributions to understand the spread of listing prices and per-square-foot rates, rather than relying on a single average."
)
dist_left, dist_right = st.columns(2, gap="large")

with dist_left:
    fig = px.histogram(
        filtered.dropna(subset=["Price_in_Lakhs"]),
        x="Price_in_Lakhs",
        nbins=35,
        marginal="box",
        title="Asking-price distribution",
        labels={"Price_in_Lakhs": "Asking price (₹ lakh)", "count": "Listings"},
        opacity=0.82,
    )
    fig.add_vline(x=median_price, line_dash="dash", line_width=2,
                  annotation_text=f"Median ₹{median_price:,.1f} L")
    fig.update_layout(height=330, margin=dict(l=5, r=8, t=42, b=8))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div class="listing-takeaway"><b>Key takeaway:</b> Half of the filtered listings are '
        f'at or below ₹{median_price:,.1f} L and half are at or above it. The interquartile '
        f'range is ₹{price.quantile(.25):,.1f}–₹{price.quantile(.75):,.1f} L.</div>',
        unsafe_allow_html=True,
    )

with dist_right:
    rate_data = filtered.dropna(subset=["Price_per_SqFt_Calculated"])
    fig = px.histogram(
        rate_data,
        x="Price_per_SqFt_Calculated",
        nbins=35,
        marginal="box",
        title="Asking rate per square foot",
        labels={"Price_per_SqFt_Calculated": "Asking rate (₹/sq ft)", "count": "Listings"},
        opacity=0.82,
    )
    if not rate_data.empty:
        fig.add_vline(x=median_rate, line_dash="dash", line_width=2,
                      annotation_text=f"Median ₹{median_rate:,.0f}/sq ft")
    fig.update_layout(height=330, margin=dict(l=5, r=8, t=42, b=8))
    st.plotly_chart(fig, use_container_width=True)
    if not rate_data.empty:
        st.markdown(
            f'<div class="listing-takeaway"><b>Key takeaway:</b> the median asking rate is '
            f'₹{median_rate:,.0f}/sq ft. A large spread can reflect differences in location, '
            f'property type, size and listing mix; it should not be read as a like-for-like valuation.</div>',
            unsafe_allow_html=True,
        )

# Configuration mix and city market map.
st.divider()
section_header(
    "5. Housing configuration and city positioning",
    "Review the mix of bedroom configurations and the relationship between city-level price, rate and listing count."
)
mix_col, scatter_col = st.columns(2, gap="large")

with mix_col:
    bhk_summary = (
        filtered.dropna(subset=["BHK"])
        .groupby("BHK", as_index=False)
        .size()
        .rename(columns={"size": "Listings"})
        .sort_values("BHK")
    )
    bhk_summary["Configuration"] = bhk_summary["BHK"].map(
        lambda value: f"{int(value)} BHK" if float(value).is_integer() else f"{value:g} BHK"
    )
    fig = px.bar(
        bhk_summary,
        x="Configuration",
        y="Listings",
        color="Listings",
        color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"],
        title="Listing mix by BHK configuration",
        text="Listings",
    )
    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=5, r=5, t=42, b=8))
    st.plotly_chart(fig, use_container_width=True)
    dominant = bhk_summary.sort_values("Listings", ascending=False).iloc[0]
    st.markdown(
        f'<div class="listing-takeaway"><b>Key takeaway:</b> {dominant["Configuration"]} '
        f'properties form the largest segment with {int(dominant["Listings"]):,} listings '
        f'({dominant["Listings"] / len(filtered):.1%} of the filtered sample). This is the '
        f'composition of the dataset, not proof of buyer demand.</div>',
        unsafe_allow_html=True,
    )

with scatter_col:
    city_plot = city_summary.dropna(subset=["Median_Rate", "Median_Price_Lakh"]).copy()
    if not city_plot.empty:
        fig = px.scatter(
            city_plot,
            x="Median_Rate",
            y="Median_Price_Lakh",
            size="Listings",
            color="City",
            hover_name="City",
            hover_data={"Listings": ":,", "Median_Area": ":,.0f"},
            title="City price positioning",
            labels={
                "Median_Rate": "Median asking rate (₹/sq ft)",
                "Median_Price_Lakh": "Median asking price (₹ lakh)",
                "Listings": "Listing count",
            },
        )
        fig.update_layout(height=320, margin=dict(l=5, r=5, t=42, b=8), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="listing-takeaway"><b>Key takeaway:</b> cities further right have higher '
            'median asking rates; cities higher up have higher median ticket prices. Bubble size '
            'represents the number of listings in the filtered sample.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("City positioning needs at least one city with valid asking-price and area data.")

# Locality-level view, if the source includes locality information.
if "Locality" in filtered.columns:
    st.divider()
    section_header(
        "6. Locality-level opportunities",
        "Identify localities with the highest and lowest observed asking rates in the selected sample. These are listing benchmarks, not investment recommendations."
    )
    locality_data = filtered.dropna(subset=["Locality", "Price_per_SqFt_Calculated"]).copy()
    locality_summary = (
        locality_data.groupby("Locality", as_index=False)
        .agg(
            Listings=("Price_in_Lakhs", "count"),
            Median_Rate=("Price_per_SqFt_Calculated", "median"),
            Median_Price=("Price_in_Lakhs", "median"),
        )
    )
    min_sample = st.slider(
        "Minimum listings per locality",
        min_value=1,
        max_value=max(1, min(30, int(locality_summary["Listings"].max()) if not locality_summary.empty else 1)),
        value=3 if not locality_summary.empty and locality_summary["Listings"].max() >= 3 else 1,
        key="locality_min_sample_v2",
    )
    locality_view = locality_summary[locality_summary["Listings"] >= min_sample]
    if not locality_view.empty:
        top_localities = locality_view.sort_values("Median_Rate", ascending=False).head(12)
        fig = px.bar(
            top_localities.sort_values("Median_Rate"),
            x="Median_Rate",
            y="Locality",
            orientation="h",
            color="Median_Rate",
            color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"],
            hover_data={"Listings": ":,", "Median_Price": ":.1f"},
            title="Highest median asking rates by locality",
            labels={"Median_Rate": "Median asking rate (₹/sq ft)", "Locality": ""},
        )
        fig.update_layout(height=340, coloraxis_showscale=False,
                          margin=dict(l=5, r=5, t=42, b=8))
        st.plotly_chart(fig, use_container_width=True)
        top_locality = locality_view.sort_values("Median_Rate", ascending=False).iloc[0]
        st.markdown(
            f'<div class="listing-takeaway"><b>Key takeaway:</b> {top_locality["Locality"]} '
            f'has the highest median asking rate (₹{top_locality["Median_Rate"]:,.0f}/sq ft) '
            f'among localities meeting the {min_sample}-listing minimum.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No localities meet the selected minimum listing count.")

# Property age and amenity/availability profiles, only when source fields support them.
if "Year_Built" in filtered.columns:
    age_data = filtered.dropna(subset=["Year_Built", "Price_in_Lakhs"]).copy()
    if not age_data.empty:
        current_year = pd.Timestamp.today().year
        age_data["Property_Age"] = (current_year - age_data["Year_Built"]).clip(lower=0)
        st.divider()
        section_header(
            "7. Property age and asking price",
            "Inspect how asking prices vary across the age profile of the listings. The relationship is descriptive and affected by location, size and property mix."
        )
        age_bins = [-1, 5, 10, 15, 20, 30, np.inf]
        age_labels = ["0–5 years", "6–10 years", "11–15 years", "16–20 years", "21–30 years", "31+ years"]
        age_data["Age_Band"] = pd.cut(
            age_data["Property_Age"],
            bins=age_bins,
            labels=age_labels,
            ordered=True,
        )
        age_plot_data = age_data.dropna(subset=["Age_Band"]).copy()
        # Sampling keeps the interactive chart responsive while the summary below
        # is calculated from the complete filtered dataset.
        if len(age_plot_data) > 15000:
            age_plot_data = age_plot_data.sample(15000, random_state=42)
        fig = px.box(
            age_plot_data,
            x="Age_Band",
            y="Price_in_Lakhs",
            color="Age_Band",
            points=False,
            title="Asking-price spread by property age band",
            labels={
                "Age_Band": "Property age",
                "Price_in_Lakhs": "Asking price (₹ lakh)",
            },
            category_orders={"Age_Band": age_labels},
        )
        fig.update_layout(
            height=330,
            showlegend=False,
            margin=dict(l=5, r=8, t=42, b=8),
            xaxis=dict(title="Property age band", tickfont=dict(size=10)),
            yaxis=dict(title="Asking price (₹ lakh)", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)
        median_age = float(age_data["Property_Age"].median())
        band_summary = (
            age_data.groupby("Age_Band", observed=False)["Price_in_Lakhs"]
            .median()
            .dropna()
        )
        if not band_summary.empty:
            lowest_band = band_summary.idxmin()
            highest_band = band_summary.idxmax()
            takeaway = (
                f"The median age is {median_age:.0f} years. Among age bands, "
                f"{lowest_band} has the lowest median asking price "
                f"(₹{band_summary.min():,.1f} L), while {highest_band} has the highest "
                f"(₹{band_summary.max():,.1f} L). Differences may reflect location and property mix."
            )
        else:
            takeaway = f"The median age in this filtered sample is {median_age:.0f} years."
        st.markdown(
            f'<div class="listing-takeaway"><b>Key takeaway:</b> {takeaway} '
            f'The box plot shows the distribution within each age band, not a causal age effect.</div>',
            unsafe_allow_html=True,
        )

# Data quality and coverage panel.
st.divider()
section_header(
    "8. Listing data quality and coverage",
    "Check missingness and the breadth of the current filtered sample before drawing conclusions."
)
quality_cols = st.columns(4)
quality_cols[0].metric("Missing asking prices", f"{filtered['Price_in_Lakhs'].isna().sum():,}")
quality_cols[1].metric("Missing area values", f"{filtered['Size_in_SqFt'].isna().sum():,}")
quality_cols[2].metric("Missing city values", f"{filtered['City'].isna().sum():,}")
quality_cols[3].metric(
    "Duplicate rows",
    f"{int(filtered.duplicated().sum()):,}",
)
with st.expander("View field completeness"):
    quality_fields = [
        column for column in [
            "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
            "Price_in_Lakhs", "Year_Built", "Furnished_Status", "Availability_Status",
            "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
        ]
        if column in filtered.columns
    ]
    completeness = pd.DataFrame({
        "Field": quality_fields,
        "Non-missing records": [int(filtered[column].notna().sum()) for column in quality_fields],
        "Missing records": [int(filtered[column].isna().sum()) for column in quality_fields],
        "Completeness": [float(filtered[column].notna().mean() * 100) for column in quality_fields],
    }).sort_values("Completeness")
    st.dataframe(
        completeness.style.format({"Completeness": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

# Listing records and export.
st.divider()
section_header(
    "9. Filtered listing records",
    f"Showing up to 2,000 of {len(filtered):,} matching records. The CSV export contains all matching rows."
)
preferred_columns = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Price_per_SqFt_Calculated", "Year_Built", "Age_of_Property",
    "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
]
display_columns = [column for column in preferred_columns if column in filtered.columns]
st.dataframe(
    filtered[display_columns].head(2000),
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download all filtered listings (CSV)",
    data=safe_export_csv(filtered[display_columns]),
    file_name="filtered_housing_listings.csv",
    mime="text/csv",
    type="primary",
)
