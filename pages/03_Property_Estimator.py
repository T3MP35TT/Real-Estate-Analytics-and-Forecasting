from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

from src.config import ARTIFACT_DIR, CLASSIFICATION_FEATURES, FORECAST_YEARS
from utils.helper import (
    city_to_hpi, find_forecast, get_hpi_forecasts,
    get_listings, money_lakh, page_header, section_header,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Property Estimator | Real Estate Intelligence", page_icon="🏠", layout="wide")

st.markdown(
    """<style>.block-container {
        padding-top: 2.65rem !important;
        padding-bottom: 2.4rem !important;
    }</style>""",
    unsafe_allow_html=True,
)
page_header(
    "PROPERTY WORKBENCH",
    "Five-year value estimator",
    "Assess a property against city-level comparable listings, inspect its market price position, and model a five-year NHB HPI scenario with transparent assumptions and an experimental screening proxy.",
)
st.info("This estimates a city-market scenario, not an independently validated individual-property resale price. The lower and upper values are empirical scenario bounds—not guaranteed outcomes.")

try:
    df = get_listings()
except Exception as exc:
    st.error(f"Could not load the property dataset: {exc}")
    st.stop()
forecasts = get_hpi_forecasts()

section_header(
    "Property assessment",
    "Enter the property profile and asking price. Results combine city-level listing comparables with a city-level HPI scenario."
)
st.subheader("1. Location and property inputs")
place1, place2 = st.columns(2)
states = sorted(df["State"].dropna().astype(str).unique())
state = place1.selectbox("State / territory", states, key="estimate_state")
city_options = sorted(df.loc[df["State"].astype(str) == state, "City"].dropna().astype(str).unique())
city = place2.selectbox("City", city_options, key="estimate_city")

with st.form("property_estimator_form", border=True):
    st.markdown("**Property characteristics**")
    c1, c2, c3 = st.columns(3)
    type_options = sorted(df["Property_Type"].dropna().astype(str).unique())
    prop_type = c1.selectbox("Property type", type_options)
    bhk_options = sorted(pd.to_numeric(df["BHK"], errors="coerce").dropna().unique().astype(int).tolist())
    bhk = c2.selectbox("BHK", bhk_options, index=bhk_options.index(2) if 2 in bhk_options else 0)
    area = c3.number_input("Built-up area (sq ft)", min_value=100, max_value=50000, value=1200, step=50)

    c4, c5, c6 = st.columns(3)
    asking_price = c4.number_input("Current asking price (₹ lakh)", min_value=1.0, max_value=100000.0, value=100.0, step=5.0)
    year_built = c5.number_input("Year built", min_value=1900, max_value=2026, value=2015, step=1)
    furnished_options = sorted(df["Furnished_Status"].dropna().astype(str).unique())
    furnished = c6.selectbox("Furnishing", furnished_options)

    c7, c8, c9 = st.columns(3)
    floor = c7.number_input("Floor number", min_value=0, max_value=200, value=2)
    total_floors = c8.number_input("Total floors", min_value=1, max_value=200, value=5)
    schools = c9.number_input("Nearby schools (count / score)", min_value=0, max_value=100, value=3)

    c10, c11, c12 = st.columns(3)
    hospitals = c10.number_input("Nearby hospitals (count / score)", min_value=0, max_value=100, value=2)
    transport_options = sorted(df["Public_Transport_Accessibility"].dropna().astype(str).unique())
    transport = c11.selectbox("Public transport accessibility", transport_options)
    parking_options = sorted(df["Parking_Space"].dropna().astype(str).unique())
    parking_raw = c12.selectbox("Parking availability", parking_options)

    c13, c14, c15 = st.columns(3)
    security_options = sorted(df["Security"].dropna().astype(str).unique())
    security_raw = c13.selectbox("Security", security_options)
    amenities_options = sorted(df["Amenities"].dropna().astype(str).unique())
    amenities = c14.selectbox("Amenities profile", amenities_options)
    facing_options = sorted(df["Facing"].dropna().astype(str).unique())
    facing = c15.selectbox("Facing", facing_options)

    c16, c17 = st.columns(2)
    owner_options = sorted(df["Owner_Type"].dropna().astype(str).unique())
    owner_type = c16.selectbox("Owner type", owner_options)
    availability_options = sorted(df["Availability_Status"].dropna().astype(str).unique())
    availability = c17.selectbox("Availability status", availability_options)

    submitted = st.form_submit_button("Calculate five-year scenario", type="primary", use_container_width=True)

render_sidebar(
    "Property Estimator",
    listings=df,
    filtered_listings=df.loc[df["City"].astype(str).eq(str(city))] if "City" in df.columns else df,
    forecasts=forecasts,
    active_filters={"State": state, "City": city, "Property type": prop_type, "Bedrooms / BHK": bhk},
    extra_metrics={"Asking price": money_lakh(asking_price), "Built-up area": f"{area:,.0f} sq ft", "Furnishing": furnished},
)

if submitted:
    hpi_city = city_to_hpi(city)
    forecast = find_forecast(city, forecasts)

    st.divider()
    section_header(
        "2. Comparable-market assessment",
        "Compare the entered property with listings in the same city, property type and bedroom configuration. These are asking-price comparables, not verified transactions."
    )

    comps = df.copy()
    comps["Price_in_Lakhs"] = pd.to_numeric(comps["Price_in_Lakhs"], errors="coerce")
    comps["Size_in_SqFt"] = pd.to_numeric(comps["Size_in_SqFt"], errors="coerce")
    comps["BHK"] = pd.to_numeric(comps["BHK"], errors="coerce")
    comps = comps[
        comps["City"].astype(str).eq(str(city))
        & comps["Property_Type"].astype(str).eq(str(prop_type))
        & comps["BHK"].eq(float(bhk))
        & comps["Price_in_Lakhs"].gt(0)
        & comps["Size_in_SqFt"].gt(0)
    ].copy()
    comps["Asking_Rate_Per_SqFt"] = comps["Price_in_Lakhs"] * 100000 / comps["Size_in_SqFt"]

    area_comps = comps[
        comps["Size_in_SqFt"].between(float(area) * 0.75, float(area) * 1.25)
    ].copy()
    if len(area_comps) >= 5:
        comps_used = area_comps
        cohort_label = "same city, type, BHK and ±25% area"
    else:
        comps_used = comps
        cohort_label = "same city, type and BHK (area band too small)"

    entered_rate = float(asking_price) * 100000 / float(area)
    if not comps_used.empty:
        comp_rate = float(comps_used["Asking_Rate_Per_SqFt"].median())
        comp_price = comp_rate * float(area) / 100000
        rate_gap_pct = ((entered_rate / comp_rate) - 1) * 100 if comp_rate > 0 else np.nan
        cohort_q25 = float(comps_used["Asking_Rate_Per_SqFt"].quantile(0.25))
        cohort_q75 = float(comps_used["Asking_Rate_Per_SqFt"].quantile(0.75))
        if rate_gap_pct > 5:
            market_position = "Above comparable median"
            position_detail = f"The entered rate is {rate_gap_pct:.1f}% above the comparable median."
        elif rate_gap_pct < -5:
            market_position = "Below comparable median"
            position_detail = f"The entered rate is {abs(rate_gap_pct):.1f}% below the comparable median."
        else:
            market_position = "Near comparable median"
            position_detail = "The entered rate is within 5% of the comparable median."
    else:
        comp_rate = comp_price = rate_gap_pct = cohort_q25 = cohort_q75 = np.nan
        market_position = "Insufficient comparables"
        position_detail = "No comparable listings were found for this city, property type and BHK combination."

    profile = pd.DataFrame([
        ("Location", f"{city}, {state}"),
        ("Property configuration", f"{prop_type} · {bhk} BHK"),
        ("Built-up area", f"{area:,.0f} sq ft"),
        ("Year built / age", f"{int(year_built)} / {max(0, 2026 - int(year_built))} years"),
        ("Floor", f"{int(floor)} of {int(total_floors)}"),
        ("Furnishing", str(furnished)),
        ("Transport access", str(transport)),
        ("Parking / security", f"{parking_raw} / {security_raw}"),
        ("Amenities / facing", f"{amenities} / {facing}"),
        ("Owner / availability", f"{owner_type} / {availability}"),
    ], columns=["Assessment input", "Selected value"])
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown("**Asking-price position**")
    if not comps_used.empty:
        a, b, c, d = st.columns(4)
        a.metric("Entered asking rate", f"₹{entered_rate:,.0f}/sq ft")
        b.metric("Comparable median rate", f"₹{comp_rate:,.0f}/sq ft")
        c.metric("Comparable-equivalent price", money_lakh(comp_price))
        d.metric("Rate vs. comparable median", f"{rate_gap_pct:+.1f}%")
        st.caption(f"Comparable cohort: {len(comps_used):,} listings · {cohort_label}.")
        st.markdown(
            f'<div class="estimator-note"><b>Market position: {market_position}.</b> '
            f'{position_detail} The comparable-equivalent price applies the cohort median asking rate '
            f'to the entered area; it is a screening benchmark, not a formal valuation.</div>',
            unsafe_allow_html=True,
        )

        rate_chart = comps_used[["Asking_Rate_Per_SqFt"]].copy()
        rate_chart = rate_chart.rename(columns={"Asking_Rate_Per_SqFt": "Asking rate (₹/sq ft)"})
        fig = px.histogram(
            rate_chart, x="Asking rate (₹/sq ft)", nbins=24,
            labels={"count": "Number of listings"},
            title="Comparable asking-rate distribution",
            opacity=0.82,
        )
        fig.add_vline(
            x=entered_rate, line_dash="dash", line_width=3,
            annotation_text="Your property", annotation_position="top"
        )
        fig.add_vline(
            x=comp_rate, line_dash="dot", line_width=2,
            annotation_text="Comparable median", annotation_position="bottom"
        )
        fig.update_layout(
            height=360, margin=dict(l=8, r=12, t=48, b=12),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Key takeaway: your asking rate is {rate_gap_pct:+.1f}% relative to the cohort median. "
            f"The middle 50% of comparable asking rates spans ₹{cohort_q25:,.0f}–₹{cohort_q75:,.0f}/sq ft."
        )
    else:
        st.warning(position_detail + " Broaden the property type/BHK selection or review the source listing coverage.")

    st.divider()
    section_header(
        "3. Five-year city-market value scenario",
        "The current asking price is scaled by the selected city's five-year NHB HPI growth multiplier. This is a city-index scenario, not a property-specific resale prediction."
    )

    if forecast is None:
        st.warning(f"No NHB forecast is available for **{city}**. No growth rate has been substituted from another city.")
    else:
        factor = pd.to_numeric(pd.Series([forecast.get("HPI_Growth_Multiplier_5Y")]), errors="coerce").iloc[0]
        if not np.isfinite(factor) or factor <= 0:
            st.error("The forecast multiplier is invalid. Rerun `python scripts/train.py` and inspect the forecast reliability report.")
        else:
            years = int(FORECAST_YEARS)
            point_value = float(asking_price) * float(factor)
            total_growth_pct = (float(factor) - 1) * 100
            annualized_growth = (float(factor) ** (1 / years) - 1) * 100 if years > 0 else np.nan
            value_change = point_value - float(asking_price)

            lower_factor = pd.to_numeric(
                pd.Series([forecast.get("HPI_Growth_Multiplier_80_Lower_Empirical", np.nan)]),
                errors="coerce"
            ).iloc[0]
            upper_factor = pd.to_numeric(
                pd.Series([forecast.get("HPI_Growth_Multiplier_80_Upper_Empirical", np.nan)]),
                errors="coerce"
            ).iloc[0]
            range_available = (
                np.isfinite(lower_factor) and np.isfinite(upper_factor)
                and lower_factor > 0 and upper_factor > 0
            )
            if range_available:
                low_value, high_value = sorted([
                    float(asking_price) * float(lower_factor),
                    float(asking_price) * float(upper_factor)
                ])
                low_value = min(low_value, point_value, high_value)
                high_value = max(low_value, point_value, high_value)
            else:
                low_value = high_value = np.nan

            st.markdown("**Headline scenario outputs**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current asking price", money_lakh(asking_price))
            k2.metric("Five-year point scenario", money_lakh(point_value), delta=f"{total_growth_pct:+.1f}% total")
            k3.metric("Implied annualized growth", f"{annualized_growth:.2f}%")
            k4.metric("Indicative value change", money_lakh(value_change), delta=f"{value_change / asking_price * 100:+.1f}%")

            if range_available:
                r1, r2, r3 = st.columns(3)
                r1.metric("Lower indicative scenario", money_lakh(low_value))
                r2.metric("Point estimate", money_lakh(point_value))
                r3.metric("Upper indicative scenario", money_lakh(high_value))
                st.caption(f"Indicative scenario span: {money_lakh(high_value - low_value)} between lower and upper empirical multipliers.")
            else:
                st.warning("The report does not contain a valid empirical lower/upper multiplier for this city.")

            # The annual path interpolates the five-year endpoint; it is not a yearly model output.
            path_rows = []
            for year in range(years + 1):
                path_value = float(asking_price) * (float(factor) ** (year / years)) if years else float(asking_price)
                path_rows.append({"Year": f"Year {year}", "Illustrative value (₹ lakh)": path_value})
            path_df = pd.DataFrame(path_rows)
            fig = px.line(
                path_df, x="Year", y="Illustrative value (₹ lakh)", markers=True,
                title="Illustrative value path from the five-year endpoint",
            )
            fig.update_traces(line_width=3, marker_size=8)
            fig.update_layout(
                height=360, margin=dict(l=8, r=12, t=48, b=12),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Key takeaway: the five-year city HPI multiplier implies {total_growth_pct:+.1f}% "
                f"cumulative change ({annualized_growth:.2f}% annualized) on the entered starting price. "
                "Intermediate annual points are mathematically interpolated, not yearly model forecasts."
            )

            if range_available:
                scenario = pd.DataFrame({
                    "Scenario": ["Lower indicative", "Point estimate", "Upper indicative"],
                    "Estimated value (₹ lakh)": [low_value, point_value, high_value],
                    "Growth from current (%)": [
                        (low_value / asking_price - 1) * 100,
                        (point_value / asking_price - 1) * 100,
                        (high_value / asking_price - 1) * 100,
                    ],
                })
                fig = px.bar(
                    scenario, x="Scenario", y="Estimated value (₹ lakh)", color="Scenario",
                    color_discrete_map={
                        "Lower indicative": "#e4a64b",
                        "Point estimate": "#147d83",
                        "Upper indicative": "#446c9b",
                    },
                    text_auto=".1f",
                    hover_data={"Growth from current (%)": ":.1f"},
                )
                fig.update_layout(showlegend=False, height=340, margin=dict(l=5, r=5, t=16, b=10))
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Key takeaway: the empirical lower/upper multipliers show how historical city-HPI "
                    "forecast errors widen the rupee scenario; they are not guaranteed resale bounds."
                )

            forecast_details = pd.DataFrame([
                ("Market index", hpi_city),
                ("Latest observed quarter", forecast.get("Latest_Quarter", "—")),
                ("Forecast endpoint", forecast.get("Forecast_Quarter", "—")),
                ("Selected forecasting method", forecast.get("Selected_Method", "—")),
                ("Five-year growth multiplier", f"{float(factor):.4f}×"),
                ("Implied CAGR", f"{annualized_growth:.2f}%"),
                ("Forecast fallback used", str(forecast.get("Forecast_Fallback_Used", False))),
            ], columns=["Forecast detail", "Value"])
            st.markdown("**Forecast provenance and model details**")
            st.dataframe(forecast_details, use_container_width=True, hide_index=True)

            fallback = str(forecast.get("Forecast_Fallback_Used", False)).lower() in {"true", "1", "yes"}
            if fallback:
                st.warning(
                    f"The city forecast used fallback method `{forecast.get('Forecast_Fallback_Method', 'last_value')}`. "
                    "Treat this estimate with extra caution."
                )

            st.markdown(
                '<div class="estimator-note"><b>Interpretation boundary.</b> '
                'The forecast applies a city-level HPI movement to the entered asking price. '
                'It does not model neighbourhood-level appreciation, building condition, rental yield, '
                'transaction costs, taxes, financing, liquidity or negotiated sale price. '
                'The empirical range is pooled historical forecast-error context, not a guaranteed 80% property-resale interval.</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    section_header(
        "4. Experimental investment-screening proxy",
        "This classifier predicts the project's constructed screening label from the entered property characteristics. It is not a profitability or investment-return model."
    )
    with st.expander("Model result and feature diagnostics", expanded=True):
        classifier_path = ARTIFACT_DIR / "classifier.joblib"
        if not classifier_path.exists():
            st.caption("Classifier artifact not found. Run `python scripts/train.py` to generate the current benchmark model.")
        else:
            clf = joblib.load(classifier_path)
            feature_names = list(getattr(clf.named_steps.get("preprocess"), "feature_names_in_", CLASSIFICATION_FEATURES))
            row_values = {
                "State": state, "City": city, "Locality": "Unknown", "Property_Type": prop_type,
                "BHK": bhk, "Size_in_SqFt": area, "Price_in_Lakhs": asking_price,
                "Price_per_SqFt": asking_price * 100000 / area,
                "Price_per_SqFt_Thousands": asking_price * 100 / area,
                "Year_Built": year_built, "Age_of_Property": max(0, 2026 - int(year_built)),
                "Furnished_Status": furnished, "Floor_No": floor, "Total_Floors": total_floors,
                "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
                "School_Density_Score": schools / (area / 1000), "Hospital_Density_Score": hospitals / (area / 1000),
                "Parking_Space": parking_raw, "Security": security_raw,
                "Parking_Space_Binary": int(str(parking_raw).strip().lower() in {"yes", "available", "true", "1"}),
                "Security_Binary": int(str(security_raw).strip().lower() in {"yes", "available", "true", "1", "gated", "cctv", "guard"}),
                "Public_Transport_Accessibility": transport, "Amenities": amenities, "Facing": facing,
                "Owner_Type": owner_type, "Availability_Status": availability,
            }
            model_input = pd.DataFrame([{name: row_values.get(name, np.nan) for name in feature_names}])
            try:
                pred = int(clf.predict(model_input)[0])
                probabilities = clf.predict_proba(model_input)[0]
                classes = list(clf.classes_)
                probability = float(probabilities[classes.index(pred)])
                result_label = "Meets screening rule" if pred == 1 else "Does not meet screening rule"
                result_col, confidence_col = st.columns(2)
                result_col.metric("Screening outcome", result_label)
                confidence_col.metric("Model confidence in proxy label", f"{probability:.1%}")
                st.progress(probability, text=f"Model probability for the predicted proxy label: {probability:.1%}")
                st.markdown(
                    f'<div class="estimator-note"><b>What this result means:</b> '
                    f'The model assigned this input profile to <b>{result_label}</b> with '
                    f'{probability:.1%} probability. This is confidence in the constructed training '
                    f'label, not the probability of profit, appreciation or a positive return.</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Inputs supplied to the screening model"):
                    input_display = pd.DataFrame(
                        [{"Feature": key, "Value": value} for key, value in row_values.items()]
                    )
                    st.dataframe(input_display, use_container_width=True, hide_index=True)
                importance_path = ARTIFACT_DIR / "classifier_importance.csv"
                if importance_path.exists():
                    imp = pd.read_csv(importance_path).head(12).sort_values("importance")
                    if not imp.empty and {"feature", "importance"}.issubset(imp.columns):
                        fig = px.bar(imp, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"])
                        fig.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=5, r=5, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Key takeaway: feature importance describes inputs used by the "
                            "constructed screening proxy; it does not identify causal drivers of profit."
                        )
            except Exception as exc:
                st.warning(f"The experimental classifier could not process this input: {exc}")


st.divider()
section_header(
    "5. Property due-diligence checklist",
    "Independent checks that should accompany any market-screening result."
)
due_diligence = pd.DataFrame([
    ("Comparable evidence", "Check recent registered transactions and active competing listings in the same city and comparable property segment."),
    ("Legal and title review", "Verify title chain, encumbrances, approvals, land use, RERA registration and pending litigation."),
    ("Physical inspection", "Assess structural condition, seepage, maintenance, age, utilities and society/building finances."),
    ("Total acquisition cost", "Include stamp duty, registration, brokerage, taxes, renovation and financing charges."),
    ("Rental economics", "Validate achievable rent, vacancy, maintenance, property tax and net rental yield independently."),
    ("Exit and liquidity", "Assess buyer depth, resale timelines, local supply pipeline and likely selling costs."),
], columns=["Due-diligence area", "What to verify"])
st.dataframe(due_diligence, use_container_width=True, hide_index=True)
st.caption(
    "The estimator is an exploratory decision-support tool. Listing comparables, city HPI scenarios "
    "and the experimental classifier answer different questions; none is a guaranteed property valuation."
)
