from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.helper import (
    get_hpi_forecasts, get_report_csv, get_report_json,
    hpi_summary, page_header, section_header,
)

from utils.sidebar import render_sidebar

st.set_page_config(page_title="Forecast Performance | Real Estate Intelligence", page_icon="📈", layout="wide")

st.markdown(
    """<style>.block-container {
        padding-top: 2.65rem !important;
        padding-bottom: 2.4rem !important;
    }</style>""",
    unsafe_allow_html=True,
)
page_header(
    "MODEL GOVERNANCE",
    "Forecast performance",
    "Review how candidate methods were selected, how they performed on held-out historical periods, and how index errors scale into indicative property-value scenarios.",
)
st.warning("Metrics on this page evaluate city-level NHB HPI forecasts or index-implied scenarios. They are not error rates measured against actual individual-property resale transactions.")

metrics = get_report_json("hpi_forecast_metrics.json")
summary = hpi_summary(metrics)
cv = get_report_csv("hpi_forecast_cv_results.csv")
holdout = get_report_csv("hpi_forecast_holdout_predictions.csv")
scenario = get_report_csv("resale_value_scenario_metrics.csv")
fallbacks = get_report_csv("hpi_forecast_fallbacks.csv")
forecasts = get_hpi_forecasts()

if not metrics:
    st.error("HPI forecast metrics were not found. Run `python scripts/train.py` from the project root.")
    st.stop()

holdout_metrics = metrics.get("final_holdout", {}).get("metrics", {})
cv_methods = metrics.get("cross_validation", {}).get("metrics_by_method", [])
if cv.empty and cv_methods:
    cv = pd.DataFrame(cv_methods)

city_options = sorted(holdout["city"].dropna().astype(str).unique()) if "city" in holdout.columns else []
selected_cities = st.multiselect(
    "Diagnostic cities",
    options=city_options,
    default=city_options,
    help="Filters the held-out prediction chart. Candidate-method CV results remain model-wide.",
    key="forecast_diagnostic_cities",
)
holdout_view = holdout.copy()
if selected_cities and "city" in holdout_view.columns:
    holdout_view = holdout_view[holdout_view["city"].astype(str).isin(selected_cities)]

render_sidebar(
    "Forecast Performance",
    forecasts=forecasts,
    model_metrics=summary,
    active_filters={"Diagnostic cities": selected_cities or "All"},
    extra_metrics={"Holdout observations": f"{len(holdout_view):,}", "CV candidates": f"{len(cv_methods):,}"},
)

section_header("Selected model", "Model chosen by rolling-origin cross-validation, subject to the configured reliability screen.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Selected method", str(summary["method"]).replace("_", " ").title())
k2.metric("CV MAPE", f"{summary['cv_mape']:.2f}%" if isinstance(summary["cv_mape"], (int, float)) else "—")
k3.metric("Holdout MAPE", f"{summary['holdout_mape']:.2f}%" if isinstance(summary["holdout_mape"], (int, float)) else "—")
k4.metric("Holdout directional accuracy", f"{summary['directional_accuracy']:.1%}" if isinstance(summary["directional_accuracy"], (int, float)) else "—")
if summary["latest_quarter"] != "—":
    st.caption(f"Historical series ends at {summary['latest_quarter']} · five-year forecast endpoint {summary['forecast_quarter']} · {summary['cities'] or '—'} cities · {summary['quarters'] or '—'} quarters.")

st.divider()
section_header("Candidate comparison", "Lower MAPE is better. Candidates excluded by the fallback reliability rule are identified separately.")
if not cv.empty and "method" in cv.columns:
    cv_view = cv.copy()
    if "selection_eligible" in cv_view:
        cv_view["Selection status"] = cv_view["selection_eligible"].map({True: "Eligible", False: "Excluded: reliability"}).fillna("Unspecified")
    if "mape_percent" in cv_view:
        cv_view = cv_view.sort_values("mape_percent")
        plot_col = "mape_percent"
    elif "mape" in cv_view:
        cv_view = cv_view.sort_values("mape")
        plot_col = "mape"
    else:
        plot_col = None
    if plot_col:
        color_col = "Selection status" if "Selection status" in cv_view.columns else None
        fig = px.bar(cv_view, x=plot_col, y="method", orientation="h", color=color_col,
                     color_discrete_map={"Eligible": "#147d83", "Excluded: reliability": "#df8d63", "Unspecified": "#7f92a5"},
                     hover_data=[c for c in ["mae_hpi_points", "rmse_hpi_points", "directional_accuracy", "fallback_rate_percent"] if c in cv_view.columns],
                     labels={plot_col: "Rolling-origin CV MAPE (%)", "method": ""})
        fig.update_layout(height=max(420, 25*len(cv_view)), margin=dict(l=5,r=5,t=10,b=10), legend_title_text="Status")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        best_candidate = cv_view.iloc[0]
        best_score = best_candidate.get(plot_col)
        st.info(
            f"**Key takeaway:** {best_candidate['method']} has the lowest candidate CV error "
            f"({float(best_score):.2f}% MAPE where the source metric is percentage-based). "
            "Eligibility and fallback reliability should be reviewed alongside error before selection."
        )
    display_cols = [c for c in ["method", "mape_percent", "mae_hpi_points", "rmse_hpi_points", "directional_accuracy", "mae_skill_vs_last_value", "fallback_rate_percent", "selection_eligible", "Selection status"] if c in cv_view.columns]
    st.dataframe(cv_view[display_cols], use_container_width=True, hide_index=True)
else:
    st.info("Candidate comparison CSV is missing, although summary metrics are present. Check the reports output path.")

with st.expander("How selection and reliability screening work"):
    design = metrics.get("candidate_selection_design", {})
    reliability = metrics.get("candidate_reliability", {})
    st.write(f"**Selection criterion:** {metrics.get('selection_metric', 'Not recorded')}.")
    st.write(f"**Candidate methods declared:** {len(metrics.get('candidates_compared', []))}.")
    st.write(f"**Reliability policy:** {reliability.get('fallback_policy', 'Not recorded')}.")
    st.write(f"**Fallback threshold:** {reliability.get('selection_fallback_threshold_percent', '—')}%.")
    if reliability.get("eligible_candidates"):
        st.write("**Eligible candidates:** " + ", ".join(reliability["eligible_candidates"]))
    if design:
        st.write(f"Screening used {design.get('cv_city_count', '—')} cities and {design.get('cv_origin_count', '—')} forecast origins. The selected candidate is subsequently evaluated on the separate temporal holdout.")

st.divider()
section_header("Held-out forecast diagnostics", "Each point compares predicted city HPI with the observed HPI at the same five-year target quarter.")
if not holdout_view.empty and {"predicted_hpi", "actual_hpi"}.issubset(holdout_view.columns):
    fig = px.scatter(holdout_view, x="actual_hpi", y="predicted_hpi", color="city" if "city" in holdout_view.columns else None,
                     hover_data=[c for c in ["origin", "target_quarter", "history_quarters"] if c in holdout.columns],
                     opacity=.75, labels={"actual_hpi": "Observed HPI", "predicted_hpi": "Predicted HPI", "city": "City"})
    low = float(np.nanmin([holdout_view["actual_hpi"].min(), holdout_view["predicted_hpi"].min()]))
    high = float(np.nanmax([holdout_view["actual_hpi"].max(), holdout_view["predicted_hpi"].max()]))
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect forecast", line=dict(color="#14253d", dash="dash", width=2)))
    fig.update_layout(height=480, margin=dict(l=5,r=5,t=10,b=10), legend_title_text="City")
    st.plotly_chart(fig, use_container_width=True)
    mean_abs_gap = (holdout_view["predicted_hpi"] - holdout_view["actual_hpi"]).abs().mean()
    st.info(
        f"**Key takeaway:** Across {len(holdout_view):,} selected held-out observations, "
        f"the mean absolute gap is {mean_abs_gap:.2f} HPI points. The diagonal represents "
        "perfect index prediction; this is not property-level resale accuracy."
    )
else:
    st.info("Held-out predictions are not available to plot. Check `reports/hpi_forecast_holdout_predictions.csv`.")

if not holdout_metrics:
    st.caption("Holdout metric details were not found in the report JSON.")
else:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Holdout HPI MAE", f"{holdout_metrics.get('mae_hpi_points', float('nan')):.2f} points")
    q2.metric("Holdout HPI RMSE", f"{holdout_metrics.get('rmse_hpi_points', float('nan')):.2f} points")
    q3.metric("Naïve baseline MAE", f"{holdout_metrics.get('naive_last_value_mae_hpi_points', float('nan')):.2f} points")
    skill = holdout_metrics.get("mae_skill_vs_last_value")
    q4.metric("MAE skill vs baseline", f"{skill:.1%}" if isinstance(skill, (int, float)) else "—")

st.divider()
section_header("Price-scaled scenario backtest", "Illustrative ₹50 lakh, ₹100 lakh and ₹200 lakh starting values; references are index-implied, not transaction prices.")
if not scenario.empty:
    scenario_view = scenario.copy()
    if {"split", "illustrative_starting_price_lakhs"}.issubset(scenario_view.columns):
        scenario_view["Starting value"] = scenario_view["illustrative_starting_price_lakhs"].map(lambda x: f"₹{x:,.0f} lakh")
        if "scenario_mape_percent" in scenario_view:
            fig = px.bar(scenario_view, x="Starting value", y="scenario_mape_percent", color="split", barmode="group",
                         color_discrete_map={"cross_validation": "#147d83", "final_holdout": "#e4a64b"},
                         labels={"scenario_mape_percent": "Scenario MAPE (%)", "split": "Evaluation split"},
                         hover_data=[c for c in ["scenario_mae_lakhs", "scenario_rmse_lakhs", "scenario_bias_lakhs", "scenario_abs_error_p90_lakhs"] if c in scenario_view.columns])
            fig.update_layout(height=380, margin=dict(l=5,r=5,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
            best_scenario = scenario_view.loc[scenario_view["scenario_mape_percent"].idxmin()]
            st.info(
                f"**Key takeaway:** The lowest scenario MAPE in this report is "
                f"{best_scenario['scenario_mape_percent']:.2f}% for the "
                f"{best_scenario.get('Starting value', 'selected starting value')} case "
                f"({best_scenario.get('split', 'evaluation split')})."
            )
    display_cols = [c for c in ["split", "illustrative_starting_price_lakhs", "city_origin_forecasts", "scenario_mae_lakhs", "scenario_rmse_lakhs", "scenario_mape_percent", "scenario_bias_lakhs", "scenario_abs_error_p90_lakhs", "last_value_baseline_mae_lakhs", "mae_skill_vs_last_value_baseline", "fallback_rate_percent"] if c in scenario_view.columns]
    st.dataframe(scenario_view[display_cols], use_container_width=True, hide_index=True)
else:
    st.info("Scenario metrics not found. Rerun `python scripts/train.py` using the updated version.")

with st.expander("Fallback audit"):
    if fallbacks.empty:
        st.success("No forecast fallback events are present in the current report, or the fallback file contains zero rows.")
    else:
        st.warning(f"{len(fallbacks):,} fallback event(s) were recorded. A fallback uses a safe alternative forecast and should be reviewed before interpreting results.")
        st.dataframe(fallbacks, use_container_width=True, hide_index=True)

st.caption("Rolling-origin windows overlap and are not independent observations. HPI-index performance should not be described as verified property-level resale performance.")
