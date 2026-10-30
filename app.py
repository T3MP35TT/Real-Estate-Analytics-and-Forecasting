from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.helper import (
    get_hpi_forecasts,
    get_listings,
    get_report_json,
    hpi_summary,
    page_header,
    section_header,
)
from utils.theme import apply_theme

st.set_page_config(
    page_title="Home | Real Estate Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Load the same shared context used by the other dashboard pages.
try:
    listings = get_listings().copy()
    listing_error = None
except Exception as exc:
    listings = pd.DataFrame()
    listing_error = str(exc)

forecasts = get_hpi_forecasts()
metrics = get_report_json("hpi_forecast_metrics.json")
hpi = hpi_summary(metrics)

# Apply the shared visual theme, but keep the Home sidebar self-contained.
# This page intentionally does not depend on utils/sidebar.py.
apply_theme("Home", sidebar_branding=False)

with st.sidebar.container(border=True):
    st.markdown("**🏡 Estate Intelligence**")
    st.caption("Property analytics · Market outlook")

    with st.container(border=True):
        st.caption("CURRENT WORKSPACE")
        st.markdown("**Home**")

    st.caption("Decision-support prototype · Not financial advice")

    with st.container(border=True):
        st.caption("PROJECT INFORMATION")
        st.markdown(
            '<div style="color:#D5AD68;font-size:.68rem;font-weight:800;'
            'letter-spacing:.10em;text-transform:uppercase;margin:8px 0 3px;">'
            'PROJECT</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Real Estate Investment Advisor**")

        info_rows = [
            ("Author", "Kartikey Singh"),
            ("Built with", "Python · Streamlit · Pandas · Plotly · Scikit-learn"),
            ("Focus", "Property analytics & HPI forecasting"),
        ]
        for label, value in info_rows:
            left, right = st.columns([1, 1.5], gap="small")
            left.markdown(
                f'<span style="color:#A1B2A7;font-size:.78rem;">{label}</span>',
                unsafe_allow_html=True,
            )
            right.markdown(
                f'<div style="text-align:right;color:#E8F0EA;font-size:.78rem;'
                f'font-weight:700;overflow-wrap:anywhere;">{value}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(
            "A data-driven decision-support project for property-listing analysis, "
            "city-level housing trends and five-year value scenarios."
        )

page_header(
    "REAL ESTATE ANALYTICS WORKSPACE",
    "Real Estate Investment Advisor",
    "A practical workspace for property-listing analysis, city-market monitoring and five-year housing-value scenarios.",
)

# Home is a launchpad with a compact data-readiness snapshot.
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listing records", f"{len(listings):,}" if not listings.empty else "—")
k2.metric("Cities in listings", f"{listings['City'].nunique():,}" if not listings.empty and "City" in listings else "—")
k3.metric("Cities with HPI forecasts", f"{forecasts['City'].nunique():,}" if not forecasts.empty and "City" in forecasts else "—")
k4.metric("Forecast holdout MAPE", f"{hpi['holdout_mape']:.1f}%" if isinstance(hpi.get("holdout_mape"), (int, float)) else "—")

if listing_error:
    st.warning(f"Property data could not be loaded: {listing_error}")
if forecasts.empty:
    st.warning("Forecast artifacts are not available yet. Run `python scripts/train.py` from the project root after confirming the configured data paths.")
else:
    st.caption(
        f"Selected HPI method: {str(hpi['method']).replace('_', ' ').title()} · "
        f"Historical data through {hpi['latest_quarter']} · Forecast endpoint {hpi['forecast_quarter']}"
    )

st.divider()
section_header("Workspace navigation", "Choose a workflow to continue. Executive Overview is a separate page from Home.")
pages = [
    ("Executive Overview", "Decision-summary KPIs, forecast health and the projected city outlook.", "pages/01_Executive_Overview.py", "🧭"),
    ("Market Insights", "Explore asking-price distributions, property segments and historical HPI.", "pages/02_Market_Insights.py", "📊"),
    ("Property Estimator", "Calculate a five-year city-HPI-based value scenario for a listing.", "pages/03_Property_Estimator.py", "🏠"),
    ("Listing Explorer", "Filter property records, review segment summaries and export results.", "pages/04_Listing_Explorer.py", "🔎"),
    ("Forecast Performance", "Inspect cross-validation, holdout metrics, scenario errors and fallbacks.", "pages/05_Forecast_Performance.py", "📈"),
    ("Methodology & Data", "Understand data sources, transformations, assumptions and limitations.", "pages/06_Methodology_and_Data.py", "📘"),
]
# Native Streamlit page links are the cards themselves, so the whole tile
# (including its title and description) is a real, clickable navigation target.
st.markdown(
    """
    <style>
    div[data-testid="stMainBlockContainer"], .block-container {
        padding-top: 2.35rem !important;
    }

    /* Turn each native page link into a KPI-style navigation card. */
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] {
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] > a,
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a {
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        justify-content: flex-start !important;
        gap: 0 !important;
        width: 100% !important;
        min-height: 164px !important;
        height: 100% !important;
        padding: 18px 20px 16px !important;
        background: linear-gradient(145deg, #17251E 0%, #111B17 100%) !important;
        border: 1px solid #2C4035 !important;
        border-top: 3px solid #D6AD68 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, .12) !important;
        color: #E8F0EA !important;
        text-decoration: none !important;
        white-space: normal !important;
        transition: transform .14s ease, border-color .14s ease,
                    box-shadow .14s ease, background .14s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a:hover {
        transform: translateY(-2px) !important;
        background: linear-gradient(145deg, #203229 0%, #14221A 100%) !important;
        border-color: #D6AD68 !important;
        border-top-color: #E5C98F !important;
        box-shadow: 0 10px 22px rgba(0, 0, 0, .22) !important;
        color: #E8F0EA !important;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a:focus-visible {
        outline: 2px solid #E5C98F !important;
        outline-offset: 3px !important;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a > div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
        min-width: 0 !important;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a > div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    /* Larger interactive card title. Style both the paragraph and its bold
       inline Markdown element because Streamlit's generated DOM varies by version. */
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a [data-testid="stMarkdownContainer"] p:first-child,
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a [data-testid="stMarkdownContainer"] p:first-child strong,
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a [data-testid="stMarkdownContainer"] p:first-child b {
        color: #E8F0EA !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        line-height: 1.35 !important;
        letter-spacing: 0 !important;
    }
    /* Keep descriptions smaller and readable, wrapping within each tile. */
    div[data-testid="stMainBlockContainer"] div[data-testid="stPageLink"] a [data-testid="stMarkdownContainer"] p + p {
        margin-top: 12px !important;
        color: #B8C9BE !important;
        font-size: .93rem !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for start in (0, 3):
    cols = st.columns(3, gap="medium")
    for col, (title, description, path, icon) in zip(cols, pages[start:start + 3]):
        with col:
            # The native page-link anchor is the card: the title, description,
            # and card background all navigate to the target page.
            st.page_link(
                path,
                label=f"**{icon} {title}**\n\n{description}",
                use_container_width=True,
                help=f"Open {title}",
            )

st.divider()
section_header("What this workspace is designed to do", "Keep the analytical workflow transparent and reproducible")
left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        '<div class="subtle-card">'
        '<h3>Market and listing analysis</h3>'
        '<p>Summarize the supplied listing snapshot, compare cities and property types, and explore asking-price distributions. Listing prices are not verified transaction prices.</p><br>'
        '<h3>City-level market outlook</h3>'
        '<p>Use quarterly NHB RESIDEX HPI history to forecast city-market movement and apply the estimated growth multiplier to a user-provided asking price.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="subtle-card">'
        '<h3>Model governance</h3>'
        '<p>Review rolling-origin validation, temporal holdout performance, candidate eligibility and forecast stability rather than relying on training fit alone.</p><br>'
        '<h3>Decision boundary</h3>'
        '<p>The five-year output is an HPI-based city-market scenario, not a property-specific resale prediction. The investment classifier uses a constructed proxy, not observed investment returns.</p>'
        '</div>',
        unsafe_allow_html=True,
    )



st.divider()
section_header(
    "Reproducibility",
    "Recreate the analytical outputs from the project root using the same documented pipeline.",
)

with st.expander("Environment setup and full refresh", expanded=False):
    st.markdown("**1. Create the environment and install dependencies**")
    st.code(
        "py -m venv .venv\n"
        ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        language="powershell",
    )

    st.markdown("**2. Recreate the exploratory data analysis reports**")
    st.code(
        ".\\.venv\\Scripts\\python.exe scripts\\run_eda.py",
        language="powershell",
    )

    st.markdown("**3. Run the unified training and evaluation pipeline**")
    st.code(
        ".\\.venv\\Scripts\\python.exe scripts\\train.py",
        language="powershell",
    )

    st.markdown("**4. Launch the dashboard**")
    st.code(
        ".\\.venv\\Scripts\\python.exe -m streamlit run app.py",
        language="powershell",
    )

    st.caption(
        "Run these commands from the repository root. The unified training script "
        "runs the NHB city-HPI forecast workflow and the listing-model benchmarks. "
        "The EDA step is a separate reporting step; it is not required just to open "
        "the dashboard when prepared outputs already exist."
    )

with st.expander("What makes a run reproducible?", expanded=False):
    st.markdown(
        """
        - **Consistent environment:** dependencies are installed from `requirements.txt`
          inside a project virtual environment.
        - **Scripted workflow:** data preparation, model comparison, tuning and evaluation
          are run through version-controlled Python scripts rather than relying only on
          manually executed notebook cells.
        - **Controlled evaluation:** model selection uses city-grouped cross-validation;
          the city holdout is reserved for final evaluation. The HPI forecast uses
          rolling-origin validation and a temporal holdout.
        - **Run record:** the training workflow records run details such as the input
          data hash, assumptions, cleaning counts and library versions in its run
          manifest, alongside evaluation outputs.
        - **Reviewable outputs:** reports and model/evaluation artifacts are written
          to the project output folders and can be inspected after a run.
        """
    )

st.caption(
    "Reproducibility note: identical results depend on using the same source data, "
    "configuration and dependency versions. The investment label is a constructed "
    "proxy, and the HPI-based value scenario is a city-market estimate—not a verified "
    "property-level resale outcome."
)

st.caption("Use the sidebar to switch pages at any time. For methodology, sources and assumptions, open Methodology & Data.")
