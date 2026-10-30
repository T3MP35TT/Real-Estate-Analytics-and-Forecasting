from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    DATA_PATH, FORECAST_YEARS, HPI_PATH, LISTING_SOURCE_URL, HPI_SOURCE_URL,
    ARTIFACT_DIR, REPORT_DIR,
)
from utils.helper import (
    get_hpi_forecasts, get_listings, get_report_json,
    hpi_summary, page_header, section_header,
)
from utils.sidebar import render_sidebar
from utils.data_manager import (
    DATASETS, audit_log, create_record, delete_record, export_dataset,
    import_csv, initialize_dataset, read_dataset, update_record,
)

st.set_page_config(
    page_title="Methodology & Data | Real Estate Intelligence",
    page_icon="📘",
    layout="wide",
)
st.markdown(
    """<style>.block-container {
        padding-top: 2.65rem !important;
        padding-bottom: 2.4rem !important;
    }</style>""",
    unsafe_allow_html=True,
)
page_header(
    "METHOD & GOVERNANCE",
    "Methodology & data",
    "A transparent guide to the data, model workflow, assumptions, evaluation and intended use of the Real Estate Investment Advisor.",
)

try:
    df = get_listings()
except Exception as exc:
    df = pd.DataFrame()
    st.warning(f"Listing dataset could not be loaded: {exc}")
forecasts = get_hpi_forecasts()
metrics = get_report_json("hpi_forecast_metrics.json")
hpi = hpi_summary(metrics)

# Sidebar is rendered only after its data inputs exist.
render_sidebar(
    "Methodology & Data",
    listings=df,
    filtered_listings=df,
    forecasts=forecasts,
    model_metrics=hpi,
    active_filters={"Dataset administration": "Managed SQLite workspace"},
)

section_header("Data inventory", "Current local files and their role in the analytics workflow")
rows = [
    {"Dataset / artifact": "Property listing dataset", "Purpose": "Descriptive listing analysis; current asking-price context", "Data level": "Property listing snapshot", "Limit": "No matched sale dates or observed resale outcomes"},
    {"Dataset / artifact": "NHB RESIDEX HPI", "Purpose": "Quarterly city-level market trend forecasting", "Data level": "City × quarter index", "Limit": "Not a property-level transaction history"},
    {"Dataset / artifact": "Forecast artifacts", "Purpose": "Five-year HPI point forecasts and empirical error multipliers", "Data level": "City market scenario", "Limit": "Uncertainty band is empirical and pooled; not a formal calibrated property interval"},
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Source references", expanded=False):
    st.markdown(f"- **Property listings:** [{LISTING_SOURCE_URL}]({LISTING_SOURCE_URL})" if LISTING_SOURCE_URL and LISTING_SOURCE_URL != "test" else "- **Property listings:** source URL is configured in `src/config.py`.")
    st.markdown(f"- **NHB RESIDEX HPI:** [{HPI_SOURCE_URL}]({HPI_SOURCE_URL})" if HPI_SOURCE_URL and HPI_SOURCE_URL != "test" else "- **NHB RESIDEX HPI:** source URL is configured in `src/config.py`.")
    st.code(
    f"Listings path: data/india_housing_prices.csv\n"
    f"HPI path: data/processed/nhb_hpi_combined.csv",
    language="text"
    )

section_header("Data management", "Select a dataset, inspect its records and manage editable source data in a separate SQLite workspace.")
st.warning(
    "Edits are stored in data/managed_data.db and do not overwrite the original source CSVs or generated forecast artifacts. "
    "Export the managed dataset when you are ready to review or publish the changes."
)

dataset_name = st.selectbox(
    "Select dataset to manage",
    list(DATASETS.keys()),
    format_func=lambda name: f"{name}  ·  {'Editable' if DATASETS[name]['editable'] else 'Read-only'}",
    key="managed_dataset_selector",
)
cfg = DATASETS[dataset_name]
st.caption(cfg["description"])

try:
    record_count, initialized_now = initialize_dataset(dataset_name)
    if initialized_now:
        st.success(f"Loaded {record_count:,} source records into the managed workspace.")
    managed, total_records = read_dataset(dataset_name, limit=1000)
except Exception as exc:
    st.error(f"Unable to initialize this dataset: {exc}")
    managed, total_records = pd.DataFrame(), 0

if not managed.empty:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Records in managed dataset", f"{total_records:,}")
    metric_cols[1].metric("Columns", f"{len(managed.columns) - 1:,}")
    metric_cols[2].metric("Management mode", "CRUD enabled" if cfg["editable"] else "Read-only")
    metric_cols[3].metric("Rows displayed", f"{len(managed):,}")

    search = st.text_input("Search records", placeholder="Search across all fields…", key="managed_search")
    try:
        records, total_records = read_dataset(dataset_name, search=search, limit=1000)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        records = pd.DataFrame()

    st.caption(f"Showing {len(records):,} matching records (maximum 1,000 per view).")
    display_frame = records.drop(columns=["_record_id"], errors="ignore")
    st.dataframe(display_frame, use_container_width=True, hide_index=True, height=330)

    st.download_button(
        "Export managed dataset (CSV)",
        data=export_dataset(dataset_name).to_csv(index=False).encode("utf-8"),
        file_name=f"{cfg['table']}_managed.csv",
        mime="text/csv",
    )

    if cfg["editable"]:
        st.divider()
        create_tab, edit_tab, delete_tab, import_tab = st.tabs(
            ["＋ Create record", "✎ Update record", "⌫ Delete record", "⇧ Import CSV"]
        )

        # Infer editor widgets from the actual SQLite dataset schema.
        table = cfg["table"]
        schema_frame, _ = read_dataset(dataset_name, limit=1)
        schema = schema_frame.drop(columns=["_record_id"], errors="ignore")
        sample = schema.iloc[0] if not schema.empty else pd.Series(dtype=object)

        def field_widget(column: str, value, key_prefix: str):
            value_text = "" if pd.isna(value) else str(value)
            if pd.api.types.is_bool_dtype(schema[column].dtype):
                return st.checkbox(column, value=bool(value) if not pd.isna(value) else False, key=f"{key_prefix}_{column}")
            if pd.api.types.is_numeric_dtype(schema[column].dtype):
                try:
                    default = float(value) if value_text else 0.0
                except (TypeError, ValueError):
                    default = 0.0
                return st.number_input(column, value=default, key=f"{key_prefix}_{column}")
            return st.text_input(column, value=value_text, key=f"{key_prefix}_{column}")

        def cast_value(column: str, value):
            dtype = schema[column].dtype
            if value == "" and (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
                return None
            if pd.api.types.is_integer_dtype(dtype):
                return int(value)
            if pd.api.types.is_float_dtype(dtype):
                return float(value)
            if pd.api.types.is_bool_dtype(dtype):
                return bool(value)
            return value

        columns = list(schema.columns)
        with create_tab:
            st.markdown("Add a new row to the selected dataset.")
            with st.form(f"create_{table}"):
                new_values = {}
                cols = st.columns(3)
                for i, column in enumerate(columns):
                    with cols[i % 3]:
                        default = sample[column] if column in sample.index else ""
                        new_values[column] = field_widget(column, default, "new")
                create_submit = st.form_submit_button("Create record", type="primary")
            if create_submit:
                try:
                    create_record(dataset_name, {k: cast_value(k, v) for k, v in new_values.items()})
                    st.success("Record created in the managed database.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Create failed: {exc}")

        with edit_tab:
            st.markdown("Choose a record from the currently loaded first 1,000 rows, then edit its fields.")
            selectable = records if not records.empty else managed
            if not selectable.empty:
                ids = selectable["_record_id"].astype(int).tolist()
                selected_id = st.selectbox("Record ID", ids, key=f"edit_id_{table}")
                selected_row = selectable.loc[selectable["_record_id"].astype(int).eq(int(selected_id))].iloc[0]
                with st.form(f"update_{table}"):
                    edit_values = {}
                    cols = st.columns(3)
                    for i, column in enumerate(columns):
                        with cols[i % 3]:
                            edit_values[column] = field_widget(column, selected_row[column], "edit")
                    update_submit = st.form_submit_button("Save changes", type="primary")
                if update_submit:
                    try:
                        update_record(dataset_name, int(selected_id), {k: cast_value(k, v) for k, v in edit_values.items()})
                        st.success(f"Record {selected_id} updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Update failed: {exc}")
            else:
                st.info("No records are available to update.")

        with delete_tab:
            st.markdown("Deletion is permanent inside the managed SQLite copy. The original source file remains untouched.")
            selectable = records if not records.empty else managed
            if not selectable.empty:
                ids = selectable["_record_id"].astype(int).tolist()
                delete_id = st.selectbox("Record ID to delete", ids, key=f"delete_id_{table}")
                confirm_delete = st.checkbox("I understand this will delete the selected managed record.", key=f"confirm_delete_{table}")
                if st.button("Delete selected record", type="primary", disabled=not confirm_delete):
                    try:
                        delete_record(dataset_name, int(delete_id))
                        st.success(f"Record {delete_id} deleted from the managed database.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
            else:
                st.info("No records are available to delete.")

        with import_tab:
            st.markdown("Import a CSV into the selected managed dataset.")
            upload = st.file_uploader("Choose CSV file", type=["csv"], key=f"upload_{table}")
            import_mode = st.radio("Import mode", ["Append records", "Replace managed dataset"], horizontal=True, key=f"import_mode_{table}")
            if import_mode == "Replace managed dataset":
                st.warning("This replaces the managed SQLite table only. It does not overwrite the original source file.")
            if upload is not None:
                if st.button("Import CSV", type="primary", key=f"import_button_{table}"):
                    try:
                        added = import_csv(dataset_name, upload, append=import_mode == "Append records")
                        st.success(f"Imported {added:,} records.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Import failed: {exc}")
    else:
        st.info("This dataset is generated by the forecasting pipeline and is intentionally read-only. Update source observations, then run `python scripts/train.py` to regenerate it.")
else:
    st.info("The selected dataset is empty or could not be loaded.")

with st.expander("Managed-data change history"):
    history = audit_log(dataset_name)
    if history.empty:
        st.caption("No changes have been recorded yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)

st.divider()
section_header("Data coverage", "Coverage values are taken from the loaded source dataset and generated forecast artifact.")
if not df.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listing rows", f"{len(df):,}")
    c2.metric("Distinct cities", f"{df['City'].nunique():,}" if "City" in df else "—")
    c3.metric("Distinct states / territories", f"{df['State'].nunique():,}" if "State" in df else "—")
    c4.metric("Cities with forecasts", f"{forecasts['City'].nunique():,}" if not forecasts.empty and "City" in forecasts else "0")
    if "State" in df.columns:
        import plotly.express as px
        state_coverage = df.groupby("State", as_index=False).size().rename(columns={"size": "Listing records"}).nlargest(15, "Listing records")
        fig = px.bar(state_coverage.sort_values("Listing records"), x="Listing records", y="State", orientation="h",
                     title="Listing coverage by state · Top 15",
                     color="Listing records", color_continuous_scale=["#b9d7d8", "#147d83", "#0e455b"])
        fig.update_layout(height=330, coloraxis_showscale=False, margin=dict(l=5, r=8, t=42, b=8))
        st.plotly_chart(fig, use_container_width=True)
        leading = state_coverage.sort_values("Listing records", ascending=False).iloc[0]
        st.info(f"**Key takeaway:** {leading['State']} has the largest representation in the source listing sample ({int(leading['Listing records']):,} records). This describes dataset coverage, not market-wide supply.")
else:
    st.info("Listing coverage unavailable because the dataset did not load.")

with st.expander("Available listing columns"):
    if not df.empty:
        st.write(", ".join(map(str, df.columns)))

st.divider()
section_header("Forecasting method", "City HPI forecasts are selected through historical five-year rolling-origin validation.")
st.markdown(
    """
    1. **Prepare each city series.** Read quarterly NHB HPI values in chronological order and require sufficient, complete, positive history.
    2. **Generate historical forecasts.** At each forecast origin, fit each candidate using only HPI available up to that date; predict 20 quarters ahead.
    3. **Select a candidate.** Compare rolling-origin cross-validation error (primarily MAPE), while excluding candidates whose numerical failures/fallback rates exceed the reliability threshold.
    4. **Evaluate a temporal holdout.** Score the chosen candidate at later historical origins that were not used for model selection.
    5. **Build city-value scenarios.** Apply the predicted city HPI growth multiplier to the entered current asking price. Historical forecast errors are also scaled to illustrative rupee values.
    """
)
st.markdown("**Scenario formula**")
st.latex(r"\widehat{V}_{t+5}=V_t\times\frac{\widehat{H}_{t+5}}{H_t}")
st.caption("V is the current asking price used as the starting value. H is the city HPI. The forecast scales the property price in line with city-index growth; it does not estimate a neighborhood or property-specific premium.")

if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected forecast method", str(hpi["method"]).replace("_", " ").title())
    c2.metric("Five-year CV MAPE", f"{hpi['cv_mape']:.2f}%" if isinstance(hpi["cv_mape"], (int, float)) else "—")
    c3.metric("Temporal holdout MAPE", f"{hpi['holdout_mape']:.2f}%" if isinstance(hpi["holdout_mape"], (int, float)) else "—")

st.divider()
section_header("Definitions and assumptions")
st.markdown(
    """
    - **Asking price:** `Price_in_Lakhs` in the listing file; it is not necessarily the transacted price.
    - **HPI:** city-level housing price index. It captures broad market movement, not variation between neighborhoods or individual homes.
    - **Forecast horizon:** five years, or 20 quarterly steps.
    - **Empirical lower / upper scenario:** pooled historical forecast-error context; it is not a formally calibrated property interval.
    - **Investment classification:** an assumption-generated proxy label. Its model probability is confidence in the constructed label, not a probability of profit.
    - **Current-price regression benchmark:** the existing listing regression target is formula-generated and is not observed five-year resale accuracy.
    """
)
with st.expander("What would be needed for a true property-level resale model?"):
    st.write("A dated dataset of completed transactions with sale prices, ideally matched to property attributes and geography, is required to train and validate a model against real resale outcomes.")
    st.write("A production decision process should also consider property condition, legal/title checks, rental yield, taxes, maintenance, financing costs, transaction costs, liquidity and local market comparables.")
st.warning("Use this dashboard for exploratory analysis and market-scenario comparison only. Do not treat the estimate or investment proxy as a guarantee of value, a valuation report, or financial advice.")
