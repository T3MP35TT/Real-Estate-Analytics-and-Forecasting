"""Shared dark real-estate theme and reusable UI helpers for Streamlit.

Place this file at ``utils/theme.py``. It uses CSS injected in the app, so no
``.streamlit/config.toml`` is required for Streamlit Community Cloud.

Usage on every page (after set_page_config):
    import streamlit as st
    from utils.theme import apply_theme, page_header, section_header

    st.set_page_config(page_title="Market Insights", page_icon="🏡", layout="wide")
    apply_theme("Market Insights")
    page_header("MARKET INTELLIGENCE", "Market Insights", "Compare asking prices and city trends.")
"""
from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping

import streamlit as st

# Dark forest / graphite / sage / brass palette.
COLORS: dict[str, str] = {
    "forest": "#68B58A",
    "forest_dark": "#10251D",
    "forest_mid": "#3F8D65",
    "forest_light": "#A8D8B9",
    "sage": "#24382F",
    "sage_deep": "#345344",
    "brass": "#D5AD68",
    "brass_light": "#F0D7A7",
    "background": "#0B1210",
    "surface": "#111B17",
    "surface_alt": "#17251E",
    "surface_hover": "#203229",
    "ink": "#E8F0EA",
    "muted": "#A1B2A7",
    "line": "#2C4035",
    "success_bg": "#193527",
    "success_text": "#A8E0B8",
    "warning_bg": "#3A2F1C",
    "warning_text": "#F2D18C",
    "danger_bg": "#3B2221",
    "danger_text": "#F1AAA2",
    "info_bg": "#1D2D3B",
    "info_text": "#ACD3F3",
}

CSS = r"""
<style>
:root {
  color-scheme: dark;
  --re-forest: #68B58A;
  --re-forest-dark: #10251D;
  --re-forest-mid: #3F8D65;
  --re-forest-light: #A8D8B9;
  --re-sage: #24382F;
  --re-sage-deep: #345344;
  --re-brass: #D5AD68;
  --re-brass-light: #F0D7A7;
  --re-bg: #0B1210;
  --re-surface: #111B17;
  --re-surface-alt: #17251E;
  --re-surface-hover: #203229;
  --re-ink: #E8F0EA;
  --re-muted: #A1B2A7;
  --re-line: #2C4035;
}

/* App canvas and layout */
html, body, [data-testid="stAppViewContainer"], .stApp,
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
  background: var(--re-bg) !important;
  color: var(--re-ink) !important;
}
[data-testid="stHeader"] { background: rgba(11,18,16,.92) !important; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container {
  max-width: 1500px;
  padding-top: 1.55rem;
  padding-bottom: 3rem;
}
[data-testid="stBottom"] { background: var(--re-bg) !important; }
[data-testid="stStatusWidget"] { background: var(--re-surface) !important; }

/* Sidebar and native multipage navigation */
[data-testid="stSidebar"] {
  background: #0E1914 !important;
  border-right: 1px solid var(--re-line);
}
[data-testid="stSidebar"] > div:first-child { background: #0E1914 !important; padding-top: 1.1rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { color: var(--re-ink) !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--re-muted) !important; }
[data-testid="stSidebarNav"] { padding-top: .4rem; }
[data-testid="stSidebarNav"] a {
  color: #C9D8CE !important; border-radius: 10px; margin: 3px 0;
  transition: background .15s ease, color .15s ease;
}
[data-testid="stSidebarNav"] a:hover { background: #1B2B22 !important; color: #FFFFFF !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: #244333 !important; color: #E7F5EB !important;
  border-left: 3px solid var(--re-brass); font-weight: 750;
}


/* Sidebar identity card and nested workspace card */
.re-sidebar-identity {
  background: linear-gradient(145deg, #14271D 0%, #101B16 100%);
  border: 1px solid #2C4035;
  border-radius: 14px;
  padding: 14px 13px 13px;
  margin: 8px 0 14px;
  box-shadow: 0 7px 18px rgba(0,0,0,.14);
}
.re-sidebar-brand {
  display: flex;
  align-items: center;
  gap: 9px;
  color: #D7E8DC;
  font-size: 1.02rem;
  font-weight: 800;
  letter-spacing: -.01em;
}
.re-sidebar-subtitle {
  color: #A1B2A7;
  font-size: .75rem;
  margin: 4px 0 12px 1px;
}
.re-sidebar-workspace {
  background: #0D1712;
  border: 1px solid #385143;
  border-radius: 10px;
  padding: 10px 11px;
  margin: 0 0 10px;
}
.re-sidebar-workspace-label {
  color: #A8D8B9;
  font-size: .64rem;
  font-weight: 820;
  letter-spacing: .14em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.re-sidebar-workspace-value {
  color: #E8F0EA;
  font-size: .86rem;
  font-weight: 700;
}
.re-sidebar-disclaimer {
  color: #A1B2A7;
  font-size: .72rem;
  line-height: 1.45;
  padding: 0 2px;
}

/* Typography */
h1, h2, h3, h4, h5, h6 { color: var(--re-ink) !important; letter-spacing: -.02em; }
h1 { font-weight: 780 !important; line-height: 1.13 !important; }
h2, h3 { font-weight: 720 !important; }
p, li, label, legend { color: var(--re-ink); line-height: 1.62; }
a { color: var(--re-forest-light) !important; }
a:hover { color: var(--re-brass-light) !important; }
small, [data-testid="stCaptionContainer"], .stCaption { color: var(--re-muted) !important; }
hr { border-color: var(--re-line) !important; }
[data-testid="stMarkdownContainer"] { color: var(--re-ink); }
code, pre { color: #D3E7DA !important; background: #101A15 !important; border-color: var(--re-line) !important; }

/* KPI / metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(145deg, #14221A 0%, #111B17 100%) !important;
  border: 1px solid var(--re-line) !important;
  border-top: 3px solid var(--re-brass) !important;
  border-radius: 14px; padding: 15px 17px;
  box-shadow: 0 8px 24px rgba(0,0,0,.15); min-height: 105px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: var(--re-muted) !important; font-size: .82rem !important; font-weight: 650 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
  color: var(--re-ink) !important; font-weight: 780 !important; letter-spacing: -.025em;
}
[data-testid="stMetricDelta"] { font-size: .8rem !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button,
[data-testid="stFormSubmitButton"] button, [data-testid="stPopover"] button {
  border-radius: 10px !important; min-height: 2.55rem; font-weight: 680;
  transition: transform .12s ease, background .15s ease, border-color .15s ease;
  background: #1B2A22 !important; color: var(--re-ink) !important;
  border: 1px solid #3A5344 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); border-color: var(--re-forest) !important; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: linear-gradient(105deg, #2D6D4A, #3F8D65) !important;
  color: #FFFFFF !important; border: 1px solid #4A9B70 !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
  background: #4A9B70 !important; border-color: var(--re-brass) !important;
}
.stDownloadButton > button { background: #17251E !important; }
button:focus-visible { outline: 2px solid var(--re-brass) !important; outline-offset: 2px; }

/* Inputs, selectboxes, text areas and BaseWeb menus */
input, textarea, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
  background: #101A15 !important; color: var(--re-ink) !important;
  border-color: #385143 !important; caret-color: var(--re-forest-light) !important;
}
input::placeholder, textarea::placeholder { color: #81968A !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
  background: #101A15 !important; border-color: #385143 !important; border-radius: 9px !important;
  color: var(--re-ink) !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] input,
[data-baseweb="select"] div { color: var(--re-ink); }
[role="listbox"], [data-baseweb="popover"], [data-baseweb="menu"],
ul[role="listbox"] {
  background: #17251E !important; border: 1px solid var(--re-line) !important;
}
[role="option"] { background: #17251E !important; color: var(--re-ink) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: #2A4937 !important; color: #FFFFFF !important; }
[data-baseweb="select"] svg { fill: var(--re-muted) !important; }
input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within { border-color: var(--re-forest) !important; box-shadow: 0 0 0 1px var(--re-forest) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: var(--re-forest) !important; border-color: var(--re-forest) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] { background: #385143 !important; }
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: var(--re-ink) !important; }
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: var(--re-ink) !important; }
[data-testid="stDateInput"] input, [data-testid="stTimeInput"] input { color: var(--re-ink) !important; }

/* Tabs, forms, expanders, alerts and data tables */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--re-line); }
[data-testid="stTabs"] button { color: var(--re-muted) !important; font-weight: 650; }
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--re-forest-light) !important; border-bottom-color: var(--re-brass) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] { color: var(--re-ink) !important; }
div[data-testid="stForm"] { background: #111B17 !important; border: 1px solid var(--re-line); border-radius: 16px; padding: 1.1rem 1.25rem; }
div[data-testid="stExpander"] { background: #111B17 !important; border: 1px solid var(--re-line) !important; border-radius: 13px; }
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] details { color: var(--re-ink) !important; }
[data-testid="stAlert"] { background: #17251E !important; color: var(--re-ink) !important; border: 1px solid var(--re-line) !important; border-radius: 12px; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--re-ink) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { background: #111B17 !important; border: 1px solid var(--re-line); border-radius: 12px; overflow: hidden; }
[data-testid="stPlotlyChart"], [data-testid="stImage"] img { border-radius: 13px; }
[data-testid="stProgressBar"] > div { background: #26382E !important; }
[data-testid="stProgressBar"] > div > div { background: var(--re-forest) !important; }
[data-testid="stPopoverBody"], [data-testid="stDialog"] > div { background: #111B17 !important; color: var(--re-ink) !important; border-color: var(--re-line) !important; }
[data-testid="stFileUploader"] section { background: #111B17 !important; border-color: var(--re-line) !important; }
[data-testid="stFileUploader"] section * { color: var(--re-ink) !important; }

/* Shared theme components */
.re-hero {
  background: radial-gradient(circle at 86% 15%, rgba(213,173,104,.15), transparent 28%),
              linear-gradient(118deg, #10251D 0%, #173A2A 56%, #1F4B34 100%);
  padding: 28px 32px 30px; border-radius: 18px; color: #F4F8F5;
  margin: 0 0 22px; box-shadow: 0 14px 34px rgba(0,0,0,.24);
  border: 1px solid #345544; position: relative; overflow: hidden;
}
.re-hero:after { content: ""; position: absolute; left: 32px; bottom: 0; width: 76px; height: 3px; background: var(--re-brass); }
.re-hero .re-eyebrow { color: var(--re-brass-light); font-size: .69rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 10px; }
.re-hero h1 { color: #F4F8F5 !important; margin: 0 0 10px; font-size: clamp(1.65rem, 2.8vw, 2.25rem); }
.re-hero p { color: #C8DCD0 !important; margin: 0; max-width: 980px; font-size: .98rem; }
.re-section-kicker { color: var(--re-forest-light); font-size: .69rem; font-weight: 820; letter-spacing: .14em; text-transform: uppercase; margin: 8px 0 4px; }
.re-section-description { color: var(--re-muted); font-size: .88rem; margin: 0 0 12px; }
.re-card { height: 100%; padding: 18px 20px; border-radius: 15px; background: linear-gradient(145deg, #17251E, #111B17); border: 1px solid var(--re-line); box-shadow: 0 7px 18px rgba(0,0,0,.15); }
.re-card h3 { color: var(--re-ink) !important; margin: 0 0 9px; font-size: 1.02rem; }
.re-card p { color: #B8C9BE !important; margin: 0; font-size: .92rem; }
.re-pill { display: inline-block; padding: 4px 9px; border-radius: 999px; font-size: .72rem; font-weight: 760; letter-spacing: .015em; background: #24382F; color: #D7E8DC; border: 1px solid #365342; }
.re-pill--success { background: #193527; color: #A8E0B8; border-color: #2B6341; }
.re-pill--warning { background: #3A2F1C; color: #F2D18C; border-color: #65502A; }
.re-pill--danger { background: #3B2221; color: #F1AAA2; border-color: #71403A; }
.re-pill--info { background: #1D2D3B; color: #ACD3F3; border-color: #31516C; }
.re-muted { color: var(--re-muted); font-size: .82rem; }

/* Responsive layout */
@media (max-width: 900px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .re-hero { padding: 23px 22px 26px; }
  .re-hero:after { left: 22px; }
}
</style>
"""


def apply_plotly_theme() -> None:
    """Set consistent dark Plotly defaults without requiring Plotly on all pages."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        palette = [COLORS["forest"], COLORS["brass"], "#85C6A0", "#68A8B0", "#B6A27E", "#A4B3A9", "#C98572"]
        px.defaults.color_discrete_sequence = palette
        px.defaults.color_continuous_scale = [
            [0.0, "#23382E"], [0.48, "#5D9877"], [1.0, COLORS["brass"]]
        ]
        px.defaults.template = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["surface"],
                font=dict(family="Inter, Aptos, Segoe UI, sans-serif", color=COLORS["ink"], size=12),
                colorway=palette,
                xaxis=dict(showgrid=True, gridcolor="#293B31", zerolinecolor=COLORS["line"], linecolor=COLORS["line"], ticks="outside", tickcolor=COLORS["line"]),
                yaxis=dict(showgrid=True, gridcolor="#293B31", zerolinecolor=COLORS["line"], linecolor=COLORS["line"], ticks="outside", tickcolor=COLORS["line"]),
                legend=dict(bgcolor="rgba(17,27,23,.94)", bordercolor=COLORS["line"], borderwidth=1, font=dict(color=COLORS["ink"])),
                margin=dict(l=14, r=14, t=55, b=18),
                hoverlabel=dict(bgcolor="#203229", bordercolor=COLORS["brass"], font_color=COLORS["ink"]),
                title=dict(font=dict(color=COLORS["ink"])),
            )
        )
    except ImportError:
        return


def apply_theme(page_name: str | None = None, *, sidebar_branding: bool = True) -> None:
    """Apply the shared dark CSS and Plotly theme to the current Streamlit page.

    Call after ``st.set_page_config`` on every app/page entry point. The style is
    injected in-page, so it does not depend on a checked-in Streamlit config file.
    """
    st.markdown(CSS, unsafe_allow_html=True)
    apply_plotly_theme()

    if sidebar_branding:
        with st.sidebar:
            workspace_html = (
                f'<div class="re-sidebar-workspace">'
                f'<div class="re-sidebar-workspace-label">Current Workspace</div>'
                f'<div class="re-sidebar-workspace-value">{escape(str(page_name))}</div>'
                f'</div>'
                if page_name else ""
            )
            st.markdown(
                '<div class="re-sidebar-identity">'
                '<div class="re-sidebar-brand">🏡 Estate Intelligence</div>'
                '<div class="re-sidebar-subtitle">Property analytics · Market outlook</div>'
                f'{workspace_html}'
                '<div class="re-sidebar-disclaimer">'
                'Decision-support prototype · Not financial advice'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )



def page_header(eyebrow: str, title: str, subtitle: str, *, icon: str | None = None) -> None:
    """Render the shared dark forest-green page banner."""
    eyebrow_html = escape(str(eyebrow))
    title_html = escape(str(title))
    subtitle_html = escape(str(subtitle))
    icon_html = f"{escape(icon)} " if icon else ""
    st.markdown(
        '<section class="re-hero">'
        f'<div class="re-eyebrow">{eyebrow_html}</div>'
        f'<h1>{icon_html}{title_html}</h1>'
        f'<p>{subtitle_html}</p>'
        '</section>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    """Render a consistent uppercase section label and optional description."""
    st.markdown(f'<div class="re-section-kicker">{escape(str(title))}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="re-section-description">{escape(str(description))}</div>', unsafe_allow_html=True)


def info_card(title: str, body: str, *, pill: str | None = None, status: str = "default") -> None:
    """Render an information card; status can be default/success/warning/danger/info."""
    allowed = {"default", "success", "warning", "danger", "info"}
    tone = status if status in allowed else "default"
    pill_html = ""
    if pill:
        pill_tone = "" if tone == "default" else f" re-pill--{tone}"
        pill_html = f'<div style="margin-bottom:9px"><span class="re-pill{pill_tone}">{escape(str(pill))}</span></div>'
    st.markdown(
        '<div class="re-card">'
        f'{pill_html}<h3>{escape(str(title))}</h3>'
        f'<p>{escape(str(body))}</p></div>',
        unsafe_allow_html=True,
    )


def status_pill(label: str, status: str = "success") -> None:
    """Render a compact, safely escaped status pill."""
    tone = status if status in {"success", "warning", "danger", "info"} else ""
    suffix = f" re-pill--{tone}" if tone else ""
    st.markdown(f'<span class="re-pill{suffix}">{escape(str(label))}</span>', unsafe_allow_html=True)


def format_inr_lakh(value: Any, decimals: int = 1) -> str:
    """Format a numeric value in Indian rupee lakhs; missing values become an em dash."""
    try:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return "—"
        return f"₹{number:,.{decimals}f} lakh"
    except (TypeError, ValueError, OverflowError):
        return "—"


def render_metric_grid(items: Iterable[Mapping[str, Any]], columns: int = 4) -> None:
    """Render native metric widgets in responsive rows.

    Each item should contain ``label`` and ``value``; optional keys are ``delta`` and
    ``help``. A maximum of six columns is allowed for readability.
    """
    values = list(items)
    if not values:
        return
    n_columns = max(1, min(int(columns), 6, len(values)))
    for start in range(0, len(values), n_columns):
        row = st.columns(min(n_columns, len(values) - start))
        for column, item in zip(row, values[start:start + n_columns]):
            kwargs: dict[str, Any] = {}
            if item.get("delta") is not None:
                kwargs["delta"] = item["delta"]
            if item.get("help") is not None:
                kwargs["help"] = item["help"]
            column.metric(str(item.get("label", "")), str(item.get("value", "—")), **kwargs)
