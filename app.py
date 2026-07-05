"""Streamlit app: rental price estimator for Santo Domingo, DR.

UI text is in Spanish (end users are Dominican real estate agents);
code, comments and identifiers stay in English per the project language
policy. The app only loads artifacts produced by train.py -- it never
trains anything, which is what keeps the free cloud deploy simple.

Layout: SaaS-style dark UI. Hero header, two-column body (property
parameters on the left, appraisal result card on the right) and a
full-width market chart. Styling lives in _CSS plus
.streamlit/config.toml; chart colors in COLORS must stay visually in
sync with the CSS custom properties.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

COLORS = {
    "card": "#1E293B",
    "accent": "#3B82F6",
    "success": "#10B981",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "bar_idle": "#334155",
    "grid": "rgba(148, 163, 184, 0.10)",
}

_CSS = """
<style>

:root {
    --card: #1E293B;
    --card-border: rgba(148, 163, 184, 0.14);
    --accent: #3B82F6;
    --accent-strong: #2563EB;
    --success: #10B981;
    --text: #F8FAFC;
    --muted: #94A3B8;
}

/* System font stack (GitHub/Vercel style): premium look with zero
   network dependency -- external font CDNs hang on TLS-inspected
   networks and freeze the first paint. */
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp button, h1, h2, h3 {
    font-family: 'Segoe UI Variable Display', 'Segoe UI', -apple-system,
        'SF Pro Display', Roboto, 'Helvetica Neue', sans-serif;
}
.stMarkdown div {
    font-family: 'Segoe UI Variable Display', 'Segoe UI', -apple-system,
        'SF Pro Display', Roboto, 'Helvetica Neue', sans-serif;
}
/* Streamlit renders some icons with a glyph font; keep it intact. */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

header[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }

[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1180px;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 2.4rem 2.5rem 3rem;
}

/* ---------- Hero ---------- */
.hero { text-align: center; margin-bottom: 2.4rem; }
.hero-badge {
    display: inline-block; padding: 0.32rem 0.95rem; border-radius: 999px;
    background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.35);
    color: #93C5FD; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em;
    margin-bottom: 1.1rem;
}
.hero h1 {
    font-size: 2.7rem; font-weight: 800; letter-spacing: -0.03em;
    color: var(--text); margin: 0 0 0.55rem; line-height: 1.12;
}
.hero h1 .accent { color: var(--accent); }
.hero p { color: var(--muted); font-size: 1.03rem; max-width: 580px; margin: 0 auto; }

/* ---------- Cards (containers created with key="card-...") ---------- */
[class*="st-key-card-"] {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.6rem 1.7rem;
    box-shadow: 0 18px 44px rgba(2, 6, 23, 0.38);
}
.st-key-card-form, .st-key-card-result { min-height: 432px; }
.card-title {
    font-size: 0.76rem; font-weight: 700; letter-spacing: 0.13em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 1.15rem;
}

/* ---------- Inputs ---------- */
[data-testid="stWidgetLabel"] p { color: var(--muted); font-size: 0.84rem; font-weight: 500; }
[data-testid="stNumberInputContainer"], div[data-baseweb="select"] > div {
    background: #0B1526 !important;
    border: 1px solid rgba(148, 163, 184, 0.20) !important;
    border-radius: 12px !important;
    transition: border-color 0.15s ease;
}
[data-testid="stNumberInputContainer"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
}
[data-testid="stNumberInputContainer"] input,
div[data-baseweb="select"] input { background: transparent !important; color: var(--text) !important; }
[data-testid="stNumberInputContainer"] button {
    background: rgba(148, 163, 184, 0.08) !important;
    border: none !important; color: var(--muted) !important;
}
[data-testid="stNumberInputContainer"] button:hover {
    background: rgba(59, 130, 246, 0.30) !important; color: #fff !important;
}

/* ---------- Primary button ---------- */
.stButton { width: 100%; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
    color: #fff; font-weight: 600; font-size: 0.98rem;
    border: none; border-radius: 12px; padding: 0.82rem 1rem;
    box-shadow: 0 10px 26px rgba(37, 99, 235, 0.35);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
    margin-top: 0.4rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 32px rgba(37, 99, 235, 0.48);
    color: #fff;
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus:not(:active) { color: #fff; }

/* ---------- Result card ---------- */
.result-label {
    font-size: 0.76rem; font-weight: 700; letter-spacing: 0.13em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 0.7rem;
}
.result-price {
    font-size: 3.3rem; font-weight: 800; letter-spacing: -0.035em;
    color: var(--text); line-height: 1.05; margin-bottom: 0.55rem;
}
.result-price .per { font-size: 1.05rem; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.result-chip {
    display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; margin-bottom: 1.2rem;
}
.chip-positive { background: rgba(16, 185, 129, 0.14); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); }
.chip-negative { background: rgba(148, 163, 184, 0.12); color: var(--muted); border: 1px solid rgba(148, 163, 184, 0.28); }
.result-divider { height: 1px; background: rgba(148, 163, 184, 0.14); margin: 0.2rem 0 1.1rem; }
.result-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.5rem 0; gap: 1rem;
}
.result-row + .result-row { border-top: 1px solid rgba(148, 163, 184, 0.08); }
.result-row .k { color: var(--muted); font-size: 0.86rem; }
.result-row .v { color: var(--text); font-size: 0.94rem; font-weight: 600; white-space: nowrap; }

.empty-state { text-align: center; padding: 4.2rem 1rem; color: var(--muted); }
.empty-state .icon { font-size: 2rem; margin-bottom: 0.8rem; opacity: 0.8; }
.empty-state p { font-size: 0.94rem; max-width: 300px; margin: 0 auto; line-height: 1.55; }

/* ---------- Chart card ---------- */
.chart-sub { color: var(--muted); font-size: 0.88rem; margin: -0.6rem 0 0.4rem; }

/* ---------- Footer ---------- */
.footer-note {
    text-align: center; color: var(--muted); font-size: 0.8rem;
    margin-top: 2.2rem; line-height: 1.6; opacity: 0.85;
}

/* ---------- Responsive ---------- */
@media (max-width: 992px) {
    [data-testid="stMainBlockContainer"] { padding: 1.5rem 1.1rem 2.2rem; }
    .hero h1 { font-size: 2rem; }
    .result-price { font-size: 2.5rem; }
    [class*="st-key-card-"] { padding: 1.2rem 1.2rem; }
}
</style>
"""


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "rental_model.pkl")


@st.cache_data
def load_metrics() -> dict:
    with open(BASE_DIR / "models" / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


def format_dop(amount: float) -> str:
    return f"RD$ {amount:,.0f}"


def format_dop_md(amount: float) -> str:
    # For markdown text only: a PAIR of "$" in st.markdown/st.write
    # starts LaTeX math and mangles the text, so escape the symbol.
    return f"RD\\$ {amount:,.0f}"


def format_dop_html(amount: float) -> str:
    # For HTML blocks rendered through st.markdown: the "&#36;" entity
    # is immune both to markdown-math and to raw-HTML passthrough.
    return f"RD&#36; {amount:,.0f}"


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">TASACIÓN INTELIGENTE</div>
            <h1>Tasador de Alquileres<span class="accent"> — Santo Domingo</span></h1>
            <p>Estime el precio mensual de alquiler de una propiedad en segundos,
            con un modelo de machine learning calibrado por sector.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_form(sectors: list[str]) -> dict:
    """Render the parameter card; return the current listing inputs."""
    st.markdown('<div class="card-title">Parámetros del inmueble</div>', unsafe_allow_html=True)
    sector = st.selectbox("Sector", sectors)
    col_a, col_b = st.columns(2)
    with col_a:
        area_m2 = st.number_input("Área (m²)", min_value=20, max_value=500, value=100)
        bedrooms = st.number_input("Habitaciones", min_value=1, max_value=6, value=2)
        bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=2)
    with col_b:
        parking_spots = st.number_input("Parqueos", min_value=0, max_value=4, value=1)
        age_years = st.number_input("Antigüedad (años)", min_value=0, max_value=60, value=5)
        furnished = st.checkbox("Amueblado")

    return {
        "sector": sector,
        "area_m2": area_m2,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spots": parking_spots,
        "furnished": int(furnished),
        "age_years": age_years,
    }


def appraise(model, listing: dict) -> float:
    # Column names must match the training schema exactly: the pipeline
    # inside the .pkl does all preprocessing (one-hot, scaling) itself.
    return float(model.predict(pd.DataFrame([listing]))[0])


def render_result_card(metrics: dict) -> None:
    appraisal = st.session_state.get("appraisal")
    st.markdown('<div class="card-title">Resultado</div>', unsafe_allow_html=True)

    if appraisal is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">—</div>
                <p>Complete los parámetros del inmueble y presione
                <b>Tasar propiedad</b> para obtener la estimación.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    estimate = appraisal["estimate"]
    # A linear model can extrapolate below zero on extreme inputs
    # (tiny, very old units in low-price sectors). Better to say so
    # than to show a RD$ 0 appraisal.
    if estimate <= 0:
        st.warning(
            "Esta combinación de características está fuera del rango "
            "típico del mercado y el modelo no puede tasarla con "
            "confianza. Ajuste los valores e intente de nuevo."
        )
        return

    rmse = metrics["rmse"]
    low, high = max(estimate - rmse, 0.0), estimate + rmse
    sector = appraisal["listing"]["sector"]
    sector_avg = metrics["avg_price_by_sector"][sector]
    delta_pct = (estimate - sector_avg) / sector_avg * 100
    chip_class = "chip-positive" if delta_pct >= 0 else "chip-negative"

    st.markdown(
        f"""
        <div class="result-label">Precio estimado</div>
        <div class="result-price">{format_dop_html(estimate)}<span class="per"> /mes</span></div>
        <div class="result-chip {chip_class}">{delta_pct:+.1f}% vs promedio de {sector}</div>
        <div class="result-divider"></div>
        <div class="result-row">
            <span class="k">Rango de confianza</span>
            <span class="v">{format_dop_html(low)} — {format_dop_html(high)}</span>
        </div>
        <div class="result-row">
            <span class="k">Precisión del modelo (R²)</span>
            <span class="v">{metrics["r2"]:.0%}</span>
        </div>
        <div class="result-row">
            <span class="k">Error medio (MAE)</span>
            <span class="v">± {format_dop_html(metrics["mae"])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_market_chart(metrics: dict) -> go.Figure:
    """Bar chart of average rent per sector, highlighting the appraised
    sector and marking the current estimate as a reference line."""
    averages = sorted(
        metrics["avg_price_by_sector"].items(), key=lambda item: item[1], reverse=True
    )
    names = [name for name, _ in averages]
    values = [value for _, value in averages]

    appraisal = st.session_state.get("appraisal")
    selected = appraisal["listing"]["sector"] if appraisal else None
    bar_colors = [
        COLORS["accent"] if name == selected else COLORS["bar_idle"] for name in names
    ]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=bar_colors,
            hovertemplate="%{x}: RD$ %{y:,.0f}<extra></extra>",
        )
    )
    if appraisal and appraisal["estimate"] > 0:
        fig.add_hline(
            y=appraisal["estimate"],
            line_dash="dash",
            line_color=COLORS["success"],
            line_width=2,
            annotation_text=f"Estimación: {format_dop(appraisal['estimate'])}",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["success"], size=13),
        )
    fig.update_layout(
        barcornerradius=8,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Segoe UI Variable Display, Segoe UI, -apple-system, sans-serif",
            color=COLORS["muted"],
            size=13,
        ),
        margin=dict(l=10, r=10, t=24, b=10),
        height=380,
        yaxis=dict(gridcolor=COLORS["grid"], tickformat=",.0f", zeroline=False),
        xaxis=dict(showgrid=False),
        hoverlabel=dict(bgcolor=COLORS["card"], font_color=COLORS["text"]),
    )
    return fig


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-note">
            Estimación orientativa, no constituye tasación oficial.<br>
            Versión demo entrenada con 500 registros sintéticos calibrados
            al mercado de alquileres de Santo Domingo.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Tasador de Alquileres - Santo Domingo",
    layout="wide",
)
st.markdown(_CSS, unsafe_allow_html=True)

model = load_model()
metrics = load_metrics()
sector_averages = metrics["avg_price_by_sector"]

render_hero()

col_form, col_result = st.columns([1.05, 0.95], gap="large")
with col_form:
    with st.container(key="card-form"):
        listing = render_form(sorted(sector_averages))
        if st.button("Tasar propiedad", type="primary", width="stretch"):
            st.session_state["appraisal"] = {
                "listing": listing,
                "estimate": appraise(model, listing),
            }
with col_result:
    with st.container(key="card-result"):
        render_result_card(metrics)

with st.container(key="card-chart"):
    st.markdown('<div class="card-title">Panorama del mercado</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-sub">Alquiler promedio por sector (RD&#36;/mes)</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_market_chart(metrics),
        width="stretch",
        config={"displayModeBar": False},
    )

render_footer()
