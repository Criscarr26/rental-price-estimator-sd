"""Streamlit app: rental price estimator for Santo Domingo, DR.

UI text is in Spanish (end users are Dominican real estate agents);
code, comments and identifiers stay in English per the project language
policy. The app only loads artifacts produced by train.py -- it never
trains anything, which is what keeps the free cloud deploy simple.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


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
    # For st.write/st.markdown only: a PAIR of "$" in markdown starts
    # LaTeX math and mangles the text, so escape the currency symbol.
    return f"RD\\$ {amount:,.0f}"


st.set_page_config(
    page_title="Tasador de Alquileres - Santo Domingo", layout="centered"
)

model = load_model()
metrics = load_metrics()
sector_averages = metrics["avg_price_by_sector"]

st.title("Tasador de Alquileres — Santo Domingo")
st.write(
    "Estime el precio mensual de alquiler de una propiedad "
    "según el sector y sus características."
)

col_left, col_right = st.columns(2)
with col_left:
    sector = st.selectbox("Sector", sorted(sector_averages))
    area_m2 = st.number_input("Área (m²)", min_value=20, max_value=500, value=100)
    bedrooms = st.number_input("Habitaciones", min_value=1, max_value=6, value=2)
    bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=2)
with col_right:
    parking_spots = st.number_input("Parqueos", min_value=0, max_value=4, value=1)
    age_years = st.number_input(
        "Antigüedad (años)", min_value=0, max_value=60, value=5
    )
    furnished = st.checkbox("Amueblado")

if st.button("Tasar propiedad", type="primary"):
    # Column names must match the training schema exactly: the pipeline
    # inside the .pkl does all preprocessing (one-hot, scaling) itself.
    listing = pd.DataFrame(
        [
            {
                "sector": sector,
                "area_m2": area_m2,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "parking_spots": parking_spots,
                "furnished": int(furnished),
                "age_years": age_years,
            }
        ]
    )
    estimate = float(model.predict(listing)[0])
    rmse = metrics["rmse"]

    # A linear model can extrapolate below zero on extreme inputs
    # (tiny, very old units in low-price sectors). Better to say so
    # than to show a RD$ 0 appraisal.
    if estimate <= 0:
        st.warning(
            "Esta combinación de características está fuera del rango "
            "típico del mercado y el modelo no puede tasarla con "
            "confianza. Ajuste los valores e intente de nuevo."
        )
    else:
        low, high = max(estimate - rmse, 0.0), estimate + rmse

        st.subheader("Precio estimado")
        st.metric("Alquiler mensual", format_dop(estimate))
        st.write(
            f"Rango de confianza (± error típico del modelo): "
            f"{format_dop_md(low)} — {format_dop_md(high)}"
        )

        sector_avg = sector_averages[sector]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(
            ["Estimación", f"Promedio {sector}"],
            [estimate, sector_avg],
            color=["#1f77b4", "#b0b7bd"],
        )
        ax.bar_label(bars, labels=[format_dop(estimate), format_dop(sector_avg)])
        ax.set_ylabel("RD$ / mes")
        ax.set_ylim(0, max(estimate, sector_avg) * 1.25)
        st.pyplot(fig)
        plt.close(fig)

st.info("Estimación orientativa, no constituye tasación oficial.")

with st.expander("Sobre el modelo"):
    total = metrics["n_train"] + metrics["n_test"]
    st.write(
        f"- Regresión lineal entrenada con {total} registros de alquiler "
        f"(datos sintéticos calibrados al mercado de Santo Domingo — versión demo)."
    )
    st.write(
        f"- R² en prueba: {metrics['r2']:.2f} · "
        f"Error promedio (MAE): {format_dop_md(metrics['mae'])}"
    )
    st.write(
        "- El rango de confianza mostrado corresponde al error típico "
        "(RMSE) del modelo en datos que nunca vio durante el entrenamiento."
    )
