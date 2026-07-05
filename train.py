"""Train the rental price model.

Reads data/rentals_sd.csv, trains a preprocessing + LinearRegression
pipeline, prints test metrics and coefficients, and saves everything
app.py needs at runtime: the fitted pipeline (models/rental_model.pkl)
and models/metrics.json, plus two diagnostic plots under reports/.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # render PNGs without needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "rentals_sd.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

CATEGORICAL = ["sector"]
NUMERIC = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "parking_spots",
    "furnished",
    "age_years",
]
TARGET = "price_dop"


def build_pipeline() -> Pipeline:
    # Preprocessing lives inside the pipeline so it is fitted only on the
    # training split (no data leakage) and ships inside the saved .pkl:
    # app.py feeds raw rows without reimplementing any transformation.
    preprocess = ColumnTransformer(
        [
            ("sector", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", StandardScaler(), NUMERIC),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def print_metrics(mae: float, rmse: float, r2: float) -> None:
    print("Test metrics (20% hold-out):")
    print(f"  MAE:  RD$ {mae:,.0f} -> on average, estimates miss the real rent by this amount")
    print(f"  RMSE: RD$ {rmse:,.0f} -> typical error weighting big misses; the app shows it as the +/- range")
    print(f"  R2:   {r2:.3f} -> the model explains {r2:.0%} of the variation in rents")


def print_coefficients(pipeline: Pipeline) -> None:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    ranked = sorted(zip(names, coefs), key=lambda item: abs(item[1]), reverse=True)

    print("\nCoefficients (DOP), sorted by absolute impact:")
    print("  sector_*: location premium vs. the average sector")
    print("  numeric features: impact per +1 standard deviation (scaled)")
    for name, coef in ranked:
        clean = name.split("__", 1)[1]  # drop the transformer prefix
        print(f"  {clean:<30} {coef:>+12,.0f}")


def save_artifacts(pipeline: Pipeline, df: pd.DataFrame, mae, rmse, r2, n_train, n_test) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "rental_model.pkl")

    # metrics.json is the contract with app.py: RMSE feeds the confidence
    # range and the sector averages feed the comparison chart.
    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "avg_price_by_sector": {
            sector: round(float(avg), 2)
            for sector, avg in df.groupby("sector")[TARGET].mean().items()
        },
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)


def save_plots(y_test: pd.Series, y_pred: np.ndarray) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="none")
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual rent (RD$)")
    ax.set_ylabel("Predicted rent (RD$)")
    ax.set_title("Actual vs. predicted rent (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Residuals should look like a shapeless cloud around zero; visible
    # structure would mean the linearity assumption is breaking down.
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted rent (RD$)")
    ax.set_ylabel("Residual (actual - predicted, RD$)")
    ax.set_title("Residuals vs. predicted rent (test set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print_metrics(mae, rmse, r2)
    print_coefficients(pipeline)
    save_artifacts(pipeline, df, mae, rmse, r2, len(X_train), len(X_test))
    save_plots(y_test, y_pred)

    print(f"\nSaved: {MODELS_DIR / 'rental_model.pkl'}")
    print(f"Saved: {MODELS_DIR / 'metrics.json'}")
    print(f"Saved: {REPORTS_DIR / 'actual_vs_predicted.png'}")
    print(f"Saved: {REPORTS_DIR / 'residuals.png'}")


if __name__ == "__main__":
    main()
