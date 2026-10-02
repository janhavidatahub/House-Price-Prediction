"""
train.py — Machine Learning Training Script for HousePriceBot
==============================================================
This script:
  1. Loads the California Housing dataset from scikit-learn
  2. Checks dataset shape and missing values
  3. Splits data into training and test sets
  4. Trains a RandomForestRegressor
  5. Evaluates the model (MAE, MSE, RMSE, R²)
  6. Saves the trained model as a .joblib file
  7. Saves evaluation metrics as metrics.json
  8. Generates and saves an Actual vs Predicted plot

Run this script ONCE before starting the FastAPI service:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Paths ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "house_price_model.joblib")
METRICS_PATH  = os.path.join(ARTIFACTS_DIR, "metrics.json")
PLOT_PATH     = os.path.join(ARTIFACTS_DIR, "actual_vs_predicted.png")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("HousePriceBot — Model Training")
print("=" * 60)
print("\n[1/7] Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame   # Pandas DataFrame with features + target

print(f"  Dataset shape : {df.shape}")
print(f"  Features      : {list(housing.feature_names)}")
print(f"  Target        : MedHouseVal  (median house value in $100,000 units)")

# ─── 2. Check Missing Values ──────────────────────────────────────────────────
print("\n[2/7] Checking for missing values...")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values found. ✓")
else:
    print(f"  Missing values detected:\n{missing[missing > 0]}")

# ─── 3. Separate Features and Target ─────────────────────────────────────────
print("\n[3/7] Separating features (X) and target (y)...")

# The eight features used by the California Housing dataset
FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

X = df[FEATURE_COLUMNS]
y = df["MedHouseVal"]   # Target: median house value in $100,000 units

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  y range : {y.min():.2f} – {y.max():.2f}  (multiply by 100,000 for USD)")

# ─── 4. Train-Test Split ──────────────────────────────────────────────────────
print("\n[4/7] Splitting data (80% train / 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")

# ─── 5. Train Models ─────────────────────────────────────────────────────────
print("\n[5/7] Training models...")

# --- Primary model: Random Forest Regressor ---
print("  → Training RandomForestRegressor (n_estimators=200) ...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# --- Optional comparison models ---
print("  → Training LinearRegression (for comparison) ...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("  → Training Ridge Regression (for comparison) ...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_preds = ridge_model.predict(X_test)

# ─── 6. Evaluate Models ───────────────────────────────────────────────────────
print("\n[6/7] Evaluating models...")

def evaluate(name, y_true, y_pred):
    """Calculate and print MAE, MSE, RMSE, R² for a given model."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)          # RMSE = sqrt(MSE)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MSE  : {mse:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")
    return {"model": name, "MAE": round(mae, 4), "MSE": round(mse, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4)}

rf_metrics    = evaluate("Random Forest Regressor", y_test, rf_preds)
lr_metrics    = evaluate("Linear Regression",       y_test, lr_preds)
ridge_metrics = evaluate("Ridge Regression",        y_test, ridge_preds)

# Save all metrics to JSON (Random Forest is the primary model)
all_metrics = {
    "primary_model": "RandomForestRegressor",
    "results": [rf_metrics, lr_metrics, ridge_metrics]
}
with open(METRICS_PATH, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n  Metrics saved → {METRICS_PATH}")

# ─── 7. Save Primary Model ────────────────────────────────────────────────────
print("\n[7/7] Saving Random Forest model ...")
joblib.dump(rf_model, MODEL_PATH)
print(f"  Model saved → {MODEL_PATH}")

# ─── 8. Generate Actual vs Predicted Plot ────────────────────────────────────
print("\nGenerating Actual vs Predicted plot ...")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, alpha=0.3, s=10, color="steelblue", label="Predictions")

# Perfect-prediction reference line
min_val = min(y_test.min(), rf_preds.min())
max_val = max(y_test.max(), rf_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")

plt.xlabel("Actual Median House Value ($100k units)")
plt.ylabel("Predicted Median House Value ($100k units)")
plt.title("Random Forest — Actual vs Predicted House Prices\n(California Housing Dataset)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()
print(f"  Plot saved → {PLOT_PATH}")

# ─── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training complete!")
print(f"  Model   : {MODEL_PATH}")
print(f"  Metrics : {METRICS_PATH}")
print(f"  Plot    : {PLOT_PATH}")
print("\nYou can now start the FastAPI service:")
print("  uvicorn app:app --reload --port 8000")
print("=" * 60)
