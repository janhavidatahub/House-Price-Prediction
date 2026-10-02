"""
app.py — FastAPI ML Service for HousePriceBot
=============================================
This service:
  - Loads the trained RandomForestRegressor model at startup
  - Exposes GET /health  → simple health check
  - Exposes POST /predict → receives house features, returns predicted price

The Node.js backend calls POST /predict whenever a user completes
the conversation flow and submits all eight feature values.

Start this service with:
    uvicorn app:app --reload --port 8000
"""

import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "house_price_model.joblib")

# ─── FastAPI application instance ─────────────────────────────────────────────
app = FastAPI(
    title="HousePriceBot ML Service",
    description=(
        "Predicts California median house values using a "
        "trained Random Forest Regressor."
    ),
    version="1.0.0",
)

# ─── Load model at startup ────────────────────────────────────────────────────
# The model is loaded once when the server starts, not on every request.
# This is more efficient than reloading it for each prediction.
model = None

@app.on_event("startup")
def load_model():
    """
    Load the trained scikit-learn model from disk when FastAPI starts.
    If the model file is missing, the service will still start but /predict
    will return a helpful error message telling the user to run train.py first.
    """
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH}")
    else:
        print(
            f"[startup] WARNING: Model file not found at {MODEL_PATH}. "
            "Please run: python train.py"
        )


# ─── Input Schema (Pydantic) ──────────────────────────────────────────────────
# Pydantic validates that all eight required fields are present and numeric.
# FastAPI uses this class to automatically parse and validate the request body.
class HouseFeatures(BaseModel):
    """
    The eight features of the California Housing dataset.
    All values must be numbers (int or float).
    """
    MedInc:     float = Field(..., example=8.3,     description="Median income in block group (in $10,000 units)")
    HouseAge:   float = Field(..., example=25.0,    description="Median house age in block group (years)")
    AveRooms:   float = Field(..., example=5.2,     description="Average number of rooms per household")
    AveBedrms:  float = Field(..., example=1.1,     description="Average number of bedrooms per household")
    Population: float = Field(..., example=800.0,   description="Block group population")
    AveOccup:   float = Field(..., example=3.0,     description="Average number of household members")
    Latitude:   float = Field(..., example=34.05,   description="Block group latitude")
    Longitude:  float = Field(..., example=-118.25, description="Block group longitude")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Node.js backend can call this to verify the ML service is running.
    """
    return {
        "status": "ok",
        "service": "house-price-ml-service",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Predict median house value from the eight California Housing features.

    The California Housing target (MedHouseVal) is expressed in units of
    $100,000. So a predicted value of 4.21 means approximately $421,000.

    Node.js sends the collected user inputs here, and this endpoint
    returns the raw prediction (in $100k units). The Node.js backend
    then multiplies by 100,000 to display the dollar amount to the user.
    """
    # Ensure model is loaded — if train.py was never run, give a clear message
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Please run `python train.py` in the ml-service directory first."
            ),
        )

    # Build a feature array in the exact order the model was trained with.
    # Order MUST match the FEATURE_COLUMNS list used in train.py.
    feature_vector = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]])

    # Run the Random Forest model
    prediction = model.predict(feature_vector)

    # prediction is a 1-element numpy array; extract the scalar value
    predicted_price = float(prediction[0])

    return {
        "predicted_price": round(predicted_price, 4),
        # Include a human-readable dollar estimate for convenience
        "estimated_usd": round(predicted_price * 100_000, 2),
        "note": (
            "predicted_price is in $100,000 units "
            "(California Housing dataset convention). "
            f"Approximately ${predicted_price * 100_000:,.0f} USD."
        ),
    }
