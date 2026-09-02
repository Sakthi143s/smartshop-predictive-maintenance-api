from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SmartShop AI Predictive Maintenance API",
    description="XGBoost-based machine failure prediction API",
    version="1.0"
)

# --------------------------------------------------
# Features used during XGBoost training
# --------------------------------------------------

FEATURES = [
    "temperature",
    "vibration",
    "current",
    "rpm",
    "load_percentage",
    "temperature_mean_6",
    "vibration_mean_6",
    "current_mean_6",
    "rpm_mean_6",
    "temperature_trend",
    "vibration_trend",
    "current_trend",
    "rpm_trend"
]

# --------------------------------------------------
# Input data structure
# --------------------------------------------------

class MachineData(BaseModel):
    machine_id: str

    temperature: float
    vibration: float
    current: float
    rpm: float
    load_percentage: float

    temperature_mean_6: float
    vibration_mean_6: float
    current_mean_6: float
    rpm_mean_6: float

    temperature_trend: float
    vibration_trend: float
    current_trend: float
    rpm_trend: float


# --------------------------------------------------
# Load trained XGBoost model lazily or on startup
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_predictive_maintenance.pkl")
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please place 'xgboost_predictive_maintenance.pkl' in the project directory."
            )
        model = joblib.load(MODEL_PATH)
    return model


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "message": "SmartShop AI Predictive Maintenance API",
        "status": "running",
        "model_loaded": model_loaded
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "SmartShop AI",
        "model": "XGBoost"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: MachineData):
    try:
        clf = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Convert input to dictionary
    input_data = data.model_dump()

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Select exactly the features used during training
    X = df[FEATURES]

    # Get failure probability
    try:
        failure_probability = float(clf.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Classification threshold (0.50)
    prediction = int(failure_probability >= 0.50)

    # Risk level categorization
    if failure_probability >= 0.75:
        risk_level = "HIGH"
    elif failure_probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "machine_id": data.machine_id,
        "failure_probability": round(failure_probability, 4),
        "failure_percentage": round(failure_probability * 100, 2),
        "prediction": prediction,
        "risk_level": risk_level,
        "prediction_horizon": "24h"
    }
