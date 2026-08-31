import os
import sys
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))

app = FastAPI(
    title="Hyperlocal Air Quality Forecasting API",
    description="Module F: Live Sensor Fusion Inference & Explainability API",
    version="1.0.0",
)

MODEL_PATH = "models/lgbm_model.txt"

# Load model globally on startup
if os.path.exists(MODEL_PATH):
    model = lgb.Booster(model_file=MODEL_PATH)
else:
    model = None


class SensorPayload(BaseModel):
    city: str
    temp_celsius: float
    humidity: float
    wind_speed: float
    wind_deg: float
    pm2_5_lag_1: float
    pm2_5_lag_2: float
    pm2_5_lag_3: float
    pm2_5_lag_6: float
    pm2_5_lag_12: float
    pm2_5_lag_24: float
    pm2_5_roll_mean_3h: float
    pm2_5_roll_std_3h: float
    pm2_5_roll_mean_6h: float
    pm2_5_roll_std_6h: float
    pm2_5_roll_mean_24h: float
    pm2_5_roll_std_24h: float
    sin_hour: float
    cos_hour: float
    u_wind: float
    v_wind: float
    day_of_week: int
    hour: int
    is_weekend: int
    pm10: float


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "project": "bds-07-air-quality",
    }


@app.post("/predict")
def predict_air_quality(payload: SensorPayload):
    if not model:
        raise HTTPException(status_code=500, detail="Model file not loaded!")

    input_data = pd.DataFrame([payload.dict()])
    features = [c for c in model.feature_name()]

    # Ensure all required features exist in payload
    X_infer = input_data[features]

    # Generate Prediction
    prediction = model.predict(X_infer)[0]

    return {
        "city": payload.city,
        "predicted_pm2_5": round(float(prediction), 2),
        "status": "SUCCESS",
    }