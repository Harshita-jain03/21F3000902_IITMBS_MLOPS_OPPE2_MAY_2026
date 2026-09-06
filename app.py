from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime, timezone
import logging
import json


# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger("heart-disease-api")


# Load trained model
model = joblib.load("models/heart_disease_model.pkl")
best_model = model.best_estimator_


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production API for heart disease prediction",
    version="1.0.0"
)


# Input schema
class PatientData(BaseModel):
    sno: int
    age: float
    gender: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# Health check
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = best_model.predict(input_data)[0]

    timestamp = datetime.now(timezone.utc).isoformat()

    # Log input features, prediction and timestamp
    log_data = {
        "event": "heart_disease_prediction",
        "input": data.model_dump(),
        "prediction": str(prediction),
        "timestamp": timestamp
    }

    logger.info(json.dumps(log_data))

    return {
        "prediction": str(prediction),
        "timestamp": timestamp
    }
