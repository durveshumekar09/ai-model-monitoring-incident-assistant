from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------
# 1. Create FastAPI app
# ---------------------------------------------------------

# FastAPI() creates the API application.
#
# This app will expose endpoints like:
# - /health
# - /predict
app = FastAPI(
    title="Fraud Risk Inference API",
    description="API for fraud-risk prediction using trained XGBoost pipeline",
    version="1.0"
)


# ---------------------------------------------------------
# 2. Project paths
# ---------------------------------------------------------

# Current file:
# ai-model-monitoring-incident-assistant/api/main.py
#
# parents[1] moves from api/ to project root.
BASE_DIR = Path(__file__).resolve().parents[1]

# Path to trained model pipeline
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fraud_xgb_model.joblib"
)


# ---------------------------------------------------------
# 3. Load trained model
# ---------------------------------------------------------

# Load the saved sklearn pipeline.
#
# The pipeline already contains:
# - preprocessing
# - one-hot encoding
# - trained XGBoost model
#
# This same model was trained in train_model.py
print("Loading trained model pipeline...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully")


# ---------------------------------------------------------
# 4. Define request schema
# ---------------------------------------------------------

# BaseModel defines the JSON input format
# expected by the API.
#
# This ensures:
# - proper input structure
# - automatic validation
# - Swagger documentation generation
#
# These fields must match the model features
# used during training.
class FraudPredictionRequest(BaseModel):

    transaction_hour: int

    merchant_category: str

    amount: float

    country: str

    channel: str

    payment_method: str

    account_age_days: int

    prior_chargebacks: int

    login_velocity_24h: int

    device_trust_score: float

    ip_risk_score: float

    merchant_risk_score: float


# ---------------------------------------------------------
# 5. Health-check endpoint
# ---------------------------------------------------------

# This endpoint checks whether the API is running.
#
# URL:
# /health
#
# Example response:
# {
#   "status": "healthy"
# }
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ---------------------------------------------------------
# 6. Prediction endpoint
# ---------------------------------------------------------

# This endpoint accepts transaction data
# and returns fraud-risk predictions.
#
# URL:
# /predict
#
# Request type:
# POST
@app.post("/predict")
def predict_fraud(request: FraudPredictionRequest):

    # ---------------------------------------------------------
    # Step 1: Convert request into dataframe
    # ---------------------------------------------------------

    # request.model_dump()
    # converts incoming JSON into Python dictionary.
    #
    # Example:
    # {
    #   "amount": 5000,
    #   "country": "AE"
    # }
    #
    # pd.DataFrame([ ... ])
    # converts it into one-row dataframe.
    input_df = pd.DataFrame(
        [request.model_dump()]
    )

    # ---------------------------------------------------------
    # Step 2: Generate fraud probability
    # ---------------------------------------------------------

    # predict_proba() returns:
    #
    # [probability_non_fraud, probability_fraud]
    #
    # [:, 1]
    # extracts fraud probability only.
    fraud_probability = (
        pipeline
        .predict_proba(input_df)[:, 1][0]
    )

    # ---------------------------------------------------------
    # Step 3: Convert probability into prediction
    # ---------------------------------------------------------

    # Same threshold used during training.
    fraud_threshold = 0.30

    if fraud_probability >= fraud_threshold:
        prediction = "High Fraud Risk"
    else:
        prediction = "Low Fraud Risk"

    # ---------------------------------------------------------
    # Step 4: Return API response
    # ---------------------------------------------------------

    return {

        "prediction": prediction,

        "fraud_probability": round(
            float(fraud_probability),
            4
        ),

        "fraud_threshold": fraud_threshold
    }


# ---------------------------------------------------------
# 7. Root endpoint
# ---------------------------------------------------------

# Simple welcome endpoint.
@app.get("/")
def home():

    return {
        "message":
            "Fraud Risk Inference API is running"
    }