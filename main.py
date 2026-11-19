
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Credit Card Default Prediction API")

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "champion_model.pkl")
pipeline = joblib.load(BASE_DIR / "preprocessing_pipeline.pkl")

FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
    'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
    'PAY_AMT5', 'PAY_AMT6', 'TOTAL_BILL', 'TOTAL_PAYMENT'
]

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "API is running successfully"}

@app.post("/predict")
def predict(data: PredictionInput):

    if len(data.features) != 25:
        raise HTTPException(status_code=400,
                            detail="Exactly 25 features are required")

    df = pd.DataFrame([data.features], columns=FEATURES)
    processed = pipeline.transform(df)

    prediction = model.predict(processed)[0]
    probabilities = model.predict_proba(processed)[0]

    return {
        "prediction": int(prediction),
        "probabilities": probabilities.tolist()
    }
