from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

app = FastAPI(title="Road Accident Count Predictor API")

class AccidentRequest(BaseModel):
    city: str
    cause_category: str
    cause_subcategory: str
    outcome: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Road Accident Count Prediction API!"}

@app.post("/predict")
def predict_count(data: AccidentRequest):
    try:
        city_enc = encoders["city"].transform([data.city])[0]
        cause_enc = encoders["cause"].transform([data.cause_category])[0]
        subcause_enc = encoders["subcause"].transform([data.cause_subcategory])[0]
        outcome_enc = encoders["outcome"].transform([data.outcome])[0]

        input_data = np.array([[city_enc, cause_enc, subcause_enc, outcome_enc]])
        prediction = model.predict(input_data)
        return {"predicted_count": round(float(prediction[0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
