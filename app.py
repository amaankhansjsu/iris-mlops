# app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# 1. Load the trained model
model = joblib.load("model.pkl")

# 2. Define input schema
class PredictRequest(BaseModel):
    inputs: List[List[float]]  # e.g. [[5.1,3.5,1.4,0.2]]

app = FastAPI()

@app.post("/predict")
def predict(req: PredictRequest):
    preds = model.predict(req.inputs).tolist()
    return {"predictions": preds}
