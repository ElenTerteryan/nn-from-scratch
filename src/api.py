from fastapi import FastAPI, HTTPException
from typing import List, Union
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

from .schemas import HeartDiseaseRequest

src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from nn_from_scratch import NeuralNetwork

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSING_PATH = os.path.join(BASE_DIR, "model", "preprocessing.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSING_PATH, "rb") as f:
    artifacts = pickle.load(f)

encoder = artifacts["encoder"]
std_scaler = artifacts["std_scaler"]
mm_scaler = artifacts["mm_scaler"]
cat_cols = artifacts["cat_cols"]
num_cols = artifacts["num_cols"]
float_cols = artifacts["float_cols"]

app = FastAPI(
    title="Heart Disease Detector API",
    version="0.1",
    description="API for predicting heart disease risk using a neural network",
)

@app.post("/predict", response_model=dict)
async def predict(data: Union[HeartDiseaseRequest, List[HeartDiseaseRequest]]):
    try:
        if isinstance(data, list):
            df = pd.DataFrame([item.dict() for item in data])
        else:
            df = pd.DataFrame([data.dict()])

        df_enc = encoder.transform(df, cat_cols)
        df_std = std_scaler.transform(df_enc, num_cols)
        df_scaled = mm_scaler.transform(df_std, float_cols)

        predictions = model.forward(df_scaled.values).ravel()

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def health_check():
    return {"version": "Heart-disease detector 0.1", "status": "OK"}
