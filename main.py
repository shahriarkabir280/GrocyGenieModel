from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from model import (
    load_or_train_model,
    predict_user_input,
    retrain_model_with_feedback,
    store_predictions,
    fetch_feedback_for_user,
)
import pandas as pd

app = FastAPI()
model = load_or_train_model()  # load model at startup

# Pydantic schema for user input
class Family(BaseModel):
    adult_male: int
    adult_female: int
    child: int

class UserInput(BaseModel):
    family: Family
    region: str
    season: str
    event: str
    stock: Dict[str, float]
    user_id: str

@app.post("/predict")
def predict(user_input: UserInput):
    global model
    try:
        input_dict = user_input.dict()
        results = predict_user_input(input_dict, model)
        store_predictions(input_dict["user_id"], results, input_dict)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_model():
    global model
    model = load_or_train_model(force_train=True)
    return {"message": "Model trained successfully."}


@app.post("/retrain/{user_id}")
def retrain(user_id: str):
    global model
    try:
        model = retrain_model_with_feedback(user_id)
        return {"message": f"Model retrained for user: {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/{user_id}")
def get_feedback(user_id: str):
    df = fetch_feedback_for_user(user_id)
    return df.to_dict(orient="records")
