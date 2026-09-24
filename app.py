from datetime import date
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import model

app = FastAPI(
    title="GrocyGenie Model API",
    description="Predict grocery depletion dates from household context and stock quantities.",
    version="1.0.0",
)

Region = Literal["urban", "rural"]
Season = Literal["winter", "spring", "summer", "autumn"]
Event = Literal["normal", "fasting", "guests", "sickness", "travel", "meal_off"]


class FamilyInput(BaseModel):
    adult_male: int = Field(ge=0)
    adult_female: int = Field(ge=0)
    child: int = Field(ge=0)


class StockAdditionInput(BaseModel):
    user_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit: str = "kg"
    purchase_date: date
    region: Region | None = None
    season: Season | None = None
    event: Event | None = None
    family: FamilyInput | None = None


class RePredictionInput(BaseModel):
    product_name: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit: str = "kg"
    region: Region
    season: Season
    event: Event
    family: FamilyInput


class RetrainRequest(BaseModel):
    user_id: str = Field(min_length=1)


class FeedbackInput(BaseModel):
    stock_id: uuid.UUID
    actual_finish_date: date


@app.on_event("startup")
def warm_model() -> None:
    model.load_model()


@app.get("/")
def read_root():
    return {"message": "GrocyGenie API is running."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model/info")
def model_info():
    return model.get_model_metadata()


@app.post("/stock/add")
def add_stock_and_predict(input_data: StockAdditionInput):
    try:
        result = model.predict_and_record_stock(input_data)
        return {
            "message": "Stock added and prediction complete.",
            "stock_id": result["stock_id"],
            "product_name": input_data.product_name,
            "predicted_finish_date": result["predicted_finish_date"],
            "daily_consumption": round(result["daily_consumption"], 4),
            "days_to_finish": round(result["days_to_finish"], 2),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to add stock and predict depletion.") from exc


@app.post("/feedback")
def record_feedback(feedback_data: FeedbackInput):
    try:
        success = model.record_actual_finish_date(
            feedback_data.stock_id,
            feedback_data.actual_finish_date,
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if success:
        return {"message": "Feedback recorded successfully."}
    raise HTTPException(status_code=404, detail=f"Stock ID {feedback_data.stock_id} not found or update failed.")


@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    try:
        result = model.retrain_model_with_feedback(request.user_id)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result["success"]:
        return {"message": result["message"]}
    raise HTTPException(status_code=400, detail=result["message"])


@app.post("/re-predict")
def recalculate_prediction(input_data: RePredictionInput):
    try:
        new_finish_date = model.recalculate_depletion(input_data)
        return {"predicted_finish_date": new_finish_date}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

