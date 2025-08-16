from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime, date
import uuid
import model  #existing model.py
import joblib
import tensorflow as tf

app = FastAPI()

# Load or train model at startup
ml_model = model.load_or_train_model()
SCALER_PATH = "/tmp/scaler.pkl"


# Request & Response Schemas

class FamilyInput(BaseModel):
    adult_male: int
    adult_female: int
    child: int

class StockAdditionInput(BaseModel):
    """Schema for adding a new stock item and getting a prediction."""
    user_id: str
    product_name: str
    quantity: float = Field(..., gt=0) # Quantity must be greater than 0
    unit: str = 'kg'
    purchase_date: date
    # Optional context fields; if not provided, they can be fetched from the user's profile
    region: Optional[str] = None
    season: Optional[str] = None
    event: Optional[str] = None
    family: Optional[FamilyInput] = None

class RePredictionInput(BaseModel):
    user_id: str
    product_name: str
    quantity: float
    unit: str
    # The following are needed for the model to predict consumption
    region: str
    season: str
    event: str
    family: FamilyInput

class RetrainRequest(BaseModel):
    user_id: str

class FeedbackInput(BaseModel):
    stock_id: uuid.UUID
    actual_finish_date: date

# -------------------------
# API Routes
# -------------------------

@app.get("/")
def read_root():
    return {"message": "✅ GrocyGenie API is running."}


@app.post("/stock/add")
def add_stock_and_predict(input_data: StockAdditionInput):
    """
    Adds a new stock item for a user and predicts its depletion date.
    This is the primary endpoint for making predictions.
    """
    try:
        # The model function now handles both prediction and DB insertion
        result = model.predict_and_record_stock(input_data)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create stock record or make prediction.")

        return {
            "message": "Stock added and prediction complete.",
            "stock_id": result['stock_id'],
            "product_name": input_data.product_name,
            "predicted_finish_date": result['predicted_finish_date']
        }

    except Exception as e:
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/feedback")
def record_feedback(feedback_data: FeedbackInput):
    """
    Receives feedback from the user about the actual finish date of a stock item.
    """
    success = model.record_actual_finish_date(
        feedback_data.stock_id,
        feedback_data.actual_finish_date
    )
    if success:
        return {"message": "Feedback recorded successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Stock ID {feedback_data.stock_id} not found or update failed.")


@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    """
    Triggers model retraining using verified feedback for a specific user.
    """
    result = model.retrain_model_with_feedback(request.user_id)
    if result["success"]:
        # Reload model and scaler globally for future predictions
        # model.ml_model = tf.keras.models.load_model(model.MODEL_PATH)
        # model.scaler = joblib.load(SCALER_PATH)
        return {"message": result["message"]}
    else:
        raise HTTPException(status_code=400, detail=result["message"])
    
@app.post("/re-predict")
def recalculate_prediction(input_data: RePredictionInput):
    """
    Calculates a new depletion date without touching the database.
    This is a pure calculation service.
    """
    try:
        # Call the new model function that only does calculations
        new_finish_date = model.recalculate_depletion(input_data)
        
        return {
            "predicted_finish_date": new_finish_date
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
