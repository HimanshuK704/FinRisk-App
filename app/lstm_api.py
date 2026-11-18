from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app import lstm_engine

router=APIRouter(prefix="/api/lstm",tags=["LSTM Deep Learning"])

class TrainRequest(BaseModel):
    period: str=Field(default="5y",pattern=r"^(1y|2y|5y)$")
    max_epochs: int=Field(default=20,ge=1,le=50)
    batch_size: int=Field(default=32,ge=8,le=128)

@router.get("/status")
def get_lstm_status():
    return lstm_engine.status()

@router.get("/metrics")
def get_lstm_metrics():
    try: return lstm_engine.metrics()
    except FileNotFoundError as e: raise HTTPException(404,str(e)) from e

@router.post("/train")
def train_lstm(req:TrainRequest):
    try: return lstm_engine.train_model(req.period,req.max_epochs,req.batch_size)
    except Exception as e: raise HTTPException(400,str(e)) from e

@router.post("/predict")
def predict_lstm():
    try: return lstm_engine.predict_latest()
    except FileNotFoundError as e: raise HTTPException(404,str(e)) from e
    except Exception as e: raise HTTPException(400,str(e)) from e
