from fastapi import APIRouter
from schemas.patient import Patient
from services.prediction_service import predict

router = APIRouter()

@router.post("/predict")
def predict_route(patient: Patient):
    return predict(patient.dict())