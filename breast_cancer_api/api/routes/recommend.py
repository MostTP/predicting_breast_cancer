from fastapi import APIRouter
from schemas.patient import Patient
from services.recommendation_service import recommend

router = APIRouter()

@router.post("/recommend")
def recommend_route(patient: Patient):
    return recommend(patient.dict())