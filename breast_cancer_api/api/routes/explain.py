from fastapi import APIRouter
from schemas.patient import Patient
from services.explain_service import explain

router = APIRouter()

@router.post("/explain")
def explain_route(patient: Patient):
    return explain(patient.dict())