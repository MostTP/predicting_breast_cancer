from fastapi import APIRouter

from core.model_metadata import get_model_metadata, get_model_metrics
from services.disclaimer_service import get_disclaimer

router = APIRouter()


@router.get("/model-info")
def model_info_route():
    return {
        "model_metadata": get_model_metadata(),
        "metrics": get_model_metrics(),
        "disclaimer": get_disclaimer()
    }
