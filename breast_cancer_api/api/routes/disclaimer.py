from fastapi import APIRouter
from ...services.disclaimer_service import get_disclaimer

router = APIRouter()

@router.get("/disclaimer")
def disclaimer_route():
    """Return the medical disclaimer for this system."""
    return get_disclaimer()
