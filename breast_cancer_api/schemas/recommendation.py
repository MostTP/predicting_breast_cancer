from pydantic import BaseModel
from typing import List, Dict, Any

class RecommendationResponse(BaseModel):
    best_treatment: Dict[str, Any]
    all_options: List[Dict[str, Any]]