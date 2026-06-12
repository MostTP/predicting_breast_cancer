from pydantic import BaseModel
from typing import List, Dict, Any

class RecommendationResponse(BaseModel):
    top_ranked_option: Dict[str, Any]
    ranked_treatment_options: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    skipped_combinations: List[Dict[str, Any]]
    warnings: List[str]
    model_metadata: Dict[str, Any]
    disclaimer: Dict[str, Any]
    best_treatment: Dict[str, Any]
    all_options: List[Dict[str, Any]]
