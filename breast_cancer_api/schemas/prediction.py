from typing import Any, Dict, List

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    estimated_outcome_probability: float
    model_score: float
    threshold: float
    probability: float
    prediction: int
    warnings: List[str]
    model_metadata: Dict[str, Any]
    disclaimer: Dict[str, Any]
